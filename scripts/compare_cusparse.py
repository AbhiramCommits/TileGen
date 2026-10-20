#!/usr/bin/env python3
"""Autotune SpMV on CUDA per matrix class and compare with cuSPARSE.

The autotuner selects a schedule per class.  The winner and the fixed naive
default are then re-timed with CUDA events (warmup + median of >=50 runs), with
host/device transfers excluded.  cuSPARSE SpMV is timed the same way through
CuPy's CSR SpMV (which dispatches to cuSPARSE).

Writes:
    build/autotune_cuda_spmv.json   full autotune records
    results/cusparse_comparison.json compact comparison table data
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilegen.autotune import BASELINE, _make_caller, prune_log, tune  # noqa: E402
from tilegen.codegen import Config, generate  # noqa: E402
from tilegen.dsl import standard_kernels  # noqa: E402
from tilegen.env import probe  # noqa: E402
from tilegen.matrices import build_corpus, corpus_classes  # noqa: E402
from tilegen.verify import (  # noqa: E402
    _inputs_for,
    _load_matrix,
    _reference,
    compile_generated,
)

MATRIX_DIR = "data/matrices"
RESULTS_DIR = "results"
FULL_DIR = "build"
KERNEL = "spmv"


def _benchmark_cuda(fn, warmup: int, iters: int) -> dict:
    """CUDA-event timing of ``fn``; no host/device copies inside the loop."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "ms_median": statistics.median(times),
        "ms_mean": statistics.mean(times),
        "ms_min": min(times),
        "ms_max": max(times),
        "runs": len(times),
    }


def _tilegen_bench(A, data, cfg: Config, workdir: str):
    kernel = standard_kernels()[KERNEL]
    gen = generate(kernel, cfg, "cuda", outdir=workdir, write=True)
    lib_path, log = compile_generated(gen, "cuda", workdir)
    if lib_path is None:
        raise RuntimeError("CUDA compilation skipped: %s" % log)
    caller, out, keep = _make_caller(lib_path, KERNEL, A, data, 8,
                                     backend="cuda")
    caller()  # correctness / warmup once, including the output copy
    return caller.launch, out, keep, gen


def _cusparse_bench(A, x):
    """Return (callable, output_holder) backed by CuPy's cuSPARSE CSR SpMV."""
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse

    crow = cp.asarray(A.indptr, dtype=cp.int64)
    col = cp.asarray(A.indices, dtype=cp.int64)
    vals = cp.asarray(A.data, dtype=cp.float64)
    mat = cpsparse.csr_matrix((vals, col, crow), shape=A.shape)
    vec = cp.asarray(x, dtype=cp.float64)
    holder = {}

    def call():
        holder["y"] = mat @ vec  # cupyx dispatches CSR SpMV to cuSPARSE

    call()  # warmup / validation call
    return call, holder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40,
                    help="autotuner variants per matrix class")
    ap.add_argument("--iters", type=int, default=100,
                    help="timed CUDA-event repetitions (>=50 required)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--mode", default="random", choices=["random", "coordinate"])
    ap.add_argument("--classes", default=",".join(corpus_classes()))
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--full-out", default=FULL_DIR)
    args = ap.parse_args(argv)

    if args.iters < 50:
        raise SystemExit("--iters must be >= 50")

    env = probe()
    if env.get("backend") != "cuda":
        raise SystemExit("CUDA backend not selected by probe(): %r" % env)
    if env.get("sm_arch"):
        os.environ.setdefault("TILEGEN_CUDA_ARCH", str(env["sm_arch"]))

    classes = [c for c in args.classes.split(",") if c]
    missing = [c for c in classes
               if not os.path.exists(os.path.join(MATRIX_DIR, "%s.npz" % c))]
    if missing:
        print("[warn] missing matrices %s; generating locally" % missing)
        build_corpus(MATRIX_DIR)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.full_out, exist_ok=True)
    workdir = os.path.join(args.full_out, "generated")

    all_results = []
    comparison = []

    for class_name in classes:
        path = os.path.join(MATRIX_DIR, "%s.npz" % class_name)
        A, loaded_class = _load_matrix(path)
        print("\n[%s] rows=%d nnz=%d  autotuning %s ..."
              % (class_name, A.shape[0], A.nnz, KERNEL))
        res = tune(KERNEL, path, budget=args.budget, mode=args.mode,
                   backend="cuda", K=8, workdir=workdir,
                   warmup=args.warmup, iters=25, seed=0)
        all_results.append(res)
        winner = res.get("winner")
        if winner is None:
            print("  no legal/verified variant")
            continue

        data = _inputs_for(KERNEL, A, 8, 0)
        ref = _reference(KERNEL, A, data)
        tuned_cfg = Config(**winner)
        default_cfg = Config(**BASELINE.to_dict())

        tuned_call, tuned_out, tuned_keep, tuned_gen = _tilegen_bench(
            A, data, tuned_cfg, workdir)
        default_call, default_out, default_keep, default_gen = _tilegen_bench(
            A, data, default_cfg, workdir)
        cus_call, cus_holder = _cusparse_bench(A, data[0])

        tuned_bench = _benchmark_cuda(tuned_call, args.warmup, args.iters)
        default_bench = _benchmark_cuda(default_call, args.warmup, args.iters)
        cus_bench = _benchmark_cuda(cus_call, args.warmup, args.iters)

        # Correctness checks on the warmup results.
        import numpy as np
        tuned_err = float(np.max(np.abs(tuned_out - ref)))
        default_err = float(np.max(np.abs(default_out - ref)))
        cus_out = cus_holder["y"].get().ravel()
        cus_err = float(np.max(np.abs(cus_out - ref)))

        row = {
            "matrix_class": class_name,
            "rows": int(A.shape[0]),
            "nnz": int(A.nnz),
            "tuned_config": winner,
            "default_config": BASELINE.to_dict(),
            "tuned": tuned_bench,
            "default": default_bench,
            "cusparse": cus_bench,
            "speedup_tuned_vs_default": (default_bench["ms_median"]
                                         / tuned_bench["ms_median"]),
            "speedup_tuned_vs_cusparse": (cus_bench["ms_median"]
                                          / tuned_bench["ms_median"]),
            "speedup_default_vs_cusparse": (cus_bench["ms_median"]
                                            / default_bench["ms_median"]),
            "max_abs_err": {
                "tuned": tuned_err,
                "default": default_err,
                "cusparse": cus_err,
            },
            "autotune_winner_ms_median": res["winner_metrics"]["ms_median"],
            "autotune_baseline_ms_min": res["baseline_ms_min"],
            "generated": {
                "tuned": tuned_gen.filename,
                "default": default_gen.filename,
            },
        }
        comparison.append(row)
        print("  tuned   %s  %.6f ms  (%.2fx vs default)"
              % (tuned_gen.filename, tuned_bench["ms_median"],
                 row["speedup_tuned_vs_default"]))
        print("  default %s  %.6f ms"
              % (default_gen.filename, default_bench["ms_median"]))
        print("  cusparse %.6f ms  -> TileGen/cuSPARSE %.2fx"
              % (cus_bench["ms_median"], row["speedup_tuned_vs_cusparse"]))

    full_payload = {
        "hardware": env,
        "baseline": BASELINE.to_dict(),
        "kernel": KERNEL,
        "search_space": {"budget": args.budget, "mode": args.mode},
        "pruned": prune_log(),
        "results": all_results,
    }
    with open(os.path.join(args.full_out, "autotune_cuda_spmv.json"), "w") as fh:
        json.dump(full_payload, fh, indent=2)

    payload = {
        "hardware": env,
        "kernel": KERNEL,
        "baseline": BASELINE.to_dict(),
        "warmup": args.warmup,
        "iterations": args.iters,
        "timer": "torch.cuda.Event (CUDA events; transfers excluded)",
        "autotune_budget": args.budget,
        "autotune_mode": args.mode,
        "classes": comparison,
    }
    out_path = os.path.join(args.out, "cusparse_comparison.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\nwrote %s and %s"
          % (out_path, os.path.join(args.full_out, "autotune_cuda_spmv.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
