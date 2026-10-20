#!/usr/bin/env python3
"""Ablate each optimization pass on CUDA: disable one pass, verify, benchmark.

For each matrix class this uses the CUDA-tuned SpMV schedule from
``results/cusparse_comparison.json``.  The full pipeline is the reference; each
row removes exactly one pass.  Contribution is ``disabled_ms / full_ms``, so
values above 1 mean the pass was helping.  A row is marked invalid if the
ablated kernel no longer matches the float64 SciPy reference.

Writes:
    results/pass_ablation.json
    results/pass_ablation.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from tilegen import passes  # noqa: E402
from tilegen.autotune import BASELINE, _make_caller  # noqa: E402
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


def _benchmark_cuda(fn, warmup: int, iters: int) -> dict:
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
        "runs": len(times),
    }


def _tolerance(A, ref) -> float:
    eps = float(np.finfo(np.float64).eps)
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    row_nnz = int(np.diff(A.indptr).max()) if A.shape[0] else 1
    return float(8.0 * eps * max(1, row_nnz) * scale + 1e-12)


def _evaluate(kernel_name, cfg, A, data, ref, skip, workdir, warmup, iters):
    kernels = standard_kernels()
    gen = generate(kernels[kernel_name], cfg, "cuda", outdir=workdir,
                   write=True, skip_passes=skip)
    lib_path, log = compile_generated(gen, "cuda", workdir)
    if lib_path is None:
        raise RuntimeError("CUDA compilation skipped: %s" % log)

    if kernel_name == "fused_spmv_axpy" and "loop_fusion" in skip:
        # Without fusion the first statement writes temporary t, not z; the
        # generated ABI therefore differs from the fused kernel's ABI and the
        # result cannot be compared against the z reference.
        return {
            "ms_median": None,
            "ms_mean": None,
            "ms_min": None,
            "runs": 0,
            "max_abs_err": None,
            "tolerance": _tolerance(A, np.asarray(ref, dtype=np.float64)),
            "verified": False,
            "reason": ("loop_fusion disabled: kernel writes temporary t instead "
                       "of fused output z"),
            "filename": gen.filename,
        }

    caller, out, keep = _make_caller(lib_path, kernel_name, A, data, 8,
                                     backend="cuda")
    caller()
    diff = np.abs(np.asarray(out, dtype=np.float64).ravel()
                  - np.asarray(ref, dtype=np.float64).ravel())
    max_abs = float(diff.max()) if diff.size else 0.0
    tol = _tolerance(A, np.asarray(ref, dtype=np.float64))
    timing = _benchmark_cuda(caller.launch, warmup, iters)
    return {
        "ms_median": timing["ms_median"],
        "ms_mean": timing["ms_mean"],
        "ms_min": timing["ms_min"],
        "runs": timing["runs"],
        "max_abs_err": max_abs,
        "tolerance": tol,
        "verified": bool(max_abs <= tol),
        "filename": gen.filename,
    }


def _load_configs(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return {row["matrix_class"]: Config(**row["tuned_config"])
            for row in data.get("classes", [])}


def _markdown_table(rows, kernel, passes_list, classes):
    lines = ["| pass | %s | mean |" % " | ".join(classes),
             "|---|%s---|" % ("---|" * len(classes))]
    for pass_name in passes_list:
        cells = []
        values = []
        for cls in classes:
            row = next((r for r in rows
                        if r["kernel"] == kernel and r["pass"] == pass_name
                        and r["matrix_class"] == cls), None)
            if row is None:
                cells.append("-")
                continue
            if not row["verified"]:
                cells.append("**INVALID**")
                continue
            value = row["speedup_contribution"]
            values.append(value)
            cells.append("%.2fx" % value)
        mean = "%.2fx" % statistics.mean(values) if values else "-"
        lines.append("| `%s` | %s | %s |"
                     % (pass_name, " | ".join(cells), mean))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--classes", default=",".join(corpus_classes()))
    ap.add_argument("--kernels", default="spmv,fused_spmv_axpy")
    ap.add_argument("--configs", default=os.path.join(
        RESULTS_DIR, "cusparse_comparison.json"))
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--full-out", default=FULL_DIR)
    args = ap.parse_args(argv)

    if args.iters < 10:
        raise SystemExit("--iters must be >= 10")

    env = probe()
    if env.get("backend") != "cuda":
        raise SystemExit("CUDA backend not selected by probe(): %r" % env)

    classes = [c for c in args.classes.split(",") if c]
    kernels = [k for k in args.kernels.split(",") if k]
    missing = [c for c in classes
               if not os.path.exists(os.path.join(MATRIX_DIR, "%s.npz" % c))]
    if missing:
        print("[warn] missing matrices %s; generating locally" % missing)
        build_corpus(MATRIX_DIR)

    tuned = _load_configs(args.configs)
    workdir = os.path.join(args.full_out, "generated")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    pass_names = [p.__name__ for p in passes.PASSES]
    rows = []

    for class_name in classes:
        path = os.path.join(MATRIX_DIR, "%s.npz" % class_name)
        A, _ = _load_matrix(path)
        data = _inputs_for("spmv", A, 8, 0)
        cfg = tuned.get(class_name, BASELINE)
        print("\n[%s] config=%s" % (class_name, cfg))
        for kernel_name in kernels:
            ref = _reference(kernel_name, A, data)
            print("  %-18s" % kernel_name)
            for pass_name in pass_names:
                # Measure the full pipeline immediately before each ablation so
                # clock drift cannot masquerade as a pass contribution.
                full = _evaluate(kernel_name, cfg, A, data, ref, [], workdir,
                                 args.warmup, args.iters)
                ablated = _evaluate(kernel_name, cfg, A, data, ref,
                                    [pass_name], workdir,
                                    args.warmup, args.iters)
                contribution = (
                    ablated["ms_median"] / full["ms_median"]
                    if ablated["ms_median"] and full["ms_median"] else None)
                row = {
                    "kernel": kernel_name,
                    "matrix_class": class_name,
                    "pass": pass_name,
                    "config": cfg.to_dict(),
                    "full": full,
                    "ablated": ablated,
                    "speedup_contribution": contribution,
                    "verified": ablated["verified"],
                }
                rows.append(row)
                if ablated["ms_median"] is None:
                    print("    without %-16s INVALID (%s)"
                          % (pass_name, ablated.get("reason", "")))
                else:
                    print("    without %-16s %.6f ms  %5.2fx  verified=%s"
                          % (pass_name, ablated["ms_median"], contribution,
                             ablated["verified"]))

    payload = {
        "hardware": env,
        "timer": "torch.cuda.Event (CUDA events; transfers excluded)",
        "warmup": args.warmup,
        "iterations": args.iters,
        "configs_from": args.configs,
        "passes": pass_names,
        "rows": rows,
    }
    json_path = os.path.join(args.out, "pass_ablation.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    md = ["# TileGen pass ablation (CUDA)", "",
          "CUDA-event timing: %d warmup + %d timed runs, median; transfers "
          "excluded. Contribution = disabled_ms / full_ms; >1 means the pass "
          "was helping. Schedules are the tuned SpMV schedules from "
          "`results/cusparse_comparison.json`." % (args.warmup, args.iters), ""]
    for kernel_name in kernels:
        md.append("## %s" % kernel_name)
        md.append("")
        md.append(_markdown_table(rows, kernel_name, pass_names, classes))
        md.append("")
    md_path = os.path.join(args.out, "pass_ablation.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))

    print("\nwrote %s and %s" % (json_path, md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
