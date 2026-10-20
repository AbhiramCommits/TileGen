#!/usr/bin/env python3
"""Ablate each optimization pass, diffing the emitted code before timing.

Why the code diff comes first
-----------------------------
If removing a pass leaves the generated source byte-identical, its contribution
is 1.00x by definition.  Timing two copies of the same kernel measures only
clock drift and cache noise, which is how an earlier version of this script
produced 0.8x/1.2x "contributions" for passes that did nothing.

For each (kernel, matrix class, pass) the script therefore:

1. emits the full-pipeline kernel and the kernel with exactly one pass skipped,
2. compares the emitted code with the diagnostic header comments stripped,
3. marks identical cases ``no change`` and does not time them,
4. for changed cases, verifies and times full/ablated variants interleaved,
   repeating the whole comparison ``--repeats`` times and reporting the
   median contribution plus its min/max spread.

Writes:
    results/pass_ablation.json
    results/pass_ablation.md
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import statistics
import sys
import time

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


# --------------------------------------------------------------------------- #
# code diffing
# --------------------------------------------------------------------------- #
def _code_body(source: str) -> str:
    """Emitted code only: drop ``//`` header/diagnostic lines."""
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("//"))


def _code_diff(full_source: str, ablated_source: str) -> dict:
    full_body = _code_body(full_source)
    ablated_body = _code_body(ablated_source)
    diff = list(difflib.unified_diff(
        full_body.splitlines(), ablated_body.splitlines(),
        fromfile="full", tofile="ablated", lineterm=""))
    added = sum(1 for line in diff
                if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff
                  if line.startswith("-") and not line.startswith("---"))
    return {
        "identical": full_body == ablated_body,
        "full_sha": hashlib.sha1(full_body.encode()).hexdigest()[:12],
        "ablated_sha": hashlib.sha1(ablated_body.encode()).hexdigest()[:12],
        "added_lines": added,
        "removed_lines": removed,
    }


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def _stats(times) -> dict:
    ordered = sorted(times)
    n = len(ordered)
    if n == 0:
        return {"ms_median": None, "ms_mean": None, "ms_min": None,
                "ms_max": None, "ms_std": None, "ms_p10": None,
                "ms_p90": None, "runs": 0}
    return {
        "ms_median": statistics.median(ordered),
        "ms_mean": statistics.mean(ordered),
        "ms_min": ordered[0],
        "ms_max": ordered[-1],
        "ms_std": statistics.stdev(ordered) if n > 1 else 0.0,
        "ms_p10": ordered[int(0.10 * (n - 1))],
        "ms_p90": ordered[int(0.90 * (n - 1))],
        "runs": n,
    }


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
    return _stats(times)


def _benchmark_cpu(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return _stats(times)


def _benchmark(fn, warmup: int, iters: int, backend: str) -> dict:
    if backend == "cuda":
        return _benchmark_cuda(fn, warmup, iters)
    return _benchmark_cpu(fn, warmup, iters)


def _tolerance(A, ref) -> float:
    eps = float(np.finfo(np.float64).eps)
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    row_nnz = int(np.diff(A.indptr).max()) if A.shape[0] else 1
    return float(8.0 * eps * max(1, row_nnz) * scale + 1e-12)


# --------------------------------------------------------------------------- #
# generation / execution
# --------------------------------------------------------------------------- #
def _prepare(kernel_name, gen, A, data, ref, workdir, backend):
    """Compile a generated kernel, run it once, and return a timed caller."""
    lib_path, log = compile_generated(gen, backend, workdir)
    if lib_path is None:
        raise RuntimeError("compilation skipped: %s" % log)

    caller, out, keep = _make_caller(lib_path, kernel_name, A, data, 8,
                                     backend=backend)
    caller()
    diff = np.abs(np.asarray(out, dtype=np.float64).ravel()
                  - np.asarray(ref, dtype=np.float64).ravel())
    max_abs = float(diff.max()) if diff.size else 0.0
    tol = _tolerance(A, np.asarray(ref, dtype=np.float64))
    return {
        "caller": caller,
        "keep": keep,
        "max_abs_err": max_abs,
        "tolerance": tol,
        "verified": bool(max_abs <= tol),
        "log": log,
        "filename": gen.filename,
    }


def _release(keep) -> None:
    for item in keep:
        free = getattr(item, "free", None)
        if callable(free):
            free()


def _load_configs(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return {row["matrix_class"]: Config(**row["tuned_config"])
            for row in data.get("classes", [])}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt_contribution(row) -> str:
    if row["status"] == "no_change":
        return "no change"
    if row["status"] != "ok":
        return "**INVALID**"
    c = row["contribution"]
    return "%.2fx (%.2f–%.2f)" % (c["median"], c["min"], c["max"])


def _markdown_table(rows, kernel, passes_list, classes):
    header = ["pass", "changed"] + list(classes) + ["mean (changed)"]
    lines = ["| %s |" % " | ".join(header),
             "|---|%s---|" % ("---|" * (len(header) - 1))]
    for pass_name in passes_list:
        cells = []
        measured = []
        changed = 0
        for cls in classes:
            row = next((r for r in rows
                        if r["kernel"] == kernel and r["pass"] == pass_name
                        and r["matrix_class"] == cls), None)
            if row is None:
                cells.append("-")
                continue
            if row["code_diff"]["identical"]:
                cells.append("no change")
            else:
                changed += 1
                if row["status"] == "ok":
                    measured.append(row["contribution"]["median"])
                cells.append(_fmt_contribution(row))
        mean = "%.2fx" % statistics.mean(measured) if measured else "-"
        lines.append("| `%s` | %d/%d | %s | %s |"
                     % (pass_name, changed, len(classes), " | ".join(cells), mean))
    return "\n".join(lines)


def _markdown_cuda_effect(rows, kernel, passes_list, classes):
    header = ["pass"] + list(classes)
    lines = ["| %s |" % " | ".join(header),
             "|---|%s" % ("---|" * (len(header) - 1))]
    for pass_name in passes_list:
        cells = []
        for cls in classes:
            row = next((r for r in rows
                        if r["kernel"] == kernel and r["pass"] == pass_name
                        and r["matrix_class"] == cls), None)
            if row is None:
                cells.append("-")
                continue
            diff = row.get("cuda_code_diff")
            if diff is None:
                cells.append("-")
            elif diff["identical"]:
                cells.append("no change")
            else:
                cells.append("changed (+%d/−%d)"
                             % (diff["added_lines"], diff["removed_lines"]))
        lines.append("| `%s` | %s |" % (pass_name, " | ".join(cells)))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100,
                    help="timed repetitions per benchmark run")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5,
                    help="interleaved full/ablated comparisons per pass")
    ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--classes", default=",".join(corpus_classes()))
    ap.add_argument("--kernels", default="spmv,fused_spmv_axpy")
    ap.add_argument("--configs", default=os.path.join(
        RESULTS_DIR, "cusparse_comparison.json"))
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--full-out", default=FULL_DIR)
    args = ap.parse_args(argv)

    if args.iters < 10:
        raise SystemExit("--iters must be >= 10")
    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2 to report a spread")

    env = probe()
    backend = env.get("backend") if args.backend == "auto" else args.backend
    if backend == "cuda" and env.get("backend") != "cuda":
        raise SystemExit("CUDA backend requested but probe() selected %r" % env)
    if backend is None:
        raise SystemExit("no usable backend from probe(): %r" % env)

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
        print("\n[%s] config=%s backend=%s" % (class_name, cfg, backend))
        for kernel_name in kernels:
            ref = _reference(kernel_name, A, data)
            print("  %-18s" % kernel_name)

            kernel_obj = standard_kernels()[kernel_name]
            full_gen = generate(kernel_obj, cfg, backend,
                                outdir=workdir, write=True)
            # Static CUDA source effect, available even on a CPU-only host.
            # This is what the old CUDA table should have been checking before
            # it timed anything.
            cuda_full_gen = (None if backend == "cuda" else
                             generate(kernel_obj, cfg, "cuda",
                                      outdir=workdir, write=False))
            full = _prepare(kernel_name, full_gen, A, data, ref, workdir,
                            backend)
            if not full["verified"]:
                raise RuntimeError("%s/%s full pipeline failed verification: "
                                   "max_abs=%g tol=%g"
                                   % (kernel_name, class_name,
                                      full["max_abs_err"], full["tolerance"]))

            for pass_name in pass_names:
                ablated_gen = generate(
                    kernel_obj, cfg, backend,
                    outdir=workdir, write=True, skip_passes=[pass_name])
                code_diff = _code_diff(full_gen.source, ablated_gen.source)
                if backend == "cuda":
                    cuda_diff = code_diff
                else:
                    cuda_ablated_gen = generate(
                        kernel_obj, cfg, "cuda", outdir=workdir, write=False,
                        skip_passes=[pass_name])
                    cuda_diff = _code_diff(cuda_full_gen.source,
                                           cuda_ablated_gen.source)

                row = {
                    "kernel": kernel_name,
                    "matrix_class": class_name,
                    "pass": pass_name,
                    "config": cfg.to_dict(),
                    "code_diff": code_diff,
                    "cuda_code_diff": cuda_diff,
                }

                if code_diff["identical"]:
                    # 1.00x by definition; do not pretend to measure it.
                    row.update({
                        "status": "no_change",
                        "verified": True,
                        "contribution": {"median": 1.0, "min": 1.0,
                                         "max": 1.0, "std": 0.0},
                        "full": None,
                        "ablated": None,
                    })
                    rows.append(row)
                    print("    without %-16s no change (identical code)"
                          % pass_name)
                    continue

                if kernel_name == "fused_spmv_axpy" and pass_name == "loop_fusion":
                    # Without fusion the first statement writes temporary t,
                    # not z; the ABI therefore differs from the fused kernel's
                    # and the z reference cannot be compared.
                    row.update({
                        "status": "invalid",
                        "verified": False,
                        "reason": ("loop_fusion disabled: kernel writes "
                                   "temporary t instead of fused output z"),
                        "contribution": None,
                        "full": None,
                        "ablated": None,
                    })
                    rows.append(row)
                    print("    without %-16s INVALID (ABI changes to t)"
                          % pass_name)
                    continue

                ablated = _prepare(kernel_name, ablated_gen, A, data, ref,
                                   workdir, backend)
                if not ablated["verified"]:
                    row.update({
                        "status": "invalid",
                        "verified": False,
                        "reason": "ablated kernel failed differential verification",
                        "full": None,
                        "ablated": None,
                        "contribution": None,
                    })
                    rows.append(row)
                    _release(ablated["keep"])
                    print("    without %-16s INVALID (reference mismatch)"
                          % pass_name)
                    continue

                full_runs, ablated_runs, ratios = [], [], []
                for _ in range(args.repeats):
                    f = _benchmark(full["caller"].launch, args.warmup,
                                   args.iters, backend)
                    a = _benchmark(ablated["caller"].launch, args.warmup,
                                   args.iters, backend)
                    full_runs.append(f)
                    ablated_runs.append(a)
                    ratios.append(a["ms_median"] / f["ms_median"])

                contribution = {
                    "median": statistics.median(ratios),
                    "mean": statistics.mean(ratios),
                    "min": min(ratios),
                    "max": max(ratios),
                    "std": statistics.stdev(ratios)
                    if len(ratios) > 1 else 0.0,
                }
                row.update({
                    "status": "ok",
                    "verified": True,
                    "full": full_runs,
                    "ablated": ablated_runs,
                    "contribution": contribution,
                })
                rows.append(row)
                print("    without %-16s %5.2fx (%.2f–%.2f)  code +%d/-%d  verified=%s"
                      % (pass_name, contribution["median"],
                         contribution["min"], contribution["max"],
                         code_diff["added_lines"],
                         code_diff["removed_lines"],
                         ablated["verified"]))
                _release(ablated["keep"])

            _release(full["keep"])

    payload = {
        "hardware": env,
        "backend": backend,
        "timer": ("torch.cuda.Event (CUDA events; transfers excluded)"
                  if backend == "cuda"
                  else "time.perf_counter (CPU, single process)"),
        "warmup": args.warmup,
        "iterations": args.iters,
        "repeats": args.repeats,
        "method": ("Diff emitted code first; identical passes are 'no change' "
                   "and are not timed. Changed cases are verified and timed "
                   "with interleaved full/ablated runs."),
        "configs_from": args.configs,
        "passes": pass_names,
        "rows": rows,
    }
    json_path = os.path.join(args.out, "pass_ablation.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    md = ["# TileGen pass ablation (%s)" % backend.upper(), "",
          "Generated code is emitted with the full pipeline and with exactly "
          "one pass skipped. The bodies are diffed with the diagnostic header "
          "comments stripped; if they are identical the cell says **no "
          "change** and that case is not timed. Changed cases run %d "
          "interleaved full/ablated comparisons of %d warmup + %d timed "
          "iterations; the cell is `median (min–max)` contribution. "
          "Contribution = disabled_ms / full_ms. Schedules are the tuned "
          "SpMV schedules from `results/cusparse_comparison.json`."
          % (args.repeats, args.warmup, args.iters), ""]
    for kernel_name in kernels:
        md.append("## %s" % kernel_name)
        md.append("")
        md.append(_markdown_table(rows, kernel_name, pass_names, classes))
        md.append("")
    md.append("`changed` counts matrix classes whose emitted code differs "
              "when the pass is disabled. `no change` means the pass is a "
              "no-op for that schedule/kernel, not that its contribution is "
              "unknown.")
    md.append("")
    md.append("## Emitted CUDA code effect (static)")
    md.append("")
    md.append("The timing table above uses the **%s** backend. These cells "
              "diff the CUDA source instead, so a pass that changes CUDA "
              "lowering is visible even when no GPU is available. `+a/−r` "
              "counts added/removed code lines outside the diagnostic "
              "header." % backend)
    md.append("")
    for kernel_name in kernels:
        md.append("### %s" % kernel_name)
        md.append("")
        md.append(_markdown_cuda_effect(rows, kernel_name, pass_names, classes))
        md.append("")
    md_path = os.path.join(args.out, "pass_ablation.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md).rstrip() + "\n")

    print("\nwrote %s and %s" % (json_path, md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
