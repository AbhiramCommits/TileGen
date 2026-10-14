#!/usr/bin/env python3
"""Run every (kernel, config, matrix-class) triple through the differential tester.

Writes results/correctness.json and prints a pass/fail table.  The deliberately
buggy spmv variant is included so the table shows the harness catching it.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilegen.codegen import Config  # noqa: E402
from tilegen.env import probe  # noqa: E402
from tilegen.matrices import build_corpus, corpus_classes  # noqa: E402
from tilegen.verify import verify  # noqa: E402

MATRIX_DIR = "data/matrices"
RESULTS_DIR = "results"

KERNELS = ["spmv", "spmm", "sddmm", "fused_spmv_axpy", "axpy"]

CONFIGS = [
    Config(tile_size=32, threads_per_row=1, unroll=1, vector_width=1, block_size=128),
    Config(tile_size=64, threads_per_row=1, unroll=4, vector_width=1, block_size=256),
    Config(tile_size=128, threads_per_row=8, unroll=2, vector_width=2, block_size=512),
]


def main() -> int:
    env = probe()
    backend = env["backend"]
    print("backend: %s (numeric reference runs on the CPU in float64)" % backend)

    # ensure the corpus exists
    missing = [c for c in corpus_classes()
               if not os.path.exists(os.path.join(MATRIX_DIR, "%s.npz" % c))]
    if missing:
        print("[warn] missing matrices %s; generating locally" % missing)
        build_corpus(MATRIX_DIR)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for class_name in corpus_classes():
        path = os.path.join(MATRIX_DIR, "%s.npz" % class_name)
        for kernel in KERNELS:
            for cfg in CONFIGS:
                r = verify(kernel, cfg, path, backend=backend)
                results.append(r)
        # buggy variant on this matrix class
        bug_cfg = Config(tile_size=32, threads_per_row=1, unroll=1, inject_bug=True)
        rb = verify("spmv", bug_cfg, path, backend=backend)
        results.append(rb)

    # print table
    print()
    header = "%-16s %-18s %-4s %8s %12s %12s %12s %6s" % (
        "matrix_class", "kernel", "ok", "rows", "nnz", "max_abs_err",
        "max_rel_err", "bug")
    print(header)
    print("-" * len(header))
    for r in results:
        print("%-16s %-18s %-4s %8d %12d %12.3e %12.3e %6s" % (
            r.matrix_class, r.kernel, "PASS" if r.ok else "FAIL", r.rows, r.nnz,
            r.max_abs_err, r.max_rel_err, "yes" if r.inject_bug else ""))

    normal = [r for r in results if not r.inject_bug]
    buggy = [r for r in results if r.inject_bug]
    summary = {
        "backend": backend,
        "hardware": env,
        "total_triples": len(results),
        "normal_triples": len(normal),
        "normal_passed": sum(1 for r in normal if r.ok),
        "bug_triples": len(buggy),
        "bug_caught": sum(1 for r in buggy if not r.ok),
        "max_abs_err": max((r.max_abs_err for r in normal), default=0.0),
        "max_rel_err": max((r.max_rel_err for r in normal), default=0.0),
        "all_normal_passed": all(r.ok for r in normal),
        "all_bugs_caught": all(not r.ok for r in buggy),
    }

    payload = {"summary": summary, "results": [r.to_dict() for r in results]}
    out = os.path.join(RESULTS_DIR, "correctness.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print()
    print("normal triples passed : %d / %d" % (summary["normal_passed"], len(normal)))
    print("injected bugs caught  : %d / %d" % (summary["bug_caught"], len(buggy)))
    print("max abs err (normal)  : %.3e" % summary["max_abs_err"])
    print("max rel err (normal)  : %.3e" % summary["max_rel_err"])
    print("wrote %s" % out)

    ok = summary["all_normal_passed"] and summary["all_bugs_caught"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
