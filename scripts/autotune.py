#!/usr/bin/env python3
"""Autotune every kernel over the matrix corpus and write results/.

Usage:
    python scripts/autotune.py --budget 40 --out results
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilegen.autotune import BASELINE, tune, write_outputs  # noqa: E402
from tilegen.env import probe  # noqa: E402
from tilegen.matrices import build_corpus, corpus_classes  # noqa: E402

MATRIX_DIR = "data/matrices"
ALL_KERNELS = ["spmv", "spmm", "sddmm", "fused_spmv_axpy", "axpy"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40,
                    help="max variants per (kernel, matrix) pair")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mode", default="random", choices=["random", "coordinate"])
    ap.add_argument("--backend", default=None, help="cpu|cuda (default: probe)")
    ap.add_argument("--kernels", default=",".join(ALL_KERNELS))
    ap.add_argument("--classes", default=",".join(corpus_classes()))
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)

    env = probe()
    backend = args.backend or env["backend"]
    kernels = [k for k in args.kernels.split(",") if k]
    classes = [c for c in args.classes.split(",") if c]

    missing = [c for c in classes if not os.path.exists(
        os.path.join(MATRIX_DIR, "%s.npz" % c))]
    if missing:
        print("[warn] missing matrices %s; generating locally" % missing)
        build_corpus(MATRIX_DIR)

    print("backend=%s budget=%d mode=%s kernels=%s classes=%s"
          % (backend, args.budget, args.mode, kernels, classes))

    all_results = []
    for class_name in classes:
        path = os.path.join(MATRIX_DIR, "%s.npz" % class_name)
        for kernel in kernels:
            res = tune(kernel, path, budget=args.budget, mode=args.mode,
                       backend=backend, iters=args.iters, warmup=args.warmup)
            all_results.append(res)
            w = res["winner"]
            if w:
                print("%-16s %-18s winner tile=%-4d tpr=%-3d unroll=%-2d vec=%-2d "
                      "block=%-4d ms=%.6f speedup=%.2fx"
                      % (class_name, kernel, w["tile_size"], w["threads_per_row"],
                         w["unroll"], w["vector_width"], w["block_size"],
                         res["winner_metrics"]["ms_median"],
                         res["winner_metrics"]["speedup_vs_naive"] or 0.0))
            else:
                print("%-16s %-18s no legal/verified variant" % (class_name, kernel))

    from tilegen.autotune import prune_log
    write_outputs(all_results, args.out, env, prune_log())
    print("\nwrote %s/autotune.json, %s/autotune.csv, %s/best_configs.json"
          % (args.out, args.out, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
