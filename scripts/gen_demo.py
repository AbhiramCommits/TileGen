#!/usr/bin/env python3
"""Generate CUDA and CPU sources for every standard kernel across a few configs.

Writes into ``build/generated/`` and prints the pass log for the fused kernel so
that the loop-fusion pass is visible in the emitted header comment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilegen.codegen import Config, generate  # noqa: E402
from tilegen.dsl import standard_kernels  # noqa: E402

OUTDIR = "build/generated"

CONFIGS = [
    Config(tile_size=32, threads_per_row=1, unroll=1, vector_width=1, block_size=256),
    Config(tile_size=64, threads_per_row=4, unroll=4, vector_width=2, block_size=256),
    Config(tile_size=128, threads_per_row=8, unroll=2, vector_width=4, block_size=512),
]


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    kernels = standard_kernels()
    written = []
    for name, k in kernels.items():
        for cfg in CONFIGS:
            for backend in ("cuda", "cpu"):
                gen = generate(k, cfg, backend, outdir=OUTDIR, write=True)
                written.append(gen.filename)
    # a deliberately buggy variant used to exercise the differential tester
    bad = generate(kernels["spmv"], Config(tile_size=32, threads_per_row=1,
                                            unroll=1, inject_bug=True),
                   "cpu", outdir=OUTDIR, write=True)
    written.append(bad.filename)

    print("Generated %d source files in %s:" % (len(written), OUTDIR))
    for f in sorted(set(written)):
        print("  %s" % f)

    print("\n=== pass log for fused_spmv_axpy (tile=64, threads_per_row=4, unroll=4) ===")
    cfg = CONFIGS[1]
    gen = generate(kernels["fused_spmv_axpy"], cfg, "cpu", outdir=OUTDIR, write=False)
    for entry in gen.pass_log:
        print("  [%-16s] %s" % (entry["pass"], entry["action"]))
    print("\nBefore fusion the kernel had two statements; after fusion it has one")
    print("loop nest with the spmv row reduction inlined into the axpy store.")

    print("\n=== CUDA markers ===")
    cuda = generate(kernels["spmv"], CONFIGS[1], "cuda", outdir=OUTDIR, write=False)
    for marker in ("__global__", "__shfl_down_sync", "#pragma unroll"):
        print("  %-20s present=%s" % (marker, marker in cuda.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
