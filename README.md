# TileGen

TileGen is a small DSL and schedule-driven compiler for sparse linear algebra.
It lowers index-notation kernels through a TACO-style IR and a pass pipeline into CUDA or OpenMP C++.
The optimization passes write an explicit loop/thread schedule, and the code generator consumes that
lowered schedule rather than reading `Config` directly. It autotunes schedules, differentially verifies
against float64 SciPy, and compares with cuSPARSE.

## Example

```python
from tilegen.dsl import Index, CSR, sparse_tensor, dense_tensor, reduce_sum, assign, kernel

i, j = Index("i"), Index("j")
A = sparse_tensor("A", CSR, shape=("M", "N"))
x = dense_tensor("x", shape=("N",))
y = dense_tensor("y", shape=("M",))
spmv = kernel("spmv", [assign(y[i], reduce_sum(j, A[i, j] * x[j]))])
```

`Kernel.pretty()`:

```text
kernel spmv {
  assign y[i] = sum(j, A[i, j] * x[j])
  // loop nest
  for i:
    for j in reduce(A row):
      assign y[i] += A[i, j] * x[j]
}
```

Emitted CUDA for the tuned `uniform_random` schedule (`T64/TPR8/U2/V2/B128`):

```cuda
__global__ void spmv_kernel(const int64_t* indptr, const int64_t* indices, const double* data,
                            const double* x, double* y, int64_t M, int64_t N, int64_t nnz) {
  int64_t TILE = 64;
  const int TPR = 8;
  int64_t lane = threadIdx.x % TPR;
  int64_t slots = blockDim.x / TPR;
  for (int64_t i0 = (int64_t)blockIdx.x * TILE; i0 < M; i0 += (int64_t)gridDim.x * TILE) {
    for (int64_t slot = threadIdx.x / TPR; slot < TILE; slot += slots) {
      int64_t i = i0 + slot;
      if (i >= M) continue;
      double acc = 0.0;
      for (int64_t p = indptr[i] + lane; p < indptr[i + 1]; p += TPR) {
        acc += (data[p] * x[indices[p]]);
      }
      #pragma unroll
      for (int off = TPR / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off, TPR);
      }
      if (lane == 0) {
        y[i] = acc;
      }
    }
  }
}
```

## Pass pipeline

`Config` is only the input to the passes. The passes lower it into the kernel's schedule IR:

| stage | what it changes | consumed by codegen |
|---|---|---|
| `canonicalize` | rewrites the expression tree (constant folding, deterministic reassociation) | yes, via the statement tree |
| `loop_fusion` | fuses a producer temporary into its consumer; refuses true dependencies | yes, via the statement tree |
| `tiling` | inserts `Tile(i, T)` for the row index | yes: codegen lowers it to an outer tile loop plus an inner stripped loop |
| `thread_mapping` | inserts `ThreadsPerRow(k)` and `Parallelize(i, block=B)` | yes: lane mapping, segmented shuffle, launch geometry |
| `unroll` | inserts `Unroll(j, U)` | yes: compressed/dense loop unrolling with remainder tail |
| `vectorize` | inserts `Vectorize(V)` when a unit-stride dense access exists | yes, on CPU dense inner loops; it is a no-op for SpMV's gather |
| `dead_code_elim` | removes dead temporaries | yes, via the statement tree |

The code generator no longer reads `tile_size`, `threads_per_row`, `unroll`, `vector_width`, or
`block_size` from `Config`. Disabling a pass therefore changes the emitted code, and if it does
not, the ablation reports `no change` instead of timing two copies of the same kernel.

## Results

Timings are from `results/cusparse_comparison.json`: Tesla T4, compute capability 7.5,
nvcc 13.0.88, CUDA events, 10 warmup + 100 timed runs, median, transfers excluded.
`tuned/default` is the fixed default `T32/TPR1/U1/V1/B128` divided by the tuned time; values
above 1 mean autotuning helped. `TileGen/cuSPARSE` above 1 means TileGen was faster.

| matrix class | rows | nnz | tuned schedule | TileGen tuned ms | TileGen default ms | cuSPARSE ms | tuned/default | TileGen/cuSPARSE |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| banded | 200000 | 1399988 | T128/TPR1/U4/V4/B512 | 0.148912 | 0.159792 | 0.186544 | 1.07x | 1.25x |
| uniform_random | 100000 | 1599890 | T64/TPR8/U2/V2/B128 | 0.200672 | 0.401376 | 0.217040 | 2.00x | 1.08x |
| power_law | 100000 | 804454 | T64/TPR8/U2/V2/B128 | 0.217552 | 0.604144 | 0.134768 | 2.78x | 0.62x |
| diagonal_heavy | 200000 | 599998 | T64/TPR1/U1/V1/B128 | 0.091872 | 0.092736 | 0.119088 | 1.01x | 1.30x |
| dense_rows | 20000 | 460083 | T32/TPR8/U8/V1/B512 | 0.139600 | 0.621264 | 0.116384 | 4.45x | 0.83x |
| blocked | 100000 | 1600000 | T32/TPR4/U4/V4/B128 | 0.145760 | 0.327712 | 0.195136 | 2.25x | 1.34x |

Differential verification from `results/correctness.json`: 90/90 normal triples passed and
6/6 injected-bug variants were caught; max abs error 3.41e-13, max rel error 4.99e-10.

Pass ablation from `results/pass_ablation.json`. The script first emits the full pipeline and
the same kernel with one pass skipped, strips the diagnostic header comments, and diffs the
remaining code. If the code is byte-identical the cell is `no change` and that case is not
timed. Changed cases are verified, then measured with 5 interleaved full/ablated comparisons of
10 warmup + 50 timed iterations; cells are median contribution with the min-max spread across
repeats. Contribution = disabled time / full time; above 1 means the pass was helping.

The timed table below uses the **CPU backend (Apple M3)** because no CUDA device was available
when the ablation was rerun. The tuned schedules are still the CUDA winners from
`results/cusparse_comparison.json`; treat the CPU magnitudes as a methodology check, not a
portability claim. The included static CUDA diff table shows which passes change emitted CUDA
on those schedules.

### spmv (CPU timing)

| pass | changed | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean (changed) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `canonicalize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `loop_fusion` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `tiling` | 6/6 | 0.93x (0.82-1.14) | 0.82x (0.49-1.04) | 0.63x (0.56-1.47) | 1.00x (0.34-1.12) | 0.61x (0.58-0.71) | 0.90x (0.87-1.00) | 0.82x |
| `thread_mapping` | 6/6 | 1.60x (1.39-1.72) | 3.09x (2.73-3.43) | 2.14x (1.34-2.87) | 2.77x (2.15-2.96) | 2.10x (1.89-2.15) | 1.33x (1.07-1.71) | 2.17x |
| `unroll` | 5/6 | 1.11x (0.84-1.29) | 0.92x (0.71-1.23) | 1.11x (0.57-1.25) | no change | 0.99x (0.98-1.00) | 1.09x (0.79-1.35) | 1.04x |
| `vectorize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `dead_code_elim` | 0/6 | no change | no change | no change | no change | no change | no change | - |

### fused_spmv_axpy (CPU timing)

| pass | changed | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean (changed) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `canonicalize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `loop_fusion` | 6/6 | **INVALID** | **INVALID** | **INVALID** | **INVALID** | **INVALID** | **INVALID** | - |
| `tiling` | 6/6 | 0.97x (0.94-1.09) | 0.88x (0.65-0.95) | 0.85x (0.52-1.25) | 1.05x (0.80-1.71) | 0.63x (0.59-0.71) | 0.92x (0.78-1.07) | 0.88x |
| `thread_mapping` | 6/6 | 1.57x (1.18-1.72) | 3.00x (2.84-3.05) | 2.85x (1.78-3.19) | 2.34x (1.45-2.49) | 2.09x (2.08-2.97) | 1.63x (1.31-1.82) | 2.25x |
| `unroll` | 5/6 | 1.16x (0.68-1.58) | 0.92x (0.73-1.04) | 1.06x (0.91-1.13) | no change | 0.97x (0.92-1.08) | 0.98x (0.92-0.99) | 1.02x |
| `vectorize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `dead_code_elim` | 0/6 | no change | no change | no change | no change | no change | no change | - |

`loop_fusion` on `fused_spmv_axpy` is marked invalid because disabling it leaves the kernel
writing temporary `t` instead of `z`; the ABI changes and the `z` reference cannot be compared.
It is a correctness pass, not a speed pass.

### Emitted CUDA code effect (static)

Because the timed table is CPU-only, the script also diffs CUDA source. `changed` means the
emitted CUDA body differs when the pass is disabled:

| pass | spmv | fused_spmv_axpy |
|---|---|---|
| `canonicalize` | no change | no change |
| `loop_fusion` | no change | changed |
| `tiling` | changed (all 6) | changed (all 6) |
| `thread_mapping` | changed (all 6) | changed (all 6) |
| `unroll` | changed on banded only (`TPR=1`); no change on the five `TPR>1` schedules | same |
| `vectorize` | no change | no change |
| `dead_code_elim` | no change | no change |

`vectorize` is a no-op here by construction: SpMV's `x[indices[p]]` is a gather and the dense
operand is rank-one, so there is no unit-stride dense axis to widen. It does affect CPU
`spmm`/`sddmm` dense inner loops.

On a CUDA host, rerun `python scripts/ablate_passes.py --backend cuda --iters 100 --warmup 50
--repeats 5` to regenerate the timed table for those same passes.

## How to run

Prerequisites: Python with `numpy`, `scipy`, `pytest`; CMake; a C++ compiler. CUDA is optional.

```bash
make setup
make matrices
make build
make test
make verify
make tune
make report
```

On a CUDA machine, the GPU suites run automatically:

```bash
python scripts/compare_cusparse.py --budget 40 --iters 100
python scripts/ablate_passes.py --iters 100 --warmup 50
```

Without a GPU, the CUDA tests skip cleanly.

## Limitations

- Results are from a Tesla T4; they are not a portability claim.
- TileGen loses to cuSPARSE on `power_law` and `dense_rows`. Both have long, high-imbalance
  rows; TileGen's best splitters leave each lane with a long serial segment and pay the
  segmented-shuffle reduction on every row, while cuSPARSE rebalances work across rows and uses
  wider memory accesses.
- `canonicalize` and `dead_code_elim` find nothing to do on the standard kernels,
  `loop_fusion` only fires on `fused_spmv_axpy`, and `vectorize` is a no-op for SpMV's gather.
  `tiling`, `thread_mapping`, and `unroll` do change emitted code, but the CPU ablation shows
  that CUDA-tuned tile sizes can hurt OpenMP code; tile sizes should be tuned per backend.
- The IR is Python-level and schedules are flat transform lists; there is no shared-memory
  staging node or typed MLIR dialect yet, and the loop split is represented by a `Tile`
  transform rather than a general affine loop nest.
- Code generation covers CSR and dense; CSC, COO, ELL, and BSR are validated but not lowered.
- The autotuner is brute-force/random or one-sweep coordinate descent, not a learned cost model.

## What I learned / what surprised me

- The first pass ablation was mostly measurement noise: several pass variants emitted identical
  CUDA, so 0.8x/1.2x "contributions" were just clock drift. Diffing the emitted code before
  timing is the fix, and it also exposed that the code generator had been ignoring the schedule.
- `loop_fusion` is a correctness pass, not a speed pass. Turn it off for `fused_spmv_axpy` and
  the kernel writes `t` instead of `z`.
- Autotuning, not pass toggling, did the real work on irregular rows: 4.45x over the
  fixed default on `dense_rows` and 2.78x on `power_law`.
- I did not expect TileGen to beat cuSPARSE on `banded`, `uniform_random`,
  `diagonal_heavy`, and `blocked`, while losing on `power_law` and `dense_rows`.
- The default one-thread-per-row schedule was especially bad on high-variance rows; that
  is the clearest argument for keeping the autotuner.

## License

MIT — see [LICENSE](LICENSE). Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
