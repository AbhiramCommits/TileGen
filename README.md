# TileGen

TileGen is a small DSL, compiler, and code generator for sparse linear algebra.
It lowers index-notation kernels through a TACO-style IR and pass pipeline into CUDA or OpenMP C++.
It autotunes schedules, differentially verifies against float64 SciPy, and compares with cuSPARSE.

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
  assign y[i] = sum(j, (A[i, j] * x[j]))
  // loop nest
  for i:
    for j in reduce(A row):
      assign y[i] = sum(j, (A[i, j] * x[j]))
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

Pass ablation from `results/pass_ablation.json` (tuned SpMV schedules, CUDA events,
50 warmup + 100 timed runs). Contribution = disabled time / full time; above 1 means the pass
was helping.

| pass | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `canonicalize` | 1.09x | 1.00x | 0.99x | 0.99x | 0.99x | 1.02x | 1.01x |
| `loop_fusion` | 1.00x | 0.99x | 0.98x | 0.98x | 0.84x | 1.00x | 0.97x |
| `tiling` | 0.99x | 1.00x | 1.01x | 1.02x | 0.99x | 1.18x | 1.03x |
| `thread_mapping` | 0.99x | 0.99x | 1.01x | 1.03x | 0.99x | 1.02x | 1.01x |
| `unroll` | 0.99x | 1.00x | 0.99x | 0.98x | 1.00x | 0.99x | 0.99x |
| `vectorize` | 1.01x | 1.00x | 0.99x | 1.02x | 1.02x | 0.99x | 1.01x |
| `dead_code_elim` | 1.00x | 1.00x | 1.01x | 1.00x | 1.01x | 1.00x | 1.00x |

Only `loop_fusion` changes the generated computation for a standard kernel. Disabling it on
`fused_spmv_axpy` leaves the kernel writing temporary `t` instead of `z`, so that ablation is
invalid rather than merely slower. The other passes are listed below as experimental / no
measured impact.

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
- Most passes have no measured impact on the standard kernels. `tiling`, `thread_mapping`,
  `unroll`, and `vectorize` only record schedule metadata; `codegen` reads `Config` directly.
  `canonicalize` and `dead_code_elim` find nothing to do on the standard kernels.
- The IR is Python-level and schedules are flat transform lists; shared-memory staging and a
  typed MLIR dialect would be the natural next step.
- Code generation covers CSR and dense; CSC, COO, ELL, and BSR are validated but not lowered.
- The autotuner is brute-force/random or one-sweep coordinate descent, not a learned cost model.

## What I learned / what surprised me

- TODO: I expected the pass list to be where the performance came from. The ablation showed
  that only `loop_fusion` changes the generated computation; the rest are mostly metadata or
  no-ops on the standard kernels.
- TODO: I was surprised that `loop_fusion` is a correctness pass, not a speed pass. Turn it off
  for `fused_spmv_axpy` and the kernel writes `t` instead of `z`.
- TODO: Autotuning, not pass toggling, did the real work on irregular rows: 4.45x over the
  fixed default on `dense_rows` and 2.78x on `power_law`.
- TODO: I did not expect TileGen to beat cuSPARSE on `banded`, `uniform_random`,
  `diagonal_heavy`, and `blocked`, while losing on `power_law` and `dense_rows`.
- TODO: The default one-thread-per-row schedule was especially bad on high-variance rows; that
  is the clearest argument for keeping the autotuner.

## License

MIT — see [LICENSE](LICENSE). Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
