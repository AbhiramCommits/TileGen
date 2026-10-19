# TileGen

TileGen is a mini domain-specific language, compiler, and code generator for sparse linear
algebra. You write a kernel in index notation plus a schedule; TileGen lowers it through a
TACO-style intermediate representation and a seven-pass optimization pipeline (canonicalization,
loop fusion, tiling, thread mapping, unrolling, vectorization, dead-code elimination), emits a
specialized CUDA kernel or an OpenMP C++ kernel, differentially verifies every generated kernel
against a reference implementation in float64, and autotunes the schedule per matrix class. A
CUDA backend is emitted and text-checked even when no GPU is present; all numerical and timing
results in this README were produced by the **CPU backend on this machine** (no NVIDIA GPU or
`nvcc` was available — see [Results](#5-results-from-this-build)).

## 1. A real DSL sample and the IR it produces

```python
from tilegen.dsl import Index, CSR, sparse_tensor, dense_tensor, scalar_tensor, \
                        reduce_sum, assign, kernel, standard_kernels

i, j = Index("i"), Index("j")
A = sparse_tensor("A", CSR, shape=("M", "N"))
x = dense_tensor("x", shape=("N",))
y = dense_tensor("y", shape=("M",))
spmv = kernel("spmv", [assign(y[i], reduce_sum(j, A[i, j] * x[j]))])

# the FUSION test case: two statements that share the index i
t     = dense_tensor("t", shape=("M",))
z     = dense_tensor("z", shape=("M",))
w     = dense_tensor("w", shape=("M",))
alpha = scalar_tensor("alpha")
beta  = scalar_tensor("beta")
fused = kernel("fused_spmv_axpy", [
    assign(t[i], reduce_sum(j, A[i, j] * x[j])),
    assign(z[i], alpha * t[i] + beta * w[i]),
])
```

`Kernel.pretty()` renders the IR as a loop nest:

```
kernel spmv {
  assign y[i] = sum(j, (A[i, j] * x[j]))
  // loop nest
  for i:
    for j in reduce(A row):
      assign y[i] = sum(j, (A[i, j] * x[j]))
}
```

Here is the **actually emitted CUDA** for `spmv` with `threads_per_row=4` (copied verbatim from
`build/generated/spmv_*.cu`; the header comment lists the applied schedule and the pass log):

```cuda
__global__ void spmv_kernel(const int64_t* indptr, const int64_t* indices, const double* data, const double* x, double* y, int64_t M, int64_t N, int64_t nnz) {
  int64_t TILE = 32;
  const int TPR = 4;
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

## 2. Architecture

```
  DSL            IR            passes                       codegen            verify        autotune   analyze
 dsl.py  --->  ir.py  --->  passes.py  ---------------->  codegen.py  ---->  verify.py  --> autotune.py --> analyze.py
 (math)       (dataclasses  (7 passes, in order:          (.cu / .cpp)       (ctypes +      (search +     (REPORT.md,
              + validate)   canonicalize, loop_fusion,                       scipy ref)     timing)       summaries)
                            tiling, thread_mapping,
                            unroll, vectorize, DCE)
                                  |
                                  v
                     C++/CUDA host harness (cpp/): ref.cpp reference kernels,
                     harness.cpp dlopen+timing+JSON, Catch2 tests (test_ref.cpp)
```

File map:

| path | role |
|---|---|
| `tilegen/env.py` | backend probing (`probe()` -> has_nvcc/has_gpu/backend/...) |
| `tilegen/matrices.py` | deterministic 6-class CSR corpus + `.npz`/`.bin` exporters |
| `tilegen/ir.py` | `Format`, `TensorDecl`, `Access/Mul/Add/Reduce`, `Stmt`, `Schedule`, `Kernel.validate()` |
| `tilegen/dsl.py` | operator-overloaded frontend, `reduce_sum`, `assign`, `standard_kernels()` |
| `tilegen/passes.py` | `PASSES` list and `run_pipeline`/`optimize` |
| `tilegen/codegen.py` | `Config`, `emit_cuda`, `emit_cpu`, `generate`, `--inject-bug` |
| `tilegen/verify.py` | `compile_generated`, `verify`, CUDA text-level checks |
| `tilegen/autotune.py` | search space, pruning, `tune`, output writers |
| `cpp/` | reference kernels, host harness, Catch2 tests, optional CUDA |
| `scripts/` | `probe_env`, `gen_demo`, `verify_all`, `autotune`, `analyze` |

## 3. How to run

Prerequisites: Python >= 3.9 with `numpy`, `scipy`, `pytest`; CMake >= 3.16 and a C++17 compiler.
CUDA and an NVIDIA GPU are optional.

```bash
make setup          # pip install -e ".[dev]" (falls back gracefully)
make matrices       # generate data/matrices/*.npz + manifest.json
make build          # cmake -S cpp -B build/cpp && cmake --build build/cpp -j
make test           # Catch2 tilegen_tests + pytest
make verify         # differential verification sweep -> results/correctness.json
make tune           # compact summary -> results/; full sweep -> build/autotune.{json,csv}
make report         # results/REPORT.md + correctness_summary.md
make all BUDGET=6   # full pipeline end-to-end with a small budget (CI uses this)
make clean
```

To exercise the **real CUDA path** on a GPU machine: ensure `nvcc` is on `PATH` and
`nvidia-smi` reports a device, then run `make all`. `tilegen.env.probe()` selects the `cuda`
backend, the harness compiles `.cu` files with `nvcc -O3 -arch=sm_<detected>`, and timing uses
`cudaEvent`; the emitted sources already contain the `__global__`, `#pragma unroll` and
`__shfl_down_sync` markers regardless of whether a GPU is present.

## 4. Matrix corpus

Generated deterministically with seed 1234 (`data/matrices/manifest.json`):

| class | rows | nnz | mean nnz/row | max nnz/row | imbalance (max/mean) |
|---|---|---|---|---|---|
| banded | 200000 | 1399988 | 7.0 | 7 | 1.00 |
| uniform_random | 100000 | 1599890 | 16.0 | 16 | 1.00 |
| power_law | 100000 | 804454 | 8.0 | 1990 | 247.37 |
| diagonal_heavy | 200000 | 599998 | 3.0 | 3 | 1.00 |
| dense_rows | 20000 | 460083 | 23.0 | 1923 | 83.59 |
| blocked | 100000 | 1600000 | 16.0 | 16 | 1.00 |

## 5. Results from this build

**Environment probe** (`python scripts/probe_env.py`, also `results/correctness.json -> hardware`):

```json
{
  "has_nvcc": false, "nvcc_version": null,
  "has_gpu": false, "gpu_name": null, "sm_arch": null,
  "cpu_name": "Apple M3", "num_cores": 8,
  "backend": "cpu"
}
```

**Which backend produced the numbers:** the `cpu` backend. No NVIDIA GPU or `nvcc` was available,
so no GPU timings exist and none are reported. The CUDA sources were still emitted and validated
with the text-level structural check (`tilegen.verify.syntax_check_cuda_text`), which is logged
honestly in `build/autotune.json` (the full sweep; `results/autotune.json` keeps the
compact best-per-class and top-5 summary) / the verification messages.

**Scale of the build (all traceable to `results/`):**

- kernels expressible in the DSL: **5** (`spmv`, `spmm`, `sddmm`, `fused_spmv_axpy`, `axpy`).
- `scripts/gen_demo.py` emitted **31** source files (15 `.cu` + 16 `.cpp`).
- the autotuner generated and compiled **200 unique (kernel, config)** variants, from **1200**
  variant attempts across the (kernel, matrix-class) pairs, with **180** illegal combinations
  pruned and logged.
- differential verification ran **96** triples: **90/90** normal triples matched the reference
  within tolerance, and **6/6** injected-bug variants were flagged. Measured errors over the
  normal triples: **max abs error 5.68e-14**, **max rel error 2.53e-10**
  (`results/correctness.json`).
- C++ reference suite: **22 assertions in 8 Catch2 test cases** (`./build/cpp/tilegen_tests`).

**Winning configuration per (kernel, matrix class)** — straight from `results/REPORT.md`
(config = Tile/ThreadsPerRow/Unroll/VectorWidth/BlockSize; speedup vs the naive baseline
T32/TPR1/U1/V1/B128):

| kernel | matrix_class | config | ms_median | speedup | GFLOP/s | GB/s |
|---|---|---|---|---|---|---|
| spmv | banded | T128/TPR1/U4/V4/B512 | 0.294792 | 1.15x | 10.54 | 138.5 |
| spmm | banded | T32/TPR32/U1/V1/B256 | 1.020583 | 1.08x | 26.28 | 148.3 |
| sddmm | banded | T32/TPR32/U1/V1/B256 | 0.859125 | 1.01x | 29.13 | 278.8 |
| fused_spmv_axpy | banded | T32/TPR32/U8/V1/B128 | 0.296042 | 1.07x | 10.87 | 149.0 |
| axpy | banded | T128/TPR2/U2/V2/B256 | 0.046041 | 1.00x | 9.99 | 119.9 |
| spmv | uniform_random | T32/TPR1/U1/V1/B128 | 0.466042 | 1.00x | 8.54 | 106.7 |
| spmm | uniform_random | T32/TPR32/U1/V1/B256 | 2.362875 | 1.10x | 12.38 | 65.4 |
| sddmm | uniform_random | T32/TPR32/U8/V1/B128 | 1.923875 | 0.94x | 16.57 | 157.9 |
| fused_spmv_axpy | uniform_random | T32/TPR1/U8/V1/B128 | 0.503500 | 1.04x | 7.02 | 89.5 |
| axpy | uniform_random | T128/TPR2/U4/V2/B256 | 0.032583 | 1.50x | 6.79 | 81.5 |
| spmv | power_law | T32/TPR32/U4/V1/B128 | 0.318084 | 0.97x | 5.43 | 70.6 |
| spmm | power_law | T64/TPR8/U1/V1/B128 | 1.676833 | 1.46x | 12.26 | 68.2 |
| sddmm | power_law | T64/TPR2/U8/V2/B512 | 1.005958 | 1.06x | 16.02 | 153.2 |
| fused_spmv_axpy | power_law | T32/TPR32/U8/V2/B512 | 0.332750 | 1.09x | 5.28 | 71.3 |
| axpy | power_law | T128/TPR1/U4/V4/B512 | 0.040416 | 1.55x | 5.83 | 69.9 |
| spmv | diagonal_heavy | T64/TPR32/U2/V1/B512 | 0.142209 | 1.23x | 10.68 | 156.6 |
| spmm | diagonal_heavy | T64/TPR8/U1/V1/B128 | 0.578500 | 1.06x | 18.16 | 118.0 |
| sddmm | diagonal_heavy | T32/TPR8/U8/V1/B512 | 0.506208 | 1.15x | 22.87 | 221.1 |
| fused_spmv_axpy | diagonal_heavy | T64/TPR8/U1/V1/B128 | 0.146334 | 1.29x | 9.32 | 149.1 |
| axpy | diagonal_heavy | T64/TPR4/U2/V2/B128 | 0.045875 | 0.99x | 9.62 | 115.4 |
| spmv | dense_rows | T32/TPR32/U4/V1/B128 | 0.253792 | 1.13x | 3.73 | 46.0 |
| spmm | dense_rows | T32/TPR1/U1/V1/B128 | 0.882917 | 1.00x | 9.24 | 48.0 |
| sddmm | dense_rows | T32/TPR4/U4/V4/B128 | 0.534250 | 1.06x | 15.56 | 148.1 |
| fused_spmv_axpy | dense_rows | T32/TPR32/U8/V1/B128 | 0.256125 | 1.04x | 3.66 | 45.9 |
| axpy | dense_rows | T32/TPR4/U2/V1/B512 | 0.025583 | 1.27x | 1.71 | 20.6 |
| spmv | blocked | T32/TPR32/U8/V2/B512 | 0.317416 | 1.04x | 11.06 | 138.3 |
| spmm | blocked | T32/TPR1/U1/V1/B128 | 1.490459 | 1.00x | 20.28 | 107.1 |
| sddmm | blocked | T32/TPR4/U8/V1/B512 | 1.102500 | 1.21x | 29.86 | 284.6 |
| fused_spmv_axpy | blocked | T32/TPR32/U8/V2/B512 | 0.333208 | 1.15x | 10.96 | 139.7 |
| axpy | blocked | T64/TPR4/U8/V1/B128 | 0.036459 | 1.23x | 6.39 | 76.7 |

Observed pattern (auto-generated, `results/REPORT.md`): high-variance classes (`power_law`,
`dense_rows`) and the uniform classes all benefit from `threads_per_row > 1`, because segmenting a
long row across a warp lets the `__shfl_down_sync` reduction hide memory latency; widening the
dense inner axis helps where the right-hand operand is contiguous; unroll factors cluster around
2–4.

**Fusion result** (fused `z = alpha*(A*x) + beta*w` vs running `spmv` then `axpy`), from
`results/REPORT.md`:

| matrix_class | unfused (spmv+axpy) ms | fused ms | fusion speedup |
|---|---|---|---|
| banded | 0.340833 | 0.296042 | 1.15x |
| blocked | 0.353875 | 0.333208 | 1.06x |
| dense_rows | 0.279375 | 0.256125 | 1.09x |
| diagonal_heavy | 0.188084 | 0.146334 | 1.29x |
| power_law | 0.358500 | 0.332750 | 1.08x |
| uniform_random | 0.498625 | 0.503500 | 0.99x |

Mean fusion speedup over the corpus: **1.11x**.

**Injected-bug detection:** `codegen` accepts `inject_bug=True`, which emits an off-by-one row
bound (`i < M - 1`). The differential tester flagged all **6/6** such variants (e.g. `spmv` on
`diagonal_heavy`: max abs error `2.26` vs tolerance `~1e-13`). This proves the harness detects
wrong kernels rather than rubber-stamping them.

## 6. Limitations and future work

- **IR:** a Python-level TACO-style IR. A real MLIR dialect would give typed passes, verifiers,
  and a shared lowering infrastructure.
- **Tile abstractions:** schedules are flat transform lists. cuTILE/Tile++-style tile
  abstractions with explicit data movement would make shared-memory staging first-class.
- **Format coverage:** code generation is fully implemented for CSR (and dense); CSC, COO, ELL and
  BSR are declarable and validated but not yet lowered by `codegen`.
- **Search:** the autotuner is brute-force/random or one-sweep coordinate descent with a variant
  budget. A learned cost model (e.g. regressing runtime on row-length statistics) would find good
  schedules in far fewer measurements.
- **CUDA numerics:** in this environment the CUDA path is code-generated and text-checked only; its
  numerical and performance results are not claimed.

## 7. Numerical notes

- **Accumulation order.** Floating-point addition is not associative. The reference sums a row in
  index order; the generated kernel may unroll, reorder, or use a segmented warp reduction, so the
  two differ by rounding noise proportional to the number of terms in the dot product.
- **Tolerance derivation.** For a length-`L` dot product the error is bounded by roughly
  `L * eps * magnitude` with `eps = 2.22e-16` for float64. TileGen therefore accepts
  `tol = 8 * eps * max_row_nnz * max|reference| + 1e-12` (the `1e-12` floor covers all-zero
  outputs). This is implemented in `tilegen/verify.py` and `tilegen/autotune.py`.
- **Empty rows** produce exactly zero (`y[i] = 0`) and are covered by a Catch2 test and a pytest
  corner case. **All-zero data** likewise produces zero, with measured error exactly `0.0`.
- **Denormals / very small magnitudes.** The relative-error denominator is floored at `1e-300` to
  avoid division blow-ups; comparisons are made against the absolute tolerance above, which is the
  numerically meaningful test for near-zero results.

## License

MIT — see [LICENSE](LICENSE). Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
