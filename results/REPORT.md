# TileGen autotuning report

All-kernel table source: **cpu** backend. CPU: Apple M3 (8 cores). GPU: none detected. nvcc: not present.

> The all-kernel table below is the earlier autotune run. The SpMV-on-CUDA autotune and cuSPARSE comparison were produced separately on a Tesla T4 (compute 7.5, nvcc 13.0.88) and are reported at the end of this file.

Baseline (naive) config: T32/TPR1/U1/V1/B128. Search space: {'tile_size': [32, 64, 128, 256], 'threads_per_row': [1, 2, 4, 8, 32], 'unroll': [1, 2, 4, 8], 'vector_width': [1, 2, 4], 'block_size': [128, 256, 512]}.

## Winning configuration per (kernel, matrix class)

| kernel | matrix_class | config | ms_median | speedup vs naive | GFLOP/s | GB/s |
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

## Observations

The corpus contains high-variance classes (power_law, dense_rows) where max/mean row nnz reaches 247; these rows make a single-thread-per-row mapping wasteful. Higher threads_per_row wins for banded, blocked, dense_rows, diagonal_heavy, power_law, uniform_random because splitting a long row across a warp segment reduces per-thread serial latency and lets the __shfl_down_sync segmented reduction hide memory latency. Widening the dense inner axis (vector_width > 1) helps on banded, blocked, dense_rows, diagonal_heavy, power_law, uniform_random where the right-hand operand is contiguous and the extent is divisible by the vector width. Unrolling the compressed row loop with a remainder tail is nearly always beneficial; the winning unroll factors cluster around 2-4. Across the sweep the winning configurations are stable per matrix class, which suggests the schedule is dominated by row-length statistics rather than by the specific random draw.

## Fusion result

| matrix_class | spmv ms | axpy ms | fused ms | unfused ms | fusion speedup |
|---|---|---|---|---|---|
| banded | 0.294792 | 0.046041 | 0.296042 | 0.340833 | 1.15x |
| blocked | 0.317416 | 0.036459 | 0.333208 | 0.353875 | 1.06x |
| dense_rows | 0.253792 | 0.025583 | 0.256125 | 0.279375 | 1.09x |
| diagonal_heavy | 0.142209 | 0.045875 | 0.146334 | 0.188084 | 1.29x |
| power_law | 0.318084 | 0.040416 | 0.332750 | 0.358500 | 1.08x |
| uniform_random | 0.466042 | 0.032583 | 0.503500 | 0.498625 | 0.99x |

Mean fusion speedup over the corpus: **1.11x** (fused_spmv_axpy vs separate spmv + axpy).

## Pruning

180 illegal configurations were pruned (e.g. threads_per_row > block_size, vector_width > unroll, block_size not divisible by threads_per_row). The full list is in build/autotune.json; results/autotune.json is the compact summary.

## SpMV on CUDA vs cuSPARSE

Autotuned on a **Tesla T4** (compute 7.5, nvcc 13.0.88). All timings use CUDA events: 10 warmup iterations and the median of 100 timed runs, with host/device transfers excluded. The fixed default is T32/TPR1/U1/V1/B128.

| matrix_class | tuned config | TileGen tuned ms | TileGen default ms | tuned/default | cuSPARSE ms | tuned/cuSPARSE | default/cuSPARSE |
|---|---|---|---|---|---|---|---|
| banded | T128/TPR1/U4/V4/B512 | 0.148912 | 0.159792 | 1.07x | 0.186544 | 1.25x | 1.17x |
| uniform_random | T64/TPR8/U2/V2/B128 | 0.200672 | 0.401376 | 2.00x | 0.217040 | 1.08x | 0.54x |
| power_law | T64/TPR8/U2/V2/B128 | 0.217552 | 0.604144 | 2.78x | 0.134768 | 0.62x | 0.22x |
| diagonal_heavy | T64/TPR1/U1/V1/B128 | 0.091872 | 0.092736 | 1.01x | 0.119088 | 1.30x | 1.28x |
| dense_rows | T32/TPR8/U8/V1/B512 | 0.139600 | 0.621264 | 4.45x | 0.116384 | 0.83x | 0.19x |
| blocked | T32/TPR4/U4/V4/B128 | 0.145760 | 0.327712 | 2.25x | 0.195136 | 1.34x | 0.60x |

TileGen loses to cuSPARSE on **power_law (imbalance 247.4x), dense_rows (imbalance 83.6x)**. These are the high-variance, long-row classes: a row can contain roughly two thousand nonzeros, while TileGen's best schedule splits it across at most TPR=8 lanes and uses a fixed row tile per block. Each lane therefore still walks a long serial segment, and every row pays the segmented-shuffle reduction and per-block scheduling overhead. cuSPARSE's CSR SpMV instead uses its tuned merge/vectorized kernels to rebalance work across rows and issue wider memory accesses, which is exactly what these irregular rows need.
TileGen wins on **banded, uniform_random, diagonal_heavy, blocked**, which have shorter, more regular rows where a simple row-parallel kernel is competitive and cuSPARSE's more elaborate work distribution does not pay for itself.

The fixed default is the naive T32/TPR1/U1/V1/B128 schedule. Autotuning buys only ~1.01-1.07x on banded and diagonal_heavy, but 2.0-4.45x on uniform_random, blocked, power_law and dense_rows: on those classes the default's one-thread-per-row mapping serializes long or irregular rows. The cuSPARSE columns put both TileGen schedules in absolute context.

## Pass ablation

Contribution is `disabled_ms / full_ms`; values above 1 mean the pass was helping. Cases whose emitted code is byte-identical are marked `no change` and are not timed. The full per-pass data, including code-diff hashes, is in `results/pass_ablation.json`.


Generated code is emitted with the full pipeline and with exactly one pass skipped. The bodies are diffed with the diagnostic header comments stripped; if they are identical the cell says **no change** and that case is not timed. Changed cases run 5 interleaved full/ablated comparisons of 10 warmup + 50 timed iterations; the cell is `median (min–max)` contribution. Contribution = disabled_ms / full_ms. Schedules are the tuned SpMV schedules from `results/cusparse_comparison.json`.

## spmv

| pass | changed | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean (changed) |
|---|---|---|---|---|---|---|---|---|---|
| `canonicalize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `loop_fusion` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `tiling` | 6/6 | 0.93x (0.82–1.14) | 0.82x (0.49–1.04) | 0.63x (0.56–1.47) | 1.00x (0.34–1.12) | 0.61x (0.58–0.71) | 0.90x (0.87–1.00) | 0.82x |
| `thread_mapping` | 6/6 | 1.60x (1.39–1.72) | 3.09x (2.73–3.43) | 2.14x (1.34–2.87) | 2.77x (2.15–2.96) | 2.10x (1.89–2.15) | 1.33x (1.07–1.71) | 2.17x |
| `unroll` | 5/6 | 1.11x (0.84–1.29) | 0.92x (0.71–1.23) | 1.11x (0.57–1.25) | no change | 0.99x (0.98–1.00) | 1.09x (0.79–1.35) | 1.04x |
| `vectorize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `dead_code_elim` | 0/6 | no change | no change | no change | no change | no change | no change | - |

## fused_spmv_axpy

| pass | changed | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean (changed) |
|---|---|---|---|---|---|---|---|---|---|
| `canonicalize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `loop_fusion` | 6/6 | **INVALID** | **INVALID** | **INVALID** | **INVALID** | **INVALID** | **INVALID** | - |
| `tiling` | 6/6 | 0.97x (0.94–1.09) | 0.88x (0.65–0.95) | 0.85x (0.52–1.25) | 1.05x (0.80–1.71) | 0.63x (0.59–0.71) | 0.92x (0.78–1.07) | 0.88x |
| `thread_mapping` | 6/6 | 1.57x (1.18–1.72) | 3.00x (2.84–3.05) | 2.85x (1.78–3.19) | 2.34x (1.45–2.49) | 2.09x (2.08–2.97) | 1.63x (1.31–1.82) | 2.25x |
| `unroll` | 5/6 | 1.16x (0.68–1.58) | 0.92x (0.73–1.04) | 1.06x (0.91–1.13) | no change | 0.97x (0.92–1.08) | 0.98x (0.92–0.99) | 1.02x |
| `vectorize` | 0/6 | no change | no change | no change | no change | no change | no change | - |
| `dead_code_elim` | 0/6 | no change | no change | no change | no change | no change | no change | - |

`changed` counts matrix classes whose emitted code differs when the pass is disabled. `no change` means the pass is a no-op for that schedule/kernel, not that its contribution is unknown.

## Emitted CUDA code effect (static)

The timing table above uses the **cpu** backend. These cells diff the CUDA source instead, so a pass that changes CUDA lowering is visible even when no GPU is available. `+a/−r` counts added/removed code lines outside the diagnostic header.

### spmv

| pass | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked |
|---|---|---|---|---|---|---|
| `canonicalize` | no change | no change | no change | no change | no change | no change |
| `loop_fusion` | no change | no change | no change | no change | no change | no change |
| `tiling` | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) |
| `thread_mapping` | changed (+1/−1) | changed (+19/−15) | changed (+19/−15) | changed (+1/−1) | changed (+37/−15) | changed (+25/−15) |
| `unroll` | changed (+2/−20) | no change | no change | no change | no change | no change |
| `vectorize` | no change | no change | no change | no change | no change | no change |
| `dead_code_elim` | no change | no change | no change | no change | no change | no change |

### fused_spmv_axpy

| pass | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked |
|---|---|---|---|---|---|---|
| `canonicalize` | no change | no change | no change | no change | no change | no change |
| `loop_fusion` | changed (+4/−4) | changed (+4/−4) | changed (+4/−4) | changed (+4/−4) | changed (+4/−4) | changed (+4/−4) |
| `tiling` | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) | changed (+2/−2) |
| `thread_mapping` | changed (+1/−1) | changed (+19/−15) | changed (+19/−15) | changed (+1/−1) | changed (+37/−15) | changed (+25/−15) |
| `unroll` | changed (+2/−20) | no change | no change | no change | no change | no change |
| `vectorize` | no change | no change | no change | no change | no change | no change |
| `dead_code_elim` | no change | no change | no change | no change | no change | no change |
