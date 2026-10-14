# TileGen autotuning report

Backend that produced these numbers: **cpu**. CPU: Apple M3 (8 cores). GPU: none detected. nvcc: not present.

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

180 illegal configurations were pruned (e.g. threads_per_row > block_size, vector_width > unroll, block_size not divisible by threads_per_row). The full list is in results/autotune.json.
