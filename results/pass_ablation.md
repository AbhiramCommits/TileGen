# TileGen pass ablation (CUDA)

CUDA-event timing: 50 warmup + 100 timed runs, median; transfers excluded. Contribution = disabled_ms / full_ms; >1 means the pass was helping. Schedules are the tuned SpMV schedules from `results/cusparse_comparison.json`.

## spmv

| pass | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean |
|---|---|---|---|---|---|---|---|
| `canonicalize` | 1.09x | 1.00x | 0.99x | 0.99x | 0.99x | 1.02x | 1.01x |
| `loop_fusion` | 1.00x | 0.99x | 0.98x | 0.98x | 0.84x | 1.00x | 0.97x |
| `tiling` | 0.99x | 1.00x | 1.01x | 1.02x | 0.99x | 1.18x | 1.03x |
| `thread_mapping` | 0.99x | 0.99x | 1.01x | 1.03x | 0.99x | 1.02x | 1.01x |
| `unroll` | 0.99x | 1.00x | 0.99x | 0.98x | 1.00x | 0.99x | 0.99x |
| `vectorize` | 1.01x | 1.00x | 0.99x | 1.02x | 1.02x | 0.99x | 1.01x |
| `dead_code_elim` | 1.00x | 1.00x | 1.01x | 1.00x | 1.01x | 1.00x | 1.00x |

## fused_spmv_axpy

| pass | banded | uniform_random | power_law | diagonal_heavy | dense_rows | blocked | mean |
|---|---|---|---|---|---|---|---|
| `canonicalize` | 0.99x | 1.00x | 1.01x | 1.02x | 1.03x | 0.99x | 1.01x |
| `loop_fusion` | **INVALID** | **INVALID** | **INVALID** | **INVALID** | **INVALID** | **INVALID** | - |
| `tiling` | 1.00x | 1.00x | 0.99x | 1.00x | 1.01x | 1.00x | 1.00x |
| `thread_mapping` | 1.02x | 1.00x | 1.10x | 0.99x | 1.03x | 0.82x | 0.99x |
| `unroll` | 1.01x | 1.02x | 0.99x | 1.02x | 1.01x | 1.00x | 1.01x |
| `vectorize` | 1.02x | 1.01x | 0.95x | 0.96x | 1.01x | 1.01x | 0.99x |
| `dead_code_elim` | 0.99x | 1.01x | 1.00x | 1.00x | 0.81x | 1.00x | 0.97x |
