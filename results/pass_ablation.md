# TileGen pass ablation (CPU)

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
