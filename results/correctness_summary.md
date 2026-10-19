# TileGen differential correctness summary

Backend for numerics: **cuda** (reference computed in float64 with numpy/scipy).

- triples checked: 96 (normal 90, injected-bug 6)
- normal triples passing tolerance: 90 / 90
- injected-bug variants correctly flagged: 6 / 6
- max absolute error (normal): 3.411e-13
- max relative error (normal): 4.993e-10

| matrix_class | kernel | ok | max_abs_err | max_rel_err | tolerance | bug |
|---|---|---|---|---|---|---|
| banded | spmv | PASS | 1.066e-14 | 1.022e-10 | 1.636e-12 |  |
| banded | spmv | PASS | 1.066e-14 | 1.022e-10 | 1.636e-12 |  |
| banded | spmv | PASS | 1.421e-14 | 4.692e-11 | 1.636e-12 |  |
| banded | spmm | PASS | 1.421e-14 | 3.389e-12 | 1.729e-12 |  |
| banded | spmm | PASS | 1.421e-14 | 3.389e-12 | 1.729e-12 |  |
| banded | spmm | PASS | 1.421e-14 | 3.389e-12 | 1.729e-12 |  |
| banded | sddmm | PASS | 4.263e-14 | 7.064e-11 | 3.229e-12 |  |
| banded | sddmm | PASS | 4.263e-14 | 7.064e-11 | 3.229e-12 |  |
| banded | sddmm | PASS | 4.263e-14 | 7.064e-11 | 3.229e-12 |  |
| banded | fused_spmv_axpy | PASS | 8.882e-15 | 2.446e-12 | 1.451e-12 |  |
| banded | fused_spmv_axpy | PASS | 8.882e-15 | 2.446e-12 | 1.451e-12 |  |
| banded | fused_spmv_axpy | PASS | 1.421e-14 | 9.052e-13 | 1.451e-12 |  |
| banded | axpy | PASS | 8.882e-16 | 3.021e-12 | 1.080e-12 |  |
| banded | axpy | PASS | 8.882e-16 | 3.021e-12 | 1.080e-12 |  |
| banded | axpy | PASS | 8.882e-16 | 3.021e-12 | 1.080e-12 |  |
| banded | spmv | FAIL | 4.796e+00 | 1.000e+00 | 1.636e-12 | yes |
| uniform_random | spmv | PASS | 7.105e-15 | 1.014e-11 | 1.603e-12 |  |
| uniform_random | spmv | PASS | 7.105e-15 | 1.014e-11 | 1.603e-12 |  |
| uniform_random | spmv | PASS | 7.105e-15 | 6.468e-12 | 1.603e-12 |  |
| uniform_random | spmm | PASS | 7.105e-15 | 1.952e-10 | 1.683e-12 |  |
| uniform_random | spmm | PASS | 7.105e-15 | 1.952e-10 | 1.683e-12 |  |
| uniform_random | spmm | PASS | 7.105e-15 | 1.952e-10 | 1.683e-12 |  |
| uniform_random | sddmm | PASS | 1.066e-14 | 2.400e-10 | 2.176e-12 |  |
| uniform_random | sddmm | PASS | 1.066e-14 | 2.400e-10 | 2.176e-12 |  |
| uniform_random | sddmm | PASS | 1.066e-14 | 2.400e-10 | 2.176e-12 |  |
| uniform_random | fused_spmv_axpy | PASS | 4.441e-15 | 8.144e-12 | 1.473e-12 |  |
| uniform_random | fused_spmv_axpy | PASS | 4.441e-15 | 8.144e-12 | 1.473e-12 |  |
| uniform_random | fused_spmv_axpy | PASS | 5.329e-15 | 1.283e-11 | 1.473e-12 |  |
| uniform_random | axpy | PASS | 8.882e-16 | 6.086e-13 | 1.199e-12 |  |
| uniform_random | axpy | PASS | 8.882e-16 | 6.086e-13 | 1.199e-12 |  |
| uniform_random | axpy | PASS | 8.882e-16 | 6.086e-13 | 1.199e-12 |  |
| uniform_random | spmv | FAIL | 5.320e+00 | 1.000e+00 | 1.603e-12 | yes |
| power_law | spmv | PASS | 7.105e-14 | 2.207e-12 | 4.131e-10 |  |
| power_law | spmv | PASS | 7.105e-14 | 2.207e-12 | 4.131e-10 |  |
| power_law | spmv | PASS | 1.350e-13 | 3.386e-12 | 4.131e-10 |  |
| power_law | spmm | PASS | 7.105e-14 | 5.978e-11 | 4.649e-10 |  |
| power_law | spmm | PASS | 7.105e-14 | 5.978e-11 | 4.649e-10 |  |
| power_law | spmm | PASS | 7.105e-14 | 5.978e-11 | 4.649e-10 |  |
| power_law | sddmm | PASS | 1.066e-14 | 2.670e-11 | 1.466e-10 |  |
| power_law | sddmm | PASS | 1.066e-14 | 2.670e-11 | 1.466e-10 |  |
| power_law | sddmm | PASS | 1.066e-14 | 2.670e-11 | 1.466e-10 |  |
| power_law | fused_spmv_axpy | PASS | 5.684e-14 | 1.631e-10 | 2.845e-10 |  |
| power_law | fused_spmv_axpy | PASS | 5.684e-14 | 1.631e-10 | 2.845e-10 |  |
| power_law | fused_spmv_axpy | PASS | 9.948e-14 | 1.157e-10 | 2.845e-10 |  |
| power_law | axpy | PASS | 8.882e-16 | 6.086e-13 | 2.580e-11 |  |
| power_law | axpy | PASS | 8.882e-16 | 6.086e-13 | 2.580e-11 |  |
| power_law | axpy | PASS | 8.882e-16 | 6.086e-13 | 2.580e-11 |  |
| power_law | spmv | FAIL | 6.072e-01 | 1.000e+00 | 4.131e-10 | yes |
| diagonal_heavy | spmv | PASS | 1.421e-14 | 4.524e-13 | 1.294e-12 |  |
| diagonal_heavy | spmv | PASS | 1.421e-14 | 4.524e-13 | 1.294e-12 |  |
| diagonal_heavy | spmv | PASS | 7.105e-15 | 2.284e-12 | 1.294e-12 |  |
| diagonal_heavy | spmm | PASS | 7.105e-15 | 3.107e-11 | 1.303e-12 |  |
| diagonal_heavy | spmm | PASS | 7.105e-15 | 3.107e-11 | 1.303e-12 |  |
| diagonal_heavy | spmm | PASS | 7.105e-15 | 3.107e-11 | 1.303e-12 |  |
| diagonal_heavy | sddmm | PASS | 5.684e-14 | 4.123e-11 | 1.941e-12 |  |
| diagonal_heavy | sddmm | PASS | 5.684e-14 | 4.123e-11 | 1.941e-12 |  |
| diagonal_heavy | sddmm | PASS | 5.684e-14 | 4.123e-11 | 1.941e-12 |  |
| diagonal_heavy | fused_spmv_axpy | PASS | 7.105e-15 | 1.760e-12 | 1.200e-12 |  |
| diagonal_heavy | fused_spmv_axpy | PASS | 7.105e-15 | 1.760e-12 | 1.200e-12 |  |
| diagonal_heavy | fused_spmv_axpy | PASS | 7.105e-15 | 1.760e-12 | 1.200e-12 |  |
| diagonal_heavy | axpy | PASS | 8.882e-16 | 3.021e-12 | 1.034e-12 |  |
| diagonal_heavy | axpy | PASS | 8.882e-16 | 3.021e-12 | 1.034e-12 |  |
| diagonal_heavy | axpy | PASS | 8.882e-16 | 3.021e-12 | 1.034e-12 |  |
| diagonal_heavy | spmv | FAIL | 6.155e+00 | 1.000e+00 | 1.294e-12 | yes |
| dense_rows | spmv | PASS | 8.527e-14 | 7.015e-13 | 4.790e-10 |  |
| dense_rows | spmv | PASS | 8.527e-14 | 7.015e-13 | 4.790e-10 |  |
| dense_rows | spmv | PASS | 3.411e-13 | 1.462e-12 | 4.790e-10 |  |
| dense_rows | spmm | PASS | 8.527e-14 | 4.993e-10 | 6.746e-10 |  |
| dense_rows | spmm | PASS | 8.527e-14 | 4.993e-10 | 6.746e-10 |  |
| dense_rows | spmm | PASS | 8.527e-14 | 4.993e-10 | 6.746e-10 |  |
| dense_rows | sddmm | PASS | 7.105e-15 | 7.785e-12 | 1.231e-10 |  |
| dense_rows | sddmm | PASS | 7.105e-15 | 7.785e-12 | 1.231e-10 |  |
| dense_rows | sddmm | PASS | 7.105e-15 | 7.785e-12 | 1.231e-10 |  |
| dense_rows | fused_spmv_axpy | PASS | 7.105e-14 | 1.067e-12 | 3.298e-10 |  |
| dense_rows | fused_spmv_axpy | PASS | 7.105e-14 | 1.067e-12 | 3.298e-10 |  |
| dense_rows | fused_spmv_axpy | PASS | 2.416e-13 | 3.880e-13 | 3.298e-10 |  |
| dense_rows | axpy | PASS | 8.882e-16 | 1.560e-12 | 2.202e-11 |  |
| dense_rows | axpy | PASS | 8.882e-16 | 1.560e-12 | 2.202e-11 |  |
| dense_rows | axpy | PASS | 8.882e-16 | 1.560e-12 | 2.202e-11 |  |
| dense_rows | spmv | FAIL | 2.260e+00 | 1.000e+00 | 4.790e-10 | yes |
| blocked | spmv | PASS | 5.329e-15 | 2.132e-11 | 1.674e-12 |  |
| blocked | spmv | PASS | 5.329e-15 | 2.132e-11 | 1.674e-12 |  |
| blocked | spmv | PASS | 7.105e-15 | 5.207e-11 | 1.674e-12 |  |
| blocked | spmm | PASS | 7.105e-15 | 4.658e-10 | 1.669e-12 |  |
| blocked | spmm | PASS | 7.105e-15 | 4.658e-10 | 1.669e-12 |  |
| blocked | spmm | PASS | 7.105e-15 | 4.658e-10 | 1.669e-12 |  |
| blocked | sddmm | PASS | 1.066e-14 | 6.347e-11 | 2.260e-12 |  |
| blocked | sddmm | PASS | 1.066e-14 | 6.347e-11 | 2.260e-12 |  |
| blocked | sddmm | PASS | 1.066e-14 | 6.347e-11 | 2.260e-12 |  |
| blocked | fused_spmv_axpy | PASS | 5.329e-15 | 7.648e-12 | 1.541e-12 |  |
| blocked | fused_spmv_axpy | PASS | 5.329e-15 | 7.648e-12 | 1.541e-12 |  |
| blocked | fused_spmv_axpy | PASS | 5.329e-15 | 6.831e-12 | 1.541e-12 |  |
| blocked | axpy | PASS | 8.882e-16 | 6.086e-13 | 1.199e-12 |  |
| blocked | axpy | PASS | 8.882e-16 | 6.086e-13 | 1.199e-12 |  |
| blocked | axpy | PASS | 8.882e-16 | 6.086e-13 | 1.199e-12 |  |
| blocked | spmv | FAIL | 3.808e+00 | 1.000e+00 | 1.674e-12 | yes |