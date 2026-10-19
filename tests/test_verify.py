"""Tests for the differential verification harness."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from tilegen.codegen import Config
from tilegen.verify import check_cuda_markers, syntax_check_cuda_text, verify
from tilegen.codegen import emit_cuda
from tilegen import passes


def _csr(rows, cols, indptr, indices, data):
    return sp.csr_matrix((np.asarray(data, dtype=np.float64),
                          np.asarray(indices, dtype=np.int64),
                          np.asarray(indptr, dtype=np.int64)),
                         shape=(rows, cols))


def test_injected_bug_is_caught(tmp_path):
    A = _csr(4, 4, [0, 2, 3, 5, 6], [0, 1, 2, 1, 2, 3],
             [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    good = verify("spmv", Config(unroll=1), A, backend="cpu", workdir=str(tmp_path))
    assert good.ok, good.message
    bad = verify("spmv", Config(unroll=1, inject_bug=True), A,
                 backend="cpu", workdir=str(tmp_path))
    assert not bad.ok
    assert bad.max_abs_err > bad.tolerance


def test_empty_row_corner_case(tmp_path):
    # middle row has no entries
    A = _csr(3, 3, [0, 1, 1, 3], [0, 1, 2], [2.0, 4.0, 5.0])
    r = verify("spmv", Config(unroll=2), A, backend="cpu", workdir=str(tmp_path))
    assert r.ok, r.message


def test_all_zero_data_corner_case(tmp_path):
    A = _csr(2, 2, [0, 2, 3], [0, 1, 1], [0.0, 0.0, 0.0])
    r = verify("spmv", Config(unroll=4), A, backend="cpu", workdir=str(tmp_path))
    assert r.ok
    assert r.max_abs_err == 0.0


def test_sddmm_and_fused_verify(tmp_path):
    A = _csr(3, 3, [0, 2, 3, 3], [0, 1, 2], [1.0, 2.0, 3.0])
    for k in ("spmm", "sddmm", "fused_spmv_axpy", "axpy"):
        r = verify(k, Config(unroll=1), A, backend="cpu", workdir=str(tmp_path))
        assert r.ok, "%s: %s" % (k, r.message)


def test_cuda_text_check_and_markers(tmp_path):
    from tilegen.dsl import standard_kernels
    cfg = Config(tile_size=32, threads_per_row=4, unroll=4)
    k = passes.run_pipeline(standard_kernels()["spmv"], cfg, backend="cuda")
    src = emit_cuda(k, cfg)
    ok, notes = syntax_check_cuda_text(src)
    assert ok, notes
    markers = check_cuda_markers(src, threads_per_row=4)
    assert markers["__global__"]
    assert markers["__shfl_down_sync"]
    assert markers["#pragma unroll"]


def test_cuda_launch_grid_covers_every_row():
    from tilegen.dsl import standard_kernels
    cfg = Config(tile_size=32, threads_per_row=4, unroll=4, block_size=256)
    spmv = passes.run_pipeline(standard_kernels()["spmv"], cfg, backend="cuda")
    src = emit_cuda(spmv, cfg)
    # Every block covers TILE rows, not blockDim rows.
    assert "int64_t blocks = (M + 32 - 1) / 32;" in src

    cfg1 = Config(tile_size=1, threads_per_row=4, unroll=1, block_size=128)
    sddmm = passes.run_pipeline(standard_kernels()["sddmm"], cfg1, backend="cuda")
    src1 = emit_cuda(sddmm, cfg1)
    # sddmm is always block-per-row, even when tiling is disabled.
    assert "int64_t blocks = M;  // block-per-row mapping" in src1


def test_cuda_unroll_pragma_precedes_the_loop():
    from tilegen.dsl import standard_kernels
    cfg = Config(tile_size=32, threads_per_row=1, unroll=4, block_size=256)
    spmv = passes.run_pipeline(standard_kernels()["spmv"], cfg, backend="cuda")
    src = emit_cuda(spmv, cfg)
    lines = src.splitlines()
    idx = lines.index("      #pragma unroll")
    assert "for (; q + 4 <= pend; q += 4)" in lines[idx + 1]
    assert "int64_t q = p;" in lines[idx - 1]
