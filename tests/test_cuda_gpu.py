"""End-to-end CUDA tests: compile with nvcc, run on the GPU, compare to SciPy.

These are skipped unless ``tilegen.env.probe()`` selects the CUDA backend, i.e.
both ``nvcc`` and a visible NVIDIA GPU are present.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tilegen.codegen import Config
from tilegen.env import probe
from tilegen.verify import verify


_ENV = probe()
_CUDA_READY = bool(
    _ENV.get("has_nvcc") and _ENV.get("has_gpu") and _ENV.get("backend") == "cuda"
)

pytestmark = pytest.mark.skipif(
    not _CUDA_READY,
    reason=(
        "CUDA GPU path requires nvcc and a visible GPU "
        "(probe: backend=%s, nvcc=%s, gpu=%s)"
        % (_ENV.get("backend"), _ENV.get("nvcc_version"), _ENV.get("gpu_name"))
    ),
)

KERNELS = ["spmv", "spmm", "sddmm", "fused_spmv_axpy", "axpy"]

# Covers every requested dimension: threads_per_row 1/2/4/8/32, several tile
# sizes, and unrolling off/on (1/2/4/8).
SCHEDULES = [
    Config(tile_size=32, threads_per_row=1, unroll=1, vector_width=1, block_size=128),
    Config(tile_size=32, threads_per_row=2, unroll=2, vector_width=1, block_size=128),
    Config(tile_size=64, threads_per_row=4, unroll=4, vector_width=2, block_size=256),
    Config(tile_size=128, threads_per_row=8, unroll=2, vector_width=2, block_size=512),
    Config(tile_size=32, threads_per_row=32, unroll=8, vector_width=2, block_size=512),
    Config(tile_size=1, threads_per_row=4, unroll=1, vector_width=1, block_size=128),
    Config(tile_size=256, threads_per_row=1, unroll=4, vector_width=1, block_size=256),
]

SCHEDULE_IDS = [
    "T%d-TPR%d-U%d-V%d-B%d"
    % (c.tile_size, c.threads_per_row, c.unroll, c.vector_width, c.block_size)
    for c in SCHEDULES
]


def _cuda_matrix() -> sp.csr_matrix:
    """A 257x257 matrix with mixed row lengths and empty rows.

    The row count is deliberately larger than the tile and block sizes so an
    under-sized launch grid (rows silently left unprocessed) is detectable.
    """
    rng = np.random.default_rng(2024)
    rows = cols = 257
    indptr = [0]
    indices = []
    data = []
    for r in range(rows):
        nnz = 0 if r % 37 == 0 else int(rng.integers(1, 25))
        if nnz:
            row_cols = np.sort(rng.choice(cols, size=nnz, replace=False))
            indices.extend(int(c) for c in row_cols)
            data.extend(float(v) for v in rng.standard_normal(nnz))
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         np.asarray(indices, dtype=np.int64),
         np.asarray(indptr, dtype=np.int64)),
        shape=(rows, cols),
    )


@pytest.fixture(scope="module")
def matrix() -> sp.csr_matrix:
    return _cuda_matrix()


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("cfg", SCHEDULES, ids=SCHEDULE_IDS)
def test_cuda_kernel_matches_float64_reference(kernel, cfg, matrix, tmp_path):
    result = verify(kernel, cfg, matrix, backend="cuda",
                    workdir=str(tmp_path), K=8)
    assert result.ok, "%s %s failed:\n%s" % (kernel, cfg, result.message)
    assert result.backend == "cuda"
    assert "nvcc compiled" in result.message, result.message
    assert result.max_abs_err <= result.tolerance


@pytest.mark.parametrize("kernel", KERNELS)
def test_cuda_injected_bug_is_caught(kernel, matrix, tmp_path):
    cfg = Config(tile_size=32, threads_per_row=4, unroll=2,
                 vector_width=1, block_size=256, inject_bug=True)
    result = verify(kernel, cfg, matrix, backend="cuda",
                    workdir=str(tmp_path), K=8)
    assert "compile error" not in result.message, result.message
    assert "nvcc compiled" in result.message, result.message
    assert not result.ok, "%s: injected bug was not detected" % kernel
    assert result.max_abs_err > result.tolerance
