"""Deterministic sparse-matrix corpus generation.

Six matrices spanning distinct *matrix classes* are generated in CSR form and
saved as ``.npz`` archives containing ``indptr``, ``indices``, ``data``,
``shape`` and ``class_name``.  A ``manifest.json`` records the statistics of
every matrix, including the *imbalance ratio* (max row nnz / mean row nnz)
which drives the autotuner's interest in row-length variance.

Everything is deterministic: each class seeds its own ``numpy`` Generator from
the fixed seed 1234, so the corpus is reproducible regardless of generation
order.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = [
    "build_corpus",
    "make_matrix",
    "matrix_stats",
    "corpus_classes",
    "save_csr_npz",
    "load_csr_npz",
    "export_matrix_bin",
    "main",
]

SEED = 1234

# Binary dump format used by the C++ harness (little-endian):
#   magic   : uint32  0x54474D31  ("TGM1")
#   rows    : int64
#   cols    : int64
#   nnz     : int64
#   indptr  : (rows + 1) x int64
#   indices : nnz x int64
#   data    : nnz x float64
_BIN_MAGIC = 0x54474D31


def _rng(name: str) -> np.random.Generator:
    """A per-class generator seeded deterministically from the global SEED."""
    # Fold the class name into the seed so each matrix is reproducible on its
    # own, independent of the order in which classes are built.
    extra = sum((i + 1) * ord(c) for i, c in enumerate(name))
    return np.random.default_rng(SEED + extra)


def _banded(n: int, diags: int, name: str) -> sp.csr_matrix:
    """Symmetric band matrix with ``diags`` diagonals (must be odd)."""
    rng = _rng(name)
    half = diags // 2
    offsets = list(range(-half, half + 1))
    mats = []
    for off in offsets:
        length = n - abs(off)
        vals = rng.standard_normal(length)
        if off == 0:
            # Strengthen the diagonal so the matrix is comfortably well scaled.
            vals = vals + 10.0
        mats.append(sp.diags(vals, offsets=off, shape=(n, n), format="csr"))
    m = mats[0]
    for extra in mats[1:]:
        m = m + extra
    m = m.tocsr()
    m.sum_duplicates()
    m.sort_indices()
    return m


def _row_var(m: sp.csr_matrix) -> None:
    """Finalize a CSR matrix in place: canonical sorted, no duplicates."""
    m.sum_duplicates()
    m.sort_indices()


def _uniform_random(n: int, cols: int, per_row: int, name: str) -> sp.csr_matrix:
    rng = _rng(name)
    counts = np.full(n, per_row, dtype=np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    total = rows.size
    cols_idx = rng.integers(0, cols, size=total, dtype=np.int64)
    data = rng.standard_normal(total)
    m = sp.coo_matrix((data, (rows, cols_idx)), shape=(n, cols)).tocsr()
    _row_var(m)
    return m


def _power_law(n: int, cols: int, alpha: float, cap: int, name: str) -> sp.csr_matrix:
    """Row nnz ~ power law with exponent ``alpha`` (high variance)."""
    rng = _rng(name)
    # Inverse-transform sampling of a discrete power law P(k) ~ k^-alpha.
    u = rng.random(n)
    u = np.clip(u, 1e-12, 1.0 - 1e-9)
    counts = np.floor(1.0 / u).astype(np.int64)  # alpha == 2.0 tail
    if abs(alpha - 2.0) > 1e-9:
        counts = np.floor(np.power(u, -1.0 / (alpha - 1.0))).astype(np.int64)
    counts = np.clip(counts, 1, cap)
    counts = np.minimum(counts, cols)
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    total = rows.size
    cols_idx = rng.integers(0, cols, size=total, dtype=np.int64)
    data = rng.standard_normal(total)
    m = sp.coo_matrix((data, (rows, cols_idx)), shape=(n, cols)).tocsr()
    _row_var(m)
    return m


def _dense_rows(n: int, cols: int, frac: float, big: int, small: int,
                name: str) -> sp.csr_matrix:
    """A small fraction of rows carry a huge nnz; the rest are sparse."""
    rng = _rng(name)
    counts = np.full(n, small, dtype=np.int64)
    n_big = max(1, int(round(frac * n)))
    big_rows = rng.choice(n, size=n_big, replace=False)
    counts[big_rows] = big
    counts = np.minimum(counts, cols)
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    total = rows.size
    cols_idx = rng.integers(0, cols, size=total, dtype=np.int64)
    data = rng.standard_normal(total)
    m = sp.coo_matrix((data, (rows, cols_idx)), shape=(n, cols)).tocsr()
    _row_var(m)
    return m


def _blocked(n: int, bs: int, blocks_per_row: int, name: str) -> sp.csr_matrix:
    """Block-sparse (BSR-style) matrix of dense ``bs x bs`` blocks."""
    if n % bs != 0:
        raise ValueError("blocked matrix requires n divisible by block size")
    rng = _rng(name)
    nb = n // bs
    rows = []
    cols = []
    data = []
    local = np.arange(bs, dtype=np.int64)
    for br in range(nb):
        bcols = {br}
        while len(bcols) < blocks_per_row:
            bcols.add(int(rng.integers(0, nb)))
        for bc in bcols:
            rr = br * bs + local
            cc = bc * bs + local
            rr_grid, cc_grid = np.meshgrid(rr, cc, indexing="ij")
            rows.append(rr_grid.ravel())
            cols.append(cc_grid.ravel())
            data.append(rng.standard_normal(bs * bs))
    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    data_arr = np.concatenate(data)
    m = sp.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(n, n)).tocsr()
    _row_var(m)
    return m


def make_matrix(class_name: str) -> sp.csr_matrix:
    """Build a single named matrix class deterministically."""
    if class_name == "banded":
        return _banded(200000, 7, "banded")
    if class_name == "uniform_random":
        return _uniform_random(100000, 100000, 16, "uniform_random")
    if class_name == "power_law":
        return _power_law(100000, 100000, 2.0, 2000, "power_law")
    if class_name == "diagonal_heavy":
        return _banded(200000, 3, "diagonal_heavy")
    if class_name == "dense_rows":
        return _dense_rows(20000, 20000, 0.01, 2000, 4, "dense_rows")
    if class_name == "blocked":
        return _blocked(100000, 8, 2, "blocked")
    raise ValueError("unknown matrix class: %s" % class_name)


def corpus_classes() -> List[str]:
    return [
        "banded",
        "uniform_random",
        "power_law",
        "diagonal_heavy",
        "dense_rows",
        "blocked",
    ]


def matrix_stats(class_name: str, m: sp.csr_matrix) -> Dict[str, object]:
    """Compute the descriptive statistics recorded in the manifest."""
    rows, cols = m.shape
    nnz = int(m.nnz)
    counts = np.diff(m.indptr).astype(np.float64)
    mean = float(counts.mean()) if counts.size else 0.0
    median = float(np.median(counts)) if counts.size else 0.0
    mx = float(counts.max()) if counts.size else 0.0
    sd = float(counts.std()) if counts.size else 0.0
    imbalance = float(mx / mean) if mean > 0 else 0.0
    return {
        "class_name": class_name,
        "rows": int(rows),
        "cols": int(cols),
        "nnz": nnz,
        "nnz_per_row_mean": mean,
        "nnz_per_row_median": median,
        "nnz_per_row_max": mx,
        "nnz_per_row_stddev": sd,
        "imbalance_ratio": imbalance,
    }


def save_csr_npz(path: str, class_name: str, m: sp.csr_matrix) -> None:
    """Persist a CSR matrix to ``.npz`` with the required fields."""
    m = m.tocsr()
    m.sum_duplicates()
    m.sort_indices()
    np.savez(
        path,
        indptr=m.indptr.astype(np.int64),
        indices=m.indices.astype(np.int64),
        data=m.data.astype(np.float64),
        shape=np.array(m.shape, dtype=np.int64),
        class_name=np.array(class_name),
    )


def load_csr_npz(path: str) -> sp.csr_matrix:
    """Load a CSR matrix written by :func:`save_csr_npz`."""
    with np.load(path, allow_pickle=False) as z:
        indptr = z["indptr"].astype(np.int64)
        indices = z["indices"].astype(np.int64)
        data = z["data"].astype(np.float64)
        shape = tuple(int(x) for x in z["shape"])
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def export_matrix_bin(path: str, m: sp.csr_matrix) -> None:
    """Write a matrix in the little-endian binary format the C++ harness reads."""
    m = m.tocsr()
    m.sum_duplicates()
    m.sort_indices()
    rows, cols = m.shape
    nnz = int(m.nnz)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", _BIN_MAGIC))
        fh.write(struct.pack("<qqq", rows, cols, nnz))
        fh.write(m.indptr.astype("<i8").tobytes(order="C"))
        fh.write(m.indices.astype("<i8").tobytes(order="C"))
        fh.write(m.data.astype("<f8").tobytes(order="C"))


def build_corpus(outdir: str) -> Dict[str, object]:
    """Generate the full corpus and write matrices + manifest.

    Returns the manifest dict.  Missing output directory is created; existing
    matrices are overwritten.
    """
    os.makedirs(outdir, exist_ok=True)
    manifest: Dict[str, object] = {"seed": SEED, "matrices": []}
    for class_name in corpus_classes():
        m = make_matrix(class_name)
        stats = matrix_stats(class_name, m)
        fname = "%s.npz" % class_name
        save_csr_npz(os.path.join(outdir, fname), class_name, m)
        export_matrix_bin(os.path.join(outdir, "%s.bin" % class_name), m)
        stats["file"] = fname
        manifest["matrices"].append(stats)
        print(
            "[matrices] %-16s rows=%-8d nnz=%-10d max/row=%-8.0f imbalance=%.2f"
            % (class_name, stats["rows"], stats["nnz"],
               stats["nnz_per_row_max"], stats["imbalance_ratio"])
        )
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("[matrices] wrote manifest.json with %d matrices" % len(manifest["matrices"]))
    return manifest


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic TileGen sparse matrix corpus."
    )
    parser.add_argument("--out", default="data/matrices",
                        help="output directory (default: data/matrices)")
    args = parser.parse_args(argv)
    build_corpus(args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
