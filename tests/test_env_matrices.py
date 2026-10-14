"""Tests for the environment probe and the matrix corpus generator."""

from __future__ import annotations

import json
import os

import numpy as np

from tilegen.env import probe
from tilegen.matrices import (
    build_corpus,
    corpus_classes,
    load_csr_npz,
    make_matrix,
    matrix_stats,
)


def test_probe_keys():
    info = probe()
    for key in [
        "has_nvcc",
        "nvcc_version",
        "has_gpu",
        "gpu_name",
        "sm_arch",
        "cpu_name",
        "num_cores",
        "backend",
    ]:
        assert key in info
    assert info["backend"] in ("cuda", "cpu")
    assert isinstance(info["num_cores"], int)
    assert info["num_cores"] >= 1


def test_corpus_roundtrip(tmp_path):
    manifest = build_corpus(str(tmp_path))
    assert len(manifest["matrices"]) == len(corpus_classes())
    assert os.path.exists(str(tmp_path / "manifest.json"))
    for entry in manifest["matrices"]:
        name = entry["class_name"]
        path = str(tmp_path / ("%s.npz" % name))
        assert os.path.exists(path)
        m = load_csr_npz(path)
        assert m.shape[0] == entry["rows"]
        assert m.nnz == entry["nnz"]
        # .bin exporter also present
        assert os.path.exists(str(tmp_path / ("%s.bin" % name)))


def test_stats_and_imbalance():
    m = make_matrix("dense_rows")
    stats = matrix_stats("dense_rows", m)
    assert stats["imbalance_ratio"] > 10.0
    counts = np.diff(m.indptr)
    assert stats["nnz_per_row_max"] == float(counts.max())


def test_determinism():
    a = make_matrix("uniform_random")
    b = make_matrix("uniform_random")
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.allclose(a.data, b.data)


def test_manifest_is_json_serialisable(tmp_path):
    build_corpus(str(tmp_path))
    with open(str(tmp_path / "manifest.json")) as fh:
        data = json.load(fh)
    assert data["seed"] == 1234
    assert len(data["matrices"]) == 6
