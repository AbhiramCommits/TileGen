"""Smoke tests for the autotuner."""

from __future__ import annotations

import json
import os

import numpy as np

from tilegen.autotune import (
    BASELINE,
    enumerate_configs,
    illegal_reason,
    legal_configs,
    prune_log,
    tune,
    write_outputs,
)
from tilegen.codegen import Config


def _small_matrix():
    indptr = np.array([0, 2, 3, 4, 6], dtype=np.int64)
    indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    return {"indptr": indptr, "indices": indices, "data": data,
            "shape": (4, 4), "class_name": "tiny"}


def test_illegal_configs_pruned_with_reasons():
    pruned = [(c, r) for c, r in enumerate_configs() if r is not None]
    assert pruned, "expected some illegal combinations"
    for _c, reason in pruned:
        assert isinstance(reason, str) and reason
    # explicit rules
    assert illegal_reason(Config(threads_per_row=8, block_size=4)) is not None
    assert illegal_reason(Config(vector_width=4, unroll=2)) is not None
    assert illegal_reason(Config(threads_per_row=32, block_size=128)) is None
    assert all(illegal_reason(c) is None for c in legal_configs())
    assert len(prune_log()) == len(pruned)


def test_tune_smoke(tmp_path):
    res = tune("spmv", _small_matrix(), budget=3, backend="cpu",
               workdir=str(tmp_path), warmup=1, iters=3)
    assert res["winner"] is not None
    assert res["records"], "expected at least one measured variant"
    measured = [r for r in res["records"] if r.get("ms_min")]
    assert measured
    m = res["winner_metrics"]
    for key in ["ms_mean", "ms_min", "ms_median", "gflops", "bandwidth_gbs",
                "max_abs_err", "ok", "speedup_vs_naive"]:
        assert key in m
    assert res["baseline"] == BASELINE.to_dict()
    assert res["baseline_ms_min"] is not None


def test_write_outputs_schema(tmp_path):
    res = tune("spmv", _small_matrix(), budget=2, backend="cpu",
               workdir=str(tmp_path / "gen"), warmup=1, iters=2)
    full = tmp_path / "full"
    write_outputs([res], str(tmp_path), {"backend": "cpu"}, prune_log(),
                  full_outdir=str(full))
    for name in ["autotune.json", "autotune.csv", "best_configs.json"]:
        assert os.path.exists(str(tmp_path / name))
    with open(str(tmp_path / "autotune.json")) as fh:
        data = json.load(fh)
    assert "results" in data and "pruned" in data and "hardware" in data
    assert "best" in data and "top5" in data
    # The tracked summary has no per-variant records; the full sweep does.
    assert all("records" not in r for r in data["results"])
    assert "tiny" in data["best"] and "spmv" in data["best"]["tiny"]
    assert data["top5"]["tiny"]
    assert len(data["top5"]["tiny"]) <= 5
    with open(str(tmp_path / "best_configs.json")) as fh:
        best = json.load(fh)
    assert "tiny" in best and "spmv" in best["tiny"]
    with open(str(full / "autotune.json")) as fh:
        raw = json.load(fh)
    assert raw["results"][0]["records"]
