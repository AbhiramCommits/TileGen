"""Schedule autotuner.

Explores a discrete space of :class:`~tilegen.codegen.Config` values, prunes
illegal combinations with logged reasons, then for each surviving config:
generates, compiles (cached by config hash), differentially verifies, and times
the kernel (5 warmup iterations, >= 20 timed iterations, reporting min and
median ms).  Effective GFLOP/s and memory bandwidth are computed, and every
variant's speedup against a fixed naive baseline is recorded.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import ctypes
import numpy as np
import scipy.sparse as sp

from . import codegen as _codegen
from .codegen import Config
from .dsl import standard_kernels
from .verify import (
    _inputs_for,
    _load_matrix,
    _reference,
    compile_generated,
    run_kernel,
)

__all__ = [
    "SEARCH_SPACE",
    "BASELINE",
    "illegal_reason",
    "enumerate_configs",
    "legal_configs",
    "tune",
]

# Discrete search space (the product is 4 * 5 * 4 * 3 * 3 = 720 raw combos).
SEARCH_SPACE: Dict[str, List[int]] = {
    "tile_size": [32, 64, 128, 256],
    "threads_per_row": [1, 2, 4, 8, 32],
    "unroll": [1, 2, 4, 8],
    "vector_width": [1, 2, 4],
    "block_size": [128, 256, 512],
}

BASELINE = Config(tile_size=32, threads_per_row=1, unroll=1, vector_width=1,
                  block_size=128)


def illegal_reason(cfg: Config) -> Optional[str]:
    """Return a human-readable reason a config is illegal, or None if legal."""
    if cfg.threads_per_row > cfg.block_size:
        return "threads_per_row (%d) > block_size (%d)" % (
            cfg.threads_per_row, cfg.block_size)
    if cfg.block_size % cfg.threads_per_row != 0:
        return "block_size (%d) not divisible by threads_per_row (%d)" % (
            cfg.block_size, cfg.threads_per_row)
    if cfg.vector_width > cfg.unroll:
        return "vector_width (%d) > unroll (%d) (unsound: no unroll slots)" % (
            cfg.vector_width, cfg.unroll)
    if cfg.vector_width not in (1, 2, 4):
        return "vector_width %d not supported" % cfg.vector_width
    return None


def enumerate_configs() -> List[Tuple[Config, Optional[str]]]:
    """All raw configs paired with their prune reason (None == legal)."""
    out = []
    keys = list(SEARCH_SPACE.keys())
    for combo in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        cfg = Config(**dict(zip(keys, combo)))
        out.append((cfg, illegal_reason(cfg)))
    return out


def legal_configs() -> List[Config]:
    return [cfg for cfg, reason in enumerate_configs() if reason is None]


def prune_log() -> List[Dict[str, str]]:
    return [{"config": str(cfg.to_dict()), "reason": reason}
            for cfg, reason in enumerate_configs() if reason is not None]


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def _flops_bytes(kernel: str, A: sp.csr_matrix, K: int) -> Tuple[float, float]:
    M, N = A.shape
    nnz = int(A.nnz)
    if kernel == "spmv":
        return 2.0 * nnz, nnz * 16.0 + (M + 1) * 8.0 + nnz * 8.0 + M * 8.0
    if kernel == "spmm":
        return 2.0 * nnz * K, nnz * 16.0 + (M + 1) * 8.0 + nnz * K * 8.0 + M * K * 8.0
    if kernel == "sddmm":
        return 2.0 * nnz * K, nnz * 16.0 + (M + 1) * 8.0 + 2.0 * nnz * K * 8.0 + nnz * 8.0
    if kernel == "fused_spmv_axpy":
        return 2.0 * nnz, nnz * 16.0 + (M + 1) * 8.0 + nnz * 8.0 + 2 * M * 8.0
    if kernel == "axpy":
        return 2.0 * M, 3.0 * M * 8.0
    raise ValueError(kernel)


def _make_caller(lib_path: str, kernel_name: str, A: sp.csr_matrix, data, K: int):
    """Build a zero-argument closure that invokes the kernel with fixed buffers."""
    M, N = A.shape
    nnz = int(A.nnz)
    lib = ctypes.CDLL(lib_path)
    fn = getattr(lib, "launch_%s" % kernel_name)
    from .verify import _SIGNATURES
    fn.argtypes = _SIGNATURES[kernel_name]

    ip = A.indptr.astype(np.int64)
    ind = A.indices.astype(np.int64)
    dat = A.data.astype(np.float64)
    x, w, B, U, V, alpha, beta = data
    p = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pi = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    keep = [ip, ind, dat, x, w, B, U, V]

    if kernel_name == "spmv":
        out = np.zeros(M, dtype=np.float64)
        args = (pi(ip), pi(ind), p(dat), p(x), p(out), M, N, nnz)
    elif kernel_name == "spmm":
        out = np.zeros((M, K), dtype=np.float64)
        args = (pi(ip), pi(ind), p(dat), p(B), p(out), M, N, K, nnz)
    elif kernel_name == "sddmm":
        out = np.zeros(nnz, dtype=np.float64)
        args = (pi(ip), pi(ind), p(dat), p(U), p(V), p(out), M, N, K, nnz)
    elif kernel_name == "fused_spmv_axpy":
        out = np.zeros(M, dtype=np.float64)
        args = (pi(ip), pi(ind), p(dat), p(x), p(w), p(out), alpha, beta, M, N, nnz)
    elif kernel_name == "axpy":
        out = np.zeros(M, dtype=np.float64)
        args = (p(x), p(w), p(out), alpha, beta, M)
    else:
        raise ValueError(kernel_name)
    keep.append(out)
    return (lambda: fn(*args)), out, keep


def _time_caller(caller, warmup: int, iters: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        caller()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        caller()
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    return (sum(times) / len(times), min(times), statistics.median(times))


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
def _candidate_configs(budget: int, mode: str, rng: random.Random) -> List[Config]:
    configs = legal_configs()
    if mode == "coordinate":
        # Greedy coordinate descent seeded at the baseline.
        best = Config(**BASELINE.to_dict())
        chosen = [best]
        for key in ["tile_size", "threads_per_row", "unroll", "vector_width", "block_size"]:
            for value in SEARCH_SPACE[key]:
                cand = Config(**{**best.to_dict(), key: value})
                if illegal_reason(cand) is None and str(cand.hash()) not in {c.hash() for c in chosen}:
                    chosen.append(cand)
                    if len(chosen) >= budget:
                        return chosen
        return chosen[:budget]
    # random sampling, always including the baseline so speedup is well defined
    pool = [c for c in configs if c.hash() != BASELINE.hash()]
    rng.shuffle(pool)
    out = [Config(**BASELINE.to_dict())]
    for c in pool:
        if len(out) >= budget:
            break
        out.append(c)
    return out


def tune(
    kernel_name: str,
    matrix,
    budget: int = 40,
    mode: str = "random",
    backend: str = "cpu",
    K: int = 8,
    workdir: str = "build/generated",
    warmup: int = 5,
    iters: int = 25,
    seed: int = 0,
) -> Dict[str, Any]:
    """Autotune one (kernel, matrix) pair; returns a dict of all variants tried."""
    rng = random.Random(seed)
    A, class_name = _load_matrix(matrix)
    data = _inputs_for(kernel_name, A, K, seed)
    ref = np.asarray(_reference(kernel_name, A, data), dtype=np.float64).ravel()
    flops, nbytes = _flops_bytes(kernel_name, A, K)
    M, N = A.shape
    nnz = int(A.nnz)

    candidates = _candidate_configs(budget, mode, rng)
    kernels = standard_kernels()

    records: List[Dict[str, Any]] = []
    baseline_ms: Optional[float] = None

    for cfg in candidates:
        rec: Dict[str, Any] = {
            "kernel": kernel_name,
            "matrix_class": class_name,
            "rows": int(M),
            "nnz": nnz,
            "config": cfg.to_dict(),
            "skipped": None,
        }
        gen = _codegen.generate(kernels[kernel_name], cfg, backend, outdir=workdir)
        try:
            lib_path, log = compile_generated(gen, backend, workdir)
        except RuntimeError as exc:
            rec["skipped"] = "compile error: %s" % exc
            records.append(rec)
            continue
        if lib_path is None:
            rec["skipped"] = "backend %s unavailable: %s" % (backend, log)
            records.append(rec)
            continue

        caller, out, keep = _make_caller(lib_path, kernel_name, A, data, K)
        caller()
        outv = np.asarray(out, dtype=np.float64).ravel()
        diff = np.abs(outv - ref)
        max_abs = float(diff.max()) if diff.size else 0.0
        eps = float(np.finfo(np.float64).eps)
        scale = float(np.abs(ref).max()) if ref.size else 0.0
        row_nnz = int(np.diff(A.indptr).max()) if A.shape[0] else 1
        tol = float(8.0 * eps * max(1, row_nnz) * scale + 1e-12)

        ms_mean, ms_min, ms_median = _time_caller(caller, warmup, iters)
        rec.update({
            "ms_mean": ms_mean,
            "ms_min": ms_min,
            "ms_median": ms_median,
            "gflops": (flops / (ms_min * 1e6)) if ms_min > 0 else 0.0,
            "bandwidth_gbs": (nbytes / (ms_min * 1e6)) if ms_min > 0 else 0.0,
            "max_abs_err": max_abs,
            "tolerance": tol,
            "ok": bool(max_abs <= tol),
            "log": log,
        })
        if cfg.hash() == BASELINE.hash():
            baseline_ms = ms_min
        records.append(rec)

    # speedup vs baseline (fall back to the slowest / first measured config)
    base = baseline_ms
    if base is None:
        measured = [r for r in records if r.get("ms_min")]
        base = measured[0]["ms_min"] if measured else None
    for r in records:
        if r.get("ms_min") and base:
            r["speedup_vs_naive"] = base / r["ms_min"]
        else:
            r["speedup_vs_naive"] = None

    measured = [r for r in records if r.get("ms_min") and r.get("ok")]
    winner = min(measured, key=lambda r: r["ms_median"]) if measured else None

    return {
        "kernel": kernel_name,
        "matrix_class": class_name,
        "baseline": BASELINE.to_dict(),
        "baseline_ms_min": base,
        "winner": winner["config"] if winner else None,
        "winner_metrics": winner if winner else None,
        "records": records,
    }


_METRIC_FIELDS = [
    "ms_mean", "ms_min", "ms_median", "gflops", "bandwidth_gbs",
    "max_abs_err", "tolerance", "ok", "speedup_vs_naive",
]

_CSV_COLS = ["matrix_class", "kernel", "tile_size", "threads_per_row", "unroll",
             "vector_width", "block_size", "ms_mean", "ms_min", "ms_median",
             "gflops", "bandwidth_gbs", "max_abs_err", "ok", "speedup_vs_naive"]


def _compact_metrics(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop verbose per-variant details (compile logs, duplicated config)."""
    if record is None:
        return None
    return {key: record.get(key) for key in _METRIC_FIELDS if key in record}


def _flat_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {"matrix_class": record["matrix_class"], "kernel": record["kernel"]}
    row.update(record["config"])
    row.update({key: record.get(key) for key in _METRIC_FIELDS if key in record})
    return row


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as fh:
        fh.write(",".join(_CSV_COLS) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(c, "")) for c in _CSV_COLS) + "\n")


def write_outputs(all_results: List[Dict[str, Any]], outdir: str, env: Dict[str, Any],
                  pruned: List[Dict[str, str]], full_outdir: str = "build") -> None:
    """Write a compact summary to ``outdir`` and the full sweep to ``full_outdir``."""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(full_outdir, exist_ok=True)

    best: Dict[str, Dict[str, Any]] = {}
    flat: List[Dict[str, Any]] = []
    compact_results: List[Dict[str, Any]] = []
    top5: Dict[str, List[Dict[str, Any]]] = {}

    for res in all_results:
        cls = res["matrix_class"]
        compact_results.append({
            "kernel": res["kernel"],
            "matrix_class": cls,
            "baseline": res["baseline"],
            "baseline_ms_min": res["baseline_ms_min"],
            "winner": res["winner"],
            "winner_metrics": _compact_metrics(res["winner_metrics"]),
        })
        best.setdefault(cls, {})[res["kernel"]] = {
            "config": res["winner"],
            "metrics": _compact_metrics(res["winner_metrics"]),
            "baseline_ms_min": res["baseline_ms_min"],
        }
        for r in res["records"]:
            if "ms_min" in r and r["ms_min"] is not None:
                flat.append(_flat_row(r))
                if r.get("ok"):
                    candidate = _flat_row(r)
                    top5.setdefault(cls, []).append(candidate)

    for cls in top5:
        top5[cls] = sorted(top5[cls], key=lambda r: r["ms_median"])[:5]

    # Full sweep (large, reproducible artifact): ignored by git under build/.
    with open(os.path.join(full_outdir, "autotune.json"), "w") as fh:
        json.dump({
            "hardware": env,
            "baseline": BASELINE.to_dict(),
            "search_space": SEARCH_SPACE,
            "pruned": pruned,
            "results": all_results,
        }, fh, indent=2)
    _write_csv(os.path.join(full_outdir, "autotune.csv"), flat)

    # Compact, tracked summary: per-pair winners plus the top five per class.
    with open(os.path.join(outdir, "autotune.json"), "w") as fh:
        json.dump({
            "hardware": env,
            "baseline": BASELINE.to_dict(),
            "search_space": SEARCH_SPACE,
            "pruned": pruned,
            "results": compact_results,
            "best": best,
            "top5": top5,
        }, fh, indent=2)

    with open(os.path.join(outdir, "best_configs.json"), "w") as fh:
        json.dump(best, fh, indent=2)

    _write_csv(os.path.join(outdir, "autotune.csv"),
               [row for cls in sorted(top5) for row in top5[cls]])
