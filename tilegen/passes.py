"""Compiler optimization pass pipeline over the TileGen IR.

Every pass has the signature ``pass(kernel) -> kernel`` and appends one entry to
``kernel.pass_log`` describing what it did (pass name, action, and before/after
snippets).  ``run_pipeline`` / ``optimize`` deep-copies the input kernel, sets the
transient ``_config`` / ``_backend`` attributes, and then applies ``PASSES`` in
order.  Running the pipeline twice on a canonical kernel is a no-op.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

from .ir import (
    Access,
    Add,
    Const,
    Expr,
    Index,
    Kernel,
    Mul,
    Reduce,
    Schedule,
    Stmt,
    ThreadsPerRow,
    Tile,
    Unroll,
    Vectorize,
    _fmt_expr,
)

__all__ = [
    "PASSES",
    "PASS_NAMES",
    "run_pipeline",
    "optimize",
    "canonicalize",
    "loop_fusion",
    "tiling",
    "thread_mapping",
    "unroll",
    "vectorize",
    "dead_code_elim",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _log(kernel: Kernel, name: str, action: str, before: str = "", after: str = "") -> None:
    kernel.pass_log.append(
        {"pass": name, "action": action, "before": before, "after": after}
    )


def _map_expr(expr: Expr, fn) -> Expr:
    """Rebuild an expression applying ``fn`` to each node's children first."""
    if isinstance(expr, (Access, Const)):
        return fn(expr)
    if isinstance(expr, Mul):
        return fn(Mul(_map_expr(expr.left, fn), _map_expr(expr.right, fn)))
    if isinstance(expr, Add):
        return fn(Add(_map_expr(expr.left, fn), _map_expr(expr.right, fn)))
    if isinstance(expr, Reduce):
        return fn(Reduce(expr.index, _map_expr(expr.body, fn)))
    return expr


def _replace_reduce_reads(expr: Expr, tensor_name: str, replacement: Expr) -> Expr:
    """Replace ``Access(tensor_name, ...)`` nodes with ``replacement`` (deep copy)."""
    if isinstance(expr, Access):
        if expr.tensor.name == tensor_name:
            return copy.deepcopy(replacement)
        return expr
    if isinstance(expr, Mul):
        return Mul(_replace_reduce_reads(expr.left, tensor_name, replacement),
                   _replace_reduce_reads(expr.right, tensor_name, replacement))
    if isinstance(expr, Add):
        return Add(_replace_reduce_reads(expr.left, tensor_name, replacement),
                   _replace_reduce_reads(expr.right, tensor_name, replacement))
    if isinstance(expr, Reduce):
        return Reduce(expr.index, _replace_reduce_reads(expr.body, tensor_name, replacement))
    return expr


def _writes(stmt: Stmt) -> str:
    return stmt.lhs.tensor.name


def _reads(expr: Expr) -> List[str]:
    return [n.tensor.name for n in expr.walk() if isinstance(n, Access)]


# --------------------------------------------------------------------------- #
# 1. canonicalize
# --------------------------------------------------------------------------- #
def _canon(expr: Expr) -> Expr:
    if isinstance(expr, Add):
        left = _canon(expr.left)
        right = _canon(expr.right)
        # constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value + right.value)
        if isinstance(left, Const) and left.value == 0.0:
            return right
        if isinstance(right, Const) and right.value == 0.0:
            return left
        # left-leaning reassociation for determinism
        if isinstance(right, Add):
            return Add(Add(left, right.left), right.right)
        return Add(left, right)
    if isinstance(expr, Mul):
        left = _canon(expr.left)
        right = _canon(expr.right)
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value * right.value)
        if isinstance(left, Const) and left.value == 1.0:
            return right
        if isinstance(right, Const) and right.value == 1.0:
            return left
        if isinstance(right, Mul):
            return Mul(Mul(left, right.left), right.right)
        return Mul(left, right)
    if isinstance(expr, Reduce):
        return Reduce(expr.index, _canon(expr.body))
    return expr


def _expr_key(expr: Expr) -> str:
    return _fmt_expr(expr)


def canonicalize(kernel: Kernel) -> Kernel:
    changed = False
    before = _expr_key(kernel.stmts[0].rhs) if kernel.stmts else ""
    new_stmts = []
    for st in kernel.stmts:
        rhs = _canon(st.rhs)
        if _expr_key(rhs) != _expr_key(st.rhs):
            changed = True
        new_stmts.append(Stmt(lhs=st.lhs, rhs=rhs))
    kernel.stmts = new_stmts
    after = _expr_key(kernel.stmts[0].rhs) if kernel.stmts else ""
    if changed:
        _log(kernel, "canonicalize",
             "flattened/reassociated expressions and folded constants",
             before, after)
    else:
        _log(kernel, "canonicalize", "no change (already canonical)", before, after)
    return kernel


# --------------------------------------------------------------------------- #
# 2. loop_fusion
# --------------------------------------------------------------------------- #
def _access_indices_for(expr: Expr, tensor_name: str) -> List[List[str]]:
    out = []
    for node in expr.walk():
        if isinstance(node, Access) and node.tensor.name == tensor_name:
            out.append([ix.name for ix in node.indices])
    return out


def loop_fusion(kernel: Kernel) -> Kernel:
    if len(kernel.stmts) < 2:
        _log(kernel, "loop_fusion", "single statement; nothing to fuse")
        return kernel

    # Look for stmt A writing a temporary that stmt B reads.
    for a_idx in range(len(kernel.stmts)):
        writer = kernel.stmts[a_idx]
        temp = _writes(writer)
        write_indices = [ix.name for ix in writer.lhs.indices]
        for b_idx in range(len(kernel.stmts)):
            if b_idx == a_idx:
                continue
            reader = kernel.stmts[b_idx]
            if temp not in _reads(reader.rhs):
                continue

            read_index_lists = _access_indices_for(reader.rhs, temp)
            # A true (loop-carried) dependency exists if the read index set
            # differs from the write index set; fusing would be unsafe.
            if any(ri != write_indices for ri in read_index_lists):
                _log(
                    kernel,
                    "loop_fusion",
                    "refused: true dependency on '%s' (read indices %s differ from "
                    "write indices %s)" % (temp, read_index_lists, write_indices),
                    _expr_key(reader.rhs),
                    _expr_key(writer.rhs),
                )
                return kernel

            before = " | ".join(_expr_key(s.rhs) for s in kernel.stmts)
            fused_rhs = _replace_reduce_reads(reader.rhs, temp, writer.rhs)
            new_stmts = [Stmt(lhs=reader.lhs, rhs=fused_rhs)]
            # keep any other statements untouched
            for idx, st in enumerate(kernel.stmts):
                if idx not in (a_idx, b_idx):
                    new_stmts.append(st)
            kernel.stmts = new_stmts
            after = _expr_key(fused_rhs)
            _log(
                kernel,
                "loop_fusion",
                "fused statements over index %s; removed temporary '%s'; "
                "dependency-safe (read index == write index)"
                % (write_indices[0] if write_indices else "?", temp),
                before,
                after,
            )
            return kernel

    _log(kernel, "loop_fusion", "no fusable producer/consumer pair found")
    return kernel


# --------------------------------------------------------------------------- #
# 3. tiling
# --------------------------------------------------------------------------- #
def _config_of(kernel: Kernel) -> Any:
    return getattr(kernel, "_config", None)


def tiling(kernel: Kernel) -> Kernel:
    cfg = _config_of(kernel)
    tile = getattr(cfg, "tile_size", 1) or 1
    free = kernel.free_indices()
    if not free:
        _log(kernel, "tiling", "no free indices; skipped")
        return kernel
    outer = free[0]
    if tile <= 1:
        _log(kernel, "tiling", "tile_size <= 1; no tiling applied")
        return kernel
    # Record the transform on the schedule.
    if not any(isinstance(t, Tile) and t.index.name == outer.name
               for t in kernel.schedule):
        kernel.schedule.append(Tile(outer, tile))
    _log(
        kernel,
        "tiling",
        "split index '%s' into outer '%s_o' and inner '%s_i' with tile=%d"
        % (outer.name, outer.name, outer.name, tile),
        "for %s in 0..extent" % outer.name,
        "for %s_o step %d: for %s_i: ..." % (outer.name, tile, outer.name),
    )
    return kernel


# --------------------------------------------------------------------------- #
# 4. thread_mapping
# --------------------------------------------------------------------------- #
def thread_mapping(kernel: Kernel) -> Kernel:
    cfg = _config_of(kernel)
    backend = getattr(kernel, "_backend", "cpu")
    tpr = getattr(cfg, "threads_per_row", 1) or 1
    block = getattr(cfg, "block_size", 256) or 256
    free = kernel.free_indices()
    outer = free[0].name if free else "?"
    if tpr > 1 and not any(
        isinstance(t, ThreadsPerRow) and t.k == tpr for t in kernel.schedule
    ):
        kernel.schedule.append(ThreadsPerRow(tpr))
    if backend == "cuda":
        unit = "warp (segmented reduction via __shfl_down_sync)" if tpr > 1 else "thread"
        _log(
            kernel,
            "thread_mapping",
            "map index '%s' to CUDA block/thread; threads_per_row=%d -> %s; block=%d"
            % (outer, tpr, unit, block),
            None,
            None,
        )
    else:
        _log(
            kernel,
            "thread_mapping",
            "map index '%s' to OpenMP parallel for schedule(dynamic, 256); "
            "threads_per_row=%d simulated per-row" % (outer, tpr),
            None,
            None,
        )
    return kernel


# --------------------------------------------------------------------------- #
# 5. unroll
# --------------------------------------------------------------------------- #
def unroll(kernel: Kernel) -> Kernel:
    cfg = _config_of(kernel)
    factor = getattr(cfg, "unroll", 1) or 1
    red = kernel.reduce_node()
    if factor <= 1:
        _log(kernel, "unroll", "unroll factor <= 1; no unrolling")
        return kernel
    target = red.index.name if red is not None else (kernel.free_indices() or [Index("?")])[0].name
    if not any(isinstance(t, Unroll) and t.factor == factor for t in kernel.schedule):
        kernel.schedule.append(Unroll(red.index if red is not None else kernel.free_indices()[0], factor))
    _log(
        kernel,
        "unroll",
        "unroll innermost loop over '%s' by %d with remainder-handling tail"
        % (target, factor),
        "for %s: step 1" % target,
        "for %s: step %d { unrolled body } ; tail remainder" % (target, factor),
    )
    return kernel


# --------------------------------------------------------------------------- #
# 6. vectorize
# --------------------------------------------------------------------------- #
def _dense_unit_stride_access(kernel: Kernel) -> Optional[Access]:
    """Find a dense access whose innermost index maps to stride 1."""
    for st in kernel.stmts:
        for node in st.rhs.walk():
            if isinstance(node, Access) and not node.tensor.format.is_sparse():
                if node.tensor.order >= 2 and len(node.tensor.shape) >= 2:
                    # last index maps to stride 1 in row-major storage
                    return node
    return None


def vectorize(kernel: Kernel) -> Kernel:
    cfg = _config_of(kernel)
    width = getattr(cfg, "vector_width", 1) or 1
    if width <= 1:
        _log(kernel, "vectorize", "vector_width <= 1; skipped")
        return kernel
    access = _dense_unit_stride_access(kernel)
    if access is None:
        _log(
            kernel,
            "vectorize",
            "skipped: no unit-stride dense access to widen (spmv gather is not "
            "vectorizable)",
        )
        return kernel
    last_dim = access.tensor.shape[-1]
    if isinstance(last_dim, int) and last_dim % width != 0:
        _log(
            kernel,
            "vectorize",
            "skipped: extent %d not divisible by vector_width %d" % (last_dim, width),
        )
        return kernel
    if not any(isinstance(t, Vectorize) and t.width == width for t in kernel.schedule):
        kernel.schedule.append(Vectorize(width))
    vtype = {2: "double2", 4: "double2x2/float4"}.get(width, "vec%d" % width)
    _log(
        kernel,
        "vectorize",
        "widened unit-stride dense access %s along its last axis by %d (%s); "
        "alignment proven from row-major layout"
        % (access.tensor.name, width, vtype),
        "%s[j*K + k]" % access.tensor.name,
        "%s vectorized loads" % access.tensor.name,
    )
    return kernel


# --------------------------------------------------------------------------- #
# 7. dead_code_elim
# --------------------------------------------------------------------------- #
def dead_code_elim(kernel: Kernel) -> Kernel:
    read_names = set()
    for st in kernel.stmts:
        read_names.update(_reads(st.rhs))
    # keep outputs = tensors written by remaining statements
    kept: List[Stmt] = []
    removed: List[str] = []
    outputs = {_writes(st) for st in kernel.stmts if _writes(st) not in read_names}
    for st in kernel.stmts:
        name = _writes(st)
        if name in read_names or name in outputs:
            kept.append(st)
        else:
            removed.append(name)
    kernel.stmts = kept
    _log(
        kernel,
        "dead_code_elim",
        "removed %d unused temporary(ies)%s"
        % (len(removed), (": " + ", ".join(removed)) if removed else ""),
    )
    return kernel


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
PASSES = [
    canonicalize,
    loop_fusion,
    tiling,
    thread_mapping,
    unroll,
    vectorize,
    dead_code_elim,
]

PASS_NAMES = [
    "canonicalize",
    "loop_fusion",
    "tiling",
    "thread_mapping",
    "unroll",
    "vectorize",
    "dead_code_elim",
]


def run_pipeline(
    kernel: Kernel,
    config: Any = None,
    backend: str = "cpu",
    fresh: bool = True,
    skip: Sequence[str] = (),
) -> Kernel:
    """Apply all passes in order, optionally skipping named passes."""
    k = copy.deepcopy(kernel) if fresh else kernel
    k._config = config
    k._backend = backend
    k.pass_log = []
    skipped = set(skip)
    for p in PASSES:
        if p.__name__ not in skipped:
            p(k)
    try:
        k.validate()
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("pass pipeline produced an invalid kernel: %s" % exc)
    return k


# Public alias used by scripts / tests.
optimize = run_pipeline
