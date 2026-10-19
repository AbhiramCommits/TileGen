"""CUDA / OpenMP code generator for TileGen kernels.

``emit_cuda`` and ``emit_cpu`` both produce a complete translation unit exposing
the same ``extern "C" void launch_<name>(...)`` ABI, so the host harness and the
ctypes-based differential tester are backend-agnostic.

The generator understands the structures the standard kernels need:

* scalar reduction over the compressed index (spmv, fused_spmv_axpy)
* reduction over the compressed index with an inner dense output index (spmm)
* sparse-pattern output with an inner dense reduction (sddmm)
* elementwise dense kernels (axpy)

and emits explicit unrolled bodies with remainder tails, optional shared-memory
tiles (CUDA, when tiling is on) and warp-level ``__shfl_down_sync`` segmented
reductions (CUDA, when ``threads_per_row > 1``).

``inject_bug=True`` deliberately generates an off-by-one row bound so the
differential tester can prove it detects wrong kernels.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import passes as _passes
from .ir import Access, Add, Const, Expr, Index, Kernel, Mul, Reduce

__all__ = ["Config", "GeneratedKernel", "generate", "emit_cuda", "emit_cpu"]


_DTYPE = {"f64": "double", "f32": "float", "double": "double", "float": "float"}


def ctype(dtype: str) -> str:
    return _DTYPE.get(dtype, "double")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    tile_size: int = 32
    threads_per_row: int = 1
    unroll: int = 1
    vector_width: int = 1
    block_size: int = 256
    dtype: str = "f64"
    inject_bug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "threads_per_row": self.threads_per_row,
            "unroll": self.unroll,
            "vector_width": self.vector_width,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "inject_bug": self.inject_bug,
        }

    def hash(self) -> str:
        key = "%d|%d|%d|%d|%d|%s|%d" % (
            self.tile_size, self.threads_per_row, self.unroll,
            self.vector_width, self.block_size, self.dtype, int(self.inject_bug),
        )
        return hashlib.sha1(key.encode()).hexdigest()[:10]


@dataclass
class GeneratedKernel:
    source: str
    filename: str
    abi_signature: str
    pass_log: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "cpu"
    kernel_name: str = ""
    config: Optional[Config] = None

    def write(self, outdir: str = "build/generated") -> str:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, self.filename)
        with open(path, "w") as fh:
            fh.write(self.source)
        return path


# --------------------------------------------------------------------------- #
# Plan analysis
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    kind: str                       # scalar_reduce | row_dense_out | sparse_out | elementwise
    out: Access
    out_tensor: Any
    A: Optional[Any] = None
    row_idx: Optional[Index] = None
    col_idx: Optional[Index] = None
    reduce: Optional[Reduce] = None
    reduce_is_compressed: bool = False
    inner_out: List[Index] = field(default_factory=list)
    value_expr: Optional[Expr] = None
    final_expr: Optional[Expr] = None
    row_extent: str = "M"


def _find_access(expr: Expr, name: str) -> Optional[Access]:
    for node in expr.walk():
        if isinstance(node, Access) and node.tensor.name == name:
            return node
    return None


def _plan(kernel: Kernel) -> Plan:
    stmt = kernel.stmts[0]
    out = stmt.lhs
    A = kernel.sparse_tensor()
    reduce = kernel.reduce_node()

    if A is None:
        return Plan(kind="elementwise", out=out, out_tensor=out.tensor,
                    value_expr=stmt.rhs, final_expr=stmt.rhs)

    a_access = _find_access(stmt.rhs, A.name)
    row_idx = a_access.indices[0]
    col_idx = a_access.indices[1]
    reduce_is_compressed = reduce is not None and reduce.index.name == col_idx.name

    lhs_names = [ix.name for ix in out.indices]
    inner_out = [ix for ix in out.indices
                 if ix.name != row_idx.name and ix.name != col_idx.name]

    if out.tensor.format.is_sparse():
        kind = "sparse_out"
    elif reduce_is_compressed and inner_out:
        kind = "row_dense_out"
    else:
        kind = "scalar_reduce"

    return Plan(
        kind=kind,
        out=out,
        out_tensor=out.tensor,
        A=A,
        row_idx=row_idx,
        col_idx=col_idx,
        reduce=reduce,
        reduce_is_compressed=reduce_is_compressed,
        inner_out=inner_out,
        value_expr=reduce.body if reduce is not None else stmt.rhs,
        final_expr=stmt.rhs,
        row_extent=str(A.shape[0]) if A.shape else "M",
    )


# --------------------------------------------------------------------------- #
# ABI
# --------------------------------------------------------------------------- #
def _abi(kernel: Kernel):
    """Return (signature, arg_names, kind_map, has_sparse)."""
    out_name = kernel.stmts[0].lhs.tensor.name
    dt = ctype(kernel.dtypes[0] if kernel.dtypes else "f64")
    arrays: List[Tuple[str, Any]] = []
    scalars: List[Any] = []
    dims: List[str] = []
    seen_dims = set()
    n_sparse_in = 0
    has_sparse = False

    for t in kernel.tensors():
        if t.order == 0:
            scalars.append(t)
            continue
        if t.format.is_sparse():
            has_sparse = True
            if t.name == out_name:
                arrays.append(("out_sparse", t))
            else:
                n_sparse_in += 1
                arrays.append(("sparse", t))
        else:
            arrays.append(("dense", t))
        for s in t.shape:
            if s not in seen_dims:
                seen_dims.add(s)
                dims.append(str(s))

    if n_sparse_in > 1:
        raise NotImplementedError("codegen supports at most one sparse input")

    decls: List[str] = []
    for kind, t in arrays:
        if kind == "sparse":
            decls.append("const int64_t* indptr")
            decls.append("const int64_t* indices")
            decls.append("const %s* data" % dt)
        elif kind == "out_sparse":
            decls.append("%s* %s" % (dt, t.name))
        else:
            const = "" if t.name == out_name else "const "
            decls.append("%s%s* %s" % (const, dt, t.name))
    for t in scalars:
        decls.append("%s %s" % (dt, t.name))
    for d in dims:
        decls.append("int64_t %s" % d)
    if has_sparse:
        decls.append("int64_t nnz")

    sig = "void launch_%s(%s)" % (kernel.name, ", ".join(decls))
    arg_names = {"arrays": arrays, "scalars": [t.name for t in scalars],
                 "dims": dims, "dtype": kernel.dtypes[0] if kernel.dtypes else "f64",
                 "has_sparse": has_sparse, "decls": decls}
    return sig, arg_names, has_sparse


# --------------------------------------------------------------------------- #
# Expression emission
# --------------------------------------------------------------------------- #
def _fmt_const(v: float) -> str:
    if v == int(v):
        return "%.1f" % v
    return repr(v)


def _strides(shape) -> List[Any]:
    st: List[Any] = [1] * len(shape)
    acc: Any = 1
    for i in range(len(shape) - 1, -1, -1):
        st[i] = acc
        s = shape[i]
        if acc == 1:
            acc = s
        elif s == 1:
            acc = acc
        else:
            acc = "%s*%s" % (acc, s)
    return st


def _offset(tensor, indices, env) -> str:
    strides = _strides(tensor.shape)
    terms = []
    for pos, idx in enumerate(indices):
        cidx = env["bind"].get(idx.name, idx.name)
        stride = strides[pos]
        if stride == 1:
            terms.append(cidx)
        else:
            terms.append("%s*%s" % (cidx, stride))
    if not terms:
        return "0"
    return " + ".join(terms)


def _cexpr(e: Expr, env) -> str:
    if isinstance(e, Access):
        t = e.tensor
        if t.order == 0:
            return t.name
        if t.format.is_sparse():
            if env.get("p") is None:
                raise ValueError("sparse access %s outside a compressed loop" % t.name)
            return "%s[%s]" % (env.get("data", "data"), env["p"])
        return "%s[%s]" % (t.name, _offset(t, e.indices, env))
    if isinstance(e, Const):
        return _fmt_const(e.value)
    if isinstance(e, Mul):
        return "(%s * %s)" % (_cexpr(e.left, env), _cexpr(e.right, env))
    if isinstance(e, Add):
        return "(%s + %s)" % (_cexpr(e.left, env), _cexpr(e.right, env))
    if isinstance(e, Reduce):
        if env.get("acc") is None:
            raise ValueError("Reduce encountered without an accumulator")
        return env["acc"]
    raise TypeError("cannot emit %r" % (e,))


def _out_offset(plan: Plan, env) -> str:
    if plan.kind == "sparse_out":
        return env["p"]
    return _offset(plan.out_tensor, plan.out.indices, env)


# --------------------------------------------------------------------------- #
# Loop-body builders (shared by CPU and CUDA via small wrappers)
# --------------------------------------------------------------------------- #
def _emit_compressed_loop(lines, ind, plan, env, unroll, body, marker=None):
    """Emit a compressed p-loop, optionally unrolled with a remainder tail."""
    row = env["bind"][plan.row_idx.name]
    pad = "  " * ind
    lines.append("%sint64_t p = indptr[%s];" % (pad, row))
    lines.append("%sint64_t pend = indptr[%s + 1];" % (pad, row))

    def set_p(pe):
        env["p"] = pe
        if plan.col_idx is not None:
            env["bind"][plan.col_idx.name] = "indices[%s]" % pe

    if unroll and unroll > 1:
        lines.append("%sint64_t q = p;" % pad)
        if marker:
            lines.append("%s%s" % (pad, marker))
        lines.append("%sfor (; q + %d <= pend; q += %d) {" % (pad, unroll, unroll))
        for o in range(unroll):
            set_p("q + %d" % o if o else "q")
            lines.append("%s{" % ("  " * (ind + 1)))
            body(lines, ind + 2, env)
            lines.append("%s}" % ("  " * (ind + 1)))
        lines.append("%s}" % pad)
        lines.append("%sfor (; q < pend; ++q) {" % pad)
        set_p("q")
        lines.append("%s{" % ("  " * (ind + 1)))
        body(lines, ind + 2, env)
        lines.append("%s}" % ("  " * (ind + 1)))
        lines.append("%s}" % pad)
    else:
        lines.append("%sfor (int64_t q = p; q < pend; ++q) {" % pad)
        set_p("q")
        body(lines, ind + 1, env)
        lines.append("%s}" % pad)
    env["p"] = None
    if plan.col_idx is not None:
        env["bind"].pop(plan.col_idx.name, None)


def _emit_dense_reduce_loop(lines, ind, extent, unroll, value_fn, marker=None):
    if unroll and unroll > 1:
        if marker:
            lines.append("%s%s" % ("  " * ind, marker))
        lines.append("%sint64_t rq = 0;" % ("  " * ind))
        lines.append("%sfor (; rq + %d <= %s; rq += %d) {" % ("  " * ind, unroll, extent, unroll))
        for o in range(unroll):
            lines.append("%sacc += %s;" % ("  " * (ind + 1), value_fn("rq + %d" % o)))
        lines.append("%s}" % ("  " * ind))
        lines.append("%sfor (; rq < %s; ++rq) {" % ("  " * ind, extent))
        lines.append("%sacc += %s;" % ("  " * (ind + 1), value_fn("rq")))
        lines.append("%s}" % ("  " * ind))
    else:
        lines.append("%sfor (int64_t kr = 0; kr < %s; ++kr) {" % ("  " * ind, extent))
        lines.append("%sacc += %s;" % ("  " * (ind + 1), value_fn("kr")))
        lines.append("%s}" % ("  " * ind))


# --------------------------------------------------------------------------- #
# CPU emitter
# --------------------------------------------------------------------------- #
def _cpu_emit(kernel: Kernel, config: Config, plan: Plan) -> str:
    dt = ctype(config.dtype)
    tile = config.tile_size if config.tile_size > 1 else 1
    unroll = config.unroll
    bug = config.inject_bug
    lines: List[str] = []
    ind = 0

    row_bound = plan.row_extent
    if bug:
        row_bound = "(%s - 1)" % plan.row_extent

    def emit_row_loops(body_fn, extra_pragma: Optional[str] = None):
        nonlocal ind
        lines.append("#pragma omp parallel for schedule(dynamic, 256)")
        if extra_pragma:
            lines.append(extra_pragma)
        if tile > 1:
            lines.append("%sfor (int64_t i_o = 0; i_o < %s; i_o += %d) {" % ("  " * ind, row_bound, tile))
            lines.append("%sint64_t i_hi = (i_o + %d < %s) ? i_o + %d : %s;" % ("  " * (ind + 1), tile, row_bound, tile, row_bound))
            lines.append("%sfor (int64_t i_i = i_o; i_i < i_hi; ++i_i) {" % ("  " * (ind + 1)))
            env = {"bind": {plan.row_idx.name: "i_i"}, "p": None, "data": "data", "acc": "acc"}
            body_fn(lines, ind + 2, env)
            lines.append("%s}" % ("  " * (ind + 1)))
            lines.append("%s}" % ("  " * ind))
        else:
            lines.append("%sfor (int64_t i = 0; i < %s; ++i) {" % ("  " * ind, row_bound))
            env = {"bind": {plan.row_idx.name: "i"}, "p": None, "data": "data", "acc": "acc"}
            body_fn(lines, ind + 1, env)
            lines.append("%s}" % ("  " * ind))

    if plan.kind == "elementwise":
        i = plan.out.indices[0]
        lines.append("#pragma omp parallel for schedule(dynamic, 256)")
        lines.append("for (int64_t i = 0; i < %s; ++i) {" % row_bound)
        env = {"bind": {i.name: "i"}, "p": None, "data": "data", "acc": None}
        lines.append("  %s[%s] = %s;" % (plan.out_tensor.name,
                                         _offset(plan.out_tensor, plan.out.indices, env),
                                         _cexpr(plan.value_expr, env)))
        lines.append("}")

    elif plan.kind == "scalar_reduce":
        def body(lines_, ind_, env):
            env["acc"] = "acc"
            lines_.append("%s%s acc = 0.0;" % ("  " * ind_, dt))
            if plan.reduce_is_compressed:
                def inner(lines2, ind2, env2):
                    lines2.append("%sacc += %s;" % ("  " * ind2, _cexpr(plan.value_expr, env2)))
                _emit_compressed_loop(lines_, ind_, plan, env, unroll, inner, marker=None)
            else:
                ext = _extent_for(plan, plan.reduce.index.name)

                def pbody(lines2, ind2, env2):
                    def vfn(kexpr):
                        env3 = dict(env2)
                        env3["bind"] = dict(env2["bind"])
                        env3["bind"][plan.reduce.index.name] = kexpr
                        return _cexpr(plan.value_expr, env3)
                    _emit_dense_reduce_loop(lines2, ind2, ext, unroll, vfn)

                _emit_compressed_loop(lines_, ind_, plan, env, 1, pbody, marker=None)
            off = _out_offset(plan, env)
            lines_.append("%s%s[%s] = %s;" % ("  " * ind_, plan.out_tensor.name, off,
                                              _cexpr(plan.final_expr, env)))
        emit_row_loops(body)

    elif plan.kind == "row_dense_out":
        inner = plan.inner_out[0]
        ext = _extent_for(plan, inner.name)

        def pbody(lines_, ind_, env):
            if config.vector_width > 1:
                lines_.append("%s#pragma omp simd" % ("  " * ind_))
            if unroll > 1:
                lines_.append("%sint64_t ku = 0;" % ("  " * ind_))
                lines_.append("%sfor (; ku + %d <= %s; ku += %d) {" % ("  " * ind_, unroll, ext, unroll))
                for o in range(unroll):
                    _emit_row_dense_write(lines_, ind_ + 1, plan, env, inner.name, "ku + %d" % o)
                lines_.append("%s}" % ("  " * ind_))
                lines_.append("%sfor (; ku < %s; ++ku) {" % ("  " * ind_, ext))
                _emit_row_dense_write(lines_, ind_ + 1, plan, env, inner.name, "ku")
                lines_.append("%s}" % ("  " * ind_))
            else:
                lines_.append("%sfor (int64_t k = 0; k < %s; ++k) {" % ("  " * ind_, ext))
                _emit_row_dense_write(lines_, ind_ + 1, plan, env, inner.name, "k")
                lines_.append("%s}" % ("  " * ind_))

        def body(lines_, ind_, env):
            _emit_compressed_loop(lines_, ind_, plan, env, unroll, pbody, marker=None)
        emit_row_loops(body)

    elif plan.kind == "sparse_out":
        ext = _extent_for(plan, plan.reduce.index.name)

        def pbody(lines_, ind_, env):
            env["acc"] = "acc"
            lines_.append("%s%s acc = 0.0;" % ("  " * ind_, dt))

            def vfn(kexpr):
                env3 = dict(env)
                env3["bind"] = dict(env["bind"])
                env3["bind"][plan.reduce.index.name] = kexpr
                return _cexpr(plan.value_expr, env3)

            _emit_dense_reduce_loop(lines_, ind_, ext, unroll, vfn, marker=None)
            lines_.append("%s%s[%s] = %s;" % ("  " * ind_, plan.out_tensor.name,
                                              env["p"], _cexpr(plan.final_expr, env)))
            env["acc"] = None

        def body(lines_, ind_, env):
            _emit_compressed_loop(lines_, ind_, plan, env, 1, pbody, marker=None)
        emit_row_loops(body)

    return "\n".join(lines)


def _emit_row_dense_write(lines, ind, plan, env, inner_name, kexpr):
    env3 = dict(env)
    env3["bind"] = dict(env["bind"])
    env3["bind"][inner_name] = kexpr
    env3["acc"] = None
    off = _offset(plan.out_tensor, plan.out.indices, env3)
    lines.append("%s%s[%s] += %s;" % ("  " * ind, plan.out_tensor.name, off,
                                      _cexpr(plan.value_expr, env3)))


def _extent_for(plan: Plan, index_name: str) -> str:
    """Resolve a dense index's extent from any tensor/access that carries it."""
    accesses: List[Access] = [plan.out]
    if plan.value_expr is not None:
        accesses += [n for n in plan.value_expr.walk() if isinstance(n, Access)]
    if plan.final_expr is not None and plan.final_expr is not plan.value_expr:
        accesses += [n for n in plan.final_expr.walk() if isinstance(n, Access)]
    for acc in accesses:
        t = acc.tensor
        if t.format.is_sparse():
            continue
        for pos, ix in enumerate(acc.indices):
            if ix.name == index_name and pos < len(t.shape):
                return str(t.shape[pos])
    return "K"


# --------------------------------------------------------------------------- #
# CUDA emitter
# --------------------------------------------------------------------------- #
def _cuda_emit(kernel: Kernel, config: Config, plan: Plan, decls: List[str]) -> str:
    dt = ctype(config.dtype)
    tile = config.tile_size
    unroll = config.unroll
    tpr = config.threads_per_row
    bug = config.inject_bug
    lines: List[str] = []

    row_bound = plan.row_extent
    if bug:
        row_bound = "(%s - 1)" % plan.row_extent

    kname = "%s_kernel" % kernel.name
    kopen = "__global__ void %s(%s) {" % (kname, ", ".join(decls))

    # ---- elementwise: one thread per element ----
    if plan.kind == "elementwise":
        i = plan.out.indices[0]
        lines.append(kopen)
        lines.append("  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;")
        lines.append("  if (i >= %s) return;" % row_bound)
        env = {"bind": {i.name: "i"}, "p": None, "data": "data", "acc": None}
        lines.append("  z[%s] = %s;" % (_offset(plan.out_tensor, plan.out.indices, env),
                                        _cexpr(plan.value_expr, env)))
        lines.append("}")
        return "\n".join(lines)

    # ---- scalar_reduce ----
    if plan.kind == "scalar_reduce":
        if tpr <= 1:
            lines.append(kopen)
            lines.append("  int64_t TILE = %d;" % tile)
            lines.append("  for (int64_t i0 = (int64_t)blockIdx.x * TILE; i0 < %s; "
                         "i0 += (int64_t)gridDim.x * TILE) {" % row_bound)
            lines.append("    for (int64_t ii = threadIdx.x; ii < TILE && i0 + ii < %s; "
                         "ii += blockDim.x) {" % row_bound)
            lines.append("      int64_t i = i0 + ii;")
            env = {"bind": {plan.row_idx.name: "i"}, "p": None, "data": "data", "acc": "acc"}
            lines.append("      %s acc = 0.0;" % dt)
            def inner(l, i_, e):
                l.append("%sacc += %s;" % ("  " * i_, _cexpr(plan.value_expr, e)))
            _emit_compressed_loop(lines, 3, plan, env, unroll, inner, marker="#pragma unroll")
            off = _out_offset(plan, env)
            lines.append("      %s[%s] = %s;" % (plan.out_tensor.name, off,
                                                _cexpr(plan.final_expr, env)))
            lines.append("    }")
            lines.append("  }")
            lines.append("}")
        else:
            lines.append(kopen)
            lines.append("  int64_t TILE = %d;" % tile)
            lines.append("  const int TPR = %d;" % tpr)
            lines.append("  int64_t lane = threadIdx.x % TPR;")
            lines.append("  int64_t slots = blockDim.x / TPR;")
            lines.append("  for (int64_t i0 = (int64_t)blockIdx.x * TILE; i0 < %s; "
                         "i0 += (int64_t)gridDim.x * TILE) {" % row_bound)
            lines.append("    for (int64_t slot = threadIdx.x / TPR; slot < TILE; slot += slots) {")
            lines.append("      int64_t i = i0 + slot;")
            lines.append("      if (i >= %s) continue;" % row_bound)
            lines.append("      %s acc = 0.0;" % dt)
            lines.append("      for (int64_t p = indptr[i] + lane; p < indptr[i + 1]; p += TPR) {")
            env = {"bind": {plan.row_idx.name: "i",
                            plan.col_idx.name: "indices[p]"},
                   "p": "p", "data": "data", "acc": "acc"}
            lines.append("        acc += %s;" % _cexpr(plan.value_expr, env))
            lines.append("      }")
            lines.append("      #pragma unroll")
            lines.append("      for (int off = TPR / 2; off > 0; off >>= 1) {")
            lines.append("        acc += __shfl_down_sync(0xffffffffu, acc, off, TPR);")
            lines.append("      }")
            lines.append("      if (lane == 0) {")
            env2 = {"bind": {plan.row_idx.name: "i"}, "p": "p", "data": "data", "acc": "acc"}
            off = _out_offset(plan, env2)
            lines.append("        %s[%s] = %s;" % (plan.out_tensor.name, off,
                                                  _cexpr(plan.final_expr, env2)))
            lines.append("      }")
            lines.append("    }")
            lines.append("  }")
            lines.append("}")
        return "\n".join(lines)

    # ---- row_dense_out (spmm): block-per-row shared-memory tiling ----
    if plan.kind == "row_dense_out":
        inner = plan.inner_out[0]
        ext = _extent_for(plan, inner.name)
        if tile > 1:
            lines.append(kopen)
            lines.append("  __shared__ %s sB[%d];" % (dt, tile))
            lines.append("  int64_t i = blockIdx.x;")
            lines.append("  if (i >= %s) return;" % row_bound)
            lines.append("  for (int64_t p = indptr[i]; p < indptr[i + 1]; ++p) {")
            lines.append("    int64_t j = indices[p];")
            lines.append("    for (int64_t k0 = 0; k0 < %s; k0 += %d) {" % (ext, tile))
            lines.append("      for (int64_t kk = threadIdx.x; kk < %d && k0 + kk < %s; "
                         "kk += blockDim.x)" % (tile, ext))
            lines.append("        sB[kk] = B[j * %s + k0 + kk];" % ext)
            lines.append("      __syncthreads();")
            lines.append("      for (int64_t kk = threadIdx.x; kk < %d && k0 + kk < %s; "
                         "kk += blockDim.x)" % (tile, ext))
            lines.append("        C[i * %s + k0 + kk] += data[p] * sB[kk];" % ext)
            lines.append("      __syncthreads();")
            lines.append("    }")
            lines.append("  }")
            lines.append("}")
        else:
            lines.append(kopen)
            lines.append("  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;")
            lines.append("  if (i >= %s) return;" % row_bound)
            lines.append("  for (int64_t p = indptr[i]; p < indptr[i + 1]; ++p) {")
            lines.append("    int64_t j = indices[p];")
            env = {"bind": {plan.row_idx.name: "i", plan.col_idx.name: "j"}, "p": "p",
                   "data": "data", "acc": None}
            # inner dense loop
            lines.append("    #pragma unroll")
            lines.append("    for (int64_t k = 0; k < %s; ++k) {" % ext)
            env["bind"][inner.name] = "k"
            off = _offset(plan.out_tensor, plan.out.indices, env)
            lines.append("      C[%s] += %s;" % (off, _cexpr(plan.value_expr, env)))
            lines.append("    }")
            lines.append("  }")
            lines.append("}")
        return "\n".join(lines)

    # ---- sparse_out (sddmm): block-per-row with shared reduction ----
    if plan.kind == "sparse_out":
        ext = _extent_for(plan, plan.reduce.index.name)
        lines.append(kopen)
        lines.append("  extern __shared__ %s sred[];" % dt)
        lines.append("  int64_t i = blockIdx.x;")
        lines.append("  if (i >= %s) return;" % row_bound)
        lines.append("  for (int64_t p = indptr[i]; p < indptr[i + 1]; ++p) {")
        lines.append("    int64_t j = indices[p];")
        lines.append("    %s acc = 0.0;" % dt)
        lines.append("    for (int64_t k = threadIdx.x; k < %s; k += blockDim.x)" % ext)
        env = {"bind": {plan.row_idx.name: "i", plan.col_idx.name: "j",
                        plan.reduce.index.name: "k"}, "p": "p", "data": "data", "acc": None}
        lines.append("      acc += %s;" % _cexpr(plan.value_expr, env))
        lines.append("    sred[threadIdx.x] = acc;")
        lines.append("    __syncthreads();")
        lines.append("    for (int s = blockDim.x / 2; s > 0; s >>= 1) {")
        lines.append("      if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];")
        lines.append("      __syncthreads();")
        lines.append("    }")
        lines.append("    if (threadIdx.x == 0) {")
        env2 = {"bind": {}, "p": "p", "data": "data", "acc": "sred[0]"}
        lines.append("      C[p] = %s;" % _cexpr(plan.final_expr, env2))
        lines.append("    }")
        lines.append("    __syncthreads();")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    raise NotImplementedError("no CUDA emitter for plan kind %r" % plan.kind)


# --------------------------------------------------------------------------- #
# File assembly
# --------------------------------------------------------------------------- #
def _header_comment(kernel: Kernel, config: Config, backend: str) -> str:
    lines = [
        "// TileGen generated kernel -- DO NOT EDIT BY HAND.",
        "// kernel      : %s" % kernel.name,
        "// backend     : %s" % backend,
        "// config      : %s" % config.to_dict(),
        "// schedule    : %s" % ", ".join(
            _fmt_transform(t) for t in kernel.schedule) if kernel.schedule else "// schedule    : (default)",
        "// pass log    :",
    ]
    if kernel.pass_log:
        for entry in kernel.pass_log:
            lines.append("//   [%s] %s" % (entry.get("pass"), entry.get("action")))
    else:
        lines.append("//   (none)")
    return "\n".join(lines)


def _fmt_transform(t) -> str:
    from .ir import Fuse, Parallelize, Split, ThreadsPerRow, Tile, Unroll, Vectorize
    if isinstance(t, Tile):
        return "Tile(%s,%d)" % (t.index.name, t.size)
    if isinstance(t, Split):
        return "Split(%s,%d)" % (t.index.name, t.factor)
    if isinstance(t, Parallelize):
        return "Parallelize(%s,%s)" % (t.index.name, t.unit)
    if isinstance(t, Unroll):
        return "Unroll(%s,%d)" % (t.index.name, t.factor)
    if isinstance(t, Fuse):
        return "Fuse(%s,%s)" % (t.index_a.name, t.index_b.name)
    if isinstance(t, ThreadsPerRow):
        return "ThreadsPerRow(%d)" % t.k
    if isinstance(t, Vectorize):
        return "Vectorize(%d)" % t.width
    return t.kind


def _launch_wrapper(kernel: Kernel, config: Config, sig: str, plan: Plan,
                    arg_names: Dict[str, Any]) -> str:
    kname = "%s_kernel" % kernel.name
    call_args = []
    for kind, t in arg_names["arrays"]:
        if kind == "sparse":
            call_args += ["indptr", "indices", "data"]
        elif kind == "out_sparse":
            call_args.append(t.name)
        else:
            call_args.append(t.name)
    call_args += arg_names["scalars"]
    call_args += arg_names["dims"]
    has_sparse = any(k == "sparse" for k, _ in arg_names["arrays"])
    if has_sparse:
        call_args.append("nnz")
    threads = min(config.block_size, 1024)
    tile = config.tile_size if config.tile_size > 0 else 1
    rows = plan.row_extent
    lines = ["  int64_t threads = %d;" % threads]
    if plan.kind == "sparse_out":
        # The sddmm kernel always maps one block to one CSR row.
        lines.append("  int64_t blocks = %s;  // block-per-row mapping" % rows)
    elif plan.kind == "row_dense_out" and config.tile_size > 1:
        lines.append("  int64_t blocks = %s;  // block-per-row mapping" % rows)
    elif plan.kind == "scalar_reduce":
        # Each block processes TILE rows, so size the grid by TILE, not by blockDim.
        lines.append("  int64_t blocks = (%s + %d - 1) / %d;" % (rows, tile, tile))
    else:
        lines.append("  int64_t blocks = (%s + threads - 1) / threads;" % rows)
    extra = ""
    if plan.kind == "sparse_out":
        extra = ", (size_t)threads * sizeof(%s)" % ctype(config.dtype)
    lines.append("  %s<<<blocks, threads%s>>>(%s);" % (kname, extra, ", ".join(call_args)))
    lines.append("  cudaDeviceSynchronize();")
    return "\n".join(lines)


def _assemble(kernel: Kernel, config: Config, backend: str, body: str, sig: str,
              plan: Plan, arg_names: Dict[str, Any]) -> str:
    header = _header_comment(kernel, config, backend)
    if backend == "cuda":
        pieces = [
            header,
            "#include <cuda_runtime.h>",
            "#include <cstdint>",
            "#include <cstddef>",
            "",
            body,
            "",
            'extern "C" %s {' % sig,
        ]
        wrapper = _launch_wrapper(kernel, config, sig, plan, arg_names)
        pieces += wrapper.splitlines()
        pieces.append("}")
        pieces.append("")
        return "\n".join(pieces)
    else:
        pieces = [
            header,
            "#include <cstdint>",
            "#include <cstddef>",
            "",
            'extern "C" %s {' % sig,
            body,
            "}",
            "",
        ]
        return "\n".join(pieces)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def emit_cpu(kernel: Kernel, config: Config) -> str:
    plan = _plan(kernel)
    sig, arg_names, _ = _abi(kernel)
    body = _cpu_emit(kernel, config, plan)
    return _assemble(kernel, config, "cpu", body, sig, plan, arg_names)


def emit_cuda(kernel: Kernel, config: Config) -> str:
    plan = _plan(kernel)
    sig, arg_names, _ = _abi(kernel)
    body = _cuda_emit(kernel, config, plan, arg_names["decls"])
    return _assemble(kernel, config, "cuda", body, sig, plan, arg_names)


def generate(
    kernel: Kernel,
    config: Config,
    backend: str = "cpu",
    outdir: Optional[str] = "build/generated",
    write: bool = True,
) -> GeneratedKernel:
    """Optimize the kernel with the pass pipeline and emit source for ``backend``."""
    optimized = _passes.run_pipeline(kernel, config, backend=backend)
    if backend == "cuda":
        source = emit_cuda(optimized, config)
        ext = "cu"
    else:
        source = emit_cpu(optimized, config)
        ext = "cpp"
    sig, _, _ = _abi(optimized)
    filename = "%s_%s.%s" % (optimized.name, config.hash(), ext)
    gen = GeneratedKernel(
        source=source,
        filename=filename,
        abi_signature=sig,
        pass_log=optimized.pass_log,
        backend=backend,
        kernel_name=optimized.name,
        config=config,
    )
    if write and outdir:
        gen.write(outdir)
    return gen
