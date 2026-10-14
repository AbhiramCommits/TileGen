"""User-facing DSL frontend.

Kernels read like math through operator overloading::

    i, j = Index("i"), Index("j")
    A = sparse_tensor("A", CSR, shape=("M", "N"))
    x = dense_tensor("x", shape=("N",))
    y = dense_tensor("y", shape=("M",))
    spmv = kernel("spmv", [assign(y[i], reduce_sum(j, A[i, j] * x[j]))])

Assignment sugar is either explicit -- ``assign(lhs, rhs)`` -- or the overloaded
``==`` operator: ``y[i] == reduce_sum(j, A[i, j] * x[j])``.  Both produce an
``ir.Stmt``.  ``reduce_sum`` (not ``sum``) avoids shadowing the builtin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .ir import (
    Access,
    Add,
    BSR,
    COO,
    CSC,
    CSR,
    Const,
    DENSE1,
    DENSE2,
    ELL,
    Format,
    Fuse,
    Index,
    Kernel,
    Mul,
    Parallelize,
    Reduce,
    SCALAR,
    Schedule,
    Split,
    Stmt,
    TensorDecl,
    ThreadsPerRow,
    Tile,
    Transform,
    Unroll,
    Vectorize,
    Assign,
)

__all__ = [
    "Index",
    "Tensor",
    "sparse_tensor",
    "dense_tensor",
    "scalar_tensor",
    "tensor",
    "reduce_sum",
    "assign",
    "kernel",
    "standard_kernels",
    "standard_kernel_names",
    "CSR",
    "CSC",
    "COO",
    "ELL",
    "BSR",
    "DENSE1",
    "DENSE2",
    "SCALAR",
    "Schedule",
    "Tile",
    "Split",
    "Parallelize",
    "Unroll",
    "Fuse",
    "ThreadsPerRow",
    "Vectorize",
    "i",
    "j",
    "k",
]


class Tensor:
    """A tensor wrapper providing index notation and scalar arithmetic."""

    def __init__(self, decl: TensorDecl):
        self.decl = decl

    # -- indexing ---------------------------------------------------------- #
    def __getitem__(self, idx: Union[Index, Tuple[Index, ...]]) -> Access:
        if isinstance(idx, tuple):
            indices = list(idx)
        else:
            indices = [idx]
        return Access(self.decl, indices)

    # -- scalar arithmetic (order-0 tensors) ------------------------------- #
    def __mul__(self, other: Any):
        return Access(self.decl, []) * other

    def __rmul__(self, other: Any):
        return other * Access(self.decl, [])

    def __add__(self, other: Any):
        return Access(self.decl, []) + other

    def __radd__(self, other: Any):
        return other + Access(self.decl, [])

    @property
    def name(self) -> str:
        return self.decl.name

    @property
    def shape(self) -> Tuple[Any, ...]:
        return self.decl.shape

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Tensor(%s, %s)" % (self.decl.name, self.decl.format.name)


def tensor(name: str, fmt: Format, shape: Sequence[Any] = (), dtype: str = "f64") -> Tensor:
    """Build a tensor with an explicit format/order."""
    order = len(fmt.levels)
    return Tensor(TensorDecl(name=name, order=order, format=fmt, dtype=dtype,
                             shape=tuple(shape)))


def sparse_tensor(name: str, fmt: Format, shape: Sequence[Any] = (),
                  dtype: str = "f64") -> Tensor:
    if not fmt.is_sparse():
        raise ValueError("sparse_tensor requires a compressed format, got %r" % fmt.name)
    return tensor(name, fmt, shape, dtype)


def dense_tensor(name: str, shape: Sequence[Any] = (), dtype: str = "f64") -> Tensor:
    order = len(shape)
    fmt = {0: SCALAR, 1: DENSE1, 2: DENSE2}.get(order)
    if fmt is None:
        fmt = Format("dense%d" % order, tuple(["dense"] * order))
    return tensor(name, fmt, shape, dtype)


def scalar_tensor(name: str, dtype: str = "f64") -> Tensor:
    return tensor(name, SCALAR, (), dtype)


# --------------------------------------------------------------------------- #
# Expression helpers
# --------------------------------------------------------------------------- #
def reduce_sum(index: Index, body: Any) -> Reduce:
    """``sum(index, body)``; named ``reduce_sum`` to avoid shadowing builtins."""
    return Reduce(index=index, body=_as_expr(body))


def assign(lhs: Access, rhs: Any) -> Stmt:
    """Build an assignment statement."""
    return Stmt(lhs=lhs, rhs=_as_expr(rhs))


def _as_expr(value: Any):
    if isinstance(value, Stmt):
        return value.rhs
    if hasattr(value, "decl") and isinstance(value, Tensor):
        return Access(value.decl, [])
    if isinstance(value, (int, float)):
        return Const(float(value))
    return value


def kernel(
    name: str,
    stmts: Union[Stmt, Sequence[Stmt]],
    schedule: Optional[Schedule] = None,
) -> Kernel:
    """Construct a validated kernel from one or more statements."""
    if isinstance(stmts, Stmt):
        stmt_list = [stmts]
    else:
        stmt_list = list(stmts)

    formats: List[Format] = []
    dtypes: List[str] = []
    seen = set()
    for st in stmt_list:
        for node in list(st.rhs.walk()) + [st.lhs]:
            if isinstance(node, Access):
                t = node.tensor
                if t.name not in seen:
                    seen.add(t.name)
                    formats.append(t.format)
                    dtypes.append(t.dtype)

    k = Kernel(name=name, stmts=stmt_list,
               schedule=schedule if schedule is not None else Schedule(),
               formats=formats, dtypes=dtypes)
    k.validate()
    return k


# --------------------------------------------------------------------------- #
# Standard kernels
# --------------------------------------------------------------------------- #
i = Index("i")
j = Index("j")
k = Index("k")

# Dimension symbols shared by the standard kernels.
M, N, K = "M", "N", "K"


def _spmv() -> Kernel:
    A = sparse_tensor("A", CSR, shape=(M, N))
    x = dense_tensor("x", shape=(N,))
    y = dense_tensor("y", shape=(M,))
    return kernel("spmv", [assign(y[i], reduce_sum(j, A[i, j] * x[j]))])


def _spmm() -> Kernel:
    A = sparse_tensor("A", CSR, shape=(M, N))
    B = dense_tensor("B", shape=(N, K))
    C = dense_tensor("C", shape=(M, K))
    return kernel("spmm", [assign(C[i, k], reduce_sum(j, A[i, j] * B[j, k]))])


def _sddmm() -> Kernel:
    A = sparse_tensor("A", CSR, shape=(M, N))
    U = dense_tensor("U", shape=(M, K))
    V = dense_tensor("V", shape=(N, K))
    C = sparse_tensor("C", CSR, shape=(M, N))
    inner = reduce_sum(k, U[i, k] * V[j, k])
    return kernel("sddmm", [assign(C[i, j], A[i, j] * inner)])


def _fused_spmv_axpy() -> Kernel:
    A = sparse_tensor("A", CSR, shape=(M, N))
    x = dense_tensor("x", shape=(N,))
    w = dense_tensor("w", shape=(M,))
    z = dense_tensor("z", shape=(M,))
    alpha = scalar_tensor("alpha")
    beta = scalar_tensor("beta")
    t = dense_tensor("t", shape=(M,))
    stmt1 = assign(t[i], reduce_sum(j, A[i, j] * x[j]))
    stmt2 = assign(z[i], alpha * t[i] + beta * w[i])
    return kernel("fused_spmv_axpy", [stmt1, stmt2])


def _axpy() -> Kernel:
    x = dense_tensor("x", shape=(M,))
    w = dense_tensor("w", shape=(M,))
    z = dense_tensor("z", shape=(M,))
    alpha = scalar_tensor("alpha")
    beta = scalar_tensor("beta")
    return kernel("axpy", [assign(z[i], alpha * x[i] + beta * w[i])])


def standard_kernel_names() -> List[str]:
    return ["spmv", "spmm", "sddmm", "fused_spmv_axpy", "axpy"]


def standard_kernels() -> Dict[str, Kernel]:
    """The kernels required to be expressible in the DSL."""
    return {
        "spmv": _spmv(),
        "spmm": _spmm(),
        "sddmm": _sddmm(),
        "fused_spmv_axpy": _fused_spmv_axpy(),
        "axpy": _axpy(),
    }
