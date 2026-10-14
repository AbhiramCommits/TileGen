"""Tests for the DSL frontend and the TileGen IR."""

from __future__ import annotations

import pytest

from tilegen.dsl import (
    CSR,
    DENSE1,
    Access,
    Index,
    Kernel,
    Mul,
    Reduce,
    Schedule,
    Stmt,
    TensorDecl,
    Tile,
    Unroll,
    Vectorize,
    assign,
    dense_tensor,
    kernel,
    reduce_sum,
    scalar_tensor,
    sparse_tensor,
    standard_kernels,
)
from tilegen.ir import Format


def test_build_all_standard_kernels():
    ks = standard_kernels()
    for name in ["spmv", "spmm", "sddmm", "fused_spmv_axpy", "axpy"]:
        assert name in ks
        assert ks[name].validate() is True


def test_spmv_ir_shape():
    k = standard_kernels()["spmv"]
    assert isinstance(k.stmt, Stmt)
    assert isinstance(k.stmt.rhs, Reduce)
    assert k.stmt.rhs.index.name == "j"
    assert isinstance(k.stmt.rhs.body, Mul)
    assert [idx.name for idx in k.free_indices()] == ["i"]
    assert [idx.name for idx in k.reduce_indices()] == ["j"]
    assert k.sparse_tensor().format.name == "csr"


def test_sddmm_ir_shape():
    k = standard_kernels()["sddmm"]
    inner = k.stmt.rhs
    assert isinstance(inner, Mul)
    assert isinstance(inner.right, Reduce)
    assert inner.right.index.name == "k"
    assert sorted(i.name for i in k.free_indices()) == ["i", "j"]


def test_fused_has_two_statements():
    k = standard_kernels()["fused_spmv_axpy"]
    assert len(k.stmts) == 2
    assert k.stmts[0].lhs.tensor.name == "t"
    assert k.stmts[1].lhs.tensor.name == "z"


def test_eq_sugar_builds_assignment():
    i, j = Index("i"), Index("j")
    A = sparse_tensor("A", CSR, shape=("M", "N"))
    x = dense_tensor("x", shape=("N",))
    y = dense_tensor("y", shape=("M",))
    stmt = y[i] == reduce_sum(j, A[i, j] * x[j])
    assert isinstance(stmt, Stmt)
    k = kernel("spmv_sugar", [stmt])
    assert k.validate() is True


def test_pretty_is_readable():
    for k in standard_kernels().values():
        text = k.pretty()
        assert "kernel" in text
        assert "for" in text
        if k.reduce_node() is not None:
            assert "sum(" in text


def test_roundtrip_to_from_dict():
    for k in standard_kernels().values():
        d = k.to_dict()
        k2 = Kernel.from_dict(d)
        assert k2.to_dict() == d


# --------------------------------------------------------------------------- #
# Validation failures
# --------------------------------------------------------------------------- #
def test_reject_unbound_lhs_index():
    i, j, kk = Index("i"), Index("j"), Index("kk")
    A = sparse_tensor("A", CSR, shape=("M", "N"))
    x = dense_tensor("x", shape=("N",))
    y = dense_tensor("y", shape=("M",))
    bad = assign(y[kk], reduce_sum(j, A[i, j] * x[j]))
    with pytest.raises(ValueError):
        kernel("bad_unbound", [bad])


def test_reject_format_order_mismatch():
    # A CSR tensor declared with two indices is fine; one index must fail.
    A = sparse_tensor("A", CSR, shape=("M", "N"))
    x = dense_tensor("x", shape=("N",))
    i = Index("i")
    bad_access = Access(A.decl, [i])  # rank 1 for an order-2 tensor
    bad = Stmt(lhs=x[i], rhs=bad_access)
    with pytest.raises(ValueError):
        kernel("bad_rank", [bad])

    # A tensor whose declared order disagrees with its format order.
    broken = TensorDecl(name="Q", order=2, format=DENSE1, dtype="f64", shape=("M", "N"))
    a = Access(broken, [i, i])
    b = dense_tensor("b", shape=("M", "N"))
    st = Stmt(lhs=b[i, i], rhs=a)
    with pytest.raises(ValueError):
        kernel("bad_fmt_order", [st])


def test_reject_unknown_schedule_index():
    k = standard_kernels()["spmv"]
    k.schedule = Schedule([Tile(Index("zz"), 32)])
    with pytest.raises(ValueError):
        k.validate()


def test_reject_bad_unroll_factor():
    k = standard_kernels()["spmv"]
    k.schedule = Schedule([Unroll(Index("i"), 0)])
    with pytest.raises(ValueError):
        k.validate()
    k.schedule = Schedule([Vectorize(0)])
    with pytest.raises(ValueError):
        k.validate()


def test_validate_returns_true_for_good_kernel():
    assert standard_kernels()["spmm"].validate() is True
