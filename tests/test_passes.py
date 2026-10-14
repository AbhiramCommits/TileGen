"""Tests for the pass pipeline and the code generator."""

from __future__ import annotations

from tilegen.codegen import Config, emit_cpu, emit_cuda, generate
from tilegen.dsl import (
    CSR,
    Index,
    assign,
    dense_tensor,
    kernel,
    reduce_sum,
    scalar_tensor,
    sparse_tensor,
    standard_kernels,
)
from tilegen import passes


def _run(k, cfg=None, backend="cpu"):
    return passes.run_pipeline(k, cfg or Config(), backend=backend)


def test_fusion_collapses_two_loop_nests():
    k = standard_kernels()["fused_spmv_axpy"]
    assert len(k.stmts) == 2
    out = _run(k)
    assert len(out.stmts) == 1
    fusion = [e for e in out.pass_log if e["pass"] == "loop_fusion"]
    assert fusion and "fused" in fusion[0]["action"]
    # the temporary 't' is gone from the fused expression
    reads = [n.tensor.name for st in out.stmts for n in st.rhs.walk()
             if hasattr(n, "tensor")]
    assert "t" not in reads


def test_fusion_refused_on_true_dependency():
    i, j, kk = Index("i"), Index("j"), Index("kk")
    A = sparse_tensor("A", CSR, shape=("M", "N"))
    x = dense_tensor("x", shape=("N",))
    w = dense_tensor("w", shape=("M",))
    t = dense_tensor("t", shape=("M",))
    z = dense_tensor("z", shape=("M",))
    alpha = scalar_tensor("alpha")
    beta = scalar_tensor("beta")
    stmt1 = assign(t[i], reduce_sum(j, A[i, j] * x[j]))
    # read t at a *different* index than it was written -> true dependency
    stmt2 = assign(z[kk], alpha * t[kk] + beta * w[kk])
    k = kernel("dependent", [stmt1, stmt2])
    out = _run(k)
    assert len(out.stmts) == 2
    fusion = [e for e in out.pass_log if e["pass"] == "loop_fusion"]
    assert fusion and "refused" in fusion[0]["action"]


def test_tiling_changes_emitted_source():
    k = standard_kernels()["spmv"]
    base = emit_cpu(passes.run_pipeline(k, Config(tile_size=1)), Config(tile_size=1))
    tiled = emit_cpu(passes.run_pipeline(k, Config(tile_size=64)), Config(tile_size=64))
    assert "i_o" not in base
    assert "i_o" in tiled
    assert base != tiled


def test_unroll_changes_emitted_source():
    k = standard_kernels()["spmv"]
    base = emit_cpu(passes.run_pipeline(k, Config(unroll=1)), Config(unroll=1))
    unrolled = emit_cpu(passes.run_pipeline(k, Config(unroll=4)), Config(unroll=4))
    assert unrolled.count("acc += ") > base.count("acc += ")
    assert "remainder" not in base.lower() or base != unrolled
    assert unrolled != base


def test_cuda_markers():
    k = standard_kernels()["spmv"]
    cfg = Config(tile_size=32, threads_per_row=4, unroll=4, block_size=256)
    src = emit_cuda(passes.run_pipeline(k, cfg, backend="cuda"), cfg)
    assert "__global__" in src
    assert "__shfl_down_sync" in src
    assert "#pragma unroll" in src
    assert "extern \"C\" void launch_spmv" in src


def test_cuda_no_shuffle_when_tpr_one():
    k = standard_kernels()["spmv"]
    cfg = Config(threads_per_row=1, unroll=1)
    src = emit_cuda(passes.run_pipeline(k, cfg, backend="cuda"), cfg)
    assert "__shfl_down_sync" not in src


def test_pipeline_idempotent_on_canonical_ir():
    k = standard_kernels()["spmv"]
    cfg = Config(tile_size=64, unroll=2, threads_per_row=1)
    once = _run(k, cfg)
    twice = _run(once, cfg)
    d1 = once.to_dict()
    d2 = twice.to_dict()
    d1.pop("pass_log")
    d2.pop("pass_log")
    assert d1 == d2


def test_canonicalize_reports_no_change_second_time():
    k = standard_kernels()["spmv"]
    first = passes.canonicalize(k)
    second = passes.canonicalize(first)
    assert "no change" in second.pass_log[-1]["action"]


def test_generate_writes_file(tmp_path):
    k = standard_kernels()["axpy"]
    gen = generate(k, Config(), "cpu", outdir=str(tmp_path), write=True)
    import os
    assert os.path.exists(str(tmp_path / gen.filename))
    assert gen.abi_signature.startswith("void launch_axpy")
