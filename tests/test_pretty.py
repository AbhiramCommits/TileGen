"""Snapshot tests for ``Kernel.pretty()``."""

from __future__ import annotations

import pytest

from tilegen.dsl import standard_kernels


SNAPSHOTS = {
    "spmv": """kernel spmv {
  assign y[i] = sum(j, A[i, j] * x[j])
  // loop nest
  for i:
    for j in reduce(A row):
      assign y[i] += A[i, j] * x[j]
}""",
    "fused_spmv_axpy": """kernel fused_spmv_axpy {
  assign t[i] = sum(j, A[i, j] * x[j])
  assign z[i] = alpha * t[i] + beta * w[i]
  // loop nest
  for i:
    for j in reduce(A row):
      assign t[i] += A[i, j] * x[j]
    assign z[i] = alpha * t[i] + beta * w[i]
}""",
}


@pytest.mark.parametrize("name", sorted(SNAPSHOTS))
def test_pretty_loop_nest_snapshot(name):
    assert standard_kernels()[name].pretty() == SNAPSHOTS[name]
