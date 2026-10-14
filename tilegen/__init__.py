"""TileGen: a mini DSL, compiler pass pipeline, and code generator for
sparse linear algebra kernels targeting CUDA and an OpenMP/serial CPU fallback.

Public entry points:
    tilegen.env        -- environment / backend probing
    tilegen.matrices   -- deterministic sparse matrix corpus generation
    tilegen.dsl        -- user-facing index-notation frontend
    tilegen.ir         -- TACO-style intermediate representation
    tilegen.passes     -- compiler optimization pass pipeline
    tilegen.codegen    -- CUDA / CPU source emission
    tilegen.verify     -- differential correctness harness
    tilegen.autotune   -- schedule autotuner
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
