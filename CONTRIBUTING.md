# Contributing to TileGen

Thanks for helping. TileGen has three extension points — a new **format**, a new **pass**, and a
new **kernel**. Each one is a small, well-scoped change touching a known set of files. Run
`make test` (Catch2 + pytest) and `python scripts/verify_all.py` before opening a PR; every new
kernel must be covered by a differential test against a reference.

## Add a new storage format (e.g. ELL, BSR, COO)

Three files to touch:

1. `tilegen/ir.py` — declare the format and register it. Add a `Format(...)` constant with the
   correct `levels` (`"dense"` / `"compressed"`) and any parameters (`ell_width`, `block_size`),
   then add it to `FORMAT_REGISTRY`. `Kernel.validate()` already checks that a format's level
   count matches the tensor order.
2. `tilegen/codegen.py` — teach the emitters to iterate the new layout. Extend `_plan` (how the
   compressed axis is discovered) and the compressed-loop builder (`_emit_compressed_loop`) /
   `_cuda_emit` to read the format's index arrays (e.g. `ell_width` for ELL, `block_size` for
   BSR). Keep the `extern "C" void launch_<name>(...)` ABI stable.
3. `tilegen/dsl.py` — expose a constructor helper (like `sparse_tensor`) so kernels can declare
   tensors in the new format. Add a reference implementation in `cpp/src/ref.cpp` +
   `cpp/include/tilegen/ref.hpp` and a Catch2 test in `cpp/tests/test_ref.cpp`.

Also add the format to the corpus/probing fixtures if you want it autotuned
(`tilegen/matrices.py`).

## Add a new compiler pass

Three files to touch:

1. `tilegen/passes.py` — write `def my_pass(kernel: Kernel) -> Kernel` that reads
   `getattr(kernel, "_config", None)` / `getattr(kernel, "_backend", "cpu")`, records an entry
   with `_log(kernel, "my_pass", action, before, after)`, and returns the kernel. Add it to the
   `PASSES` list and `PASS_NAMES` in the right position. Passes must be idempotent on canonical IR.
2. `tilegen/codegen.py` — if the pass changes the loop structure, make the generator honour the
   corresponding transform (e.g. a new `Config` field and emission of the transformed loop nest).
3. `tests/test_passes.py` — assert the pass changes the emitted source as expected, refuses the
   transformation on an unsafe case, and is idempotent.

## Add a new kernel

Three files to touch:

1. `tilegen/dsl.py` — build the kernel in index notation in `standard_kernels()` (and
   `standard_kernel_names()`), e.g. `y[i] = reduce_sum(j, A[i, j] * x[j])`. `kernel(...)` runs
   `validate()` for you.
2. `tilegen/codegen.py` — make sure `_plan`/`_abi` produce a sensible plan and ABI for the new
   expression shape, and add the kernel's `ctypes` signature to `_SIGNATURES` in
   `tilegen/verify.py`.
3. `tilegen/verify.py` + `cpp/src/ref.cpp` — add a numpy/scipy reference in `_reference` and a
   plainly-written C++ reference, then add the kernel to `KERNELS` in `scripts/verify_all.py`
   and to the autotuner's kernel list. Add a small hand-checked unit test in `cpp/tests/test_ref.cpp`
   and a pytest case in `tests/test_verify.py`.

## Style and ground rules

- C++17, no external sparse libraries (numpy/scipy only on the Python side).
- The CPU backend must always work with no GPU and no `nvcc`; CUDA must degrade to text-level
  checks, never to a crash or a fabricated number.
- Never invent benchmark numbers. If hardware is unavailable, say so.
- Every pass and kernel change needs a test; run `make all BUDGET=6` to confirm the whole pipeline
  still runs green.
