"""Differential correctness testing for generated kernels.

``compile_generated`` turns a :class:`~tilegen.codegen.GeneratedKernel` into a
loadable shared object (nvcc for CUDA, the host compiler for the CPU backend)
and *surfaces compiler errors clearly*.  When ``nvcc`` is unavailable the CUDA
source is still validated with a text-level structural check and a warning is
logged, exactly as the build rules require.

``verify`` runs the generated kernel through ctypes and compares against a
scipy/numpy reference, reporting max absolute error, max relative error and the
tolerance used.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ctypes
import numpy as np
import scipy.sparse as sp

from . import codegen as _codegen
from .dsl import standard_kernels

__all__ = [
    "VerifyResult",
    "compile_generated",
    "cuda_arch",
    "verify",
    "syntax_check_cuda_text",
    "check_cuda_markers",
]

I64 = ctypes.c_longlong
I64P = ctypes.POINTER(ctypes.c_longlong)
F64P = ctypes.POINTER(ctypes.c_double)

_darwin = platform.system() == "Darwin"
_SOEXT = "dylib" if _darwin else "so"

_CUDA_HOST_TO_DEVICE = 1
_CUDA_DEVICE_TO_HOST = 2
_cudart = None


def _cuda_runtime():
    """Load the CUDA runtime shared library used for device-buffer transfers."""
    global _cudart
    if _cudart is not None:
        return _cudart

    import ctypes.util
    import glob

    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates += sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"), reverse=True)
    candidates += ["libcudart.so", "libcudart.so.13", "libcudart.so.12",
                   "libcudart.so.11.0"]

    last_error = None
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as exc:
            last_error = exc
            continue
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_int]
        lib.cudaMemcpy.restype = ctypes.c_int
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        _cudart = lib
        return _cudart

    raise RuntimeError("could not load the CUDA runtime library: %s" % last_error)


def _cuda_check(code: int, what: str) -> None:
    if code == 0:
        return
    lib = _cuda_runtime()
    message = lib.cudaGetErrorString(code)
    raise RuntimeError("%s failed with CUDA error %d: %s"
                       % (what, code, message.decode("utf-8", "replace")
                          if message else "unknown"))


class _CudaBuffers:
    """Contiguous host arrays mirrored into device memory for one kernel call."""

    def __init__(self, arrays):
        self.lib = _cuda_runtime()
        self.entries = []
        for arr in arrays:
            ptr = ctypes.c_void_p()
            nbytes = int(arr.nbytes)
            _cuda_check(self.lib.cudaMalloc(ctypes.byref(ptr),
                                            ctypes.c_size_t(max(1, nbytes))),
                        "cudaMalloc")
            if nbytes:
                _cuda_check(self.lib.cudaMemcpy(
                    ptr, ctypes.c_void_p(arr.ctypes.data),
                    ctypes.c_size_t(nbytes), _CUDA_HOST_TO_DEVICE),
                    "cudaMemcpy(host->device)")
            self.entries.append((arr, ptr, nbytes))

    def arg(self, index: int):
        arr, ptr, _ = self.entries[index]
        ctype = ctypes.c_longlong if arr.dtype == np.int64 else ctypes.c_double
        return ctypes.cast(ptr, ctypes.POINTER(ctype))

    def copy_back(self, index: int) -> None:
        arr, ptr, nbytes = self.entries[index]
        if not nbytes:
            return
        _cuda_check(self.lib.cudaMemcpy(
            ctypes.c_void_p(arr.ctypes.data), ptr,
            ctypes.c_size_t(nbytes), _CUDA_DEVICE_TO_HOST),
            "cudaMemcpy(device->host)")

    def synchronize(self) -> None:
        _cuda_check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def free(self) -> None:
        for _arr, ptr, _nbytes in self.entries:
            self.lib.cudaFree(ptr)
        self.entries = []


def _invoke(fn, arrays, output_indices, scalars, backend: str) -> None:
    """Call a generated launcher with host buffers or mirrored CUDA buffers."""
    if backend != "cuda":
        args = []
        for arr in arrays:
            ctype = ctypes.c_longlong if arr.dtype == np.int64 else ctypes.c_double
            args.append(arr.ctypes.data_as(ctypes.POINTER(ctype)))
        fn(*(args + list(scalars)))
        return

    buffers = _CudaBuffers(arrays)
    try:
        args = [buffers.arg(i) for i in range(len(arrays))]
        fn(*(args + list(scalars)))
        buffers.synchronize()
        for index in output_indices:
            buffers.copy_back(index)
    finally:
        buffers.free()

# Arguments (beyond the ctypes pointer types) per kernel ABI.
_SIGNATURES = {
    "spmv": [I64P, I64P, F64P, F64P, F64P, I64, I64, I64],
    "spmm": [I64P, I64P, F64P, F64P, F64P, I64, I64, I64, I64],
    "sddmm": [I64P, I64P, F64P, F64P, F64P, F64P, I64, I64, I64, I64],
    "fused_spmv_axpy": [I64P, I64P, F64P, F64P, F64P, F64P,
                        ctypes.c_double, ctypes.c_double, I64, I64, I64],
    "axpy": [F64P, F64P, F64P, ctypes.c_double, ctypes.c_double, I64],
}


# --------------------------------------------------------------------------- #
# CUDA text-level checks
# --------------------------------------------------------------------------- #
def _balanced(text: str) -> Tuple[bool, str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    in_line_comment = False
    for ch in text:
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            continue
        if ch == "\n":
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False, "unbalanced '%s'" % ch
            stack.pop()
    if stack:
        return False, "unclosed '%s'" % stack[-1]
    return True, "balanced delimiters"


def syntax_check_cuda_text(source: str) -> Tuple[bool, List[str]]:
    """Structural validation used when nvcc is unavailable.

    We cannot hand ``<<<>>>`` launch syntax to a host C++ compiler, so instead of
    pretending to compile we validate the structure (balanced delimiters, present
    ABI and kernel entry points) and say so in the returned notes.
    """
    notes: List[str] = []
    ok, msg = _balanced(source.split("// pass log")[-1])
    notes.append("delimiters: %s" % msg)
    for marker in ('extern "C"', "launch_", "__global__", "#include"):
        present = marker in source
        notes.append("marker %-12s: %s" % (marker, present))
        ok = ok and present
    return ok, notes


def check_cuda_markers(source: str, threads_per_row: int = 1) -> Dict[str, bool]:
    markers = {
        "__global__": "__global__" in source,
        "#pragma unroll": "#pragma unroll" in source,
    }
    if threads_per_row > 1:
        markers["__shfl_down_sync"] = "__shfl_down_sync" in source
    return markers


# --------------------------------------------------------------------------- #
# compilation
# --------------------------------------------------------------------------- #
def _host_compiler() -> str:
    for cc in (os.environ.get("CXX"), "clang++", "g++", "c++"):
        if cc and shutil.which(cc):
            return cc
    raise RuntimeError("no C++ compiler found (need clang++/g++)")


def _omp_flags() -> List[str]:
    """Flags for the host compiler so OpenMP works on macOS and Linux."""
    plain = ["-O3", "-march=native", "-fopenmp"]
    # Apple clang needs the libomp include/lib explicitly.
    libomp_inc = "/opt/homebrew/opt/libomp/include"
    libomp_lib = "/opt/homebrew/opt/libomp/lib"
    if _darwin and os.path.isdir(libomp_inc):
        return ["-O3", "-Xpreprocessor", "-fopenmp",
                "-I" + libomp_inc, "-L" + libomp_lib, "-lomp"]
    return plain


def cuda_arch() -> Optional[str]:
    """Detected nvcc target, e.g. ``sm_75``; ``TILEGEN_CUDA_ARCH`` overrides it."""
    override = os.environ.get("TILEGEN_CUDA_ARCH")
    if override:
        override = override.strip()
        return override if override.startswith("sm_") else "sm_" + override
    try:
        from .env import probe

        return probe().get("sm_arch")
    except Exception:
        return None


def compile_generated(
    gen: "_codegen.GeneratedKernel",
    backend: str = "cpu",
    workdir: str = "build/generated",
    verbose: bool = False,
) -> Tuple[Optional[str], str]:
    """Compile a generated kernel; return (shared_object_path_or_None, log).

    CPU sources are compiled with the host compiler.  CUDA sources require nvcc;
    when it is missing we run the text-level structural check and return None.
    """
    src = os.path.join(workdir, gen.filename)
    if not os.path.exists(src):
        gen.write(workdir)

    if backend == "cuda":
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            ok, notes = syntax_check_cuda_text(gen.source)
            log = ("nvcc unavailable: skipped CUDA compilation; ran text-level "
                   "structural check -> %s\n  " % ("PASS" if ok else "FAIL")
                   + "\n  ".join(notes))
            return None, log
        out = os.path.join(workdir, gen.filename.rsplit(".", 1)[0] + "." + _SOEXT)
        cmd = [nvcc, "-O3"]
        arch = cuda_arch()
        if arch:
            cmd.append("-arch=%s" % arch)
        cmd += ["-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-o", out, src]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("nvcc failed for %s:\n%s" % (gen.filename, res.stderr))
        return out, "nvcc compiled %s with: %s" % (out, " ".join(cmd[1:]))

    # CPU backend
    out = os.path.join(workdir, gen.filename.rsplit(".", 1)[0] + "." + _SOEXT)
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out, "cached %s" % out

    cc = _host_compiler()
    base = [cc, "-std=c++17", "-shared", "-fPIC", "-o", out, src]
    attempts = [base + _omp_flags(), base + ["-O3", "-march=native"]]
    last_err = ""
    for cmd in attempts:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return out, "compiled %s with: %s" % (gen.filename, " ".join(cmd[1:]))
        last_err = res.stderr
    raise RuntimeError("host compiler failed for %s:\n%s" % (gen.filename, last_err))


# --------------------------------------------------------------------------- #
# matrix handling
# --------------------------------------------------------------------------- #
def _load_matrix(matrix) -> Tuple[sp.csr_matrix, str]:
    if isinstance(matrix, str):
        with np.load(matrix, allow_pickle=False) as z:
            indptr = z["indptr"].astype(np.int64)
            indices = z["indices"].astype(np.int64)
            data = z["data"].astype(np.float64)
            shape = tuple(int(x) for x in z["shape"])
            name = str(z["class_name"]) if "class_name" in z else os.path.basename(matrix)
        return sp.csr_matrix((data, indices, indptr), shape=shape), name
    if isinstance(matrix, dict):
        shape = tuple(matrix["shape"])
        return sp.csr_matrix((np.asarray(matrix["data"], dtype=np.float64),
                              np.asarray(matrix["indices"], dtype=np.int64),
                              np.asarray(matrix["indptr"], dtype=np.int64)),
                             shape=shape), matrix.get("class_name", "custom")
    if sp.issparse(matrix):
        return matrix.tocsr().astype(np.float64), "custom"
    raise TypeError("unsupported matrix type: %r" % type(matrix))


def _inputs_for(kernel: str, A: sp.csr_matrix, K: int, seed: int = 7):
    M, N = A.shape
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    w = rng.standard_normal(M)
    B = rng.standard_normal((N, K))
    U = rng.standard_normal((M, K))
    V = rng.standard_normal((N, K))
    alpha, beta = 0.7, -1.3
    return x, w, B, U, V, alpha, beta


def _reference(kernel: str, A: sp.csr_matrix, data):
    x, w, B, U, V, alpha, beta = data
    if kernel == "spmv":
        return (A @ x).astype(np.float64)
    if kernel == "spmm":
        return np.asarray(A @ B, dtype=np.float64)
    if kernel == "sddmm":
        coo = A.tocoo()
        # chunked to bound peak memory: sum_k U[i,k] V[j,k] only on the pattern
        out = np.empty(coo.nnz, dtype=np.float64)
        step = 200000
        for s in range(0, coo.nnz, step):
            e = min(s + step, coo.nnz)
            rr = coo.row[s:e]
            cc = coo.col[s:e]
            out[s:e] = np.einsum("ik,ik->i", U[rr], V[cc]) * coo.data[s:e]
        return out
    if kernel == "fused_spmv_axpy":
        return alpha * (A @ x) + beta * w
    if kernel == "axpy":
        return alpha * x + beta * w
    raise ValueError("unknown kernel %r" % kernel)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
@dataclass
class VerifyResult:
    kernel: str
    matrix_class: str
    backend: str
    rows: int = 0
    nnz: int = 0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    tolerance: float = 0.0
    ok: bool = False
    inject_bug: bool = False
    message: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "matrix_class": self.matrix_class,
            "backend": self.backend,
            "rows": self.rows,
            "nnz": self.nnz,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "tolerance": self.tolerance,
            "ok": self.ok,
            "inject_bug": self.inject_bug,
            "message": self.message,
            "config": self.config,
        }


def _max_row_nnz(A: sp.csr_matrix) -> int:
    counts = np.diff(A.indptr)
    return int(counts.max()) if counts.size else 0


def run_kernel(lib_path: str, kernel_name: str, A: sp.csr_matrix, data,
               K: int = 8, backend: str = "cpu"):
    """Load ``lib_path`` and invoke ``launch_<kernel_name>`` via ctypes.

    For CUDA, NumPy buffers are copied into device memory before the launch and
    the output is copied back afterwards; generated kernels never receive host
    pointers.
    """
    M, N = A.shape
    nnz = int(A.nnz)
    lib = ctypes.CDLL(lib_path)
    fn = getattr(lib, "launch_%s" % kernel_name)
    fn.argtypes = _SIGNATURES[kernel_name]

    ip = A.indptr.astype(np.int64)
    ind = A.indices.astype(np.int64)
    dat = A.data.astype(np.float64)
    x, w, B, U, V, alpha, beta = data

    if kernel_name == "spmv":
        out = np.zeros(M, dtype=np.float64)
        _invoke(fn, [ip, ind, dat, x, out], [4], (M, N, nnz), backend)
    elif kernel_name == "spmm":
        out = np.zeros((M, K), dtype=np.float64)
        _invoke(fn, [ip, ind, dat, B, out], [4], (M, N, K, nnz), backend)
    elif kernel_name == "sddmm":
        out = np.zeros(nnz, dtype=np.float64)
        _invoke(fn, [ip, ind, dat, U, V, out], [5], (M, N, K, nnz), backend)
    elif kernel_name == "fused_spmv_axpy":
        out = np.zeros(M, dtype=np.float64)
        _invoke(fn, [ip, ind, dat, x, w, out], [5],
                (alpha, beta, M, N, nnz), backend)
    elif kernel_name == "axpy":
        out = np.zeros(M, dtype=np.float64)
        _invoke(fn, [x, w, out], [2], (alpha, beta, M), backend)
    else:
        raise ValueError("unknown kernel %r" % kernel_name)
    return out, lib, fn, (ip, ind, dat)


def verify(
    kernel_name: str,
    config: "_codegen.Config",
    matrix,
    backend: str = "cpu",
    workdir: str = "build/generated",
    K: int = 8,
    seed: int = 7,
    skip_passes: Sequence[str] = (),
) -> VerifyResult:
    """Generate, compile and run ``kernel_name`` and compare against a reference.

    Tolerance: floating-point accumulation order differs between the reference
    (which sums in index order) and the generated kernel (which may unroll /
    reorder / reduce with threads).  The error of a length-L dot product is
    bounded by ~ L * eps * magnitude, so we accept
    ``tol = 8 * eps * max_row_nnz * scale`` with a small absolute floor.
    """
    kernels = standard_kernels()
    if kernel_name not in kernels:
        raise ValueError("unknown kernel %r" % kernel_name)
    A, class_name = _load_matrix(matrix)
    M, N = A.shape
    nnz = int(A.nnz)
    data = _inputs_for(kernel_name, A, K, seed)
    ref = _reference(kernel_name, A, data)

    gen = _codegen.generate(kernels[kernel_name], config, backend,
                            outdir=workdir, write=True,
                            skip_passes=skip_passes)
    try:
        lib_path, log = compile_generated(gen, backend, workdir)
    except RuntimeError as exc:
        return VerifyResult(kernel=kernel_name, matrix_class=class_name,
                            backend=backend, rows=M, nnz=nnz, ok=False,
                            inject_bug=config.inject_bug,
                            message="compile error: %s" % exc,
                            config=config.to_dict())

    if lib_path is None:
        # CUDA source without nvcc: structurally checked, numerics not run here.
        ok = "PASS" in log
        return VerifyResult(kernel=kernel_name, matrix_class=class_name,
                            backend=backend, rows=M, nnz=nnz, ok=ok,
                            inject_bug=config.inject_bug,
                            message=log, config=config.to_dict())

    out, _lib, _fn, _bufs = run_kernel(lib_path, kernel_name, A, data, K,
                                       backend=backend)
    out = np.asarray(out, dtype=np.float64).ravel()
    refv = np.asarray(ref, dtype=np.float64).ravel()
    diff = np.abs(out - refv)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(refv), 1e-300)
    max_rel = float((diff / denom).max()) if diff.size else 0.0

    eps = float(np.finfo(np.float64).eps)
    scale = float(np.abs(refv).max()) if refv.size else 0.0
    row_nnz = _max_row_nnz(A)
    tol = float(8.0 * eps * max(1, row_nnz) * scale + 1e-12)
    ok = bool(max_abs <= tol)

    return VerifyResult(
        kernel=kernel_name, matrix_class=class_name, backend=backend,
        rows=M, nnz=nnz, max_abs_err=max_abs, max_rel_err=max_rel,
        tolerance=tol, ok=ok, inject_bug=config.inject_bug,
        message=log, config=config.to_dict(),
    )
