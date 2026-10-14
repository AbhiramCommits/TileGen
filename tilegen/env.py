"""Environment and backend detection.

The whole toolchain degrades gracefully: this module never raises.  If a CUDA
compiler or a visible GPU is absent we simply report that fact and the caller
falls back to the CPU backend.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from typing import Dict, List, Optional

__all__ = ["probe", "cpu_name", "num_cores", "pretty"]

_TIMEOUT = 10


def _run(cmd: List[str]) -> Optional[str]:
    """Run a command, returning stripped stdout or None on any failure."""
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=_TIMEOUT,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _nvcc_version() -> Optional[str]:
    """Return a compact nvcc version string, e.g. '12.4.131', or None."""
    exe = shutil.which("nvcc")
    if exe is None:
        return None
    text = _run([exe, "--version"])
    if not text:
        return None
    m = re.search(r"release\s+([0-9]+\.[0-9]+)(?:,\s*V([0-9]+\.[0-9]+\.[0-9]+))?", text)
    if m:
        return m.group(2) if m.group(2) else m.group(1)
    m = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", text)
    return m.group(1) if m else text.splitlines()[0]


def _gpu_info() -> Dict[str, object]:
    """Query nvidia-smi for the first GPU's name and compute capability."""
    info: Dict[str, object] = {"has_gpu": False, "gpu_name": None, "sm_arch": None}
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return info
    text = _run([exe, "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if not text:
        return info
    first = text.splitlines()[0].strip()
    if not first:
        return info
    parts = [p.strip() for p in first.split(",")]
    if len(parts) >= 2 and parts[1]:
        cap = parts[1]
        sm = cap.replace(".", "")
        info["has_gpu"] = True
        info["gpu_name"] = parts[0]
        info["sm_arch"] = "sm_%s" % sm
    elif parts and parts[0]:
        info["has_gpu"] = True
        info["gpu_name"] = parts[0]
    return info


def cpu_name() -> str:
    """Best-effort human readable CPU model name."""
    system = platform.system()
    try:
        if system == "Darwin":
            text = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if text:
                return text
            # Apple Silicon reports the model under hw.model; brand_string may be empty.
            model = _run(["sysctl", "-n", "hw.model"])
            return model or platform.processor() or "unknown"
        if system == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as fh:
                    for line in fh:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        return platform.processor() or platform.machine() or "unknown"
    except Exception:
        return "unknown"


def num_cores() -> int:
    try:
        n = os.cpu_count()
        return int(n) if n else 1
    except Exception:
        return 1


def probe() -> Dict[str, object]:
    """Detect the toolchain and pick a backend.

    Returns a dict with keys: has_nvcc, nvcc_version, has_gpu, gpu_name,
    sm_arch, cpu_name, num_cores, backend.
    """
    result: Dict[str, object] = {
        "has_nvcc": False,
        "nvcc_version": None,
        "has_gpu": False,
        "gpu_name": None,
        "sm_arch": None,
        "cpu_name": "unknown",
        "num_cores": 1,
        "backend": "cpu",
    }

    try:
        exe = shutil.which("nvcc")
        result["has_nvcc"] = exe is not None
        if exe is not None:
            result["nvcc_version"] = _nvcc_version() or "unknown"
    except Exception:
        pass

    try:
        result.update(_gpu_info())
    except Exception:
        pass

    try:
        result["cpu_name"] = cpu_name()
    except Exception:
        pass

    try:
        result["num_cores"] = num_cores()
    except Exception:
        pass

    # CUDA backend requires *both* a compiler and a visible device.
    if result["has_nvcc"] and result["has_gpu"]:
        result["backend"] = "cuda"
    else:
        result["backend"] = "cpu"

    return result


def pretty(info: Optional[Dict[str, object]] = None) -> str:
    """Render a probe dict as a small human-readable block."""
    if info is None:
        info = probe()
    lines = [
        "TileGen environment probe",
        "-------------------------",
        "backend      : %s" % info.get("backend"),
        "cpu_name     : %s" % info.get("cpu_name"),
        "num_cores    : %s" % info.get("num_cores"),
        "has_nvcc     : %s" % info.get("has_nvcc"),
        "nvcc_version : %s" % info.get("nvcc_version"),
        "has_gpu      : %s" % info.get("has_gpu"),
        "gpu_name     : %s" % info.get("gpu_name"),
        "sm_arch      : %s" % info.get("sm_arch"),
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(probe(), indent=2))
