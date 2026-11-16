from __future__ import annotations

from .assistant.config import child_environment
import ctypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuStatus:
    name: str
    total_bytes: int
    free_bytes: int


def memory_available() -> int | None:
    if not hasattr(ctypes, "windll"):
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)
    except (OSError, AttributeError):
        pass
    return None


def _smi_fields(columns: str, executable: str, expected: int) -> list[str] | None:
    """First GPU's requested CSV fields, or None when the probe is unusable."""
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={columns}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
            env=child_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    fields = [field.strip() for field in line.split(",")]
    return fields if len(fields) == expected else None


def gpu_status(executable: str = "nvidia-smi") -> GpuStatus | None:
    fields = _smi_fields("name,memory.total,memory.free", executable, 3)
    if fields is None:
        return None
    try:
        return GpuStatus(fields[0], int(fields[1]) * 1024**2, int(fields[2]) * 1024**2)
    except ValueError:
        return None


def disk_available(path: str | Path) -> int:
    candidate = Path(path).expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free


# Compute capability -> marketing generation, for scheme ordering/labeling only.
# This is informational and deliberately broader than bootstrap.cuda_package()'s
# engine-download whitelist (which gates 7.0/7.5/8.0/8.6/8.9/12.0 and raises).
_GPU_GENERATIONS = {
    12.0: "Blackwell · RTX 50",
    10.0: "Blackwell · 数据中心",
    9.0: "Hopper",
    8.9: "Ada · RTX 40",
    8.6: "Ampere · RTX 30",
    8.0: "Ampere · A100",
    7.5: "Turing · RTX 20",
    7.0: "Volta",
}


def detect_gpu(executable: str = "nvidia-smi") -> dict | None:
    """Non-fatal GPU generation probe used to order/label quantization schemes.

    Returns None on any failure (no NVIDIA driver, timeout, malformed output) so
    AMD/CPU/headless machines degrade to bpw/quality ordering instead of erroring.
    Unlike cuda_package(), this never raises and never gates a download; it only
    reports hardware facts. No performance number is inferred here.
    """
    fields = _smi_fields("name,compute_cap,memory.total,driver_version", executable, 4)
    if fields is None:
        return None
    try:
        cap = float(fields[1])
        total = int(fields[2]) * 1024**2
    except ValueError:
        return None
    return {
        "name": fields[0],
        "generation": _GPU_GENERATIONS.get(cap, f"compute {cap:g}"),
        "compute_cap": cap,
        "total_bytes": total,
        "driver": fields[3],
    }
