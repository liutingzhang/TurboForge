"""Read-only deployment diagnostics, including an actionable support boundary."""
from __future__ import annotations
import importlib.util
import platform
from pathlib import Path
import shutil
import sys
from .bootstrap import cuda_package, local_tools


def check(root: Path) -> dict:
    errors = []
    try:
        package = cuda_package()
    except RuntimeError as exc:
        package = None
        errors.append(str(exc))
    try:
        tools = local_tools()
    except (RuntimeError, OSError, ValueError) as exc:
        tools = None
        errors.append(str(exc))
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        errors.append("The project disk has less than 1 GiB free; move the whole folder to a disk with enough space and retry")
    return {"ready": not errors, "errors": errors, "platform": platform.platform(),
            "python": sys.version.split()[0], "cuda_package": package, "disk_free_bytes": free,
            "runtime": str(tools[1]) if tools else "The official engine is downloaded automatically on first conversion; unreleased adaptations require the bundle",
            "dependencies": {name: "installed" if importlib.util.find_spec(name) else "installed on demand"
                             for name in ("flask", "torch", "numpy", "safetensors", "transformers")}}
