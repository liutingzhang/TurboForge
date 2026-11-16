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
        errors.append("项目磁盘剩余空间不足 1 GiB，请将整个文件夹移到空间充足的磁盘后重试")
    return {"ready": not errors, "errors": errors, "platform": platform.platform(),
            "python": sys.version.split()[0], "cuda_package": package, "disk_free_bytes": free,
            "runtime": str(tools[1]) if tools else "首次转换自动下载官方引擎；未发布的适配需使用整合包",
            "dependencies": {name: "已安装" if importlib.util.find_spec(name) else "按需自动安装"
                             for name in ("flask", "torch", "numpy", "safetensors", "transformers")}}
