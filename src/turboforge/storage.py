"""Explicit cleanup limited to failed, managed build outputs; never dependencies."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import stat

from .manifest import read_json_atomic

FAILED = {"failed", "blocked", "validation-failed", "quality-rejected", "interrupted"}


def _size(directory: Path) -> int:
    total = 0
    for item in directory.rglob("*"):
        info = item.lstat()
        if item.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("The build directory contains links; automatic deletion is not offered")
        if item.is_file():
            total += info.st_size
    return total


def inventory(root: Path) -> dict:
    root = root.resolve()
    items = []
    builds = root / "builds"
    if builds.is_dir() and builds.resolve().parent == root and not (getattr(builds.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT) and not builds.is_symlink():
        for folder in sorted(builds.iterdir()):
            if not folder.is_dir() or folder.resolve().parent != builds.resolve() or folder.is_symlink() or getattr(folder.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            manifest = folder / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                status = read_json_atomic(manifest).get("status", "unknown")
                size = _size(folder)
            except (OSError, ValueError):
                continue
            items.append({"id": folder.name, "status": status, "bytes": size, "removable": status in FAILED})
    return {"items": items, "free_bytes": shutil.disk_usage(root).free,
            "note": "Only builds managed by this project are listed. Only failed artifacts may be cleaned; successful models, original models, dependency environments and download caches are all kept."}


def remove_failed(root: Path, identifiers: list[str]) -> dict:
    if not identifiers or len(identifiers) > 100 or any(not isinstance(value, str) or not re.fullmatch(r"[\w.-]{1,180}", value) or value.startswith(".") for value in identifiers):
        raise ValueError("Select valid failed builds")
    allowed = {item["id"] for item in inventory(root)["items"] if item["removable"]}
    if not set(identifiers) <= allowed:
        raise ValueError("The selected items changed or are not cleanable failed artifacts; rescan")
    builds = (root / "builds").resolve()
    targets = [(builds / name).resolve() for name in set(identifiers)]
    if any(target.parent != builds for target in targets):
        raise ValueError("The cleanup path is not inside the build directory")
    # Recheck links immediately before deleting, using the same shell/runtime.
    for target in targets:
        _size(target)
        shutil.rmtree(target)
    return inventory(root)
