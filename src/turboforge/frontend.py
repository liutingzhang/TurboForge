"""Content-based frontend freshness; ZIP timestamps must not trigger Node installs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def source_digest(web: Path) -> str:
    sources = [web/name for name in ("package.json", "package-lock.json", "index.html", "vite.config.ts", "tsconfig.json")]
    sources += [path for path in (web/"src").rglob("*") if path.is_file()]
    digest = hashlib.sha256()
    for path in sorted(sources):
        if path.is_file():
            digest.update(path.relative_to(web).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def frontend_current(web: Path) -> bool:
    marker = web/"dist/build-receipt.json"
    if not marker.is_file() or not (web/"dist/index.html").is_file():
        return False
    try:
        receipt = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(receipt, dict) or not isinstance(receipt.get("files"), dict) or "index.html" not in receipt["files"]:
        return False
    if receipt.get("source_sha256") != source_digest(web):
        return False
    dist = (web/"dist").resolve()
    for name, expected in receipt["files"].items():
        if not isinstance(name, str) or not isinstance(expected, str):
            return False
        asset = (dist/name).resolve()
        if not asset.is_relative_to(dist) or not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            return False
    return True
