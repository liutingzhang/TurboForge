from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .domain import ForgeConfig, ModelInspection, MODEL_RESOURCE_FILES

# Keep local polling outside replacement; Windows filesystem/AV handles can
# also briefly outlive another process's read, so replacement retries only
# those transient sharing/access errors, with a bounded deadline.
_JSON_LOCK = threading.Lock()


def read_json_atomic(path: Path) -> Any:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            with _JSON_LOCK:
                return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 1.0
        while True:
            try:
                with _JSON_LOCK:
                    os.replace(temporary_name, path)
                break
            except PermissionError as exc:
                if os.name != "nt" or exc.winerror not in (5, 32) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def source_content_identity(path: Path) -> dict[str, Any]:
    """Record missing inputs without preventing creation of a blocked manifest."""
    path = path.expanduser().resolve()
    if path.is_file():
        return {"type": "file", "sha256": file_sha256(path)}
    if path.is_dir():
        return {
            "type": "directory",
            "files": {
                item.relative_to(path).as_posix(): file_sha256(item)
                for item in sorted(path.rglob("*")) if item.is_file()
            },
        }
    return {"type": "missing"}


class BuildManifest:
    def __init__(self, output_dir: Path, build_id: str | None = None) -> None:
        self.output_dir = output_dir
        self.build_id = build_id or uuid.uuid4().hex
        self.path = output_dir / "manifest.json"
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "build_id": self.build_id,
            "status": "created",
            "history": [],
        }

    @classmethod
    def create(
        cls,
        output_dir: str | Path,
        config: ForgeConfig,
        inspection: ModelInspection,
        plan: dict[str, Any],
    ) -> "BuildManifest":
        output = Path(output_dir).expanduser().resolve()
        source = Path(inspection.path).resolve()
        if output == source or (source.is_file() and output == source.parent):
            raise ValueError("Output directory must differ from the original model location")
        if source.is_dir() and source in output.parents:
            raise ValueError("Output directory cannot be created inside the original model directory")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "reports", "scripts"):
            (output / name).mkdir(exist_ok=True)
        manifest = cls(output)
        manifest.data.update({
            "status": "planned",
            "created_at": utc_now(),
            "config": asdict(config),
            "source": inspection.to_dict(),
            "source_identity": {
                "configuration": inspection.config,
                "resources": {name:file_sha256(source / name) for name in MODEL_RESOURCE_FILES
                              if source.is_dir() and (source/name).is_file()},
            },
            "additional_source_identity": {
                name: source_content_identity(Path(path)) for name, path in config.sources.items()
            },
            "source_files": [
                {
                    **item,
                    "modified_ns": (source / item["path"]).stat().st_mtime_ns
                    if source.is_dir() else source.stat().st_mtime_ns,
                }
                for item in inspection.files
            ],
            "plan": plan,
            "quality_loss": {
                "quantization": "Recorded by the selected backend; nominal bits are not measured quality",
                "jailbreak": config.jailbreak,
                "nvfp4_dequantization_is_approximate": inspection.input_kind.value == "nvfp4",
            },
            "artifacts": {},
            "measurements": {"performance": None, "quality": None},
        })
        manifest.save()
        return manifest

    def transition(self, status: str, detail: str, **fields: Any) -> None:
        self.data["status"] = status
        self.data["history"].append({"at": utc_now(), "status": status, "detail": detail})
        self.data.update(fields)
        self.save()

    def add_artifact(self, name: str, path: Path, *, digest: bool = False) -> None:
        resolved = path.resolve()
        if self.output_dir not in resolved.parents:
            raise ValueError("Manifest artifacts must stay inside the build output directory")
        value: dict[str, Any] = {
            "path": resolved.relative_to(self.output_dir).as_posix(),
            "bytes": resolved.stat().st_size,
        }
        if digest:
            value["sha256"] = file_sha256(resolved)
        self.data["artifacts"][name] = value
        self.save()

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        write_json_atomic(self.path, self.data)
