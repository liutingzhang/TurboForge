from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .domain import TensorSummary


class SafetensorsFormatError(ValueError):
    pass


def read_tensor_summaries(path: Path) -> tuple[list[TensorSummary], dict[str, Any]]:
    """Read only a safetensors header; tensor payloads stay untouched."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafetensorsFormatError(f"{path}: truncated safetensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > 256 * 1024 * 1024:
            raise SafetensorsFormatError(
                f"{path}: invalid or unreasonable header size {header_size}"
            )
        data_start = 8 + header_size
        if data_start > file_size:
            raise SafetensorsFormatError(f"{path}: header extends past end of file")
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsFormatError(f"{path}: malformed JSON header") from exc
        if not isinstance(header, dict):
            raise SafetensorsFormatError(f"{path}: header must be a JSON object")
        metadata = header.get("__metadata__", {})
        summaries: list[TensorSummary] = []
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(value, dict):
                raise SafetensorsFormatError(f"{path}: invalid tensor entry {name!r}")
            try:
                dtype = value["dtype"]
                shape = value["shape"]
                start, end = value["data_offsets"]
            except (KeyError, TypeError, ValueError) as exc:
                raise SafetensorsFormatError(
                    f"{path}: incomplete tensor entry {name!r}"
                ) from exc
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start <= end <= file_size - data_start
            ):
                raise SafetensorsFormatError(
                    f"{path}: invalid dtype, shape or data offsets for {name!r}"
                )
            summaries.append(TensorSummary(name, dtype, shape, end - start))
        return summaries, metadata if isinstance(metadata, dict) else {}
