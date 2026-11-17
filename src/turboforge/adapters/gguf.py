"""Read-only GGUF header inspection (standard library only).

Parses the GGUF v2/v3 metadata KV block and the tensor directory so a GGUF
file is auto-recognized like a safetensors directory: architecture, quant
label (general.file_type), tensor count and MTP/DFlash2 companions. It never
reads weight payloads and never modifies the file.

The GGUF is a *secondary* weight source for the pinned converter: the plan
still requires the original HF directory as --model (see planner). This
adapter only makes the input visible and honest about what it is.
"""

from __future__ import annotations

import struct
import re
from pathlib import Path
from typing import Any, BinaryIO

from ..domain import Diagnostic, InputKind, ModelInspection, Severity, TensorSummary
from .base import InputAdapter

# llama.cpp llama_ftype enum values (stable public spec).
_FTYPE_LABELS = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "IQ3_XXS", 22: "IQ1_S", 23: "IQ4_NL", 24: "IQ3_S",
    25: "IQ2_S", 26: "IQ4_XS", 27: "IQ1_M", 28: "BF16", 32: "TQ1_0", 33: "TQ2_0",
}

_SCALAR_FORMATS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}


class GGUFFormatError(ValueError):
    pass


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise GGUFFormatError("truncated GGUF header")
    return data


def _read_string(stream: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read_exact(stream, 8))
    if length > 64 * 1024 * 1024:
        raise GGUFFormatError("implausible GGUF string length")
    return _read_exact(stream, length).decode("utf-8", errors="replace")


def _skip_value(stream: BinaryIO, value_type: int) -> None:
    if value_type == 8:
        _read_string(stream)
    elif value_type == 9:
        (elem_type,) = struct.unpack("<I", _read_exact(stream, 4))
        (count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            _skip_value(stream, elem_type)
    elif value_type in _SCALAR_FORMATS:
        _, size = _SCALAR_FORMATS[value_type]
        _read_exact(stream, size)
    else:
        raise GGUFFormatError(f"unknown GGUF metadata value type {value_type}")


def _read_value(stream: BinaryIO, value_type: int) -> Any:
    if value_type == 8:
        return _read_string(stream)
    if value_type == 9:
        _skip_value(stream, value_type)
        return None
    if value_type in _SCALAR_FORMATS:
        fmt, size = _SCALAR_FORMATS[value_type]
        return struct.unpack(fmt, _read_exact(stream, size))[0]
    raise GGUFFormatError(f"unknown GGUF metadata value type {value_type}")


def _nominal_bits(label: str) -> float | None:
    if label in ("F16", "BF16"):
        return 16.0
    if label == "F32":
        return 32.0
    digits = ""
    for char in label:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return float(digits) if digits else None


_EXPERT_SUFFIXES = (".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight")


def _read_directory(stream: BinaryIO, label: str, *, keep_metadata: bool
                    ) -> tuple[dict[str, Any], list[tuple[str, list[int], int, int]]]:
    """Read one shard header: metadata (optional) and tensor directory.

    Returns ``(metadata, [(name, dims, type_id, offset), ...])``. Only the first
    shard's metadata is kept; later shards still have to be walked to find
    auxiliary heads stored entirely in a split file.
    """
    if _read_exact(stream, 4) != b"GGUF":
        raise GGUFFormatError(f"{label}: not a GGUF file (bad magic)")
    (version,) = struct.unpack("<I", _read_exact(stream, 4))
    if version not in (2, 3):
        raise GGUFFormatError(f"unsupported GGUF version {version}")
    (tensor_count,) = struct.unpack("<Q", _read_exact(stream, 8))
    (kv_count,) = struct.unpack("<Q", _read_exact(stream, 8))
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _read_string(stream)
        (value_type,) = struct.unpack("<I", _read_exact(stream, 4))
        value = _read_value(stream, value_type)
        if keep_metadata and value is not None:
            metadata[key] = value
    tensors = []
    for _ in range(tensor_count):
        name = _read_string(stream)
        (n_dims,) = struct.unpack("<I", _read_exact(stream, 4))
        dims = list(struct.unpack(f"<{n_dims}Q", _read_exact(stream, 8 * n_dims)))
        (type_id,) = struct.unpack("<I", _read_exact(stream, 4))
        (offset,) = struct.unpack("<Q", _read_exact(stream, 8))
        tensors.append((name, dims, type_id, offset))
    return metadata, tensors


class GGUFAdapter(InputAdapter):
    name = "gguf-header"

    def matches(self, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() == ".gguf"
        return path.is_dir() and any(path.glob("*.gguf"))

    def inspect(self, path: Path) -> ModelInspection:
        path = path.expanduser().resolve()
        if path.is_file():
            shards = [path]
            match = re.fullmatch(r"(.+)-\d{5}-of-(\d{5})\.gguf", path.name, re.I)
            if match:
                shards = [path.parent / f"{match[1]}-{number:05}-of-{int(match[2]):05}.gguf"
                          for number in range(1, int(match[2]) + 1)]
                if any(not shard.is_file() for shard in shards):
                    raise GGUFFormatError("GGUF shards are incomplete; cannot determine whether the speculative heads are complete")
        else:
            shards = sorted(path.glob("*.gguf"), key=lambda p: p.as_posix().lower())
        if not shards:
            raise GGUFFormatError(f"No .gguf file found in {path}")

        with shards[0].open("rb") as stream:
            metadata, directory = _read_directory(stream, shards[0].name, keep_metadata=True)
        # Auxiliary heads may be stored entirely in the last split shard.
        # Inspect their directories too; never infer absence from shard 1 only.
        combined = list(directory)
        for shard in shards[1:]:
            with shard.open("rb") as stream:
                _, extra = _read_directory(stream, str(shard), keep_metadata=False)
            combined.extend(extra)

        tensor_names = [name for name, _, _, _ in combined]
        tensor_types = {type_id for _, _, type_id, _ in combined}
        expert_types = [type_id for name, _, type_id, _ in combined
                        if name.endswith(_EXPERT_SUFFIXES)]
        # Sample byte sizes are derived from shard 1's directory only: offsets
        # in other split shards are relative to their own data section.
        offsets = sorted(((name, offset, dims) for name, dims, _, offset in directory),
                         key=lambda item: item[1])
        total = shards[0].stat().st_size
        samples = []
        for index, (name, offset, dims) in enumerate(offsets[:24]):
            end = offsets[index + 1][1] if index + 1 < len(offsets) else total
            samples.append(TensorSummary(name, "gguf_block", dims, max(end - offset, 0)))

        architecture = str(metadata.get("general.architecture", "unknown"))
        file_type = metadata.get("general.file_type")
        label = _FTYPE_LABELS.get(file_type, f"ftype_{file_type}") if isinstance(file_type, int) else "unknown"
        # Pinned Prism GGUF type id, from tools.convert.sources.gguf.TYPE_PQ2_0.
        # A filename containing 'ternary' is not evidence of this storage format.
        if 142 in tensor_types:
            label = "PQ2_0"
        bits = _nominal_bits(label)

        diagnostics = [Diagnostic(
            "gguf-recognized",
            f"GGUF recognized: architecture={architecture}, quant={label}"
            + (f" (~{bits:g} bit nominal)" if bits else "")
            + f", {len(combined)} tensors across {len(shards)} file(s)",
            Severity.INFO,
        )]
        speculative: dict[str, Any] = {"training": "not_supported"}
        if any("mtp" in name.lower() or ".nextn." in name.lower() for name in tensor_names):
            speculative["mtp"] = {"detected": True, "compatibility_verified": False}
        if any("dflash2" in name.lower() for name in tensor_names):
            speculative["dflash2"] = {"detected": True, "compatibility_verified": False}
        elif any("dflash" in name.lower() for name in tensor_names):
            speculative["dflash_hint"] = "A DFlash name was detected; this is not proof of DFlash2 compatibility"

        return ModelInspection(
            path=str(path),
            input_kind=InputKind.GGUF,
            architecture=architecture,
            model_type=architecture,
            weight_precision=label,
            tensor_count=len(combined),
            tensor_bytes=sum(shard.stat().st_size for shard in shards),
            element_count=0,
            shard_count=len(shards),
            dtype_counts={label: len(combined)},
            tensor_samples=samples,
            quantization={
                "gguf_file_type": file_type,
                "native_gsq_q2": bool(expert_types) and all(type_id == 42 for type_id in expert_types),
                "gguf_quant_label": label,
                "ternary_pq2": 142 in tensor_types,
                "origin_repositories": [metadata[key] for key in (
                    "general.base_model.0.repo_url", "general.source.huggingface.repository", "general.source.url"
                ) if isinstance(metadata.get(key), str)],
                "nominal_bits": bits,
            },
            tokenizer_files=[],
            speculative_components=speculative,
            files=[{"path": shard.name, "bytes": shard.stat().st_size} for shard in shards],
            diagnostics=diagnostics,
            config={},
        )
