from __future__ import annotations

import json
import struct
from pathlib import Path


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I32": 4, "F8_E4M3": 1}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * _DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        payload.extend(b"\0" * size)
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(payload)


def write_config(path: Path, **overrides) -> None:
    value = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 8,
        "num_hidden_layers": 1,
    }
    value.update(overrides)
    (path / "config.json").write_text(json.dumps(value), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def write_gguf(path: Path, *, architecture: str = "qwen3", file_type: int = 15,
               tensors: dict[str, list[int]] | None = None, tensor_type: int = 12) -> None:
    """Minimal valid GGUF v3: metadata KV, tensor directory, zeroed payload.

    Enough for header inspection tests; the payload is never parsed.
    """
    tensors = tensors if tensors is not None else {"blk.0.attn_q.weight": [8, 8]}
    body = bytearray()
    body += b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", len(tensors)) + struct.pack("<Q", 2)
    for key, value in (("general.architecture", architecture),):
        encoded_key = key.encode("utf-8")
        encoded_value = value.encode("utf-8")
        body += struct.pack("<Q", len(encoded_key)) + encoded_key
        body += struct.pack("<I", 8)
        body += struct.pack("<Q", len(encoded_value)) + encoded_value
    key = b"general.file_type"
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 4) + struct.pack("<I", file_type)
    offset = 0
    for name, dims in tensors.items():
        encoded = name.encode("utf-8")
        body += struct.pack("<Q", len(encoded)) + encoded
        body += struct.pack("<I", len(dims))
        body += struct.pack(f"<{len(dims)}Q", *dims)
        body += struct.pack("<I", tensor_type)  # header tests never decode payload
        body += struct.pack("<Q", offset)
        offset += 64
    body += b"\0" * offset
    path.write_bytes(bytes(body))
