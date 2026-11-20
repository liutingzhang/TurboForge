"""Deterministic weight edits shared by the jailbreak processors.

Everything here operates on high-precision safetensors shards with torch and
never touches the source directory: each edit reads a shard, transforms the
matching tensors and writes a new shard into the output directory. The math is
split out from the IO so it can be unit-tested on synthetic tensors without a
real checkpoint or a GPU.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from ..domain import MODEL_RESOURCE_FILES

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:  # torch/safetensors are optional; only abliteration needs them
    torch = None  # type: ignore[assignment]
    load_file = save_file = None  # type: ignore[assignment]


# Matrices that write into the residual stream: their output axis is the hidden
# dimension, so a refusal direction is removed from that axis. Embeddings are
# handled separately because the hidden dimension is their row axis.
# out_proj.weight covers GDN / linear-attention output projections (Qwen3.5
# hybrid) and generic MultiheadAttention; the shape[0]==hidden guard below keeps
# vision-tower projections (different hidden) from being touched.
DEFAULT_OUTPUT_SUFFIXES = ("o_proj.weight", "down_proj.weight", "out_proj.weight")
DEFAULT_EMBEDDING_SUFFIXES = ("embed_tokens.weight",)

_RESOURCE_GLOBS = (
    "config.json", "generation_config.json", "tokenizer*.json", "tokenizer.model",
    "*.txt", "chat_template.jinja", "preprocessor_config.json",
    "*.safetensors.index.json",
)


def require_dependencies() -> None:
    global torch, load_file, save_file
    if torch is None or load_file is None:
        try:
            import torch as runtime_torch
            from safetensors.torch import load_file as runtime_load, save_file as runtime_save
        except ImportError as exc:
            raise RuntimeError("abliteration requires torch and safetensors") from exc
        torch, load_file, save_file = runtime_torch, runtime_load, runtime_save


def model_shards(model_dir: Path) -> list[Path]:
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in {model_dir}")
    return shards


def hidden_size(model_dir: Path) -> int:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    value = config.get("hidden_size")
    if not isinstance(value, int) or value <= 0:
        # Multimodal checkpoints (e.g. Qwen3.5) keep the text tower's hidden_size
        # under text_config; the refusal direction lives in that text residual.
        text = config.get("text_config")
        if isinstance(text, dict):
            value = text.get("hidden_size")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{model_dir}/config.json has no usable hidden_size")
    return value


def copy_resources(src: Path, dst: Path) -> None:
    """Copy non-weight model resources so the edited dir stays loadable."""
    dst.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    for pattern in (*MODEL_RESOURCE_FILES, *_RESOURCE_GLOBS):
        for path in src.glob(pattern):
            if path.is_file() and path not in seen:
                shutil.copy2(path, dst / path.name)
                seen.add(path)
    from ..speculative import copy_companions
    copy_companions(src,dst)


def rewrite_shards(
    src: Path, dst: Path, transform: Callable[[str, torch.Tensor], torch.Tensor]
) -> list[str]:
    """Apply transform to every tensor, shard by shard, into dst.

    Only tensors the transform actually changes are reported. The source shard
    is read once and never written back.
    """
    dst.mkdir(parents=True, exist_ok=True)
    changed: list[str] = []
    for shard in model_shards(src):
        tensors = load_file(str(shard))
        out: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            new = transform(name, tensor)
            if new is not tensor:
                changed.append(name)
            out[name] = new
        save_file(out, str(dst / shard.name))
    return changed


# -- abliteration math -------------------------------------------------------


def refusal_direction(harmful_mean: torch.Tensor, harmless_mean: torch.Tensor) -> torch.Tensor:
    """Mean-activation difference, normalized to unit length (Arditi et al.)."""
    return normalize_direction(harmful_mean - harmless_mean)


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    if not isinstance(direction, torch.Tensor) or direction.is_complex():
        raise ValueError("refusal direction must be a real tensor")
    direction = direction.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not direction.numel() or not torch.isfinite(direction).all():
        raise ValueError("refusal direction must be nonempty and finite")
    scale = direction.abs().max()
    if scale == 0:
        raise ValueError("refusal direction must be nonzero")
    direction = direction / scale
    return direction / direction.norm()


def orthogonalize_output(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove `direction` from the output axis of W [out, in]."""
    dtype = weight.dtype
    w = weight.to(torch.float32)
    r = direction.to(torch.float32)
    projection = r @ w                      # [in]
    edited = w - torch.outer(r, projection)  # [out, in]
    return edited.to(dtype)


def orthogonalize_embedding(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove `direction` from each row of E [vocab, hidden]."""
    dtype = weight.dtype
    w = weight.to(torch.float32)
    r = direction.to(torch.float32)
    coefficients = w @ r                    # [vocab]
    edited = w - torch.outer(coefficients, r)
    return edited.to(dtype)


def abliteration_transform(
    direction: torch.Tensor,
    output_suffixes: tuple[str, ...] = DEFAULT_OUTPUT_SUFFIXES,
    embedding_suffixes: tuple[str, ...] = DEFAULT_EMBEDDING_SUFFIXES,
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    direction = normalize_direction(direction)

    def transform(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2 or not tensor.is_floating_point():
            return tensor
        if name.endswith(output_suffixes) and tensor.shape[0] == direction.numel():
            return orthogonalize_output(tensor, direction)
        if name.endswith(embedding_suffixes) and tensor.shape[1] == direction.numel():
            return orthogonalize_embedding(tensor, direction)
        return tensor

    return transform
