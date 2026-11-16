"""Model architecture vocabulary shared by every layer.

One module answers three questions about a Hugging Face config: which verified
architecture tuple it belongs to, whether it routes MoE or dense, and what text
geometry it declares. Inspection adapters only report raw metadata; the recipe
registry, planner, precheck and serving profile all read the config through
these helpers instead of re-implementing the ``text_config`` fallback.

The tuples are the exact accepted values of ``config["architectures"][0]`` and
are re-exported by ``backends.registry`` for existing callers.
"""

from __future__ import annotations

from typing import Any

QWEN35_DENSE = ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
QWEN35_MOE = ("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration")
FLASH_NEXT = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
GEMMA4 = ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
FLINT = ("FlintForCausalLM",)
DEEPSEEK_V41 = ("DeepseekV41ForCausalLM",)

# MoE/Dense classification reuses the verified tuples above and the expert-count
# fields the converter already reads; it adds no new probe. An architecture
# outside both sets that declares no experts is "unknown", never a guess.
MOE_ARCHITECTURES = frozenset((*QWEN35_MOE, *FLASH_NEXT, *DEEPSEEK_V41))
DENSE_ARCHITECTURES = frozenset((*QWEN35_DENSE, *GEMMA4, *FLINT))
_EXPERT_FIELDS = ("num_experts", "num_local_experts", "moe_intermediate_size")


def text_config(config: dict) -> dict:
    """The text sub-config for multimodal checkpoints, else the top level."""
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def architecture_names(config: dict) -> list[str]:
    """``config["architectures"]`` as a list; malformed values yield no names."""
    value = config.get("architectures")
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def hidden_size(config: dict) -> int | None:
    """Declared text hidden size, or None when absent/malformed."""
    value = text_config(config).get("hidden_size")
    return value if isinstance(value, int) and value > 0 else None


def max_context(config: dict, default: int = 2048) -> int:
    """Declared context limit, used to clamp serving and evaluation."""
    value = text_config(config).get("max_position_embeddings", default)
    return value if isinstance(value, int) and value > 0 else default


def architecture_kind(config: dict) -> str:
    """Classify a model config as "moe", "dense" or "unknown".

    The verified architecture tuples are authoritative; expert-count fields in
    the text config are the fallback for architectures outside those tuples.
    """
    architectures = architecture_names(config)
    if any(item in MOE_ARCHITECTURES for item in architectures):
        return "moe"
    if any(item in DENSE_ARCHITECTURES for item in architectures):
        return "dense"
    raw = text_config(config)
    if any(type(raw.get(key)) is int and raw[key] > 0 for key in _EXPERT_FIELDS):
        return "moe"
    return "unknown"
