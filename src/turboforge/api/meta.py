"""Single source of truth for the web UI's choices.

Everything the frontend renders as a dropdown, a required/optional marker or a
recipe detail is derived here from the same registry and processors the CLI
uses. The UI never hardcodes a choice, so a backend change cannot silently
desynchronize the frontend.

This module is pure (no Flask) so it can be unit-tested directly.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any

from ..backends.compiler import probe_upstream
from ..backends.registry import PINNED_UPSTREAM_REVISION, RECIPES
from ..backends.hf_quantization import HF_RECIPES
from ..domain import (
    JAILBREAK_CHOICES,
    InputKind,
)
from ..local_endpoint import default_serving_port
from ..jailbreak.processors import PROCESSORS
from ..system import gpu_status
from ..runner import DEFAULT_NINFER_BIN


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# -- human-facing descriptions ----------------------------------------------

_JAILBREAK_INFO = {
    "none": ("不越狱", "保持原始权重，直接量化转换"),
    "heretic": ("Heretic 一键越狱", "自动调用本地 heretic（pip install -U heretic-llm）；"
                 "独立窗口运行，结束时把权重保存到构建目录后自动继续量化"),
    "abliteration": ("Abliteration 一键越狱", "拒绝方向正交化（torch，逐分片）；仅高精度权重，"
                     "可用预计算方向或校准提示词"),
}

_DEVICE_INFO = {
    "cuda": ("CUDA（GPU）", "在 GPU 上执行分组量化；最快，需 CUDA 版 torch"),
    "cpu": ("CPU", "tools.convert 支持 --device cpu，但本机无实测计时，预计远慢于 CUDA"),
}

_INPUT_KIND_INFO = {
    "safetensors": "HF safetensors 目录",
    "nvfp4": "NVFP4 量化检查点",
    "gguf": "GGUF 块文件",
    "ninfer": "NInfer 产物",
    "unknown": "未知 / 无法识别",
}


# -- jailbreak parameter schemas ---------------------------------------------
# required: unconditional. required_when_env_missing: the param becomes required
# when that optional module is absent (the frontend evaluates this against env).
# advanced: grouped under a collapsed "advanced" section in the UI.

def _param(key: str, label: str, type_: str, *, required: bool = False,
           required_when_env_missing: str | None = None, help_: str = "",
           placeholder: str = "", default: Any = None,
           advanced: bool = False) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": type_,
        "required": required,
        "required_when_env_missing": required_when_env_missing,
        "help": help_,
        "placeholder": placeholder,
        "default": default,
        "advanced": advanced,
    }


_PARAM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "none": [],
    "heretic": [],
    "abliteration": [
        _param(
            "direction_file", "拒绝方向向量", "path",
            required_when_env_missing="transformers",
            help_="预计算拒绝方向（.npy / .pt，长度等于 hidden_size，可由 heretic 导出）。"
                  "未安装 transformers 时必填；已安装时可留空，由校准提示词自动采集。",
            placeholder="D:/vectors/refusal_direction.npy",
        ),
        _param(
            "layer", "校准层索引", "int",
            help_="采集激活的隐藏层索引；留空时取 num_hidden_layers // 2。",
            placeholder="例如 32", advanced=True,
        ),
        _param(
            "harmful_prompts", "有害校准提示词", "string_list",
            help_="每行一条；留空使用内置默认集。仅在需要校准时使用。",
            advanced=True,
        ),
        _param(
            "harmless_prompts", "无害校准提示词", "string_list",
            help_="每行一条；留空使用内置默认集。仅在需要校准时使用。",
            advanced=True,
        ),
        _param(
            "output_suffixes", "输出投影后缀", "string_list",
            help_="要做正交化的输出张量后缀；留空使用默认集。高级选项。",
            advanced=True,
        ),
        _param(
            "embedding_suffixes", "嵌入张量后缀", "string_list",
            help_="要处理的嵌入张量后缀；留空使用默认集。高级选项。",
            advanced=True,
        ),
    ],
}


def _recipe_dict(recipe) -> dict[str, Any]:
    return {
        "recipe": recipe.recipe,
        "engine": recipe.engine,
        "label": recipe.label,
        "input_kinds": [k.value for k in recipe.input_kinds],
        "architectures": list(recipe.architectures),
        "storage_formats": list(recipe.storage_formats),
        "nominal_bits": recipe.nominal_bits,
        "typical_bpw": recipe.typical_bpw,
        "extra_sources": list(recipe.extra_sources),
        "components": recipe.components,
        "needs_cuda": recipe.needs_cuda,
        "needs_gpu": recipe.needs_gpu,
        "notes": list(recipe.notes),
    }


def _choice(value: str, info: dict[str, tuple[str, str]]) -> dict[str, str]:
    label, description = info[value]
    return {"value": value, "label": label, "description": description}


def build_meta() -> dict[str, Any]:
    """Assemble the full metadata payload served at GET /api/meta."""
    env = {
        "torch": _module_available("torch"),
        "transformers": _module_available("transformers"),
        "numpy": _module_available("numpy"),
        "safetensors": _module_available("safetensors"),
        "flask": _module_available("flask"),
    }

    upstream = probe_upstream()
    env["torch_cuda"] = bool(upstream.cuda_available)
    gpu = gpu_status()

    jailbreak = []
    for name in JAILBREAK_CHOICES:
        processor = PROCESSORS[name]
        label, description = _JAILBREAK_INFO[name]
        jailbreak.append({
            "name": name,
            "label": label,
            "description": description,
            "capability": asdict(processor.capability),
            "params": _PARAM_SCHEMAS[name],
        })

    return {
        "pinned_revision": PINNED_UPSTREAM_REVISION,
        "choices": {
            "jailbreak": jailbreak,
            "device": [_choice(v, _DEVICE_INFO) for v in ("cuda", "cpu")],
            "input_kind": [
                {"value": k.value, "label": _INPUT_KIND_INFO[k.value]} for k in InputKind
            ],
        },
        "recipes": [_recipe_dict(r) for r in RECIPES],
        "hf_recipes": [_recipe_dict(r) for r in HF_RECIPES],
        "env": env,
        "upstream": {
            "root": str(upstream.root) if upstream.root else None,
            "found": upstream.found,
            "revision": upstream.revision,
            "revision_matches_pin": upstream.revision_matches_pin,
            "torch_available": upstream.torch_available,
            "numpy_available": upstream.numpy_available,
            "diagnostics": [asdict(d) for d in upstream.diagnostics],
        },
        "gpu": (
            {"name": gpu.name, "total_bytes": gpu.total_bytes, "free_bytes": gpu.free_bytes}
            if gpu else None
        ),
        "engine": {"root": os.environ.get("TURBOFORGE_NINFER_BIN", str(DEFAULT_NINFER_BIN)), "available": (Path(os.environ.get("TURBOFORGE_NINFER_BIN", str(DEFAULT_NINFER_BIN))) / "ninfer-serve.exe").is_file()},
        "defaults": {
            "target": "auto",
            "jailbreak": "none",
            "device": "cuda" if upstream.cuda_available else "cpu",
            "upstream_dir": str(upstream.root),
            "ninfer_dir": os.environ.get("TURBOFORGE_NINFER_BIN", str(DEFAULT_NINFER_BIN)),
            "allow_approximate_requantization": False,
            "run_validation": True,
            "run_benchmark": False,
            "runtime": {"max_context": 32768, "max_concurrency": 1, "port": default_serving_port()},
            # kv_dtype is intentionally absent: the runner picks it from the
            # geometry (bf16 on the small GDN GGUF bridge for faster MTP, rk4v4 on
            # the native 27B/35B route for VRAM). Pinning a default here would drift
            # from that choice; the UI offers "auto" (omit) plus explicit overrides.
            "benchmark": {
                "repetitions": 5,
                "n_prompt": [512, 2048],
                "n_gen": [128],
                "perplexity_context": 4096,
                "perplexity_stride": 512,
            },
        },
    }
