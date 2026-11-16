from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ..domain import Diagnostic, InputKind, ModelInspection, Severity, TensorSummary, QUANTIZATION_CONFIG_FILES
from ..safetensors_io import SafetensorsFormatError, read_tensor_summaries
from ..speculative import companions, component_name
from .base import InputAdapter


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
)
_QUANTIZED_DTYPES = {
    "F4",
    "F8_E4M3",
    "F8_E5M2",
    "I8",
    "U8",
    "I16",
    "I32",
    "U32",
    "I64",
}
_SCALE_MARKERS = (
    "weight_scale",
    "weight_global_scale",
    "weight_quantizer",
    "input_scale",
    "activation_scale",
)


def _contains_nvfp4(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.lower().replace("-", "").replace("_", "")
        return "nvfp4" in normalized
    if isinstance(value, dict):
        return any(_contains_nvfp4(key) or _contains_nvfp4(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_nvfp4(item) for item in value)
    return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SafetensorsFormatError(f"{path}: expected a JSON object")
    return value


def _natural_key(path: Path) -> str:
    return path.as_posix().lower()


def read_hf_config(root: Path) -> dict[str, Any]:
    config = _read_json(root/"config.json")
    if not config.get("quantization_config"):
        for name in QUANTIZATION_CONFIG_FILES:
            if (root/name).is_file():
                config["quantization_config"] = _read_json(root/name)
                break
    return config


class SafetensorsAdapter(InputAdapter):
    name = "hf-safetensors"

    def matches(self, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() == ".safetensors"
        return path.is_dir() and (
            (path / "model.safetensors.index.json").is_file()
            or any(path.glob("*.safetensors"))
        )

    def inspect(self, path: Path) -> ModelInspection:
        path = path.expanduser().resolve()
        root = path if path.is_dir() else path.parent
        shards = self._shards(path)
        config = read_hf_config(root) if (root / "config.json").is_file() else {}
        quant_config = config.get("quantization_config", {})

        tensors: list[TensorSummary] = []
        metadata: dict[str, Any] = {}
        files: list[dict[str, Any]] = []
        diagnostics: list[Diagnostic] = []
        seen: dict[str, str] = {}
        for shard in shards:
            shard_tensors, shard_metadata = read_tensor_summaries(shard)
            relative = shard.relative_to(root).as_posix()
            files.append({"path": relative, "bytes": shard.stat().st_size})
            for tensor in shard_tensors:
                if tensor.name in seen:
                    diagnostics.append(Diagnostic(
                        "duplicate-tensor",
                        f"Tensor {tensor.name!r} occurs in both {seen[tensor.name]} and {relative}",
                        Severity.ERROR,
                    ))
                seen[tensor.name] = relative
            tensors.extend(shard_tensors)
            metadata.update(shard_metadata)

        if not tensors:
            diagnostics.append(Diagnostic(
                "empty-model", "No tensor entries were found", Severity.ERROR
            ))

        architecture_values = config.get("architectures", [])
        architecture = (
            architecture_values[0]
            if isinstance(architecture_values, list) and architecture_values
            else config.get("model_type", "unknown")
        )
        model_type = str(config.get("model_type", "unknown"))
        dtype_counts = dict(Counter(tensor.dtype for tensor in tensors))
        scale_tensors = [
            tensor.name for tensor in tensors
            if any(marker in tensor.name.lower() for marker in _SCALE_MARKERS)
        ]
        quantized_tensors = [
            tensor for tensor in tensors
            if tensor.dtype in _QUANTIZED_DTYPES
            or "weight_packed" in tensor.name.lower()
        ]
        config_nvfp4 = _contains_nvfp4(quant_config) or _contains_nvfp4(metadata)
        layout_verified = bool(config_nvfp4 and scale_tensors and quantized_tensors)
        filename_nvfp4 = "nvfp4" in path.name.lower()
        if filename_nvfp4 and not config_nvfp4:
            diagnostics.append(Diagnostic(
                "nvfp4-name-is-not-evidence",
                "The path mentions NVFP4 but the model metadata does not; the input was not reclassified by filename",
                Severity.WARNING,
            ))
        if config_nvfp4 and not layout_verified:
            diagnostics.append(Diagnostic(
                "nvfp4-layout-unverified",
                "NVFP4 metadata exists, but packed weights and/or block/global scale tensors are missing or unrecognized; a compatible NVFP4 weight adapter is required",
                Severity.ERROR,
            ))

        input_kind = InputKind.NVFP4 if config_nvfp4 else InputKind.SAFETENSORS
        precision = self._precision(tensors, input_kind)
        method = str(quant_config.get("quant_method",quant_config.get("method",""))).lower() if isinstance(quant_config,dict) else ""
        if method in ("awq","gptq"):
            precision = f"{method}_{quant_config.get('bits',quant_config.get('w_bit','unknown'))}bit"
        quantization = dict(quant_config) if isinstance(quant_config, dict) else {"value": quant_config}
        quantization.update({
            "nvfp4_metadata_detected": config_nvfp4,
            "nvfp4_layout_verified": layout_verified,
            "scale_tensor_count": len(scale_tensors),
            "quantized_tensor_count": len(quantized_tensors),
            "dequantization_restores_original_precision": False,
        })
        if input_kind == InputKind.NVFP4:
            diagnostics.append(Diagnostic(
                "nvfp4-information-loss",
                "NVFP4 dequantization produces approximate weights and cannot restore original BF16/FP16 information; any requantization is a second lossy quantization",
                Severity.WARNING,
            ))

        tokenizer_files = [name for name in _TOKENIZER_FILES if (root / name).is_file()]
        if not tokenizer_files:
            diagnostics.append(Diagnostic(
                "tokenizer-missing",
                "No tokenizer resources were found beside the weights; conversion may require a complete Hugging Face model directory",
                Severity.WARNING,
            ))
        if not config:
            diagnostics.append(Diagnostic(
                "config-missing", "config.json is missing", Severity.ERROR
            ))

        speculative = self._speculative_components(
            root, config, shards, [tensor.name for tensor in tensors], seen
        )
        known_files = {item["path"] for item in files}
        for component in ("mtp", "dflash2"):
            directory = speculative.get(component, {}).get("source_path")
            if not directory:
                continue
            for file in Path(directory).rglob("*"):
                if file.is_file() and (file.suffix == ".safetensors" or file.name == "config.json"):
                    relative = file.relative_to(root).as_posix()
                    if relative not in known_files:
                        files.append({"path": relative, "bytes": file.stat().st_size})
                        known_files.add(relative)
        return ModelInspection(
            path=str(path),
            input_kind=input_kind,
            architecture=str(architecture),
            model_type=model_type,
            weight_precision=precision,
            tensor_count=len(tensors),
            tensor_bytes=sum(tensor.byte_size for tensor in tensors),
            element_count=sum(math.prod(tensor.shape) for tensor in tensors),
            shard_count=len(shards),
            dtype_counts=dtype_counts,
            tensor_samples=tensors[:24],
            quantization=quantization,
            tokenizer_files=tokenizer_files,
            speculative_components=speculative,
            license=config.get("license") if isinstance(config.get("license"), str) else None,
            files=files,
            diagnostics=diagnostics,
            config=config,
        )

    def _shards(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            index = _read_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise SafetensorsFormatError(f"{index_path}: invalid or empty weight_map")
            names = set()
            for name in weight_map.values():
                if not isinstance(name, str) or not name:
                    raise SafetensorsFormatError(f"{index_path}: invalid shard reference {name!r}")
                candidate = (path / name).resolve()
                if path.resolve() not in candidate.parents or candidate.suffix.lower() != ".safetensors":
                    raise SafetensorsFormatError(f"{index_path}: unsafe shard reference {name!r}")
                if not candidate.is_file():
                    raise SafetensorsFormatError(f"Referenced weight shard is missing: {candidate}")
                names.add(name)
            shards = [path / name for name in names]
            for extra in path.glob("*.safetensors"):
                if extra.name not in names and any(component_name(t.name) for t in read_tensor_summaries(extra)[0]):
                    shards.append(extra)
        else:
            shards = list(path.glob("*.safetensors"))
            if (path / "adapter_config.json").is_file() and not any(
                item.name.startswith("model") for item in shards
            ):
                raise SafetensorsFormatError(
                    "This directory looks like a LoRA adapter, not a base model; select the base model directory instead"
                )
        if not shards:
            raise SafetensorsFormatError(f"No safetensors weights found in {path}")
        return sorted(shards, key=_natural_key)

    @staticmethod
    def _precision(tensors: list[TensorSummary], kind: InputKind) -> str:
        if kind == InputKind.NVFP4:
            return "nvfp4"
        bytes_by_dtype: Counter[str] = Counter()
        for tensor in tensors:
            bytes_by_dtype[tensor.dtype] += tensor.byte_size
        if not bytes_by_dtype:
            return "unknown"
        dominant = bytes_by_dtype.most_common(1)[0][0]
        return {
            "BF16": "bf16",
            "F16": "fp16",
            "F32": "fp32",
            "F64": "fp64",
            "F8_E4M3": "fp8_e4m3",
            "F8_E5M2": "fp8_e5m2",
        }.get(dominant, dominant.lower())

    @staticmethod
    def _speculative_components(root: Path, config: dict[str, Any], shards: list[Path],
                                tensor_names: list[str], tensor_files: dict[str, str]) -> dict[str, Any]:
        embedded: dict[str, list[str]] = {"mtp": [], "dflash2": []}
        for name in tensor_names:
            component = component_name(name)
            if component is not None:
                embedded[component[0]].append(name)

        result: dict[str, Any] = {"training": "not_supported"}
        for component, names in embedded.items():
            if names:
                # tools.convert.qwen3_5 reads MTP straight from the base
                # checkpoint's "mtp." tensors when no companion source is given,
                # so embedded components are detected by tensor prefix -- not by
                # a sidecar filename or a config key, which small dense Qwen3.5
                # checkpoints (e.g. 0.8B) do not have.
                result[component] = {
                    "detected": True,
                    "compatibility_verified": False,
                    "source": "base_checkpoint",
                    "embedded_tensor_count": len(names),
                }
        has_mtp_config = any(
            key in config for key in ("num_nextn_predict_layers", "mtp_config", "num_mtp_layers")
        )
        if has_mtp_config and not embedded["mtp"]:
            result["mtp_hint"] = "配置声明 MTP，但当前权重内未找到 mtp.* 张量；不会自动启用"

        for component, source in companions(root).items():
            if result.get(component, {}).get("detected"):
                for name in embedded[component]:
                    shard = (root / tensor_files[name]).resolve()
                    if Path(source) not in shard.parents:
                        raise ValueError(f"模型同时内嵌并附带独立 {component.upper()}，请明确整理来源，不能猜测覆盖")
            result[component] = {
                "detected": True,
                "compatibility_verified": False,
                "source": "companion_directory",
                "source_path": source,
            }

        component_files = [
            item.name for item in root.iterdir()
            if item.is_file() and any(marker in item.name.lower() for marker in ("mtp", "dflash"))
        ]
        if component_files:
            result["files"] = component_files
        result["source_files"] = [
            item.name for item in shards
            if any(marker in item.name.lower() for marker in ("mtp", "dflash"))
        ]
        return result
