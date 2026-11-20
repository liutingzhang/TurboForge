from __future__ import annotations

from pathlib import Path
import json

from .adapters.base import InputAdapter
from .adapters.gguf import GGUFAdapter
from .adapters.safetensors import SafetensorsAdapter
from .domain import Diagnostic, ModelInspection, InputKind, Severity
from .model_sources import normalize_reference
from .safetensors_io import SafetensorsFormatError, read_tensor_summaries
from .speculative import component_name


class ModelInspector:
    def __init__(self, adapters: list[InputAdapter] | None = None) -> None:
        self.adapters = adapters or [SafetensorsAdapter(), GGUFAdapter()]

    def inspect(self, path: str | Path, *, metadata_only: bool = False) -> ModelInspection:
        candidate = Path(normalize_reference(str(path))).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Input model does not exist: {candidate}")
        if metadata_only:
            if not candidate.is_dir():
                raise ValueError("原始模型来源必须是目录，请选择包含 config.json 与分词器文件的原始 HF 模型目录，或填写原始仓库链接")
            if not (candidate / "config.json").is_file():
                raise ValueError("所选原始模型目录缺少 config.json。GGUF 权重所在文件夹不一定是原始模型目录；请选择含配置与分词器文件的原始 HF 模型目录，或填写原始仓库链接")
            config = json.loads((candidate / "config.json").read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("config.json 必须是对象")
            tokenizer = [name for name in ("tokenizer.json", "tokenizer.model") if (candidate / name).is_file()]
            if not tokenizer:
                raise ValueError("GGUF 导入需要原始 HF 目录中的 config.json 与 tokenizer.json/tokenizer.model")
            speculative, files, diagnostics = self._metadata_components(candidate, config)
            architectures = config.get("architectures", [])
            return ModelInspection(path=str(candidate), input_kind=InputKind.SAFETENSORS,
                                   architecture=architectures[0] if architectures else config.get("model_type", "unknown"),
                                   config=config, tokenizer_files=tokenizer, weight_precision="metadata_only",
                                   speculative_components=speculative,
                                   files=files, diagnostics=diagnostics)
        for adapter in self.adapters:
            if adapter.matches(candidate):
                return adapter.inspect(candidate)
        raise ValueError(
            f"No input adapter recognizes {candidate}. Provide a complete model directory, not a renamed file extension."
        )

    @staticmethod
    def _metadata_components(candidate: Path, config: dict) -> tuple[dict, list[dict], list[Diagnostic]]:
        """Unreadable weight headers must not silently imply absent auxiliary heads."""
        root = candidate.resolve()
        shards = set(root.glob("*.safetensors"))
        index_path = root / "model.safetensors.index.json"
        weight_map = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8-sig"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_path}: invalid weight_map")
            for name in weight_map.values():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{index_path}: invalid shard reference")
                shard = (root / name).resolve()
                if root not in shard.parents or shard.suffix.lower() != ".safetensors":
                    raise ValueError(f"{index_path}: unsafe shard reference {name!r}")
                if shard.is_file():
                    shards.add(shard)

        tensor_files: dict[str, str] = {}
        files: dict[str, Path] = {}
        diagnostics: list[Diagnostic] = []
        for shard in sorted(shards):
            try:
                summaries, _ = read_tensor_summaries(shard)
            except (OSError, SafetensorsFormatError) as exc:
                diagnostics.append(Diagnostic(
                    "speculative-inspection-unverified",
                    f"无法核实已有权重中的 MTP/DFlash2，不能假定没有推测头并继续转换：{exc}",
                    Severity.ERROR,
                ))
                continue
            relative = shard.relative_to(root).as_posix()
            files[relative] = shard
            for tensor in summaries:
                if tensor.name in tensor_files:
                    diagnostics.append(Diagnostic("duplicate-tensor", f"重复来源张量：{tensor.name}", Severity.ERROR))
                tensor_files[tensor.name] = relative
        for name in weight_map:
            if component_name(name) and name not in tensor_files:
                diagnostics.append(Diagnostic(
                    "speculative-inspection-unverified",
                    f"索引声明的推测头张量 {name} 缺失，不能假定没有推测头并继续转换",
                    Severity.ERROR,
                ))
        speculative = SafetensorsAdapter._speculative_components(
            root, config, sorted(shards), list(tensor_files), tensor_files,
        )
        for component in ("mtp", "dflash2"):
            directory = speculative.get(component, {}).get("source_path")
            if directory:
                for file in Path(directory).rglob("*"):
                    if file.is_file() and (file.suffix == ".safetensors" or file.name == "config.json"):
                        files[file.relative_to(root).as_posix()] = file
        return speculative, [{"path": name, "bytes": file.stat().st_size} for name, file in sorted(files.items())], diagnostics
