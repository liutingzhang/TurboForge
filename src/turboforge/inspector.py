from __future__ import annotations

from pathlib import Path
import json

from .adapters.base import InputAdapter
from .adapters.gguf import GGUFAdapter
from .adapters.safetensors import SafetensorsAdapter
from .domain import ModelInspection, InputKind
from .model_sources import normalize_reference


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
            architectures = config.get("architectures", [])
            return ModelInspection(path=str(candidate), input_kind=InputKind.SAFETENSORS,
                                   architecture=architectures[0] if architectures else config.get("model_type", "unknown"),
                                   config=config, tokenizer_files=tokenizer, weight_precision="metadata_only")
        for adapter in self.adapters:
            if adapter.matches(candidate):
                return adapter.inspect(candidate)
        raise ValueError(
            f"No input adapter recognizes {candidate}. Provide a complete model directory, not a renamed file extension."
        )
