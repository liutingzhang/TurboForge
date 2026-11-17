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
                raise ValueError("The original model source must be a directory; select the original HF model directory containing config.json and tokenizer files, or enter the original repository link")
            if not (candidate / "config.json").is_file():
                raise ValueError("The selected original model directory lacks config.json. The folder holding GGUF weights is not necessarily the original model directory; select the original HF model directory with config and tokenizer files, or enter the original repository link")
            config = json.loads((candidate / "config.json").read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("config.json must be an object")
            tokenizer = [name for name in ("tokenizer.json", "tokenizer.model") if (candidate / name).is_file()]
            if not tokenizer:
                raise ValueError("GGUF import needs config.json and tokenizer.json/tokenizer.model from the original HF directory")
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
