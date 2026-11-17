from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InputKind(str, Enum):
    SAFETENSORS = "safetensors"
    NVFP4 = "nvfp4"
    GGUF = "gguf"
    NINFER = "ninfer"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# TurboForge is a conversion tool: the only user-facing axes are the quantization
# target and the jailbreak processor. Speculative components (MTP / DFlash2) are
# detected and carried through automatically, never configured or trained.
JAILBREAK_CHOICES = ("none", "heretic", "abliteration")
QUANTIZATION_CONFIG_FILES = ("quantize_config.json", "quant_config.json", "hf_quant_config.json")
MODEL_RESOURCE_FILES = frozenset(("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "chat_template.json", "added_tokens.json", "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json", *QUANTIZATION_CONFIG_FILES))



@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity = Severity.INFO
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Callers pass lists for readability; keep one canonical shape so the
        # manifest and reports always serialize alternatives the same way.
        object.__setattr__(self, "alternatives", tuple(self.alternatives))


@dataclass
class TensorSummary:
    name: str
    dtype: str
    shape: list[int]
    byte_size: int


@dataclass
class ModelInspection:
    path: str
    input_kind: InputKind
    architecture: str = "unknown"
    model_type: str = "unknown"
    weight_precision: str = "unknown"
    tensor_count: int = 0
    tensor_bytes: int = 0
    element_count: int = 0
    shard_count: int = 0
    dtype_counts: dict[str, int] = field(default_factory=dict)
    tensor_samples: list[TensorSummary] = field(default_factory=list)
    quantization: dict[str, Any] = field(default_factory=dict)
    tokenizer_files: list[str] = field(default_factory=list)
    speculative_components: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def source_path(self) -> Path:
        return Path(self.path)

    @property
    def usable(self) -> bool:
        return not any(item.severity == Severity.ERROR for item in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["usable"] = self.usable
        result["source_path"] = str(self.source_path)
        return result


@dataclass(frozen=True)
class ProcessorCapability:
    name: str
    supported_architectures: tuple[str, ...]
    input_precisions: tuple[str, ...]
    supports_nvfp4: bool
    requires_high_precision: bool
    requires_calibration: bool
    supports_chunking: bool
    min_ram_bytes: int | None
    min_vram_bytes: int | None
    output_precision: str
    can_quantize_afterwards: bool
    implementation: str
    available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class BackendCapability:
    name: str
    target: str
    input_kinds: tuple[InputKind, ...]
    architectures: tuple[str, ...]
    nominal_bits: float | None
    typical_bpw: float | None
    implementation: str
    available: bool
    unavailable_reason: str | None = None
    verified_conversion: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class ForgeConfig:
    input_path: str
    output_dir: str
    target: str = "auto"
    jailbreak: str = "none"
    jailbreak_params: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    allow_approximate_requantization: bool = False
    source_revision: str | None = None
    # Named extra weights for the converter, e.g. {"quantized": NVFP4 dir,
    # "ternary": PQ2_0 GGUF, "gguf": block GGUF, "imatrix": calibration file}.
    # These never replace input_path: the Hugging Face checkpoint stays --model.
    sources: dict[str, str] = field(default_factory=dict)
    device: str = "cuda"
    run_validation: bool = True
    # Opt-in post-build measurement. Reports are always written; when this is
    # false every metric stays not_measured rather than being estimated.
    run_benchmark: bool = False
    # BenchmarkEngine knobs: repetitions, n_prompt, n_gen, kv_dtype, corpus,
    # perplexity_text / perplexity_corpus / perplexity_context / perplexity_stride,
    # baseline_artifact (for a quantization-loss perplexity delta).
    benchmark: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] = field(default_factory=dict)
    upstream_dir: str | None = None
    ninfer_dir: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ForgeConfig":
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a JSON object")
        # Keys starting with "_" are documentation for humans and never fields.
        payload = {k: v for k, v in value.items() if not k.startswith("_")}
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(unknown)}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject a malformed choice loudly instead of defaulting it away."""
        if not self.input_path:
            raise ValueError("input_path is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.jailbreak not in JAILBREAK_CHOICES:
            raise ValueError(
                f"jailbreak must be one of {', '.join(JAILBREAK_CHOICES)}, "
                f"got {self.jailbreak!r}"
            )
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be cuda or cpu, got {self.device!r}")
        for name in ("input_path", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty path")
        for name in ("runtime", "benchmark", "sources", "jailbreak_params", "quantization"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
        for name in ("perplexity_text", "perplexity_corpus", "baseline_artifact", "corpus"):
            if name in self.benchmark and (not isinstance(self.benchmark[name],str) or not self.benchmark[name].strip()):
                raise ValueError(f"Evaluation file {name} must be a non-empty path")
        for name in ("upstream_dir", "ninfer_dir"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty path")
        for name, value in self.sources.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Source {name} must be a non-empty path")
        for name in ("max_context", "max_concurrency", "port"):
            if name in self.runtime:
                value = self.runtime[name]
                if type(value) is not int or value < 1 or (name == "port" and value > 65535):
                    raise ValueError(f"{name} must be a valid positive integer")
        if self.runtime.get("speculative", "auto") not in ("auto", "none", "mtp", "dflash2"):
            raise ValueError("speculative only supports auto, none, mtp, dflash2")
        for name in ("run_validation", "run_benchmark", "allow_approximate_requantization"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if "auto_quality" in self.benchmark:
            if type(self.benchmark["auto_quality"]) is not bool:
                raise ValueError("auto_quality must be a boolean")
            if self.benchmark["auto_quality"] and (not self.target.startswith("hf_") or self.target == "hf_packed_import" or not self.run_benchmark):
                raise ValueError("Automatic quality comparison only supports new HF quantization with benchmarking enabled; other routes need a same-stage baseline and corpus")
            if self.benchmark["auto_quality"] and (self.benchmark.get("perplexity_text") or self.benchmark.get("perplexity_corpus")):
                raise ValueError("Choose either the automatic evaluation set or a manual evaluation corpus, not both")
        if "max_ppl_relative_increase" in self.benchmark:
            import math
            limit = self.benchmark["max_ppl_relative_increase"]
            if type(limit) not in (int, float) or not math.isfinite(limit) or limit < 0:
                raise ValueError("The quality threshold must be a non-negative finite number")
            if not self.run_benchmark or (not self.target.startswith("hf_") and not self.benchmark.get("baseline_artifact")) or not (self.benchmark.get("auto_quality") or self.benchmark.get("perplexity_text") or self.benchmark.get("perplexity_corpus")):
                raise ValueError("Enabling the quality gate requires benchmarking plus a same-processing-stage baseline artifact and evaluation corpus")
        if self.benchmark.get("perplexity_text") and self.benchmark.get("perplexity_corpus"):
            raise ValueError("Choose either perplexity text or a corpus manifest, not both")
        for name in ("repetitions", "perplexity_context", "perplexity_stride"):
            if name in self.benchmark and (type(self.benchmark[name]) is not int or self.benchmark[name] < 1):
                raise ValueError(f"{name} must be a positive integer")
        if "perplexity_context" in self.benchmark and "perplexity_stride" in self.benchmark and self.benchmark["perplexity_stride"] >= self.benchmark["perplexity_context"]:
            raise ValueError("PPL stride must be smaller than context")
        for name in ("n_prompt", "n_gen"):
            if name in self.benchmark:
                values = self.benchmark[name]
                if not isinstance(values, (list, tuple)) or not values or any(type(v) is not int or v < 1 for v in values):
                    raise ValueError(f"{name} must be a non-empty list of positive integers")


@dataclass
class ResourceEstimate:
    input_bytes: int
    estimated_output_bytes: int | None
    temporary_disk_bytes: int
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None
    available_disk_bytes: int
    available_ram_bytes: int | None
    available_vram_bytes: int | None
    assumptions: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass
class PlanStep:
    name: str
    implementation: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgePlan:
    config: ForgeConfig
    inspection: ModelInspection
    steps: list[PlanStep]
    diagnostics: list[Diagnostic]
    resources: ResourceEstimate | None = None
    backend: str | None = None
    processor: str | None = None
    # Serving profile (KV dtype, speculation, clamped context, launcher flags).
    # These are serving choices, not conversion choices: the artifact is identical
    # regardless. Computed by the shared serving module so the preview, launcher
    # and benchmark agree. None when no recipe was selected.
    serving: dict[str, Any] | None = None

    @property
    def executable(self) -> bool:
        return bool(self.steps) and not any(
            item.severity == Severity.ERROR for item in self.diagnostics
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["executable"] = self.executable
        return result
