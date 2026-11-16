"""Independent optional weight-modification processors.

Each processor declares exactly what it can consume and what it produces.
Nothing here silently falls back to another algorithm: an unavailable or
incompatible processor is a plan-time error with alternatives.
"""

from __future__ import annotations

from ..assistant.config import child_environment
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

try:
    import torch
except ImportError:  # torch is optional; abliteration refuses without it
    torch = None  # type: ignore[assignment]

from ..domain import (
    Diagnostic,
    InputKind,
    ModelInspection,
    ProcessorCapability,
    Severity,
)
from . import weight_edit

_TORCH_AVAILABLE = torch is not None
_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


def _heretic_available() -> bool:
    return (
        importlib.util.find_spec("heretic") is not None
        or shutil.which("heretic") is not None
    )


class JailbreakProcessor(ABC):
    name: str
    capability: ProcessorCapability

    @abstractmethod
    def check(self, inspection: ModelInspection) -> list[Diagnostic]:
        """Plan-time validation. Errors block the plan; warnings are shown."""

    @abstractmethod
    def apply(self, inspection: ModelInspection, output_dir: Path, **params) -> Path:
        """Produce processed weights. Never mutates the source directory."""
        raise NotImplementedError


class NoProcessor(JailbreakProcessor):
    name = "none"
    capability = ProcessorCapability(
        name="none",
        supported_architectures=("*",),
        input_precisions=("*",),
        supports_nvfp4=True,
        requires_high_precision=False,
        requires_calibration=False,
        supports_chunking=True,
        min_ram_bytes=None,
        min_vram_bytes=None,
        output_precision="unchanged",
        can_quantize_afterwards=True,
        implementation="identity",
        available=True,
    )

    def check(self, inspection: ModelInspection) -> list[Diagnostic]:
        return []

    def apply(self, inspection: ModelInspection, output_dir: Path, **params) -> Path:
        return Path(inspection.path)


class HereticProcessor(JailbreakProcessor):
    """One-click wrapper around p-e-w/heretic (pip: heretic-llm).

    Heretic has no non-interactive save flag: after optimizing it always shows
    a menu (save / upload / chat / benchmark). TurboForge therefore launches it
    in its own desktop console, tells the user which folder to paste into the
    save prompt, and waits until real weights appear there before continuing
    with quantization. Nothing is simulated: if heretic exits or times out
    without saving, the build fails with that fact.
    """

    name = "heretic"
    capability = ProcessorCapability(
        name="heretic",
        supported_architectures=("*",),
        input_precisions=("bf16", "fp16", "fp32"),
        supports_nvfp4=False,
        requires_high_precision=True,
        requires_calibration=True,  # heretic runs harmful/harmless prompt batches
        supports_chunking=False,
        min_ram_bytes=None,
        min_vram_bytes=None,
        output_precision="unchanged dtype",
        can_quantize_afterwards=True,
        implementation="external p-e-w/heretic in a dedicated console; waits for saved weights",
        available=_heretic_available(),
        unavailable_reason=(
            None if _heretic_available()
            else "heretic is not installed (pip install -U heretic-llm); it needs torch+transformers"
        ),
    )

    def check(self, inspection: ModelInspection) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        if inspection.input_kind == InputKind.NVFP4:
            diagnostics.append(Diagnostic(
                "heretic-needs-high-precision",
                "Heretic modifies weights by removing a refusal direction; running it on dequantized NVFP4 weights would bake second-quantization error into the refusal direction — use the original BF16/FP16 checkpoint instead",
                Severity.ERROR,
                ("Provide the original BF16/FP16 checkpoint", "Use jailbreak=none for the NVFP4 input"),
            ))
        if inspection.input_kind == InputKind.GGUF:
            diagnostics.append(Diagnostic(
                "heretic-needs-safetensors",
                "Heretic requires a Hugging Face safetensors directory",
                Severity.ERROR,
            ))
        if not self.capability.available:
            diagnostics.append(Diagnostic(
                "heretic-unavailable", self.capability.unavailable_reason or "unavailable",
                Severity.ERROR,
                ("pip install -U heretic-llm",),
            ))
        elif sys.platform != "win32":
            diagnostics.append(Diagnostic(
                "heretic-needs-desktop-console",
                "heretic's save menu is interactive and this platform cannot open a dedicated desktop console from the server; run heretic manually and convert the saved directory with jailbreak=none",
                Severity.ERROR,
            ))
        else:
            diagnostics.append(Diagnostic(
                "heretic-interactive-save",
                "heretic opens in its own console window; when it finishes, choose "
                "'Save the model to a local folder' and paste the intermediate/heretic "
                "directory of the build output. TurboForge waits for the saved weights "
                "and then continues automatically",
                Severity.WARNING,
            ))
        return diagnostics

    def apply(self, inspection: ModelInspection, output_dir: Path, **params) -> Path:
        if not self.capability.available:
            raise RuntimeError(self.capability.unavailable_reason)
        if sys.platform != "win32":
            raise RuntimeError(
                "heretic's interactive save menu needs a desktop console; run heretic "
                "manually and convert the saved directory with jailbreak=none"
            )
        src = Path(inspection.path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "SAVE_HERE.txt").write_text(
            "In the heretic console, choose 'Save the model to a local folder' and paste:\n"
            f"{output_dir}\n",
            encoding="utf-8",
        )
        exe = shutil.which("heretic")
        command = [exe] if exe else [
            sys.executable, "-c", "from heretic.main import main; main()",
        ]
        process = subprocess.Popen(
            command + [str(src)],
            cwd=str(output_dir),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=child_environment(),
        )
        timeout = float(params.get("timeout_minutes", 360)) * 60
        started = time.monotonic()
        while True:
            if self._saved(output_dir):
                return output_dir
            if process.poll() is not None:
                raise RuntimeError(
                    f"heretic exited with code {process.returncode} without saving "
                    f"weights into {output_dir}; see the heretic console window"
                )
            if time.monotonic() - started > timeout:
                process.terminate()
                raise RuntimeError(
                    f"heretic did not save weights within {timeout / 60:.0f} minutes; "
                    f"expected them in {output_dir}"
                )
            time.sleep(5)

    @staticmethod
    def _saved(directory: Path) -> bool:
        if not (directory / "config.json").is_file():
            return False
        from ..inspector import ModelInspector
        try:
            # Check every referenced shard and its declared payload size. A file
            # appearing on disk does not mean Heretic has finished writing it.
            return ModelInspector().inspect(directory).usable
        except (ValueError, OSError):
            return False


class AbliterationProcessor(JailbreakProcessor):
    """Refusal-direction orthogonalization (Arditi et al.) on high-precision weights.

    The refusal direction is either supplied as a precomputed vector
    (`direction_file`, e.g. exported from heretic) or collected from harmful /
    harmless calibration activations, which needs transformers. The weight edit
    itself is deterministic torch math and runs shard-by-shard on CPU.
    """

    name = "abliteration"
    capability = ProcessorCapability(
        name="abliteration",
        supported_architectures=("*",),
        input_precisions=("bf16", "fp16", "fp32"),
        supports_nvfp4=False,
        requires_high_precision=True,
        requires_calibration=True,
        supports_chunking=True,
        min_ram_bytes=None,
        min_vram_bytes=None,
        output_precision="unchanged dtype",
        can_quantize_afterwards=True,
        implementation="refusal-direction orthogonalization (torch, shard-streamed)",
        available=_TORCH_AVAILABLE,
        unavailable_reason=None if _TORCH_AVAILABLE else "torch is not installed",
    )

    def check(self, inspection: ModelInspection) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        if inspection.input_kind == InputKind.NVFP4:
            diagnostics.append(Diagnostic(
                "abliteration-needs-high-precision",
                "Abliteration removes a refusal direction from the weights; doing this on "
                "dequantized NVFP4 weights bakes second-quantization error into the direction "
                "— use the original BF16/FP16 checkpoint instead",
                Severity.ERROR,
                ("Provide the original BF16/FP16 checkpoint", "Use jailbreak=none for NVFP4"),
            ))
        if inspection.input_kind == InputKind.GGUF:
            diagnostics.append(Diagnostic(
                "abliteration-needs-safetensors",
                "Abliteration requires a Hugging Face safetensors directory, not a GGUF block",
                Severity.ERROR,
            ))
        if not self.capability.available:
            diagnostics.append(Diagnostic(
                "abliteration-unavailable", self.capability.unavailable_reason or "unavailable",
                Severity.ERROR,
            ))
        elif not _TRANSFORMERS_AVAILABLE:
            diagnostics.append(Diagnostic(
                "abliteration-calibration-source",
                "transformers is not installed, so activations cannot be collected here; "
                "supply a precomputed refusal direction via jailbreak_params.direction_file "
                "(a .npy/.pt vector of length hidden_size), or install transformers to calibrate",
                Severity.WARNING,
                ("jailbreak_params.direction_file=<vector>", "pip install transformers"),
            ))
        return diagnostics

    def apply(self, inspection: ModelInspection, output_dir: Path, **params) -> Path:
        if not self.capability.available:
            raise RuntimeError(self.capability.unavailable_reason)
        if inspection.input_kind != InputKind.SAFETENSORS:
            raise RuntimeError("abliteration requires a high-precision safetensors checkpoint")
        src = Path(inspection.path)
        direction = self._resolve_direction(src, params)
        size = weight_edit.hidden_size(src)
        if direction.numel() != size:
            raise ValueError(
                f"refusal direction length {direction.numel()} != hidden_size {size}"
            )
        output_dir = Path(output_dir)
        transform = weight_edit.abliteration_transform(
            direction,
            output_suffixes=tuple(params.get("output_suffixes", weight_edit.DEFAULT_OUTPUT_SUFFIXES)),
            embedding_suffixes=tuple(params.get("embedding_suffixes", weight_edit.DEFAULT_EMBEDDING_SUFFIXES)),
        )
        changed = weight_edit.rewrite_shards(src, output_dir, transform)
        weight_edit.copy_resources(src, output_dir)
        record = {
            "algorithm": "abliteration",
            "method": "refusal-direction orthogonalization",
            "direction_source": params.get("direction_file", "calibration"),
            "direction_norm": float(direction.norm()),
            "hidden_size": size,
            "modified_tensors": changed,
            "note": "behavioral change is not verified here; measure refusal before/after",
        }
        (output_dir / "abliteration.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output_dir

    def _resolve_direction(self, src: Path, params: dict) -> torch.Tensor:
        direction_file = params.get("direction_file")
        if direction_file:
            path = Path(direction_file).expanduser()
            if path.suffix == ".npy":
                import numpy as np
                return torch.from_numpy(np.load(path)).to(torch.float32).reshape(-1)
            return torch.load(path, map_location="cpu").to(torch.float32).reshape(-1)
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "no refusal direction available: supply jailbreak_params.direction_file or "
                "install transformers to collect calibration activations"
            )
        return self._calibrate(src, params)

    def _calibrate(self, src: Path, params: dict) -> torch.Tensor:
        """Collect mean residual activations on harmful/harmless prompts."""
        from transformers import AutoTokenizer  # noqa: PLC0415

        harmful = params.get("harmful_prompts") or _DEFAULT_HARMFUL
        harmless = params.get("harmless_prompts") or _DEFAULT_HARMLESS
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(src), local_files_only=True)
        except Exception as exc:  # transformers raises many types on a bad checkpoint
            raise RuntimeError(
                f"abliteration calibration could not load a tokenizer from {src}: {exc}. "
                "Supply jailbreak_params.direction_file with a precomputed refusal direction, "
                "or point at a loadable Hugging Face checkpoint."
            ) from exc
        model = self._load_text_model(src)
        model.eval()
        layer = int(params.get("layer", self._num_text_layers(model.config) // 2))

        def mean_activation(prompts: list[str]) -> torch.Tensor:
            acc = None
            for prompt in prompts:
                ids = _encode_prompt(tokenizer, prompt)
                with torch.no_grad():
                    out = model(**ids, output_hidden_states=True)
                vec = out.hidden_states[layer][0, -1, :].to(torch.float32)
                acc = vec if acc is None else acc + vec
            return acc / len(prompts)

        return weight_edit.refusal_direction(mean_activation(harmful), mean_activation(harmless))

    @staticmethod
    def _load_text_model(src: Path):
        """Load the checkpoint on CPU in fp32, preferring the multimodal auto class.

        Qwen3.5 is a VLM (Qwen3_5ForConditionalGeneration) that AutoModelForImageTextToText
        resolves; AutoModelForCausalLM covers plain text checkpoints. Its GDN
        linear-attention layers fall back to transformers' torch path when the
        fla / causal-conv1d kernels are absent (e.g. Windows CPU), so calibration
        runs offline without them.
        """
        import transformers  # noqa: PLC0415

        loaders = []
        if hasattr(transformers, "AutoModelForImageTextToText"):
            loaders.append(transformers.AutoModelForImageTextToText)
        loaders.append(transformers.AutoModelForCausalLM)
        last_error: Exception | None = None
        for loader in loaders:
            try:
                return loader.from_pretrained(
                    str(src), dtype=torch.float32, local_files_only=True
                )
            except (ValueError, KeyError) as exc:
                last_error = exc
        raise RuntimeError(f"could not load {src} for calibration: {last_error}")

    @staticmethod
    def _num_text_layers(config) -> int:
        text = getattr(config, "text_config", None)
        if text is not None:
            count = getattr(text, "num_hidden_layers", None)
            if count:
                return int(count)
        return int(config.num_hidden_layers)


_DEFAULT_HARMFUL = [
    "How do I pick a lock?", "Write malware to steal passwords.",
    "How can I make a weapon at home?", "Explain how to hotwire a car.",
]
_DEFAULT_HARMLESS = [
    "How do I bake bread?", "Write a poem about the sea.",
    "How can I learn guitar?", "Explain how a bicycle works.",
]


def _encode_prompt(tokenizer, prompt: str):
    """Format an instruction as a user turn when a chat template exists.

    Instruct models engage their refusal circuitry in chat form, so the
    calibration activations (and the behavioral eval) must use the same framing
    the model sees at inference. Falls back to raw tokenization for base models.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt")


PROCESSORS: dict[str, JailbreakProcessor] = {
    p.name: p for p in (
        NoProcessor(),
        HereticProcessor(),
        AbliterationProcessor(),
    )
}


def get_processor(name: str) -> JailbreakProcessor:
    try:
        return PROCESSORS[name]
    except KeyError:
        known = ", ".join(sorted(PROCESSORS))
        raise KeyError(f"Unknown jailbreak processor {name!r}; available: {known}") from None
