"""Native AWQ/GPTQ checkpoints, separate from the NInfer storage/runtime.

Contracts checked against the PyPI GPTQModel 7.10.0 source distribution.
No packed checkpoint is dequantized and silently quantized a second time.
This module is stdlib-only so precheck never installs tensor libraries.
"""
from __future__ import annotations

from ..architecture import architecture_names, hidden_size, max_context, text_config
from ..assistant.config import child_environment
import os
from pathlib import Path
import subprocess
import sys

from ..domain import Diagnostic, ForgeConfig, ForgePlan, InputKind, PlanStep, Severity
from ..local_endpoint import default_serving_port
from .registry import RecipeSpec

GPTQMODEL_VERSION = "7.10.0"
HF_DEFAULT_BACKEND = "auto"
# Used by planning, selection and assistant knowledge; a disabled route must not
# reappear because a separate help table was missed during a capability change.
AWQ_DISABLED_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
AWQ_DISABLED_REASON = "New Qwen3.5 AWQ quantization did not pass local quality acceptance and is disabled; choose GPTQ or NInfer. Existing AWQ files support only verified native import; quantization is never redone automatically."
HF_CORE_VERSIONS = {"transformers":"5.20.2","datasets":"5.2.0","safetensors":"0.8.0","torchao":"0.18.0","pypcre":"0.6.4"}
HF_ACCEL_VERSIONS = {"triton-windows":"3.8.0.post29","flash-linear-attention":"0.6.0","fla-core":"0.6.0","einops":"0.8.2"}
HF_ARCHITECTURES = (
    "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
)
HF_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3", "qwen3_5", "qwen3_5_text")
_MODEL_TYPE_BY_ARCHITECTURE = dict(zip(HF_ARCHITECTURES,("llama","mistral","qwen2","qwen3","qwen3_5_text","qwen3_5")))
HF_RECIPES = tuple(
    RecipeSpec(f"hf_{method}_{bits}", f"{method.upper()} {bits} bit · HF native format",
               (InputKind.SAFETENSORS,), HF_ARCHITECTURES, (f"hf_{method}",), float(bits), None,
               notes=("GPTQModel calibration and native packing; uses separate HF inference, not converted to NInfer.",
                      "The calibration set is downloaded automatically, with batch size 1, CPU activation cache and disk offload; VRAM and speed follow the actual reports."),
               engine="gptqmodel")
    for method, widths in (("awq", (4,)), ("gptq", (2, 3, 4, 8))) for bits in widths
) + (RecipeSpec("hf_packed_import", "Retain AWQ/GPTQ native format · HF engine",
                (InputKind.SAFETENSORS,), HF_ARCHITECTURES, ("hf_packed",), None, None,
                notes=("Reuses existing packed weights without dequantization or bit-width change; arbitrary vendor-reordered layouts are not supported.",), engine="gptqmodel"),)


def is_hf(target: str) -> bool:
    return any(recipe.recipe == target for recipe in HF_RECIPES)


def quant_method(config: dict) -> str:
    quant = config.get("quantization_config") or {}
    if not isinstance(quant,dict):
        raise ValueError("quantization_config must be an object")
    return str(quant.get("quant_method", quant.get("method", ""))).lower()


def supported(config: dict) -> bool:
    architectures = architecture_names(config)
    hidden = hidden_size(config)
    return type(hidden) is int and hidden % 32 == 0 and any(
        _MODEL_TYPE_BY_ARCHITECTURE.get(item) == config.get("model_type")
        for item in architectures
    )


def check_packed(config: dict) -> None:
    quant = config.get("quantization_config") or {}
    method = quant_method(config)
    for aliases in (("quant_method","method"),("bits","w_bit"),("group_size","q_group_size"),("checkpoint_format","format")):
        values = [str(quant[key]).lower() for key in aliases if key in quant]
        if len(set(values)) > 1:
            raise ValueError("Conflicting quantization metadata fields: "+" / ".join(aliases))
    if method == "awq" and "version" in quant:
        for key in ("checkpoint_format","format"):
            if key in quant and str(quant[key]).lower() != str(quant["version"]).lower():
                raise ValueError("The AWQ version conflicts with the declared packing layout")
    if str(quant.get("pack_dtype","int32")).lower() not in ("int32","torch.int32"):
        raise ValueError("Only INT32 packing is currently supported")
    bits = quant.get("bits", quant.get("w_bit"))
    fmt = str(quant.get("checkpoint_format", quant.get("format", quant.get("version", "gemm" if method == "awq" else "gptq")))).lower()
    if method == "awq" and (bits != 4 or fmt != "gemm"):
        raise ValueError("AWQ import supports only the 4-bit GEMM native layout; GEMV/Marlin/other reordered layouts cannot pass as GEMM")
    if method == "gptq" and (type(bits) is not int or bits not in (2, 3, 4, 8) or fmt not in ("gptq", "gptq_v2")):
        raise ValueError("GPTQ import supports only the 2/3/4/8-bit GPTQ/GPTQ_V2 native layouts")
    if method not in ("awq", "gptq"):
        raise ValueError("Not a supported AWQ/GPTQ native checkpoint")
    group = quant.get("group_size",quant.get("q_group_size",128))
    if type(group) is not int or group not in (-1,32,64,128):
        raise ValueError("Native AWQ/GPTQ import supports only -1/32/64/128 group sizes")
    if quant.get("dynamic") or quant.get("activation") or quant.get("adapter"):
        raise ValueError("This import does not accept mixed bit widths, extra activation quantization or adapter configs; provide a standard AWQ/GPTQ checkpoint")


def recipes_for_config(config: dict) -> list[RecipeSpec]:
    if not supported(config):
        return []
    if quant_method(config) in ("awq", "gptq"):
        check_packed(config)
        return [HF_RECIPES[-1]]
    if config.get("quantization_config"):
        return []
    return [recipe for recipe in HF_RECIPES[:-1] if not (recipe.recipe == "hf_awq_4" and config.get("model_type") in AWQ_DISABLED_MODEL_TYPES)]


def validate_options(config: ForgeConfig) -> None:
    if config.device != "cuda":
        raise ValueError("The AWQ/GPTQ native route needs CUDA; auto mode prepares the GPU environment")
    values = config.quantization
    allowed = {"calibration_file", "samples", "sequence_length", "group_size", "backend"}
    if set(values) - allowed:
        raise ValueError("Unknown AWQ/GPTQ options: " + ", ".join(sorted(set(values) - allowed)))
    for key, default, low, high in (("samples",128,1,4096),("sequence_length",512,32,4096)):
        value = values.get(key,default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer in {low}–{high}")
    if type(values.get("group_size",128)) is not int or values.get("group_size",128) not in (32,64,128):
        raise ValueError("AWQ/GPTQ group size supports only 32, 64, 128")
    if values.get("backend","auto") not in ("torch","auto"):
        raise ValueError("The AWQ/GPTQ inference backend supports only torch or auto")
    if values.get("calibration_file") and (not isinstance(values["calibration_file"],str) or not Path(values["calibration_file"]).expanduser().is_file()):
        raise ValueError("The calibration text file does not exist; select a UTF-8 text file. Leave empty to download WikiText-2 automatically")
    if values.get("calibration_file") and config.benchmark.get("perplexity_text"):
        calibration = Path(values["calibration_file"]).expanduser().read_text(encoding="utf-8-sig")
        evaluation_path = Path(config.benchmark["perplexity_text"]).expanduser()
        if not evaluation_path.is_file():
            raise ValueError("The quality evaluation text file does not exist")
        evaluation = evaluation_path.read_text(encoding="utf-8-sig")
        if calibration == evaluation:
            raise ValueError("Calibration and quality evaluation cannot use corpora with identical content")
    if config.runtime.get("speculative","auto") not in ("auto","none") or config.sources:
        raise ValueError("The HF AWQ/GPTQ route does not yet run MTP/DFlash2 or NInfer extra sources; choose off or auto (speculation is off on this route)")
    if config.runtime.get("max_concurrency",1) != 1:
        raise ValueError("The HF AWQ/GPTQ local service currently supports single-request concurrency; set it to 1")
    if set(config.runtime) - {"speculative","max_context","max_concurrency","port"}:
        raise ValueError("The HF route does not accept NInfer-specific cache or decoding flags; use the HF default runtime parameters")
    if config.jailbreak == "heretic":
        raise ValueError("For HF auto quantization use none or Abliteration; Heretic's interactive save does not apply to this route")
    if not config.run_validation:
        raise ValueError("The HF native route must validate GPU load and generation; validation cannot be skipped")
    if config.benchmark.get("perplexity_text") and not config.run_benchmark:
        raise ValueError("HF PPL evaluation requires run_benchmark to be enabled")
    if config.benchmark.get("repetitions",3) > 30 or len(config.benchmark.get("n_gen",[64])) != 1:
        raise ValueError("HF speed tests repeat at most 30 times, and n_gen accepts only one length")
    unsupported = set(config.benchmark) - {"auto_quality","perplexity_text","perplexity_context","perplexity_stride","max_ppl_relative_increase","repetitions","n_gen"}
    if unsupported:
        raise ValueError("HF evaluation automatically uses the same-stage original weights as the baseline and supports text evaluation only; unsupported: " + ", ".join(sorted(unsupported)))


def plan(config: ForgeConfig, inspection) -> ForgePlan:
    diagnostics = list(inspection.diagnostics)
    try:
        validate_options(config)
        if any(inspection.speculative_components.get(name,{}).get("detected") for name in ("mtp","dflash2")) and config.runtime.get("speculative","auto") != "none":
            raise ValueError("This model ships MTP/DFlash2, but the HF engine does not yet run these components. To keep the acceleration choose the NInfer route; if you only need native AWQ/GPTQ with losslessly archived heads, explicitly turn speculative decoding off. No automatic downgrade")
        candidates = recipes_for_config(inspection.config)
        if config.target == "hf_awq_4" and inspection.config.get("model_type") in AWQ_DISABLED_MODEL_TYPES:
            raise ValueError(AWQ_DISABLED_REASON)
        if config.target not in [recipe.recipe for recipe in candidates]:
            raise ValueError("The selected AWQ/GPTQ route is incompatible with this model's architecture, precision or packed layout")
        if config.target != "hf_packed_import" and inspection.weight_precision not in ("bf16","fp16","fp32"):
            raise ValueError("AWQ/GPTQ calibration needs original BF16/FP16/FP32 weights")
        if config.target == "hf_packed_import" and config.jailbreak != "none":
            raise ValueError("Existing AWQ/GPTQ weights cannot be abliterated directly; quantize from the original high-precision weights")
        if config.target == "hf_packed_import" and "max_ppl_relative_increase" in config.benchmark:
            raise ValueError("The already-quantized input lacks same-stage original weights and cannot serve as its own quality-gate baseline")
        output, source = Path(config.output_dir).resolve(), Path(inspection.path).resolve()
        if output == source or source in output.parents or output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("The output must be a new empty directory outside the original model directory")
        if not inspection.tokenizer_files:
            raise ValueError("AWQ/GPTQ needs complete tokenizer files")
        if not Path(inspection.path).is_dir():
            raise ValueError("For AWQ/GPTQ select a complete model directory, not a single Safetensors shard")
        if inspection.architecture == "Qwen3_5ForConditionalGeneration" and any(not (source/name).is_file() for name in ("preprocessor_config.json","video_preprocessor_config.json")):
            raise ValueError("The Qwen3.5 HF route lacks image/video processor config; provide the original repository link in auto mode to complete it. No manual weight download is needed")
        if config.runtime.get("max_context",2048) < 32:
            raise ValueError("The HF local service context must be at least 32 tokens")
        if config.benchmark.get("perplexity_text") or config.benchmark.get("auto_quality"):
            limit = max_context(inspection.config)
            context = min(config.benchmark.get("perplexity_context",512),config.runtime.get("max_context",2048),limit)
            stride = config.benchmark.get("perplexity_stride",context//2)
            if not 0 < stride < context:
                raise ValueError("The PPL stride must be smaller than the effective evaluation context")
            from ..system import memory_available
            available = memory_available()
            required = inspection.element_count * 4 + 1024**3
            if available is not None and required > available:
                raise ValueError(f"The quality baseline needs about {required/1024**3:.1f} GiB of available RAM (FP32 original weights plus headroom), which is not currently available; close other programs or explicitly skip the quality baseline")
        if config.quantization.get("sequence_length",512) > max_context(inspection.config):
            raise ValueError("The calibration sample length exceeds the model's declared context limit")
    except ValueError as exc:
        diagnostics.append(Diagnostic("hf-configuration", str(exc), Severity.ERROR))
    diagnostics.append(Diagnostic("hf-native-runtime", "This route outputs an HF native checkpoint using the separate GPTQModel engine; it does not run NInfer's MTP/DFlash2. The new route has no local large-model measurements yet.", Severity.WARNING))
    context = min(config.runtime.get("max_context",2048), max_context(inspection.config))
    serving = {"engine":"gptqmodel", "backend":config.quantization.get("backend","auto"), "max_context":context,
               "max_concurrency":1, "port":config.runtime.get("port") or default_serving_port(), "launcher_flags":[], "speculative":"none",
               "kv_dtype":"Determined by the HF model dtype","context_ceiling":text_config(inspection.config).get("max_position_embeddings"),"spec":None,"draft_tokens":0,"embedding_q4":False}
    selected = next((r for r in HF_RECIPES if r.recipe == config.target),None)
    return ForgePlan(config, inspection,
                     [PlanStep("jailbreak",config.jailbreak,"skipped" if config.jailbreak == "none" else "planned"),
                      PlanStep("quantize+compile",config.target,"planned",{"engine":"gptqmodel", "nominal_bits":selected.nominal_bits if selected else None,
                               "label":selected.label if selected else config.target,"device":"cuda","components":"text","storage_formats":list(selected.storage_formats) if selected else []}),
                      PlanStep("speculative","none","skipped",{"reason":"The HF route does not package speculative decoding"}),
                      PlanStep("validate","HF GPU load and generation","planned")],
                     diagnostics, backend=config.target,processor=config.jailbreak,serving=serving)


def environment_python() -> Path:
    from ..bootstrap import cache_root
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    if os.environ.get("TURBOFORGE_HF_HOME"):
        return Path(os.environ["TURBOFORGE_HF_HOME"]).expanduser()/"hf-env"/suffix
    if os.name != "nt":
        return cache_root()/"hf-env"/suffix
    # Keep multi-GB GPU dependencies off the project/output disk by default.
    return Path(os.environ["LOCALAPPDATA"])/"NInferTurboForge/hf-env-py31318"/suffix


def ensure_environment(emit=print) -> Path:
    from ..bootstrap import run, cuda_package, cache_root
    if emit is print:
        emit = lambda message: print(message,flush=True)
    python = environment_python()
    if not python.is_file():
        from .hf_environment import bootstrap_python
        run([str(bootstrap_python(python.parent.parent.parent,emit)),"-m","venv",str(python.parent.parent)],emit)
    # Probe exact package/API contracts without importing heavyweight libraries
    # into the Web process or upgrading the existing NInfer worker environment.
    probe_env = child_environment({**os.environ,"PYTHONIOENCODING":"utf-8"})
    if os.name == "nt":
        version_check = subprocess.run([str(python),"-c","import sys;assert (3,13,18) <= sys.version_info < (3,14)"],capture_output=True,text=True,encoding="utf-8",env=probe_env)
        if version_check.returncode:
            raise RuntimeError("This custom HF environment uses an old Python; no download or upgrade was performed. Choose a new TURBOFORGE_HF_HOME so the script prepares a Python 3.13.18 environment automatically; the original environment and model are kept.")
    probe_code = "from importlib.metadata import version; import torch, torchvision; from gptqmodel import GPTQModel, QuantizeConfig; assert version('gptqmodel') == '7.10.0'; assert torch.__version__ == '2.14.1+cu130'; assert version('torchvision') == '0.29.1+cu130'; assert torch.cuda.is_available(); assert torch.version.cuda"
    probe_code += "; assert all(version(name) == expected for name,expected in " + repr(HF_CORE_VERSIONS) + ".items())"
    probe_code += "; import os; assert os.name != 'nt' or torch.cuda.get_device_capability()[0] < 8 or all(version(name) == expected for name,expected in " + repr(HF_ACCEL_VERSIONS) + ".items())"
    probe = subprocess.run([str(python),"-c",probe_code],capture_output=True,text=True,encoding="utf-8",errors="replace", env=probe_env)
    if probe.returncode:
        package = cuda_package()
        if package != "13":
            raise RuntimeError("The AWQ/GPTQ automatic GPU environment currently requires an NVIDIA driver with CUDA 13 support (580+); the existing NInfer CUDA 12 route is unaffected")
        version = "2.14.1"  # Verified cu130 Windows wheels for CPython 3.10–3.14.
        emit("Installing the AWQ/GPTQ environment and official CUDA 13 PyTorch separately; the existing conversion environment is not modified")
        run([str(python),"-m","pip","install","--upgrade","pip"],emit)
        wheels = []
        if os.name == "nt":
            from .hf_wheels import cuda_wheels
            wheels = cuda_wheels(python,cache_root()/"downloads/hf-wheels",emit)
        run([str(python),"-m","pip","install",*([str(path) for path in wheels] if wheels else [f"torch=={version}","torchvision==0.29.1"]),"--index-url","https://download.pytorch.org/whl/cu130"],emit)
        from .hf_environment import install_pcre
        install_pcre(python,emit)
        run([str(python),"-m","pip","install",f"gptqmodel=={GPTQMODEL_VERSION}","flask>=3.0",f"torch=={version}+cu130","torchvision==0.29.1+cu130","transformers==5.20.2","datasets==5.2.0","safetensors==0.8.0","torchao==0.18.0","pypcre==0.6.4"],emit)
        if os.name == "nt":
            capability = subprocess.run([str(python),"-c","import torch; print(torch.cuda.get_device_capability()[0])"],capture_output=True,text=True,encoding="utf-8",check=True,env=probe_env)
            if int(capability.stdout.strip()) >= 8:
                emit("Automatically preparing Triton and linear-attention acceleration kernels; the first run compiles and caches them")
                run([str(python),"-m","pip","install",*[name+"=="+version for name,version in HF_ACCEL_VERSIONS.items()]],emit)
        run([str(python),"-m","pip","check"],emit)
        probe = subprocess.run([str(python),"-c",probe_code],capture_output=True,text=True,encoding="utf-8",errors="replace", env=probe_env)
        if probe.returncode:
            raise RuntimeError("AWQ/GPTQ environment import or CUDA check failed: " + probe.stderr[-2000:])
        for wheel in wheels:
            wheel.unlink()
            wheel.with_name(wheel.name+".sha256").unlink(missing_ok=True)
    return python


def server_command(config: dict, output: Path, profile: dict) -> list[str]:
    from .hf_worker import verify_checkpoint
    checkpoint = output/"model-hf"
    verify_checkpoint(checkpoint)
    try:
        python = ensure_environment()
    except OSError as exc:
        raise RuntimeError("HF runtime preparation was interrupted; downloaded parts are kept for resuming on the next start. Check the specific download error in the launch window, then retry") from exc
    return [str(python),"-m","turboforge.backends.hf_worker","serve","--model",str(checkpoint),
            "--port",str(profile["port"]),"--context",str(profile["max_context"]),
            "--backend",config.get("quantization",{}).get("backend","auto")]
