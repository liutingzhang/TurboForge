"""Read-only guided choices from the same recipe registry used by execution."""
from pathlib import Path
import json
import re
from .architecture import (
    DEEPSEEK_V41,
    FLASH_NEXT,
    GEMMA4,
    QWEN35_DENSE,
    QWEN35_MOE,
    architecture_kind,
    architecture_names,
    hidden_size,
    text_config,
)
from .bootstrap import cuda_package, read_url
from .domain import InputKind
from .inspector import ModelInspector
from .adapters.safetensors import _contains_nvfp4
from .model_sources import ModelResolver, reference_parts, normalize_reference
from .preparation import check_config_support
from .backends.registry import (
    RECIPES,
    geometry_admits,
    gpu_scheme_note as _gpu_scheme_note,
    gpu_scheme_order as _gpu_scheme_order,
)
from .domain import QUANTIZATION_CONFIG_FILES
from .system import detect_gpu


SOURCE_FIELDS = {
    "quantized": {"label":"NVFP4 quantized checkpoint", "kind":"directory"},
    "gguf": {"label":"GGUF weights", "kind":"file"},
    "ternary": {"label":"PQ2_0 ternary GGUF", "kind":"file"},
    "imatrix": {"label":"Importance matrix (imatrix)", "kind":"file"},
    "engram": {"label":"Engram table GGUF", "kind":"file"},
}


# GPU-native format notes/ordering live in backends.registry (they depend on the
# storage formats defined there); they are imported above as presentation-only
# helpers and never override an explicit user target.


# Advisory runtime selection. TurboForge still emits .ninfer (or HF AWQ/GPTQ) and
# launches its bundled engine; these hints never download or start another runtime.
# Basalt is a Strata fork built only for consumer Blackwell (sm_120, compute 12.0).
# Datacenter Blackwell is compute 10.0 and is not sm_120. Strata covers RTX 20/30/40/50,
# AMD, Intel Arc (Linux) and Apple Silicon. Strata reads GGUF, Basalt reads .basalt.
_BASALT_COMPUTE_CAP = 12.0
# Published model lists (README, verified 2026-11-15): Basalt ships MoE support
# only for Flash-Next and DeepSeek-V4.1-Flash; Qwen3.6-35B-A3B is Strata-only
# even on RTX 50. An MoE config outside both lists gets no engine promise.
_BASALT_MOE_ARCHITECTURES = frozenset((*FLASH_NEXT, *DEEPSEEK_V41))
_STRATA_MOE_ARCHITECTURES = frozenset((*FLASH_NEXT, *DEEPSEEK_V41, *QWEN35_MOE))
_MOE_ENGINE_ADVISORY = ("Engine-selection advice only, no engine integration; TurboForge produces .ninfer for the companion NInfer engine, "
                        "Strata uses GGUF and Basalt uses .basalt; each needs its own source/conversion and they cannot be fed to each other directly.")
_BASALT_URL = "https://github.com/jesdga95/basalt"
_STRATA_URL = "https://github.com/Niko1221/Strata"
# fusion-kvmem publishes an Apache-2.0 license and Windows sm_120/89/86 builds.
# ninfer-kvmem-actkv has no LICENSE file, so it is not offered.
_FUSION_URL = "https://github.com/1314521gjy/ninfer-fusion-kvmem"
_AGENT_BAT = ("Hard ordering for the assistant: first ask the user two separate questions, 'context length (positive integer of tokens)' and 'simultaneous requests (concurrency, positive integer)'; "
               "do not write the launcher script before getting the numbers, and do not assume them for the user. After the answers, generate exactly one .bat with an ASCII file name, "
               "using cmd.exe only: first line @echo off, then chcp 65001, setlocal EnableExtensions, cd /d \"%~dp0\"; "
               "quote every path; every failure branch echoes the reason, then pause, then exit /b 1; also pause after the engine command finishes, "
               "to avoid the double-click window closing instantly on Win10/Win11. Do not use PowerShell syntax or parameters not verified in the corresponding entry.")
_BASALT_STEPS = (_AGENT_BAT + "Search J12 in the help center. Verified: basalt serve --config <json>; context goes through the command-line --max-context; "
                 "the release package contains neither CUDA 13 nor Python; provide them yourself. Concurrency has no verified config field; if the answer is greater than 1, "
                 "you must tell the user this run starts with a single request; never write concurrency into an unverified field.")
_STRATA_STEPS = (_AGENT_BAT + "Search J13 in the help center. The only verified items are strata launch, --family, --model, --plan; "
                 "no --context or concurrency flag has been verified. After getting the two numbers, if the engine --help has no matching flag, "
                 "the bat uses only verified flags and explicitly states the context and concurrency that were not passed; never invent parameters.")
_FUSION_STEPS = (_AGENT_BAT + "Search J14 in the help center; the entry contains the full commands and troubleshooting. "
                 "Summary: (1) In Releases pick engine-v0.11.0-kvmem-20261008; on RTX 50 use ninfer-serve-120a.exe, "
                 "40 series -89, 30 series -86, and verify SHA256; (2) put the .ninfer produced on this page and the engine on an English path; "
                 "(3) set the KV environment variables per the docs then launch; on a 12GB card start with the documented --kv-capacity; "
                 "(4) open another terminal, visit /v1/models and have it answer an arithmetic question. That repository is Apache-2.0; "
                 "the unlicensed melis2023/ninfer-kvmem-actkv is not recommended.")
# Shown up front for an RTX 50 card whose MoE is NOT in Basalt's adapted list.
_BASALT_FASTER_UNADAPTED = ("Basalt is usually faster on RTX 50 (sm_120), but this model is not in its published MoE list "
                            "(only Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash). "
                            "Check yourself whether a compatible .basalt exists; if not, use the default engine below.")


def _moe_engine_recommendation(config: dict, gpu: dict | None) -> dict | None:
    if architecture_kind(config) != "moe":
        return None
    architectures = architecture_names(config)
    in_basalt = any(a in _BASALT_MOE_ARCHITECTURES for a in architectures)
    in_strata = any(a in _STRATA_MOE_ARCHITECTURES for a in architectures)
    cap = gpu["compute_cap"] if gpu else None
    rtx50 = cap == _BASALT_COMPUTE_CAP
    if rtx50 and in_basalt:
        return {
            "engine": "basalt", "label": "Basalt",
            "reason": f"This card {gpu['generation']} (compute {cap:g}) is consumer Blackwell sm_120, and this MoE is in Basalt's published list; officially native NVFP4 and faster on RTX 50",
            "requirements": "NVIDIA RTX 50 (sm_120) + CUDA 13 + driver R580+; Linux x64 or Win10 1903+ x64; the release package contains no CUDA/Python, provide them yourself; experts that do not fit go to system RAM/SSD",
            "format": ".basalt (v2, self-contained: weights/tokenizer/MTP/vision projection)",
            "models": "Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash",
            "alternative": "Strata also supports this card and model, in GGUF format; a cross-platform alternative",
            "url": _BASALT_URL, "steps": _BASALT_STEPS,
            "advisory": _MOE_ENGINE_ADVISORY,
        }
    compat_check = ""
    if rtx50:
        compat_check = _BASALT_FASTER_UNADAPTED
        if in_strata:
            reason = (f"This card {gpu['generation']} (compute {cap:g}) is RTX 50 and Basalt is usually faster, but the published list does not include this model; "
                      "Strata has published the matching GGUF, so Strata is the default recommendation")
            alternative = "Basalt's published MoE models are only Flash-Next and DeepSeek-V4.1-Flash; switch only after confirming a compatible .basalt yourself"
        else:
            reason = (f"This card {gpu['generation']} (compute {cap:g}) is RTX 50 and Basalt is usually faster, but this MoE is not in the published list; "
                      "Strata has broader coverage, so Strata is the default recommendation")
            alternative = "Basalt supports only RTX 50 (sm_120) and the model must be in the published list; switch only after confirming compatibility yourself"
            compat_check += " This model is also not in Strata's published list; defer to both engines' official repositories before running."
    elif cap is not None:
        reason = f"This card {gpu['generation']} (compute {cap:g}) is not RTX 50 sm_120; Basalt cannot run, so Strata is chosen for its coverage of RTX 20/30/40/50 and multiple platforms"
        alternative = "Basalt supports only NVIDIA RTX 50 (sm_120, compute 12.0); data-center Blackwell (compute 10.0) is also not applicable"
    else:
        reason = ("No NVIDIA GPU detected (possibly AMD/Intel/Apple or no discrete GPU); Strata supports NVIDIA RTX 20/30/40/50, "
                  "AMD, Intel Arc (Linux) and Apple Silicon; Basalt is limited to RTX 50")
        alternative = "Basalt is limited to NVIDIA RTX 50 (sm_120); it becomes an option only with an RTX 50 series card"
    if not in_strata and not rtx50:
        reason += "; this architecture is not in either engine's published model list; defer to the official repositories before running"
    return {
        "engine": "strata", "label": "Strata",
        "reason": reason,
        "compat_check": compat_check,
        "requirements": "From 12GB-class VRAM; RAM by tier (Qwen3.8-Flash-Next Q2_0 about 37.6GB, Coder about 32GB, 64GB+ recommended for full residency; weights counted as RAM+VRAM combined); three-level GPU→RAM→SSD expert cache; NVIDIA needs CUDA 13 / driver 580+",
        "format": "GGUF (standard quantization, ISTA-DASLab / Unsloth etc.)",
        "models": "Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, DeepSeek-V4.1-Flash",
        "alternative": alternative,
        "url": _STRATA_URL, "steps": _STRATA_STEPS,
        "advisory": _MOE_ENGINE_ADVISORY,
    }


def _dense_runtime(gpu: dict | None) -> dict:
    """Dense artifacts stay .ninfer; fusion-kvmem is the suggested long-context runner."""
    card = f"This card {gpu['generation']} (compute {gpu['compute_cap']:g})" if gpu else "No NVIDIA GPU detected"
    return {
        "engine": "fusion-kvmem", "label": "ninfer-fusion-kvmem",
        "reason": f"{card}. For Dense models, run the .ninfer produced on this page separately with ninfer-fusion-kvmem: it keeps the KV window on the GPU according to VRAM and the rest in RAM, suited to long context.",
        "requirements": "Windows x64 prebuilt packages are split by architecture: RTX 50 uses sm_120a, RTX 40 sm_89, RTX 30 sm_86; for other GPUs check the repository README first. Experimental branch; on the first run defer to the engine logs.",
        "format": ".ninfer (the same as this page's artifact; no separate GGUF/.basalt download needed)",
        "models": "The Dense NInfer geometries covered by the repository notes, including Qwen3.5 / Bonsai 27B ternary",
        "alternative": "'Start model API' on this page still uses the companion NInfer engine and does not depend on this repository",
        "url": _FUSION_URL, "steps": _FUSION_STEPS,
        "advisory": "Selection advice only, no integration. Download and run fusion-kvmem separately; the LICENSE-less ninfer-kvmem-actkv is not recommended.",
    }


def runtime_recommendation(config: dict, gpu: dict | None) -> dict | None:
    """Dense → fusion-kvmem; MoE on RTX 50 → Basalt when published, else Strata; other MoE → Strata."""
    kind = architecture_kind(config)
    if kind == "dense":
        return _dense_runtime(gpu)
    if kind == "moe":
        return _moe_engine_recommendation(config, gpu)
    return None


def _target_pool(config: dict, route: str | None,
                 source_quantization: dict | None) -> tuple[list, list]:
    """Verified NInfer recipes and HF native recipes for this config."""
    from .backends.hf_quantization import recipes_for_config, quant_method
    architectures = architecture_names(config)
    candidates = [
        r for r in RECIPES if InputKind.SAFETENSORS in r.input_kinds
        and any(a in r.architectures for a in architectures)
        and geometry_admits(r, hidden_size(config))
        and (route is None or route in r.extra_sources)
        and (r.recipe != "qwen3_5_gguf_standalone"
             or (source_quantization or {}).get("gguf_file_type") == 7)
        and (r.recipe != "qwen3_8_flash_next_gsq_q2" or source_quantization is None
             or source_quantization.get("native_gsq_q2"))
    ]
    hf = recipes_for_config(config) if route is None else []
    if quant_method(config) in ("awq", "gptq"):
        candidates = []
    return candidates, hf


def _target_entry(recipe, cap: float | None, provided: tuple[str, ...]) -> dict:
    return {
        "value": recipe.recipe,
        "label": recipe.label,
        "bits": recipe.nominal_bits,
        "bpw": recipe.typical_bpw,
        "formats": list(recipe.storage_formats),
        "available": True,
        "engine": recipe.engine,
        "required_sources": list(recipe.extra_sources),
        "missing_sources": [key for key in recipe.extra_sources if key not in provided],
        "gpu_note": _gpu_scheme_note(recipe, cap),
    }


def choices(config: dict, components: dict | None = None, *, route: str | None = None, provided: tuple[str,...] = (), source_quantization: dict | None = None) -> dict:
    native, hf = _target_pool(config, route, source_quantization)
    native_available = bool(native)
    gpu = detect_gpu()
    cap = gpu["compute_cap"] if gpu else None
    native_first = any(_gpu_scheme_order(recipe, cap)[0] == 0 for recipe in native + hf)
    candidates = sorted(native + hf, key=lambda r: _gpu_scheme_order(r, cap))
    explicit = (route == "quantized" and len(candidates) > 1) or (not native_available and bool(hf))
    optional = {}
    architectures = architecture_names(config)
    if any(arch in (*QWEN35_DENSE,*QWEN35_MOE,*FLASH_NEXT,*GEMMA4) for arch in architectures):
        optional["mtp"] = {"label":"External MTP component", "kind":"model"}
    if any(arch in (*QWEN35_DENSE,*QWEN35_MOE) for arch in architectures):
        optional["dflash2"] = {"label":"External DFlash2 checkpoint", "kind":"directory"}
    if any(arch in FLASH_NEXT for arch in architectures):
        optional["ngram"] = {"label":"n-gram table", "kind":"file"}
    recommendation = runtime_recommendation(config, gpu)
    return {"detected_components":[name for name in ("mtp","dflash2") if (components or {}).get(name,{}).get("detected")],"optional_sources":optional,
            "architecture_kind":architecture_kind(config), "gpu":gpu,
            "scheme_order":"native-format" if native_first else "bits-per-weight",
            "runtime_engine":recommendation,
            "moe_engine":recommendation if architecture_kind(config) == "moe" else None,
            "targets":[_target_entry(r, cap, provided) for r in candidates],
            "provided_sources":list(provided), "requires_explicit_target":explicit, "source_fields":SOURCE_FIELDS,
            "speculation":["auto","none"] + [key for key in ("mtp","dflash2") if (components or {}).get(key,{}).get("detected")],
            "unsupported":[{"name":"Runtime format and bit width","reason":"AWQ/GPTQ uses a separate HF engine and keeps the native format; it is not equivalent to NInfer Q4. Only wired-in architectures and packed layouts are shown; arbitrary bit widths and vendor-reordered formats remain unsupported."}] + ([{"name":"New Qwen3.5 AWQ quantization","reason":"GPTQModel 7.10 showed clear quality degradation in local Qwen3.5 acceptance, so new AWQ quantization is disabled; choose GPTQ or NInfer. Existing AWQ is native import only."}] if config.get("model_type") in ("qwen3_5","qwen3_5_text") else []),
            "quality_note":"The quality gate only covers same-corpus PPL; it is not a guarantee of every task capability."}


def _unsupported(config: dict) -> bool:
    from .backends.hf_quantization import quant_method
    return quant_method(config) in ("awq","gptq")


def _remote_config(reference: str) -> dict:
    repository, revision, filename = reference_parts(reference)
    if not repository or filename:
        raise ValueError("The original source must be a complete model repository link")
    resolver = ModelResolver(Path.cwd(), lambda _: None)
    info = resolver.info(repository, revision)
    if not any(item["rfilename"] == "config.json" for item in info.get("siblings", [])):
        raise ValueError("The original repository lacks config.json")
    if not any(item["rfilename"] in ("tokenizer.json", "tokenizer.model") for item in info.get("siblings", [])):
        raise ValueError("The original repository lacks tokenizer.json / tokenizer.model; provide an original model repository that includes a tokenizer")
    return _remote_metadata(repository,info)


def _remote_metadata(repository: str, info: dict) -> dict:
    config = json.loads(read_url(f"https://huggingface.co/{repository}/resolve/{info['sha']}/config.json"))
    if not config.get("quantization_config"):
        for name in QUANTIZATION_CONFIG_FILES:
            if any(item["rfilename"] == name for item in info.get("siblings", [])):
                config["quantization_config"] = json.loads(read_url(f"https://huggingface.co/{repository}/resolve/{info['sha']}/{name}"))
                break
    return config


def precheck(model: str, original_model: str | None = None) -> dict:
    model = normalize_reference(model)
    original_model = normalize_reference(original_model) if original_model else None
    path = Path(model).expanduser()
    repository, revision, filename = reference_parts(model)
    remote = not path.exists() and repository
    if not path.exists() and not remote:
        return {"ready":False,"message":"This model cannot be found; reselect the model folder or GGUF file."}
    try:
        cuda_package()
        inspector = ModelInspector()
        if remote:
            info = ModelResolver(Path.cwd(), lambda _: None).info(repository, revision)
            ggufs = [item["rfilename"] for item in info.get("siblings", []) if item["rfilename"].endswith(".gguf") and not any(tag in item["rfilename"].lower() for tag in ("mmproj","mtp","dflash","vision"))
                     and (not re.search(r"-\d{5}-of-\d{5}\.gguf$",item["rfilename"]) or "-00001-of-" in item["rfilename"])]
            has_weights = any(item["rfilename"].endswith(".safetensors") and "/" not in item["rfilename"] for item in info.get("siblings", []))
            if filename or (ggufs and not has_weights):
                if not filename and len(ggufs) > 1:
                    return {"ready":False,"message":"Select the specific GGUF file in the repository; precision is never changed automatically.","action":"choose_file",
                            "input_candidates":[{"name":name,"reference":f"https://huggingface.co/{repository}/resolve/{info['sha']}/{name}"} for name in ggufs]}
                if filename and not any(item["rfilename"] == filename for item in info.get("siblings", [])):
                    raise ValueError("The selected repository file does not exist")
                base = original_model
                if not base:
                    card = info.get("cardData") or {}
                    origin = card.get("base_model")
                    base = origin if isinstance(origin,str) else origin[0] if isinstance(origin,list) and len(origin)==1 and isinstance(origin[0],str) else None
                if not base:
                    return {"ready":False,"message":"This GGUF repository does not declare a unique original source; the original model's config and tokenizer information must be supplied.","action":"provide_original"}
                try:
                    config = inspector.inspect(base,metadata_only=True).config if Path(base).exists() else _remote_config(base)
                except (ValueError, OSError) as exc:
                    return {"ready":False,"message":str(exc),"action":"provide_original"}
                route = "gguf"  # Actual ternary layout is confirmed from the downloaded header.
                result = choices(config, route=route, provided=(route,))
                if not result["targets"]: return {"ready":False,"message":"There is no direct GGUF import route for this model size; change the model to process to the original high-precision model directory or repository link and recheck.","action":"choose_original"}
                return {"ready":True,"remote":True,"can_edit_weights":False,"message":"The GGUF's actual storage format will be verified and imported at the source precision.",**result}
            config = _remote_metadata(repository,info)
            if _unsupported(config):
                result = choices(config)
                if not result["targets"]: raise ValueError("This AWQ/GPTQ architecture has no wired-in HF route")
                return {"ready":True,"remote":True,"can_edit_weights":False,"message":"The native AWQ/GPTQ bit width is retained and the HF engine is used; the actual packed tensors are verified after download.",**result}
            if _contains_nvfp4(config.get("quantization_config", {})):
                result = choices(config,route="quantized",provided=("quantized",))
                if not result["targets"]: raise ValueError("This quantization layout or architecture has no verified import route")
                return {"ready":True,"remote":True,"can_edit_weights":False,"message":"The actual NVFP4 tensors and source will be verified; a file name is never treated as proof of compatibility.",**result}
            check_config_support(config)
            return {"ready":True,"remote":True,"can_edit_weights":not bool(config.get("quantization_config")),"message":"The pinned revision's config has been verified; weights have not been downloaded yet.",**choices(config)}
        inspection = inspector.inspect(path)
        if original_model and inspection.input_kind not in (InputKind.GGUF, InputKind.NVFP4):
            return {"ready":False,"message":"The model to process is already an original HF model; clear 'Original model source' and recheck. That field is only for GGUF / NVFP4 import."}
        if inspection.input_kind == InputKind.NVFP4 and not inspection.quantization.get("nvfp4_layout_verified"):
            raise ValueError("The NVFP4 lacks verifiable packed weights or scale tensors; provide a complete checkpoint")
        if inspection.input_kind in (InputKind.GGUF, InputKind.NVFP4):
            directory = path if path.is_dir() else path.parent
            from .gguf_standalone import can_prepare, prepare_resources
            if can_prepare(path, inspection):
                from .bootstrap import cache_root
                generated = prepare_resources(path, cache_root())
                config = inspector.inspect(generated, metadata_only=True).config
                result = choices(config, route='gguf', provided=('gguf',), source_quantization=inspection.quantization)
                from .backends.registry import get_recipe, serving_context_ceiling
                ceiling = serving_context_ceiling(get_recipe("qwen3_5_gguf_standalone"), config)
                limit = f"The adapted engine's context ceiling is {ceiling}. " if ceiling else "This geometry has no known adapted-engine context ceiling."
                return {**result,"optional_sources":{},"ready":True,"can_edit_weights":False,
                        **({"recommended_max_context":ceiling} if ceiling else {}),
                        "message":"Config, tokenizer and chat template were generated from the GGUF embedded information; no original model directory is needed. Weights will be decoded and re-encoded as Q8_0; quality needs separate acceptance; " + limit}
            base = original_model
            if not base and inspection.input_kind == InputKind.GGUF and (directory/"config.json").is_file() and any((directory/name).is_file() for name in ("tokenizer.json","tokenizer.model")):
                base = str(directory)
            if not base: base = ModelResolver(directory,lambda _:None).origin(directory,inspection.quantization)
            if not base:
                return {"ready":False,"message":"The selected quantized model lacks a confirmable original source; the original model's config and tokenizer information must be supplied.","action":"provide_original"}
            try:
                config = inspector.inspect(base,metadata_only=inspection.input_kind == InputKind.GGUF).config if Path(base).exists() else _remote_config(base)
            except (ValueError, OSError) as exc:
                return {"ready":False,"message":str(exc),"action":"provide_original"}
            route = "quantized" if inspection.input_kind == InputKind.NVFP4 else "ternary" if inspection.quantization.get("ternary_pq2") else "gguf"
            result = choices(config,inspection.speculative_components,route=route,provided=(route,),source_quantization=inspection.quantization)
            if not result["targets"]: return {"ready":False,"message":"The config was recognized, but the current converter has no direct GGUF / NVFP4 import route for this model size. Change the model to process to the original high-precision model directory or repository link and recheck the available quantization plans.","action":"choose_original"}
            return {"ready":True,"can_edit_weights":False,"message":"Import route identified; the source bit width stays unchanged.",**result}
        if _unsupported(inspection.config):
            from .backends.hf_worker import check_checkpoint_layout
            check_checkpoint_layout(path if path.is_dir() else path.parent)
            result = choices(inspection.config)
            if not result["targets"]: raise ValueError("This AWQ/GPTQ architecture has no wired-in HF route")
            if not inspection.usable: raise ValueError("The AWQ/GPTQ checkpoint is incomplete")
            return {"ready":True,"can_edit_weights":False,"message":"Existing quantized weights are retained; validate after choosing the HF native retention route. No requantization.",**result}
        check_config_support(inspection.config)
        if not inspection.usable: raise ValueError("The model files are incomplete; select a model directory containing config, tokenizer and weights")
        return {"ready":True,"can_edit_weights":inspection.weight_precision in ("bf16","fp16","fp32"),"message":"Model recognized; choose a processing plan.",**choices(inspection.config,inspection.speculative_components)}
    except (ValueError,OSError,RuntimeError) as exc:
        return {"ready":False,"message":str(exc)}
