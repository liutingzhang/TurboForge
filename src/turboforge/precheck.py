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
    "quantized": {"label":"NVFP4 量化检查点", "kind":"directory"},
    "gguf": {"label":"GGUF 权重", "kind":"file"},
    "ternary": {"label":"PQ2_0 三值 GGUF", "kind":"file"},
    "imatrix": {"label":"重要性矩阵（imatrix）", "kind":"file"},
    "engram": {"label":"Engram 表 GGUF", "kind":"file"},
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
_MOE_ENGINE_ADVISORY = ("选型建议，不集成引擎；TurboForge 产出 .ninfer 供配套 NInfer 引擎，"
                        "Strata 用 GGUF、Basalt 用 .basalt，需各自来源/转换，不能直接互喂。")
_BASALT_URL = "https://github.com/jesdga95/basalt"
_STRATA_URL = "https://github.com/Niko1221/Strata"
# fusion-kvmem publishes an Apache-2.0 license and Windows sm_120/89/86 builds.
# ninfer-kvmem-actkv has no LICENSE file, so it is not offered.
_FUSION_URL = "https://github.com/1314521gjy/ninfer-fusion-kvmem"
_AGENT_BAT = ("给助手的硬性顺序：先用两句独立问题询问用户「上下文长度（token 正整数）」和「同时请求数（并发，正整数）」，"
               "得到数字前不要写启动脚本，也不要替用户假设。回答后只生成一个 ASCII 文件名的 .bat，"
               "只用 cmd.exe：首行 @echo off，接着 chcp 65001、setlocal EnableExtensions、cd /d \"%~dp0\"；"
               "所有路径加引号；每个失败分支都 echo 原因后 pause，再 exit /b 1；引擎命令结束后也 pause，"
               "避免 Win10/Win11 双击闪退。不要用 PowerShell 语法、不要用未在对应条目核对过的参数。")
_BASALT_STEPS = (_AGENT_BAT + "帮助中心搜 J12。已核对：basalt serve --config <json>，上下文用命令行 --max-context；"
                 "发布包不含 CUDA 13 与 Python，需自备。并发没有已核对的配置字段，问到后若大于 1，"
                 "必须告诉用户本次按单请求启动，不能把并发写进未核对字段。")
_STRATA_STEPS = (_AGENT_BAT + "帮助中心搜 J13。已核对的只有 strata launch、--family、--model、--plan；"
                 "没有核对 --context 或并发旗标。问到两个数字后，若引擎 --help 没有对应旗标，"
                 "bat 只使用已核对旗标，并明确写出未传入的上下文和并发，不能发明参数。")
_FUSION_STEPS = (_AGENT_BAT + "帮助中心搜 J14，条目内含完整命令与排错。"
                 "概要：① Releases 选 engine-v0.11.0-kvmem-20261008，RTX 50 下 ninfer-serve-120a.exe、"
                 "40 系 -89、30 系 -86，并核对 SHA256；② 把本页产出的 .ninfer 和引擎放英文路径；"
                 "③ 按文档设 KV 环境变量后启动，12GB 卡先用文档里的 --kv-capacity；"
                 "④ 另开终端访问 /v1/models 并让它回答一道算术题。该仓库 Apache-2.0；"
                 "无许可证的 melis2023/ninfer-kvmem-actkv 不推荐。")
# Shown up front for an RTX 50 card whose MoE is NOT in Basalt's adapted list.
_BASALT_FASTER_UNADAPTED = ("Basalt 在 RTX 50（sm_120）上通常更快，但本模型不在它已发布的 MoE 列表"
                            "（仅 Qwen3.8-Flash-Next、DeepSeek-V4.1-Flash）。"
                            "请自行核对有没有兼容的 .basalt；没有就用下面的默认引擎。")


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
            "reason": f"本卡 {gpu['generation']}（compute {cap:g}）是消费级 Blackwell sm_120，且该 MoE 在 Basalt 已发布列表内，官方口径原生 NVFP4、在 RTX 50 上更快",
            "requirements": "NVIDIA RTX 50（sm_120）+ CUDA 13 + 驱动 R580+；Linux x64 或 Win10 1903+ x64；release 包不含 CUDA/Python，需自备；放不下的专家进系统内存/SSD",
            "format": ".basalt（v2，自包含：权重/分词器/MTP/视觉投影）",
            "models": "Qwen3.8-Flash-Next、DeepSeek-V4.1-Flash",
            "alternative": "Strata 也支持本卡与该模型，格式是 GGUF，可作跨平台备选",
            "url": _BASALT_URL, "steps": _BASALT_STEPS,
            "advisory": _MOE_ENGINE_ADVISORY,
        }
    compat_check = ""
    if rtx50:
        compat_check = _BASALT_FASTER_UNADAPTED
        if in_strata:
            reason = (f"本卡 {gpu['generation']}（compute {cap:g}）是 RTX 50，Basalt 通常更快，但已发布列表不含本模型；"
                      "Strata 已发布对应 GGUF，默认推荐 Strata")
            alternative = "Basalt 已发布的 MoE 只有 Flash-Next 与 DeepSeek-V4.1-Flash；自行确认有兼容 .basalt 后再换"
        else:
            reason = (f"本卡 {gpu['generation']}（compute {cap:g}）是 RTX 50，Basalt 通常更快，但该 MoE 不在已发布列表；"
                      "Strata 覆盖更广，默认推荐 Strata")
            alternative = "Basalt 只支持 RTX 50（sm_120）且模型须在已发布列表内；自行确认兼容后再换"
            compat_check += " 本模型也不在 Strata 已发布列表，运行前以两引擎官方仓库为准。"
    elif cap is not None:
        reason = f"本卡 {gpu['generation']}（compute {cap:g}）不是 RTX 50 的 sm_120；Basalt 跑不了，故选覆盖 RTX 20/30/40/50 与多平台的 Strata"
        alternative = "Basalt 只支持 NVIDIA RTX 50（sm_120，compute 12.0），数据中心 Blackwell（compute 10.0）也不适用"
    else:
        reason = ("未检测到 NVIDIA 显卡（可能为 AMD/Intel/Apple 或无独显）；Strata 支持 NVIDIA RTX 20/30/40/50、"
                  "AMD、Intel Arc（Linux）、Apple Silicon，Basalt 只限 RTX 50")
        alternative = "Basalt 只限 NVIDIA RTX 50（sm_120）；换用 RTX 50 系才可选"
    if not in_strata and not rtx50:
        reason += "；该架构不在两引擎已发布模型列表，运行前以官方仓库为准"
    return {
        "engine": "strata", "label": "Strata",
        "reason": reason,
        "compat_check": compat_check,
        "requirements": "12GB 级显存起；内存按档位（Qwen3.8-Flash-Next Q2_0 约 37.6GB、Coder 约 32GB、完整建议 64GB+，权重 RAM+VRAM 合计）；GPU→RAM→SSD 三级专家缓存；NVIDIA 需 CUDA 13/驱动 580+",
        "format": "GGUF（标准量化，ISTA-DASLab / Unsloth 等）",
        "models": "Qwen3.8-Flash-Next、Qwen3.6-35B-A3B、DeepSeek-V4.1-Flash",
        "alternative": alternative,
        "url": _STRATA_URL, "steps": _STRATA_STEPS,
        "advisory": _MOE_ENGINE_ADVISORY,
    }


def _dense_runtime(gpu: dict | None) -> dict:
    """Dense artifacts stay .ninfer; fusion-kvmem is the suggested long-context runner."""
    card = f"本卡 {gpu['generation']}（compute {gpu['compute_cap']:g}）" if gpu else "未检测到 NVIDIA 显卡"
    return {
        "engine": "fusion-kvmem", "label": "ninfer-fusion-kvmem",
        "reason": f"{card}。Dense 模型建议用 ninfer-fusion-kvmem 单独运行本页产出的 .ninfer：它按显存把 KV 窗口放在 GPU、其余放内存，适合长上下文。",
        "requirements": "Windows x64 预编译包按架构分：RTX 50 用 sm_120a、RTX 40 用 sm_89、RTX 30 用 sm_86；其他显卡先看仓库 README。实验分支，首次以引擎日志为准。",
        "format": ".ninfer（与本页产物相同，不需要另下 GGUF/.basalt）",
        "models": "仓库说明覆盖的 Dense NInfer 几何，含 Qwen3.5 / Bonsai 27B 三值",
        "alternative": "本页「启动模型服务」仍用配套 NInfer 引擎，不依赖这个仓库",
        "url": _FUSION_URL, "steps": _FUSION_STEPS,
        "advisory": "选型建议，不集成。fusion-kvmem 单独下载运行；无 LICENSE 的 ninfer-kvmem-actkv 不推荐。",
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
        optional["mtp"] = {"label":"外部 MTP 组件", "kind":"model"}
    if any(arch in (*QWEN35_DENSE,*QWEN35_MOE) for arch in architectures):
        optional["dflash2"] = {"label":"外部 DFlash2 检查点", "kind":"directory"}
    if any(arch in FLASH_NEXT for arch in architectures):
        optional["ngram"] = {"label":"n-gram 表", "kind":"file"}
    recommendation = runtime_recommendation(config, gpu)
    return {"detected_components":[name for name in ("mtp","dflash2") if (components or {}).get(name,{}).get("detected")],"optional_sources":optional,
            "architecture_kind":architecture_kind(config), "gpu":gpu,
            "scheme_order":"native-format" if native_first else "bits-per-weight",
            "runtime_engine":recommendation,
            "moe_engine":recommendation if architecture_kind(config) == "moe" else None,
            "targets":[_target_entry(r, cap, provided) for r in candidates],
            "provided_sources":list(provided), "requires_explicit_target":explicit, "source_fields":SOURCE_FIELDS,
            "speculation":["auto","none"] + [key for key in ("mtp","dflash2") if (components or {}).get(key,{}).get("detected")],
            "unsupported":[{"name":"运行格式与位宽","reason":"AWQ/GPTQ 使用独立 HF 引擎，保留原生格式；不等同于 NInfer Q4。仅显示已接入的架构与打包布局，任意位宽、厂商重排格式仍不支持。"}] + ([{"name":"Qwen3.5 AWQ 新量化","reason":"GPTQModel 7.10 在本机 Qwen3.5 验收中出现明显质量退化，已停用新量化；请选择 GPTQ 或 NInfer。已有 AWQ 仅作原生导入。"}] if config.get("model_type") in ("qwen3_5","qwen3_5_text") else []),
            "quality_note":"质量门禁仅覆盖同语料 PPL；不等同于所有任务能力保证。"}


def _unsupported(config: dict) -> bool:
    from .backends.hf_quantization import quant_method
    return quant_method(config) in ("awq","gptq")


def _remote_config(reference: str) -> dict:
    repository, revision, filename = reference_parts(reference)
    if not repository or filename:
        raise ValueError("原始来源必须是完整模型仓库链接")
    resolver = ModelResolver(Path.cwd(), lambda _: None)
    info = resolver.info(repository, revision)
    if not any(item["rfilename"] == "config.json" for item in info.get("siblings", [])):
        raise ValueError("原始仓库缺少 config.json")
    if not any(item["rfilename"] in ("tokenizer.json", "tokenizer.model") for item in info.get("siblings", [])):
        raise ValueError("原始仓库缺少 tokenizer.json / tokenizer.model，请提供含分词器的原始模型仓库")
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
        return {"ready":False,"message":"找不到这个模型，请重新选择模型文件夹或 GGUF 文件。"}
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
                    return {"ready":False,"message":"请选择仓库中的具体 GGUF 文件，不会擅自更换精度。","action":"choose_file",
                            "input_candidates":[{"name":name,"reference":f"https://huggingface.co/{repository}/resolve/{info['sha']}/{name}"} for name in ggufs]}
                if filename and not any(item["rfilename"] == filename for item in info.get("siblings", [])):
                    raise ValueError("所选仓库文件不存在")
                base = original_model
                if not base:
                    card = info.get("cardData") or {}
                    origin = card.get("base_model")
                    base = origin if isinstance(origin,str) else origin[0] if isinstance(origin,list) and len(origin)==1 and isinstance(origin[0],str) else None
                if not base:
                    return {"ready":False,"message":"此 GGUF 仓库未声明唯一原始来源，需要补充原始模型的配置与分词器信息。","action":"provide_original"}
                try:
                    config = inspector.inspect(base,metadata_only=True).config if Path(base).exists() else _remote_config(base)
                except (ValueError, OSError) as exc:
                    return {"ready":False,"message":str(exc),"action":"provide_original"}
                route = "gguf"  # Actual ternary layout is confirmed from the downloaded header.
                result = choices(config, route=route, provided=(route,))
                if not result["targets"]: return {"ready":False,"message":"当前模型尺寸没有 GGUF 直接导入路线，请将待处理模型改为原始高精度模型目录或仓库链接重新检查。","action":"choose_original"}
                return {"ready":True,"remote":True,"can_edit_weights":False,"message":"将核对 GGUF 实际存储格式，按来源精度导入。",**result}
            config = _remote_metadata(repository,info)
            if _unsupported(config):
                result = choices(config)
                if not result["targets"]: raise ValueError("此 AWQ/GPTQ 架构没有接入 HF 路线")
                return {"ready":True,"remote":True,"can_edit_weights":False,"message":"将保留原生 AWQ/GPTQ 位宽，使用 HF 引擎；下载后核对实际 packed 张量。",**result}
            if _contains_nvfp4(config.get("quantization_config", {})):
                result = choices(config,route="quantized",provided=("quantized",))
                if not result["targets"]: raise ValueError("此量化布局或架构没有已验证导入路线")
                return {"ready":True,"remote":True,"can_edit_weights":False,"message":"将核对 NVFP4 实际张量与来源；不会把文件名当作兼容证明。",**result}
            check_config_support(config)
            return {"ready":True,"remote":True,"can_edit_weights":not bool(config.get("quantization_config")),"message":"已核对指定版本的配置，尚未下载权重。",**choices(config)}
        inspection = inspector.inspect(path)
        if original_model and inspection.input_kind not in (InputKind.GGUF, InputKind.NVFP4):
            return {"ready":False,"message":"当前待处理模型已是原始 HF 模型，请清空“原始模型来源”后重新检查；该字段只用于 GGUF / NVFP4 导入。"}
        if inspection.input_kind == InputKind.NVFP4 and not inspection.quantization.get("nvfp4_layout_verified"):
            raise ValueError("NVFP4 缺少可核实的 packed 权重或缩放张量，请提供完整检查点")
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
                limit = f"当前适配引擎上下文上限为 {ceiling}。" if ceiling else "此几何没有已知的适配引擎上下文上限。"
                return {**result,"optional_sources":{},"ready":True,"can_edit_weights":False,
                        **({"recommended_max_context":ceiling} if ceiling else {}),
                        "message":"已从 GGUF 内嵌信息生成配置、分词器和聊天模板，无需原始模型目录。权重将解码后重编码为 Q8_0，质量需另外验收；" + limit}
            base = original_model
            if not base and inspection.input_kind == InputKind.GGUF and (directory/"config.json").is_file() and any((directory/name).is_file() for name in ("tokenizer.json","tokenizer.model")):
                base = str(directory)
            if not base: base = ModelResolver(directory,lambda _:None).origin(directory,inspection.quantization)
            if not base:
                return {"ready":False,"message":"已选量化模型缺少可确认的原始来源，需要补充原始模型的配置与分词器信息。","action":"provide_original"}
            try:
                config = inspector.inspect(base,metadata_only=inspection.input_kind == InputKind.GGUF).config if Path(base).exists() else _remote_config(base)
            except (ValueError, OSError) as exc:
                return {"ready":False,"message":str(exc),"action":"provide_original"}
            route = "quantized" if inspection.input_kind == InputKind.NVFP4 else "ternary" if inspection.quantization.get("ternary_pq2") else "gguf"
            result = choices(config,inspection.speculative_components,route=route,provided=(route,),source_quantization=inspection.quantization)
            if not result["targets"]: return {"ready":False,"message":"配置已识别，但当前转换器没有此模型尺寸的 GGUF / NVFP4 直接导入路线。请将待处理模型改为原始高精度模型目录或仓库链接，重新检查可用量化方案。","action":"choose_original"}
            return {"ready":True,"can_edit_weights":False,"message":"已识别导入路线，来源位宽保持不变。",**result}
        if _unsupported(inspection.config):
            from .backends.hf_worker import check_checkpoint_layout
            check_checkpoint_layout(path if path.is_dir() else path.parent)
            result = choices(inspection.config)
            if not result["targets"]: raise ValueError("此 AWQ/GPTQ 架构没有接入 HF 路线")
            if not inspection.usable: raise ValueError("AWQ/GPTQ 检查点不完整")
            return {"ready":True,"can_edit_weights":False,"message":"保留已有量化权重，选择 HF 原生保留路线后验证；不会重新量化。",**result}
        check_config_support(inspection.config)
        if not inspection.usable: raise ValueError("模型文件不完整，请选择包含配置、分词器与权重的模型目录")
        return {"ready":True,"can_edit_weights":inspection.weight_precision in ("bf16","fp16","fp32"),"message":"模型已识别，请选择处理方案。",**choices(inspection.config,inspection.speculative_components)}
    except (ValueError,OSError,RuntimeError) as exc:
        return {"ready":False,"message":str(exc)}
