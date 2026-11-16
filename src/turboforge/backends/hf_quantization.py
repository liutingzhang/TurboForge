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
AWQ_DISABLED_REASON = "Qwen3.5 AWQ 新量化未通过本机质量验收，已停用；请选择 GPTQ 或 NInfer。已有 AWQ 文件仅支持校验后的原生导入，不自动重做量化。"
HF_CORE_VERSIONS = {"transformers":"5.20.2","datasets":"5.2.0","safetensors":"0.8.0","torchao":"0.18.0","pypcre":"0.6.4"}
HF_ACCEL_VERSIONS = {"triton-windows":"3.8.0.post29","flash-linear-attention":"0.6.0","fla-core":"0.6.0","einops":"0.8.2"}
HF_ARCHITECTURES = (
    "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
)
HF_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3", "qwen3_5", "qwen3_5_text")
_MODEL_TYPE_BY_ARCHITECTURE = dict(zip(HF_ARCHITECTURES,("llama","mistral","qwen2","qwen3","qwen3_5_text","qwen3_5")))
HF_RECIPES = tuple(
    RecipeSpec(f"hf_{method}_{bits}", f"{method.upper()} {bits} bit · HF 原生格式",
               (InputKind.SAFETENSORS,), HF_ARCHITECTURES, (f"hf_{method}",), float(bits), None,
               notes=("GPTQModel 校准与原生打包；使用独立 HF 推理，不转换成 NInfer。",
                      "自动下载校准集，批量 1、CPU 激活缓存与磁盘卸载；显存和速度以实际报告为准。"),
               engine="gptqmodel")
    for method, widths in (("awq", (4,)), ("gptq", (2, 3, 4, 8))) for bits in widths
) + (RecipeSpec("hf_packed_import", "保留 AWQ/GPTQ 原生格式 · HF 引擎",
                (InputKind.SAFETENSORS,), HF_ARCHITECTURES, ("hf_packed",), None, None,
                notes=("复用已有 packed 权重，不反量化、不改变位宽；不支持任意厂商重排布局。",), engine="gptqmodel"),)


def is_hf(target: str) -> bool:
    return any(recipe.recipe == target for recipe in HF_RECIPES)


def quant_method(config: dict) -> str:
    quant = config.get("quantization_config") or {}
    if not isinstance(quant,dict):
        raise ValueError("quantization_config 必须是对象")
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
            raise ValueError("量化元数据字段冲突："+" / ".join(aliases))
    if method == "awq" and "version" in quant:
        for key in ("checkpoint_format","format"):
            if key in quant and str(quant[key]).lower() != str(quant["version"]).lower():
                raise ValueError("AWQ version 与打包布局声明冲突")
    if str(quant.get("pack_dtype","int32")).lower() not in ("int32","torch.int32"):
        raise ValueError("当前仅支持 INT32 打包")
    bits = quant.get("bits", quant.get("w_bit"))
    fmt = str(quant.get("checkpoint_format", quant.get("format", quant.get("version", "gemm" if method == "awq" else "gptq")))).lower()
    if method == "awq" and (bits != 4 or fmt != "gemm"):
        raise ValueError("AWQ 导入仅支持 4-bit GEMM 原生布局；GEMV/Marlin/其他重排布局不能冒充 GEMM")
    if method == "gptq" and (type(bits) is not int or bits not in (2, 3, 4, 8) or fmt not in ("gptq", "gptq_v2")):
        raise ValueError("GPTQ 导入仅支持 2/3/4/8-bit 的 GPTQ/GPTQ_V2 原生布局")
    if method not in ("awq", "gptq"):
        raise ValueError("不是受支持的 AWQ/GPTQ 原生检查点")
    group = quant.get("group_size",quant.get("q_group_size",128))
    if type(group) is not int or group not in (-1,32,64,128):
        raise ValueError("原生 AWQ/GPTQ 导入仅支持 -1/32/64/128 分组")
    if quant.get("dynamic") or quant.get("activation") or quant.get("adapter"):
        raise ValueError("当前导入不接受混合位宽、额外激活量化或适配器配置，请提供标准 AWQ/GPTQ 检查点")


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
        raise ValueError("AWQ/GPTQ 原生路线需要 CUDA；自动模式会准备 GPU 环境")
    values = config.quantization
    allowed = {"calibration_file", "samples", "sequence_length", "group_size", "backend"}
    if set(values) - allowed:
        raise ValueError("未知 AWQ/GPTQ 选项：" + ", ".join(sorted(set(values) - allowed)))
    for key, default, low, high in (("samples",128,1,4096),("sequence_length",512,32,4096)):
        value = values.get(key,default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} 必须为 {low}–{high} 的整数")
    if type(values.get("group_size",128)) is not int or values.get("group_size",128) not in (32,64,128):
        raise ValueError("AWQ/GPTQ 分组仅支持 32、64、128")
    if values.get("backend","auto") not in ("torch","auto"):
        raise ValueError("AWQ/GPTQ 推理后端仅支持 torch 或 auto")
    if values.get("calibration_file") and (not isinstance(values["calibration_file"],str) or not Path(values["calibration_file"]).expanduser().is_file()):
        raise ValueError("校准文本文件不存在，请选择 UTF-8 文本；留空则自动下载 WikiText-2")
    if values.get("calibration_file") and config.benchmark.get("perplexity_text"):
        calibration = Path(values["calibration_file"]).expanduser().read_text(encoding="utf-8-sig")
        evaluation_path = Path(config.benchmark["perplexity_text"]).expanduser()
        if not evaluation_path.is_file():
            raise ValueError("质量评测文本文件不存在")
        evaluation = evaluation_path.read_text(encoding="utf-8-sig")
        if calibration == evaluation:
            raise ValueError("校准与质量评测不能使用相同内容的语料")
    if config.runtime.get("speculative","auto") not in ("auto","none") or config.sources:
        raise ValueError("HF AWQ/GPTQ 路线暂不执行 MTP/DFlash2 或 NInfer 附加来源，请选择关闭或自动（此路线关闭推测）")
    if config.runtime.get("max_concurrency",1) != 1:
        raise ValueError("HF AWQ/GPTQ 本地服务目前支持单请求并发，请设为 1")
    if set(config.runtime) - {"speculative","max_context","max_concurrency","port"}:
        raise ValueError("HF 路线不接受 NInfer 专用缓存或解码旗标，请使用 HF 默认运行参数")
    if config.jailbreak == "heretic":
        raise ValueError("HF 自动量化请使用 none 或 Abliteration，Heretic 的交互保存不适用于此路线")
    if not config.run_validation:
        raise ValueError("HF 原生路线必须验证 GPU 可加载与生成，不能跳过验证")
    if config.benchmark.get("perplexity_text") and not config.run_benchmark:
        raise ValueError("HF PPL 评测需要开启 run_benchmark")
    if config.benchmark.get("repetitions",3) > 30 or len(config.benchmark.get("n_gen",[64])) != 1:
        raise ValueError("HF 速度测试最多重复 30 次，n_gen 只接受一个长度")
    unsupported = set(config.benchmark) - {"auto_quality","perplexity_text","perplexity_context","perplexity_stride","max_ppl_relative_increase","repetitions","n_gen"}
    if unsupported:
        raise ValueError("HF 评测使用同阶段原始权重自动作为对照，仅支持文本评测；不支持：" + ", ".join(sorted(unsupported)))


def plan(config: ForgeConfig, inspection) -> ForgePlan:
    diagnostics = list(inspection.diagnostics)
    try:
        validate_options(config)
        if any(inspection.speculative_components.get(name,{}).get("detected") for name in ("mtp","dflash2")) and config.runtime.get("speculative","auto") != "none":
            raise ValueError("此模型自带 MTP/DFlash2，但 HF 引擎尚未执行这些组件。要保留加速请选择 NInfer 路线；如只需原生 AWQ/GPTQ 与无损归档头，请明确关闭推测解码，不能自动降级")
        candidates = recipes_for_config(inspection.config)
        if config.target == "hf_awq_4" and inspection.config.get("model_type") in AWQ_DISABLED_MODEL_TYPES:
            raise ValueError(AWQ_DISABLED_REASON)
        if config.target not in [recipe.recipe for recipe in candidates]:
            raise ValueError("所选 AWQ/GPTQ 路线与此模型架构、精度或 packed 布局不兼容")
        if config.target != "hf_packed_import" and inspection.weight_precision not in ("bf16","fp16","fp32"):
            raise ValueError("AWQ/GPTQ 校准需要原始 BF16/FP16/FP32 权重")
        if config.target == "hf_packed_import" and config.jailbreak != "none":
            raise ValueError("已有 AWQ/GPTQ 权重不能直接越狱；请用原始高精度权重量化")
        if config.target == "hf_packed_import" and "max_ppl_relative_increase" in config.benchmark:
            raise ValueError("已有量化输入缺少同阶段原始权重，不能作为自身的质量门禁对照")
        output, source = Path(config.output_dir).resolve(), Path(inspection.path).resolve()
        if output == source or source in output.parents or output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("输出必须是原模型目录之外的新空目录")
        if not inspection.tokenizer_files:
            raise ValueError("AWQ/GPTQ 需要完整分词器文件")
        if not Path(inspection.path).is_dir():
            raise ValueError("AWQ/GPTQ 请选择完整模型目录，不能只提供一个 Safetensors 分片")
        if inspection.architecture == "Qwen3_5ForConditionalGeneration" and any(not (source/name).is_file() for name in ("preprocessor_config.json","video_preprocessor_config.json")):
            raise ValueError("Qwen3.5 HF 路线缺少图像/视频处理器配置；请在自动模式提供原始仓库链接以补齐，无需手工下载权重")
        if config.runtime.get("max_context",2048) < 32:
            raise ValueError("HF 本地服务上下文至少为 32 token")
        if config.benchmark.get("perplexity_text") or config.benchmark.get("auto_quality"):
            limit = max_context(inspection.config)
            context = min(config.benchmark.get("perplexity_context",512),config.runtime.get("max_context",2048),limit)
            stride = config.benchmark.get("perplexity_stride",context//2)
            if not 0 < stride < context:
                raise ValueError("PPL stride 必须小于实际生效的评测上下文")
            from ..system import memory_available
            available = memory_available()
            required = inspection.element_count * 4 + 1024**3
            if available is not None and required > available:
                raise ValueError(f"质量对照需要约 {required/1024**3:.1f} GiB 可用内存（含 FP32 原权重及余量），当前不足；请关闭其他程序或明确跳过质量对照")
        if config.quantization.get("sequence_length",512) > max_context(inspection.config):
            raise ValueError("校准样本长度超过模型声明的上下文上限")
    except ValueError as exc:
        diagnostics.append(Diagnostic("hf-configuration", str(exc), Severity.ERROR))
    diagnostics.append(Diagnostic("hf-native-runtime", "此路线输出 HF 原生检查点，使用独立 GPTQModel 引擎；不会执行 NInfer 的 MTP/DFlash2。新路线尚未做本机大模型实测。", Severity.WARNING))
    context = min(config.runtime.get("max_context",2048), max_context(inspection.config))
    serving = {"engine":"gptqmodel", "backend":config.quantization.get("backend","auto"), "max_context":context,
               "max_concurrency":1, "port":config.runtime.get("port") or default_serving_port(), "launcher_flags":[], "speculative":"none",
               "kv_dtype":"由 HF 模型 dtype 决定","context_ceiling":text_config(inspection.config).get("max_position_embeddings"),"spec":None,"draft_tokens":0,"embedding_q4":False}
    selected = next((r for r in HF_RECIPES if r.recipe == config.target),None)
    return ForgePlan(config, inspection,
                     [PlanStep("jailbreak",config.jailbreak,"skipped" if config.jailbreak == "none" else "planned"),
                      PlanStep("quantize+compile",config.target,"planned",{"engine":"gptqmodel", "nominal_bits":selected.nominal_bits if selected else None,
                               "label":selected.label if selected else config.target,"device":"cuda","components":"text","storage_formats":list(selected.storage_formats) if selected else []}),
                      PlanStep("speculative","none","skipped",{"reason":"HF 路线未封装推测解码"}),
                      PlanStep("validate","HF GPU 加载与生成","planned")],
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
            raise RuntimeError("此自定义 HF 环境使用旧 Python，未执行下载或升级。请选择新的 TURBOFORGE_HF_HOME，让脚本自动准备 Python 3.13.18 环境；原环境与模型保留。")
    probe_code = "from importlib.metadata import version; import torch, torchvision; from gptqmodel import GPTQModel, QuantizeConfig; assert version('gptqmodel') == '7.10.0'; assert torch.__version__ == '2.14.1+cu130'; assert version('torchvision') == '0.29.1+cu130'; assert torch.cuda.is_available(); assert torch.version.cuda"
    probe_code += "; assert all(version(name) == expected for name,expected in " + repr(HF_CORE_VERSIONS) + ".items())"
    probe_code += "; import os; assert os.name != 'nt' or torch.cuda.get_device_capability()[0] < 8 or all(version(name) == expected for name,expected in " + repr(HF_ACCEL_VERSIONS) + ".items())"
    probe = subprocess.run([str(python),"-c",probe_code],capture_output=True,text=True,encoding="utf-8",errors="replace", env=probe_env)
    if probe.returncode:
        package = cuda_package()
        if package != "13":
            raise RuntimeError("AWQ/GPTQ 自动 GPU 环境目前要求支持 CUDA 13 的 NVIDIA 驱动（580+）；NInfer 原有 CUDA 12 路线不受影响")
        version = "2.14.1"  # Verified cu130 Windows wheels for CPython 3.10–3.14.
        emit("独立安装 AWQ/GPTQ 环境与官方 CUDA 13 PyTorch，不修改已有转换环境")
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
                emit("自动准备 Triton 与线性注意力加速内核；首次运行会编译并缓存")
                run([str(python),"-m","pip","install",*[name+"=="+version for name,version in HF_ACCEL_VERSIONS.items()]],emit)
        run([str(python),"-m","pip","check"],emit)
        probe = subprocess.run([str(python),"-c",probe_code],capture_output=True,text=True,encoding="utf-8",errors="replace", env=probe_env)
        if probe.returncode:
            raise RuntimeError("AWQ/GPTQ 环境导入或 CUDA 检查失败：" + probe.stderr[-2000:])
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
        raise RuntimeError("HF 运行环境准备中断；已下载部分会保留供下次启动续传，请查看启动窗口的具体下载错误后重试") from exc
    return [str(python),"-m","turboforge.backends.hf_worker","serve","--model",str(checkpoint),
            "--port",str(profile["port"]),"--context",str(profile["max_context"]),
            "--backend",config.get("quantization",{}).get("backend","auto")]
