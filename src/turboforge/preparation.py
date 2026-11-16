"""Prepare everything required by a model-only conversion request."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import socket
import uuid

from .architecture import architecture_names, hidden_size
from .bootstrap import cache_root, ensure_dependencies, ensure_tools, cuda_package, Emit
from .domain import ForgeConfig
from .local_endpoint import default_serving_port
from .model_sources import ModelResolver, repo_id, normalize_reference


def check_config_support(config: dict) -> None:
    """Check downloaded metadata before fetching multi-GiB model weights."""
    from .backends.registry import recipes_for, geometry_admits
    from .domain import InputKind
    from .bootstrap import local_tools
    hidden = hidden_size(config)
    recipes = [recipe for architecture in architecture_names(config)
               for recipe in recipes_for(InputKind.SAFETENSORS, architecture)
               if geometry_admits(recipe, hidden)]
    if not recipes:
        from .backends.hf_quantization import recipes_for_config
        if recipes_for_config(config):
            return
        raise ValueError("此模型架构或几何没有已接入转换路径；尚未下载大体积权重，请更换支持的模型")
    if all(recipe.recipe.startswith("qwen3_5_gguf") for recipe in recipes):
        tools = local_tools()
        if not tools or not (tools[0] / "tools/convert/qwen3_5_gguf.py").is_file():
            raise ValueError("此小模型需要适配引擎；请使用包含适配引擎的 Windows 整合包。已在下载模型权重前停止。")


def prepare(model: str, options: dict, emit: Emit) -> ForgeConfig:
    model = normalize_reference(model)
    root = cache_root()
    # Validate choices before installing or downloading anything.
    allowed = {"target", "jailbreak", "jailbreak_params", "sources", "runtime", "output_dir", "run_benchmark", "benchmark", "original_model", "quantization"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError("未知自动转换选项：" + ", ".join(sorted(unknown)))
    original = options.get("original_model")
    if original is not None and (not isinstance(original,str) or not original.strip()):
        raise ValueError("原始来源必须是非空路径或仓库链接")
    preliminary = ForgeConfig.from_dict({"input_path": model, "output_dir": str(root / "builds/pending"),
                                         **{key:value for key,value in options.items() if key != "original_model"}, "device": "cpu"})
    if preliminary.jailbreak == "heretic":
        raise ValueError("Heretic 的交互保存步骤不能无人值守；自动模式请选择 Abliteration，高级模式仍可使用 Heretic")
    if not Path(model).expanduser().exists() and not repo_id(model):
        raise ValueError("找不到模型；请选择本机模型，或填写 Hugging Face 仓库链接")
    for key in ("perplexity_text","perplexity_corpus","baseline_artifact","corpus"):
        value = preliminary.benchmark.get(key)
        if value and (not isinstance(value,str) or not Path(value).expanduser().is_file()):
            raise ValueError(f"评测文件 {key} 不存在或不是文件，请重新选择")
    from .backends.registry import get_recipe, geometry_admits
    selected_recipe = None
    if preliminary.target != "auto":
        try:
            selected_recipe = get_recipe(preliminary.target)
        except KeyError as exc:
            raise ValueError("指定量化方案不存在，请选择受支持方案") from exc
    for name, value in preliminary.sources.items():
        if not Path(value).expanduser().exists():
            raise ValueError(f"附加来源 {name} 不存在")
    def check_requested(config):
        if selected_recipe and selected_recipe.engine == "gptqmodel":
            from .backends.hf_quantization import recipes_for_config
            if preliminary.target not in [r.recipe for r in recipes_for_config(config)]:
                raise ValueError("所选 AWQ/GPTQ 路线与仓库模型不兼容，已在下载权重前停止")
            return
        check_config_support(config)
        if selected_recipe and (not any(arch in selected_recipe.architectures for arch in architecture_names(config))
                                or not geometry_admits(selected_recipe, hidden_size(config))):
            raise ValueError("指定量化方案与仓库模型不兼容，已在下载权重前停止")
    emit("先检查系统、显卡与驱动兼容性")
    cuda_package()
    emit("识别模型并自动获取必需的分词器与来源文件")
    hf_mode = bool(selected_recipe and selected_recipe.engine == "gptqmodel")
    if hf_mode:
        from .backends.hf_quantization import validate_options
        preliminary.device = "cuda"
        validate_options(preliminary)
    resolved = ModelResolver(root, emit, check_config=check_requested, hf_mode=hf_mode).resolve(model, options.get("original_model"))
    from .inspector import ModelInspector
    inspected = ModelInspector().inspect(resolved.model, metadata_only=bool(resolved.sources.get("gguf") or resolved.sources.get("ternary")))
    if hf_mode:
        from .backends.hf_quantization import plan
        if resolved.sources:
            raise ValueError("HF AWQ/GPTQ 路线仅接受原始高精度或明确的原生 packed 检查点，不能忽略 GGUF/NVFP4 附加权重")
        values = asdict(preliminary)
        values.update(input_path=str(resolved.model.resolve()), device="cuda")
        if not options.get("output_dir"):
            values["output_dir"] = str(root/"builds"/f"{resolved.model.name}-{uuid.uuid4().hex[:8]}")
        if "port" not in preliminary.runtime:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1",0));port = probe.getsockname()[1]
            values["runtime"] = {**preliminary.runtime,"port":port}
        config = ForgeConfig.from_dict(values)
        checked = plan(config,inspected)
        if not checked.executable:
            raise ValueError("；".join(d.message for d in checked.diagnostics if d.severity.value == "error"))
        if preliminary.target == "hf_packed_import":
            from .backends.hf_worker import check_checkpoint_layout
            check_checkpoint_layout(resolved.model)
        emit("已选择 HF 原生 AWQ/GPTQ 路线，将自动准备独立环境；不下载 NInfer 引擎。输出："+config.output_dir)
        return config
    from .planner import ForgePlanner
    candidates = [recipe for recipe in ForgePlanner().available_recipes(inspected) if recipe.engine == "ninfer"]
    for name, value in preliminary.sources.items():
        if name in resolved.sources and Path(value).expanduser().resolve() != Path(resolved.sources[name]).expanduser().resolve():
            raise ValueError("导入输入与附加来源冲突，请重新选择主输入")
    provided = {name:str(Path(value).expanduser().resolve()) for name,value in {**resolved.sources, **preliminary.sources}.items()}
    runnable = [recipe for recipe in candidates if all(name in provided for name in recipe.extra_sources)]
    for name, value in provided.items():
        if not Path(value).expanduser().exists():
            raise ValueError(f"附加来源 {name} 不存在，请选择本机文件或目录")
    if not runnable:
        raise ValueError("此输入没有可直接执行的转换路径。该架构可能只支持 GGUF 导入，请直接提供其受支持的模型格式；已在下载引擎前停止。")
    if preliminary.target == "auto" and "quantized" in resolved.sources and sum("quantized" in r.extra_sources for r in runnable) > 1:
        raise ValueError("此架构有多种 NVFP4 布局，请明确选择与检查点匹配的导入方案；不会猜测生产工具")
    if preliminary.target != "auto":
        selected = next((recipe for recipe in candidates if recipe.recipe == preliminary.target), None)
        if selected is None:
            raise ValueError("指定方案与当前模型不兼容，请重新选择可用方案；不会自动替换")
        missing = [name for name in selected.extra_sources if name not in provided]
        if missing:
            raise ValueError("指定方案缺少来源：" + "、".join(missing))
    if preliminary.target == "qwen3_8_flash_next_gsq_q2" and provided.get("gguf"):
        from .adapters.gguf import GGUFAdapter
        if not GGUFAdapter().inspect(Path(provided["gguf"])).quantization.get("native_gsq_q2"):
            raise ValueError("GSQ 原生 2-bit 只导入已有 Q2_0 专家权重，不能把普通 Q4 等文件在此路线改成 2-bit")
    if options.get("original_model") and preliminary.jailbreak != "none":
        raise ValueError("导入量化权重不能叠加越狱，请使用原始模型执行越狱后重新量化")
    small_model = bool(candidates) and all(recipe.recipe.startswith("qwen3_5_gguf") for recipe in candidates)
    emit("准备匹配的转换器和推理引擎")
    source, engine = ensure_tools(root, emit, small_model=small_model)
    emit("检查并准备所需依赖")
    # This runs in a dedicated worker interpreter; no loaded web-process DLLs
    # are replaced when pip installs or upgrades optional tensor libraries.
    ensure_dependencies(preliminary.jailbreak, emit)
    if provided.get("imatrix") and Path(provided["imatrix"]).suffix.lower() == ".gguf":
        from .bootstrap import digest, run
        import hashlib
        identity = hashlib.sha256((digest(Path(provided["imatrix"])) + digest(resolved.model/"config.json")).encode()).hexdigest()
        adapted = root / "imatrix" / (identity + ".safetensors")
        if not adapted.is_file():
            adapted.parent.mkdir(parents=True, exist_ok=True)
            emit("使用官方适配器将 imatrix GGUF 映射为当前模型的输入通道顺序")
            pending = adapted.with_suffix(".safetensors.part")
            run([os.sys.executable,"-m","tools.convert.imatrix","--gguf",provided["imatrix"],"--config",str(resolved.model/"config.json"),"--out",str(pending)],emit,cwd=source)
            from safetensors import safe_open
            with safe_open(str(pending),framework="pt",device="cpu") as vectors:
                if not list(vectors.keys()):
                    raise ValueError("imatrix 没有与当前模型匹配的校准向量")
            pending.replace(adapted)
        provided["imatrix"] = str(adapted)
    os.environ["TURBOFORGE_NINFER_SRC"] = str(source)
    os.environ["TURBOFORGE_NINFER_BIN"] = str(engine)
    import torch
    values = asdict(preliminary)
    values.update(input_path=str(resolved.model.resolve()), upstream_dir=str(source), ninfer_dir=str(engine),
                  device="cuda" if torch.cuda.is_available() else "cpu",
                  sources={name:str(Path(value).expanduser().resolve()) for name,value in provided.items()}, run_validation=True)
    if "port" not in preliminary.runtime:
        with socket.socket() as probe:
            try:
                port = default_serving_port()
                probe.bind(("127.0.0.1", port))
            except OSError:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        values["runtime"] = {**preliminary.runtime, "port": port}
    if not options.get("output_dir"):
        name = resolved.model.stem if resolved.model.is_file() else resolved.model.name
        values["output_dir"] = str(root / "builds" / f"{name}-{uuid.uuid4().hex[:8]}")
    config = ForgeConfig.from_dict(values)
    emit("自动准备完成，开始量化与 GPU 验证。输出：" + config.output_dir)
    return config
