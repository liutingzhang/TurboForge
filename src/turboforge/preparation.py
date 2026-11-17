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
        raise ValueError("This model architecture or geometry has no wired-in conversion path; large weights have not been downloaded yet. Switch to a supported model")
    if all(recipe.recipe.startswith("qwen3_5_gguf") for recipe in recipes):
        tools = local_tools()
        if not tools or not (tools[0] / "tools/convert/qwen3_5_gguf.py").is_file():
            raise ValueError("This small model needs the adapted engine; use the Windows bundle that includes it. Stopped before downloading model weights.")


def prepare(model: str, options: dict, emit: Emit) -> ForgeConfig:
    model = normalize_reference(model)
    root = cache_root()
    # Validate choices before installing or downloading anything.
    allowed = {"target", "jailbreak", "jailbreak_params", "sources", "runtime", "output_dir", "run_benchmark", "benchmark", "original_model", "quantization"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError("Unknown auto-conversion options: " + ", ".join(sorted(unknown)))
    original = options.get("original_model")
    if original is not None and (not isinstance(original,str) or not original.strip()):
        raise ValueError("The original source must be a non-empty path or repository link")
    preliminary = ForgeConfig.from_dict({"input_path": model, "output_dir": str(root / "builds/pending"),
                                         **{key:value for key,value in options.items() if key != "original_model"}, "device": "cpu"})
    if preliminary.jailbreak == "heretic":
        raise ValueError("Heretic's interactive save step cannot run unattended; choose Abliteration in auto mode. Heretic remains available in advanced mode")
    if not Path(model).expanduser().exists() and not repo_id(model):
        raise ValueError("Model not found; select a local model, or enter a Hugging Face repository link")
    for key in ("perplexity_text","perplexity_corpus","baseline_artifact","corpus"):
        value = preliminary.benchmark.get(key)
        if value and (not isinstance(value,str) or not Path(value).expanduser().is_file()):
            raise ValueError(f"Evaluation file {key} does not exist or is not a file; reselect it")
    from .backends.registry import get_recipe, geometry_admits
    selected_recipe = None
    if preliminary.target != "auto":
        try:
            selected_recipe = get_recipe(preliminary.target)
        except KeyError as exc:
            raise ValueError("The specified quantization plan does not exist; choose a supported plan") from exc
    for name, value in preliminary.sources.items():
        if not Path(value).expanduser().exists():
            raise ValueError(f"Extra source {name} does not exist")
    def check_requested(config):
        if selected_recipe and selected_recipe.engine == "gptqmodel":
            from .backends.hf_quantization import recipes_for_config
            if preliminary.target not in [r.recipe for r in recipes_for_config(config)]:
                raise ValueError("The selected AWQ/GPTQ route is incompatible with the repository model; stopped before downloading weights")
            return
        check_config_support(config)
        if selected_recipe and (not any(arch in selected_recipe.architectures for arch in architecture_names(config))
                                or not geometry_admits(selected_recipe, hidden_size(config))):
            raise ValueError("The specified quantization plan is incompatible with the repository model; stopped before downloading weights")
    emit("First check system, GPU and driver compatibility")
    cuda_package()
    emit("Identify the model and automatically fetch the required tokenizer and source files")
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
            raise ValueError("The HF AWQ/GPTQ route only accepts original high-precision or explicit native packed checkpoints; GGUF/NVFP4 extra weights cannot be ignored")
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
            raise ValueError("; ".join(d.message for d in checked.diagnostics if d.severity.value == "error"))
        if preliminary.target == "hf_packed_import":
            from .backends.hf_worker import check_checkpoint_layout
            check_checkpoint_layout(resolved.model)
        emit("Selected the HF native AWQ/GPTQ route; an isolated environment will be prepared automatically. The NInfer engine is not downloaded. Output: "+config.output_dir)
        return config
    from .planner import ForgePlanner
    candidates = [recipe for recipe in ForgePlanner().available_recipes(inspected) if recipe.engine == "ninfer"]
    for name, value in preliminary.sources.items():
        if name in resolved.sources and Path(value).expanduser().resolve() != Path(resolved.sources[name]).expanduser().resolve():
            raise ValueError("The import input conflicts with the extra sources; reselect the main input")
    provided = {name:str(Path(value).expanduser().resolve()) for name,value in {**resolved.sources, **preliminary.sources}.items()}
    runnable = [recipe for recipe in candidates if all(name in provided for name in recipe.extra_sources)]
    for name, value in provided.items():
        if not Path(value).expanduser().exists():
            raise ValueError(f"Extra source {name} does not exist; select a local file or directory")
    if not runnable:
        raise ValueError("This input has no directly executable conversion path. The architecture may only support GGUF import; provide its supported model format directly. Stopped before downloading the engine.")
    if preliminary.target == "auto" and "quantized" in resolved.sources and sum("quantized" in r.extra_sources for r in runnable) > 1:
        raise ValueError("This architecture has multiple NVFP4 layouts; explicitly choose the import plan matching the checkpoint. The producer tool is never guessed")
    if preliminary.target != "auto":
        selected = next((recipe for recipe in candidates if recipe.recipe == preliminary.target), None)
        if selected is None:
            raise ValueError("The specified plan is incompatible with the current model; reselect an available plan. No automatic substitution")
        missing = [name for name in selected.extra_sources if name not in provided]
        if missing:
            raise ValueError("The specified plan is missing sources: " + ", ".join(missing))
    if preliminary.target == "qwen3_8_flash_next_gsq_q2" and provided.get("gguf"):
        from .adapters.gguf import GGUFAdapter
        if not GGUFAdapter().inspect(Path(provided["gguf"])).quantization.get("native_gsq_q2"):
            raise ValueError("GSQ native 2-bit only imports existing Q2_0 expert weights; ordinary Q4 etc. files cannot be turned into 2-bit on this route")
    if options.get("original_model") and preliminary.jailbreak != "none":
        raise ValueError("Imported quantized weights cannot be combined with abliteration; run abliteration on the original model and requantize afterwards")
    small_model = bool(candidates) and all(recipe.recipe.startswith("qwen3_5_gguf") for recipe in candidates)
    emit("Prepare the matching converter and inference engine")
    source, engine = ensure_tools(root, emit, small_model=small_model)
    emit("Check and prepare the required dependencies")
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
            emit("Use the official adapter to map the imatrix GGUF to the current model's input-channel order")
            pending = adapted.with_suffix(".safetensors.part")
            run([os.sys.executable,"-m","tools.convert.imatrix","--gguf",provided["imatrix"],"--config",str(resolved.model/"config.json"),"--out",str(pending)],emit,cwd=source)
            from safetensors import safe_open
            with safe_open(str(pending),framework="pt",device="cpu") as vectors:
                if not list(vectors.keys()):
                    raise ValueError("The imatrix has no calibration vectors matching the current model")
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
    emit("Automatic preparation complete; starting quantization and GPU validation. Output: " + config.output_dir)
    return config
