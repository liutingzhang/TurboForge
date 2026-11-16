"""Isolated GPTQModel worker and loopback-only OpenAI-compatible text API.

Heavy imports happen only in the dedicated HF environment. The stdlib receipt
validator is also used by the Web process before starting a completed build.
"""
from __future__ import annotations

from ..assistant.config import child_environment
import argparse
import array
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

from ..domain import ForgeConfig
from ..local_endpoint import default_serving_port
from ..manifest import BuildManifest, file_sha256, read_json_atomic, write_json_atomic
from .hf_quantization import GPTQMODEL_VERSION, HF_DEFAULT_BACKEND, check_packed, environment_python, ensure_environment, plan, quant_method


def _checkpoint_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "turboforge-checkpoint.json")


def check_checkpoint_layout(directory: Path) -> dict:
    from ..adapters.safetensors import read_hf_config
    config = read_hf_config(directory)
    check_packed(config)
    from ..safetensors_io import read_tensor_summaries
    tensors = {}
    for path in directory.glob("*.safetensors"):
        summaries, _ = read_tensor_summaries(path)
        for tensor in summaries:
            if tensor.name in tensors:
                raise ValueError("AWQ/GPTQ 重复张量：" + tensor.name)
            tensors[tensor.name] = tensor
    from ..speculative import tensors as tensor_locations
    locations = tensor_locations(directory)
    packed = [name[:-len(".qweight")] for name in tensors if name.endswith(".qweight")]
    if not packed:
        raise ValueError("保存结果没有 qweight，不能认定为原生 AWQ/GPTQ 检查点")
    for prefix in packed:
        if tensors[prefix+".qweight"].dtype != "I32":
            raise ValueError("当前仅支持 INT32 打包：" + prefix)
        for suffix in (".scales", ".qzeros"):
            if prefix+suffix not in tensors:
                raise ValueError("量化权重缺少配套张量：" + prefix+suffix)
        weight, scale, zero = (tensors[prefix+suffix] for suffix in (".qweight",".scales",".qzeros"))
        quant = config["quantization_config"]
        bits = quant.get("bits",quant.get("w_bit"))
        group = quant.get("group_size",quant.get("q_group_size",128))
        if len(weight.shape) != 2 or len(scale.shape) != 2 or len(zero.shape) != 2 or zero.dtype != "I32" or scale.dtype not in ("F16","BF16","F32"):
            raise ValueError("量化张量类型或维度无效："+prefix)
        groups,n = scale.shape
        k = weight.shape[0] if quant_method(config) == "awq" else weight.shape[0]*32//bits
        expected_weight = [k,n//8] if quant_method(config) == "awq" else [k//32*bits,n]
        if k <= 0 or n <= 0 or k % 32 or n % 32 or groups != (1 if group == -1 else (k+group-1)//group) or weight.shape != expected_weight or zero.shape != [groups,n//32*bits]:
            raise ValueError("量化 packed/scale/zero 几何不匹配："+prefix)
        index = tensors.get(prefix+".g_idx")
        if quant.get("desc_act") or index is not None:
            if index is None or index.dtype != "I32" or index.shape != [k]:
                raise ValueError("GPTQ act-order 缺少有效 g_idx："+prefix)
        for tensor in (weight, scale, zero, *([index] if index is not None else [])):
            size = 2 if tensor.dtype in ("F16","BF16") else 4
            if locations[tensor.name]["bytes"] != math.prod(tensor.shape)*size:
                raise ValueError("量化张量声明与实际字节数不符："+tensor.name)
        _check_packed_values(locations[prefix+".scales"], scale.dtype, name=prefix+".scales")
        if index is not None:
            _check_packed_values(locations[prefix+".g_idx"], "I32", groups, prefix+".g_idx")
    return {"method":quant_method(config),"packed_layers":len(packed)}


def _check_packed_values(item: dict, dtype: str, groups: int = 0, name: str = "") -> None:
    """Bounded CPU checks; affine scales are positive, unlike signed GGML scales."""
    code = "i" if dtype == "I32" else "I" if dtype == "F32" else "H"
    exponent = {"F16":0x7c00,"BF16":0x7f80,"F32":0x7f800000}.get(dtype)
    sign = 0x80000000 if dtype == "F32" else 0x8000
    with item["file"].open("rb") as stream:
        stream.seek(item["offset"])
        remaining = item["bytes"]
        while remaining:
            data = stream.read(min(remaining,1024*1024))
            if not data:
                raise ValueError("量化张量数据被截断")
            values = array.array(code)
            values.frombytes(data)
            if sys.byteorder != "little": values.byteswap()
            if dtype == "I32":
                invalid = any(value < 0 or value >= groups for value in values)
            else:
                invalid = any(value == 0 or value & sign or (value & exponent) == exponent for value in values)
            if invalid:
                reason = "量化 g_idx 超出分组范围" if dtype == "I32" else "量化 scales 必须为有限正数，不能为零、负数、NaN 或 Inf"
                raise ValueError(reason+"："+name)
            remaining -= len(data)


def checkpoint_receipt(directory: Path) -> Path:
    layout = check_checkpoint_layout(directory)
    files = {p.relative_to(directory).as_posix():file_sha256(p) for p in _checkpoint_files(directory)}
    receipt = directory/"turboforge-checkpoint.json"
    write_json_atomic(receipt,{"engine":"gptqmodel","version":GPTQMODEL_VERSION,**layout,"files":files})
    return receipt


def verify_checkpoint(directory: Path) -> None:
    manifest_path = directory.parent/"manifest.json"
    if manifest_path.is_file() and read_json_atomic(manifest_path).get("status") != "validated":
        raise ValueError("此构建尚未通过全部验证或质量门禁，不能直接启动 HF 检查点")
    receipt = read_json_atomic(directory/"turboforge-checkpoint.json")
    files = receipt.get("files")
    if not isinstance(files,dict) or "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("AWQ/GPTQ 检查点清单不完整")
    for name, digest in files.items():
        path = (directory/name).resolve()
        if directory.resolve() not in path.parents or not path.is_file() or file_sha256(path) != digest:
            raise ValueError("AWQ/GPTQ 文件损坏或被更改：" + name)
    check_checkpoint_layout(directory)


def _environment() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build(config: ForgeConfig, inspection) -> BuildManifest:
    from .hf_quantization import validate_options
    config.validate()
    validate_options(config)
    for name in ("input_path","output_dir"):
        setattr(config,name,str(Path(getattr(config,name)).expanduser().resolve()))
    if config.quantization.get("calibration_file"):
        config.quantization["calibration_file"] = str(Path(config.quantization["calibration_file"]).expanduser().resolve())
    if config.benchmark.get("perplexity_text"):
        config.benchmark["perplexity_text"] = str(Path(config.benchmark["perplexity_text"]).expanduser().resolve())
    effective_plan = plan(config, inspection)
    if not effective_plan.executable:
        manifest = BuildManifest.create(config.output_dir,config,inspection,effective_plan.to_dict())
        manifest.transition("blocked","；".join(d.message for d in effective_plan.diagnostics if d.severity.value == "error"))
        return manifest
    if config.target == "hf_packed_import":
        check_checkpoint_layout(Path(config.input_path))
    python = environment_python()
    # Keep all HF dependencies outside the NInfer worker and the Web process.
    if Path(sys.executable).resolve() != python.resolve():
        python = ensure_environment()
        with tempfile.TemporaryDirectory(prefix="turboforge-hf-") as temporary:
            request = Path(temporary)/"config.json"
            write_json_atomic(request,asdict(config))
            result = subprocess.run([str(python),"-m",__name__,"build","--config",str(request)],env=child_environment(_environment()))
        manifest = BuildManifest(Path(config.output_dir))
        if not manifest.path.is_file():
            raise RuntimeError(f"AWQ/GPTQ 进程退出 {result.returncode}，没有生成构建记录")
        manifest.data = read_json_atomic(manifest.path)
        if result.returncode and manifest.data["status"] not in ("failed","blocked","validation-failed","quality-rejected"):
            manifest.transition("failed",f"AWQ/GPTQ 独立进程退出 {result.returncode}")
        return manifest
    output = Path(config.output_dir).expanduser().resolve()
    manifest = BuildManifest.create(output,config,inspection,effective_plan.to_dict())
    write_json_atomic(output/"plan.json",effective_plan.to_dict())
    if not effective_plan.executable:
        manifest.transition("blocked","；".join(d.message for d in effective_plan.diagnostics if d.severity.value == "error"))
        return manifest
    with (output/"logs/convert.log").open("w",encoding="utf-8",buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            _build(config,inspection,effective_plan,manifest)
        except Exception as exc:
            traceback.print_exc()
            manifest.transition("failed",f"{type(exc).__name__}: {exc}")
    return manifest


def calibration_examples(tokenizer, options: dict, output: Path) -> list[dict]:
    import torch
    count, length = options.get("samples",128), options.get("sequence_length",512)
    if options.get("calibration_file"):
        path = Path(options["calibration_file"]).expanduser().resolve()
        text = path.read_text(encoding="utf-8-sig")
        identity = {"file":str(path),"sha256":file_sha256(path)}
    else:
        from ..evaluation import WIKITEXT_REVISION
        from datasets import load_dataset
        revision = WIKITEXT_REVISION
        dataset = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train",revision=revision)
        text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
        identity = {"dataset":"Salesforce/wikitext","configuration":"wikitext-2-raw-v1","split":"train","revision":revision,
                    "text_sha256":hashlib.sha256(text.encode()).hexdigest()}
    # The corpus may exceed model context; only bounded sample windows are fed.
    ids = tokenizer(text,add_special_tokens=False,verbose=False)["input_ids"]
    identity["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if len(ids) < count*length:
        raise ValueError(f"校准语料只有 {len(ids)} token，{count} 个不重叠样本至少需要 {count*length}；请增加语料或减少样本")
    generator = random.Random(0)
    starts = generator.sample(range(0,len(ids)-length+1,length),count)
    write_json_atomic(output/"calibration.json",{**identity,"samples":count,"sequence_length":length,"seed":0,"starts":starts})
    return [{"input_ids":torch.tensor([ids[start:start+length]],dtype=torch.long),
             "attention_mask":torch.ones((1,length),dtype=torch.long)} for start in starts]


def _backend(name: str):
    from gptqmodel import BACKEND
    return BACKEND.TORCH if name == "torch" else BACKEND.AUTO


def _load(checkpoint: Path, backend: str):
    from gptqmodel import GPTQModel
    return GPTQModel.load(str(checkpoint),device="cuda:0",backend=_backend(backend),trust_remote_code=False,
                          profile="low_memory")


def _release() -> None:
    import torch
    # Call only after all strong references have been removed by the caller.
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def cpu_reference_kernels(model_type: str):
    """Transformers 5.20.2's FLA wrapper does not dispatch CPU inputs safely."""
    if model_type not in ("qwen3_5","qwen3_5_text"):
        yield
        return
    import importlib
    import inspect
    module=importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    originals={name:getattr(module,name) for name in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule")}
    try:
        for name,function in originals.items():
            setattr(module,name,inspect.unwrap(function))
        yield
    finally:
        for name,function in originals.items(): setattr(module,name,function)


def _ppl(model, ids: list[int], context: int, stride: int, device: str) -> tuple[float,int]:
    import torch
    if stride >= context or context < 2 or len(ids) < 2:
        raise ValueError("PPL 需要至少两个 token，且 stride 必须小于 context")
    loss, scored, previous_end = 0.0,0,0
    with torch.inference_mode():
        for start in range(0,len(ids),stride):
            end = min(start+context,len(ids))
            input_ids = torch.tensor([ids[start:end]],device=device)
            labels = input_ids.clone()
            target_length = end-previous_end
            if target_length < labels.shape[1]:
                labels[:,:-target_length] = -100
            logits = model(input_ids=input_ids,use_cache=False).logits
            count = int((labels[:,1:] != -100).sum().item())
            loss += torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),reduction="sum").item()
            scored += count
            previous_end = end
            if end == len(ids):
                break
    value = math.exp(loss/scored)
    if not math.isfinite(value):
        raise ValueError("PPL 不是有限数")
    return value,scored


def _build(config, inspection, effective_plan, manifest) -> None:
    if config.quantization.get("backend","auto") == "torch":
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer
    from ..jailbreak.processors import get_processor
    from ..reports import quality_gate, quality_report, performance_report, measurement_summary, write_reports
    from ..benchmark import BenchResult
    output = manifest.output_dir
    source = Path(config.input_path).resolve()
    if not source.is_dir():
        source = source.parent
    checkpoint = output/"model-hf"
    if not torch.cuda.is_available():
        raise RuntimeError("AWQ/GPTQ GPU 路线需要可用 CUDA PyTorch")
    torch.manual_seed(0)
    if config.jailbreak != "none":
        manifest.transition("jailbreaking","先编辑高精度权重，再校准量化")
        work = output/"intermediate"/config.jailbreak
        work.mkdir(parents=True)
        source = Path(get_processor(config.jailbreak).apply(inspection,work,**config.jailbreak_params))
        manifest.transition("jailbroken","高精度权重编辑完成")
    tokenizer = AutoTokenizer.from_pretrained(str(source),trust_remote_code=False,local_files_only=True)
    from ..evaluation import evaluation_text, AUTO_EVALUATION_TOKENS
    evaluation_corpus, evaluation_identity = evaluation_text(config.benchmark, output)
    conversion_start = time.monotonic()
    quantization_peak = None
    manifest.transition("converting","GPTQModel 原生打包；没有二次量化",converter={"engine":"gptqmodel","version":GPTQMODEL_VERSION,"recipe":config.target})
    if config.target == "hf_packed_import":
        checkpoint.mkdir()
        from ..domain import MODEL_RESOURCE_FILES
        for name in MODEL_RESOURCE_FILES | {"model.safetensors.index.json","quantize_config.json"}:
            path = source/name
            if path.is_file():
                shutil.copy2(path,checkpoint/name)
        for item in inspection.files:
            path = source/item["path"]
            target = checkpoint/item["path"]
            target.parent.mkdir(parents=True,exist_ok=True)
            try:
                if path.suffix == ".safetensors": os.link(path,target)
                else: shutil.copy2(path,target)
            except OSError:
                shutil.copy2(path,target)
    else:
        method, bits = config.target.split("_")[1:]
        calibration = calibration_examples(tokenizer,config.quantization,output)
        if evaluation_corpus is not None:
            if read_json_atomic(output/"calibration.json").get("text_sha256") == evaluation_identity["text_sha256"]:
                raise ValueError("校准与质量评测不能使用相同内容的语料")
        qconfig = QuantizeConfig(method=method,format="gemm" if method == "awq" else "gptq",
                                 bits=int(bits),group_size=config.quantization.get("group_size",128),
                                 sym=method == "gptq",desc_act=method == "gptq",static_groups=False,act_group_aware=False,device="cuda:0",
                                 offload_to_disk=True,offload_to_disk_path=str(output/"offload"),
                                 calibration_data_device="cpu",fallback=None,pack_dtype="int32")
        # GPTQModel 7.10 forwards local_files_only into the shell constructor;
        # it is not a supported keyword there. Source is already a local directory.
        # Legacy GPTQ stores FP16 scales. Calibrate in the same supported dtype
        # so BF16-only tiny ranges are not packed with scales that become zero.
        # The quality reference remains the untouched source; heads are raw-copied.
        model = GPTQModel.load(str(source),quantize_config=qconfig,trust_remote_code=False,
                              dtype=torch.float16)
        torch.cuda.reset_peak_memory_stats()
        model.quantize(calibration,batch_size=1,backend=_backend("torch"),calibration_data_min_length=10)
        model.save(str(checkpoint),max_shard_size="2GB")
        torch.cuda.synchronize()
        quantization_peak = torch.cuda.max_memory_allocated()
        # GPTQModel saves tokenizer then processor; saving the tokenizer again
        # can overwrite the processor's chat template (7.10 writer contract).
        del model,calibration
        _release()
        # Scratch is never a delivered model. A failed build keeps it available
        # for explicit cleanup; a successful save no longer needs another copy.
        scratch = output/"offload"
        if scratch.exists():
            if scratch.resolve().parent != output.resolve() or scratch.is_symlink():
                raise ValueError("卸载缓存路径越出本次输出目录，拒绝清理")
            shutil.rmtree(scratch)
        manifest.add_artifact("calibration",output/"calibration.json",digest=True)
    from ..domain import MODEL_RESOURCE_FILES
    for name in MODEL_RESOURCE_FILES:
        if (source/name).is_file() and not (checkpoint/name).exists():
            shutil.copy2(source/name,checkpoint/name)
    from ..speculative import preserve_hf
    retention = preserve_hf(source,checkpoint)
    manifest.data["speculative_retention"] = retention
    receipt = checkpoint_receipt(checkpoint)
    manifest.add_artifact("hf_checkpoint",receipt,digest=True)
    manifest.transition("converted","HF 原生检查点已保存",converter={"engine":"gptqmodel","version":GPTQMODEL_VERSION,
                         "recipe":config.target,"seconds":round(time.monotonic()-conversion_start,2),"checkpoint":"model-hf"})
    backend = config.quantization.get("backend",HF_DEFAULT_BACKEND)
    model = _load(checkpoint,backend)
    device = "cuda:0"
    model.model.eval()
    manifest.transition("validating","HF GPU 加载与生成")
    inputs = tokenizer("Reply with one word: OK",return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model.model(**inputs,use_cache=False).logits
        if not torch.isfinite(logits).all().item():
            raise ValueError("量化检查点 GPU logits 含 NaN/Inf，不能交付")
        del logits
        generated = model.generate(**inputs,max_new_tokens=16,do_sample=False,return_dict_in_generate=False)
    new_tokens = generated[0,inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens,skip_special_tokens=True)
    if not text.strip():
        raise ValueError("GPU 加载成功但未生成有效文本")
    (output/"logs/smoke.log").write_text(text,encoding="utf-8")
    performance = performance_report({},runtime=effective_plan.serving,environment={"engine":"gptqmodel","backend":backend,"gpu":torch.cuda.get_device_name(0)})
    performance["metrics"]["quantization_peak_vram_bytes"] = {"status":"measured" if quantization_peak is not None else "not_measured",
        "value":quantization_peak,"source":"torch.cuda.max_memory_allocated",
        "note":"本次校准量化到保存结束的 PyTorch 分配峰值；含加载后驻留权重，不代表整张显卡或其他进程"}
    if config.run_benchmark:
        manifest.transition("benchmarking","HF 实际生成速度测试")
        repetitions = config.benchmark.get("repetitions",3)
        inputs = tokenizer("Explain the advantages of running language models locally in detail.",return_tensors="pt").to(device)
        maximum = min(config.benchmark.get("n_gen",[64])[0],effective_plan.serving["max_context"]-inputs["input_ids"].shape[1])
        if maximum < 1:
            raise ValueError("服务上下文不足以完成生成测试")
        rates = []
        with torch.inference_mode():
            warmup = model.generate(**inputs,min_new_tokens=min(4,maximum),max_new_tokens=min(4,maximum),do_sample=False,return_dict_in_generate=False)
            del warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(repetitions):
                torch.cuda.synchronize();started = time.monotonic()
                tokens = model.generate(**inputs,min_new_tokens=maximum,max_new_tokens=maximum,do_sample=False,return_dict_in_generate=False)
                torch.cuda.synchronize()
                if tokens.shape[-1]-inputs["input_ids"].shape[-1] != maximum:
                    raise ValueError("速度测试未生成要求的固定 token 数，不能报告为完整测速")
                rates.append((tokens.shape[-1]-inputs["input_ids"].shape[-1])/(time.monotonic()-started))
                del tokens
        performance["metrics"]["decode_tok_s"] = {"status":"measured","value":{"end_to_end_generation":sum(rates)/len(rates)},"source":"GPTQModel.generate",
            "note":"含短提示预填充的端到端生成吞吐；不是纯解码内核速度，不能直接与 ninfer_bench 对比"}
        performance["generation_benchmark"] = {"tokens_per_run":maximum,"prompt_tokens":inputs["input_ids"].shape[-1],"repetitions":repetitions,"warmup_new_tokens":min(4,maximum)}
        performance["metrics"]["peak_vram_bytes"] = {"status":"measured","value":torch.cuda.max_memory_allocated(),"source":"torch.cuda.max_memory_allocated",
            "note":"仅 PyTorch 当前进程分配峰值，包含驻留权重；不代表整张显卡或量化阶段峰值"}
    quality = quality_report(jailbreak=config.jailbreak,runtime=effective_plan.serving)
    if evaluation_corpus is not None:
        corpus = evaluation_corpus
        ids = tokenizer(corpus,add_special_tokens=False)["input_ids"]
        if config.benchmark.get("auto_quality"):
            ids = ids[:AUTO_EVALUATION_TOKENS + 1]
        context = min(config.benchmark.get("perplexity_context",512),effective_plan.serving["max_context"])
        stride = config.benchmark.get("perplexity_stride",context//2)
        ppl,scored = _ppl(model.model,ids,context,stride,device)
        del model,generated,new_tokens,inputs
        _release()
        baseline = None
        if config.target != "hf_packed_import":
            current_source = {item["path"]:(Path(config.input_path)/item["path"]).stat() for item in manifest.data["source_files"]}
            if config.jailbreak == "none" and any(current_source[item["path"]].st_size != item["bytes"] or current_source[item["path"]].st_mtime_ns != item["modified_ns"] for item in manifest.data["source_files"]):
                raise ValueError("原始权重在量化后发生变更，不能作为同阶段质量对照")
            from transformers import AutoModelForCausalLM,AutoModelForImageTextToText
            loader = AutoModelForImageTextToText if inspection.architecture.endswith("ForConditionalGeneration") else AutoModelForCausalLM
            original = loader.from_pretrained(str(source),dtype=torch.float32,device_map={"":"cpu"},trust_remote_code=False,local_files_only=True).eval()
            with cpu_reference_kernels(inspection.model_type):
                baseline,baseline_count = _ppl(original,ids,context,stride,"cpu")
            if scored != baseline_count:
                raise ValueError("对照与量化模型 PPL 的评分 token 不一致")
            del original;gc.collect()
        def result(value):
            return BenchResult("GPTQModel.forward",[],0,0,"","",{"perplexity":value})
        quality = quality_report(perplexity=result(ppl),baseline_perplexity=result(baseline) if baseline is not None else None,
                                 jailbreak=config.jailbreak,runtime=effective_plan.serving)
        quality["metrics"]["quantization_loss"]["source"] = "GPTQModel.forward / Transformers CPU reference"
        quality["evaluation"] = {**evaluation_identity,"corpus_sha256":hashlib.sha256(corpus.encode()).hexdigest(),"tokens_scored":scored,
                                  "context":context,"stride":stride,"baseline":"同阶段高精度权重（CPU FP32 计算）" if baseline is not None else None}
    else:
        del model,generated,new_tokens,inputs
        _release()
    status = "validated"
    if "max_ppl_relative_increase" in config.benchmark:
        quality["gate"] = quality_gate(quality,config.benchmark["max_ppl_relative_increase"])
        if quality["gate"]["status"] != "passed":
            status = "quality-rejected"
    for name,path in write_reports(output/"reports",performance,quality).items():
        manifest.add_artifact("report_"+name,path)
    manifest.transition(status,"HF GPU 验证与所选评测完成",measurements=measurement_summary(performance,quality))
    if status == "validated":
        from .hf_quantization import server_command
        command = server_command(asdict(config),output,effective_plan.serving)
        # Relative model path allows relocating a completed build; the managed
        # Python environment remains local to this installation.
        rendered = subprocess.list2cmdline(command)
        rendered = rendered.replace(subprocess.list2cmdline([str(checkpoint)]),'"%~dp0..\\model-hf"')
        python_path = str(Path(__file__).resolve().parents[2]).replace("%", "%%")
        lines = [
            "@echo off",
            "chcp 65001 >nul",
            "setlocal EnableExtensions",
            'cd /d "%~dp0"',
            f'set "PYTHONPATH={python_path}"',
            rendered,
            "echo.",
            "echo HF server stopped. Press any key to close.",
            "pause >nul",
        ]
        launcher = output/"scripts/run.bat"
        launcher.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        manifest.add_artifact("launcher",launcher)


def create_text_app(loaded,tokenizer,name: str,context: int):
    import torch
    from flask import Flask,Response,jsonify,request,stream_with_context
    import threading
    lock = threading.Lock()
    app = Flask("TurboForge HF")

    @app.get("/v1/models")
    def models():
        return jsonify({"object":"list","data":[{"id":name,"object":"model","owned_by":"local"}]})

    @app.post("/v1/chat/completions")
    def completions():
        body = request.get_json(silent=True)
        if not isinstance(body,dict):
            return jsonify({"error":{"message":"请求必须为 JSON 对象","type":"invalid_request_error"}}),400
        if type(body.get("stream",False)) is not bool:
            return jsonify({"error":{"message":"stream 必须为布尔值","type":"invalid_request_error"}}),400
        if body.get("tools") or body.get("tool_choice") or body.get("response_format") or body.get("stop") or body.get("n",1) != 1:
            return jsonify({"error":{"message":"HF 本地服务当前支持单条文本对话，不支持工具、结构化输出或 stop 参数","type":"invalid_request_error"}}),400
        messages = body.get("messages")
        if not isinstance(messages,list) or not messages or any(not isinstance(m,dict) or m.get("role") not in ("system","user","assistant") or not isinstance(m.get("content"),str) for m in messages):
            return jsonify({"error":{"message":"messages 仅支持 system/user/assistant 文本消息","type":"invalid_request_error"}}),400
        maximum = body.get("max_tokens",128)
        temperature = body.get("temperature",0)
        if type(maximum) is not int or maximum < 1 or type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature < 0:
            return jsonify({"error":{"message":"max_tokens 或 temperature 无效","type":"invalid_request_error"}}),400
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        else:
            prompt = "\n".join(m["role"]+": "+m["content"] for m in messages)+"\nassistant:"
        inputs = tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
        count = inputs["input_ids"].shape[-1]
        if count+maximum > context:
            return jsonify({"error":{"message":f"输入与输出总长度超过 {context} token","type":"invalid_request_error"}}),400
        kwargs = {"max_new_tokens":maximum,"do_sample":temperature > 0,"return_dict_in_generate":False}
        if temperature > 0:
            kwargs["temperature"] = temperature
        if not lock.acquire(blocking=False):
            return jsonify({"error":{"message":"模型正在处理另一个请求","type":"rate_limit_error"}}),429
        released = False
        release_guard = threading.Lock()
        def release():
            nonlocal released
            with release_guard:
                if not released:
                    released=True;lock.release()
        completion_id = "chatcmpl-"+uuid.uuid4().hex
        if body.get("stream"):
            try:
                from transformers import TextIteratorStreamer,StoppingCriteria,StoppingCriteriaList
                streamer = TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True,timeout=60)
            except Exception:
                release()
                raise
            cancelled = threading.Event()
            class Cancelled(StoppingCriteria):
                def __call__(self,input_ids,scores,**extra):
                    return torch.full((input_ids.shape[0],),cancelled.is_set(),device=input_ids.device,dtype=torch.bool)
            outcome = {}
            def generate():
                try:
                    with torch.inference_mode():
                        tokens = loaded.generate(**inputs.to("cuda:0"),**kwargs,streamer=streamer,stopping_criteria=StoppingCriteriaList([Cancelled()]))
                    outcome["tokens"] = tokens.shape[-1]-count
                except Exception as exc:
                    outcome["error"] = str(exc)
                    app.logger.exception("HF streaming generation failed")
                    streamer.end()
                finally:
                    release()
            thread = threading.Thread(target=generate,daemon=True)
            def events():
                def chunk(delta,finish=None):
                    return {"id":completion_id,"object":"chat.completion.chunk","created":int(time.time()),"model":name,
                            "choices":[{"index":0,"delta":delta,"finish_reason":finish}]}
                try:
                    thread.start()
                    yield "data: "+json.dumps(chunk({"role":"assistant"}),ensure_ascii=False)+"\n\n"
                    for piece in streamer:
                        yield "data: "+json.dumps(chunk({"content":piece}),ensure_ascii=False)+"\n\n"
                    thread.join(timeout=60)
                    if thread.is_alive():
                        raise TimeoutError("HF generation did not finish")
                    if outcome.get("error"):
                        yield "data: "+json.dumps({"error":{"message":outcome["error"],"type":"server_error"}})+"\n\n"
                    else:
                        yield "data: "+json.dumps(chunk({},"length" if outcome.get("tokens") == maximum else "stop"))+"\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    app.logger.exception("HF streaming response failed")
                    yield "data: "+json.dumps({"error":{"message":str(exc) or "生成超时","type":"server_error"}})+"\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    cancelled.set()
                    if thread.ident is None:
                        release()
            response = Response(stream_with_context(events()),mimetype="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            def close():
                cancelled.set()
                if thread.ident is None:
                    release()
            response.call_on_close(close)
            return response
        try:
            with torch.inference_mode():
                output = loaded.generate(**inputs.to("cuda:0"),**kwargs)
            tokens = output[0,count:]
            text = tokenizer.decode(tokens,skip_special_tokens=True)
            return jsonify({"id":completion_id,"object":"chat.completion","created":int(time.time()),"model":name,
                            "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"length" if len(tokens)==maximum else "stop"}],
                            "usage":{"prompt_tokens":count,"completion_tokens":len(tokens),"total_tokens":count+len(tokens)}})
        except RuntimeError as exc:
            app.logger.exception("HF generation failed")
            return jsonify({"error":{"message":str(exc),"type":"server_error"}}),500
        finally:
            release()
    return app


def serve(model: Path,port: int,context: int,backend: str) -> None:
    if backend == "torch":
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from transformers import AutoTokenizer
    verify_checkpoint(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model),trust_remote_code=False,local_files_only=True)
    loaded = _load(model,backend)
    loaded.model.eval()
    app = create_text_app(loaded,tokenizer,model.parent.name,context)
    app.run(host="127.0.0.1",port=port,threaded=True,use_reloader=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command",required=True)
    build_args = commands.add_parser("build");build_args.add_argument("--config",type=Path,required=True)
    serve_args = commands.add_parser("serve");serve_args.add_argument("--model",type=Path,required=True)
    serve_args.add_argument("--port",type=int,default=default_serving_port());serve_args.add_argument("--context",type=int,default=2048)
    serve_args.add_argument("--backend",choices=("torch","auto"),default="auto")
    args = parser.parse_args()
    if args.command == "build":
        from ..inspector import ModelInspector
        config = ForgeConfig.from_dict(read_json_atomic(args.config))
        manifest = build(config,ModelInspector().inspect(config.input_path))
        return 0 if manifest.data["status"] == "validated" else 1
    serve(args.model.resolve(),args.port,args.context,args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
