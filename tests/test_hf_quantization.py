"""CPU/header-only contract checks for the new native HF route; no model downloads."""
from dataclasses import asdict
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers import write_config, write_safetensors as write_zero_safetensors
from turboforge.backends.hf_quantization import (
    HF_RECIPES, check_packed, plan, recipes_for_config, validate_options,
)
from turboforge.backends.hf_worker import checkpoint_receipt, verify_checkpoint
from turboforge.backends.registry import get_recipe
from turboforge.domain import ForgeConfig, InputKind, ModelInspection
from turboforge.inspector import ModelInspector
from turboforge.precheck import choices


def write_safetensors(path, tensors):
    write_zero_safetensors(path,tensors)
    data = bytearray(path.read_bytes())
    length = struct.unpack("<Q",data[:8])[0]
    header = json.loads(data[8:8+length])
    for name, value in header.items():
        if name.endswith(".scales"):
            begin,end = value["data_offsets"]
            data[8+length+begin:8+length+end] = struct.pack("<e",1)*((end-begin)//2)
    path.write_bytes(data)


def packed_fixture(directory, dtype="F16", desc_act=False):
    write_config(directory,**model_config(quantization_config={"quant_method":"gptq","bits":4,"group_size":32,"desc_act":desc_act}))
    path=directory/"model.safetensors"
    write_zero_safetensors(path,{"layer.qweight":("I32",[8,64]),"layer.scales":(dtype,[2,64]),
                                "layer.qzeros":("I32",[2,8]),"layer.g_idx":("I32",[64])})
    mutate_tensor(path,"layer.scales",{"F16":struct.pack("<e",1),"BF16":struct.pack("<H",0x3f80),"F32":struct.pack("<f",1)}[dtype],repeat=True)
    return path


def mutate_tensor(path,name,payload,repeat=False):
    data=bytearray(path.read_bytes());length=struct.unpack("<Q",data[:8])[0]
    item=json.loads(data[8:8+length])[name];begin,end=item["data_offsets"]
    if repeat: payload=payload*((end-begin)//len(payload))
    data[8+length+begin:8+length+begin+len(payload)]=payload
    path.write_bytes(data)


@pytest.mark.parametrize("dtype,bad",[("F16",0),("F16",0x8000),("F16",0xbc00),("F16",0x7c00),("F16",0x7e00),
                                      ("BF16",0x7f80),("BF16",0x7fc0),("F32",0x7f800000),("F32",0x7fc00000)])
def test_nonfinite_or_nonpositive_affine_scale_rejected(tmp_path,dtype,bad):
    path=packed_fixture(tmp_path,dtype)
    mutate_tensor(path,"layer.scales",struct.pack("<I" if dtype == "F32" else "<H",bad))
    with pytest.raises(ValueError,match="scales"):
        checkpoint_receipt(tmp_path)


@pytest.mark.parametrize("desc_act",[False,True])
@pytest.mark.parametrize("value",[-1,2])
def test_out_of_range_group_index_rejected_even_without_act_order(tmp_path,desc_act,value):
    path=packed_fixture(tmp_path,desc_act=desc_act)
    mutate_tensor(path,"layer.g_idx",struct.pack("<i",value))
    with pytest.raises(ValueError,match="g_idx"):
        checkpoint_receipt(tmp_path)


def test_packed_payload_must_match_shape(tmp_path):
    path=packed_fixture(tmp_path)
    data=path.read_bytes();length=struct.unpack("<Q",data[:8])[0];header=json.loads(data[8:8+length])
    header["layer.scales"]["data_offsets"][1]-=2
    encoded=json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q",len(encoded))+encoded+data[8+length:])
    with pytest.raises(ValueError,match="actual byte count"):
        checkpoint_receipt(tmp_path)


@pytest.mark.parametrize("extra",[{"w_bit":8},{"method":"awq"},{"q_group_size":64},{"format":"marlin"},{"pack_dtype":"int8"}])
def test_conflicting_quantization_metadata_rejected(extra):
    with pytest.raises(ValueError):
        check_packed(model_config(quantization_config={"quant_method":"gptq","bits":4,"group_size":32,"checkpoint_format":"gptq",**extra}))


def test_calibration_cannot_inflate_sample_count_with_overlapping_windows(tmp_path):
    from turboforge.backends.hf_worker import calibration_examples
    path=tmp_path/"calibration.txt";path.write_text("fixture",encoding="utf-8")
    with pytest.raises(ValueError,match="non-overlapping samples"):
        calibration_examples(lambda *a,**kw:{"input_ids":list(range(640))},
                             {"calibration_file":str(path),"samples":128,"sequence_length":512},tmp_path)
    samples=calibration_examples(lambda *a,**kw:{"input_ids":list(range(1024))},
                                {"calibration_file":str(path),"samples":2,"sequence_length":512},tmp_path)
    assert set(samples[0]["input_ids"].flatten().tolist()).isdisjoint(samples[1]["input_ids"].flatten().tolist())
    assert json.loads((tmp_path/"calibration.json").read_text())["text_sha256"]


def test_equal_calibration_and_evaluation_content_rejected_before_install(tmp_path):
    first=tmp_path/"train.txt";second=tmp_path/"test.txt"
    first.write_text("same content",encoding="utf-8-sig");second.write_text("same content",encoding="utf-8")
    with pytest.raises(ValueError,match="identical content"):
        validate_options(forge(tmp_path,run_benchmark=True,quantization={"calibration_file":str(first)},benchmark={"perplexity_text":str(second)}))


def test_effective_ppl_context_checked_before_quantization(tmp_path):
    config=forge(tmp_path,run_benchmark=True,runtime={"max_context":64},benchmark={"perplexity_text":"test.txt","perplexity_stride":128})
    config.target="hf_gptq_4"
    inspection=ModelInspection(path=config.input_path,input_kind=InputKind.SAFETENSORS,
        config=model_config(),weight_precision="bf16",tokenizer_files=["tokenizer.json"])
    result=plan(config,inspection)
    assert not result.executable and any("stride" in d.message for d in result.diagnostics)


def model_config(**updates):
    return {"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5",
            "text_config":{"hidden_size":1024,"max_position_embeddings":32768},**updates}


def forge(tmp_path,**updates):
    source = tmp_path/"source";source.mkdir(exist_ok=True)
    return ForgeConfig(input_path=str(source),output_dir=str(tmp_path/"output"),target="hf_awq_4",**updates)


@pytest.mark.parametrize("method,widths",[("awq",{4}),("gptq",{2,3,4,8})])
def test_real_native_targets_are_separate_from_ninfer(method,widths):
    actual = [r for r in recipes_for_config(model_config(architectures=["Qwen3ForCausalLM"],model_type="qwen3",hidden_size=1024)) if r.recipe.startswith("hf_"+method)]
    assert {r.nominal_bits for r in actual} == widths
    assert all(r.engine == "gptqmodel" and get_recipe(r.recipe) == r for r in actual)


def test_precheck_advertises_native_backend_and_no_fake_arbitrary_bits():
    result = choices(model_config())
    assert {r["value"] for r in result["targets"] if r["engine"] == "gptqmodel"} == {r.recipe for r in HF_RECIPES[:-1] if r.recipe != "hf_awq_4"}
    assert not any(r["value"] == "hf_awq_2" for r in result["targets"])


@pytest.mark.parametrize("quant",[
    {"quant_method":"awq","bits":4,"version":"GEMV"},
    {"quant_method":"gptq","bits":4,"checkpoint_format":"marlin"},
    {"quant_method":"gptq","bits":4,"group_size":256},
    {"quant_method":"awq","bits":3,"version":"GEMM"},
    {"method":"gptq","bits":4,"dynamic":{"layer":{"bits":8}}},
])
def test_unsupported_native_layout_is_rejected(quant):
    with pytest.raises(ValueError):
        check_packed(model_config(quantization_config=quant))


def test_existing_awq_is_never_requantized_or_offered_ninfer():
    result = choices(model_config(quantization_config={"quant_method":"awq","bits":4,"version":"GEMM"}))
    assert [r["value"] for r in result["targets"]] == ["hf_packed_import"]
    assert result["requires_explicit_target"]


def test_failed_qwen35_awq_creation_is_not_offered_or_executable(tmp_path):
    result=choices(model_config())
    assert not any(item['value']=='hf_awq_4' for item in result['targets'])
    config=forge(tmp_path)
    inspection=ModelInspection(path=config.input_path,input_kind=InputKind.SAFETENSORS,
        config=model_config(),weight_precision='bf16',tokenizer_files=['tokenizer.json'])
    assert any('is disabled' in item.message for item in plan(config,inspection).diagnostics)


@pytest.mark.parametrize('status',['failed','quality-rejected','converted','validating'])
def test_direct_serve_cannot_bypass_recorded_validation_or_quality_failure(tmp_path,status):
    model=tmp_path/'model-hf';model.mkdir()
    packed_fixture(model);checkpoint_receipt(model)
    (tmp_path/'manifest.json').write_text(json.dumps({'status':status}),encoding='utf-8')
    with pytest.raises(ValueError,match='cannot be started directly'):
        verify_checkpoint(model)


@pytest.mark.parametrize("quant",[{"samples":True},{"group_size":True},{"samples":0},{"sequence_length":8192},{"backend":"made_up"},{"arbitrary":1}])
def test_bad_calibration_options_rejected_before_dependency_install(tmp_path,quant):
    with pytest.raises(ValueError):
        validate_options(forge(tmp_path,quantization=quant))


@pytest.mark.parametrize("updates",[
    {"runtime":{"speculative":"mtp"}}, {"sources":{"dflash2":"anything"}},
    {"run_validation":False}, {"runtime":{"max_concurrency":2}},
    {"benchmark":{"baseline_artifact":"ninfer-file"}},
])
def test_hf_does_not_silently_ignore_incompatible_features(tmp_path,updates):
    with pytest.raises(ValueError):
        validate_options(forge(tmp_path,**updates))


def test_hf_quality_gate_uses_automatic_reference(tmp_path):
    corpus=tmp_path/"evaluation.txt";corpus.write_text("evaluation",encoding="utf-8")
    config=forge(tmp_path,run_benchmark=True,benchmark={"perplexity_text":str(corpus),"max_ppl_relative_increase":.1})
    config.validate()
    validate_options(config)
    with pytest.raises(ValueError,match="baseline artifact"):
        ForgeConfig.from_dict({**asdict(config),"target":"qwen3_5_gguf"})


def test_hf_plan_does_not_depend_on_ninfer_and_declares_spec_disabled(tmp_path):
    config=forge(tmp_path)
    config.target="hf_gptq_4"
    inspection=ModelInspection(path=config.input_path,input_kind=InputKind.SAFETENSORS,
                               config=model_config(),weight_precision="bf16",tokenizer_files=["tokenizer.json"])
    result=plan(config,inspection)
    assert result.executable
    assert result.serving["engine"] == "gptqmodel" and result.serving["speculative"] == "none"
    assert not any("ninfer" in s.implementation for s in result.steps)


def test_import_cannot_use_itself_as_high_precision_reference(tmp_path):
    config=forge(tmp_path,run_benchmark=True,benchmark={"perplexity_text":"evaluation.txt","max_ppl_relative_increase":.1})
    config.target="hf_packed_import"
    inspection=ModelInspection(path=config.input_path,input_kind=InputKind.SAFETENSORS,
        config=model_config(quantization_config={"quant_method":"awq","bits":4}),tokenizer_files=["tokenizer.json"])
    result=plan(config,inspection)
    assert not result.executable
    assert any("its own quality-gate baseline" in d.message for d in result.diagnostics)


@pytest.mark.parametrize("method,bits",[("awq",4),("gptq",2),("gptq",3),("gptq",4),("gptq",8)])
def test_packed_receipt_checks_shapes_and_detects_changed_files(tmp_path,method,bits):
    config=model_config(quantization_config={"quant_method":method,"bits":bits,"group_size":32})
    (tmp_path/"config.json").write_text(json.dumps(config),encoding="utf-8")
    write_safetensors(tmp_path/"model.safetensors",{
        "layer.qweight":("I32",[64,8] if method == "awq" else [2*bits,64]),
        "layer.scales":("F16",[2,64]),"layer.qzeros":("I32",[2,2*bits]),
    })
    checkpoint_receipt(tmp_path);verify_checkpoint(tmp_path)
    (tmp_path/"config.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="corrupt or altered"):
        verify_checkpoint(tmp_path)


def test_receipt_rejects_fake_dtype_and_missing_act_order(tmp_path):
    config=model_config(quantization_config={"quant_method":"gptq","bits":4,"group_size":32,"desc_act":True})
    (tmp_path/"config.json").write_text(json.dumps(config),encoding="utf-8")
    write_safetensors(tmp_path/"model.safetensors",{
        "layer.qweight":("I32",[8,64]),"layer.scales":("F16",[2,64]),"layer.qzeros":("I32",[2,8])})
    with pytest.raises(ValueError,match="g_idx"):
        checkpoint_receipt(tmp_path)


def test_awq_prepare_never_downloads_ninfer_engine(tmp_path):
    from turboforge.preparation import prepare
    from turboforge.model_sources import ModelSource
    source=tmp_path/"model";source.mkdir()
    write_config(source,**model_config(architectures=["Qwen3ForCausalLM"],model_type="qwen3",hidden_size=1024))
    for name in ("preprocessor_config.json","video_preprocessor_config.json"):
        (source/name).write_text("{}",encoding="utf-8")
    write_safetensors(source/"model.safetensors",{"weight":("BF16",[32,32])})
    with patch("turboforge.preparation.cache_root",return_value=tmp_path), patch("turboforge.preparation.cuda_package"), \
         patch("turboforge.preparation.ModelResolver.resolve",return_value=ModelSource(source)), \
         patch("turboforge.preparation.ensure_tools") as tools,patch("turboforge.preparation.ensure_dependencies") as deps:
        result=prepare(str(source),{"target":"hf_awq_4"},lambda _:None)
    assert result.target == "hf_awq_4" and result.runtime["port"] > 0
    tools.assert_not_called();deps.assert_not_called()


def test_ppl_counts_each_target_once_on_overlapping_windows():
    import torch
    from turboforge.backends.hf_worker import _ppl
    class Uniform:
        def __call__(self,input_ids,use_cache):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape,16)))
    ppl,count=_ppl(Uniform(),[1]*33,context=12,stride=5,device="cpu")
    assert ppl == pytest.approx(16,rel=1e-6) and count == 32


@pytest.mark.parametrize("reference_vocab,expected",[(16,"validated"),(4,"quality-rejected"),(0,"failed")])
def test_hf_driver_quality_gate_and_launcher_without_gpu(tmp_path,monkeypatch,reference_vocab,expected):
    """Exercise orchestration with explicit package doubles, not real quantization."""
    import sys
    import torch
    from turboforge.backends import hf_worker as worker
    source=tmp_path/"source";source.mkdir()
    write_config(source,**model_config())
    for name in ("preprocessor_config.json","video_preprocessor_config.json"):
        (source/name).write_text("{}",encoding="utf-8")
    write_safetensors(source/"model.safetensors",{"weight":("BF16",[64,64])})
    corpus=tmp_path/"corpus.txt";corpus.write_text("fixed evaluation text",encoding="utf-8")
    config=ForgeConfig(input_path=str(source),output_dir=str(tmp_path/"output with spaces"),target="hf_gptq_4",
        run_benchmark=True,benchmark={"perplexity_text":str(corpus),"perplexity_context":32,"perplexity_stride":16,"max_ppl_relative_increase":.1})
    class Batch(dict):
        def to(self,device): return self
    class Uniform:
        def __init__(self,vocab): self.vocab=vocab
        def __call__(self,input_ids,use_cache):
            return SimpleNamespace(logits=torch.full((*input_ids.shape,self.vocab),float("nan") if reference_vocab == 0 else 0.))
        def eval(self): return self
    class Tokenizer:
        chat_template=None
        def __call__(self,text,**kwargs):
            if kwargs.get("return_tensors"): return Batch(input_ids=torch.tensor([[1,2]]))
            return {"input_ids":[1]*50}
        def decode(self,tokens,**kwargs): return "ok"
        def save_pretrained(self,directory): (Path(directory)/"tokenizer.json").write_text("{}",encoding="utf-8")
    class Model:
        model=Uniform(8)
        def generate(self,input_ids,**kwargs):
            return torch.cat([input_ids,torch.ones((1,kwargs.get('min_new_tokens',2)),dtype=torch.long)],dim=1)
        def quantize(self,calibration,**kwargs): assert kwargs["batch_size"] == 1
        def save(self,directory,**kwargs):
            directory=Path(directory);directory.mkdir()
            write_config(directory,**model_config(quantization_config={"quant_method":"gptq","bits":4,"group_size":32}))
            write_safetensors(directory/"model.safetensors",{
                "layer.qweight":("I32",[8,64]),"layer.scales":("F16",[2,64]),"layer.qzeros":("I32",[2,8])})
    qconfigs=[]
    def quantize_config(**kwargs): qconfigs.append(kwargs);return SimpleNamespace(**kwargs)
    def load(*args,**kwargs):
        assert "local_files_only" not in kwargs  # 7.10 leaks it into model.__init__.
        return Model()
    fake_gptq=SimpleNamespace(GPTQModel=SimpleNamespace(load=load),QuantizeConfig=quantize_config)
    fake_transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args,**kwargs:Uniform(reference_vocab)),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *args,**kwargs:Uniform(reference_vocab)))
    from importlib.machinery import ModuleSpec
    fake_transformers.__spec__=ModuleSpec("transformers",loader=None)
    monkeypatch.setitem(sys.modules,"gptqmodel",fake_gptq);monkeypatch.setitem(sys.modules,"transformers",fake_transformers)
    monkeypatch.setattr(worker,"environment_python",lambda:Path(sys.executable))
    monkeypatch.setattr(worker,"_load",lambda *args:Model());monkeypatch.setattr(worker,"_backend",lambda name:name)
    monkeypatch.setattr(worker,"_release",lambda:None)
    from contextlib import nullcontext
    monkeypatch.setattr(worker,"cpu_reference_kernels",lambda _:nullcontext())
    import itertools
    ticks=itertools.count()
    monkeypatch.setattr(worker,"time",SimpleNamespace(monotonic=lambda:next(ticks)/10))
    original_ppl=worker._ppl
    monkeypatch.setattr(worker,"_ppl",lambda model,ids,context,stride,device:original_ppl(model,ids,context,stride,"cpu"))
    def calibration(tokenizer,options,output):
        (output/"calibration.json").write_text("{}",encoding="utf-8");return []
    monkeypatch.setattr(worker,"calibration_examples",calibration)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda index:"unit-test double")
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:123)
    monkeypatch.setattr("turboforge.backends.hf_quantization.server_command",lambda config,output,profile:[sys.executable,"--model",str(output/"model-hf")])
    manifest=worker.build(config,ModelInspector().inspect(source))
    assert manifest.data["status"] == expected,manifest.data["history"]
    assert (Path(config.output_dir)/"scripts/run.bat").is_file() == (expected == "validated")
    assert qconfigs[0]["fallback"] is None and qconfigs[0]["offload_to_disk"]
    assert qconfigs[0]["method"] == "gptq" and qconfigs[0]["format"] == "gptq"
    if expected == "validated":
        assert '"%~dp0..\\model-hf"' in (Path(config.output_dir)/"scripts/run.bat").read_text(encoding="utf-8")


def test_bad_hf_model_does_not_install_dependencies(tmp_path,monkeypatch):
    from turboforge.backends import hf_worker as worker
    config=forge(tmp_path)
    inspection=ModelInspection(path=config.input_path,input_kind=InputKind.SAFETENSORS,config={})
    with patch.object(worker,"ensure_environment") as environment:
        result=worker.build(config,inspection)
    assert result.data["status"] == "blocked"
    environment.assert_not_called()


def test_legacy_gptq_sidecar_is_identified_without_mutating_config(tmp_path):
    from turboforge.adapters.safetensors import read_hf_config
    config=model_config()
    path=tmp_path/"config.json";path.write_text(json.dumps(config),encoding="utf-8")
    original=path.read_bytes()
    (tmp_path/"quantize_config.json").write_text(json.dumps({"bits":4,"quant_method":"gptq","group_size":32}),encoding="utf-8")
    resolved=read_hf_config(tmp_path)
    assert recipes_for_config(resolved)[0].recipe == "hf_packed_import"
    assert path.read_bytes() == original


@pytest.mark.parametrize("stream",[False,True])
def test_text_api_json_and_sse_without_gpu(monkeypatch,stream):
    import sys,queue
    import torch
    from turboforge.backends.hf_worker import create_text_app
    class Batch(dict):
        def to(self,device): return self
    class Tokenizer:
        chat_template=None
        def __call__(self,*args,**kwargs): return Batch(input_ids=torch.tensor([[1,2]]))
        def decode(self,*args,**kwargs): return "ok"
    class Streamer:
        def __init__(self,*args,**kwargs): self.queue=queue.Queue()
        def __iter__(self): return self
        def __next__(self):
            value=self.queue.get(timeout=2)
            if value is None: raise StopIteration
            return value
        def end(self): self.queue.put(None)
    class Model:
        def generate(self,input_ids,**kwargs):
            if kwargs.get("streamer"):
                kwargs["streamer"].queue.put("ok");kwargs["streamer"].end()
            return torch.tensor([[1,2,3]])
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(TextIteratorStreamer=Streamer,StoppingCriteria=object,StoppingCriteriaList=list))
    app=create_text_app(Model(),Tokenizer(),"fixture",64)
    client=app.test_client()
    assert client.get("/v1/models").json["data"][0]["id"] == "fixture"
    for _ in range(2):
        response=client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"hello"}],"max_tokens":4,"stream":stream})
        assert response.status_code == 200
        if stream:
            assert '"content": "ok"' in response.text and "data: [DONE]" in response.text
        else:
            assert response.json["choices"][0]["message"]["content"] == "ok"
        response.close()
    for bad in ([],{"messages":[]},{"messages":[{"role":"user","content":[]}]}):
        assert client.post("/v1/chat/completions",json=bad).status_code == 400
    assert client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"x"}],"max_tokens":100}).status_code == 400
