"""Header/CPU-only regressions: never run a model or repeat a GPU benchmark."""
import json
from pathlib import Path
import struct

import pytest

from tests.helpers import write_config,write_safetensors,write_gguf
from turboforge.inspector import ModelInspector
from turboforge.speculative import prepare_sources,preserve_hf,tensors,verify_ninfer
from turboforge.backends.registry import get_recipe
from turboforge.planner import ForgePlanner


def base(root):
    root.mkdir(exist_ok=True)
    write_config(root,architectures=["Qwen3_5ForCausalLM"],model_type="qwen3_5_text",hidden_size=1024)
    write_safetensors(root/"model.safetensors",{"model.norm.weight":("BF16",[32])})


def test_unindexed_mtp_is_detected_and_materialized_without_touching_input(tmp_path):
    base(tmp_path)
    write_safetensors(tmp_path/"head.safetensors",{"mtp.fc.weight":("BF16",[32,64])})
    index=tmp_path/"model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map":{"model.norm.weight":"model.safetensors"}}),encoding="utf-8")
    before=index.read_bytes()
    inspection=ModelInspector().inspect(tmp_path)
    assert inspection.speculative_components["mtp"]["detected"]
    assert len(inspection.files) == 2
    result=prepare_sources(tmp_path,{},tmp_path/"work")
    assert "mtp.fc.weight" in tensors(Path(result["mtp"]))
    assert before == index.read_bytes()


def test_wrapped_mtp_prefix_is_normalized_losslessly(tmp_path):
    base(tmp_path)
    write_safetensors(tmp_path/"head.safetensors",{"model.mtp.fc.weight":("F16",[32,64])})
    result=prepare_sources(tmp_path,{},tmp_path/"work")
    assert "mtp.fc.weight" in tensors(Path(result["mtp"]))
    assert "model.mtp.fc.weight" in tensors(tmp_path)


def test_bundled_dflash2_is_added_to_plan_even_when_execution_disabled(tmp_path):
    base(tmp_path)
    companion=tmp_path/"dflash2";companion.mkdir()
    write_config(companion,architectures=["DFlash2DraftModel"])
    write_safetensors(companion/"model.safetensors",{"fc.weight":("BF16",[32,64])})
    inspection=ModelInspector().inspect(tmp_path)
    assert inspection.speculative_components["dflash2"]["source_path"] == str(companion.resolve())
    assert "dflash2/model.safetensors" in {item["path"] for item in inspection.files}
    components=ForgePlanner._effective_components(get_recipe("qwen3_5_gguf_q4_0"),inspection,{})
    assert "dflash2" in components.split(",")
    assert prepare_sources(tmp_path,{},tmp_path/"work")["dflash2"] == str(companion.resolve())


def test_embedded_dflash2_requires_explicit_config(tmp_path):
    base(tmp_path)
    write_safetensors(tmp_path/"head.safetensors",{"dflash2.fc.weight":("BF16",[32,64])})
    with pytest.raises(ValueError,match="dflash2_config"):
        prepare_sources(tmp_path,{},tmp_path/"work")
    config=json.loads((tmp_path/"config.json").read_text(encoding="utf-8"))
    config["dflash2_config"]={"architectures":["DFlash2DraftModel"]}
    (tmp_path/"config.json").write_text(json.dumps(config),encoding="utf-8")
    result=prepare_sources(tmp_path,{},tmp_path/"work2")
    assert "fc.weight" in tensors(Path(result["dflash2"]))


def test_hf_export_restores_missing_auxiliary_head_and_updates_index(tmp_path):
    source=tmp_path/"source";target=tmp_path/"output";base(source);base(target)
    write_safetensors(source/"head.safetensors",{"mtp.fc.weight":("BF16",[32,64])})
    original=(source/"head.safetensors").read_bytes()
    result=preserve_hf(source,target)
    assert result["status"] == "retained"
    assert result["tensors"]["mtp.fc.weight"]["dtype"] == "BF16"
    index=json.loads((target/"model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"]["mtp.fc.weight"] == "speculative-preserved.safetensors"
    assert (source/"head.safetensors").read_bytes() == original
    assert result["runtime_acceleration"] == "not_supported_in_hf_runtime"


def test_hf_rejects_head_precision_change_instead_of_silently_accepting(tmp_path):
    source=tmp_path/"source";target=tmp_path/"output";base(source);base(target)
    write_safetensors(source/"head.safetensors",{"mtp.fc.weight":("BF16",[32,64])})
    write_safetensors(target/"head.safetensors",{"mtp.fc.weight":("F16",[32,64])})
    with pytest.raises(ValueError,match="改动"):
        preserve_hf(source,target)


def test_hf_existing_unindexed_head_becomes_discoverable(tmp_path):
    source=tmp_path/"source";target=tmp_path/"output";base(source);base(target)
    for directory in (source,target):
        write_safetensors(directory/"head.safetensors",{"mtp.fc.weight":("BF16",[32,64])})
    preserve_hf(source,target)
    index=json.loads((target/"model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"]["mtp.fc.weight"] == "head.safetensors"
    assert not (target/"speculative-preserved.safetensors").exists()


def artifact(path,component="mtp"):
    names=[component+"/layers/0/attention/query",component+"/layers/0/mlp/up","text/output_head"]
    if component == "mtp":
        names += ["mtp/"+name for name in ("input_projection","embedding_norm","hidden_norm","final_norm")]
        names += ["mtp/layers/0/"+name for name in ("input_norm","post_attention_norm","attention/gate","attention/key","attention/value","attention/query_norm","attention/key_norm","attention/output","mlp/gate","mlp/down")]
    if component == "dflash2":
        names += [component+"/candidate_selector/"+name for name in ("hidden_projection","predecessor_codebook","successor_codebook")]
        names += ["dflash2/"+name for name in ("feature_projection","context_norm","final_norm")]
        names += ["dflash2/layers/0/"+name for name in ("input_norm","post_attention_norm","attention/key","attention/value","attention/output","attention/query_norm","attention/key_norm","attention/context_key","attention/context_value","mlp/gate","mlp/down","attention_conv/base_kernel","attention_conv/kernel_projection","mlp_conv/base_kernel","mlp_conv/kernel_projection")]
    directory={"components":{"text":{},component:{"config":{"num_hidden_layers":1,"architectures":["Qwen3_5MTP"] if component == "mtp" else ["DFlash2DraftModel"]}}},"objects":[{"id":"weight/0","shape":[32,32],"format":"bf16","kind":"tensor","offset":0,"bytes":2048}],
               "files":[{"path":None,"payload_bytes":2048}],
               "bindings":{name:{"object":"weight/0"} for name in names},
               "uses":[{"parameter":"text/output_head","input":component+"/final_hidden"}]}
    save_artifact(path,directory)
    return directory


def save_artifact(path,directory):
    encoded=json.dumps(directory).encode()
    raw=struct.pack("<8sQ16s",b"NINFER\x00\x03",len(encoded),b"0"*16)+encoded
    path.write_bytes(raw+b"\0"*((-len(raw))%4096)+b"\0"*2048)


@pytest.mark.parametrize("component",["mtp","dflash2"])
def test_artifact_audit_requires_real_component_and_shared_head(tmp_path,component):
    path=tmp_path/"model.ninfer";directory=artifact(path,component)
    assert verify_ninfer(path,[component])[component]["shared_output_connected"]
    directory["uses"]=[];save_artifact(path,directory)
    with pytest.raises(ValueError,match="输出投影"):
        verify_ninfer(path,[component])


def test_artifact_metadata_only_head_is_rejected(tmp_path):
    path=tmp_path/"model.ninfer";directory=artifact(path)
    del directory["bindings"]["mtp/layers/0/attention/query"]
    save_artifact(path,directory)
    with pytest.raises(ValueError,match="attention"):
        verify_ninfer(path,["mtp"])


def test_mtp_in_last_gguf_shard_is_detected(tmp_path):
    first=tmp_path/"model-00001-of-00002.gguf";last=tmp_path/"model-00002-of-00002.gguf"
    write_gguf(first,tensors={"blk.0.attn_q.weight":[32,32]})
    write_gguf(last,tensors={"blk.64.nextn.eh_proj.weight":[32,64]})
    inspection=ModelInspector().inspect(first)
    assert inspection.speculative_components["mtp"]["detected"]
    assert inspection.tensor_count == 2


def test_hf_cannot_silently_disable_builtin_head(tmp_path):
    from turboforge.backends.hf_quantization import plan
    from turboforge.domain import ForgeConfig
    base(tmp_path)
    write_safetensors(tmp_path/"head.safetensors",{"mtp.fc.weight":("BF16",[32,64])})
    inspection=ModelInspector().inspect(tmp_path)
    config=ForgeConfig(str(tmp_path),str(tmp_path.parent/(tmp_path.name+"-output")),target="hf_gptq_4")
    result=plan(config,inspection)
    assert not result.executable
    assert any("不能自动降级" in item.message for item in result.diagnostics)
    config.runtime={"speculative":"none"}
    assert plan(config,inspection).executable


def test_custom_quantizer_keeps_draft_heads_q8_and_control_projections_direct(monkeypatch):
    from types import SimpleNamespace
    import torch
    pytest.importorskip("gguf")
    upstream=Path("D:/llm/ninfer-upstream")
    if not upstream.is_dir(): pytest.skip("optional pinned converter checkout unavailable")
    monkeypatch.syspath_prepend(str(upstream))
    from tools.convert.sources.logical import LogicalSource
    from turboforge.backends.gguf_quantization import _configure
    source=LogicalSource((32,32),"fixture",lambda begin,end:torch.zeros(end-begin))
    names=["text/layers/0/attention/query","mtp/input_projection","dflash2/layers/0/attention/query",
           "dflash2/layers/0/attention_conv/kernel_projection","dflash2/candidate_selector/hidden_projection"]
    parameters={name:SimpleNamespace(shape=(32,32),source=source,projection=True,direct_format="bf16") for name in names}
    class Recorder:
        def __init__(self): self.assigned={};self.shared={}
        def assign(self,name,**kwargs): self.assigned[name]=kwargs
        def share(self,name,target): self.shared[name]=target
    recipe=Recorder()
    _configure(SimpleNamespace(parameters=parameters,components={"dflash2":{"config":{"num_hidden_layers":1}}}),recipe,"MXFP4")
    assert recipe.assigned[names[0]]["format"] == "gguf_mxfp4"
    assert recipe.assigned[names[1]]["format"] == "gguf_q8_0"
    assert recipe.assigned[names[2]]["format"] == "gguf_q8_0"
    assert recipe.assigned[names[3]]["format"] == "bf16"
    assert recipe.assigned[names[4]]["source"] is source
    assert recipe.shared["dflash2/layers/0/attention/context_key"] == "dflash2/layers/0/attention/key"


def test_q8_bridge_gets_dflash_control_override():
    from turboforge.backends.compiler import NInferCompiler
    command=NInferCompiler().build_command(get_recipe("qwen3_5_gguf"),Path("model"),Path("output"),components="text,mtp,dflash2")
    assert command[command.index("--override")+1].endswith("gguf_quantization.py:preserve_draft_formats")


@pytest.mark.parametrize("type_name",["Q4_0","Q5_0","Q8_0","MXFP4"])
def test_ggml_codec_rejects_nonfinite_and_overflow_without_rejecting_signed_scales(monkeypatch,type_name):
    import numpy as np
    pytest.importorskip("gguf")
    upstream=Path("D:/llm/ninfer-upstream")
    if not upstream.is_dir(): pytest.skip("optional pinned converter checkout unavailable")
    monkeypatch.syspath_prepend(str(upstream))
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from turboforge.backends.gguf_quantization import checked_quantize
    qtype=getattr(GGMLQuantizationType,type_name)
    rows=np.linspace(-16,15,32,dtype=np.float32).reshape(1,32)
    assert np.isfinite(dequantize(checked_quantize(rows,qtype),qtype)).all()
    rows[0,0]=np.nan
    with pytest.raises(ValueError,match="输入"):
        checked_quantize(rows,qtype)
    if type_name != "MXFP4":
        with pytest.raises(ValueError,match="溢出"):
            checked_quantize(np.full((1,32),1e30,dtype=np.float32),qtype)


def test_explicit_imported_head_also_normalizes_unindexed_aliases(tmp_path):
    source=tmp_path/"source";head=tmp_path/"head";base(source);base(head)
    write_safetensors(head/"aux.safetensors",{"model.mtp.fc.weight":("BF16",[32,64])})
    result=prepare_sources(source,{"mtp":str(head)},tmp_path/"work")
    assert Path(result["mtp"]) != head
    assert "mtp.fc.weight" in tensors(Path(result["mtp"]))


def test_nvfp4_import_keeps_heads_from_the_actual_quantized_input(tmp_path):
    from types import SimpleNamespace
    from turboforge.model_sources import ModelResolver
    original=tmp_path/"original";quantized=tmp_path/"quantized";base(original);base(quantized)
    inspection=SimpleNamespace(speculative_components={"mtp":{"source":"base_checkpoint","detected":True}})
    result=ModelResolver.nvfp4_sources(original,quantized,inspection)
    assert result.sources["mtp"] == str(quantized)
    assert result.sources["quantized"] == str(quantized)


def test_gemma_assistant_keeps_its_own_output_head(tmp_path):
    path=tmp_path/"gemma.ninfer";directory=artifact(path)
    directory["components"]["mtp"]["config"]["architectures"]=["Gemma4AssistantForCausalLM"]
    directory["bindings"]["mtp/output_head"]={"object":"weight/0"}
    directory["uses"]=[{"parameter":"mtp/output_head","input":"mtp/final_hidden"}]
    save_artifact(path,directory)
    result=verify_ninfer(path,["mtp"])["mtp"]
    assert result["output_head_connected"] and not result["shared_output_connected"]


def test_hf_export_of_indexed_companion_mtp_is_loadable_without_ambiguous_duplicates(tmp_path):
    source=tmp_path/"source";target=tmp_path/"output";base(source);base(target)
    companion=source/"mtp";companion.mkdir()
    write_config(companion,architectures=["Qwen3_5ForCausalLM"],model_type="qwen3_5_text",hidden_size=1024)
    write_safetensors(companion/"model.safetensors",{"mtp.fc.weight":("BF16",[32,64])})
    (source/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"model.norm.weight":"model.safetensors","mtp.fc.weight":"mtp/model.safetensors"}}),encoding="utf-8")
    assert ModelInspector().inspect(source).speculative_components["mtp"]["detected"]
    preserve_hf(source,target)
    inspection=ModelInspector().inspect(target)
    assert inspection.usable and inspection.speculative_components["mtp"]["detected"]
    assert "mtp.fc.weight" in tensors(target)
    assert (target/"speculative-archive/mtp/config.json").is_file()
