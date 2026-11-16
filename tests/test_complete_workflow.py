import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tests.helpers import write_config, write_gguf
from turboforge.precheck import choices, precheck
from turboforge.model_sources import ModelResolver, reference_parts
from turboforge.domain import InputKind, ForgeConfig
from turboforge.api.builds import BuildManager
from turboforge.manifest import write_json_atomic, read_json_atomic
from turboforge.runner import BuildRunner
from turboforge.planner import ForgePlanner
from turboforge.backends.compiler import UpstreamStatus


def test_every_required_source_has_a_guided_input():
    from turboforge.backends.registry import RECIPES
    from turboforge.precheck import SOURCE_FIELDS
    assert {key for recipe in RECIPES for key in recipe.extra_sources} <= SOURCE_FIELDS.keys()
    result=choices({"architectures":["Qwen3_5ForCausalLM"],"hidden_size":5120})
    for key in ("qwen3_8_27b_imatrix","bonsai2_27b_ternary","qwen3_8_27b_nvfp4_nvidia"):
        option=next(t for t in result["targets"] if t["value"]==key)
        assert option["available"] and option["missing_sources"]


def test_import_route_never_offers_raw_requantization():
    result=choices({"architectures":["Qwen3_5ForCausalLM"],"hidden_size":5120},route="gguf",provided=("gguf",))
    assert result["targets"] and all("gguf" in t["required_sources"] for t in result["targets"])
    assert all(not t["missing_sources"] for t in result["targets"])
    assert choices({"architectures":["Qwen3_5ForCausalLM"],"hidden_size":5120},route="quantized")["requires_explicit_target"]


def test_reference_parser_preserves_revision_and_filename():
    assert reference_parts("https://huggingface.co/Org/Model/resolve/dev%2Ftest/a%20b.gguf")==("Org/Model","dev/test","a b.gguf")


def test_missing_split_shard_is_refused_before_download(tmp_path):
    resolver=ModelResolver(tmp_path,lambda _:None)
    info={"sha":"a"*40,"siblings":[{"rfilename":"model-00001-of-00002.gguf","size":10}]}
    with patch.object(resolver,"info",return_value=info),patch("turboforge.model_sources.download") as download:
        with pytest.raises(ValueError,match="缺少分片"):resolver.snapshot("Org/Model")
    download.assert_not_called()


def test_multiple_gguf_precisions_are_not_silently_changed(tmp_path):
    resolver=ModelResolver(tmp_path,lambda _:None)
    info={"sha":"a"*40,"siblings":[{"rfilename":"Q4.gguf","size":10},{"rfilename":"Q8.gguf","size":20}]}
    with patch.object(resolver,"info",return_value=info),patch("turboforge.model_sources.download") as download:
        with pytest.raises(ValueError,match="具体 GGUF"):resolver.snapshot("Org/Model")
    download.assert_not_called()


def test_explicit_original_source_retains_gguf_file_path(tmp_path):
    gguf=tmp_path/"m.gguf";gguf.touch()
    base=tmp_path/"base";base.mkdir()
    inspection=SimpleNamespace(input_kind=InputKind.GGUF,config={},quantization={})
    with patch("turboforge.inspector.ModelInspector.inspect",return_value=inspection):
        source=ModelResolver(tmp_path,lambda _:None).resolve(str(gguf),str(base))
    assert source.model==base and source.sources=={"gguf":str(gguf)}


def test_unverified_nvfp4_does_not_download_original_weights(tmp_path):
    inspection=SimpleNamespace(input_kind=InputKind.NVFP4,config={},quantization={})
    resolver=ModelResolver(tmp_path,lambda _:None)
    with patch("turboforge.inspector.ModelInspector.inspect",return_value=inspection),patch.object(resolver,"snapshot") as fetch:
        with pytest.raises(ValueError,match="缩放张量"):resolver.resolve(str(tmp_path))
    fetch.assert_not_called()


def test_finished_job_survives_manager_restart(tmp_path,monkeypatch):
    monkeypatch.setenv("TURBOFORGE_HOME",str(tmp_path))
    def work(self,record,model,options):
        record.update(status="validated",output_dir=str(tmp_path/"out"))
    with patch.object(BuildManager,"_auto_work",work):
        manager=BuildManager();job=manager.start(None,model="example")
        state=tmp_path/"jobs"/job/"state.json"
        for _ in range(100):
            if read_json_atomic(state)["finished"]:break
            time.sleep(.02)
        else:raise AssertionError("worker did not persist completion")
    assert BuildManager().status(job)["status"]=="validated"


def test_interrupted_or_broken_job_cannot_be_started(tmp_path,monkeypatch):
    monkeypatch.setenv("TURBOFORGE_HOME",str(tmp_path));job="a"*32
    write_json_atomic(tmp_path/"jobs"/job/"state.json",{"build_id":job,"finished":False,"status":"converting","output_dir":"","error":None,"setup_log":[]})
    manager=BuildManager();assert manager.status(job)["status"]=="interrupted"
    with pytest.raises(RuntimeError):manager.launch(job)
    assert manager.status("../../other") is None


@pytest.mark.parametrize("raise_error",[False,True])
def test_quality_failure_never_writes_launcher(tmp_path,raise_error):
    from tests.helpers import write_safetensors
    model=tmp_path/"model";model.mkdir();write_config(model,architectures=["Qwen3_5ForCausalLM"],hidden_size=5120)
    write_safetensors(model/"model.safetensors",{"model.weight":("BF16",[8,8])})
    runner=BuildRunner(ForgePlanner(upstream=UpstreamStatus(tmp_path,True,"test",True,True,True,cuda_available=False)))
    config=ForgeConfig(str(model),str(tmp_path/"out"),device="cpu")
    def convert(config,plan,out,manifest,source):
        path=out/"m.ninfer";path.write_bytes(b"model");return path
    def report(artifact,out,config,manifest,plan):
        if raise_error:raise ValueError("bad quality provenance")
        manifest.transition("quality-rejected","PPL exceeds threshold")
    with patch.object(runner,"_run_converter",side_effect=convert),patch.object(runner,"_smoke_validate",return_value=(True,"ok")),patch.object(runner,"_write_reports",side_effect=report):
        result=runner.build(config)
    assert result.data["status"]==("failed" if raise_error else "quality-rejected")
    assert not (tmp_path/"out/scripts/run.bat").exists()


def test_remote_precheck_uses_exact_requested_revision(tmp_path):
    config={"architectures":["Qwen3_5ForCausalLM"],"hidden_size":5120}
    info={"sha":"b"*40,"siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"}]}
    with patch("turboforge.precheck.cuda_package"),patch("turboforge.precheck.ModelResolver.info",return_value=info) as meta,patch("turboforge.precheck.read_url",return_value=json.dumps(config).encode()) as fetch:
        result=precheck("https://huggingface.co/Org/Model/tree/dev")
    assert result["ready"]
    meta.assert_called_once_with("Org/Model","dev")
    assert "/"+"b"*40+"/config.json" in fetch.call_args.args[0]


def test_remote_multiple_gguf_precheck_returns_concrete_file_choices():
    info={"sha":"b"*40,"siblings":[{"rfilename":"Q4.gguf"},{"rfilename":"Q8.gguf"}]}
    with patch("turboforge.precheck.cuda_package"),patch("turboforge.precheck.ModelResolver.info",return_value=info):
        result=precheck("Org/Model")
    assert not result["ready"] and result["action"]=="choose_file"
    assert len(result["input_candidates"])==2
    assert all("/resolve/"+"b"*40 in item["reference"] for item in result["input_candidates"])


def test_fp8_metadata_is_not_misclassified_as_nvfp4():
    config={"architectures":["Qwen4ExpForCausalLM"],"hidden_size":2048,"quantization_config":{"quant_method":"fp8"}}
    info={"sha":"b"*40,"siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"}]}
    with patch("turboforge.precheck.cuda_package"),patch("turboforge.precheck.ModelResolver.info",return_value=info),patch("turboforge.precheck.read_url",return_value=json.dumps(config).encode()):
        result=precheck("Org/Model")
    assert result["ready"] and not result["can_edit_weights"]
    assert not result["provided_sources"]


def test_legacy_completed_worker_result_is_restored(tmp_path,monkeypatch):
    monkeypatch.setenv("TURBOFORGE_HOME",str(tmp_path));job="f"*32
    write_json_atomic(tmp_path/"jobs"/job/"result.json",{"status":"quality-rejected","output_dir":str(tmp_path/"out")})
    assert BuildManager().status(job)["status"]=="quality-rejected"


def test_broken_record_is_visible_and_does_not_return_other_reports(tmp_path,monkeypatch):
    monkeypatch.setenv("TURBOFORGE_HOME",str(tmp_path));job="f"*32
    write_json_atomic(tmp_path/"jobs"/job/"state.json",[])
    manager=BuildManager();assert manager.status(job)["status"]=="crashed"
    assert manager.report(job,"quality") is None


def test_flash_without_ngram_source_does_not_import_unavailable_table(tmp_path):
    from turboforge.backends.registry import get_recipe
    inspection=SimpleNamespace(speculative_components={})
    assert ForgePlanner._effective_components(get_recipe("qwen3_8_flash_next"),inspection,{})=="text"
    assert ForgePlanner._effective_components(get_recipe("qwen3_8_flash_next"),inspection,{"ngram":"table"})=="text,ngram"


def test_invalid_api_payloads_return_400_instead_of_internal_errors():
    from turboforge.api.server import create_app
    client=create_app().test_client()
    assert client.post("/api/inspect",json=["bad"]).status_code==400
    assert client.post("/api/inspect",json={"input_path":42}).status_code==400
    assert client.post("/api/pick-path",json=["bad"]).status_code==400
    assert client.post("/api/precheck",json={"model":"m","original_model":42}).status_code==400


def test_global_server_controller_can_stop_a_different_task():
    from turboforge.api.server import create_app
    app=create_app();manager=app.extensions["turboforge_build_manager"]
    with patch.object(manager,"active_servers",return_value=[]) as active,patch.object(manager,"stop_servers") as stop:
        client=app.test_client()
        assert client.get("/api/servers").json=={"servers":[]}
        assert client.delete("/api/servers").status_code==200
        stop.assert_called_once()
        assert active.call_count==2


def test_non_gsq_gguf_does_not_offer_native_gsq_q2():
    config={"architectures":["Qwen4ExpForCausalLM"]}
    result=choices(config,route="gguf",provided=("gguf",),source_quantization={"native_gsq_q2":False})
    assert result["targets"]
    assert "qwen3_8_flash_next_gsq_q2" not in {target["value"] for target in result["targets"]}
    supported=choices(config,route="gguf",provided=("gguf",),source_quantization={"native_gsq_q2":True})
    assert "qwen3_8_flash_next_gsq_q2" in {target["value"] for target in supported["targets"]}
