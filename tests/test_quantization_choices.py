from pathlib import Path
import pytest
from turboforge.backends.registry import get_recipe, geometry_admits
from turboforge.backends.compiler import NInferCompiler
from turboforge.domain import ForgeConfig
from turboforge.reports import quality_gate


def test_selected_bits_use_real_custom_quantizer():
    recipe=get_recipe("qwen3_5_gguf_q4_0")
    assert recipe.nominal_bits==4
    assert geometry_admits(recipe,1024)
    assert not geometry_admits(recipe,5120)
    command=NInferCompiler().build_command(recipe,Path("model"),Path("out"),device="cpu")
    assert any("gguf_quantization.py:q4_0" in value for value in command)


def test_quality_gate_does_not_pass_missing_or_excessive_loss():
    quality={"metrics":{"perplexity":{"status":"not_measured"},"baseline_perplexity":{"status":"measured","value":10}}}
    assert quality_gate(quality,.1)["status"]=="not_measured"
    quality["metrics"]["perplexity"]={"status":"measured","value":12}
    assert quality_gate(quality,.1)["status"]=="failed"
    assert quality_gate(quality,.3)["status"]=="passed"


def test_quality_gate_requires_paired_evaluation():
    with pytest.raises(ValueError,match="质量门禁"):
        ForgeConfig.from_dict({"input_path":"m","output_dir":"o","benchmark":{"max_ppl_relative_increase":.1}})


def test_ppl_parser_reads_overall_score_not_model_filename():
    from turboforge.benchmark import BenchmarkEngine
    text="Perplexity result\nartifact: model0.8b\noverall 887 1.131042 3.098884\n"
    assert BenchmarkEngine._parse_ppl(text)=={"perplexity":3.098884}
    assert BenchmarkEngine._parse_ppl("Perplexity result\nartifact: model0.8b")=={}


def test_quality_gate_rejects_changed_baseline_file(tmp_path):
    import json
    from types import SimpleNamespace
    from turboforge.runner import BuildRunner
    baseline=tmp_path/"base.ninfer"
    baseline.write_bytes(b"changed")
    (tmp_path/"manifest.json").write_text(json.dumps({"source_files":[],"config":{"jailbreak":"none"},"artifacts":{"ninfer_artifact":{"sha256":"wrong"}}}))
    config=ForgeConfig("m","o",run_benchmark=True,benchmark={"max_ppl_relative_increase":.1,"baseline_artifact":str(baseline),"perplexity_text":"corpus.txt"})
    with pytest.raises(ValueError,match="校验值不一致"):
        BuildRunner()._validate_quality_reference(config,SimpleNamespace(data={"status":"validated","source_files":[]}))
