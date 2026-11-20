import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.helpers import write_config, write_safetensors
from turboforge.inspector import ModelInspector
from turboforge.manifest import BuildManifest
from turboforge.runner import BuildRunner
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


_EXTRA_SOURCES = ("gguf", "ternary", "quantized", "engram", "ngram", "imatrix", "mtp", "dflash2", "vision", "dspark")


@pytest.fixture
def quality_reference(tmp_path):
    """Real tiny manifests/artifacts; no converter, worker or GPU is started."""
    source = tmp_path / "model"
    source.mkdir()
    write_config(source)
    write_safetensors(source / "model.safetensors", {"weight": ("BF16", [2, 2])})
    sources, payloads = {}, {}
    for name in _EXTRA_SOURCES:
        path = tmp_path / name
        if name in ("quantized", "mtp", "dflash2"):
            path.mkdir()
            (path / "config.json").write_text("{}", encoding="utf-8")
            shard_dir = path / "shards"
            shard_dir.mkdir()
            payload = shard_dir / "weight.safetensors"
        else:
            payload = path
        payload.write_bytes(b"original")
        sources[name] = str(path)
        payloads[name] = payload
    baseline_path = tmp_path / "reference" / "model.ninfer"
    config = ForgeConfig(str(source), str(tmp_path / "current"), sources=sources, run_benchmark=True,
                         benchmark={"max_ppl_relative_increase": .1, "baseline_artifact": str(baseline_path),
                                    "perplexity_text": str(tmp_path / "corpus.txt")})
    inspection = ModelInspector().inspect(source)
    reference = BuildManifest.create(tmp_path / "reference", replace(config, output_dir=str(tmp_path / "reference")), inspection, {})
    baseline_path.write_bytes(b"baseline artifact")
    reference.add_artifact("ninfer_artifact", baseline_path, digest=True)
    current = BuildManifest.create(config.output_dir, config, inspection, {})
    # This validator does not use an engine or planner; avoid environment probes.
    runner = BuildRunner.__new__(BuildRunner)
    return runner, config, reference, current, payloads, inspection


def test_quality_reference_accepts_matching_content_despite_extra_source_mtime_changes(quality_reference):
    runner, config, reference, current, payloads, _ = quality_reference
    for payload in payloads.values():
        stat = payload.stat()
        os.utime(payload, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    assert runner._validate_quality_reference(config, current)["build_id"] == reference.build_id


@pytest.mark.parametrize("name", _EXTRA_SOURCES)
@pytest.mark.parametrize("replace_before_current_manifest", [False, True])
def test_quality_reference_rejects_same_path_same_size_same_mtime_content_replacement(
        tmp_path, quality_reference, name, replace_before_current_manifest):
    runner, config, _, current, payloads, inspection = quality_reference
    payload = payloads[name]
    stat = payload.stat()
    payload.write_bytes(b"replaced")
    os.utime(payload, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert payload.stat().st_size == stat.st_size
    assert payload.stat().st_mtime_ns == stat.st_mtime_ns
    if replace_before_current_manifest:
        output = tmp_path / "after-replacement"
        config = replace(config, output_dir=str(output))
        current = BuildManifest.create(output, config, inspection, {})
    with pytest.raises(ValueError, match="内容指纹不一致"):
        runner._validate_quality_reference(config, current)


@pytest.mark.parametrize("change", ["add", "remove", "rename", "resource", "type"])
def test_quality_reference_rejects_directory_inventory_or_type_changes(quality_reference, change):
    runner, config, _, current, payloads, _ = quality_reference
    directory = Path(config.sources["quantized"])
    payload = payloads["quantized"]
    if change == "add":
        (directory / "new-scales.bin").write_bytes(b"new")
    elif change == "remove":
        payload.unlink()
    elif change == "rename":
        payload.rename(payload.with_name("renamed.safetensors"))
    elif change == "resource":
        (directory / "config.json").write_text('{"changed": true}', encoding="utf-8")
    else:
        file_source = Path(config.sources["gguf"])
        file_source.unlink()
        file_source.mkdir()
        (file_source / "weight").write_bytes(b"original")
    with pytest.raises(ValueError, match="内容指纹不一致"):
        runner._validate_quality_reference(config, current)


@pytest.mark.parametrize("missing_evidence", ["reference", "current", "entry"])
def test_quality_reference_explicitly_refuses_legacy_or_partial_content_evidence(quality_reference, missing_evidence):
    runner, config, reference, current, _, _ = quality_reference
    if missing_evidence == "reference":
        del reference.data["additional_source_identity"]
    elif missing_evidence == "current":
        del current.data["additional_source_identity"]
    else:
        del reference.data["additional_source_identity"]["gguf"]
    reference.save()
    with pytest.raises(ValueError, match="缺少额外来源内容指纹.*旧记录"):
        runner._validate_quality_reference(config, current)


def test_quality_reference_without_extra_sources_still_requires_new_evidence(quality_reference):
    runner, config, reference, current, _, _ = quality_reference
    config.sources = {}
    reference.data["config"]["sources"] = {}
    current.data["additional_source_identity"] = {}
    reference.data["additional_source_identity"] = {}
    reference.save()
    assert runner._validate_quality_reference(config, current)["build_id"] == reference.build_id
    del reference.data["additional_source_identity"]
    reference.save()
    with pytest.raises(ValueError, match="旧记录"):
        runner._validate_quality_reference(config, current)


def test_quality_reference_rejects_missing_source_even_with_matching_missing_records(quality_reference):
    runner, config, reference, current, payloads, _ = quality_reference
    payloads["gguf"].unlink()
    reference.data["additional_source_identity"]["gguf"] = {"type": "missing"}
    current.data["additional_source_identity"]["gguf"] = {"type": "missing"}
    reference.save()
    with pytest.raises(ValueError, match="额外来源不存在"):
        runner._validate_quality_reference(config, current)


def test_quality_reference_validation_is_only_used_for_an_explicit_gate(quality_reference, monkeypatch):
    runner, config, _, current, _, _ = quality_reference
    config.benchmark = {}
    monkeypatch.setattr("turboforge.manifest.source_content_identity", lambda _: pytest.fail("unexpected hashing"))
    assert runner._validate_quality_reference(config, current) is None


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
