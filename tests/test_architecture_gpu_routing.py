from turboforge.backends.registry import (
    architecture_kind,
    QWEN35_DENSE,
    QWEN35_MOE,
    FLASH_NEXT,
    GEMMA4,
    FLINT,
    DEEPSEEK_V41,
)


def test_verified_moe_architectures_are_moe():
    for arch in (*QWEN35_MOE, *FLASH_NEXT, *DEEPSEEK_V41):
        assert architecture_kind({"architectures": [arch]}) == "moe"


def test_verified_dense_architectures_are_dense():
    for arch in (*QWEN35_DENSE, *GEMMA4, *FLINT):
        assert architecture_kind({"architectures": [arch]}) == "dense"


def test_expert_fields_identify_moe_outside_verified_tuples():
    for key in ("num_experts", "num_local_experts", "moe_intermediate_size"):
        assert architecture_kind({"architectures": ["SomeNewForCausalLM"], key: 8}) == "moe"


def test_nested_text_config_expert_fields_identify_moe():
    config = {"architectures": ["SomeNewForConditionalGeneration"],
              "text_config": {"num_experts": 64, "hidden_size": 2048}}
    assert architecture_kind(config) == "moe"


def test_architecture_tuple_wins_over_absent_expert_fields():
    # A verified dense architecture stays dense even with no expert fields.
    assert architecture_kind({"architectures": [QWEN35_DENSE[0]]}) == "dense"


def test_zero_or_missing_expert_fields_do_not_imply_moe():
    assert architecture_kind({"architectures": ["SomeNewForCausalLM"], "num_experts": 0}) == "unknown"
    assert architecture_kind({"architectures": ["SomeNewForCausalLM"]}) == "unknown"


def test_undecidable_config_is_unknown_not_a_guess():
    assert architecture_kind({}) == "unknown"
    assert architecture_kind({"architectures": []}) == "unknown"
    assert architecture_kind({"model_type": "llama"}) == "unknown"


import subprocess
import turboforge.system as system


class _FakeResult:
    def __init__(self, stdout):
        self.stdout = stdout


def _mock_smi(monkeypatch, stdout=None, exc=None):
    def fake_run(*args, **kwargs):
        if exc is not None:
            raise exc
        return _FakeResult(stdout)
    monkeypatch.setattr(system.subprocess, "run", fake_run)


def test_detect_gpu_blackwell(monkeypatch):
    _mock_smi(monkeypatch, "NVIDIA GeForce RTX 5070 Ti, 12.0, 12288, 580.97\n")
    gpu = system.detect_gpu()
    assert gpu["generation"] == "Blackwell · RTX 50"
    assert gpu["compute_cap"] == 12.0
    assert gpu["total_bytes"] == 12288 * 1024**2
    assert gpu["driver"] == "580.97"


def test_detect_gpu_ada_and_ampere(monkeypatch):
    _mock_smi(monkeypatch, "NVIDIA GeForce RTX 4090, 8.9, 24576, 560.00\n")
    assert system.detect_gpu()["generation"] == "Ada · RTX 40"
    _mock_smi(monkeypatch, "NVIDIA GeForce RTX 3060, 8.6, 12288, 550.00\n")
    assert system.detect_gpu()["generation"] == "Ampere · RTX 30"


def test_detect_gpu_unknown_capability_reports_raw_number(monkeypatch):
    _mock_smi(monkeypatch, "Some Future GPU, 13.5, 1024, 999.00\n")
    gpu = system.detect_gpu()
    assert gpu["compute_cap"] == 13.5
    assert gpu["generation"] == "compute 13.5"


def test_detect_gpu_no_driver_returns_none(monkeypatch):
    _mock_smi(monkeypatch, exc=OSError("nvidia-smi not found"))
    assert system.detect_gpu() is None


def test_detect_gpu_timeout_returns_none(monkeypatch):
    _mock_smi(monkeypatch, exc=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10))
    assert system.detect_gpu() is None


def test_detect_gpu_malformed_or_empty_returns_none(monkeypatch):
    _mock_smi(monkeypatch, "only,two\n")
    assert system.detect_gpu() is None
    _mock_smi(monkeypatch, "NVIDIA, notanumber, 12288, 580.0\n")
    assert system.detect_gpu() is None
    _mock_smi(monkeypatch, "\n")
    assert system.detect_gpu() is None


from types import SimpleNamespace
from unittest.mock import patch
from turboforge.precheck import choices, _gpu_scheme_note
from turboforge.backends.registry import FMT_FP8


def _gpu(cap):
    return {"name": "Test GPU", "generation": "x", "compute_cap": cap,
            "total_bytes": 12 * 1024**3, "driver": "580.0"}


_DENSE_27B = {"architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 5120}


def test_choices_blackwell_prioritizes_native_nvfp4():
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        result = choices(_DENSE_27B)
    assert result["architecture_kind"] == "dense"
    assert result["gpu"]["compute_cap"] == 12.0
    first = result["targets"][0]
    assert "nvfp4" in first["value"]
    assert first["gpu_note"] == "本卡原生 FP4 加速"
    ninfer = [t["value"] for t in result["targets"] if t["engine"] == "ninfer"]
    assert ninfer.index("bonsai2_27b_ternary") < ninfer.index("qwen3_8_27b")


def test_choices_ada_labels_nvfp4_as_non_native():
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(8.9)):
        result = choices(_DENSE_27B)
    nvfp4 = next(t for t in result["targets"] if t["value"] == "qwen3_8_27b_nvfp4")
    assert nvfp4["gpu_note"] == "本卡无原生 FP4，需反量化，未必提速"
    ninfer = [t["value"] for t in result["targets"] if t["engine"] == "ninfer"]
    assert ninfer.index("bonsai2_27b_ternary") < ninfer.index("qwen3_8_27b_nvfp4")


def test_choices_without_gpu_degrades_to_bpw_and_no_notes():
    with patch("turboforge.precheck.detect_gpu", return_value=None):
        result = choices(_DENSE_27B)
    assert result["gpu"] is None
    assert result["architecture_kind"] == "dense"
    assert result["scheme_order"] == "bits-per-weight"
    assert all(t["gpu_note"] is None for t in result["targets"])


def test_choices_identifies_moe():
    config = {"architectures": ["Qwen3_5MoeForCausalLM"], "hidden_size": 2048}
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        result = choices(config)
    assert result["architecture_kind"] == "moe"


def test_explicit_layout_choice_requires_explicit_target():
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        result = choices(_DENSE_27B, route="quantized", provided=("quantized",))
    assert result["requires_explicit_target"]
    # Ordering/labels are presentation only; no target is flagged as the single
    # auto recommendation (that stays with the planner's unchanged auto policy).
    assert all("recommended" not in t for t in result["targets"])
    assert result["targets"][0]["gpu_note"] == "本卡原生 FP4 加速"


def test_gpu_note_fp8_and_mxfp4_native_thresholds():
    fp8 = SimpleNamespace(storage_formats=(FMT_FP8,), extra_sources=())
    assert _gpu_scheme_note(fp8, 8.9) == "本卡原生 FP8 加速"
    assert _gpu_scheme_note(fp8, 8.6) == "本卡无原生 FP8，需反量化，未必提速"
    mxfp4 = SimpleNamespace(storage_formats=("gguf_mxfp4",), extra_sources=())
    assert _gpu_scheme_note(mxfp4, 12.0) == "本卡原生 MXFP4 加速"
    assert _gpu_scheme_note(mxfp4, 8.9) == "本卡无原生 MXFP4，需反量化，未必提速"
    groupwise = SimpleNamespace(storage_formats=("q4_g64_fp16",), extra_sources=())
    assert _gpu_scheme_note(groupwise, 12.0) is None


_MOE_35B = {"architectures": ["Qwen3_5MoeForCausalLM"], "hidden_size": 2048}
_MOE_FLASH_NEXT = {"architectures": ["Qwen4ExpForCausalLM"], "hidden_size": 2048}
_MOE_UNKNOWN = {"architectures": ["SomeNewMoeForCausalLM"], "num_experts": 64, "hidden_size": 2048}


def test_moe_engine_recommendation_blackwell_prefers_basalt():
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        result = choices(_MOE_FLASH_NEXT)
    assert result["architecture_kind"] == "moe"
    rec = result["moe_engine"]
    assert rec["engine"] == "basalt" and rec["label"] == "Basalt"
    assert rec is result["runtime_engine"]
    assert "github.com/jesdga95/basalt" in rec["url"] and "J12" in rec["steps"]
    assert "sm_120" in rec["reason"]
    assert "不集成" in rec["advisory"] and ".ninfer" in rec["advisory"]
    # Adapted model needs no compatibility banner.
    assert not rec.get("compat_check")


def test_moe_engine_recommendation_35b_a3b_is_strata_even_on_blackwell():
    # Basalt's published model list has no Qwen3.6-35B-A3B; Strata ships its
    # Unsloth GGUF, so the model family must override the GPU-only heuristic.
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        rec = choices(_MOE_35B)["moe_engine"]
    assert rec["engine"] == "strata"
    assert "已发布列表不含本模型" in rec["reason"]
    assert "Qwen3.6-35B-A3B" in rec["models"]
    # Blackwell + unadapted: surface the faster-Basalt compatibility decision up
    # front instead of silently routing away from it.
    assert "更快" in rec["compat_check"] and "自行核对" in rec["compat_check"]


def test_moe_engine_recommendation_unknown_family_gets_no_engine_promise():
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        rec = choices(_MOE_UNKNOWN)["moe_engine"]
    assert rec["engine"] == "strata"
    assert "更快" in rec["compat_check"] and "不在 Strata 已发布列表" in rec["compat_check"]
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(8.9)):
        rec = choices(_MOE_UNKNOWN)["moe_engine"]
    assert rec["engine"] == "strata"
    assert not rec.get("compat_check")
    assert "不在两引擎已发布模型列表" in rec["reason"]


def test_datacenter_blackwell_is_not_consumer_rtx50():
    # Compute 10.0 is datacenter Blackwell, not the RTX 50 sm_120 Basalt targets.
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(10.0)):
        rec = choices(_MOE_FLASH_NEXT)["runtime_engine"]
    assert rec["engine"] == "strata"
    assert "sm_120" in rec["reason"]
    assert not rec.get("compat_check")


def test_moe_engine_recommendation_non_blackwell_prefers_strata():
    # DC Blackwell (10.0) and Hopper (9.0) are not consumer sm_120, so Strata.
    for cap in (10.0, 9.0, 8.9, 8.6, 8.0, 7.5):
        for config in (_MOE_35B, _MOE_FLASH_NEXT):
            with patch("turboforge.precheck.detect_gpu", return_value=_gpu(cap)):
                rec = choices(config)["runtime_engine"]
            assert rec["engine"] == "strata"
            assert "J13" in rec["steps"] and rec["url"].endswith("/Strata")
            assert "Basalt" in rec["alternative"]
            assert not rec.get("compat_check")
    with patch("turboforge.precheck.detect_gpu", return_value=None):
        rec = choices(_MOE_35B)["moe_engine"]
    assert rec["engine"] == "strata"
    assert "未检测到 NVIDIA" in rec["reason"]
    assert not rec.get("compat_check")


def test_dense_recommends_fusion_kvmem_and_unknown_recommends_nothing():
    with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
        rec = choices(_DENSE_27B)["runtime_engine"]
        assert rec["engine"] == "fusion-kvmem"
        assert rec["url"] == "https://github.com/1314521gjy/ninfer-fusion-kvmem"
        assert "J14" in rec["steps"] and "Apache-2.0" in rec["steps"]
        assert choices(_DENSE_27B)["scheme_order"] == "native-format"
        assert choices({"architectures": ["SomeNewForCausalLM"]})["runtime_engine"] is None


def test_moe_engine_recommendation_covers_all_moe_architectures():
    # Basalt only for its published MoE families; Qwen3.6-35B-A3B stays Strata.
    for arch in (*FLASH_NEXT, *DEEPSEEK_V41):
        with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
            rec = choices({"architectures": [arch], "hidden_size": 2048})["moe_engine"]
        assert rec is not None and rec["engine"] == "basalt"
        assert not rec.get("compat_check")
    for arch in QWEN35_MOE:
        with patch("turboforge.precheck.detect_gpu", return_value=_gpu(12.0)):
            rec = choices({"architectures": [arch], "hidden_size": 2048})["moe_engine"]
        assert rec is not None and rec["engine"] == "strata"
        assert rec.get("compat_check")