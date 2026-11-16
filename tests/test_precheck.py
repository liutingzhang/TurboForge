import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from turboforge.precheck import precheck
from turboforge.domain import InputKind


def test_missing_input_has_one_actionable_reason():
    with tempfile.TemporaryDirectory() as root:
        result = precheck(str(Path(root) / "missing"))
    assert not result["ready"] and "重新选择" in result["message"]


def test_gguf_without_origin_does_not_offer_weight_editing():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "model.gguf"
        path.touch()
        inspection = SimpleNamespace(input_kind=InputKind.GGUF, quantization={})
        with patch("turboforge.precheck.cuda_package"), patch("turboforge.precheck.ModelInspector.inspect", return_value=inspection):
            result = precheck(str(path))
    assert not result["ready"]
    assert result["action"] == "provide_original"
    assert "原始模型" in result["message"]


def test_valid_hf_can_advance_without_installing_dependencies():
    with tempfile.TemporaryDirectory() as root:
        inspection = SimpleNamespace(input_kind=InputKind.SAFETENSORS, config={}, usable=True, speculative_components={}, weight_precision="bf16")
        with patch("turboforge.precheck.cuda_package"), patch("turboforge.precheck.ModelInspector.inspect", return_value=inspection), patch("turboforge.precheck.check_config_support"):
            result = precheck(root)
    assert result["ready"] and result["can_edit_weights"]


def test_gguf_directory_without_config_requests_original_source():
    from tests.helpers import write_gguf
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "model.gguf"
        write_gguf(path)
        with patch("turboforge.precheck.cuda_package"):
            result = precheck(str(path), root)
    assert not result["ready"]
    assert result["action"] == "provide_original"
    assert "config.json" in result["message"]
    assert "No such file" not in result["message"]


def test_small_gguf_requires_switching_input_to_original_model():
    from tests.helpers import write_gguf, write_config
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "model.gguf"
        write_gguf(path)
        base = Path(root) / "original"
        base.mkdir()
        write_config(base, architectures=["Qwen3_5ForConditionalGeneration"], hidden_size=1024)
        with patch("turboforge.precheck.cuda_package"):
            result = precheck(str(path), str(base))
    assert not result["ready"]
    assert result["action"] == "choose_original"


def test_clean_gguf_can_use_explicit_remote_metadata_without_local_config():
    from tests.helpers import write_gguf
    config = {"architectures":["Qwen3_5ForCausalLM"], "hidden_size":5120}
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "model.gguf"
        write_gguf(path)
        with patch("turboforge.precheck.cuda_package"), patch("turboforge.precheck._remote_config", return_value=config) as remote:
            result = precheck(str(path), "org/original")
    remote.assert_called_once_with("org/original")
    assert result["ready"]
    assert any(t["value"] == "qwen3_8_27b_gguf" for t in result["targets"])
