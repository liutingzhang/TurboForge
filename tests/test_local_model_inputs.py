"""Local paths and original-source contracts must agree across entry points."""
from pathlib import Path
from unittest.mock import patch

import pytest

from turboforge.inspector import ModelInspector
from turboforge.model_sources import ModelResolver
from turboforge.precheck import precheck
from tests.helpers import write_config, write_gguf, write_safetensors


def hf_model(root):
    root.mkdir()
    write_config(root, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
    write_safetensors(root / "model.safetensors", {"weight": ("BF16", [2, 2])})
    return root


@pytest.mark.parametrize("wrap", [lambda p: f'"{p}"', lambda p: f"  {p}  ", lambda p: f" '{p}' "])
def test_pasted_local_model_works_in_check_and_resolution(tmp_path, wrap):
    model = hf_model(tmp_path / "本机模型 with spaces")
    reference = wrap(model)
    with patch("turboforge.precheck.cuda_package"):
        result = precheck(reference)
    assert result["ready"], result
    resolved = ModelResolver(tmp_path / "cache", lambda _: None).resolve(reference)
    assert resolved.model == model
    assert ModelInspector().inspect(reference).usable


def test_pasted_path_reaches_build_resolution(tmp_path):
    from turboforge.preparation import prepare
    model = hf_model(tmp_path / "model")
    with patch("turboforge.preparation.cache_root", return_value=tmp_path / "cache"), patch("turboforge.preparation.cuda_package"), patch("turboforge.preparation.ModelResolver.resolve", side_effect=RuntimeError("resolution reached")) as resolve:
        with pytest.raises(RuntimeError, match="resolution reached"):
            prepare(f' "{model}" ', {}, lambda _: None)
    assert resolve.call_args.args[0] == str(model)


def test_gguf_metadata_does_not_read_unused_safetensors(tmp_path):
    base = tmp_path / "original"
    base.mkdir()
    write_config(base, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
    (base / "model.safetensors").write_bytes(b"not downloaded completely")
    gguf = tmp_path / "weights.gguf"
    write_gguf(gguf)
    with patch("turboforge.precheck.cuda_package"):
        result = precheck(str(gguf), f' "{base}" ')
    assert result["ready"], result
    resolved = ModelResolver(tmp_path / "cache", lambda _: None).resolve(str(gguf), f' "{base}" ')
    assert resolved.model == base
    assert resolved.sources == {"gguf": str(gguf)}


def test_metadata_requires_tokenizer_even_if_weights_exist(tmp_path):
    base = hf_model(tmp_path / "original")
    (base / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        ModelInspector().inspect(base, metadata_only=True)


def test_resolver_rejects_gguf_folder_as_original_before_install(tmp_path):
    gguf = tmp_path / "weights.gguf"
    write_gguf(gguf)
    with pytest.raises(ValueError, match="config.json"):
        ModelResolver(tmp_path / "cache", lambda _: None).resolve(str(gguf), str(tmp_path))


def test_hf_input_with_extra_original_is_rejected_during_check(tmp_path):
    model = hf_model(tmp_path / "model")
    with patch("turboforge.precheck.cuda_package"):
        result = precheck(str(model), str(model))
    assert not result["ready"]
    assert "清空" in result["message"]


def test_remote_gguf_and_local_source_have_same_validation(tmp_path):
    gguf_info = {"sha":"a" * 40, "siblings":[{"rfilename":"model.gguf"}]}
    with patch("turboforge.precheck.cuda_package"), patch.object(ModelResolver, "info", return_value=gguf_info):
        result = precheck("org/gguf", str(tmp_path))
    assert not result["ready"]
    assert result["action"] == "provide_original"
    assert "config.json" in result["message"]


def test_remote_original_without_tokenizer_is_not_ready():
    info = {"sha":"a" * 40, "siblings":[{"rfilename":"config.json"}]}
    from turboforge.precheck import _remote_config
    with patch.object(ModelResolver, "info", return_value=info):
        with pytest.raises(ValueError, match="tokenizer"):
            _remote_config("org/base")


def test_local_path_pasted_through_api_is_accepted(tmp_path):
    from turboforge.api.server import create_app
    model = hf_model(tmp_path / "model")
    with patch.dict("os.environ", {}, clear=True):
        app = create_app(assistant_root=tmp_path / "assistant")
    try:
        with patch("turboforge.precheck.cuda_package"):
            response = app.test_client().post("/api/precheck", json={"model": f'  "{model}"  '})
        assert response.status_code == 200
        assert response.json["ready"], response.json
    finally:
        app.extensions["turboforge_build_manager"].close()
