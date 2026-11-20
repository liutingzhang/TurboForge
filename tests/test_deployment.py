from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from turboforge.bootstrap import cuda_package, local_tools
from turboforge.preparation import check_config_support
from turboforge.windows_runtime import imports, collect


class CompatibilityTest(unittest.TestCase):
    def test_driver_gpu_matrix(self):
        for capability, driver, expected in [("7.0", "580.99", "12"), ("7.5", "580.99", "13"), ("8.6", "550.99", "12"), ("12.0", "615.99", "13"), ("12.0", "570.99", None), ("6.1", "580.99", None), ("8.9", "527.00", None), ("9.0", "615.99", None)]:
            with self.subTest(capability=capability, driver=driver), patch("turboforge.bootstrap.sys.platform", "win32"), patch("turboforge.bootstrap.platform.machine", return_value="AMD64"), patch("turboforge.bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0, capability + ", " + driver)):
                if expected:
                    self.assertEqual(cuda_package(), expected)
                else:
                    with self.assertRaises(RuntimeError): cuda_package()

    def test_missing_driver_is_actionable(self):
        with patch("turboforge.bootstrap.sys.platform", "win32"), patch("turboforge.bootstrap.platform.machine", return_value="AMD64"), patch("turboforge.bootstrap.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "官方驱动"): cuda_package()

    def test_small_remote_model_without_runtime_stops_before_weights(self):
        with patch("turboforge.bootstrap.local_tools", return_value=None):
            with self.assertRaisesRegex(ValueError, "下载模型权重前停止"):
                check_config_support({"architectures":["Qwen3_5ForConditionalGeneration"], "text_config":{"hidden_size":1024}})

    def test_unknown_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "没有已接入转换路径"):
            check_config_support({"architectures":["Unknown"]})

    def test_broken_bundle_does_not_fall_back_to_host_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "runtime").mkdir()
            (root / "runtime/bundle.json").write_text(json.dumps({"files":{"bin/ninfer.exe":"bad"}}))
            with patch("turboforge.bootstrap.__file__", str(root / "src/turboforge/bootstrap.py")), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "损坏或缺失"): local_tools()

    def test_remote_small_model_checks_runtime_before_download(self):
        from turboforge.model_sources import ModelResolver
        with tempfile.TemporaryDirectory() as temp:
            resolver = ModelResolver(Path(temp), lambda _: None, check_config_support)
            info = {"sha":"a" * 40, "siblings":[{"rfilename":"config.json", "size":100}, {"rfilename":"model.safetensors", "size":100000000000}]}
            def download(url, destination, **kwargs):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"architectures":["Qwen3_5ForConditionalGeneration"], "text_config":{"hidden_size":1024}}))
                return destination
            with patch.object(resolver, "info", return_value=info), patch("turboforge.model_sources.download", side_effect=download) as fetch, patch("turboforge.bootstrap.local_tools", return_value=None):
                with self.assertRaisesRegex(ValueError, "下载模型权重前停止"):
                    resolver.snapshot("Org/Small")
                self.assertEqual(fetch.call_count, 1)

    def test_install_failure_surfaces_after_readable_download_progress(self):
        from turboforge.bootstrap import run
        messages = []
        with self.assertRaisesRegex(RuntimeError, "退出码 9"):
            run([sys.executable, "-c", "print('Progress 1048576 of 2097152');raise SystemExit(9)"], messages.append)
        self.assertTrue(any("50%" in message for message in messages))

    def test_missing_gguf_is_installed_with_converter_dependencies(self):
        from turboforge.bootstrap import ensure_dependencies
        versions = {"torch":"2.14.1", "numpy":"2.2.6", "safetensors":"0.8.0", "pip":"26.2.2"}
        with patch("turboforge.bootstrap.importlib.util.find_spec", side_effect=lambda name: None if name == "gguf" else object()), patch("turboforge.bootstrap.importlib.metadata.version", side_effect=versions.__getitem__), patch("turboforge.bootstrap.run") as install:
            ensure_dependencies("none", lambda _: None)
        install.assert_called_once()
        self.assertIn("gguf>=0.19,<1", install.call_args.args[0])


def small_hf_config():
    return {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
            "hidden_size": 1024, "max_position_embeddings": 32768}


def test_missing_small_adapter_allows_valid_hf_fallback(tmp_path):
    from turboforge.precheck import choices, precheck
    from tests.helpers import write_config, write_safetensors
    config = small_hf_config()
    write_config(tmp_path, **config)
    write_safetensors(tmp_path/"model.safetensors", {"weight": ("BF16", [2, 2])})
    with patch("turboforge.bootstrap.local_tools", return_value=(tmp_path, tmp_path)), \
         patch("turboforge.precheck.cuda_package"), patch("turboforge.precheck.detect_gpu", return_value=None):
        check_config_support(config)
        options = choices(config)
        result = precheck(str(tmp_path))
    assert result["ready"] and result["can_edit_weights"]
    assert options["requires_explicit_target"]
    assert {item["engine"] for item in options["targets"]} == {"gptqmodel"}
    assert {item["value"] for item in options["targets"]} == {"hf_gptq_2", "hf_gptq_3", "hf_gptq_4", "hf_gptq_8"}


def test_unusable_local_tools_do_not_block_valid_hf_metadata():
    with patch("turboforge.bootstrap.local_tools", side_effect=RuntimeError("damaged bundle")):
        check_config_support(small_hf_config())


def test_present_adapter_preserves_ninfer_routes(tmp_path):
    from turboforge.precheck import choices
    adapter = tmp_path/"tools/convert/qwen3_5_gguf.py"
    adapter.parent.mkdir(parents=True); adapter.touch()
    with patch("turboforge.bootstrap.local_tools", return_value=(tmp_path, tmp_path)), \
         patch("turboforge.precheck.detect_gpu", return_value=None):
        check_config_support(small_hf_config())
        result = choices(small_hf_config())
    assert any(item["value"] == "qwen3_5_gguf" for item in result["targets"])
    assert not result["requires_explicit_target"]


def test_remote_auto_and_explicit_unavailable_ninfer_stop_before_weights(tmp_path):
    import pytest
    from turboforge.preparation import prepare
    config = small_hf_config()
    info = {"sha": "c"*40, "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors", "size": 100000000000}]}
    def metadata_only_download(url, destination, **kwargs):
        assert url.endswith("/config.json"), "must not download weights"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(config), encoding="utf-8")
    for target, reason in (("auto", "明确选择 HF"), ("qwen3_5_gguf", "指定 NInfer")):
        with patch("turboforge.preparation.cache_root", return_value=tmp_path), \
             patch("turboforge.preparation.cuda_package"), \
             patch("turboforge.model_sources.ModelResolver.info", return_value=info), \
             patch("turboforge.model_sources.download", side_effect=metadata_only_download) as download, \
             patch("turboforge.bootstrap.local_tools", return_value=None), \
             patch("turboforge.preparation.ensure_tools") as tools, \
             patch("turboforge.preparation.ensure_dependencies") as deps:
            with pytest.raises(ValueError, match=reason):
                prepare("Org/Small", {"target": target}, lambda _: None)
        assert download.call_count == 1
        tools.assert_not_called(); deps.assert_not_called()


def test_local_auto_cannot_reselect_missing_ninfer_adapter(tmp_path):
    import pytest
    from turboforge.preparation import prepare
    from tests.helpers import write_config, write_safetensors
    source = tmp_path/"model"; source.mkdir()
    write_config(source, **small_hf_config())
    write_safetensors(source/"model.safetensors", {"weight": ("BF16", [2, 2])})
    original = (source/"config.json").read_bytes()
    for target, reason in (("auto", "明确选择 HF"), ("qwen3_5_gguf", "指定 NInfer")):
        with patch("turboforge.preparation.cache_root", return_value=tmp_path/"cache"), \
             patch("turboforge.preparation.cuda_package"), \
             patch("turboforge.planner.probe_upstream"), \
             patch("turboforge.bootstrap.local_tools", return_value=None), \
             patch("turboforge.model_sources.download", side_effect=AssertionError("no downloads")), \
             patch("turboforge.preparation.ensure_tools") as tools, \
             patch("turboforge.preparation.ensure_dependencies") as deps:
            with pytest.raises(ValueError, match=reason):
                prepare(str(source), {"target": target}, lambda _: None)
        tools.assert_not_called(); deps.assert_not_called()
    assert (source/"config.json").read_bytes() == original


def test_explicit_hf_prepare_works_without_small_ninfer_adapter(tmp_path):
    from turboforge.preparation import prepare
    from tests.helpers import write_config, write_safetensors
    source = tmp_path/"model"; source.mkdir()
    write_config(source, **small_hf_config())
    write_safetensors(source/"model.safetensors", {"weight": ("BF16", [2, 2])})
    with patch("turboforge.preparation.cache_root", return_value=tmp_path/"cache"), \
         patch("turboforge.preparation.cuda_package"), \
         patch("turboforge.bootstrap.local_tools", return_value=None), \
         patch("turboforge.model_sources.download", side_effect=AssertionError("no downloads")), \
         patch("turboforge.preparation.ensure_tools") as tools, \
         patch("turboforge.preparation.ensure_dependencies") as deps:
        result = prepare(str(source), {"target": "hf_gptq_4"}, lambda _: None)
    assert result.target == "hf_gptq_4" and result.device == "cuda"
    assert result.upstream_dir is None and result.ninfer_dir is None
    tools.assert_not_called(); deps.assert_not_called()


def test_auto_prepare_pins_existing_policy_after_adapter_filter(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from turboforge.preparation import prepare
    from tests.helpers import write_config, write_safetensors
    source = tmp_path/"model"; source.mkdir()
    write_config(source, architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text", hidden_size=5120)
    write_safetensors(source/"model.safetensors", {"weight": ("BF16", [2, 2])})
    # Keep all runtime/environment effects confined to test doubles and temp.
    monkeypatch.setenv("TURBOFORGE_NINFER_SRC", str(tmp_path))
    monkeypatch.setenv("TURBOFORGE_NINFER_BIN", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with patch("turboforge.preparation.cache_root", return_value=tmp_path/"cache"), \
         patch("turboforge.preparation.cuda_package"), patch("turboforge.planner.probe_upstream"), \
         patch("turboforge.bootstrap.local_tools", return_value=None), \
         patch("turboforge.preparation.ensure_tools", return_value=(tmp_path, tmp_path)) as tools, \
         patch("turboforge.preparation.ensure_dependencies") as deps:
        result = prepare(str(source), {}, lambda _: None)
    assert result.target == "qwen3_8_27b"
    tools.assert_called_once_with(tmp_path/"cache", unittest.mock.ANY, small_model=False)
    deps.assert_called_once()


class RuntimeTest(unittest.TestCase):
    def test_non_pe_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "fake.exe"
            p.write_bytes(b"not an executable")
            with self.assertRaises(ValueError): imports(p)

    def test_missing_transitive_dll_blocks_distribution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "engine.exe").touch()
            with patch("turboforge.windows_runtime.imports", return_value=["turboforge-missing.dll"]), patch.dict(os.environ, {"PATH":"", "SystemRoot":temp}):
                with self.assertRaisesRegex(RuntimeError, "缺少运行库"): collect(root)
