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
            with self.assertRaisesRegex(RuntimeError, "official driver"): cuda_package()

    def test_small_remote_model_without_runtime_stops_before_weights(self):
        with patch("turboforge.bootstrap.local_tools", return_value=None):
            with self.assertRaisesRegex(ValueError, "Stopped before downloading model weights"):
                check_config_support({"architectures":["Qwen3_5ForConditionalGeneration"], "text_config":{"hidden_size":1024}})

    def test_unknown_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no wired-in conversion path"):
            check_config_support({"architectures":["Unknown"]})

    def test_broken_bundle_does_not_fall_back_to_host_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "runtime").mkdir()
            (root / "runtime/bundle.json").write_text(json.dumps({"files":{"bin/ninfer.exe":"bad"}}))
            with patch("turboforge.bootstrap.__file__", str(root / "src/turboforge/bootstrap.py")), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "corrupt or missing"): local_tools()

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
                with self.assertRaisesRegex(ValueError, "Stopped before downloading model weights"):
                    resolver.snapshot("Org/Small")
                self.assertEqual(fetch.call_count, 1)

    def test_install_failure_surfaces_after_readable_download_progress(self):
        from turboforge.bootstrap import run
        messages = []
        with self.assertRaisesRegex(RuntimeError, "exit code 9"):
            run([sys.executable, "-c", "print('Progress 1048576 of 2097152');raise SystemExit(9)"], messages.append)
        self.assertTrue(any("50%" in message for message in messages))

    def test_missing_gguf_is_installed_with_converter_dependencies(self):
        from turboforge.bootstrap import ensure_dependencies
        versions = {"torch":"2.14.1", "numpy":"2.2.6", "safetensors":"0.8.0", "pip":"26.2.2"}
        with patch("turboforge.bootstrap.importlib.util.find_spec", side_effect=lambda name: None if name == "gguf" else object()), patch("turboforge.bootstrap.importlib.metadata.version", side_effect=versions.__getitem__), patch("turboforge.bootstrap.run") as install:
            ensure_dependencies("none", lambda _: None)
        install.assert_called_once()
        self.assertIn("gguf>=0.19,<1", install.call_args.args[0])


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
                with self.assertRaisesRegex(RuntimeError, "missing runtime"): collect(root)
