"""The shared serving profile keeps the plan preview, launcher and benchmark in
agreement. These tests pin the geometry-driven choices and assert that plan.serving
equals what the runner actually emits for the same config (no drift)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from turboforge import serving
from turboforge.backends.compiler import UpstreamStatus
from turboforge.backends.registry import (
    GDN_BRIDGE_CONTEXT_CEILING,
    PINNED_UPSTREAM_REVISION,
    get_recipe,
)
from turboforge.domain import ForgeConfig
from turboforge.inspector import ModelInspector
from turboforge.planner import ForgePlanner
from turboforge.runner import BuildRunner

from tests.helpers import write_config, write_safetensors


def make_upstream(root: Path) -> UpstreamStatus:
    return UpstreamStatus(
        root=root, found=True, revision=PINNED_UPSTREAM_REVISION,
        revision_matches_pin=True, torch_available=True, numpy_available=True,
        diagnostics=[],
    )


def test_default_serving_port_follows_allowlist(monkeypatch):
    from turboforge import local_endpoint

    class Fake:
        allowed_local_origins = ("http://127.0.0.1:23456",)

    monkeypatch.setattr(local_endpoint, "ConfigStore", lambda: Fake())
    assert local_endpoint.default_serving_port() == 23456

    class Empty:
        allowed_local_origins = ()

    monkeypatch.setattr(local_endpoint, "ConfigStore", lambda: Empty())
    assert local_endpoint.default_serving_port() == local_endpoint.FALLBACK_SERVING_PORT


class ServingMathTest(unittest.TestCase):
    GDN = {"hidden_size": 1024, "linear_num_value_heads": 16}

    def test_kv_dtype_by_geometry(self) -> None:
        bridge = get_recipe("qwen3_5_gguf")
        native = get_recipe("qwen3_8_27b")
        self.assertEqual(serving.kv_dtype(bridge, self.GDN), "bf16")
        self.assertEqual(serving.kv_dtype(native, {"hidden_size": 5120}), "rk4v4")
        # A bridge without the GDN head signature has no known ceiling -> rk4v4.
        self.assertEqual(serving.kv_dtype(bridge, {"hidden_size": 1024}), "rk4v4")

    def test_spec_choice_dflash2_wins(self) -> None:
        self.assertEqual(serving.spec_choice(True, True), ("dflash2", 7))
        self.assertEqual(serving.spec_choice(True, False), ("mtp", 3))
        self.assertEqual(serving.spec_choice(False, True), ("dflash2", 7))
        self.assertIsNone(serving.spec_choice(False, False))

    def test_effective_context_clamps(self) -> None:
        bridge = get_recipe("qwen3_5_gguf")
        self.assertEqual(
            serving.effective_context(bridge, self.GDN, {"max_context": 32768}),
            GDN_BRIDGE_CONTEXT_CEILING,
        )
        self.assertEqual(serving.effective_context(bridge, self.GDN, {"max_context": 512}), 512)

    def test_launcher_flags_bridge_omits_embedding_q4(self) -> None:
        flags = serving.launcher_flags(get_recipe("qwen3_5_gguf"), self.GDN, True, False)
        self.assertNotIn("--embedding-q4", flags)
        self.assertIn("--kv-dtype bf16", flags)
        self.assertIn("--spec mtp", flags)
        self.assertIn("--draft-tokens 3", flags)
        self.assertNotIn("--lm-head-draft", flags)

    def test_launcher_flags_native_keeps_embedding_q4(self) -> None:
        flags = serving.launcher_flags(get_recipe("qwen3_8_27b"), {"hidden_size": 5120}, False, False)
        self.assertIn("--embedding-q4", flags)
        self.assertIn("--kv-dtype rk4v4", flags)
        self.assertNotIn("--spec mtp", flags)


class PlanServingConsistencyTest(unittest.TestCase):
    """plan.serving must equal what the runner emits for the same config."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        write_config(self.root, architectures=["Qwen3_5ForCausalLM"],
                     hidden_size=1024, linear_num_value_heads=16)
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
            "mtp.fc.weight": ("BF16", [8, 16]),
            "mtp.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8]),
            "mtp.norm.weight": ("BF16", [8]),
        })

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_plan_serving_matches_runner(self) -> None:
        planner = ForgePlanner(inspector=ModelInspector(), upstream=make_upstream(self.root))
        config = ForgeConfig(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto", jailbreak="none", device="cpu",
            runtime={"max_context": 32768, "max_concurrency": 2, "port": 8123},
        )
        plan = planner.plan(config)
        self.assertIsNotNone(plan.serving)
        self.assertEqual(plan.serving["kv_dtype"], "bf16")
        self.assertEqual(plan.serving["spec"], "mtp")
        self.assertEqual(plan.serving["draft_tokens"], 3)
        self.assertEqual(plan.serving["max_context"], GDN_BRIDGE_CONTEXT_CEILING)
        self.assertEqual(plan.serving["context_ceiling"], GDN_BRIDGE_CONTEXT_CEILING)
        self.assertEqual(plan.serving["max_concurrency"], 2)
        self.assertEqual(plan.serving["port"], 8123)
        self.assertFalse(plan.serving["embedding_q4"])
        # The runner's launcher flags / kv / context must equal the preview's.
        runner = BuildRunner.__new__(BuildRunner)
        self.assertEqual(runner._launcher_flags(plan), plan.serving["launcher_flags"])
        self.assertEqual(runner._kv_dtype(plan), plan.serving["kv_dtype"])
        self.assertEqual(runner._effective_context(config, plan), plan.serving["max_context"])

    def test_plan_serving_none_when_no_recipe(self) -> None:
        planner = ForgePlanner(inspector=ModelInspector(), upstream=make_upstream(self.root))
        config = ForgeConfig(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="qwen3_8_27b", jailbreak="none", device="cpu",
        )
        plan = planner.plan(config)
        self.assertFalse(plan.executable)
        self.assertIsNone(plan.backend)
        self.assertIsNone(plan.serving)


if __name__ == "__main__":
    unittest.main()