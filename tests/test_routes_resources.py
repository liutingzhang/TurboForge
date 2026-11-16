from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from turboforge.backends.compiler import UpstreamStatus
from turboforge.backends.registry import (
    PINNED_UPSTREAM_REVISION,
    get_recipe,
    to_capability,
)
from turboforge.domain import ForgeConfig
from turboforge.inspector import ModelInspector
from turboforge.planner import ForgePlanner
from turboforge.resources import ResourceEstimator

from tests.helpers import write_config, write_safetensors, write_gguf


def make_upstream(root: Path) -> UpstreamStatus:
    return UpstreamStatus(
        root=root, found=True, revision=PINNED_UPSTREAM_REVISION,
        revision_matches_pin=True, torch_available=True, numpy_available=True,
        diagnostics=[],
    )


class GgufRouteTest(unittest.TestCase):
    """GGUF is a secondary source beside the HF checkpoint, so the inspected
    input is SAFETENSORS; the GGUF recipes must still be reachable."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        write_config(self.root, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        self.gguf = Path(self.temp.name) / "Qwen3.8-27B-Q4.gguf"
        write_gguf(self.gguf)
        self.planner = ForgePlanner(
            inspector=ModelInspector(), upstream=make_upstream(self.root)
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **overrides) -> ForgeConfig:
        values = dict(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="qwen3_8_27b_gguf",
            jailbreak="none",
        )
        values.update(overrides)
        return ForgeConfig(**values)

    def errors(self, plan) -> set[str]:
        return {d.code for d in plan.diagnostics if d.severity.value == "error"}

    def test_gguf_target_with_source_is_executable(self) -> None:
        plan = self.planner.plan(self.config(sources={"gguf": str(self.gguf)}))
        self.assertTrue(plan.executable)
        self.assertEqual(plan.backend, "qwen3_8_27b_gguf")

    def test_gguf_target_without_source_is_blocked(self) -> None:
        plan = self.planner.plan(self.config())
        self.assertFalse(plan.executable)
        self.assertIn("recipe-source-missing", self.errors(plan))

    def test_gguf_not_auto_selected_without_source(self) -> None:
        plan = self.planner.plan(self.config(target="auto"))
        self.assertTrue(plan.executable)
        self.assertNotIn("gguf", plan.backend)


class ResourceEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        write_config(self.root, architectures=["Qwen3_5ForCausalLM"])
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        self.inspection = ModelInspector().inspect(self.root)
        self.output = Path(self.temp.name) / "build"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def capability(self, recipe_name: str):
        return to_capability(
            get_recipe(recipe_name),
            upstream_available=True, torch_available=True,
        )

    def test_serving_vram_estimate_is_labeled_and_conservative(self) -> None:
        mib = 1024**2
        self.assertIsNone(ResourceEstimator._serving_vram(None))
        self.assertIsNone(ResourceEstimator._serving_vram(0))
        self.assertEqual(ResourceEstimator._serving_vram(1000), 1000 + 512 * mib)
        large = 8 * 1024**3
        self.assertEqual(ResourceEstimator._serving_vram(large), large + int(large * 0.15))

    def test_groupwise_route_estimates_vram_and_flags_unverified_ram(self) -> None:
        estimate = ResourceEstimator().estimate(
            self.inspection, self.capability("qwen3_8_27b"), self.output,
            jailbreak="none", streaming=False,
        )
        self.assertIsNotNone(estimate.peak_vram_bytes)
        self.assertIsNone(estimate.peak_ram_bytes)
        codes = {d.code for d in estimate.diagnostics}
        self.assertIn("ram-profile-unverified", codes)
        self.assertTrue(any("serving estimate" in a for a in estimate.assumptions))
        # The old false claim that the build is blocked must be gone.
        self.assertFalse(any("remains blocked" in a for a in estimate.assumptions))
        self.assertFalse(any(d.severity.value == "error" for d in estimate.diagnostics))

    def test_streaming_route_estimates_peak_ram(self) -> None:
        estimate = ResourceEstimator().estimate(
            self.inspection, self.capability("qwen3_8_27b_gguf"), self.output,
            jailbreak="none", streaming=True,
        )
        self.assertIsNotNone(estimate.peak_ram_bytes)
        self.assertNotIn("ram-profile-unverified",
                         {d.code for d in estimate.diagnostics})


if __name__ == "__main__":
    unittest.main()
