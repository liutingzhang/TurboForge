"""Native-acceleration routing for small Qwen3.5 dense checkpoints.

The native groupwise weight kernels are compile-time locked to the 27B/35B
geometries, so a small model (e.g. 0.8B, hidden 1024) must route to the
geometry-generic GGUF Q8_0 bridge and import its embedded MTP tensors for
speculative decoding. These tests pin that routing, the honest marking, and the
serve-time flags, using the same facts verified against the pinned converter.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from turboforge.backends.compiler import UpstreamStatus
from turboforge.backends.registry import (
    FMT_GGUF_Q8_0,
    GDN_BRIDGE_CONTEXT_CEILING,
    NATIVE_DENSE_HIDDEN,
    PINNED_UPSTREAM_REVISION,
    geometry_admits,
    get_recipe,
    serving_context_ceiling,
)
from turboforge.domain import (
    ForgeConfig,
    ForgePlan,
    InputKind,
    ModelInspection,
    PlanStep,
)
from turboforge.inspector import ModelInspector
from turboforge.manifest import BuildManifest
from turboforge.planner import ForgePlanner
from turboforge.runner import BuildRunner

from tests.helpers import write_config, write_safetensors


def make_upstream(root: Path) -> UpstreamStatus:
    return UpstreamStatus(
        root=root, found=True, revision=PINNED_UPSTREAM_REVISION,
        revision_matches_pin=True, torch_available=True, numpy_available=True,
        diagnostics=[],
    )


class GeometryGateTest(unittest.TestCase):
    def test_bridge_admits_any_hidden(self) -> None:
        bridge = get_recipe("qwen3_5_gguf")
        self.assertIn(FMT_GGUF_Q8_0, bridge.storage_formats)
        for hidden in (768, 1024, 2048, 5120):
            self.assertTrue(geometry_admits(bridge, hidden))

    def test_native_dense_locked_to_27b(self) -> None:
        for name in ("qwen3_8_27b", "qwen3_6_27b", "qwen3_8_27b_nvfp4",
                     "bonsai2_27b_ternary", "qwen3_8_27b_gguf"):
            recipe = get_recipe(name)
            self.assertTrue(geometry_admits(recipe, NATIVE_DENSE_HIDDEN), name)
            self.assertFalse(geometry_admits(recipe, 1024), name)

    def test_native_moe_locked_to_35b(self) -> None:
        recipe = get_recipe("qwen3_6_35b_a3b")
        self.assertTrue(geometry_admits(recipe, 2048))
        self.assertFalse(geometry_admits(recipe, 1024))

    def test_unknown_hidden_is_not_blocked(self) -> None:
        # Missing geometry must not invent a block; the converter validates it.
        self.assertTrue(geometry_admits(get_recipe("qwen3_8_27b"), None))

    def test_explicit_whitelist_overrides_default(self) -> None:
        restricted = replace(get_recipe("qwen3_5_gguf"), hidden_sizes=(1024,))
        self.assertTrue(geometry_admits(restricted, 1024))
        self.assertFalse(geometry_admits(restricted, 5120))


class SmallModelRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, *, mtp: bool = False, multimodal: bool = False) -> None:
        tensors = {"model.layers.0.q_proj.weight": ("BF16", [8, 8])}
        if mtp:
            tensors.update({
                "mtp.fc.weight": ("BF16", [8, 16]),
                "mtp.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8]),
                "mtp.norm.weight": ("BF16", [8]),
            })
        if multimodal:
            write_config(self.root, architectures=["Qwen3_5ForConditionalGeneration"],
                         text_config={"hidden_size": 1024})
        else:
            write_config(self.root, architectures=["Qwen3_5ForCausalLM"], hidden_size=1024)
        write_safetensors(self.root / "model.safetensors", tensors)

    def _planner(self) -> ForgePlanner:
        return ForgePlanner(inspector=ModelInspector(), upstream=make_upstream(self.root))

    def _config(self, **overrides) -> ForgeConfig:
        values = dict(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto", jailbreak="none",
        )
        values.update(overrides)
        return ForgeConfig(**values)

    def test_small_dense_only_offers_bridge(self) -> None:
        self._write()
        inspection = ModelInspector().inspect(self.root)
        names = {r.recipe for r in self._planner().available_recipes(inspection)}
        self.assertEqual(names, {"qwen3_5_gguf", "qwen3_5_gguf_standalone", "qwen3_5_gguf_q4_0", "qwen3_5_gguf_q5_0", "qwen3_5_gguf_mxfp4"})

    def test_multimodal_text_config_hidden_is_read(self) -> None:
        self._write(multimodal=True)
        inspection = ModelInspector().inspect(self.root)
        names = {r.recipe for r in self._planner().available_recipes(inspection)}
        self.assertEqual(names, {"qwen3_5_gguf", "qwen3_5_gguf_standalone", "qwen3_5_gguf_q4_0", "qwen3_5_gguf_q5_0", "qwen3_5_gguf_mxfp4"})

    def test_auto_selects_bridge_with_honest_diagnostic(self) -> None:
        self._write()
        plan = self._planner().plan(self._config())
        self.assertTrue(plan.executable)
        self.assertEqual(plan.backend, "qwen3_5_gguf")
        self.assertIn("auto-selected-gguf-bridge", {d.code for d in plan.diagnostics})

    def test_embedded_mtp_detected_and_components_extended(self) -> None:
        self._write(mtp=True)
        inspection = ModelInspector().inspect(self.root)
        mtp = inspection.speculative_components["mtp"]
        self.assertTrue(mtp["detected"])
        self.assertEqual(mtp["source"], "base_checkpoint")
        self.assertEqual(mtp["embedded_tensor_count"], 3)
        plan = self._planner().plan(self._config())
        self.assertTrue(plan.executable)
        step = next(s for s in plan.steps if s.name == "quantize+compile")
        self.assertEqual(step.details["components"], "text,mtp")
        spec = next(s for s in plan.steps if s.name == "speculative")
        self.assertEqual(spec.status, "planned")
        self.assertTrue(spec.details["mtp"])

    def test_no_mtp_keeps_text_components(self) -> None:
        self._write()
        plan = self._planner().plan(self._config())
        step = next(s for s in plan.steps if s.name == "quantize+compile")
        self.assertEqual(step.details["components"], "text")
        spec = next(s for s in plan.steps if s.name == "speculative")
        self.assertEqual(spec.status, "skipped")

    def test_explicit_native_target_refused_for_small_geometry(self) -> None:
        self._write()
        plan = self._planner().plan(self._config(target="qwen3_8_27b"))
        self.assertFalse(plan.executable)
        self.assertIn("target-incompatible",
                      {d.code for d in plan.diagnostics if d.severity.value == "error"})
        self.assertIsNone(plan.backend)


class LauncherSpecFlagsTest(unittest.TestCase):
    def _plan(self, status: str, details: dict) -> ForgePlan:
        config = ForgeConfig(input_path="x", output_dir="y")
        inspection = ModelInspection(path="x", input_kind=InputKind.SAFETENSORS)
        steps = [PlanStep("speculative", "import detected components", status, details)]
        return ForgePlan(config=config, inspection=inspection, steps=steps, diagnostics=[])

    def test_mtp_flags_do_not_force_lm_head_draft(self) -> None:
        plan = self._plan("planned", {"mtp": True, "dflash2": False})
        flags = BuildRunner._speculative_flags(plan)
        self.assertEqual(flags, ["--spec mtp", "--draft-tokens 3"])
        self.assertNotIn("--lm-head-draft", flags)

    def test_dflash2_wins_when_both_present(self) -> None:
        plan = self._plan("planned", {"mtp": True, "dflash2": True})
        self.assertEqual(BuildRunner._speculative_flags(plan),
                         ["--spec dflash2", "--draft-tokens 7"])

    def test_skipped_spec_yields_no_flags(self) -> None:
        self.assertEqual(BuildRunner._speculative_flags(self._plan("skipped", {})), [])

    def test_absent_spec_step_yields_no_flags(self) -> None:
        config = ForgeConfig(input_path="x", output_dir="y")
        inspection = ModelInspection(path="x", input_kind=InputKind.SAFETENSORS)
        plan = ForgePlan(config=config, inspection=inspection, steps=[], diagnostics=[])
        self.assertEqual(BuildRunner._speculative_flags(plan), [])


class ContextCapTest(unittest.TestCase):
    """The 0.8B GDN bridge serves native MTP only up to the engine's attention
    profile ceiling; the launcher/smoke/bench must never request more."""

    GDN = {"hidden_size": 1024, "linear_num_value_heads": 16}

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        write_config(self.root, architectures=["Qwen3_5ForCausalLM"], **self.GDN)
        write_safetensors(self.root / "model.safetensors",
                          {"model.layers.0.q_proj.weight": ("BF16", [8, 8])})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _planner(self) -> ForgePlanner:
        return ForgePlanner(inspector=ModelInspector(), upstream=make_upstream(self.root))

    def _config(self, **overrides) -> ForgeConfig:
        values = dict(input_path=str(self.root),
                      output_dir=str(Path(self.temp.name) / "build"),
                      target="auto", jailbreak="none", device="cpu")
        values.update(overrides)
        return ForgeConfig(**values)

    def test_ceiling_only_for_gdn_bridge_geometry(self) -> None:
        bridge = get_recipe("qwen3_5_gguf")
        self.assertEqual(serving_context_ceiling(bridge, self.GDN), GDN_BRIDGE_CONTEXT_CEILING)
        self.assertEqual(serving_context_ceiling(bridge, {"text_config": self.GDN}),
                         GDN_BRIDGE_CONTEXT_CEILING)
        # No GDN heads -> no known cap; native recipes are tuned for long context.
        self.assertIsNone(serving_context_ceiling(bridge, {"hidden_size": 1024}))
        self.assertIsNone(serving_context_ceiling(get_recipe("qwen3_8_27b"), self.GDN))

    def test_planner_warns_when_request_exceeds_ceiling(self) -> None:
        plan = self._planner().plan(self._config(runtime={"max_context": 32768}))
        codes = {d.code for d in plan.diagnostics}
        self.assertIn("context-capped-gdn-geometry", codes)
        warn = next(d for d in plan.diagnostics if d.code == "context-capped-gdn-geometry")
        self.assertEqual(warn.severity.value, "warning")

    def test_planner_silent_when_request_within_ceiling(self) -> None:
        plan = self._planner().plan(self._config(runtime={"max_context": 512}))
        self.assertNotIn("context-capped-gdn-geometry", {d.code for d in plan.diagnostics})

    def test_runner_effective_context_clamps(self) -> None:
        plan = self._planner().plan(self._config(runtime={"max_context": 32768}))
        runner = BuildRunner.__new__(BuildRunner)
        self.assertEqual(runner._effective_context(plan.config, plan), GDN_BRIDGE_CONTEXT_CEILING)
        within = self._planner().plan(self._config(runtime={"max_context": 512}))
        self.assertEqual(runner._effective_context(within.config, within), 512)

    def test_launcher_uses_capped_context(self) -> None:
        config = self._config(runtime={"max_context": 32768})
        plan = self._planner().plan(config)
        out = Path(self.temp.name) / "build"
        out.mkdir()
        manifest = BuildManifest.create(out, config, plan.inspection, plan.to_dict())
        artifact = out / "model.ninfer"
        artifact.write_bytes(b"ninfer")
        runner = BuildRunner.__new__(BuildRunner)
        runner.ninfer_dir = Path(self.temp.name) / "ninfer"
        runner._write_launcher(artifact, out, config, manifest, plan)
        body = (out / "scripts" / "run.bat").read_text(encoding="ascii")
        self.assertIn(f"--max-context {GDN_BRIDGE_CONTEXT_CEILING}", body)
        self.assertIn(f"--kv-capacity {GDN_BRIDGE_CONTEXT_CEILING}", body)
        self.assertNotIn("--max-context 32768", body)
        self.assertNotIn("--embedding-q4", body)
        # bf16 KV is the measured-faster MTP choice on this tiny-cache geometry.
        self.assertIn("--kv-dtype bf16", body)
        self.assertNotIn("--kv-dtype rk4v4", body)


class LauncherStorageFlagsTest(unittest.TestCase):
    """--embedding-q4 needs native Q8_G32 embedding storage; the GGUF Q8_0 bridge
    stores GGML Q8_0, so serve refuses the flag and it must be omitted there."""

    def _plan(self, backend: str, hidden: int) -> ForgePlan:
        config = ForgeConfig(input_path="x", output_dir="y")
        inspection = ModelInspection(path="x", input_kind=InputKind.SAFETENSORS,
                                     config={"hidden_size": hidden})
        steps = [PlanStep("speculative", "none detected", "skipped", {})]
        return ForgePlan(config=config, inspection=inspection, steps=steps,
                         diagnostics=[], backend=backend)

    def test_bridge_omits_embedding_q4(self) -> None:
        runner = BuildRunner.__new__(BuildRunner)
        flags = runner._launcher_flags(self._plan("qwen3_5_gguf", 1024))
        self.assertNotIn("--embedding-q4", flags)
        self.assertIn("--kv-dtype rk4v4", flags)
        self.assertIn("--gdn-state-fp16", flags)

    def test_native_keeps_embedding_q4(self) -> None:
        runner = BuildRunner.__new__(BuildRunner)
        flags = runner._launcher_flags(self._plan("qwen3_8_27b", 5120))
        self.assertIn("--embedding-q4", flags)


if __name__ == "__main__":
    unittest.main()