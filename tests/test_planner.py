from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from turboforge.backends.compiler import UpstreamStatus
from turboforge.backends.registry import (
    ALL_RECIPE_NAMES,
    PINNED_UPSTREAM_REVISION,
    get_recipe,
)
from turboforge.domain import ForgeConfig
from turboforge.inspector import ModelInspector
from turboforge.planner import ForgePlanner

from tests.helpers import write_config, write_gguf, write_safetensors


def make_upstream(root: Path, *, torch: bool = True) -> UpstreamStatus:
    return UpstreamStatus(
        root=root, found=True, revision=PINNED_UPSTREAM_REVISION,
        revision_matches_pin=True, torch_available=torch, numpy_available=True,
        diagnostics=[],
    )


class PlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        write_config(self.root, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        self.planner = ForgePlanner(
            inspector=ModelInspector(), upstream=make_upstream(self.root)
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **overrides) -> ForgeConfig:
        values = dict(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto",
            jailbreak="none",
        )
        values.update(overrides)
        return ForgeConfig(**values)

    def errors(self, plan) -> dict[str, str]:
        return {d.code: d.message for d in plan.diagnostics if d.severity.value == "error"}

    def test_auto_selects_lowest_bpw_verified_recipe(self) -> None:
        plan = self.planner.plan(self.config())
        self.assertTrue(plan.executable)
        # ternary/imatrix need unsupplied sources; among runnable recipes the
        # lowest-bpw verified target is qwen3_8_27b.
        self.assertEqual(plan.backend, "qwen3_8_27b")
        self.assertIn("auto-selected-lowest-bpw", {d.code for d in plan.diagnostics})
        step = next(s for s in plan.steps if s.name == "quantize+compile")
        self.assertEqual(step.status, "planned")

    def test_explicit_incompatible_target_is_refused_with_alternatives(self) -> None:
        plan = self.planner.plan(self.config(target="gemma4_gguf"))
        self.assertFalse(plan.executable)
        self.assertIn("target-incompatible", self.errors(plan))
        diagnostic = next(
            d for d in plan.diagnostics if d.code == "target-incompatible"
        )
        self.assertTrue(diagnostic.alternatives)
        # Nothing was silently swapped: the build backend stays unset.
        self.assertIsNone(plan.backend)

    def test_bonsai2_without_ternary_source_is_blocked(self) -> None:
        plan = self.planner.plan(self.config(target="bonsai2_27b_ternary"))
        self.assertFalse(plan.executable)
        errors = self.errors(plan)
        self.assertIn("recipe-source-missing", errors)
        step = next(s for s in plan.steps if s.name == "quantize+compile")
        self.assertEqual(step.status, "blocked")

    def test_bonsai2_with_ternary_source_is_executable(self) -> None:
        gguf = Path(self.temp.name) / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
        gguf.write_bytes(b"\0" * 16)
        plan = self.planner.plan(self.config(
            target="bonsai2_27b_ternary",
            sources={"ternary": str(gguf)},
        ))
        self.assertTrue(plan.executable)
        self.assertEqual(plan.backend, "bonsai2_27b_ternary")

    def test_auto_prefers_bonsai2_when_ternary_supplied(self) -> None:
        gguf = Path(self.temp.name) / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
        gguf.write_bytes(b"\0" * 16)
        plan = self.planner.plan(self.config(sources={"ternary": str(gguf)}))
        self.assertTrue(plan.executable)
        self.assertEqual(plan.backend, "bonsai2_27b_ternary")
        self.assertIn("auto-selected-bonsai2", {d.code for d in plan.diagnostics})

    def test_auto_prefers_gguf_import_when_gguf_supplied(self) -> None:
        gguf = Path(self.temp.name) / "model-Q4_K_M.gguf"
        write_gguf(gguf)
        plan = self.planner.plan(self.config(sources={"gguf": str(gguf)}))
        self.assertTrue(plan.executable)
        self.assertEqual(plan.backend, "qwen3_8_27b_gguf")
        self.assertIn("auto-selected-gguf", {d.code for d in plan.diagnostics})

    def test_jailbreak_with_imported_weights_is_refused(self) -> None:
        gguf = Path(self.temp.name) / "model-Q4_K_M.gguf"
        write_gguf(gguf)
        plan = self.planner.plan(self.config(
            jailbreak="abliteration",
            jailbreak_params={"direction_file": str(self.root / "dir.npy")},
            sources={"gguf": str(gguf)},
        ))
        self.assertFalse(plan.executable)
        self.assertIn("jailbreak-not-applied-to-imported-weights", self.errors(plan))

    def test_gguf_standalone_input_is_refused_with_alternatives(self) -> None:
        gguf = Path(self.temp.name) / "model-Q4_K_M.gguf"
        write_gguf(gguf)
        plan = self.planner.plan(self.config(input_path=str(gguf)))
        self.assertFalse(plan.executable)
        diagnostic = next(
            d for d in plan.diagnostics if d.code == "gguf-secondary-source"
        )
        self.assertTrue(diagnostic.alternatives)

    def test_speculative_step_is_automatic_and_never_trains(self) -> None:
        plan = self.planner.plan(self.config())
        step = next(s for s in plan.steps if s.name == "speculative")
        self.assertEqual(step.status, "skipped")
        self.assertEqual(step.implementation, "none detected")

    def test_heretic_unavailable_is_explicit_error(self) -> None:
        plan = self.planner.plan(self.config(jailbreak="heretic"))
        # heretic not installed in this environment -> explicit error, never a
        # silent fallback to none.
        self.assertIn("heretic-unavailable", self.errors(plan))

    def test_removed_jailbreak_choices_are_rejected(self) -> None:
        for removed in ("existing", "lora_merge"):
            with self.assertRaises(ValueError):
                self.planner.plan(self.config(jailbreak=removed))

    def test_missing_extra_source_file_is_an_error(self) -> None:
        plan = self.planner.plan(self.config(
            target="qwen3_8_27b_imatrix",
            sources={"imatrix": str(self.root / "absent.imatrix")},
        ))
        self.assertIn("source-missing", self.errors(plan))

    def test_registry_covers_all_official_recipes(self) -> None:
        expected = {
            "qwen3_8_flash_next", "qwen3_6_27b", "qwen3_6_27b_nvfp4",
            "qwen3_8_27b", "qwen3_8_27b_q6", "qwen3_8_27b_imatrix",
            "qwen3_8_27b_nvfp4", "qwen3_8_27b_nvfp4_nvidia",
            "qwen3_8_27b_nvfp4_orcarouter", "qwen3_6_35b_a3b",
            "qwen3_6_35b_a3b_nvfp4", "bonsai2_27b_ternary",
            "qwen3_8_27b_gguf", "qwen3_8_flash_next_gguf",
            "qwen3_8_flash_next_gsq_q2", "gemma4_gguf", "flint_gguf",
            "deepseek_v41_gguf", "qwen3_5_gguf_standalone", "qwen3_5_gguf", "qwen3_5_gguf_q4_0", "qwen3_5_gguf_q5_0", "qwen3_5_gguf_mxfp4",
        }
        self.assertEqual(set(ALL_RECIPE_NAMES), expected)

    def test_unknown_recipe_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_recipe("qwen2_7b_q4")

    def test_removed_preserve_target_is_refused(self) -> None:
        plan = self.planner.plan(self.config(target="preserve"))
        self.assertIn("target-incompatible", self.errors(plan))


class Nvfp4PlannerTest(unittest.TestCase):
    """Route 2: NVFP4 input must not be treated as restorable precision."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "nvfp4"
        self.root.mkdir()
        write_config(self.root,
                     architectures=["Qwen3_5ForCausalLM"],
                     quantization_config={"quant_method": "modelopt",
                                          "quant_algo": "NVFP4"})
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("U8", [8, 4]),
            "model.layers.0.q_proj.weight_scale": ("F8_E4M3", [8, 2]),
            "model.layers.0.q_proj.weight_global_scale": ("F32", [1]),
        })
        self.planner = ForgePlanner(
            inspector=ModelInspector(), upstream=make_upstream(self.root)
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_nvfp4_as_input_is_refused_as_secondary_source(self) -> None:
        quantized = Path(self.temp.name) / "quantized"
        quantized.mkdir()
        config = ForgeConfig(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto",
            sources={"quantized": str(quantized)},
        )
        plan = self.planner.plan(config)
        self.assertFalse(plan.executable)
        self.assertIn("nvfp4-secondary-source",
                      {d.code for d in plan.diagnostics})

    def test_auto_picks_verbatim_nvfp4_import_with_bf16_model(self) -> None:
        base = Path(self.temp.name) / "base"
        base.mkdir()
        write_config(base, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
        write_safetensors(base / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        config = ForgeConfig(
            input_path=str(base),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto",
            sources={"quantized": str(self.root)},
        )
        plan = self.planner.plan(config)
        self.assertTrue(plan.executable)
        self.assertIn("nvfp4", plan.backend)
        self.assertIn("auto-selected-nvfp4", {d.code for d in plan.diagnostics})

    def test_weight_modification_on_nvfp4_requires_original_precision(self) -> None:
        config = ForgeConfig(
            input_path=str(self.root),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto",
            jailbreak="heretic",
        )
        plan = self.planner.plan(config)
        codes = {d.code for d in plan.diagnostics}
        self.assertIn("heretic-needs-high-precision", codes)
        self.assertFalse(plan.executable)

    def test_nvfp4_with_jailbreak_none_imports_verbatim(self) -> None:
        base = Path(self.temp.name) / "base"
        base.mkdir()
        write_config(base, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
        write_safetensors(base / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        config = ForgeConfig(
            input_path=str(base),
            output_dir=str(Path(self.temp.name) / "build"),
            target="auto",
            jailbreak="none",
            sources={"quantized": str(self.root)},
        )
        plan = self.planner.plan(config)
        self.assertTrue(plan.executable)
        self.assertIn("nvfp4", plan.backend)


if __name__ == "__main__":
    unittest.main()
