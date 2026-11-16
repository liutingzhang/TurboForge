"""Web API contract tests.

These lock the JSON shapes the frontend depends on, so a backend change cannot
silently desynchronize the UI. Flask is optional; the module skips without it.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

try:
    import flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

from turboforge.backends.compiler import UpstreamStatus
from turboforge.backends.registry import ALL_RECIPE_NAMES, PINNED_UPSTREAM_REVISION
from turboforge.domain import JAILBREAK_CHOICES
from turboforge.inspector import ModelInspector
from turboforge.planner import ForgePlanner

from tests.helpers import write_config, write_safetensors


def _fake_upstream(root: Path) -> UpstreamStatus:
    return UpstreamStatus(
        root=root, found=True, revision=PINNED_UPSTREAM_REVISION,
        revision_matches_pin=True, torch_available=True, numpy_available=True,
        diagnostics=[],
    )


@unittest.skipUnless(_HAS_FLASK, "flask is not installed")
class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        from turboforge.api.server import create_app

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        write_config(self.root, architectures=["Qwen3_5ForCausalLM"], hidden_size=5120)
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        self.output = Path(self.temp.name) / "build"
        planner = ForgePlanner(
            inspector=ModelInspector(), upstream=_fake_upstream(self.root)
        )
        self.app = create_app(planner=planner)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **overrides) -> dict:
        values = {
            "input_path": str(self.root),
            "output_dir": str(self.output),
            "target": "auto",
            "jailbreak": "none",
        }
        values.update(overrides)
        return values

    # -- meta ----------------------------------------------------------------

    def test_meta_exposes_every_choice_from_the_backend(self) -> None:
        response = self.client.get("/api/meta")
        self.assertEqual(response.status_code, 200)
        meta = response.get_json()
        choices = meta["choices"]
        # Conversion-only contract: no optimization/speculative axes remain.
        self.assertNotIn("optimization", choices)
        self.assertNotIn("speculative", choices)
        self.assertEqual(
            [c["name"] for c in choices["jailbreak"]], list(JAILBREAK_CHOICES)
        )
        self.assertEqual([c["value"] for c in choices["device"]], ["cuda", "cpu"])
        self.assertIn("gguf", [c["value"] for c in choices["input_kind"]])
        # every choice carries a human label + description (no bare dropdowns)
        for item in choices["device"]:
            self.assertTrue(item["label"])
            self.assertTrue(item["description"])
        for item in choices["jailbreak"]:
            self.assertTrue(item["label"])
            self.assertTrue(item["description"])

    def test_meta_recipes_match_registry(self) -> None:
        meta = self.client.get("/api/meta").get_json()
        names = [r["recipe"] for r in meta["recipes"]]
        self.assertEqual(set(names), set(ALL_RECIPE_NAMES))
        for recipe in meta["recipes"]:
            for key in ("label", "nominal_bits", "typical_bpw", "extra_sources",
                        "needs_gpu", "input_kinds", "notes"):
                self.assertIn(key, recipe)

    def test_auto_build_requires_only_a_model_reference(self) -> None:
        from unittest.mock import patch
        manager = self.app.extensions['turboforge_build_manager']
        with patch.object(manager, '_auto_work') as work:
            response = self.client.post('/api/auto-build', json={'model': 'Org/Model'})
            self.assertEqual(response.status_code, 200)
            self._wait_finished(response.get_json()['build_id'])
            self.assertEqual(work.call_args.args[1], 'Org/Model')
            self.assertEqual(work.call_args.args[2], {})

    def test_auto_build_rejects_malformed_requests(self) -> None:
        for body in ({}, {'model': 42}, {'model': 'Org/Model', 'options': []}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/api/auto-build', json=body).status_code, 400)

    def test_meta_jailbreak_param_required_flags(self) -> None:
        meta = self.client.get("/api/meta").get_json()
        jailbreak = {j["name"]: j for j in meta["choices"]["jailbreak"]}
        self.assertEqual(jailbreak["heretic"]["params"], [])
        abl = {p["key"]: p for p in jailbreak["abliteration"]["params"]}
        self.assertEqual(abl["direction_file"]["required_when_env_missing"], "transformers")
        self.assertFalse(abl["direction_file"]["required"])
        # each processor declares its capability for the compatibility panel
        for name in JAILBREAK_CHOICES:
            self.assertIn("capability", jailbreak[name])
            self.assertIn("available", jailbreak[name]["capability"])

    def test_meta_env_and_defaults_present(self) -> None:
        meta = self.client.get("/api/meta").get_json()
        for key in ("torch", "transformers", "numpy", "safetensors"):
            self.assertIn(key, meta["env"])
        self.assertEqual(meta["defaults"]["target"], "auto")
        self.assertNotIn("optimization", meta["defaults"])
        self.assertNotIn("speculative", meta["defaults"])
        self.assertIn("max_context", meta["defaults"]["runtime"])
        self.assertIn("upstream", meta)
        self.assertIn("pinned_revision", meta)

    # -- inspect -------------------------------------------------------------

    def test_inspect_returns_inspection_and_legal_recipes(self) -> None:
        response = self.client.post("/api/inspect", json={"input_path": str(self.root)})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["inspection"]["architecture"], "Qwen3_5ForCausalLM")
        self.assertEqual(body["inspection"]["input_kind"], "safetensors")
        self.assertTrue(body["recipes"])

    def test_inspect_missing_path_is_400(self) -> None:
        response = self.client.post("/api/inspect", json={"input_path": str(self.root / "nope")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_inspect_without_input_path_is_400(self) -> None:
        response = self.client.post("/api/inspect", json={})
        self.assertEqual(response.status_code, 400)

    # -- plan ----------------------------------------------------------------

    def test_plan_executable_with_capabilities(self) -> None:
        response = self.client.post("/api/plan", json=self.config())
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["plan"]["executable"])
        self.assertEqual(body["plan"]["backend"], "qwen3_8_27b")
        self.assertTrue(body["capabilities"])
        for capability in body["capabilities"]:
            self.assertIn("name", capability)
            self.assertIn("available", capability)
        # steps carry their params under details (serialization contract)
        step = next(s for s in body["plan"]["steps"] if s["name"] == "quantize+compile")
        self.assertIn("details", step)
        self.assertIn("typical_bpw", step["details"])

    def test_plan_incompatible_target_is_not_executable(self) -> None:
        response = self.client.post("/api/plan", json=self.config(target="gemma4_gguf"))
        self.assertEqual(response.status_code, 200)
        plan = response.get_json()["plan"]
        self.assertFalse(plan["executable"])
        codes = {d["code"] for d in plan["diagnostics"]}
        self.assertIn("target-incompatible", codes)

    def test_plan_invalid_config_is_400(self) -> None:
        response = self.client.post("/api/plan", json=self.config(input_path=""))
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_plan_unknown_field_is_400(self) -> None:
        response = self.client.post("/api/plan", json=self.config(bogus_field=1))
        self.assertEqual(response.status_code, 400)

    # -- build ---------------------------------------------------------------

    def _wait_finished(self, build_id: str, timeout: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.client.get(f"/api/build/{build_id}").get_json()
            if body.get("finished"):
                return body
            time.sleep(0.05)
        self.fail("build did not finish in time")

    def test_build_blocked_plan_records_manifest(self) -> None:
        response = self.client.post(
            "/api/build", json=self.config(target="qwen3_8_27b_nvfp4")
        )
        self.assertEqual(response.status_code, 200)
        build_id = response.get_json()["build_id"]
        body = self._wait_finished(build_id)
        self.assertEqual(body["status"], "blocked")
        self.assertIsNotNone(body["manifest"])
        self.assertTrue(body["manifest"]["history"])

    def test_build_unknown_id_is_404(self) -> None:
        self.assertEqual(self.client.get("/api/build/does-not-exist").status_code, 404)
        self.assertEqual(self.client.get("/api/build/does-not-exist/log").status_code, 404)
        self.assertEqual(
            self.client.get("/api/build/does-not-exist/report?name=performance").status_code, 404
        )

    def test_report_invalid_name_is_400(self) -> None:
        response = self.client.post(
            "/api/build", json=self.config(target="qwen3_8_27b_nvfp4")
        )
        build_id = response.get_json()["build_id"]
        self._wait_finished(build_id)
        self.assertEqual(
            self.client.get(f"/api/build/{build_id}/report?name=bogus").status_code, 400
        )

    def test_report_absent_is_200_null_not_404(self) -> None:
        # A blocked/failed build never writes reports; absence must be a normal
        # 200 + null so the browser console stays free of expected-404 noise.
        response = self.client.post(
            "/api/build", json=self.config(target="qwen3_8_27b_nvfp4")
        )
        build_id = response.get_json()["build_id"]
        self._wait_finished(build_id)
        for name in ("performance", "quality"):
            resp = self.client.get(f"/api/build/{build_id}/report?name={name}")
            self.assertEqual(resp.status_code, 200)
            self.assertIsNone(resp.get_json()["report"])

    def test_build_invalid_config_is_400(self) -> None:
        response = self.client.post("/api/build", json=self.config(output_dir=""))
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
