from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from turboforge.domain import InputKind, ModelInspection, Severity
from turboforge.jailbreak import weight_edit
from turboforge.jailbreak.processors import get_processor


def _inspection(path: Path, kind: InputKind = InputKind.SAFETENSORS) -> ModelInspection:
    return ModelInspection(path=str(path), input_kind=kind, architecture="qwen2",
                           weight_precision="bf16")


def _make_model(root: Path, hidden: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
         "hidden_size": hidden, "num_hidden_layers": 2}), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    g = torch.Generator().manual_seed(0)
    save_file({
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden, generator=g),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, hidden, generator=g),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(hidden, hidden, generator=g),
    }, str(root / "model-00001-of-00002.safetensors"))
    save_file({
        "model.embed_tokens.weight": torch.randn(8, hidden, generator=g),
    }, str(root / "model-00002-of-00002.safetensors"))
    return root


class TestAbliterationMath(unittest.TestCase):
    def test_refusal_direction_normalized(self):
        h = torch.tensor([3.0, 0.0, 0.0])
        n = torch.tensor([1.0, 0.0, 0.0])
        r = weight_edit.refusal_direction(h, n)
        self.assertAlmostEqual(float(r.norm()), 1.0, places=5)
        self.assertGreater(float(r[0]), 0.99)

    def test_identical_means_rejected(self):
        with self.assertRaises(ValueError):
            weight_edit.refusal_direction(torch.ones(4), torch.ones(4))

    def test_output_orthogonalization_removes_direction(self):
        r = torch.tensor([1.0, 0.0, 0.0, 0.0])
        w = torch.randn(4, 6)
        edited = weight_edit.orthogonalize_output(w, r)
        # The component of every column along r must vanish.
        self.assertTrue(torch.allclose(r @ edited, torch.zeros(6), atol=1e-5))
        # The orthogonal complement is preserved.
        self.assertTrue(torch.allclose(edited[1:], w[1:], atol=1e-5))

    def test_embedding_orthogonalization_removes_direction(self):
        r = torch.tensor([0.0, 1.0, 0.0, 0.0])
        e = torch.randn(8, 4)
        edited = weight_edit.orthogonalize_embedding(e, r)
        self.assertTrue(torch.allclose(edited @ r, torch.zeros(8), atol=1e-5))


class TestAbliterationProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.src = _make_model(self.base / "model")
        self.out = self.base / "edited"
        self.proc = get_processor("abliteration")

    def tearDown(self):
        self.tmp.cleanup()

    def test_available_and_capability(self):
        self.assertTrue(self.proc.capability.available)
        self.assertTrue(self.proc.capability.supports_chunking)
        self.assertFalse(self.proc.capability.supports_nvfp4)

    def test_nvfp4_rejected(self):
        diags = self.proc.check(_inspection(self.src, InputKind.NVFP4))
        self.assertTrue(any(d.severity == Severity.ERROR for d in diags))

    def test_apply_with_direction_file(self):
        import numpy as np
        direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dfile = self.base / "direction.npy"
        np.save(dfile, direction)
        before = load_file(str(self.src / "model-00001-of-00002.safetensors"))
        result = self.proc.apply(_inspection(self.src), self.out, direction_file=str(dfile))
        self.assertEqual(Path(result), self.out)
        after = load_file(str(self.out / "model-00001-of-00002.safetensors"))
        r = torch.from_numpy(direction)
        key = "model.layers.0.self_attn.o_proj.weight"
        self.assertTrue(torch.allclose(r @ after[key], torch.zeros(4), atol=1e-4))
        # gate_proj is not a residual-writing matrix and must be untouched.
        self.assertTrue(torch.equal(after["model.layers.0.mlp.gate_proj.weight"],
                                    before["model.layers.0.mlp.gate_proj.weight"]))
        # Source is never modified.
        src_now = load_file(str(self.src / "model-00001-of-00002.safetensors"))
        self.assertTrue(torch.equal(src_now[key], before[key]))
        # Resources copied and a record written.
        self.assertTrue((self.out / "config.json").is_file())
        record = json.loads((self.out / "abliteration.json").read_text())
        self.assertIn(key, record["modified_tensors"])

    def test_nonunit_direction_is_normalized(self):
        dfile = self.base / "direction.pt"
        torch.save(torch.tensor([2.0, 0.0, 0.0, 0.0]), dfile)
        self.proc.apply(_inspection(self.src), self.out, direction_file=str(dfile))
        after = load_file(str(self.out / "model-00001-of-00002.safetensors"))
        self.assertTrue(torch.allclose(after["model.layers.0.self_attn.o_proj.weight"][0], torch.zeros(4)))
        record = json.loads((self.out / "abliteration.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(record["direction_norm"], 1.0)

    def test_invalid_direction_rejected_before_writing(self):
        for vector in (torch.zeros(4), torch.tensor([float("nan"), 0, 0, 0]),
                       torch.tensor([float("inf"), 0, 0, 0]), torch.ones(4, dtype=torch.complex64)):
            with self.subTest(vector=vector):
                dfile = self.base / "direction.pt"
                torch.save(vector, dfile)
                with self.assertRaises(ValueError):
                    self.proc.apply(_inspection(self.src), self.out, direction_file=str(dfile))
                self.assertFalse(self.out.exists())

    def test_all_model_resources_are_copied(self):
        from turboforge.domain import MODEL_RESOURCE_FILES
        for name in MODEL_RESOURCE_FILES:
            if not (self.src / name).exists():
                (self.src / name).write_text("{}", encoding="utf-8")
        weight_edit.copy_resources(self.src, self.out)
        for name in MODEL_RESOURCE_FILES:
            self.assertEqual((self.src / name).read_bytes(), (self.out / name).read_bytes())

    def test_apply_without_direction_raises(self):
        with self.assertRaises(RuntimeError):
            self.proc.apply(_inspection(self.src), self.out)


class TestMultimodalGeometry(unittest.TestCase):
    """Qwen3.5 keeps the text tower's geometry under text_config, not top level."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {"hidden_size": 1024, "num_hidden_layers": 24},
            "vision_config": {"hidden_size": 768},
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_hidden_size_reads_text_config(self):
        self.assertEqual(weight_edit.hidden_size(self.root), 1024)

    def test_hidden_size_prefers_top_level(self):
        (self.root / "config.json").write_text(json.dumps({
            "hidden_size": 2048, "text_config": {"hidden_size": 1024},
        }), encoding="utf-8")
        self.assertEqual(weight_edit.hidden_size(self.root), 2048)

    def test_num_text_layers_reads_text_config(self):
        from turboforge.jailbreak.processors import AbliterationProcessor

        class _Cfg:
            num_hidden_layers = 99
            text_config = type("T", (), {"num_hidden_layers": 24})()

        self.assertEqual(AbliterationProcessor._num_text_layers(_Cfg()), 24)

    def test_num_text_layers_falls_back_to_top_level(self):
        from turboforge.jailbreak.processors import AbliterationProcessor

        class _Cfg:
            num_hidden_layers = 28

        self.assertEqual(AbliterationProcessor._num_text_layers(_Cfg()), 28)


class TestHereticProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.proc = get_processor("heretic")

    def tearDown(self):
        self.tmp.cleanup()

    def test_availability_is_reported_honestly(self):
        diags = self.proc.check(_inspection(self.base))
        codes = {d.code for d in diags}
        if self.proc.capability.available:
            self.assertNotIn("heretic-unavailable", codes)
        else:
            self.assertIn("heretic-unavailable", codes)
            self.assertTrue(any(
                d.severity == Severity.ERROR for d in diags
                if d.code == "heretic-unavailable"
            ))

    def test_nvfp4_and_gguf_rejected(self):
        for kind in (InputKind.NVFP4, InputKind.GGUF):
            diags = self.proc.check(_inspection(self.base, kind))
            self.assertTrue(any(d.severity == Severity.ERROR for d in diags))

    @unittest.skipIf(get_processor("heretic").capability.available,
                     "heretic is installed; the refusal path is not reachable")
    def test_apply_without_heretic_raises(self):
        with self.assertRaises(RuntimeError):
            self.proc.apply(_inspection(self.base), self.base / "out")

    def test_saved_detection_requires_config_and_weights(self):
        out = self.base / "out"
        out.mkdir()
        self.assertFalse(self.proc._saved(out))
        (out / "config.json").write_text("{}", encoding="utf-8")
        self.assertFalse(self.proc._saved(out))
        (out / "model.safetensors").write_bytes(b"\0")
        self.assertFalse(self.proc._saved(out), "partially written weights must not start conversion")
        (out / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
        save_file({"model.layers.0.weight": torch.ones(2, 2)}, out / "model.safetensors")
        self.assertTrue(self.proc._saved(out))


if __name__ == "__main__":
    unittest.main()
