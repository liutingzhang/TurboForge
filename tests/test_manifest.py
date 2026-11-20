from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from turboforge.domain import ForgeConfig, InputKind, ModelInspection
from turboforge.manifest import BuildManifest, file_sha256, source_content_identity


def inspection_at(path: Path) -> ModelInspection:
    return ModelInspection(
        path=str(path), input_kind=InputKind.SAFETENSORS,
        architecture="Qwen3_5ForCausalLM", weight_precision="bf16",
        files=[{"path": "model.safetensors", "bytes": 128}],
    )


class ManifestGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "model"
        self.source.mkdir()
        (self.source / "model.safetensors").write_bytes(b"\0" * 128)
        self.config = ForgeConfig(
            input_path=str(self.source),
            output_dir=str(self.base / "build"),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_output_inside_source_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            BuildManifest.create(self.source / "out", self.config,
                                 inspection_at(self.source), {})

    def test_output_equal_to_source_file_parent_is_refused(self) -> None:
        single = self.base / "model.safetensors"
        single.write_bytes(b"\0")
        config = ForgeConfig(input_path=str(single), output_dir=str(self.base))
        with self.assertRaises(ValueError):
            BuildManifest.create(self.base, config, inspection_at(single), {})

    def test_nonempty_output_is_refused(self) -> None:
        out = self.base / "build"
        out.mkdir()
        (out / "old.ninfer").write_bytes(b"\0")
        with self.assertRaises(FileExistsError):
            BuildManifest.create(out, self.config,
                                 inspection_at(self.source), {})

    def test_manifest_records_history_and_artifacts(self) -> None:
        manifest = BuildManifest.create(self.base / "build", self.config,
                                        inspection_at(self.source), {})
        self.assertEqual(manifest.data["status"], "planned")
        artifact = manifest.output_dir / "model.ninfer"
        artifact.write_bytes(b"\1" * 64)
        manifest.add_artifact("ninfer_artifact", artifact, digest=True)
        manifest.transition("validated", "smoke ok")
        saved = json.loads(manifest.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "validated")
        self.assertIn("sha256", saved["artifacts"]["ninfer_artifact"])
        self.assertEqual(saved["artifacts"]["ninfer_artifact"]["bytes"], 64)

    def test_additional_sources_record_file_and_directory_content(self) -> None:
        gguf = self.base / "weights.gguf"
        gguf.write_bytes(b"gguf content")
        quantized = self.base / "quantized"
        nested = quantized / "shards"
        nested.mkdir(parents=True)
        (quantized / "config.json").write_text("{}", encoding="utf-8")
        (nested / "weights.bin").write_bytes(b"packed weights")
        self.config.sources = {"gguf": str(gguf), "quantized": str(quantized)}
        manifest = BuildManifest.create(self.base / "build", self.config, inspection_at(self.source), {})
        saved = json.loads(manifest.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["additional_source_identity"], {
            "gguf": {"type": "file", "sha256": file_sha256(gguf)},
            "quantized": {"type": "directory", "files": {
                "config.json": file_sha256(quantized / "config.json"),
                "shards/weights.bin": file_sha256(nested / "weights.bin"),
            }},
        })

    def test_missing_additional_source_is_not_hashed(self) -> None:
        self.config.sources = {"gguf": str(self.base / "absent.gguf")}
        with patch("turboforge.manifest.file_sha256", side_effect=AssertionError("must not hash missing input")):
            manifest = BuildManifest.create(self.base / "build", self.config, inspection_at(self.source), {})
        self.assertEqual(manifest.data["additional_source_identity"], {"gguf": {"type": "missing"}})
        manifest.transition("blocked", "source-missing")
        self.assertEqual(json.loads(manifest.path.read_text(encoding="utf-8"))["status"], "blocked")

    def test_source_identity_changes_with_directory_inventory(self) -> None:
        source = self.base / "quantized"
        source.mkdir()
        weight = source / "weight"
        weight.write_bytes(b"before")
        original = source_content_identity(source)
        weight.write_bytes(b"after!")
        self.assertNotEqual(original, source_content_identity(source))
        weight.write_bytes(b"before")
        self.assertEqual(original, source_content_identity(source))
        (source / "scale").write_bytes(b"scale")
        self.assertNotEqual(original, source_content_identity(source))
        (source / "scale").unlink()
        weight.rename(source / "renamed")
        self.assertNotEqual(original, source_content_identity(source))

    def test_no_additional_sources_have_explicit_empty_identity(self) -> None:
        manifest = BuildManifest.create(self.base / "build", self.config, inspection_at(self.source), {})
        self.assertEqual(manifest.data["additional_source_identity"], {})

    def test_artifact_outside_output_is_refused(self) -> None:
        manifest = BuildManifest.create(self.base / "build", self.config,
                                        inspection_at(self.source), {})
        with self.assertRaises(ValueError):
            manifest.add_artifact("evil", self.source / "model.safetensors")


if __name__ == "__main__":
    unittest.main()
