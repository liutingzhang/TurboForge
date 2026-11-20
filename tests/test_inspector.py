from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from turboforge.domain import InputKind, Severity
from turboforge.inspector import ModelInspector
from turboforge.safetensors_io import SafetensorsFormatError, read_tensor_summaries

from tests.helpers import write_config, write_safetensors


class SafetensorsInspectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inspector = ModelInspector()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_sharded_bf16_model(self) -> None:
        write_config(self.root)
        write_safetensors(self.root / "model-00001-of-00002.safetensors", {
            "model.layers.0.q_proj.weight": ("BF16", [8, 8]),
        })
        write_safetensors(self.root / "model-00002-of-00002.safetensors", {
            "model.embed_tokens.weight": ("BF16", [16, 8]),
        })
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 384},
            "weight_map": {
                "model.layers.0.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            },
        }), encoding="utf-8")

        result = self.inspector.inspect(self.root)

        self.assertEqual(result.input_kind, InputKind.SAFETENSORS)
        self.assertEqual(result.weight_precision, "bf16")
        self.assertEqual(result.shard_count, 2)
        self.assertEqual(result.tensor_count, 2)
        self.assertEqual(result.tensor_bytes, 384)
        self.assertEqual(result.architecture, "Qwen2ForCausalLM")
        self.assertTrue(result.usable)

    def test_nvfp4_requires_config_scales_and_packed_weights(self) -> None:
        write_config(self.root, quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        })
        write_safetensors(self.root / "model.safetensors", {
            "model.layers.0.q_proj.weight": ("U8", [8, 4]),
            "model.layers.0.q_proj.weight_scale": ("F8_E4M3", [8, 2]),
            "model.layers.0.q_proj.weight_global_scale": ("F32", [1]),
        })

        result = self.inspector.inspect(self.root)

        self.assertEqual(result.input_kind, InputKind.NVFP4)
        self.assertTrue(result.quantization["nvfp4_layout_verified"])
        self.assertEqual(result.quantization["scale_tensor_count"], 2)
        self.assertFalse(result.quantization["dequantization_restores_original_precision"])
        self.assertIn("nvfp4-information-loss", {item.code for item in result.diagnostics})

    def test_nvfp4_filename_alone_does_not_reclassify_weights(self) -> None:
        write_config(self.root)
        path = self.root / "model-nvfp4.safetensors"
        write_safetensors(path, {"weight": ("BF16", [2, 2])})

        result = self.inspector.inspect(path)

        self.assertEqual(result.input_kind, InputKind.SAFETENSORS)
        self.assertEqual(result.weight_precision, "bf16")
        self.assertIn("nvfp4-name-is-not-evidence", {item.code for item in result.diagnostics})

    def test_nvfp4_metadata_without_layout_is_an_error(self) -> None:
        write_config(self.root, quantization_config={"quant_algo": "NVFP4"})
        write_safetensors(self.root / "model.safetensors", {"weight": ("BF16", [2, 2])})

        result = self.inspector.inspect(self.root)

        self.assertFalse(result.usable)
        diagnostic = next(item for item in result.diagnostics if item.code == "nvfp4-layout-unverified")
        self.assertEqual(diagnostic.severity, Severity.ERROR)

    def test_missing_shard_fails_instead_of_partially_inspecting(self) -> None:
        write_config(self.root)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "missing.safetensors"},
        }), encoding="utf-8")

        with self.assertRaisesRegex(SafetensorsFormatError, "missing"):
            self.inspector.inspect(self.root)

    def test_tensor_offsets_cannot_exceed_file(self) -> None:
        path = self.root / "broken.safetensors"
        header = json.dumps({"weight": {
            "dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 1000],
        }}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 32)

        with self.assertRaisesRegex(SafetensorsFormatError, "offsets"):
            read_tensor_summaries(path)

    def test_metadata_only_with_missing_base_shards_still_finds_actual_head(self) -> None:
        write_config(self.root)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"model.weight": "not-downloaded.safetensors"},
        }), encoding="utf-8")
        write_safetensors(self.root / "head.safetensors", {"mtp.fc.weight": ("BF16", [2, 2])})
        result = self.inspector.inspect(self.root, metadata_only=True)
        self.assertEqual(result.weight_precision, "metadata_only")
        self.assertTrue(result.speculative_components["mtp"]["detected"])
        self.assertFalse(result.speculative_components["mtp"]["compatibility_verified"])
        self.assertEqual({item["path"] for item in result.files}, {"head.safetensors"})

    def test_metadata_only_config_and_filename_are_not_head_evidence(self) -> None:
        write_config(self.root, num_mtp_layers=1, dflash2_config={"architectures": ["DFlash2DraftModel"]})
        write_safetensors(self.root / "mtp-dflash2.safetensors", {"weight": ("BF16", [2, 2])})
        result = self.inspector.inspect(self.root, metadata_only=True)
        self.assertFalse(result.speculative_components.get("mtp", {}).get("detected", False))
        self.assertFalse(result.speculative_components.get("dflash2", {}).get("detected", False))
        self.assertEqual(result.speculative_components["training"], "not_supported")
        self.assertIn("mtp_hint", result.speculative_components)

    def test_metadata_only_corrupt_present_head_is_explicitly_gated(self) -> None:
        write_config(self.root)
        (self.root / "head.safetensors").write_bytes(b"broken")
        result = self.inspector.inspect(self.root, metadata_only=True)
        self.assertFalse(result.usable)
        self.assertEqual(result.diagnostics[0].code, "speculative-inspection-unverified")
        self.assertEqual(result.diagnostics[0].severity, Severity.ERROR)
        self.assertEqual(result.config["model_type"], "qwen2")

    def test_metadata_only_reads_existing_indexed_subdirectory_head(self) -> None:
        write_config(self.root)
        nested = self.root / "shards"
        nested.mkdir()
        write_safetensors(nested / "head.safetensors", {"dflash2.fc.weight": ("BF16", [2, 2])})
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"model.weight": "missing.safetensors", "dflash2.fc.weight": "shards/head.safetensors"},
        }), encoding="utf-8")
        result = self.inspector.inspect(self.root, metadata_only=True)
        self.assertTrue(result.speculative_components["dflash2"]["detected"])
        self.assertFalse(result.speculative_components["dflash2"]["compatibility_verified"])
        self.assertEqual({item["path"] for item in result.files}, {"shards/head.safetensors"})

    def test_metadata_only_gates_indexed_heads_missing_from_actual_weights(self) -> None:
        write_config(self.root)
        write_safetensors(self.root / "model.safetensors", {"weight": ("BF16", [2, 2])})
        for component in ("mtp", "dflash2"):
            for shard in ("missing.safetensors", "model.safetensors"):
                with self.subTest(component=component, shard=shard):
                    (self.root / "model.safetensors.index.json").write_text(json.dumps({
                        "weight_map": {component + ".fc.weight": shard},
                    }), encoding="utf-8")
                    result = self.inspector.inspect(self.root, metadata_only=True)
                    self.assertFalse(result.usable)
                    self.assertIn("speculative-inspection-unverified", {d.code for d in result.diagnostics})
                    self.assertFalse(result.speculative_components.get(component, {}).get("detected", False))

    def test_metadata_only_does_not_follow_unsafe_index_references(self) -> None:
        write_config(self.root)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"mtp.fc.weight": "../external.safetensors"},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe shard reference"):
            self.inspector.inspect(self.root, metadata_only=True)

    def test_adapter_directory_is_not_a_base_model(self) -> None:
        (self.root / "adapter_config.json").write_text("{}", encoding="utf-8")
        write_safetensors(self.root / "adapter_model.safetensors", {"lora_A": ("BF16", [2, 2])})

        with self.assertRaisesRegex(SafetensorsFormatError, "LoRA adapter"):
            self.inspector.inspect(self.root)


if __name__ == "__main__":
    unittest.main()
