"""GGUF header adapter: auto-recognition without torch, honest about limits.

The adapter parses only the metadata KV block and tensor directory of a GGUF
v2/v3 file. These tests build synthetic headers (no real weights) and lock the
recognition contract the UI and planner rely on.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from turboforge.adapters.gguf import GGUFAdapter, GGUFFormatError
from turboforge.domain import InputKind
from turboforge.inspector import ModelInspector

from tests.helpers import write_gguf


class GgufAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.adapter = GGUFAdapter()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_matches_file_and_directory(self) -> None:
        gguf = self.base / "model-Q4_K_M.gguf"
        write_gguf(gguf)
        self.assertTrue(self.adapter.matches(gguf))
        self.assertTrue(self.adapter.matches(self.base))
        self.assertFalse(self.adapter.matches(self.base / "absent.gguf"))

    def test_recognizes_architecture_quant_and_tensors(self) -> None:
        gguf = self.base / "model-Q4_K_M.gguf"
        write_gguf(gguf, architecture="qwen3", file_type=15, tensors={
            "blk.0.attn_q.weight": [8, 8],
            "blk.0.attn_k.weight": [8, 8],
            "token_embd.weight": [16, 8],
        })
        inspection = self.adapter.inspect(gguf)
        self.assertEqual(inspection.input_kind, InputKind.GGUF)
        self.assertEqual(inspection.architecture, "qwen3")
        self.assertEqual(inspection.weight_precision, "Q4_K_M")
        self.assertEqual(inspection.tensor_count, 3)
        self.assertEqual(inspection.quantization["nominal_bits"], 4.0)
        self.assertEqual(inspection.shard_count, 1)
        self.assertTrue(inspection.usable)
        self.assertIn("gguf-recognized", {d.code for d in inspection.diagnostics})
        # element counts are not invented for block formats
        self.assertEqual(inspection.element_count, 0)

    def test_ternary_label_recognized(self) -> None:
        gguf = self.base / "model-PQ2_0.gguf"
        write_gguf(gguf, file_type=33)
        inspection = self.adapter.inspect(gguf)
        self.assertEqual(inspection.weight_precision, "TQ2_0")
        self.assertEqual(inspection.quantization["nominal_bits"], 2.0)

    def test_mtp_and_dflash_detected_from_tensor_names(self) -> None:
        gguf = self.base / "model-mtp.gguf"
        write_gguf(gguf, tensors={
            "blk.0.attn_q.weight": [8, 8],
            "mtp.layers.0.eh_proj.weight": [8, 8],
            "dflash.conv.weight": [8, 8],
        })
        inspection = self.adapter.inspect(gguf)
        self.assertTrue(inspection.speculative_components["mtp"]["detected"])
        self.assertNotIn("dflash2",inspection.speculative_components)
        self.assertIn("dflash_hint",inspection.speculative_components)
        self.assertEqual(inspection.speculative_components["training"], "not_supported")

    def test_bad_magic_rejected(self) -> None:
        bad = self.base / "bad.gguf"
        bad.write_bytes(b"NOPE" + b"\0" * 64)
        with self.assertRaisesRegex(GGUFFormatError, "magic"):
            self.adapter.inspect(bad)

    def test_unsupported_version_rejected(self) -> None:
        bad = self.base / "bad.gguf"
        bad.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\0" * 64)
        with self.assertRaisesRegex(GGUFFormatError, "version"):
            self.adapter.inspect(bad)

    def test_truncated_header_rejected(self) -> None:
        gguf = self.base / "model.gguf"
        write_gguf(gguf)
        truncated = self.base / "truncated.gguf"
        truncated.write_bytes(gguf.read_bytes()[:12])
        with self.assertRaisesRegex(GGUFFormatError, "truncated"):
            self.adapter.inspect(truncated)

    def test_inspector_routes_gguf_to_the_adapter(self) -> None:
        gguf = self.base / "model-Q4_K_M.gguf"
        write_gguf(gguf)
        inspection = ModelInspector().inspect(gguf)
        self.assertEqual(inspection.input_kind, InputKind.GGUF)
        self.assertEqual(inspection.weight_precision, "Q4_K_M")


if __name__ == "__main__":
    unittest.main()
