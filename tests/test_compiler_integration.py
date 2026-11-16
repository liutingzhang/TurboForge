"""End-to-end conversion on tiny weights.

Uses the same shape as upstream tests/convert/test_cli.py: a 1-layer Qwen3.5
source directory plus a custom recipe file, converted on CPU through the exact
`python -m tools.convert` command that NInferCompiler builds. Official recipes
refuse this geometry on purpose; this test proves the plumbing, not a recipe.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from turboforge.backends.compiler import NInferCompiler, probe_upstream


TINY_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "hidden_size": 128,
    "vocab_size": 8,
    "num_hidden_layers": 1,
    "max_position_embeddings": 128,
    "layer_types": ["full_attention"],
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "intermediate_size": 24,
    "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
}

TINY_RECIPE = """import torch
from tools.convert.sources.logical import array_source

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        value = torch.ones(parameter.shape, dtype=torch.bfloat16)
        recipe.assign(name, source=array_source(value, "turboforge-integration"))
"""


def upstream_ok() -> bool:
    status = probe_upstream()
    return status.found and status.torch_available and status.numpy_available


@unittest.skipUnless(upstream_ok(), "needs the ninfer-all tree and torch")
class CompilerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "config.json").write_text(json.dumps(TINY_CONFIG))
        for name, value in {
            "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
            "tokenizer_config.json": {},
            "generation_config.json": {},
        }.items():
            (self.source / name).write_text(json.dumps(value))
        # tools.convert collects frontend resources from the source directory;
        # a chat template is required, so give it one instead of letting the
        # loader fail on a missing file.
        (self.source / "chat_template.jinja").write_text("{{ messages }}")
        (self.root / "recipe.py").write_text(TINY_RECIPE)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_end_to_end_conversion_produces_readable_artifact(self) -> None:
        compiler = NInferCompiler()
        artifact = self.root / "tiny.ninfer"
        log_path = self.root / "convert.log"
        result = compiler.run(
            str(self.root / "recipe.py"),
            self.source, artifact,
            name="turboforge-integration",
            device="cpu",
            log_path=log_path,
            timeout=600,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(artifact.is_file(), msg="".join(result["tail"]))
        self.assertGreater(artifact.stat().st_size, 0)
        self.assertTrue(log_path.is_file())

        # The artifact must be readable by the official reader, not just exist.
        upstream = str(compiler.upstream.root)
        sys.path.insert(0, upstream)
        try:
            from tools.artifact.reader import Artifact
            with Artifact(artifact) as opened:
                self.assertEqual(
                    opened.directory.metadata.get("name"),
                    "turboforge-integration",
                )
                self.assertIn("text", opened.directory.components)
        finally:
            sys.path.remove(upstream)
            for module in [m for m in sys.modules if m.startswith("tools.")]:
                del sys.modules[module]

    def test_missing_recipe_source_is_rejected_before_running(self) -> None:
        from turboforge.backends.registry import get_recipe
        compiler = NInferCompiler()
        recipe = get_recipe("qwen3_8_27b_imatrix")
        with self.assertRaisesRegex(ValueError, "imatrix"):
            compiler.build_command(recipe, self.source, self.root / "x.ninfer")


if __name__ == "__main__":
    unittest.main()
