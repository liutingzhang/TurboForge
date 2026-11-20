"""The CLI must forward --jailbreak-param into ForgeConfig.jailbreak_params.

Without this, abliteration cannot receive its required direction_file from the
command line, so the flag has to be parsed, de-duplicated and threaded through
_config_from_args, and exposed on both the build and plan subcommands.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from turboforge import cli
from turboforge.domain import ForgeConfig

_BASE = ["--input", "D:/models/Qwen3.8-27B", "--output", "D:/builds/out"]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli._add_common_build_args(parser)
    return parser.parse_args(argv)


class CliJailbreakParamTest(unittest.TestCase):
    def test_abliteration_direction_file_flows_into_config(self):
        args = _parse(_BASE + [
            "--jailbreak", "abliteration",
            "--jailbreak-param", "direction_file=D:/vectors/qwen38.npy",
        ])
        config = cli._config_from_args(args)
        self.assertIsInstance(config, ForgeConfig)
        self.assertEqual(config.jailbreak, "abliteration")
        self.assertEqual(config.jailbreak_params, {"direction_file": "D:/vectors/qwen38.npy"})

    def test_abliteration_advanced_params_flow_into_config(self):
        args = _parse(_BASE + [
            "--jailbreak", "abliteration",
            "--jailbreak-param", "layer=32",
            "--jailbreak-param", "harmful_prompts=a\nb",
        ])
        config = cli._config_from_args(args)
        self.assertEqual(config.jailbreak, "abliteration")
        self.assertEqual(config.jailbreak_params, {"layer": "32", "harmful_prompts": "a\nb"})

    def test_no_param_yields_empty_dict(self):
        config = cli._config_from_args(_parse(_BASE + ["--jailbreak", "none"]))
        self.assertEqual(config.jailbreak_params, {})

    def test_duplicate_key_is_rejected(self):
        args = _parse(_BASE + [
            "--jailbreak", "abliteration",
            "--jailbreak-param", "direction_file=a.npy",
            "--jailbreak-param", "direction_file=b.npy",
        ])
        with self.assertRaises(SystemExit) as ctx:
            cli._config_from_args(args)
        self.assertIn("--jailbreak-param", str(ctx.exception))

    def test_malformed_entries_are_rejected(self):
        for bad in ("direction_file", "=D:/x.npy", "direction_file="):
            with self.subTest(entry=bad):
                args = _parse(_BASE + ["--jailbreak", "abliteration", "--jailbreak-param", bad])
                with self.assertRaises(SystemExit):
                    cli._config_from_args(args)

    def test_value_may_contain_equals_sign(self):
        args = _parse(_BASE + [
            "--jailbreak", "abliteration",
            "--jailbreak-param", "direction_file=D:/a=b",
        ])
        config = cli._config_from_args(args)
        self.assertEqual(config.jailbreak_params, {"direction_file": "D:/a=b"})

    def test_sources_still_parsed_alongside_params(self):
        args = _parse(_BASE + [
            "--jailbreak", "abliteration",
            "--jailbreak-param", "direction_file=D:/vectors/qwen38.npy",
            "--source", "gguf=D:/models/q4.gguf",
        ])
        config = cli._config_from_args(args)
        self.assertEqual(config.jailbreak_params, {"direction_file": "D:/vectors/qwen38.npy"})
        self.assertEqual(config.sources, {"gguf": "D:/models/q4.gguf"})

    def test_removed_optimization_and_speculative_flags_are_rejected(self):
        for flag in ("--optimization", "--speculative"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    _parse(_BASE + [flag, "speed"])

    def test_flag_is_exposed_on_build_and_plan(self):
        for command in ("build", "plan"):
            with self.subTest(command=command):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    with self.assertRaises(SystemExit) as ctx:
                        cli.main([command, "--help"])
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("--jailbreak-param", buf.getvalue())


class ModuleEntrypointTest(unittest.TestCase):
    def run_cli(self, *args):
        from turboforge.assistant.config import child_environment

        env = child_environment()
        source = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run([sys.executable, "-m", "turboforge", *args],
                cwd=directory, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_module_help_works_outside_repository(self):
        for args in (("--help",), ("build", "--help"), ("plan", "--help"), ("webui", "--help")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage: turboforge", result.stdout)

    def test_module_dispatches_recipes_with_json_output(self):
        result = self.run_cli("recipes", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual({item["recipe"] for item in data}, {recipe.recipe for recipe in cli.RECIPES})

    def test_module_preserves_parser_error_exit_code(self):
        result = self.run_cli("not-a-subcommand")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_module_preserves_command_error_exit_code(self):
        result = self.run_cli("inspect", "missing-model.safetensors", "--json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)

    def test_module_propagates_cli_main_return_value(self):
        with patch.object(cli, "main", return_value=7) as main:
            with self.assertRaises(SystemExit) as caught:
                runpy.run_module("turboforge", run_name="__main__")
        self.assertEqual(caught.exception.code, 7)
        main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()