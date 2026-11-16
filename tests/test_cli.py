"""The CLI must forward --jailbreak-param into ForgeConfig.jailbreak_params.

Without this, abliteration cannot receive its required direction_file from the
command line, so the flag has to be parsed, de-duplicated and threaded through
_config_from_args, and exposed on both the build and plan subcommands.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest

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


if __name__ == "__main__":
    unittest.main()