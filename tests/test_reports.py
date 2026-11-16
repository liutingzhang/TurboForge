from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from turboforge.benchmark import BenchResult
from turboforge.domain import ForgeConfig, ForgePlan, InputKind, ModelInspection
from turboforge.manifest import BuildManifest
from turboforge.reports import (
    MEASURED,
    NOT_MEASURED,
    REFUSED,
    measurement_summary,
    performance_report,
    quality_report,
    write_reports,
)
from turboforge.runner import BuildRunner

from tests.helpers import write_config


def bench_result(metrics: dict, returncode: int = 0) -> BenchResult:
    return BenchResult(
        executable="ninfer_bench.exe",
        command=["ninfer_bench.exe", "--weights", "m.ninfer"],
        returncode=returncode,
        seconds=12.3,
        stdout_tail="ok",
        stderr_tail="" if returncode == 0 else "missing executable: ninfer_bench.exe",
        metrics=metrics,
    )


def ppl_result(value: float | None, returncode: int = 0) -> BenchResult:
    metrics = {"perplexity": value} if value is not None else {}
    return BenchResult(
        executable="ninfer-perplexity.exe",
        command=["ninfer-perplexity.exe", "m.ninfer"],
        returncode=returncode,
        seconds=40.0,
        stdout_tail=f"perplexity = {value}",
        stderr_tail="",
        metrics=metrics,
    )


class PerformanceReportTest(unittest.TestCase):
    def test_measured_throughput_keeps_provenance(self) -> None:
        result = bench_result({
            "tg128_decode_output_tok_s": 180.4,
            "pp512_prefill_tok_s": 900.0,
            "tg128_spec_acceptance_length": 3.2,
        })
        report = performance_report({"ninfer_bench": result},
                                    runtime={"max_context": 32768})
        metrics = report["metrics"]
        self.assertEqual(metrics["decode_tok_s"]["status"], MEASURED)
        self.assertEqual(metrics["decode_tok_s"]["value"], {"tg128": 180.4})
        self.assertEqual(metrics["decode_tok_s"]["source"], "ninfer_bench.exe")
        self.assertEqual(metrics["prefill_tok_s"]["value"], {"pp512": 900.0})
        self.assertEqual(metrics["speculative"]["status"], MEASURED)
        # Context is a chosen parameter, never presented as a measurement.
        self.assertEqual(metrics["context_tokens"]["status"], "configured")
        # No VRAM/RAM/offload number is invented.
        for key in ("peak_vram_bytes", "peak_ram_bytes", "cpu_offload"):
            self.assertEqual(metrics[key]["status"], NOT_MEASURED)
            self.assertIsNone(metrics[key]["value"])

    def test_refused_tool_is_recorded_not_estimated(self) -> None:
        report = performance_report({"ninfer_bench": bench_result({}, returncode=-1)})
        decode = report["metrics"]["decode_tok_s"]
        self.assertEqual(decode["status"], REFUSED)
        self.assertIsNone(decode["value"])

    def test_no_benchmark_stage_is_not_measured(self) -> None:
        report = performance_report({})
        self.assertEqual(report["metrics"]["decode_tok_s"]["status"], NOT_MEASURED)
        self.assertEqual(report["metrics"]["ttft_seconds"]["status"], NOT_MEASURED)

    def test_ttft_only_from_a_real_probe(self) -> None:
        report = performance_report({}, probe={"ttft_seconds": 0.42})
        self.assertEqual(report["metrics"]["ttft_seconds"]["status"], MEASURED)
        self.assertEqual(report["metrics"]["ttft_seconds"]["value"], 0.42)


class QualityReportTest(unittest.TestCase):
    def test_unrun_evaluations_stay_not_measured(self) -> None:
        report = quality_report(jailbreak="abliteration")
        metrics = report["metrics"]
        self.assertEqual(metrics["perplexity"]["status"], NOT_MEASURED)
        for key in ("code", "math", "instruction_following", "long_context",
                    "output_consistency", "jailbreak_behavior_change"):
            self.assertEqual(metrics[key]["status"], NOT_MEASURED)
            self.assertIsNone(metrics[key]["value"])
        self.assertEqual(report["jailbreak_processor"], "abliteration")

    def test_perplexity_measured_and_loss_separate_from_jailbreak(self) -> None:
        report = quality_report(
            perplexity=ppl_result(8.5),
            baseline_perplexity=ppl_result(7.0),
            jailbreak="heretic",
        )
        metrics = report["metrics"]
        self.assertEqual(metrics["perplexity"]["status"], MEASURED)
        self.assertEqual(metrics["perplexity"]["value"], 8.5)
        self.assertEqual(metrics["quantization_loss"]["status"], MEASURED)
        self.assertAlmostEqual(metrics["quantization_loss"]["value"], 1.5)
        # A perplexity delta must never be reported as a jailbreak behavior change.
        self.assertEqual(metrics["jailbreak_behavior_change"]["status"], NOT_MEASURED)

    def test_loss_needs_both_sides(self) -> None:
        report = quality_report(perplexity=ppl_result(8.5))
        self.assertEqual(report["metrics"]["quantization_loss"]["status"], NOT_MEASURED)


class WriteReportsTest(unittest.TestCase):
    def test_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            perf = performance_report({"ninfer_bench": bench_result(
                {"tg128_decode_output_tok_s": 180.4})})
            qual = quality_report()
            paths = write_reports(Path(temp) / "reports", perf, qual)
            self.assertEqual(len(paths), 4)
            for path in paths.values():
                self.assertTrue(path.is_file())
            loaded = json.loads(paths["performance_json"].read_text(encoding="utf-8"))
            self.assertEqual(loaded["metrics"]["decode_tok_s"]["value"], {"tg128": 180.4})
            md = paths["quality_md"].read_text(encoding="utf-8")
            self.assertIn("not_measured", md)
            summary = measurement_summary(perf, qual)
            self.assertEqual(summary["performance"]["decode_tok_s"]["status"], MEASURED)
            self.assertEqual(summary["quality"]["perplexity"]["status"], NOT_MEASURED)


class RunnerReportWiringTest(unittest.TestCase):
    """The build output must always contain honest reports (§13)."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        write_config(self.model)
        self.out = self.root / "build"
        self.config = ForgeConfig(input_path=str(self.model), output_dir=str(self.out))
        self.inspection = ModelInspection(
            path=str(self.model), input_kind=InputKind.SAFETENSORS,
            architecture="Qwen2ForCausalLM",
        )
        self.manifest = BuildManifest.create(self.out, self.config, self.inspection, {})
        self.artifact = self.out / "model.ninfer"
        self.artifact.write_bytes(b"ninfer")
        # Avoid constructing the full planner/upstream; only ninfer_dir is used.
        self.runner = BuildRunner.__new__(BuildRunner)
        self.runner.ninfer_dir = self.root / "ninfer"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_reports_written_even_without_benchmark(self) -> None:
        plan = ForgePlan(config=self.config, inspection=self.inspection,
                         steps=[], diagnostics=[], serving={"max_context": 1024})
        self.runner._write_reports(
            self.artifact, self.out, self.config, self.manifest, plan
        )
        perf_path = self.out / "reports" / "performance.json"
        qual_path = self.out / "reports" / "quality.json"
        self.assertTrue(perf_path.is_file())
        self.assertTrue(qual_path.is_file())
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        self.assertEqual(perf["metrics"]["decode_tok_s"]["status"], NOT_MEASURED)
        self.assertEqual(perf["metrics"]["context_tokens"]["value"], 1024)
        measurements = self.manifest.data["measurements"]
        self.assertEqual(measurements["performance"]["decode_tok_s"]["status"], NOT_MEASURED)
        self.assertIn("report_performance_json", self.manifest.data["artifacts"])
        self.assertIn("report_quality_md", self.manifest.data["artifacts"])

    def test_missing_bench_exe_is_refused_not_faked(self) -> None:
        self.config.run_benchmark = True
        plan = ForgePlan(config=self.config, inspection=self.inspection,
                         steps=[], diagnostics=[])
        self.runner._write_reports(
            self.artifact, self.out, self.config, self.manifest, plan
        )
        perf = json.loads((self.out / "reports" / "performance.json").read_text(encoding="utf-8"))
        self.assertEqual(perf["metrics"]["decode_tok_s"]["status"], REFUSED)
        self.assertIsNone(perf["metrics"]["decode_tok_s"]["value"])


if __name__ == "__main__":
    unittest.main()
