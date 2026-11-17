"""Report Generator: honest performance and quality reports for a build.

Every metric carries an explicit status:

  measured      a tool actually ran on this machine and produced the number;
                the tool name and command line are kept for reproducibility.
  configured    a serving parameter the user chose (not a measurement).
  refused       the tool was invoked but could not run (missing exe, timeout).
  not_measured  no tool produced this number; the reason is recorded.

Nothing here is estimated, simulated, or borrowed from another GPU. Quantization
loss (a perplexity delta) and jailbreak behavior change (a paired behavioral
eval) are reported on separate axes so one is never mistaken for the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark import BenchResult
from .manifest import utc_now, write_json_atomic

MEASURED = "measured"
CONFIGURED = "configured"
REFUSED = "refused"
NOT_MEASURED = "not_measured"

_POLICY = (
    "Every 'measured' value was produced by running the named tool on this "
    "machine. 'refused' and 'not_measured' values are recorded honestly and are "
    "never filled in with estimates, simulations, or other-GPU results."
)


def _entry(status: str, value: Any = None, *, source: str | None = None,
           command: list[str] | None = None, note: str | None = None) -> dict:
    item: dict[str, Any] = {"status": status, "value": value}
    if source is not None:
        item["source"] = source
    if command is not None:
        item["command"] = command
    if note is not None:
        item["note"] = note
    return item


def _by_suffix(metrics: dict, suffix: str) -> dict:
    """Group ninfer_bench metrics by their test label (tg128, pp512, ...)."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith(suffix) and isinstance(value, (int, float)):
            label = key[: -len(suffix)].rstrip("_") or "value"
            out[label] = float(value)
    return out


def _tool_entry(result: BenchResult | None, value: Any, *, present: bool,
                absent_note: str) -> dict:
    """Wrap a value that depends on a tool run, keeping the run's provenance."""
    if result is None:
        return _entry(NOT_MEASURED, note=absent_note)
    if result.returncode == -1:
        return _entry(REFUSED, source=result.executable, command=result.command,
                      note=result.stderr_tail.strip()[-400:] or "tool could not run")
    if not present:
        return _entry(NOT_MEASURED, source=result.executable, command=result.command,
                      note="tool ran but produced no parseable value for this metric")
    return _entry(MEASURED, value, source=result.executable, command=result.command)


def performance_report(
    results: dict[str, BenchResult],
    *,
    probe: dict | None = None,
    runtime: dict | None = None,
    environment: dict | None = None,
) -> dict:
    """Assemble the performance report from real tool runs only.

    `results` maps a tool name to its BenchResult (e.g. ninfer_bench). `probe` is
    an optional ServeProbe.to_dict() from a running server (the TTFT source).
    `runtime` holds the chosen serving parameters; `environment` the machine
    identity captured at report time. Neither is a performance measurement.
    """
    runtime = runtime or {}
    bench = results.get("ninfer_bench")
    metrics = bench.metrics if bench is not None and bench.ok else {}
    decode = _by_suffix(metrics, "_decode_output_tok_s")
    prefill = _by_suffix(metrics, "_prefill_tok_s")
    spec = {k: v for k, v in metrics.items()
            if "_spec_" in k and isinstance(v, (int, float))}

    absent = "benchmark stage not run (enable with run_benchmark / --bench)"
    ttft = None
    if probe is not None:
        ttft = probe.get("ttft_seconds")
    ttft_entry = (
        _entry(MEASURED, ttft, source="serve_probe.probe",
               note="time to first streamed token from a running server")
        if ttft is not None else
        _entry(NOT_MEASURED, note="no running-server probe was performed")
    )

    spec_note = metrics.get("_corpus_caveat")
    speculative = (
        _entry(MEASURED, spec, source=bench.executable, command=bench.command,
               note=spec_note)
        if spec else
        _entry(NOT_MEASURED, note="no speculative component was benchmarked")
    )

    return {
        "recorded_at": utc_now(),
        "policy": _POLICY,
        "environment": environment or {},
        "metrics": {
            "decode_tok_s": _tool_entry(bench, decode, present=bool(decode),
                                        absent_note=absent),
            "prefill_tok_s": _tool_entry(bench, prefill, present=bool(prefill),
                                         absent_note=absent),
            "ttft_seconds": ttft_entry,
            "peak_vram_bytes": _entry(
                NOT_MEASURED,
                note="ninfer_bench does not report VRAM; no sampler is wired, "
                     "so no peak is invented"),
            "peak_ram_bytes": _entry(
                NOT_MEASURED,
                note="host RAM peak is not sampled during the benchmark run"),
            "cpu_offload": _entry(
                NOT_MEASURED,
                note="offload residency is not exposed by the benchmark tool"),
            "context_tokens": _entry(
                CONFIGURED, runtime.get("max_context"),
                note="serving parameter chosen by the user, not a measurement"),
            "speculative": speculative,
        },
        "runs": {name: result.to_dict() for name, result in results.items()},
    }


def quality_report(
    *,
    perplexity: BenchResult | None = None,
    baseline_perplexity: BenchResult | None = None,
    jailbreak: str = "none",
    runtime: dict | None = None,
) -> dict:
    """Assemble the quality report; unrun evaluations stay not_measured.

    Quantization loss is a perplexity delta and is only computed when both the
    artifact and a baseline were measured on the same corpus/context. Jailbreak
    behavior change is a separate axis that needs a paired behavioral eval; it
    is never inferred from perplexity.
    """
    runtime = runtime or {}

    def ppl_value(result: BenchResult | None) -> float | None:
        if result is not None and result.ok:
            value = result.metrics.get("perplexity")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    ppl = ppl_value(perplexity)
    baseline = ppl_value(baseline_perplexity)
    ppl_entry = _tool_entry(
        perplexity, ppl, present=ppl is not None,
        absent_note="perplexity stage not run (supply benchmark.perplexity_text "
                    "or benchmark.perplexity_corpus)",
    )
    baseline_entry = _tool_entry(
        baseline_perplexity, baseline, present=baseline is not None,
        absent_note="no baseline artifact was measured for comparison",
    )

    if ppl is not None and baseline is not None:
        loss = _entry(
            MEASURED, round(ppl - baseline, 6),
            source="ninfer-perplexity.exe",
            note="artifact PPL minus baseline PPL; valid only if both used the "
                 "same corpus, context and stride",
        )
    else:
        loss = _entry(
            NOT_MEASURED,
            note="needs both artifact and baseline perplexity on identical input",
        )

    behavioral = {
        name: _entry(NOT_MEASURED, note=note)
        for name, note in (
            ("output_consistency", "no consistency harness is wired"),
            ("code", "no code-evaluation dataset/judge is wired"),
            ("math", "no math-reasoning dataset/judge is wired"),
            ("instruction_following", "no instruction-following eval is wired"),
            ("long_context", "no long-context eval is wired"),
        )
    }

    return {
        "recorded_at": utc_now(),
        "policy": _POLICY,
        "separation": (
            "Quantization loss is measured as a perplexity delta. Jailbreak "
            "behavior change is a distinct axis requiring a paired pre/post "
            "behavioral evaluation; it is not derived from perplexity."
        ),
        "jailbreak_processor": jailbreak,
        "context_tokens": _entry(CONFIGURED, runtime.get("max_context"),
                                 note="serving parameter, not a measurement"),
        "metrics": {
            "perplexity": ppl_entry,
            "baseline_perplexity": baseline_entry,
            "quantization_loss": loss,
            **behavioral,
            "jailbreak_behavior_change": _entry(
                NOT_MEASURED,
                note="requires paired behavioral eval on pre/post weights; "
                     f"processor '{jailbreak}' is recorded as provenance only",
            ),
        },
        "runs": {
            name: result.to_dict()
            for name, result in (
                ("ninfer_perplexity", perplexity),
                ("ninfer_perplexity_baseline", baseline_perplexity),
            )
            if result is not None
        },
    }


def measurement_summary(performance: dict, quality: dict) -> dict:
    """Compact, honest headline for manifest.measurements."""
    perf = performance["metrics"]
    qual = quality["metrics"]

    def headline(entry: dict) -> dict:
        out = {"status": entry["status"]}
        if entry["status"] == MEASURED and entry["value"] is not None:
            out["value"] = entry["value"]
        return out

    return {
        "performance": {
            "decode_tok_s": headline(perf["decode_tok_s"]),
            "prefill_tok_s": headline(perf["prefill_tok_s"]),
            "ttft_seconds": headline(perf["ttft_seconds"]),
        },
        "quality": {
            "perplexity": headline(qual["perplexity"]),
            "quantization_loss": headline(qual["quantization_loss"]),
        },
    }


def _render_md(title: str, report: dict) -> str:
    lines = [f"# {title}", "", f"Recorded: {report['recorded_at']}", "",
             report["policy"], ""]
    if report.get("separation"):
        lines += [report["separation"], ""]
    env = report.get("environment") or {}
    if env:
        lines += ["## Environment", ""]
        lines += [f"- {key}: {value}" for key, value in env.items()]
        lines += [""]
    lines += ["## Metrics", "", "| metric | status | value | note |",
              "| --- | --- | --- | --- |"]
    for name, entry in report["metrics"].items():
        value = entry.get("value")
        value_text = "" if value is None else str(value)
        note = (entry.get("note") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {entry['status']} | {value_text} | {note} |")
    lines += [""]
    runs = report.get("runs") or {}
    if runs:
        lines += ["## Tool runs (reproducibility)", ""]
        for name, run in runs.items():
            lines.append(f"### {name}")
            lines.append(f"- exit: {run['returncode']}  seconds: {run['seconds']}")
            lines.append("- command: `" + " ".join(run["command"]) + "`")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(report_dir: str | Path, performance: dict, quality: dict) -> dict[str, Path]:
    """Persist performance/quality reports as JSON + Markdown; return the paths."""
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, title, report in (
        ("performance", "Performance report", performance),
        ("quality", "Quality report", quality),
    ):
        json_path = directory / f"{name}.json"
        write_json_atomic(json_path, report)
        md_path = directory / f"{name}.md"
        md_path.write_text(_render_md(title, report), encoding="utf-8")
        paths[f"{name}_json"] = json_path
        paths[f"{name}_md"] = md_path
    return paths

def quality_gate(quality: dict, limit: float) -> dict:
    import math
    metrics = quality["metrics"]
    current = metrics["perplexity"]
    baseline = metrics["baseline_perplexity"]
    if any(entry.get("status") != "measured" for entry in (current, baseline)):
        return {"status":"not_measured", "note":"Quality measurement did not succeed; a pass cannot be asserted"}
    value, reference = current["value"], baseline["value"]
    if not all(isinstance(x, (int,float)) and math.isfinite(x) and x > 0 for x in (value,reference)):
        return {"status":"not_measured", "note":"Perplexity is invalid; a pass cannot be asserted"}
    increase = value / reference - 1
    return {"status":"passed" if increase <= limit else "failed", "relative_increase":increase,
            "maximum_relative_increase":limit, "note":"Same-corpus PPL comparison; does not represent every task's ability or abliteration effect"}
