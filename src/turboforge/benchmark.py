"""Real measurement only.

Every number a TurboForge report contains comes from a tool that was actually
executed: ninfer_bench.exe for throughput, ninfer-perplexity.exe for quality.
A benchmark that could not run is recorded as refused, never estimated.
"""

from __future__ import annotations

from .assistant.config import child_environment
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import utc_now, write_json_atomic


@dataclass
class BenchResult:
    executable: str
    command: list[str]
    returncode: int
    seconds: float
    stdout_tail: str
    stderr_tail: str
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "executable": self.executable,
            "command": self.command,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 2),
            "metrics": self.metrics,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


# `ninfer_bench` table rows look like:
#   pp512    123.45 t/s ...   tg128   48.12 t/s ...
# We keep every "<label> <value> t/s" pair and let the caller interpret them.
_BENCH_ROW = re.compile(r"((?:pp|tg|pg)[^\s]*)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:t/s|tok/s)", re.I)


def parse_bench_output(text: str) -> dict:
    """Extract numbers from ninfer_bench `-o json` (or fall back to the table).

    Schema observed from the installed ninfer_bench (2026-11-08.1):
      {"schema_version": 18, "tests": [{"label": "tg128", "kind": "tg",
                    "decode_output_tok_s_mean": 179.8,
                    "prefill_tok_s_mean": ..., "speculative": {...}, ...}]}
    Empty return means "no numbers were parsed"; callers record that honestly.
    """
    metrics: dict = {}
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    rows = None
    if isinstance(data, dict):
        rows = data.get("tests") or data.get("results") or data.get("rows") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("name") or "").lower()
            if not label:
                continue
            for key, value in row.items():
                if key.endswith("_mean") and isinstance(value, (int, float)):
                    metrics[f"{label}_{key[:-5]}"] = float(value)
            spec = row.get("speculative")
            if isinstance(spec, dict):
                for key in ("acceptance_rate", "acceptance_length",
                            "drafted_tokens", "accepted_tokens", "rounds"):
                    if isinstance(spec.get(key), (int, float)):
                        metrics[f"{label}_spec_{key}"] = spec[key]
        if metrics:
            return metrics
    # Table fallback.
    for label, value in _BENCH_ROW.findall(text):
        key = label.lower().replace("=", "_") + "_tok_s"
        metrics.setdefault(key, float(value))
    return metrics


# The committed upstream corpus. Real text, but 682 distinct ids tiled to 65k
# tokens; upstream's own docstring says speculative acceptance measured on it is
# meaningless (a drafter predicts it perfectly). Throughput numbers stay valid.
DEFAULT_CORPUS_CANDIDATES = (
    Path(r"D:\llm\ninfer-upstream\bench\fixtures\bench_corpus.ids"),
)


class BenchmarkEngine:
    """Drives the executables installed beside ninfer.exe."""

    def __init__(self, ninfer_dir: Path, corpus: Path | None = None) -> None:
        self.ninfer_dir = Path(ninfer_dir).expanduser().resolve()
        self.corpus = corpus or next(
            (c for c in DEFAULT_CORPUS_CANDIDATES if c.is_file()), None
        )

    def bench(
        self,
        artifact: Path,
        *,
        n_prompt: tuple[int, ...] = (512, 2048),
        n_gen: tuple[int, ...] = (128,),
        repetitions: int = 5,
        warmup: int = 1,
        kv_dtype: str = "rk4v4",
        spec: str | None = None,
        draft_tokens: int | None = None,
        extra: tuple[str, ...] = (),
        timeout: int = 3600,
    ) -> BenchResult:
        exe = self.ninfer_dir / "ninfer_bench.exe"
        if not exe.is_file():
            return BenchResult("ninfer_bench.exe", [], -1, 0.0,
                               "", f"missing executable: {exe}")
        command = [
            str(exe), "--weights", str(artifact.expanduser().resolve()),
            "-p", ",".join(str(p) for p in n_prompt),
            "-n", ",".join(str(g) for g in n_gen),
            "-r", str(repetitions),
            "--warmup", str(warmup),
            "--kv-dtype", kv_dtype,
            "-o", "json",
        ]
        if self.corpus is not None:
            command += ["--corpus", str(self.corpus.expanduser().resolve())]
        if spec:
            command += ["--spec", spec]
            if draft_tokens:
                command += ["--draft-tokens", str(draft_tokens)]
        command += list(extra)
        result = self._run("ninfer_bench", command, timeout,
                           parser=parse_bench_output)
        if spec and self.corpus is not None:
            result.metrics["_corpus_caveat"] = (
                "speculative acceptance on the committed tiled corpus is not "
                "meaningful (upstream docstring); treat decode tok/s as valid, "
                "acceptance ratios as not"
            )
        return result

    def perplexity(
        self,
        artifact: Path,
        *,
        text_file: Path | None = None,
        corpus_manifest: Path | None = None,
        context: int = 4096,
        stride: int = 512,
        extra: tuple[str, ...] = (),
        timeout: int = 3600,
    ) -> BenchResult:
        exe = self.ninfer_dir / "ninfer-perplexity.exe"
        if not exe.is_file():
            return BenchResult("ninfer-perplexity.exe", [], -1, 0.0,
                               "", f"missing executable: {exe}")
        if text_file is None and corpus_manifest is None:
            raise ValueError("perplexity needs --text or --corpus input")
        if not 1 <= stride < context:
            raise ValueError("PPL stride 必须大于 0 且小于 context")
        command = [str(exe), str(artifact.expanduser().resolve()), "--context", str(context), "--stride", str(stride)]
        if text_file is not None:
            command += ["--text", str(text_file.expanduser().resolve())]
        else:
            command += ["--corpus", str(corpus_manifest.expanduser().resolve())]
        command += list(extra)
        return self._run("ninfer-perplexity", command, timeout,
                         parser=self._parse_ppl)

    # -- internals ------------------------------------------------------------

    def _run(self, name: str, command: list[str], timeout: int, parser=None) -> BenchResult:
        import time
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
                cwd=self.ninfer_dir,
                env=child_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            return BenchResult(name, command, -1, time.monotonic() - started,
                               str(exc.stdout or "")[-2000:],
                               f"timed out after {timeout}s")
        except OSError as exc:
            return BenchResult(name, command, -1, time.monotonic() - started,
                               "", f"could not start {name}: {exc}")
        metrics = parser(proc.stdout) if parser and proc.returncode == 0 else {}
        return BenchResult(
            name, command, proc.returncode, time.monotonic() - started,
            proc.stdout[-4000:], proc.stderr[-4000:], metrics,
        )

    @staticmethod
    def _parse_ppl(text: str) -> dict:
        number = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
        match = re.search(r"^\s*overall\s+\d+\s+\S+\s+" + number + r"\s*$", text, re.M)
        if match is None:
            match = re.search(r"\bperplexity\s*[:=]\s*" + number, text, re.I)
        if match:
            return {"perplexity": float(match.group(1))}
        return {}


def record_report(report_dir: Path, results: dict[str, BenchResult]) -> Path:
    """Persist raw measurements + command lines into the given directory."""
    report = {
        "recorded_at": utc_now(),
        "policy": "Every metric below was produced by running the named tool on "
                  "this machine; refused or failed runs are recorded, not filled in.",
        "results": {name: result.to_dict() for name, result in results.items()},
    }
    path = Path(report_dir) / "benchmark.json"
    write_json_atomic(path, report)
    return path
