"""Build execution: plan gate, jailbreak stage, converter, artifacts, validation.

The original input is never modified. Every stage writes into the output
directory, and every transition is recorded in the manifest so a failed build
leaves an accurate trail instead of a silent half-artifact.
"""

from __future__ import annotations

from .assistant.config import child_environment
import json
import os
import subprocess
from pathlib import Path

from . import serving
from .backends.compiler import NInferCompiler
from .backends.registry import RecipeSpec, get_recipe
from .benchmark import BenchmarkEngine
from .domain import ForgeConfig, ForgePlan
from .jailbreak.processors import get_processor
from .local_endpoint import default_serving_port
from .manifest import BuildManifest
from .planner import ForgePlanner
from .reports import (
    measurement_summary,
    performance_report,
    quality_report,
    write_reports,
)
from .system import gpu_status


# Engine binaries used by the generated launcher, smoke validation and benchmark.
# Defaults to the installed package; override with TURBOFORGE_NINFER_BIN to point
# at a locally built engine (e.g. one carrying geometry ports the release lacks).
_BUNDLED_ENGINE = Path(__file__).resolve().parents[2] / "runtime/bin"
DEFAULT_NINFER_BIN = Path(os.environ.get("TURBOFORGE_NINFER_BIN", str(_BUNDLED_ENGINE) if _BUNDLED_ENGINE.is_dir() else r"D:\llm\ninfer"))


def _bat_path(value: str | Path) -> str:
    """Escape a path for a generated batch file (percent is the only variable)."""
    return str(value).replace("%", "%%")


class BuildRunner:
    def __init__(self, planner: ForgePlanner | None = None,
                 ninfer_dir: Path | str = DEFAULT_NINFER_BIN) -> None:
        self.planner = planner or ForgePlanner()
        self.compiler = NInferCompiler(self.planner.upstream)
        # Absolute so the generated launcher works after it cd's to its own folder.
        self.ninfer_dir = Path(ninfer_dir).expanduser().resolve()

    def build(self, config: ForgeConfig) -> BuildManifest:
        config.validate()
        from .backends.hf_quantization import is_hf
        if is_hf(config.target):
            from .backends.hf_worker import build
            return build(config,self.planner.inspector.inspect(config.input_path))
        for name in ("input_path", "output_dir", "upstream_dir", "ninfer_dir"):
            if getattr(config,name):
                setattr(config,name,str(Path(getattr(config,name)).expanduser().resolve()))
        config.sources = {name:str(Path(value).expanduser().resolve()) for name,value in config.sources.items()}
        for name in ("perplexity_text","perplexity_corpus","baseline_artifact","corpus"):
            if name in config.benchmark:
                config.benchmark[name] = str(Path(config.benchmark[name]).expanduser().resolve())
        if config.upstream_dir:
            from .backends.compiler import probe_upstream
            self.planner = ForgePlanner(self.planner.inspector, probe_upstream(Path(config.upstream_dir)))
            self.compiler = NInferCompiler(self.planner.upstream)
        if config.ninfer_dir:
            self.ninfer_dir = Path(config.ninfer_dir)
        plan = self.planner.plan(config)
        output_dir = Path(config.output_dir).expanduser().resolve()
        manifest = self._open_manifest(output_dir, config, plan)
        if not plan.executable:
            manifest.transition(
                "blocked",
                self._errors(plan) or "the plan has no executable steps",
                plan_diagnostics=[
                    {"code": d.code, "message": d.message, "severity": d.severity.value,
                     "alternatives": list(d.alternatives)}
                    for d in plan.diagnostics
                ],
            )
            return manifest

        try:
            reference = self._validate_quality_reference(config, manifest)
            model_dir = self._run_jailbreak(config, plan, output_dir, manifest)
            if config.jailbreak != "none":
                from .manifest import file_sha256
                processed = {file.name:file_sha256(file) for file in model_dir.glob("*.safetensors")}
                manifest.data["processed_weights"] = processed
                manifest.save()
                if reference is not None and (not processed or reference.get("processed_weights") != processed):
                    raise ValueError("实际越狱后权重与对照不一致，不能作为纯量化对照")
            artifact = self._run_converter(config, plan, output_dir, manifest, model_dir)
        except (RuntimeError, NotImplementedError, OSError, ValueError) as exc:
            manifest.transition("failed", f"{type(exc).__name__}: {exc}")
            return manifest

        if config.run_validation:
            manifest.transition("validating", "ninfer.exe smoke generation")
            ok, detail = self._smoke_validate(
                artifact, output_dir / "logs" / "smoke.log",
                max_context=min(self._effective_context(config, plan), 4096),
                flags=self._launcher_flags(plan),
            )
            manifest.transition("validated" if ok else "validation-failed", detail)
        else:
            manifest.transition("converted", "validation skipped by configuration")
        try:
            self._write_reports(artifact, output_dir, config, manifest, plan)
        except (RuntimeError, OSError, ValueError) as exc:
            manifest.transition("failed", f"评测失败：{type(exc).__name__}: {exc}")
            return manifest
        if manifest.data["status"] in ("validated", "converted"):
            self._write_launcher(artifact, output_dir, config, manifest, plan)
        return manifest

    # -- stages --------------------------------------------------------------

    def _open_manifest(self, output_dir: Path, config: ForgeConfig,
                       plan: ForgePlan) -> BuildManifest:
        # BuildManifest.create refuses a non-empty directory so an earlier build
        # is never overwritten; it owns directory creation, so the plan is
        # written only after the manifest exists.
        manifest = BuildManifest.create(output_dir, config, plan.inspection, plan.to_dict())
        for name in ("intermediate",):
            (output_dir / name).mkdir(exist_ok=True)
        plan_file = output_dir / "plan.json"
        plan_file.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        manifest.add_artifact("plan", plan_file)
        return manifest

    def _run_jailbreak(self, config: ForgeConfig, plan: ForgePlan, output_dir: Path,
                       manifest: BuildManifest) -> Path:
        """Return the directory the converter should use as --model."""
        processor = get_processor(config.jailbreak)
        model_dir = Path(plan.inspection.path)
        if processor.name == "none":
            return model_dir
        work_dir = output_dir / "intermediate" / processor.name
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest.transition("jailbreaking", f"processor {processor.name}")
        processed = processor.apply(
            plan.inspection, work_dir, **config.jailbreak_params
        )
        manifest.add_artifact(f"jailbreak_{processor.name}", Path(processed))
        manifest.transition("jailbroken", f"processed weights at {processed}")
        return Path(processed)

    def _run_converter(self, config: ForgeConfig, plan: ForgePlan, output_dir: Path,
                       manifest: BuildManifest, model_dir: Path) -> Path:
        from .speculative import prepare_sources,verify_ninfer
        recipe = get_recipe(plan.backend)
        step = next((s for s in plan.steps if s.name == "quantize+compile"), None)
        components = step.details.get("components", recipe.components) if step else recipe.components
        requested = Path(config.output_dir).name or "model"
        artifact = output_dir / f"{requested}.ninfer"
        manifest.transition("converting", f"recipe {recipe.recipe}",
                            converter={"recipe": recipe.recipe,
                                       "components": components,
                                       "device": config.device,
                                       "log": "logs/convert.log"})
        result = self.compiler.run(
            recipe, model_dir, artifact,
            sources={k:Path(v) for k,v in prepare_sources(model_dir,config.sources,output_dir/"intermediate/speculative-input").items()},
            components=components,
            name=requested,
            device=config.device,
            log_path=output_dir / "logs" / "convert.log",
        )
        produced = self._resolve_artifact(artifact)
        expected = [name for name in ("mtp","dflash2") if name in (components or "").split(",")]
        retention = verify_ninfer(produced,expected) if expected else {}
        manifest.data["speculative_retention"] = retention
        manifest.add_artifact("ninfer_artifact", produced, digest=True)
        manifest.transition(
            "converted", f"wrote {produced.name}",
            converter={"recipe": recipe.recipe, "components": components,
                       "device": config.device,
                       "seconds": round(result["seconds"], 1),
                       "log": "logs/convert.log"},
        )
        return produced

    def _write_launcher(self, artifact: Path, output_dir: Path, config: ForgeConfig,
                        manifest: BuildManifest, plan: ForgePlan) -> None:
        context = self._effective_context(config, plan)
        concurrency = int(config.runtime.get("max_concurrency", 1))
        port = int(config.runtime.get("port") or default_serving_port())
        server = _bat_path(self.ninfer_dir / "ninfer-serve.exe")
        relative = _bat_path(artifact.relative_to(output_dir).as_posix())
        flags = [_bat_path(flag) for flag in self._launcher_flags(plan)]
        lines = [
            "@echo off",
            "chcp 65001 >nul",
            "setlocal EnableExtensions",
            'cd /d "%~dp0"',
            "rem Generated by TurboForge. These flags are serving choices only;",
            "rem the artifact is identical without them. See manifest.json for provenance.",
            f'set "SERVER={server}"',
            f'set "MODEL=%~dp0..\\{relative}"',
            'if not exist "%SERVER%" (',
            '  echo Missing NInfer server: %SERVER%',
            "  pause",
            "  exit /b 1",
            ")",
            'if not exist "%MODEL%" (',
            '  echo Missing model artifact: %MODEL%',
            "  pause",
            "  exit /b 1",
            ")",
            '"%SERVER%" "%MODEL%" ^',
            f"  --max-context {context} --kv-capacity {context} ^",
            f"  --max-concurrency {concurrency} ^",
        ]
        lines += [f"  {flag} ^" for flag in flags]
        lines += [
            f"  --host 127.0.0.1 --port {port}",
            "echo.",
            "echo NInfer server stopped. Press any key to close.",
            "pause >nul",
            "",
        ]
        # One CRLF per line: a bare \n inside a cmd line continuation misparses.
        body = "\r\n".join(lines)
        launcher = output_dir / "scripts" / "run.bat"
        launcher.write_text(body, encoding="utf-8")
        manifest.add_artifact("launcher", launcher)

    @staticmethod
    def _plan_basis(plan: ForgePlan) -> tuple[RecipeSpec | None, bool, bool]:
        """The recipe and the components the build actually imported."""
        recipe = get_recipe(plan.backend) if plan.backend else None
        step = next((s for s in plan.steps if s.name == "speculative"), None)
        planned = step is not None and step.status == "planned"
        return (
            recipe,
            bool(step.details.get("mtp")) if planned else False,
            bool(step.details.get("dflash2")) if planned else False,
        )

    def _launcher_flags(self, plan: ForgePlan) -> list[str]:
        """Serving flags for this artifact's storage route, plus speculation.

        Delegates to the shared serving profile so the launcher, the plan preview
        and the benchmark never disagree about what the artifact will run.
        """
        recipe, mtp, dflash2 = self._plan_basis(plan)
        return serving.launcher_flags(recipe, plan.inspection.config, mtp, dflash2)

    def _kv_dtype(self, plan: ForgePlan) -> str:
        """KV storage chosen for the geometry's MTP speed/VRAM tradeoff."""
        recipe, _, _ = self._plan_basis(plan)
        return serving.kv_dtype(recipe, plan.inspection.config)

    @staticmethod
    def _spec_choice(plan: ForgePlan) -> tuple[str, int] | None:
        """The drafter the build actually imported, as (backend, draft_tokens).

        None when no speculative component was planned, so neither the launcher
        nor the benchmark invents a --spec flag for an artifact without one.
        """
        _, mtp, dflash2 = BuildRunner._plan_basis(plan)
        return serving.spec_choice(mtp, dflash2)

    @classmethod
    def _speculative_flags(cls, plan: ForgePlan) -> list[str]:
        """Serve-time speculation, only when the build really imported a drafter.

        Passing --spec to an artifact without that component is refused at
        startup rather than ignored, so a skipped or blocked stage yields no flag.
        --lm-head-draft is deliberately not forced: the engine's ProposalHead::Auto
        already picks the optimized proposal head when the artifact stores one and
        the full head otherwise. Forcing it is refused on artifacts without a
        proposal head, e.g. the small-model GGUF Q8_0 MTP bridge.
        """
        choice = cls._spec_choice(plan)
        if choice is None:
            return []
        backend, draft_tokens = choice
        return [
            f"--spec {backend}",
            f"--draft-tokens {draft_tokens}",
        ]

    @staticmethod
    def _context_ceiling(plan: ForgePlan) -> int | None:
        recipe, _, _ = BuildRunner._plan_basis(plan)
        return serving.context_ceiling(recipe, plan.inspection.config)

    def _effective_context(self, config: ForgeConfig, plan: ForgePlan) -> int:
        """Serving context clamped to what the engine supports for this geometry.

        The planner emits the matching warning, so the preview and the generated
        launcher never disagree about the context the artifact can actually serve.
        """
        recipe, _, _ = self._plan_basis(plan)
        return serving.effective_context(recipe, plan.inspection.config, config.runtime)

    def _validate_quality_reference(self, config: ForgeConfig, manifest: BuildManifest) -> dict | None:
        if "max_ppl_relative_increase" in config.benchmark:
            from .manifest import read_json_atomic
            baseline_path = Path(config.benchmark["baseline_artifact"])
            provenance = baseline_path.parent / "manifest.json"
            if not provenance.is_file():
                raise ValueError("质量门禁需要 TurboForge 生成的对照产物及 manifest，以核对模型来源和处理阶段")
            reference = read_json_atomic(provenance)
            if reference.get("source_files") != manifest.data.get("source_files") or reference.get("config", {}).get("jailbreak") != config.jailbreak:
                raise ValueError("对照模型的权重来源或越狱处理阶段不同，不能把混合变化标成量化质量损失")
            artifact_info = reference.get("artifacts", {}).get("ninfer_artifact", {})
            from .manifest import file_sha256
            if not artifact_info.get("sha256") or file_sha256(baseline_path) != artifact_info["sha256"]:
                raise ValueError("对照产物与其 manifest 校验值不一致，质量门禁不能通过")
            if not reference.get("source_identity") or reference["source_identity"] != manifest.data.get("source_identity"):
                raise ValueError("对照缺少匹配的配置/分词器指纹，请用相同模型资料重新生成对照产物")
            if reference.get("config", {}).get("jailbreak_params", {}) != config.jailbreak_params:
                raise ValueError("对照模型的越狱参数不同，不能作为纯量化质量对照")
            imported = ("gguf", "ternary", "quantized", "engram", "ngram")
            if {key:reference.get("config", {}).get("sources", {}).get(key) for key in imported} != {key:config.sources.get(key) for key in imported}:
                raise ValueError("对照使用了不同的导入权重或附加表，不能认定为相同来源")
            return reference
        return None

    def _write_reports(self, artifact: Path, output_dir: Path, config: ForgeConfig,
                       manifest: BuildManifest, plan: ForgePlan) -> None:
        """Always emit performance + quality reports; measure only when asked.

        With run_benchmark off, every metric stays not_measured (never estimated).
        With it on, ninfer_bench / ninfer-perplexity actually run and their real
        numbers, command lines and refusals are recorded.
        """
        prior_status = manifest.data["status"]
        results: dict = {}
        perplexity = None
        baseline = None
        if config.run_benchmark:
            manifest.transition("benchmarking", "ninfer_bench / ninfer-perplexity")
            opts = config.benchmark
            corpus = Path(opts["corpus"]) if opts.get("corpus") else None
            engine = BenchmarkEngine(self.ninfer_dir, corpus=corpus)
            choice = self._spec_choice(plan)
            spec = choice[0] if choice else None
            draft = choice[1] if choice else None
            n_prompt = tuple(opts.get("n_prompt", (512, 2048)))
            ceiling = self._context_ceiling(plan)
            if ceiling is not None:
                # ninfer_bench derives max_context from the largest prompt; keep it
                # under the geometry ceiling so the bench starts instead of refusing.
                n_prompt = tuple(min(p, ceiling - 8) for p in n_prompt)
            results["ninfer_bench"] = engine.bench(
                artifact,
                n_prompt=n_prompt,
                n_gen=tuple(opts.get("n_gen", (128,))),
                repetitions=int(opts.get("repetitions", 5)),
                kv_dtype=opts.get("kv_dtype") or self._kv_dtype(plan),
                spec=spec,
                draft_tokens=draft,
            )
            ppl_kwargs = dict(
                context=min(int(opts.get("perplexity_context", 4096)), ceiling) if ceiling else int(opts.get("perplexity_context", 4096)),
                stride=int(opts.get("perplexity_stride", 512)),
            )
            if opts.get("perplexity_text") or opts.get("perplexity_corpus"):
                perplexity = engine.perplexity(
                    artifact,
                    text_file=Path(opts["perplexity_text"]) if opts.get("perplexity_text") else None,
                    corpus_manifest=Path(opts["perplexity_corpus"]) if opts.get("perplexity_corpus") else None,
                    extra=serving.perplexity_flags(self._launcher_flags(plan)),
                    **ppl_kwargs,
                )
            if opts.get("baseline_artifact"):
                from .manifest import read_json_atomic
                reference_file = Path(opts["baseline_artifact"]).parent / "manifest.json"
                reference_flags = [flag for flag in self._launcher_flags(plan) if not flag.startswith("--embedding-q4")]
                if reference_file.is_file():
                    reference_flags = read_json_atomic(reference_file).get("plan", {}).get("serving", {}).get("launcher_flags", reference_flags)
                baseline = engine.perplexity(
                    Path(opts["baseline_artifact"]),
                    text_file=Path(opts["perplexity_text"]) if opts.get("perplexity_text") else None,
                    corpus_manifest=Path(opts["perplexity_corpus"]) if opts.get("perplexity_corpus") else None,
                    extra=serving.perplexity_flags(reference_flags),
                    **ppl_kwargs,
                )

        gpu = gpu_status()
        environment = {
            "gpu": gpu.name if gpu else "not detected",
            "gpu_total_bytes": gpu.total_bytes if gpu else None,
            "ninfer_dir": str(self.ninfer_dir),
        }
        performance = performance_report(
            results, runtime=plan.serving or config.runtime, environment=environment
        )
        quality = quality_report(
            perplexity=perplexity, baseline_perplexity=baseline,
            jailbreak=config.jailbreak, runtime=plan.serving or config.runtime,
        )
        if "max_ppl_relative_increase" in config.benchmark:
            from .reports import quality_gate
            quality["gate"] = quality_gate(quality, config.benchmark["max_ppl_relative_increase"])
            if quality["gate"]["status"] != "passed":
                prior_status = "quality-rejected"
        paths = write_reports(output_dir / "reports", performance, quality)
        for name, path in paths.items():
            manifest.add_artifact(f"report_{name}", path)
        manifest.transition(
            prior_status, "performance/quality reports written",
            measurements=measurement_summary(performance, quality),
        )

    def _smoke_validate(self, artifact: Path, log_path: Path,
                        max_context: int = 4096, flags: list[str] | None = None) -> tuple[bool, str]:
        exe = self.ninfer_dir / "ninfer.exe"
        if not exe.is_file():
            return False, f"ninfer.exe not found at {exe}; cannot validate the artifact"
        try:
            import shlex
            profile = [part for flag in (flags or []) if flag.split()[0] not in serving.SERVER_ONLY_FLAGS for part in shlex.split(flag)]
            result = subprocess.run(
                [str(exe), str(artifact), "--prompt", "Reply with the single word: ok",
                 "--max-new", "16", "--max-context", str(max_context),
                 "--log-level", "warning", "--raw-output", *profile],
                capture_output=True, text=True, timeout=900,
                encoding="utf-8", errors="replace",
                env=child_environment(),
            )
        except subprocess.TimeoutExpired:
            return False, "smoke generation timed out after 900s"
        except OSError as exc:
            return False, f"could not launch ninfer.exe: {exc}"
        log_path.write_text(
            f"exit={result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}", encoding="utf-8",
        )
        if result.returncode != 0:
            errors = [line for line in result.stderr.splitlines() if "error:" in line.lower()]
            return False, (
                f"ninfer.exe exited {result.returncode}: "
                f"{'; '.join(errors)[:800] or result.stderr.strip()[-400:] or 'no stderr'}"
            )
        if not result.stdout.strip():
            return False, "ninfer.exe loaded the artifact but generated no text"
        return True, f"smoke generation ok ({len(result.stdout.strip())} chars)"

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _resolve_artifact(artifact: Path) -> Path:
        if artifact.exists():
            return artifact
        # The writer may add its own extension; accept exactly one candidate.
        candidates = sorted(artifact.parent.glob(artifact.name + "*"))
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            f"converter reported success but no single artifact matches {artifact}"
        )

    @staticmethod
    def _errors(plan: ForgePlan) -> str:
        return "; ".join(
            d.message for d in plan.diagnostics if d.severity.value == "error"
        )
