"""Command-line front end.

Subcommands:
  inspect  identify architecture, precision, NVFP4 layout and speculative parts
  plan     show the full route, compatibility, resources and risks of a build
  recipes  list every verified conversion recipe
  build    execute a plan (plan is re-derived and re-gated inside)
  bench    run ninfer_bench against an existing artifact and record the report

The CLI never asks for low-level converter flags beyond the documented user
choices; everything else is derived by the planner and shown before execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends.registry import RECIPES
from .benchmark import BenchmarkEngine, record_report
from .domain import (
    JAILBREAK_CHOICES,
    ForgeConfig,
)
from .planner import ForgePlanner
from .runner import DEFAULT_NINFER_BIN, BuildRunner


def _print_json(value) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def _add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True,
                        help="model directory or .safetensors file")
    parser.add_argument("--output", required=True, help="build output directory")
    parser.add_argument("--target", default="auto",
                        help="recipe name or 'auto' (default: Bonsai 2 ternary when a "
                             "ternary source is supplied, else the input's import route, "
                             "else the lowest-bpw verified groupwise target)")
    parser.add_argument("--jailbreak", default="none", choices=JAILBREAK_CHOICES)
    parser.add_argument("--jailbreak-param", action="append", default=[], metavar="KEY=VALUE",
                        help="processor parameter, e.g. direction_file=<vector> for "
                             "abliteration (repeatable)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                        help="extra named weights: quantized=, ternary=, gguf=, imatrix=")
    parser.add_argument("--allow-approximate-requantization", action="store_true",
                        help="accept stacked loss when modifying already-quantized input")
    parser.add_argument("--max-context", type=int, default=32768,
                        help="launcher context size (serving choice only)")


def _config_from_args(args: argparse.Namespace) -> ForgeConfig:
    sources = {}
    for entry in args.source:
        name, separator, path = entry.partition("=")
        if not separator or not name or not path or name in sources:
            raise SystemExit(f"--source requires unique NAME=PATH entries, got {entry!r}")
        sources[name] = path
    jailbreak_params: dict[str, str] = {}
    for entry in args.jailbreak_param:
        key, separator, value = entry.partition("=")
        if not separator or not key or not value or key in jailbreak_params:
            raise SystemExit(f"--jailbreak-param requires unique KEY=VALUE entries, got {entry!r}")
        jailbreak_params[key] = value
    return ForgeConfig(
        input_path=args.input,
        output_dir=args.output,
        target=args.target,
        jailbreak=args.jailbreak,
        jailbreak_params=jailbreak_params,
        device=args.device,
        sources=sources,
        allow_approximate_requantization=args.allow_approximate_requantization,
        runtime={"max_context": args.max_context},
    )


def _render_plan(plan) -> None:
    print(f"input      : {plan.inspection.path}")
    print(f"kind       : {plan.inspection.input_kind.value}  "
          f"arch={plan.inspection.architecture}  precision={plan.inspection.weight_precision}")
    print(f"executable : {plan.executable}")
    print("\nsteps:")
    for step in plan.steps:
        print(f"  [{step.status:7}] {step.name}: {step.implementation}")
    if plan.resources is not None:
        r = plan.resources
        mib = 1024 * 1024
        print(f"\nresources  : input {r.input_bytes // mib} MiB, "
              f"temp disk ~{r.temporary_disk_bytes // mib} MiB, "
              f"free disk {r.available_disk_bytes // mib} MiB")
        if r.estimated_output_bytes is not None:
            print(f"             estimated artifact ~{r.estimated_output_bytes // mib} MiB")
    errors = [d for d in plan.diagnostics if d.severity.value == "error"]
    warnings = [d for d in plan.diagnostics if d.severity.value == "warning"]
    infos = [d for d in plan.diagnostics if d.severity.value == "info"]
    for group, label in ((errors, "ERRORS"), (warnings, "WARNINGS"), (infos, "NOTES")):
        if group:
            print(f"\n{label}:")
            for item in group:
                print(f"  - [{item.code}] {item.message}")
                for alt in item.alternatives:
                    print(f"      alternative: {alt}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="turboforge",
        description="TurboForge: inspect, quantize and validate local models for NInfer, Basalt or Strata",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_auto = sub.add_parser("auto", help="Provide only the model; dependencies and tools are downloaded automatically and conversion runs")
    p_auto.add_argument("model", help="A local path, Hugging Face org/model name, or repository link")
    p_auto.add_argument("--jailbreak", choices=("none", "abliteration"), default="none")
    p_auto.add_argument("--bench", action="store_true")
    p_auto.add_argument("--target", default="auto", help="An NInfer recipe, or hf_awq_4 / hf_gptq_2/3/4/8 / hf_packed_import")
    p_auto.add_argument("--speculative",choices=("auto","none","mtp","dflash2"),default="auto")
    p_auto.add_argument("--output", default=None)

    p_inspect = sub.add_parser("inspect", help="identify an input model")
    p_inspect.add_argument("input")
    p_inspect.add_argument("--json", action="store_true")

    p_recipes = sub.add_parser("recipes", help="list verified conversion recipes")
    p_recipes.add_argument("--json", action="store_true")
    p_recipes.add_argument("--hf", action="store_true", help="List the AWQ/GPTQ routes of the standalone HF engine")

    p_plan = sub.add_parser("plan", help="derive and display a build plan")
    _add_common_build_args(p_plan)
    p_plan.add_argument("--json", action="store_true")

    p_build = sub.add_parser("build", help="execute a build")
    _add_common_build_args(p_build)
    p_build.add_argument("--no-validation", action="store_true",
                         help="skip the ninfer.exe smoke generation after conversion")
    p_build.add_argument("--bench", action="store_true",
                         help="run ninfer_bench/perplexity after conversion and "
                              "fill the performance/quality reports with real numbers")

    p_run = sub.add_parser("run-config", help="execute a build from a JSON config file")
    p_run.add_argument("config_file")
    p_run.add_argument("--no-validation", action="store_true")
    p_run.add_argument("--bench", action="store_true",
                       help="force the benchmark stage on (overrides the config file)")

    p_webui = sub.add_parser("webui", help="serve the local configuration page (needs flask)")
    p_webui.add_argument("--host", default="127.0.0.1")
    p_webui.add_argument("--port", type=int, default=8090)

    p_bench = sub.add_parser("bench", help="measure an existing .ninfer artifact")
    p_bench.add_argument("artifact")
    p_bench.add_argument("--report-dir", default=None,
                         help="directory for benchmark.json (default: beside artifact)")
    p_bench.add_argument("--ninfer-dir", default=str(DEFAULT_NINFER_BIN))
    p_bench.add_argument("--spec", default=None, choices=(None, "mtp", "dflash", "dflash2"))
    p_bench.add_argument("--draft-tokens", type=int, default=None)
    p_bench.add_argument("--kv-dtype", default="rk4v4")
    p_bench.add_argument("--repetitions", type=int, default=5)
    p_bench.add_argument("--corpus", default=None,
                         help="token-id corpus for ninfer_bench "
                              "(default: upstream bench/fixtures/bench_corpus.ids if present)")

    args = parser.parse_args(argv)

    if args.command == "auto":
        from .api.builds import BuildManager
        import time
        options = {"jailbreak": args.jailbreak, "run_benchmark": args.bench, "target":args.target,"runtime":{"speculative":args.speculative}}
        if args.output:
            options["output_dir"] = args.output
        manager = BuildManager()
        job = manager.start(None, model=args.model, options=options)
        last = ""
        while True:
            state = manager.status(job)
            lines = state.get("setup_log", [])
            message = state["status"] + (": " + lines[-1] if lines else "")
            if message != last:
                print(message, flush=True)
                last = message
            if state["finished"]:
                print("Output directory: " + state["output_dir"])
                if state["error"]:
                    print(state["error"], file=sys.stderr)
                return 0 if state["status"] == "validated" else 1
            time.sleep(.5)

    if args.command == "inspect":
        from .inspector import ModelInspector
        try:
            inspection = ModelInspector().inspect(args.input)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            _print_json(inspection.to_dict())
        else:
            print(f"path       : {inspection.path}")
            print(f"kind       : {inspection.input_kind.value}")
            print(f"architecture: {inspection.architecture}")
            print(f"precision  : {inspection.weight_precision}")
            print(f"shards     : {inspection.shard_count}")
            print(f"tensors    : {inspection.tensor_count} "
                  f"({inspection.tensor_bytes / 1024**3:.2f} GiB)")
            if inspection.quantization:
                print("quantization:")
                for key in ("nvfp4_metadata_detected", "nvfp4_layout_verified",
                            "scale_tensor_count", "quantized_tensor_count"):
                    if key in inspection.quantization:
                        print(f"  {key}: {inspection.quantization[key]}")
            for item in inspection.diagnostics:
                print(f"  [{item.severity.value:7}] {item.message}")
        return 0

    if args.command == "recipes":
        from .backends.hf_quantization import HF_RECIPES
        data = [
            {
                "recipe": r.recipe, "label": r.label,
                "inputs": [k.value for k in r.input_kinds],
                "architectures": list(r.architectures),
                "storage_formats": list(r.storage_formats),
                "nominal_bits": r.nominal_bits,
                "typical_bpw": r.typical_bpw,
                "extra_sources": list(r.extra_sources),
                "notes": list(r.notes),
            }
            for r in (HF_RECIPES if args.hf else RECIPES)
        ]
        if args.json:
            _print_json(data)
        else:
            for item in data:
                src = f"  [needs --source {','.join(item['extra_sources'])}]" if item["extra_sources"] else ""
                print(f"{item['recipe']:32} {item['label']}{src}")
        return 0

    if args.command == "plan":
        config = _config_from_args(args)
        plan = ForgePlanner().plan(config)
        if args.json:
            _print_json(plan.to_dict())
        else:
            _render_plan(plan)
        return 0 if plan.executable else 1

    if args.command in ("build", "run-config"):
        if args.command == "build":
            config = _config_from_args(args)
        else:
            path = Path(args.config_file).expanduser()
            try:
                config = ForgeConfig.from_dict(
                    json.loads(path.read_text(encoding="utf-8-sig"))
                )
            except (OSError, ValueError, KeyError) as exc:
                print(f"error: cannot load config {path}: {exc}", file=sys.stderr)
                return 2
        config.run_validation = not args.no_validation
        config.run_benchmark = config.run_benchmark or args.bench
        runner = BuildRunner()
        manifest = runner.build(config)
        status = manifest.data["status"]
        print(f"build status: {status}")
        print(f"output dir  : {manifest.output_dir}")
        artifacts = manifest.data.get("artifacts", {})
        for name, item in artifacts.items():
            print(f"  artifact {name}: {item['path']} ({item['bytes']} bytes)")
        errors = [e["detail"] for e in manifest.data["history"]
                  if e["status"] in ("blocked", "failed")]
        for error in errors:
            print(f"  ! {error[:400]}")
        return 0 if status in ("validated", "converted") else 1

    if args.command == "webui":
        from .webui import main as webui_main
        return webui_main(["--host", args.host, "--port", str(args.port)])

    if args.command == "bench":
        artifact = Path(args.artifact).expanduser().resolve()
        if not artifact.is_file():
            print(f"error: artifact not found: {artifact}", file=sys.stderr)
            return 2
        engine = BenchmarkEngine(
            Path(args.ninfer_dir),
            corpus=Path(args.corpus) if args.corpus else None,
        )
        result = engine.bench(
            artifact,
            spec=args.spec,
            draft_tokens=args.draft_tokens,
            kv_dtype=args.kv_dtype,
            repetitions=args.repetitions,
        )
        report_dir = Path(args.report_dir or artifact.parent)
        report_path = record_report(report_dir, {"ninfer_bench": result})
        print(f"bench exit={result.returncode} in {result.seconds:.1f}s")
        if result.metrics:
            for key, value in sorted(result.metrics.items()):
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f} tok/s")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  no throughput metrics parsed (see report for tool output)")
        print(f"report: {report_path}")
        return 0 if result.ok else 1

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
