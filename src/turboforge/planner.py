"""Plan builder: joins the inspection, jailbreak capability, recipe registry and
resource estimate into one explicit ForgePlan.

The planner never widens a user choice. An unsupported combination becomes an
error diagnostic naming the reason and the verified alternatives; execution is
refused until the user picks one.
"""

from __future__ import annotations

from pathlib import Path

from . import selection, serving
from .architecture import hidden_size
from .backends.compiler import probe_upstream
from .backends.registry import (
    RecipeSpec,
    geometry_admits,
    recipes_for,
    to_capability,
)
from .domain import (
    BackendCapability,
    Diagnostic,
    ForgeConfig,
    ForgePlan,
    InputKind,
    ModelInspection,
    PlanStep,
    Severity,
)
from .inspector import ModelInspector
from .jailbreak.processors import JailbreakProcessor, get_processor
from .resources import ResourceEstimator


class ForgePlanner:
    def __init__(self, inspector: ModelInspector | None = None, upstream=None) -> None:
        self.inspector = inspector or ModelInspector()
        self.upstream = upstream or probe_upstream()

    def available_recipes(self, inspection: ModelInspection) -> list[RecipeSpec]:
        from .backends.hf_quantization import recipes_for_config, quant_method
        hf = recipes_for_config(inspection.config)
        if quant_method(inspection.config) in ("awq", "gptq"):
            return hf
        hidden = hidden_size(inspection.config)
        return [
            recipe for recipe in recipes_for(inspection.input_kind, inspection.architecture)
            if geometry_admits(recipe, hidden)
        ] + hf

    def capabilities(
        self, inspection: ModelInspection, config: ForgeConfig | None = None
    ) -> list[BackendCapability]:
        """Every verified target for this model, with its availability and reason."""
        sources = tuple((config.sources if config else {}).keys())
        return [
            to_capability(
                recipe,
                upstream_available=self.upstream.found,
                torch_available=self.upstream.torch_available,
                sources_provided=sources,
            )
            for recipe in self.available_recipes(inspection)
        ]

    def plan(self, config: ForgeConfig) -> ForgePlan:
        config.validate()
        if config.upstream_dir and Path(config.upstream_dir).resolve() != self.upstream.root:
            return ForgePlanner(self.inspector, probe_upstream(Path(config.upstream_dir).resolve())).plan(config)
        diagnostics: list[Diagnostic] = []
        steps: list[PlanStep] = []

        try:
            inspection = self.inspector.inspect(
                config.input_path,
                metadata_only=bool(config.sources.get("gguf") or config.sources.get("ternary")),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            diagnostics.append(Diagnostic("input-unreadable", str(exc), Severity.ERROR))
            return ForgePlan(config, self._empty_inspection(config), [], diagnostics)

        from .backends import hf_quantization
        if hf_quantization.is_hf(config.target):
            return hf_quantization.plan(config, inspection)
        if hf_quantization.quant_method(inspection.config) in ("awq", "gptq"):
            return ForgePlan(config, inspection, [], [Diagnostic(
                "packed-needs-hf",
                "已有 AWQ/GPTQ 请选择 HF 原生保留路线，不能直接交给 NInfer。",
                Severity.ERROR,
            )])

        diagnostics.extend(self._preflight(config, inspection))
        processor = self._plan_jailbreak(config, inspection, steps, diagnostics)
        recipe = self._plan_target(config, inspection, steps, diagnostics)
        spec_step = self._speculative_step(config, inspection, recipe)
        resources = serving_summary = None
        if recipe is not None:
            resources, serving_summary = self._assess_target(
                config, inspection, recipe, spec_step, diagnostics
            )
        steps.append(spec_step)
        steps.append(PlanStep("validate", "ninfer load + bench", "planned" if config.run_validation else "skipped", {
            "checks": [
                "artifact readable by the NInfer reader",
                "ninfer.exe smoke generation on GPU",
                "ninfer_bench decode and prefill throughput",
            ],
            "note": "no result is reported as measured until these tools actually ran",
        }))

        return ForgePlan(
            config=config,
            inspection=inspection,
            steps=steps,
            diagnostics=diagnostics,
            resources=resources,
            backend=recipe.recipe if recipe else None,
            processor=processor.name if processor else None,
            serving=serving_summary,
        )

    def _preflight(self, config: ForgeConfig,
                   inspection: ModelInspection) -> list[Diagnostic]:
        """Environment, source and output checks that precede any selection."""
        found = [
            *inspection.diagnostics,
            *self.upstream.diagnostics,
            *selection.source_diagnostics(config),
            *selection.secondary_source_diagnostics(inspection),
        ]
        output = Path(config.output_dir).expanduser().resolve()
        source = Path(inspection.path).resolve()
        if output == source or source in output.parents or (source.is_file() and output == source.parent):
            found.append(Diagnostic("output-overlaps-input", "输出目录必须与原模型分开，且不能放在原模型目录内。", Severity.ERROR))
        elif output.exists() and (not output.is_dir() or any(output.iterdir())):
            found.append(Diagnostic("output-not-empty", "输出目录已存在且非空。请填写一个新的目录名，避免覆盖之前的产物。", Severity.ERROR))
        if config.ninfer_dir:
            required = ["ninfer-serve.exe"]
            if config.run_validation:
                required.append("ninfer.exe")
            if config.run_benchmark:
                required.append("ninfer_bench.exe")
            missing = [name for name in required if not (Path(config.ninfer_dir) / name).is_file()]
            if missing:
                found.append(Diagnostic("engine-missing", "引擎目录缺少：" + "、".join(missing), Severity.ERROR,
                                        ("在第一步选择包含 NInfer 可执行文件与 DLL 的目录。",)))
        if config.device == "cuda" and self.upstream.cuda_available is False:
            found.append(Diagnostic("torch-cuda-unavailable", "当前 Python 的 torch 无法使用 CUDA，不能在 GPU 上转换。", Severity.ERROR,
                                    ("在第二步选择 CPU 转换；生成的模型仍使用 NVIDIA GPU 推理。", "或为当前 Python 安装匹配驱动的 CUDA 版 torch。")))
        return found

    def _assess_target(self, config: ForgeConfig, inspection: ModelInspection,
                       recipe: RecipeSpec, spec_step: PlanStep,
                       diagnostics: list[Diagnostic]
                       ) -> tuple[object | None, dict | None]:
        """Serving profile and resource estimate for the selected recipe."""
        requested = config.runtime.get("speculative", "auto")
        if requested in ("mtp", "dflash2") and not spec_step.details.get(requested):
            diagnostics.append(Diagnostic("speculative-component-missing", f"模型没有兼容的 {requested.upper()} 组件，请选择自动或关闭；程序不会自行训练或替换。", Severity.ERROR))
        serving_summary = serving.serving_summary(
            recipe, inspection.config, config.runtime,
            mtp=bool(spec_step.details.get("mtp")),
            dflash2=bool(spec_step.details.get("dflash2")),
        )
        capability = to_capability(
            recipe,
            upstream_available=self.upstream.found,
            torch_available=self.upstream.torch_available,
            sources_provided=tuple(config.sources),
        )
        resources = ResourceEstimator().estimate(
            inspection, capability, config.output_dir,
            jailbreak=config.jailbreak,
            streaming=not capability.available or not recipe.needs_gpu,
            calibration=config.jailbreak == "abliteration" and not config.jailbreak_params.get("direction_file"),
            sources=config.sources,
        )
        diagnostics.extend(resources.diagnostics)
        if recipe.needs_gpu and config.device == "cpu":
            # tools.convert does accept --device cpu (torch runs it), but no
            # CPU timing for grouped_absmax has been measured on this machine,
            # so the request stays legal and is flagged rather than blocked.
            diagnostics.append(Diagnostic(
                "device-unverified-cpu",
                f"{recipe.recipe} quantizes on the selected --device; a CPU path exists "
                "but has no measured timing on this machine and is expected to be far "
                "slower than CUDA",
                Severity.WARNING,
                ("--device cuda",),
            ))
        ceiling = serving.context_ceiling(recipe, inspection.config)
        if ceiling is not None:
            requested_context = int(config.runtime.get("max_context", 32768))
            if requested_context > ceiling:
                diagnostics.append(Diagnostic(
                    "context-capped-gdn-geometry",
                    f"Serving context capped to {ceiling} (requested {requested_context}) for this "
                    "GDN bridge geometry. The local engine's chunked causal-attention "
                    "split-capacity profile is ported only up to 1024 visible keys; above "
                    "it the engine refuses to start. Native MTP speculative decoding stays "
                    "active within the cap, and the launcher, smoke test and benchmark all "
                    "use the capped context",
                    Severity.WARNING,
                    (f"max_context={ceiling}",),
                ))
        return resources, serving_summary

    # -- steps ---------------------------------------------------------------

    def _plan_jailbreak(self, config: ForgeConfig, inspection: ModelInspection,
                        steps: list[PlanStep], diagnostics: list[Diagnostic]
                        ) -> JailbreakProcessor | None:
        try:
            processor = get_processor(config.jailbreak)
        except KeyError as exc:
            diagnostics.append(Diagnostic("jailbreak-unknown", str(exc), Severity.ERROR))
            return None

        found = processor.check(inspection)
        diagnostics.extend(found)
        if processor.capability.requires_high_precision and inspection.weight_precision not in ("bf16", "fp16", "fp32"):
            diagnostics.append(Diagnostic("jailbreak-precision", "越狱需要原始 BF16/FP16/FP32 权重；当前输入不符合。", Severity.ERROR))
        if config.jailbreak == "abliteration":
            from .jailbreak.processors import _transformers_available
            direction = config.jailbreak_params.get("direction_file")
            if direction and (not isinstance(direction, str) or not Path(direction).is_file()):
                diagnostics.append(Diagnostic("direction-file-missing", "拒绝方向文件不存在，请重新选择 .npy/.pt 文件。", Severity.ERROR))
            elif not direction and not _transformers_available():
                diagnostics.append(Diagnostic("calibration-unavailable", "自动标定需要 transformers；请安装后重启，或提供已有拒绝方向向量。", Severity.ERROR))
        blocked = any(item.severity == Severity.ERROR for item in found)
        capability = processor.capability
        steps.append(PlanStep(
            "jailbreak", processor.name, "blocked" if blocked else "planned",
            {
                "supported_architectures": list(capability.supported_architectures),
                "input_precisions": list(capability.input_precisions),
                "supports_nvfp4_input": capability.supports_nvfp4,
                "requires_high_precision": capability.requires_high_precision,
                "requires_calibration": capability.requires_calibration,
                "supports_chunking": capability.supports_chunking,
                "output_precision": capability.output_precision,
                "can_quantize_afterwards": capability.can_quantize_afterwards,
                "implementation": capability.implementation,
                "params": config.jailbreak_params,
            },
        ))
        if inspection.input_kind == InputKind.NVFP4 and processor.name != "none":
            diagnostics.append(Diagnostic(
                "nvfp4-requantization-loss",
                "Applying weight modification to dequantized NVFP4 weights and then "
                "quantizing again stacks two independent losses; the manifest records this "
                "as a second quantization, not as a restore of the original precision",
                Severity.WARNING if config.allow_approximate_requantization else Severity.ERROR,
                ("Provide the original BF16/FP16 checkpoint for weight modification",
                 "Set allow_approximate_requantization to accept the stacked loss explicitly"),
            ))
        return processor

    def _plan_target(self, config: ForgeConfig, inspection: ModelInspection,
                     steps: list[PlanStep], diagnostics: list[Diagnostic]
                     ) -> RecipeSpec | None:
        candidates = self.available_recipes(inspection)
        if not candidates:
            diagnostics.append(Diagnostic(
                "no-verified-recipe",
                f"No verified conversion recipe exists for input={inspection.input_kind.value} "
                f"architecture={inspection.architecture!r}. The pinned converter (revision "
                f"{self.upstream.revision or 'unknown'}) accepts Qwen3.5 dense/MoE, "
                "Qwen3.8-Flash-Next, Gemma 4, Flint and DeepSeek-V4.1 only. A model that "
                "NInfer can *serve* is not necessarily one this converter can *produce*.",
                Severity.ERROR,
                ("Use a checkpoint with a supported architecture",
                 "Publish weights in a format an official recipe already reads"),
            ))
            return None

        recipe = selection.select_target(config, inspection, candidates, diagnostics)
        if recipe is None:
            return None
        retained = set((selection.retained_components(recipe, inspection, config.sources) or "text").split(","))
        for name in ("mtp","dflash2"):
            if (inspection.speculative_components.get(name,{}).get("detected") or config.sources.get(name)) and name not in retained:
                diagnostics.append(Diagnostic("speculative-would-be-dropped",f"当前路线无法保留输入自带的 {name.upper()}，已阻止转换；请选择能封装该组件的路线，不能静默丢弃。",Severity.ERROR))

        blocked = not (self.upstream.found and self.upstream.torch_available)
        missing = [s for s in recipe.extra_sources if s not in config.sources]
        if missing:
            diagnostics.append(Diagnostic(
                "recipe-source-missing",
                f"{recipe.recipe} requires --source {', '.join(missing)}; "
                "the recipe does not synthesize these weights",
                Severity.ERROR,
                tuple(f"--source {name}=<path>" for name in missing),
            ))
            blocked = True

        imported = [s for s in recipe.extra_sources if s in ("quantized", "gguf", "ternary")]
        if imported and config.jailbreak != "none":
            diagnostics.append(Diagnostic(
                "jailbreak-not-applied-to-imported-weights",
                f"{recipe.recipe} imports its weights from --source {', '.join(imported)}; "
                "weight edits made by the jailbreak processor on the HF base would not "
                "reach those imported tensors, so the artifact would not actually be "
                "jailbroken. TurboForge refuses to present that as a successful jailbreak",
                Severity.ERROR,
                ("Use jailbreak=none and import a published already-processed checkpoint",
                 "Jailbreak the original BF16/FP16 checkpoint without a secondary weight source"),
            ))
            blocked = True

        steps.append(PlanStep(
            "quantize+compile", recipe.recipe, "blocked" if blocked else "planned",
            {
                "label": recipe.label,
                "nominal_bits": recipe.nominal_bits,
                "typical_bpw": recipe.typical_bpw,
                "storage_formats": list(recipe.storage_formats),
                "extra_sources": list(recipe.extra_sources),
                "components": selection.retained_components(recipe, inspection, config.sources),
                "device": config.device,
                "notes": list(recipe.notes),
            },
        ))
        return recipe

    @staticmethod
    def _effective_components(recipe: RecipeSpec, inspection: ModelInspection,
                              sources: dict | None = None) -> str | None:
        """Compatibility wrapper for the selection policy's component retention."""
        return selection.retained_components(recipe, inspection, sources)

    def _speculative_step(self, config: ForgeConfig, inspection: ModelInspection, recipe: RecipeSpec | None = None) -> PlanStep:
        """MTP / DFlash2 are carried through when the input already has them.

        There is no user-facing switch: components that ship with the checkpoint
        are imported by the converter, and nothing is ever trained.
        """
        found = inspection.speculative_components
        has_mtp = bool(found.get("mtp", {}).get("detected"))
        has_dflash2 = bool(found.get("dflash2", {}).get("detected"))
        if recipe is not None:
            components = (self._effective_components(recipe, inspection, config.sources) or "text").split(",")
            has_mtp = "mtp" in components
            has_dflash2 = "dflash2" in components
        requested = config.runtime.get("speculative", "auto")
        if requested == "mtp": has_dflash2 = False
        if requested == "dflash2": has_mtp = False
        if config.runtime.get("speculative") == "none":
            return PlanStep("speculative", "已保留组件，服务时关闭推测解码以节省显存", "skipped", {"mtp": False, "dflash2": False})
        if has_mtp or has_dflash2:
            return PlanStep("speculative", "import detected components", "planned", {
                "mtp": has_mtp,
                "dflash2": has_dflash2,
                "training": "never; only existing components are imported",
                "note": "TurboForge only detects existing components; geometry compatibility "
                        "(layer count, hidden size, version) is checked by the upstream "
                        "converter at import time and is not independently re-verified here",
            })
        return PlanStep("speculative", "none detected", "skipped", {
            "note": "no MTP or DFlash2 weights found beside the checkpoint, so this "
                    "stage is skipped rather than trained",
        })

    @staticmethod
    def _empty_inspection(config: ForgeConfig) -> ModelInspection:
        return ModelInspection(path=config.input_path, input_kind=InputKind.UNKNOWN)
