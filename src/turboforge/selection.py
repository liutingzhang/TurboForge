"""Target selection: which verified recipe a request resolves to.

Kept separate from the planner's pipeline so the policy is one readable seam.
Selection only ranks and validates recipes the registry already declared legal
for the inspected model, plus the input's own import route; it never invents a
conversion path, and an unsupported request becomes an error diagnostic naming
the verified alternatives rather than a silent substitution.
"""

from __future__ import annotations

from pathlib import Path

from .adapters.gguf import GGUFAdapter
from .backends.registry import (
    ALL_RECIPE_NAMES,
    FLASH_NEXT,
    FMT_GGUF_Q8_0,
    QWEN35_DENSE,
    QWEN35_MOE,
    RecipeSpec,
)
from .domain import Diagnostic, ForgeConfig, InputKind, ModelInspection, Severity


def select_target(config: ForgeConfig, inspection: ModelInspection,
                  candidates: list[RecipeSpec],
                  diagnostics: list[Diagnostic]) -> RecipeSpec | None:
    """Resolve ``config.target`` against the verified candidates."""
    if config.target == "auto":
        return auto_select(config, inspection, candidates, diagnostics)
    for recipe in candidates:
        if recipe.recipe == config.target:
            return recipe
    diagnostics.append(Diagnostic(
        "target-incompatible",
        f"Target {config.target!r} is not verified for input={inspection.input_kind.value} "
        f"architecture={inspection.architecture!r}. Verified for this model: "
        + (", ".join(r.recipe for r in candidates) or "none")
        + f". All verified recipes in this build: {', '.join(ALL_RECIPE_NAMES)}",
        Severity.ERROR,
        tuple(r.recipe for r in candidates),
    ))
    return None


def auto_select(config: ForgeConfig, inspection: ModelInspection,
                candidates: list[RecipeSpec],
                diagnostics: list[Diagnostic]) -> RecipeSpec:
    """Default policy: Bonsai 2 ternary first, then the input's own import
    route, then the lowest-bpw verified groupwise target.

    Ranking only chooses among recipes that are already legal for this model
    and whose extra sources were supplied; it never invents a path.
    """
    def bits(recipe: RecipeSpec) -> float:
        return recipe.typical_bpw if recipe.typical_bpw is not None else 99.0

    pool = [r for r in candidates if r.engine == "ninfer"
            and all(s in config.sources for s in r.extra_sources)]
    if not pool:
        diagnostics.append(Diagnostic(
            "no-runnable-target",
            "Every verified target for this model needs a source that was not supplied "
            "(quantized / ternary / gguf / imatrix)",
            Severity.ERROR,
            tuple(f"--source {s}=<path>" for r in candidates for s in r.extra_sources),
        ))
        return candidates[0]

    ternary = [r for r in pool if "ternary" in r.extra_sources]
    gguf = [r for r in pool if "gguf" in r.extra_sources]
    nvfp4 = [r for r in pool if "quantized" in r.extra_sources]
    if ternary:
        chosen = ternary[0]
        diagnostics.append(Diagnostic(
            "auto-selected-bonsai2",
            f"Default policy selected {chosen.recipe} (Bonsai 2 ternary, "
            f"~{chosen.typical_bpw} bpw): the published ternary source was supplied",
            Severity.INFO,
        ))
        return chosen
    if gguf:
        chosen = gguf[0]
        diagnostics.append(Diagnostic(
            "auto-selected-gguf",
            ("从 GGUF 解码并重编码为 Q8_0；配置和分词器可从内嵌元数据生成，质量未验证。"
             if chosen.recipe == 'qwen3_5_gguf_standalone' else
             f"A GGUF source was supplied, so the default is {chosen.recipe}: blocks are imported verbatim at the GGUF's own precision"),
            Severity.INFO,
        ))
        return chosen
    if nvfp4:
        chosen = nvfp4[0]
        diagnostics.append(Diagnostic(
            "auto-selected-nvfp4",
            f"An NVFP4 source was supplied, so the default is {chosen.recipe}: packed "
            "codes are imported verbatim; pick another NVFP4 layout recipe explicitly "
            "if the checkpoint uses a different producer layout",
            Severity.INFO,
        ))
        return chosen

    bridge_pool = all(FMT_GGUF_Q8_0 in r.storage_formats for r in pool)
    chosen = (next((r for r in pool if r.recipe == "qwen3_5_gguf"), min(pool, key=bits))
              if bridge_pool else min(pool, key=bits))
    if FMT_GGUF_Q8_0 in chosen.storage_formats:
        diagnostics.append(Diagnostic(
            "auto-selected-gguf-bridge",
            f"Default is {chosen.recipe}: the geometry-generic GGUF Q8_0 bridge. The "
            "native groupwise Q4/Q5/Q6/Q8 weight kernels are compile-time locked to the "
            "27B/35B geometries, so a small dense checkpoint runs on the GGML Q8_0 bridge "
            "instead. MTP speculative decoding is imported when the checkpoint carries it",
            Severity.INFO,
        ))
    else:
        diagnostics.append(Diagnostic(
            "auto-selected-lowest-bpw",
            f"No ternary/GGUF/NVFP4 source supplied; default is the lowest-bpw verified "
            f"groupwise target {chosen.recipe} (~{chosen.typical_bpw} bpw). Supply "
            "--source ternary=<published PQ2_0 GGUF> to make Bonsai 2 (2 bit) the default",
            Severity.INFO,
        ))
    return chosen


def retained_components(recipe: RecipeSpec, inspection: ModelInspection,
                        sources: dict | None = None) -> str | None:
    """The ``--components`` value the converter will actually receive.

    tools.convert.qwen3_5 imports the base checkpoint's embedded mtp.* tensors
    only when --components includes mtp; the recipe default is text, so a
    detected MTP component is appended here. This is computed in the plan (not
    only at execution) so the preview and the build never disagree.
    """
    sources = sources or {}
    base = recipe.components
    if recipe.architectures == FLASH_NEXT and base and not sources.get("ngram"):
        base = ",".join(part for part in base.split(",") if part != "ngram")
    has_mtp = bool(inspection.speculative_components.get("mtp", {}).get("detected"))
    bundled_mtp = inspection.speculative_components.get("mtp", {}).get("source_path")
    import_key = next((key for key in ("gguf", "ternary")
                       if key in recipe.extra_sources and sources.get(key)), None)
    if import_key:
        try:
            imported = GGUFAdapter().inspect(Path(sources[import_key]))
        except (ValueError, OSError):
            # The source diagnostic reports this format error to the user.
            has_mtp = False
        else:
            has_mtp = bool(imported.speculative_components.get("mtp", {}).get("detected"))
    if (has_mtp or sources.get("mtp") or bundled_mtp) and recipe.architectures in (
            QWEN35_DENSE, QWEN35_MOE, FLASH_NEXT):
        parts = (base or "text").split(",")
        if "mtp" not in parts:
            parts.append("mtp")
        base = ",".join(parts)
    if sources.get("mtp") and recipe.recipe == "gemma4_gguf":
        base = (base or "text") + ",mtp"
    has_dflash2 = bool(inspection.speculative_components.get("dflash2", {}).get("detected"))
    if (sources.get("dflash2") or has_dflash2) and recipe.architectures in (QWEN35_DENSE, QWEN35_MOE):
        base = (base or "text") + ",dflash2"
    return base


def source_diagnostics(config: ForgeConfig) -> list[Diagnostic]:
    """Validate supplied named sources before they reach the converter."""
    diagnostics = []
    for name, value in config.sources.items():
        path = Path(value).expanduser()
        if not path.exists():
            diagnostics.append(Diagnostic(
                "source-missing",
                f"--source {name}={value} does not exist",
                Severity.ERROR,
            ))
        elif name == "gguf":
            try:
                GGUFAdapter().inspect(path)
            except (ValueError, OSError) as exc:
                diagnostics.append(Diagnostic("gguf-source-invalid", f"GGUF 来源无法读取：{exc}", Severity.ERROR))
    return diagnostics


def secondary_source_diagnostics(inspection: ModelInspection) -> list[Diagnostic]:
    """GGUF and NVFP4 checkpoints are secondary sources, never --model.

    Verified against the pinned upstream: the GGUF block recipes read every
    text tensor from --source gguf and need only config/tokenizer from
    --model; the NVFP4 recipes read the BF16 embedding (and other non-packed
    tensors) from --model, so the original BF16/FP16 directory is required.
    """
    if inspection.input_kind == InputKind.GGUF:
        return [Diagnostic(
            "gguf-secondary-source",
            "A GGUF cannot be converted standalone: the pinned converter takes the "
            "original Hugging Face directory as --model (for this route only its "
            "config/tokenizer files are needed) and the GGUF as --source gguf=. The "
            "bit width is decided by the GGUF itself and is imported verbatim; to "
            "change precision, convert the original BF16/FP16 checkpoint instead",
            Severity.ERROR,
            ("--input <original HF dir> --source gguf=<this file>",
             "Use the original BF16/FP16 checkpoint to pick another quantization"),
        )]
    if inspection.input_kind == InputKind.NVFP4:
        return [Diagnostic(
            "nvfp4-secondary-source",
            "An NVFP4 checkpoint is imported verbatim via --source quantized=; the "
            "original BF16/FP16 directory must stay --model because the recipes read "
            "the BF16 embedding and other non-packed tensors from it. NVFP4 output is "
            "4.25 bit and cannot be re-quantized to other widths on this route",
            Severity.ERROR,
            ("--input <original BF16 dir> --source quantized=<this checkpoint>",
             "Use the original BF16/FP16 checkpoint to pick another quantization"),
        )]
    return []
