"""Verified NInfer conversion recipes mirrored from tools/convert.

Every entry was read from the pinned upstream tree, not guessed. The registry is
the single place that decides what TurboForge may offer as a target precision;
a recipe absent here is reported as unsupported rather than attempted.

Upstream revision: 6eb311fd3d4c6b6a69f321de0becc4664553d76f
(the commit in BUILD-INFO.json of the installed NInfer 2026-11-08.1 package).

Recipe name -> module of record:
  tools/convert/official_recipes.py  (11)
  tools/convert/ternary.py           bonsai2_27b_ternary
  tools/convert/gguf_blocks.py       qwen3_8_27b_gguf
  tools/convert/qwen4_exp_gguf.py    qwen3_8_flash_next_gguf, qwen3_8_flash_next_gsq_q2
  tools/convert/gemma4.py            gemma4_gguf
  tools/convert/flint.py             flint_gguf
  tools/convert/deepseek_v41.py      deepseek_v41_gguf
"""

from __future__ import annotations

from dataclasses import dataclass

from ..architecture import (
    DEEPSEEK_V41,
    FLASH_NEXT,
    FLINT,
    GEMMA4,
    QWEN35_DENSE,
    QWEN35_MOE,
    architecture_kind,
    text_config,
)
from ..domain import BackendCapability, InputKind

PINNED_UPSTREAM_REVISION = "6eb311fd3d4c6b6a69f321de0becc4664553d76f"

# Architecture tuples live in turboforge.architecture (the shared vocabulary);
# they are re-exported here because this registry is their first consumer.

# tools/convert/__main__.py always opens --model with SafetensorsSource, so a
# GGUF or an NVFP4 checkpoint is never a standalone input: the Hugging Face
# safetensors directory stays --model for config.json and the tokenizer, and the
# other weights arrive as a named --source.
GGUF_IS_SECONDARY = (
    "GGUF is a secondary source, not a standalone input: the Hugging Face "
    "safetensors directory remains --model for config/tokenizer and the GGUF is "
    "passed as --source gguf="
)
NVFP4_IS_SECONDARY = (
    "NVFP4 weights arrive as --source quantized=<checkpoint>; the BF16 safetensors "
    "directory is still --model. The converter imports the packed codes verbatim "
    "(import_encoded), so no re-quantization happens on this route"
)

# Storage formats as named by tools/artifact/formats.py, used for display only.
FMT_Q4 = "q4_g64_fp16"
FMT_Q5 = "q5_g64_fp16"
FMT_Q6 = "q6_g64_fp16"
FMT_Q8 = "q8_g32_fp16"
FMT_GGUF_Q8_0 = "gguf_q8_0"
FMT_FP8 = "fp8_e4m3fn_row_bf16"
FMT_NVFP4 = "nvfp4"
FMT_T2 = "t2_g128_fp16"


@dataclass(frozen=True)
class RecipeSpec:
    """One official recipe and the exact CLI shape it requires.

    nominal_bits is the storage format's bit width; typical_bpw is the measured
    whole-model average including higher-precision tensors. They are never the
    same number and neither is the file size.
    """

    recipe: str
    label: str
    input_kinds: tuple[InputKind, ...]
    architectures: tuple[str, ...]
    storage_formats: tuple[str, ...]
    nominal_bits: float | None
    typical_bpw: float | None
    extra_sources: tuple[str, ...] = ()
    components: str | None = None
    needs_cuda: bool = False
    notes: tuple[str, ...] = ()
    # Optional explicit hidden-size whitelist for the runtime's native kernels.
    # When None, geometry_admits() applies the default rule: a dense Qwen3.5 recipe
    # whose weights are NOT the GGUF bridge is locked to the 27B geometry (hidden 5120),
    # because the groupwise Q4/Q5/Q6/Q8/T2/NVFP4 weight kernels are compile-time
    # instantiated for those row counts only. The GGUF bridge admits any K divisible by 32.
    hidden_sizes: tuple[int, ...] | None = None
    custom_function: str | None = None
    engine: str = "ninfer"

    @property
    def needs_gpu(self) -> bool:
        """Groupwise quantization runs on --device; direct import can use CPU."""
        return self.needs_cuda or not self.extra_sources


RECIPES: tuple[RecipeSpec, ...] = (
    RecipeSpec("qwen3_5_gguf_standalone", "GGUF standalone import · Q8_0 (decode then re-encode)",
               (InputKind.SAFETENSORS,), QWEN35_DENSE,
               (FMT_GGUF_Q8_0, "bf16", "fp32"), 8.0, None,
               extra_sources=("gguf",), components="text", hidden_sizes=(1024,),
               custom_function="standalone_qwen35",
               notes=("Resources are generated from the GGUF's embedded metadata; no original HF checkpoint is needed. Q8 weights are decoded and re-encoded, not a byte-for-byte lossless import.",)),
    RecipeSpec("qwen3_5_gguf_q4_0", "GGML Q4_0 · Q8 embedding and output head", (InputKind.SAFETENSORS,), QWEN35_DENSE,
               ("gguf_q4_0", FMT_GGUF_Q8_0, "bf16", "fp32"), 4.0, None, components="text",
               notes=("Real GGML Q4_0 block quantization; embedding and output head stay Q8, not AWQ.",), custom_function="q4_0", hidden_sizes=(1024,)),
    RecipeSpec("qwen3_5_gguf_q5_0", "GGML Q5_0 · Q8 embedding and output head", (InputKind.SAFETENSORS,), QWEN35_DENSE,
               ("gguf_q5_0", FMT_GGUF_Q8_0, "bf16", "fp32"), 5.0, None, components="text",
               notes=("Real GGML Q5_0 block quantization; embedding and output head stay Q8, not AWQ.",), custom_function="q5_0", hidden_sizes=(1024,)),
    RecipeSpec("qwen3_5_gguf_mxfp4", "GGML MXFP4 · Q8 embedding and output head", (InputKind.SAFETENSORS,), QWEN35_DENSE,
               ("gguf_mxfp4", FMT_GGUF_Q8_0, "bf16", "fp32"), 4.0, None, components="text",
               notes=("Real MXFP4 block floating point; not NVFP4 or AWQ.",), custom_function="mxfp4", hidden_sizes=(1024,)),
    # --- Route 1: original BF16/FP16 safetensors, dense Qwen3.5 ------------
    RecipeSpec(
        "qwen3_8_27b", "Qwen3.8-27B groupwise Q4/Q5",
        (InputKind.SAFETENSORS,), QWEN35_DENSE,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), 4.0, 4.9,
        needs_cuda=True,
        notes=("Official dense recipe: Q8 vocab, Q6 head, Q4 attention, Q5 the rest",
               "grouped_absmax runs on the selected --device"),
    ),
    RecipeSpec(
        "qwen3_8_27b_q6", "Qwen3.8-27B groupwise, Q6 gate/up",
        (InputKind.SAFETENSORS,), QWEN35_DENSE,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), 4.5, 5.3,
        needs_cuda=True,
        notes=("Same shape as qwen3_8_27b with Q6 gate/up: slower and larger, better quality",),
    ),
    RecipeSpec(
        "qwen3_6_27b", "Qwen3.6-27B groupwise",
        (InputKind.SAFETENSORS,), QWEN35_DENSE,
        (FMT_Q6, FMT_Q5, FMT_Q4), 4.0, 4.9,
        needs_cuda=True,
        notes=("Q6 vocab and head; requires the Qwen3.6-27B geometry",),
    ),
    RecipeSpec(
        "qwen3_8_27b_imatrix", "Qwen3.8-27B imatrix-guided Q4",
        (InputKind.SAFETENSORS,), QWEN35_DENSE,
        (FMT_Q6, FMT_Q5, FMT_Q4), 4.0, 4.7,
        extra_sources=("imatrix",), needs_cuda=True,
        notes=("grouped_search with signed scales, weighted by an imatrix file built with "
               "tools.convert.imatrix; sized for one 24 GB card",),
    ),
    RecipeSpec(
        "qwen3_5_gguf", "Qwen3.5 small-model GGUF Q8_0 bridge (MTP-capable)",
        (InputKind.SAFETENSORS,), QWEN35_DENSE,
        (FMT_GGUF_Q8_0, "bf16", "fp32"), 8.0, None,
        components="text",
        notes=("Geometry-generic route for small Qwen3.5 dense checkpoints (e.g. 0.8B, hidden "
               "1024): every projection is stored as GGML Q8_0 and runs on the GGUF bridge, "
               "because the native groupwise Q4/Q5/Q6/Q8 weight kernels are compile-time "
               "locked to the 27B/35B row counts. GDN a/b control stays BF16",
               "MTP is imported when the checkpoint carries it (--components text,mtp); the "
               "engine's MTP attention-input path was ported to the 0.8B geometry, so native "
               "speculative decoding runs. Measured decode gain for the 0.8B is modest (~1.09x, "
               "426->466 tok/s on an RTX 5070 Ti Laptop): the tiny baseline is already near the "
               "memory-bandwidth limit and the large-vocab draft head adds overhead. Speculation "
               "helps large models far more (27B ~3.4x); it is not a guaranteed win at this size",
               "Quantization is CPU torch/numpy (--device cpu); no CUDA toolkit needed to convert"),
    ),
    # --- Route 1: original BF16/FP16 safetensors, MoE ----------------------
    RecipeSpec(
        "qwen3_6_35b_a3b", "Qwen3.6-35B-A3B MoE groupwise",
        (InputKind.SAFETENSORS,), QWEN35_MOE,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), 4.0, 5.3,
        needs_cuda=True,
        notes=("Q8 vocab and non-expert matrices; per-layer Q4/Q5/Q6 routed experts",),
    ),
    RecipeSpec(
        "qwen3_8_flash_next", "Qwen3.8-Flash-Next groupwise",
        (InputKind.SAFETENSORS,), FLASH_NEXT,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), 4.0, 5.4,
        components="text,ngram",
        needs_cuda=True,
        notes=("Accepts BF16 or block-FP8 HF checkpoints; the n-gram table comes from "
               "--source ngram= and is written as FP8",),
    ),
    # --- Route 2: NVFP4 checkpoint, imported verbatim ----------------------
    # The inspected --model stays the original BF16 directory (recipes read the
    # BF16 embedding and other non-packed tensors from it); the NVFP4 checkpoint
    # is --source quantized. NVFP4 input kind is also listed so an inspected
    # NVFP4 checkpoint sees these targets, but the planner refuses it as a
    # standalone --model (secondary-source diagnostics).
    RecipeSpec(
        "qwen3_8_27b_nvfp4", "Qwen3.8-27B NVFP4 import (compressed-tensors)",
        (InputKind.SAFETENSORS, InputKind.NVFP4), QWEN35_DENSE,
        (FMT_NVFP4, FMT_FP8, FMT_Q8), 4.25, 4.5,
        extra_sources=("quantized",),
        notes=(NVFP4_IS_SECONDARY,
               "NVFP4 MLPs for layers <56, per-row FP8 elsewhere, FP8 embedding"),
    ),
    RecipeSpec(
        "qwen3_8_27b_nvfp4_nvidia", "Qwen3.8-27B NVFP4 import (ModelOpt AutoQuant)",
        (InputKind.SAFETENSORS, InputKind.NVFP4), QWEN35_DENSE,
        (FMT_NVFP4, FMT_FP8, FMT_Q8), 4.25, 4.5,
        extra_sources=("quantized",),
        notes=(NVFP4_IS_SECONDARY,
               "nvidia/Qwen3.8-27B-NVFP4 layout: every text MLP is NVFP4, attention and "
               "GDN are per-row FP8; the NVFP4 output head is dequantised and re-quantised "
               "to FP8 because the runtime registers the vocabulary projection for FP8 only"),
    ),
    RecipeSpec(
        "qwen3_8_27b_nvfp4_orcarouter", "Qwen3.8-27B NVFP4 import (GPTQ compressed-tensors)",
        (InputKind.SAFETENSORS, InputKind.NVFP4), QWEN35_DENSE,
        (FMT_NVFP4, FMT_FP8), 4.25, 4.5,
        extra_sources=("quantized",),
        notes=(NVFP4_IS_SECONDARY,
               "orcarouter/Qwen3.8-27B-Uncensored-NVFP4 layout; MLPs of layers 0..55 are "
               "NVFP4 packed, the last eight layers' MLPs and all attention/GDN are per-row "
               "FP8, embedding and head stay BF16 and are re-quantised"),
    ),
    RecipeSpec(
        "qwen3_6_27b_nvfp4", "Qwen3.6-27B NVFP4 import",
        (InputKind.SAFETENSORS, InputKind.NVFP4), QWEN35_DENSE,
        (FMT_NVFP4, FMT_Q8), 4.25, 4.5,
        extra_sources=("quantized",),
        notes=(NVFP4_IS_SECONDARY,
               "Sensitive early-layer attention projections keep their direct representation"),
    ),
    RecipeSpec(
        "qwen3_6_35b_a3b_nvfp4", "Qwen3.6-35B-A3B MoE NVFP4 import",
        (InputKind.SAFETENSORS, InputKind.NVFP4), QWEN35_MOE,
        (FMT_NVFP4, FMT_Q8, FMT_Q6), 4.25, 4.5,
        extra_sources=("quantized",),
        notes=(NVFP4_IS_SECONDARY,
               "Routed and shared experts are imported as NVFP4; everything else keeps the "
               "groupwise representation so the two MoE artifacts differ in one mechanism"),
    ),
    # --- Bonsai 2 Ternary: imported, never trained by TurboForge -----------
    RecipeSpec(
        "bonsai2_27b_ternary", "Bonsai 2 Ternary 27B (PQ2_0 import)",
        (InputKind.SAFETENSORS,), QWEN35_DENSE,
        (FMT_T2, FMT_Q4, FMT_Q8), 2.0, 2.4,
        extra_sources=("ternary",),
        notes=("Fastest verified 27B target at 2.125 bits: requires "
               "--source ternary=Ternary-Bonsai-2-27B-PQ2_0.gguf and the exact "
               "Qwen3.8-27B Dense geometry (hidden 5120, 64 layers). TurboForge imports "
               "published ternary weights and Hadamard sign vectors; it never trains them. "
               "Not every model has such a GGUF, so Bonsai 2 is a candidate, not a default",),
    ),
    # --- GGUF block import (secondary source beside the HF checkpoint) -----
    # The inspected --model is always the HF safetensors directory (the converter
    # opens it with SafetensorsSource), so these recipes match SAFETENSORS input
    # and become runnable only when --source gguf= is supplied. They also match
    # InputKind.GGUF so the GGUF header adapter can list them, but a standalone
    # GGUF --model is refused by the planner's secondary-source diagnostics.
    RecipeSpec(
        "qwen3_8_27b_gguf", "Qwen3.8-27B GGUF block import",
        (InputKind.SAFETENSORS, InputKind.GGUF), QWEN35_DENSE,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), None, None,
        extra_sources=("gguf",),
        notes=(GGUF_IS_SECONDARY,
               "Weight formats come from the GGUF blocks themselves, so precision is decided "
               "by the source file, not by this recipe; an -mtp GGUF is needed for the MTP "
               "component. Requires the Qwen3.8-27B Dense geometry"),
    ),
    RecipeSpec(
        "qwen3_8_flash_next_gguf", "Qwen3.8-Flash-Next GGUF import",
        (InputKind.SAFETENSORS, InputKind.GGUF), FLASH_NEXT,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), None, None,
        extra_sources=("gguf",), components="text,ngram",
        notes=(GGUF_IS_SECONDARY,),
    ),
    RecipeSpec(
        "qwen3_8_flash_next_gsq_q2", "Flash-Next native Q2 (Q2_0 expert source only)",
        (InputKind.SAFETENSORS, InputKind.GGUF), FLASH_NEXT,
        ("q2_native",), 2.0, None,
        extra_sources=("gguf",), components="text,ngram",
        notes=(GGUF_IS_SECONDARY,
               "Native Q2 codec route; lowest verified bpw for Flash-Next"),
    ),
    RecipeSpec(
        "gemma4_gguf", "Gemma 4 GGUF import",
        (InputKind.SAFETENSORS, InputKind.GGUF), GEMMA4,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), None, None,
        extra_sources=("gguf",),
        notes=(GGUF_IS_SECONDARY,
               "--components text,mtp adds the assistant from --source mtp=, "
               "text,vision the tower from --source vision=MMPROJ.gguf"),
    ),
    RecipeSpec(
        "flint_gguf", "Flint GGUF import",
        (InputKind.SAFETENSORS, InputKind.GGUF), FLINT,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), None, None,
        extra_sources=("gguf",),
        notes=(GGUF_IS_SECONDARY, "Flint text component only"),
    ),
    RecipeSpec(
        "deepseek_v41_gguf", "DeepSeek-V4.1 GGUF + Engram import",
        (InputKind.SAFETENSORS, InputKind.GGUF), DEEPSEEK_V41,
        (FMT_Q8, FMT_Q6, FMT_Q5, FMT_Q4), None, None,
        extra_sources=("gguf", "engram"), components="text,engram",
        notes=(GGUF_IS_SECONDARY,
               "Text from --source gguf=SHARD1.gguf and the Engram table from "
               "--source engram=ENGRAM.gguf; optional dspark and vision sources"),
    ),
)

ALL_RECIPE_NAMES = tuple(r.recipe for r in RECIPES)


def recipes_for(input_kind: InputKind, architecture: str) -> list[RecipeSpec]:
    """Verified candidates for one inspected model. Empty means unsupported."""
    return [
        recipe for recipe in RECIPES
        if input_kind in recipe.input_kinds and architecture in recipe.architectures
    ]


# The native groupwise weight GEMMs (Q4/Q5/Q6/Q8/T2/NVFP4/FP8 dense matmul) are
# compile-time instantiated for the 27B dense and 35B-A3B MoE row counts only.
# The GGUF Q8_0 bridge (FMT_GGUF_Q8_0) instead runs on GGML kernels that accept
# any K divisible by 32, so it is the geometry-generic route for small models.
NATIVE_DENSE_HIDDEN = 5120  # Qwen3.8-27B dense
NATIVE_MOE_HIDDEN = 2048    # Qwen3.6-35B-A3B MoE


def geometry_admits(recipe: RecipeSpec, hidden_size: int | None) -> bool:
    """Whether the runtime's kernels can serve this recipe at this hidden size.

    A recipe absent from this gate would convert but fail to load, so the planner
    filters candidates with it rather than offering a target the engine cannot
    run. hidden_size None means the inspection could not read it; we do not invent
    a block on missing data and leave geometry validation to the upstream converter.
    """
    if hidden_size is None:
        return True
    if recipe.hidden_sizes is not None:
        return hidden_size in recipe.hidden_sizes
    if FMT_GGUF_Q8_0 in recipe.storage_formats:
        return True  # generic GGML Q8_0 bridge, any K % 32 == 0
    if recipe.architectures == QWEN35_DENSE:
        return hidden_size == NATIVE_DENSE_HIDDEN
    if recipe.architectures == QWEN35_MOE:
        return hidden_size == NATIVE_MOE_HIDDEN
    return True


# The local engine's GDN causal-attention split-capacity profile is ported for the
# 0.8B-class geometry (16 linear value heads) only up to the ChunkedSmallT crossover
# at 1024 visible keys (kThreeChunkPromptVisibleKeys in causal_softmax_attention.cpp).
# Above it, target finalization throws "causal_softmax_attention split capacity:
# invalid profile" at init, so a launcher/smoke/bench asking for more context fails
# to start. Native MTP speculation is fully active at or below this ceiling (measured:
# serve decode 875 tok/s, mtp accepted 11/11). Cap serving context honestly rather
# than emit a launcher that crashes; long-context support needs the chunked attention
# profile ported. See docs/NINFER_CUDA_BUILD_AND_CONVERT.md.
GDN_BRIDGE_CONTEXT_CEILING = 1024


def serving_context_ceiling(recipe: RecipeSpec, config: dict) -> int | None:
    """Max serving context the local engine supports for this geometry, or None.

    None means no known cap: the native 27B/35B geometries are tuned for long
    context. The geometry is read from the same text_config the converter uses.
    """
    if FMT_GGUF_Q8_0 not in recipe.storage_formats:
        return None
    text = text_config(config)
    if text.get("hidden_size") == 1024 and text.get("linear_num_value_heads") == 16:
        return GDN_BRIDGE_CONTEXT_CEILING
    return None


# -- GPU-native format presentation -----------------------------------------
# Which storage family this recipe's weights belong to. Presentation only: the
# notes below state whether the detected tensor cores execute the format
# natively, never a measured speedup, and never override an explicit target.
_FP_FAMILY_NATIVE_CAP = {"nvfp4": 10.0, "mxfp4": 10.0, "fp8": 8.9}
_FP_FAMILY_LABEL = {"nvfp4": "FP4", "mxfp4": "MXFP4", "fp8": "FP8"}


def scheme_family(recipe: RecipeSpec) -> str:
    """The storage family used to explain hardware-native support."""
    formats = recipe.storage_formats
    if FMT_NVFP4 in formats:
        return "nvfp4"
    if "gguf_mxfp4" in formats:
        return "mxfp4"
    if FMT_FP8 in formats:
        return "fp8"
    if "ternary" in recipe.extra_sources or FMT_T2 in formats:
        return "ternary"
    if FMT_GGUF_Q8_0 in formats or "gguf" in recipe.extra_sources:
        return "gguf"
    return "groupwise"


def gpu_scheme_note(recipe: RecipeSpec, compute_cap: float | None) -> str | None:
    """One-line hardware note for the UI, or None for formats without one."""
    if compute_cap is None:
        return None
    family = scheme_family(recipe)
    required = _FP_FAMILY_NATIVE_CAP.get(family)
    if required is None:
        return None
    label = _FP_FAMILY_LABEL[family]
    if compute_cap >= required:
        return f"Native {label} acceleration on this GPU"
    return f"No native {label} on this GPU; dequantization is required and speedup is not guaranteed"


def gpu_scheme_order(recipe: RecipeSpec, compute_cap: float | None) -> tuple[int, float]:
    """Sort key: hardware-native formats first, then lowest bits-per-weight."""
    bpw = recipe.typical_bpw if recipe.typical_bpw is not None else 99.0
    required = _FP_FAMILY_NATIVE_CAP.get(scheme_family(recipe))
    native = 0 if (compute_cap is not None and required is not None
                   and compute_cap >= required) else 1
    return (native, bpw)


def get_recipe(name: str) -> RecipeSpec:
    from .hf_quantization import HF_RECIPES
    for recipe in (*RECIPES, *HF_RECIPES):
        if recipe.recipe == name:
            return recipe
    raise KeyError(
        f"Unknown recipe {name!r}; verified recipes: {', '.join(ALL_RECIPE_NAMES)}"
    )


def to_capability(
    recipe: RecipeSpec,
    *,
    upstream_available: bool,
    torch_available: bool,
    sources_provided: tuple[str, ...] = (),
) -> BackendCapability:
    """Turn a registry entry into a capability declaration for the plan/CLI.

    Availability is decided by facts that can be checked now: the upstream tree,
    its torch dependency, and whether the recipe's required extra sources were
    supplied. A recipe that is merely unexercised on this machine stays marked
    verified_conversion=True (it is official code) but may still be unavailable.
    """
    missing = tuple(s for s in recipe.extra_sources if s not in sources_provided)
    if recipe.engine == "gptqmodel":
        return BackendCapability(recipe.recipe,recipe.label,recipe.input_kinds,recipe.architectures,
                                 recipe.nominal_bits,recipe.typical_bpw,"GPTQModel dedicated environment",
                                 True,verified_conversion=False,notes=recipe.notes)
    reasons = []
    if not upstream_available:
        reasons.append("the NInfer source tree with tools.convert is not available")
    if not torch_available:
        reasons.append("torch is not installed, which tools.convert imports")
    if missing:
        reasons.append("missing required source(s): " + ", ".join(missing))
    return BackendCapability(
        name=recipe.recipe,
        target=recipe.label,
        input_kinds=recipe.input_kinds,
        architectures=recipe.architectures,
        nominal_bits=recipe.nominal_bits,
        typical_bpw=recipe.typical_bpw,
        implementation=f"python -m tools.convert --recipe {recipe.recipe}",
        available=not reasons,
        unavailable_reason="; ".join(reasons) or None,
        verified_conversion=True,
        notes=recipe.notes,
    )
