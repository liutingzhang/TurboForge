"""Serving profile shared by the plan preview, the generated launcher and the bench.

These are serving choices, not conversion choices: the artifact bytes are identical
regardless. Computing them in one place keeps the preview, run.bat and ninfer_bench
in agreement, so the UI can show exactly what the build will run without drift.
"""

from __future__ import annotations

from .backends.registry import RecipeSpec, serving_context_ceiling
from .local_endpoint import default_serving_port

# Runtime flags emitted into the generated launcher. They mirror the verified
# tuned profile of the installed package. Speculation is added only when the build
# actually imported a draft component, because --spec on an artifact without one
# is refused at startup rather than ignored.
LAUNCHER_FLAGS = (
    "--kv-dtype rk4v4",
    "--prefill-cublas",
    "--prefill-chunk 2048",
    "--gdn-state-fp16",
    "--embedding-q4",
    "--device-profile auto",
    "--wddm-evictable-budget",
    "--host-cache-mib 0",
)

# --embedding-q4 requantizes the token embedding to Q4 at load, which the engine
# only allows for native row-split Q8_G32 storage. The GGUF Q8_0 bridge stores
# GGML Q8_0, so serve refuses the flag there; it is omitted on that route.
NATIVE_ONLY_LAUNCHER_FLAGS = ("--embedding-q4",)
SERVER_ONLY_FLAGS = ("--device-profile", "--host-cache-mib")

# Verified token-embedding assignments in official_recipes.py. Ternary and
# GGUF imports, FP8 NVFP4 embeddings, Q6 and imatrix Q4 embeddings do not meet
# --embedding-q4's native row-split Q8 storage contract.
EMBEDDING_Q4_RECIPES = frozenset(("qwen3_8_27b", "qwen3_8_27b_q6",
    "qwen3_6_35b_a3b", "qwen3_6_27b_nvfp4", "qwen3_6_35b_a3b_nvfp4"))


def context_ceiling(recipe: RecipeSpec | None, config: dict) -> int | None:
    """Max serving context the engine supports for this geometry, or None."""
    if recipe is None:
        return None
    return serving_context_ceiling(recipe, config)


def kv_dtype(recipe: RecipeSpec | None, config: dict) -> str:
    """KV storage chosen for the geometry's MTP speed/VRAM tradeoff.

    On the small GDN GGUF bridge the KV cache is tiny (~14 MiB at 1024 ctx), so
    rk4v4 saves negligible VRAM but measurably cuts MTP acceptance and decode
    throughput (greedy 782->527 tok/s on the 0.8B). bf16 KV is faster there. The
    native 27B/35B route keeps rk4v4 because its KV cache is large enough for the
    VRAM saving to matter.
    """
    return "bf16" if context_ceiling(recipe, config) is not None else "rk4v4"


def spec_choice(mtp: bool, dflash2: bool) -> tuple[str, int] | None:
    """The drafter to serve, as (backend, draft_tokens), or None.

    DFlash2 is the default priority when both exist, not a claim that it is
    faster for every model. Explicit choices filter the candidates beforehand.
    Nothing is trained; only imported components run.
    """
    if dflash2:
        return ("dflash2", 7)
    if mtp:
        return ("mtp", 3)
    return None


def effective_context(recipe: RecipeSpec | None, config: dict, runtime: dict) -> int:
    """Serving context clamped to what the engine supports for this geometry."""
    requested = int(runtime.get("max_context", 32768))
    ceiling = context_ceiling(recipe, config)
    return min(requested, ceiling) if ceiling is not None else requested


def launcher_flags(recipe: RecipeSpec | None, config: dict,
                   mtp: bool, dflash2: bool) -> list[str]:
    """Serving flags valid for this artifact's storage route, plus speculation."""
    flags = list(LAUNCHER_FLAGS)
    if recipe is None or recipe.recipe not in EMBEDDING_Q4_RECIPES:
        flags = [f for f in flags if f not in NATIVE_ONLY_LAUNCHER_FLAGS]
    kv = kv_dtype(recipe, config)
    flags = [f"--kv-dtype {kv}" if f.startswith("--kv-dtype") else f for f in flags]
    choice = spec_choice(mtp, dflash2)
    if choice is not None:
        backend, draft_tokens = choice
        flags += [f"--spec {backend}", f"--draft-tokens {draft_tokens}"]
    return flags


def serving_summary(recipe: RecipeSpec | None, config: dict, runtime: dict,
                    *, mtp: bool, dflash2: bool) -> dict:
    """The full serving profile for the plan preview (read-only in the UI).

    Derived from the same geometry facts the launcher and benchmark use, so the
    preview never disagrees with what the build actually runs. ``mtp`` and
    ``dflash2`` are the components the build will actually import.
    """
    choice = spec_choice(mtp, dflash2)
    ceiling = context_ceiling(recipe, config)
    return {
        "kv_dtype": kv_dtype(recipe, config),
        "spec": choice[0] if choice else None,
        "draft_tokens": choice[1] if choice else None,
        "max_context": effective_context(recipe, config, runtime),
        "context_ceiling": ceiling,
        "max_concurrency": int(runtime.get("max_concurrency", 1)),
        "port": int(runtime.get("port") or default_serving_port()),
        "embedding_q4": recipe is not None and recipe.recipe in EMBEDDING_Q4_RECIPES,
        "launcher_flags": launcher_flags(recipe, config, mtp, dflash2),
    }


def perplexity_flags(flags: list[str]) -> tuple[str, ...]:
    """Only quality-affecting flags accepted by the pinned scorer CLI."""
    import shlex
    supported = {"--kv-dtype", "--prefill-cublas", "--gdn-state-fp16", "--embedding-q4"}
    return tuple(part for flag in flags if flag.split()[0] in supported for part in shlex.split(flag))
