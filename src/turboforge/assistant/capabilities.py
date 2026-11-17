"""Public static facts derived from executable capability rules, never page text."""
from __future__ import annotations

from ..backends.hf_quantization import (
    AWQ_DISABLED_MODEL_TYPES, AWQ_DISABLED_REASON, GPTQMODEL_VERSION,
    HF_ACCEL_VERSIONS, HF_CORE_VERSIONS, HF_DEFAULT_BACKEND,
)
from ..evaluation import AUTO_EVALUATION_TOKENS, WIKITEXT_REVISION


def capability_entries() -> list[dict]:
    source = ["src/turboforge/backends/hf_quantization.py"]
    return [
        {"id": "capability.local_model_input", "title": "Valid local model path not detected, missing config.json, and what to put in Original model source",
         "body": "A local path means the PC running the TurboForge service. Click Select folder / Select GGUF file to fill the path, or paste it manually; paired quotes and leading/trailing spaces from Windows copy-as-path are supported, and spaces inside the directory are preserved. A path existing does not mean the model files are complete or the conversion route is supported. Put a complete original HF checkpoint in \"Model\" and leave its \"Original model source\" empty; for GGUF pick the specific file. Supported self-contained files generate their resources automatically; other routes need a matching config.json and tokenizer.json/tokenizer.model, which can come from a separate local original HF directory or an explicit original repository. Do not treat a folder containing only GGUF as the original source again; GGUF import does not require original Safetensors when only metadata is needed. A single-file Qwen3.5 dense Q8_0 GGUF with hidden_size=1024 and complete embedded metadata can generate config.json, tokenizer, and chat template directly, without an original directory; other files of unknown origin must not have their config guessed from the file name. When there is no matching import route, use the original full-precision model as the input model and re-inspect instead of pretending support by adding only config.json; unsupported hardware or models must stop explicitly.",
         "topic": "input", "aliases": ["local model invalid", "path is fine", "local directory", "config.json not found", "what to fill in original model source", "no upload does not mean no input"],
         "source_refs": ["src/turboforge/model_sources.py", "src/turboforge/inspector.py", "src/turboforge/precheck.py", "src/turboforge/preparation.py", "web/src/App.tsx", "web/src/components/PathInput.tsx"]},
        {"id": "capability.clean_gguf", "title": "A clean machine with only one GGUF: where does the original model directory come from",
         "body": "The release bundle contains no models, but a supported single-file GGUF does not need the developer's local original model directory. A complete Qwen3.5 dense Q8_0 GGUF with hidden_size=1024 can generate frontend resources in the gguf-resources cache from its embedded structure parameters, vocabulary, BPE merges, special tokens, and chat template, and convert with the GGUF as the only weight source. After selecting the GGUF, leave Original model source empty; this route is also recognized when an older version mistakenly filled the GGUF's own directory and it has no config.json. The recipe qwen3_5_gguf_standalone decodes and re-encodes to Q8_0, which is not byte-for-byte lossless, so quality must be verified separately; the current adapted engine caps context at 1024, applied automatically after the guided check. Files with missing fields, duplicate tokens, or unverified MTP stop explicitly — no guessed config and no silently dropped heads. Other architectures/precisions/shards are outside the current standalone import validation. The first run still prepares dependencies and the adapted engine; the generated resources, conversion, GPU load, and text generation of this 0.8B Q8_0 have been measured, but a full deployment on a clean PC has not.",
         "topic": "input", "aliases": ["clean environment with only one gguf", "clean PC", "no original model directory", "0.8B GGUF import", "does adding config.json make it work"],
         "source_refs": ["src/turboforge/gguf_standalone.py", "src/turboforge/model_sources.py", "src/turboforge/precheck.py", "src/turboforge/preparation.py", "src/turboforge/bootstrap.py", "src/turboforge/backends/registry.py"]},
        {"id": "capability.auto_quality", "title": "HF automatic quality gate and independent evaluation corpus",
         "body": f"In guided mode, new HF quantization enables the quality gate by default; no evaluation file needs to be provided. The default is WikiText-2 test at pinned revision {WIKITEXT_REVISION}, up to {AUTO_EVALUATION_TOKENS} tokens; calibration uses train, and hashes and scores are recorded. The reference is the CPU FP32 PPL of this run's post-processing, pre-quantization weights, which costs extra RAM/time. NInfer still needs a matching baseline artifact and corpus; a packed import cannot prove its own quality. General English PPL does not guarantee results on every task.",
         "topic": "quality", "aliases": ["quality gate evaluation file automatic evaluation independent corpus"],
         "source_refs": ["src/turboforge/evaluation.py", "src/turboforge/domain.py", "web/src/App.tsx"]},
        {"id": "capability.offline_start", "title": "Can it start without a network, and what does offline use need",
         "body": "The first start needs the internet to install missing Python/web dependencies; conversion also downloads uncached models, environments, and calibration/evaluation sets. Once fully prepared, local models and offline help work. Remote AI Q&A still needs a connection to the configured service; a clean PC cannot be promised to work offline out of the box.",
         "topic": "start", "aliases": ["offline start disconnected no network"], "source_refs": ["scripts/start_web.py", "src/turboforge/bootstrap.py"]},
        {"id": "capability.awq_restrictions", "title": "Qwen3.5 AWQ support and why it is disabled",
         "body": f"GPTQModel {GPTQMODEL_VERSION}: {AWQ_DISABLED_REASON} Disabled model_type: {', '.join(AWQ_DISABLED_MODEL_TYPES)}. Other architectures cannot claim measured quality on this basis either.",
         "topic": "quantization", "aliases": ["can Qwen3.5 do AWQ", "qwen3_5 AWQ", "AWQ option is gone"], "source_refs": source},
        {"id": "capability.hf_backend", "title": "GPTQ/AWQ default inference backend and environment dependencies",
         "body": f"The current default backend is {HF_DEFAULT_BACKEND.upper()}; GPTQModel picks an available kernel for the current device and layout, and it is not guaranteed to be faster for every model. Torch is the explicit compatibility option. Dependency versions: GPTQModel {GPTQMODEL_VERSION}, {HF_CORE_VERSIONS}; Windows acceleration dependencies: {HF_ACCEL_VERSIONS}. The script prepares an isolated environment automatically, with a PyPcre wheel for the supported Python version, so no manual C++ compiler install is needed.",
         "topic": "resources", "aliases": ["which backend does GPTQ use by default", "do I need to install a compiler myself", "AUTO Torch Triton FLA", "PyPcre torchvision"], "source_refs": source},
    ]


def public_capabilities() -> dict:
    return {entry["id"]: entry["body"] for entry in capability_entries()}