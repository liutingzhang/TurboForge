# TurboForge AI assistant knowledge draft and question coverage table

Date: 2026-11-14. Companion [refactor plan](AI_ASSISTANT_REFACTOR_PLAN.md), [diagnostic catalog](AI_ASSISTANT_DIAGNOSTICS.md). This document is a content draft to be wired in, not a live knowledge service.

## 1. Content usage contract

This document answers against the current source code; re-check it when the product or third-party versions change. In future each table row becomes one or more stable-ID knowledge entries, keeping the applicable route, code source, and evidence level; the whole document is never stuffed into every prompt turn.

Current fact sources: `precheck.py`, `planner.py`, `backends/registry.py`, `backends/hf_quantization.py`, `backends/hf_worker.py`, `serving.py`, `api/builds.py`, `reports.py`, plus the version-matched `SUPPORTED_CAPABILITIES.md / AWQ_GPTQ.md / SPECULATIVE_RETENTION.md / QUANTIZATION_HARDENING.md`. Historical counts and old route limits in existing docs must not override current code.

Answer shape: respond in one sentence first, then give 1–3 action steps; show the part relevant to the current user first and fold the theory away. When the task snapshot cannot confirm something, use conditional sentences — never pretend to have read the user's hardware, files, or reports.

## 2. Core conclusions that must stay fixed

1. TurboForge handles model preparation, conversion, validation, and managed startup; inference is executed by the corresponding NInfer/HF engine.
2. Input identified, plan runnable, conversion succeeded, GPU validation passed, quality passed, service ready, and quality fit for the purpose are different conclusions.
3. Available options depend on architecture, geometry, input format, recipe, additional sources, and the local environment; "a family is supported" does not mean every size/version is supported.
4. NInfer and HF AWQ/GPTQ are different routes with different artifacts, dependencies, and runtime capabilities; loaders are not mixed.
5. Lower bits often reduce weight storage, but do not guarantee unchanged quality, more speed, or a proportional drop in total VRAM.
6. Main bit width, average bits per weight, on-disk file size, and runtime VRAM are four different quantities.
7. Importing quantized blocks as-is does not restore original precision, and does not mean every conversion chain is completely risk-free.
8. Bonsai 2 requires an existing matching ternary source; the project does not train an arbitrary BF16 model into a ternary model.
9. Detection, retention, runtime enablement, and measured speedup of MTP/DFlash2 must each be proven separately; the current HF service does not execute them.
10. A same-corpus PPL gate covers only that test setup; it cannot guarantee code, math, long-text, or instruction-following ability.
11. Resource estimates and historical results must not pose as current measurements; unknown shows as unknown, and failure is never rewritten as success.
12. AI configuration is not conversion-model configuration: the assistant can use an external model and does not need to wait for the model being converted to run successfully.

## 3. Quantization methods and selection knowledge

### 3.1 Route comparison

| Method | What the project does now | What it needs | Main trade-offs and boundaries |
|---|---|---|---|
| BF16/FP16 original weights | Serves as full-precision input and weight-processing source | Complete HF config/tokenizer/weights | High RAM/disk cost; NInfer has no general BF16 preserve recipe, which does not mean HF cannot read a full-precision reference internally |
| NInfer native groupwise mixed quantization | Uses registered recipes to encode different tensors | Matching architecture and geometry | Q4/Q5/Q6 in the name is not necessarily a uniform bit width across the model; some kernels are locked to large-model geometry |
| GGML Q8_0 bridge | Wired for applicable geometries such as small models | Full-precision input allowed by the recipe | The current small-model auto path often picks it; this must not be explained as an 8-bit default for all models always |
| GGML Q4_0 / Q5_0 local encoding | Adapted for Qwen3.5 0.8B hidden 1024 | Matching current geometry | Embeddings and output head keep Q8 and other mixed formats; low bits need an explicit choice and must not be generalized to arbitrary sizes |
| MXFP4 local encoding | 4-bit floating-point encoding for currently wired geometries | Matching recipe and engine | Different from NVFP4 and INT4; never treat them as interchangeable just because "all are 4" |
| NVFP4 import | Validates and keeps the corresponding packed/scale layout | Complete quantized checkpoint, matching original files, explicit producer layout | About 4.25 bpw is this route's basis; the file name does not prove the layout, and BF16 true values cannot be restored |
| GGUF import | Imports supported existing quantized blocks | Complete GGUF shards and matching config/tokenizer | Precision is decided by the input; it is not a general requantizer from any GGUF to any bit |
| imatrix recipe | Uses an adapted importance matrix to assist quantization | A matrix matching the model/tensors | A matrix is not ordinary corpus; matching and actual effect need verification |
| Bonsai 2 / PQ2_0 | Imports an existing matching ternary source | The corresponding 27B Dense geometry, a published PQ2_0 GGUF | Not general INT2, and no new ternary weights are trained; other tensors/heads may still be higher precision |
| GSQ Q2 | Imports existing Q2_0 expert weights for Flash-Next | Specific model geometry and native source | Not quantizing an arbitrary MoE to 2-bit, and not the same as Bonsai ternary |
| AWQ 4-bit HF route | `hf_awq_4` generates the native format with real calibration | Original full-precision model, calibration text, isolated GPTQModel environment | An activation-aware method, not ordinary Q4; currently supports the GEMM layout, and real-GPU acceptance is still pending |
| GPTQ 2/3/4/8-bit HF route | The corresponding `hf_gptq_*` runs real calibration | Supported architecture, calibration files, isolated environment | Supported bit widths and packing are limited by code; low-bit results depend on model and data, with no guarantee of beating other algorithms |
| Existing AWQ/GPTQ retention | `hf_packed_import` validates and keeps | A complete supported packed checkpoint | No second quantization, no bit change, no full-precision weight editing; unwired vendor-reordered layouts are rejected |

The above is a product capability summary; not every combination has been measured on this machine's GPU. For specific sources and scope see the [capability matrix](SUPPORTED_CAPABILITIES.md) and [HF route](AWQ_GPTQ.md).

### 3.2 Simple selection tree

```text
Inspect the model first
  ├─ Unsupported/incomplete → fix the source or switch to a supported input; do not blindly download large files
  ├─ Existing GGUF/NVFP4 → use an available import route, keep the original precision; supply matching original files
  ├─ Existing AWQ/GPTQ → HF native retention, confirm the layout; do not send to NInfer
  └─ Original full precision
       ├─ No special target → see the auto recommendation and resource estimate from the current inspection
       ├─ Need to save VRAM → compare available lower bpw, context, and concurrency, and run a quality test
       ├─ Explicitly need AWQ/GPTQ → choose an HF recipe, accepting the isolated environment and current capability limits
       └─ Existing matching ternary source → optionally import via Bonsai; do not run a separate full-precision abliteration
```

When answering "which should I pick", look at least at: input format, actual available options, free GPU VRAM/system RAM/disk, purpose and context, and quality requirements; when something is missing, give a safe conditional suggestion first — never casually prescribe 4-bit.

## 4. User question catalog and answer points

The 131 entries below are the content coverage baseline (first version 120; the architecture-routing refactor added B11/C11; MoE engine selection added B12/C12/C13/C14/D11/J11; the three-route plain-language run steps added J12/J13/J14). Every entry should be searchable in the help center, retrievable, and linked to a business area. Fixed error codes are covered item by item in the diagnostic catalog.

### A. First use and environment (A01–A10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| A01 | I know nothing about models — what do I do first? | Double-click the launcher → pick a local model/public HF link → inspect → choose the recommendation → automatic preparation and conversion → check validation → start; explain the single necessary input for each step now |
| A02 | Must I install Python, Node, or CUDA? | The bundle and source development differ; the one-click script prepares Python/page/dependencies as needed, and a built bundle needs no Node; the driver must work, and no arbitrary environment is guaranteed to auto-repair |
| A03 | Are AMD, Mac, or CPU-only supported? | The current one-click target is Windows x64 + supported NVIDIA; a CPU conversion option exists but has no speed measurement for this model, and it does not mean all GPU inference can switch to CPU |
| A04 | How long online and how much download? | First-time dependencies/models can be large; later runs reuse the verified cache; explain stages from the task log — never estimate completion minutes from the model name |
| A05 | I already installed these dependencies — why download again? | Check the actual worker/dedicated environment and versions; system Python is not the task environment; HF and NInfer environments are separate |
| A06 | The window closes after double-click and the page will not open? | Read the startup failure text and `.turboforge/logs/`, run `Start TurboForge.bat --check`; troubleshoot unbuilt page, permissions, network, and port separately |
| A07 | 8090 will not open — is it broken? | One-click start may pick another free port; use the actual URL shown in the startup window; the model API port defaults to the port of the first origin in the .env allowlist (template 18080) and is a different service |
| A08 | Do I need administrator rights or a driver upgrade? | Read the specific error first; never elevate by default. When the driver is unsupported, follow the official driver requirements — the script does not modify drivers automatically |
| A09 | Can I close the command-line window? | The current startup window hosts the page service; closing it interrupts tasks and the owned model process may be released with it; distinguish refreshing the page from closing the service |
| A10 | Can I use it without configuring AI? | Yes; model conversion and the help center are independent; AI is an explanation entry, not a runtime prerequisite |

### B. Input, download, and model identification (B01–B12)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| B01 | What goes in the local model path, and why is an existing path still invalid? | The path belongs to the PC running the TurboForge service; select or paste it, paired quotes and leading/trailing spaces are supported, and spaces inside the directory are preserved. For a complete HF fill the model directory and leave Original model source empty; for GGUF pick the specific file. Path exists, files complete, and route supported are three independent checks — handle the explicit missing item |
| B02 | Which kind of Hugging Face link do I enter? | Use a model repository or an explicit file/revision link; with multiple GGUFs pick the specific file and keep the explicit version; not an arbitrary web page link |
| B03 | Why does a private or gated model fail to download? | Verify legitimate repository access and license; never ask the user to send an access token to the assistant, and never promise to bypass restrictions |
| B04 | Can a clean environment with only one GGUF convert? | The release bundle contains no models and cannot depend on the developer's existing directories. First check whether it is a supported standalone import; for other routes verify the original repository — a declaration is parsed automatically, otherwise the user provides the original repository link to fetch only config/tokenizer. A complete Qwen3.5-0.8B Q8_0 single file is supported to generate resources from embedded parameters/vocabulary/merges/template and import standalone, with no original directory; it decodes and re-encodes to Q8_0, is not lossless, and caps context at 1024; missing metadata or other unsupported layouts still must not guess config |
| B05 | Why does a GGUF still need the original model, and what if config.json is missing? | The supported self-contained Qwen3.5-0.8B Q8_0 GGUF generates config/tokenizer/template automatically; leave Original model source empty. For other routes the original model source should be a matching HF directory with config.json and tokenizer.json/tokenizer.model, or the original repository; a directory containing only GGUF does not qualify. GGUF import does not require original Safetensors when only metadata is read; NVFP4 may need original full-precision tensors. Supplying files still must pass architecture/size support checks — config must not be forged |
| B06 | What are shards, and can I continue with one missing? | Multiple files together form the complete checkpoint; a missing shard stops the run — supply it from the original repository; never forge tensors |
| B07 | The file name says NVFP4 — why was it not recognized? | Real evidence is required: quantization config, packed weights, scale/block layout; the name alone is not enough |
| B08 | Same model name — why can others convert it but I cannot? | Check revision, architecture/geometry, actual format, source completeness, and engine version. The small-model NInfer route needs an adapted bundle; the developer's local adaptation must not be treated as a capability of the plain official engine everywhere; an unknown environment cannot be promised as accepted |
| B09 | Can an existing `.ninfer` change bits again? | It is not a general `.ninfer` requantizer now; see the available input routes and prefer going back to original weights or a compatible import source |
| B10 | The download broke — must I restart it? | Current downloads support resume and verified-cache reuse, so it can be re-initiated; an interrupted build does not mean conversion can always resume in place — keep the diagnostics |
| B11 | How are MoE (mixture of experts) and Dense architectures identified automatically? | During inspection the backend decides MoE (mixture of experts, multi-expert FFN) or Dense from the config's architectures and expert fields (num_experts, num_local_experts, moe_intermediate_size); when it cannot decide it shows unknown. The result is used only to rank and label quantization/acceleration plans and show the processing route — it does not by itself change the artifact binary; available recipes are still decided by architecture, geometry, and input format, and being identified as MoE does not mean any MoE inference engine is wired |
| B12 | After a MoE model is produced, which inference engine is recommended dynamically by GPU model? | When identified as MoE, the backend gives a community inference-engine recommendation under the double condition "local GPU (compute_cap from nvidia-smi) + model on the engine's published list": only NVIDIA Blackwell (compute≥12.0, RTX 50 series, sm_120) AND a Basalt-published MoE (Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash) recommends Basalt (optimized for that architecture, native NVFP4, officially faster on Blackwell); MoE not on Basalt's adapted list such as Qwen3.6-35B-A3B defaults to Strata even on Blackwell, but a prominent compatibility note (compat_check, card turns warning-colored) is placed at the very top of the recommendation card: "Basalt is faster on this card, but this model is not on Basalt's adapted list (only Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.8-27B). Verify yourself whether you can obtain or convert a compatible .basalt source before deciding to switch to the faster Basalt; if not, use the default engine below" — handing the "is the faster Basalt compatible" decision to the user instead of quietly sidestepping it; if that MoE is also not on Strata's published list, the note appends "defer to both engines' official repositories". Other NVIDIA (40/30/20 series), AMD, Intel, Apple, or no NVIDIA detected recommends Strata (broad support; Basalt only runs on Blackwell), with no compatibility note then (Basalt simply cannot run); a MoE architecture on neither published list gets Strata plus a "defer to the official repositories" note. The recommendation includes reason, hardware requirements, input format, supported models, alternatives, and official links, shown above the quantization plans. This is only a selection-advice display layer: it does not integrate engines, change the .ninfer artifact, or change engine download gates; Dense and unknown get no such recommendation. Differences in C12, hardware/format in C13/C14, resource boundaries in D11, format relations in J11, plain-language run steps in J12 (Basalt)/J13 (Strata)/J14 (Dense ninfer+kvmem) |

### C. Quantization concepts and routes (C01–C14)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| C01 | What is quantization — does it change the model? | Representing weights with less information can reduce storage and may affect output; the source directory is not overwritten, and the output is a new artifact |
| C02 | How should I understand 2/3/4/5/6/8-bit? | An encoding-granularity metric, not a whole-model quality ranking; mixed plans and extra metadata make the actual bpw different |
| C03 | I picked 4-bit — why is the size not a quarter of the original? | Original dtype, mixed precision, scales, embeddings, heads, alignment, etc. affect size; reason from the real artifact, not the nominal bit width |
| C04 | Is 4-bit always faster than 8-bit? | Not necessarily; it depends on kernel, GPU, batch, context, bandwidth, and decode overhead; compare measurements under identical conditions |
| C05 | Is AWQ the same as ordinary Q4? | No; algorithm, calibration, packing, and runtime kernel differ; this project takes the isolated HF route |
| C06 | Which is better, GPTQ or AWQ? | There is no project-wide uniform conclusion; compare currently supported bit widths, task data, quality, and runtime cost — never fabricate a uniform ranking |
| C07 | Are MXFP4, NVFP4, and INT4 interchangeable? | Not just because they are 4-bit; floating-point encoding, scale, and layout contracts differ — use the exact recipe |
| C08 | Are Bonsai and GPTQ 2-bit the same thing? | No; ternary-source import and GPTQ calibrated encoding differ; available models, formats, and quality evidence are each independent |
| C09 | Is GGUF a quantization method? | It is a container/format system that can hold different tensor encodings; "GGUF" does not state a specific bit width or algorithm |
| C10 | Can restoring low bits to high bits recover ability? | Raising storage precision does not recover lost information; a second quantization may lose more — prefer the original full-precision source |
| C11 | Why are plans ranked by GPU generation, with NVFP4 recommended on RTX 50? | During inspection nvidia-smi reads compute_cap to identify the GPU generation: Blackwell (≥10.0, RTX 50 series) natively supports FP4/NVFP4, Ada/Hopper (≥8.9, RTX 40 series) natively supports FP8, Ampere (RTX 30 series) has no native FP8/FP4. The quantization plan list puts plans natively supported by this card first, then sorts by bpw, and uses gpu_note to say "native FP4 acceleration on this card" or "no native FP4 on this card; dequantization needed, not necessarily faster". Ranking and labeling are display advice only — they do not change the artifact or engine download gates, and do not replace the backend's "auto recommendation"; without GPU info it falls back to bpw sorting with no generation label. Native support is only a hardware capability; actual gains depend on measurements under identical conditions |
| C12 | What is the difference between Strata and Basalt, and which GPUs/scenarios does each suit? | Both are MIT-licensed open-source local MoE inference engines; Basalt is a fork of Strata. Strata (Niko1221/Strata) supports NVIDIA RTX 20/30/40/50, AMD, Intel Arc (Linux), Apple Silicon, takes standard GGUF, tiers experts GPU→RAM→SSD, has a one-click desktop app installer, and suits non-Blackwell/cross-platform/low-VRAM-large-RAM. Basalt (jesdga95/basalt) is NVIDIA Blackwell only (sm_120, compute 12.0) + CUDA 13 + driver R580+, takes self-contained .basalt v2, native NVFP4, officially fastest on Blackwell; its release package contains no CUDA/Python so you provide them, and it suits RTX 50 users chasing maximum speed. Both are OpenAI/Anthropic API compatible. Recommend by GPU + model: Blackwell AND model on Basalt's adapted list (Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash) → Basalt (faster); Blackwell but model not adapted (e.g. Qwen3.6-35B-A3B) → default Strata, with a prominent compatibility note at the top of the card: "Basalt is faster but not adapted; verify compatibility yourself before deciding to switch"; other GPUs → Strata (Basalt cannot run). Speed numbers are for specific models/settings only and are not generalized into a universal guarantee |
| C13 | Does Basalt only support Blackwell GPUs, and what are the hardware/format requirements? | Yes. Basalt hard-requires an NVIDIA Blackwell GPU (compute capability 12.0 / sm_120) + CUDA 13; the system must be Linux x64 (AVX2, GCC13+, CMake3.24+) or Windows 10 1903+ x64 (MSVC/VS2022, Ninja), driver R580 branch or newer (Linux minimum 580.95.05). Non-Blackwell (40/30/20 series, AMD, Intel, Apple) cannot run it — use Strata instead. The release package (basalt-vX.Y.Z-windows-x64.zip / linux-x64.tar.gz) contains no CUDA libraries or Python environment; you must provide the CUDA 13 runtime (cudart, cuBLAS) on PATH and a Python 3.10+ .venv. Input is self-contained .basalt v2 (weights/tokenizer/MTP/vision projection/int8 tables); expert formats are Unsloth UD quants, Q8_0, NVFP4. Supports Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.8-27B (dense, added in v0.4.0); does NOT include Qwen3.6-35B-A3B (that MoE is published by Strata only, so Blackwell defaults to Strata with a "Basalt is faster but not adapted; verify compatibility yourself" note at the top of the card). Experts that do not fit go to system RAM/SSD, with multi-GPU expert tiers. Start with bin/basalt serve --config basalt_config.json, default 127.0.0.1:8080 |
| C14 | Which GPUs, VRAM, and RAM tiers does Strata support? | Strata supports NVIDIA GeForce RTX 20/30/40/50 series, AMD Radeon RX 7700/7800/7900, RX 9060/9070, AI PRO R9700 (RX 6800/6900 experimental), Intel Arc (Linux), Apple Silicon M1+, AMD Ryzen AI Max Strix Halo, NVIDIA DGX Spark (Linux). A 12GB-class consumer card is enough (README measurements include RTX 5070 12GB, RTX 3060 12GB); NVIDIA needs CUDA 13/driver 580+. RAM by tier (numbers = combined weight RAM+VRAM, leave extra OS headroom, estimates): Qwen3.8-Flash-Next ISTA-DASLab Q2_0 about 37.6GB (fastest), IQ2_XS 39.2GB, IQ3_XXS 47GB, IQ3_S 54.8GB (best quality); Coder-only encoding about 32GB RAM; Unsloth 4–8bit needs 72–136GB. Experts tier GPU→RAM→SSD, takes standard GGUF; the provider recommends ISTA-DASLab→Unsloth→Gyro, with the user's explicit choice first. strata launch --plan only previews without downloading; one-click install via the desktop app or install.ps1/sh, OpenAI/Anthropic API. Supports Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, DeepSeekV4.1-Flash |

### D. Plan selection and resources (D01–D11)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| D01 | Is the auto recommendation optimal? | It is a usable default under the wired rules, not a guarantee of best quality/speed; explain this run's actual auto-selection diagnostic |
| D02 | Why is the bit width I want missing? | Limited by architecture/geometry/input/format/engine; only available recipes are shown; the AI must not create an unimplemented route |
| D03 | Can 12GB VRAM run a 27B? | It depends on the specific model/bits, context, KV, speculative components, and free VRAM; historical Bonsai results do not guarantee every 27B |
| D04 | Are conversion RAM and inference VRAM the same? | Different stages/resources; explain the basis of the planner's current estimate — an unknown conversion peak is never filled with a number |
| D05 | Why does disk need to be larger than the final file? | Original download, extraction, intermediate processing weights, temporary output, and the final file occupy space at the same time |
| D06 | What if memory is not enough? | First see whether it is RAM/VRAM/disk; close occupying programs, switch to a suitable smaller model/recipe, adjust context/concurrency; lowering bits does not guarantee solving conversion-stage RAM |
| D07 | Can offloading to CPU guarantee it runs? | Look at the specific engine's current capabilities and parameters; a resource warning must not be explained as the project already providing arbitrary offload automatically |
| D08 | Why can the output location not be reused? | To prevent overwriting the original or existing artifacts; create a new separate empty output directory — never "solve" it by deleting the original model |
| D09 | I want quality first — how should I choose? | Start from the currently available higher-fidelity plans and a same-corpus baseline, and test your own task; do not treat bit width as the only criterion |
| D10 | Do parameter changes require re-conversion? | Distinguish weight quantization/processing parameters from runtime parameters; weight changes need a redo, serving parameters usually do not change weights, but explain the current generation-script/plan binding per the actual flow |
| D11 | What are the VRAM/RAM boundaries of the MoE engine recommendation — is a 12GB Blackwell enough? | Recommending by GPU only solves "which engine to pick", not "can it run/run fast". MoE expert weights mostly live in system RAM/SSD, with VRAM holding hot experts + attention; the threshold is the combination of VRAM + RAM + disk, not VRAM alone. Local example RTX 5070 Ti Laptop (Blackwell, 12GB) + about 31GB RAM: hits Basalt (driver≥R580), but under 12GB VRAM large MoE experts mostly land in RAM/SSD; Strata Qwen3.8-Flash-Next Q2_0 needs about 37.6GB (RAM+VRAM), Coder about 32GB — this machine's RAM is tight, so pick a lower tier/leave enough headroom, with no guarantee of smoothly running the full 125B. Basalt likewise needs enough RAM for experts that do not fit in VRAM. First pick a pack by the engine's RAM tier table, use --plan/preview to see the resource plan, and measured speed is valid only for that specific setup. When resources are insufficient, switch to a smaller model/lower tier/add RAM — never speculate a guarantee |

### E. Weight processing, calibration, and sources (E01–E10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| E01 | How do I choose among no abliteration, Abliteration, and Heretic? | Default none keeps the input; Abliteration does refusal-direction processing; Heretic needs an interactive save; none guarantees quality on every task |
| E02 | Why can GGUF/NVFP4 not be abliterated again? | Current import uses the quantized source directly; editing the HF base does not change imported tensors; to process, start from full precision first |
| E03 | Does an existing Heretic model need processing again? | Choose none when there is no need, to avoid stacked changes; the name alone cannot prove the processing method or quality |
| E04 | What is a direction file? | A refusal-direction vector matching the model's hidden size; validated against the specific supported format/source — never casually use another model's vector |
| E05 | Why does Heretic pop up a separate window? | The existing tool needs interaction and an artifact save; without saving complete weights in that window it will not continue pretending to be done |
| E06 | Is the calibration corpus for chatting? | It is used to estimate statistics for quantization/processing, not to automatically train arbitrary new knowledge; corpora at different stages serve different purposes |
| E07 | Can AWQ/GPTQ run without a corpus? | It can currently auto-prepare WikiText-2 train at a pinned revision; this needs network and resources, records sampling/hashes, and may not be the best fit for a domain task |
| E08 | Are more calibration samples always better? | Data representativeness, length, and count affect time and statistics; no monotonic improvement is promised — verify per task |
| E09 | What are group size, batch, and CPU activations? | Explain quantization grouping/processing batch and memory trade-offs; current HF group choice follows metadata, and CPU/disk offload adds time and RAM/disk pressure |
| E10 | Are imatrix, Engram, n-gram, and MTP sources the same? | No; they are respectively an importance matrix, an architecture-specific table, or speculative components — fill them per the current missing-source fields, never substituting one for another |

### F. MTP/DFlash2 and runtime parameters (F01–F10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| F01 | What is speculative decoding? | An acceleration mechanism where a draft proposes tokens and the main model verifies them; it needs compatible components and implementation, and does not add ability to a model out of thin air |
| F02 | My model has no head — can it train one automatically? | Not currently; you can provide a supported external component, or choose auto/off — it cannot be force-enabled |
| F03 | After quantization/format conversion can MTP still be used, and why is it kept but not faster? | A compatible route keeps existing heads; kept does not mean enabled/faster; HF currently only archives, and execution needs a compatible NInfer. Look at runtime, the actual engine, acceptance, and a same-condition baseline |
| F04 | Can HF AWQ/GPTQ use MTP? | The current HF service does not execute it; auto with heads is blocked — to execute, use a compatible NInfer; when explicitly turned off it can archive only |
| F05 | Can MTP and DFlash2 be swapped freely? | No; architecture, geometry, weights, and config contracts differ; use only detected/compatible components |
| F06 | Are more draft tokens always faster? | Not guaranteed; more drafts add cost, and the gain depends on acceptance and hardware; test under identical conditions with available parameters |
| F07 | What is context, and why is there a 1024 cap? | Input and generation are bounded by the token window/current adapted engine; that small-model path has a specific cap, not a site-wide uniform cap for all models |
| F08 | Is KV cache bit width the same as weight quantization? | Different objects; the runtime cache affects VRAM and numerics; changing KV settings does not mean the main weights are requantized |
| F09 | What does high concurrency affect? | It increases simultaneous requests and resource overhead; the HF local service currently has generation concurrency 1 — do not assume it scaled just from a UI number |
| F10 | Will stopping the model and re-converting lose files? | What stops is the service process; normal artifacts are kept; the current page requires releasing the owned model service before converting, and the AI must not stop it on its own |

### G. Build process and failures (G01–G10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| G01 | It keeps saying "preparing" — is it stuck? | Look at the current download/install log, file growth, and errors — do not judge by time; the first run may be a large download, and no fixed progress is promised |
| G02 | If validation fails, does conversion still count as successful? | There may be a file but GPU use did not pass; runnability cannot be inferred from file existence; check the smoke-stage error |
| G03 | Difference between converted and validated? | converted only means converted (validation may have been skipped); validated has the corresponding validation evidence; direct start requires this run's verified state |
| G04 | Will refreshing the page lose the task? | Existing task state is recoverable; a backend restart/process interruption does not mean background execution continues — check the actual task record |
| G05 | Can I convert several at once, or convert while running? | The current BuildManager limits concurrent builds/service conflicts; wait per the UI hint or manually stop the owned service |
| G06 | Why does it say Torch CUDA is unavailable? | Confirm the actual worker's Torch and driver; having a GPU in the system does not mean that Python can use CUDA; prefer the one-click diagnostic log |
| G07 | Can the version-mismatch warning be ignored? | Distinguish warning/error; a version difference means insufficient compatibility evidence — a matching bundle/pinned engine reproduces more easily |
| G08 | After an interrupted build, which step resumes? | Current downloads are reusable, but an arbitrary conversion resume point is not guaranteed; keep logs, rebuild to a new output per task state, and never tamper with the manifest |
| G09 | The error shows only one line — what do I do? | Expand the current stage log and copy the redacted error with context; do not submit whole-drive paths, tokens, or complete model weights |
| G10 | The AI says it can continue, but the button is still disabled? | The backend gate decides; the AI can only explain — list the disable reason and locate it, and never let the user edit the state file to bypass |

### H. Quality reports (H01–H10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| H01 | The model lost quality after quantization and got dumber — what now, and what is PPL? | Low bit width and unsuitable calibration corpus can increase quality loss; first look at the same-corpus PPL gate and real task performance, and if needed raise the bit width and use domain-matched independent calibration and evaluation sets. PPL is predictive perplexity on a given corpus, not accuracy on every task |
| H02 | How is the PPL increase computed? | `(current PPL / baseline PPL - 1) × 100%`; only a valid same-condition baseline makes it meaningful |
| H03 | Must the baseline model be exactly identical? | Source, tokenizer, processing stage, corpus, and evaluation settings should match; the pre/post-abliteration difference must not be treated as pure quantization loss |
| H04 | Is the 10% gate an official safe value? | It is a user-set acceptance threshold, not a universal quality guarantee; validate against your own application |
| H05 | Why can quality-rejected not start? | The currently chosen quality requirement was not met or measurement failed, and the system keeps the real report; adjust the plan/corpus and re-evaluate — editing the manifest directly is not advised |
| H06 | Does no PPL number mean zero loss? | No; look at not_measured/refused/measurement error — unknown is not 0 |
| H07 | Why does the HF reference run on CPU? | The current full-precision reference path uses CPU FP32 while the quantized model uses GPU; it costs RAM/time, and dtype/device differences are also a comparison limitation |
| H08 | Can a packed model use itself as the baseline? | It cannot serve as a known full-precision baseline proving quantization lossless; without a confirmable reference the corresponding quality proof is refused |
| H09 | Can the calibration corpus also be the evaluation set? | Overlap should be avoided, following the current implementation's data isolation and traceability settings; otherwise results may be too optimistic |
| H10 | PPL passed but chat got worse — what now? | Add real-task tests for code/math/long context/instruction following; PPL does not cover every dimension, and checking the template and parameters is also necessary |

### I. Speed and resource reports (I01–I10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| I01 | Difference between prefill and decode? | The former processes the input prompt, the latter outputs token by token; the two tok/s numbers are not compared interchangeably |
| I02 | First token is slow but later ones fast — normal? | First load, input length, and prefill affect the first token; only measurement records can separate the specific bottleneck |
| I03 | measured/configured/refused/not_measured? | Respectively real measurement, configuration only, refused, and not measured; a configured number must not be advertised as measured performance |
| I04 | Why is my speed worse than the README? | GPU, model/revision, quantization, KV, context, speculation, concurrency, and test method differ; establish a same-condition baseline first |
| I05 | A 27B can do 180 tok/s — can I? | That is a specific historical RTX 5070 Ti Laptop+Bonsai+MTP3 measurement and cannot be extrapolated; cite it with source/conditions |
| I06 | Does a high acceptance rate mean MTP is good? | Corpus and method affect results; acceptance on a historical tiled corpus does not represent real requests |
| I07 | Is the Torch peak the GPU's total usage? | No; the current HF metric basis is Torch's allocation peak, excluding all process/driver overhead or the quantization-stage total peak |
| I08 | It was estimated to fit — why actual OOM? | Beyond weights there are KV, activations, temporary buffers, other applications, and stage differences; estimates have assumptions — diagnose by the actual stage |
| I09 | Can two reports of the same quantization be compared directly? | Hardware, input, context, KV, concurrency, and speculation must also align; different sampling settings are not evidence of algorithm superiority |
| I10 | How do I verify faster without obviously worse? | Fix input/runtime config for a speed baseline, then look at same-corpus quality and real tasks; show measurement conditions and limitations with the results |

### J. Artifact startup and application integration (J01–J14)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| J01 | After conversion, how do I actually run it? | Confirm this run is validated/gate passed → click Start model API → wait for ready → copy the returned base_url → fill the model ID in the client → verify with a short sentence |
| J02 | What opens a `.ninfer`? | A matching NInfer engine/the page-generated launch script; not every GGUF/HF loader recognizes it |
| J03 | Where is the AWQ/GPTQ artifact and how do I run it? | Current output is `model-hf/`, via the isolated HF worker; after validation, start from the page or the generated `scripts/run.bat` — do not load with NInfer |
| J04 | Why does it say the port is occupied? | Stop the occupying service you clearly own, or change the port per the current flow and update/rebuild the plan; never let the AI kill unknown processes |
| J05 | starting never changes? | A process starting is not ready; look at the current serve.log, model loading, VRAM, port, and exit state |
| J06 | What goes in base_url in the client? | Copy the actual service-returned `http://127.0.0.1:<port>/v1`; if the client wants a full endpoint, append `/chat/completions` once |
| J07 | How do I fill the model name? | Read the real id from that service's `/v1/models`; not a guess from the file name or UI label; when a service does not support listing, fill it manually per config |
| J08 | Does the local API need a key? | The current built-in HF loopback service has no such auth requirement; when a client forces one, use its required placeholder value, but that is not authentication; other engines/upstreams follow their actual service config |
| J09 | Can it connect to chat software, images, or tool calls? | Text-compatible clients can integrate per the interface; the current HF does not support images, tools, structured output, or arbitrary parameters — text compatibility does not promise every feature |
| J10 | Can another computer access it? | Default loopback is local-only; public deployment needs added authentication/permissions/network protection — simply changing host does not make it safely done |
| J11 | Can the produced .ninfer run directly in Strata or Basalt, and which should I use after conversion? | No. .ninfer is only for the matching NInfer; Strata needs a separate GGUF download and Basalt a separate .basalt — neither can be fed .ninfer directly. The page's "Inference engine recommendation" gives one by architecture and GPU, with repository links. Dense uses ninfer-fusion-kvmem (J14, https://github.com/1314521gjy/ninfer-fusion-kvmem, Apache-2.0; melis2023/ninfer-kvmem-actkv has no LICENSE and is not recommended). MoE with compute exactly 12.0 AND model Flash-Next or DeepSeek-V4.1-Flash uses Basalt (J12, https://github.com/jesdga95/basalt). Other MoE uses Strata (J13, https://github.com/Niko1221/Strata); datacenter compute 10.0 is not RTX 50. These recommendations do not integrate or auto-download. |
| J12 | Complete plain-language steps to run MoE with Basalt on an RTX 50 card? | Applies to: compute exactly 12.0, Win10 1903+ or Win11, model Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash. compute 10.0 and Qwen3.6-35B-A3B go to J13. Never treat .ninfer as .basalt. ① Provide CUDA 13, driver R580+, Python 3.10+ yourself; open https://github.com/jesdga95/basalt/releases and download basalt-v0.4.0-windows-x64.zip. ② Separately download the matching .basalt v2. ③ First ask separately for context tokens and concurrency; do not write the script before getting positive integers. ④ basalt_config.json contains only the verified structure: version=2, models[0].model=the user's absolute .basalt path, gpu="0", kv per model default (Flash-Next fp16, DeepSeek fp8), server.port=8080. Writing an unverified concurrency field is forbidden; when concurrency > 1 the bat first echoes "This run starts with a single request". ⑤ start-basalt.bat uses cmd only: @echo off, chcp 65001, setlocal EnableExtensions, cd /d "%~dp0"; if the engine or model is missing then echo, pause, exit /b 1. Launch line: "bin\\basalt.exe" serve --config "%~dp0basalt_config.json" --max-context user-context. End with pause. ⑥ Open http://127.0.0.1:8080/v1/models then send a short chat. |
| J13 | Complete plain-language steps to run MoE with Strata on a non-RTX-50 card (or Qwen3.6-35B-A3B)? | Applies to: MoE with compute not 12.0, and Qwen3.6-35B-A3B on any card. Never rename .ninfer to GGUF. ① Open https://github.com/Niko1221/Strata/releases and install Strata-0.2.13-windows-x64-setup.exe; confirm strata.exe works. ② Separately download GGUF: 35B-A3B uses Unsloth UD-Q4_K_XL, UD-Q6_K_XL or Q8_0; Flash-Next uses ISTA-DASLab Q2_0, IQ2_XS, IQ3_XXS, IQ3_S. ③ First ask separately for context tokens and concurrency. The only verified command is strata launch --family --model --plan; no context or concurrency flag has been verified. ④ start-strata.bat uses cmd only: @echo off, chcp 65001, setlocal EnableExtensions, cd /d "%~dp0"; if where strata.exe fails or the GGUF does not exist then echo, pause, exit /b 1. Before launching, echo the user's requested context and concurrency and state "the currently verified command cannot pass these two". The preview command writes only strata.exe launch --family user-family --model user-tier --plan. After user confirmation a second bat drops --plan. Adding unverified flags is forbidden. End with pause. ⑤ Use the address printed by the process to open /v1/models then send a short chat. NVIDIA needs CUDA 13, driver 580+. |
| J14 | Complete plain-language steps to run a Dense model with ninfer+kvmem? | Applies to: Dense .ninfer produced by TurboForge. Recommended: https://github.com/1314521gjy/ninfer-fusion-kvmem (Apache-2.0). https://github.com/melis2023/ninfer-kvmem-actkv has no LICENSE and is not recommended. ① Use only Releases tag engine-v0.11.0-kvmem-20261008. On RTX 50 use ninfer-serve-120a.exe, 40 series -89, 30 series -86, and verify SHA256SUMS.txt. Put engine and model on an English path. ② The assistant must first ask separately for context token count and concurrency count; do not write the script before getting positive integers. ③ Write start-fusion.bat, cmd only: @echo off, chcp 65001, setlocal EnableExtensions, cd /d "%~dp0"; if not exist engine or model then echo, pause, exit /b 1. In the same bat set NINFER_KV_WINDOW=16384, NINFER_KV_RETRIEVE=8192, NINFER_HOST_PAGEABLE=1, NINFER_KV_REUSE_HOSTBACKED=1; for a ternary model also set NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=64. The launch line writes the user's context into --max-context and the user's concurrency into --max-concurrency, with --host 127.0.0.1 --port 8091 --kv-dtype k8v4 --host-kv-mib 16384 --prefill-chunk 1024 --max-shared-prefixes 0. On a 12GB card use --kv-capacity 8192; only larger VRAM may use the documented 17920; --host-kv-mib 16384 needs at least 16GB free RAM — if not enough, ask the user first then reduce. Last line pause. ④ Do not add --spec yet. After the window shows engine ready, open another terminal to http://127.0.0.1:8091/v1/models, then ask "17+25, reply with the number only". When the request is longer than --kv-capacity the answer is untrustworthy — switch to the repository's gateway\\chat_gateway.py. ⑤ Troubleshooting: missing DLL → put CUDA 13/FFmpeg/libcurl beside the exe; insufficient VRAM → lower --kv-capacity first; CUDA Graph failure → add --no-cuda-graph; a Chinese path that cannot be read → switch to an English path. |

### K. AI's own configuration and privacy (K01–K10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| K01 | Where do the AI's URL, model, and key come from? | Provided by the user's chosen vendor/self-hosted service; fill the actual compatible API root address, model ID, and access key; this project bundles no quota |
| K02 | Is the assistant the same as the model I want to convert? | They can be independent; an external assistant helps local conversion; when needed it can also connect to an already-started compatible local model, but it cannot act as a usable assistant before it itself runs |
| K03 | Why does the connection test fail? | Check in order: address root path, model ID, auth/permissions, quota, network/certificates, and protocol capability; explain per the real error — never let the user paste the key into chat |
| K04 | Is a key in `.env` absolutely never leaked? | Plaintext config still needs backend isolation, ignore/packaging exclusion, file permissions, and log redaction; it must not go in VITE variables or the frontend bundle |
| K05 | Where is the key entered in the dialog saved? | The first-version design defaults to backend session memory, lost on restart/expiry; only after an admin enables it and the user explicitly chooses is it saved to the local `.env` |
| K06 | Why is the config different after refresh/restart? | Explain the actual source by the priority of session override, environment variables, `.env`; restart clears memory; environment variables may override the just-saved file |
| K07 | Which page content is sent to the cloud? | Selected text, the question, and the redacted business summary checked in the preview; logs are not attached by default and can be edited/deleted before sending; the target domain is shown clearly |
| K08 | Can I keep everything off the external network entirely? | Choose an explicitly allowed local compatible service or use offline help only; a third-party service's retention policy depends on that service — the project cannot guarantee it for them |
| K09 | Will the AI change config or run commands automatically? | The first version is read-only and navigation; answers do not execute; normal business operations still use the existing buttons and gates |
| K10 | What if the AI answers wrong, finds nothing, or runs out of quota? | Look at knowledge citations/actual diagnostics and use offline help; provide the redacted question and request ID; never treat an AI answer as proof of successful execution |

### L. Maintenance, migration, and unknown problems (L01–L10)

| ID | How a user might ask | Answer points / next step |
|---|---|---|
| L01 | Can I move the artifact to another machine and just run it? | It needs the same format support, matching engine/runtime/driver, and paths; generated scripts may contain the original path and need checking — cross-machine plug-and-play is not guaranteed |
| L02 | Which files can I delete to save space? | Distinguish original model, download cache, intermediate weights, final artifact, and dedicated environment; stop tasks/services and check dependencies first — never auto-delete in an answer |
| L03 | Will user data be packed into a shared bundle? | The release script should explicitly exclude models/cache/secrets/sessions; go by the actual ZIP inspection, not just `.gitignore` |
| L04 | The source updated — why does the AI still describe old features? | Check the running package version and knowledge version; business changes update entries and coverage tests in sync — never feed stale docs to the model as-is |
| L05 | How do I report a problem so maintainers understand? | Include version/route, redacted model identifier, stage/error code, relevant log excerpt, and request ID; no keys/weights/private full paths |
| L06 | Why did it succeed before and fail now? | Compare model revision/file hashes, tool versions, environment and resources, runtime parameters; do not attribute it to the quantization algorithm without evidence |
| L07 | Where are historical reports, and can they be reproduced? | Look at the actual location of the current manifest/artifacts; reproduction needs the corresponding inputs/versions/commands — never graft numbers from another build |
| L08 | Can I ignore a warning? | Analyze that warning's specific risk and stage; a warning does not always block, but an error/quality rejection must not be skipped as an ordinary hint |
| L09 | My error is not in the help — what now? | Show the unknown error and stage, collect bounded redacted evidence; provide general path/permission/resource/service diagnostics — never guess a nonexistent code |
| L10 | Is there a mode that "guarantees zero difficulty for every model"? | Not currently; the dynamic support matrix, specific errors, and in-app help lower the operation barrier, but external permission/hardware/algorithm limits cannot be removed |

## 5. From successful conversion to the first chat: in-app guidance body

### 5.1 Normal path

1. On this build's card confirm the state is "Validation passed". If the quality gate is on, also check the gate result. Do not just look at the output file existing.
2. Click "Start model API" and wait for `ready`. If the model is large, check the startup log first instead of clicking start repeatedly.
3. Copy the actual `base_url` returned by the page. Open a client that supports text Chat Completions and fill the model ID with the value returned by that service's `/v1/models`; authentication follows that service's actual config.
4. Send one short question first to verify input/output, then increase context gradually. On error, expand this build's `logs/serve.log` or the actual log location shown on the page.

Later the help center can add a **"test one sentence" entry specifically for the business model service**: after the user explicitly clicks it, the backend calls the current ready service and shows the real response. That entry and the "assistant upstream connection test" are two different features — an external assistant's answer must not pose as the local model running. This plan first completes guidance and existing-interface integration; if that test entry is implemented it should be a separate explicit work item reusing the managed-service limits.

### 5.2 Start decision

| Current fact | What the user sees | Next step |
|---|---|---|
| Preparing/converting/validating | This run's validation is not finished | Watch the stage log and wait for the real result |
| converted | A conversion artifact exists, but this run did not complete GPU validation | Enable validation to a new output per the current flow, or explicitly use the script for manual diagnosis; no way to fake validated is provided |
| validation-failed | The artifact did not pass current GPU validation | Look at smoke/engine/resource errors |
| quality-rejected | The current quality requirement did not pass | Look at baseline/corpus/threshold and the failure cause, adjust and redo — do not edit the result file |
| validated and startable | The current start gate is satisfied | Click start and wait for ready |
| ready | The service passed the health check | Configure a client and make one real text request |
| exited | The process exited | Check serve.log; do not keep showing "model available" |

## 6. Answer examples and evidence constraints

**Question: Is 4-bit the best deal?**

"It usually reduces weight storage, but whether it suits you depends on this model's available recipes, free VRAM, and quality requirements. Establish a baseline with the current auto recommendation first; if VRAM is still tight, compare available lower bit widths and check quality with the same corpus and real tasks. Your specific recommendation now follows the inspection result."

**Question: The MTP head was kept — why did AWQ not get faster?**

"The current HF AWQ/GPTQ service only archives auxiliary heads and does not execute MTP/DFlash2. If you need actual speculative execution, look at the NInfer routes available for this model; if you only need to export AWQ, explicitly choose to turn speculation off. Kept intact does not mean enabled." Basis: `SPECULATIVE_RETENTION.md` and current HF serving capability.

**Question: PPL is 9.90% above baseline — is the model ruined?**

"This number alone cannot decide the model is unusable. It means that under this run's specified corpus and settings, PPL increased 9.90% versus the baseline; whether it passes depends on your set threshold and real task performance. First confirm the baseline and scoring settings match, then look at the gate result, and add tests with your own tasks." Without this run's report, never directly claim "passed".

**Question: Out of memory — is lowering bits enough?**

"First confirm whether what is short is system RAM or GPU VRAM. Lowering output bits may reduce runtime weights, but conversion may still load full-precision input, calibration activations, and intermediate data. Look at the error stage and resource card first, then decide whether to switch to a smaller model, free resources, or choose another currently supported path."

## 7. Content maintenance and acceptance mapping

| Source / coverage object | Corresponding content | Maintenance rule |
|---|---|---|
| `ForgeConfig` and page fields | B–F, parameter terms | Every editable field has explanation, effect, limits, default source, and error location |
| `RECIPES / HF_RECIPES / precheck.choices` | C–D, route table | Options come from runtime metadata; a new recipe updates the knowledge scope in sync |
| `Diagnostic.code` | Companion diagnostic catalog | Compare the auto-extracted set with knowledge IDs; a new code without an entry fails CI |
| Build/service states | G, J, start decision table | A new state explains in sync whether it is terminal/startable, keeping the backend's original gate |
| reports metrics | H–I | Units, state, source, measurement conditions, and non-comparable conditions are complete |
| Runtime exceptions | Companion runtime error classification, A/B/G/J/K | Record category and original cause; an unknown error is never forced into "resolved" |
| AI config and privacy | K | Must match actual storage/session/deployment mode; never advertise unimplemented protection |
| Historical test reports | Relevant measured entries | Label history, hardware, and boundaries; without re-measurement never change to "all currently verified" |

Before going live, use the full A01–L10 question set as the manual acceptance checklist, with representative rewrites and known errors as automatic retrieval regression; sampled coverage does not mean every answer was verified item by item. After implementation, fill in each knowledge ID's source, reviewer/date, and verification status. All entries are currently "draft pending integration"; this document does not claim any AI Q&A evaluation was run.