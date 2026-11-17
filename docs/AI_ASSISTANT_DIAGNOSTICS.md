# AI assistant diagnostic and runtime exception catalog

Date: 2026-11-14. Status: knowledge draft. Companion [refactor plan](AI_ASSISTANT_REFACTOR_PLAN.md), [120 user questions](AI_ASSISTANT_KNOWLEDGE.md).

The fixed diagnostic codes below are taken from the string-literal arguments of `Diagnostic(...)` in current `src/turboforge/**/*.py`, extracted via AST and cross-checked. This is not the full set of runtime exceptions: many download/calibration/generation failures propagate through `ValueError / RuntimeError / OSError` or third-party return content and must be classified separately under part 2.

## 1. Fixed diagnostic codes: meaning and action

Severity follows the `severity` actually returned by the backend this run; the same explanation must not auto-upgrade a warning to an error or downgrade an error to a suggestion. Automatic selection and successful identification are information, not "errors" in the UI. Dynamic message values and task-specific numbers/suggestions in alternatives should be kept after redaction.

### Input format and integrity

| Code | Plain explanation | Next step | Source |
|---|---|---|---|
| `input-unreadable` | The model path or content cannot be read | Check the local path, access permissions, and file integrity | `planner.py` |
| `empty-model` | The directory has no usable Safetensors weights | Select a complete model directory and check whether the download finished | `adapters/safetensors.py` |
| `duplicate-tensor` | Multiple shards declare duplicate weights; the source may be mixed | Re-check shards against the same revision; do not delete unknown tensors directly | `adapters/safetensors.py` |
| `config-missing` | The architecture config is missing | Supply a config.json matching the weights | `adapters/safetensors.py` |
| `tokenizer-missing` | The tokenizer is missing | Get tokenizer files from the same source; a random model's tokenizer cannot substitute | `adapters/safetensors.py` |
| `gguf-recognized` | GGUF recognized; not an error | See its actual encoding, shards, and available import routes | `adapters/gguf.py` |
| `gguf-source-invalid` | The additional GGUF cannot be read or parsed | Check the specific file, shards, and format integrity; keep the underlying error | `planner.py` |
| `gguf-secondary-source` | On the manual route the GGUF should be an additional source, with original metadata supplied | Use auto guidance to verify the source, or specify the original directory and the gguf source field | `planner.py` |
| `nvfp4-secondary-source` | NVFP4 should be imported as a quantized additional source | Supply matching original files and set the quantized source | `planner.py` |
| `nvfp4-name-is-not-evidence` | The name contains NVFP4 but there is no complete format evidence | Go by the actual config, packed tensors, and scales; renaming to bypass is not allowed | `adapters/safetensors.py` |
| `nvfp4-layout-unverified` | The current NVFP4 layout has not passed a supported contract validation | Get a complete, supported checkpoint/producer layout | `adapters/safetensors.py` |
| `nvfp4-information-loss` | NVFP4 has already lost some original numeric information | Accept the precision boundary of the current source; for original precision, obtain the original weights | `adapters/safetensors.py` |

### Quantization, sources, and output

| Code | Plain explanation | Next step | Source |
|---|---|---|---|
| `no-verified-recipe` | The current architecture/geometry has no wired conversion plan | See the current support list and switch to a supported model; do not guess a similar recipe | `planner.py` |
| `target-incompatible` | The specified plan does not match the model | Go back to the available options and reselect; the AI must not replace an explicit choice on its own | `planner.py` |
| `no-runnable-target` | A recipe exists, but required sources were not provided | Supply ternary/gguf/quantized/imatrix and other files per the missing list | `planner.py` |
| `recipe-source-missing` | This recipe lacks a required additional source | Supply the files this recipe explicitly requires, then preview the plan | `planner.py` |
| `source-missing` | The entered source path does not exist | Reselect the correct file/directory and confirm the drive is mounted | `planner.py` |
| `packed-needs-hf` | Existing AWQ/GPTQ cannot be handed to NInfer directly | Choose the HF packed retention route or start over from full precision | `planner.py` |
| `nvfp4-requantization-loss` | Converting from existing low precision to another quantization adds loss | Prefer original full precision; only actually supported approximate routes can discuss explicit consent, and a switch cannot lift all layout limits | `planner.py` |
| `output-overlaps-input` | The output overlaps the original model directory, risking overwrite | Choose a new empty directory outside the original model | `planner.py` |
| `output-not-empty` | The output directory already has files; the program refuses to overwrite | Create a new output directory and keep existing artifacts | `planner.py` |
| `auto-selected-bonsai2` | A matching ternary source exists, so Bonsai was selected automatically | Show the actual recipe and source; this run did not train a ternary model | `planner.py` |
| `auto-selected-gguf` | Import automatically follows the GGUF's original precision | Explain that precision is decided by the input; no promise of re-bitting | `planner.py` |
| `auto-selected-nvfp4` | An available NVFP4 import was selected automatically | It still must match the actual producer layout; resolve ambiguity by explicit page choice | `planner.py` |
| `auto-selected-gguf-bridge` | The current geometry uses the GGML Q8_0 bridge by default | Explain the native fixed-geometry kernel limit; do not generalize it to a Q8 default for all models | `planner.py` |
| `auto-selected-lowest-bpw` | A lower bpw was picked among current candidates by the default rule | Show this run's candidates and the quality/resource trade-off; do not claim best quality | `planner.py` |

### Environment and resources

| Code | Plain explanation | Next step | Source |
|---|---|---|---|
| `engine-missing` | The selected engine directory lacks required executables | Use a matching bundle or re-prepare the current engine | `planner.py` |
| `upstream-missing` | The converter source is not at the specified location | Prepare automatically or select a matching source directory in advanced settings | `backends/compiler.py` |
| `upstream-revision-differs` | The actual converter differs from the checked revision | Review the revision difference and prefer the matching version; do not claim an untested version is compatible | `backends/compiler.py` |
| `upstream-revision-unknown` | The converter revision cannot be confirmed | Check the source marker/version and record it as unknown; never auto-treat it as matching | `backends/compiler.py` |
| `torch-missing` | The current conversion environment lacks torch | Use one-click preparation and check the actual worker environment | `backends/compiler.py` |
| `numpy-missing` | The current conversion environment lacks numpy | Use one-click preparation to fix that environment | `backends/compiler.py` |
| `torch-cuda-unavailable` | The current torch cannot use CUDA | Check the actual Python, Torch CUDA build, driver, and device | `planner.py` |
| `device-unverified-cpu` | CPU conversion was selected, but there is no speed evidence for this route yet | It can be explained as likely slower; never give fabricated durations, and it does not mean CPU inference is supported | `planner.py` |
| `ram-profile-unverified` | This route has no reliable conversion RAM peak model | Observe actual usage on the first run; never fill unknown with zero or "enough" | `resources.py` |
| `disk-insufficient` | Temporary disk demand exceeds free space | Switch to an output drive with more free space; never delete user files automatically | `resources.py` |
| `ram-insufficient` | Modeled system memory demand exceeds what is available | Free RAM, use a smaller model or a verified low-memory route; do not just promise that lowering bits works | `resources.py` |
| `gpu-not-detected` | NVIDIA VRAM cannot be queried | Check the GPU/driver and the current runtime environment; distinguish conversion from inference stages | `resources.py` |
| `vram-insufficient` | Estimated runtime VRAM may not fit | Adjust context/KV/concurrency, lower bpw, and other usage; this is not a measured conversion peak | `resources.py` |
| `context-capped-gdn-geometry` | The current adapted engine caps context for this geometry | Show the backend's actual effective cap; for long-text needs switch to a compatible plan — changing the number alone cannot break the kernel | `planner.py` |

### Weight processing and calibration

| Code | Plain explanation | Next step | Source |
|---|---|---|---|
| `jailbreak-unknown` | The processing method name is invalid | Choose a currently available item among none/heretic/abliteration | `planner.py` |
| `jailbreak-precision` | The current precision does not support full-precision weight processing | Obtain supported original BF16/FP16/FP32 or choose no processing | `planner.py` |
| `jailbreak-not-applied-to-imported-weights` | Editing the HF base does not affect imported quantized tensors | Import an already-processed source and choose none, or process full precision first and then quantize | `planner.py` |
| `direction-file-missing` | The specified direction file does not exist | Reselect a matching vector, or use available automatic calibration | `planner.py` |
| `calibration-unavailable` | Automatic direction calibration lacks transformers or other conditions | Fix the correct environment, or provide a matching existing direction | `planner.py` |
| `heretic-needs-high-precision` | Heretic needs full-precision input | Use a matching full-precision source; never force it on packed input | `jailbreak/processors.py` |
| `heretic-needs-safetensors` | The current input container is unsuitable for Heretic processing | Select a complete supported Safetensors model directory | `jailbreak/processors.py` |
| `heretic-unavailable` | The current environment has no usable heretic-llm | Prepare it per the existing install instructions, or choose an available method | `jailbreak/processors.py` |
| `heretic-needs-desktop-console` | The current environment cannot complete the required desktop interaction | Use advanced mode in a supported Windows desktop environment, or choose an unattended method | `jailbreak/processors.py` |
| `heretic-interactive-save` | You must operate in an external window and save the processed weights | Follow the prompt to save complete weights to the agreed directory and wait for real completion | `jailbreak/processors.py` |
| `abliteration-needs-high-precision` | Abliteration needs supported full-precision weights | Switch to a full-precision source or turn processing off | `jailbreak/processors.py` |
| `abliteration-needs-safetensors` | The current format cannot be rewritten by this processor | Use a complete HF Safetensors directory | `jailbreak/processors.py` |
| `abliteration-unavailable` | Processing dependencies are incomplete | Check that worker's actual torch/numpy and other dependencies, not just the system environment | `jailbreak/processors.py` |
| `abliteration-calibration-source` | Collecting the direction using the specified/default prompts | Explain the direction source and processing stage; successful calibration is not a quality guarantee for every task | `jailbreak/processors.py` |

### Speculative components and the HF route

| Code | Plain explanation | Next step | Source |
|---|---|---|---|
| `speculative-component-missing` | A compatible component was explicitly requested but not detected | Choose auto/off, or provide a supported matching source; never train/forge heads | `planner.py` |
| `speculative-would-be-dropped` | The current route cannot keep the input's own components | Switch to a route that keeps them; never finish by silently dropping heads | `planner.py` |
| `hf-native-runtime` | The output goes to the standalone HF engine, not NInfer | Use the HF launch flow; MTP/DFlash2 are not executed, and the new route has not completed real-GPU acceptance | `backends/hf_quantization.py` |
| `hf-configuration` | HF parameter/source/layout validation failed | Show this run's specific message, then map it to the corresponding category in part 2; do not just say "switch to auto" | `backends/hf_quantization.py` |

The last four items come from `planner.py` (first two) and `backends/hf_quantization.py` (last two). When converting them into runtime entries, fill in the source code references.

## 2. Runtime exceptions without a fixed Diagnostic code

The categories below are **proposed knowledge IDs**, not existing unified backend error codes. During implementation keep the real exception cause and request ID, and gradually provide stable codes at the API boundary; do not derive every cause by matching whole sentences.

| Proposed knowledge ID / scenario | Existing source / typical symptom | What to answer and next step |
|---|---|---|
| `runtime.model.not_found` | precheck/preparation: does not exist locally and is not a valid repository | Paired quotes and leading/trailing spaces are supported; confirm the path belongs to the PC running the service, and select an existing directory/file or a valid HF repository |
| `runtime.model.source_unknown` | An orphan GGUF/metadata has no original source, or the source is not a complete repository | The user supplies a verifiable original HF repository or a directory with config.json and tokenizer; a directory containing only GGUF does not qualify; when there is no import route for that size, use the original full-precision model instead of just adding config |
| `runtime.model.incomplete` | Missing config, tokenizer, weights/index/shards | List the specific missing items and revision, and supply them from the same source |
| `runtime.model.unsupported_geometry` | Unsupported architecture/geometry rejected before downloading large weights | This is a missing implementation, not a network fault; switch to a supported input — repeated downloads do not help |
| `runtime.model.engine_port_required` | A small model needs an adapted bundle | Use a version with the corresponding adapted engine; do not promise that a plain upstream download has it |
| `runtime.download.access` | HTTP 401/403, gated models, permissions/license | Verify legitimate access; never ask for keys or guide bypassing |
| `runtime.download.not_found` | The file/revision does not exist | Check the full link, explicit version, and actual files; distinguish the source of the 404 |
| `runtime.download.network` | DNS/SSL/proxy/timeout/interruption | Explain the network stage and keep the safe error; cache/resume can be reused, but never disable TLS verification |
| `runtime.download.integrity` | Checksum failure or missing shards | Re-fetch the specific corrupted items and re-verify; never skip SHA-256 |
| `runtime.fs.permission` | Output, cache, or task records are not writable | Check space/current-user permissions and choose a writable separate directory; never elevate by default |
| `runtime.hf.layout` | GEMV/Marlin/BitBLAS/extra activations/mixed bit widths and others are not wired | Provide a standard compatible layout or original full precision; never pose as GEMM |
| `runtime.hf.packed_integrity` | qweight/scales/qzeros/g_idx dtype, shape, value, or index anomalies | Check the complete checkpoint and quantization config; never fill values or remove checks on your own |
| `runtime.hf.config_conflict` | Different quantization config files or method declarations conflict | Use a consistent checkpoint from one source, state the conflicting fields, and never pick one arbitrarily to swallow the error |
| `runtime.hf.calibration` | Too few samples, length over the window, missing text, calibration failure | Supply a valid corpus/reduce length or adjust within supported parameters; never fall back to another algorithm |
| `runtime.hf.calibration_eval_overlap` | Evaluation and calibration data are identical or overlap | Use an independent scoring corpus and keep records traceable; never disable validation and then claim reliable quality |
| `runtime.hf.parameters` | group, backend, concurrency, context, PPL stride, repetitions, etc. are invalid | Show the actual fields, values, and allowed ranges and locate the control; packed original group may be -1, not to be confused with new-calibration 32/64/128 |
| `runtime.hf.environment` | Isolated environment missing, install failure, CUDA/import self-check failure | See the actual error in `.turboforge/hf-env`; current code pins versions/driver requirements per this bundle's instructions — never upgrade an existing environment arbitrarily |
| `runtime.hf.speculation` | HF does not run heads; auto with heads is blocked | To execute them choose a supported NInfer; for archiving only, the user must explicitly turn speculation off — heads are never dropped automatically |
| `runtime.weight.interactive` | Heretic not saved, interaction closed, timeout | Check the external window and agreed artifact; after a successful save continue per the real flow |
| `runtime.source.imatrix` | The matrix has no matching vectors | Use a matrix adapted to this model; never fill ordinary text as imatrix |
| `runtime.source.conflict` | The main input and additional sources conflict | Re-decide a single route and its sources, keep the user's explicit choice, and never overwrite silently |
| `runtime.convert.process_exit` | Converter non-zero exit/crash | Locate the first relevant error in convert.log; distinguish resources, format, and engine — do not just repeat the exit code |
| `runtime.convert.no_artifact` | The process ended but the expected artifact was not generated | The build failed; exit code 0 must not be filled as success — check the output/converter interface |
| `runtime.convert.oom` | Conversion/calibration RAM or CUDA OOM | Analyze memory by stage, choose a supported low-memory config/smaller model, and keep the real resource limits |
| `runtime.speculation.binding` | Head tensor/connection/shape/physical-range/continuation-volume validation failed | Use a matching complete component or a supported route; never skip validation and then show "kept" |
| `runtime.validate.failure` | GPU smoke failure, non-finite logits | See smoke/worker logs and the engine environment; having a file does not mean it can start |
| `runtime.quality.reference` | Baseline source, processing stage, fingerprint/hash mismatch | Choose a real baseline at the same stage; never mix pre/post-abliteration or the artifact itself |
| `runtime.quality.measurement` | PPL tool missing/failed, scoring tokens inconsistent | Quality is unknown and the failure is kept; fix the test conditions — never fill zero loss |
| `runtime.quality.threshold` | quality-rejected | Explain the actual PPL/relative increase/threshold, choose another plan and re-accept; never edit the state by hand |
| `runtime.build.concurrent` | An existing build/model service is occupying resources | Wait for the task to finish, or let the user manually stop the service they clearly own; never kill unknown processes automatically |
| `runtime.build.record` | unknown build id/incomplete record structure/interruption | Refresh the actual state and check output records; a lost record must not be treated as success |
| `runtime.serve.validation_required` | Not validated this run, so it cannot start | Point to this run's validation problem; converted or skipped validation cannot start directly |
| `runtime.serve.artifact_missing` | The model was moved/deleted, or validation is abnormal | Check the actual artifact or rebuild per the existing flow; an empty file must not fool the existence check |
| `runtime.serve.port_conflict` | The specified port is already in use | Confirm ownership and manually stop that service, or change the port and rebuild the plan per the current flow |
| `runtime.serve.not_ready` | starting, process exited, health check not passed | See serve.log, loading, and VRAM; a process existing is not ready |
| `runtime.serve.request` | HF does not support tools/images/parameters; total tokens over the limit | Use text messages and supported parameters, shorten input/output, or switch to a service that actually supports it |
| `runtime.serve.busy` | HF concurrency is 1; another request is being processed | Wait/stop your own request; never promise that adding threads scales GPU concurrency |
| `runtime.serve.generation` | Timeout, generation error, mid-stream error | Keep the returned text and show it as unfinished, check the specific log, and never fabricate a complete answer |
| `runtime.ai.config` | Proposed: URL/model/key missing or invalid | Open the three settings in assistant settings and explain the actual effective source |
| `runtime.ai.auth_quota` | Proposed: 401/403/429 | Distinguish authentication/permission/quota/concurrency; never call every 429 "insufficient balance" |
| `runtime.ai.protocol` | Proposed: 404, invalid JSON/SSE, incompatible format | Check the API root address, model ID, text protocol, and streaming capability |
| `runtime.ai.timeout` | Proposed: connection/first-byte/total timeout, cancellation/stream break | Keep the question draft; the user decides whether to retry — never auto-repeat billing mid-stream |
| `runtime.ai.context` | Proposed: 413/context over limit, selection too large | Shrink the selection/log/history and state what is dropped; the user can start a new session |
| `runtime.ai.security` | Proposed: CSRF/URL/session permission denied | Explain the current deployment policy and the allowed secure configuration path; never treat turning protection off as a general fix |

## 3. Catalog maintenance and acceptance method

1. Collect `Diagnostic` string codes via AST, deduplicate, and diff against this code table/the future runtime entry set; a new code must get body text and linked pages. Do not only grep the current frontend `guidance.ts` — it covers only some diagnostics.
2. `Diagnostic` code references may be refactored; generate source_refs during implementation to prevent hand-written line numbers from going stale. This table lists module names; all relative locations are under `src/turboforge/`.
3. Scan raise/HTTP errors/task terminal states at the main boundaries and classify per part 2; such exceptions are not a finite static code set, and full coverage of third-party errors is not claimed.
4. Every runtime help card contains meaning, applicable stage, evidence, actionable steps, and when more files are needed; never just "please retry". Redact the user's on-site error text and logs before quoting.
5. At release, convert this table into packageable knowledge entries and verify exact diagnostic retrieval and in-place UI explanation; this pass delivers content and coverage only — no LLM/GPU/browser acceptance was run.