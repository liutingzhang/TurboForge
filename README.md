# TurboForge

When a new model comes out, running it on your own GPU — fast enough, with a long context — takes two steps: turn the weights into something this machine can run, then hand that file to the right inference engine. TurboForge does the first half. Dense artifacts go to NInfer; MoE goes to Basalt or Strata depending on the GPU.

[Chinese README (main branch)](https://github.com/liutingzhang/TurboForge/blob/main/README.md) · [Apache-2.0](LICENSE) · [Third-party notices](NOTICE) · [Configuration template](.env.example)

> This branch is the **English edition**: the web UI, built-in help, and assistant are English-only. The Chinese edition lives on `main`.

**Windows 10/11 x64 + an NVIDIA GPU. Double-click, then in the browser: pick a model → inspect → choose quantization → convert → validate → start the API.**

## What you get

- **Identification** from config, geometry, and weights — not from the file name. Hugging Face Safetensors, supported NVFP4, and GGUF.
- **Quantization choices** limited to recipes this model and engine can actually run. Native FP4/FP8 moves up the list when the GPU supports it.
- **A local service**: environments are prepared for you, the GPU load is checked, speed and quality are optional, then an OpenAI-compatible API starts.
- **An engine recommendation**: ninfer-fusion-kvmem for dense models and long context; Basalt for published MoE models on RTX 50; Strata for other MoE models.

A chat app opens a model someone else already prepared. TurboForge turns your own, fine-tuned, or abliterated weights into a model service you can reproduce on this PC.

## Install and open

The release is split into two ZIP files; both are required:

1. Download `TurboForge-Windows-x64-english-part1.zip` and `TurboForge-Windows-x64-english-part2.zip` into the same folder. GitHub's 2 GB per-file limit is why the release is split.
2. Extract both into the same parent folder, for example `D:\`; choose Merge when Windows asks. You should end up with `D:\TurboForge`.
3. Open `TurboForge` and double-click **`Start TurboForge.bat`**. Leave the window open; closing it stops the service.
4. If the page does not open, use the address printed in that window. The default is `http://127.0.0.1:8090`. If 8090 is taken, another port is chosen.

Uninstall: delete the whole `TurboForge` folder. If you used the HF route, also delete `%LOCALAPPDATA%\NInferTurboForge`, which holds downloaded Python/HF environments. Models and build records live inside `TurboForge`, so deleting the folder is a full cleanup.

The first run needs the internet and an NVIDIA driver that is already installed. You do not need Git, Node.js, Python, or the CUDA Toolkit beforehand. The two parts total about 2.8 GB (1.2 GB + 1.6 GB) and include source, the web page, Python 3.13.18, and a matched NInfer engine. They do not include models or personal keys. Checksums are in `SHA256SUMS.txt`; both parts are required.

Check the machine without converting:

```powershell
.\"Start TurboForge.bat" --check
```

The report is `.turboforge/deployment.json`. Logs are in `.turboforge/logs/`.

To run from source, create a virtual environment. Conversion also needs a matched NInfer source and engine: point `TURBOFORGE_NINFER_SRC` at the directory containing `tools/convert`, and `TURBOFORGE_NINFER_BIN` at the engine directory. The bundle needs neither.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe scripts\start_web.py
```

## Six steps to a local API

1. **Pick a model**  
   Enter a local folder, a GGUF file, or a Hugging Face `org/name`. Leave "original model source" empty for a full HF folder. A single-file Qwen3.5 dense Q8_0 GGUF with hidden size 1024 can be used as-is. Other GGUF files need the original model folder or repository.

2. **Inspect**  
   The page reads architecture, size, shards, and quantization, plus the GPU name, compute capability, and VRAM. It continues when the combination is supported.

3. **Choose quantization**  
   Pick `auto` or one recipe from the list. Only adapted recipes appear. If several NVFP4 layouts exist, pick one. Weights that are already quantized are imported; they are not edited again as if they were full precision.

4. **Convert**  
   The first run downloads and checks dependencies. Outputs, logs, and records go to `.turboforge/builds/`. Leave RAM and temporary disk space; the final file size is not the whole cost.

5. **Validate**  
   A pass means the GPU can load the model. Quality and speed are reported only when they were measured. One Qwen3.5-0.8B GPTQ 4-bit run recorded PPL 20.6995 → 21.7993 (+5.31%) and 21.27 tok/s over three 64-token generations.

6. **Start**  
   Open `scripts/run.bat` in the build folder, then confirm the model server:

```powershell
curl.exe http://127.0.0.1:18080/v1/models
```

Use the port on the results page. The default comes from the first address in `TURBOFORGE_AI_ALLOWED_LOCAL_ORIGINS` in `.env`; the template uses `18080`. Port `8090` is the web page, not the model API.

Example call:

```powershell
curl.exe http://127.0.0.1:18080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"<id from the previous call>\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain KV cache in one sentence\"}]}"
```

## Which engine to use after conversion

| Your model | Use | File |
| --- | --- | --- |
| Dense | [ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem): KV window on the GPU, the rest in RAM | `.ninfer` |
| MoE on RTX 50 (compute 12.0), and the model is Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash | [Basalt](https://github.com/jesdga95/basalt) | `.basalt` |
| Other MoE, including Qwen3.6-35B-A3B | [Strata](https://github.com/Niko1221/Strata) | GGUF |

The page shows the repository link and the launch steps. The three file types are not drop-in replacements. Datacenter Blackwell (compute 10.0) uses Strata, not Basalt.

The standalone Qwen3.5-0.8B GGUF path is limited to 1024 tokens; the page fills that in. For a longer context, use ninfer-fusion-kvmem's KV tiers. Set concurrency when you start the server, after checking VRAM.

## While you use it

- The HF path keeps existing MTP/DFlash2 files. To actually run those heads, use a compatible NInfer server.
- New Qwen3.5 quantization uses GPTQ or NInfer. An existing standard AWQ checkpoint can be imported.
- The quality gate uses English WikiText by default. Bring your own text for other languages, code, or math.
- Built for one person on one PC. Add your own auth and access control before exposing it on the public internet.

## If something fails

| What you see | What to do |
| --- | --- |
| The window closes at once | Read the error, then run `Start TurboForge.bat --check` |
| The page does not open | Use the address printed in the window, not a memorized 8090 |
| The API does not answer | Use the port on the results page and open `/v1/models` first |
| Disk is full | Delete failed builds in the page; finished models and originals stay |
| It fails after you moved the folder | Delete that virtual environment and start again; keep models and builds |

## Related projects

- [ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem): dense models, long context
- [Strata](https://github.com/Niko1221/Strata): MoE on RTX 20/30/40/50
- [Basalt](https://github.com/jesdga95/basalt): published MoE models on RTX 50
- [NInfer upstream](https://github.com/iamwavecut/ninfer-all): conversion and engine source

Licenses and model lists belong to those repositories. `ninfer-kvmem-actkv` has no license file and is not recommended.