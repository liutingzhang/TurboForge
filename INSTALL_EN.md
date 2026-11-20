# TurboForge clean Windows installation bundle

Chinese and English documentation and launchers are provided. Both editions contain the same application; the UI and some runtime messages are currently Chinese. Choose one complete language set. Never mix languages, release versions, or files from an older installation.

## 1. Download and verify

- Download ALL `TurboForge-Windows-x64-english-part*.zip` files and `SHA256SUMS.txt` from the same release directory.
- These are **independent ZIP archives**, not `.z01` volumes. Every part must be downloaded and extracted. A single part is not a standalone installation.
- Each ZIP is limited to 1,800,000,000 bytes, below GitHub's requirement that each release asset be smaller than 2 GiB (2,147,483,648 bytes).
- In PowerShell, run `Get-FileHash .\*english-part*.zip -Algorithm SHA256`. Compare every filename and hash with `SHA256SUMS.txt`. Do not start an incomplete or mismatched download.

GitHub release limits: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

## 2. Extract and start

Extract ALL parts into the same new, empty destination, merging their contents into one `TurboForge` directory. Do not create a separate installation for each part, run from inside a ZIP, or overwrite an older directory containing models and jobs.

From the download directory:

```powershell
Get-ChildItem '*english-part*.zip' | Sort-Object Name | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath '.\installed' -Force
}
```

Open `installed\TurboForge` and double-click **Start TurboForge.bat**. The Chinese launcher is **启动 TurboForge.bat**. Keep the command window open. The browser opens automatically; otherwise, use the local URL printed in the window. The default port is 8090; an available port is selected when it is occupied.

The first launch creates an isolated Python environment under the installation and downloads Web dependencies. No personal `.env` is shipped. Offline help requires no API key. The optional AI assistant requires your own service configuration and credentials; no publisher credentials are included.

## 3. Requirements and first-run downloads

- Windows x64, not ARM64 or 32-bit Windows. Use a writable directory outside Program Files.
- No preinstalled Python, Git, Node.js, or CUDA Toolkit is required. A pinned Python runtime and prebuilt Web assets are included.
- Model conversion and GPU validation require a supported NVIDIA GPU, adequate VRAM, and a working driver. Drivers are not supplied by this bundle. RTX 50-series GPUs and the HF GPTQ route require NVIDIA driver 580 or newer.
- Internet access is still required for first launch and initial route setup. The separate HF GPTQ environment, PyTorch, and other packages may require several GB of downloads. Models are supplied separately. This is not a fully offline distribution.
- Allow at least 15 GB for basic installation and environments, plus additional capacity for models, quantization dependencies, intermediate weights, and quality references. ZIP size does not represent total working-space requirements.

Check the environment without converting a model:

```powershell
.\"Start TurboForge.bat" --check
```

## 4. Support boundaries

- Select a model and run the model check before choosing one of the available routes. An unsupported request is not silently replaced.
- Small Qwen3.5 NInfer routes require an additional adapter that is not present in this bundled upstream runtime. Complete original HF checkpoints can use the integrated HF GPTQ routes. A GGUF is not an original high-precision HF checkpoint.
- Detecting MTP/DFlash2 does not prove that an engine can execute those heads. Unverifiable retention or missing components blocks conversion. Follow the explicit instruction to disable speculative decoding when choosing HF.
- Quality gates assess their selected corpus and metrics, not every model capability. Large-model GPU conversion is outside the scope of this packaging/startup acceptance run.
- Tests on the current host do not substitute for testing a fresh Windows VM or every GPU combination. Check the actual deployment and build reports.

## 5. Included files and privacy

The bundle retains backend source in `src`, frontend source and build configuration in `web`, startup scripts, required engines/DLLs in `runtime/bin`, corresponding converter source in `runtime/source`, the pinned interpreter archive in `runtime/python`, prebuilt pages in `runtime/web`, licenses, bilingual guides, and checksums. The prebuilt pages are required runtime assets, not personal data; they avoid a first-run Node.js installation.

The distribution excludes `dist`, `.env` (including environment templates), `.git`, `.github`, development caches, `node_modules`, tests, worktrees, old releases, private configuration, models, jobs, logs, and chat caches. Required license texts are retained. `runtime/source/.turboforge-source.json` records public upstream provenance only, not user settings.

After startup, the generated `.turboforge` directory contains the installation user's own environments, jobs, and logs. It is not shipped publisher data. Do not redistribute this directory with your credentials or models.

## 6. Troubleshooting

- Missing scripts, DLLs, pages, or damaged receipts: verify every ZIP hash, then extract every part into a new empty directory.
- Installation failure: check internet/proxy access, free disk space, write permissions, and security-software messages. Inspect `.turboforge/logs`; do not disable system-wide protection.
- Unsupported GPU/driver: install the appropriate official NVIDIA driver. A working Web page does not establish support for arbitrary model conversion.
- Unreachable page: keep the launcher window open and use its printed URL, rather than assuming port 8090.
- Before sharing diagnostics, redact usernames, local paths, tokens, and private model information. Never attach `.env` or assistant credentials.
