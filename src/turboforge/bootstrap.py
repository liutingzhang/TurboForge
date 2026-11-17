"""Managed downloads and tool installation. This module uses only the stdlib."""
from __future__ import annotations

from .assistant.config import child_environment
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable
import urllib.error
import urllib.request
import zipfile

Emit = Callable[[str], None]
RELEASE = "2026-11-12.4"
REVISION = "e2e41d4050c448253f3d69d380809a41f67b8ab4"
REPOSITORY = "https://github.com/iamwavecut/ninfer-all"
PACKAGES = {
    "13": ("ninfer-windows-x64-cuda13.4-sm75-80-86-89-120a-e2e41d4050c4", (
        "1f2ed2f761b9a3837564716c954ce502e54111de540a7ff888780d18fad95427",
        "0ac71c363e9ac0c92b6d3b5c7a8dff0dde98cc881799386352a04e787db248ee")),
    "12": ("ninfer-windows-x64-cuda12.9-sm70-75-80-86-89-e2e41d4050c4", (
        "a93d26eee17c1e5bcafa933c247b0a1c92551001053e511374c2e169ffb0c416",
        "59a50345a2d986c37bafc340a679251119e1d6feb63d6d44ec2da4622cbc1949")),
}


def cache_root() -> Path:
    root = Path(os.environ.get("TURBOFORGE_HOME", Path(__file__).resolve().parents[2] / ".turboforge"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TurboForge/0.3"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def download(url: str, destination: Path, *, sha256: str | None = None,
             size: int | None = None, emit: Emit = print) -> Path:
    """Resume interrupted downloads; promote only complete, checked files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".sha256")
    expected = sha256 or (receipt.read_text().strip() if receipt.is_file() else None)
    if destination.is_file() and expected and digest(destination) == expected:
        emit(f"Reusing verified download: {destination.name}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "TurboForge/0.3"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        completed = re.fullmatch(r"bytes \*/(\d+)", exc.headers.get("Content-Range", "")) if exc.code == 416 else None
        if completed and offset and offset == int(completed[1]) and (size is None or size == offset):
            actual = digest(partial)
            if sha256 and actual != sha256:
                partial.unlink()
                raise RuntimeError(f"{destination.name} failed SHA-256 verification; the corrupt temporary file was removed") from exc
            partial.replace(destination)
            receipt.write_text(actual, encoding="ascii")
            return destination
        if exc.code in (401, 403):
            raise RuntimeError(f"Download refused (HTTP {exc.code}): {url}. Private or gated models require access approval first.") from exc
        raise
    with response:
        append = response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("The server returned a mismatched resume position; stopped to avoid corrupting the file")
        if not append:
            offset = 0
        remaining = int(response.headers.get("Content-Length", "0"))
        total = size or (offset + remaining if remaining else None)
        if total and shutil.disk_usage(destination.parent).free < max(0, total - offset) + 128 * 1024**2:
            raise RuntimeError(f"Insufficient disk space to download {destination.name}; expected total size {total / 1024**3:.2f} GiB")
        emit(f"Downloading {destination.name}" + (f" ({total / 1024**2:.0f} MiB)" if total else ""))
        last = 0.0
        with partial.open("ab" if append else "wb") as out:
            received = offset
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
                received += len(chunk)
                if time.monotonic() - last > 2:
                    emit(f"{destination.name}: {received / 1024**2:.0f} MiB" + (f" / {total / 1024**2:.0f} MiB" if total else ""))
                    last = time.monotonic()
        if total and received != total:
            raise RuntimeError(f"{destination.name} download incomplete; starting again resumes from the breakpoint")
    actual = digest(partial)
    if sha256 and actual != sha256:
        partial.unlink()
        raise RuntimeError(f"{destination.name} failed SHA-256 verification; the corrupt temporary file was removed")
    partial.replace(destination)
    receipt.write_text(actual, encoding="ascii")
    return destination


def safe_member(root: Path, name: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/"))
    target = (root / str(parts)).resolve()
    if parts.is_absolute() or ":" in name or root.resolve() not in target.parents:
        raise ValueError(f"The archive contains an out-of-bounds path: {name}")
    return target


def extract_zip(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        required = sum(info.file_size for info in source.infolist())
        if shutil.disk_usage(output.parent).free < required + 128 * 1024**2:
            raise RuntimeError("Insufficient space to extract")
        for info in source.infolist():
            target = safe_member(output, info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links in the archive are not accepted")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)


def run(command: list[str], emit: Emit, *, cwd: Path | None = None, env: dict | None = None) -> None:
    last_progress = 0.0
    with subprocess.Popen(command, cwd=cwd, env=child_environment(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            progress = re.fullmatch(r"Progress (\d+) of (\d+)\s*", line)
            if progress:
                received, total = map(int, progress.groups())
                if time.monotonic() - last_progress >= 2 or received == total:
                    emit(f"Dependency download: {received / 1024**2:.1f} MiB" + (f" / {total / 1024**2:.1f} MiB ({received / total:.0%})" if total else ""))
                    last_progress = time.monotonic()
            else:
                emit(line.rstrip())
        code = process.wait()
    if code:
        raise RuntimeError(f"The automatic install step failed (exit code {code}); see the log above")


def ensure_dependencies(jailbreak: str, emit: Emit = print) -> None:
    requirements = {"torch": ("torch>=2.14.1,<3", (2, 14, 1)), "numpy": ("numpy>=1.26,<3", (1, 26, 0)),
                    "safetensors": ("safetensors>=0.4.5,<1", (0, 4, 5)),
                    "gguf": ("gguf>=0.19,<1", (0, 19, 0))}
    if jailbreak == "abliteration":
        requirements["transformers"] = ("transformers>=5.9,<6", (5, 9, 0))
    if jailbreak == "heretic":
        requirements["heretic"] = ("heretic-llm", (0, 0, 0))
    missing = []
    for module, (spec, minimum) in requirements.items():
        if importlib.util.find_spec(module) is None:
            missing.append(spec)
            continue
        try:
            installed = importlib.metadata.version("heretic-llm" if module == "heretic" else module)
            numbers = tuple(int(n) for n in re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", installed).groups(default="0"))
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)
            continue
        maximum_major = {"torch": 3, "numpy": 3, "safetensors": 1, "gguf": 1, "transformers": 6}.get(module)
        if numbers < minimum or (maximum_major is not None and numbers[0] >= maximum_major):
            missing.append(spec)
    if missing:
        emit("Automatically installing conversion dependencies: " + ", ".join(missing))
        progress = ["--progress-bar", "raw"] if int(importlib.metadata.version("pip").split(".")[0]) >= 26 else []
        run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *progress, *missing], emit)
        importlib.invalidate_caches()
    else:
        emit("Conversion dependencies are ready; no repeat download needed")


def ensure_node(root: Path, emit: Emit = print) -> Path:
    existing = shutil.which("npm.cmd") or shutil.which("npm")
    node = shutil.which("node")
    if existing and node:
        version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=15, env=child_environment())
        numbers = tuple(int(value) for value in version.stdout.strip().lstrip("v").split(".")) if version.returncode == 0 else ()
        if numbers and ((numbers[0] == 20 and numbers >= (20,19)) or numbers >= (22,12)):
            return Path(existing)
    releases = json.loads(read_url("https://nodejs.org/dist/index.json"))
    version = next(item["version"] for item in releases if item.get("lts") and "win-x64-zip" in item["files"])
    name = f"node-{version}-win-x64"
    checks = read_url(f"https://nodejs.org/dist/{version}/SHASUMS256.txt").decode()
    checksum = next(line.split()[0] for line in checks.splitlines() if line.split()[-1] == name + ".zip")
    destination = root / "node" / name
    if not (destination / "npm.cmd").is_file():
        archive = download(f"https://nodejs.org/dist/{version}/{name}.zip", root / "downloads" / (name + ".zip"), sha256=checksum, emit=emit)
        extract_zip(archive, root / "node")
    os.environ["PATH"] = str(destination) + os.pathsep + os.environ.get("PATH", "")
    return destination / "npm.cmd"


def local_tools() -> tuple[Path, Path] | None:
    """Reuse explicit paths or a matching existing local installation."""
    bundled = Path(__file__).resolve().parents[2] / "runtime"
    manifest = bundled / "bundle.json"
    if manifest.is_file() and not any(os.environ.get(name) for name in ("TURBOFORGE_NINFER_SRC", "TURBOFORGE_NINFER_BIN")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        required = {"bin/ninfer.exe", "bin/ninfer-serve.exe", "bin/ninfer_bench.exe", "source/tools/convert/__main__.py"}
        if not required.issubset(data["files"]):
            raise RuntimeError("The bundle manifest is corrupt or missing required files; re-extract the complete release package")
        for name, expected in data["files"].items():
            path = safe_member(bundled, name)
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"Bundle file corrupt or missing: {name}. Re-extract the complete release package.")
        if data.get("gpu_capabilities"):
            result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, env=child_environment())
            if result.returncode or not result.stdout.strip() or float(result.stdout.splitlines()[0]) not in data["gpu_capabilities"]:
                return None  # The official package will be selected for this GPU.
        return bundled / "source", bundled / "bin"
    source_value = os.environ.get("TURBOFORGE_NINFER_SRC")
    if not source_value:
        return None
    source = Path(source_value)
    if not (source / "tools/convert/__main__.py").is_file():
        return None
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, env=child_environment()).stdout.strip() if shutil.which("git") else ""
    engines = [Path(os.environ["TURBOFORGE_NINFER_BIN"])] if os.environ.get("TURBOFORGE_NINFER_BIN") else [source.parent / "ninfer-ported", source.parent / "ninfer"]
    for engine in engines:
        if not all((engine / name).is_file() for name in ("ninfer.exe", "ninfer-serve.exe", "ninfer_bench.exe")):
            continue
        info = engine / "BUILD-INFO.json"
        if info.is_file():
            data = json.loads(info.read_text(encoding="utf-8"))
            if revision and data.get("source_revision") == revision:
                return source, engine
        elif engine.name == "ninfer-ported" and revision == "6eb311fd3d4c6b6a69f321de0becc4664553d76f":
            version = subprocess.run([str(engine / "ninfer.exe"), "--version"], capture_output=True, text=True, timeout=15, env=child_environment())
            if version.returncode == 0 and "2026-11-08.1" in version.stdout:
                return source, engine
    return None


def cuda_package() -> str:
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("One-click deployment supports Windows x64; the current system or processor architecture is not supported")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15,
                                env=child_environment())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot detect the NVIDIA driver. Install the GPU vendor's official driver, then double-click to start again; the CUDA Toolkit is not required. https://www.nvidia.com/Download/index.aspx") from exc
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("No usable NVIDIA driver detected; the inference engine cannot be selected")
    capability, driver = result.stdout.splitlines()[0].split(",")
    capability = float(capability)
    if capability not in (7.0, 7.5, 8.0, 8.6, 8.9, 12.0):
        raise RuntimeError(f"This GPU's compute capability {capability} is outside the support range of the published Windows engines")
    if int(driver.strip().split(".")[0]) >= 580 and capability >= 7.5:
        return "13"
    if capability >= 12:
        raise RuntimeError("The RTX 50 series needs an NVIDIA driver with CUDA 13 support; update the GPU driver and retry")
    if tuple(int(n) for n in driver.strip().split(".")[:2]) < (528, 33):
        raise RuntimeError("The current NVIDIA driver is too old; CUDA 12 engines need at least Windows driver 528.33. Update the driver and retry")
    return "12"


def ensure_tools(root: Path, emit: Emit = print, *, small_model: bool = False) -> tuple[Path, Path]:
    installed = local_tools()
    if installed:
        emit("Reusing the matching engine and converter already on this machine: " + str(installed[1]))
        return installed
    if sys.platform != "win32":
        raise RuntimeError("The current automatic installer targets Windows x64")
    source = ensure_source(root, emit)
    if small_model and not (source / "tools/convert/qwen3_5_gguf.py").is_file():
        raise RuntimeError("The officially downloadable engines do not yet include a verified adaptation for this small model; stopped before downloading the large engine. Auto mode can reuse an adapted version already on this machine.")
    package = cuda_package()
    name, hashes = PACKAGES[package]
    install = root / "tools" / name
    marker = install / "installation.json"
    if marker.is_file():
        data = json.loads(marker.read_text())
        engine = install / data["engine"]
    else:
        emit(f"Automatically fetching NInfer {RELEASE} / CUDA {package}; the release package includes the CUDA runtime, no Toolkit install needed")
        parts = [download(f"{REPOSITORY}/releases/download/{RELEASE}/{name}.7z.{i:03}",
                          root / "downloads" / f"{name}.7z.{i:03}", sha256=checksum, emit=emit)
                 for i, checksum in enumerate(hashes, 1)]
        extractor = download("https://www.7-zip.org/a/7zr.exe", root / "downloads/7zr.exe", emit=emit)
        listing = subprocess.run([str(extractor), "l", "-slt", str(parts[0])], capture_output=True, text=True, errors="replace", check=True, env=child_environment()).stdout
        entries = listing.split("----------", 1)
        if len(entries) != 2:
            raise RuntimeError("Cannot read the engine archive directory")
        for line in entries[1].splitlines():
            if line.startswith("Path = "):
                safe_member(install, line[7:])
            if line.startswith(("Symbolic Link = ", "Hard Link = ")):
                raise RuntimeError("The engine archive contains unsupported links")
        sizes = [int(value) for value in re.findall(r"^Size = (\d+)$", entries[1], re.M)]
        if shutil.disk_usage(root).free < sum(sizes) + 256 * 1024**2:
            raise RuntimeError("Insufficient space to extract the engine; downloaded volumes are kept for next time")
        with tempfile.TemporaryDirectory(prefix="engine-", dir=root) as temporary:
            staging = Path(temporary)
            run([str(extractor), "x", str(parts[0]), f"-o{staging}", "-y"], emit)
            candidates = sorted(staging.rglob("ninfer-serve.exe"), key=lambda p: len(p.parts))
            if not candidates:
                raise RuntimeError("The extracted release package has no ninfer-serve.exe")
            relative = candidates[0].parent.relative_to(staging)
            if install.exists():
                raise RuntimeError("The engine install directory already has unconfirmed content; check the install log")
            install.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(install))
            marker.write_text(json.dumps({"release": RELEASE, "revision": REVISION, "engine": str(relative)}), encoding="utf-8")
            engine = install / relative
    if not all((engine / name).is_file() for name in ("ninfer.exe", "ninfer-serve.exe", "ninfer_bench.exe")):
        raise RuntimeError("Engine files are incomplete; automatic preparation did not succeed")
    return source, engine


def ensure_source(root: Path, emit: Emit = print) -> Path:
    source = root / "sources" / REVISION
    if not (source / "tools/convert/__main__.py").is_file():
        archive = download(f"https://codeload.github.com/iamwavecut/ninfer-all/zip/{REVISION}",
                           root / "downloads" / f"ninfer-{REVISION}.zip", emit=emit)
        with tempfile.TemporaryDirectory(prefix="source-", dir=root) as temporary:
            staging = Path(temporary)
            extract_zip(archive, staging)
            unpacked = staging / f"ninfer-all-{REVISION}"
            if not (unpacked / "tools/convert/__main__.py").is_file():
                raise RuntimeError("The source archive lacks the converter entry point")
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(unpacked), str(source))
        (source / ".turboforge-source.json").write_text(json.dumps({"revision": REVISION, "archive_sha256": digest(archive)}), encoding="utf-8")
    return source
