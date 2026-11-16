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
    request = urllib.request.Request(url, headers={"User-Agent": "NInfer-TurboForge/0.3"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def download(url: str, destination: Path, *, sha256: str | None = None,
             size: int | None = None, emit: Emit = print) -> Path:
    """Resume interrupted downloads; promote only complete, checked files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".sha256")
    expected = sha256 or (receipt.read_text().strip() if receipt.is_file() else None)
    if destination.is_file() and expected and digest(destination) == expected:
        emit(f"复用已校验下载：{destination.name}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "NInfer-TurboForge/0.3"}
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
                raise RuntimeError(f"{destination.name} SHA-256 校验失败，损坏的临时文件已清除") from exc
            partial.replace(destination)
            receipt.write_text(actual, encoding="ascii")
            return destination
        if exc.code in (401, 403):
            raise RuntimeError(f"下载被拒绝（HTTP {exc.code}）：{url}。私有或受限模型需要先获得访问权限。") from exc
        raise
    with response:
        append = response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("服务器返回了不匹配的续传位置，已停止以避免损坏文件")
        if not append:
            offset = 0
        remaining = int(response.headers.get("Content-Length", "0"))
        total = size or (offset + remaining if remaining else None)
        if total and shutil.disk_usage(destination.parent).free < max(0, total - offset) + 128 * 1024**2:
            raise RuntimeError(f"下载 {destination.name} 的磁盘空间不足，预计总大小 {total / 1024**3:.2f} GiB")
        emit(f"下载 {destination.name}" + (f"（{total / 1024**2:.0f} MiB）" if total else ""))
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
            raise RuntimeError(f"{destination.name} 下载未完成；再次开始将从断点继续")
    actual = digest(partial)
    if sha256 and actual != sha256:
        partial.unlink()
        raise RuntimeError(f"{destination.name} SHA-256 校验失败，损坏的临时文件已清除")
    partial.replace(destination)
    receipt.write_text(actual, encoding="ascii")
    return destination


def safe_member(root: Path, name: str) -> Path:
    parts = PurePosixPath(name.replace("\\", "/"))
    target = (root / str(parts)).resolve()
    if parts.is_absolute() or ":" in name or root.resolve() not in target.parents:
        raise ValueError(f"归档中含越界路径：{name}")
    return target


def extract_zip(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        required = sum(info.file_size for info in source.infolist())
        if shutil.disk_usage(output.parent).free < required + 128 * 1024**2:
            raise RuntimeError("解压空间不足")
        for info in source.infolist():
            target = safe_member(output, info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("不接受归档中的符号链接")
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
                    emit(f"依赖下载：{received / 1024**2:.1f} MiB" + (f" / {total / 1024**2:.1f} MiB（{received / total:.0%}）" if total else ""))
                    last_progress = time.monotonic()
            else:
                emit(line.rstrip())
        code = process.wait()
    if code:
        raise RuntimeError(f"自动安装步骤失败（退出码 {code}），详见上方日志")


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
        emit("自动安装转换依赖：" + ", ".join(missing))
        progress = ["--progress-bar", "raw"] if int(importlib.metadata.version("pip").split(".")[0]) >= 26 else []
        run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *progress, *missing], emit)
        importlib.invalidate_caches()
    else:
        emit("转换依赖已就绪，无需重复下载")


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
            raise RuntimeError("整合包清单损坏或缺失必要文件，请重新解压完整发布包")
        for name, expected in data["files"].items():
            path = safe_member(bundled, name)
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"整合包文件损坏或缺失：{name}。请重新解压完整发布包。")
        if data.get("gpu_capabilities"):
            result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, env=child_environment())
            if result.returncode or not result.stdout.strip() or float(result.stdout.splitlines()[0]) not in data["gpu_capabilities"]:
                return None  # The official package will be selected for this GPU.
        return bundled / "source", bundled / "bin"
    source = Path(os.environ.get("TURBOFORGE_NINFER_SRC", r"D:\llm\ninfer-upstream"))
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
        raise RuntimeError("一键部署支持 Windows x64；当前系统或处理器架构不受支持")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15,
                                env=child_environment())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("无法检测 NVIDIA 驱动。请安装显卡厂商官方驱动后重新双击启动；无需安装 CUDA Toolkit。https://www.nvidia.com/Download/index.aspx") from exc
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("未检测到可用的 NVIDIA 驱动，无法选择推理引擎")
    capability, driver = result.stdout.splitlines()[0].split(",")
    capability = float(capability)
    if capability not in (7.0, 7.5, 8.0, 8.6, 8.9, 12.0):
        raise RuntimeError(f"此 GPU 的计算能力 {capability} 不在已发布 Windows 引擎的支持范围内")
    if int(driver.strip().split(".")[0]) >= 580 and capability >= 7.5:
        return "13"
    if capability >= 12:
        raise RuntimeError("RTX 50 系列需要支持 CUDA 13 的 NVIDIA 驱动，请更新显卡驱动后重试")
    if tuple(int(n) for n in driver.strip().split(".")[:2]) < (528, 33):
        raise RuntimeError("当前 NVIDIA 驱动过旧；CUDA 12 引擎至少需要 Windows 驱动 528.33，请更新驱动后重试")
    return "12"


def ensure_tools(root: Path, emit: Emit = print, *, small_model: bool = False) -> tuple[Path, Path]:
    installed = local_tools()
    if installed:
        emit("复用本机已有匹配引擎与转换器：" + str(installed[1]))
        return installed
    if sys.platform != "win32":
        raise RuntimeError("当前自动安装器面向 Windows x64")
    source = ensure_source(root, emit)
    if small_model and not (source / "tools/convert/qwen3_5_gguf.py").is_file():
        raise RuntimeError("官方可下载引擎尚未包含此小模型的已验证适配；已在下载大体积引擎前停止。当前自动模式可复用本机已有适配版。")
    package = cuda_package()
    name, hashes = PACKAGES[package]
    install = root / "tools" / name
    marker = install / "installation.json"
    if marker.is_file():
        data = json.loads(marker.read_text())
        engine = install / data["engine"]
    else:
        emit(f"自动获取 NInfer {RELEASE} / CUDA {package}；发行包包含 CUDA 运行库，无需安装 Toolkit")
        parts = [download(f"{REPOSITORY}/releases/download/{RELEASE}/{name}.7z.{i:03}",
                          root / "downloads" / f"{name}.7z.{i:03}", sha256=checksum, emit=emit)
                 for i, checksum in enumerate(hashes, 1)]
        extractor = download("https://www.7-zip.org/a/7zr.exe", root / "downloads/7zr.exe", emit=emit)
        listing = subprocess.run([str(extractor), "l", "-slt", str(parts[0])], capture_output=True, text=True, errors="replace", check=True, env=child_environment()).stdout
        entries = listing.split("----------", 1)
        if len(entries) != 2:
            raise RuntimeError("无法读取引擎归档目录")
        for line in entries[1].splitlines():
            if line.startswith("Path = "):
                safe_member(install, line[7:])
            if line.startswith(("Symbolic Link = ", "Hard Link = ")):
                raise RuntimeError("引擎归档含不支持的链接")
        sizes = [int(value) for value in re.findall(r"^Size = (\d+)$", entries[1], re.M)]
        if shutil.disk_usage(root).free < sum(sizes) + 256 * 1024**2:
            raise RuntimeError("引擎解压空间不足，下载分卷已保留供下次使用")
        with tempfile.TemporaryDirectory(prefix="engine-", dir=root) as temporary:
            staging = Path(temporary)
            run([str(extractor), "x", str(parts[0]), f"-o{staging}", "-y"], emit)
            candidates = sorted(staging.rglob("ninfer-serve.exe"), key=lambda p: len(p.parts))
            if not candidates:
                raise RuntimeError("发行包解压后没有 ninfer-serve.exe")
            relative = candidates[0].parent.relative_to(staging)
            if install.exists():
                raise RuntimeError("引擎安装目录已有未确认内容，请检查安装日志")
            install.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(install))
            marker.write_text(json.dumps({"release": RELEASE, "revision": REVISION, "engine": str(relative)}), encoding="utf-8")
            engine = install / relative
    if not all((engine / name).is_file() for name in ("ninfer.exe", "ninfer-serve.exe", "ninfer_bench.exe")):
        raise RuntimeError("引擎文件不完整，自动准备未成功")
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
                raise RuntimeError("源码归档缺少转换器入口")
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(unpacked), str(source))
        (source / ".turboforge-source.json").write_text(json.dumps({"revision": REVISION, "archive_sha256": digest(archive)}), encoding="utf-8")
    return source
