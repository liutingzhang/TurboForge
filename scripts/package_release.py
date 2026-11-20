"""Build a complete Windows source + runtime release, without local environments."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from turboforge.bootstrap import local_tools
from turboforge.windows_runtime import collect
from turboforge.distribution import release_path
from turboforge.assistant.config import child_environment


def prepare_python_runtime() -> Path:
    from turboforge.bootstrap import cache_root, download
    spec = json.loads((ROOT/"src/turboforge/data/python-runtime.json").read_text(encoding="utf-8"))
    bundled = os.environ.get("TURBOFORGE_RELEASE_PYTHON")
    if bundled:
        path = Path(bundled)
        if path.stat().st_size != spec["size"] or _digest(path) != spec["sha256"]:
            raise RuntimeError("Bundled Python archive does not match the pinned receipt")
        return path
    return download(spec["url"],cache_root()/"downloads"/spec["filename"],
                    sha256=spec["sha256"],size=spec["size"])


def _digest(value: Path | bytes) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    checksum = hashlib.sha256()
    with value.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _write_parts(files: dict[str, Path | bytes], output: Path, max_bytes: int | None,
                 *, replace: bool = False, language: str = "zh") -> list[Path]:
    if max_bytes is not None and not 65536 <= max_bytes < 2 * 1024**3:
        raise ValueError("Each part limit must be at least 64 KiB and strictly below 2 GiB")
    groups, group, estimated = [], [], 0
    limit = max_bytes - min(4 * 1024**2, max_bytes // 10) if max_bytes else None
    for name, value in sorted(files.items()):
        size = len(value) if isinstance(value, bytes) else value.stat().st_size
        bound = size + size // 1000 + 1024 + len(name.encode("utf-8")) * 2
        if limit and bound > limit:
            raise RuntimeError("A file exceeds the part limit: " + name)
        if limit and estimated + bound > limit:
            groups.append(group)
            group, estimated = [], 0
        group.append((name, value))
        estimated += bound
    if group:
        groups.append(group)
    if max_bytes:
        output.mkdir(parents=True, exist_ok=True)
        label = "chinese" if language == "zh" else "english"
        paths = [output / f"TurboForge-Windows-x64-{label}-part{i+1}.zip" for i in range(len(groups))]
        sums = output / "SHA256SUMS.txt"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        paths, sums = [output], output.with_suffix(output.suffix + ".sha256")
    partials = [path.with_suffix(path.suffix + ".part") for path in paths]
    if any(path.exists() for path in partials) or (not replace and any(path.exists() for path in [*paths, sums])):
        raise RuntimeError("Delivery already exists; choose a new output directory")
    try:
        for group, partial in zip(groups, partials):
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for name, value in group:
                    if isinstance(value, bytes):
                        archive.writestr("TurboForge/" + name, value)
                    else:
                        with value.open("rb") as src, archive.open("TurboForge/" + name, "w", force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            if max_bytes and partial.stat().st_size > max_bytes:
                raise RuntimeError("Compressed ZIP exceeded its configured limit")
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip():
                    raise RuntimeError("Release package CRC check failed")
            print("Checked part: " + partial.name, flush=True)
        for partial, path in zip(partials, paths):
            os.replace(partial, path)
        sums.write_text("".join(_digest(path) + "  " + path.name + "\n" for path in paths), encoding="utf-8")
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    return paths


def package(output: Path, *, replace: bool = False, language: str = "zh", max_bytes: int | None = None) -> Path | list[Path]:
    from turboforge.frontend import frontend_current
    if not frontend_current(ROOT/"web"):
        raise RuntimeError("The frontend lacks a valid content receipt; build the pages and run scripts/frontend_receipt.py first")
    source_value = os.environ.get("TURBOFORGE_RELEASE_SOURCE")
    engine_value = os.environ.get("TURBOFORGE_RELEASE_ENGINE")
    if source_value and engine_value:
        source, engine = Path(source_value), Path(engine_value)
    else:
        tools = local_tools()
        if tools is None:
            raise RuntimeError("No matching engine and source available to package; complete a local conversion acceptance first")
        source, engine = tools
    version = subprocess.run([str(engine / "ninfer.exe"), "--version"], capture_output=True, text=True, check=True, env=child_environment()).stdout
    architectures = re.search(r",\s*([0-9a, -]+)\)", version)
    if architectures:
        capabilities = sorted({int(value.rstrip("a")) / 10 for value in re.findall(r"\d+a?", architectures[1])})
    else:
        # Shared-runtime builds print a one-line version; BUILD-INFO records the same
        # architecture list the archive was compiled for.
        info = engine / "BUILD-INFO.json"
        if not info.is_file():
            raise RuntimeError("Cannot confirm the engine build architecture; a release with compatibility claims cannot be produced")
        capabilities = sorted({int(value.rstrip("a")) / 10 for value in json.loads(info.read_text(encoding="utf-8"))["architectures"]})
    files = {}
    python_archive = prepare_python_runtime()
    files["runtime/python/"+python_archive.name] = python_archive
    # Runtime needs the app, the built page, its frontend sources, and the two
    # launcher scripts. Tests, docs, configs, CI and release tooling stay out.
    launcher_scripts = {"start_web.py", "start_web.ps1", "frontend_receipt.py"}
    for directory in ("src", "scripts", "web/dist", "web/src", "web/public"):
        for path in (ROOT / directory).rglob("*"):
            if directory == "scripts" and path.name not in launcher_scripts:
                continue
            name = path.relative_to(ROOT).as_posix()
            if directory == "web/dist":
                name = "runtime/web/" + path.relative_to(ROOT / directory).as_posix()
            if path.is_file() and release_path(name):
                files[name] = path
    for name in ("启动 TurboForge.bat", "Start TurboForge.bat", "INSTALL_ZH.md", "INSTALL_EN.md", "README.md", "README_EN.md", "pyproject.toml", "LICENSE", "NOTICE", "requirements-web.txt", "web/package.json", "web/package-lock.json", "web/index.html", "web/vite.config.ts", "web/tsconfig.json"):
        path = ROOT / name
        if path.is_file():
            files[name] = path
    # Shared-runtime engines load their kernels from backends/<cuda>/ beside the
    # launcher, so the bundled layout must keep that relative structure.
    for path in engine.rglob("*"):
        name = "runtime/bin/" + path.relative_to(engine).as_posix()
        if path.is_file() and release_path(name) and path.suffix.lower() in (".exe", ".dll", ".json", ".txt"):
            files[name] = path
    # Runtime redistributables resolved from the host (MSVC family) go beside them.
    for path in collect(engine).values():
        if path.is_file() and release_path(path.name) and path.suffix.lower() in (".exe", ".dll", ".json", ".txt"):
            files.setdefault("runtime/bin/" + path.name, path)
    for path in engine.iterdir():
        if path.is_file() and any(word in path.name.lower() for word in ("license", "eula", "notice")) and release_path(path.name):
            files["runtime/licenses/" + path.name] = path
    # A staged engine keeps its full license set (engine LICENSE, CUDA texts)
    # in a sibling licenses directory; reuse it instead of relying on the host.
    staged_licenses = engine.parent / "licenses"
    if staged_licenses.is_dir():
        for path in staged_licenses.iterdir():
            if path.is_file() and release_path(path.name):
                files.setdefault("runtime/licenses/" + path.name, path)
    # Preserve licenses for every bundled frontend dependency, including tooling.
    for path in (ROOT/"web/node_modules").rglob("*"):
        if path.is_file() and path.name.lower().startswith(("license", "licence", "notice", "copying")):
            name = "licenses/web/" + path.relative_to(ROOT/"web/node_modules").as_posix().replace("/node_modules/", "/dependencies/")
            if release_path(name):
                files[name] = path
    cuda = os.environ.get("CUDA_PATH")
    if cuda:
        for name in ("EULA.txt", "LICENSE"):
            license_path = Path(cuda) / name
            if license_path.is_file():
                files["runtime/licenses/CUDA-" + name] = license_path
    tracked = subprocess.run(["git", "-C", str(source), "ls-files", "-z"], capture_output=True, check=True, env=child_environment()).stdout.decode().split("\0") if (source / ".git").exists() else []
    paths = [source / name for name in tracked if name] if tracked else [p for p in source.rglob("*") if p.is_file()]
    if tracked:
        added = subprocess.run(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"], capture_output=True, check=True, env=child_environment()).stdout.decode().split("\0")
        paths += [source / name for name in added if name and Path(name).suffix.lower() in (".cu", ".cuh", ".c", ".cpp", ".h", ".hpp", ".cmake", ".py")]
    paths += list((source / "tools").rglob("*.py"))  # include the tested local converter port
    blocked_parts = {"tests", "test", "docs", "examples", "example", "__pycache__"}
    for path in paths:
        relative = path.relative_to(source)
        name = relative.as_posix()
        if any(part.lower() in blocked_parts for part in relative.parts):
            continue
        if relative.suffix.lower() in {".md", ".rst"}:
            continue
        if path.is_file() and release_path(name) and relative.name not in {"CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"}:
            files["runtime/source/" + name] = path
    if "runtime/source/tools/convert/__main__.py" not in files:
        raise RuntimeError("The converter is incomplete")
    if not (ROOT / "web/dist/index.html").is_file():
        raise RuntimeError("Run the launcher script to build the pages automatically before packaging")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < sum(p.stat().st_size for p in files.values()) + 256 * 1024**2:
        raise RuntimeError("Insufficient disk space for the release package")
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, env=child_environment()).stdout.strip() if (source / ".git").exists() else json.loads((source / ".turboforge-source.json").read_text())["revision"]
    files["runtime/source/.turboforge-source.json"] = json.dumps({"revision": revision, "local_modifications_included": True}).encode()
    files["START_HERE.txt"] = (("请先阅读 INSTALL_ZH.md。将同一语言的全部 ZIP 解压到同一目录，再双击 启动 TurboForge.bat。\n"
                                "首次运行需要联网；程序界面主要为中文。不要混用新旧发布包。\n") if language == "zh" else
                               ("Read INSTALL_EN.md first. Extract ALL ZIP parts of this language into the same folder, then double-click Start TurboForge.bat.\n"
                                "First launch requires internet. The application UI is currently Chinese. Do not mix release versions.\n")).encode("utf-8")
    project_checksums = {name: _digest(value) for name, value in files.items()}
    checksums = {name.removeprefix("runtime/"): digest for name, digest in project_checksums.items() if name.startswith("runtime/")}
    files["runtime/bundle.json"] = json.dumps({"revision": revision, "gpu_capabilities": capabilities, "version": version.strip(), "files": checksums}, indent=2).encode()
    project_checksums["runtime/bundle.json"] = _digest(files["runtime/bundle.json"])
    files["release-files.json"] = json.dumps({"language": language, "files": project_checksums}, indent=2).encode()
    paths = _write_parts(files, output, max_bytes, replace=replace, language=language)
    return paths if max_bytes else paths[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--max-part-bytes", type=int, default=1_800_000_000)
    args = parser.parse_args()
    try:
        for language in (("zh", "en") if args.language == "both" else (args.language,)):
            paths = package(args.output / language, language=language, max_bytes=args.max_part_bytes)
            print(json.dumps([str(path) for path in paths]), flush=True)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
