"""Build a complete Windows source + runtime release, without local environments."""
from __future__ import annotations
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
from turboforge.distribution import private_path
from turboforge.assistant.config import child_environment


def prepare_python_runtime() -> Path:
    from turboforge.bootstrap import cache_root, download
    spec = json.loads((ROOT/"src/turboforge/data/python-runtime.json").read_text(encoding="utf-8"))
    return download(spec["url"],cache_root()/"downloads"/spec["filename"],
                    sha256=spec["sha256"],size=spec["size"])


def package(output: Path, *, replace: bool = False) -> Path:
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
            if path.is_file() and not private_path(path.relative_to(ROOT).as_posix()):
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ("启动 TurboForge.bat", "README.md", "README_EN.md", "pyproject.toml", "LICENSE", "NOTICE", ".env.example", "requirements-web.txt", "web/package.json", "web/package-lock.json", "web/index.html", "web/vite.config.ts", "web/tsconfig.json"):
        path = ROOT / name
        if path.is_file():
            files[name] = path
    # Shared-runtime engines load their kernels from backends/<cuda>/ beside the
    # launcher, so the bundled layout must keep that relative structure.
    for path in engine.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".exe", ".dll", ".json", ".txt"):
            files["runtime/bin/" + path.relative_to(engine).as_posix()] = path
    # Runtime redistributables resolved from the host (MSVC family) go beside them.
    for path in collect(engine).values():
        if path.is_file() and path.suffix.lower() in (".exe", ".dll", ".json", ".txt"):
            files.setdefault("runtime/bin/" + path.name, path)
    for path in engine.iterdir():
        if path.is_file() and any(word in path.name.lower() for word in ("license", "eula", "notice")):
            files["runtime/licenses/" + path.name] = path
    # Preserve licenses for every bundled frontend dependency, including tooling.
    for path in (ROOT/"web/node_modules").rglob("*"):
        if path.is_file() and path.name.lower().startswith(("license", "licence", "notice", "copying")):
            files["licenses/web/" + path.relative_to(ROOT/"web/node_modules").as_posix()] = path
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
        if path.is_file() and not private_path(name) and relative.name not in {"CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"}:
            files["runtime/source/" + name] = path
    if "runtime/source/tools/convert/__main__.py" not in files:
        raise RuntimeError("The converter is incomplete")
    if not (ROOT / "web/dist/index.html").is_file():
        raise RuntimeError("Run the launcher script to build the pages automatically before packaging")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise RuntimeError("The release package already exists; specify a new filename to avoid overwriting an existing delivery")
    if shutil.disk_usage(output.parent).free < sum(p.stat().st_size for p in files.values()) + 256 * 1024**2:
        raise RuntimeError("Insufficient disk space for the release package")
    partial = output.with_suffix(output.suffix + ".part")
    checksums = {}
    project_checksums = {}
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            checksum = hashlib.sha256()
            with path.open("rb") as src, archive.open("TurboForge/" + name, "w", force_zip64=True) as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    checksum.update(chunk); dst.write(chunk)
            if name.startswith("runtime/"):
                checksums[name[len("runtime/"):]] = checksum.hexdigest()
            project_checksums[name] = checksum.hexdigest()
            if name.endswith(".exe"):
                print("Packaged: " + name, flush=True)
        revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, env=child_environment()).stdout.strip() if (source / ".git").exists() else json.loads((source / ".turboforge-source.json").read_text())["revision"]
        marker = json.dumps({"revision": revision, "local_modifications_included": True})
        archive.writestr("TurboForge/runtime/source/.turboforge-source.json", marker)
        checksums["source/.turboforge-source.json"] = hashlib.sha256(marker.encode()).hexdigest()
        project_checksums["runtime/source/.turboforge-source.json"] = checksums["source/.turboforge-source.json"]
        bundle = json.dumps({"revision": revision, "gpu_capabilities": capabilities, "version": version.strip(), "files": checksums}, indent=2)
        archive.writestr("TurboForge/runtime/bundle.json", bundle)
        project_checksums["runtime/bundle.json"] = hashlib.sha256(bundle.encode()).hexdigest()
        archive.writestr("TurboForge/release-files.json", json.dumps({"files":project_checksums},indent=2))
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():
            raise RuntimeError("Release package CRC check failed; the existing delivery was not replaced")
    os.replace(partial, output)
    checksum = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): checksum.update(chunk)
    output.with_suffix(output.suffix+".sha256").write_text(checksum.hexdigest()+"  "+output.name+"\n",encoding="ascii")
    print("Release package: " + str(output), flush=True)
    return output


if __name__ == "__main__":
    try:
        args = [value for value in sys.argv[1:] if value != "--replace"]
        package(Path(args[0]) if args else ROOT / "dist/TurboForge-Windows-x64.zip", replace="--replace" in sys.argv)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
