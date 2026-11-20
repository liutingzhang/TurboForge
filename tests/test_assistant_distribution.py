import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

import pytest

from scripts import verify_release

from turboforge.distribution import private_path


def test_every_worker_process_filters_ai_credentials():
    root = Path(__file__).resolve().parents[1]
    missing = []
    for path in [*(root / "src/turboforge").rglob("*.py"), root / "scripts/start_web.py", root / "scripts/package_release.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        sanitized_variables = {n.targets[0].id for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "child_environment"}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess" and node.func.attr in ("run", "Popen", "check_call", "check_output")):
                continue
            env = next((k.value for k in node.keywords if k.arg == "env"), None)
            if not ((isinstance(env, ast.Call) and isinstance(env.func, ast.Name) and env.func.id == "child_environment")
                    or (isinstance(env, ast.Name) and env.id in sanitized_variables)):
                missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not missing, missing


def test_private_path_policy():
    for path in (".env", "src/.env.production", "scripts/secrets/a.json", "web/dist/key.pem", "a/.git/config", "logs/request.log", "../.env", "a\\.env.local"):
        assert private_path(path)
    assert private_path(".env.example")
    assert not private_path(".env.example", allow_template=True)
    assert not private_path("src/turboforge/assistant/knowledge/entries.json")


def test_actual_release_zip_excludes_secrets_in_every_recursive_branch(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("assistant_release_test", root / "scripts/package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = tmp_path / "project"
    source, engine = tmp_path / "upstream", tmp_path / "engine"
    for path, text in {
        fake / "web/dist/index.html": "<html>page</html>",
        fake / "src/turboforge/assistant/knowledge/entries.json": "[]",
        fake / ".env.example": "TURBOFORGE_AI_API_KEY=\n",
        fake / ".env": "CANARY-DO-NOT-SHIP",
        fake / "src/.env.production": "CANARY-DO-NOT-SHIP",
        fake / "scripts/.env": "CANARY-DO-NOT-SHIP",
        fake / "web/dist/.env": "CANARY-DO-NOT-SHIP",
        fake / "web/dist/secrets/config.json": "CANARY-DO-NOT-SHIP",
        source / "tools/convert/__main__.py": "# converter",
        source / ".env": "CANARY-DO-NOT-SHIP",
        source / ".turboforge-source.json": '{"revision":"test"}',
        engine / "ninfer.exe": "fake executable fixture",
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", fake)
    from scripts.frontend_receipt import write_receipt
    write_receipt(fake/"web")
    python_archive = tmp_path / "python-runtime.tar.gz"
    python_archive.write_bytes(b"fixture")
    monkeypatch.setattr(module, "prepare_python_runtime", lambda:python_archive)
    monkeypatch.setattr(module, "local_tools", lambda: (source, engine))
    monkeypatch.setattr(module, "collect", lambda _: {})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="NInfer (fixture, 89)", returncode=0))
    output = module.package(tmp_path / "release.zip")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "TurboForge/.env.example" not in names
        assert "TurboForge/runtime/web/index.html" in names
        assert not any("/dist/" in name for name in names)
        assert "TurboForge/src/turboforge/assistant/knowledge/entries.json" in names
        assert not any("CANARY-DO-NOT-SHIP" in archive.read(name).decode("utf-8", errors="ignore") for name in names)


def _json_bytes(value):
    return json.dumps(value).encode("utf-8")


@pytest.fixture
def release_files(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(verify_release, "ROOT", workspace)
    files = {name: b"fixture\n" for name in verify_release.REQUIRED}
    files.pop("release-files.json")
    files.pop("runtime/bundle.json")
    files.update({
        "src/turboforge/__main__.py": b"from .cli import main\nraise SystemExit(main())\n",
        "runtime/python/python-fixture.zip": b"fixture Python archive",
        "runtime/bin/ninfer.exe": b"fixture executable",
        "runtime/bin/ninfer-serve.exe": b"fixture server",
        "runtime/bin/ninfer_bench.exe": b"fixture benchmark",
        "runtime/bin/backends/13/ninfer.dll": b"fixture CUDA backend",
        "runtime/source/tools/convert/__main__.py": b"# fixture converter\n",
        "runtime/source/.turboforge-source.json": _json_bytes({"revision": "fixture"}),
        "licenses/web/sample-package/LICENSE": b"fixture license",
    })
    files["src/turboforge/data/python-runtime.json"] = _json_bytes({
        "filename": "python-fixture.zip", "version": "3.13.18",
        "sha256": hashlib.sha256(files["runtime/python/python-fixture.zip"]).hexdigest(),
    })
    files["runtime/web/build-receipt.json"] = _json_bytes({"files": {
        "index.html": hashlib.sha256(files["runtime/web/index.html"]).hexdigest(),
    }})
    files["runtime/bundle.json"] = _json_bytes({"files": {
        name.removeprefix("runtime/"): hashlib.sha256(raw).hexdigest()
        for name, raw in files.items() if name.startswith("runtime/")
    }})
    _refresh_release_manifest(files)
    return files


def _refresh_release_manifest(files):
    files["release-files.json"] = _json_bytes({"files": {
        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items() if name != "release-files.json"
    }})


def _write_sums(paths, destination=None):
    destination = destination or paths[0].parent / "SHA256SUMS.txt"
    destination.write_bytes(("\ufeff" + "".join(
        hashlib.sha256(path.read_bytes()).hexdigest().upper() + " *" + path.name + "\r\n" for path in paths
    )).encode("utf-8"))
    return destination


def _write_release(tmp_path, files, *, split=True):
    paths = [tmp_path / "TurboForge-Windows-x64-chinese-part1.zip",
             tmp_path / "TurboForge-Windows-x64-chinese-part2.zip"] if split else [tmp_path / "release.zip"]
    for index, path in enumerate(paths):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            for name, raw in files.items():
                part = int(name.startswith("runtime/") and name != "runtime/bundle.json") if split else 0
                if part == index:
                    archive.writestr("TurboForge/" + name, raw)
    if split:
        _write_sums(paths)
    else:
        paths[0].with_suffix(".zip.sha256").write_text(
            hashlib.sha256(paths[0].read_bytes()).hexdigest() + "  release.zip\n", encoding="ascii"
        )
    return paths


def test_independent_release_parts_verify_as_one_complete_delivery(tmp_path, release_files):
    paths = _write_release(tmp_path, release_files)
    result = verify_release.verify(paths)
    assert result["crc_and_file_hashes"] == "passed"
    assert result["checksums"] == "passed"
    assert result["python"] == "3.13.18"
    assert result["entries"] == len(release_files)
    assert result["bytes"] == sum(path.stat().st_size for path in paths)
    assert [item["release"] for item in result["releases"]] == [path.name for path in paths]
    assert verify_release.verify(list(reversed(paths)))["entries"] == len(release_files)


def test_single_release_keeps_legacy_sidecar_and_result(tmp_path, release_files):
    path = _write_release(tmp_path, release_files, split=False)[0]
    result = verify_release.verify(path)
    assert result["release"] == path.name
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["entries"] == len(release_files)
    path.with_suffix(".zip.sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(AssertionError, match="ZIP sidecar mismatch"):
        verify_release.verify(path)


def test_release_cli_accepts_all_parts_and_explicit_sums(tmp_path, release_files, capsys):
    paths = _write_release(tmp_path, release_files)
    sums = tmp_path / "checksums.txt"
    (tmp_path / "SHA256SUMS.txt").replace(sums)
    assert verify_release.main([*(str(path) for path in paths), "--checksums", str(sums)]) == 0
    result = json.loads(capsys.readouterr().out)
    report = verify_release.ROOT / "reports/RELEASE_VERIFICATION.json"
    assert json.loads(report.read_text(encoding="utf-8")) == result
    assert result["checksums"] == "passed"


def test_release_cli_reports_missing_part_without_success_report(tmp_path, release_files, capsys):
    paths = _write_release(tmp_path, release_files)
    assert verify_release.main([str(paths[0]), "--checksums", str(tmp_path / "SHA256SUMS.txt")]) == 1
    assert "ZIP set" in capsys.readouterr().err
    assert not (verify_release.ROOT / "reports/RELEASE_VERIFICATION.json").exists()


@pytest.mark.parametrize("problem", ["wrong", "missing", "duplicate", "malformed", "unsafe"])
def test_part_checksums_reject_invalid_sha256sums(tmp_path, release_files, problem):
    paths = _write_release(tmp_path, release_files)
    sums = tmp_path / "SHA256SUMS.txt"
    lines = sums.read_text(encoding="utf-8-sig").splitlines()
    if problem == "wrong":
        lines[1] = "0" * 64 + "  " + paths[1].name
    elif problem == "missing":
        lines.pop()
    elif problem == "duplicate":
        lines.append(lines[1])
    elif problem == "malformed":
        lines[1] = "not-a-sha256  " + paths[1].name
    else:
        lines[1] = "0" * 64 + "  ../" + paths[1].name
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA256SUMS"):
        verify_release.verify(paths)


def test_sha256sums_supports_utf8_filenames_with_spaces(tmp_path, release_files):
    paths = _write_release(tmp_path, release_files)
    paths[1] = paths[1].replace(tmp_path / "TurboForge 中文 part2.zip")
    _write_sums(paths)
    assert verify_release.verify(paths)["checksums"] == "passed"


@pytest.mark.parametrize("missing", ["runtime/bin/backends/13/ninfer.dll", "web/src/App.tsx"])
def test_joint_release_rejects_missing_content(tmp_path, release_files, missing):
    del release_files[missing]
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match="Missing deliverables|Incomplete release digest list"):
        verify_release.verify(paths)


def test_joint_release_rejects_corrupt_digest_in_second_part(tmp_path, release_files):
    release_files["runtime/bin/backends/13/ninfer.dll"] = b"changed backend"
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match="Digest mismatch: runtime/bin/backends/13/ninfer.dll"):
        verify_release.verify(paths)


def test_joint_release_checks_crc_in_second_part(tmp_path, release_files):
    paths = _write_release(tmp_path, release_files)
    with zipfile.ZipFile(paths[1]) as archive:
        info = archive.getinfo("TurboForge/runtime/bin/backends/13/ninfer.dll")
    raw = bytearray(paths[1].read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", raw, info.header_offset + 26)
    offset = info.header_offset + 30 + name_length + extra_length
    raw[offset] ^= 1
    paths[1].write_bytes(raw)
    _write_sums(paths)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        verify_release.verify(paths)


@pytest.mark.parametrize("name", ["runtime/source/.env", "runtime/bin/secrets/token.json", "web/dist/key.pem"])
def test_joint_release_rejects_private_files_before_reading_them(tmp_path, release_files, monkeypatch, name):
    release_files[name] = b"fixture private content must not be read"
    _refresh_release_manifest(release_files)
    paths = _write_release(tmp_path, release_files)
    original = zipfile.ZipFile.open

    def guarded_open(archive, member, *args, **kwargs):
        filename = member.filename if isinstance(member, zipfile.ZipInfo) else member
        assert filename != "TurboForge/" + name, "Private member was read"
        return original(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    with pytest.raises(AssertionError, match="Private files included"):
        verify_release.verify(paths)


@pytest.mark.parametrize("name", ["runtime/source/tests/test_convert.py", "scripts/package_release.py"])
def test_joint_release_rejects_development_files(tmp_path, release_files, name):
    release_files[name] = b"# development fixture\n"
    _refresh_release_manifest(release_files)
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match="Development files included"):
        verify_release.verify(paths)


@pytest.mark.parametrize("content,message", [
    (b"# z:/release-test-private/personal-checkout\n", "Personal development paths"),
    (_json_bytes({"path": "C:\\Users\\release-test-private\\personal-checkout"}), "Personal development paths"),
    (b"# sk-proj-" + b"A" * 40 + b"\n", "Secret content"),
    (b"-----BEGIN PRIVATE KEY-----\nfixture\n", "Secret content"),
])
def test_joint_release_scans_text_in_second_part(tmp_path, release_files, monkeypatch, content, message):
    monkeypatch.setattr(verify_release, "PERSONAL_PATHS", ("Z:/release-test-private", "C:/Users/release-test-private"))
    release_files["runtime/source/fixture.py"] = content
    _refresh_release_manifest(release_files)
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match=message) as caught:
        verify_release.verify(paths)
    assert "A" * 40 not in str(caught.value)


def test_joint_release_text_scan_catches_chunk_boundary_tokens(tmp_path, release_files):
    release_files["runtime/source/fixture.py"] = b"#" * (1024 * 1024 - 5) + b"sk-proj-" + b"A" * 40
    _refresh_release_manifest(release_files)
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match="Secret content"):
        verify_release.verify(paths)


@pytest.mark.parametrize("name", ["TurboForge/../escape.txt", "Elsewhere/file.txt", "TurboForge/runtime\\file.txt"])
def test_joint_release_rejects_unsafe_members_in_second_part(tmp_path, release_files, name):
    paths = _write_release(tmp_path, release_files)
    with zipfile.ZipFile(paths[1], "a") as archive:
        archive.writestr(name, b"fixture")
    if "\\" in name:
        # ZipInfo normalizes OS separators on Windows; emulate a foreign ZIP
        # by changing the same-length local and central header names directly.
        paths[1].write_bytes(paths[1].read_bytes().replace(name.replace("\\", "/").encode(), name.encode()))
    _write_sums(paths)
    with pytest.raises(AssertionError, match="Unsafe ZIP path"):
        verify_release.verify(paths)


def test_joint_release_rejects_overlapping_files(tmp_path, release_files):
    paths = _write_release(tmp_path, release_files)
    with zipfile.ZipFile(paths[1], "a") as archive:
        archive.writestr("TurboForge/README.md", release_files["README.md"])
    _write_sums(paths)
    with pytest.raises(AssertionError, match="duplicate entries"):
        verify_release.verify(paths)


@pytest.mark.parametrize("receipt,message", [("runtime/bundle.json", "Runtime receipt mismatch"),
                                              ("runtime/web/build-receipt.json", "Frontend receipt mismatch"),
                                              ("src/turboforge/data/python-runtime.json", "Python runtime mismatch")])
def test_joint_release_retains_internal_receipt_checks(tmp_path, release_files, receipt, message):
    data = json.loads(release_files[receipt])
    if "files" in data:
        data["files"][next(iter(data["files"]))] = "0" * 64
    else:
        data["sha256"] = "0" * 64
    release_files[receipt] = _json_bytes(data)
    if receipt.startswith("runtime/") and receipt != "runtime/bundle.json":
        # The runtime receipt covers runtime/web, so keep it consistent to reach
        # the more specific frontend check.
        bundle = json.loads(release_files["runtime/bundle.json"])
        bundle["files"][receipt.removeprefix("runtime/")] = hashlib.sha256(release_files[receipt]).hexdigest()
        release_files["runtime/bundle.json"] = _json_bytes(bundle)
    _refresh_release_manifest(release_files)
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match=message):
        verify_release.verify(paths)


def test_joint_release_retains_workspace_consistency_checks(tmp_path, release_files):
    (verify_release.ROOT / "README.md").write_bytes(b"different workspace content")
    paths = _write_release(tmp_path, release_files)
    with pytest.raises(AssertionError, match="Workspace changed: README.md"):
        verify_release.verify(paths)


def _release_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("assistant_release_parts_test", root / "scripts/package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_path_policy():
    from turboforge.distribution import release_path
    for name in (".env", ".env.example", "a/.github/workflows/ci.yml", "web/dist/index.html",
                 "runtime/web/dist/x.js", "web/node_modules/pkg/index.js", "tests/test_x.py",
                 "runtime/source/tests/x.py", "models/a.gguf", "a/b.safetensors", "x.pyc",
                 "runtime/source/.turboforge-source.json", "jobs/state.json", "a/logs/x.txt"):
        assert not release_path(name), name
    for name in ("runtime/web/index.html", "runtime/web/build-receipt.json",
                 "licenses/web/sample-package/LICENSE", "runtime/bin/ninfer.exe",
                 "src/turboforge/api/server.py", "INSTALL_ZH.md", "Start TurboForge.bat"):
        assert release_path(name), name


def test_write_parts_splits_under_limit_with_complete_union(tmp_path):
    import os
    module = _release_module()
    files = {f"runtime/bin/file{i}.bin": os.urandom(40_000) for i in range(5)}
    files["README.md"] = b"readme"
    paths = module._write_parts(files, tmp_path / "zh", 100_000, language="zh")
    assert len(paths) >= 2
    assert all(path.stat().st_size <= 100_000 for path in paths)
    assert [path.name for path in paths] == [f"TurboForge-Windows-x64-chinese-part{i+1}.zip" for i in range(len(paths))]
    seen = {}
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                assert name not in seen
                seen[name] = archive.read(name)
    assert seen == {"TurboForge/" + name: value for name, value in files.items()}
    sums = (tmp_path / "zh/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert sums == [hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name for path in paths]
    assert not list((tmp_path / "zh").glob("*.part"))


def test_write_parts_english_names_and_single_group(tmp_path):
    module = _release_module()
    paths = module._write_parts({"README.md": b"readme"}, tmp_path / "en", 100_000, language="en")
    assert [path.name for path in paths] == ["TurboForge-Windows-x64-english-part1.zip"]


def test_write_parts_rejects_oversized_file_and_invalid_limits(tmp_path):
    import os
    module = _release_module()
    with pytest.raises(RuntimeError, match="exceeds the part limit"):
        module._write_parts({"big.bin": os.urandom(200_000)}, tmp_path / "out", 100_000)
    for limit in (65535, 2 * 1024**3):
        with pytest.raises(ValueError):
            module._write_parts({"a.txt": b"a"}, tmp_path / "out", limit)


def test_write_parts_refuses_to_overwrite_existing_delivery(tmp_path):
    module = _release_module()
    output = tmp_path / "zh"
    output.mkdir()
    (output / "TurboForge-Windows-x64-chinese-part1.zip").write_bytes(b"old")
    with pytest.raises(RuntimeError, match="already exists"):
        module._write_parts({"README.md": b"readme"}, output, 100_000)
    assert (output / "TurboForge-Windows-x64-chinese-part1.zip").read_bytes() == b"old"
    (output / "TurboForge-Windows-x64-chinese-part1.zip").unlink()
    (output / "TurboForge-Windows-x64-chinese-part1.zip.part").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="already exists"):
        module._write_parts({"README.md": b"readme"}, output, 100_000)
