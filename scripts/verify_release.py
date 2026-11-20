"""Check a whole ZIP or merged independent part ZIPs, without extraction."""
from __future__ import annotations

import argparse
import codecs
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from turboforge.distribution import private_path, release_path

REQUIRED = {"README.md", "README_EN.md", "LICENSE", "NOTICE",
            "requirements-web.txt", "pyproject.toml", "启动 TurboForge.bat", "Start TurboForge.bat",
            "INSTALL_ZH.md", "INSTALL_EN.md", "START_HERE.txt",
            "scripts/start_web.py", "scripts/start_web.ps1",
            "src/turboforge/data/python-runtime.json", "src/turboforge/data/ensure_python.ps1",
            "web/package-lock.json", "web/src/App.tsx", "runtime/web/index.html", "runtime/web/build-receipt.json",
            "runtime/bundle.json", "release-files.json"}
FORBIDDEN_PREFIXES = ("tests/", "docs/", "configs/", ".github/", "runtime/source/docs/", "runtime/source/tests/",
                      "runtime/source/examples/", "runtime/source/CONTRIBUTING.md", "runtime/source/SECURITY.md",
                      "runtime/source/CHANGELOG.md")
FORBIDDEN_NAMES = {"CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "PROJECT_SUMMARY.md", ".gitignore", "web/README.md",
                   "scripts/build_assistant_knowledge.py", "scripts/build_assistant_browser_fixture.mjs",
                   "scripts/check_clean_start.py", "scripts/package_release.py", "scripts/verify_release.py"}
EXCEPTIONS = {"runtime/source/.turboforge-source.json"}
TEXT_SUFFIXES = {".py", ".ps1", ".bat", ".md", ".txt", ".json", ".toml", ".tsx", ".ts", ".yml", ".yaml",
                 ".js", ".mjs", ".html", ".css", ".cu", ".cuh", ".c", ".cpp", ".h", ".hpp", ".cmake"}
PERSONAL_PATHS = (ROOT.as_posix(), Path.home().as_posix())
SECRET_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}"
)


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _verify_checksums(paths: list[Path], checksums: Path | None) -> list[str]:
    if checksums is None and len(paths) == 1:
        expected = paths[0].with_suffix(paths[0].suffix + ".sha256").read_text(encoding="ascii").split()[0]
        actual = _sha256(paths[0])
        assert expected == actual, "ZIP sidecar mismatch"
        return [actual]
    if checksums is None:
        assert len({path.resolve().parent for path in paths}) == 1, "Parts need a shared SHA256SUMS.txt or --checksums"
        checksums = paths[0].parent / "SHA256SUMS.txt"
    expected = {}
    for line in checksums.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?(.+)", line)
        assert match is not None, "Malformed SHA256SUMS line"
        digest, name = match.groups()
        name = name.removeprefix("./")
        assert name not in (".", "..") and "/" not in name and "\\" not in name and ":" not in name, "Unsafe SHA256SUMS filename"
        assert name not in expected, "Duplicate SHA256SUMS filename: " + name
        expected[name] = digest.lower()
    assert {name for name in expected if name.lower().endswith(".zip")} == {path.name for path in paths}, "SHA256SUMS ZIP set does not match supplied parts"
    actual = []
    for path in paths:
        digest = _sha256(path)
        assert expected.get(path.name) == digest, "SHA256SUMS mismatch: " + path.name
        actual.append(digest)
    return actual


def _check_text(name: str, text: str) -> None:
    # JSON and generated JS may escape the backslashes of a Windows path.
    unescaped = text.replace("\\\\", "\\").replace("\\", "/").casefold()
    assert not any(marker.casefold() in unescaped for marker in PERSONAL_PATHS), "Personal development paths included: " + name
    assert SECRET_PATTERN.search(text) is None, "Secret content included: " + name


def _member_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo, name: str) -> str:
    checksum = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("ignore") if Path(name).suffix.lower() in TEXT_SUFFIXES else None
    overlap = ""
    with archive.open(info) as stream:
        # Reading to EOF also checks ZIP CRC, including zero-length members.
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
            if decoder is not None:
                text = overlap + decoder.decode(chunk)
                _check_text(name, text)
                overlap = text[-512:]
        if decoder is not None:
            _check_text(name, overlap + decoder.decode(b"", final=True))
    return checksum.hexdigest()


def verify(path: Path | Sequence[Path], *, checksums: Path | None = None) -> dict:
    """Verify independent ZIPs jointly against their checksums and release receipts."""
    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(value) for value in path]
    assert paths, "No release ZIPs supplied"
    assert all(value.stat().st_size < 2 * 1024**3 for value in paths), "GitHub release assets must be smaller than 2 GiB"
    assert len({value.resolve() for value in paths}) == len(paths), "Duplicate release ZIP"
    assert len({value.name for value in paths}) == len(paths), "Duplicate release ZIP filename"
    digests = _verify_checksums(paths, Path(checksums) if checksums is not None else None)
    with ExitStack() as stack:
        members = {}
        entry_count = 0
        for value in paths:
            archive = stack.enter_context(zipfile.ZipFile(value))
            names = archive.namelist()
            entry_count += len(names)
            assert len(names) == len(set(names)), "ZIP contains duplicate entries"
            for info in archive.infolist():
                # ZipInfo.filename normalizes Windows separators and truncates NULs;
                # validate the original central-directory name before that happens.
                name = info.orig_filename
                assert (name == info.filename and name.startswith("TurboForge/")
                        and "\\" not in name and ":" not in name and "\x00" not in name
                        and all(part not in (".", "..", "") for part in name.rstrip("/").split("/"))), "Unsafe ZIP path"
                relative = name[len("TurboForge/"):]
                if info.is_dir():
                    assert not (relative in FORBIDDEN_NAMES or relative.startswith(FORBIDDEN_PREFIXES)), "Development files included: " + relative
                    assert not private_path(relative) or relative in EXCEPTIONS, "Private files included"
                    assert release_path(relative) or relative in EXCEPTIONS, "Non-delivery directory included: " + relative
                    continue
                assert relative not in members, "ZIP parts contain duplicate entries: " + relative
                members[relative] = (archive, info)
        relative = set(members)
        assert REQUIRED <= relative, "Missing deliverables: " + str(sorted(REQUIRED - relative))
        leaked = sorted(name for name in relative if name in FORBIDDEN_NAMES or name.startswith(FORBIDDEN_PREFIXES))
        assert not leaked, "Development files included: " + str(leaked)
        assert all(not private_path(name) or name in EXCEPTIONS for name in relative), "Private files included"
        assert all(release_path(name) or name in EXCEPTIONS for name in relative), "Non-delivery files included"

        def read_json(name: str) -> dict:
            archive, info = members[name]
            raw = archive.read(info)
            _check_text(name, raw.decode("utf-8"))
            return json.loads(raw)

        manifest = read_json("release-files.json")["files"]
        assert set(manifest) == relative - {"release-files.json"}, "Incomplete release digest list"
        for name, expected in manifest.items():
            archive, info = members[name]
            assert _member_digest(archive, info, name) == expected, "Digest mismatch: " + name
            local = ROOT / name
            if not name.startswith(("runtime/", "licenses/")) and local.is_file():
                assert _sha256(local) == expected, "Workspace changed: " + name
        bundle = read_json("runtime/bundle.json")
        for name, expected in bundle["files"].items():
            assert manifest.get("runtime/" + name) == expected, "Runtime receipt mismatch"
        receipt = read_json("runtime/web/build-receipt.json")
        for name, expected in receipt["files"].items():
            assert manifest.get("runtime/web/" + name) == expected, "Frontend receipt mismatch"
        spec = read_json("src/turboforge/data/python-runtime.json")
        assert manifest.get("runtime/python/" + spec["filename"]) == spec["sha256"], "Python runtime mismatch"
    result = {"entries": entry_count, "crc_and_file_hashes": "passed", "private_paths": "excluded",
              "workspace": "matches", "frontend": "complete", "python": spec["version"], "runtime": "complete"}
    if len(paths) == 1:
        result.update(release=paths[0].name, bytes=paths[0].stat().st_size, sha256=digests[0])
    else:
        result.update(releases=[{"release": value.name, "bytes": value.stat().st_size, "sha256": digest}
                                for value, digest in zip(paths, digests)], bytes=sum(value.stat().st_size for value in paths),
                      checksums="passed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", type=Path, nargs="+", help="whole ZIP or all independent part ZIPs")
    parser.add_argument("--checksums", "--sha256sums", type=Path, help="SHA256SUMS.txt (default: beside part ZIPs)")
    args = parser.parse_args(argv)
    try:
        result = verify(args.zips, checksums=args.checksums)
    except (AssertionError, OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as exc:
        print("Release verification failed: " + str(exc), file=sys.stderr)
        return 1
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/RELEASE_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
