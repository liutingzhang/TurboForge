"""Check release contents, every digest and workspace consistency without extraction."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from turboforge.distribution import private_path


def verify(path: Path) -> dict:
    required = {"README.md", "LICENSE", "NOTICE",
                "requirements-web.txt", "pyproject.toml", ".env.example", "Start TurboForge.bat",
                "scripts/start_web.py", "scripts/start_web.ps1",
                "src/turboforge/data/python-runtime.json", "src/turboforge/data/ensure_python.ps1",
                "web/package-lock.json", "web/src/App.tsx", "web/dist/index.html", "web/dist/build-receipt.json",
                "runtime/bundle.json"}
    forbidden_prefixes = ("tests/", "docs/", "configs/", ".github/", "runtime/source/docs/", "runtime/source/tests/",
                          "runtime/source/examples/", "runtime/source/CONTRIBUTING.md", "runtime/source/SECURITY.md",
                          "runtime/source/CHANGELOG.md")
    forbidden_names = {"CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "PROJECT_SUMMARY.md", ".gitignore", "web/README.md",
                       "scripts/build_assistant_knowledge.py", "scripts/build_assistant_browser_fixture.mjs",
                       "scripts/check_clean_start.py", "scripts/package_release.py", "scripts/verify_release.py"}
    exceptions = {".env.example", "runtime/source/.turboforge-source.json"}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "ZIP contains duplicate entries"
        assert all(name.startswith("TurboForge/") and ".." not in PurePosixPath(name).parts for name in names), "Unsafe ZIP path"
        relative = {name[len("TurboForge/"):] for name in names}
        assert required <= relative, "Missing deliverables: " + str(sorted(required-relative))
        leaked = sorted(name for name in relative if name in forbidden_names or name.startswith(forbidden_prefixes))
        assert not leaked, "Development files included: " + str(leaked)
        assert all(not private_path(name) or name in exceptions for name in relative), "Private files included"
        markers = ("D:\\llm", "D:/llm", "D:\\xiangmu", "D:/xiangmu", "C:\\Users\\huazuo", "C:/Users/huazuo")
        leaked_paths = []
        for name in relative:
            if name.startswith(("src/", "scripts/", "web/")) or "/" not in name:
                if name.lower().endswith((".py", ".ps1", ".bat", ".md", ".txt", ".json", ".toml", ".tsx", ".ts", ".yml", ".yaml")):
                    text = archive.read("TurboForge/" + name).decode("utf-8", "ignore")
                    if any(marker in text for marker in markers):
                        leaked_paths.append(name)
        assert not leaked_paths, "Personal development paths included: " + str(sorted(leaked_paths))
        manifest = json.loads(archive.read("TurboForge/release-files.json"))["files"]
        assert set(manifest) == relative - {"release-files.json"}, "Incomplete release digest list"
        for name, expected in manifest.items():
            checksum = hashlib.sha256()
            with archive.open("TurboForge/"+name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b""): checksum.update(chunk)
            assert checksum.hexdigest() == expected, "Digest mismatch: " + name
            local = ROOT/name
            if not name.startswith(("runtime/", "licenses/")) and local.is_file():
                assert hashlib.sha256(local.read_bytes()).hexdigest() == expected, "Workspace changed: " + name
        bundle = json.loads(archive.read("TurboForge/runtime/bundle.json"))
        for name, expected in bundle["files"].items():
            assert manifest["runtime/"+name] == expected, "Runtime receipt mismatch"
        receipt = json.loads(archive.read("TurboForge/web/dist/build-receipt.json"))
        for name, expected in receipt["files"].items():
            assert manifest["web/dist/"+name] == expected, "Frontend receipt mismatch"
        spec = json.loads(archive.read("TurboForge/src/turboforge/data/python-runtime.json"))
        assert manifest["runtime/python/"+spec["filename"]] == spec["sha256"], "Python runtime mismatch"
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): checksum.update(chunk)
    assert path.with_suffix(path.suffix+".sha256").read_text(encoding="ascii").split()[0] == checksum.hexdigest(), "ZIP sidecar mismatch"
    return {"release":path.name,"bytes":path.stat().st_size,"sha256":checksum.hexdigest(),"entries":len(names),
            "crc_and_file_hashes":"passed","private_paths":"excluded","workspace":"matches",
            "frontend":"complete","python":spec["version"],"runtime":"complete"}


if __name__ == "__main__":
    result=verify(Path(sys.argv[1]))
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports/RELEASE_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
