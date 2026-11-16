import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zipfile

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
        assert "TurboForge/.env.example" in names
        assert "TurboForge/src/turboforge/assistant/knowledge/entries.json" in names
        assert not any("CANARY-DO-NOT-SHIP" in archive.read(name).decode("utf-8", errors="ignore") for name in names)
