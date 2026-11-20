import json
from pathlib import Path
import subprocess
import os

import pytest

from turboforge.storage import inventory, remove_failed
from turboforge.assistant.config import child_environment


def build(root: Path, name: str, status: str) -> Path:
    folder = root/'builds'/name
    folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps({'status':status}))
    (folder/'model.bin').write_bytes(b'failed-or-successful-model')
    return folder


def test_cleanup_removes_only_selected_failed_managed_outputs(tmp_path):
    failed = build(tmp_path,'failed-test','failed')
    kept = build(tmp_path,'successful','validated')
    original = tmp_path/'models'/'source.bin'; original.parent.mkdir(); original.write_bytes(b'original')
    assert len(inventory(tmp_path)['items']) == 2
    with pytest.raises(ValueError): remove_failed(tmp_path,['successful'])
    with pytest.raises(ValueError): remove_failed(tmp_path,['../models'])
    remove_failed(tmp_path,['failed-test'])
    assert not failed.exists() and kept.exists() and original.read_bytes() == b'original'


def test_recovered_interrupted_manifest_becomes_removable(tmp_path, monkeypatch):
    from turboforge.api.builds import BuildManager
    from turboforge.manifest import read_json_atomic, write_json_atomic
    monkeypatch.setenv("TURBOFORGE_HOME", str(tmp_path))
    folder = build(tmp_path, "interrupted-output", "converting")
    job = "a" * 32
    state = tmp_path / "jobs" / job / "state.json"
    write_json_atomic(state, {"build_id": job, "finished": False, "status": "converting",
        "output_dir": str(folder), "error": None, "setup_log": []})
    manager = BuildManager()
    result = manager.storage_inventory()
    assert result["items"][0]["status"] == "interrupted"
    assert result["items"][0]["removable"]
    assert read_json_atomic(state)["finished"]
    history = read_json_atomic(folder / "manifest.json")["history"]
    assert history[-1]["status"] == "interrupted"
    assert BuildManager().status(job)["status"] == "interrupted"
    manager.clean_failed_outputs(["interrupted-output"])
    assert not folder.exists()


@pytest.mark.parametrize("crash", [False, True])
def test_worker_exit_settles_nonterminal_manifest(tmp_path, monkeypatch, crash):
    from types import SimpleNamespace
    from turboforge.api import builds
    from turboforge.domain import ForgeConfig
    from turboforge.manifest import read_json_atomic
    monkeypatch.setenv("TURBOFORGE_HOME", str(tmp_path))
    folder = build(tmp_path, "worker-output", "converting")
    pending = []
    monkeypatch.setattr(builds, "threading", SimpleNamespace(
        Lock=builds.threading.Lock,
        Thread=lambda *, target, daemon: SimpleNamespace(start=lambda: pending.append(target))))
    def run(_config):
        if crash:
            raise RuntimeError("fixture worker crash")
        return SimpleNamespace(data={"status": "converting"})
    monkeypatch.setattr(builds, "BuildRunner", lambda **kwargs: SimpleNamespace(build=run))
    manager = builds.BuildManager()
    job = manager.start(ForgeConfig(str(tmp_path / "model"), str(folder)))
    pending[0]()
    state = read_json_atomic(tmp_path / "jobs" / job / "state.json")
    assert state["finished"] and state["status"] == ("crashed" if crash else "failed")
    assert read_json_atomic(folder / "manifest.json")["status"] == "failed"
    assert manager.storage_inventory()["items"][0]["removable"]


def test_settlement_never_changes_successful_manifest(tmp_path):
    from turboforge.api.builds import BuildManager
    folder = build(tmp_path, "success", "validated")
    manifest = folder / "manifest.json"
    before = manifest.read_bytes()
    assert not BuildManager()._settle_manifest(str(folder), "interrupted", "stopped")
    assert manifest.read_bytes() == before


@pytest.mark.skipif(os.name != 'nt', reason='Windows junction boundary')
def test_junction_is_never_followed_for_cleanup(tmp_path):
    outside = build(tmp_path/'outside','source','failed')
    managed = tmp_path/'managed'; (managed/'builds').mkdir(parents=True)
    junction = managed/'builds'/'alias'
    subprocess.run(['cmd','/c','mklink','/J',str(junction),str(outside)],check=True,capture_output=True,env=child_environment())
    try:
        assert inventory(managed)['items'] == []
        with pytest.raises(ValueError): remove_failed(managed,['alias'])
        assert (outside/'model.bin').exists()
    finally:
        # Removing only the junction itself, never its target.
        os.rmdir(junction)
