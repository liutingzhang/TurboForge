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
