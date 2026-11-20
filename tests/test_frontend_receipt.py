import os
import pytest

from scripts.frontend_receipt import write_receipt
from turboforge.frontend import frontend_current


def test_archive_timestamps_do_not_require_node_but_source_changes_do(tmp_path):
    (tmp_path/'src').mkdir(); source=tmp_path/'src/App.tsx'; source.write_text('source')
    (tmp_path/'dist').mkdir(); asset=tmp_path/'dist/index.html'; asset.write_text('<html>built</html>')
    write_receipt(tmp_path)
    os.utime(source,(2000000000,2000000000))
    assert frontend_current(tmp_path)
    source.write_text('modified')
    assert not frontend_current(tmp_path)


def test_clean_release_serves_runtime_web_without_dist_or_node(tmp_path, monkeypatch):
    from turboforge.frontend import built_directory
    from turboforge.api import server
    web = tmp_path / 'web'
    (web / 'src').mkdir(parents=True)
    (web / 'src/App.tsx').write_text('source')
    (web / 'dist').mkdir()
    (web / 'dist/index.html').write_text('<html>clean release</html>')
    write_receipt(web)
    (tmp_path / 'runtime').mkdir()
    (web / 'dist').rename(tmp_path / 'runtime/web')
    assert frontend_current(web)
    assert built_directory(web) == tmp_path / 'runtime/web'
    monkeypatch.delenv('TURBOFORGE_WEB_DIST', raising=False)
    monkeypatch.setattr(server, '__file__', str(tmp_path / 'src/turboforge/api/server.py'))
    assert server._dist_dir() == tmp_path / 'runtime/web'
    client = server.create_app(assistant_root=tmp_path).test_client()
    assert b'clean release' in client.get('/').data
    (web / 'src/App.tsx').write_text('modified')
    assert not frontend_current(web)


def test_missing_or_modified_built_asset_requires_rebuild(tmp_path):
    (tmp_path/'dist').mkdir(); asset=tmp_path/'dist/index.html'; asset.write_text('built')
    write_receipt(tmp_path)
    asset.write_text('modified')
    assert not frontend_current(tmp_path)


@pytest.mark.parametrize("text", ['{broken', '[]', '{"files":[]}'])
def test_corrupt_build_receipt_triggers_rebuild_instead_of_crashing(tmp_path,text):
    (tmp_path/'dist').mkdir(); (tmp_path/'dist/index.html').write_text('built')
    (tmp_path/'dist/build-receipt.json').write_text(text)
    assert not frontend_current(tmp_path)
