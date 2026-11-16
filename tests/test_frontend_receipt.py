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
