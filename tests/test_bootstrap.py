from __future__ import annotations

import hashlib
import http.server
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile
import urllib.error
from email.message import Message

from turboforge.bootstrap import download, extract_zip, ensure_source, REVISION
from turboforge.model_sources import ModelResolver, repo_id
from turboforge.manifest import read_json_atomic, write_json_atomic
from turboforge.domain import InputKind


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b'checked model bytes' * 1000
        test = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                test.requests += 1
                offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].rstrip('-'))
                test.last_offset = offset
                self.send_response(206 if offset else 200)
                if offset:
                    self.send_header('Content-Range', f'bytes {offset}-{len(test.payload)-1}/{len(test.payload)}')
                self.send_header('Content-Length', str(len(test.payload)-offset))
                self.end_headers()
                self.wfile.write(test.payload[offset:])
            def log_message(self, *args):
                pass
        self.requests = self.last_offset = 0
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/model'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.temp.cleanup()

    def test_partial_download_resumes_and_verified_cache_avoids_network(self):
        destination = self.root / 'weights.bin'
        destination.with_suffix('.bin.part').write_bytes(self.payload[:100])
        checksum = hashlib.sha256(self.payload).hexdigest()
        download(self.url, destination, sha256=checksum, size=len(self.payload), emit=lambda _: None)
        self.assertEqual(self.last_offset, 100)
        self.assertEqual(destination.read_bytes(), self.payload)
        download(self.url, destination, sha256=checksum, emit=lambda _: None)
        self.assertEqual(self.requests, 1)

    def test_bad_checksum_is_not_promoted_to_completed_file(self):
        destination = self.root / 'bad.bin'
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            download(self.url, destination, sha256='0' * 64, emit=lambda _: None)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix('.bin.part').exists())

    def test_archive_cannot_escape_install_directory(self):
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as out:
            out.writestr('../escape.py', 'bad')
        with self.assertRaises(ValueError):
            extract_zip(archive, self.root / 'install')
        self.assertFalse((self.root / 'escape.py').exists())

    def test_official_archive_install_is_pinned_and_reused(self):
        archive = self.root / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as out:
            out.writestr(f'ninfer-all-{REVISION}/tools/convert/__main__.py', '# converter')
        with patch('turboforge.bootstrap.download', return_value=archive) as fetch:
            source = ensure_source(self.root, lambda _: None)
            self.assertEqual(ensure_source(self.root, lambda _: None), source)
            self.assertEqual(fetch.call_count, 1)
        self.assertEqual(json.loads((source / '.turboforge-source.json').read_text())['revision'], REVISION)

    def test_worker_can_replace_manifest_while_parent_polls(self):
        path = self.root / 'manifest.json'
        write_json_atomic(path, {'sequence': 0})
        script = "from pathlib import Path;from turboforge.manifest import write_json_atomic;p=Path(__import__('sys').argv[1]);[write_json_atomic(p,{'sequence':i}) for i in range(1,100)]"
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        process = subprocess.Popen([sys.executable, '-c', script, str(path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            self.assertIn('sequence', read_json_atomic(path))
        stdout, stderr = process.communicate()
        self.assertEqual(process.returncode, 0, stderr.decode(errors='replace'))
        self.assertEqual(read_json_atomic(path)['sequence'], 99)

    def test_completed_partial_can_resume_after_server_returns_416(self):
        destination=self.root/"finished.bin"
        destination.with_name(destination.name+".part").write_bytes(self.payload)
        headers=Message();headers["Content-Range"]=f"bytes */{len(self.payload)}"
        error=urllib.error.HTTPError(self.url,416,"complete",headers,None)
        with patch("turboforge.bootstrap.urllib.request.urlopen",side_effect=error):
            download(self.url,destination,sha256=hashlib.sha256(self.payload).hexdigest(),emit=lambda _:None)
        self.assertEqual(destination.read_bytes(),self.payload)


class ModelSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.resolver = ModelResolver(self.root, lambda _: None)

    def tearDown(self):
        self.temp.cleanup()

    def test_local_paths_are_not_mistaken_for_repository_names(self):
        self.assertIsNone(repo_id('D:/models/name'))
        self.assertIsNone(repo_id('../private'))
        self.assertEqual(repo_id('Qwen/Qwen3.5-0.8B'), 'Qwen/Qwen3.5-0.8B')
        self.assertEqual(repo_id('https://huggingface.co/Qwen/Test/tree/main'), 'Qwen/Test')

    def test_explicit_gguf_link_keeps_requested_file(self):
        info = {'sha': 'a' * 40, 'siblings': [
            {'rfilename': 'small-Q2.gguf', 'size': 10},
            {'rfilename': 'wanted-Q4.gguf', 'size': 20},
        ]}
        def fetch(url, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'header')
            return destination
        with patch.object(self.resolver, 'info', return_value=info), patch('turboforge.model_sources.download', side_effect=fetch) as fetcher:
            folder = self.resolver.snapshot('Org/Model', filename='wanted-Q4.gguf')
        self.assertTrue((folder / 'wanted-Q4.gguf').is_file())
        self.assertFalse((folder / 'small-Q2.gguf').exists())
        self.assertEqual(fetcher.call_count, 1)

    def test_unsupported_hf_architecture_stops_before_large_weight_download(self):
        info = {'sha': 'a' * 40, 'siblings': [
            {'rfilename': 'config.json', 'size': 30},
            {'rfilename': 'model.safetensors', 'size': 100000000000},
        ]}
        def fetch(url, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{"architectures":["NotSupported"]}')
            return destination
        with patch.object(self.resolver, 'info', return_value=info), patch('turboforge.model_sources.download', side_effect=fetch) as fetcher:
            with self.assertRaisesRegex(ValueError, '架构'):
                self.resolver.snapshot('Org/Model')
        self.assertEqual(fetcher.call_count, 1)

    def test_only_missing_resources_are_requested_for_gguf_base(self):
        info = {'sha': 'b' * 40, 'siblings': [
            {'rfilename': 'config.json', 'size': 10},
            {'rfilename': 'tokenizer.json', 'size': 10},
            {'rfilename': 'model.safetensors', 'size': 100000000000},
        ]}
        def fetch(url, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{}')
            return destination
        with patch.object(self.resolver, 'info', return_value=info), patch('turboforge.model_sources.download', side_effect=fetch) as fetcher:
            self.resolver.snapshot('Org/Model', resources_only=True)
        self.assertEqual(fetcher.call_count, 2)

    def test_ambiguous_origin_is_refused_instead_of_guessing(self):
        (self.root / 'config.json').write_text('{"_name_or_path":"Org/A", "base_model_name_or_path":"Org/B"}')
        with self.assertRaisesRegex(ValueError, '多个'):
            self.resolver.origin(self.root)

    def test_tokenizer_config_alone_gets_hydrated_without_changing_source(self):
        model, resources = self.root / 'model', self.root / 'resources'
        model.mkdir(); resources.mkdir()
        original = '{"_name_or_path":"Org/Original"}'
        (model / 'config.json').write_text(original)
        (model / 'tokenizer_config.json').write_text('{}')
        weights = model / 'model.safetensors'
        weights.write_bytes(b'weights')
        (resources / 'tokenizer.json').write_text('{}')
        inspection = SimpleNamespace(input_kind=InputKind.SAFETENSORS, tokenizer_files=['tokenizer_config.json'], config={})
        with patch('turboforge.inspector.ModelInspector.inspect', return_value=inspection), patch.object(self.resolver, 'snapshot', return_value=resources):
            first = self.resolver.resolve(str(model)).model
            self.assertTrue((first / 'tokenizer.json').is_file())
            (first / 'config.json').write_text('{"edited":true}')
            self.assertEqual((model / 'config.json').read_text(), original)
            weights.write_bytes(b'changed weights')
            second = self.resolver.resolve(str(model)).model
            self.assertNotEqual(first, second)
            self.assertEqual((second / weights.name).read_bytes(), b'changed weights')

    def test_missing_model_fails_before_dependency_install(self):
        from turboforge.preparation import prepare
        with patch('turboforge.preparation.cache_root', return_value=self.root), patch('turboforge.preparation.ensure_dependencies') as install:
            with self.assertRaisesRegex(ValueError, '找不到模型'):
                prepare(str(self.root / 'missing'), {}, lambda _: None)
        install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
