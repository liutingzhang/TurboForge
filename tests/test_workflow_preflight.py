from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from turboforge.backends.compiler import UpstreamStatus
from turboforge.backends.registry import get_recipe
from turboforge.domain import ForgeConfig
from turboforge.inspector import ModelInspector
from turboforge.manifest import BuildManifest, read_json_atomic, write_json_atomic
from turboforge.planner import ForgePlanner
from turboforge.runner import BuildRunner
from turboforge import serving
from turboforge.api.builds import BuildManager
from tests.helpers import write_config, write_safetensors, write_gguf


class WorkflowPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        write_config(self.model, architectures=['Qwen3_5ForCausalLM'], hidden_size=5120)
        write_safetensors(self.model / 'model.safetensors', {'model.weight': ('BF16', [8, 8])})
        upstream = UpstreamStatus(self.root, True, 'test', True, True, True, cuda_available=False)
        self.planner = ForgePlanner(upstream=upstream)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **values):
        return ForgeConfig(input_path=str(self.model), output_dir=str(self.root / 'out'), device=values.pop('device', 'cpu'), **values)

    def test_cpu_torch_cannot_plan_cuda_conversion(self):
        plan = self.planner.plan(self.config(device='cuda'))
        self.assertFalse(plan.executable)
        self.assertIn('torch-cuda-unavailable', {d.code for d in plan.diagnostics})
        self.assertTrue(self.planner.plan(self.config()).executable)

    def test_existing_output_is_blocked_before_build(self):
        out = self.root / 'out'
        out.mkdir()
        (out / 'keep.txt').write_text('keep')
        plan = self.planner.plan(self.config())
        self.assertFalse(plan.executable)
        self.assertEqual((out / 'keep.txt').read_text(), 'keep')

    def test_config_or_filename_alone_does_not_enable_speculation(self):
        write_config(self.model, architectures=['Qwen3_5ForCausalLM'], hidden_size=5120, num_mtp_layers=1)
        (self.model / 'dflash2-notes.txt').write_text('not weights')
        plan = self.planner.plan(self.config())
        self.assertIsNone(plan.serving['spec'])
        self.assertEqual(next(s for s in plan.steps if s.name == 'quantize+compile').details['components'], None)

    def test_explicit_dflash_source_reaches_converter_and_launcher(self):
        companion = self.root / 'draft'
        companion.mkdir()
        plan = self.planner.plan(self.config(sources={'dflash2': str(companion)}))
        self.assertEqual(next(s for s in plan.steps if s.name == 'quantize+compile').details['components'], 'text,dflash2')
        self.assertEqual(plan.serving['spec'], 'dflash2')
        self.assertIn('--spec dflash2', BuildRunner(self.planner)._launcher_flags(plan))

    def test_spec_off_is_consistent_and_keeps_imported_mtp(self):
        write_safetensors(self.model / 'model.safetensors', {'mtp.fc.weight': ('BF16', [8, 8])})
        plan = self.planner.plan(self.config(runtime={'speculative': 'none'}))
        self.assertEqual(next(s for s in plan.steps if s.name == 'quantize+compile').details['components'], 'text,mtp')
        self.assertIsNone(plan.serving['spec'])
        self.assertFalse(any(flag.startswith('--spec ') for flag in BuildRunner(self.planner)._launcher_flags(plan)))

    def test_gguf_accepts_metadata_only_hf_directory(self):
        (self.model / 'model.safetensors').unlink()
        gguf = self.root / 'weights.gguf'
        write_gguf(gguf)
        plan = self.planner.plan(self.config(sources={'gguf': str(gguf)}))
        self.assertTrue(plan.executable, plan.diagnostics)
        self.assertEqual(plan.inspection.weight_precision, 'metadata_only')
        self.assertGreater(plan.resources.temporary_disk_bytes, gguf.stat().st_size)

    def test_missing_import_source_build_keeps_blocked_manifest(self):
        config = self.config(sources={'gguf': str(self.root / 'absent.gguf')})
        runner = BuildRunner(self.planner)
        with patch.object(runner.compiler, 'run', side_effect=AssertionError('blocked build must not convert')):
            manifest = runner.build(config)
        self.assertEqual(manifest.data['status'], 'blocked')
        self.assertEqual(manifest.data['additional_source_identity'], {'gguf': {'type': 'missing'}})
        self.assertIn('source-missing', {d['code'] for d in manifest.data['plan_diagnostics']})

    def test_unreadable_base_head_is_blocked_despite_valid_imported_weights(self):
        (self.model / 'model.safetensors').write_bytes(b'incomplete checkpoint')
        gguf = self.root / 'weights.gguf'
        write_gguf(gguf)
        config = self.config(sources={'gguf': str(gguf)})
        runner = BuildRunner(self.planner)
        with patch.object(runner.compiler, 'run', side_effect=AssertionError('unverified heads must not convert')):
            manifest = runner.build(config)
        self.assertEqual(manifest.data['status'], 'blocked')
        self.assertIn('speculative-inspection-unverified', {d['code'] for d in manifest.data['plan_diagnostics']})

    def test_fp8_q6_ternary_and_gguf_embeddings_do_not_get_q4_flag(self):
        for name in ('qwen3_8_27b_nvfp4', 'qwen3_6_27b', 'bonsai2_27b_ternary', 'qwen3_8_27b_gguf', 'qwen3_8_27b_imatrix'):
            with self.subTest(recipe=name):
                self.assertNotIn('--embedding-q4', serving.launcher_flags(get_recipe(name), {}, False, False))

    def test_gguf_mtp_comes_from_nextn_tensors_not_hf_base(self):
        gguf = self.root / 'weights.gguf'
        write_gguf(gguf, tensors={'blk.64.nextn.eh_proj.weight': [8, 8]})
        plan = self.planner.plan(self.config(sources={'gguf': str(gguf)}))
        self.assertEqual(plan.serving['spec'], 'mtp')
        write_gguf(gguf)
        write_safetensors(self.model / 'model.safetensors', {'mtp.fc.weight': ('BF16', [8, 8])})
        plan = self.planner.plan(self.config(sources={'gguf': str(gguf)}))
        self.assertIsNone(plan.serving['spec'], 'HF MTP cannot invent a missing GGUF MTP block')
        self.assertTrue(plan.inspection.speculative_components['mtp']['detected'])
        self.assertFalse(plan.executable, 'A detected HF head cannot be silently discarded by a GGUF import')
        self.assertIn('speculative-would-be-dropped', {d.code for d in plan.diagnostics})

    def test_pq2_is_detected_from_tensor_storage_not_filename(self):
        gguf = self.root / 'model.gguf'
        write_gguf(gguf, tensor_type=142)
        inspected = ModelInspector().inspect(gguf)
        self.assertTrue(inspected.quantization['ternary_pq2'])
        self.assertEqual(inspected.weight_precision, 'PQ2_0')
        misleading = self.root / 'Bonsai-ternary-PQ2_0.gguf'
        write_gguf(misleading)
        self.assertFalse(ModelInspector().inspect(misleading).quantization['ternary_pq2'])

    def test_automatic_calibration_exposes_full_fp32_ram_cost(self):
        inspection = ModelInspector().inspect(self.model)
        inspection.element_count = 27_000_000_000
        from turboforge.resources import ResourceEstimator
        capability = self.planner.capabilities(inspection)[0]
        with patch('turboforge.resources.memory_available', return_value=32 * 1024**3):
            estimate = ResourceEstimator().estimate(inspection, capability, self.root / 'out', calibration=True)
        self.assertGreater(estimate.peak_ram_bytes, 100 * 1024**3)
        self.assertIn('ram-insufficient', {d.code for d in estimate.diagnostics})

    def test_bad_runtime_and_benchmark_inputs_are_rejected(self):
        for override in ({'runtime': {'port': 70000}}, {'runtime': {'max_context': None}},
                         {'run_validation': 'false'}, {'benchmark': {'n_gen': []}},
                         {'benchmark': {'perplexity_text': 'a', 'perplexity_corpus': 'b'}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.config(**override).validate()

    def test_concurrent_polling_does_not_break_atomic_manifest_writes(self):
        path = self.root / 'manifest.json'
        write_json_atomic(path, {'sequence': 0})
        errors = []
        def writer():
            try:
                for i in range(1, 100):
                    write_json_atomic(path, {'sequence': i})
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=writer)
        thread.start()
        while thread.is_alive():
            self.assertIn('sequence', read_json_atomic(path))
        thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(read_json_atomic(path)['sequence'], 99)

    def test_final_worker_error_is_not_hidden_by_stale_manifest(self):
        out = self.root / 'out'
        out.mkdir()
        write_json_atomic(out / 'manifest.json', {'status': 'planned'})
        manager = BuildManager()
        manager._builds['test'] = {'finished': True, 'status': 'crashed', 'error': 'test error', 'output_dir': str(out)}
        self.assertEqual(manager.status('test')['status'], 'crashed')

    def test_gpu_smoke_checks_speculation_without_server_only_flag(self):
        engine = self.root / 'engine'
        engine.mkdir()
        (engine / 'ninfer.exe').write_bytes(b'test')
        runner = BuildRunner(self.planner, ninfer_dir=engine)
        with patch('turboforge.runner.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='ok', stderr='')) as run:
            ok, _ = runner._smoke_validate(self.root / 'model.ninfer', self.root / 'smoke.log',
                flags=['--kv-dtype bf16', '--spec mtp', '--draft-tokens 3', '--device-profile auto', '--host-cache-mib 0'])
        self.assertTrue(ok)
        command = run.call_args.args[0]
        self.assertIn('--spec', command)
        self.assertIn('mtp', command)
        self.assertNotIn('--device-profile', command)
        self.assertNotIn('--host-cache-mib', command)

    def test_generated_launcher_supports_chinese_paths(self):
        config = self.config()
        plan = self.planner.plan(config)
        output = self.root / 'out'
        manifest = BuildManifest.create(output, config, plan.inspection, plan.to_dict())
        artifact = output / '模型.ninfer'
        artifact.write_bytes(b'test')
        BuildRunner(self.planner, ninfer_dir=self.root / '引擎')._write_launcher(artifact, output, config, manifest, plan)
        body = (output / 'scripts/run.bat').read_text(encoding='utf-8')
        self.assertIn('模型.ninfer', body)
        self.assertIn('引擎', body)
        self.assertIn('chcp 65001', body)
        # Failures must keep the window open instead of flashing closed.
        self.assertIn('cd /d "%~dp0"', body)
        self.assertIn('if not exist "%SERVER%"', body)
        self.assertIn('if not exist "%MODEL%"', body)
        self.assertIn('pause', body)

    def test_root_startup_batch_keeps_window_on_failure(self):
        bat = Path(__file__).resolve().parents[1] / '启动 TurboForge.bat'
        body = bat.read_text(encoding='utf-8')
        self.assertIn('chcp 65001', body)
        self.assertIn('start_web.ps1', body)
        self.assertIn('pause', body)
        self.assertIn('if errorlevel 1', body)


if __name__ == '__main__':
    unittest.main()
