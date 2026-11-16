import json
import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from turboforge.backends import hf_environment as environment
from turboforge.manifest import file_sha256
from turboforge.model_sources import ModelResolver
from tests.helpers import write_config,write_safetensors


def test_bundled_windows_wheels_have_verified_content_and_licenses():
    root=Path(environment.__file__).parents[1]/'data/hf-wheels'
    for name,digest in environment.WHEELS.values():
        assert file_sha256(root/name)==digest
    assert (root/'PCRE2-LICENSE.txt').is_file() and (root/'PyPcre-LICENSE.txt').is_file()


def test_corrupt_wheel_is_never_installed():
    with (patch.object(environment.os,'name','nt'),patch.object(environment.subprocess,'run',return_value=SimpleNamespace(stdout='3.10')),
          patch('turboforge.manifest.file_sha256',return_value='corrupt'),patch('turboforge.bootstrap.run') as install):
        with pytest.raises(RuntimeError,match='校验失败'):
            environment.install_pcre(Path('python.exe'),lambda _:None)
        install.assert_not_called()


def test_invalid_artifact_does_not_trigger_environment_install(tmp_path,monkeypatch):
    from turboforge.backends.hf_quantization import server_command
    with patch('turboforge.backends.hf_worker.verify_checkpoint',side_effect=ValueError('quality gate failed')), \
         patch('turboforge.backends.hf_quantization.ensure_environment') as install:
        with pytest.raises(ValueError,match='quality gate'):
            server_command({},tmp_path,{})
        install.assert_not_called()


def test_probe_uses_utf8_and_checks_pinned_runtime(tmp_path,monkeypatch):
    from turboforge.backends import hf_quantization as backend
    python=tmp_path/'python.exe';python.touch()
    monkeypatch.setattr(backend,'environment_python',lambda:python)
    with patch.object(backend.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as probe:
        assert backend.ensure_environment(lambda _:None)==python
    assert probe.call_args.kwargs['env']['PYTHONIOENCODING']=='utf-8'
    code=probe.call_args.args[0][-1]
    assert 'torchvision' in code and '5.20.2' in code


def test_new_gpu_environment_uses_patched_environment_and_preserves_legacy_files(tmp_path,monkeypatch):
    from turboforge.backends.hf_quantization import environment_python
    monkeypatch.delenv('TURBOFORGE_HF_HOME',raising=False)
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path/'user'))
    monkeypatch.setattr('turboforge.bootstrap.cache_root',lambda:tmp_path/'project')
    assert environment_python()==tmp_path/'user/NInferTurboForge/hf-env-py31318/Scripts/python.exe'
    legacy=tmp_path/'project/hf-env/Scripts/python.exe'
    legacy.parent.mkdir(parents=True);legacy.touch()
    assert environment_python()==tmp_path/'user/NInferTurboForge/hf-env-py31318/Scripts/python.exe'
    assert legacy.is_file()


def test_python311_uses_existing_managed_313_for_binary_dependencies(tmp_path,monkeypatch):
    python=tmp_path/'NInferTurboForge/Python31318/python.exe'
    python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    with patch.object(environment.os,'name','nt'),patch.object(environment.sys,'version_info',(3,11,9)),patch.object(environment.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
        assert environment.bootstrap_python(tmp_path,lambda _:None)==python


def test_hf_missing_processors_hydrate_from_recorded_repo_without_mutating_source(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    write_config(source,architectures=['Qwen3_5ForConditionalGeneration'],model_type='qwen3_5',text_config={'hidden_size':1024})
    write_safetensors(source/'model.safetensors',{'weight':('BF16',[32,32])})
    (source/'.turboforge-model.json').write_text(json.dumps({'repo':'org/model','card':{},'revision':'a'*40}))
    resources=tmp_path/'resources';resources.mkdir()
    for name in ('preprocessor_config.json','video_preprocessor_config.json'):
        (resources/name).write_text('{}')
    resolver=ModelResolver(tmp_path/'cache',lambda _:None,hf_mode=True)
    assert resolver.origin(source) is None  # An NVFP4 base must not become its own quantized repo.
    monkeypatch.setattr(resolver,'snapshot',lambda repo,**kw:resources)
    result=resolver.resolve(str(source))
    assert result.model != source
    assert (result.model/'video_preprocessor_config.json').is_file()
    assert not (source/'preprocessor_config.json').exists()
    assert file_sha256(result.model/'model.safetensors')==file_sha256(source/'model.safetensors')


def test_cpu_reference_uses_torch_and_restores_accelerated_functions_after_error():
    from turboforge.backends.hf_worker import cpu_reference_kernels
    def reference(): return 'torch'
    @functools.wraps(reference)
    def accelerated(): raise ValueError('CPU pointer cannot be accessed from Triton')
    module=SimpleNamespace(torch_chunk_gated_delta_rule=accelerated,torch_recurrent_gated_delta_rule=accelerated)
    with patch('importlib.import_module',return_value=module):
        with pytest.raises(RuntimeError,match='test interruption'):
            with cpu_reference_kernels('qwen3_5'):
                assert module.torch_chunk_gated_delta_rule()=='torch'
                raise RuntimeError('test interruption')
    assert module.torch_chunk_gated_delta_rule is accelerated
    assert module.torch_recurrent_gated_delta_rule is accelerated
