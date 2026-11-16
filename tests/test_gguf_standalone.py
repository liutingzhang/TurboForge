"""Resource recovery uses GGUF contents without a sibling HF checkpoint."""
import json
import struct
from pathlib import Path
from unittest.mock import patch
import pytest
from turboforge.gguf_standalone import prepare_resources, configuration, read_metadata
from turboforge.model_sources import ModelResolver
from turboforge.precheck import precheck


def metadata():
    values = {'general.architecture':'qwen35', 'general.file_type':7,
        'tokenizer.ggml.model':'gpt2', 'tokenizer.ggml.pre':'qwen2',
        'tokenizer.ggml.tokens':['a','b','ab','<|endoftext|>','<|im_start|>','<|im_end|>'],
        'tokenizer.ggml.token_type':[1,1,1,3,3,3], 'tokenizer.ggml.merges':['a b'],
        'tokenizer.chat_template':'{{ messages[0]["content"] }}',
        'tokenizer.ggml.bos_token_id':3,'tokenizer.ggml.eos_token_id':5,'tokenizer.ggml.padding_token_id':3}
    values.update({'qwen35.'+k:v for k,v in {
        'embedding_length':1024, 'block_count':24, 'feed_forward_length':3584,
        'context_length':262144,'attention.head_count':8,'attention.head_count_kv':2,
        'attention.key_length':256,'attention.layer_norm_rms_epsilon':1e-6,
        'rope.freq_base':10000000.,'rope.dimension_count':64,'rope.dimension_sections':[11,11,10,0],
        'ssm.conv_kernel':4,'ssm.state_size':128,'ssm.group_count':16,
        'ssm.time_step_rank':16,'ssm.inner_size':2048,'full_attention_interval':4}.items()})
    return values


def write_header(path, values):
    def string(value):
        value=value.encode('utf-8'); return struct.pack('<Q',len(value))+value
    def scalar(value):
        if isinstance(value,str): return 8,string(value)
        if isinstance(value,float): return 6,struct.pack('<f',value)
        return 4,struct.pack('<I',value)
    body=bytearray(struct.pack('<4sIQQ',b'GGUF',3,0,len(values)))
    for key,value in values.items():
        body+=string(key)
        if isinstance(value,list):
            kind=scalar(value[0])[0]
            body+=struct.pack('<IIQ',9,kind,len(value))+b''.join(scalar(v)[1] for v in value)
        else:
            kind,data=scalar(value);body+=struct.pack('<I',kind)+data
    path.write_bytes(body)


def test_embedded_resources_are_generated_without_original_directory(tmp_path):
    path=tmp_path/'only.gguf'; write_header(path,metadata())
    folder=prepare_resources(path,tmp_path/'cache')
    config=json.loads((folder/'config.json').read_text())
    tokenizer=json.loads((folder/'tokenizer.json').read_text())
    assert config['hidden_size']==1024 and config['linear_value_head_dim']==128
    assert config['rope_parameters']['mrope_section']==[11,11,10]
    assert tokenizer['model']['vocab']['<|im_end|>']==5
    assert tokenizer['model']['merges']==[['a','b']]
    assert (folder/'chat_template.jinja').read_text()==metadata()['tokenizer.chat_template']
    assert not (tmp_path/'config.json').exists()
    assert not list(folder.glob('*.safetensors'))
    with patch('turboforge.precheck.cuda_package'), patch('turboforge.bootstrap.cache_root',return_value=tmp_path/'cache'):
        result=precheck(str(path),str(tmp_path))
    assert result['ready'],result
    assert result['recommended_max_context']==1024
    assert result['targets'][0]['value']=='qwen3_5_gguf_standalone'
    with patch('turboforge.model_sources.read_url',side_effect=AssertionError('must not download original weights')):
        resolved=ModelResolver(tmp_path/'cache',lambda _:None).resolve(str(path))
    assert resolved.sources=={'gguf':str(path)}
    assert resolved.model==folder


def test_self_contained_gguf_ignores_bogus_original_source(tmp_path):
    path=tmp_path/'only.gguf'; write_header(path,metadata())
    bogus=tmp_path/'wrong-source'; bogus.mkdir()
    with patch('turboforge.precheck.cuda_package'), patch('turboforge.bootstrap.cache_root',return_value=tmp_path/'cache'):
        result=precheck(str(path),str(bogus))
    assert result['ready'],result
    assert result['targets'][0]['value']=='qwen3_5_gguf_standalone'
    with patch('turboforge.model_sources.read_url',side_effect=AssertionError('must not download original weights')):
        resolved=ModelResolver(tmp_path/'cache',lambda _:None).resolve(str(path),str(bogus))
    assert resolved.sources=={'gguf':str(path)}


@pytest.mark.parametrize('key', ['tokenizer.ggml.tokens','tokenizer.ggml.merges','tokenizer.chat_template','qwen35.rope.dimension_sections','qwen35.rope.freq_base'])
def test_missing_metadata_is_not_guessed(key):
    values=metadata();values.pop(key)
    with pytest.raises(ValueError): configuration(values)


def test_duplicate_vocabulary_is_rejected(tmp_path):
    values=metadata();values['tokenizer.ggml.tokens'][1]='a'
    path=tmp_path/'duplicate.gguf';write_header(path,values)
    with pytest.raises(ValueError,match='重复'):prepare_resources(path,tmp_path/'cache')


def test_wrong_geometry_is_not_silently_accepted():
    values=metadata();values['qwen35.embedding_length']=2048
    with pytest.raises(ValueError,match='1024'):configuration(values)
