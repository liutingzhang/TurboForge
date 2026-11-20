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
    adapter = tmp_path/'tools/convert/qwen3_5_gguf.py'
    adapter.parent.mkdir(parents=True, exist_ok=True); adapter.touch()
    with patch('turboforge.precheck.cuda_package'), patch('turboforge.precheck.detect_gpu',return_value=None), \
         patch('turboforge.bootstrap.local_tools',return_value=(tmp_path,tmp_path)), \
         patch('turboforge.bootstrap.cache_root',return_value=tmp_path/'cache'):
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
    adapter = tmp_path/'tools/convert/qwen3_5_gguf.py'
    adapter.parent.mkdir(parents=True, exist_ok=True); adapter.touch()
    with patch('turboforge.precheck.cuda_package'), patch('turboforge.precheck.detect_gpu',return_value=None), \
         patch('turboforge.bootstrap.local_tools',return_value=(tmp_path,tmp_path)), \
         patch('turboforge.bootstrap.cache_root',return_value=tmp_path/'cache'):
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


def remote_header(tmp_path, values=None, tensor_name='token_embd.weight'):
    """Small real header with a tensor directory, never a model payload."""
    path=tmp_path/'remote.gguf';write_header(path, metadata() if values is None else values)
    body=bytearray(path.read_bytes())
    struct.pack_into('<Q',body,8,1)
    name=tensor_name.encode()
    body+=struct.pack('<Q',len(name))+name+struct.pack('<IQQIQ',2,1024,6,8,0)
    return bytes(body)


class RangeResponse:
    def __init__(self, body, *, status=206, content_range=None):
        import io
        self.stream=io.BytesIO(body);self.status=status;self.read_sizes=[]
        self.headers={'Content-Range':content_range or f'bytes 0-{len(body)-1}/{len(body)+1000000000}'}

    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,size):
        assert size >= 0, 'unbounded model download'
        self.read_sizes.append(size)
        return self.stream.read(size)


def remote_check(tmp_path, response, *, name='untrusted Q4.gguf', original=None, adapted=True):
    adapter=tmp_path/'tools/convert/qwen3_5_gguf.py'
    adapter.parent.mkdir(parents=True,exist_ok=True)
    if adapted: adapter.touch()
    info={'sha':'b'*40,'siblings':[{'rfilename':name,'size':1000000000}]}
    reference=f'https://huggingface.co/Org/GGUF/blob/dev/{name}'
    with patch('turboforge.precheck.cuda_package'), patch('turboforge.precheck.detect_gpu',return_value=None), \
         patch('turboforge.precheck.ModelResolver.info',return_value=info) as meta, \
         patch('turboforge.bootstrap.local_tools',return_value=(tmp_path,tmp_path)), \
         patch('turboforge.precheck.urllib.request.urlopen',return_value=response) as fetch, \
         patch('turboforge.model_sources.download',side_effect=AssertionError('no model downloads')), \
         patch('turboforge.bootstrap.cache_root',side_effect=AssertionError('no persistent cache writes')), \
         patch('turboforge.precheck._remote_config',side_effect=AssertionError('self-contained GGUF needs no original')):
        result=precheck(reference,original)
    meta.assert_called_once_with('Org/GGUF','dev')
    request=fetch.call_args.args[0]
    assert request.full_url.endswith('/'+'b'*40+'/'+name.replace(' ','%20'))
    assert request.get_header('Range')=='bytes=0-16777215'
    return result


def test_remote_self_contained_q8_returns_same_route_without_original_or_payload(tmp_path):
    body=remote_header(tmp_path)
    response=RangeResponse(body+b'WEIGHT_PAYLOAD_MUST_NOT_BE_READ')
    result=remote_check(tmp_path,response,original=str(tmp_path/'nonexistent'))
    assert result['ready'],result
    assert result['remote'] and not result['can_edit_weights']
    assert result['provided_sources']==['gguf']
    assert [t['value'] for t in result['targets']]==['qwen3_5_gguf_standalone']
    assert result['recommended_max_context']==1024 and result['optional_sources']=={}
    assert '尚未下载或验证' in result['message']
    assert response.stream.tell()==len(body)
    assert not (tmp_path/'gguf-resources').exists()


@pytest.mark.parametrize('key',['tokenizer.chat_template','tokenizer.ggml.tokens','tokenizer.ggml.token_type','qwen35.rope.dimension_sections'])
def test_remote_missing_embedded_metadata_never_claims_standalone(tmp_path,key):
    values=metadata();values.pop(key)
    result=remote_check(tmp_path,RangeResponse(remote_header(tmp_path,values)))
    assert not result['ready'] and result['action']=='provide_original'
    assert not result.get('targets')


@pytest.mark.parametrize('change', ['duplicate_tokens','bad_merge','bad_special_id','mtp','split','not_q8'])
def test_remote_invalid_resources_are_not_guessed(tmp_path,change):
    values=metadata();name='model.gguf';tensor='token_embd.weight'
    if change=='duplicate_tokens': values['tokenizer.ggml.tokens'][1]='a'
    elif change=='bad_merge': values['tokenizer.ggml.merges']=['a b ab']
    elif change=='bad_special_id': values['tokenizer.ggml.eos_token_id']=999
    elif change=='mtp': tensor='blk.0.nextn.weight'
    elif change=='split': values['split.count']=2;name='model-00001-of-00002.gguf'
    elif change=='not_q8': values['general.file_type']=15
    result=remote_check(tmp_path,RangeResponse(remote_header(tmp_path,values,tensor)),name=name)
    assert not result['ready'] and result['action']=='provide_original'


@pytest.mark.parametrize('status,extent',[(200,None),(206,'bytes 1-100/1000000'),(206,'bytes 0-16777216/1000000000'),(206,'bytes 0-99/*')])
def test_remote_range_refusal_never_reads_response_body(tmp_path,status,extent):
    response=RangeResponse(remote_header(tmp_path),status=status,content_range=extent)
    result=remote_check(tmp_path,response)
    assert not result['ready'] and result['action']=='provide_original'
    assert response.read_sizes==[]


def test_remote_header_limit_and_truncation_are_actionable(tmp_path):
    body=remote_header(tmp_path)
    response=RangeResponse(body[:-1])
    result=remote_check(tmp_path,response)
    assert not result['ready'] and result['action']=='provide_original'
    assert sum(response.read_sizes)<=16*1024*1024
    # A declared huge string cannot trigger a huge read/allocation.
    huge=struct.pack('<4sIQQQ',b'GGUF',3,1,1,32*1024*1024)
    response=RangeResponse(huge,content_range='bytes 0-16777215/1000000000')
    result=remote_check(tmp_path,response)
    assert not result['ready']
    assert max(response.read_sizes)<=24


def test_remote_missing_adapter_does_not_offer_hf_for_quantized_source(tmp_path):
    result=remote_check(tmp_path,RangeResponse(remote_header(tmp_path)),adapted=False)
    assert not result['ready'] and result['action']=='choose_original'
    assert '适配器' in result['message']


def test_local_standalone_missing_adapter_is_not_ready_with_empty_targets(tmp_path):
    path=tmp_path/'only.gguf';write_header(path,metadata())
    with patch('turboforge.precheck.cuda_package'),patch('turboforge.precheck.detect_gpu',return_value=None), \
         patch('turboforge.bootstrap.local_tools',return_value=None), \
         patch('turboforge.bootstrap.cache_root',return_value=tmp_path/'cache'):
        result=precheck(str(path))
    assert not result['ready'] and result['action']=='choose_original'
