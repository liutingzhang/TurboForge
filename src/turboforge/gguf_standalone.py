"""Recover Qwen3.5 text resources from an embedded, self-contained GGUF.

No checkpoint paths, repository names or model dimensions are guessed.
"""
from pathlib import Path
import hashlib
import json
import struct
import math

from .adapters.gguf import _read_exact, _read_string, _read_value
from .domain import InputKind, ModelInspection
from .manifest import write_json_atomic

QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
              r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def can_prepare(path: Path, inspection: ModelInspection) -> bool:
    if (inspection.input_kind != InputKind.GGUF or inspection.quantization.get('gguf_file_type') != 7
            or inspection.architecture != 'qwen35' or not path.is_file()):
        return False
    # Embedded metadata is authoritative here; a missing or mistyped original source must never block a self-contained GGUF.
    return read_metadata(path).get('qwen35.embedding_length') == 1024


def read_metadata(path: Path) -> dict:
    with path.open('rb') as stream:
        magic, version, _, count = struct.unpack('<4sIQQ', _read_exact(stream, 24))
        if magic != b'GGUF' or version not in (2, 3):
            raise ValueError('Standalone import needs a valid GGUF v2/v3 file')
        if count > 100_000:
            raise ValueError('Too many GGUF metadata entries')
        metadata = {}
        for _ in range(count):
            key = _read_string(stream)
            if key in metadata:
                raise ValueError('Duplicate GGUF metadata key: ' + key)
            kind = struct.unpack('<I', _read_exact(stream, 4))[0]
            if kind == 9:
                element, size = struct.unpack('<IQ', _read_exact(stream, 12))
                if element == 9 or size > 10_000_000:
                    raise ValueError('Unsupported GGUF metadata array type or length')
                value = [_read_value(stream, element) for _ in range(size)]
            else:
                value = _read_value(stream, kind)
            metadata[key] = value
    return metadata


def configuration(metadata: dict) -> dict:
    if metadata.get('split.count', 1) != 1:
        raise ValueError('This standalone import route currently supports only a complete single-file GGUF, not shards')
    if metadata.get('general.architecture') != 'qwen35' or metadata.get('general.file_type') != 7:
        raise ValueError('Current standalone resource generation supports Qwen3.5 dense Q8_0 GGUF')
    def positive(key):
        value = metadata.get('qwen35.' + key)
        if type(value) is not int or value <= 0:
            raise ValueError('GGUF lacks a valid geometry parameter: qwen35.' + key)
        return value
    def positive_float(key):
        value = metadata.get('qwen35.' + key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('GGUF lacks a valid geometry parameter: qwen35.' + key)
        return value
    hidden = positive('embedding_length')
    if hidden != 1024:
        raise ValueError('This standalone import route is currently validated for hidden_size=1024')
    tokens = metadata.get('tokenizer.ggml.tokens')
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        raise ValueError('The GGUF does not embed a complete vocabulary; a tokenizer cannot be generated')
    if metadata.get('tokenizer.ggml.model') != 'gpt2' or metadata.get('tokenizer.ggml.pre') != 'qwen2':
        raise ValueError('The GGUF embedded tokenizer is not the supported Qwen2 BPE scheme')
    if not metadata.get('tokenizer.ggml.merges') or not metadata.get('tokenizer.chat_template'):
        raise ValueError('The GGUF lacks BPE merges or a chat template; complete frontend resources cannot be generated')
    head = positive('attention.key_length')
    rotary = positive('rope.dimension_count')
    if rotary > head or rotary % 2:
        raise ValueError('Invalid GGUF rotary position-embedding dimension')
    sections = metadata.get('qwen35.rope.dimension_sections')
    if not isinstance(sections, list) or len(sections) < 3 or any(type(v) is not int or v < 0 for v in sections) or sum(sections) != rotary // 2:
        raise ValueError('The GGUF lacks verifiable MRoPE section information')
    if any(sections[3:]):
        raise ValueError('This MRoPE section layout is not yet supported')
    value_heads = positive('ssm.time_step_rank')
    inner = positive('ssm.inner_size')
    if inner % value_heads:
        raise ValueError('GGUF GDN dimensions are not divisible')
    return {
        'architectures': ['Qwen3_5ForCausalLM'], 'model_type': 'qwen3_5_text',
        'hidden_size': hidden, 'vocab_size': len(tokens),
        'num_hidden_layers': positive('block_count'),
        'intermediate_size': positive('feed_forward_length'),
        'max_position_embeddings': positive('context_length'),
        'num_attention_heads': positive('attention.head_count'),
        'num_key_value_heads': positive('attention.head_count_kv'), 'head_dim': head,
        'full_attention_interval': positive('full_attention_interval'),
        'linear_conv_kernel_dim': positive('ssm.conv_kernel'),
        'linear_key_head_dim': positive('ssm.state_size'),
        'linear_value_head_dim': inner // value_heads,
        'linear_num_key_heads': positive('ssm.group_count'),
        'linear_num_value_heads': value_heads,
        'rms_norm_eps': positive_float('attention.layer_norm_rms_epsilon'),
        'hidden_act': 'silu', 'attention_bias': False, 'attn_output_gate': True,
        'mamba_ssm_dtype': 'float32',
        'rope_parameters': {'rope_theta': positive_float('rope.freq_base'),
                            'partial_rotary_factor': rotary / head,
                            'mrope_interleaved': True, 'mrope_section': sections[:3]},
    }


def prepare_resources(path: Path, root: Path) -> Path:
    metadata = read_metadata(path)
    config = configuration(metadata)
    # A missing output tensor explicitly represents tied embeddings in GGUF.
    # Tensor names are inspected rather than inferred from the model's filename.
    from .adapters.gguf import _read_string as read_string
    with path.open('rb') as stream:
        _, _, tensor_count, count = struct.unpack('<4sIQQ', _read_exact(stream, 24))
        for _ in range(count):
            read_string(stream)
            _read_value(stream, struct.unpack('<I', _read_exact(stream, 4))[0])
        names = set()
        for _ in range(tensor_count):
            names.add(read_string(stream))
            rank = struct.unpack('<I', _read_exact(stream, 4))[0]
            if not 1 <= rank <= 4:
                raise ValueError('Unsupported GGUF tensor dimensionality')
            _read_exact(stream, rank * 8 + 12)
    config['tie_word_embeddings'] = 'output.weight' not in names
    if any('nextn' in name for name in names):
        raise ValueError('MTP import for standalone GGUF is not yet validated; the speculative head cannot be silently dropped')
    tokens = metadata['tokenizer.ggml.tokens']
    types = metadata.get('tokenizer.ggml.token_type')
    if not isinstance(types, list) or len(types) != len(tokens) or any(type(t) is not int or t not in range(1, 7) for t in types):
        raise ValueError('The GGUF lacks token_type; special tokens cannot be verified')
    if len(set(tokens)) != len(tokens):
        raise ValueError('The GGUF vocabulary contains duplicate tokens; a tokenizer cannot be generated unambiguously')
    added = [{'id': i, 'content': token, 'single_word': False, 'lstrip': False,
              'rstrip': False, 'normalized': False, 'special': kind == 3}
             for i, (token, kind) in enumerate(zip(tokens, types)) if kind in (3, 4)]
    byte_level = {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}
    merges = []
    for merge in metadata['tokenizer.ggml.merges']:
        pieces = merge.split(' ')
        if len(pieces) != 2:
            raise ValueError('A GGUF BPE merge must contain two tokens')
        merges.append(pieces)
    tokenizer = {'version': '1.0', 'truncation': None, 'padding': None,
                 'added_tokens': added, 'normalizer': {'type': 'NFC'},
                 'pre_tokenizer': {'type': 'Sequence', 'pretokenizers': [
                     {'type': 'Split', 'pattern': {'Regex': QWEN_SPLIT}, 'behavior': 'Isolated', 'invert': False}, byte_level]},
                 'post_processor': byte_level, 'decoder': byte_level,
                 'model': {'type': 'BPE', 'dropout': None, 'unk_token': None,
                           'continuing_subword_prefix': '', 'end_of_word_suffix': '',
                           'fuse_unk': False, 'byte_fallback': False, 'ignore_merges': False,
                           'vocab': {token: i for i, token in enumerate(tokens)}, 'merges': merges}}
    special = {}
    for name in ('bos', 'eos', 'padding'):
        index = metadata.get('tokenizer.ggml.' + name + '_token_id')
        if type(index) is not int or not 0 <= index < len(tokens):
            raise ValueError('GGUF lacks a valid special-token ID: ' + name)
        special[('pad' if name == 'padding' else name) + '_token'] = tokens[index]
    tokenizer_config = {**special, 'add_bos_token': metadata.get('tokenizer.ggml.add_bos_token', False),
                        'add_prefix_space': False, 'chat_template': metadata['tokenizer.chat_template'],
                        'added_tokens_decoder': {str(t['id']): t for t in added}}
    if special['pad_token'] != '<|endoftext|>' or tokenizer_config['add_bos_token'] is not False:
        raise ValueError('The GGUF special-token rules are incompatible with the current Qwen3.5 frontend')
    identity = hashlib.sha256((str(path.resolve()) + str(path.stat().st_size) + str(path.stat().st_mtime_ns)).encode()).hexdigest()[:20]
    target = root / 'gguf-resources' / identity
    target.mkdir(parents=True, exist_ok=True)
    for name, value in {'config.json': config, 'tokenizer.json': tokenizer,
                        'tokenizer_config.json': tokenizer_config,
                        'generation_config.json': {'eos_token_id': metadata['tokenizer.ggml.eos_token_id'],
                                                   'pad_token_id': metadata['tokenizer.ggml.padding_token_id']},
                        'gguf-source.json': {'source': str(path.resolve()), 'metadata_generated': True}}.items():
        write_json_atomic(target / name, value)
    (target / 'chat_template.jinja').write_text(metadata['tokenizer.chat_template'], encoding='utf-8')
    return target
