"""NInfer custom recipes using gguf's real block quantizers, never AWQ."""
import numpy as np
import torch
from gguf import GGMLQuantizationType
from gguf.quants import quantize, dequantize
from tools.convert.methods import import_encoded, cast_direct
from tools.convert.sources.logical import EncodedRows, LogicalSource


def checked_quantize(rows, qtype):
    if not np.isfinite(rows).all():
        raise ValueError("GGML 量化输入含 NaN/Inf")
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        blocks = quantize(rows,qtype)
        restored = dequantize(blocks,qtype)
    if not np.isfinite(restored).all():
        raise ValueError("GGML 量化块解码产生 NaN/Inf，拒绝保存溢出权重")
    return blocks


def _configure(model, recipe, type_name):
    for name, parameter in model.parameters.items():
        if name.startswith(("dflash/","dflash2/")) and name.endswith(("/attention_conv/kernel_projection","/mlp_conv/kernel_projection","/candidate_selector/hidden_projection")):
            continue
        if name.endswith(("gdn/a_projection", "gdn/b_projection")):
            continue
        if not parameter.projection and name != "text/token_embedding":
            continue
        if len(parameter.shape) != 2 or parameter.shape[1] % 32:
            raise ValueError(f"{name}: GGML block encoding requires K divisible by 32")
        # Preserve embeddings and output heads, including the MTP head, in Q8.
        keep_q8 = name == "text/token_embedding" or name.endswith("output_head") or name.startswith(("mtp/","dflash/","dflash2/"))
        selected = "Q8_0" if keep_q8 else type_name
        fmt = "gguf_" + selected.lower()
        qtype = getattr(GGMLQuantizationType, selected)
        source = parameter.source
        def encoded(begin, end, source=source, fmt=fmt, qtype=qtype):
            rows = np.ascontiguousarray(source.rows(begin, end).to(torch.float32).numpy())
            blocks = checked_quantize(rows, qtype)
            return EncodedRows(fmt, torch.from_numpy(blocks.copy()), torch.empty(0, dtype=torch.float16))
        wrapped = LogicalSource(source.shape, source.label + "[" + fmt + "]", source.read_values, encoded)
        recipe.assign(name, format=fmt, method=import_encoded, source=wrapped)
    preserve_draft_formats(model,recipe,{})


def preserve_draft_formats(model,recipe,sources):
    """DFlash2 control projections/codebooks retain the native direct contract."""
    for name,parameter in model.parameters.items():
        if name.startswith(("dflash/","dflash2/")) and name.endswith(("/attention_conv/kernel_projection","/mlp_conv/kernel_projection","/candidate_selector/hidden_projection")):
            recipe.assign(name,format=parameter.direct_format,layout="auto",method=cast_direct,source=parameter.source)
    for backend in ("dflash","dflash2"):
        if backend in model.components:
            for layer in range(model.components[backend]["config"]["num_hidden_layers"]):
                for role in ("key","value"):
                    prefix=f"{backend}/layers/{layer}/attention/"
                    recipe.share(prefix+"context_"+role,prefix+role)


def q4_0(model, recipe, sources):
    _configure(model, recipe, "Q4_0")


def q5_0(model, recipe, sources):
    _configure(model, recipe, "Q5_0")


def mxfp4(model, recipe, sources):
    _configure(model, recipe, "MXFP4")


def standalone_qwen35(model, recipe, sources):
    """Use the GGUF as the only weight store; no original HF weights are read."""
    import atexit
    from tools.convert.sources.gguf_source import (
        GGUFSource, HFAliasSource, standard_dense_name_map, qwen35_linear_attention_name_map,
    )
    raw = GGUFSource(sources['gguf'].path)
    atexit.register(raw.close)
    config = model.config
    if config['hidden_size'] != 1024 or raw.metadata().get('general.file_type') != 7:
        raise ValueError('此独立导入配方仅支持 Qwen3.5 hidden_size=1024 的 Q8_0 GGUF')
    names, transforms, shapes = standard_dense_name_map(
        config['num_hidden_layers'], tie_word_embeddings=config['tie_word_embeddings'], text_prefix='model.')
    for layer, kind in enumerate(config['layer_types']):
        if kind != 'linear_attention':
            continue
        mapping, changes, reshapes = qwen35_linear_attention_name_map(
            layer, num_k_heads=config['linear_num_key_heads'], num_v_heads=config['linear_num_value_heads'],
            head_k_dim=config['linear_key_head_dim'], head_v_dim=config['linear_value_head_dim'], text_prefix='model.')
        names.update(mapping)
        transforms.update(changes)
        shapes.update(reshapes)
    alias = HFAliasSource(raw, names, transforms=transforms, shapes=shapes)
    alias.config = sources['base'].config
    for name, parameter in model.parameters.items():
        source = model.source(name, alias)
        if (parameter.projection or name == 'text/token_embedding') and not name.endswith(('gdn/a_projection', 'gdn/b_projection')):
            def encoded(begin, end, source=source):
                rows = np.ascontiguousarray(source.rows(begin, end).to(torch.float32).numpy())
                blocks = checked_quantize(rows, GGMLQuantizationType.Q8_0)
                return EncodedRows('gguf_q8_0', torch.from_numpy(blocks.copy()), torch.empty(0, dtype=torch.float16))
            wrapped = LogicalSource(source.shape, source.label + '[GGUF->Q8_0]', source.read_values, encoded)
            recipe.assign(name, format='gguf_q8_0', method=import_encoded, source=wrapped)
        else:
            recipe.assign(name, format=parameter.direct_format, method=cast_direct, source=source)
