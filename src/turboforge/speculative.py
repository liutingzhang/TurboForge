"""Lossless auxiliary-head IO and header-only component retention checks.

The source checkpoint is immutable. Detection, storage retention, runtime
availability and measured speedup are deliberately separate facts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct

from .domain import MODEL_RESOURCE_FILES
from .manifest import file_sha256, write_json_atomic
from .safetensors_io import read_tensor_summaries

COMPONENTS = ("mtp", "dflash2")


def copy_companions(source: Path, destination: Path) -> None:
    for component,directory in companions(source).items():
        origin=Path(directory)
        for path in origin.rglob("*"):
            if not path.is_file() or (path.suffix != ".safetensors" and path.name not in MODEL_RESOURCE_FILES and not path.name.endswith(".safetensors.index.json")):
                continue
            if origin not in path.resolve().parents: raise ValueError("组件链接指向来源目录之外")
            target=destination/component/path.relative_to(origin)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copy2(path,target)


def component_name(name: str) -> tuple[str,str] | None:
    for component in COMPONENTS:
        for prefix in (component+".", "model."+component+".", "model.language_model."+component+"."):
            if name.startswith(prefix):
                return component, component+"."+name[len(prefix):]
    return None


def companions(directory: Path) -> dict[str,str]:
    found = {}
    for component in COMPONENTS:
        candidates = []
        for path in (directory/component,directory/"components"/component):
            if not (path/"config.json").is_file() or not any(path.glob("*.safetensors")):
                continue
            config = json.loads((path/"config.json").read_text(encoding="utf-8-sig"))
            if component == "dflash2" and config.get("architectures") != ["DFlash2DraftModel"]:
                raise ValueError(f"{path}: DFlash2 目录缺少明确的 DFlash2DraftModel 配置")
            if component == "mtp" and not any(component_name(t.name) for file in path.glob("*.safetensors") for t in read_tensor_summaries(file)[0]):
                raise ValueError(f"{path}: MTP 目录没有实际 MTP 张量")
            candidates.append(path.resolve())
        if len(candidates) > 1:
            raise ValueError(f"{component.upper()} 存在多个自带来源，请明确指定，不会随意选取")
        if candidates:
            found[component] = str(candidates[0])
    return found


def tensors(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        read_tensor_summaries(path)
        with path.open("rb") as stream:
            length = struct.unpack("<Q",stream.read(8))[0]
            header = json.loads(stream.read(length))
        # Validate offsets/types with the same boundary reader as inspection.
        for name,value in header.items():
            if name == "__metadata__":
                continue
            if name in result:
                raise ValueError("重复来源张量："+name)
            result[name] = {"file":path,"dtype":value["dtype"],"shape":value["shape"],
                            "offset":8+length+value["data_offsets"][0],"bytes":value["data_offsets"][1]-value["data_offsets"][0]}
    return result


def _hash_tensor(item: dict) -> str:
    digest = hashlib.sha256()
    remaining = item["bytes"]
    with item["file"].open("rb") as stream:
        stream.seek(item["offset"])
        while remaining:
            block = stream.read(min(remaining,4*1024**2))
            if not block:
                raise ValueError("组件张量被截断")
            digest.update(block);remaining-=len(block)
    return digest.hexdigest()


def write_tensors(destination: Path, selected: dict) -> None:
    header,offset = {},0
    for name,item in selected.items():
        header[name] = {"dtype":item["dtype"],"shape":item["shape"],"data_offsets":[offset,offset+item["bytes"]]}
        offset+=item["bytes"]
    encoded = json.dumps(header,separators=(",",":")).encode()
    encoded += b" "*((-len(encoded))%8)
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending = destination.with_suffix(destination.suffix+".part")
    with pending.open("wb") as output:
        output.write(struct.pack("<Q",len(encoded)));output.write(encoded)
        for item in selected.values():
            remaining=item["bytes"]
            with item["file"].open("rb") as source:
                source.seek(item["offset"])
                while remaining:
                    block=source.read(min(remaining,4*1024**2))
                    if not block: raise ValueError("组件来源被截断")
                    output.write(block);remaining-=len(block)
    pending.replace(destination)


def prepare_sources(directory: Path, supplied: dict[str,str], work: Path) -> dict[str,str]:
    directory = directory if directory.is_dir() else directory.parent
    result = {**companions(directory),**supplied}
    base_inventory = tensors(directory)
    for component in COMPONENTS:
        origin=Path(result[component]) if component in result else directory
        if not origin.is_dir(): continue  # GGUF MTP imports are handled by the upstream GGUF reader.
        inventory=tensors(origin) if origin != directory else base_inventory
        index=origin/"model.safetensors.index.json"
        mapped=json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.is_file() else {
            name:item["file"].name for name,item in inventory.items() if item["file"].name == "model.safetensors"}
        selected = {name:item for name,item in inventory.items() if component_name(name) and component_name(name)[0] == component}
        if not selected:
            continue
        if component == "mtp" and all(name in mapped and component_name(name)[1] == name for name in selected):
            continue
        target = work/component
        target.mkdir(parents=True,exist_ok=True)
        if component == "dflash2":
            config = json.loads((origin/"config.json").read_text(encoding="utf-8-sig")).get("dflash2_config")
            if not isinstance(config,dict) or config.get("architectures") != ["DFlash2DraftModel"]:
                raise ValueError("内嵌 DFlash2 张量缺少 dflash2_config，不能猜测其配置")
        else:
            config = json.loads((origin/"config.json").read_text(encoding="utf-8-sig"))
        canonical = {}
        for name,item in selected.items():
            renamed=component_name(name)[1]
            if component == "dflash2": renamed=renamed[len("dflash2."):]
            if renamed in canonical: raise ValueError("组件名称规范化发生冲突："+renamed)
            canonical[renamed]=item
        write_tensors(target/"model.safetensors",canonical)
        write_json_atomic(target/"config.json",config)
        result[component]=str(target.resolve())
    return result


def preserve_hf(directory: Path, checkpoint: Path) -> dict:
    original, exported = tensors(directory),tensors(checkpoint)
    bundled=companions(directory)
    if "mtp" in bundled:
        for name,item in tensors(Path(bundled["mtp"])).items():
            if component_name(name) and component_name(name)[0] == "mtp":
                if name in original: raise ValueError("MTP 有多个来源，不能无声覆盖")
                original[name]=item
    missing,record = {},{}
    for name,item in original.items():
        component = component_name(name)
        if component is None: continue
        digest=_hash_tensor(item)
        if name in exported:
            other=exported[name]
            if (item["shape"],item["dtype"],digest) != (other["shape"],other["dtype"],_hash_tensor(other)):
                raise ValueError("量化改动了应保留的推测头："+name)
        else:
            missing[name]=item
        record[name]={"component":component[0],"dtype":item["dtype"],"shape":item["shape"],"sha256":digest}
    if missing:
        shard=checkpoint/"speculative-preserved.safetensors"
        if shard.exists(): raise ValueError("推测头保留文件已存在，拒绝覆盖")
        write_tensors(shard,missing)
    existing_index=checkpoint/"model.safetensors.index.json"
    mapped=json.loads(existing_index.read_text(encoding="utf-8")).get("weight_map",{}) if existing_index.is_file() else {
        name:item["file"].name for name,item in exported.items() if item["file"].name == "model.safetensors"}
    if missing or set(record)-mapped.keys():
        all_tensors=tensors(checkpoint)
        write_json_atomic(checkpoint/"model.safetensors.index.json",{
            "metadata":{"total_size":sum(item["bytes"] for item in all_tensors.values())},
            "weight_map":{name:item["file"].name for name,item in all_tensors.items()}})
    sidecars={}
    for component,path in bundled.items():
        origin=Path(path); target=checkpoint/"speculative-archive"/component if component == "mtp" else checkpoint/component
        for file in origin.rglob("*"):
            if not file.is_file() or (file.suffix != ".safetensors" and file.name not in MODEL_RESOURCE_FILES and not file.name.endswith(".safetensors.index.json")):
                continue
            if origin not in file.resolve().parents: raise ValueError("组件目录链接指向模型之外")
            relative=file.relative_to(origin)
            destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists() and file_sha256(destination) != file_sha256(file): raise ValueError("组件文件冲突："+str(relative))
            if not destination.exists(): shutil.copy2(file,destination)
            sidecars[destination.relative_to(checkpoint).as_posix()]=file_sha256(file)
    source_config=json.loads((directory/"config.json").read_text(encoding="utf-8-sig"))
    config_path=checkpoint/"config.json";config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    for scope in (None,"text_config"):
        source=source_config if scope is None else source_config.get(scope,{})
        destination=config if scope is None else config.setdefault(scope,{}) if scope in source_config else {}
        for key in ("mtp_num_hidden_layers","mtp_use_dedicated_embeddings","mtp","mtp_config","num_mtp_layers","num_nextn_predict_layers","dflash2_config"):
            if key in source: destination[key]=source[key]
    if record or sidecars:
        write_json_atomic(config_path,config)
    result={"status":"retained" if record or sidecars else "not_present","tensors":record,"companion_files":sidecars,
            "runtime_acceleration":"not_supported_in_hf_runtime"}
    write_json_atomic(checkpoint/"speculative-retention.json",result)
    return result


def verify_ninfer(path: Path, expected: list[str]) -> dict:
    with path.open("rb") as stream:
        raw=stream.read(32)
        if len(raw) != 32: raise ValueError("NInfer 组件审计：文件头不完整")
        magic,length,artifact_id=struct.unpack("<8sQ16s",raw)
        if magic != b"NINFER\x00\x03" or not 0 < length <= min(path.stat().st_size-32,256*1024**2):
            raise ValueError("NInfer 组件审计：目录格式或长度无效")
        directory=json.loads(stream.read(length))
    components=directory["components"];bindings=directory["bindings"]
    objects={item["id"]:item for item in directory["objects"]}
    files=directory["files"]
    total_payload=sum(item["payload_bytes"] for item in files)
    for index,item in enumerate(files):
        file=path if index == 0 else (path.parent/item["path"]).resolve()
        if index and path.parent.resolve() not in file.parents: raise ValueError("组件续卷路径越界")
        framing=((32+length+4095)//4096)*4096 if index == 0 else 4096
        if not file.is_file() or file.stat().st_size != framing+item["payload_bytes"]:
            raise ValueError("推测组件产物/续卷不完整")
        if index:
            with file.open("rb") as stream: header=stream.read(32)
            if len(header) != 32 or struct.unpack("<8sQ16s",header) != (b"NINPRT\x00\x03",index,artifact_id):
                raise ValueError("推测组件续卷不属于当前产物")
    result={}
    for component in expected:
        if component not in components: raise ValueError("转换丢失组件："+component)
        names=[name for name in bindings if name.startswith(component+"/")]
        if not names: raise ValueError("组件只有声明，没有实际权重绑定："+component)
        config=components[component].get("config",{})
        architectures=config.get("architectures",[])
        output_head="mtp/output_head" if component == "mtp" and "Gemma4AssistantForCausalLM" in architectures else "text/output_head"
        required=[]
        if component == "mtp" and any(arch in ("Qwen3_5MTP","Qwen3_5MoeMTP") for arch in architectures):
            required=["mtp/"+name for name in ("input_projection","embedding_norm","hidden_norm","final_norm")]
            required += ["mtp/layers/0/"+name for name in ("input_norm","post_attention_norm","attention/query","attention/gate","attention/key","attention/value","attention/query_norm","attention/key_norm","attention/output")]
            if "Qwen3_5MTP" in architectures:
                required += ["mtp/layers/0/mlp/"+name for name in ("gate","up","down")]
        if component == "dflash2":
            layers=config.get("num_hidden_layers")
            if type(layers) is not int or layers < 1: raise ValueError("DFlash2 缺少有效层数")
            required=["dflash2/"+name for name in ("feature_projection","context_norm","final_norm")]
            for layer in range(layers):
                prefix=f"dflash2/layers/{layer}/"
                required += [prefix+name for name in ("input_norm","post_attention_norm","attention/query","attention/key","attention/value","attention/output","attention/query_norm","attention/key_norm","attention/context_key","attention/context_value","mlp/gate","mlp/up","mlp/down","attention_conv/base_kernel","attention_conv/kernel_projection","mlp_conv/base_kernel","mlp_conv/kernel_projection")]
        if any(name not in bindings for name in required):
            raise ValueError("推测头缺少权重："+", ".join(name for name in required if name not in bindings))
        for name in names+[output_head]:
            if name not in bindings: raise ValueError("缺少共享输出头："+name)
            binding=bindings[name]
            refs=binding.get("parts",[binding])
            if not refs or any(ref.get("object") not in objects for ref in refs): raise ValueError("组件权重绑定无效："+name)
            for ref in refs:
                obj=objects[ref["object"]]
                if obj["kind"] != "tensor" or obj["bytes"] <= 0 or not 0 <= obj["offset"] <= total_payload-obj["bytes"]:
                    raise ValueError("推测组件权重数据范围无效："+name)
        if not any(use.get("input") == component+"/final_hidden" and use.get("parameter") == output_head for use in directory["uses"]):
            raise ValueError("推测头未连接共享输出投影："+component)
        if component == "dflash2":
            for suffix in ("hidden_projection","predecessor_codebook","successor_codebook"):
                name=component+"/candidate_selector/"+suffix
                if name not in bindings: raise ValueError("DFlash2 缺少 selector 权重："+suffix)
            for name in names:
                if name.endswith(("/attention_conv/kernel_projection","/mlp_conv/kernel_projection","/candidate_selector/hidden_projection","/candidate_selector/predecessor_codebook","/candidate_selector/successor_codebook")):
                    refs=bindings[name].get("parts",[bindings[name]])
                    if any(objects[ref["object"]]["format"] != "bf16" for ref in refs): raise ValueError("DFlash2 控制投影/码本未保留 BF16："+name)
        if not any("/attention/" in name for name in names) or not any("/mlp/" in name or "/moe/" in name for name in names):
            raise ValueError("推测头缺少完整 attention/MLP 权重："+component)
        result[component]={"retained":True,"binding_count":len(names),"output_head_connected":True,"shared_output_connected":output_head == "text/output_head",
                           "runtime_geometry":"converter_checked; engine validation tracked separately","speedup":"not_inferred_from_presence"}
    return result
