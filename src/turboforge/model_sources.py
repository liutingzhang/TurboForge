"""Resolve one model reference and fetch only its required model resources."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import urllib.parse
from typing import Callable

from .bootstrap import Emit, download, read_url, safe_member

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
from .domain import MODEL_RESOURCE_FILES, QUANTIZATION_CONFIG_FILES
_RESOURCES = MODEL_RESOURCE_FILES


def normalize_reference(value: str) -> str:
    """Accept paths pasted with Windows 'Copy as path' quotes."""
    value = value.strip()
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        value = value[1:-1].strip()
    return value


def repo_id(value: str) -> str | None:
    value = normalize_reference(value)
    if value.startswith("https://huggingface.co/"):
        value = "/".join(urllib.parse.urlparse(value).path.strip("/").split("/")[:2])
    return value if _REPO.fullmatch(value) else None


def reference_parts(reference: str) -> tuple[str | None, str | None, str | None]:
    reference = normalize_reference(reference)
    repository = repo_id(reference)
    parts = urllib.parse.urlparse(reference).path.strip("/").split("/") if reference.startswith("https://") else []
    revision = urllib.parse.unquote(parts[3]) if len(parts) >= 4 and parts[2] in ("tree", "resolve", "blob") else None
    filename = urllib.parse.unquote("/".join(parts[4:])) if len(parts) > 4 and parts[2] in ("resolve", "blob") else None
    return repository, revision, filename


@dataclass
class ModelSource:
    model: Path
    sources: dict[str, str] = field(default_factory=dict)


class ModelResolver:
    def __init__(self, root: Path, emit: Emit, check_config: Callable[[dict], None] | None = None, *, hf_mode: bool = False):
        self.root = root
        self.emit = emit
        self.check_config = check_config
        self.hf_mode = hf_mode

    def info(self, repository: str, revision: str | None = None) -> dict:
        suffix = "/revision/" + urllib.parse.quote(revision, safe="") if revision else ""
        value = json.loads(read_url(f"https://huggingface.co/api/models/{repository}{suffix}?blobs=true"))
        if not re.fullmatch(r"[a-f0-9]{40}", value.get("sha", "")):
            raise ValueError("模型仓库没有有效的固定版本")
        return value

    @staticmethod
    def nvfp4_sources(base: Path, checkpoint: Path, inspection) -> ModelSource:
        from .speculative import companions
        sources={"quantized":str(checkpoint),**companions(checkpoint)}
        for name in ("mtp","dflash2"):
            if inspection.speculative_components.get(name,{}).get("source") == "base_checkpoint":
                sources[name]=str(checkpoint)
        return ModelSource(base,sources)

    def snapshot(self, repository: str, *, resources_only: bool = False, revision: str | None = None,
                 filename: str | None = None) -> Path:
        info = self.info(repository, revision)
        revision = info["sha"]
        target = self.root / "models" / repository.replace("/", "--") / revision
        files = info.get("siblings", [])
        resources = [item for item in files if item["rfilename"] in _RESOURCES]
        selected = list(resources)
        gguf_file = None
        if not resources_only:
            weights = [item for item in files if item["rfilename"].endswith(".safetensors") and "/" not in item["rfilename"]]
            if weights and not (filename and filename.endswith(".gguf")):
                config_item = next((item for item in resources if item["rfilename"] == "config.json"), None)
                if not config_item:
                    raise ValueError("模型仓库缺少 config.json，无法确认架构")
                config_path = target / "config.json"
                download(f"https://huggingface.co/{repository}/resolve/{revision}/config.json", config_path, emit=self.emit)
                from .backends.registry import RECIPES
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if not config.get("quantization_config"):
                    for sidecar in QUANTIZATION_CONFIG_FILES:
                        if any(item["rfilename"] == sidecar for item in resources):
                            download(f"https://huggingface.co/{repository}/resolve/{revision}/{sidecar}",target/sidecar,emit=self.emit)
                            config["quantization_config"] = json.loads((target/sidecar).read_text(encoding="utf-8-sig"))
                            break
                from .backends.hf_quantization import quant_method, check_packed, HF_ARCHITECTURES
                if quant_method(config) in ("awq", "gptq"):
                    if not self.hf_mode:
                        raise ValueError("AWQ/GPTQ 需要选择 HF 原生保留路线，不能交给 NInfer；尚未下载权重")
                    check_packed(config)
                if self.check_config:
                    self.check_config(config)
                supported = {arch for recipe in RECIPES for arch in recipe.architectures}
                if self.hf_mode:
                    supported.update(HF_ARCHITECTURES)
                if not any(arch in supported for arch in config.get("architectures", [])):
                    raise ValueError("该模型架构尚不受转换器支持，已在下载大权重前停止")
                selected += weights + [item for item in files if item["rfilename"] == "model.safetensors.index.json"]
            else:
                ggufs = [item for item in files if item["rfilename"].lower().endswith(".gguf")
                         and not any(tag in item["rfilename"].lower() for tag in ("mmproj", "dflash", "mtp", "vision", "engram", "imatrix", "dspark"))]
                if not ggufs:
                    raise ValueError("仓库没有可转换的 Safetensors 或文本 GGUF 权重")
                # Candidate selection is based on declared size. Actual precision
                # and architecture are read from the downloaded header, never the name.
                firsts = [item for item in ggufs if not re.search(r"-\d{5}-of-\d{5}\.gguf$", item["rfilename"])
                          or "-00001-of-" in item["rfilename"]]
                if not firsts:
                    raise ValueError("GGUF 仓库缺少第一分片")
                if filename:
                    chosen = next((item for item in ggufs if item["rfilename"] == filename), None)
                    if chosen is None:
                        raise ValueError("模型链接指定的 GGUF 文件不存在")
                    filename = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-00001-of-\1.gguf", filename)
                    chosen = next((item for item in firsts if item["rfilename"] == filename), None)
                    if chosen is None:
                        raise ValueError("所选 GGUF 缺少第一分片")
                else:
                    if len(firsts) != 1:
                        raise ValueError("仓库包含多个 GGUF 精度，请选择具体 GGUF 文件链接；不会擅自改用最小文件")
                    chosen = firsts[0]
                name = chosen["rfilename"]
                gguf_file = name
                match = re.search(r"-00001-of-(\d{5})\.gguf$", name)
                if match:
                    prefix = name[:match.start()]
                    selected += [item for item in ggufs if re.fullmatch(re.escape(prefix) + r"-\d{5}-of-" + match[1] + r"\.gguf", item["rfilename"])]
                    shard_numbers = {int(re.search(r"-(\d{5})-of-", item["rfilename"])[1]) for item in selected if item in ggufs}
                    if shard_numbers != set(range(1, int(match[1]) + 1)):
                        raise ValueError("GGUF 仓库缺少分片，已在下载前停止")
                else:
                    selected.append(chosen)
                self.emit("自动选择仓库中的较小文本 GGUF：" + name)
        if not resources_only:
            for component in ("mtp","dflash2"):
                prefixes=[prefix for prefix in (component+"/","components/"+component+"/") if any(item["rfilename"] == prefix+"config.json" for item in files)]
                if len(prefixes) > 1: raise ValueError("仓库提供多份 "+component.upper()+"，请明确整理来源，不能猜测")
                for prefix in prefixes:
                    companion_config=json.loads(read_url(f"https://huggingface.co/{repository}/resolve/{revision}/{prefix}config.json"))
                    if component == "dflash2" and companion_config.get("architectures") != ["DFlash2DraftModel"]:
                        raise ValueError("仓库 dflash2 目录没有 DFlash2DraftModel 声明")
                    selected += [item for item in files if item["rfilename"].startswith(prefix) and (item["rfilename"].endswith(".safetensors") or Path(item["rfilename"]).name in _RESOURCES or item["rfilename"].endswith(".safetensors.index.json"))]
        total = sum(item.get("size", item.get("lfs", {}).get("size", 0)) for item in selected)
        self.root.mkdir(parents=True, exist_ok=True)
        pending = sum(item.get("size", item.get("lfs", {}).get("size", 0)) for item in selected
                      if not (target / item["rfilename"]).is_file())
        if shutil.disk_usage(self.root).free < pending + 256 * 1024**2:
            raise RuntimeError(f"模型所需下载约 {total / 1024**3:.2f} GiB，缓存盘空间不足")
        self.emit(f"获取 {repository}@{revision[:12]}，本次所需文件约 {total / 1024**3:.2f} GiB")
        for item in selected:
            name = item["rfilename"]
            destination = safe_member(target, name)
            lfs = item.get("lfs") or {}
            download(f"https://huggingface.co/{repository}/resolve/{revision}/{urllib.parse.quote(name)}", destination,
                     sha256=lfs.get("sha256"), size=item.get("size", lfs.get("size")), emit=self.emit)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".turboforge-model.json").write_text(json.dumps({"repo": repository, "revision": revision,
            "card": info.get("cardData") or {}, "resources_only": resources_only, "gguf_file": gguf_file}), encoding="utf-8")
        return target

    def origin(self, directory: Path, metadata: dict | None = None, *, allow_self: bool = False) -> str | None:
        candidates = list((metadata or {}).get("origin_repositories", []))
        receipt = directory / ".turboforge-model.json"
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            base = saved.get("card", {}).get("base_model")
            candidates += [base] if isinstance(base, str) else base if isinstance(base, list) else []
        config = directory / "config.json"
        if config.is_file():
            data = json.loads(config.read_text(encoding="utf-8"))
            candidates += [data.get("base_model_name_or_path", ""), data.get("_name_or_path", "")]
        known = list(dict.fromkeys(repo_id(value) for value in candidates if isinstance(value, str) and repo_id(value)))
        if len(known) == 1:
            return known[0]
        if len(known) > 1:
            raise ValueError("模型声明了多个不同的原始来源，无法安全自动选择。请直接提供原始模型的 Hugging Face 仓库链接。")
        if allow_self and receipt.is_file() and repo_id(saved.get("repo", "")):
            return saved["repo"]
        return None

    def resolve(self, reference: str, original_model: str | None = None) -> ModelSource:
        from .inspector import ModelInspector
        from .domain import InputKind
        reference = normalize_reference(reference)
        original_model = normalize_reference(original_model) if original_model else None
        path = Path(reference).expanduser()
        if not path.exists():
            repository = repo_id(reference)
            if repository is None:
                raise ValueError("找不到模型；请选择本机模型，或填写 Hugging Face 的组织/模型名、仓库链接")
            repository, revision, filename = reference_parts(reference)
            path = self.snapshot(repository, revision=revision, filename=filename)
            receipt = json.loads((path / ".turboforge-model.json").read_text(encoding="utf-8"))
            if receipt.get("gguf_file"):
                path = safe_member(path, receipt["gguf_file"])
        inspection = ModelInspector().inspect(path)
        from .backends.hf_quantization import quant_method,check_packed
        if quant_method(inspection.config) in ("awq", "gptq"):
            if not self.hf_mode:
                raise ValueError("AWQ/GPTQ 需要选择 HF 原生保留路线，不能直接交给 NInfer")
            check_packed(inspection.config)
        if inspection.input_kind == InputKind.NVFP4 and not inspection.quantization.get("nvfp4_layout_verified"):
            raise ValueError("NVFP4 packed 权重或缩放张量不完整，已在下载原始模型前停止")
        from .gguf_standalone import can_prepare, prepare_resources
        if can_prepare(path, inspection):
            self.emit('从 GGUF 内嵌信息生成配置与分词器，不下载原始模型权重')
            return ModelSource(prepare_resources(path, self.root), {'gguf':str(path)})
        if original_model:
            if inspection.input_kind not in (InputKind.GGUF, InputKind.NVFP4):
                raise ValueError("原始来源目录仅适用于 GGUF/NVFP4 导入")
            base = Path(original_model).expanduser()
            if not base.exists():
                repository, revision, filename = reference_parts(original_model)
                if not repository or filename:
                    raise ValueError("原始来源必须是模型文件夹或 Hugging Face 仓库链接")
                base = self.snapshot(repository, revision=revision, resources_only=inspection.input_kind == InputKind.GGUF)
            if not base.is_dir():
                raise ValueError("原始模型来源必须是目录")
            ModelInspector().inspect(base, metadata_only=inspection.input_kind == InputKind.GGUF)
            key = "quantized" if inspection.input_kind == InputKind.NVFP4 else "ternary" if inspection.quantization.get("ternary_pq2") else "gguf"
            return self.nvfp4_sources(base,path,inspection) if inspection.input_kind == InputKind.NVFP4 else ModelSource(base, {key:str(path)})
        if inspection.input_kind == InputKind.GGUF:
            container = path if path.is_dir() else path.parent
            if (container / "config.json").is_file() and any((container / n).is_file() for n in ("tokenizer.json", "tokenizer.model")):
                base = container
            else:
                origin = self.origin(path if path.is_dir() else path.parent, inspection.quantization)
                if not origin:
                    raise ValueError("此 GGUF 没有可确认的原始模型来源，无法凭文件名猜测分词器。请改为提供发布它的 Hugging Face 仓库链接。")
                base = self.snapshot(origin, resources_only=True)
            key = "ternary" if inspection.quantization.get("ternary_pq2") else "gguf"
            return ModelSource(base, {key: str(path)})
        if inspection.input_kind == InputKind.NVFP4:
            origin = self.origin(path)
            if not origin:
                raise ValueError("此 NVFP4 缺少原始模型来源，请提供其 Hugging Face 仓库链接以自动获取来源元数据")
            self.emit("NVFP4 导入需要原始高精度嵌入，将自动获取明确标注的原始检查点")
            return self.nvfp4_sources(self.snapshot(origin),path,inspection)
        directory = path if path.is_dir() else path.parent
        needs_processor = self.hf_mode and inspection.architecture == "Qwen3_5ForConditionalGeneration" and any(not (directory/name).is_file() for name in ("preprocessor_config.json","video_preprocessor_config.json"))
        if needs_processor or not any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
            path = path if path.is_dir() else path.parent
            origin = self.origin(path, allow_self=self.hf_mode)
            if not origin:
                raise ValueError("此模型缺少分词器或处理器配置且未声明来源，请提供其 Hugging Face 仓库链接")
            resources = self.snapshot(origin, resources_only=True)
            # Keep source immutable; use hard links on the same drive rather than
            # duplicating all weights. Cross-drive copying is explicit and checked.
            hydrated = self.root / "hydrated" / hashlib_path(path)
            hydrated.mkdir(parents=True, exist_ok=True)
            for item in path.iterdir():
                if item.is_file() and (item.name in _RESOURCES or item.suffix == ".safetensors" or item.name == "model.safetensors.index.json") and not (hydrated / item.name).exists():
                    try:
                        import os
                        if item.suffix == ".safetensors":
                            os.link(item, hydrated / item.name)
                        else:
                            shutil.copy2(item, hydrated / item.name)
                    except OSError:
                        if shutil.disk_usage(hydrated).free < item.stat().st_size:
                            raise RuntimeError("跨盘准备完整模型副本时磁盘空间不足")
                        shutil.copy2(item, hydrated / item.name)
            for item in resources.iterdir():
                if item.name in _RESOURCES and not (hydrated / item.name).exists():
                    shutil.copy2(item, hydrated / item.name)
            from .speculative import copy_companions
            copy_companions(path,hydrated)
            path = hydrated
        return ModelSource(path)


def hashlib_path(path: Path) -> str:
    import hashlib
    entries = [(item.name, item.stat().st_size, item.stat().st_mtime_ns)
               for item in sorted(path.iterdir()) if item.is_file()
               and (item.suffix == ".safetensors" or item.name in _RESOURCES or item.name == "model.safetensors.index.json")]
    return hashlib.sha256(json.dumps([str(path.resolve()), entries]).encode()).hexdigest()[:20]
