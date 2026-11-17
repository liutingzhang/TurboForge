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
            raise ValueError("The model repository has no valid pinned revision")
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
                    raise ValueError("The model repository lacks config.json; the architecture cannot be confirmed")
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
                        raise ValueError("AWQ/GPTQ must use the HF native retention route and cannot be handed to NInfer; weights have not been downloaded yet")
                    check_packed(config)
                if self.check_config:
                    self.check_config(config)
                supported = {arch for recipe in RECIPES for arch in recipe.architectures}
                if self.hf_mode:
                    supported.update(HF_ARCHITECTURES)
                if not any(arch in supported for arch in config.get("architectures", [])):
                    raise ValueError("This model architecture is not yet supported by the converter; stopped before downloading large weights")
                selected += weights + [item for item in files if item["rfilename"] == "model.safetensors.index.json"]
            else:
                ggufs = [item for item in files if item["rfilename"].lower().endswith(".gguf")
                         and not any(tag in item["rfilename"].lower() for tag in ("mmproj", "dflash", "mtp", "vision", "engram", "imatrix", "dspark"))]
                if not ggufs:
                    raise ValueError("The repository has no convertible Safetensors or text GGUF weights")
                # Candidate selection is based on declared size. Actual precision
                # and architecture are read from the downloaded header, never the name.
                firsts = [item for item in ggufs if not re.search(r"-\d{5}-of-\d{5}\.gguf$", item["rfilename"])
                          or "-00001-of-" in item["rfilename"]]
                if not firsts:
                    raise ValueError("The GGUF repository is missing the first shard")
                if filename:
                    chosen = next((item for item in ggufs if item["rfilename"] == filename), None)
                    if chosen is None:
                        raise ValueError("The GGUF file specified by the model link does not exist")
                    filename = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-00001-of-\1.gguf", filename)
                    chosen = next((item for item in firsts if item["rfilename"] == filename), None)
                    if chosen is None:
                        raise ValueError("The selected GGUF is missing the first shard")
                else:
                    if len(firsts) != 1:
                        raise ValueError("The repository contains multiple GGUF precisions; select a specific GGUF file link. The smallest file is never substituted automatically")
                    chosen = firsts[0]
                name = chosen["rfilename"]
                gguf_file = name
                match = re.search(r"-00001-of-(\d{5})\.gguf$", name)
                if match:
                    prefix = name[:match.start()]
                    selected += [item for item in ggufs if re.fullmatch(re.escape(prefix) + r"-\d{5}-of-" + match[1] + r"\.gguf", item["rfilename"])]
                    shard_numbers = {int(re.search(r"-(\d{5})-of-", item["rfilename"])[1]) for item in selected if item in ggufs}
                    if shard_numbers != set(range(1, int(match[1]) + 1)):
                        raise ValueError("The GGUF repository is missing shards; stopped before downloading")
                else:
                    selected.append(chosen)
                self.emit("Automatically selected the smaller text GGUF in the repository: " + name)
        if not resources_only:
            for component in ("mtp","dflash2"):
                prefixes=[prefix for prefix in (component+"/","components/"+component+"/") if any(item["rfilename"] == prefix+"config.json" for item in files)]
                if len(prefixes) > 1: raise ValueError("The repository provides multiple "+component.upper()+" copies; organize the sources explicitly, no guessing")
                for prefix in prefixes:
                    companion_config=json.loads(read_url(f"https://huggingface.co/{repository}/resolve/{revision}/{prefix}config.json"))
                    if component == "dflash2" and companion_config.get("architectures") != ["DFlash2DraftModel"]:
                        raise ValueError("The repository dflash2 directory has no DFlash2DraftModel declaration")
                    selected += [item for item in files if item["rfilename"].startswith(prefix) and (item["rfilename"].endswith(".safetensors") or Path(item["rfilename"]).name in _RESOURCES or item["rfilename"].endswith(".safetensors.index.json"))]
        total = sum(item.get("size", item.get("lfs", {}).get("size", 0)) for item in selected)
        self.root.mkdir(parents=True, exist_ok=True)
        pending = sum(item.get("size", item.get("lfs", {}).get("size", 0)) for item in selected
                      if not (target / item["rfilename"]).is_file())
        if shutil.disk_usage(self.root).free < pending + 256 * 1024**2:
            raise RuntimeError(f"The model needs about {total / 1024**3:.2f} GiB of downloads; the cache disk has insufficient space")
        self.emit(f"Fetching {repository}@{revision[:12]}; about {total / 1024**3:.2f} GiB of files needed this run")
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
            raise ValueError("The model declares multiple different original sources; a safe automatic choice is not possible. Provide the original model's Hugging Face repository link directly.")
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
                raise ValueError("Model not found; select a local model, or enter a Hugging Face org/model name or repository link")
            repository, revision, filename = reference_parts(reference)
            path = self.snapshot(repository, revision=revision, filename=filename)
            receipt = json.loads((path / ".turboforge-model.json").read_text(encoding="utf-8"))
            if receipt.get("gguf_file"):
                path = safe_member(path, receipt["gguf_file"])
        inspection = ModelInspector().inspect(path)
        from .backends.hf_quantization import quant_method,check_packed
        if quant_method(inspection.config) in ("awq", "gptq"):
            if not self.hf_mode:
                raise ValueError("AWQ/GPTQ must use the HF native retention route and cannot be handed directly to NInfer")
            check_packed(inspection.config)
        if inspection.input_kind == InputKind.NVFP4 and not inspection.quantization.get("nvfp4_layout_verified"):
            raise ValueError("The NVFP4 packed weights or scale tensors are incomplete; stopped before downloading the original model")
        from .gguf_standalone import can_prepare, prepare_resources
        if can_prepare(path, inspection):
            self.emit('Generating config and tokenizer from the GGUF embedded information; original model weights are not downloaded')
            return ModelSource(prepare_resources(path, self.root), {'gguf':str(path)})
        if original_model:
            if inspection.input_kind not in (InputKind.GGUF, InputKind.NVFP4):
                raise ValueError("The original source directory only applies to GGUF/NVFP4 import")
            base = Path(original_model).expanduser()
            if not base.exists():
                repository, revision, filename = reference_parts(original_model)
                if not repository or filename:
                    raise ValueError("The original source must be a model folder or a Hugging Face repository link")
                base = self.snapshot(repository, revision=revision, resources_only=inspection.input_kind == InputKind.GGUF)
            if not base.is_dir():
                raise ValueError("The original model source must be a directory")
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
                    raise ValueError("This GGUF has no confirmable original model source; the tokenizer cannot be guessed from the file name. Provide the Hugging Face repository link that published it instead.")
                base = self.snapshot(origin, resources_only=True)
            key = "ternary" if inspection.quantization.get("ternary_pq2") else "gguf"
            return ModelSource(base, {key: str(path)})
        if inspection.input_kind == InputKind.NVFP4:
            origin = self.origin(path)
            if not origin:
                raise ValueError("This NVFP4 lacks an original model source; provide its Hugging Face repository link to fetch the source metadata automatically")
            self.emit("NVFP4 import needs the original high-precision embedding; the explicitly labeled original checkpoint will be fetched automatically")
            return self.nvfp4_sources(self.snapshot(origin),path,inspection)
        directory = path if path.is_dir() else path.parent
        needs_processor = self.hf_mode and inspection.architecture == "Qwen3_5ForConditionalGeneration" and any(not (directory/name).is_file() for name in ("preprocessor_config.json","video_preprocessor_config.json"))
        if needs_processor or not any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
            path = path if path.is_dir() else path.parent
            origin = self.origin(path, allow_self=self.hf_mode)
            if not origin:
                raise ValueError("This model lacks a tokenizer or processor config and declares no source; provide its Hugging Face repository link")
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
                            raise RuntimeError("Insufficient disk space while preparing a full model copy across drives")
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
