"""Compile reviewed help into distributable records; --check detects stale data."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from turboforge.assistant.capabilities import capability_entries
from turboforge.backends.registry import PINNED_UPSTREAM_REVISION
DEST = ROOT / "src/turboforge/assistant/knowledge/entries.json"
TOPICS = dict(zip("ABCDEFGHIJKL", ("start", "input", "quantization", "resources", "calibration",
    "speculation", "errors", "quality", "performance", "serving", "connection", "maintenance")))


def source_sha256(path: Path) -> str:
    """Hash UTF-8 text with universal newlines, independent of Git checkout EOLs."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def compile_entries() -> list[dict]:
    source = ROOT / "docs/AI_ASSISTANT_KNOWLEDGE.md"
    diagnostics = ROOT / "docs/AI_ASSISTANT_DIAGNOSTICS.md"
    refs: dict[str, list[str]] = {}
    for path in (ROOT / "src/turboforge").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Diagnostic":
                value = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "code"), None)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    refs.setdefault(value.value, []).append(path.relative_to(ROOT).as_posix())
    entries = []

    def add(key: str, title: str, body: str, topic: str, references: list[str], aliases: list[str] | None = None):
        entries.append({"id": key, "title": title, "body": body, "topic": topic,
            "aliases": aliases or [], "applies_to": ["TurboForge"],
            "product_version": re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1),
            "upstream_revision": PINNED_UPSTREAM_REVISION,
            "evidence_level": "code_documented", "source_refs": sorted(set(references)),
            "source_sha256": {name: source_sha256(ROOT / name) for name in sorted(set(references))},
            "related_ids": []})

    for entry in capability_entries():
        add(entry["id"], entry["title"], entry["body"], entry["topic"], entry["source_refs"], entry["aliases"])
        entries[-1]["evidence_level"] = "runtime_rules"

    for line in source.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if cells and re.fullmatch(r"[A-L]\d{2}", cells[0]):
            key, title, body = cells
            add(key, title, body, TOPICS[key[0]], [source.relative_to(ROOT).as_posix()])
    for line in diagnostics.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if not cells:
            continue
        key = cells[0].strip("`")
        if key in refs:
            add(key, cells[1], cells[1] + "。下一步：" + cells[2], "errors", refs[key], [key])
        elif re.fullmatch(r"runtime\.[a-z_.]+", key):
            add(key, cells[1].removeprefix("拟新增："), cells[2], "errors", [diagnostics.relative_to(ROOT).as_posix()], [key])
    field_source = ROOT / "docs/AI_ASSISTANT_FIELDS.md"
    field_aliases = set()
    for line in field_source.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if cells and cells[0].startswith("field."):
            key, names, explanation, constraints, topic = cells
            aliases = names.split("、")
            field_aliases.update(aliases)
            add(key, names, explanation + "。默认/约束：" + constraints, topic,
                [field_source.relative_to(ROOT).as_posix()], aliases)
        elif cells and cells[0].startswith("state."):
            key, title, explanation, launch = cells
            add(key, title, explanation + "。启动：" + launch, "serving" if key.startswith("state.server") else "errors",
                [field_source.relative_to(ROOT).as_posix()], title.split(" / "))
    for path in (ROOT / "web/src/App.tsx", ROOT / "web/src/components/ConfigPanel.tsx"):
        labels = set(re.findall(r'<Field\b[^>]*?\blabel="([^"]+)"', path.read_text(encoding="utf-8")))
        if labels - field_aliases:
            raise ValueError("Uncovered page fields: " + repr(sorted(labels - field_aliases)))
    meta = ROOT / "src/turboforge/api/meta.py"
    for node in ast.walk(ast.parse(meta.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_param" and len(node.args) >= 3:
            key, title, kind = (ast.literal_eval(value) for value in node.args[:3])
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            add("parameter." + key, title, kwargs.get("help_", "") + "。参数类型：" + kind +
                "；默认取当前元数据/处理器计算结果，不推测缺失值。", "calibration", [meta.relative_to(ROOT).as_posix()], [title, key])
    precheck = ROOT / "src/turboforge/precheck.py"
    purposes = {"quantized":"已有 NVFP4 检查点，需匹配生产者布局", "gguf":"已量化 GGUF 的完整文件与分片",
        "ternary":"已发布且匹配几何的 PQ2_0 三值来源，不训练新权重", "imatrix":"与当前模型张量匹配的重要性矩阵",
        "engram":"指定架构要求的 Engram 表", "ngram":"支持架构的 n-gram 表，未提供时按真实路线处理",
        "mtp":"与当前模型兼容的外部 MTP 组件，不重新训练", "dflash2":"带明确兼容声明的 DFlash2 检查点，不等同于普通 DFlash"}
    sources = {}
    for node in ast.walk(ast.parse(precheck.read_text(encoding="utf-8-sig"))):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "SOURCE_FIELDS":
            sources.update(ast.literal_eval(node.value))
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "optional" and isinstance(target.slice, ast.Constant):
            sources[target.slice.value] = ast.literal_eval(node.value)
    for key, info in sources.items():
        add("source." + key, info["label"], purposes[key] + "。默认不提供；仅在当前配方/功能需要时补齐，实际来源与形状须通过后端校验。",
            "speculation" if key in ("mtp", "dflash2") else "input", [precheck.relative_to(ROOT).as_posix()],
            [info["label"], key, "来源 · " + key, "来源 " + key])
    expected = {f"{letter}{i:02}" for letter in TOPICS for i in range(1, 11)} | set(refs)
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)) or not expected.issubset(ids):
        raise ValueError("Knowledge coverage is incomplete or duplicated: " + str(sorted(expected - set(ids))))
    return entries


if __name__ == "__main__":
    payload = json.dumps(compile_entries(), ensure_ascii=False, indent=2) + "\n"
    if "--check" in sys.argv:
        if not DEST.is_file() or DEST.read_text(encoding="utf-8") != payload:
            raise SystemExit("Assistant knowledge is stale; run scripts/build_assistant_knowledge.py")
        print("Assistant knowledge is current")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(payload, encoding="utf-8")
        print("Compiled assistant knowledge records")
