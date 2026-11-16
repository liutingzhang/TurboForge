"""Reproducible, disjoint automatic evaluation data; no model imports."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .manifest import write_json_atomic

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
AUTO_EVALUATION_TOKENS = 2048


def evaluation_text(benchmark: dict, output: Path) -> tuple[str | None, dict]:
    if benchmark.get("auto_quality"):
        from datasets import load_dataset
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                            split="test", revision=WIKITEXT_REVISION)
        text = "\n\n".join(row["text"] for row in rows if row["text"].strip())
        identity = {"dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
                    "revision": WIKITEXT_REVISION, "split": "test",
                    "token_limit": AUTO_EVALUATION_TOKENS,
                    "note": "固定测试集开头最多 2048 token；独立于默认 train 校准，不代表全任务能力"}
    elif benchmark.get("perplexity_text"):
        text = Path(benchmark["perplexity_text"]).read_text(encoding="utf-8-sig")
        identity = {"source": "user_text"}
    else:
        return None, {}
    if not text.strip():
        raise ValueError("质量评测语料为空")
    identity["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    write_json_atomic(output / "evaluation.json", identity)
    return text, identity
