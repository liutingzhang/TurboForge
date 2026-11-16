"""Small deterministic Chinese/English help index, with no remote dependency."""
from __future__ import annotations

from importlib.resources import files
import json
import math
import re
from collections import Counter

from .capabilities import capability_entries

STOP_TERMS = {"什么", "怎么", "如何", "为什么", "为什", "么模", "可以", "是否", "是不", "不是", "需要", "能用", "还能", "用吗", "了吗", "时候", "一下", "样吗", "么样", "只有", "有一", "动吗", "能启", "么办"}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\d+)\s*[-_]?\s*(?:bits?|位)", r"\1bit", text)
    text = text.replace("提速", "加速").replace("提升", "加速")
    for value in ("变笨", "变傻", "胡言乱语", "答非所问", "能力下降"):
        text = text.replace(value, "质量 损失")
    return text


def terms(text: str) -> set[str]:
    text = normalize(text)
    words = set(re.findall(r"[a-z0-9_][a-z0-9_.-]*", text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return words - STOP_TERMS


class KnowledgeIndex:
    def __init__(self):
        self.entries = json.loads(files("turboforge.assistant").joinpath("knowledge/entries.json").read_text(encoding="utf-8"))
        # Runtime facts replace their distributed copies after a capability change.
        live = {e["id"]: e for e in capability_entries()}
        self.entries = [{**e, **live.pop(e["id"], {})} for e in self.entries]
        self.entries.extend({**e, "evidence_level": "runtime_rules"} for e in live.values())
        self.by_id = {entry["id"]: entry for entry in self.entries}
        self.index = {e["id"]: (terms(e["title"] + " " + " ".join(e["aliases"])), terms(e["body"])) for e in self.entries}
        self.frequencies = Counter(t for title, body in self.index.values() for t in title | body)

    def search(self, query: str, *, topic: str = "", limit: int = 5, context: str = "") -> list[dict]:
        if not query.strip():
            return [e for e in self.entries if not topic or e["topic"] == topic][:limit]
        query_terms = terms(query)
        context_terms = terms(context)
        ranked = []
        for entry in self.entries:
            if topic and entry["topic"] != topic:
                continue
            title, body = self.index[entry["id"]]
            exact = entry["id"].lower() in query_terms
            matched = query_terms & (title | body)
            if not matched:
                continue
            # IDF prevents generic Chinese fragments from outranking MTP/AWQ etc.
            score = 1000 * exact + sum(
                math.log(1 + len(self.entries) / self.frequencies[t]) *
                (3 if t in title else 1) * (4 if re.search(r"[a-z0-9]", t) else 1)
                for t in matched)
            if score:
                score += min(2, len(context_terms & (title | body)))
                ranked.append((score, entry["id"], entry))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        if not ranked or ranked[0][0] < 3:
            return []
        return [row[2] for row in ranked[:limit] if row[0] >= max(3, ranked[0][0] * .25)]
