"""Bounded allowlisted business evidence, separate from untrusted selected text."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re

from .errors import AssistantError
from .capabilities import public_capabilities


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<key hidden>")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"'<>]+", r"\1<hidden>", text)
    text = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|secret|password|authorization)\s*[=:]\s*[\"']?)[^\s,;\"']+", r"\1<hidden>", text)
    text = re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]{12,}", "<key hidden>", text)
    text = re.sub(r"(?i)([?&](?:key|token|api_key|access_token)=)[^&\s]+", r"\1<hidden>", text)
    text = re.sub(r"[A-Za-z]:[\\/][^\n\r\"'<>|]*", "<local path>", text)
    text = re.sub(r"(?<![:\w/])/(?:home|Users|mnt|tmp|var|root)/[^\n\r\"'<>|]*", "<local path>", text)
    return text


def redact_stream_prefix(pending: str, secrets: tuple[str, ...]) -> tuple[str, str]:
    """Emit complete text, retaining only a possible split credential or path.

    ASCII tokens may be key fragments; ordinary prose need not wait for
    an arbitrary 256-character buffer. Whitespace closes generic credentials.
    """
    cut = len(pending)
    tail = re.search(r"[A-Za-z0-9_:/\\?&=.'\"-]+$", pending)
    if tail:
        cut = tail.start()
    # A path can contain spaces; keep it until the same boundary used by redact.
    path = re.search(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|tmp|var|root)/)[^\n\r\"'<>|]*$", pending)
    if path:
        cut = min(cut, path.start())
    marker = re.search(r"(?i)(?:bearer\s+|(?:api[_-]?key|access[_-]?token|secret|password|authorization)\s*[=:]\s*[\"']?)[^\s,;\"']*$", pending)
    if marker:
        cut = min(cut, marker.start())
    for secret in secrets:
        for length in range(min(len(secret) - 1, len(pending)), 0, -1):
            if pending.endswith(secret[:length]):
                cut = min(cut, len(pending) - length)
                break
    return redact(pending[:cut], secrets), pending[cut:]


def clean(value, secrets: tuple[str, ...], *, depth: int = 0):
    if depth > 7:
        return "<depth limit>"
    if isinstance(value, str):
        return redact(value[:6000], secrets)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, list):
        return [clean(item, secrets, depth=depth + 1) for item in value[:30]]
    if isinstance(value, dict):
        return {str(k): clean(v, secrets, depth=depth + 1) for k, v in list(value.items())[:40]
            if not re.search(r"(?i)(path|directory|_dir$|key|tokenizer_files|password|secret|command|environment)", str(k))}
    return None


def validate_chat(body: dict) -> dict:
    allowed = {"question", "selection", "build_id", "conversation_id", "request_id", "include", "captured_at", "page_summary"}
    if set(body) - allowed:
        raise AssistantError("chat_input", "The assistant request contains unknown fields")
    question = body.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise AssistantError("chat_input", "The question must be 1–2000 characters")
    selections = body.get("selection", [])
    if not isinstance(selections, list) or len(selections) > 5:
        raise AssistantError("chat_input", "At most 5 selection blocks can be attached")
    total = 0
    for item in selections:
        if not isinstance(item, dict) or set(item) - {"id", "title", "text", "code"}:
            raise AssistantError("chat_input", "Invalid selection format")
        for key, limit in (("id", 180), ("title", 180), ("text", 6000), ("code", 180)):
            value = item.get(key, "")
            if not isinstance(value, str) or len(value) > limit:
                raise AssistantError("chat_input", "Selection content is too large or invalid")
        total += len(item.get("text", ""))
    if total > 6000:
        raise AssistantError("chat_input", "Selections exceed 6000 characters in total; narrow the scope", 413)
    for key in ("conversation_id", "build_id", "request_id"):
        value = body.get(key)
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value)):
            raise AssistantError("chat_input", "Invalid conversation or build identifier")
    include = body.get("include", {})
    if not isinstance(include, dict) or set(include) - {"business_summary", "recent_logs"} or any(not isinstance(v, bool) for v in include.values()):
        raise AssistantError("chat_input", "Invalid attached-material selection")
    if not isinstance(body.get("captured_at", ""), str) or len(body.get("captured_at", "")) > 80:
        raise AssistantError("chat_input", "Invalid capture time")
    summary = body.get("page_summary", {})
    if not isinstance(summary, dict) or len(json.dumps(summary)) > 12000:
        raise AssistantError("chat_input", "Page summary is too large or invalid")
    return body


def business_snapshot(manager, session, body: dict) -> dict:
    """Only existing owned build IDs can read authoritative private build details."""
    include = body.get("include", {})
    build_id = body.get("build_id")
    snapshot = {"captured_at": datetime.now(timezone.utc).isoformat(), "build_id": build_id,
        "selected_at": body.get("captured_at", ""), "authority": "no_build_evidence",
        "capability_rules": public_capabilities()}
    secrets = (session.config.api_key, session.default.api_key)
    if include.get("business_summary", True):
        allowed = ("architecture", "input_kind", "weight_precision", "target", "engine", "speculative", "stage", "diagnostics")
        snapshot["untrusted_page_summary"] = clean({k: v for k, v in body.get("page_summary", {}).items() if k in allowed}, secrets)
    if build_id is None:
        if include.get("recent_logs"):
            raise AssistantError("no_build", "No logs from this build are available to attach yet")
        return snapshot
    if build_id not in session.builds:
        raise AssistantError("build_forbidden", "This build does not belong to this assistant session; create a new build on the current page", 403)
    record = manager.status(build_id)
    if record is None:
        raise AssistantError("build_missing", "The record for this build no longer exists", 404)
    snapshot["authority"] = "backend_build_record"
    if include.get("business_summary", True):
        manifest = record.get("manifest") or {}
        config = manifest.get("config") or record.get("prepared_config") or {}
        plan = manifest.get("plan") or {}
        source = manifest.get("source") or {}
        snapshot["business"] = clean({"status": record.get("status"), "finished": record.get("finished"),
            "model": {k: source.get(k) for k in ("architecture", "input_kind", "weight_precision")},
            "config": {k: config.get(k) for k in ("target", "jailbreak", "quantization", "runtime", "run_validation", "run_benchmark")},
            "diagnostics": plan.get("diagnostics", []), "serving": plan.get("serving"),
            "performance": manager.report(build_id, "performance"), "quality": manager.report(build_id, "quality")}, secrets)
    if include.get("recent_logs"):
        log = redact(manager.log(build_id) or "", secrets)
        lines = log.splitlines()
        tail = "\n".join(lines[-120:])
        raw = tail.encode("utf-8")
        snapshot["recent_logs"] = raw[-12288:].decode("utf-8", errors="ignore")
        snapshot["logs_truncated"] = len(lines) > 120 or len(raw) > 12288
    return snapshot
