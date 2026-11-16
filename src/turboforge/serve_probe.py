"""Measure a RUNNING NInfer server over its OpenAI-compatible endpoint.

This is the measurement path used when the artifact is already served (no free
VRAM to launch a second Engine). It reports only what was actually observed:
TTFT, wall time, completion tokens and achieved decode throughput. Speculation
acceptance is read from /stats when the server exposes it, and is otherwise
reported as unavailable rather than inferred.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ServeProbe:
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_seconds: float | None = None
    wall_seconds: float | None = None
    decode_tokens_per_second: float | None = None
    accepted_tokens_per_round: float | None = None
    text_preview: str = ""
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": self.ttft_seconds,
            "wall_seconds": self.wall_seconds,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "accepted_tokens_per_round": self.accepted_tokens_per_round,
            "text_preview": self.text_preview,
            "diagnostics": self.diagnostics,
        }


def _stats(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/stats", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def probe(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: int = 600,
) -> ServeProbe:
    """One timed streaming completion; numbers come only from this run."""
    result = ServeProbe(model=model)
    before = _stats(base_url.rstrip("/"))
    decode_before = (before or {}).get("counters", {}).get("committed_decode_tokens")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    text_parts: list[str] = []
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            first_event = True
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if first_event:
                    result.ttft_seconds = time.perf_counter() - started
                    first_event = False
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
    except OSError as exc:
        result.diagnostics.append(f"endpoint error: {exc}")
        return result
    result.wall_seconds = time.perf_counter() - started
    result.text_preview = "".join(text_parts)[:200]

    after = _stats(base_url.rstrip("/"))
    decode_after = (after or {}).get("counters", {}).get("committed_decode_tokens")
    if decode_before is not None and decode_after is not None:
        result.completion_tokens = decode_after - decode_before
        decode_seconds = (result.wall_seconds or 0.0) - (result.ttft_seconds or 0.0)
        if decode_seconds > 0 and result.completion_tokens > 0:
            result.decode_tokens_per_second = result.completion_tokens / decode_seconds
        rounds = (after or {}).get("counters", {}).get("decode_rounds", 0) - \
                 (before or {}).get("counters", {}).get("decode_rounds", 0)
        if rounds > 0 and result.completion_tokens:
            result.accepted_tokens_per_round = result.completion_tokens / rounds
    else:
        result.diagnostics.append("/stats unavailable; token counts not measured")
    return result


if __name__ == "__main__":
    import argparse
    from .local_endpoint import default_serving_port

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=f"http://127.0.0.1:{default_serving_port()}")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain in one short paragraph why the sky is blue.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    outcome = probe(args.url, args.model, args.prompt, max_tokens=args.max_tokens)
    print(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
