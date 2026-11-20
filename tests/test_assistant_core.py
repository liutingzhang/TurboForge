"""Assistant boundary regressions; local capture service never calls a real provider."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread, Timer
import time
from unittest.mock import patch

import pytest

from turboforge.assistant.client import ChatClient, RequestControl, target_for
from turboforge.assistant.config import ConfigStore, ConnectionConfig, child_environment, from_payload, parse_dotenv
from turboforge.assistant.errors import AssistantError
from turboforge.assistant.retrieval import KnowledgeIndex


def payload(**changes):
    return {"base_url": "https://llm.example/v1", "model": "test-model", "auth_mode": "bearer",
        "secret_action": "replace", "api_key": "CANARY-PRIVATE-KEY", **changes}


def test_knowledge_matches_every_code_and_all_faq():
    from scripts.build_assistant_knowledge import compile_entries, DEST
    assert json.loads(DEST.read_text(encoding="utf-8")) == compile_entries()
    index = KnowledgeIndex()
    assert len([k for k in index.by_id if len(k) == 3 and k[0] in "ABCDEFGHIJKL"]) == 131
    assert index.search("为什么报错 packed-needs-hf")[0]["id"] == "packed-needs-hf"
    assert index.search("AWQ 跟普通 Q4 一样吗")[0]["id"] == "C05"
    assert index.search("PPL 增幅怎么计算")[0]["id"] == "H02"


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_knowledge_source_hash_is_newline_equivalent(tmp_path, newline):
    import hashlib
    from scripts.build_assistant_knowledge import source_sha256

    source = tmp_path / "source.py"
    text = '# 中文源码\nvalue = "unchanged"\n'
    source.write_bytes(text.replace("\n", newline).encode("utf-8"))
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert source_sha256(source) == expected
    source.write_bytes(text.replace("unchanged", "changed").replace("\n", newline).encode("utf-8"))
    assert source_sha256(source) != expected


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_compiled_knowledge_is_identical_across_checkout_newlines(tmp_path, monkeypatch, newline):
    from scripts import build_assistant_knowledge as builder

    # A complete small input tree keeps this EOL regression independent of
    # concurrently edited production diagnostics and the pending entries.json.
    sources = {
        "pyproject.toml": 'version = "0.3.0"\n',
        "docs/AI_ASSISTANT_KNOWLEDGE.md": "".join(
            f"| {letter}{number:02} | 问题 | 回答 |\n" for letter in builder.TOPICS for number in range(1, 11)),
        "docs/AI_ASSISTANT_DIAGNOSTICS.md": "| fixture-error | 诊断 | 建议 |\n| runtime.fixture | 拟新增：说明 | 后续 |\n",
        "docs/AI_ASSISTANT_FIELDS.md": "| field.model | 模型 | 模型目录 | 必填 | input |\n| state.server | 服务 | 状态 | 启动 |\n",
        "src/turboforge/diagnostic.py": 'Diagnostic("fixture-error", "回归")\n',
        "src/turboforge/api/meta.py": '_param("fixture", "模型", "string", help_="帮助")\n',
        "src/turboforge/precheck.py": 'SOURCE_FIELDS = {"mtp": {"label": "MTP"}}\noptional["gguf"] = {"label": "GGUF"}\n',
        "web/src/App.tsx": '<Field label="模型" />\n',
        "web/src/components/ConfigPanel.tsx": '<Field label="模型" />\n',
        "scripts/start_web.py": "# 启动脚本\n",
    }
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "capability_entries", lambda: [{
        "id": "capability.fixture", "title": "能力", "body": "帮助", "topic": "start", "aliases": [],
        "source_refs": ["scripts/start_web.py", "web/src/App.tsx", "src/turboforge/diagnostic.py"],
    }])
    for name, text in sources.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    expected = builder.compile_entries()
    for name, text in sources.items():
        (tmp_path / name).write_bytes(text.replace("\n", newline).encode("utf-8"))
    assert builder.compile_entries() == expected


@pytest.mark.parametrize("question,expected", [
    ("量化后MTP还能用吗", {"F03", "F04"}),
    ("GPTQ默认用什么后端", {"capability.hf_backend"}),
    ("Qwen3.5可以做AWQ吗", {"capability.awq_restrictions"}),
    ("为什么模型量化后变笨了", {"H01", "H02", "H03", "H04"}),
    ("4bit是不是一定更快", {"C04", "I01", "I02"}),
    ("校准文本只有一小段够吗", {"runtime.hf.calibration"}),
    ("MTP头还在为什么速度没提升", {"F03"}),
    ("没网络还能启动吗", {"capability.offline_start"}),
    ("本机模型路径没问题却提示找不到config.json", {"capability.local_model_input"}),
    ("空白环境只下一个gguf，原始模型目录从哪里来", {"capability.clean_gguf"}),
    ("什么是MoE混合专家架构", {"B11"}),
    ("50系显卡为什么推荐NVFP4", {"C11"}),
    ("我的显卡支持原生FP4吗", {"C11"}),
    ("方案按什么排序，推荐是怎么来的", {"C11"}),
    ("MoE模型产出后按显卡推荐哪个推理引擎", {"B12"}),
    ("Strata和Basalt有什么区别", {"C12"}),
    ("Basalt只支持Blackwell显卡吗", {"C13"}),
    ("Strata支持哪些显卡和内存档位", {"C14"}),
    ("MoE引擎推荐显存内存够吗", {"D11"}),
    ("产出的ninfer能用Strata或Basalt跑吗", {"J11"}),
    ("50系显卡用Basalt跑MoE的完整步骤", {"J12"}),
    ("非50系显卡用Strata跑MoE的完整步骤", {"J13"}),
    ("Dense模型用ninfer加kvmem跑起来的步骤", {"J14"}),

])
def test_natural_questions_retrieve_relevant_help_first(question, expected):
    results = KnowledgeIndex().search(question)
    assert results and results[0]["id"] in expected


def test_unrelated_question_does_not_offer_generic_help():
    assert KnowledgeIndex().search("今天天气怎么样") == []


def test_streaming_redacts_split_credentials_without_delaying_chinese():
    from turboforge.assistant.context import redact, redact_stream_prefix
    for text in ("中文回答立即出现。", "answer CANARY-PRIVATE-KEY done。",
                 "Authorization: Bearer PRIVATEVALUE\n结束。", "api_key=VALUE\n安全。",
                 "路径 C:/Users/private folder/file\n结束。", "密钥中文秘密结束。"):
        pending, emitted = "", ""
        for char in text:
            safe, pending = redact_stream_prefix(pending + char, ("CANARY-PRIVATE-KEY", "中文秘密"))
            emitted += safe
        result = emitted + redact(pending, ("CANARY-PRIVATE-KEY", "中文秘密"))
        for private in ("CANARY-PRIVATE-KEY", "PRIVATEVALUE", "VALUE", "Users", "private folder", "中文秘密"):
            assert private not in result
        assert result.endswith(text[-3:])
    assert redact_stream_prefix("这条简短回答应立刻显示。", ("CANARY-PRIVATE-KEY",))[0] == "这条简短回答应立刻显示。"


def test_dotenv_is_literal_strict_and_secret_not_public():
    assert parse_dotenv('\ufeff# comment\r\nTURBOFORGE_AI_MODEL="a$NOT_EXPANDED" # comment\r\n')["TURBOFORGE_AI_MODEL"] == "a$NOT_EXPANDED"
    assert parse_dotenv("TURBOFORGE_AI_API_KEY='abc#def'\n")["TURBOFORGE_AI_API_KEY"] == "abc#def"
    with pytest.raises(AssistantError):
        parse_dotenv("TURBOFORGE_AI_MODEL=x\nTURBOFORGE_AI_MODEL=y")
    with pytest.raises(AssistantError):
        parse_dotenv("TURBOFORGE_AI_MDOEL=x")
    config = from_payload(payload(), ConnectionConfig())
    assert "CANARY" not in repr(config)
    assert "CANARY" not in json.dumps(config.public())


def test_key_actions_and_origin_boundary():
    current = from_payload(payload(), ConnectionConfig())
    keep = {k: v for k, v in payload(secret_action="keep").items() if k != "api_key"}
    assert from_payload(keep, current).api_key == current.api_key
    with pytest.raises(AssistantError, match="origin"):
        from_payload({**keep, "base_url": "https://evil.example/v1"}, current)
    with pytest.raises(AssistantError):
        from_payload(payload(base_url="https://name:secret@llm.example/v1"), current)
    with pytest.raises(AssistantError):
        from_payload(payload(base_url="https://llm.example/v1?token=secret"), current)
    with pytest.raises(AssistantError):
        from_payload(payload(base_url="https://llm.example/v1/chat/completions"), current)


def test_persistence_permissions_and_environment_priority():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".env").write_text("OTHER_SETTING=untouched\nTURBOFORGE_AI_ALLOW_UI_PERSIST=true\n", encoding="utf-8")
        store = ConfigStore(root, {"TURBOFORGE_AI_MODEL": "env-model"})
        config = from_payload(payload(), store.default)
        result = store.persist(config)
        assert result.model == "env-model" and result.source == "environment"
        text = (root / ".env").read_text(encoding="utf-8")
        assert "OTHER_SETTING=untouched" in text
        assert parse_dotenv(text)["TURBOFORGE_AI_API_KEY"] == config.api_key
        old = text
        with patch("turboforge.assistant.config.restrict_file", side_effect=OSError("denied")):
            with pytest.raises(AssistantError):
                store.persist(config)
        assert (root / ".env").read_text(encoding="utf-8") == old
        assert not list(root.glob(".assistant-config-*"))


def test_child_environment_strips_assistant_values():
    with patch.dict("os.environ", {"TURBOFORGE_AI_API_KEY": "CANARY", "PATH": "unchanged"}):
        env = child_environment()
        assert "TURBOFORGE_AI_API_KEY" not in env and env["PATH"] == "unchanged"


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "::ffff:127.0.0.1", "10.0.0.1", "169.254.169.254", "0.0.0.0"])
def test_resolved_private_addresses_blocked(address):
    config = from_payload(payload(), ConnectionConfig())
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", (address, 443))]):
        with pytest.raises(AssistantError):
            target_for(config, ())


def test_every_dns_answer_validated_and_selected_ip_pinned():
    config = from_payload(payload(), ConnectionConfig())
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("127.0.0.1", 443))]):
        with pytest.raises(AssistantError):
            target_for(config, ())
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
        target = target_for(config, ())
        assert target.ip == "8.8.8.8" and target.host == "llm.example" and target.secure


class CaptureHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.server.captured = (self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        mode = self.server.mode
        if isinstance(mode, int):
            self.send_response(mode)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if mode != "json" else "application/json")
        self.end_headers()
        if mode == "json":
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}).encode())
        else:
            if mode == "stall":
                self.wfile.flush()
                time.sleep(2)
                return
            text = 'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\r\n\r\n'
            if mode == "sse":
                text += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            if mode == "invalid":
                text = 'data: invalid\n\n'
            if mode == "error":
                text = 'data: {"error":{"message":"CANARY must never be reflected"}}\n\n'
            for byte in text.encode():
                self.wfile.write(bytes([byte]))


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    server.mode = "sse"
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server
    server.shutdown()
    server.server_close()
    worker.join()


def local_client(server, *, stream=True):
    base = f"http://127.0.0.1:{server.server_port}"
    config = from_payload(payload(base_url=base + "/v1", stream=stream), ConnectionConfig())
    return ChatClient((base,)), config


@pytest.mark.parametrize("mode,stream", [("sse", True), ("json", False)])
def test_real_http_text_and_stream_secret_stays_in_auth(upstream, mode, stream):
    upstream.mode = mode
    client, config = local_client(upstream, stream=stream)
    messages = [{"role": "user", "content": "hello"}]
    events = list(client.events(config, messages, RequestControl()))
    assert events[0] == ("delta", {"text": "answer"})
    assert events[-1][0] == "done"
    path, headers, request = upstream.captured
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer CANARY-PRIVATE-KEY"
    assert "CANARY" not in json.dumps(request)
    assert request["messages"] == messages


@pytest.mark.parametrize("mode", [401, 403, 404, 429, 500, 302, "invalid", "eof", "error"])
def test_upstream_errors_are_public_and_not_silent(upstream, mode):
    upstream.mode = mode
    client, config = local_client(upstream)
    with pytest.raises(AssistantError) as caught:
        list(client.events(config, [{"role": "user", "content": "hello"}], RequestControl()))
    assert "CANARY" not in str(caught.value)
    assert caught.value.code != "cancelled"


def test_total_timeout_interrupts_blocked_response(upstream):
    upstream.mode = "stall"
    client, config = local_client(upstream)
    started = time.monotonic()
    # Disable the timer callback to reproduce scheduler delay deterministically;
    # socket timeout must still be classified by the monotonic total deadline.
    with patch("turboforge.assistant.client.Timer"), pytest.raises(AssistantError) as caught:
        list(client.events(replace(config, timeout_seconds=1), [{"role":"user","content":"hello"}], RequestControl()))
    assert caught.value.code == "timeout"
    assert time.monotonic() - started < 1.8


def test_explicit_cancel_interrupts_blocked_response(upstream):
    upstream.mode = "stall"
    client, config = local_client(upstream)
    control = RequestControl()
    timer = Timer(0.2, control.interrupt)
    timer.start()
    try:
        with pytest.raises(AssistantError) as caught:
            list(client.events(config, [{"role":"user","content":"hello"}], control))
        assert caught.value.code == "cancelled"
        assert control.connection is None
    finally:
        timer.cancel()


@pytest.mark.parametrize("value", [True, 1.5, [], None])
def test_integer_settings_reject_runtime_nonintegers(value):
    with pytest.raises(AssistantError):
        from_payload(payload(context_tokens=value), ConnectionConfig())
