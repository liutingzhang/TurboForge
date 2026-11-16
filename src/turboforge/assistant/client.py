"""Bounded Chat Completions transport: pinned DNS, TLS hostname, no redirects/proxy."""
from __future__ import annotations

from dataclasses import dataclass
import http.client
import ipaddress
import json
import socket
import ssl
from threading import BoundedSemaphore, Event, Lock, Thread, Timer
import time
from urllib.parse import urlsplit

from .config import ConnectionConfig, validate_connection
from .errors import AssistantError

_DNS_SLOTS = BoundedSemaphore(4)


def resolve_addresses(host: str, port: int, control: RequestControl | None = None):
    if not _DNS_SLOTS.acquire(blocking=False):
        raise AssistantError("dns_busy", "域名解析繁忙，请稍后重试", 429)
    result = []
    finished = Event()

    def resolve():
        try:
            result.append(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError:
            result.append(None)
        finally:
            _DNS_SLOTS.release()
            finished.set()

    Thread(target=resolve, daemon=True).start()
    deadline = time.monotonic() + 5
    while not finished.wait(0.1):
        if control:
            control.check()
        if time.monotonic() >= deadline:
            raise AssistantError("dns_timeout", "模型域名解析超时，请检查 DNS 与网络", 504)
    if not result[0]:
        raise AssistantError("network", "无法解析模型服务域名，请检查地址与网络", 502)
    return result[0]


def origin(value: str) -> str:
    url = urlsplit(value)
    try:
        port = url.port or (443 if url.scheme == "https" else 80)
    except ValueError:
        raise AssistantError("base_url", "Base URL 端口无效") from None
    host = (url.hostname or "").lower()
    return f"{url.scheme.lower()}://{'[' + host + ']' if ':' in host else host}:{port}"


@dataclass(frozen=True)
class Target:
    host: str
    port: int
    ip: str
    secure: bool
    path: str


def target_for(config: ConnectionConfig, allowed_local: tuple[str, ...], control: RequestControl | None = None) -> Target:
    validate_connection(config)
    url = urlsplit(config.base_url)
    port = url.port or (443 if url.scheme == "https" else 80)
    local_allowed = origin(config.base_url) in {origin(x) for x in allowed_local}
    try:
        addresses = resolve_addresses(url.hostname or "", port, control)
    except OSError:
        raise AssistantError("network", "无法解析模型服务域名，请检查地址与网络", 502) from None
    ips = sorted({address[4][0] for address in addresses})
    if not ips:
        raise AssistantError("network", "模型服务没有可连接地址", 502)
    for address in ips:
        ip = ipaddress.ip_address(address)
        ip = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip
        if not ip.is_global and not (local_allowed and ip.is_loopback):
            raise AssistantError("unsafe_url", "此服务解析到私网、本机或保留地址；本机模型需管理员允许精确 origin", 403)
        if local_allowed and not ip.is_loopback:
            raise AssistantError("unsafe_url", "本机例外只能解析到 loopback，不能用于远端或其他内网地址", 403)
    if url.scheme != "https" and not local_allowed:
        raise AssistantError("unsafe_url", "远端模型服务必须使用 HTTPS", 403)
    if config.auth_mode == "none" and not local_allowed:
        raise AssistantError("unsafe_url", "无需 Key 模式仅允许管理员指定的本机服务", 403)
    return Target(url.hostname or "", port, ips[0], url.scheme == "https", url.path.rstrip("/") + "/chat/completions")


class PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, target: Target, timeout: float):
        super().__init__(target.host, target.port, timeout=timeout)
        self.target = target

    def connect(self):
        self.sock = socket.create_connection((self.target.ip, self.target.port), self.timeout)
        if self.target.secure:
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.target.host)


class RequestControl:
    def __init__(self):
        self.cancelled = Event()
        self.expired = Event()
        self.lock = Lock()
        self.connection: http.client.HTTPConnection | None = None
        self.sock: socket.socket | None = None
        self.timer: Timer | None = None
        self.deadline: float | None = None

    def bind(self, connection: http.client.HTTPConnection):
        with self.lock:
            self.connection = connection
        self.check()

    def bind_socket(self):
        with self.lock:
            if self.connection is not None:
                self.sock = self.connection.sock
        self.check()

    def interrupt(self, *, expired: bool = False):
        (self.expired if expired else self.cancelled).set()
        with self.lock:
            sock = self.sock
            connection = self.connection
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection:
            connection.close()

    def check(self):
        if self.cancelled.is_set():
            raise AssistantError("cancelled", "已停止读取回答；上游仍可能计费", 409)
        if self.expired.is_set() or (self.deadline is not None and time.monotonic() >= self.deadline):
            raise AssistantError("timeout", "模型服务超过总时限，请缩短问题或检查服务", 504)

    def close(self):
        if self.timer:
            self.timer.cancel()
        with self.lock:
            if self.connection:
                self.connection.close()
            self.connection = None
            self.sock = None


def sse_data(response: http.client.HTTPResponse):
    """Yield complete SSE data frames; bound individual lines/frames and full response."""
    data: list[str] = []
    size = total = 0
    while True:
        raw = response.readline(65537)
        if not raw:
            if data:
                raise AssistantError("upstream_protocol", "模型流在事件结束前断开", 502)
            return
        total += len(raw)
        size += len(raw)
        if len(raw) > 65536 or size > 65536 or total > 2 * 1024 * 1024:
            raise AssistantError("upstream_protocol", "模型流响应超过大小限制", 502)
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            raise AssistantError("upstream_protocol", "模型流不是有效 UTF-8", 502) from None
        if not line:
            if data:
                yield "\n".join(data)
            data, size = [], 0
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


class ChatClient:
    def __init__(self, allowed_local: tuple[str, ...] = ()):
        self.allowed_local = allowed_local

    def events(self, config: ConnectionConfig, messages: list[dict], control: RequestControl):
        started = time.monotonic()
        control.deadline = started + config.timeout_seconds
        control.timer = Timer(config.timeout_seconds, lambda: control.interrupt(expired=True))
        control.timer.daemon = True
        control.timer.start()
        response = None
        try:
            target = target_for(config, self.allowed_local, control)
            control.check()
            connection = PinnedHTTPConnection(target, timeout=min(15, config.timeout_seconds))
            control.bind(connection)
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if config.stream else "application/json"}
            if config.auth_mode == "bearer":
                headers["Authorization"] = "Bearer " + config.api_key
            body = json.dumps({"model": config.model, "messages": messages, "stream": config.stream,
                "max_tokens": config.max_output_tokens}, ensure_ascii=False).encode("utf-8")
            connection.connect()
            control.bind_socket()
            connection.request("POST", target.path, body, headers)
            response = connection.getresponse()
            control.check()
            if response.status != 200:
                public = {401: "模型服务鉴权失败，请检查 API Key", 403: "模型服务拒绝访问，请检查权限",
                    404: "模型服务路径或模型不存在，请检查根地址与模型 ID", 413: "模型服务认为上下文过大，请缩小选区或新建对话",
                    429: "模型服务限流或额度不足，请检查服务账户与并发"}
                message = public.get(response.status, "模型服务返回异常状态，请检查服务状态")
                if 300 <= response.status < 400:
                    message = "模型服务返回重定向，已阻止转发凭据；请填写最终 HTTPS 根地址"
                raise AssistantError("upstream_" + str(response.status), message, response.status if response.status in (401, 403, 404, 413, 429) else 502)
            if config.stream:
                if "text/event-stream" not in response.getheader("Content-Type", "").lower():
                    raise AssistantError("upstream_protocol", "服务未返回 SSE，请检查流式能力或在设置中明确选择非流式", 502)
                finished = False
                received_text = False
                finish_reason = None
                usage = None
                for data in sse_data(response):
                    control.check()
                    if data.strip() == "[DONE]":
                        finished = True
                        break
                    event = self._json(data)
                    if event.get("error"):
                        raise AssistantError("upstream_stream", "模型服务在生成中返回错误；回答未完成", 502)
                    choices = event.get("choices")
                    if not isinstance(choices, list):
                        raise AssistantError("upstream_protocol", "模型 SSE 缺少 choices", 502)
                    if event.get("usage") is not None:
                        usage = self._usage(event["usage"])
                    if not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                        raise AssistantError("upstream_protocol", "模型 SSE 文本增量格式无效", 502)
                    content = choice["delta"].get("content")
                    if content is not None and not isinstance(content, str):
                        raise AssistantError("upstream_protocol", "模型 SSE 只支持文本回答", 502)
                    if content:
                        received_text = True
                        yield "delta", {"text": content}
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])[:80]
                control.check()
                if not finished:
                    raise AssistantError("upstream_eof", "模型流意外结束，回答未完成，请手动重试", 502)
                if not received_text:
                    if finish_reason == "length":
                        raise AssistantError("output_budget", "模型输出预算已用尽且没有正文，请提高输出上限或显式选择其他模型", 502)
                    raise AssistantError("empty_answer", "模型流结束但没有正文，请检查模型能力与输出预算", 502)
                control.check()
                yield "done", {"finish_reason": finish_reason or "stop", "usage": usage}
            else:
                raw = response.read(2 * 1024 * 1024 + 1)
                control.check()
                if len(raw) > 2 * 1024 * 1024:
                    raise AssistantError("upstream_protocol", "模型响应超过大小限制", 502)
                event = self._json(raw)
                try:
                    text = event["choices"][0]["message"]["content"]
                    if not isinstance(text, str) or not text:
                        if event["choices"][0].get("finish_reason") == "length":
                            raise AssistantError("output_budget", "模型输出预算已用尽且没有正文，请提高输出上限或显式选择其他模型", 502)
                        raise ValueError
                    reason = event["choices"][0].get("finish_reason", "stop")
                except (KeyError, IndexError, TypeError, ValueError):
                    raise AssistantError("upstream_protocol", "模型未返回有效文本回答", 502) from None
                yield "delta", {"text": text}
                yield "done", {"finish_reason": str(reason)[:80], "usage": self._usage(event.get("usage"))}
            control.check()
            if time.monotonic() - started > config.timeout_seconds:
                raise AssistantError("timeout", "模型服务超过总时限", 504)
        except AssistantError:
            control.check()
            raise
        except TimeoutError:
            control.check()
            raise AssistantError("response_timeout", "连接或等待模型响应超时，请检查服务，未自动重试", 504) from None
        except (OSError, http.client.HTTPException, ValueError):
            control.check()
            raise AssistantError("network", "模型连接、证书或响应读取失败，请检查服务；未自动重试", 502) from None
        finally:
            if response:
                response.close()
            control.close()

    @staticmethod
    def _json(value: str | bytes) -> dict:
        try:
            result = json.loads(value)
        except (ValueError, UnicodeDecodeError):
            raise AssistantError("upstream_protocol", "模型服务返回无效 JSON", 502) from None
        if not isinstance(result, dict):
            raise AssistantError("upstream_protocol", "模型响应必须是 JSON 对象", 502)
        return result

    @staticmethod
    def _usage(value) -> dict | None:
        if not isinstance(value, dict):
            return None
        result = {key: value[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(value.get(key), int) and not isinstance(value[key], bool) and value[key] >= 0}
        return result or None
