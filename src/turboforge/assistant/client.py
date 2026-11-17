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
        raise AssistantError("dns_busy", "DNS resolution is busy; try again shortly", 429)
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
            raise AssistantError("dns_timeout", "Model hostname resolution timed out; check DNS and the network", 504)
    if not result[0]:
        raise AssistantError("network", "Cannot resolve the model service hostname; check the address and network", 502)
    return result[0]


def origin(value: str) -> str:
    url = urlsplit(value)
    try:
        port = url.port or (443 if url.scheme == "https" else 80)
    except ValueError:
        raise AssistantError("base_url", "Invalid Base URL port") from None
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
        raise AssistantError("network", "Cannot resolve the model service hostname; check the address and network", 502) from None
    ips = sorted({address[4][0] for address in addresses})
    if not ips:
        raise AssistantError("network", "The model service has no connectable address", 502)
    for address in ips:
        ip = ipaddress.ip_address(address)
        ip = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip
        if not ip.is_global and not (local_allowed and ip.is_loopback):
            raise AssistantError("unsafe_url", "This service resolves to a private, loopback, or reserved address; local models require the administrator to allow the exact origin", 403)
        if local_allowed and not ip.is_loopback:
            raise AssistantError("unsafe_url", "The local exception may only resolve to loopback; it cannot be used for remote or other intranet addresses", 403)
    if url.scheme != "https" and not local_allowed:
        raise AssistantError("unsafe_url", "Remote model services must use HTTPS", 403)
    if config.auth_mode == "none" and not local_allowed:
        raise AssistantError("unsafe_url", "Keyless mode is only allowed for administrator-specified local services", 403)
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
            raise AssistantError("cancelled", "Stopped reading the answer; the upstream may still bill", 409)
        if self.expired.is_set() or (self.deadline is not None and time.monotonic() >= self.deadline):
            raise AssistantError("timeout", "The model service exceeded the total time limit; shorten the question or check the service", 504)

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
                raise AssistantError("upstream_protocol", "The model stream disconnected before the event finished", 502)
            return
        total += len(raw)
        size += len(raw)
        if len(raw) > 65536 or size > 65536 or total > 2 * 1024 * 1024:
            raise AssistantError("upstream_protocol", "The model stream response exceeded the size limit", 502)
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            raise AssistantError("upstream_protocol", "The model stream is not valid UTF-8", 502) from None
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
                public = {401: "Model service authentication failed; check the API Key", 403: "The model service denied access; check permissions",
                    404: "The model service path or model does not exist; check the base URL and model ID", 413: "The model service considers the context too large; shrink the selection or start a new conversation",
                    429: "The model service is rate-limited or out of quota; check the service account and concurrency"}
                message = public.get(response.status, "The model service returned an abnormal status; check the service")
                if 300 <= response.status < 400:
                    message = "The model service returned a redirect; forwarding credentials was blocked. Enter the final HTTPS base URL"
                raise AssistantError("upstream_" + str(response.status), message, response.status if response.status in (401, 403, 404, 413, 429) else 502)
            if config.stream:
                if "text/event-stream" not in response.getheader("Content-Type", "").lower():
                    raise AssistantError("upstream_protocol", "The service did not return SSE; check streaming support or explicitly choose non-streaming in settings", 502)
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
                        raise AssistantError("upstream_stream", "The model service returned an error during generation; the answer did not complete", 502)
                    choices = event.get("choices")
                    if not isinstance(choices, list):
                        raise AssistantError("upstream_protocol", "Model SSE is missing choices", 502)
                    if event.get("usage") is not None:
                        usage = self._usage(event["usage"])
                    if not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                        raise AssistantError("upstream_protocol", "Invalid model SSE text-delta format", 502)
                    content = choice["delta"].get("content")
                    if content is not None and not isinstance(content, str):
                        raise AssistantError("upstream_protocol", "Model SSE only supports text answers", 502)
                    if content:
                        received_text = True
                        yield "delta", {"text": content}
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])[:80]
                control.check()
                if not finished:
                    raise AssistantError("upstream_eof", "The model stream ended unexpectedly and the answer is incomplete; retry manually", 502)
                if not received_text:
                    if finish_reason == "length":
                        raise AssistantError("output_budget", "The model output budget was exhausted with no body; raise the output limit or explicitly choose another model", 502)
                    raise AssistantError("empty_answer", "The model stream ended with no body; check model capability and the output budget", 502)
                control.check()
                yield "done", {"finish_reason": finish_reason or "stop", "usage": usage}
            else:
                raw = response.read(2 * 1024 * 1024 + 1)
                control.check()
                if len(raw) > 2 * 1024 * 1024:
                    raise AssistantError("upstream_protocol", "The model response exceeded the size limit", 502)
                event = self._json(raw)
                try:
                    text = event["choices"][0]["message"]["content"]
                    if not isinstance(text, str) or not text:
                        if event["choices"][0].get("finish_reason") == "length":
                            raise AssistantError("output_budget", "The model output budget was exhausted with no body; raise the output limit or explicitly choose another model", 502)
                        raise ValueError
                    reason = event["choices"][0].get("finish_reason", "stop")
                except (KeyError, IndexError, TypeError, ValueError):
                    raise AssistantError("upstream_protocol", "The model did not return a valid text answer", 502) from None
                yield "delta", {"text": text}
                yield "done", {"finish_reason": str(reason)[:80], "usage": self._usage(event.get("usage"))}
            control.check()
            if time.monotonic() - started > config.timeout_seconds:
                raise AssistantError("timeout", "The model service exceeded the total time limit", 504)
        except AssistantError:
            control.check()
            raise
        except TimeoutError:
            control.check()
            raise AssistantError("response_timeout", "Connecting or waiting for the model response timed out; check the service. No automatic retry", 504) from None
        except (OSError, http.client.HTTPException, ValueError):
            control.check()
            raise AssistantError("network", "Model connection, certificate, or response reading failed; check the service. No automatic retry", 502) from None
        finally:
            if response:
                response.close()
            control.close()

    @staticmethod
    def _json(value: str | bytes) -> dict:
        try:
            result = json.loads(value)
        except (ValueError, UnicodeDecodeError):
            raise AssistantError("upstream_protocol", "The model service returned invalid JSON", 502) from None
        if not isinstance(result, dict):
            raise AssistantError("upstream_protocol", "The model response must be a JSON object", 502)
        return result

    @staticmethod
    def _usage(value) -> dict | None:
        if not isinstance(value, dict):
            return None
        result = {key: value[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(value.get(key), int) and not isinstance(value[key], bool) and value[key] >= 0}
        return result or None
