"""Backend-only connection configuration and restricted atomic dotenv storage."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from collections.abc import Mapping
from threading import RLock
from urllib.parse import urlsplit

from .errors import AssistantError

PREFIX = "TURBOFORGE_AI_"
ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {"ENABLED": "true", "BASE_URL": "", "MODEL": "", "API_KEY": "", "AUTH_MODE": "bearer",
    "STREAM": "true", "CONTEXT_TOKENS": "8192", "MAX_OUTPUT_TOKENS": "1024", "TIMEOUT_SECONDS": "90",
    "ALLOW_UI_PERSIST": "false", "ALLOWED_LOCAL_ORIGINS": ""}


def child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    # The AI key is irrelevant to every conversion, installer and model worker.
    return {k: v for k, v in (os.environ if base is None else base).items() if not k.upper().startswith(PREFIX)}


def parse_dotenv(text: str) -> dict[str, str]:
    values = {}
    for line in text.lstrip("\ufeff").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, raw = line.partition("=")
        key = key.strip()
        if not key.startswith(PREFIX):
            continue
        if not sep or key in values or key[len(PREFIX):] not in DEFAULTS:
            raise AssistantError("config_file", "助手 .env 键重复、未知或格式错误，请检查模板")
        raw = raw.strip()
        try:
            if raw.startswith('"'):
                value, offset = json.JSONDecoder().raw_decode(raw)
                if not isinstance(value, str) or (raw[offset:].strip() and not raw[offset:].lstrip().startswith("#")):
                    raise ValueError
            elif raw.startswith("'"):
                match = re.fullmatch(r"'([^']*)'\s*(?:#.*)?", raw)
                if not match:
                    raise ValueError
                value = match[1]
            else:
                value = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
            if "\n" in value or "\r" in value or "\x00" in value:
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            raise AssistantError("config_file", "助手 .env 值格式错误；双引号使用 JSON 转义，不支持多行或变量展开") from None
        values[key] = value
    return values


def boolean(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise AssistantError("config_value", "助手布尔配置只能使用 true 或 false")
    return value.lower() == "true"


def integer(value: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AssistantError("config_value", "助手数字配置必须是整数")
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise AssistantError("config_value", "助手数字配置无效") from None
    if not low <= number <= high:
        raise AssistantError("config_value", f"助手数字配置必须在 {low}–{high} 之间")
    return number


@dataclass(frozen=True)
class ConnectionConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    auth_mode: str = "bearer"
    stream: bool = True
    context_tokens: int = 8192
    max_output_tokens: int = 1024
    timeout_seconds: int = 90
    enabled: bool = True
    source: str = "default"

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model and (self.api_key or self.auth_mode == "none"))

    def public(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "has_api_key": bool(self.api_key),
            "auth_mode": self.auth_mode, "stream": self.stream, "enabled": self.enabled,
            "context_tokens": self.context_tokens, "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds, "configured": self.configured, "source": self.source}


def validate_connection(config: ConnectionConfig, *, required: bool = True) -> ConnectionConfig:
    if any(len(v) > limit or any(c in v for c in "\r\n\x00") for v, limit in
           ((config.base_url, 2048), (config.model, 256), (config.api_key, 8192))):
        raise AssistantError("config_value", "连接配置过长或包含非法控制字符")
    try:
        url = urlsplit(config.base_url)
        url.port
    except ValueError:
        raise AssistantError("base_url", "Base URL 无效") from None
    if config.base_url and (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment or "\\" in config.base_url
            or any(c.isspace() for c in config.base_url)):
        raise AssistantError("base_url", "Base URL 只允许不含凭据、查询参数或片段的 HTTP(S) 接口根地址")
    if url.path.rstrip("/").endswith(("/chat/completions", "/models")):
        raise AssistantError("base_url", "请填写接口根地址（如 /v1），不要填写 /chat/completions 或 /models")
    if config.auth_mode not in ("bearer", "none"):
        raise AssistantError("auth_mode", "鉴权方式必须为 bearer 或本地无 Key")
    if required and not config.configured:
        raise AssistantError("not_configured", "请配置 Base URL、模型名称和 API Key；本机无鉴权服务需明确选择无需 Key")
    if config.auth_mode == "none" and config.api_key:
        raise AssistantError("auth_mode", "无需 Key 模式请明确清除 Key")
    if config.max_output_tokens >= config.context_tokens:
        raise AssistantError("context_budget", "输出 token 上限必须小于上下文窗口")
    return config


def from_payload(payload: dict, current: ConnectionConfig) -> ConnectionConfig:
    allowed = {"base_url", "model", "api_key", "secret_action", "auth_mode", "stream", "context_tokens",
        "max_output_tokens", "timeout_seconds", "scope"}
    if set(payload) - allowed:
        raise AssistantError("config_value", "包含未知连接配置字段")
    for key in ("base_url", "model", "auth_mode"):
        if not isinstance(payload.get(key), str):
            raise AssistantError("config_value", "请完整填写 Base URL、模型名和鉴权方式")
    base_url = payload["base_url"].strip().rstrip("/")
    validate_connection(replace(current, base_url=base_url), required=False)
    action = payload.get("secret_action")
    if action == "replace":
        secret = payload.get("api_key")
        if not isinstance(secret, str) or not secret.strip():
            raise AssistantError("config_value", "替换 Key 时请输入新的 API Key")
        secret = secret.strip()
    elif action == "clear":
        secret = ""
    elif action == "keep":
        if urlsplit(base_url).netloc.lower() != urlsplit(current.base_url).netloc.lower() or urlsplit(base_url).scheme != urlsplit(current.base_url).scheme:
            raise AssistantError("key_origin", "更换服务 origin 后不能沿用原 Key，请重新输入或明确清除")
        secret = current.api_key
    else:
        raise AssistantError("config_value", "请明确选择保留、替换或清除 Key")
    if action != "replace" and "api_key" in payload:
        raise AssistantError("config_value", "仅替换操作可以携带 Key")
    stream = payload.get("stream", current.stream)
    if not isinstance(stream, bool):
        raise AssistantError("config_value", "stream 必须是布尔值")
    config = replace(current, base_url=base_url, model=payload["model"].strip(), api_key=secret,
        auth_mode=payload["auth_mode"], stream=stream, source="session",
        context_tokens=integer(payload.get("context_tokens", current.context_tokens), 256, 1048576),
        max_output_tokens=integer(payload.get("max_output_tokens", current.max_output_tokens), 16, 32768),
        timeout_seconds=integer(payload.get("timeout_seconds", current.timeout_seconds), 5, 300))
    return validate_connection(config)


def restrict_file(path: Path) -> None:
    if os.name == "nt":
        import csv
        result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True, env=child_environment())
        sid = next(csv.reader([result.stdout.strip()]))[-1]
        if not re.fullmatch(r"S-1-[0-9-]+", sid):
            raise OSError("Cannot resolve current user SID")
        for args in (("/reset",), ("/inheritance:r", "/grant:r", f"*{sid}:F", "*S-1-5-18:F")):
            subprocess.run(["icacls", str(path), *args], capture_output=True, check=True, env=child_environment())
    else:
        path.chmod(0o600)


class ConfigStore:
    def __init__(self, root: Path = ROOT, environ: dict[str, str] | None = None):
        self.path = root / ".env"
        self.environ = dict(os.environ if environ is None else environ)
        self.lock = RLock()
        self.reload()

    def reload(self) -> None:
        try:
            text = self.path.read_text(encoding="utf-8-sig") if self.path.is_file() else ""
        except (OSError, UnicodeError):
            raise AssistantError("config_file", "无法读取助手 .env；业务功能不受影响") from None
        self.file_values = parse_dotenv(text)
        values = {PREFIX + k: v for k, v in DEFAULTS.items()}
        values.update(self.file_values)
        values.update({k: v for k, v in self.environ.items() if k.startswith(PREFIX)})
        v = lambda name: values[PREFIX + name]
        source = "environment" if any(PREFIX + k in self.environ for k in ("BASE_URL", "MODEL", "API_KEY")) else "dotenv" if self.file_values else "default"
        self.default = validate_connection(ConnectionConfig(v("BASE_URL").strip().rstrip("/"), v("MODEL").strip(), v("API_KEY").strip(),
            v("AUTH_MODE"), boolean(v("STREAM")), integer(v("CONTEXT_TOKENS"), 256, 1048576),
            integer(v("MAX_OUTPUT_TOKENS"), 16, 32768), integer(v("TIMEOUT_SECONDS"), 5, 300), boolean(v("ENABLED")), source), required=False)
        self.allow_persist = boolean(v("ALLOW_UI_PERSIST"))
        self.allowed_local_origins = tuple(x.strip().rstrip("/") for x in v("ALLOWED_LOCAL_ORIGINS").split(",") if x.strip())

    def persist(self, config: ConnectionConfig) -> ConnectionConfig:
        if not self.allow_persist:
            raise AssistantError("persistence_disabled", "管理员未开启本机 .env 保存；可使用本次会话配置", 403)
        with self.lock:
            path = None
            try:
                old = self.path.read_text(encoding="utf-8-sig") if self.path.exists() else ""
                parse_dotenv(old)
                mapping = {PREFIX + k.upper(): str(v).lower() if isinstance(v, bool) else str(v)
                    for k, v in {"base_url": config.base_url, "model": config.model, "api_key": config.api_key,
                        "auth_mode": config.auth_mode, "stream": config.stream, "context_tokens": config.context_tokens,
                        "max_output_tokens": config.max_output_tokens, "timeout_seconds": config.timeout_seconds}.items()}
                kept = [line for line in old.splitlines() if line.partition("=")[0].strip() not in mapping]
                content = "\n".join(kept + [k + "=" + json.dumps(v, ensure_ascii=False) for k, v in mapping.items()]) + "\n"
                fd, name = tempfile.mkstemp(prefix=".assistant-config-", dir=self.path.parent)
                os.close(fd)
                path = Path(name)
                restrict_file(path)  # No secret bytes are written before permissions are restricted.
                with path.open("w", encoding="utf-8", newline="\n") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(path, self.path)
                self.reload()
                return self.default
            except (OSError, subprocess.SubprocessError):
                raise AssistantError("config_write", "无法安全保存 .env，请检查权限/空间；原配置保留，可继续临时模式") from None
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
