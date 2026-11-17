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
            raise AssistantError("config_file", "Assistant .env keys are duplicated, unknown, or malformed; check the template")
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
            raise AssistantError("config_file", "Assistant .env value is malformed; double quotes use JSON escaping, and multiline or variable expansion is not supported") from None
        values[key] = value
    return values


def boolean(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise AssistantError("config_value", "Assistant boolean settings only accept true or false")
    return value.lower() == "true"


def integer(value: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AssistantError("config_value", "Assistant numeric settings must be integers")
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise AssistantError("config_value", "Invalid assistant numeric setting") from None
    if not low <= number <= high:
        raise AssistantError("config_value", f"Assistant numeric settings must be between {low} and {high}")
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
        raise AssistantError("config_value", "Connection settings are too long or contain illegal control characters")
    try:
        url = urlsplit(config.base_url)
        url.port
    except ValueError:
        raise AssistantError("base_url", "Invalid Base URL") from None
    if config.base_url and (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment or "\\" in config.base_url
            or any(c.isspace() for c in config.base_url)):
        raise AssistantError("base_url", "Base URL must be an HTTP(S) API root without credentials, query parameters, or fragments")
    if url.path.rstrip("/").endswith(("/chat/completions", "/models")):
        raise AssistantError("base_url", "Enter the API root (e.g. /v1), not /chat/completions or /models")
    if config.auth_mode not in ("bearer", "none"):
        raise AssistantError("auth_mode", "Auth mode must be bearer or local keyless")
    if required and not config.configured:
        raise AssistantError("not_configured", "Configure Base URL, model name, and API Key; for a local service without auth, explicitly choose keyless")
    if config.auth_mode == "none" and config.api_key:
        raise AssistantError("auth_mode", "In keyless mode, explicitly clear the Key")
    if config.max_output_tokens >= config.context_tokens:
        raise AssistantError("context_budget", "The output token limit must be smaller than the context window")
    return config


def from_payload(payload: dict, current: ConnectionConfig) -> ConnectionConfig:
    allowed = {"base_url", "model", "api_key", "secret_action", "auth_mode", "stream", "context_tokens",
        "max_output_tokens", "timeout_seconds", "scope"}
    if set(payload) - allowed:
        raise AssistantError("config_value", "Unknown connection setting fields")
    for key in ("base_url", "model", "auth_mode"):
        if not isinstance(payload.get(key), str):
            raise AssistantError("config_value", "Fill in Base URL, model name, and auth mode completely")
    base_url = payload["base_url"].strip().rstrip("/")
    validate_connection(replace(current, base_url=base_url), required=False)
    action = payload.get("secret_action")
    if action == "replace":
        secret = payload.get("api_key")
        if not isinstance(secret, str) or not secret.strip():
            raise AssistantError("config_value", "Enter the new API Key when replacing the Key")
        secret = secret.strip()
    elif action == "clear":
        secret = ""
    elif action == "keep":
        if urlsplit(base_url).netloc.lower() != urlsplit(current.base_url).netloc.lower() or urlsplit(base_url).scheme != urlsplit(current.base_url).scheme:
            raise AssistantError("key_origin", "After changing the service origin the old Key cannot be reused; re-enter it or explicitly clear it")
        secret = current.api_key
    else:
        raise AssistantError("config_value", "Explicitly choose to keep, replace, or clear the Key")
    if action != "replace" and "api_key" in payload:
        raise AssistantError("config_value", "Only the replace action may carry a Key")
    stream = payload.get("stream", current.stream)
    if not isinstance(stream, bool):
        raise AssistantError("config_value", "stream must be a boolean")
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
            raise AssistantError("config_file", "Cannot read the assistant .env; core features are unaffected") from None
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
            raise AssistantError("persistence_disabled", "The administrator has not enabled local .env saving; use the session configuration instead", 403)
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
                raise AssistantError("config_write", "Cannot safely save .env; check permissions/disk space. The previous configuration is kept and session mode still works") from None
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
