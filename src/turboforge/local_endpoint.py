"""Default local model endpoint derived from the assistant's allowlist.

Users configure the port in one place: TURBOFORGE_AI_ALLOWED_LOCAL_ORIGINS in
.env or the environment. The first listed origin's explicit port becomes the
default serving port, so the page default, plan preview, generated launcher and
benchmark stay on the same port when the user moves it.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from .assistant.config import ConfigStore
from .assistant.errors import AssistantError

# Only used when the allowlist is empty, malformed, or has no explicit port.
FALLBACK_SERVING_PORT = 18080


def default_serving_port() -> int:
    """Port from the first allowlisted local origin, else the fallback."""
    try:
        origins = ConfigStore().allowed_local_origins
    except (AssistantError, OSError, ValueError, UnicodeError):
        return FALLBACK_SERVING_PORT
    for value in origins:
        try:
            port = urlsplit(value).port
        except ValueError:
            continue
        if port:
            return port
    return FALLBACK_SERVING_PORT
