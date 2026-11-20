"""Ephemeral opaque sessions, owned conversations and bounded active requests."""
from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import re
from threading import RLock
import time

from .client import RequestControl
from .config import ConnectionConfig
from .errors import AssistantError


@dataclass
class Conversation:
    id: str
    boundary: tuple[str | None, str, str]
    messages: list[dict] = field(default_factory=list)


@dataclass
class Session:
    id: str
    csrf: str
    default: ConnectionConfig
    owner: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    touched: float = field(default_factory=time.monotonic)
    override: ConnectionConfig | None = None
    builds: set[str] = field(default_factory=set)
    conversations: dict[str, Conversation] = field(default_factory=dict)
    requests: dict[str, RequestControl] = field(default_factory=dict)
    cancelled_requests: dict[str, float] = field(default_factory=dict)
    tested_config: tuple[str, ConnectionConfig, float] | None = None

    @property
    def config(self) -> ConnectionConfig:
        return self.override or self.default


class SessionStore:
    def __init__(self):
        self.lock = RLock()
        self.sessions: dict[str, Session] = {}

    def get(self, token: str | None, default: ConnectionConfig, owner: str | None = None) -> Session:
        with self.lock:
            now = time.monotonic()
            stale = [key for key, session in self.sessions.items() if now - session.touched > 1800 and not session.requests]
            for key in stale:
                del self.sessions[key]
            session = self.sessions.get(token or "")
            if session is None:
                if len(self.sessions) >= 256:
                    raise AssistantError("session_limit", "本机助手会话已满，请稍后重试", 429)
                session = Session(secrets.token_urlsafe(32), secrets.token_urlsafe(32), default)
                if owner and re.fullmatch(r"[A-Za-z0-9_-]{43}", owner):
                    session.owner = owner
                self.sessions[session.id] = session
            if session.tested_config is not None and now - session.tested_config[2] > 300:
                session.tested_config = None
            session.touched = now
            return session

    def conversation(self, session: Session, identifier: str | None, build_id: str | None) -> Conversation:
        config = session.config
        boundary = (build_id, config.base_url, config.model)
        with self.lock:
            if identifier is not None:
                conversation = session.conversations.get(identifier)
                if conversation is None:
                    raise AssistantError("conversation_missing", "对话已过期或不属于本会话，请新建对话", 404)
                if conversation.boundary != boundary:
                    raise AssistantError("context_changed", "模型、服务或任务已变化，请新建对话，旧答案仍属于旧任务", 409)
                return conversation
            if len(session.conversations) >= 20:
                del session.conversations[next(iter(session.conversations))]
            conversation = Conversation(secrets.token_urlsafe(18), boundary)
            session.conversations[conversation.id] = conversation
            return conversation

    def acquire(self, session: Session, identifier: str | None = None) -> tuple[str, RequestControl]:
        with self.lock:
            self._expire_cancellations(session)
            if identifier and identifier in session.cancelled_requests:
                del session.cancelled_requests[identifier]
                raise AssistantError("cancelled", "已停止读取回答；上游仍可能计费", 409)
            if session.requests or sum(len(s.requests) for s in self.sessions.values()) >= 4:
                raise AssistantError("busy", "助手正在处理其他请求，请等待或停止当前回答", 429)
            identifier = identifier or secrets.token_urlsafe(18)
            control = RequestControl()
            session.requests[identifier] = control
            return identifier, control

    def release(self, session: Session, identifier: str):
        with self.lock:
            control = session.requests.pop(identifier, None)
            if control:
                control.close()
            session.touched = time.monotonic()

    def cancel(self, session: Session, identifier: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier):
            raise AssistantError("chat_input", "请求标识无效")
        with self.lock:
            control = session.requests.get(identifier)
            if control is None:
                self._expire_cancellations(session)
                if identifier not in session.cancelled_requests and len(session.cancelled_requests) >= 128:
                    raise AssistantError("cancel_limit", "取消请求过多，请稍后重试", 429)
                session.cancelled_requests[identifier] = time.monotonic()
        if control is not None:
            control.interrupt()

    @staticmethod
    def _expire_cancellations(session: Session):
        now = time.monotonic()
        for identifier, created in list(session.cancelled_requests.items()):
            if now - created > 300:
                del session.cancelled_requests[identifier]
