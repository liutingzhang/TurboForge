"""Same-origin optional assistant API. Existing conversion endpoints stay independent."""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit

from flask import Blueprint, Response, g, jsonify, request, stream_with_context

from ..assistant.config import ConfigStore, ConnectionConfig, ROOT, from_payload
from ..assistant.context import business_snapshot, redact, validate_chat
from ..assistant.errors import AssistantError
from ..assistant.retrieval import KnowledgeIndex
from ..assistant.service import AssistantService
from ..assistant.sessions import SessionStore

COOKIE = "turboforge_assistant"


def loopback(value: str) -> bool:
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def register_assistant(app, manager, root: Path | None = None):
    blueprint = Blueprint("assistant", __name__, url_prefix="/api/assistant")
    sessions = SessionStore()
    knowledge = KnowledgeIndex()
    service = None
    init_error = None
    try:
        config_store = ConfigStore(root or ROOT)
        service = AssistantService(config_store, manager)
        service.sessions = sessions
    except AssistantError as exc:
        init_error = exc
    app.extensions["turboforge_assistant"] = service
    app.extensions["turboforge_assistant_sessions"] = sessions

    def ready():
        if service is None:
            raise init_error or AssistantError("unavailable", "Assistant initialization failed; check the configuration", 503)
        return service

    @blueprint.route("/storage", methods=["GET", "DELETE"])
    def storage():
        from ..bootstrap import cache_root
        from ..storage import inventory
        if request.method == "GET":
            return jsonify(inventory(cache_root()))
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) != {"ids"} or not isinstance(body["ids"], list):
            return jsonify({"error": "Select the failed builds to delete"}), 400
        try:
            return jsonify(manager.clean_failed_outputs(body["ids"]))
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.before_request
    def establish_session():
        if not request.path.startswith("/api/"):
            return None
        is_assistant = request.path.startswith("/api/assistant")
        if not is_assistant and not (request.method == "POST" and request.path in ("/api/build", "/api/auto-build")):
            return None
        host = urlsplit(request.host_url).hostname or ""
        if is_assistant:
            if not loopback(host) or not loopback(request.remote_addr or ""):
                return jsonify({"error": "This first assistant release is local-only; public deployment needs full authentication and permission work", "code": "local_only"}), 403
            if request.headers.get("Sec-Fetch-Site") == "cross-site":
                return jsonify({"error": "Cross-site assistant request rejected", "code": "origin"}), 403
            incoming_origin = request.headers.get("Origin")
            allowed = {request.host_url.rstrip("/")}
            # Explicit opt-in development origin; no CORS response headers are emitted.
            dev_origin = app.config.get("ASSISTANT_DEV_ORIGIN")
            if dev_origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
                allowed.add(dev_origin)
            if incoming_origin is not None and incoming_origin not in allowed:
                return jsonify({"error": "Assistant request origin is not allowed", "code": "origin"}), 403
        try:
            session = sessions.get(request.cookies.get(COOKIE), service.config_store.default if service else ConnectionConfig())
        except AssistantError as exc:
            if is_assistant:
                return jsonify({"error": str(exc), "code": exc.code}), exc.status
            return None
        g.assistant_session = session
        if is_assistant and request.method not in ("GET", "HEAD", "OPTIONS"):
            token = request.headers.get("X-Assistant-CSRF", "")
            if not incoming_origin or not secrets.compare_digest(token, session.csrf):
                return jsonify({"error": "Session expired or missing same-origin CSRF check; refresh the assistant", "code": "csrf"}), 403
            if request.method != "DELETE" and request.mimetype != "application/json":
                return jsonify({"error": "Assistant write requests must use JSON", "code": "content_type"}), 415
        return None

    @app.after_request
    def finish_session(response):
        session = getattr(g, "assistant_session", None)
        if session is not None:
            if request.path in ("/api/build", "/api/auto-build") and request.method == "POST" and response.status_code == 200:
                body = response.get_json(silent=True) or {}
                if isinstance(body.get("build_id"), str):
                    with sessions.lock:
                        session.builds.add(body["build_id"])
            if request.cookies.get(COOKIE) != session.id:
                response.set_cookie(COOKIE, session.id, httponly=True, samesite="Strict", secure=request.is_secure, path="/api")
        if request.path.startswith("/api/assistant"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.errorhandler(AssistantError)
    def public_error(exc):
        return jsonify({"error": str(exc), "code": exc.code}), exc.status

    def body():
        raw = request.stream.read(65537)
        if len(raw) > 65536:
            raise AssistantError("request_size", "Assistant request exceeds 64KiB; shrink the content", 413)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise AssistantError("request_json", "The request is not valid JSON") from None
        if not isinstance(value, dict):
            raise AssistantError("request_json", "The request must be a JSON object")
        return value

    @blueprint.get("/config")
    def configuration():
        session = g.assistant_session
        config = session.config.public()
        config.update({"csrf_token": session.csrf, "allow_persist": bool(service and service.config_store.allow_persist),
            "config_error": str(init_error) if init_error else None})
        return jsonify(config)

    @blueprint.post("/config/test")
    def test_configuration():
        current = ready()
        config = from_payload(body(), g.assistant_session.config)
        result = current.test_connection(g.assistant_session, config)
        token = secrets.token_urlsafe(24)
        with sessions.lock:
            g.assistant_session.tested_config = (token, config, time.monotonic())
        return jsonify({**result, "tested_config_id": token})

    @blueprint.put("/config")
    def update_configuration():
        current = ready()
        value = body()
        if "tested_config_id" in value:
            tested = g.assistant_session.tested_config
            if set(value) - {"tested_config_id", "scope"} or not isinstance(value["tested_config_id"], str) or not tested or not secrets.compare_digest(value["tested_config_id"], tested[0]) or time.monotonic() - tested[2] > 300:
                raise AssistantError("test_expired", "The tested configuration expired or does not belong to this session; test again")
            config = tested[1]
        else:
            config = from_payload(value, g.assistant_session.config)
        scope = value.get("scope", "session")
        if scope not in ("session", "dotenv"):
            raise AssistantError("config_scope", "Configuration scope must be session or dotenv")
        if scope == "dotenv":
            if not current.config_store.allow_persist:
                raise AssistantError("persistence_disabled", "The administrator has not enabled local .env saving; use the session configuration instead", 403)
            current.test_connection(g.assistant_session, config)
            config = current.config_store.persist(config)
            current.client.allowed_local = current.config_store.allowed_local_origins
            with sessions.lock:
                g.assistant_session.default = config
                g.assistant_session.override = None
        else:
            with sessions.lock:
                g.assistant_session.override = config
        with sessions.lock:
            g.assistant_session.tested_config = None
        return jsonify({**config.public(), "saved_scope": scope,
            "notice": "File saved; current environment variables still take priority" if scope == "dotenv" and config.source == "environment" else None})

    @blueprint.delete("/config/session")
    def clear_configuration():
        current = ready()
        with sessions.lock:
            g.assistant_session.default = current.config_store.default
            g.assistant_session.override = None
        return jsonify(g.assistant_session.config.public())

    @blueprint.get("/help")
    def help_search():
        query = request.args.get("q", "")
        topic = request.args.get("topic", "")
        if len(query) > 2000 or len(topic) > 100:
            raise AssistantError("search_size", "Search content is too long")
        limit = 30 if query else 500
        entries = knowledge.search(query, topic=topic, limit=limit)
        return jsonify({"entries": entries, "total": len(entries)})

    @blueprint.get("/help/<identifier>")
    def help_entry(identifier):
        entry = knowledge.by_id.get(identifier)
        if entry is None:
            raise AssistantError("help_missing", "No such help entry", 404)
        return jsonify(entry)

    @blueprint.post("/chat")
    def chat():
        current = ready()
        session = g.assistant_session
        _, _, events = current.answer(session, body())
        if session.config.stream:
            def stream():
                try:
                    for event, data in events:
                        yield "event: " + event + "\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                finally:
                    events.close()
            return Response(stream_with_context(stream()), mimetype="text/event-stream",
                headers={"X-Accel-Buffering": "no"})
        result = {"text": ""}
        for event, data in events:
            if event == "meta":
                result.update(data)
            elif event == "delta":
                result["text"] += data["text"]
            elif event == "error":
                return jsonify(data), data["http_status"]
            elif event == "done":
                result.update(data)
        return jsonify(result)

    @blueprint.post("/context/preview")
    def preview():
        value = validate_chat(body())
        session = g.assistant_session
        secret_values = (session.config.api_key, session.default.api_key)
        return jsonify({"question": redact(value["question"], secret_values),
            "selection": [{k: redact(v, secret_values) for k, v in block.items()} for block in value.get("selection", [])],
            "evidence": business_snapshot(manager, session, value)})

    @blueprint.post("/chat/<identifier>/cancel")
    def cancel(identifier):
        ready().sessions.cancel(g.assistant_session, identifier)
        return jsonify({"cancelled": True, "notice": "Stopped reading; the upstream may still bill"})

    @blueprint.route("/conversations/<identifier>", methods=["GET", "DELETE"])
    def conversation(identifier):
        session = g.assistant_session
        with sessions.lock:
            entry = session.conversations.get(identifier)
            if entry is None:
                raise AssistantError("conversation_missing", "Conversation expired or does not belong to this session", 404)
            if request.method == "DELETE":
                if session.requests:
                    raise AssistantError("busy", "Stop the current answer before deleting the conversation", 409)
                del session.conversations[identifier]
                return jsonify({"deleted": True})
            return jsonify({"conversation_id": entry.id, "build_id": entry.boundary[0], "model": entry.boundary[2],
                "turns": entry.messages})

    app.register_blueprint(blueprint)
