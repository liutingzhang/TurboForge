"""Flask application: JSON API + the built single-page frontend on one origin.

Serving the frontend and the API from the same origin removes CORS and any
frontend/backend address drift. In development, Vite proxies /api here; in
production the built web/dist is served directly.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from ..backends.registry import recipes_for
from ..domain import ForgeConfig
from ..inspector import ModelInspector
from ..planner import ForgePlanner
from .builds import BuildManager
from .meta import _recipe_dict, build_meta


def _dist_dir() -> Path:
    override = os.environ.get("TURBOFORGE_WEB_DIST")
    if override:
        return Path(override)
    # src/turboforge/api/server.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3] / "web" / "dist"


_BUILD_NOTICE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TurboForge</title>
<style>
:root{color-scheme:dark}
body{margin:0;min-height:100vh;display:grid;place-items:center;
  font-family:system-ui,"Segoe UI",sans-serif;color:#e6edf3;
  background:radial-gradient(1200px 600px at 20% -10%,#2a1c0e 0%,transparent 60%),
             radial-gradient(900px 500px at 100% 0%,#0e2233 0%,transparent 55%),#0e1116}
.card{max-width:640px;padding:40px;border:1px solid #2b3844;border-radius:16px;
  background:#161b22cc;box-shadow:0 20px 60px #0008}
h1{font-size:22px;margin:0 0 8px}
.spark{color:#ff9838}
p{color:#9aa7b4;line-height:1.7}
code{display:block;background:#0b0f13;border:1px solid #223041;border-radius:8px;
  padding:14px 16px;margin:12px 0;font-family:ui-monospace,Consolas,monospace;
  font-size:13px;color:#7ee787;white-space:pre}
a{color:#38bdf8}
</style></head><body><div class="card">
<h1><span class="spark">&#9670;</span> TurboForge</h1>
<p>The frontend has not been built yet. The API is ready; build the UI to use the full GUI:</p>
<code>cd web
npm install
npm run build</code>
<p>Refresh this page after the build. For development run <code style="display:inline;padding:2px 6px">npm run dev</code>
(Vite proxies /api to this service).</p>
</div></body></html>"""


def create_app(planner: ForgePlanner | None = None, *, assistant_root: Path | None = None):
    try:
        from flask import Flask, jsonify, request, send_from_directory
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("the webui needs flask: pip install flask") from exc

    app = Flask(__name__, static_folder=None)
    app.config["JSON_AS_ASCII"] = False
    planner = planner or ForgePlanner()
    manager = BuildManager(planner)
    app.extensions["turboforge_build_manager"] = manager
    from .assistant import loopback, register_assistant
    from urllib.parse import urlsplit

    @app.before_request
    def local_request_boundary():
        # Loopback binding alone does not prevent DNS rebinding or cross-site
        # requests from another website open in the same browser.
        if not loopback(urlsplit(request.host_url).hostname or "") or not loopback(request.remote_addr or ""):
            return jsonify({"error": "This tool is local-only; use the localhost address shown in the launcher window"}), 403
        if request.path.startswith("/api/"):
            incoming = request.headers.get("Origin")
            if request.headers.get("Sec-Fetch-Site") == "cross-site" or incoming and incoming != request.host_url.rstrip("/"):
                return jsonify({"error": "Cross-site operation rejected; retry from the local page"}), 403
            if request.method in ("POST", "PUT", "PATCH") and request.content_length and request.mimetype != "application/json":
                return jsonify({"error": "Operation parameters must use JSON"}), 415

    register_assistant(app, manager, assistant_root)
    inspector = ModelInspector()

    def _config_error(exc: Exception):
        return jsonify({"error": f"invalid configuration: {exc}"}), 400

    # -- API -----------------------------------------------------------------

    @app.get("/api/meta")
    def api_meta():
        return jsonify(build_meta())

    @app.post("/api/auto-build")
    def api_auto_build():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"].strip():
            return jsonify({"error": "Provide a model path or Hugging Face model name"}), 400
        options = body.get("options", {})
        if not isinstance(options, dict):
            return jsonify({"error": "options must be an object"}), 400
        try:
            return jsonify({"build_id": manager.start(None, model=body["model"].strip(), options=options)})
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/pick-path")
    def api_pick_path():
        from .native import pick_path
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error":"Select a file or folder"}), 400
        try:
            return jsonify({"path": pick_path(body.get("kind"))})
        except ValueError as exc:
            return _config_error(exc)
        except (RuntimeError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/servers", methods=["GET", "DELETE"])
    def api_servers():
        if request.method == "DELETE":
            manager.stop_servers()
        return jsonify({"servers":manager.active_servers()})

    @app.route("/api/build/<build_id>/server", methods=["GET", "POST", "DELETE"])
    def api_model_server(build_id: str):
        try:
            if request.method == "POST":
                return jsonify(manager.launch(build_id))
            if request.method == "DELETE":
                return jsonify(manager.stop(build_id))
            return jsonify(manager.server_status(build_id))
        except (RuntimeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/precheck")
    def api_precheck():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"].strip():
            return jsonify({"error": "Select a model first"}), 400
        from ..precheck import precheck
        original = body.get("original_model")
        if original is not None and (not isinstance(original, str) or not original.strip()):
            return jsonify({"error":"The original source must be a non-empty path or repository link"}), 400
        return jsonify(precheck(body["model"].strip(), original))

    @app.post("/api/inspect")
    def api_inspect():
        body = request.get_json(force=True, silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error":"Model inspection parameters must be an object"}), 400
        input_path = body.get("input_path")
        if not isinstance(input_path, str) or not input_path.strip():
            return jsonify({"error": "input_path is required"}), 400
        try:
            inspection = inspector.inspect(input_path, metadata_only=body.get("metadata_only") is True)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            recipes = planner.available_recipes(inspection)
        except ValueError as exc:
            return _config_error(exc)
        return jsonify({
            "inspection": inspection.to_dict(),
            "recipes": [
                {
                    "recipe": r.recipe, "label": r.label,
                    "nominal_bits": r.nominal_bits, "typical_bpw": r.typical_bpw,
                    "extra_sources": list(r.extra_sources), "needs_gpu": r.needs_gpu,
                }
                for r in recipes
            ],
        })

    @app.post("/api/plan")
    def api_plan():
        try:
            config = ForgeConfig.from_dict(request.get_json(force=True))
        except (ValueError, KeyError, TypeError) as exc:
            return _config_error(exc)
        plan = planner.plan(config)
        capabilities = planner.capabilities(plan.inspection, config)
        # Same geometry gate the plan uses, so the target dropdown never offers a
        # recipe the planner would refuse (e.g. native 27B kernels on a 0.8B model).
        recipes = [_recipe_dict(r) for r in planner.available_recipes(plan.inspection)]
        return jsonify({
            "plan": plan.to_dict(),
            "capabilities": [asdict(c) for c in capabilities],
            "recipes": recipes,
        })

    @app.post("/api/build")
    def api_build():
        try:
            config = ForgeConfig.from_dict(request.get_json(force=True))
        except (ValueError, KeyError, TypeError) as exc:
            return _config_error(exc)
        try:
            build_id = manager.start(config)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"build_id": build_id})

    @app.get("/api/build/<build_id>")
    def api_status(build_id: str):
        record = manager.status(build_id)
        if record is None:
            return jsonify({"error": "unknown build id"}), 404
        return jsonify(record)

    @app.get("/api/build/<build_id>/log")
    def api_log(build_id: str):
        log = manager.log(build_id)
        if log is None:
            return jsonify({"error": "unknown build id"}), 404
        return jsonify({"log": log})

    @app.get("/api/build/<build_id>/report")
    def api_report(build_id: str):
        name = request.args.get("name", "performance")
        if manager.status(build_id) is None:
            return jsonify({"error": "unknown build id"}), 404
        try:
            report = manager.report(build_id, name)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        # Absence is a normal state (failed build / metric not run), not an error;
        # 200 + null keeps the browser console free of expected-404 noise.
        return jsonify({"report": report})

    # -- frontend ------------------------------------------------------------

    dist = _dist_dir()
    if dist.is_dir():
        @app.get("/")
        def index():
            return send_from_directory(dist, "index.html")

        @app.get("/<path:filename>")
        def assets(filename: str):
            from ..distribution import private_path
            if private_path(filename):
                return "Not found", 404
            target = (dist / filename).resolve()
            if dist.resolve() not in target.parents:
                return "Not found", 404
            if target.is_file():
                return send_from_directory(dist, filename)
            return send_from_directory(dist, "index.html")
    else:
        @app.get("/")
        def index():
            return _BUILD_NOTICE

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turboforge webui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    app = create_app()
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True, load_dotenv=False)
    finally:
        app.extensions["turboforge_build_manager"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
