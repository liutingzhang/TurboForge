"""Background build execution for the web API.

The runner already writes manifest.json atomically on every transition and
streams the converter log to logs/convert.log, so live progress is read straight
from those files -- no change to the build pipeline is needed. Only one build
runs at a time to avoid GPU/disk contention.
"""

from __future__ import annotations

from ..assistant.config import child_environment
import json
import re
import threading
import shlex
import socket
import subprocess
import sys
import os
import urllib.request
from pathlib import Path
from typing import Any

from ..domain import ForgeConfig
from ..local_endpoint import default_serving_port
from ..runner import BuildRunner
from ..manifest import read_json_atomic, write_json_atomic


class BuildManager:
    def __init__(self, planner=None) -> None:
        self.planner = planner
        self._builds: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._servers: dict[str, subprocess.Popen] = {}
        self._server_job = None

    def is_running(self) -> bool:
        with self._lock:
            return any(not b["finished"] for b in self._builds.values())

    def clean_failed_outputs(self, identifiers: list[str]) -> dict:
        from ..bootstrap import cache_root
        from ..storage import remove_failed
        with self._lock:
            if any(not b["finished"] for b in self._builds.values()) or any(p.poll() is None for p in self._servers.values()):
                raise ValueError("Wait for the conversion to finish and stop the model server before cleaning failed artifacts")
            return remove_failed(cache_root(), identifiers)

    def start(self, config: ForgeConfig | None, *, model: str | None = None, options: dict | None = None) -> str:
        import uuid

        build_id = uuid.uuid4().hex
        record = {
            "build_id": build_id,
            "finished": False,
            "status": "starting",
            "output_dir": str(Path(config.output_dir).expanduser().resolve()) if config else "",
            "setup_log": [],
            "error": None,
        }
        with self._lock:
            if any(not b["finished"] for b in self._builds.values()):
                raise RuntimeError("A build is already running; wait for it to finish")
            if any(p.poll() is None for p in self._servers.values()):
                raise RuntimeError("Stop the model server started from this page to free the GPU before converting")
            self._builds[build_id] = record

        try:
            self._persist(record)
        except OSError as exc:
            with self._lock:
                self._builds.pop(build_id, None)
            raise RuntimeError("Cannot write the build record; check cache disk space and directory permissions") from exc

        def work() -> None:
            try:
                effective = config
                if model is not None:
                    self._auto_work(record, model, options or {})
                else:
                    manifest = BuildRunner(planner=self.planner).build(effective)
                    with self._lock:
                        record["status"] = manifest.data["status"]
            except Exception as exc:  # record, never crash the worker
                with self._lock:
                    record["status"] = "crashed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    record["finished"] = True
                self._persist(record)

        threading.Thread(target=work, daemon=True).start()
        return build_id

    def _auto_work(self, record: dict, model: str, options: dict) -> None:
        from ..bootstrap import cache_root
        from ..manifest import write_json_atomic
        root = cache_root()
        job = root / "jobs" / record["build_id"]
        job.mkdir(parents=True, exist_ok=True)
        request, result = job / "request.json", job / "result.json"
        write_json_atomic(request, {"model": model, "options": options})
        def emit(message: str) -> None:
            with self._lock:
                record["setup_log"] = [*record["setup_log"][-199:], message]
                if not record["output_dir"]:
                    record["status"] = "preparing"
        worker = root / ("worker-env-py31318" if os.name == "nt" else "worker-env")
        python = worker / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python.is_file():
            emit("Creating a dedicated conversion environment automatically; system Python packages are not modified")
            from ..bootstrap import run
            from ..backends.hf_environment import bootstrap_python
            run([str(bootstrap_python(root,emit)), "-m", "venv", str(worker)], emit)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONIOENCODING"] = "utf-8"
        with (job / "preparation.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen([str(python), "-m", "turboforge.auto_worker", str(request), str(result)],
                                  env=child_environment(environment), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    if line.startswith("@@CONFIG@@"):
                        prepared = json.loads(line[len("@@CONFIG@@"):])
                        with self._lock:
                            record["prepared_config"] = prepared
                            record["output_dir"] = prepared["output_dir"]
                        self._persist(record)
                    else:
                        log.write(line); log.flush(); emit(line.rstrip())
                code = process.wait()
        outcome = read_json_atomic(result) if result.is_file() else {}
        if code or "error" in outcome:
            raise RuntimeError(outcome.get("error", f"The automatic preparation process exited with {code}; check the preparation log"))
        with self._lock:
            record["status"] = outcome["status"]
            record["output_dir"] = outcome["output_dir"]

    def launch(self, build_id: str) -> dict:
        from ..runner import DEFAULT_NINFER_BIN
        record = self.status(build_id)
        if not record or not record["finished"] or record["status"] != "validated":
            raise RuntimeError("Only builds that passed GPU validation can be started")
        manifest = record["manifest"]
        if not manifest or manifest.get("status") != "validated":
            raise RuntimeError("The build record is missing or incomplete; check the output directory and rebuild")
        config = manifest["config"]
        profile = manifest["plan"]["serving"]
        port = profile["port"]
        engine_dir = Path(config.get("ninfer_dir") or DEFAULT_NINFER_BIN)
        hf = profile.get("engine") == "gptqmodel"
        if hf:
            from ..backends.hf_quantization import server_command
            command = server_command(config,Path(record["output_dir"]),profile)
            engine_dir = Path(record["output_dir"])
        else:
            artifact = manifest["artifacts"]["ninfer_artifact"]["path"]
            if not Path(artifact).is_absolute():
                artifact = str(Path(record["output_dir"]) / artifact)
            if not Path(artifact).is_file():
                raise RuntimeError("The model artifact was moved or deleted; rebuild")
            retained=next((step.get("details",{}).get("components","") for step in manifest.get("plan",{}).get("steps",[]) if step.get("name") == "quantize+compile"),"")
            expected=[name for name in ("mtp","dflash2") if name in (retained or "").split(",") or profile.get("spec") == name]
            if expected:
                from ..speculative import verify_ninfer
                verify_ninfer(Path(artifact),expected)
            command = [str(engine_dir / "ninfer-serve.exe"), artifact,
                       "--max-context", str(profile["max_context"]), "--kv-capacity", str(profile["max_context"]),
                       "--max-concurrency", str(profile["max_concurrency"]),
                       "--host", "127.0.0.1", "--port", str(port)]
            command += [part for flag in profile["launcher_flags"] for part in shlex.split(flag)]
        with self._lock:
            if any(not build["finished"] for build in self._builds.values()):
                raise RuntimeError("A conversion is running; wait for it to finish before starting the model server")
            if any(p.poll() is None for p in self._servers.values()):
                raise RuntimeError("A model server from this page is already running; stop it first")
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise RuntimeError(f"Port {port} is in use; stop the occupying service or change the serving port and rebuild") from exc
            if self._server_job is None:
                from ..process_owner import ServerJob
                self._server_job = ServerJob()
            with (Path(record["output_dir"]) / "logs" / "serve.log").open("w", encoding="utf-8") as log:
                environment = dict(os.environ)
                if hf:
                    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + environment.get("PYTHONPATH","")
                process = subprocess.Popen(command, cwd=engine_dir, stdout=log, stderr=subprocess.STDOUT,env=child_environment(environment))
                try:
                    self._server_job.attach(process.pid)
                except OSError:
                    process.terminate();process.wait(timeout=5)
                    raise
                self._servers[build_id] = process
        return self.server_status(build_id)

    def active_servers(self) -> list[dict]:
        with self._lock:
            active = [build_id for build_id,process in self._servers.items() if process.poll() is None]
        return [{"build_id":build_id,**self.server_status(build_id)} for build_id in active]

    def stop_servers(self) -> None:
        with self._lock:
            active = list(self._servers)
        for build_id in active:
            self.stop(build_id)

    def server_status(self, build_id: str) -> dict:
        with self._lock:
            process = self._servers.get(build_id)
        record = self._record(build_id)
        if record is None:
            raise ValueError("unknown build id")
        manifest = self._read_manifest(record["output_dir"])
        port = (manifest or {}).get("plan", {}).get("serving", {}).get("port") or default_serving_port()
        base_url = f"http://127.0.0.1:{port}/v1"
        status = "stopped"
        if process is not None:
            status = "starting" if process.poll() is None else "exited"
            if status == "starting":
                try:
                    with urllib.request.urlopen(base_url + "/models", timeout=1):
                        status = "ready"
                except OSError:
                    pass
        return {"status": status, "base_url": base_url,
                "log_path": str(Path(record["output_dir"]) / "logs" / "serve.log")}

    def stop(self, build_id: str) -> dict:
        with self._lock:
            process = self._servers.pop(build_id, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return self.server_status(build_id)

    def close(self) -> None:
        """Release only model servers started by this app instance."""
        with self._lock:
            processes = list(self._servers.values())
            self._servers.clear()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        if self._server_job is not None:
            self._server_job.close()
            self._server_job = None

    def _persist(self, record: dict) -> None:
        from ..bootstrap import cache_root
        with self._lock:
            snapshot = dict(record)
        write_json_atomic(cache_root()/"jobs"/record["build_id"]/"state.json", snapshot)

    def _record(self, build_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._builds.get(build_id)
            if record:
                return dict(record)
        if not re.fullmatch(r"[a-f0-9]{32}", build_id):
            return None
        from ..bootstrap import cache_root
        state = cache_root()/"jobs"/build_id/"state.json"
        try:
            if state.is_file():
                record = read_json_atomic(state)
            else:
                # Import completed results produced before state.json existed.
                legacy = state.with_name("result.json")
                if not legacy.is_file():
                    return None
                outcome = read_json_atomic(legacy)
                if not isinstance(outcome,dict) or not ("error" in outcome or isinstance(outcome.get("status"),str) and isinstance(outcome.get("output_dir"),str)):
                    raise ValueError("The legacy build record is incomplete")
                record = {"build_id":build_id,"finished":True,"status":outcome.get("status","crashed"),"output_dir":outcome.get("output_dir",""),"setup_log":[],"error":outcome.get("error")}
            if not isinstance(record,dict) or record.get("build_id") != build_id or type(record.get("finished")) is not bool or not isinstance(record.get("output_dir"),str) or not isinstance(record.get("status"),str):
                raise ValueError("The build record is incomplete")
        except (OSError, ValueError) as exc:
            record = {"build_id":build_id,"finished":True,"status":"crashed","output_dir":"","setup_log":[],"error":f"The build record cannot be read; start over: {exc}"}
        if not record["finished"]:
            record.update(finished=True, status="interrupted", error="The previous local service closed before the build finished. Check the logs; restarting reuses the download cache.")
        with self._lock:
            self._builds[build_id] = record
        return dict(record)

    def status(self, build_id: str) -> dict[str, Any] | None:
        record = self._record(build_id)
        if record is None:
            return None
        manifest = self._read_manifest(record["output_dir"]) if record["output_dir"] else None
        if manifest is not None and not record["finished"]:
            record["status"] = manifest.get("status", record["status"])
        record["manifest"] = manifest
        return record

    def log(self, build_id: str, tail: int = 200) -> str | None:
        record = self._record(build_id)
        if record is None:
            return None
        setup = "\n".join(record.get("setup_log", []))
        if not record["output_dir"]:
            return setup
        path = Path(record["output_dir"]) / "logs" / "convert.log"
        if not path.is_file():
            return setup
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return setup + "\n" + "\n".join(lines[-tail:])

    def report(self, build_id: str, name: str) -> dict[str, Any] | None:
        if name not in ("performance", "quality"):
            raise ValueError("report name must be 'performance' or 'quality'")
        record = self._record(build_id)
        if record is None:
            return None
        if not record["output_dir"]:
            return None
        path = Path(record["output_dir"]) / "reports" / f"{name}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_manifest(output_dir: str) -> dict[str, Any] | None:
        if not output_dir:
            return None
        path = Path(output_dir) / "manifest.json"
        if not path.is_file():
            return None
        try:
            return read_json_atomic(path)
        except (OSError, ValueError):
            return None
