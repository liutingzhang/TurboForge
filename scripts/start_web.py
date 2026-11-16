"""Open the UI immediately; model-specific dependencies download in the job."""
from __future__ import annotations

import json
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import webbrowser
import urllib.request
import urllib.parse
import time


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from turboforge.bootstrap import cache_root, ensure_node
    from turboforge.deployment import check
    from turboforge.assistant.config import child_environment
    print("[1/3] 检查部署环境…", flush=True)
    diagnostics = check(root)
    report = cache_root() / "deployment.json"
    report.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("自检报告：" + str(report), flush=True)
    for error in diagnostics["errors"]:
        print("需要处理：" + error, flush=True)
    if "--check" in sys.argv[1:]:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
        return 0 if diagnostics["ready"] else 1
    if "--prepare-hf" in sys.argv[1:]:
        from turboforge.backends.hf_quantization import ensure_environment
        print("自动准备 HF 环境，不转换模型、不重新测速。",flush=True)
        print("HF Python："+str(ensure_environment()),flush=True)
        return 0
    print("[2/3] 自动准备页面与依赖，首次下载请保持窗口开启…", flush=True)
    requirements = root / "requirements-web.txt"
    expected = [line.split("==", 1) for line in requirements.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    try:
        matching = all(importlib.metadata.version(name) == version for name, version in expected)
    except importlib.metadata.PackageNotFoundError:
        matching = False
    if not matching:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True, env=child_environment())
    web = root / "web"
    from turboforge.frontend import frontend_current
    if not frontend_current(web):
        npm = str(ensure_node(cache_root()))
        subprocess.run([npm, "ci"], cwd=root / "web", check=True, env=child_environment())
        subprocess.run([npm, "run", "build"], cwd=root / "web", check=True, env=child_environment())
        from frontend_receipt import write_receipt
        write_receipt(web)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 8090))
            port = 8090
        except OSError:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    print("[3/3] 启动本机页面…", flush=True)
    url = f"http://127.0.0.1:{port}"
    print(f"TurboForge: {url}\n请保留此窗口，关闭窗口将中断构建。", flush=True)
    target_url = url + ("?model=" + urllib.parse.quote(sys.argv[1]) if len(sys.argv) > 1 else "")
    def open_when_ready():
        for _ in range(100):
            try:
                with urllib.request.urlopen(url, timeout=1):
                    webbrowser.open(target_url)
                    return
            except OSError:
                time.sleep(.2)
    if os.environ.get("TURBOFORGE_NO_BROWSER") != "1":
        threading.Thread(target=open_when_ready, daemon=True).start()
    from turboforge.api.server import main as serve
    return serve(["--port", str(port)])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
