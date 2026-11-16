"""Local desktop file selection, isolated from Flask's worker threads."""
from __future__ import annotations

from ..assistant.config import child_environment
import json
import subprocess
import sys
import threading

_DIALOG_LOCK = threading.Lock()
_SCRIPT = '''
import json, sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
options = {'parent': root, 'title': 'TurboForge — 选择本机路径'}
try:
    path = filedialog.askdirectory(**options) if sys.argv[1] == 'directory' else filedialog.askopenfilename(**options)
    print(json.dumps({'path': path or None}))
finally:
    root.destroy()
'''


def pick_path(kind: str) -> str | None:
    if kind not in ("directory", "file"):
        raise ValueError("仅支持选择文件或目录")
    try:
        with _DIALOG_LOCK:
            result = subprocess.run([sys.executable, "-c", _SCRIPT, kind],
                                    capture_output=True, text=True, encoding="utf-8", timeout=300,
                                    env=child_environment())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("选择窗口已超时，请重试或粘贴路径") from exc
    if result.returncode:
        raise RuntimeError("无法打开本机选择窗口；请直接粘贴路径。Python 需带 tkinter，且服务需在本机桌面运行。")
    return json.loads(result.stdout)["path"]
