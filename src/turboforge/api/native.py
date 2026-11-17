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
options = {'parent': root, 'title': 'TurboForge — Select a local path'}
try:
    path = filedialog.askdirectory(**options) if sys.argv[1] == 'directory' else filedialog.askopenfilename(**options)
    print(json.dumps({'path': path or None}))
finally:
    root.destroy()
'''


def pick_path(kind: str) -> str | None:
    if kind not in ("directory", "file"):
        raise ValueError("Only file or directory selection is supported")
    try:
        with _DIALOG_LOCK:
            result = subprocess.run([sys.executable, "-c", _SCRIPT, kind],
                                    capture_output=True, text=True, encoding="utf-8", timeout=300,
                                    env=child_environment())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("The selection dialog timed out; retry or paste the path") from exc
    if result.returncode:
        raise RuntimeError("Cannot open the local selection dialog; paste the path directly. Python needs tkinter and the service must run on the local desktop.")
    return json.loads(result.stdout)["path"]
