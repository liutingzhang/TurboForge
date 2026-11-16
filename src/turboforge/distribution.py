"""Secret/private path policy shared by release packaging and static serving."""
from pathlib import PurePosixPath


def private_path(name: str, *, allow_template: bool = False) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    for part in parts:
        lowered = part.lower()
        if allow_template and lowered == ".env.example":
            continue
        if lowered.startswith(".") or lowered in ("secrets", "private", "logs", "assistant-sessions", "assistant-cache", "__pycache__"):
            return True
        if lowered.endswith((".pem", ".key", ".log")):
            return True
    return False
