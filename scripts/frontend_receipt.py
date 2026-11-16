"""Run after Vite to bind shipped assets to their exact sources."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from turboforge.frontend import source_digest


def write_receipt(web: Path) -> None:
    files = {p.relative_to(web/"dist").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (web/"dist").rglob("*") if p.is_file() and p.name != "build-receipt.json"}
    (web/"dist/build-receipt.json").write_text(json.dumps({"source_sha256":source_digest(web),"files":files},indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    write_receipt(ROOT/"web")
