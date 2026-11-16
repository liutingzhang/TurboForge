"""Fresh process for installing optional dependencies and executing a build."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    from .manifest import write_json_atomic
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        from .preparation import prepare
        config = prepare(request["model"], request["options"], lambda message: print(message, flush=True))
        print("@@CONFIG@@" + json.dumps(asdict(config), ensure_ascii=False), flush=True)
        from .runner import BuildRunner
        manifest = BuildRunner().build(config)
        write_json_atomic(result_path, {"status": manifest.data["status"], "output_dir": config.output_dir})
        return 0
    except Exception as exc:
        write_json_atomic(result_path, {"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
