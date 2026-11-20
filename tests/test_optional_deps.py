"""The core package must import without torch, and the torch-dependent
processors must report themselves unavailable rather than crash at import.

This guards the dependency-check contract in the processor capabilities: the
`available=False / unavailable_reason="torch is not installed"` branch has to be
reachable, so the `none` route stays usable on a torch-free install.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_CHILD = r"""
import sys, importlib.abc

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch.") or fullname == "safetensors.torch":
            raise ImportError("blocked: " + fullname)
        return None

sys.meta_path.insert(0, _Blocker())

import turboforge.cli  # the whole import chain must survive without torch
from turboforge.jailbreak import weight_edit
from turboforge.jailbreak.processors import PROCESSORS

assert weight_edit.torch is None
abl = PROCESSORS["abliteration"]
heretic = PROCESSORS["heretic"]
assert abl.capability.available is False
assert "torch" in (abl.capability.unavailable_reason or "")
assert heretic.capability.available is False
assert "heretic" in (heretic.capability.unavailable_reason or "")
assert PROCESSORS["none"].capability.available is True
sys.meta_path.pop(0)
if importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"):
    assert abl.capability.available is True
    weight_edit.require_dependencies()
    assert weight_edit.torch is not None
    assert weight_edit.load_file is not None
print("OK")
"""


class OptionalDependencyTest(unittest.TestCase):
    def test_package_imports_and_torch_processors_refuse_without_torch(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True, text=True, env=env, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()