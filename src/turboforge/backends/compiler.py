"""NInferCompiler: drives the official tools.convert pipeline of a pinned
ninfer-all checkout. TurboForge never re-implements the conversion math; it
locates the upstream tree, validates the recipe, and runs it as a subprocess
with captured logs.
"""

from __future__ import annotations

from ..assistant.config import child_environment
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..domain import Diagnostic, Severity
from .registry import PINNED_UPSTREAM_REVISION, RecipeSpec, get_recipe

__all__ = [
    "DEFAULT_UPSTREAM",
    "NInferCompiler",
    "UpstreamStatus",
    "get_recipe",
    "probe_upstream",
]


_BUNDLED_SOURCE = Path(__file__).resolve().parents[3] / "runtime/source"
DEFAULT_UPSTREAM = _BUNDLED_SOURCE


@dataclass
class UpstreamStatus:
    root: Path | None
    found: bool
    revision: str | None
    revision_matches_pin: bool
    torch_available: bool
    numpy_available: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    cuda_available: bool | None = None


def probe_upstream(root: Path | None = None) -> UpstreamStatus:
    root = root or Path(os.environ.get("TURBOFORGE_NINFER_SRC", DEFAULT_UPSTREAM))
    diagnostics: list[Diagnostic] = []
    found = (root / "tools" / "convert" / "__main__.py").is_file()
    revision = None
    matches = False
    if not found:
        diagnostics.append(Diagnostic(
            "upstream-missing",
            f"NInfer source tree not found at {root}; clone iamwavecut/ninfer-all or set TURBOFORGE_NINFER_SRC",
            Severity.ERROR,
        ))
    else:
        marker = root / ".turboforge-source.json"
        if marker.is_file():
            revision = json.loads(marker.read_text(encoding="utf-8"))["revision"]
            matches = revision == PINNED_UPSTREAM_REVISION
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=15,
                env=child_environment(),
            )
            if result.returncode == 0:
                revision = result.stdout.strip()
                matches = revision == PINNED_UPSTREAM_REVISION
                if not matches:
                    diagnostics.append(Diagnostic(
                        "upstream-revision-differs",
                        f"Upstream is at {revision[:12]}, recipes were validated against {PINNED_UPSTREAM_REVISION[:12]}; recipe behavior may differ",
                        Severity.WARNING,
                    ))
        except (OSError, subprocess.SubprocessError):
            if revision is not None:
                pass  # Official source archives carry their pinned revision marker.
            else:
                diagnostics.append(Diagnostic(
                    "upstream-revision-unknown", "无法确认转换器源码版本", Severity.WARNING,
                ))
    if revision and not matches and not any(d.code == "upstream-revision-differs" for d in diagnostics):
        diagnostics.append(Diagnostic("upstream-revision-differs", f"当前自动安装源码 {revision[:12]} 与原配方核对版本不同；将由真实转换与 GPU 验证确认结果", Severity.WARNING))
    import importlib.util
    torch_ok = importlib.util.find_spec("torch") is not None
    numpy_available = importlib.util.find_spec("numpy") is not None
    if not torch_ok:
        diagnostics.append(Diagnostic(
            "torch-missing",
            "tools.convert imports torch at module level; install torch (CUDA build for GPU-side quantization methods) before building artifacts",
            Severity.ERROR,
        ))
    if not numpy_available:
        diagnostics.append(Diagnostic(
            "numpy-missing", "tools.convert requires numpy", Severity.ERROR
        ))
    cuda_available = False
    if torch_ok:
        import torch
        cuda_available = torch.cuda.is_available()
    return UpstreamStatus(root, found, revision, matches, torch_ok, numpy_available,
                          diagnostics, cuda_available)


class NInferCompiler:
    def __init__(self, upstream: UpstreamStatus | None = None, python: str | None = None) -> None:
        self.upstream = upstream or probe_upstream()
        self.python = python or sys.executable

    def build_command(
        self,
        recipe: RecipeSpec | str,
        model: Path,
        out: Path,
        *,
        sources: dict[str, Path] | None = None,
        components: str | None = None,
        name: str | None = None,
        device: str = "cuda",
        rows_per_chunk: int = 512,
        proposal: bool = False,
    ) -> list[str]:
        sources = sources or {}
        if isinstance(recipe, RecipeSpec):
            if recipe.engine != "ninfer":
                raise ValueError("AWQ/GPTQ 原生路线由 HF worker 执行，不能交给 NInfer 转换器")
            missing = [s for s in recipe.extra_sources if s not in sources]
            if missing:
                raise ValueError(
                    f"Recipe {recipe.recipe} requires source(s) {', '.join(missing)}; "
                    "provide them explicitly (e.g. quantized=<NVFP4 checkpoint dir>)"
                )
            recipe_ref = recipe.recipe
            if recipe.custom_function:
                recipe_ref = str(Path(__file__).with_name("gguf_quantization.py")) + ":" + recipe.custom_function
        else:
            # A user recipe file reference (FILE or FILE:function), exactly as
            # tools.convert accepts it; the official recipes always go through
            # the RecipeSpec path above.
            recipe_ref = recipe
        command = [
            self.python, "-m", "tools.convert",
            "--model", str(model.expanduser().resolve()),
            "--recipe", recipe_ref,
            "--out", str(out.expanduser().resolve()),
            "--device", device,
            "--rows-per-chunk", str(rows_per_chunk),
        ]
        for source_name, source_path in sorted(sources.items()):
            command += ["--source", f"{source_name}={source_path.expanduser().resolve()}"]
        if components:
            command += ["--components", components]
            if isinstance(recipe,RecipeSpec) and recipe.recipe == "qwen3_5_gguf" and "dflash2" in components.split(","):
                command += ["--override",str(Path(__file__).with_name("gguf_quantization.py"))+":preserve_draft_formats"]
        if name:
            command += ["--name", name]
        if proposal:
            command.append("--proposal")
        return command

    def run(
        self,
        recipe: RecipeSpec | str,
        model: Path,
        out: Path,
        *,
        sources: dict[str, Path] | None = None,
        components: str | None = None,
        name: str | None = None,
        device: str = "cuda",
        log_path: Path | None = None,
        timeout: int | None = None,
    ) -> dict:
        if not self.upstream.found:
            raise RuntimeError("NInfer upstream tree is missing; conversion is impossible")
        if not self.upstream.torch_available:
            raise RuntimeError("torch is not installed; tools.convert cannot run")
        command = self.build_command(
            recipe, model, out, sources=sources, components=components,
            name=name, device=device,
        )
        started = time.monotonic()
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            cwd=self.upstream.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_environment(environment),
        )
        lines: list[str] = []
        log_stream = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
        try:
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                if log_stream:
                    log_stream.write(line)
                    log_stream.flush()
            process.stdout.close()
            code = process.wait(timeout=timeout)
        finally:
            if log_stream:
                log_stream.close()
        elapsed = time.monotonic() - started
        result = {
            "command": command,
            "cwd": str(self.upstream.root),
            "returncode": code,
            "seconds": elapsed,
            "log_path": str(log_path) if log_path else None,
            "tail": lines[-20:],
        }
        if code != 0:
            raise RuntimeError(
                f"tools.convert failed with exit code {code}; last lines: "
                + "".join(lines[-5:])
            )
        if not out.exists() and not out.with_suffix(".ninfer").exists():
            # pipeline prints "wrote <out>" and the writer may append .ninfer
            raise RuntimeError(f"tools.convert exited 0 but produced no artifact at {out}")
        return result
