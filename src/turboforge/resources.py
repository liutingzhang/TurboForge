from __future__ import annotations

from pathlib import Path

from .domain import (
    BackendCapability,
    Diagnostic,
    ModelInspection,
    ResourceEstimate,
    Severity,
)
from .system import disk_available, gpu_status, memory_available


_MIB = 1024**2


class ResourceEstimator:
    """Conservative preflight estimates; never presented as measured usage."""

    def estimate(
        self,
        inspection: ModelInspection,
        backend: BackendCapability,
        output_dir: str | Path,
        *,
        jailbreak: str = "none",
        streaming: bool = False,
        measured_bytes_per_second: float | None = None,
        calibration: bool = False,
        sources: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        source_bytes = max(inspection.tensor_bytes, sum(
            int(item.get("bytes", 0)) for item in inspection.files
        ))
        output_bytes = self._output_size(inspection, backend)
        if output_bytes is None and sources and sources.get("gguf"):
            source = Path(sources["gguf"])
            if source.is_file():
                output_bytes = source.stat().st_size + 64 * _MIB
        processing_copy = jailbreak != "none"
        temporary = source_bytes + (output_bytes or 0) if processing_copy else output_bytes or 0
        temporary += 128 * _MIB
        if streaming:
            peak_ram: int | None = min(source_bytes, 2 * 1024**3) + 512 * _MIB
            ram_assumption = "The selected converter declares streaming/chunked weights; actual allocator overhead can still vary"
        else:
            peak_ram = None
            ram_assumption = ("No verified peak-RAM model exists for this backend; the converter "
                              "manages its own host memory and no number is invented")
        if calibration:
            peak_ram = inspection.element_count * 4 + 1024 * _MIB
            ram_assumption = "Automatic calibration loads the full model in FP32 on the CPU; the estimate includes weights plus 1 GiB headroom, and the activation peak can still be higher"

        gpu = gpu_status()
        available_ram = memory_available()
        available_disk = disk_available(output_dir)
        peak_vram = self._serving_vram(output_bytes)
        assumptions = [
            "Output size is an estimate from tensor bytes and target bpw, not a promised artifact size",
            "Nominal bits, bits-per-weight and physical file size are distinct",
            ram_assumption,
            "Peak VRAM is a conservative serving estimate (resident weights + KV/activation overhead), not a measured or conversion-time peak",
            "No conversion-time value is invented without a prior local measurement",
        ]
        diagnostics = []
        if peak_ram is None:
            diagnostics.append(Diagnostic(
                "ram-profile-unverified",
                "Peak host RAM for this backend is not modeled; the converter manages its own "
                "memory. Watch RAM on the first run instead of relying on an invented estimate",
                Severity.WARNING,
            ))
        if temporary > available_disk:
            diagnostics.append(Diagnostic(
                "disk-insufficient",
                f"Estimated temporary disk requirement is {temporary / _MIB:.0f} MiB but only {available_disk / _MIB:.0f} MiB is available",
                Severity.ERROR,
                ("Choose an output drive with more free space",),
            ))
        if peak_ram is not None and available_ram is not None and peak_ram > available_ram:
            diagnostics.append(Diagnostic(
                "ram-insufficient",
                f"Estimated peak RAM is {peak_ram / _MIB:.0f} MiB but only {available_ram / _MIB:.0f} MiB is available",
                Severity.ERROR,
                ("Use a verified streaming converter or a smaller model",),
            ))
        if gpu is None:
            diagnostics.append(Diagnostic(
                "gpu-not-detected",
                "NVIDIA memory could not be queried; GPU-dependent stages must perform their own preflight",
                Severity.WARNING,
            ))
        elif peak_vram is not None and peak_vram > gpu.free_bytes:
            diagnostics.append(Diagnostic(
                "vram-insufficient",
                f"Estimated serving VRAM is {peak_vram / _MIB:.0f} MiB but only "
                f"{gpu.free_bytes / _MIB:.0f} MiB is free on {gpu.name}; the artifact may not "
                "fully reside on the GPU and CPU offload would reduce decode speed",
                Severity.WARNING,
                ("Choose a lower-bpw target", "Reduce max_context / KV cache",
                 "Close other GPU applications"),
            ))
        if measured_bytes_per_second is not None and measured_bytes_per_second <= 0:
            raise ValueError("measured_bytes_per_second must be positive")
        estimate = ResourceEstimate(
            input_bytes=source_bytes,
            estimated_output_bytes=output_bytes,
            temporary_disk_bytes=temporary,
            peak_ram_bytes=peak_ram,
            peak_vram_bytes=peak_vram,
            available_disk_bytes=available_disk,
            available_ram_bytes=available_ram,
            available_vram_bytes=gpu.free_bytes if gpu else None,
            assumptions=assumptions,
            diagnostics=diagnostics,
        )
        return estimate

    @staticmethod
    def _output_size(inspection: ModelInspection, backend: BackendCapability) -> int | None:
        if backend.typical_bpw is None:
            if backend.name == "qwen3_5_gguf":
                return int(inspection.element_count * 1.25) + 64 * _MIB
            return None
        if not inspection.element_count:
            return None
        return int(inspection.element_count * backend.typical_bpw / 8) + 64 * _MIB

    @staticmethod
    def _serving_vram(output_bytes: int | None) -> int | None:
        if not output_bytes:
            return None
        overhead = max(512 * _MIB, int(output_bytes * 0.15))
        return output_bytes + overhead
