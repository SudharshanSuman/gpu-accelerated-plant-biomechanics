"""Warp device selection and reporting.

The project targets an RTX 4070, but the code must also run on a machine where
the CUDA driver is unavailable (headless CI, a laptop, or a workstation whose
driver has not been reloaded after a kernel update). Every module therefore
asks for a device through `resolve_device()` instead of hard-coding "cuda".
"""

from __future__ import annotations

import warp as wp

_initialised = False


def ensure_init() -> None:
    global _initialised
    if not _initialised:
        wp.init()
        _initialised = True


def resolve_device(requested: str | None = None) -> str:
    """Return a usable Warp device string.

    `requested` of None means "best available": the CUDA device if the driver
    is up, otherwise the CPU. Asking explicitly for "cuda" when no driver is
    present is an error rather than a silent downgrade, because a benchmark
    that quietly ran on the CPU would be worse than no benchmark at all.
    """
    ensure_init()

    if requested is None:
        return "cuda:0" if wp.is_cuda_available() else "cpu"

    if requested.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError(
            "CUDA device requested but no CUDA driver is available. "
            "Check `nvidia-smi`; pass device='cpu' to run without a GPU."
        )
    return requested


def describe_device(device: str) -> str:
    ensure_init()
    d = wp.get_device(device)
    if d.is_cuda:
        free, total = wp.get_device_memory_info(d) if hasattr(wp, "get_device_memory_info") else (0, 0)
        extra = f", {total / 1e9:.1f} GB VRAM" if total else ""
        return f"{d.alias} -> {d.name}{extra}"
    return f"{d.alias} -> {d.name} (CPU fallback; GPU timings will not be meaningful)"


def device_report() -> str:
    ensure_init()
    lines = [f"Warp {wp.config.version}", f"CUDA available: {wp.is_cuda_available()}"]
    for d in wp.get_devices():
        lines.append(f"  device {d.alias}: {d.name}")
    return "\n".join(lines)
