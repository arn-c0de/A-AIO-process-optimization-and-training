from __future__ import annotations

from dataclasses import dataclass

from gui.utils.system_stats import SystemStatsSampler


@dataclass(frozen=True)
class SystemStats:
    cpu_percent: float | None
    ram_percent: float | None
    gpu_percent: float | None


def read_system_stats(sampler: SystemStatsSampler) -> SystemStats:
    """Sample CPU/RAM/GPU usage without blocking.

    GPU utilization comes from the sampler's background nvidia-smi poller, so
    this is safe to call from the Tk main loop.
    """
    ram = sampler.ram_info()
    return SystemStats(
        cpu_percent=sampler.cpu_percent(),
        ram_percent=ram[0] if ram is not None else None,
        gpu_percent=sampler.gpu_percent(),
    )


def format_system_stats(stats: SystemStats) -> str:
    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0f}%"

    return f"CPU {fmt(stats.cpu_percent)} | GPU {fmt(stats.gpu_percent)} | RAM {fmt(stats.ram_percent)}"
