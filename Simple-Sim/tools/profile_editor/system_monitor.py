from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SystemStats:
    cpu_percent: Optional[float]
    ram_percent: Optional[float]
    gpu_percent: Optional[float]


def _cpu_percent() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass

    try:
        load1, _load5, _load15 = os.getloadavg()
        cpus = max(1, os.cpu_count() or 1)
        return max(0.0, min(999.0, (load1 / cpus) * 100.0))
    except Exception:
        return None


def _ram_percent() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().percent)
    except Exception:
        pass

    try:
        mem_total = 0.0
        mem_avail = 0.0
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail = float(line.split()[1])
        if mem_total <= 0.0:
            return None
        used = max(0.0, mem_total - mem_avail)
        return (used / mem_total) * 100.0
    except Exception:
        return None


def _gpu_percent_nvidia() -> Optional[float]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        res = subprocess.run(
            [nvsmi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=0.8,
            check=False,
        )
        if res.returncode != 0:
            return None
        line = (res.stdout or "").strip().splitlines()
        if not line:
            return None
        vals = []
        for x in line:
            x = x.strip()
            if not x:
                continue
            vals.append(float(x))
        if not vals:
            return None
        return sum(vals) / len(vals)
    except Exception:
        return None


def read_system_stats() -> SystemStats:
    return SystemStats(
        cpu_percent=_cpu_percent(),
        ram_percent=_ram_percent(),
        gpu_percent=_gpu_percent_nvidia(),
    )


def format_system_stats(stats: SystemStats) -> str:
    def fmt(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.0f}%"

    return f"CPU {fmt(stats.cpu_percent)} | GPU {fmt(stats.gpu_percent)} | RAM {fmt(stats.ram_percent)}"
