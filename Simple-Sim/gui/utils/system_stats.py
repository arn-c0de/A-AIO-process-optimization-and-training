"""Lightweight CPU/RAM/GPU usage readers for the GUI status bar (Linux /proc + nvidia-smi)."""

from __future__ import annotations

import subprocess
import threading
from typing import Optional, Tuple


def read_ram_info() -> Optional[Tuple[float, int, int]]:
    """Return (used_percent, used_bytes, total_bytes) from /proc/meminfo, or None."""
    try:
        total_kb, avail_kb = None, None
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                if total_kb is not None and avail_kb is not None:
                    break
        if total_kb is None or avail_kb is None or total_kb <= 0:
            return None
        used_kb = total_kb - avail_kb
        return 100.0 * (used_kb / total_kb), used_kb * 1024, total_kb * 1024
    except Exception:
        return None


def read_gpu_percent(timeout_s: float = 0.5) -> Optional[float]:
    """Return GPU utilization (first GPU) via nvidia-smi, or None."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out = (proc.stdout or "").strip()
        if not out:
            return None
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


class SystemStatsSampler:
    """Stateful sampler: CPU usage is computed as a delta between calls.

    GPU utilization is polled by a background thread so the (up to 0.5 s)
    nvidia-smi call never blocks the Tk main loop.
    """

    def __init__(self, gpu_interval_s: float = 1.0) -> None:
        self._cpu_prev_total: Optional[int] = None
        self._cpu_prev_idle: Optional[int] = None
        self._gpu_pct: Optional[float] = None
        self._gpu_interval_s = gpu_interval_s
        self._stop = threading.Event()
        self._gpu_thread: Optional[threading.Thread] = None

    def cpu_percent(self) -> Optional[float]:
        try:
            with open("/proc/stat", encoding="utf-8") as f:
                line = f.readline().strip()
            parts = line.split()
            if not parts or parts[0] != "cpu" or len(parts) < 5:
                return None
            nums = [int(x) for x in parts[1:]]
            total = sum(nums)
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
            if self._cpu_prev_total is None or self._cpu_prev_idle is None:
                self._cpu_prev_total, self._cpu_prev_idle = total, idle
                return None
            dt = total - self._cpu_prev_total
            di = idle - self._cpu_prev_idle
            self._cpu_prev_total, self._cpu_prev_idle = total, idle
            return 100.0 * (1.0 - (di / dt)) if dt > 0 else None
        except Exception:
            return None

    def ram_info(self) -> Optional[Tuple[float, int, int]]:
        return read_ram_info()

    def gpu_percent(self) -> Optional[float]:
        """Return the most recent GPU sample (starts the polling thread on first use)."""
        if self._gpu_thread is None:
            self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
            self._gpu_thread.start()
        return self._gpu_pct

    def stop(self) -> None:
        self._stop.set()

    def _gpu_loop(self) -> None:
        while not self._stop.is_set():
            self._gpu_pct = read_gpu_percent()
            self._stop.wait(self._gpu_interval_s)
