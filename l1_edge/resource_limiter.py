"""Resource limiter for CPU/MEM thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LimitResult:
    ok: bool
    reason: str | None


class ResourceLimiter:
    def __init__(
        self,
        cpu_percent: float = 85.0,
        mem_percent: float = 85.0,
        gpu_mem_percent: float | None = None,
    ):
        self.cpu_percent = cpu_percent
        self.mem_percent = mem_percent
        self.gpu_mem_percent = gpu_mem_percent

    def check(self, metrics: dict) -> LimitResult:
        cpu = metrics.get("cpu_percent")
        mem = metrics.get("mem_percent")
        if cpu is not None and cpu > self.cpu_percent:
            return LimitResult(False, f"cpu_percent={cpu} exceeds {self.cpu_percent}")
        if mem is not None and mem > self.mem_percent:
            return LimitResult(False, f"mem_percent={mem} exceeds {self.mem_percent}")
        if self.gpu_mem_percent is not None:
            gpus = metrics.get("gpus") or []
            for gpu in gpus:
                used = gpu.get("mem_percent")
                if used is not None and used > self.gpu_mem_percent:
                    return LimitResult(
                        False,
                        f"gpu_mem_percent={used} exceeds {self.gpu_mem_percent}",
                    )
        return LimitResult(True, None)
