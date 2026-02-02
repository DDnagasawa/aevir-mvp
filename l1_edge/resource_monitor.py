"""Resource monitor for CPU/MEM/GPU usage."""

from __future__ import annotations

import os
import time

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class ResourceMonitor:
    def read(self) -> dict:
        now = int(time.time())
        metrics = {"ts": now}
        if psutil:
            metrics.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "mem_percent": psutil.virtual_memory().percent,
                    "mem_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "cpu_cores": psutil.cpu_count(logical=True),
                }
            )
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        metrics.setdefault("cpu_percent", None)
        metrics.setdefault("mem_percent", None)
        metrics["load1"] = load

        if torch and torch.cuda.is_available():
            gpu_metrics = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                total_gb = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(idx) / (1024**3)
                percent = (allocated / total_gb * 100) if total_gb else None
                gpu_metrics.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_gb": round(total_gb, 2),
                        "allocated_gb": round(allocated, 2),
                        "mem_percent": round(percent, 2) if percent is not None else None,
                        "cuda_capability": f"{props.major}.{props.minor}",
                    }
                )
            metrics["gpus"] = gpu_metrics
            metrics["cuda_version"] = torch.version.cuda
        return metrics

    def hardware_profile(self) -> dict:
        profile = {
            "tflops": None,
            "vram_gb": None,
            "cuda_version": None,
            "rocm_version": None,
            "gpu_name": None,
            "gpu_count": 0,
            "cpu_cores": None,
            "ram_gb": None,
        }
        if psutil:
            profile["cpu_cores"] = psutil.cpu_count(logical=True)
            profile["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        if torch and torch.cuda.is_available():
            profile["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            profile["gpu_name"] = props.name
            profile["vram_gb"] = round(props.total_memory / (1024**3), 2)
            profile["cuda_version"] = torch.version.cuda
        return profile

