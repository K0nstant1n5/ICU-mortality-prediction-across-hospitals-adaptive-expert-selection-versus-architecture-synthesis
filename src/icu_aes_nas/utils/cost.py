"""Cost accounting: parameters, FLOPs, latency, peak memory and wall-clock time.

Every training run appends one record to the cost ledger (`artifacts/cost_ledger.jsonl`).
GPU-hours are summed per method from this ledger (thesis cost axis, proposal §12.10).
"""

from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .io import append_jsonl


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, example: dict) -> int | None:
    """Forward FLOPs for one batch using PyTorch's built-in counter (no extra dependency)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(example)
        return int(counter.get_total_flops())
    except Exception:  # some custom ops are not countable; cost table then shows n/a
        return None


@torch.no_grad()
def measure_latency(model: nn.Module, example: dict, repeats: int = 50, warmup: int = 5) -> float:
    """Median forward latency in milliseconds."""
    model.eval()
    device = next(model.parameters()).device
    times = []
    for i in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


@contextmanager
def track_run(ledger: str | Path, method: str, run_id: str, **extra):
    """Times a block and appends a cost record, including on failure."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    record = {"method": method, "run_id": run_id, "host": socket.gethostname(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "device": device_name(), **extra}
    status = "ok"
    try:
        yield record
    except BaseException:
        status = "failed"
        raise
    finally:
        wall = time.time() - t0
        n_gpu = 1 if torch.cuda.is_available() else 0
        record.update({
            "status": status,
            "wall_s": round(wall, 2),
            "gpu_hours": round(wall * n_gpu / 3600, 5),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if n_gpu else None,
        })
        append_jsonl(record, ledger)
