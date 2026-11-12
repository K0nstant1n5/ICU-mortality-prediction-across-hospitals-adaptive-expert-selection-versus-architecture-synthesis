"""Helpers for running inside 8-hour SLURM jobs: time budgets and self-resubmission."""

from __future__ import annotations

import os
import shlex
import subprocess
import time

from .logging import get_logger

log = get_logger(__name__)

_START = time.time()


def in_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def seconds_left(default_budget_s: float) -> float:
    """Seconds until this job must stop. Uses SLURM's end time when available, else a fixed budget."""
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end:
        return float(end) - time.time()
    return default_budget_s - (time.time() - _START)


def resubmit_self(script: str | None = None, extra_env: dict[str, str] | None = None) -> str | None:
    """Submit the same sbatch script again (used by the NAS controller when its budget is not spent)."""
    script = script or os.environ.get("IAN_SBATCH_SCRIPT")
    if not script or not in_slurm():
        log.info("Not resubmitting (no SLURM or no script path).")
        return None
    env = os.environ.copy()
    env.update(extra_env or {})
    cmd = ["sbatch", "--parsable", "--export=ALL", script]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout.strip()
    log.info("Resubmitted %s as job %s", shlex.join(cmd), out)
    return out
