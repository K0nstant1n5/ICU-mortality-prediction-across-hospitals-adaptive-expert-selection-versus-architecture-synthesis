import os
import sys
from functools import wraps

import hydra

from ..pipelines.registry import get_pipeline
from ..training.train import EXIT_INCOMPLETE, Incomplete


def entry(fn):
    """Hydra main with the package config, pipeline check and the 'resubmit me' exit code."""

    @hydra.main(config_path="../conf", config_name="config", version_base="1.3")
    @wraps(fn)
    def _main(cfg):
        get_pipeline(cfg.pipeline)
        try:
            result = fn(cfg)
        except Incomplete as e:
            print(f"INCOMPLETE: {e}", flush=True)
            sys.exit(EXIT_INCOMPLETE)
        if result == "incomplete":
            sys.exit(EXIT_INCOMPLETE)
        return result

    return _main


os.environ.setdefault("HYDRA_FULL_ERROR", "1")
