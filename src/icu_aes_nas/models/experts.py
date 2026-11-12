"""Expert definitions come from `conf/model/*.yaml`; this turns them into NetworkSpecs."""

from __future__ import annotations

from omegaconf import OmegaConf

from ..constants import DYNAMIC_VARS
from ..core.network import NetworkSpec
from ..data.preprocess import FittedPreprocessor


def spec_from_cfg(model_cfg, pp: FittedPreprocessor, use_mask: bool) -> NetworkSpec:
    s = OmegaConf.to_container(model_cfg.spec, resolve=True)
    return NetworkSpec(
        d_model=s["d_model"], stem=s["stem"], blocks=s["blocks"], head=s["head"],
        static_mode=s.get("static_mode", "add"), dropout=s.get("dropout", 0.1),
        n_vars=len(DYNAMIC_VARS), static_dim=pp.static_dim, use_mask=use_mask, task_type=pp.task_type,
    )
