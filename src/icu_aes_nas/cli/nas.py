from ..aes.run import aes_dir
from ..nas.finalize import finalize
from ..nas.search import NASRunner, summary
from ..utils.io import read_json
from ..utils.logging import get_logger
from ._hydra import entry

log = get_logger(__name__)


def _prior(cfg):
    if cfg.nas.prior != "aes":
        return None
    path = aes_dir(cfg) / "nas_prior_seed0.json"
    if not path.exists():
        raise FileNotFoundError(f"nas.prior=aes needs {path}; run ian-aes first")
    return read_json(path)["family_frequency"]


@entry
def main(cfg):
    mode = cfg.cli.nas_mode
    if mode == "search":
        return NASRunner(cfg, _prior(cfg)).run()
    if mode == "finalize":
        return finalize(cfg)
    if mode == "summary":
        log.info("%s", summary(cfg))
        return summary(cfg)
    raise ValueError(mode)


if __name__ == "__main__":
    main()
