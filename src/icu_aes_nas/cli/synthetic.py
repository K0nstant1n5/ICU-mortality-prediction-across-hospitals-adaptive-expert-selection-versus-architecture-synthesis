"""Write fabricated data (no real patients) in the real folder layout, for smoke tests and dry runs."""

from ..data.synthetic import make_synthetic
from ._hydra import entry


@entry
def main(cfg):
    sources = tuple(s for s, c in cfg.data.sources.items() if c.enabled)
    make_synthetic(cfg.paths.data, n_patients=int(cfg.cli.synthetic_patients), sources=sources)


if __name__ == "__main__":
    main()
