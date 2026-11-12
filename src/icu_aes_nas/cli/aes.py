from ..aes.run import run_aes
from ._hydra import entry


@entry
def main(cfg):
    return run_aes(cfg, seed=int(cfg.seed))


if __name__ == "__main__":
    main()
