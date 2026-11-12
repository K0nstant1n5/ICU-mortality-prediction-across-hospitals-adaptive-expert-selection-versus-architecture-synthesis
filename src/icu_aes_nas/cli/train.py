from ..training.train import train_expert
from ._hydra import entry


@entry
def main(cfg):
    return train_expert(cfg)


if __name__ == "__main__":
    main()
