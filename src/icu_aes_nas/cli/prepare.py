from ..data.prepare import prepare_task
from ._hydra import entry


@entry
def main(cfg):
    prepare_task(cfg)


if __name__ == "__main__":
    main()
