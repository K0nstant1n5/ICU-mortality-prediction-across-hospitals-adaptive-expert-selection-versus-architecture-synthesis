from ..aes import soup
from ._hydra import entry


@entry
def main(cfg):
    stage = cfg.cli.soup_stage
    if stage == "pretrain":
        return soup.pretrain(cfg)
    if stage == "ingredient":
        return soup.ingredient(cfg, int(cfg.cli.ingredient))
    if stage == "combine":
        return soup.combine(cfg)
    raise ValueError(stage)


if __name__ == "__main__":
    main()
