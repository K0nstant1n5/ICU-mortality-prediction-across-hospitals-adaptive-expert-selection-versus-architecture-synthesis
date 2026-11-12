from ..eval.report import evaluate
from ._hydra import entry


@entry
def main(cfg):
    res = evaluate(cfg, seed=int(cfg.seed))
    print(res["table"].to_string(index=False))


if __name__ == "__main__":
    main()
