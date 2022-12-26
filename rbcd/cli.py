import logging
from logging import FileHandler, Formatter, StreamHandler
from typing import Optional

import click
import pandas as pd

from rbcd import (
    Config,
    EnsembleInference,
    InferenceConfig,
    Model,
    Search,
    SetupFolds,
    Train,
    Validate,
    read_csv,
    seed_everything,
)

stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(Formatter("%(message)s"))
file_handler = FileHandler(f"app.log")
logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    pass


@click.command("setup-folds")
@click.option("-c", "--config-path")
def setup_folds(
    config_path: str,
) -> None:
    config = Config.load(config_path)
    setup_folds = SetupFolds(
        seed=config.seed,
        n_splits=config.n_splits,
    )
    train_df = pd.read_csv("/store/train.csv")
    setup_folds(train_df)
    setup_folds.save("/store")


@click.command()
@click.option("-c", "--config-path")
def train(
    config_path: str,
    data_path: str = "/store",
    limit: Optional[int] = None,
) -> None:
    print("aaaa")
    cfg = Config.load(config_path)
    print(cfg)
    seed_everything(cfg.seed)
    train = Train(cfg)
    train(limit=limit)


@click.command()
@click.option("-c", "--config-path")
@click.option("-l", "--limit", type=int)
def validate(
    config_path: str,
    data_path: str = "/store",
    limit: Optional[int] = None,
) -> None:
    cfg = Config.load(config_path)
    seed_everything(cfg.seed)
    validate = Validate(cfg)
    model = Model.load(cfg).to(cfg.device)
    res = validate(model, enable_find_threshold=True)
    print(res)


@click.command()
@click.option("-c", "--config-path")
@click.option("-l", "--limit", type=int)
def validate_all(
    config_path: str,
    data_path: str = "/store",
    limit: Optional[int] = None,
) -> None:
    cfg = Config.load(config_path)
    df = read_csv("/store/train.csv")
    print(df)
    seed_everything(cfg.seed)
    validate = Validate(cfg, df=df)
    model = Model.load(cfg).to(cfg.device)
    loss, score, auc, binary_score, thr = validate(model, enable_find_threshold=True)
    print(
        f"loss: {loss:.4f}, score: {score:.4f}, auc: {auc:.4f}, binary_score: {binary_score:.4f}, thr: {thr:.4f}"
    )


@click.command()
@click.option("-c", "--config-path")
@click.option("-t", "--n-trials", type=int, default=10)
def search(
    config_path: str,
    n_trials: int,
) -> None:
    cfg = Config.load(config_path)
    search = Search(n_trials=n_trials, cfg=cfg, limit=cfg.search_limit, logger=logger)
    search()


@click.command("inference")
@click.option("-c", "--config-path", type=str)
def inference(
    config_path: str,
) -> None:
    cfg = InferenceConfig.load(config_path)
    logger.info(cfg)
    inference = EnsembleInference(cfg)
    sub = inference()
    sub.to_csv("submission.csv", index=False)
    print(sub)


cli.add_command(setup_folds)
cli.add_command(train)
cli.add_command(validate)
cli.add_command(validate_all)
cli.add_command(search)
cli.add_command(inference)

if __name__ == "__main__":
    cli()
