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
    resize_all_images,
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
@click.option("-b", "--batch-size", default=None, type=int)
@click.option("-l", "--limit", type=int)
def validate(
    config_path: str,
    data_path: str = "/store",
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    cfg = Config.load(config_path)
    cfg.batch_size = batch_size or cfg.batch_size
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
    res = validate(model, enable_find_threshold=True)
    print(res)


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


@click.command()
@click.option("-c", "--csv-path", type=str)
@click.option("-i", "--input-dir", type=str)
@click.option("-o", "--output-dir", type=str)
@click.option("-s", "--size", type=int)
@click.option("-w", "--workers", type=int)
def resize(
    csv_path: str,
    input_dir: str,
    output_dir: str,
    size: int,
    workers: int,
) -> None:
    df = read_csv(csv_path)
    resize_all_images(
        df,
        input_dir=input_dir,
        output_dir=output_dir,
        image_size=size,
        n_jobs=workers,
    )


cli.add_command(setup_folds)
cli.add_command(train)
cli.add_command(validate)
cli.add_command(validate_all)
cli.add_command(search)
cli.add_command(inference)
cli.add_command(resize)

if __name__ == "__main__":
    cli()
