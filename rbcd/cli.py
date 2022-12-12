from typing import Optional

import click
import pandas as pd

from rbcd import Config, Model, SetupFolds, Train, Validate, seed_everything


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
@click.option("-l", "--limit", type=int)
def train(
    config_path: str,
    data_path: str = "/store",
    limit: Optional[int] = None,
) -> None:
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
    loss, score, auc = validate(model, enable_find_threshold=True)
    print(f"loss: {loss:.4f}, score: {score:.4f}, auc: {auc:.4f}")


cli.add_command(setup_folds)
cli.add_command(train)
cli.add_command(validate)

if __name__ == "__main__":
    cli()
