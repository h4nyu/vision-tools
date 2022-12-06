import click
import pandas as pd

from rbcd import Config, SetupFolds, Train


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


@click.command("train")
@click.option("-c", "--config-path")
def train(
    config_path: str,
    data_path: str = "/store",
) -> None:
    cfg = Config.load(config_path)
    train = Train(cfg)
    train()


cli.add_command(setup_folds)
cli.add_command(train)

if __name__ == "__main__":
    cli()
