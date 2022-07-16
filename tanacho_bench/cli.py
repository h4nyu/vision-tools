from dataclasses import asdict
from pprint import pprint

import click
from predictor import (
    Config,
    LitModelNoNet,
    ScoringService,
    Search,
    check_folds,
    eda,
    evaluate,
    kfold,
    preprocess,
    preview_dataset,
    train,
)


@click.group()
def cli() -> None:
    pass


@click.command()
@click.option("-c", "--config-path", required=True)
def show(config_path: str) -> None:
    cfg = Config.load(config_path)
    pprint(cfg.__dict__, width=1)


@click.command()
@click.option("-i", "--input", required=True)
@click.option("-m", "--model_path", default="model")
@click.option("-d", "--reference_path", default="/app/datasets/train")
@click.option("-r", "--reference_meta_path", default="/app/datasets/train_meta.json")
def predict(
    input: str, reference_path: str, reference_meta_path: str, model_path: str
) -> None:
    res = ScoringService.get_model(
        model_path=model_path,
        reference_path=reference_path,
        reference_meta_path=reference_meta_path,
    )
    if res:
        ScoringService.predict(input)


@click.command()
@click.option("-c", "--config-path")
def search(config_path: str) -> None:
    cfg = Config.load(config_path)
    data = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    folds = kfold(cfg=cfg, rows=data["rows"])
    search = Search(n_trials=10, cfg=cfg, fold=folds[cfg.fold])
    search()


@click.command("train")
@click.option("-c", "--config-path")
def _train(config_path: str) -> None:
    cfg = Config.load(config_path)
    data = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    folds = kfold(cfg=cfg, rows=data["rows"])
    train(cfg=cfg, fold=folds[cfg.fold])


@click.command("fine")
@click.option("-c", "--config-path")
def fine(config_path: str) -> None:
    cfg = Config.load(config_path)
    fine_cfg = cfg.fine_cfg
    model = LitModelNoNet.load_from_checkpoint(cfg.checkpoint_path, cfg=fine_cfg)
    data = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    folds = kfold(cfg=cfg, rows=data["rows"])
    train(cfg=fine_cfg, fold=folds[cfg.fold], model=model)


@click.command("evaluate")
@click.option("-c", "--config-path")
@click.option("-f", "--fine", default=False)
def _evaluate(config_path: str, fine: bool) -> None:
    cfg = Config.load(config_path)
    if fine:
        cfg = cfg.fine_cfg

    print(cfg.checkpoint_path)
    data = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    folds = kfold(cfg=cfg, rows=data["rows"])
    evaluate(cfg=cfg, fold=folds[cfg.fold])


cli.add_command(predict)
cli.add_command(show)
cli.add_command(search)
cli.add_command(_evaluate)
cli.add_command(_train)
cli.add_command(fine)

if __name__ == "__main__":
    cli()
