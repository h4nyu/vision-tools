import json
import os
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
    invert_lr,
    kfold,
    preprocess,
    preview_dataset,
    setup_fold,
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
    fold = setup_fold(cfg)
    search = Search(n_trials=10, cfg=cfg, fold=fold)
    search()


@click.command("train")
@click.option("-c", "--config-path")
def _train(config_path: str) -> None:
    cfg = Config.load(config_path)
    fold = setup_fold(cfg)
    train(cfg=cfg, fold=fold)


@click.command("evaluate")
@click.option("-c", "--config-path")
def _evaluate(config_path: str) -> None:
    cfg = Config.load(config_path)
    fold = setup_fold(cfg)
    evaluate(cfg=cfg, fold=fold)


@click.command("preview")
@click.option("-c", "--config-path")
@click.option("-i", "--image-path")
def preview(config_path: str, image_path: str) -> None:
    cfg = Config.load(config_path)
    rows = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    preview_rows = []
    for row in rows:
        if row["image_path"].endswith(image_path):
            preview_rows.append(row)
    preview_dataset(
        cfg=cfg, rows=preview_rows, path=f"outputs/{image_path.replace('/','-')}"
    )


@click.command()
def flip() -> None:
    rows = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    new_rows = invert_lr(rows)
    with open("/app/datasets/extend_meta.json", "w") as f:
        json.dump(new_rows, f)


cli.add_command(predict)
cli.add_command(show)
cli.add_command(search)
cli.add_command(_evaluate)
cli.add_command(_train)
cli.add_command(preview)
cli.add_command(flip)

if __name__ == "__main__":
    cli()
