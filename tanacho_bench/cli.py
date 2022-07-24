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
    SearchRegistry,
    check_folds,
    compare_sample_pair,
    eda,
    evaluate,
    extend_dataset,
    kfold,
    preprocess,
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


@click.command("search-registry")
@click.option("-c", "--config-path")
def search_registry(config_path: str) -> None:
    cfg = Config.load(config_path)
    fold = setup_fold(cfg)
    search = SearchRegistry(n_trials=10, cfg=cfg, fold=fold)
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


@click.command("pair")
@click.option("-c", "--config-path")
@click.option("-r", "--reference-path")
@click.option("-t", "--target-path")
def pair(config_path: str, reference_path: str, target_path: str) -> None:
    cfg = Config.load(config_path)
    rows = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    preview_rows = []
    for row in rows:
        last_2_path = "/".join(row["image_path"].split("/")[-2:])
        if last_2_path == reference_path:
            preview_rows.append(row)
        if last_2_path == target_path:
            preview_rows.append(row)
    for row in preview_rows:
        print(row["image_path"])
    compare_sample_pair(
        cfg=cfg,
        reference=preview_rows[0],
        target=preview_rows[1],
        path=f"outputs/{reference_path.replace('/','-')}-{target_path.replace('/','-')}.png",
    )


@click.command()
def extend() -> None:
    rows = preprocess(
        image_dir="/app/datasets/train",
        meta_path="/app/datasets/train_meta.json",
    )
    new_rows = extend_dataset(rows)
    with open("/app/datasets/extend_meta.json", "w") as f:
        json.dump(new_rows, f)


@click.command("check-fold")
@click.option("-c", "--config-path")
def check_fold(
    config_path: str,
) -> None:
    cfg = Config.load(config_path)
    fold = setup_fold(cfg)


cli.add_command(predict)
cli.add_command(show)
cli.add_command(search)
cli.add_command(_evaluate)
cli.add_command(_train)
cli.add_command(pair)
cli.add_command(extend)
cli.add_command(search_registry)
cli.add_command(check_fold)

if __name__ == "__main__":
    cli()
