import click
from predictor import (
    Config,
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
@click.option("-i", "--input", required=True)
def show_config(input: str) -> None:
    cfg = Config.load(input)
    print(cfg.checkpoint_path)


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


cli.add_command(predict)
cli.add_command(show_config)

if __name__ == "__main__":
    cli()
