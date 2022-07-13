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
def predict(input: str) -> None:
    res = ScoringService.get_model(
        model_path="models",
        reference_path="../datasets/train",
        reference_meta_path="../datasets/train_meta.json",
    )
    if res:
        ScoringService.predict(input)


cli.add_command(predict)

if __name__ == "__main__":
    cli()
