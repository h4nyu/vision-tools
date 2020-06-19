import typer
from .pipeline import eda_bboxes, train, pre_submit, submit

app = typer.Typer()

app.command()(eda_bboxes)

app.command()(train)
app.command()(pre_submit)
app.command()(submit)
