from logging import getLogger, StreamHandler, Formatter, INFO, FileHandler
import argparse
from .train import trainer
from .predict import predictor

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(INFO)
handler_format = Formatter("%(asctime)s|%(name)s|%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="")
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser("train")
parser_train.add_argument("--epochs", type=int, dest="epochs")
parser_train.set_defaults(handler=trainer)

parser_submit = subparsers.add_parser("predict")
parser_submit.set_defaults(handler=predictor)


def main() -> None:
    args = parser.parse_args()
    if args.handler is not None:
        kwargs = vars(args)
        handler = args.handler
        kwargs.pop("handler")
        handler(**kwargs)
    else:
        parser.print_help()
