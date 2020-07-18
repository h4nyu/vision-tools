import argparse
from .train import train
from .predict import predict

parser = argparse.ArgumentParser(description="")
subparsers = parser.add_subparsers()
parser_train = subparsers.add_parser("train")
parser_train.add_argument("-e", "--epochs", type=int, default=100, dest="epochs")
parser_train.set_defaults(handler=train)

parser_predict = subparsers.add_parser("predict")
parser_predict.set_defaults(handler=predict)


def main() -> None:
    args = parser.parse_args()
    if args.handler is not None:
        kwargs = vars(args)
        handler = args.handler
        kwargs.pop("handler")
        handler(**kwargs)
    else:
        parser.print_help()
