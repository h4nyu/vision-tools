from bench.bms.data import read_train_rows
from bench.bms.data import read_train_rows
from bench.bms.config import Config

config = Config()


def test_read_train_rows() -> None:
    read_train_rows(config, nrows=10)
