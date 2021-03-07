from bench.kuzushiji.data import read_rows, read_code_map
from bench.kuzushiji import config


def test_read_rows() -> None:
    rows = read_rows(config.root_dir)
    print(rows)
