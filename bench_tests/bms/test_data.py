from bench.bms.data import read_train_rows, Row, InChI
from bench.bms.config import Config
from cytoolz.curried import pipe, map, filter

config = Config()


def test_read() -> None:
    rows = read_train_rows(config, 999)
    inchis = pipe(rows, map(lambda x: x.inchi), list)
    inchis_has_6layer = pipe(inchis, filter(lambda x: len(x.num_layers) == 7), list)
    for i in inchis_has_6layer:
        print(i.name)
        print(i.connections)
        break


# def test_row() -> None:
#     row = InChI(
#         value="InChI=1S/C13H20OS/c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14/h5-7,9,11,14H,8H2,1-4H3"
#     )
