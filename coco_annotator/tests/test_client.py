import pytest
from coco_annotator import CocoAnnotator
from toolz.curried import pipe, map, filter


@pytest.fixture
def client() -> CocoAnnotator:
    c =  CocoAnnotator()
    c.login(username='admin', password='admin')
    return c


def test_image(client: CocoAnnotator) -> None:
    rows = client.image.filter(limit=10)
    pipe(rows, map(lambda x: {
        "id": x['id'],
        "file_name": x['file_name'],
    }), list)

