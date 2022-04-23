import pytest
from nanoid import generate
from toolz.curried import filter, map, pipe

from coco_annotator import CocoAnnotator


@pytest.fixture
def client() -> CocoAnnotator:
    c = CocoAnnotator()
    c.login(username="admin", password="admin")
    return c


def test_filter_image(client: CocoAnnotator) -> None:
    rows = client.image.filter(limit=10)
    pipe(
        rows,
        map(
            lambda x: {
                "id": x["id"],
                "file_name": x["file_name"],
            }
        ),
        list,
    )


def test_filter_category(client: CocoAnnotator) -> None:
    client.category.filter(limit=10)


def test_crd_category(client: CocoAnnotator) -> None:
    new_row = client.category.create(
        {
            "name": generate(),
            "supercategory": "test",
        }
    )
    saved_row = client.category.get(new_row["id"])
    assert saved_row == new_row

    client.category.delete(new_row["id"])

    deleted_row = client.category.get(new_row["id"])
    assert deleted_row["deleted"] == True


def test_filter_annotation(client: CocoAnnotator) -> None:
    rows = client.annotation.filter(limit=10)


def test_delete_annotation(client: CocoAnnotator) -> None:
    client.annotation.delete(1)


def test_get_annotation(client: CocoAnnotator) -> None:
    client.annotation.get(1)
