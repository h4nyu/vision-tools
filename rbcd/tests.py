from sklearn.metrics import f1_score

from rbcd import pfbeta


def test_pfbeta() -> None:
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert pfbeta(labels, predictions) == f1_score(labels, predictions)
