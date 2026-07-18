import pytest

from tiny_crcv.tree import fit_tree


def test_shallow_tree_learns_and_serializes_a_threshold() -> None:
    rows = [{"surprise": value} for value in [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]]
    labels = [0, 0, 0, 1, 1, 1]
    tree = fit_tree(
        rows,
        labels,
        features=["surprise"],
        max_depth=1,
        min_leaf=2,
    )
    assert tree.feature == "surprise"
    assert tree.threshold == pytest.approx(0.55)
    assert tree.predict({"surprise": 0.25}) == 0
    assert tree.predict({"surprise": 0.85}) == 1
    assert tree.to_dict()["right"]["positives"] == 3


def test_tree_rejects_invalid_training_inputs() -> None:
    with pytest.raises(ValueError):
        fit_tree([], [], features=["x"], max_depth=1, min_leaf=1)
