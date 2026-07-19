import math

import pytest

from tiny_crcv.core import aurac
from tiny_crcv.fast_baselines import (
    discrete_semantic_entropy_3,
    lexical_disagreement_3,
    lexical_jaccard_distance,
    semantic_clusters_3,
)


def judgment(left: int, right: int, risk: float) -> dict:
    return {
        "sample_a": left,
        "sample_b": right,
        "normalized_no_probability": risk,
    }


def test_lexical_disagreement_uses_normalized_word_sets() -> None:
    assert lexical_jaccard_distance("The capital is Taipei.", "Taipei capital") == pytest.approx(0.5)
    result = lexical_disagreement_3(["Taipei", "Taipei.", "New Taipei"])
    assert [item["distance"] for item in result["pair_distances"]] == pytest.approx(
        [0.0, 0.5, 0.5]
    )
    assert result["score"] == pytest.approx(1 / 3)


def test_discrete_semantic_entropy_for_one_two_and_three_clusters() -> None:
    one = [judgment(0, 1, 0.1), judgment(0, 2, 0.2), judgment(1, 2, 0.1)]
    assert semantic_clusters_3(one) == [[0, 1, 2]]
    assert discrete_semantic_entropy_3(one)["score"] == pytest.approx(0.0)

    two = [judgment(0, 1, 0.1), judgment(0, 2, 0.9), judgment(1, 2, 0.8)]
    expected_two = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
    assert semantic_clusters_3(two) == [[0, 1], [2]]
    assert discrete_semantic_entropy_3(two)["score"] == pytest.approx(expected_two)

    three = [judgment(0, 1, 0.9), judgment(0, 2, 0.9), judgment(1, 2, 0.9)]
    assert semantic_clusters_3(three) == [[0], [1], [2]]
    assert discrete_semantic_entropy_3(three)["score"] == pytest.approx(math.log(3))


def test_semantic_clustering_uses_connected_components() -> None:
    transitive = [judgment(0, 1, 0.1), judgment(0, 2, 0.9), judgment(1, 2, 0.1)]
    assert semantic_clusters_3(transitive) == [[0, 1, 2]]


def test_aurac_is_mean_accuracy_over_low_risk_prefixes() -> None:
    labels = [0, 1, 0]
    assert aurac(labels, [0.1, 0.3, 0.2]) == pytest.approx((1.0 + 1.0 + 2 / 3) / 3)
    assert aurac(labels, [0.3, 0.1, 0.2]) == pytest.approx((0.0 + 0.5 + 2 / 3) / 3)


def test_aurac_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        aurac([], [])
    with pytest.raises(ValueError):
        aurac([0], [0.1, 0.2])
