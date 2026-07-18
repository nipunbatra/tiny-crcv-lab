import pytest

from tiny_crcv.core import (
    answer_matches,
    auroc,
    bootstrap_auroc_difference_ci,
    calibrate_threshold,
    compute_features,
    macro_f1,
    normalize_answer,
    rolling_sample_std,
)


def test_answer_normalization_and_alias_matching() -> None:
    assert normalize_answer("Gabriel García-Márquez") == "gabriel garcia marquez"
    assert answer_matches("The capital is Paris, in France.", ["Paris"])
    assert not answer_matches("The answer is Parisian.", ["Paris"])


def test_rolling_std_and_crcv_features() -> None:
    assert rolling_sample_std([1.0, 1.0, 1.0], 3) == [0.0]
    stable = compute_features([0.9] * 6, [None, 0.5, 0.5, 0.5, 0.5, 0.5], window=3)
    volatile = compute_features(
        [0.9] * 6, [None, 0.0, 1.0, 0.0, 1.0, 0.0], window=3
    )
    assert volatile["crcv_mean"] > stable["crcv_mean"]


def test_auroc_and_macro_f1() -> None:
    labels = [0, 0, 1, 1]
    assert auroc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert macro_f1(labels, labels) == 1.0
    difference, low, high = bootstrap_auroc_difference_ci(
        labels, [0.1, 0.2, 0.8, 0.9], [0.4, 0.3, 0.2, 0.1], samples=100
    )
    assert difference == 1.0
    assert low <= difference <= high


def test_calibration_uses_high_scores_as_positive() -> None:
    result = calibrate_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < result.threshold < 0.8
    assert result.macro_f1 == 1.0


def test_invalid_window() -> None:
    with pytest.raises(ValueError):
        rolling_sample_std([1.0, 2.0], 1)
