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


def test_simple_token_surprise_features() -> None:
    features = compute_features([1.0, 0.5, 0.25, 0.125], [None, 1.0, 1.0, 1.0], window=3)
    surprises = [0.0, 0.69314718056, 1.38629436112, 2.07944154168]
    assert features["mean_nll"] == pytest.approx(sum(surprises) / 4)
    assert features["top3_token_surprise"] == pytest.approx(sum(surprises[1:]) / 3)
    assert features["worst_token_surprise"] == pytest.approx(surprises[-1])
    assert features["surprise_spread"] > 0


def test_distribution_and_hidden_state_features() -> None:
    features = compute_features(
        [0.8, 0.4, 0.2],
        [None, 0.3, 0.6],
        window=5,
        token_margins=[0.7, 0.2, 0.05],
        token_entropies=[0.1, 0.4, 0.7],
        hidden_cosine_distances=[None, 0.02, 0.08],
        hidden_norms=[1.0, 2.0, 3.0],
    )
    assert features["token_ambiguity_mean"] == pytest.approx((0.3 + 0.8 + 0.95) / 3)
    assert features["token_entropy_top3"] == pytest.approx(0.4)
    assert features["hidden_cosine_max"] == pytest.approx(0.08)
    assert features["hidden_norm_variability"] == pytest.approx(1.0)


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
