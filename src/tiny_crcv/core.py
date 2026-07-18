"""Pure scoring, labeling, and evaluation functions for the tiny CRCV study."""

from __future__ import annotations

import math
import random
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def normalize_answer(text: str) -> str:
    """Normalize an answer for the benchmark's intentionally simple lexical judge."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_matches(generation: str, accepted_answers: Sequence[str]) -> bool:
    """Return True when a normalized gold answer occurs as a whole phrase."""
    normalized_generation = f" {normalize_answer(generation)} "
    return any(
        f" {normalize_answer(answer)} " in normalized_generation
        for answer in accepted_answers
        if normalize_answer(answer)
    )


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def rolling_sample_std(values: Sequence[float], window: int) -> list[float]:
    """Sample standard deviation over complete trailing windows only."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(values) < 2:
        return [0.0]
    effective_window = min(window, len(values))
    return [
        sample_std(values[end - effective_window : end])
        for end in range(effective_window, len(values) + 1)
    ]


def compute_features(
    confidences: Sequence[float],
    shifts: Sequence[float | None],
    *,
    window: int,
) -> dict[str, float | int]:
    """Aggregate token-level signals into auditable answer-level scores.

    ``confidences[t]`` and ``shifts[t]`` refer to the same generation step.
    The first shift is normally ``None`` because there is no preceding hidden state.
    """
    if len(confidences) != len(shifts):
        raise ValueError("confidences and shifts must have the same length")

    valid_pairs = [
        (float(confidence), float(shift))
        for confidence, shift in zip(confidences, shifts, strict=True)
        if shift is not None and math.isfinite(shift)
    ]
    valid_confidences = [confidence for confidence, _ in valid_pairs]
    valid_shifts = [shift for _, shift in valid_pairs]
    couplings = [confidence * shift for confidence, shift in valid_pairs]

    crcv_windows = rolling_sample_std(couplings, window)
    confidence_windows = rolling_sample_std(valid_confidences, window)
    shift_windows = rolling_sample_std(valid_shifts, window)
    finite_confidences = [max(float(value), 1e-12) for value in confidences]
    token_surprises = [-math.log(value) for value in finite_confidences]
    largest_surprises = sorted(token_surprises, reverse=True)[:3]

    return {
        "crcv_mean": statistics.fmean(crcv_windows),
        "crcv_max": max(crcv_windows),
        "confidence_variance_mean": statistics.fmean(confidence_windows),
        "shift_variance_mean": statistics.fmean(shift_windows),
        "mean_nll": statistics.fmean(token_surprises) if token_surprises else 0.0,
        "top3_token_surprise": statistics.fmean(largest_surprises)
        if largest_surprises
        else 0.0,
        "worst_token_surprise": max(token_surprises, default=0.0),
        "surprise_spread": sample_std(token_surprises),
        "answer_tokens": len(confidences),
    }


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC by pairwise comparisons, including half-credit for ties."""
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires at least one example from each class")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Unweighted mean of the positive- and negative-class F1 values."""
    class_f1s: list[float] = []
    for target in (0, 1):
        tp = sum(y == target and p == target for y, p in zip(labels, predictions, strict=True))
        fp = sum(y != target and p == target for y, p in zip(labels, predictions, strict=True))
        fn = sum(y == target and p != target for y, p in zip(labels, predictions, strict=True))
        denominator = 2 * tp + fp + fn
        class_f1s.append((2 * tp / denominator) if denominator else 0.0)
    return statistics.fmean(class_f1s)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    macro_f1: float


def calibrate_threshold(labels: Sequence[int], scores: Sequence[float]) -> ThresholdResult:
    """Choose a high-score-means-hallucination threshold on calibration data."""
    if not scores:
        raise ValueError("cannot calibrate an empty score sequence")
    unique_scores = sorted(set(float(score) for score in scores))
    epsilon = max(1e-12, (unique_scores[-1] - unique_scores[0]) * 1e-9)
    candidates = [unique_scores[0] - epsilon]
    candidates.extend((left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:]))
    candidates.append(unique_scores[-1] + epsilon)

    best: ThresholdResult | None = None
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        result = ThresholdResult(threshold=threshold, macro_f1=macro_f1(labels, predictions))
        if best is None or (result.macro_f1, result.threshold) > (best.macro_f1, best.threshold):
            best = result
    assert best is not None
    return best


def bootstrap_auroc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    samples: int = 2_000,
    seed: int = 20260718,
) -> tuple[float, float]:
    """Return a deterministic percentile 95% bootstrap interval for AUROC."""
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    estimates: list[float] = []
    for _ in range(samples):
        draw = [rng.choice(indices) for _ in indices]
        draw_labels = [labels[index] for index in draw]
        if len(set(draw_labels)) < 2:
            continue
        estimates.append(auroc(draw_labels, [scores[index] for index in draw]))
    if not estimates:
        raise ValueError("bootstrap produced no samples containing both classes")
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return lower, upper


def bootstrap_auroc_difference_ci(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    samples: int = 2_000,
    seed: int = 20260718,
) -> tuple[float, float, float]:
    """Paired bootstrap interval for AUROC(A) minus AUROC(B)."""
    point_difference = auroc(labels, scores_a) - auroc(labels, scores_b)
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    differences: list[float] = []
    for _ in range(samples):
        draw = [rng.choice(indices) for _ in indices]
        draw_labels = [labels[index] for index in draw]
        if len(set(draw_labels)) < 2:
            continue
        differences.append(
            auroc(draw_labels, [scores_a[index] for index in draw])
            - auroc(draw_labels, [scores_b[index] for index in draw])
        )
    if not differences:
        raise ValueError("bootstrap produced no samples containing both classes")
    differences.sort()
    lower = differences[int(0.025 * (len(differences) - 1))]
    upper = differences[int(0.975 * (len(differences) - 1))]
    return point_difference, lower, upper


def confusion_counts(labels: Iterable[int], predictions: Iterable[int]) -> dict[str, int]:
    pairs = list(zip(labels, predictions, strict=True))
    return {
        "true_negative": sum(y == 0 and p == 0 for y, p in pairs),
        "false_positive": sum(y == 0 and p == 1 for y, p in pairs),
        "false_negative": sum(y == 1 and p == 0 for y, p in pairs),
        "true_positive": sum(y == 1 and p == 1 for y, p in pairs),
    }
