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
    token_margins: Sequence[float] | None = None,
    token_entropies: Sequence[float] | None = None,
    hidden_cosine_distances: Sequence[float | None] | None = None,
    hidden_norms: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Aggregate token-level signals into auditable answer-level scores.

    ``confidences[t]`` and ``shifts[t]`` refer to the same generation step.
    The first shift is normally ``None`` because there is no preceding hidden state.
    """
    if len(confidences) != len(shifts):
        raise ValueError("confidences and shifts must have the same length")
    optional_signals = {
        "token_margins": token_margins,
        "token_entropies": token_entropies,
        "hidden_cosine_distances": hidden_cosine_distances,
        "hidden_norms": hidden_norms,
    }
    for name, values in optional_signals.items():
        if values is not None and len(values) != len(confidences):
            raise ValueError(f"{name} must have the same length as confidences")

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
    valid_surprise_shift = [
        token_surprises[index] * float(shift)
        for index, shift in enumerate(shifts)
        if shift is not None and math.isfinite(shift)
    ]
    valid_uncertainty_shift = [
        (1.0 - finite_confidences[index]) * float(shift)
        for index, shift in enumerate(shifts)
        if shift is not None and math.isfinite(shift)
    ]
    margins = [float(value) for value in (token_margins or [])]
    ambiguities = [1.0 - value for value in margins]
    entropies = [float(value) for value in (token_entropies or [])]
    cosine_distances = [
        float(value)
        for value in (hidden_cosine_distances or [])
        if value is not None and math.isfinite(value)
    ]
    norms = [float(value) for value in (hidden_norms or [])]

    def mean_or_zero(values: Sequence[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def top3_mean(values: Sequence[float]) -> float:
        return mean_or_zero(sorted(values, reverse=True)[:3])

    return {
        "crcv_mean": statistics.fmean(crcv_windows),
        "crcv_max": max(crcv_windows),
        "confidence_variance_mean": statistics.fmean(confidence_windows),
        "shift_variance_mean": statistics.fmean(shift_windows),
        "mean_nll": statistics.fmean(token_surprises) if token_surprises else 0.0,
        "top3_token_surprise": statistics.fmean(largest_surprises)
        if largest_surprises
        else 0.0,
        "top4_token_surprise": mean_or_zero(sorted(token_surprises, reverse=True)[:4]),
        "skip_first_top3_surprise": top3_mean(token_surprises[1:]),
        "worst_token_surprise": max(token_surprises, default=0.0),
        "surprise_spread": sample_std(token_surprises),
        "surprise_shift_top3": top3_mean(valid_surprise_shift),
        "uncertainty_shift_top3": top3_mean(valid_uncertainty_shift),
        "token_ambiguity_mean": mean_or_zero(ambiguities),
        "token_ambiguity_max": max(ambiguities, default=0.0),
        "token_ambiguity_top3": top3_mean(ambiguities),
        "token_entropy_mean": mean_or_zero(entropies),
        "token_entropy_max": max(entropies, default=0.0),
        "token_entropy_top3": top3_mean(entropies),
        "hidden_cosine_mean": mean_or_zero(cosine_distances),
        "hidden_cosine_max": max(cosine_distances, default=0.0),
        "hidden_cosine_top3": top3_mean(cosine_distances),
        "hidden_norm_mean": mean_or_zero(norms),
        "hidden_norm_variability": sample_std(norms),
        "answer_tokens": len(confidences),
    }


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC by pairwise comparisons, including half-credit for ties."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        raise ValueError("AUROC requires at least one example from each class")

    groups: dict[float, list[int]] = {}
    for label, score in zip(labels, scores, strict=True):
        counts = groups.setdefault(float(score), [0, 0])
        counts[int(label)] += 1

    wins = 0.0
    lower_negatives = 0
    for score in sorted(groups):
        negative_count, positive_count = groups[score]
        wins += positive_count * (lower_negatives + 0.5 * negative_count)
        lower_negatives += negative_count
    return wins / (positives * negatives)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute tie-aware average precision for high-score-means-positive risks."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have the same length")
    positive_count = sum(label == 1 for label in labels)
    if positive_count == 0:
        raise ValueError("average precision requires at least one positive example")

    score_groups: dict[float, list[int]] = {}
    for label, score in zip(labels, scores, strict=True):
        score_groups.setdefault(float(score), []).append(int(label))

    seen = 0
    true_positives = 0
    result = 0.0
    for score in sorted(score_groups, reverse=True):
        group = score_groups[score]
        group_positives = sum(label == 1 for label in group)
        seen += len(group)
        true_positives += group_positives
        result += (group_positives / positive_count) * (true_positives / seen)
    return result


def selective_accuracy(
    labels: Sequence[int], scores: Sequence[float], coverage: float
) -> float:
    """Accuracy among the lowest-risk fraction, where label 1 means incorrect."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have the same length")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    keep = max(1, round(len(labels) * coverage))
    ranked = sorted(range(len(labels)), key=lambda index: (scores[index], index))[:keep]
    return sum(labels[index] == 0 for index in ranked) / keep


def aurac(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean retained-answer accuracy over every non-empty coverage level.

    Answers are ordered from lowest to highest risk.  For each retained prefix
    of size 1..N, compute strict-label accuracy, then average those N values.
    This is the discrete area under the rejection-accuracy curve used here.
    """
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have the same length")
    ranked = sorted(range(len(labels)), key=lambda index: (scores[index], index))
    correct = 0
    accuracies: list[float] = []
    for retained, index in enumerate(ranked, start=1):
        correct += int(labels[index] == 0)
        accuracies.append(correct / retained)
    return statistics.fmean(accuracies)


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
