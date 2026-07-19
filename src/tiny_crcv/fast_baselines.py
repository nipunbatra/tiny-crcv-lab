"""Small post-hoc baselines computed from the frozen fresh-QA traces."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tiny_crcv.core import normalize_answer
from tiny_crcv.probe import HiddenProbe, fit_hidden_probe


TRACE_ENSEMBLE_FEATURES = (
    "top3_token_surprise",
    "mean_nll",
    "token_entropy_top3",
    "token_ambiguity_top3",
    "crcv_mean",
    "surprise_shift_top3",
    "hidden_cosine_top3",
    "answer_tokens",
)


def _word_set(text: str) -> set[str]:
    return set(normalize_answer(text).split())


def lexical_jaccard_distance(left: str, right: str) -> float:
    """One minus Jaccard overlap over normalized word-token sets."""
    left_words = _word_set(left)
    right_words = _word_set(right)
    union = left_words | right_words
    if not union:
        return 0.0
    return 1.0 - len(left_words & right_words) / len(union)


def lexical_disagreement_3(samples: Sequence[str]) -> dict[str, Any]:
    if len(samples) != 3:
        raise ValueError("lexical_disagreement_3 requires exactly three samples")
    pairs = ((0, 1), (0, 2), (1, 2))
    distances = [
        {
            "sample_a": left,
            "sample_b": right,
            "distance": lexical_jaccard_distance(samples[left], samples[right]),
        }
        for left, right in pairs
    ]
    return {
        "pair_distances": distances,
        "score": statistics.fmean(item["distance"] for item in distances),
    }


def semantic_clusters_3(
    pair_judgments: Sequence[Mapping[str, float | int]],
    *,
    not_equivalent_threshold: float = 0.5,
) -> list[list[int]]:
    """Cluster three answers using connected same-answer judgments."""
    if len(pair_judgments) != 3:
        raise ValueError("semantic_clusters_3 requires exactly three pair judgments")
    parent = list(range(3))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for judgment in pair_judgments:
        risk = float(judgment["normalized_no_probability"])
        if risk < not_equivalent_threshold:
            union(int(judgment["sample_a"]), int(judgment["sample_b"]))

    grouped: dict[int, list[int]] = {}
    for index in range(3):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda cluster: cluster[0])


def discrete_semantic_entropy_3(
    pair_judgments: Sequence[Mapping[str, float | int]],
    *,
    not_equivalent_threshold: float = 0.5,
) -> dict[str, Any]:
    clusters = semantic_clusters_3(
        pair_judgments,
        not_equivalent_threshold=not_equivalent_threshold,
    )
    probabilities = [len(cluster) / 3.0 for cluster in clusters]
    terms = [-probability * math.log(probability) for probability in probabilities]
    return {
        "threshold": not_equivalent_threshold,
        "clusters": clusters,
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "probabilities": probabilities,
        "entropy_terms": terms,
        "score": sum(terms),
    }


def trace_feature_vector(
    features: Mapping[str, float | int],
    keys: Sequence[str] = TRACE_ENSEMBLE_FEATURES,
) -> list[float]:
    return [float(features[key]) for key in keys]


def fit_trace_logistic(
    feature_rows: Sequence[Mapping[str, float | int]],
    labels: Sequence[int],
    *,
    seed: int,
    keys: Sequence[str] = TRACE_ENSEMBLE_FEATURES,
) -> HiddenProbe:
    return fit_hidden_probe(
        [trace_feature_vector(row, keys) for row in feature_rows],
        labels,
        seed=seed,
        learning_rate=0.03,
        steps=800,
        weight_decay=0.01,
    )


def explain_trace_logistic(
    probe: HiddenProbe,
    features: Mapping[str, float | int],
    keys: Sequence[str] = TRACE_ENSEMBLE_FEATURES,
) -> dict[str, Any]:
    vector = trace_feature_vector(features, keys)
    standardized = probe.standardize([vector])[0]
    contributions = standardized * probe.weights
    logit = float(probe.bias + contributions.sum())
    risk = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (
        1.0 + math.exp(logit)
    )
    terms = [
        {
            "feature": key,
            "raw_value": float(vector[index]),
            "calibration_mean": float(probe.mean[index]),
            "calibration_scale": float(probe.scale[index]),
            "standardized_value": float(standardized[index]),
            "weight": float(probe.weights[index]),
            "contribution": float(contributions[index]),
        }
        for index, key in enumerate(keys)
    ]
    return {
        "bias": probe.bias,
        "terms": terms,
        "logit": logit,
        "sigmoid_risk": risk,
    }


def predict_trace_logistic(
    probe: HiddenProbe,
    feature_rows: Sequence[Mapping[str, float | int]],
    keys: Sequence[str] = TRACE_ENSEMBLE_FEATURES,
) -> np.ndarray:
    return probe.predict_risk([trace_feature_vector(row, keys) for row in feature_rows])
