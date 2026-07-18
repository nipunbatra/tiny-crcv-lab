"""A dependency-free, deterministic shallow decision tree for tiny experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass
class TreeNode:
    count: int
    positives: int
    probability: float
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def predict(self, row: Mapping[str, float | int]) -> float:
        node = self
        while not node.is_leaf:
            assert node.feature is not None and node.threshold is not None
            child = node.left if float(row[node.feature]) <= node.threshold else node.right
            assert child is not None
            node = child
        return node.probability

    def path(self, row: Mapping[str, float | int]) -> list[str]:
        node = self
        result: list[str] = []
        while not node.is_leaf:
            assert node.feature is not None and node.threshold is not None
            value = float(row[node.feature])
            goes_left = value <= node.threshold
            result.append(
                f"{node.feature}={value:.6g} "
                f"{'≤' if goes_left else '>'} {node.threshold:.6g}"
            )
            child = node.left if goes_left else node.right
            assert child is not None
            node = child
        result.append(
            f"leaf: {node.positives}/{node.count} calibration candidates hallucinated"
        )
        return result

    def to_dict(self) -> dict:
        result = {
            "count": self.count,
            "positives": self.positives,
            "hallucination_probability": self.probability,
        }
        if not self.is_leaf:
            result.update(
                {
                    "feature": self.feature,
                    "threshold": self.threshold,
                    "left_if": "value <= threshold",
                    "left": self.left.to_dict() if self.left else None,
                    "right_if": "value > threshold",
                    "right": self.right.to_dict() if self.right else None,
                }
            )
        return result


def _gini(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    positive_rate = sum(labels) / len(labels)
    return 1.0 - positive_rate**2 - (1.0 - positive_rate) ** 2


def fit_tree(
    rows: Sequence[Mapping[str, float | int]],
    labels: Sequence[int],
    *,
    features: Sequence[str],
    max_depth: int,
    min_leaf: int,
) -> TreeNode:
    """Fit a CART-style tree using weighted Gini reduction and stable tie breaks."""
    if len(rows) != len(labels) or not rows:
        raise ValueError("rows and labels must be non-empty and have the same length")
    if max_depth < 0 or min_leaf < 1:
        raise ValueError("max_depth must be non-negative and min_leaf must be positive")

    indices = list(range(len(rows)))

    def build(selected: list[int], depth: int) -> TreeNode:
        selected_labels = [int(labels[index]) for index in selected]
        positives = sum(selected_labels)
        node = TreeNode(
            count=len(selected),
            positives=positives,
            probability=positives / len(selected),
        )
        if depth >= max_depth or positives in {0, len(selected)}:
            return node

        base = _gini(selected_labels)
        best: tuple[float, int, float, list[int], list[int]] | None = None
        for feature_index, feature in enumerate(features):
            values = sorted({float(rows[index][feature]) for index in selected})
            thresholds = [
                (left + right) / 2 for left, right in zip(values, values[1:])
            ]
            for threshold in thresholds:
                left_indices = [
                    index for index in selected if float(rows[index][feature]) <= threshold
                ]
                right_indices = [index for index in selected if index not in left_indices]
                if len(left_indices) < min_leaf or len(right_indices) < min_leaf:
                    continue
                impurity = (
                    len(left_indices)
                    * _gini([int(labels[index]) for index in left_indices])
                    + len(right_indices)
                    * _gini([int(labels[index]) for index in right_indices])
                ) / len(selected)
                gain = base - impurity
                candidate = (gain, -feature_index, -threshold, left_indices, right_indices)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

        if best is None or best[0] <= 1e-12:
            return node
        _, negative_feature_index, negative_threshold, left_indices, right_indices = best
        node.feature = features[-negative_feature_index]
        node.threshold = -negative_threshold
        node.left = build(left_indices, depth + 1)
        node.right = build(right_indices, depth + 1)
        return node

    return build(indices, 0)
