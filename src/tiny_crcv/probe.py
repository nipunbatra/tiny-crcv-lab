"""A tiny, fixed logistic probe for mean hidden states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class HiddenProbe:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    training_loss: float
    steps: int

    def standardize(self, hidden_states: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(hidden_states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("hidden states do not match the fitted probe dimension")
        return (values - self.mean) / self.scale

    def decision_function(self, hidden_states: Sequence[Sequence[float]]) -> np.ndarray:
        return self.standardize(hidden_states) @ self.weights + self.bias

    def predict_risk(self, hidden_states: Sequence[Sequence[float]]) -> np.ndarray:
        logits = self.decision_function(hidden_states).astype(np.float64)
        return np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )

    def explain(self, hidden_state: Sequence[float], *, top_k: int = 8) -> dict[str, Any]:
        standardized = self.standardize([hidden_state])[0]
        contributions = standardized * self.weights
        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        logit = float(self.bias + contributions.sum())
        risk = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (
            1.0 + math.exp(logit)
        )
        selected = [
            {
                "dimension": int(index),
                "raw_hidden": float(hidden_state[index]),
                "calibration_mean": float(self.mean[index]),
                "calibration_scale": float(self.scale[index]),
                "standardized_hidden": float(standardized[index]),
                "weight": float(self.weights[index]),
                "contribution": float(contributions[index]),
            }
            for index in order
        ]
        shown_sum = sum(item["contribution"] for item in selected)
        return {
            "bias": self.bias,
            "top_contributions": selected,
            "other_dimensions_contribution": float(contributions.sum() - shown_sum),
            "logit": logit,
            "sigmoid_risk": risk,
        }


def fit_hidden_probe(
    hidden_states: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    seed: int = 20260726,
    learning_rate: float = 0.03,
    steps: int = 800,
    weight_decay: float = 0.01,
) -> HiddenProbe:
    """Fit the predeclared standardized, class-balanced one-layer sigmoid probe."""
    import torch

    values = np.asarray(hidden_states, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(targets) or len(values) == 0:
        raise ValueError("hidden states and labels must be aligned non-empty arrays")
    counts = np.bincount(targets.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("probe fitting requires both label classes")

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    standardized = ((values - mean) / scale).astype(np.float32)

    torch.manual_seed(seed)
    model = torch.nn.Linear(standardized.shape[1], 1, bias=True, device="cpu")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    inputs = torch.from_numpy(standardized)
    expected = torch.from_numpy(targets).reshape(-1, 1)
    class_weights = len(targets) / (2.0 * counts)
    sample_weights = torch.from_numpy(class_weights[targets.astype(np.int64)]).reshape(-1, 1)

    loss_value = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, expected, reduction="none"
        )
        loss = (losses * sample_weights).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())

    weights = model.weight.detach().numpy()[0].astype(np.float32)
    bias = float(model.bias.detach().item())
    return HiddenProbe(
        mean=mean.astype(np.float32),
        scale=scale,
        weights=weights,
        bias=bias,
        training_loss=loss_value,
        steps=steps,
    )
