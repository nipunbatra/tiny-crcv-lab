import numpy as np
import pytest

from tiny_crcv.probe import fit_hidden_probe


def test_hidden_probe_learns_and_explains_a_simple_signal() -> None:
    hidden = [[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    probe = fit_hidden_probe(hidden, [0, 0, 1, 1], steps=200, seed=4)
    risks = probe.predict_risk(hidden)
    assert list(risks) == sorted(risks)
    explanation = probe.explain(hidden[-1], top_k=2)
    assert explanation["sigmoid_risk"] == pytest.approx(risks[-1])
    assert explanation["logit"] == pytest.approx(
        explanation["bias"]
        + sum(item["contribution"] for item in explanation["top_contributions"])
        + explanation["other_dimensions_contribution"]
    )
    assert np.isfinite(probe.training_loss)


def test_hidden_probe_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both label classes"):
        fit_hidden_probe([[0.0], [1.0]], [0, 0], steps=1)
