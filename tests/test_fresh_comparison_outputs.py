import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tiny_crcv.core import auroc, calibrate_threshold, compute_features


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/fresh_qa_600.jsonl"
PROTOCOL = ROOT / "experiments/fresh_qa_600_protocol.json"
METHODS = (
    "top3_token_surprise",
    "p_true",
    "hidden_logistic_probe",
    "semantic_disagreement_3",
    "answer_tokens",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.mark.parametrize("model", ["instruct", "base"])
def test_fresh_outputs_reconstruct_every_published_score(model: str) -> None:
    output = ROOT / "outputs" / f"fresh_qa_qwen05b_{model}"
    source = read_jsonl(DATA)
    rows = read_jsonl(output / "predictions.jsonl")
    metrics = json.loads((output / "metrics.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())

    assert len(source) == len(rows) == 600
    assert [row["id"] for row in source] == [row["id"] for row in rows]
    assert metadata["data_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert metadata["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert metadata["resolved_revision"] != "main"

    for row in rows:
        recomputed = compute_features(
            row["confidences"],
            row["hidden_shifts"],
            window=metadata["window"],
            token_margins=row["token_margins"],
            token_entropies=row["token_entropies"],
            hidden_cosine_distances=row["hidden_cosine_distances"],
            hidden_norms=row["hidden_norms"],
        )
        assert row["features"] == pytest.approx(recomputed)
        assert row["method_scores"]["top3_token_surprise"] == pytest.approx(
            recomputed["top3_token_surprise"]
        )
        judgment = row["p_true_judgment"]
        expected_p_false = judgment["no_probability"] / (
            judgment["yes_probability"] + judgment["no_probability"]
        )
        assert row["method_scores"]["p_true"] == pytest.approx(expected_p_false)
        assert row["method_scores"]["semantic_disagreement_3"] == pytest.approx(
            np.mean(
                [item["normalized_no_probability"] for item in row["pair_judgments"]]
            )
        )
        assert row["method_scores"]["answer_tokens"] == len(row["token_pieces"])

        explanation = row["probe_explanation"]
        reconstructed_logit = (
            explanation["bias"]
            + sum(
                item["contribution"] for item in explanation["top_contributions"]
            )
            + explanation["other_dimensions_contribution"]
        )
        assert explanation["logit"] == pytest.approx(reconstructed_logit, abs=1e-5)
        reconstructed_risk = 1 / (1 + math.exp(-explanation["logit"]))
        assert explanation["sigmoid_risk"] == pytest.approx(reconstructed_risk)
        assert row["method_scores"]["hidden_logistic_probe"] == pytest.approx(
            reconstructed_risk, abs=5e-7
        )

    calibration = [row for row in rows if row["split"] == "calibration"]
    held_out = [row for row in rows if row["split"] == "test"]
    labels = [row["is_hallucination"] for row in held_out]
    for method in METHODS:
        calibration_threshold = calibrate_threshold(
            [row["is_hallucination"] for row in calibration],
            [row["method_scores"][method] for row in calibration],
        )
        published = metrics["methods"][method]
        assert published["threshold"] == pytest.approx(calibration_threshold.threshold)
        assert published["held_out"]["auroc"] == pytest.approx(
            auroc(labels, [row["method_scores"][method] for row in held_out])
        )


@pytest.mark.parametrize("model", ["instruct", "base"])
def test_hidden_probe_standardization_uses_calibration_only(model: str) -> None:
    output = ROOT / "outputs" / f"fresh_qa_qwen05b_{model}"
    rows = read_jsonl(output / "predictions.jsonl")
    artifacts = np.load(output / "hidden_probe.npz")
    ids = [str(value) for value in artifacts["ids"]]
    calibration_indices = [
        index for index, row in enumerate(rows) if row["split"] == "calibration"
    ]
    hidden = artifacts["mean_hidden"]
    expected_mean = hidden[calibration_indices].mean(axis=0)
    expected_scale = hidden[calibration_indices].std(axis=0)
    expected_scale = np.where(expected_scale < 1e-6, 1.0, expected_scale)

    assert ids == [row["id"] for row in rows]
    assert hidden.shape == (600, 896)
    assert artifacts["weights"].shape == (896,)
    assert np.allclose(artifacts["calibration_mean"], expected_mean, atol=1e-6)
    assert np.allclose(artifacts["calibration_scale"], expected_scale, atol=1e-6)
