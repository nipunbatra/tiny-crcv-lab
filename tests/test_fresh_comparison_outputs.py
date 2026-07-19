import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tiny_crcv.core import auroc, calibrate_threshold, compute_features
from tiny_crcv.fast_baselines import (
    TRACE_ENSEMBLE_FEATURES,
    discrete_semantic_entropy_3,
    lexical_disagreement_3,
)


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/fresh_qa_600.jsonl"
PROTOCOL = ROOT / "experiments/fresh_qa_600_protocol.json"
ORIGINAL_METHODS = (
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
    frozen_metrics = json.loads((output / "metrics_frozen_five.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())

    assert len(source) == len(rows) == 600
    assert [row["id"] for row in source] == [row["id"] for row in rows]
    assert metadata["data_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert metadata["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert metadata["resolved_revision"] != "main"
    assert metrics["original_frozen_method_order"] == list(ORIGINAL_METHODS)
    assert len(metrics["scalar_method_order"]) == 24
    assert len(metrics["all_method_order"]) == 31
    assert list(metrics["methods"]) == metrics["all_method_order"]
    assert list(frozen_metrics["methods"]) == list(ORIGINAL_METHODS)

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
        for key in metrics["scalar_method_order"]:
            assert row["method_scores"][key] == pytest.approx(recomputed[key])
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

        lexical = lexical_disagreement_3(
            [item["answer"] for item in row["stochastic_answers"]]
        )
        assert row["lexical_disagreement_explanation"] == pytest.approx(lexical)
        assert row["method_scores"]["lexical_disagreement_3"] == pytest.approx(
            lexical["score"]
        )
        semantic_entropy = discrete_semantic_entropy_3(row["pair_judgments"])
        assert row["semantic_entropy_explanation"] == pytest.approx(semantic_entropy)
        assert row["method_scores"]["discrete_semantic_entropy_3"] == pytest.approx(
            semantic_entropy["score"]
        )

        trace_explanation = row["trace_logistic_explanation"]
        trace_logit = trace_explanation["bias"] + sum(
            item["contribution"] for item in trace_explanation["terms"]
        )
        assert trace_explanation["logit"] == pytest.approx(trace_logit, abs=1e-6)
        trace_risk = 1 / (1 + math.exp(-trace_logit))
        assert trace_explanation["sigmoid_risk"] == pytest.approx(trace_risk, abs=1e-6)
        assert row["method_scores"]["trace_logistic_8"] == pytest.approx(
            trace_risk, abs=1e-6
        )
        assert [item["feature"] for item in trace_explanation["terms"]] == list(
            TRACE_ENSEMBLE_FEATURES
        )

        leaf = row["trace_tree_path"][-1]
        positives, count = leaf.removeprefix("leaf: ").split()[0].split("/")
        assert row["method_scores"]["trace_tree_depth2"] == pytest.approx(
            int(positives) / int(count)
        )

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
    for method in metrics["all_method_order"]:
        calibration_threshold = calibrate_threshold(
            [row["is_hallucination"] for row in calibration],
            [row["method_scores"][method] for row in calibration],
        )
        published = metrics["methods"][method]
        assert published["threshold"] == pytest.approx(calibration_threshold.threshold)
        assert published["held_out"]["auroc"] == pytest.approx(
            auroc(labels, [row["method_scores"][method] for row in held_out])
        )

    assert metrics["posthoc_paired_intervals_above_zero"] == []


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
