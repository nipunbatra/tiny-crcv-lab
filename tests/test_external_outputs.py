import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from tiny_crcv.core import compute_features


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/halueval_qa_50.jsonl"
OUTPUT = ROOT / "outputs/halueval_qwen05b_100"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_frozen_halueval_slice_and_outputs_are_paired() -> None:
    source = read_jsonl(DATA)
    predictions = read_jsonl(OUTPUT / "predictions.jsonl")
    assert len(source) == len(predictions) == 100
    assert {row["id"] for row in source} == {row["id"] for row in predictions}

    source_by_id = {row["id"]: row for row in source}
    pairs: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        original = source_by_id[prediction["id"]]
        for key in ("pair_id", "split", "knowledge", "question", "candidate_answer", "is_hallucination"):
            assert prediction[key] == original[key]
        pairs[prediction["pair_id"]].append(prediction)

    assert len(pairs) == 50
    assert sum(rows[0]["split"] == "calibration" for rows in pairs.values()) == 25
    assert sum(rows[0]["split"] == "test" for rows in pairs.values()) == 25
    for rows in pairs.values():
        assert len(rows) == 2
        assert {row["is_hallucination"] for row in rows} == {0, 1}
        assert len({row["split"] for row in rows}) == 1


def test_external_metadata_hashes_the_exact_inputs() -> None:
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    protocol = ROOT / "experiments/halueval_qa_50_protocol.json"
    assert metadata["data_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert metadata["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert metadata["resolved_revision"] != "main"


def test_published_features_recompute_from_raw_token_signals() -> None:
    predictions = read_jsonl(OUTPUT / "predictions.jsonl")
    for row in predictions:
        recomputed = compute_features(
            row["confidences"],
            row["hidden_shifts"],
            window=5,
            token_margins=row["token_margins"],
            token_entropies=row["token_entropies"],
            hidden_cosine_distances=row["hidden_cosine_distances"],
            hidden_norms=row["hidden_norms"],
        )
        assert recomputed.keys() == row["features"].keys()
        for key, value in recomputed.items():
            assert value == pytest.approx(row["features"][key], abs=1e-12), (row["id"], key)


def test_scalar_selection_and_length_warning_match_saved_rows() -> None:
    predictions = read_jsonl(OUTPUT / "predictions.jsonl")
    metrics = json.loads((OUTPUT / "metrics.json").read_text())
    best_calibration = max(metric["calibration_auroc"] for metric in metrics["scores"].values())
    assert metrics["selected_scalar"]["calibration_auroc"] == best_calibration
    assert metrics["selected_scalar"]["score_key"] == "skip_first_top3_surprise"

    test_pairs: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        if row["split"] == "test":
            test_pairs[row["pair_id"]].append(row)
    longer = 0
    for rows in test_pairs.values():
        right = next(row for row in rows if row["is_hallucination"] == 0)
        hallucinated = next(row for row in rows if row["is_hallucination"] == 1)
        longer += hallucinated["features"]["answer_tokens"] > right["features"]["answer_tokens"]
    assert longer == metrics["length_diagnostics"]["test_pairs_hallucinated_longer"] == 25
