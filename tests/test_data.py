import json
import hashlib
import random
from collections import Counter
from pathlib import Path


def test_question_set_is_fixed_and_balanced() -> None:
    path = Path(__file__).parents[1] / "data" / "questions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    assert Counter(row["split"] for row in rows) == {"calibration": 50, "test": 50}
    assert all(row["question"] and row["answers"] for row in rows)
    assert all(row["difficulty"] in {"easy", "medium", "hard"} for row in rows)


def test_fresh_qa_set_matches_the_predeclared_sampling_protocol() -> None:
    root = Path(__file__).parents[1]
    protocol_path = root / "experiments" / "fresh_qa_600_protocol.json"
    data_path = root / "data" / "fresh_qa_600.jsonl"
    manifest = json.loads(
        (root / "data" / "fresh_qa_600_manifest.json").read_text()
    )
    protocol = json.loads(protocol_path.read_text())
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line]

    assert len(rows) == 600
    assert len({row["id"] for row in rows}) == 600
    assert Counter(row["source_dataset"] for row in rows) == {
        "nq_open": 200,
        "trivia_qa": 200,
        "truthful_qa": 200,
    }
    assert Counter(row["split"] for row in rows) == {"calibration": 300, "test": 300}
    assert all(row["question"] and row["answers"] for row in rows)
    assert manifest["protocol_sha256"] == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    assert manifest["data_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()

    rows_by_dataset = {
        key: [row for row in rows if row["source_dataset"] == key]
        for key in ("nq_open", "trivia_qa", "truthful_qa")
    }
    for dataset in protocol["datasets"]:
        expected = sorted(
            random.Random(dataset["sampling_seed"]).sample(
                range(dataset["population_rows"]), dataset["sample_rows"]
            )
        )
        assert [row["source_index"] for row in rows_by_dataset[dataset["key"]]] == expected
