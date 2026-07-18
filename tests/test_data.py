import json
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
