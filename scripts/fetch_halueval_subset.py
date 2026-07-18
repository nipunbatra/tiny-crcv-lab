"""Fetch the predeclared 50-pair HaluEval-QA slice via Dataset Viewer."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API = "https://datasets-server.huggingface.co/rows"
DATASET = "pminervini/HaluEval"
CONFIG = "qa"
SPLIT = "data"
PROTOCOL = Path("experiments/halueval_qa_50_protocol.json")


def fetch_page(offset: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": 100,
        }
    )
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": "tiny-crcv-lab/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return payload["rows"]


def build_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    selected = protocol["sampling"]["source_indices"]
    calibration = set(protocol["sampling"]["calibration_pair_indices"])
    pages: dict[int, list[dict[str, Any]]] = {}
    for page_offset in sorted({(index // 100) * 100 for index in selected}):
        pages[page_offset] = fetch_page(page_offset)

    source_rows = {
        item["row_idx"]: item["row"]
        for page in pages.values()
        for item in page
        if item["row_idx"] in selected
    }
    missing = sorted(set(selected) - set(source_rows))
    if missing:
        raise RuntimeError(f"Dataset Viewer did not return selected rows: {missing}")

    result: list[dict[str, Any]] = []
    for source_index in selected:
        source = source_rows[source_index]
        common = {
            "pair_id": f"hqa-{source_index:05d}",
            "source_index": source_index,
            "source_dataset": "HaluEval QA",
            "split": "calibration" if source_index in calibration else "test",
            "knowledge": source["knowledge"],
            "question": source["question"],
        }
        result.extend(
            [
                {
                    **common,
                    "id": f"hqa-{source_index:05d}-right",
                    "candidate_kind": "right",
                    "candidate_answer": source["right_answer"],
                    "is_hallucination": 0,
                },
                {
                    **common,
                    "id": f"hqa-{source_index:05d}-hallucinated",
                    "candidate_kind": "hallucinated",
                    "candidate_answer": source["hallucinated_answer"],
                    "is_hallucination": 1,
                },
            ]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=Path("data/halueval_qa_50.jsonl"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    rows = build_rows(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(
        f"Wrote {len(rows)} candidates from {len(rows) // 2} HaluEval pairs to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
