"""Fetch the predeclared 600-question fresh QA benchmark via Dataset Viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


API = "https://datasets-server.huggingface.co/rows"
DEFAULT_PROTOCOL = Path("experiments/fresh_qa_600_protocol.json")
DEFAULT_OUTPUT = Path("data/fresh_qa_600.jsonl")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fetch_page(dataset: dict[str, Any], offset: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset["dataset"],
            "config": dataset["config"],
            "split": dataset["source_split"],
            "offset": offset,
            "length": 100,
        }
    )
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": "tiny-crcv-lab/0.2"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    return payload["rows"]


def selected_indices(dataset: dict[str, Any]) -> list[int]:
    rng = random.Random(int(dataset["sampling_seed"]))
    return sorted(
        rng.sample(range(int(dataset["population_rows"])), int(dataset["sample_rows"]))
    )


def accepted_answers(dataset_key: str, row: dict[str, Any]) -> list[str]:
    if dataset_key == "nq_open":
        values = row["answer"]
    elif dataset_key == "trivia_qa":
        values = row["answer"]["aliases"]
    elif dataset_key == "truthful_qa":
        values = row["correct_answers"]
    else:
        raise ValueError(f"unsupported dataset key: {dataset_key}")
    answers = [str(value).strip() for value in values if str(value).strip()]
    if not answers:
        raise ValueError(f"row from {dataset_key} has no accepted answer")
    return list(dict.fromkeys(answers))


def fetch_dataset(
    dataset: dict[str, Any], *, split_seed: int, dataset_position: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = selected_indices(dataset)
    offsets = sorted({(index // 100) * 100 for index in selected})
    with ThreadPoolExecutor(max_workers=8) as executor:
        pages = list(executor.map(lambda offset: fetch_page(dataset, offset), offsets))
    source_rows = {
        int(item["row_idx"]): item["row"]
        for page in pages
        for item in page
        if int(item["row_idx"]) in selected
    }
    missing = sorted(set(selected) - set(source_rows))
    if missing:
        raise RuntimeError(f"Dataset Viewer omitted {dataset['key']} indices: {missing}")

    split_order = list(selected)
    random.Random(split_seed + dataset_position).shuffle(split_order)
    calibration = set(split_order[: len(split_order) // 2])
    result: list[dict[str, Any]] = []
    for source_index in selected:
        source = source_rows[source_index]
        result.append(
            {
                "id": f"fresh-{dataset['key']}-{source_index:05d}",
                "source_dataset": dataset["key"],
                "source_repo": dataset["dataset"],
                "source_config": dataset["config"],
                "source_split": dataset["source_split"],
                "source_index": source_index,
                "split": "calibration" if source_index in calibration else "test",
                "question": str(source["question"]).strip(),
                "answers": accepted_answers(dataset["key"], source),
            }
        )
    manifest = {
        "key": dataset["key"],
        "dataset": dataset["dataset"],
        "config": dataset["config"],
        "source_split": dataset["source_split"],
        "selected_source_indices": selected,
        "calibration_source_indices": sorted(calibration),
        "test_source_indices": sorted(set(selected) - calibration),
        "viewer_pages_fetched": len(offsets),
    }
    return result, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")

    protocol = read_json(args.protocol)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    split_seed = int(protocol["sampling"]["split_seed"])
    for position, dataset in enumerate(protocol["datasets"]):
        dataset_rows, manifest = fetch_dataset(
            dataset, split_seed=split_seed, dataset_position=position
        )
        rows.extend(dataset_rows)
        manifests.append(manifest)
        print(
            f"Fetched {dataset['key']}: {len(dataset_rows)} rows "
            f"from {manifest['viewer_pages_fetched']} viewer pages",
            flush=True,
        )

    expected = int(protocol["sampling"]["total_questions"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, received {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(payload)
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "resolved entirely from the predeclared sampling algorithm",
                "protocol": str(args.protocol),
                "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "rows": len(rows),
                "datasets": manifests,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
