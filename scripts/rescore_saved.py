"""Recompute answer-level features and reports from saved token traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_crcv.benchmark import evaluate, render_report, write_json
from tiny_crcv.core import compute_features


def rescore(output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.json"
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    metadata = json.loads(metadata_path.read_text())
    previous_metrics = json.loads(metrics_path.read_text())
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines() if line]

    for row in rows:
        row["features"] = compute_features(
            row["confidences"],
            row["hidden_shifts"],
            window=int(metadata["window"]),
            token_margins=row.get("token_margins"),
            token_entropies=row.get("token_entropies"),
            hidden_cosine_distances=row.get("hidden_cosine_distances"),
            hidden_norms=row.get("hidden_norms"),
        )

    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    metrics = evaluate(rows)
    for key in ("runtime", "answers_at_token_limit"):
        if key in previous_metrics:
            metrics[key] = previous_metrics[key]
    write_json(metrics_path, metrics)
    (output_dir / "report.md").write_text(render_report(metadata, metrics))
    print(f"Rescored {len(rows)} saved predictions in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for output_dir in args.output_dirs:
        rescore(output_dir)


if __name__ == "__main__":
    main()
