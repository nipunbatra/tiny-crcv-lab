"""Fit the same predeclared shallow tree to existing saved-generation traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tiny_crcv.benchmark import SCORES, write_json
from tiny_crcv.core import (
    auroc,
    bootstrap_auroc_ci,
    calibrate_threshold,
    confusion_counts,
    macro_f1,
)
from tiny_crcv.tree import fit_tree


FEATURES = [
    "top3_token_surprise",
    "token_entropy_top3",
    "token_ambiguity_top3",
    "surprise_shift_top3",
    "hidden_cosine_top3",
    "answer_tokens",
]
MODELS = {
    "instruct": Path("outputs/qwen05b_100"),
    "base": Path("outputs/qwen05b_base_100"),
}


def evaluate_tree(rows: list[dict[str, Any]], depth: int, seed: int) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    tree = fit_tree(
        [row["features"] for row in calibration],
        [int(row["is_hallucination"]) for row in calibration],
        features=FEATURES,
        max_depth=depth,
        min_leaf=8,
    )
    calibration_labels = [int(row["is_hallucination"]) for row in calibration]
    test_labels = [int(row["is_hallucination"]) for row in test]
    calibration_scores = [tree.predict(row["features"]) for row in calibration]
    test_scores = [tree.predict(row["features"]) for row in test]
    threshold = calibrate_threshold(calibration_labels, calibration_scores)
    test_predictions = [int(score >= threshold.threshold) for score in test_scores]
    return {
        "calibration_auroc": auroc(calibration_labels, calibration_scores),
        "calibration_macro_f1": threshold.macro_f1,
        "threshold": threshold.threshold,
        "test_auroc": auroc(test_labels, test_scores),
        "test_auroc_ci_95": list(
            bootstrap_auroc_ci(test_labels, test_scores, seed=seed)
        ),
        "test_macro_f1": macro_f1(test_labels, test_predictions),
        "test_confusion": confusion_counts(test_labels, test_predictions),
        "rules": tree.to_dict(),
    }


def main() -> None:
    result = {
        "status": "exploratory; existing held-out splits were inspected previously",
        "protocol": {
            "features": FEATURES,
            "primary_depth": 2,
            "diagnostic_depth": 1,
            "minimum_leaf_candidates": 8,
            "training_split": "calibration only",
        },
        "models": {},
    }
    for model_index, (model, output_dir) in enumerate(MODELS.items()):
        rows = [
            json.loads(line)
            for line in (output_dir / "predictions.jsonl").read_text().splitlines()
            if line
        ]
        metrics = json.loads((output_dir / "metrics.json").read_text())
        best_scalar_key = max(
            SCORES,
            key=lambda key: (
                metrics["scores"][key]["calibration_auroc"],
                -list(SCORES).index(key),
            ),
        )
        result["models"][model] = {
            "selected_scalar": {
                "score_key": best_scalar_key,
                **metrics["scores"][best_scalar_key],
            },
            "stump": evaluate_tree(rows, 1, 20260780 + model_index * 2),
            "depth2_tree": evaluate_tree(rows, 2, 20260781 + model_index * 2),
        }

    output = Path("outputs/shallow_tree_results.json")
    write_json(output, result)
    print(f"Wrote {output}")
    for model, values in result["models"].items():
        scalar = values["selected_scalar"]
        tree = values["depth2_tree"]
        print(
            f"{model}: calibration-selected scalar {scalar['score_key']} "
            f"test={scalar['test_auroc']:.3f}; depth-2 tree test={tree['test_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
