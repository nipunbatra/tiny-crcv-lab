"""Score the frozen HaluEval-QA candidate subset and evaluate tiny detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tiny_crcv.benchmark import SCORES, evaluate, write_json
from tiny_crcv.core import (
    auroc,
    bootstrap_auroc_ci,
    calibrate_threshold,
    compute_features,
    confusion_counts,
    macro_f1,
)
from tiny_crcv.model import WhiteBoxGenerator
from tiny_crcv.tree import TreeNode, fit_tree


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def paired_accuracy(
    rows: list[dict[str, Any]], score: Callable[[dict[str, Any]], float], split: str
) -> float:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["split"] == split:
            grouped.setdefault(row["pair_id"], []).append(row)
    credits: list[float] = []
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {row["is_hallucination"] for row in pair} != {0, 1}:
            raise ValueError(f"{pair_id} is not one right/hallucinated pair")
        positive = next(row for row in pair if row["is_hallucination"] == 1)
        negative = next(row for row in pair if row["is_hallucination"] == 0)
        positive_score = score(positive)
        negative_score = score(negative)
        credits.append(
            1.0
            if positive_score > negative_score
            else 0.5
            if positive_score == negative_score
            else 0.0
        )
    return statistics.fmean(credits)


def tree_result(
    rows: list[dict[str, Any]], tree: TreeNode, *, seed: int
) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    calibration_labels = [int(row["is_hallucination"]) for row in calibration]
    test_labels = [int(row["is_hallucination"]) for row in test]
    calibration_scores = [tree.predict(row["features"]) for row in calibration]
    test_scores = [tree.predict(row["features"]) for row in test]
    threshold = calibrate_threshold(calibration_labels, calibration_scores)
    predictions = [int(value >= threshold.threshold) for value in test_scores]
    test_auroc = auroc(test_labels, test_scores)
    return {
        "calibration_auroc": auroc(calibration_labels, calibration_scores),
        "calibration_macro_f1": threshold.macro_f1,
        "threshold": threshold.threshold,
        "test_auroc": test_auroc,
        "test_auroc_ci_95": list(
            bootstrap_auroc_ci(test_labels, test_scores, seed=seed)
        ),
        "test_macro_f1": macro_f1(test_labels, predictions),
        "test_confusion": confusion_counts(test_labels, predictions),
        "calibration_pairwise_accuracy": paired_accuracy(
            rows, lambda row: tree.predict(row["features"]), "calibration"
        ),
        "test_pairwise_accuracy": paired_accuracy(
            rows, lambda row: tree.predict(row["features"]), "test"
        ),
        "rules": tree.to_dict(),
    }


def length_residual_diagnostic(
    rows: list[dict[str, Any]], score_key: str
) -> dict[str, Any]:
    """Post-hoc: remove a calibration-fitted linear log-length trend."""
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    x_values = [math.log1p(float(row["features"]["answer_tokens"])) for row in calibration]
    y_values = [float(row["features"][score_key]) for row in calibration]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean

    def residual(row: dict[str, Any]) -> float:
        expected = intercept + slope * math.log1p(
            float(row["features"]["answer_tokens"])
        )
        return float(row["features"][score_key]) - expected

    calibration_labels = [int(row["is_hallucination"]) for row in calibration]
    test_labels = [int(row["is_hallucination"]) for row in test]
    calibration_scores = [residual(row) for row in calibration]
    test_scores = [residual(row) for row in test]
    threshold = calibrate_threshold(calibration_labels, calibration_scores)
    test_predictions = [int(score >= threshold.threshold) for score in test_scores]
    return {
        "status": "post-hoc confound diagnostic; not predeclared",
        "score_key": score_key,
        "length_transform": "ln(1 + answer_tokens)",
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_auroc": auroc(calibration_labels, calibration_scores),
        "threshold": threshold.threshold,
        "test_auroc": auroc(test_labels, test_scores),
        "test_auroc_ci_95": list(
            bootstrap_auroc_ci(test_labels, test_scores, seed=20260772)
        ),
        "test_macro_f1": macro_f1(test_labels, test_predictions),
        "test_confusion": confusion_counts(test_labels, test_predictions),
        "calibration_pairwise_accuracy": paired_accuracy(
            rows, residual, "calibration"
        ),
        "test_pairwise_accuracy": paired_accuracy(rows, residual, "test"),
    }


def render_report(metrics: dict[str, Any]) -> str:
    best = metrics["selected_scalar"]
    tree = metrics["depth2_tree"]
    stump = metrics["stump"]
    length = metrics["scores"]["answer_tokens"]
    residual = metrics["length_diagnostics"]["selected_scalar_residual"]
    return "\n".join(
        [
            "# HaluEval-QA 50-pair external benchmark",
            "",
            "This frozen subset contains 25 calibration and 25 held-out question pairs. "
            "Each pair contributes the supplied right and hallucinated candidate answer.",
            "The model teacher-forces each candidate under HaluEval's supplied knowledge; "
            "this is grounded candidate discrimination, not free generation.",
            "",
            "## Held-out results",
            "",
            "| Detector | Selection | AUROC | 95% CI | Macro-F1 | Pairwise accuracy |",
            "|---|---|---:|---:|---:|---:|",
            f"| {best['display_name']} | best scalar on calibration | "
            f"{best['test_auroc']:.3f} | {best['test_auroc_ci_95'][0]:.3f}–"
            f"{best['test_auroc_ci_95'][1]:.3f} | {best['test_macro_f1']:.3f} | "
            f"{best['test_pairwise_accuracy']:.3f} |",
            f"| Decision stump | diagnostic | {stump['test_auroc']:.3f} | "
            f"{stump['test_auroc_ci_95'][0]:.3f}–{stump['test_auroc_ci_95'][1]:.3f} | "
            f"{stump['test_macro_f1']:.3f} | {stump['test_pairwise_accuracy']:.3f} |",
            f"| Depth-2 tree | predeclared primary tree | {tree['test_auroc']:.3f} | "
            f"{tree['test_auroc_ci_95'][0]:.3f}–{tree['test_auroc_ci_95'][1]:.3f} | "
            f"{tree['test_macro_f1']:.3f} | {tree['test_pairwise_accuracy']:.3f} |",
            f"| Answer length | confound baseline | {length['test_auroc']:.3f} | "
            f"{length['test_auroc_ci_95'][0]:.3f}–{length['test_auroc_ci_95'][1]:.3f} | "
            f"{length['test_macro_f1']:.3f} | {length['test_pairwise_accuracy']:.3f} |",
            f"| {best['display_name']} after log-length residualization | post-hoc diagnostic | "
            f"{residual['test_auroc']:.3f} | {residual['test_auroc_ci_95'][0]:.3f}–"
            f"{residual['test_auroc_ci_95'][1]:.3f} | {residual['test_macro_f1']:.3f} | "
            f"{residual['test_pairwise_accuracy']:.3f} |",
            "",
            "Pairwise accuracy asks whether the hallucinated candidate receives a higher "
            "risk score than the right candidate for the same question; ties receive half credit.",
            f"In the held-out split, the hallucinated candidate was longer in "
            f"{metrics['length_diagnostics']['test_pairs_hallucinated_longer']}/25 pairs. "
            "The residualized row is a post-hoc diagnostic, not a new confirmatory result.",
            "",
            "## Limits",
            "",
            "- Only 25 held-out question pairs are used, so uncertainty is wide.",
            "- HaluEval hallucinated answers are synthetic and often longer than right answers.",
            "- Teacher-forced candidate discrimination is not the same task as detecting errors "
            "in the model's own free generation.",
            "- The tree has only two levels and six predeclared inputs, but 50 calibration "
            "candidates is still a very small training set.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/halueval_qa_50.jsonl"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/halueval_qa_50_protocol.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/halueval_qwen05b_100")
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.overwrite:
        raise SystemExit(f"{predictions_path} exists; pass --overwrite to replace it")

    source_rows = load_jsonl(args.data)
    protocol = json.loads(args.protocol.read_text())
    if len(source_rows) != 100:
        raise ValueError("the frozen protocol requires exactly 100 candidate rows")
    generator = WhiteBoxGenerator(
        args.model,
        revision=args.revision,
        device=args.device,
        layer=args.layer,
        prompt_style="instruct",
    )

    predictions: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        trace = generator.score_candidate(
            source["knowledge"], source["question"], source["candidate_answer"]
        )
        features = compute_features(
            trace.confidences,
            trace.shifts,
            window=args.window,
            token_margins=trace.token_margins,
            token_entropies=trace.token_entropies,
            hidden_cosine_distances=trace.hidden_cosine_distances,
            hidden_norms=trace.hidden_norms,
        )
        row = {
            **source,
            "token_ids": trace.token_ids,
            "token_pieces": trace.token_pieces,
            "confidences": trace.confidences,
            "hidden_shifts": trace.shifts,
            "token_margins": trace.token_margins,
            "token_entropies": trace.token_entropies,
            "hidden_cosine_distances": trace.hidden_cosine_distances,
            "hidden_norms": trace.hidden_norms,
            "features": features,
            "elapsed_seconds": trace.elapsed_seconds,
        }
        predictions.append(row)
        print(
            f"[{index:03d}/{len(source_rows):03d}] {source['id']} "
            f"{trace.elapsed_seconds:.2f}s ({features['answer_tokens']} tokens)",
            flush=True,
        )

    metrics = evaluate(predictions)
    metrics["label_definition"] = "1 = HaluEval hallucinated candidate; 0 = HaluEval right candidate"
    for score_key in SCORES:
        metrics["scores"][score_key]["calibration_pairwise_accuracy"] = paired_accuracy(
            predictions, lambda row, key=score_key: float(row["features"][key]), "calibration"
        )
        metrics["scores"][score_key]["test_pairwise_accuracy"] = paired_accuracy(
            predictions, lambda row, key=score_key: float(row["features"][key]), "test"
        )

    best_scalar_key = max(
        SCORES,
        key=lambda key: (metrics["scores"][key]["calibration_auroc"], -list(SCORES).index(key)),
    )
    metrics["selected_scalar"] = {
        "score_key": best_scalar_key,
        **metrics["scores"][best_scalar_key],
    }

    def hallucinated_longer(split: str) -> int:
        pairs: dict[str, list[dict[str, Any]]] = {}
        for row in predictions:
            if row["split"] == split:
                pairs.setdefault(row["pair_id"], []).append(row)
        return sum(
            next(row for row in pair if row["is_hallucination"] == 1)["features"]["answer_tokens"]
            > next(row for row in pair if row["is_hallucination"] == 0)["features"]["answer_tokens"]
            for pair in pairs.values()
        )

    metrics["length_diagnostics"] = {
        "status": "post-hoc confound analysis",
        "calibration_pairs_hallucinated_longer": hallucinated_longer("calibration"),
        "test_pairs_hallucinated_longer": hallucinated_longer("test"),
        "selected_scalar_residual": length_residual_diagnostic(
            predictions, best_scalar_key
        ),
    }

    tree_features = protocol["tree_protocol"]["features"]
    calibration = [row for row in predictions if row["split"] == "calibration"]
    calibration_rows = [row["features"] for row in calibration]
    calibration_labels = [int(row["is_hallucination"]) for row in calibration]
    stump = fit_tree(
        calibration_rows,
        calibration_labels,
        features=tree_features,
        max_depth=1,
        min_leaf=8,
    )
    depth2 = fit_tree(
        calibration_rows,
        calibration_labels,
        features=tree_features,
        max_depth=2,
        min_leaf=8,
    )
    metrics["stump"] = tree_result(predictions, stump, seed=20260770)
    metrics["depth2_tree"] = tree_result(predictions, depth2, seed=20260771)
    metrics["tree_features"] = tree_features

    for row in predictions:
        row["tree_score"] = depth2.predict(row["features"])
        row["tree_path"] = depth2.path(row["features"])

    runtime = [row["elapsed_seconds"] for row in predictions]
    metrics["runtime"] = {
        "total_scoring_seconds": sum(runtime),
        "mean_seconds_per_candidate": statistics.fmean(runtime),
        "median_seconds_per_candidate": statistics.median(runtime),
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": generator.resolved_revision,
        "device": generator.device,
        "layer": args.layer,
        "window": args.window,
        "mode": "teacher_forced_grounded_candidate_scoring",
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "metadata.json", metadata)
    (args.output_dir / "report.md").write_text(render_report(metrics))
    print(render_report(metrics), flush=True)


if __name__ == "__main__":
    main()
