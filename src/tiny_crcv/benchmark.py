"""Command-line benchmark for the tiny CRCV hallucination detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiny_crcv.core import (
    answer_matches,
    auroc,
    bootstrap_auroc_difference_ci,
    bootstrap_auroc_ci,
    calibrate_threshold,
    compute_features,
    confusion_counts,
    macro_f1,
)
from tiny_crcv.model import WhiteBoxGenerator


SCORES = {
    "crcv_mean": "CRCV (primary)",
    "crcv_max": "CRCV maximum",
    "mean_nll": "Mean token surprise",
    "confidence_variance_mean": "Confidence variability",
    "shift_variance_mean": "Hidden-shift variability",
    "answer_tokens": "Answer length",
    "top3_token_surprise": "Top-3 token surprise",
    "worst_token_surprise": "Worst-token surprise",
    "surprise_spread": "Token-surprise spread",
}


def optional_auroc(labels: list[int], scores: list[float]) -> float | None:
    return auroc(labels, scores) if len(set(labels)) == 2 else None


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique")
    if {row["split"] for row in rows} != {"calibration", "test"}:
        raise ValueError("selected questions must contain calibration and test rows")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def evaluate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = [row for row in predictions if row["split"] == "calibration"]
    test = [row for row in predictions if row["split"] == "test"]
    results: dict[str, Any] = {}

    for seed_offset, (score_key, display_name) in enumerate(SCORES.items()):
        calibration_labels = [int(row["is_hallucination"]) for row in calibration]
        calibration_scores = [float(row["features"][score_key]) for row in calibration]
        test_labels = [int(row["is_hallucination"]) for row in test]
        test_scores = [float(row["features"][score_key]) for row in test]

        threshold = calibrate_threshold(calibration_labels, calibration_scores)
        test_predictions = [int(score >= threshold.threshold) for score in test_scores]
        test_auroc = optional_auroc(test_labels, test_scores)
        if test_auroc is None:
            ci_low, ci_high = None, None
        else:
            ci_low, ci_high = bootstrap_auroc_ci(
                test_labels,
                test_scores,
                seed=20260718 + seed_offset,
            )
        results[score_key] = {
            "display_name": display_name,
            "direction": "higher score predicts an incorrect answer",
            "calibration_auroc": optional_auroc(calibration_labels, calibration_scores),
            "calibration_macro_f1": threshold.macro_f1,
            "threshold": threshold.threshold,
            "test_auroc": test_auroc,
            "test_auroc_ci_95": [ci_low, ci_high],
            "test_macro_f1": macro_f1(test_labels, test_predictions),
            "test_confusion": confusion_counts(test_labels, test_predictions),
        }

    comparison = None
    improvement_comparison = None
    test_labels = [int(row["is_hallucination"]) for row in test]
    if len(set(test_labels)) == 2:
        difference, difference_low, difference_high = bootstrap_auroc_difference_ci(
            test_labels,
            [float(row["features"]["crcv_mean"]) for row in test],
            [float(row["features"]["confidence_variance_mean"]) for row in test],
        )
        comparison = {
            "name": "primary CRCV minus confidence variability",
            "test_auroc_difference": difference,
            "test_auroc_difference_ci_95": [difference_low, difference_high],
        }
        previous_best_key = max(
            (
                "crcv_mean",
                "crcv_max",
                "mean_nll",
                "confidence_variance_mean",
                "shift_variance_mean",
                "answer_tokens",
            ),
            key=lambda key: float(results[key]["test_auroc"]),
        )
        improvement, improvement_low, improvement_high = bootstrap_auroc_difference_ci(
            test_labels,
            [float(row["features"]["top3_token_surprise"]) for row in test],
            [float(row["features"][previous_best_key]) for row in test],
            seed=20260719,
        )
        improvement_comparison = {
            "name": f"top-3 token surprise minus previous-best {SCORES[previous_best_key]}",
            "baseline_score_key": previous_best_key,
            "test_auroc_difference": improvement,
            "test_auroc_difference_ci_95": [improvement_low, improvement_high],
        }

    return {
        "label_definition": "1 = generated answer did not contain any normalized gold alias",
        "calibration_examples": len(calibration),
        "calibration_hallucinations": sum(row["is_hallucination"] for row in calibration),
        "test_examples": len(test),
        "test_hallucinations": sum(row["is_hallucination"] for row in test),
        "scores": results,
        "paired_comparison": comparison,
        "improvement_comparison": improvement_comparison,
    }


def interpretation(metrics: dict[str, Any]) -> str:
    candidate = metrics["scores"]["top3_token_surprise"]
    value = candidate["test_auroc"]
    low, high = candidate["test_auroc_ci_95"]
    if value is None:
        return "Top-3 token surprise AUROC is undefined because the held-out split contains only one label class."
    if high < 0.5:
        finding = "Top-3 token surprise was inversely associated with wrong answers on this test."
    elif low > 0.5:
        finding = "Top-3 token surprise separated wrong from correct answers better than chance on this test."
    else:
        finding = "Top-3 token surprise did not demonstrate better-than-chance separation on this small test."
    return f"{finding} Its held-out AUROC was {value:.3f} (bootstrap 95% CI {low:.3f}-{high:.3f})."


def render_report(metadata: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Tiny CRCV benchmark report",
        "",
        interpretation(metrics),
        "",
        "## Setup",
        "",
        f"- Model: `{metadata['model_id']}` at revision `{metadata['resolved_revision']}`",
        f"- Device: `{metadata['device']}`; greedy decoding; at most {metadata['max_new_tokens']} answer tokens",
        f"- Questions: {metrics['calibration_examples']} calibration + {metrics['test_examples']} held out",
        f"- Window: {metadata['window']} generated steps; hidden layer: {metadata['layer']}",
        "- Operational label: an answer is wrong when no normalized accepted alias appears in the generated sentence",
        "",
        "## Held-out results",
        "",
        "| Score | AUROC | Bootstrap 95% CI | Macro-F1 | Calibration threshold |",
        "|---|---:|---:|---:|---:|",
    ]
    for score_key in SCORES:
        result = metrics["scores"][score_key]
        low, high = result["test_auroc_ci_95"]
        auroc_text = "n/a" if result["test_auroc"] is None else f"{result['test_auroc']:.3f}"
        ci_text = "n/a" if low is None else f"{low:.3f}-{high:.3f}"
        lines.append(
            f"| {result['display_name']} | {auroc_text} | "
            f"{ci_text} | {result['test_macro_f1']:.3f} | "
            f"{result['threshold']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"The held-out set contained {metrics['test_hallucinations']} incorrect and "
            f"{metrics['test_examples'] - metrics['test_hallucinations']} correct generations.",
            "AUROC uses the predeclared direction 'higher score = more likely wrong'; 0.5 is chance.",
            "Thresholds were chosen only on the calibration split by maximum macro-F1.",
            "",
            (
                "Primary CRCV minus confidence-variability AUROC: "
                f"{metrics['paired_comparison']['test_auroc_difference']:.3f} "
                f"(paired bootstrap 95% CI "
                f"{metrics['paired_comparison']['test_auroc_difference_ci_95'][0]:.3f} to "
                f"{metrics['paired_comparison']['test_auroc_difference_ci_95'][1]:.3f})."
                if metrics["paired_comparison"] is not None
                else "The paired AUROC comparison is undefined because the test split has one class."
            ),
            (
                "Top-3 token surprise minus the previous-best original-score AUROC: "
                f"{metrics['improvement_comparison']['test_auroc_difference']:.3f} "
                f"(paired bootstrap 95% CI "
                f"{metrics['improvement_comparison']['test_auroc_difference_ci_95'][0]:.3f} to "
                f"{metrics['improvement_comparison']['test_auroc_difference_ci_95'][1]:.3f})."
                if metrics["improvement_comparison"] is not None
                else "The improvement comparison is undefined because the test split has one class."
            ),
            "",
            "## What this establishes",
            "",
            "This is an exploratory follow-up on one tiny model, not a validated detector or an untouched confirmatory test.",
            "The saved predictions include each answer, gold aliases, token confidences, hidden-state shifts, and aggregate scores.",
            "The largest limitations are the 50-question held-out sample, one model/layer, greedy decoding, and lexical correctness labels.",
            "A follow-up should manually audit label errors, repeat across seeds/models, and use a larger human-aligned benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def render_error_audit(predictions: list[dict[str, Any]]) -> str:
    wrong = [row for row in predictions if row["is_hallucination"]]
    lines = [
        "# Incorrect-answer audit",
        "",
        "These are all generations labeled incorrect by the normalized alias rule.",
        "Review this table before interpreting detector metrics.",
        "",
        "| ID | Split | Question | Accepted | Generated answer |",
        "|---|---|---|---|---|",
    ]
    for row in wrong:
        cells = [
            row["id"],
            row["split"],
            row["question"],
            "; ".join(row["answers"]),
            row["generated_answer"],
        ]
        escaped = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen05b_100"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--prompt-style", choices=["instruct", "base"], default="instruct")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window < 2:
        raise SystemExit("--window must be at least 2")

    questions = load_questions(args.questions, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.overwrite:
        raise SystemExit(f"{predictions_path} exists; pass --overwrite to replace it")

    generator = WhiteBoxGenerator(
        args.model,
        revision=args.revision,
        device=args.device,
        layer=args.layer,
        prompt_style=args.prompt_style,
    )

    import torch
    import transformers

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": generator.resolved_revision,
        "device": generator.device,
        "layer": args.layer,
        "window": args.window,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "prompt_style": args.prompt_style,
        "question_file": str(args.questions),
        "question_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    write_json(args.output_dir / "metadata.json", metadata)

    predictions: list[dict[str, Any]] = []
    with predictions_path.open("w") as output_file:
        for index, question in enumerate(questions, start=1):
            trace = generator.generate(question["question"], max_new_tokens=args.max_new_tokens)
            correct = answer_matches(trace.answer, question["answers"])
            row = {
                **question,
                "generated_answer": trace.answer,
                "correct": correct,
                "is_hallucination": int(not correct),
                "token_ids": trace.token_ids,
                "token_pieces": trace.token_pieces,
                "confidences": trace.confidences,
                "hidden_shifts": trace.shifts,
                "features": compute_features(
                    trace.confidences,
                    trace.shifts,
                    window=args.window,
                ),
                "elapsed_seconds": trace.elapsed_seconds,
            }
            predictions.append(row)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{index:03d}/{len(questions):03d}] {question['id']} "
                f"{'correct' if correct else 'WRONG'} {trace.elapsed_seconds:.2f}s: {trace.answer}",
                flush=True,
            )

    metrics = evaluate(predictions)
    latencies = [row["elapsed_seconds"] for row in predictions]
    metrics["runtime"] = {
        "total_generation_seconds": sum(latencies),
        "mean_seconds_per_question": statistics.fmean(latencies),
        "median_seconds_per_question": statistics.median(latencies),
    }
    metrics["answers_at_token_limit"] = sum(
        row["features"]["answer_tokens"] == args.max_new_tokens for row in predictions
    )
    write_json(args.output_dir / "metrics.json", metrics)
    (args.output_dir / "report.md").write_text(render_report(metadata, metrics))
    (args.output_dir / "errors.md").write_text(render_error_audit(predictions))
    print("\n" + interpretation(metrics), flush=True)
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
