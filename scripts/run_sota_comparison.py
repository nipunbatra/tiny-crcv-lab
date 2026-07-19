"""Run the frozen 600-question comparison of four tiny hallucination scores."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tiny_crcv.core import (
    answer_matches,
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    bootstrap_auroc_difference_ci,
    calibrate_threshold,
    compute_features,
    confusion_counts,
    macro_f1,
    selective_accuracy,
)
from tiny_crcv.model import WhiteBoxGenerator
from tiny_crcv.probe import HiddenProbe, fit_hidden_probe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments/fresh_qa_600_protocol.json"
DEFAULT_DATA = ROOT / "data/fresh_qa_600.jsonl"
METHODS = {
    "top3_token_surprise": {
        "display_name": "Top-3 token surprise",
        "formula": "mean(largest 3 values of -ln p(chosen greedy token))",
    },
    "p_true": {
        "display_name": "P(False self-check)",
        "formula": "p(No) / (p(Yes) + p(No)) under the fixed correctness prompt",
    },
    "hidden_logistic_probe": {
        "display_name": "Mean-hidden linear probe",
        "formula": "sigmoid(b + sum_j w_j × ((h_j - mean_j) / std_j))",
    },
    "semantic_disagreement_3": {
        "display_name": "Three-answer disagreement",
        "formula": "mean of three pairwise normalized p(No) same-answer judgments",
    },
    "answer_tokens": {
        "display_name": "Answer length (confound)",
        "formula": "number of tokens in the greedy answer",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    """Return a stable repo-relative path for relative or absolute CLI inputs."""
    return str(path.resolve().relative_to(ROOT))


def optional_auroc(labels: list[int], scores: list[float]) -> float | None:
    return auroc(labels, scores) if len(set(labels)) == 2 else None


def select_rows(
    rows: list[dict[str, Any]], limit_per_dataset_split: int | None
) -> list[dict[str, Any]]:
    if limit_per_dataset_split is None:
        return rows
    selected: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["source_dataset"], row["split"])
        if counts.get(key, 0) < limit_per_dataset_split:
            selected.append(row)
            counts[key] = counts.get(key, 0) + 1
    return selected


def trace_fields(trace: Any) -> dict[str, Any]:
    return {
        "generated_answer": trace.answer,
        "token_ids": trace.token_ids,
        "token_pieces": trace.token_pieces,
        "confidences": trace.confidences,
        "hidden_shifts": trace.shifts,
        "token_margins": trace.token_margins,
        "token_entropies": trace.token_entropies,
        "hidden_cosine_distances": trace.hidden_cosine_distances,
        "hidden_norms": trace.hidden_norms,
        "generation_seconds": trace.elapsed_seconds,
    }


def score_question(
    generator: WhiteBoxGenerator,
    question: dict[str, Any],
    *,
    row_index: int,
    generation: dict[str, Any],
    window: int,
) -> dict[str, Any]:
    prompt = question["question"]
    greedy = generator.generate(
        prompt, max_new_tokens=int(generation["max_new_tokens"])
    )
    correct = answer_matches(greedy.answer, question["answers"])
    features = compute_features(
        greedy.confidences,
        greedy.shifts,
        window=window,
        token_margins=greedy.token_margins,
        token_entropies=greedy.token_entropies,
        hidden_cosine_distances=greedy.hidden_cosine_distances,
        hidden_norms=greedy.hidden_norms,
    )

    correctness = generator.score_correctness(prompt, greedy.answer)
    samples = []
    for sample_index in range(int(generation["stochastic_samples"])):
        seed = int(generation["sampling_seed"]) + row_index * 10 + sample_index
        trace = generator.generate(
            prompt,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=True,
            seed=seed,
            temperature=float(generation["sampling_temperature"]),
            top_k=int(generation["sampling_top_k"]),
            top_p=float(generation["sampling_top_p"]),
        )
        samples.append(
            {
                "sample_index": sample_index,
                "seed": seed,
                "answer": trace.answer,
                "elapsed_seconds": trace.elapsed_seconds,
            }
        )

    pair_judgments = []
    for left, right in itertools.combinations(range(len(samples)), 2):
        judgment = generator.score_equivalence(
            prompt, samples[left]["answer"], samples[right]["answer"]
        )
        pair_judgments.append(
            {
                "sample_a": left,
                "sample_b": right,
                "yes_probability": judgment.yes_probability,
                "no_probability": judgment.no_probability,
                "normalized_no_probability": judgment.normalized_no_probability,
                "yes_token_id": judgment.yes_token_id,
                "no_token_id": judgment.no_token_id,
                "elapsed_seconds": judgment.elapsed_seconds,
            }
        )
    disagreement = statistics.fmean(
        item["normalized_no_probability"] for item in pair_judgments
    )

    return {
        **question,
        "row_index": row_index,
        **trace_fields(greedy),
        "correct": correct,
        "is_hallucination": int(not correct),
        "features": features,
        "p_true_judgment": {
            "prompt_question": prompt,
            "proposed_answer": greedy.answer,
            "yes_probability": correctness.yes_probability,
            "no_probability": correctness.no_probability,
            "normalized_no_probability": correctness.normalized_no_probability,
            "yes_token_id": correctness.yes_token_id,
            "no_token_id": correctness.no_token_id,
            "elapsed_seconds": correctness.elapsed_seconds,
        },
        "stochastic_answers": samples,
        "pair_judgments": pair_judgments,
        "method_scores": {
            "top3_token_surprise": features["top3_token_surprise"],
            "p_true": correctness.normalized_no_probability,
            "semantic_disagreement_3": disagreement,
            "answer_tokens": features["answer_tokens"],
        },
        "method_seconds": {
            "top3_token_surprise": 0.0,
            "p_true": correctness.elapsed_seconds,
            "semantic_disagreement_3": sum(
                item["elapsed_seconds"] for item in samples
            )
            + sum(item["elapsed_seconds"] for item in pair_judgments),
            "answer_tokens": 0.0,
        },
        "_mean_hidden": greedy.mean_hidden,
    }


def fit_and_attach_probe(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[HiddenProbe, float, float]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    spec = protocol["methods"]["hidden_logistic_probe"]
    started = time.perf_counter()
    probe = fit_hidden_probe(
        [row["_mean_hidden"] for row in calibration],
        [row["is_hallucination"] for row in calibration],
        seed=int(spec["seed"]),
        learning_rate=0.03,
        steps=800,
        weight_decay=0.01,
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    risks = probe.predict_risk([row["_mean_hidden"] for row in rows])
    for row, risk in zip(rows, risks, strict=True):
        row["method_scores"]["hidden_logistic_probe"] = float(risk)
        row["probe_explanation"] = probe.explain(row["_mean_hidden"])
    score_seconds = time.perf_counter() - started
    per_row_seconds = score_seconds / len(rows)
    for row in rows:
        row["method_seconds"]["hidden_logistic_probe"] = per_row_seconds
    return probe, fit_seconds, score_seconds


def metric_slice(
    rows: list[dict[str, Any]],
    method: str,
    threshold: float,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    labels = [int(row["is_hallucination"]) for row in rows]
    scores = [float(row["method_scores"][method]) for row in rows]
    predictions = [int(score >= threshold) for score in scores]
    area = optional_auroc(labels, scores)
    if area is None:
        interval: list[float | None] = [None, None]
    else:
        interval = list(
            bootstrap_auroc_ci(
                labels,
                scores,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )
    return {
        "examples": len(rows),
        "incorrect": sum(labels),
        "correct": len(labels) - sum(labels),
        "auroc": area,
        "auroc_ci_95": interval,
        "average_precision": average_precision(labels, scores) if sum(labels) else None,
        "macro_f1": macro_f1(labels, predictions),
        "confusion": confusion_counts(labels, predictions),
        "selective_accuracy": {
            str(coverage): selective_accuracy(labels, scores, coverage)
            for coverage in (0.9, 0.8, 0.7)
        },
    }


def evaluate(
    rows: list[dict[str, Any]], protocol: dict[str, Any], probe_times: tuple[float, float]
) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    bootstrap_samples = int(protocol["evaluation"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["evaluation"]["bootstrap_seed"])
    test_labels = [int(row["is_hallucination"]) for row in test]
    baseline_scores = [
        float(row["method_scores"]["top3_token_surprise"]) for row in test
    ]
    results: dict[str, Any] = {}

    for method_index, (method, description) in enumerate(METHODS.items()):
        calibration_labels = [int(row["is_hallucination"]) for row in calibration]
        calibration_scores = [
            float(row["method_scores"][method]) for row in calibration
        ]
        threshold = calibrate_threshold(calibration_labels, calibration_scores)
        held_out = metric_slice(
            test,
            method,
            threshold.threshold,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + method_index,
        )
        by_dataset = {}
        for dataset_index, dataset in enumerate(
            sorted({row["source_dataset"] for row in test})
        ):
            dataset_rows = [row for row in test if row["source_dataset"] == dataset]
            by_dataset[dataset] = metric_slice(
                dataset_rows,
                method,
                threshold.threshold,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 100 + method_index * 10 + dataset_index,
            )

        paired = None
        if method != "top3_token_surprise" and len(set(test_labels)) == 2:
            difference, low, high = bootstrap_auroc_difference_ci(
                test_labels,
                [float(row["method_scores"][method]) for row in test],
                baseline_scores,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 200 + method_index,
            )
            paired = {
                "auroc_difference": difference,
                "auroc_difference_ci_95": [low, high],
                "excludes_zero": low > 0 or high < 0,
                "reliably_improves": low > 0,
            }
        seconds = [float(row["method_seconds"][method]) for row in rows]
        results[method] = {
            **description,
            "direction": "higher means more likely incorrect",
            "threshold": threshold.threshold,
            "calibration": {
                "examples": len(calibration),
                "incorrect": sum(calibration_labels),
                "auroc": optional_auroc(calibration_labels, calibration_scores),
                "macro_f1": threshold.macro_f1,
            },
            "held_out": held_out,
            "by_dataset": by_dataset,
            "paired_vs_top3_token_surprise": paired,
            "incremental_runtime": {
                "mean_seconds_per_question": statistics.fmean(seconds),
                "median_seconds_per_question": statistics.median(seconds),
                "total_seconds": sum(seconds),
            },
        }

    held_out_areas = {
        key: value["held_out"]["auroc"]
        for key, value in results.items()
        if value["held_out"]["auroc"] is not None and key != "answer_tokens"
    }
    descriptive_best = max(held_out_areas, key=held_out_areas.get)
    reliable_improvements = [
        key
        for key, value in results.items()
        if value["paired_vs_top3_token_surprise"]
        and value["paired_vs_top3_token_surprise"]["reliably_improves"]
    ]
    return {
        "label_definition": "1 = greedy answer contains no normalized accepted answer alias",
        "calibration_examples": len(calibration),
        "held_out_examples": len(test),
        "calibration_incorrect": sum(row["is_hallucination"] for row in calibration),
        "held_out_incorrect": sum(row["is_hallucination"] for row in test),
        "methods": results,
        "probe_training": {
            "fit_seconds": probe_times[0],
            "scoring_seconds_all_questions": probe_times[1],
            "calibration_only": True,
        },
        "descriptive_best_non_confound": descriptive_best,
        "reliable_improvements_over_top3": reliable_improvements,
        "interpretation_rule": protocol["interpretation_rule"],
    }


def render_report(metadata: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Fresh 600-question tiny-detector comparison",
        "",
        f"Model: `{metadata['model_id']}` at `{metadata['resolved_revision']}`.",
        f"Data: {metrics['calibration_examples']} calibration + "
        f"{metrics['held_out_examples']} held out across NQ-Open, TriviaQA, and TruthfulQA.",
        "",
        "| Method | Held-out AUROC | 95% CI | AP | Macro-F1 | Mean extra seconds/question |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in metrics["methods"].items():
        held = result["held_out"]
        low, high = held["auroc_ci_95"]
        lines.append(
            f"| {result['display_name']} | {held['auroc']:.3f} | {low:.3f}–{high:.3f} | "
            f"{held['average_precision']:.3f} | {held['macro_f1']:.3f} | "
            f"{result['incremental_runtime']['mean_seconds_per_question']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Descriptively highest non-confound score: "
            f"**{metrics['methods'][metrics['descriptive_best_non_confound']]['display_name']}**.",
            (
                "Methods with a paired 95% AUROC-difference interval above zero versus "
                "top-3 surprise: " + ", ".join(metrics["reliable_improvements_over_top3"])
                if metrics["reliable_improvements_over_top3"]
                else "No method met the predeclared reliable-improvement rule versus top-3 surprise."
            ),
            "",
            "These are reference-free risk signals, not factual verifiers. Labels use transparent "
            "alias matching and require manual audit for paraphrases and dataset ambiguity.",
            "Thresholds were fit on pooled calibration data and frozen for all held-out slices.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-key",
        required=True,
        help="Model key declared in the selected protocol.",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-per-dataset-split", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(args.protocol)
    try:
        model_spec = next(
            item for item in protocol["models"] if item["key"] == args.model_key
        )
    except StopIteration as error:
        available = ", ".join(item["key"] for item in protocol["models"])
        raise ValueError(
            f"model key {args.model_key!r} is not in the protocol; choose from {available}"
        ) from error
    legacy_output_name = (
        f"fresh_qa_qwen05b_{args.model_key}"
        if args.protocol.resolve() == DEFAULT_PROTOCOL.resolve()
        else f"fresh_qa_{args.model_key}"
    )
    output_name = model_spec.get("output_name", legacy_output_name)
    output_dir = args.output_dir or ROOT / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    final_paths = [
        output_dir / "predictions.jsonl",
        output_dir / "metrics.json",
        output_dir / "metadata.json",
        output_dir / "hidden_probe.npz",
        output_dir / "report.md",
    ]
    if args.overwrite:
        for path in [progress_path, *final_paths]:
            path.unlink(missing_ok=True)
    elif (output_dir / "predictions.jsonl").exists():
        raise SystemExit(f"{output_dir} is complete; pass --overwrite to replace it")

    questions = select_rows(read_jsonl(args.data), args.limit_per_dataset_split)
    expected_splits = {row["split"] for row in questions}
    if expected_splits != {"calibration", "test"}:
        raise ValueError("selected rows must contain calibration and held-out examples")
    generator = WhiteBoxGenerator(
        model_spec["id"],
        revision=model_spec.get("revision", "main"),
        device=args.device,
        prompt_style=model_spec["prompt_style"],
    )
    import torch
    import transformers

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": repo_relative(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "data": repo_relative(args.data),
        "data_sha256": sha256(args.data),
        "model_key": args.model_key,
        "model_id": model_spec["id"],
        "requested_revision": model_spec.get("revision", "main"),
        "resolved_revision": generator.resolved_revision,
        "prompt_style": model_spec["prompt_style"],
        "device": generator.device,
        "window": args.window,
        "generation": protocol["generation"],
        "rows": len(questions),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    write_json(output_dir / "metadata.json", metadata)

    saved = read_jsonl(progress_path)
    by_id = {row["id"]: row for row in saved}
    if len(saved) != len(by_id):
        raise ValueError("progress file contains duplicate question IDs")
    mode = "a" if saved else "w"
    with progress_path.open(mode) as progress:
        for row_index, question in enumerate(questions):
            if question["id"] in by_id:
                continue
            result = score_question(
                generator,
                question,
                row_index=row_index,
                generation=protocol["generation"],
                window=args.window,
            )
            by_id[result["id"]] = result
            progress.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.flush()
            print(
                f"[{len(by_id):03d}/{len(questions):03d}] {result['id']} "
                f"{'correct' if result['correct'] else 'WRONG'}: {result['generated_answer']}",
                flush=True,
            )

    rows = [by_id[question["id"]] for question in questions]
    probe, fit_seconds, score_seconds = fit_and_attach_probe(rows, protocol)
    np.savez_compressed(
        output_dir / "hidden_probe.npz",
        ids=np.asarray([row["id"] for row in rows]),
        mean_hidden=np.asarray([row["_mean_hidden"] for row in rows], dtype=np.float32),
        calibration_mean=probe.mean,
        calibration_scale=probe.scale,
        weights=probe.weights,
        bias=np.asarray([probe.bias], dtype=np.float32),
    )

    metrics = evaluate(rows, protocol, (fit_seconds, score_seconds))
    metrics["runtime"] = {
        "mean_greedy_generation_seconds": statistics.fmean(
            row["generation_seconds"] for row in rows
        ),
        "total_greedy_generation_seconds": sum(row["generation_seconds"] for row in rows),
        "answers_at_token_limit": sum(
            row["features"]["answer_tokens"]
            == int(protocol["generation"]["max_new_tokens"])
            for row in rows
        ),
    }
    write_json(output_dir / "metrics.json", metrics)
    public_rows = []
    for row in rows:
        public_row = {key: value for key, value in row.items() if key != "_mean_hidden"}
        public_rows.append(public_row)
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_rows)
    )
    (output_dir / "report.md").write_text(render_report(metadata, metrics))
    progress_path.unlink()
    print(f"Complete: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
