"""Evaluate the post-hoc fast-baseline extension on frozen fresh-QA traces."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from tiny_crcv.benchmark import SCORES
from tiny_crcv.core import (
    aurac,
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    bootstrap_auroc_difference_ci,
    calibrate_threshold,
    confusion_counts,
    macro_f1,
    selective_accuracy,
)
from tiny_crcv.fast_baselines import (
    TRACE_ENSEMBLE_FEATURES,
    discrete_semantic_entropy_3,
    explain_trace_logistic,
    fit_trace_logistic,
    lexical_disagreement_3,
    predict_trace_logistic,
    trace_feature_vector,
)
from tiny_crcv.tree import fit_tree


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/fresh_qa_600_extension_protocol.json"
MODEL_DIRS = {
    "instruct": ROOT / "outputs/fresh_qa_qwen05b_instruct",
    "base": ROOT / "outputs/fresh_qa_qwen05b_base",
}


SCALAR_FORMULAS = {
    "crcv_mean": "mean over complete trailing windows of sample-SD(p_t × r_t)",
    "crcv_max": "max over complete trailing windows of sample-SD(p_t × r_t)",
    "mean_nll": "mean_t(−ln p_t)",
    "confidence_variance_mean": "mean over trailing windows of sample-SD(p_t)",
    "shift_variance_mean": "mean over trailing windows of sample-SD(r_t)",
    "answer_tokens": "number of tokens in the greedy answer",
    "top3_token_surprise": "mean(largest 3 values of −ln p_t)",
    "worst_token_surprise": "max_t(−ln p_t)",
    "surprise_spread": "sample-SD_t(−ln p_t)",
    "top4_token_surprise": "mean(largest 4 values of −ln p_t)",
    "skip_first_top3_surprise": "mean(largest 3 values of −ln p_t for t > 1)",
    "surprise_shift_top3": "mean(largest 3 values of (−ln p_t) × r_t)",
    "uncertainty_shift_top3": "mean(largest 3 values of (1 − p_t) × r_t)",
    "token_entropy_top3": "mean(largest 3 normalized full-vocabulary entropies H_t)",
    "token_entropy_mean": "mean_t(H_t)",
    "token_entropy_max": "max_t(H_t)",
    "token_ambiguity_top3": "mean(largest 3 values of 1 − (p_(1),t − p_(2),t))",
    "token_ambiguity_mean": "mean_t(1 − (p_(1),t − p_(2),t))",
    "token_ambiguity_max": "max_t(1 − (p_(1),t − p_(2),t))",
    "hidden_cosine_mean": "mean_t(1 − cosine(h_t, h_(t−1)))",
    "hidden_cosine_max": "max_t(1 − cosine(h_t, h_(t−1)))",
    "hidden_cosine_top3": "mean(largest 3 values of 1 − cosine(h_t, h_(t−1)))",
    "hidden_norm_mean": "mean_t(RMS(h_t))",
    "hidden_norm_variability": "sample-SD_t(RMS(h_t))",
}

SCALAR_FAMILIES = {
    "crcv_mean": "CRCV",
    "crcv_max": "CRCV",
    "mean_nll": "token probability",
    "confidence_variance_mean": "token probability",
    "shift_variance_mean": "hidden dynamics",
    "answer_tokens": "control",
    "top3_token_surprise": "token probability",
    "worst_token_surprise": "token probability",
    "surprise_spread": "token probability",
    "top4_token_surprise": "token probability",
    "skip_first_top3_surprise": "token probability",
    "surprise_shift_top3": "coupled",
    "uncertainty_shift_top3": "coupled",
    "token_entropy_top3": "distribution",
    "token_entropy_mean": "distribution",
    "token_entropy_max": "distribution",
    "token_ambiguity_top3": "distribution",
    "token_ambiguity_mean": "distribution",
    "token_ambiguity_max": "distribution",
    "hidden_cosine_mean": "hidden dynamics",
    "hidden_cosine_max": "hidden dynamics",
    "hidden_cosine_top3": "hidden dynamics",
    "hidden_norm_mean": "hidden dynamics",
    "hidden_norm_variability": "hidden dynamics",
}

EXTRA_METHODS = {
    "p_true": {
        "display_name": "P(False self-check)",
        "formula": "p(No) / (p(Yes) + p(No)) under the fixed correctness prompt",
        "family": "P(True)-style self-evaluation",
        "implementation_note": "One extra pass through the same checkpoint.",
    },
    "hidden_logistic_probe": {
        "display_name": "Mean-hidden linear probe",
        "formula": "sigmoid(b + sum_j w_j × ((h_j − mean_j) / std_j))",
        "family": "supervised internal state",
        "implementation_note": "896 final-layer mean-hidden values; calibration-only fit.",
    },
    "semantic_disagreement_3": {
        "display_name": "Three-answer semantic disagreement",
        "formula": "mean of three pairwise normalized p(No) same-answer judgments",
        "family": "sample consistency",
        "implementation_note": "Three generations plus three same-model judge passes.",
    },
    "lexical_disagreement_3": {
        "display_name": "Three-answer lexical disagreement",
        "formula": "mean over pairs of 1 − Jaccard(normalized word-token sets)",
        "family": "SelfCheckGPT-inspired",
        "implementation_note": "Three generations; no judge passes.",
    },
    "discrete_semantic_entropy_3": {
        "display_name": "Three-sample discrete semantic entropy proxy",
        "formula": "−sum_c (n_c / 3) ln(n_c / 3) after same-answer clustering",
        "family": "semantic-entropy proxy",
        "implementation_note": "Budget proxy: 3 samples and same-answer judgments, not the paper's 10-sample bidirectional-entailment setup.",
    },
    "trace_logistic_8": {
        "display_name": "Eight-feature trace logistic",
        "formula": "sigmoid(b + sum_k w_k × ((x_k − mean_k) / std_k))",
        "family": "supervised trace ensemble",
        "implementation_note": "Eight fixed cheap features; calibration-only fit; no hyperparameter search.",
    },
    "trace_tree_depth2": {
        "display_name": "Depth-2 trace tree",
        "formula": "CART-style branches over eight fixed trace features; leaf wrong-rate is the risk",
        "family": "supervised trace ensemble",
        "implementation_note": "Maximum depth 2 and at least 20 calibration rows per leaf.",
    },
}

SCALAR_METHOD_ORDER = list(SCORES)
EXTRA_METHOD_ORDER = list(EXTRA_METHODS)
ALL_METHOD_ORDER = SCALAR_METHOD_ORDER + EXTRA_METHOD_ORDER
HEADLINE_METHOD_ORDER = [
    "top3_token_surprise",
    "crcv_mean",
    "mean_nll",
    "token_entropy_top3",
    "p_true",
    "lexical_disagreement_3",
    "discrete_semantic_entropy_3",
    "semantic_disagreement_3",
    "hidden_logistic_probe",
    "trace_logistic_8",
    "trace_tree_depth2",
    "answer_tokens",
]
ORIGINAL_FROZEN_METHODS = [
    "top3_token_surprise",
    "p_true",
    "hidden_logistic_probe",
    "semantic_disagreement_3",
    "answer_tokens",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def method_catalog(hidden_dimensions: int) -> dict[str, dict[str, str]]:
    catalog = {
        key: {
            "display_name": SCORES[key],
            "formula": SCALAR_FORMULAS[key],
            "family": SCALAR_FAMILIES[key],
            "implementation_note": "Computed from the existing greedy generation trace; no extra model pass.",
        }
        for key in SCALAR_METHOD_ORDER
    }
    catalog.update({key: dict(value) for key, value in EXTRA_METHODS.items()})
    catalog["hidden_logistic_probe"]["implementation_note"] = (
        f"{hidden_dimensions} final-layer mean-hidden values; calibration-only fit."
    )
    return catalog


def optional_auroc(labels: list[int], scores: list[float]) -> float | None:
    return auroc(labels, scores) if len(set(labels)) == 2 else None


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
    interval: list[float | None] = [None, None]
    if area is not None:
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
        "aurac": aurac(labels, scores),
        "selective_accuracy": {
            str(coverage): selective_accuracy(labels, scores, coverage)
            for coverage in (0.9, 0.8, 0.7)
        },
    }


def attach_scores(
    rows: list[dict[str, Any]],
    *,
    model: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    for row in rows:
        for key in SCALAR_METHOD_ORDER:
            row["method_scores"][key] = float(row["features"][key])
            row["method_seconds"][key] = 0.0

        samples = [item["answer"] for item in row["stochastic_answers"]]
        lexical = lexical_disagreement_3(samples)
        semantic_entropy = discrete_semantic_entropy_3(row["pair_judgments"])
        row["lexical_disagreement_explanation"] = lexical
        row["semantic_entropy_explanation"] = semantic_entropy
        row["method_scores"]["lexical_disagreement_3"] = lexical["score"]
        row["method_scores"]["discrete_semantic_entropy_3"] = semantic_entropy["score"]
        sample_seconds = sum(item["elapsed_seconds"] for item in row["stochastic_answers"])
        judge_seconds = sum(item["elapsed_seconds"] for item in row["pair_judgments"])
        row["method_seconds"]["lexical_disagreement_3"] = sample_seconds
        row["method_seconds"]["discrete_semantic_entropy_3"] = sample_seconds + judge_seconds

    calibration = [row for row in rows if row["split"] == "calibration"]
    probe = fit_trace_logistic(
        [row["features"] for row in calibration],
        [int(row["is_hallucination"]) for row in calibration],
        seed=seed,
    )
    started = time.perf_counter()
    risks = predict_trace_logistic(probe, [row["features"] for row in rows])
    logistic_scoring_seconds = time.perf_counter() - started
    for row, risk in zip(rows, risks, strict=True):
        row["method_scores"]["trace_logistic_8"] = float(risk)
        row["method_seconds"]["trace_logistic_8"] = logistic_scoring_seconds / len(rows)
        row["trace_logistic_explanation"] = explain_trace_logistic(
            probe, row["features"]
        )

    tree = fit_tree(
        [row["features"] for row in calibration],
        [int(row["is_hallucination"]) for row in calibration],
        features=TRACE_ENSEMBLE_FEATURES,
        max_depth=2,
        min_leaf=20,
    )
    started = time.perf_counter()
    tree_risks = [tree.predict(row["features"]) for row in rows]
    tree_scoring_seconds = time.perf_counter() - started
    for row, risk in zip(rows, tree_risks, strict=True):
        row["method_scores"]["trace_tree_depth2"] = float(risk)
        row["method_seconds"]["trace_tree_depth2"] = tree_scoring_seconds / len(rows)
        row["trace_tree_path"] = tree.path(row["features"])

    artifact = {
        "model": model,
        "feature_order": list(TRACE_ENSEMBLE_FEATURES),
        "logistic": {
            "calibration_mean": probe.mean.tolist(),
            "calibration_scale": probe.scale.tolist(),
            "weights": probe.weights.tolist(),
            "bias": probe.bias,
            "training_loss": probe.training_loss,
            "steps": probe.steps,
            "seed": seed,
        },
        "tree": tree.to_dict(),
    }
    timings = {
        "trace_logistic_scoring_seconds_all_600": logistic_scoring_seconds,
        "trace_tree_scoring_seconds_all_600": tree_scoring_seconds,
    }
    return artifact, timings


def evaluate(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    catalog: dict[str, dict[str, str]],
    timing: dict[str, float],
) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    bootstrap_samples = int(protocol["evaluation"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["evaluation"]["bootstrap_seed"])
    calibration_labels = [int(row["is_hallucination"]) for row in calibration]
    test_labels = [int(row["is_hallucination"]) for row in test]
    baseline_scores = [
        float(row["method_scores"]["top3_token_surprise"]) for row in test
    ]
    results: dict[str, Any] = {}

    for method_index, method in enumerate(ALL_METHOD_ORDER):
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
                bootstrap_seed=bootstrap_seed + 1000 + method_index * 10 + dataset_index,
            )

        paired = None
        if method != "top3_token_surprise" and len(set(test_labels)) == 2:
            difference, low, high = bootstrap_auroc_difference_ci(
                test_labels,
                [float(row["method_scores"][method]) for row in test],
                baseline_scores,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 2000 + method_index,
            )
            paired = {
                "auroc_difference": difference,
                "auroc_difference_ci_95": [low, high],
                "excludes_zero": low > 0 or high < 0,
                "reliably_improves": low > 0,
            }

        seconds = [float(row["method_seconds"][method]) for row in rows]
        results[method] = {
            **catalog[method],
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

    eligible = {
        key: results[key]["held_out"]["auroc"]
        for key in ALL_METHOD_ORDER
        if key != "answer_tokens" and results[key]["held_out"]["auroc"] is not None
    }
    best = max(eligible, key=eligible.get)
    improves = [
        key
        for key in ALL_METHOD_ORDER
        if results[key]["paired_vs_top3_token_surprise"]
        and results[key]["paired_vs_top3_token_surprise"]["reliably_improves"]
    ]
    return {
        "extension_status": protocol["status"],
        "label_definition": "1 = greedy answer contains no normalized accepted answer alias",
        "calibration_examples": len(calibration),
        "held_out_examples": len(test),
        "calibration_incorrect": sum(calibration_labels),
        "held_out_incorrect": sum(test_labels),
        "original_frozen_method_order": ORIGINAL_FROZEN_METHODS,
        "headline_method_order": HEADLINE_METHOD_ORDER,
        "scalar_method_order": SCALAR_METHOD_ORDER,
        "all_method_order": ALL_METHOD_ORDER,
        "trace_ensemble_features": list(TRACE_ENSEMBLE_FEATURES),
        "methods": results,
        "posthoc_descriptive_best_non_confound": best,
        "posthoc_paired_intervals_above_zero": improves,
        "interpretation": protocol["interpretation"],
        "extension_runtime": timing,
    }


def render_report(model: str, metrics: dict[str, Any]) -> str:
    lines = [
        "# Post-hoc fast-baseline extension",
        "",
        f"Checkpoint: **{model}**. This is exploratory because the held-out outputs were already inspected.",
        "",
        "## Headline methods",
        "",
        "| Method | Family | AUROC | 95% CI | AURAC | Macro-F1 | Extra s/question |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in metrics["headline_method_order"]:
        method = metrics["methods"][key]
        held = method["held_out"]
        low, high = held["auroc_ci_95"]
        lines.append(
            f"| {method['display_name']} | {method['family']} | {held['auroc']:.3f} | "
            f"{low:.3f}–{high:.3f} | {held['aurac']:.3f} | {held['macro_f1']:.3f} | "
            f"{method['incremental_runtime']['mean_seconds_per_question']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## All trace-only scalars",
            "",
            "| Scalar | Family | AUROC | 95% CI | Macro-F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for key in metrics["scalar_method_order"]:
        method = metrics["methods"][key]
        held = method["held_out"]
        low, high = held["auroc_ci_95"]
        lines.append(
            f"| {method['display_name']} | {method['family']} | {held['auroc']:.3f} | "
            f"{low:.3f}–{high:.3f} | {held['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Descriptively highest non-length method: `{metrics['posthoc_descriptive_best_non_confound']}`.",
            "Do not read that as confirmation or SOTA: model, task, label, sampling budget, and split differ from published studies.",
            "",
        ]
    )
    return "\n".join(lines)


def process_model(
    model: str,
    output_dir: Path,
    *,
    protocol: dict[str, Any],
    seed: int,
) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    frozen_metrics_path = output_dir / "metrics_frozen_five.json"
    if not frozen_metrics_path.exists():
        current = read_json(metrics_path)
        if "extension_status" in current:
            raise ValueError(
                f"{metrics_path} is already extended but {frozen_metrics_path} is absent"
            )
        write_json(frozen_metrics_path, current)

    rows = read_jsonl(predictions_path)
    if len(rows) != 600:
        raise ValueError(f"expected 600 rows in {predictions_path}, got {len(rows)}")
    with np.load(output_dir / "hidden_probe.npz") as probe_artifact:
        hidden_dimensions = int(probe_artifact["mean_hidden"].shape[1])
    artifact, timings = attach_scores(rows, model=model, seed=seed)
    metrics = evaluate(
        rows,
        protocol=protocol,
        catalog=method_catalog(hidden_dimensions),
        timing=timings,
    )
    previous = read_json(frozen_metrics_path)
    if "runtime" in previous:
        metrics["runtime"] = previous["runtime"]
    write_jsonl(predictions_path, rows)
    write_json(metrics_path, metrics)
    write_json(output_dir / "fast_baseline_artifacts.json", artifact)
    (output_dir / "extension_report.md").write_text(render_report(model, metrics))
    print(
        f"{model}: best={metrics['posthoc_descriptive_best_non_confound']} "
        f"AUROC={metrics['methods'][metrics['posthoc_descriptive_best_non_confound']]['held_out']['auroc']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="both",
        help="Legacy built-in model key, 'both', or a label used with --output-dir.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(args.protocol)
    seeds = protocol["added_methods"]["trace_logistic_8"]["seeds"]
    if args.output_dir is not None:
        if args.model == "both":
            raise ValueError("--output-dir requires a single --model label")
        selected = {args.model: args.output_dir}
    elif args.model == "both":
        selected = MODEL_DIRS
    else:
        if args.model not in MODEL_DIRS:
            raise ValueError(
                f"unknown built-in model {args.model!r}; pass --output-dir for a new model"
            )
        selected = {args.model: MODEL_DIRS[args.model]}
    for model, output_dir in selected.items():
        seed = args.seed if args.seed is not None else int(seeds[model])
        process_model(
            model,
            output_dir,
            protocol=protocol,
            seed=seed,
        )


if __name__ == "__main__":
    main()
