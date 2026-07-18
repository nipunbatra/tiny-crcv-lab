"""Screen cheap, interpretable scores using only saved generation traces.

This is an exploratory search. Candidates are ranked by Instruct calibration
AUROC; held-out and Base-model results are reported but never used to mutate the
question set or model outputs.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tiny_crcv.core import auroc, rolling_sample_std


@dataclass(frozen=True)
class Candidate:
    display_name: str
    formula: str
    input_points: str
    compute: Callable[[dict[str, Any]], float]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def top_mean(values: list[float], count: int) -> float:
    return mean(sorted(values, reverse=True)[:count])


def rolling_mean_max(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    effective = min(window, len(values))
    return max(mean(values[end - effective : end]) for end in range(effective, len(values) + 1))


def trace(row: dict[str, Any]) -> dict[str, list[float]]:
    confidences = [max(float(value), 1e-12) for value in row["confidences"]]
    surprises = [-math.log(value) for value in confidences]
    pairs = [
        (confidence, surprises[index], float(shift))
        for index, (confidence, shift) in enumerate(
            zip(confidences, row["hidden_shifts"], strict=True)
        )
        if shift is not None and math.isfinite(shift)
    ]
    shifts = [shift for _, _, shift in pairs]
    confidence_couplings = [confidence * shift for confidence, _, shift in pairs]
    surprise_couplings = [surprise * shift for _, surprise, shift in pairs]
    uncertainty_couplings = [(1.0 - confidence) * shift for confidence, _, shift in pairs]
    return {
        "confidences": confidences,
        "surprises": surprises,
        "shifts": shifts,
        "confidence_couplings": confidence_couplings,
        "surprise_couplings": surprise_couplings,
        "uncertainty_couplings": uncertainty_couplings,
    }


def variability(values: list[float]) -> float:
    return mean(rolling_sample_std(values, 5))


CANDIDATES: dict[str, Candidate] = {
    "top2_token_surprise": Candidate(
        "Top-2 token surprise",
        "mean(top 2 of −ln(cₜ))",
        "selected-token confidence",
        lambda row: top_mean(trace(row)["surprises"], 2),
    ),
    "top4_token_surprise": Candidate(
        "Top-4 token surprise",
        "mean(top 4 of −ln(cₜ))",
        "selected-token confidence",
        lambda row: top_mean(trace(row)["surprises"], 4),
    ),
    "skip_first_top3_surprise": Candidate(
        "Top-3 surprise after first token",
        "mean(top 3 of −ln(cₜ), t > 1)",
        "selected-token confidence and token position",
        lambda row: top_mean(trace(row)["surprises"][1:], 3),
    ),
    "surprise_burst_3": Candidate(
        "Three-token surprise burst",
        "max trailing-3 mean(−ln(cₜ))",
        "selected-token confidence and token position",
        lambda row: rolling_mean_max(trace(row)["surprises"], 3),
    ),
    "low_confidence_fraction_05": Candidate(
        "Fraction below 50% confidence",
        "mean(1[cₜ < 0.5])",
        "selected-token confidence",
        lambda row: mean([float(value < 0.5) for value in trace(row)["confidences"]]),
    ),
    "low_confidence_fraction_08": Candidate(
        "Fraction below 80% confidence",
        "mean(1[cₜ < 0.8])",
        "selected-token confidence",
        lambda row: mean([float(value < 0.8) for value in trace(row)["confidences"]]),
    ),
    "surprise_shift_mean": Candidate(
        "Surprise × shift mean",
        "mean((−ln(cₜ))rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: mean(trace(row)["surprise_couplings"]),
    ),
    "surprise_shift_max": Candidate(
        "Surprise × shift maximum",
        "max((−ln(cₜ))rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: max(trace(row)["surprise_couplings"], default=0.0),
    ),
    "surprise_shift_top3": Candidate(
        "Top-3 surprise × shift",
        "mean(top 3 of (−ln(cₜ))rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: top_mean(trace(row)["surprise_couplings"], 3),
    ),
    "surprise_shift_variability": Candidate(
        "Surprise × shift variability",
        "mean trailing-window sample-SD((−ln(cₜ))rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: variability(trace(row)["surprise_couplings"]),
    ),
    "uncertainty_shift_mean": Candidate(
        "Uncertainty × shift mean",
        "mean((1−cₜ)rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: mean(trace(row)["uncertainty_couplings"]),
    ),
    "uncertainty_shift_max": Candidate(
        "Uncertainty × shift maximum",
        "max((1−cₜ)rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: max(trace(row)["uncertainty_couplings"], default=0.0),
    ),
    "uncertainty_shift_top3": Candidate(
        "Top-3 uncertainty × shift",
        "mean(top 3 of (1−cₜ)rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: top_mean(trace(row)["uncertainty_couplings"], 3),
    ),
    "uncertainty_shift_variability": Candidate(
        "Uncertainty × shift variability",
        "mean trailing-window sample-SD((1−cₜ)rₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: variability(trace(row)["uncertainty_couplings"]),
    ),
    "hidden_shift_mean": Candidate(
        "Hidden-shift mean",
        "mean(rₜ)",
        "normalized hidden shift",
        lambda row: mean(trace(row)["shifts"]),
    ),
    "hidden_shift_max": Candidate(
        "Hidden-shift maximum",
        "max(rₜ)",
        "normalized hidden shift",
        lambda row: max(trace(row)["shifts"], default=0.0),
    ),
    "confidence_shift_mean": Candidate(
        "Confidence × shift mean",
        "mean(cₜrₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: mean(trace(row)["confidence_couplings"]),
    ),
    "confidence_shift_max": Candidate(
        "Confidence × shift maximum",
        "max(cₜrₜ)",
        "selected-token confidence and normalized hidden shift",
        lambda row: max(trace(row)["confidence_couplings"], default=0.0),
    ),
}


MODELS = {
    "instruct": Path("outputs/qwen05b_100/predictions.jsonl"),
    "base": Path("outputs/qwen05b_base_100/predictions.jsonl"),
}


def evaluate(rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, float]:
    result: dict[str, float] = {}
    for split in ("calibration", "test"):
        selected = [row for row in rows if row["split"] == split]
        result[f"{split}_auroc"] = auroc(
            [int(row["is_hallucination"]) for row in selected],
            [candidate.compute(row) for row in selected],
        )
    return result


def main() -> None:
    started = time.perf_counter()
    rows_by_model = {
        model: [json.loads(line) for line in path.read_text().splitlines() if line]
        for model, path in MODELS.items()
    }
    results = {
        key: {
            "display_name": candidate.display_name,
            "formula": candidate.formula,
            "input_points": candidate.input_points,
            "models": {
                model: evaluate(rows, candidate) for model, rows in rows_by_model.items()
            },
        }
        for key, candidate in CANDIDATES.items()
    }
    ranking = sorted(
        CANDIDATES,
        key=lambda key: results[key]["models"]["instruct"]["calibration_auroc"],
        reverse=True,
    )
    output = {
        "protocol": {
            "status": "exploratory; the original held-out split has been inspected in prior iterations",
            "selection_rule": "rank candidates by Instruct calibration AUROC",
            "constraints": "O(answer tokens), no extra model calls, no learned classifier",
            "candidate_count": len(CANDIDATES),
            "runtime_seconds": time.perf_counter() - started,
        },
        "ranking": ranking,
        "candidates": results,
    }
    output_path = Path("outputs/simple_metric_search.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {output_path} in {output['protocol']['runtime_seconds']:.3f}s")
    for rank, key in enumerate(ranking, start=1):
        values = results[key]["models"]
        print(
            f"{rank:02d} {key:34s} "
            f"I-cal={values['instruct']['calibration_auroc']:.3f} "
            f"I-test={values['instruct']['test_auroc']:.3f} "
            f"B-cal={values['base']['calibration_auroc']:.3f} "
            f"B-test={values['base']['test_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
