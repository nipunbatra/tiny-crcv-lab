"""Apply the frozen cross-model replication rule to Qwen and SmolLM2 results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QWEN_PATH = ROOT / "outputs/fresh_qa_qwen05b_instruct/metrics.json"
SMOL_PATH = ROOT / "outputs/fresh_qa_smollm2_360m_instruct/metrics.json"
PROTOCOL_PATH = ROOT / "experiments/fresh_qa_600_smollm2_replication_protocol.json"
OUTPUT_PATH = ROOT / "outputs/fresh_qa_cross_model_replication.json"
REPORT_PATH = ROOT / "outputs/fresh_qa_cross_model_replication.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def finite_area(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def direction_count(method: dict[str, Any]) -> int:
    return sum(
        finite_area(dataset["auroc"]) is not None and float(dataset["auroc"]) > 0.5
        for dataset in method["by_dataset"].values()
    )


def main() -> None:
    qwen = read_json(QWEN_PATH)
    smol = read_json(SMOL_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if qwen["all_method_order"] != smol["all_method_order"]:
        raise ValueError("model outputs do not expose the same ordered method set")

    rows = []
    for key in qwen["all_method_order"]:
        qwen_method = qwen["methods"][key]
        smol_method = smol["methods"][key]
        qwen_auroc = finite_area(qwen_method["held_out"]["auroc"])
        smol_auroc = finite_area(smol_method["held_out"]["auroc"])
        qwen_slices = direction_count(qwen_method)
        smol_slices = direction_count(smol_method)
        replicated = bool(
            qwen_auroc is not None
            and smol_auroc is not None
            and qwen_auroc > 0.5
            and smol_auroc > 0.5
            and qwen_slices >= 2
            and smol_slices >= 2
        )
        qwen_ci = qwen_method["held_out"]["auroc_ci_95"]
        smol_ci = smol_method["held_out"]["auroc_ci_95"]
        rows.append(
            {
                "key": key,
                "display_name": qwen_method["display_name"],
                "family": qwen_method["family"],
                "qwen_auroc": qwen_auroc,
                "qwen_auroc_ci_95": qwen_ci,
                "smollm2_auroc": smol_auroc,
                "smollm2_auroc_ci_95": smol_ci,
                "qwen_slices_above_chance": qwen_slices,
                "smollm2_slices_above_chance": smol_slices,
                "replicated": replicated,
                "both_ci_lower_above_chance": bool(
                    qwen_ci[0] is not None
                    and smol_ci[0] is not None
                    and qwen_ci[0] > 0.5
                    and smol_ci[0] > 0.5
                ),
                "worst_model_auroc": min(qwen_auroc, smol_auroc)
                if qwen_auroc is not None and smol_auroc is not None
                else None,
            }
        )

    non_control = [row for row in rows if row["key"] != "answer_tokens"]
    ranked = sorted(
        non_control,
        key=lambda row: row["worst_model_auroc"]
        if row["worst_model_auroc"] is not None
        else -1,
        reverse=True,
    )
    replicated = [row["key"] for row in rows if row["replicated"]]
    strict = [row["key"] for row in rows if row["both_ci_lower_above_chance"]]
    result = {
        "status": protocol["status"],
        "rule": protocol["interpretation_rule"],
        "models": {
            "qwen_instruct": {
                "id": "Qwen/Qwen2.5-0.5B-Instruct",
                "parameters": "0.5B",
                "held_out_examples": qwen["held_out_examples"],
                "held_out_incorrect": qwen["held_out_incorrect"],
            },
            "smollm2_instruct": {
                "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "parameters": "360M",
                "held_out_examples": smol["held_out_examples"],
                "held_out_incorrect": smol["held_out_incorrect"],
            },
        },
        "method_count": len(rows),
        "replicated_method_count": len(replicated),
        "replicated_methods": replicated,
        "both_ci_lower_above_chance": strict,
        "best_worst_model_method": ranked[0]["key"],
        "best_worst_model_auroc": ranked[0]["worst_model_auroc"],
        "rows": rows,
        "interpretation": protocol["interpretation"],
    }
    write_json(OUTPUT_PATH, result)

    lines = [
        "# Cross-model replication",
        "",
        f"Replicated by the frozen rule: **{len(replicated)} / {len(rows)} methods**.",
        f"Best worst-model AUROC: **{ranked[0]['display_name']} ({ranked[0]['worst_model_auroc']:.3f})**.",
        "",
        "| Method | Qwen AUROC | SmolLM2 AUROC | Qwen slices > .5 | Smol slices > .5 | Rule passes |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: item["worst_model_auroc"]
        if item["worst_model_auroc"] is not None
        else -1,
        reverse=True,
    ):
        lines.append(
            f"| {row['display_name']} | {row['qwen_auroc']:.3f} | "
            f"{row['smollm2_auroc']:.3f} | {row['qwen_slices_above_chance']}/3 | "
            f"{row['smollm2_slices_above_chance']}/3 | {'yes' if row['replicated'] else 'no'} |"
        )
    lines.extend(["", protocol["interpretation"], ""])
    REPORT_PATH.write_text("\n".join(lines))
    print(
        f"replicated={len(replicated)}/{len(rows)} "
        f"best_worst={ranked[0]['key']}:{ranked[0]['worst_model_auroc']:.3f}"
    )


if __name__ == "__main__":
    main()
