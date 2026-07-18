import json
from pathlib import Path

import pytest

from tiny_crcv.core import compute_features


@pytest.mark.parametrize("folder", ["qwen05b_100", "qwen05b_base_100"])
def test_saved_features_recompute_from_raw_traces(folder: str) -> None:
    root = Path(__file__).parents[1]
    output_dir = root / "outputs" / folder
    metadata = json.loads((output_dir / "metadata.json").read_text())
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
        if line
    ]

    assert len(rows) == 100
    for row in rows:
        recomputed = compute_features(
            row["confidences"],
            row["hidden_shifts"],
            window=metadata["window"],
            token_margins=row["token_margins"],
            token_entropies=row["token_entropies"],
            hidden_cosine_distances=row["hidden_cosine_distances"],
            hidden_norms=row["hidden_norms"],
        )
        assert row["features"] == pytest.approx(recomputed)
