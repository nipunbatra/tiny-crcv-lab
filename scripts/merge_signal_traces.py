"""Merge newly captured scalar token signals into identical saved generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SIGNAL_KEYS = (
    "token_margins",
    "token_entropies",
    "hidden_cosine_distances",
    "hidden_norms",
)


def merge(target_path: Path, source_path: Path) -> None:
    targets = [json.loads(line) for line in target_path.read_text().splitlines() if line]
    sources = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    if len(targets) != len(sources):
        raise ValueError(f"row-count mismatch: {target_path} vs {source_path}")

    for target, source in zip(targets, sources, strict=True):
        identity = ("id", "token_ids", "generated_answer", "is_hallucination")
        if any(target[key] != source[key] for key in identity):
            raise ValueError(f"generation mismatch for {target['id']}")
        for key in SIGNAL_KEYS:
            target[key] = source[key]

    target_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in targets)
    )
    print(f"Merged {len(targets)} traces from {source_path} into {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    merge(args.target, args.source)


if __name__ == "__main__":
    main()
