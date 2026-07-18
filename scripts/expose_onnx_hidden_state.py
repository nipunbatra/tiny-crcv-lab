#!/usr/bin/env python3
"""Expose Qwen's final normalized hidden state as an ONNX graph output.

The onnx-community Qwen2.5 causal-LM exports compute this tensor immediately
before ``lm_head`` but only publish logits and KV-cache tensors. Adding an
output ValueInfo does not change the computation or weights; it makes the
existing activation available to ONNX Runtime (and therefore Transformers.js).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


HIDDEN_OUTPUT = "/model/norm/Mul_1_output_0"


def expose_hidden_state(source: Path, destination: Path, *, element_type: int) -> None:
    model = onnx.load(source, load_external_data=True)
    produced = {name for node in model.graph.node for name in node.output}
    if HIDDEN_OUTPUT not in produced:
        raise ValueError(f"Expected tensor {HIDDEN_OUTPUT!r} was not found")

    existing = {value.name for value in model.graph.output}
    if HIDDEN_OUTPUT not in existing:
        model.graph.output.append(
            helper.make_tensor_value_info(
                HIDDEN_OUTPUT,
                element_type,
                ["batch_size", "sequence_length", 896],
            )
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16"),
        default="auto",
        help="Activation type; auto selects float16 for q4f16 filenames",
    )
    args = parser.parse_args()
    dtype = args.dtype
    if dtype == "auto":
        dtype = "float16" if "q4f16" in args.source.name else "float32"
    element_type = TensorProto.FLOAT16 if dtype == "float16" else TensorProto.FLOAT
    expose_hidden_state(args.source, args.destination, element_type=element_type)
    print(f"Patched model written to {args.destination}")


if __name__ == "__main__":
    main()
