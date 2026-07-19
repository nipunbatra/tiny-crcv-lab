# Fresh 600-question tiny-detector comparison

Model: `HuggingFaceTB/SmolLM2-360M-Instruct` at `c38281e01d0c0b0c36eac2f5bcb5b51fa2e803fc`.
Data: 300 calibration + 300 held out across NQ-Open, TriviaQA, and TruthfulQA.

| Method | Held-out AUROC | 95% CI | AP | Macro-F1 | Mean extra seconds/question |
|---|---:|---:|---:|---:|---:|
| Top-3 token surprise | 0.715 | 0.630–0.793 | 0.921 | 0.577 | 0.0000 |
| P(False self-check) | 0.591 | 0.492–0.681 | 0.873 | 0.550 | 0.0281 |
| Mean-hidden linear probe | 0.648 | 0.558–0.742 | 0.884 | 0.595 | 0.0001 |
| Three-answer disagreement | 0.548 | 0.457–0.639 | 0.859 | 0.518 | 0.7695 |
| Answer length (confound) | 0.519 | 0.428–0.611 | 0.850 | 0.497 | 0.0000 |

Descriptively highest non-confound score: **Top-3 token surprise**.
No method met the predeclared reliable-improvement rule versus top-3 surprise.

These are reference-free risk signals, not factual verifiers. Labels use transparent alias matching and require manual audit for paraphrases and dataset ambiguity.
Thresholds were fit on pooled calibration data and frozen for all held-out slices.
