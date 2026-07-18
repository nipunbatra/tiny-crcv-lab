# Fresh 600-question tiny-detector comparison

Model: `Qwen/Qwen2.5-0.5B` at `060db6499f32faf8b98477b0a26969ef7d8b9987`.
Data: 300 calibration + 300 held out across NQ-Open, TriviaQA, and TruthfulQA.

| Method | Held-out AUROC | 95% CI | AP | Macro-F1 | Mean extra seconds/question |
|---|---:|---:|---:|---:|---:|
| Top-3 token surprise | 0.599 | 0.494–0.703 | 0.939 | 0.527 | 0.0000 |
| P(False self-check) | 0.628 | 0.523–0.728 | 0.942 | 0.536 | 0.1128 |
| Mean-hidden linear probe | 0.545 | 0.433–0.655 | 0.925 | 0.502 | 0.0001 |
| Three-answer disagreement | 0.595 | 0.491–0.691 | 0.941 | 0.488 | 2.8537 |
| Answer length (confound) | 0.528 | 0.430–0.632 | 0.912 | 0.552 | 0.0000 |

Descriptively highest non-confound score: **P(False self-check)**.
No method met the predeclared reliable-improvement rule versus top-3 surprise.

These are reference-free risk signals, not factual verifiers. Labels use transparent alias matching and require manual audit for paraphrases and dataset ambiguity.
Thresholds were fit on pooled calibration data and frozen for all held-out slices.
