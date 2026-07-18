# Fresh 600-question tiny-detector comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`.
Data: 300 calibration + 300 held out across NQ-Open, TriviaQA, and TruthfulQA.

| Method | Held-out AUROC | 95% CI | AP | Macro-F1 | Mean extra seconds/question |
|---|---:|---:|---:|---:|---:|
| Top-3 token surprise | 0.656 | 0.558–0.743 | 0.910 | 0.584 | 0.0000 |
| P(False self-check) | 0.603 | 0.508–0.696 | 0.886 | 0.506 | 0.1297 |
| Mean-hidden linear probe | 0.549 | 0.453–0.641 | 0.866 | 0.538 | 0.0001 |
| Three-answer disagreement | 0.621 | 0.515–0.721 | 0.891 | 0.594 | 1.9013 |
| Answer length (confound) | 0.540 | 0.432–0.644 | 0.856 | 0.568 | 0.0000 |

Descriptively highest non-confound score: **Top-3 token surprise**.
No method met the predeclared reliable-improvement rule versus top-3 surprise.

These are reference-free risk signals, not factual verifiers. Labels use transparent alias matching and require manual audit for paraphrases and dataset ambiguity.
Thresholds were fit on pooled calibration data and frozen for all held-out slices.
