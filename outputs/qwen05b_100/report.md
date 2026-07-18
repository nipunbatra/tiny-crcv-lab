# Tiny CRCV benchmark report

Top-3 token surprise separated wrong from correct answers better than chance on this test. Its held-out AUROC was 0.817 (bootstrap 95% CI 0.684-0.931).

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Device: `mps`; greedy decoding; at most 24 answer tokens
- Questions: 50 calibration + 50 held out
- Window: 5 generated steps; hidden layer: -1
- Operational label: an answer is wrong when no normalized accepted alias appears in the generated sentence

## Held-out results

| Score | AUROC | Bootstrap 95% CI | Macro-F1 | Calibration threshold |
|---|---:|---:|---:|---:|
| CRCV (primary) | 0.594 | 0.419-0.766 | 0.570 | 0.257274 |
| CRCV maximum | 0.674 | 0.506-0.830 | 0.587 | 0.397006 |
| Mean token surprise | 0.730 | 0.568-0.865 | 0.670 | 0.292744 |
| Confidence variability | 0.777 | 0.632-0.904 | 0.678 | 0.155704 |
| Hidden-shift variability | 0.453 | 0.285-0.621 | 0.538 | 0.222815 |
| Answer length | 0.626 | 0.454-0.787 | 0.525 | 12.5 |
| Top-3 token surprise | 0.817 | 0.684-0.931 | 0.668 | 1.02219 |
| Worst-token surprise | 0.752 | 0.593-0.889 | 0.632 | 1.11853 |
| Token-surprise spread | 0.771 | 0.618-0.904 | 0.633 | 0.535516 |

The held-out set contained 15 incorrect and 35 correct generations.
AUROC uses the predeclared direction 'higher score = more likely wrong'; 0.5 is chance.
Thresholds were chosen only on the calibration split by maximum macro-F1.

Primary CRCV minus confidence-variability AUROC: -0.183 (paired bootstrap 95% CI -0.345 to -0.018).
Top-3 token surprise minus the previous-best original-score AUROC: 0.040 (paired bootstrap 95% CI -0.056 to 0.128).

## What this establishes

This is an exploratory follow-up on one tiny model, not a validated detector or an untouched confirmatory test.
The saved predictions include each answer, gold aliases, token confidences, hidden-state shifts, and aggregate scores.
The largest limitations are the 50-question held-out sample, one model/layer, greedy decoding, and lexical correctness labels.
A follow-up should manually audit label errors, repeat across seeds/models, and use a larger human-aligned benchmark.
