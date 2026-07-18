# Tiny CRCV benchmark report

CRCV did not demonstrate better-than-chance separation on this small test. Its held-out AUROC was 0.508 (bootstrap 95% CI 0.328-0.677).

## Setup

- Model: `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Device: `mps`; greedy decoding; at most 24 answer tokens
- Questions: 50 calibration + 50 held out
- Window: 5 generated steps; hidden layer: -1
- Operational label: an answer is wrong when no normalized accepted alias appears in the generated sentence

## Held-out results

| Score | AUROC | Bootstrap 95% CI | Macro-F1 | Calibration threshold |
|---|---:|---:|---:|---:|
| CRCV (primary) | 0.508 | 0.328-0.677 | 0.449 | 0.166918 |
| CRCV maximum | 0.486 | 0.317-0.655 | 0.451 | 0.442997 |
| Mean token surprise | 0.666 | 0.501-0.807 | 0.584 | 0.60343 |
| Confidence variability | 0.598 | 0.442-0.768 | 0.451 | 0.215774 |
| Hidden-shift variability | 0.540 | 0.365-0.705 | 0.532 | 0.215889 |
| Answer length | 0.484 | 0.367-0.593 | 0.383 | 24 |

The held-out set contained 19 incorrect and 31 correct generations.
AUROC uses the predeclared direction 'higher score = more likely wrong'; 0.5 is chance.
Thresholds were chosen only on the calibration split by maximum macro-F1.

Primary CRCV minus confidence-variability AUROC: -0.090 (paired bootstrap 95% CI -0.228 to 0.044).

## What this establishes

This is a falsifiable smoke test of the document's coupling hypothesis on one tiny model, not a validated detector.
The saved predictions include each answer, gold aliases, token confidences, hidden-state shifts, and aggregate scores.
The largest limitations are the 50-question held-out sample, one model/layer, greedy decoding, and lexical correctness labels.
A follow-up should manually audit label errors, repeat across seeds/models, and use a larger human-aligned benchmark.
