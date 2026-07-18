# Tiny CRCV benchmark report

Top-3 token entropy did not demonstrate better-than-chance separation on this small test. Its held-out AUROC was 0.633 (bootstrap 95% CI 0.473-0.783).

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
| Top-3 token surprise | 0.728 | 0.574-0.858 | 0.667 | 1.45037 |
| Worst-token surprise | 0.737 | 0.589-0.868 | 0.672 | 1.75567 |
| Token-surprise spread | 0.666 | 0.502-0.819 | 0.601 | 0.527268 |
| Top-4 token surprise | 0.701 | 0.543-0.839 | 0.625 | 1.33201 |
| Top-3 surprise after first token | 0.693 | 0.521-0.837 | 0.603 | 1.52144 |
| Top-3 surprise × hidden shift | 0.567 | 0.395-0.726 | 0.495 | 1.09526 |
| Top-3 uncertainty × hidden shift | 0.497 | 0.340-0.668 | 0.458 | 0.580783 |
| Top-3 token entropy | 0.633 | 0.473-0.783 | 0.587 | 0.372545 |
| Mean token entropy | 0.666 | 0.499-0.810 | 0.600 | 0.133171 |
| Maximum token entropy | 0.581 | 0.403-0.756 | 0.556 | 0.425273 |
| Top-3 token ambiguity | 0.723 | 0.575-0.855 | 0.554 | 0.963816 |
| Mean token ambiguity | 0.654 | 0.481-0.804 | 0.587 | 0.431661 |
| Maximum token ambiguity | 0.655 | 0.491-0.800 | 0.509 | 0.991083 |
| Mean hidden cosine change | 0.324 | 0.178-0.476 | 0.355 | 0.614792 |
| Maximum hidden cosine change | 0.480 | 0.295-0.667 | 0.493 | 0.950125 |
| Top-3 hidden cosine change | 0.435 | 0.263-0.611 | 0.504 | 0.894712 |
| Mean hidden-state RMS norm | 0.643 | 0.476-0.798 | 0.643 | 8.71378 |
| Hidden-state norm variability | 0.319 | 0.161-0.484 | 0.454 | 1.18857 |

The held-out set contained 19 incorrect and 31 correct generations.
AUROC uses the predeclared direction 'higher score = more likely wrong'; 0.5 is chance.
Thresholds were chosen only on the calibration split by maximum macro-F1.

Primary CRCV minus confidence-variability AUROC: -0.090 (paired bootstrap 95% CI -0.228 to 0.044).
Top-3 token surprise minus the previous-best original-score AUROC: 0.063 (paired bootstrap 95% CI -0.066 to 0.192).
Top-3 token entropy minus top-3 token-surprise AUROC: -0.095 (paired bootstrap 95% CI -0.231 to 0.027).

## What this establishes

This is an exploratory follow-up on one tiny checkpoint, not a validated detector or an untouched confirmatory test.
The saved predictions include each answer, gold aliases, token-distribution signals, hidden-state signals, and aggregate scores.
The largest limitations are the 50-question held-out sample, one architecture/layer, greedy decoding, and lexical correctness labels.
A follow-up should manually audit label errors, repeat across seeds/models, and use a larger human-aligned benchmark.
