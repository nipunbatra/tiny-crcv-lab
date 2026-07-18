# Tiny CRCV benchmark report

Top-3 token entropy separated wrong from correct answers better than chance on this test. Its held-out AUROC was 0.825 (bootstrap 95% CI 0.699-0.929).

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
| Top-4 token surprise | 0.815 | 0.661-0.937 | 0.668 | 0.869943 |
| Top-3 surprise after first token | 0.827 | 0.686-0.939 | 0.740 | 0.593955 |
| Top-3 surprise × hidden shift | 0.800 | 0.653-0.915 | 0.760 | 0.751286 |
| Top-3 uncertainty × hidden shift | 0.792 | 0.638-0.919 | 0.689 | 0.598683 |
| Top-3 token entropy | 0.825 | 0.699-0.929 | 0.689 | 0.215948 |
| Mean token entropy | 0.743 | 0.603-0.869 | 0.520 | 0.070725 |
| Maximum token entropy | 0.785 | 0.637-0.906 | 0.689 | 0.301896 |
| Top-3 token ambiguity | 0.779 | 0.629-0.902 | 0.733 | 0.740088 |
| Mean token ambiguity | 0.640 | 0.482-0.792 | 0.561 | 0.267982 |
| Maximum token ambiguity | 0.777 | 0.613-0.900 | 0.589 | 0.853756 |
| Mean hidden cosine change | 0.352 | 0.182-0.538 | 0.440 | 0.572485 |
| Maximum hidden cosine change | 0.579 | 0.399-0.748 | 0.510 | 0.956028 |
| Top-3 hidden cosine change | 0.509 | 0.306-0.705 | 0.603 | 0.879057 |
| Mean hidden-state RMS norm | 0.672 | 0.484-0.847 | 0.566 | 8.83993 |
| Hidden-state norm variability | 0.352 | 0.171-0.546 | 0.384 | 0.795834 |

The held-out set contained 15 incorrect and 35 correct generations.
AUROC uses the predeclared direction 'higher score = more likely wrong'; 0.5 is chance.
Thresholds were chosen only on the calibration split by maximum macro-F1.

Primary CRCV minus confidence-variability AUROC: -0.183 (paired bootstrap 95% CI -0.345 to -0.018).
Top-3 token surprise minus the previous-best original-score AUROC: 0.040 (paired bootstrap 95% CI -0.056 to 0.128).
Top-3 token entropy minus top-3 token-surprise AUROC: 0.008 (paired bootstrap 95% CI -0.056 to 0.076).

## What this establishes

This is an exploratory follow-up on one tiny checkpoint, not a validated detector or an untouched confirmatory test.
The saved predictions include each answer, gold aliases, token-distribution signals, hidden-state signals, and aggregate scores.
The largest limitations are the 50-question held-out sample, one architecture/layer, greedy decoding, and lexical correctness labels.
A follow-up should manually audit label errors, repeat across seeds/models, and use a larger human-aligned benchmark.
