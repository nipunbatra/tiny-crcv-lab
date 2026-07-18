# HaluEval-QA 50-pair external benchmark

This frozen subset contains 25 calibration and 25 held-out question pairs. Each pair contributes the supplied right and hallucinated candidate answer.
The model teacher-forces each candidate under HaluEval's supplied knowledge; this is grounded candidate discrimination, not free generation.

## Held-out results

| Detector | Selection | AUROC | 95% CI | Macro-F1 | Pairwise accuracy |
|---|---|---:|---:|---:|---:|
| Top-3 surprise after first token | best scalar on calibration | 0.994 | 0.976–1.000 | 0.940 | 1.000 |
| Decision stump | diagnostic | 0.900 | 0.821–0.975 | 0.899 | 0.900 |
| Depth-2 tree | predeclared primary tree | 0.968 | 0.913–0.998 | 0.899 | 0.980 |
| Answer length | confound baseline | 0.958 | 0.900–0.996 | 0.859 | 1.000 |
| Top-3 surprise after first token after log-length residualization | post-hoc diagnostic | 0.787 | 0.644–0.915 | 0.756 | 0.800 |

Pairwise accuracy asks whether the hallucinated candidate receives a higher risk score than the right candidate for the same question; ties receive half credit.
In the held-out split, the hallucinated candidate was longer in 25/25 pairs. The residualized row is a post-hoc diagnostic, not a new confirmatory result.

## Limits

- Only 25 held-out question pairs are used, so uncertainty is wide.
- HaluEval hallucinated answers are synthetic and often longer than right answers.
- Teacher-forced candidate discrimination is not the same task as detecting errors in the model's own free generation.
- The tree has only two levels and six predeclared inputs, but 50 calibration candidates is still a very small training set.
