# Post-hoc fast-baseline extension

Checkpoint: **smollm2_instruct**. This is exploratory because the held-out outputs were already inspected.

## Headline methods

| Method | Family | AUROC | 95% CI | AURAC | Macro-F1 | Extra s/question |
|---|---|---:|---:|---:|---:|---:|
| Top-3 token surprise | token probability | 0.715 | 0.635–0.793 | 0.257 | 0.577 | 0.000000 |
| CRCV (primary) | CRCV | 0.588 | 0.503–0.674 | 0.187 | 0.488 | 0.000000 |
| Mean token surprise | token probability | 0.712 | 0.629–0.787 | 0.255 | 0.595 | 0.000000 |
| Top-3 token entropy | distribution | 0.720 | 0.633–0.799 | 0.266 | 0.614 | 0.000000 |
| P(False self-check) | P(True)-style self-evaluation | 0.591 | 0.499–0.686 | 0.200 | 0.550 | 0.028105 |
| Three-answer lexical disagreement | SelfCheckGPT-inspired | 0.691 | 0.606–0.767 | 0.245 | 0.588 | 0.683648 |
| Three-sample discrete semantic entropy proxy | semantic-entropy proxy | 0.489 | 0.455–0.514 | 0.162 | 0.458 | 0.769485 |
| Three-answer semantic disagreement | sample consistency | 0.548 | 0.457–0.638 | 0.196 | 0.518 | 0.769485 |
| Mean-hidden linear probe | supervised internal state | 0.648 | 0.553–0.733 | 0.228 | 0.595 | 0.000081 |
| Eight-feature trace logistic | supervised trace ensemble | 0.724 | 0.632–0.804 | 0.268 | 0.633 | 0.000001 |
| Depth-2 trace tree | supervised trace ensemble | 0.666 | 0.582–0.740 | 0.258 | 0.577 | 0.000001 |
| Answer length | control | 0.519 | 0.420–0.610 | 0.169 | 0.497 | 0.000000 |

## All trace-only scalars

| Scalar | Family | AUROC | 95% CI | Macro-F1 |
|---|---|---:|---:|---:|
| CRCV (primary) | CRCV | 0.588 | 0.503–0.674 | 0.488 |
| CRCV maximum | CRCV | 0.597 | 0.511–0.683 | 0.542 |
| Mean token surprise | token probability | 0.712 | 0.629–0.787 | 0.595 |
| Confidence variability | token probability | 0.658 | 0.578–0.739 | 0.570 |
| Hidden-shift variability | hidden dynamics | 0.521 | 0.437–0.602 | 0.471 |
| Answer length | control | 0.519 | 0.420–0.610 | 0.497 |
| Top-3 token surprise | token probability | 0.715 | 0.635–0.793 | 0.577 |
| Worst-token surprise | token probability | 0.745 | 0.661–0.819 | 0.639 |
| Token-surprise spread | token probability | 0.731 | 0.644–0.811 | 0.625 |
| Top-4 token surprise | token probability | 0.707 | 0.623–0.785 | 0.596 |
| Top-3 surprise after first token | token probability | 0.688 | 0.600–0.768 | 0.579 |
| Top-3 surprise × hidden shift | coupled | 0.660 | 0.566–0.741 | 0.567 |
| Top-3 uncertainty × hidden shift | coupled | 0.633 | 0.541–0.723 | 0.570 |
| Top-3 token entropy | distribution | 0.720 | 0.633–0.799 | 0.614 |
| Mean token entropy | distribution | 0.722 | 0.646–0.798 | 0.598 |
| Maximum token entropy | distribution | 0.723 | 0.640–0.799 | 0.622 |
| Top-3 token ambiguity | distribution | 0.634 | 0.542–0.720 | 0.585 |
| Mean token ambiguity | distribution | 0.683 | 0.600–0.760 | 0.560 |
| Maximum token ambiguity | distribution | 0.645 | 0.555–0.728 | 0.548 |
| Mean hidden cosine change | hidden dynamics | 0.390 | 0.314–0.470 | 0.479 |
| Maximum hidden cosine change | hidden dynamics | 0.495 | 0.406–0.573 | 0.499 |
| Top-3 hidden cosine change | hidden dynamics | 0.442 | 0.359–0.527 | 0.466 |
| Mean hidden-state RMS norm | hidden dynamics | 0.517 | 0.427–0.606 | 0.524 |
| Hidden-state norm variability | hidden dynamics | 0.535 | 0.449–0.626 | 0.495 |

Descriptively highest non-length method: `worst_token_surprise`.
Do not read that as confirmation or SOTA: model, task, label, sampling budget, and split differ from published studies.
