# Cross-model replication

Replicated by the frozen rule: **25 / 31 methods**.
Best worst-model AUROC: **Token-surprise spread (0.706)**.

| Method | Qwen AUROC | SmolLM2 AUROC | Qwen slices > .5 | Smol slices > .5 | Rule passes |
|---|---:|---:|---:|---:|:---:|
| Token-surprise spread | 0.706 | 0.731 | 3/3 | 3/3 | yes |
| Maximum token entropy | 0.696 | 0.723 | 3/3 | 3/3 | yes |
| Eight-feature trace logistic | 0.695 | 0.724 | 3/3 | 3/3 | yes |
| Three-answer lexical disagreement | 0.717 | 0.691 | 3/3 | 3/3 | yes |
| Worst-token surprise | 0.689 | 0.745 | 3/3 | 3/3 | yes |
| Mean token surprise | 0.663 | 0.712 | 3/3 | 3/3 | yes |
| Top-3 token entropy | 0.661 | 0.720 | 3/3 | 3/3 | yes |
| Top-3 surprise × hidden shift | 0.667 | 0.660 | 3/3 | 3/3 | yes |
| Confidence variability | 0.675 | 0.658 | 3/3 | 3/3 | yes |
| Top-3 token surprise | 0.656 | 0.715 | 3/3 | 3/3 | yes |
| Mean token entropy | 0.653 | 0.722 | 3/3 | 3/3 | yes |
| Top-4 token surprise | 0.648 | 0.707 | 3/3 | 3/3 | yes |
| Top-3 surprise after first token | 0.644 | 0.688 | 3/3 | 3/3 | yes |
| Top-3 uncertainty × hidden shift | 0.657 | 0.633 | 3/3 | 3/3 | yes |
| Maximum token ambiguity | 0.632 | 0.645 | 3/3 | 3/3 | yes |
| Mean token ambiguity | 0.619 | 0.683 | 3/3 | 3/3 | yes |
| Depth-2 trace tree | 0.610 | 0.666 | 2/3 | 3/3 | yes |
| CRCV maximum | 0.606 | 0.597 | 3/3 | 3/3 | yes |
| P(False self-check) | 0.603 | 0.591 | 2/3 | 3/3 | yes |
| CRCV (primary) | 0.634 | 0.588 | 3/3 | 3/3 | yes |
| Top-3 token ambiguity | 0.584 | 0.634 | 2/3 | 3/3 | yes |
| Mean-hidden linear probe | 0.549 | 0.648 | 2/3 | 3/3 | yes |
| Three-answer semantic disagreement | 0.621 | 0.548 | 3/3 | 3/3 | yes |
| Hidden-state norm variability | 0.611 | 0.535 | 2/3 | 2/3 | yes |
| Hidden-shift variability | 0.557 | 0.521 | 2/3 | 2/3 | yes |
| Answer length | 0.540 | 0.519 | 1/3 | 1/3 | no |
| Maximum hidden cosine change | 0.517 | 0.495 | 1/3 | 1/3 | no |
| Three-sample discrete semantic entropy proxy | 0.500 | 0.489 | 1/3 | 1/3 | no |
| Mean hidden-state RMS norm | 0.473 | 0.517 | 1/3 | 3/3 | no |
| Top-3 hidden cosine change | 0.528 | 0.442 | 1/3 | 2/3 | no |
| Mean hidden cosine change | 0.555 | 0.390 | 3/3 | 0/3 | no |

This prospectively tests new-model transfer on an existing question set. It does not establish state of the art, a new-data replication, or factual verification. Model, prompt, labels, sampling budget, and split remain mandatory context.
