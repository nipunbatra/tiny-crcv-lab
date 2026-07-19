# Post-hoc fast-baseline extension

Checkpoint: **instruct**. This is exploratory because the held-out outputs were already inspected.

## Headline methods

| Method | Family | AUROC | 95% CI | AURAC | Macro-F1 | Extra s/question |
|---|---|---:|---:|---:|---:|---:|
| Top-3 token surprise | token probability | 0.656 | 0.560–0.747 | 0.226 | 0.584 | 0.000000 |
| CRCV (primary) | CRCV | 0.634 | 0.543–0.712 | 0.194 | 0.573 | 0.000000 |
| Mean token surprise | token probability | 0.663 | 0.567–0.754 | 0.223 | 0.597 | 0.000000 |
| Top-3 token entropy | distribution | 0.661 | 0.570–0.750 | 0.224 | 0.585 | 0.000000 |
| P(False self-check) | P(True)-style self-evaluation | 0.603 | 0.505–0.697 | 0.192 | 0.506 | 0.129678 |
| Three-answer lexical disagreement | SelfCheckGPT-inspired | 0.717 | 0.623–0.801 | 0.244 | 0.571 | 1.461757 |
| Three-sample discrete semantic entropy proxy | semantic-entropy proxy | 0.500 | 0.434–0.553 | 0.176 | 0.461 | 1.901344 |
| Three-answer semantic disagreement | sample consistency | 0.621 | 0.519–0.719 | 0.210 | 0.594 | 1.901344 |
| Mean-hidden linear probe | supervised internal state | 0.549 | 0.453–0.644 | 0.172 | 0.538 | 0.000073 |
| Eight-feature trace logistic | supervised trace ensemble | 0.695 | 0.603–0.777 | 0.238 | 0.600 | 0.000002 |
| Depth-2 trace tree | supervised trace ensemble | 0.610 | 0.526–0.697 | 0.213 | 0.560 | 0.000001 |
| Answer length | control | 0.540 | 0.438–0.651 | 0.197 | 0.568 | 0.000000 |

## All trace-only scalars

| Scalar | Family | AUROC | 95% CI | Macro-F1 |
|---|---|---:|---:|---:|
| CRCV (primary) | CRCV | 0.634 | 0.543–0.712 | 0.573 |
| CRCV maximum | CRCV | 0.606 | 0.516–0.705 | 0.564 |
| Mean token surprise | token probability | 0.663 | 0.567–0.754 | 0.597 |
| Confidence variability | token probability | 0.675 | 0.594–0.751 | 0.568 |
| Hidden-shift variability | hidden dynamics | 0.557 | 0.461–0.654 | 0.568 |
| Answer length | control | 0.540 | 0.438–0.651 | 0.568 |
| Top-3 token surprise | token probability | 0.656 | 0.560–0.747 | 0.584 |
| Worst-token surprise | token probability | 0.689 | 0.592–0.776 | 0.611 |
| Token-surprise spread | token probability | 0.706 | 0.620–0.790 | 0.616 |
| Top-4 token surprise | token probability | 0.648 | 0.552–0.742 | 0.587 |
| Top-3 surprise after first token | token probability | 0.644 | 0.545–0.736 | 0.569 |
| Top-3 surprise × hidden shift | coupled | 0.667 | 0.575–0.754 | 0.596 |
| Top-3 uncertainty × hidden shift | coupled | 0.657 | 0.565–0.745 | 0.562 |
| Top-3 token entropy | distribution | 0.661 | 0.570–0.750 | 0.585 |
| Mean token entropy | distribution | 0.653 | 0.558–0.739 | 0.634 |
| Maximum token entropy | distribution | 0.696 | 0.609–0.777 | 0.590 |
| Top-3 token ambiguity | distribution | 0.584 | 0.478–0.676 | 0.567 |
| Mean token ambiguity | distribution | 0.619 | 0.514–0.714 | 0.596 |
| Maximum token ambiguity | distribution | 0.632 | 0.536–0.719 | 0.584 |
| Mean hidden cosine change | hidden dynamics | 0.555 | 0.451–0.652 | 0.598 |
| Maximum hidden cosine change | hidden dynamics | 0.517 | 0.410–0.624 | 0.555 |
| Top-3 hidden cosine change | hidden dynamics | 0.528 | 0.426–0.627 | 0.598 |
| Mean hidden-state RMS norm | hidden dynamics | 0.473 | 0.379–0.570 | 0.462 |
| Hidden-state norm variability | hidden dynamics | 0.611 | 0.514–0.704 | 0.577 |

Descriptively highest non-length method: `lexical_disagreement_3`.
Do not read that as confirmation or SOTA: model, task, label, sampling budget, and split differ from published studies.
