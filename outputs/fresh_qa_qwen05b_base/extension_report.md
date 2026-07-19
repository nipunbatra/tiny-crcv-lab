# Post-hoc fast-baseline extension

Checkpoint: **base**. This is exploratory because the held-out outputs were already inspected.

## Headline methods

| Method | Family | AUROC | 95% CI | AURAC | Macro-F1 | Extra s/question |
|---|---|---:|---:|---:|---:|---:|
| Top-3 token surprise | token probability | 0.599 | 0.489–0.705 | 0.119 | 0.527 | 0.000000 |
| CRCV (primary) | CRCV | 0.537 | 0.422–0.646 | 0.122 | 0.552 | 0.000000 |
| Mean token surprise | token probability | 0.579 | 0.456–0.698 | 0.120 | 0.516 | 0.000000 |
| Top-3 token entropy | distribution | 0.606 | 0.509–0.700 | 0.113 | 0.501 | 0.000000 |
| P(False self-check) | P(True)-style self-evaluation | 0.628 | 0.525–0.728 | 0.115 | 0.536 | 0.112848 |
| Three-answer lexical disagreement | SelfCheckGPT-inspired | 0.601 | 0.474–0.717 | 0.117 | 0.522 | 2.435995 |
| Three-sample discrete semantic entropy proxy | semantic-entropy proxy | 0.600 | 0.507–0.686 | 0.115 | 0.476 | 2.853661 |
| Three-answer semantic disagreement | sample consistency | 0.595 | 0.497–0.693 | 0.112 | 0.488 | 2.853661 |
| Mean-hidden linear probe | supervised internal state | 0.545 | 0.429–0.654 | 0.112 | 0.502 | 0.000073 |
| Eight-feature trace logistic | supervised trace ensemble | 0.627 | 0.510–0.727 | 0.141 | 0.533 | 0.000001 |
| Depth-2 trace tree | supervised trace ensemble | 0.609 | 0.505–0.709 | 0.143 | 0.525 | 0.000001 |
| Answer length | control | 0.528 | 0.429–0.638 | 0.129 | 0.552 | 0.000000 |

## All trace-only scalars

| Scalar | Family | AUROC | 95% CI | Macro-F1 |
|---|---|---:|---:|---:|
| CRCV (primary) | CRCV | 0.537 | 0.422–0.646 | 0.552 |
| CRCV maximum | CRCV | 0.522 | 0.392–0.648 | 0.552 |
| Mean token surprise | token probability | 0.579 | 0.456–0.698 | 0.516 |
| Confidence variability | token probability | 0.627 | 0.519–0.726 | 0.550 |
| Hidden-shift variability | hidden dynamics | 0.573 | 0.460–0.683 | 0.552 |
| Answer length | control | 0.528 | 0.429–0.638 | 0.552 |
| Top-3 token surprise | token probability | 0.599 | 0.489–0.705 | 0.527 |
| Worst-token surprise | token probability | 0.635 | 0.528–0.742 | 0.539 |
| Token-surprise spread | token probability | 0.630 | 0.518–0.730 | 0.582 |
| Top-4 token surprise | token probability | 0.596 | 0.492–0.695 | 0.502 |
| Top-3 surprise after first token | token probability | 0.635 | 0.530–0.732 | 0.548 |
| Top-3 surprise × hidden shift | coupled | 0.619 | 0.508–0.721 | 0.525 |
| Top-3 uncertainty × hidden shift | coupled | 0.569 | 0.457–0.672 | 0.537 |
| Top-3 token entropy | distribution | 0.606 | 0.509–0.700 | 0.501 |
| Mean token entropy | distribution | 0.577 | 0.450–0.701 | 0.474 |
| Maximum token entropy | distribution | 0.626 | 0.517–0.723 | 0.532 |
| Top-3 token ambiguity | distribution | 0.563 | 0.470–0.658 | 0.469 |
| Mean token ambiguity | distribution | 0.517 | 0.379–0.655 | 0.503 |
| Maximum token ambiguity | distribution | 0.651 | 0.553–0.745 | 0.546 |
| Mean hidden cosine change | hidden dynamics | 0.490 | 0.384–0.597 | 0.527 |
| Maximum hidden cosine change | hidden dynamics | 0.515 | 0.400–0.636 | 0.527 |
| Top-3 hidden cosine change | hidden dynamics | 0.500 | 0.391–0.610 | 0.527 |
| Mean hidden-state RMS norm | hidden dynamics | 0.552 | 0.450–0.655 | 0.490 |
| Hidden-state norm variability | hidden dynamics | 0.629 | 0.524–0.729 | 0.527 |

Descriptively highest non-length method: `token_ambiguity_max`.
Do not read that as confirmation or SOTA: model, task, label, sampling budget, and split differ from published studies.
