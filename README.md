# Tiny CRCV Lab

An inspectable experiment asking one question:

> On short factual answers from a 0.49B open-weight model, does instability in
> confidence × hidden-state movement predict which answers are wrong?

**Result:** token uncertainty transfers across two small model families. On the
same 300 held-out questions, token-surprise spread reached 0.706 AUROC for
Qwen2.5-0.5B-Instruct and 0.731 for SmolLM2-360M-Instruct. The original frozen
top-3 surprise baseline reached 0.656 and 0.715. CRCV transferred more weakly
(0.634 / 0.588), while pure hidden-cosine movement did not. Treat every score as
triage, not factual verification.

[Open the beginner-friendly browser lab](https://nipunbatra.github.io/tiny-crcv-lab/).
The default view explains the result in plain language; **Evidence** first shows
the cross-family replication, 12 headline baselines, and the complete 31-method
audit on Qwen Instruct, SmolLM2 Instruct, Qwen Base, NQ-Open, TriviaQA, and
TruthfulQA. It can reconstruct CRCV, sampled
consistency, semantic-entropy, shallow-tree, logistic, probe, and token-level
scores for any of the 600 questions from raw saved inputs. The earlier 100-row
sweep and HaluEval slice remain in one collapsed section. **Try it** performs
inference inside the browser and exposes every raw per-token input.

[Open the 70-page animated presentation](https://nipunbatra.github.io/tiny-crcv-lab/talk/s/tiny-hallucination-detector).
It defines the exact question-plus-answer detection task up front, shows real
correct and wrong benchmark outputs, and works one held-out correct/wrong pair
through tokenization, generation, CRCV, eight other detector calculations,
calibration, selective answering, expanded baselines, and published-method
context. Every detector value in the worked example comes from the saved
experiment traces. Press **F** for fullscreen presentation mode.

This is a research prototype, not a production detector. The fresh comparison
uses 300 questions for thresholds/probe fitting and reports results on 300 held
out questions. Every answer, self-check probability, stochastic sample, pair
judgment, probe contribution, and token-level signal is saved for audit.

The same audit is available as a fully static browser app. It runs on GitHub
Pages and sends no prompt or model trace to an application server. The live
WebGPU/WASM console remains the separately verified Qwen path; the SmolLM2 panel
exposes all saved benchmark traces and calculations.

## Run the browser lab

```bash
npm ci
npm run test:web
npm run dev
```

The summary results load immediately; the larger fresh trace files load only
when the per-question inspector opens. The **Try it** panel downloads
a roughly 750 MB quantized public ONNX model on first use. WebGPU and WASM now
deliberately use the same q4 graph; the earlier automatic q4f16 switch could
change greedy tokens on some GPU adapters. The page exposes Auto, WebGPU, and
WASM runtime controls and caches the patched graph when browser quota permits.

The public ONNX exports normally expose only logits and KV-cache tensors. They
still compute the final normalized hidden state immediately before `lm_head`.
The app makes that existing 896-value tensor an additional graph output with a
small in-browser protobuf edit. It generates with Transformers.js's standard
deterministic path while intercepting the exact forward-pass outputs used to
select each token. Sampling and the model card's repetition penalty are disabled,
so displayed confidence is the raw distribution that chose the token. There is
no second, numerically independent replay. A repeated-trigram guard visibly
invalidates degenerate output. This is not a KV-cache proxy and adds no
neural-network layer.

Install the isolated presentation workspace once with `npm ci --prefix talk`.
`npm run build` then creates both the lab and `/talk/` presentation under
`dist/`. The included GitHub Actions workflow installs both workspaces, tests,
builds, and deploys that directory to GitHub Pages; enable **Pages → GitHub
Actions** in the repository settings.

## Method in one minute

For every greedily generated answer token `t`, the same forward pass records:

1. selected-token probability `c_t`, top-two probability margin, and full-vocab
   entropy normalized by `ln(vocabulary size)`;
2. the 896-value final-layer state, its RMS norm, normalized L2 movement, and
   cosine change from the preceding step;
3. simple answer summaries such as means, maxima, top-3/top-4 averages, local
   variability, and confidence × movement couplings.

CRCV forms `s_t = c_t * r_t` and takes the sample standard deviation of `s_t`
inside complete trailing five-step windows. The browser shows the inputs,
formula, numerical substitution, and result for all 24 evaluated summaries.

The Instruct model receives a direct “short answer, no explanation” chat prompt;
the Base checkpoint receives `Question: ...\nAnswer:`. The original bracketed
sentence frame was removed after a runtime feasibility trace showed that this
tiny model copied the placeholder. That repair happened before either complete
run, probe fitting, or aggregate metric calculation and is explicitly recorded
as a protocol amendment.
Higher scores are declared more likely to be wrong before looking at test data.
Confidence variability, hidden-shift variability, token surprise, and answer
length are evaluated as baselines. Correctness is an intentionally transparent
lexical proxy: at least one normalized accepted answer must occur in the output.

The first follow-up added three confidence-only summaries, with no learned
classifier:

- top-3 token surprise: the mean of the three largest `-ln(c_t)` values;
- worst-token surprise: the largest `-ln(c_t)` value;
- surprise spread: sample standard deviation of all `-ln(c_t)` values.

The next bounded sweep added top-4/skip-first variants plus equally cheap margin,
normalized-entropy, cosine-change, hidden-norm, and uncertainty × movement
summaries. Top-3 entropy was 0.008 AUROC above top-3 surprise on Instruct, with a
paired 95% interval of -0.056 to 0.076; on Base it was 0.095 lower. That is a
useful warning: full-distribution entropy is intuitive, but selected-token
surprise was more robust across these two checkpoints. Because the same
50-question held-out set has now been inspected during iteration, all additions
must be confirmed on fresh questions.

## Fresh 600-question comparison

The sampling indices, calibration/test split, models, generation settings,
methods, probe optimizer, thresholds, metrics, and improvement rule were frozen
in `experiments/fresh_qa_600_protocol.json`. Each model answered 200 NQ-Open,
200 TriviaQA, and 200 TruthfulQA questions. Within each dataset, 100 questions
were calibration and 100 were held out.

| Method | Instruct AUROC (95% CI) | Base AUROC (95% CI) | Mean extra local cost |
|---|---:|---:|---:|
| Top-3 token surprise | **0.656 (0.558–0.743)** | 0.599 (0.494–0.703) | none |
| P(False) self-check | 0.603 (0.508–0.696) | **0.628 (0.523–0.728)** | 0.130 / 0.113 s |
| Mean-hidden linear probe | 0.549 (0.453–0.641) | 0.545 (0.433–0.655) | one dot product after fitting |
| Three-answer disagreement | 0.621 (0.515–0.721) | 0.595 (0.491–0.691) | 1.901 / 2.854 s |
| Answer length control | 0.540 (0.432–0.644) | 0.528 (0.430–0.632) | none |

No added method's paired bootstrap AUROC-difference interval was entirely above
zero versus frozen top-3 surprise. P(False) was descriptively best on Base, but
its +0.029 difference had a 95% interval from -0.083 to +0.150. The practical v0
therefore remains top-3 surprise, with answer length always reported and
P(False) optional as a second diagnostic. The probe and three-sample method add
complexity or latency without established benefit here.

### Post-hoc 31-method extension

After the five-method held-out outputs had already been inspected, a second
protocol froze a broader audit in
`experiments/fresh_qa_600_extension_protocol.json`. It reuses the unchanged 600
questions and saved traces, reports all 24 pre-existing scalar features, and
adds two low-budget literature-inspired scores plus calibration-only 8-feature
logistic and depth-2 tree baselines. Because this happened after test inspection,
the rankings below are hypotheses for a new confirmatory split, not validated
winners.

| Representative extension method | Instruct AUROC (95% CI) | Base AUROC (95% CI) | Extra local cost |
|---|---:|---:|---:|
| Three-answer lexical disagreement | **0.717 (0.623–0.801)** | 0.601 (0.474–0.717) | 1.462 / 2.436 s |
| Token-surprise spread | 0.706 (0.620–0.790) | 0.630 (0.518–0.730) | none |
| Maximum token entropy | 0.696 (0.609–0.777) | 0.626 (0.517–0.723) | none |
| Eight-feature trace logistic | 0.695 (0.603–0.777) | 0.627 (0.510–0.727) | about 2 / 1 μs |
| Maximum token ambiguity | 0.632 (0.536–0.719) | **0.651 (0.553–0.745)** | none |
| CRCV mean | 0.634 (0.543–0.712) | 0.537 (0.422–0.646) | none |
| Three-sample discrete semantic-entropy proxy | 0.500 (0.434–0.553) | 0.600 (0.507–0.686) | 1.901 / 2.854 s |

No extension method's paired 95% AUROC-difference interval was entirely above
zero versus top-3 surprise. The Instruct cost frontier makes surprise spread the
most interesting free follow-up; the Base result instead favors maximum token
ambiguity. The disagreement is evidence against one checkpoint-independent
magic score.

### Prospective SmolLM2 cross-model replication

Before generating any SmolLM2 answer, the checkpoint revision, chat prompt,
generation settings, all 31 methods, calibration fits, thresholds, metrics, and
transfer rule were frozen in
`experiments/fresh_qa_600_smollm2_replication_protocol.json`. The 600 questions
and prior Qwen results were already known, so this is prospective new-model
evidence on reused data—not a second untouched-dataset confirmation.

| Representative method | Qwen Instruct AUROC | SmolLM2 Instruct AUROC | Frozen transfer rule |
|---|---:|---:|:---:|
| Token-surprise spread | **0.706** | **0.731** | passes |
| Maximum token entropy | 0.696 | 0.723 | passes |
| Eight-feature trace logistic | 0.695 | 0.724 | passes |
| Three-answer lexical disagreement | 0.717 | 0.691 | passes |
| Worst-token surprise | 0.689 | 0.745 | passes |
| Top-3 token surprise | 0.656 | 0.715 | passes |
| CRCV mean | 0.634 | 0.588 | passes |
| Mean hidden cosine change | 0.555 | 0.390 | **fails** |

Twenty-five of 31 methods pass the predeclared direction rule: pooled AUROC
above 0.5 for both models and at least two of three dataset slices above 0.5 for
each model. Nineteen methods also have pooled 95% interval lower bounds above
0.5 on both models. Surprise spread has the highest worst-model AUROC (0.706),
so it is now the lead free candidate alongside the frozen top-3 anchor. Its
paired improvement interval versus top-3 still crosses zero on SmolLM2; this is
not a statistically established win.

The three-sample lexical score adapts the consistency idea from
[SelfCheckGPT](https://aclanthology.org/2023.emnlp-main.557/). The discrete
semantic-entropy score is explicitly a budget proxy: the
[Nature method](https://www.nature.com/articles/s41586-024-07421-0) normally uses
ten QA generations and bidirectional-entailment clustering. SAR, INSIDE /
EigenScore, and Lookback Lens are explained but not assigned fake scores because
the frozen traces lack, respectively, relevance passes, hidden embeddings for
sampled answers, and attention to a grounding passage.

The label is intentionally mechanical: a normalized accepted answer must occur
as a whole phrase. This is especially strict for TruthfulQA's sentence-length
answers. Base had no alias matches in the 100-question held-out TruthfulQA slice,
so that slice's AUROC is undefined; the site displays “—” and exposes every row.
These results evaluate strict answer matching, not human-adjudicated factuality.

## Detector-building protocol

Yes: for a one-score detector, the clean process is to define the score first,
fit its operational cutoff on calibration data, and report performance once on
held-out data. This repository uses two related but distinct quantities:

- **AUROC needs no cutoff.** It measures how often a randomly chosen wrong
  answer receives a higher risk score than a randomly chosen correct answer.
- **Macro-F1 and the confusion matrix need a cutoff.** The cutoff that maximizes
  calibration macro-F1 is frozen, then applied unchanged to the test split.

If several scores are candidates, selecting among them also consumes calibration
data. The HaluEval experiment therefore chooses the scalar with the highest
calibration AUROC and never reselects it using held-out AUROC. Repeatedly inventing
features after viewing a test split turns that split into development data; a
fresh test set is then required.

A small decision tree is a reasonable multi-feature experiment because its
rules remain inspectable. Here the tree is deliberately restricted to depth 2,
at least 8 calibration candidates per leaf, and six predeclared inputs. It learns
both the feature thresholds and the branch structure on calibration data only.
That added flexibility did not reliably help: on saved free generations the
depth-2 tree changed held-out AUROC from 0.825 to 0.660 for Instruct and from
0.666 to 0.706 for Base. It therefore remains a diagnostic, not the default.

## External HaluEval-QA slice

The protocol, seeds, model, allowed tree features, and sample indices were
committed before downloading the selected examples. The slice contains 50
HaluEval QA pairs: 25 pairs for calibration and 25 pairs held out. Each question
contributes its supplied knowledge, one right candidate, and one hallucinated
candidate, giving 100 exact token traces. Whole pairs always stay in the same
split.

| Detector | Held-out AUROC | 95% CI | Macro-F1 | Pair accuracy |
|---|---:|---:|---:|---:|
| Calibration-selected scalar: skip-first top-3 surprise | **0.994** | 0.976–1.000 | 0.940 | 1.000 |
| One-split stump | 0.900 | 0.821–0.975 | 0.899 | 0.900 |
| Depth-2 tree | 0.968 | 0.913–0.998 | 0.899 | 0.980 |
| Answer length control | 0.958 | 0.900–0.996 | 0.859 | 1.000 |
| Selected scalar after log-length residualization (post-hoc) | 0.787 | 0.644–0.915 | 0.756 | 0.800 |

The headline 0.994 is heavily confounded: the hallucinated candidate is longer
in all 25 held-out pairs. The residualized analysis fits and removes a
log-answer-length trend using calibration data, but it was designed after seeing
the length pattern and is therefore only a warning. HaluEval also uses synthetic
adversarial answers. This run tests whether Qwen assigns different token signals
to supplied right and hallucinated candidates under grounding context; it does
not test detection on Qwen's own free generation.

## Reproduce

The default model is
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
an Apache-2.0 model with 0.49B parameters. The first full run downloads roughly
1 GB of weights. Python dependencies are locked by `uv.lock`.

```bash
uv sync --extra dev
uv run pytest -q
uv run tiny-crcv --overwrite
uv run tiny-crcv --model Qwen/Qwen2.5-0.5B --prompt-style base \
  --output-dir outputs/qwen05b_base_100 --overwrite
```

The included raw token traces can be rescored after adding an answer-level
formula without rerunning either model:

```bash
uv run python scripts/rescore_saved.py \
  outputs/qwen05b_100 outputs/qwen05b_base_100
```

The bounded 18-formula search over the original saved confidence/shift traces is
also reproducible. It writes every candidate and split result—there is no hidden
winner-only notebook:

```bash
uv run python scripts/search_simple_metrics.py
```

Recreate the frozen external slice, score its exact candidates, and compare the
shallow trees with the scalar baselines:

```bash
uv run python scripts/fetch_halueval_subset.py
uv run python scripts/run_halueval_subset.py --overwrite
uv run python scripts/evaluate_shallow_trees.py
```

Recreate the frozen fresh sample and run the three stronger small baselines:

```bash
uv run python scripts/fetch_fresh_qa.py --overwrite
HF_HUB_OFFLINE=1 uv run python scripts/run_sota_comparison.py \
  --model-key instruct --overwrite
HF_HUB_OFFLINE=1 uv run python scripts/run_sota_comparison.py \
  --model-key base --overwrite

# Recompute the post-hoc 31-method audit from those frozen saved traces.
uv run python scripts/evaluate_fast_baselines.py

# Reproduce the cross-family run and its prospectively frozen 31-method audit.
uv run python scripts/run_sota_comparison.py \
  --protocol experiments/fresh_qa_600_smollm2_replication_protocol.json \
  --model-key smollm2_instruct --device mps
uv run python scripts/evaluate_fast_baselines.py \
  --model smollm2_instruct \
  --output-dir outputs/fresh_qa_smollm2_360m_instruct \
  --protocol experiments/fresh_qa_600_smollm2_replication_protocol.json \
  --seed 20260802
uv run python scripts/evaluate_cross_model_replication.py
```

Omit `HF_HUB_OFFLINE=1` on the first run so Hugging Face can download the model.
The fresh outputs include `hidden_probe.npz`, which preserves all 600 mean hidden
states plus calibration statistics and weights; the browser shows the largest
per-question contributions and the exact residual from the other dimensions.

The published files are in `outputs/halueval_qwen05b_100/`; the exact sampling
contract is `experiments/halueval_qa_50_protocol.json`. The browser's paired
candidate microscope exposes all 24 answer-level scores, every raw per-token
input, every numerical substitution, and the exact tree path.

To sanity-check standard Transformers.js generation with the same q4 ONNX model:

```bash
npm run check:onnx
```

For a quick end-to-end check before the full benchmark:

```bash
uv run tiny-crcv --limit 10 --output-dir outputs/smoke_10 --overwrite
```

Outputs:

- `predictions.jsonl`: raw generations, labels, token probabilities, margins,
  normalized entropies, hidden-state signals, and all answer-level scores.
- `metrics.json`: thresholds, AUROC with bootstrap intervals, macro-F1, and
  confusion matrices.
- `report.md`: a short human-readable result and limitations.
- `errors.md`: every answer labeled wrong, in a table for manual label audit.
- `metadata.json`: exact model commit, software versions, device, and dataset
  hash.

## Result from the included 100-question run

On the 50 held-out questions, 15 answers were labeled wrong and 35 correct.

| Score | Held-out AUROC | Bootstrap 95% CI |
|---|---:|---:|
| Skip-first top-3 surprise (exploratory) | **0.827** | **0.686–0.939** |
| Top-3 normalized entropy (exploratory) | **0.825** | **0.699–0.929** |
| Top-3 token surprise (follow-up) | 0.817 | 0.684–0.931 |
| Surprise × hidden shift, top-3 | 0.800 | 0.653–0.915 |
| Top-two ambiguity, top-3 | 0.779 | 0.629–0.902 |
| Token-surprise spread | 0.771 | 0.618–0.904 |
| CRCV mean (predeclared primary) | 0.594 | 0.419–0.766 |
| Confidence variability | 0.777 | 0.632–0.904 |

The primary CRCV statistic did not establish better-than-chance discrimination.
It also trailed the confidence-only baseline by 0.183 AUROC (paired bootstrap
95% CI: 0.018 to 0.345 worse). This tiny experiment therefore does **not**
validate the CRCV hypothesis. The small differences among the top-ranked
exploratory scores include zero in paired bootstrap comparisons. The complete
run took 24.4 seconds of generation on Apple MPS, excluding model loading.

The included base-model run is intentionally the same small stress test. Its
primary CRCV AUROC was 0.508 (95% CI 0.328–0.677), while top-3 token surprise was
0.728 and worst-token surprise was 0.737. The base model frequently continued
the prompt and hit the 24-token
limit (79/100 answers), so this comparison also measures instruction-following
and is not an architecture-only ablation. The browser UI exposes those outputs
instead of hiding the confound.

## Why WebGPU can return the hidden state

For Qwen2.5-0.5B the relevant ONNX node is
`/model/norm/Mul_1_output_0`, with shape `[batch, sequence, 896]`. It feeds
directly into `/lm_head/MatMul`. Exporters omit it from the public output
contract to reduce returned data; WebGPU itself has no restriction against
returning it. For offline inspection, the equivalent model-file transformation
is reproducible with:

```bash
uv run python scripts/expose_onnx_hidden_state.py \
  path/to/model_q4.onnx path/to/model_q4_hidden.onnx
```

## Scope and honest limitations

- A wrong short answer is used as an operational hallucination label.
- Only one small architecture, its Instruct/Base checkpoints, one layer, and
  greedy primary decoding are tested.
- The fresh held-out set has 300 examples per checkpoint, but very few strict
  alias matches, so class imbalance and wide intervals remain important.
- The external HaluEval slice has only 25 held-out pairs and a severe answer-
  length/style confound.
- Alias matching can mislabel nuanced answers; inspect `predictions.jsonl`.
- The result is useful even if CRCV performs at or below chance: that falsifies
  this tiny version and prevents premature detector claims.
