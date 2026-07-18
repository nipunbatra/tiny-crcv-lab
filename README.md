# Tiny CRCV Lab

An inspectable experiment asking one question:

> On short factual answers from a 0.49B open-weight model, does instability in
> confidence × hidden-state movement predict which answers are wrong?

**Result:** CRCV was not convincing, but several tiny distribution-based scores
were useful. On the Instruct checkpoint, skipping the first token and averaging
the three largest surprises reached 0.827 held-out AUROC; top-3 normalized token
entropy reached 0.825; the earlier top-3 selected-token surprise reached 0.817;
and CRCV reached 0.594. These are exploratory rankings on a reused 50-question
test split—not a validated detector. A separately frozen 50-pair slice of the
real [HaluEval QA benchmark](https://github.com/RUCAIBox/HaluEval) is now included
as an external, teacher-forced candidate-discrimination check.

[Open the beginner-friendly browser lab](https://nipunbatra.github.io/tiny-crcv-lab/).
The default view explains the result in plain language; **Evidence** compares all
24 scalar methods side by side on the Instruct run, Base stress check, and
HaluEval slice, then exposes each formula, confidence interval, and worked
example. It also contains all 100 saved generations for each model; **Try it**
performs inference inside the browser and exposes every raw per-token input.

This is a research smoke test, not a production detector. It uses 100 fixed
questions, calibrates score thresholds on 50, and reports results once on 50
held-out questions. Every answer and token-level signal is saved for audit.

The same audit is available as a fully static browser app. It runs on GitHub
Pages, uses Transformers.js with WebGPU (or WASM), and sends no prompt or model
trace to an application server.

## Run the browser lab

```bash
npm ci
npm run test:web
npm run dev
```

The saved 100-question results load immediately. The **Try it** panel downloads
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

`npm run build` creates `dist/`. The included GitHub Actions workflow tests,
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

The model is asked to place its short answer inside a fixed, content-neutral
sentence frame. This creates enough generation steps for a window without
inviting extra factual claims. The primary answer score is the mean of those
windowed standard deviations.
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
  greedy decoding are tested.
- The held-out set has only 50 examples, so uncertainty will be wide.
- The external HaluEval slice has only 25 held-out pairs and a severe answer-
  length/style confound.
- Alias matching can mislabel nuanced answers; inspect `predictions.jsonl`.
- The result is useful even if CRCV performs at or below chance: that falsifies
  this tiny version and prevents premature detector claims.
