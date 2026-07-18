# Tiny CRCV Lab

An inspectable experiment asking one question:

> On short factual answers from a 0.49B open-weight model, does instability in
> confidence × hidden-state movement predict which answers are wrong?

**Result:** not convincingly in this small test. The proposed CRCV score reached
0.594 AUROC, while a simpler confidence-variability score reached 0.777. The
experiment is useful because it makes that negative result easy to inspect.

[Open the beginner-friendly browser lab](https://nipunbatra.github.io/tiny-crcv-lab/).
The default view explains the result in plain language; **Explore details**
contains all 100 generations and arithmetic; **Run locally** performs inference
inside the browser.

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

The saved 100-question results load immediately. The **Run in browser** panel
downloads a quantized public ONNX model on first use (about 483 MB with WebGPU
or 750 MB with WASM) and caches it when the browser has enough quota.

The public ONNX exports normally expose only logits and KV-cache tensors. They
still compute the final normalized hidden state immediately before `lm_head`.
The app makes that existing 896-value tensor an additional graph output with a
small in-browser protobuf edit, then computes the same CRCV definition as the
Python code. This is not a KV-cache proxy and adds no neural-network layer.

`npm run build` creates `dist/`. The included GitHub Actions workflow tests,
builds, and deploys that directory to GitHub Pages; enable **Pages → GitHub
Actions** in the repository settings.

## Method in one minute

For every greedily generated answer token `t`:

1. Record the probability `c_t` assigned to the selected token.
2. Record the final-layer state used to predict it.
3. Compute normalized movement `r_t` from the preceding generation step.
4. Form `s_t = c_t * r_t`.
5. Compute the sample standard deviation of `s_t` in trailing five-step windows.

The model is asked to place its short answer inside a fixed, content-neutral
sentence frame. This creates enough generation steps for a window without
inviting extra factual claims. The primary answer score is the mean of those
windowed standard deviations.
Higher scores are declared more likely to be wrong before looking at test data.
Confidence variability, hidden-shift variability, token surprise, and answer
length are evaluated as baselines. Correctness is an intentionally transparent
lexical proxy: at least one normalized accepted answer must occur in the output.

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

For a quick end-to-end check before the full benchmark:

```bash
uv run tiny-crcv --limit 10 --output-dir outputs/smoke_10 --overwrite
```

Outputs:

- `predictions.jsonl`: raw generations, labels, token probabilities, hidden
  shifts, and all answer-level scores.
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
| CRCV mean (predeclared primary) | 0.594 | 0.419–0.766 |
| CRCV maximum (secondary) | 0.674 | 0.506–0.830 |
| Confidence variability | **0.777** | **0.632–0.904** |

The primary CRCV statistic did not establish better-than-chance discrimination.
It also trailed the confidence-only baseline by 0.183 AUROC (paired bootstrap
95% CI: 0.018 to 0.345 worse). This tiny experiment therefore does **not**
validate the CRCV hypothesis; it says the simpler confidence-only detector is
the better starting point for the next iteration. The complete run took 24.4
seconds of generation on Apple MPS, excluding model loading.

The included base-model run is intentionally the same small stress test. Its
primary CRCV AUROC was 0.508 (95% CI 0.328–0.677), while mean token surprise was
0.666. The base model frequently continued the prompt and hit the 24-token
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
- Only one model, one layer, and greedy decoding are tested.
- The held-out set has only 50 examples, so uncertainty will be wide.
- Alias matching can mislabel nuanced answers; inspect `predictions.jsonl`.
- The result is useful even if CRCV performs at or below chance: that falsifies
  this tiny version and prevents premature detector claims.
