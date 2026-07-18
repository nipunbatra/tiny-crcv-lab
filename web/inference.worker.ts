/// <reference lib="webworker" />

import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers';
import { BROWSER_MODEL_DTYPE, DETERMINISTIC_GENERATION } from './lib/inference-config';
import { liveFeatures } from './lib/metrics';
import { exposeHiddenOutput, HIDDEN_NAME } from './lib/onnx-patch';
import type { LiveToken, ModelKind, WorkerToMain } from './types';

type RunMessage = {
  type: 'run';
  model: ModelKind;
  question: string;
  runtime: 'webgpu' | 'wasm';
  maxNewTokens: number;
};

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const send = (message: WorkerToMain) => ctx.postMessage(message);

function modelRepo(kind: ModelKind): string {
  return kind === 'instruct'
    ? 'onnx-community/Qwen2.5-0.5B-Instruct'
    : 'onnx-community/Qwen2.5-0.5B-ONNX';
}

async function downloadBlob(url: string): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Model download failed (${response.status})`);
  const total = Number(response.headers.get('content-length') ?? 0);
  if (!response.body) return response.blob();
  let loaded = 0;
  const tracked = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      loaded += chunk.byteLength;
      send({
        type: 'progress',
        file: url.split('/').at(-1) ?? 'model.onnx',
        loaded,
        total,
        progress: total ? (loaded / total) * 100 : 0,
      });
      controller.enqueue(chunk);
    },
  }));
  return new Response(tracked, { headers: { 'content-type': 'application/octet-stream' } }).blob();
}

function installPatchedCache(): void {
  env.useBrowserCache = false;
  env.useCustomCache = true;
  const cachePromise = caches.open('tiny-crcv-models-v1').catch(() => null);
  env.customCache = {
    async match(request: string) {
      const url = String(request);
      const isRemote = url.startsWith('https://');
      const isOnnx = /\/onnx\/model_q4(?:f16)?\.onnx(?:\?|$)/.test(url);
      const cache = await cachePromise;
      if (!isRemote) return undefined;
      const patchedKey = `${url}${url.includes('?') ? '&' : '?'}tiny-crcv-hidden=v1`;
      const cached = await cache?.match(patchedKey);
      if (cached) return cached;
      if (!isOnnx) return cache?.match(url);

      send({ type: 'status', phase: 'Model', detail: 'downloading public ONNX weights' });
      const original = await downloadBlob(url);
      send({ type: 'status', phase: 'Graph', detail: 'exposing the existing final hidden-state tensor' });
      const patched = await exposeHiddenOutput(original, url.includes('q4f16') ? 10 : 1);
      const response = new Response(patched, {
        headers: { 'content-type': 'application/octet-stream', 'content-length': String(patched.size) },
      });
      try { await cache?.put(patchedKey, response.clone()); } catch { /* Quota is optional; this run can continue. */ }
      return response;
    },
    async put(request: string, response: Response) {
      try { await (await cachePromise)?.put(String(request), response); } catch { /* Small config caching is optional. */ }
    },
  };
}

function halfToNumber(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function valueAt(data: ArrayLike<number>, index: number, type: string): number {
  return type === 'float16' && data instanceof Uint16Array ? halfToNumber(data[index]) : Number(data[index]);
}

function lastVector(tensor: any): Float32Array {
  const data = tensor.data as ArrayLike<number>;
  const width = tensor.dims.at(-1) as number;
  const offset = data.length - width;
  const result = new Float32Array(width);
  for (let i = 0; i < width; i += 1) result[i] = valueAt(data, offset + i, tensor.type);
  return result;
}

function normalizedShift(current: Float32Array, previous: Float32Array | null): number | null {
  if (!previous) return null;
  let squaredDelta = 0;
  let squaredPrevious = 0;
  for (let index = 0; index < current.length; index += 1) {
    squaredDelta += (current[index] - previous[index]) ** 2;
    squaredPrevious += previous[index] ** 2;
  }
  return Math.sqrt(squaredDelta) / (Math.sqrt(squaredPrevious) + 1e-8);
}

function hiddenObservables(current: Float32Array, previous: Float32Array | null): { hiddenShift: number | null; hiddenCosineDistance: number | null; hiddenNorm: number } {
  let squaredCurrent = 0;
  let squaredPrevious = 0;
  let dot = 0;
  for (let index = 0; index < current.length; index += 1) {
    squaredCurrent += current[index] ** 2;
    if (previous) {
      squaredPrevious += previous[index] ** 2;
      dot += current[index] * previous[index];
    }
  }
  const hiddenNorm = Math.sqrt(squaredCurrent / current.length);
  if (!previous) return { hiddenShift: null, hiddenCosineDistance: null, hiddenNorm };
  const cosine = dot / (Math.sqrt(squaredCurrent) * Math.sqrt(squaredPrevious) + 1e-8);
  return {
    hiddenShift: normalizedShift(current, previous),
    hiddenCosineDistance: Math.max(0, Math.min(2, 1 - cosine)),
    hiddenNorm,
  };
}

function greedyToken(logits: any): { tokenId: number; confidence: number; margin: number; entropy: number } {
  const data = logits.data as ArrayLike<number>;
  const vocab = logits.dims.at(-1) as number;
  const offset = data.length - vocab;
  let tokenId = 0;
  let maximum = Number.NEGATIVE_INFINITY;
  let secondMaximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < vocab; index += 1) {
    const value = valueAt(data, offset + index, logits.type);
    if (value > maximum) {
      secondMaximum = maximum;
      maximum = value;
      tokenId = index;
    } else if (value > secondMaximum) {
      secondMaximum = value;
    }
  }
  let denominator = 0;
  let weightedShiftedLogit = 0;
  for (let index = 0; index < vocab; index += 1) {
    const shifted = valueAt(data, offset + index, logits.type) - maximum;
    const weight = Math.exp(shifted);
    denominator += weight;
    weightedShiftedLogit += weight * shifted;
  }
  const confidence = 1 / denominator;
  const secondConfidence = Math.exp(secondMaximum - maximum) / denominator;
  const entropy = (Math.log(denominator) - weightedShiftedLogit / denominator) / Math.log(vocab);
  return { tokenId, confidence, margin: confidence - secondConfidence, entropy };
}

function repetitionWarning(tokenIds: number[]): string | null {
  if (tokenIds.length < 8) return null;
  const trigrams = tokenIds.slice(0, -2).map((_, index) => tokenIds.slice(index, index + 3).join(','));
  const duplicateFraction = 1 - new Set(trigrams).size / trigrams.length;
  return duplicateFraction >= 0.25
    ? 'Repeated token patterns detected. Treat this as a degenerate generation, not a meaningful factual answer.'
    : null;
}

function eosIds(model: any, tokenizer: any): Set<number> {
  const configured = model.generation_config?.eos_token_id ?? model.config?.eos_token_id ?? tokenizer.eos_token_id;
  return new Set(Array.isArray(configured) ? configured.map(Number) : [Number(configured)]);
}

function promptInputs(tokenizer: any, model: ModelKind, question: string): any {
  if (model === 'base') {
    return tokenizer(`Question: ${question}\nAnswer with one short factual sentence and no explanation:`);
  }
  return tokenizer.apply_chat_template([
    {
      role: 'system',
      content: 'Answer factual questions directly and state your best answer even if uncertain. Follow the requested sentence frame exactly and add no facts.',
    },
    {
      role: 'user',
      content: `${question}\nReplace only the bracketed text in this frame: 'The requested answer is [short answer], stated as my best factual response.' Add no explanation or supporting detail.`,
    },
  ], { add_generation_prompt: true, return_dict: true });
}

async function selectDtype(runtime: RunMessage['runtime']): Promise<{ dtype: 'q4'; detail: string }> {
  if (runtime === 'wasm') return { dtype: BROWSER_MODEL_DTYPE, detail: 'WASM q4' };
  const gpu = (ctx.navigator as any).gpu;
  const adapter = await gpu?.requestAdapter();
  if (!adapter) throw new Error('WebGPU was selected, but this browser could not provide a GPU adapter. Choose WASM and retry.');
  // Keep WebGPU and WASM on the same q4 graph. The smaller q4f16 graph can
  // change greedy tokens on some adapters, which makes backend comparisons and
  // factual demonstrations untrustworthy.
  return { dtype: BROWSER_MODEL_DTYPE, detail: 'WebGPU q4 (same graph as WASM)' };
}

async function run(message: RunMessage): Promise<void> {
  installPatchedCache();
  const repo = modelRepo(message.model);
  const selected = await selectDtype(message.runtime);
  const dtype = selected.dtype;
  send({ type: 'status', phase: 'Tokenizer', detail: `loading ${repo}` });
  const tokenizer = await AutoTokenizer.from_pretrained(repo);
  send({ type: 'status', phase: 'Runtime', detail: `initializing ${selected.detail}` });
  const model: any = await AutoModelForCausalLM.from_pretrained(repo, {
    dtype,
    device: message.runtime,
    progress_callback: (event: any) => {
      if (event.status === 'progress' && event.progress != null) {
        send({ type: 'progress', file: event.file ?? 'model', loaded: event.loaded ?? 0, total: event.total ?? 0, progress: event.progress });
      }
    },
  });
  if (!model.sessions.model.outputNames.includes(HIDDEN_NAME)) {
    throw new Error('The ONNX graph loaded without the requested hidden-state output. Clear this site’s model cache and retry.');
  }
  try {
    send({ type: 'ready', model: message.model, runtime: message.runtime, dtype });
    send({ type: 'status', phase: 'Inference', detail: 'greedy decoding and capturing the same forward-pass signals' });

    const start = performance.now();
    const generationInputs: any = promptInputs(tokenizer, message.model, message.question);
    const promptLength = generationInputs.input_ids.dims.at(-1) as number;
    const tracedTokens: LiveToken[] = [];
    let previousHidden: Float32Array | null = null;
    const originalForward = model.forward.bind(model);

    // Capture logits and hidden state from the exact forward call used by
    // Transformers.js generation. This avoids a second numerically independent
    // replay, which can disagree on GPU when two logits are close.
    model.forward = async (inputs: any) => {
      const outputs: any = await originalForward(inputs);
      const { tokenId, confidence, margin, entropy } = greedyToken(outputs.logits);
      const hidden = lastVector(outputs[HIDDEN_NAME]);
      tracedTokens.push({
        tokenId,
        token: tokenizer.decode([tokenId], { skip_special_tokens: false }),
        confidence,
        margin,
        entropy,
        ...hiddenObservables(hidden, previousHidden),
      });
      previousHidden = hidden;
      return outputs;
    };

    let generated: any;
    try {
      generated = await model.generate({
        ...generationInputs,
        ...DETERMINISTIC_GENERATION,
        max_new_tokens: message.maxNewTokens,
      });
    } finally {
      model.forward = originalForward;
    }

    const sequenceTensor = generated.sequences ?? generated;
    const fullSequence = (sequenceTensor.tolist()[0] as bigint[]).map(Number);
    const generatedWithEos = fullSequence.slice(promptLength);
    sequenceTensor.dispose?.();
    if (tracedTokens.length < generatedWithEos.length) {
      throw new Error(`Generation produced ${generatedWithEos.length} tokens but exposed only ${tracedTokens.length} signal steps.`);
    }
    generatedWithEos.forEach((generatedTokenId, index) => {
      if (tracedTokens[index].tokenId !== generatedTokenId) {
        throw new Error(`Generation signal mismatch at step ${index + 1} (${generatedTokenId} vs ${tracedTokens[index].tokenId}).`);
      }
    });

    const eos = eosIds(model, tokenizer);
    const tokens = tracedTokens.slice(0, generatedWithEos.length).filter((token) => !eos.has(token.tokenId));
    tokens.forEach((token) => send({ type: 'token', token }));
    const generatedIds = tokens.map((token) => token.tokenId);
    const answer = tokenizer.decode(generatedIds, { skip_special_tokens: true }).trim();
    const elapsedMs = performance.now() - start;
    send({ type: 'result', answer, tokens, features: liveFeatures(tokens), elapsedMs, qualityWarning: repetitionWarning(generatedIds) });
  } finally {
    await model.dispose();
  }
}

ctx.onmessage = (event: MessageEvent<RunMessage>) => {
  if (event.data.type !== 'run') return;
  run(event.data).catch((error: unknown) => {
    send({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  });
};
