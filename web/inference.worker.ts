/// <reference lib="webworker" />

import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers';
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

function greedyToken(logits: any): { tokenId: number; confidence: number } {
  const data = logits.data as ArrayLike<number>;
  const vocab = logits.dims.at(-1) as number;
  const offset = data.length - vocab;
  let tokenId = 0;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < vocab; index += 1) {
    const value = valueAt(data, offset + index, logits.type);
    if (value > maximum) { maximum = value; tokenId = index; }
  }
  let denominator = 0;
  for (let index = 0; index < vocab; index += 1) {
    denominator += Math.exp(valueAt(data, offset + index, logits.type) - maximum);
  }
  return { tokenId, confidence: 1 / denominator };
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

async function run(message: RunMessage): Promise<void> {
  installPatchedCache();
  const repo = modelRepo(message.model);
  const dtype = message.runtime === 'webgpu' ? 'q4f16' : 'q4';
  send({ type: 'status', phase: 'Tokenizer', detail: `loading ${repo}` });
  const tokenizer = await AutoTokenizer.from_pretrained(repo);
  send({ type: 'status', phase: 'Runtime', detail: `initializing ${message.runtime.toUpperCase()} (${dtype})` });
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
  send({ type: 'ready', model: message.model, runtime: message.runtime, dtype });
  send({ type: 'status', phase: 'Inference', detail: 'greedy decoding and tracing token states' });

  const start = performance.now();
  let modelInputs: any = promptInputs(tokenizer, message.model, message.question);
  const allInputIds: bigint[][] = modelInputs.input_ids.tolist();
  const generatedIds: number[] = [];
  const tokens: LiveToken[] = [];
  let previousHidden: Float32Array | null = null;
  const eos = eosIds(model, tokenizer);
  const generationConfig = model.generation_config ?? {};

  for (let step = 0; step < message.maxNewTokens; step += 1) {
    modelInputs = model.prepare_inputs_for_generation(allInputIds, modelInputs, generationConfig);
    const outputs: any = await model.forward(modelInputs);
    const { tokenId, confidence } = greedyToken(outputs.logits);
    if (eos.has(tokenId)) break;
    const hidden = lastVector(outputs[HIDDEN_NAME]);
    const token: LiveToken = {
      tokenId,
      token: tokenizer.decode([tokenId], { skip_special_tokens: false }),
      confidence,
      hiddenShift: normalizedShift(hidden, previousHidden),
    };
    previousHidden = hidden;
    generatedIds.push(tokenId);
    tokens.push(token);
    send({ type: 'token', token });
    allInputIds[0].push(BigInt(tokenId));
    modelInputs = model._update_model_kwargs_for_generation({
      generated_input_ids: [[BigInt(tokenId)]],
      outputs,
      model_inputs: modelInputs,
      is_encoder_decoder: false,
    });
  }

  const answer = tokenizer.decode(generatedIds, { skip_special_tokens: true }).trim();
  const elapsedMs = performance.now() - start;
  await model.dispose();
  send({ type: 'result', answer, tokens, features: liveFeatures(tokens), elapsedMs });
}

ctx.onmessage = (event: MessageEvent<RunMessage>) => {
  if (event.data.type !== 'run') return;
  run(event.data).catch((error: unknown) => {
    send({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  });
};
