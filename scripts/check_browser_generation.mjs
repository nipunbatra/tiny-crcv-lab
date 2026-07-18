#!/usr/bin/env node

import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers';

const kind = process.argv[2] === 'base' ? 'base' : 'instruct';
const question = process.argv.slice(3).join(' ') || 'What is the capital of France?';
const repo = kind === 'base'
  ? 'onnx-community/Qwen2.5-0.5B-ONNX'
  : 'onnx-community/Qwen2.5-0.5B-Instruct';

env.cacheDir = process.env.TRANSFORMERS_CACHE || '/private/tmp/tiny-crcv-transformers-cache';

function promptInputs(tokenizer) {
  if (kind === 'base') {
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

console.log(`Loading ${repo} with Transformers.js ONNX/q4 (Node CPU provider)…`);
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const model = await AutoModelForCausalLM.from_pretrained(repo, { dtype: 'q4', device: 'cpu' });
const inputs = promptInputs(tokenizer);
const promptLength = inputs.input_ids.dims.at(-1);
const output = await model.generate({
  ...inputs,
  do_sample: false,
  repetition_penalty: 1,
  temperature: 1,
  top_k: 0,
  top_p: 1,
  max_new_tokens: 24,
});
const sequence = output.sequences ?? output;
const ids = sequence.tolist()[0].map(Number).slice(promptLength);
const answer = tokenizer.decode(ids, { skip_special_tokens: true }).trim();

console.log(`Question: ${question}`);
console.log(`Answer: ${answer}`);
console.log(`Generated token IDs: ${ids.join(', ')}`);

if (/one short factual sentence(?: and)?/i.test(answer)) {
  throw new Error('Degenerate prompt-fragment repetition detected.');
}
if (question === 'What is the capital of France?' && !/Paris/i.test(answer)) {
  throw new Error('France sanity check did not contain Paris.');
}

await model.dispose();
console.log('PASS: standard Transformers.js generation produced a non-degenerate answer.');
