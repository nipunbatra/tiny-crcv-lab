import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { AutoTokenizer, env } from '@huggingface/transformers';

env.cacheDir = './work/hf-js-cache';
const tokenizer = await AutoTokenizer.from_pretrained('onnx-community/Qwen2.5-0.5B-Instruct');
const questions = (await readFile('data/questions.jsonl', 'utf8'))
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));

const output = {};
for (const item of questions) {
  const encoded = tokenizer(item.question, { add_special_tokens: false });
  const ids = Array.from(encoded.input_ids.data, Number);
  output[item.id] = ids.map((id) => ({
    id,
    piece: tokenizer.decode([id], { skip_special_tokens: false }),
  }));
}

await mkdir('web/data', { recursive: true });
await writeFile('web/data/question-tokens.json', `${JSON.stringify(output)}\n`);
console.log(`Wrote exact Qwen tokenization for ${questions.length} questions.`);
