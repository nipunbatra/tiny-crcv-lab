#!/usr/bin/env node

import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const destination = resolve(root, 'public/data');
mkdirSync(destination, { recursive: true });

for (const model of ['instruct', 'base']) {
  copyFileSync(
    resolve(root, `outputs/fresh_qa_qwen05b_${model}/predictions.jsonl`),
    resolve(destination, `fresh_qa_qwen05b_${model}.jsonl`),
  );
}

console.log('Prepared lazy-loaded fresh QA traces in public/data.');
