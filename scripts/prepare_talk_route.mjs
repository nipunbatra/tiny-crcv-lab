import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const talkRoot = resolve(repoRoot, 'public/talk');
const indexPath = resolve(talkRoot, 'index.html');
const routePath = resolve(talkRoot, 's/tiny-hallucination-detector');
const presenterPath = resolve(routePath, 'presenter');

const indexHtml = await readFile(indexPath, 'utf8');
const titledHtml = indexHtml.replace(
  '<title>open-slide</title>',
  '<title>Can a Tiny Model Warn Us When It Is Wrong?</title>',
);

await writeFile(indexPath, titledHtml);
await mkdir(routePath, { recursive: true });
await mkdir(presenterPath, { recursive: true });
await copyFile(indexPath, resolve(routePath, 'index.html'));
await copyFile(indexPath, resolve(presenterPath, 'index.html'));
