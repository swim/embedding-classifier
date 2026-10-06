import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { builtinModules } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';

import { truncateText as fromEmbedder } from '../src/embedder.ts';
import { truncateText } from '../src/index.ts';

/** Every module reachable from `entry` through relative imports, with their import specifiers. */
function importGraph(entry: string): Map<string, string[]> {
  const seen = new Map<string, string[]>();
  const visit = (file: string) => {
    if (seen.has(file)) return;
    const src = readFileSync(file, 'utf8');
    const specs = [...src.matchAll(/(?:^|\n)\s*(?:import|export)\s[^;]*?from\s*'([^']+)'|import\(\s*'([^']+)'\s*\)/g)].map((m) => m[1] ?? m[2]);
    seen.set(file, specs);
    for (const s of specs) if (s.startsWith('.')) visit(resolve(dirname(file), s));
  };
  visit(entry);
  return seen;
}
const builtins = new Set(builtinModules);
const nodeImports = (entry: string) => [...importGraph(entry)].flatMap(([file, specs]) => specs.filter((s) => s.startsWith('node:') || builtins.has(s.split('/')[0])).map((s) => `${file}: ${s}`));
const SRC = join(import.meta.dirname, '..', 'src');

test('the package root imports no Node built-ins, so serving code runs on edge and browser runtimes', () => {
  assert.ok(importGraph(join(SRC, 'index.ts')).size > 10, 'the walker found the source modules');
  assert.deepEqual(nodeImports(join(SRC, 'index.ts')), []);
});

test('the check catches Node built-ins: /embedder (Node-only by design) is flagged', () => {
  assert.ok(nodeImports(join(SRC, 'embedder.ts')).some((s) => s.endsWith('node:fs')));
});

test('truncateText from the root is the one /embedder uses (training and serving truncate alike)', () => {
  assert.equal(truncateText, fromEmbedder);
  assert.equal(truncateText('ab😀c', 3), 'ab');
});
