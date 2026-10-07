import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test, type TestContext } from 'node:test';

import { CachedEmbedder, hashEmbedding } from '../src/embedder.ts';

const tempDir = (t: TestContext) => { const dir = mkdtempSync(join(tmpdir(), 'ec-cache-')); t.after(() => rmSync(dir, { recursive: true, force: true })); return dir; };
const writer = (cachePath: string, format: 'json' | 'binary', name: string, count: number) => new Promise<number | null>((resolve) => {
  spawn(process.execPath, [...process.execArgv, join(import.meta.dirname, 'support', 'cache-writer.ts'), cachePath, format, name, String(count)], { stdio: 'inherit' }).on('exit', resolve);
});
const texts = (name: string, count: number) => Array.from({ length: count }, (_, i) => `${name} text number ${i}`);
/** Reads every text back; counts vectors that aren't the text's own and texts that had to be embedded again. */
const audit = async (cachePath: string, format: 'json' | 'binary', all: string[]) => {
  let reembedded = 0;
  const reader = new CachedEmbedder({ cachePath, format, dimensions: 8, log: () => {}, embedBatch: async (ts) => { reembedded += ts.length; return ts.map((t) => hashEmbedding(t, 8)); } });
  const got = await reader.embedMany(all);
  const wrong = got.filter((v, i) => v.some((x, j) => Math.abs(x - hashEmbedding(all[i], 8)[j]) > 1e-6)).length;
  return { wrong, reembedded };
};

for (const format of ['binary', 'json'] as const) {
  test(`cached embedder, format '${format}': two processes writing one cache never mix up or lose entries`, async (t) => {
    const cachePath = join(tempDir(t), 'shared');
    assert.deepEqual(await Promise.all([writer(cachePath, format, 'alpha', 600), writer(cachePath, format, 'beta', 600)]), [0, 0]);
    assert.deepEqual(await audit(cachePath, format, [...texts('alpha', 600), ...texts('beta', 600)]), { wrong: 0, reembedded: 0 });
  });
}

test("cached embedder, format 'binary': a run and its rerun return identical vectors", async (t) => {
  const cachePath = join(tempDir(t), 'precision');
  // Two words: values like 1/sqrt(2) aren't exact in float32, so a double-returning provider would differ.
  const make = () => new CachedEmbedder({ cachePath, format: 'binary', dimensions: 8, log: () => {}, embed: async (s) => hashEmbedding(s, 8) });
  const first = await make().embedMany(['two words']);
  const second = await make().embedMany(['two words']);
  assert.deepEqual(first, second);
});

test("cached embedder, format 'binary': a two-file cache is converted, dropping a cut-off key line", async (t) => {
  const cachePath = join(tempDir(t), 'legacy');
  const d = 8, keyOf = (s: string) => createHash('sha256').update(s, 'utf8').digest('hex');
  const words = ['one', 'two', 'three'];
  const rows = new Float32Array(words.flatMap((w) => hashEmbedding(w, d)));
  writeFileSync(`${cachePath}.f32`, new Uint8Array(rows.buffer));
  // The third key was cut off mid-line by an interrupted write.
  writeFileSync(`${cachePath}.keys`, `${keyOf('one')}\n${keyOf('two')}\n${keyOf('three').slice(0, 20)}`);
  const logs: string[] = [];
  let reembedded: string[] = [];
  const c = new CachedEmbedder({ cachePath, format: 'binary', dimensions: d, log: (l) => logs.push(l), embed: async (s) => { reembedded.push(s); return hashEmbedding(s, d); } });
  assert.match(logs.join(), /converted/);
  await c.embedMany(words);
  assert.deepEqual(reembedded, ['three'], 'the two complete entries were kept; the cut-off one is embedded again');
  reembedded = [];
  await new CachedEmbedder({ cachePath, format: 'binary', dimensions: d, log: () => {}, embed: async (s) => { reembedded.push(s); return hashEmbedding(s, d); } }).embedMany(words);
  assert.deepEqual(reembedded, []);
  assert.equal(readFileSync(`${cachePath}.keys`, 'utf8').length > 0, true, 'the old files are left in place');
});

test('cached embedder: a lock left by a process that is gone is broken; a live one times out with a clear error', async (t) => {
  const cachePath = join(tempDir(t), 'locked');
  writeFileSync(`${cachePath}.bin.lock`, '999999\n'); // no such process
  await new CachedEmbedder({ cachePath, format: 'binary', dimensions: 8, log: () => {}, embed: async (s) => hashEmbedding(s, 8) }).embedMany(['a']);
  writeFileSync(`${cachePath}.bin.lock`, `${process.ppid}\n`); // a live process holds it
  assert.throws(() => new CachedEmbedder({ cachePath, format: 'binary', dimensions: 8, lockTimeoutMs: 50, log: () => {}, embed: async (s) => hashEmbedding(s, 8) }), /is locked/);
});
