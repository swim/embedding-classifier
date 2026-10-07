import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import { CachedEmbedder, hashEmbedding, truncateText } from '../src/embedder.ts';
import {
  assertRoundTrip,
  buildArtifact,
  evaluateHead,
  gateHead,
  pickThreshold,
  publishPlan,
  reportMarkdown,
  trainHeads,
  validateArtifact,
  type ClassifierArtifact,
  type HeadPolicy,
  type Split,
  type WeakInput,
} from '../src/index.ts';

/** A temporary directory removed when the test finishes. */
function tempDir(t: { after: (fn: () => void) => void }): string {
  const dir = mkdtempSync(join(tmpdir(), 'embedder-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}

const recall: HeadPolicy = { kind: 'recall', mode: 'heuristic', targetRecall: 0.95, designRecall: 0.98, maxFalseAlarm: 0.05, minPositives: 150, minRecallLower: 0.9, minPositiveGroups: 30 };
const precision: HeadPolicy = { kind: 'precision', mode: 'heuristic', targetPrecision: 0.8 };

test('recall threshold never exceeds the false-alarm budget, even with an outlier positive', () => {
  const neg = Array.from({ length: 20 }, (_, i) => 0.01 * (i + 1));
  const pos = [0.9, 0.85, 0.8, 0.95, 0.001];
  const t = pickThreshold(recall, [...pos, ...neg], [...pos.map(() => 1), ...neg.map(() => 0)]);
  const falseAlarms = neg.filter((p) => p >= t).length;
  assert.ok(falseAlarms <= Math.floor(0.05 * neg.length), `threshold ${t} lets ${falseAlarms} negatives fire`);
  assert.ok(t > 0.19, 'the outlier positive must not drag the threshold down');
});

test('thresholds keep a margin above the gate', () => {
  const p = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
  const y = p.map(() => 1);
  assert.equal(pickThreshold(recall, p, y), 0.1); // 98% of 11 positives -> all 11 -> lowest
  assert.equal(pickThreshold(precision, p, y), 0.8, 'precision heads use the target precision');
  assert.throws(() => pickThreshold({ ...recall, designRecall: 0.9 } as HeadPolicy, p, y), /designRecall/);
});

test('evaluation compares against a baseline and gates on what the classifier adds', () => {
  const ev = evaluateHead({ p: [0.1, 0.95, 0.01], y: [1, 1, 0], w: [1, 1, 1], threshold: 0.9, baseline: [true, false, false], slices: { source: ['a', 'b', 'a'] } });
  assert.deepEqual(ev.vs_baseline, { caught_by_both: 0, caught_by_baseline_only: 1, caught_by_classifier_only: 1, missed_by_both: 0 });
  assert.equal(ev.recall, 0.5);
  assert.equal(ev.combined_recall, 1);
  assert.deepEqual(Object.keys(ev.slices), ['source=a', 'source=b']);
  const { failures, warnings } = gateHead(recall, ev);
  assert.ok(failures.some((g) => g.startsWith('only 2 test positives')));
  assert.ok(failures.some((g) => g.startsWith('test recall 0.500')));
  assert.ok(!failures.some((g) => g.includes('adds no value')));
  assert.ok(warnings.some((w) => w.includes('only 2 distinct group(s)')));
  const useless = evaluateHead({ p: [0.1, 0.1], y: [1, 1], w: [1, 1], threshold: 0.9, baseline: [true, true] });
  assert.ok(gateHead(recall, useless).failures.some((g) => g.includes('adds no value')));
  assert.ok(!gateHead({ ...recall, mustBeatBaseline: false } as HeadPolicy, useless).failures.some((g) => g.includes('adds no value')));
});

test('a precision head that never fires gets an explicit gate message citing the evaluated threshold', () => {
  const ev = evaluateHead({ p: [0.1, 0.2], y: [1, 0], w: [1, 1], threshold: 0.93 });
  assert.match(gateHead(precision, ev).failures[0], /never fires on the test set at threshold 0.93/);
});

test('thin support warns by default and fails with an explicit minimum', () => {
  const ev = evaluateHead({ p: [0.9, 0.1, 0.1, 0.1], y: [1, 0, 0, 0], w: [1, 1, 1, 1], threshold: 0.8 });
  assert.equal(ev.fired, 1);
  assert.ok(gateHead(precision, ev).warnings.some((w) => w.includes('only 1 fired test examples')));
  assert.ok(gateHead({ ...precision, minFired: 5 } as HeadPolicy, ev).failures.some((f) => f.includes('fires on only 1')));
  const recallNoMin = { kind: 'recall', targetRecall: 0.5, designRecall: 0.5 } as HeadPolicy;
  assert.ok(gateHead(recallNoMin, ev).warnings.some((w) => w.includes('only 1 test positives and no minPositives')));
  assert.ok(!gateHead({ ...recallNoMin, minPositives: 1 } as HeadPolicy, ev).warnings.some((w) => w.includes('minPositives')));
});

test('evaluateHead rejects misaligned inputs', () => {
  assert.throws(() => evaluateHead({ p: [0.9, 0.1], y: [1, 0], w: [1, 1], threshold: 0.5, baseline: [true] }), /baseline has 1 entries but p has 2/);
  assert.throws(() => evaluateHead({ p: [0.9, 0.1], y: [1, 0], w: [1, 1], threshold: 0.5, slices: { s: ['a'] } }), /slices.s has 1/);
});

test('publishing: failing models can be shadow candidates but never promoted', () => {
  assert.deepEqual(publishPlan({ gatesPassed: true, promote: true }), { role: 'promoted' });
  assert.match((publishPlan({ gatesPassed: false, promote: true }) as { error: string }).error, /refusing to promote/);
  assert.deepEqual(publishPlan({ gatesPassed: false, shadowCandidate: true }), { role: 'shadow-candidate' });
  assert.match((publishPlan({ gatesPassed: true, shadowCandidate: true, blockedReason: 'fake' }) as { error: string }).error, /refusing to publish: fake/);
  assert.match((publishPlan({ gatesPassed: true, promote: true, shadowCandidate: true }) as { error: string }).error, /choose one/);
  assert.match((publishPlan({ gatesPassed: false }) as { error: string }).error, /gates failed/);
  assert.deepEqual(publishPlan({ gatesPassed: false, allowFailingGates: true }), { role: null });
  assert.deepEqual(publishPlan({ gatesPassed: true }), { role: null });
});

/** Two separable clusters in 8 dims with a little label noise - enough to exercise the whole pipeline. */
function dataset(n: number, seed = 1) {
  let s = seed;
  const rand = () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
  const X: number[][] = [];
  const y: (0 | 1)[] = [];
  const split: Split[] = [];
  for (let i = 0; i < n; i++) {
    const label = rand() < 0.3 ? 1 : 0;
    X.push(Array.from({ length: 8 }, (_, j) => (j < 2 ? (label ? 1.5 : -1.5) : 0) + rand() * 2 - 1));
    y.push(rand() < 0.02 ? ((1 - label) as 0 | 1) : label);
    split.push((['train', 'train', 'calibration', 'test'] as const)[i % 4]);
  }
  return { X, y, split };
}

test('trainHeads end to end: fits, gates, and the serialised artifact scores identically', () => {
  const { X, y, split } = dataset(2400);
  const lines: string[] = [];
  const result = trainHeads({
    X, split, C: 1, log: (l) => lines.push(l),
    heads: [
      { name: 'urgent', y, prevalence: 0.05, policy: { ...recall, targetRecall: 0.9, designRecall: 0.95, minPositiveGroups: undefined } as HeadPolicy },
      { name: 'spam', y, prevalence: 0.05, policy: { kind: 'precision', mode: 'heuristic', targetPrecision: 0.5 } },
    ],
    slices: { third: X.map((_, i) => String(i % 3)) },
  });
  assert.equal(lines.filter((l) => /positive/.test(l)).length, 2, 'one split summary per head');
  assert.equal(lines.filter((l) => /auto chose/.test(l)).length, 2, "type 'auto' (the default) reports its choice per head");
  const ev = result.evaluation.urgent!;
  assert.ok(ev.recall >= 0.9, `recall ${ev.recall}`);
  assert.ok(ev.false_alarm_rate < 0.1, `false alarms ${ev.false_alarm_rate}`);
  assert.deepEqual(Object.keys(ev.slices), ['third=0', 'third=1', 'third=2']);
  assert.deepEqual(result.failures, [], result.failures.join('; '));
  assert.equal(result.heads.urgent!.review_floor, result.heads.urgent!.threshold * 0.5);

  // buildArtifact carries whatever the heads need (a knn or stack head's reference) and the gates.
  const artifact: ClassifierArtifact<'urgent' | 'spam'> = buildArtifact(result, { version: 'v1', createdAt: '', embedding: { model_id: 'test', dimensions: 8, normalize: false } });
  assertRoundTrip(artifact, X, result.testProbabilities);
  assert.throws(() => assertRoundTrip({ ...artifact, heads: { ...artifact.heads, urgent: { ...artifact.heads.urgent!, bias: 1 } } }, X, result.testProbabilities), /differs/);
  // The router pairing record: stored under training.router, validated, malformed hashes refused.
  const paired = buildArtifact(result, { version: 'v1', createdAt: '', embedding: artifact.embedding, router: { ruleSetHash: 'f'.repeat(64) } });
  assert.deepEqual(validateArtifact(JSON.parse(JSON.stringify(paired))).training?.router, { ruleSetHash: 'f'.repeat(64) });
  assert.throws(() => buildArtifact(result, { version: 'v1', embedding: artifact.embedding, router: { ruleSetHash: 'nope' } }), /ruleSetHash/);
  const report = reportMarkdown(artifact, { title: 'Test model', baselineName: 'rules' });
  assert.match(report, /^# Test model/);
  assert.match(report, /## urgent/);
});

test('trainHeads refuses a split without both classes', () => {
  const { X, split } = dataset(40);
  assert.throws(() => trainHeads({ X, split, heads: [{ name: 'h', y: X.map(() => 0), prevalence: 0.1, policy: precision }] }), /both positive and negative/);
});

test('trainHeads rejects misaligned or invalid inputs before fitting', () => {
  const { X, y, split } = dataset(400);
  const head = { name: 'h', y, prevalence: 0.05, policy: precision };
  assert.throws(() => trainHeads({ X, split, heads: [head], groups: ['g'] }), /groups has 1 entries but X has 400/);
  assert.throws(() => trainHeads({ X, split, heads: [head], slices: { src: ['a', 'b'] } }), /slices.src has 2 entries/);
  assert.throws(() => trainHeads({ X, split, heads: [{ ...head, baseline: [true] }] }), /h: baseline has 1 entries/);
  assert.throws(() => trainHeads({ X, split: split.map((s, i) => (i === 7 ? ('validation' as Split) : s)), heads: [head] }), /split\[7\] is "validation"/);
  assert.throws(() => trainHeads({ X, split, heads: [{ ...head, prevalence: 1.5 }] }), /h: prevalence must be strictly between 0 and 1/);
  assert.throws(() => trainHeads({ X, split, heads: [head], background: { X: [], maxRate: { h: 0.01 } } }), /background.X must be non-empty/);
});

test('trainHeads with isotonic calibration round-trips and records the shipped threshold', () => {
  const { X, y, split } = dataset(1200);
  const result = trainHeads({ X, split, calibration: 'isotonic', heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.8, designRecall: 0.9, minPositives: 30 } }] });
  assert.equal(result.heads.h!.calibration.method, 'isotonic');
  assert.equal(result.evaluation.h!.threshold, result.heads.h!.threshold);
  const artifact: ClassifierArtifact<'h'> = { version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: result.heads };
  assertRoundTrip(artifact, X, result.testProbabilities);
});

test('a budget-raised precision threshold is what the gate reports', () => {
  const { X, y, split } = dataset(800);
  const r = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'precision', targetPrecision: 0.3 } }], background: { X: X.filter((_, i) => y[i] === 1).slice(0, 100), maxRate: { h: 0 } } });
  const t = r.heads.h!.threshold;
  assert.ok(t > 0.3);
  assert.ok(r.failures.some((f) => f.includes(`threshold ${t}`)), r.failures.join('; '));
});

test('the report tolerates a partial evaluation from storage', () => {
  const artifact = { version: 'v', created_at: '', embedding: { model_id: 'm', dimensions: 1, normalize: true },
    heads: { h: { weights: [1], bias: 0, calibration: { method: 'platt' as const, a: 1, c: 0 }, threshold: 0.5, review_floor: 0.25 } },
    evaluation: { h: { n: 1 }, gone: { n: 2 } } };
  const report = reportMarkdown(artifact);
  assert.match(report, /## h/);
  assert.match(report, /recall \(95% CI\) \| n\/a/);
  assert.doesNotMatch(report, /## gone/);
});

test('hash embeddings are deterministic and normalised', () => {
  const a = hashEmbedding('my kid will not sleep', 64);
  assert.deepEqual(a, hashEmbedding('my kid will not sleep', 64));
  assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-12);
});

test('cached embedder: embeds each distinct (truncated) text once and persists the cache', async (t) => {
  const cachePath = join(tempDir(t), 'nested', 'cache.json');
  const calls: string[] = [];
  const embed = async (text: string) => {
    calls.push(text);
    return hashEmbedding(text, 4);
  };
  const first = new CachedEmbedder({ cachePath, embed, maxChars: 5, concurrency: 3, checkpointEvery: 1, log: () => {} });
  const out = await first.embedMany(['hello world', 'hello there', 'other']);
  assert.deepEqual(calls.sort(), ['hello', 'other'], 'truncated texts are the cache key and what is embedded');
  assert.deepEqual(out[0], out[1]);
  const second = new CachedEmbedder({ cachePath, embed, maxChars: 5, log: () => {} });
  await second.embedMany(['hello again']);
  assert.equal(calls.length, 2, 'served from the persisted cache');
  const { readdirSync } = await import('node:fs');
  const { dirname } = await import('node:path');
  assert.deepEqual(readdirSync(dirname(cachePath)).filter((f) => f.endsWith('.tmp')), [], 'no temporary files left behind');
});

test('background budget: thresholds rise until the head fires on at most maxRate of ordinary traffic, before gating', async () => {
  const { budgetThreshold } = await import('../src/index.ts');
  const bg = Array.from({ length: 100 }, (_, i) => i / 100); // 0.00 .. 0.99
  assert.equal(budgetThreshold(0.5, bg, 0.05) > 0.94, true, 'at most 5 of 100 may reach it');
  assert.equal(budgetThreshold(0.5, bg, 0.05) <= 0.95, true);
  assert.equal(budgetThreshold(0.99, bg, 0.05), 0.99, 'never lowered');

  const { X, y, split } = dataset(2400);
  const heads = (maxRate?: number) => trainHeads({
    X, split, heads: [{ name: 'urgent', y, prevalence: 0.05, policy: { kind: 'recall', mode: 'heuristic', targetRecall: 0.5, designRecall: 0.9, maxFalseAlarm: 0.2 } as HeadPolicy }],
    background: maxRate === undefined ? undefined : { X: X.filter((_, i) => y[i] === 0).slice(0, 400), maxRate: { urgent: maxRate } },
  });
  const loose = heads(), tight = heads(0.001);
  assert.ok(tight.heads.urgent!.threshold >= loose.heads.urgent!.threshold);
  assert.ok(tight.evaluation.urgent!.background_rate! <= 0.001);
  assert.ok(tight.evaluation.urgent!.recall <= loose.evaluation.urgent!.recall, 'test metrics reflect the shipped threshold');
});

test('cached embedder: batches for providers that embed many texts per call', async (t) => {
  const cachePath = join(tempDir(t), 'cache.json');
  const calls: number[] = [];
  const embedder = new CachedEmbedder({ cachePath, batchSize: 3, log: () => {}, embedBatch: async (texts) => { calls.push(texts.length); return texts.map((t) => hashEmbedding(t, 4)); } });
  const out = await embedder.embedMany(['a', 'b', 'c', 'd', 'e', 'a']);
  assert.deepEqual(calls.sort(), [2, 3], 'five distinct texts in batches of at most 3');
  assert.deepEqual(out[0], out[5]);
  assert.notEqual(out[0], out[5], 'duplicates are separate copies');
  out[0][0] = 99;
  assert.notEqual((await embedder.embedMany(['a']))[0][0], 99, 'mutating a result does not touch the cache');
  assert.throws(() => new CachedEmbedder({ cachePath }), /embed or embedBatch/);
});

test('cached embedder: dimensions are checked on load and on new embeddings', async (t) => {
  const cachePath = join(tempDir(t), 'cache.json');
  await new CachedEmbedder({ cachePath, log: () => {}, embed: async (s) => hashEmbedding(s, 4) }).embedMany(['a']);
  assert.throws(() => new CachedEmbedder({ cachePath, dimensions: 8, embed: async (s) => hashEmbedding(s, 8) }), /4-dimensional vector, expected 8/);
  const fresh = new CachedEmbedder({ cachePath: join(tempDir(t), 'c.json'), dimensions: 8, log: () => {}, embed: async (s) => hashEmbedding(s, 4) });
  await assert.rejects(fresh.embedMany(['b']), /embedding provider has a 4-dimensional vector/);
});

test('truncation never splits a surrogate pair', () => {
  assert.equal(truncateText('ab😀c', 3), 'ab');
  assert.equal(truncateText('ab😀c', 4), 'ab😀');
  assert.equal(truncateText('abc', 5), 'abc');
  assert.equal(truncateText('abc'), 'abc');
});

test('trainHeads: weak positives are capped, train-only, and reported', () => {
  const { X, y, split } = dataset(1200);
  const base = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: precision }] });
  assert.deepEqual(base.weakLabels, {});
  const fresh = dataset(400, 7);
  const weakX = fresh.X.filter((_, i) => fresh.y[i] === 1);
  const weak = { X: weakX, weights: weakX.map(() => 0.9), rules: weakX.map((_, i) => (i % 2 ? 'risk.r1' : 'risk.r2')), source: { rule_set_version: 'v1', rule_set_hash: 'abc123' } };
  const lines: string[] = [];
  const result = trainHeads({ X, split, log: (l) => lines.push(l), heads: [{ name: 'h', y, prevalence: 0.05, policy: precision, weak, maxWeakShare: 0.25 }] });
  const s = result.weakLabels.h!;
  const gold = y.filter((v, i) => v === 1 && split[i] === 'train').length;
  assert.equal(s.count, weakX.length);
  assert.equal(s.gold_train_positives, gold);
  assert.ok(Math.abs(s.weight - 0.25 * gold) < 1e-9 && s.scale < 1, `weight ${s.weight} capped at 0.25 × ${gold}`);
  assert.equal(s.by_rule!['risk.r1'].count + s.by_rule!['risk.r2'].count, weakX.length);
  assert.equal(s.rule_set_hash, 'abc123');
  assert.ok(lines.some((l) => l.includes('weak positives')));
  assert.notDeepEqual(result.heads.h!.weights, base.heads.h!.weights, 'weak positives change the fit');
  const uncapped = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: precision, weak: { X: weakX.slice(0, 3), weights: [0.5, 0.5, 0.5] } }] });
  assert.equal(uncapped.weakLabels.h!.scale, 1);

  const artifact: ClassifierArtifact<'h'> = { version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: result.heads, evaluation: result.evaluation, training: { weak_labels: result.weakLabels } };
  assertRoundTrip(artifact, X, result.testProbabilities);
  const report = reportMarkdown(artifact);
  assert.match(report, /Weak positives \(train only\): \d+, weight .* from rule set v1 \(abc123\)/);
  assert.match(report, /\| risk\.r1 \| \d+ \|/);

  const head = (w: WeakInput) => () => trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: precision, weak: w }] });
  const testRow = X[split.indexOf('test')];
  assert.throws(head({ ...weak, X: [testRow], weights: [1], rules: undefined }), /weak.X\[0\] is a calibration, test or background example/);
  assert.throws(head({ ...weak, weights: weak.weights.map(() => 1.5) }), /weak.weights\[0\] must be in \[0, 1\]/);
  assert.throws(head({ ...weak, X: [[1, 2]], weights: [1], rules: undefined }), /weak.X\[0\] has 2 dimensions but X has 8/);
  assert.throws(head({ ...weak, rules: ['x'] }), /weak.rules has 1 entries/);
  const trainRow = X[split.indexOf('train')];
  assert.doesNotThrow(head({ X: [trainRow], weights: [1] }), 'a train row may also be a weak example');
  const withBackground = () => trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: precision, weak: { X: [weakX[0]], weights: [1] } }], background: { X: [weakX[0]], maxRate: {} } });
  assert.throws(withBackground, /background example/);
});

test("cached embedder, format 'binary': append-only checkpoints, resume, repair after an interrupted append", async (t) => {
  const { appendFileSync, readFileSync, statSync } = await import('node:fs');
  const cachePath = join(tempDir(t), 'nested', 'cache');
  const calls: string[] = [];
  const embed = async (text: string) => { calls.push(text); return hashEmbedding(text, 4); };
  assert.throws(() => new CachedEmbedder({ cachePath, embed, format: 'binary' }), /needs dimensions/);
  const first = new CachedEmbedder({ cachePath, embed, dimensions: 4, format: 'binary', checkpointEvery: 1, concurrency: 1, log: () => {} });
  const out = await first.embedMany(['alpha', 'beta', 'gamma']);
  assert.deepEqual(out[1], hashEmbedding('beta', 4).map(Math.fround), 'stored as float32');
  assert.equal(statSync(`${cachePath}.f32`).size, 3 * 4 * 4, 'three float32 rows');
  assert.equal(readFileSync(`${cachePath}.keys`, 'utf8').split('\n').filter(Boolean).length, 3);
  // A second instance serves the stored vectors and appends only new ones.
  const second = new CachedEmbedder({ cachePath, embed, dimensions: 4, format: 'binary', log: () => {} });
  await second.embedMany(['alpha', 'delta']);
  assert.deepEqual(calls, ['alpha', 'beta', 'gamma', 'delta'], 'alpha came from the cache');
  assert.equal(statSync(`${cachePath}.f32`).size, 4 * 4 * 4);
  // An interrupted append: a partial vector with no key. The next load trims it and keeps going.
  appendFileSync(`${cachePath}.f32`, new Uint8Array(6));
  const third = new CachedEmbedder({ cachePath, embed, dimensions: 4, format: 'binary', log: () => {} });
  assert.equal(statSync(`${cachePath}.f32`).size, 4 * 4 * 4, 'trimmed back to the last matched entry');
  const [eps] = await third.embedMany(['epsilon']);
  const fourth = new CachedEmbedder({ cachePath, embed, dimensions: 4, format: 'binary', log: () => {} });
  assert.deepEqual((await fourth.embedMany(['epsilon']))[0], eps, 'the appended vector lines up with its key');
  assert.equal(calls.filter((c) => c === 'epsilon').length, 1);
  // A different-sized model can't read it: its rows wouldn't line up.
  assert.throws(() => new CachedEmbedder({ cachePath, embed, format: 'binary', dimensions: 0 }), /needs dimensions/);
});

test('trainHeads records whether each head\'s fits converged (model and Platt calibrator)', () => {
  const { X, y, split } = dataset(1200);
  const result = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 } }] });
  assert.deepEqual(result.convergence.h, { model: true, calibration: true });
  assert.ok(!result.warnings.some((w) => w.includes('did not converge')));
});

test('buildArtifact records the gates from the result; validateArtifact enforces them and refuses inconsistent ones', async () => {
  const { buildArtifact, validateArtifact } = await import('../src/index.ts');
  const { X, y, split } = dataset(1200);
  const embedding = { model_id: 't', dimensions: 8, normalize: false };
  const good = buildArtifact(trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.8 } }] }), { version: 'v1', embedding });
  assert.equal(good.gates!.passed, good.gates!.failures.length === 0);
  // Without sampled records a precision head can have no guarantee: it fails unless 'heuristic' is chosen.
  const unguaranteed = trainHeads({ X, split, heads: [{ name: 'p', y, prevalence: 0.05, policy: { kind: 'precision', targetPrecision: 0.5 } }] });
  assert.ok(unguaranteed.failures.some((f) => /no precision guarantee without sampled calibration records/.test(f)));
  const chosen = trainHeads({ X, split, heads: [{ name: 'p', y, prevalence: 0.05, policy: { kind: 'precision', targetPrecision: 0.5, mode: 'heuristic' } }] });
  assert.ok(!chosen.failures.some((f) => /no precision guarantee/.test(f)) && chosen.warnings.some((w) => /carries no guarantee/.test(w)));
  const failing = buildArtifact(unguaranteed, { version: 'v2', embedding });
  assert.equal(failing.gates!.passed, false);
  validateArtifact(failing, { mode: 'shadow' });
  assert.throws(() => validateArtifact(failing, { mode: 'enforce' }), /gates did not pass/);
  assert.throws(() => validateArtifact({ ...failing, gates: undefined }, { mode: 'enforce' }), /gates did not pass \(or were not recorded\)/);
  assert.throws(() => validateArtifact({ ...failing, gates: { ...failing.gates!, passed: true } }), /passed but record/);
});
