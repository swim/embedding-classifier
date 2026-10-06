import assert from 'node:assert/strict';
import { test } from 'node:test';

import { assertRoundTrip, decide, trainHeads, validateArtifact, type ClassifierArtifact, type HeadPolicy, type Split, type TrainResult } from '../src/index.ts';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}

/** A linear world; `cleared` marks rows a (simulated) dismissal rule clears: mostly negatives, plus a share of positives. */
function world(positiveShareCleared: number, n = 2400, seed = 5) {
  const rand = rng(seed);
  const X: number[][] = [], y: Array<0 | 1> = [], split: Split[] = [], rows: boolean[] = [];
  for (let i = 0; i < n; i++) {
    const label = rand() < 0.3 ? 1 : 0;
    X.push(Array.from({ length: 8 }, (_, j) => (j < 2 ? (label ? 1.2 : -1.2) : 0) + rand() * 2 - 1));
    y.push(label);
    split.push((['train', 'train', 'calibration', 'test'] as const)[i % 4]);
    rows.push(label ? rand() < positiveShareCleared : rand() < 0.6);
  }
  return { X, y, split, rows };
}
const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 };
const train = (w: ReturnType<typeof world>, dismissal = true): TrainResult<'h'> => trainHeads<'h'>({
  X: w.X, split: w.split,
  heads: [{ name: 'h', y: w.y, prevalence: 0.3, policy, ...(dismissal ? { dismissal: { rows: w.rows, ruleSet: 'rules-v1', maxRate: 0.05, certified: 3 } } : {}) }],
});

test('a positive the rules dismissed counts as a miss: test recall includes the rules', () => {
  const w = world(0.04);
  const withRules = train(w), without = train(w, false);
  const pTest = withRules.testProbabilities.h!;
  // Cleared test rows are left out of the round-trip probabilities (runtime never scores them)...
  assert.ok(pTest.idx.every((i) => !w.rows[i]));
  // ...but the evaluation counts them: recall can't exceed the share of test positives not cleared.
  const testPos = w.X.map((_, i) => i).filter((i) => w.split[i] === 'test' && w.y[i] === 1);
  const notCleared = testPos.filter((i) => !w.rows[i]).length / testPos.length;
  assert.ok(withRules.evaluation.h!.recall <= notCleared + 1e-12, `recall ${withRules.evaluation.h!.recall} vs ${notCleared}`);
  // The threshold compensates for the cleared positives (here it sits lower than without rules).
  assert.ok(withRules.heads.h!.threshold <= without.heads.h!.threshold + 1e-12);
  assert.deepEqual(withRules.heads.h!.dismissal, { rule_set: 'rules-v1', max_rate: 0.05, certified: 3 });
  const artifact: ClassifierArtifact<'h'> = { version: 'v1', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: withRules.heads };
  assertRoundTrip(artifact, w.X, withRules.testProbabilities);
  assert.throws(() => validateArtifact({ ...artifact, heads: { h: { ...artifact.heads.h!, dismissal: { rule_set: 'x', max_rate: 2, certified: 1 } } } }), /invalid dismissal record/);
});

test('rules that clear too many positives fail the head instead of firing on cleared messages', () => {
  const result = train(world(0.5));
  assert.ok(result.failures.some((f) => f.includes('dismissal rules cleared')), result.failures.join('; '));
  assert.ok(result.heads.h!.threshold > 0);
});

test('decide: a dismissed head needs no score, never fires and never suppresses', () => {
  const heads = { urgent: { weights: [1], bias: 0, calibration: { method: 'platt' as const, a: 1, c: 0 }, threshold: 0.5, review_floor: 0.25 }, faq: { weights: [1], bias: 0, calibration: { method: 'platt' as const, a: 1, c: 0 }, threshold: 0.5, review_floor: 0.25 } };
  const artifact: ClassifierArtifact<'urgent' | 'faq'> = { version: 'v1', created_at: '', embedding: { model_id: 't', dimensions: 1, normalize: false }, heads };
  const policy = { priority: ['urgent', 'faq'] as const, suppress: [{ when: ['urgent' as const], heads: ['faq' as const] }] };
  assert.throws(() => decide(artifact, { faq: 0.9 }, policy), /no finite score/);
  assert.deepEqual(decide(artifact, { faq: 0.9 }, policy, ['urgent']), { head: 'faq', reason: 'above_threshold' });
  assert.deepEqual(decide(artifact, {}, policy, ['urgent', 'faq']), { head: null, reason: 'none' });
});
