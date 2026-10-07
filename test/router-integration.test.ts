import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  decide,
  missingPolicyHeads,
  prepareScoring,
  routerRuleSetHash,
  scoreEmbedding,
  validateArtifact,
  type ClassifierArtifact,
  type DecisionHeads,
} from '../src/index.ts';
import { encodeReference } from '../src/heads.ts';

const HASH = 'a'.repeat(64);

const knnArtifact = (rows: number[][]): ClassifierArtifact<'h'> => ({
  version: 't', created_at: '', embedding: { model_id: 'm', dimensions: 2, normalize: false },
  reference: encodeReference(rows, { h: rows.map((_, i) => (i % 2 ? 1 : 0)) }),
  heads: { h: { weights: [1], bias: 0, features: { kind: 'knn', k: 1 }, calibration: { method: 'platt', a: 1, c: 0 }, threshold: 0.5, review_floor: 0.2 } },
});

test('a float32 reference holding NaN or Infinity is rejected at load, not scored as NaN', () => {
  assert.doesNotThrow(() => validateArtifact(knnArtifact([[1, 0], [0, 1]])));
  assert.throws(() => validateArtifact(knnArtifact([[1, 0], [Number.NaN, 1]])), /reference row 1 has a non-finite value at index 0/);
  assert.throws(() => validateArtifact(knnArtifact([[Infinity, 0], [0, 1]])), /reference row 0 has a non-finite value/);
});

test('training.router: read strictly, never guessed', () => {
  const base = knnArtifact([[1, 0], [0, 1]]);
  assert.equal(routerRuleSetHash(base), undefined);
  assert.equal(routerRuleSetHash({ training: { router: { ruleSetHash: HASH } } }), HASH);
  for (const router of [null, 'x', { ruleSetHash: 'ABC' }, { ruleSetHash: HASH.toUpperCase() }, { rule_set: HASH }]) {
    assert.throws(() => validateArtifact({ ...base, training: { router } }), /training\.router/);
  }
});

test('prepareScoring builds the reference state once and scoring is unchanged', () => {
  const a = knnArtifact([[1, 0], [0, 1], [0.8, 0.6], [0.6, 0.8]]);
  const before = scoreEmbedding(a, [0.9, 0.1]);
  prepareScoring(Object.freeze(a));
  assert.deepEqual(scoreEmbedding(a, [0.9, 0.1]), before);
});

test('decide and missingPolicyHeads accept a threshold/review-floor projection', () => {
  const heads: DecisionHeads<'a' | 'b'> = { heads: { a: { threshold: 0.6, review_floor: 0.3 }, b: { threshold: 1.0000000000000002, review_floor: 0.5 } } };
  assert.deepEqual(decide(heads, { a: 0.7, b: 0.9 }, { priority: ['b', 'a'] }), { head: 'a', reason: 'above_threshold' });
  assert.deepEqual(decide(heads, { a: 0.1 }, { priority: ['b', 'a'] }, [], 'b'), { head: 'b', reason: 'rule' }, 'a firing rule acts on a disabled head');
  assert.deepEqual(missingPolicyHeads(heads, { priority: ['a', 'c' as 'a'] }), ['c']);
});
