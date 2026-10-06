import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeReference, encodeReference, fitFeatures, foldOf, knnFeature, dot, type FeatureTraining } from '../src/heads.ts';
import { assertRoundTrip, headProbability, scoreEmbedding, trainHeads, validateArtifact, type ClassifierArtifact, type HeadPolicy, type Split, type TrainResult } from '../src/index.ts';

const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.9, designRecall: 0.95 };

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}
const unit = (v: number[]) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };

/**
 * Labelled embeddings on the unit sphere. 'clustered': positives are three tight clusters with
 * negative clusters between them (no hyperplane separates them; neighbours do). 'diffuse': the label
 * is the side of a hyperplane (a linear head's home ground).
 */
function world(shape: 'clustered' | 'diffuse', n = 900, seed = 3) {
  const rand = rng(seed), d = 16;
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const centre = () => unit(Array.from({ length: d }, gauss));
  const pos = [centre(), centre(), centre()];
  const neg = [...Array.from({ length: 12 }, centre), unit(pos[0].map((v, j) => v + pos[1][j])), unit(pos[1].map((v, j) => v + pos[2][j])), unit(pos[0].map((v, j) => v + pos[2][j]))];
  const w = centre();
  const X: number[][] = [], y: Array<0 | 1> = [], split: Split[] = [], groups: string[] = [];
  for (let i = 0; i < n; i++) {
    if (shape === 'clustered') {
      const label = rand() < 0.2 ? 1 : 0;
      const c = label ? pos[i % 3] : neg[i % neg.length];
      X.push(unit(c.map((v) => v + 0.18 * gauss())));
      y.push(label);
    } else {
      const x = unit(Array.from({ length: d }, gauss));
      X.push(x);
      y.push(dot(w, x) + 0.05 * gauss() > 0.25 ? 1 : 0);
    }
    split.push((['train', 'train', 'calibration', 'test'] as const)[i % 4]);
    groups.push(`g${i}`);
  }
  return { X, y, split, groups };
}

const artifactOf = (result: TrainResult<'h'>, dims = 16): ClassifierArtifact<'h'> => ({
  version: 'v1', created_at: '', embedding: { model_id: 'test', dimensions: dims, normalize: true },
  heads: result.heads, ...(result.reference ? { reference: result.reference } : {}),
});

test('reference encoding round-trips float32 rows exactly, whatever the padding', () => {
  for (const rows of [1, 2, 3, 5]) {
    const data = Array.from({ length: rows }, (_, i) => Array.from({ length: 3 }, (_, j) => Math.fround(Math.sin(i * 7 + j) * 0.37)));
    const ref = encodeReference(data, { h: data.map((_, i) => (i % 2 ? 1 : 0)) });
    assert.deepEqual(decodeReference(ref).map((r) => Array.from(r)), data);
  }
});

test('int8 references: about 4x smaller, rows within one quantisation step, and the artifact still round-trips exactly', () => {
  const { X, y, split, groups } = world('clustered', 600);
  const f32 = trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'knn' }] });
  const int8 = trainHeads<'h'>({ X, split, groups, referenceEncoding: 'int8', heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'knn' }] });
  assert.equal(int8.reference!.encoding, 'int8');
  assert.ok(int8.reference!.data.length < f32.reference!.data.length / 3.9);
  const a = decodeReference(f32.reference!), b = decodeReference(int8.reference!);
  a.forEach((row, i) => row.forEach((v, t) => assert.ok(Math.abs(v - b[i][t]) <= int8.reference!.scales![i] / 2 + 1e-7)));
  const artifact = artifactOf(int8);
  assertRoundTrip(artifact, X, int8.testProbabilities);
  assert.throws(() => validateArtifact({ ...artifact, reference: { ...artifact.reference!, scales: undefined } }), /one positive scale per row/);
});

test("type 'auto' chooses a neighbour head for a clustered label and stays linear for a diffuse one", () => {
  for (const [shape, expectLinear] of [['clustered', false], ['diffuse', true]] as const) {
    const { X, y, split, groups } = world(shape);
    const result: TrainResult<'h'> = trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'auto' }] });
    const choice = result.headChoice.h!;
    assert.equal(choice.chosen === 'linear', expectLinear, `${shape}: chose ${choice.chosen} (${JSON.stringify(choice.costs)})`);
    assert.equal(!!result.heads.h!.features, !expectLinear);
    assert.equal(!!result.reference, !expectLinear, 'the reference ships only when a head uses it');
    // Deterministic: the same input gives the same head.
    const again: TrainResult<'h'> = trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'auto' }] });
    assert.deepEqual(again.heads.h, result.heads.h);
  }
});

test("autoMargin: a non-linear head must beat linear's cross-validated cost by the margin", () => {
  const { X, y, split, groups } = world('clustered');
  const base = trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'auto' }] });
  const { costs } = base.headChoice.h!;
  const best = Math.min(costs.knn, costs.stack);
  assert.ok(best < costs.linear * 0.8, 'the clustered world clears the default 20% margin');
  // A margin larger than the gain keeps linear.
  const strict = trainHeads<'h'>({ X, split, groups, autoMargin: 1 - best / costs.linear + 0.01, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'auto' }] });
  assert.equal(strict.headChoice.h!.chosen, 'linear');
  assert.throws(() => trainHeads<'h'>({ X, split, groups, autoMargin: 1, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'auto' }] }), /autoMargin/);
});

test('knn and stack heads: the saved artifact scores exactly as evaluated, and needs its reference', () => {
  const { X, y, split, groups } = world('clustered');
  for (const type of ['knn', 'stack'] as const) {
    const result: TrainResult<'h'> = trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type }] });
    assert.equal(result.heads.h!.features!.kind, type);
    assert.equal(result.heads.h!.weights.length, type === 'knn' ? 1 : 3);
    const artifact = artifactOf(result);
    assertRoundTrip(artifact, X, result.testProbabilities);
    assert.ok(result.evaluation.h!.recall >= 0.85, `${type}: recall ${result.evaluation.h!.recall}`);
    // Without its reference the artifact is refused, and the linear-only helper refuses the head.
    assert.throws(() => validateArtifact({ ...artifact, reference: undefined }), /reference has no labels for it/);
    assert.throws(() => headProbability(result.heads.h!, X[0]), /needs the artifact's reference/);
    assert.ok(Number.isFinite(scoreEmbedding(validateArtifact(JSON.parse(JSON.stringify(artifact))), X[1]).h!));
  }
});

test('training rows get out-of-fold features: never scored against their own fold', () => {
  const { X, y, groups } = world('clustered', 300);
  const rows = X.map((_, i) => i);
  const reference = decodeReference(encodeReference(X, { h: y }));
  const t: FeatureTraining = { rows, y, weights: rows.map(() => 1), keys: groups, X, reference, referenceKeys: groups, referenceLabels: y, C: 1, seed: 0 };
  const fitted = fitFeatures('knn', t);
  rows.forEach((i, n) => {
    const f = foldOf(groups[i]);
    const keep = (j: number) => foldOf(groups[j]) !== f;
    const sims = reference.map((r) => dot(X[i], r));
    const expected = knnFeature(sims, rows.filter((j) => y[j] === 1 && keep(j)), rows.filter((j) => y[j] === 0 && keep(j)), 10);
    assert.equal(fitted.train[n][0], expected);
  });
  // A row scored against the full reference would find itself (cosine 1): the out-of-fold feature doesn't.
  const selfMatch = rows.filter((i, n) => y[i] === 1 && fitted.train[n][0] === fitted.apply(X[i])[0]).length;
  assert.ok(selfMatch < rows.filter((i) => y[i] === 1).length / 10, `${selfMatch} positives scored as if in-sample`);
});

test('validateArtifact checks feature widths and the reference', () => {
  const { X, y, split, groups } = world('clustered', 400);
  const artifact = artifactOf(trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'knn' }] }));
  validateArtifact(artifact);
  assert.throws(() => validateArtifact({ ...artifact, heads: { h: { ...artifact.heads.h!, weights: [1, 2] } } }), /2 weights but its knn features have 1/);
  assert.throws(() => validateArtifact({ ...artifact, reference: { ...artifact.reference!, data: artifact.reference!.data.slice(4) } }), /wrong length/);
  assert.throws(() => validateArtifact({ ...artifact, reference: { ...artifact.reference!, labels: { h: artifact.reference!.labels.h.map(() => 0 as const) } } }), /both positive and negative/);
  assert.throws(() => validateArtifact({ ...artifact, reference: { ...artifact.reference!, dims: 8 } }), /8 dimensions/);
});

test('linear heads are untouched: no reference, no features; weak positives stay linear-only', () => {
  const { X, y, split, groups } = world('diffuse', 400);
  const result = trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy }] });
  assert.equal(result.reference, undefined);
  assert.equal(result.heads.h!.features, undefined);
  assert.deepEqual(result.headChoice, {});
  assert.throws(() => trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'knn', weak: { X: [X[0]], weights: [1] } }] }), /linear heads only/);
  assert.throws(() => trainHeads<'h'>({ X, split, groups, heads: [{ name: 'h', y, prevalence: 0.2, policy, type: 'tree' as never }] }), /unknown head type/);
});
