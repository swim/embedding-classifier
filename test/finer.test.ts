import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decisionFunction } from '@liquidau/solvers';

import { assertRoundTrip, buildArtifact, finerProbabilities, fitFinerModel, scoreEmbedding, trainHeads, validateArtifact, type HeadPolicy, type Split } from '../src/index.ts';

const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.9, designRecall: 0.95 };

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}
const unit = (v: number[]) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };

/**
 * Twelve finer labels (intents), each a cluster on the unit sphere; the route is the union of four of
 * them, spread around the sphere so no hyperplane separates the route from the rest.
 */
function world(n: number, seed: number, centresSeed = 11) {
  const crand = rng(centresSeed), rand = rng(seed), d = 16;
  const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const centres = Array.from({ length: 12 }, () => unit(Array.from({ length: d }, () => gauss(crand))));
  const route = new Set([0, 3, 6, 9]);
  const X: number[][] = [], y: Array<0 | 1> = [], finer: string[] = [], split: Split[] = [], groups: string[] = [];
  for (let i = 0; i < n; i++) {
    const k = i % 12;
    X.push(unit(centres[k].map((v) => v + 0.32 * gauss(rand))));
    y.push(route.has(k) ? 1 : 0);
    finer.push(`intent-${k}`);
    split.push((['train', 'train', 'calibration', 'test'] as const)[Math.floor(i / 12) % 4]);
    groups.push(`g${seed}-${i}`);
  }
  return { X, y, finer, split, groups };
}

const embedding = { model_id: 'test', dimensions: 16, normalize: true };
/** False alarms at the target recall, with an oracle threshold, on held-out rows. */
function falseAlarms(scores: number[], y: Array<0 | 1>, recall: number): number {
  const order = scores.map((s, i) => [s, y[i]] as const).sort((a, b) => b[0] - a[0]);
  const pos = y.filter((v) => v === 1).length, neg = y.length - pos;
  let tp = 0, fp = 0;
  for (const [, v] of order) { if (v) tp++; else fp++; if (tp / pos >= recall) return fp / neg; }
  return 1;
}

test('a finer head stores the shared model, validates, and runtime scoring reproduces evaluation exactly', () => {
  const { X, y, finer, split, groups } = world(1440, 1);
  const result = trainHeads<'route'>({ X, split, groups, finer: { labels: finer }, heads: [{ name: 'route', y, prevalence: 1 / 3, policy, finer: true }] });
  const spec = result.heads.route!;
  assert.equal(spec.features?.kind, 'finer');
  assert.equal(spec.weights.length, 17, 'one weight per dimension plus the route score');
  assert.equal(result.finer!.classes.length, 12);
  assert.equal(result.reference, undefined, 'a finer head needs no reference');
  const artifact = buildArtifact(result, { version: 'v1', embedding });
  assert.ok(artifact.finer);
  validateArtifact(JSON.parse(JSON.stringify(artifact)));
  assertRoundTrip(artifact, X, result.testProbabilities);
  // Route rates: exactly the route's union, smoothed toward the overall rate.
  const rates = (spec.features as { route_rates: number[] }).route_rates;
  result.finer!.classes.forEach((c, k) => assert.equal(rates[k] > 0.5, [0, 3, 6, 9].includes(Number(c.split('-')[1])), `${c}: ${rates[k]}`));
});

test('on a route that is a union of finer labels, the finer head has fewer false alarms than the embedding alone', () => {
  const { finer, y, ...train } = world(1440, 2), fresh = world(1200, 3);
  const finerResult = trainHeads<'route'>({ ...train, finer: { labels: finer }, heads: [{ name: 'route', y, prevalence: 1 / 3, policy, finer: true }] });
  const plain = trainHeads<'route'>({ ...train, heads: [{ name: 'route', y, prevalence: 1 / 3, policy, type: 'linear' }] });
  const score = (r: typeof plain) => { const a = buildArtifact(r, { version: 'v1', embedding }); return fresh.X.map((x) => scoreEmbedding(a, x).route!); };
  const faFiner = falseAlarms(score(finerResult), fresh.y, 0.9), faPlain = falseAlarms(score(plain), fresh.y, 0.9);
  assert.ok(faFiner < faPlain, `finer ${faFiner} vs plain ${faPlain}`);
});

test('training rows get out-of-fold finer probabilities; other rows the stored model', () => {
  const { X, finer, groups } = world(480, 4);
  const rows = X.map((_, i) => i).filter((i) => i % 2 === 0);
  const fit = fitFinerModel(X, finer, rows, (i) => groups[i]);
  let differ = 0;
  for (const i of rows) {
    const full = finerProbabilities(fit.model, X[i]), oof = fit.outOfFold.get(i)!;
    assert.ok(Math.abs(oof.reduce((a, b) => a + b, 0) - 1) < 1e-9);
    if (full.some((p, k) => Math.abs(p - oof[k]) > 1e-9)) differ++;
  }
  assert.equal(differ, rows.length, 'every fitted row is scored out of fold');
  assert.equal(fit.model.rows, rows.length);
});

test('finer is refused without labels, with a knn or stack head, or when the artifact is inconsistent', () => {
  const { X, y, finer, split, groups } = world(720, 5);
  const head = { name: 'route' as const, y, prevalence: 1 / 3, policy, finer: true };
  assert.throws(() => trainHeads<'route'>({ X, split, groups, heads: [head] }), /needs TrainInput.finer/);
  assert.throws(() => trainHeads<'route'>({ X, split, groups, finer: { labels: finer }, heads: [{ ...head, type: 'knn' }] }), /type knn is not supported/);
  assert.throws(() => trainHeads<'route'>({ X, split, groups, finer: { labels: finer.map(() => 'one') }, heads: [head] }), /at least two/);
  const result = trainHeads<'route'>({ X, split, groups, finer: { labels: finer }, heads: [head] });
  const artifact = JSON.parse(JSON.stringify(buildArtifact(result, { version: 'v1', embedding })));
  assert.throws(() => validateArtifact({ ...artifact, finer: undefined }), /no finer model/);
  const short = { ...artifact, heads: { route: { ...artifact.heads.route, features: { ...artifact.heads.route.features, route_rates: [0.5] } } } };
  assert.throws(() => validateArtifact(short), /one route rate/);
  assert.throws(() => validateArtifact({ ...artifact, finer: { ...artifact.finer, coef: artifact.finer.coef.slice(1) } }), /rows of 16/);
});

test('standardize fits on standardised features and folds them into raw weights', () => {
  const { X: X0, y, split, groups } = world(960, 6);
  // Uneven scales, which change what the L2 penalty shrinks.
  const X = X0.map((x) => x.map((v, j) => v * (1 + j)));
  const d = X[0].length, mean = new Array<number>(d).fill(0), sd = new Array<number>(d).fill(0);
  for (const x of X) x.forEach((v, j) => (mean[j] += v / X.length));
  for (const x of X) x.forEach((v, j) => (sd[j] += (v - mean[j]) ** 2 / X.length));
  const Z = X.map((x) => x.map((v, j) => (v - mean[j]) / Math.sqrt(sd[j])));
  const head = { name: 'route' as const, y, prevalence: 1 / 3, policy, type: 'linear' as const };
  const a = trainHeads<'route'>({ X, split, groups, C: 0.01, standardize: true, heads: [head] });
  const b = trainHeads<'route'>({ X: Z, split, groups, C: 0.01, heads: [head] });
  const c = trainHeads<'route'>({ X, split, groups, C: 0.01, heads: [head] });
  X.forEach((x, i) => {
    const za = decisionFunction({ coef: a.heads.route!.weights, intercept: a.heads.route!.bias }, x);
    const zb = decisionFunction({ coef: b.heads.route!.weights, intercept: b.heads.route!.bias }, Z[i]);
    assert.ok(Math.abs(za - zb) < 1e-8, `row ${i}: ${za} vs ${zb}`);
  });
  assert.notDeepEqual(a.heads.route!.weights, c.heads.route!.weights);
  assertRoundTrip(buildArtifact(a, { version: 'v1', embedding }), X, a.testProbabilities);
});
