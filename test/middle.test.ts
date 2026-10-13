import assert from 'node:assert/strict';
import { test } from 'node:test';

import { STATIC_SCHEMA } from '@liquidau/text-preprocessing';

import {
  assembleMiddle, assertRoundTrip, cascadeCost, cascadeInput, checkTierPairing, chooseCascade, edgeHeadDigest, middleHeadDigest, scoreMiddle, trainEdge, trainHeads, trainMiddle,
  validateArtifact, validateMiddle, type ClassifierArtifact, type EmbeddingSpec, type HeadPolicy, type Split,
} from '../src/index.ts';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}
const WORDS = ['card', 'stolen', 'lost', 'fraud', 'balance', 'transfer', 'hours', 'branch', 'statement', 'fee', 'limit', 'app'];
/** Positives often mention "stolen"/"fraud"; X (main), mid (middle) and vec (static) carry noisy signals of decreasing strength. */
function world(n = 4000, seed = 7) {
  const rand = rng(seed);
  const texts: string[] = [], y: Array<0 | 1> = [], split: Split[] = [], groups: string[] = [], X: number[][] = [], mid: number[][] = [], vec: number[][] = [];
  for (let i = 0; i < n; i++) {
    const label = rand() < 0.2 ? 1 : 0;
    const words = Array.from({ length: 6 }, () => WORDS[Math.floor(rand() * WORDS.length)]).filter((w) => w !== 'stolen' && w !== 'fraud');
    if (label && rand() < 0.3) words.splice(Math.floor(rand() * words.length), 0, rand() < 0.5 ? 'stolen' : 'fraud');
    texts.push(`${words.join(' ')} #${i}`);
    y.push(label);
    split.push((['train', 'train', 'calibration', 'test'] as const)[i % 4]);
    groups.push(`g${i}`);
    X.push(Array.from({ length: 6 }, (_, j) => (j < 3 ? (label ? 1.2 : -1.2) : 0) + rand() * 2 - 1));
    mid.push(Array.from({ length: 5 }, (_, j) => (j < 2 ? (label ? 0.6 : -0.6) : 0) + rand() * 2 - 1));
    vec.push(Array.from({ length: 4 }, (_, j) => (j === 0 ? (label ? 0.2 : -0.2) : 0) + rand() - 0.5));
  }
  return { texts, y, split, groups, X, mid, vec };
}
const identity = { schema: STATIC_SCHEMA, modelId: 'fixture', dimensions: 4, maxTokens: 512, tokenizerSha256: 'a'.repeat(64), weightsSha256: 'b'.repeat(64) } as const;
const midSpec: EmbeddingSpec = { model_id: 'middle-fixture', dimensions: 5, normalize: false, precision: 'fp32' };
const policy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 } as const satisfies HeadPolicy;

/** The edge, a host-trained middle head, and the certified middle tier behind the edge's first candidate. */
function tiers(w = world()) {
  const edge = trainEdge<'h'>({ texts: w.texts, split: w.split, groups: w.groups, scorers: ['lexical', 'static'], static: { identity, vectors: w.vec }, heads: [{ name: 'h', y: w.y, maxDismissed: 0.5 * (1 - policy.targetRecall) }] });
  const midHead = trainHeads<'h'>({ X: w.mid, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, type: 'linear' }] }).heads.h!;
  const middle = trainMiddle<'h'>({ embedding: midSpec, vectors: w.mid, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, model: { coef: midHead.weights, intercept: midHead.bias }, targetRecall: policy.targetRecall, edge: edge.rungs.h![0] }] });
  return { w, edge, middle };
}

test('cascadeCost never splits tied scores', () => {
  // Four rows tied at 1: recall 0.5 is only reachable by taking all of them (two false alarms of two).
  assert.equal(cascadeCost([1, 1, 1, 1, 0], [1, 0, 0, 1, 1], 0.5), 1);
  assert.equal(cascadeCost([3, 2, 1, 0], [1, 1, 0, 0], 1), 0);
  assert.equal(cascadeCost([0, 0], [0, 0], 0.5), 1, 'no positives: unreachable');
});

test('chooseCascade keeps an informative middle tier and drops one that only adds false alarms', () => {
  const rand = rng(3), n = 3000;
  const y = Array.from({ length: n }, () => (rand() < 0.2 ? 1 : 0) as 0 | 1);
  const edge = y.map((v) => (v ? 1 : -1) * 0.3 + rand() * 2 - 1);
  const head = y.map((v) => (v ? 1 : -1) * 0.4 + rand() * 2 - 1);
  const budgets = { edgeFire: 0.005, edgeDismiss: 0.1, middleFire: 0.01, middleDismiss: 0.15 };
  const good = chooseCascade({ y, edge, middle: y.map((v) => (v ? 1 : -1) * 3 + rand() - 0.5), head, budgets, targetRecall: 0.8 });
  assert.equal(good.cascade, true, JSON.stringify(good.costs));
  // A middle score that is noise: its 0.5% extra fires and its dismissals of positives cost the head.
  const bad = chooseCascade({ y, edge, middle: y.map(() => rand()), head, budgets: { ...budgets, middleDismiss: 0.3 }, targetRecall: 0.8 });
  assert.equal(bad.cascade, false, JSON.stringify(bad.costs));
  assert.ok(bad.costs.cascade > bad.costs.twoTier);
  assert.throws(() => chooseCascade({ y, edge: edge.slice(1), middle: edge, head, budgets, targetRecall: 0.8 }), /same length/);
});

test('trainMiddle certifies cumulative budgets behind the edge and decides only what the edge left', () => {
  const { w, edge, middle } = tiers();
  const r = middle.heads.h!, er = edge.rungs.h![0].rows;
  assert.ok(r.model.fire !== null || r.model.dismiss !== null);
  assert.equal(r.model.claims.prior_edge_sha256, edge.rungs.h![0].sha256);
  assert.equal(r.model.claims.max_false_alarms, 0.01);
  assert.ok(Math.abs(r.model.claims.max_dismissed - 0.75 * (1 - policy.targetRecall)) < 1e-12);
  // Every edge decision is kept; the middle adds decisions only on rows the edge left.
  for (let i = 0; i < w.y.length; i++) {
    if (er.fired[i]) assert.ok(r.rows.fired[i]);
    if (er.dismissed[i]) assert.ok(r.rows.dismissed[i]);
    assert.ok(!(r.rows.fired[i] && r.rows.dismissed[i]));
  }
  const added = r.rows.fired.filter((f, i) => f && !er.fired[i]).length + r.rows.dismissed.filter((d, i) => d && !er.dismissed[i]).length;
  assert.ok(added > 0, 'the middle decides some rows');
  // Cumulative point estimates on calibration sit inside the certified budgets.
  const cal = w.split.map((s, i) => (s === 'calibration' ? i : -1)).filter((i) => i >= 0);
  const neg = cal.filter((i) => w.y[i] === 0), pos = cal.filter((i) => w.y[i] === 1);
  assert.ok(neg.filter((i) => r.rows.fired[i]).length / neg.length <= 0.01);
  assert.ok(pos.filter((i) => r.rows.dismissed[i]).length / pos.length <= r.model.claims.max_dismissed);
  // Runtime scoring reproduces the training decisions.
  const doc = middle.middle;
  for (let i = 0; i < 200; i++) if (!er.fired[i] && !er.dismissed[i]) {
    const d = scoreMiddle(doc, w.mid[i]).h!.decision;
    assert.equal(d === 'fire', r.rows.fired[i]);
    assert.equal(d === 'dismiss', r.rows.dismissed[i]);
  }
  assert.equal(r.sha256, middleHeadDigest(doc, 'h'));
  assert.ok(r.trainScores.every((v, i) => (w.split[i] === 'train' ? Number.isFinite(v) : Number.isNaN(v))));
  validateMiddle(JSON.parse(JSON.stringify(doc)));
});

test('trainHeads with a cascade: the choice is recorded, the head is certified behind both tiers, and the release pairs', () => {
  const { w, edge, middle } = tiers();
  const casc = cascadeInput(edge, middle, 'h')!;
  const edgeRows = { ...edge.rungs.h![0].rows, edgeSha256: edge.rungs.h![0].sha256 };
  const res = trainHeads<'h'>({ X: w.X, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: edgeRows, cascade: casc }] });
  const choice = res.cascadeChoice.h!;
  assert.ok(choice.costs && Number.isFinite(choice.costs.twoTier) && Number.isFinite(choice.costs.cascade));
  assert.equal(choice.cascade, choice.costs.cascade <= 1.1 * choice.costs.twoTier + 0.001);
  const spec = res.heads.h!;
  assert.equal(spec.edge!.model_sha256, edge.rungs.h![0].sha256);
  const doc = assembleMiddle(middle, res.cascadeChoice);
  const edgeDoc = edge.edge;
  const artifact: ClassifierArtifact<'h'> = { version: 'v1', created_at: '', embedding: { model_id: 't', dimensions: 6, normalize: false }, heads: res.heads };
  if (choice.cascade) {
    assert.deepEqual(spec.middle, { model_sha256: middleHeadDigest(doc!, 'h') });
    assert.ok(res.testProbabilities.h!.idx.every((i) => !casc.rows.fired[i] && !casc.rows.dismissed[i]), 'rows either tier decided are never scored at runtime');
  } else assert.equal(doc, null);
  assertRoundTrip(artifact, w.X, res.testProbabilities);
  validateArtifact(JSON.parse(JSON.stringify(artifact)));
  assert.deepEqual(checkTierPairing(artifact, edgeDoc, doc), []);

  // Forced: the middle kept regardless, and the pairing holes are caught.
  const forced = trainHeads<'h'>({ X: w.X, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: edgeRows, cascade: { ...casc, choice: 'cascade' } }] });
  assert.deepEqual(forced.cascadeChoice.h, { cascade: true, forced: true, criterion: 'fixed by HeadInput.cascade.choice' });
  // The head's threshold is chained to the edge's two claims and the middle's two.
  const g = forced.heads.h!.guarantee!;
  if (g.delta !== undefined) assert.deepEqual(g.chain, { claims: 4, ladder: 1, joint_delta: Math.min(1, 5 * g.delta) });
  const twoTier = trainHeads<'h'>({ X: w.X, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: edgeRows, cascade: { ...casc, choice: 'edge' } }] }).heads.h!.guarantee!;
  if (twoTier.delta !== undefined) assert.equal(twoTier.chain?.claims, 2);
  const fa: ClassifierArtifact<'h'> = { ...artifact, heads: forced.heads };
  const mdoc = assembleMiddle(middle, forced.cascadeChoice)!;
  assert.deepEqual(checkTierPairing(fa, edgeDoc, mdoc), []);
  assert.match(checkTierPairing(fa, edgeDoc, null).join(), /has no middle tier/);
  assert.match(checkTierPairing(artifact.heads.h!.middle ? { ...fa, heads: { h: { ...fa.heads.h!, middle: undefined } } } : artifact, edgeDoc, mdoc).join(), /not certified with it/);
  const otherEdge = { ...edgeDoc, heads: { h: { ...edgeDoc.heads.h!, fire: (edgeDoc.heads.h!.fire ?? 0) + 1 } } };
  assert.notEqual(edgeHeadDigest(otherEdge, 'h'), edgeHeadDigest(edgeDoc, 'h'));
  assert.match(checkTierPairing(fa, otherEdge, mdoc).join(), /certified behind edge model/);
  const moved = { ...mdoc, heads: { h: { ...mdoc.heads.h!, fire: (mdoc.heads.h!.fire ?? 0) + 1 } } };
  assert.match(checkTierPairing(fa, edgeDoc, moved).join(), /certified with middle model/, 'thresholds are part of the digest');
  assert.equal(fa.heads.h!.middle!.model_sha256, mdoc && middleHeadDigest(mdoc, 'h'));
  assert.throws(() => validateArtifact({ ...fa, heads: { h: { ...fa.heads.h!, edge: undefined } } }), /middle record but no edge record/);

  const none = trainHeads<'h'>({ X: w.X, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: edgeRows, cascade: { ...casc, choice: 'edge' } }] });
  assert.equal(none.heads.h!.middle, undefined);
  assert.equal(none.cascadeChoice.h!.cascade, false);
});

test('cascade refusals: no edge, a ladder, a precision head, rows that drop edge decisions, bad documents', () => {
  const { w, edge, middle } = tiers(world(2400, 9));
  const casc = cascadeInput(edge, middle, 'h')!;
  const edgeRows = { ...edge.rungs.h![0].rows, edgeSha256: edge.rungs.h![0].sha256 };
  const head = { name: 'h' as const, y: w.y, prevalence: 0.2, policy };
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, cascade: casc }] }), /cascade needs the edge/);
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, edge: [edgeRows, edgeRows], cascade: casc }] }), /single edge candidate/);
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, policy: { kind: 'precision', targetPrecision: 0.7 }, edge: edgeRows, cascade: casc }] }), /recall heads only/);
  const firstDismissed = edgeRows.dismissed.indexOf(true);
  assert.ok(firstDismissed >= 0, 'the edge dismisses some rows');
  const dropped = { ...casc, rows: { ...casc.rows, dismissed: casc.rows.dismissed.map((d, i) => (i === firstDismissed ? false : d)) } };
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, edge: edgeRows, cascade: dropped }] }), /keep every edge decision/);
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, edge: edgeRows, cascade: { ...casc, scores: { ...casc.scores, edge: casc.scores.edge.slice(1) } } }] }), /cascade.scores.edge has/);
  // Documents.
  const doc = JSON.parse(JSON.stringify(middle.middle));
  assert.throws(() => validateMiddle({ ...doc, embedding: { ...doc.embedding, precision: undefined } }), /precision is required/);
  assert.throws(() => validateMiddle({ ...doc, embedding: { ...doc.embedding, context: { model_id: 'c', dimensions: 2, normalize: true } } }), /message only/);
  assert.throws(() => validateMiddle({ ...doc, heads: { h: { ...doc.heads.h, model: { coef: [1], intercept: 0 } } } }), /linear model over the middle embedding's 5 dimensions/);
  assert.throws(() => validateMiddle({ ...doc, heads: { h: { ...doc.heads.h, claims: { ...doc.heads.h.claims, prior_edge_sha256: 'x' } } } }), /claims must be/);
  assert.throws(() => validateMiddle({ ...doc, heads: { h: { ...doc.heads.h, fire: null, dismiss: null } } }), /decides nothing/);
  assert.throws(() => scoreMiddle(doc, [0, 0, 0]), /3 values, the middle encoder 5/);
  assert.throws(() => scoreMiddle(doc, [0, 0, 0, 0, NaN]), /non-finite/);
  assert.throws(() => trainMiddle<'h'>({ embedding: { ...midSpec, precision: undefined }, vectors: w.mid, split: w.split, heads: [] }), /precision is required/);
});

test('a firing budget equal to the edge\'s makes a dismiss-only middle tier', () => {
  const { w, edge } = tiers();
  const midHead = trainHeads<'h'>({ X: w.mid, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, type: 'linear' }] }).heads.h!;
  const r = trainMiddle<'h'>({ embedding: midSpec, vectors: w.mid, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, model: { coef: midHead.weights, intercept: midHead.bias }, targetRecall: policy.targetRecall, edge: edge.rungs.h![0], maxFalseAlarms: edge.rungs.h![0].model.claims.max_false_alarms }] }).heads.h!;
  assert.equal(r.model.fire, null);
  assert.ok(r.model.dismiss !== null);
  const er = edge.rungs.h![0].rows;
  assert.ok(r.rows.fired.every((f, i) => f === er.fired[i]), 'no fires beyond the edge\'s');
  assert.equal(r.budgets.middleFire, r.budgets.edgeFire);
  assert.throws(() => trainMiddle<'h'>({ embedding: midSpec, vectors: w.mid, split: w.split, heads: [{ name: 'h', y: w.y, model: { coef: midHead.weights, intercept: midHead.bias }, targetRecall: policy.targetRecall, edge: edge.rungs.h![0], maxFalseAlarms: 0.001 }] }), /at least the edge/);
});

test('a PAC guarantee behind the edge and the middle records its chain: 4 earlier claims, a joint delta of 5δ', () => {
  const { w, edge, middle } = tiers();
  const casc = cascadeInput(edge, middle, 'h')!;
  const edgeRows = { ...edge.rungs.h![0].rows, edgeSha256: edge.rungs.h![0].sha256 };
  const pac: HeadPolicy = { kind: 'recall', targetRecall: 0.8, mode: 'conformal-pac', delta: 0.05 };
  const res = trainHeads<'h'>({ X: w.X, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy: pac, edge: edgeRows, cascade: { ...casc, choice: 'cascade' } }] });
  const g = res.heads.h!.guarantee!;
  assert.equal(g.kind, 'pac');
  assert.deepEqual(g.chain, { claims: 4, ladder: 1, joint_delta: 0.25 });
  const alone = trainHeads<'h'>({ X: w.X, split: w.split, groups: w.groups, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy: pac }] });
  assert.equal(alone.heads.h!.guarantee!.chain, undefined, 'a head certified alone has no chain');
});
