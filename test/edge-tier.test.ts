import assert from 'node:assert/strict';
import { test } from 'node:test';

import { STATIC_SCHEMA } from '@liquidau/text-preprocessing';

import { assembleEdge, assertRoundTrip, CalibrationRemainderError, checkEdgePairing, edgeDigest, edgeHeadDigest, edgeLadder, scoreEdge, trainEdge, trainHeads, validateArtifact, validateEdge, type ClassifierArtifact, type HeadPolicy, type Split } from '../src/index.ts';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}
const WORDS = ['card', 'stolen', 'lost', 'fraud', 'balance', 'transfer', 'hours', 'branch', 'statement', 'fee', 'limit', 'app'];
/** Positives mention "stolen"/"fraud" most of the time; the static vector and X carry a noisy signal. */
function world(n = 3000, seed = 3) {
  const rand = rng(seed);
  const texts: string[] = [], y: Array<0 | 1> = [], split: Split[] = [], groups: string[] = [], X: number[][] = [], vec: number[][] = [];
  for (let i = 0; i < n; i++) {
    const label = rand() < 0.2 ? 1 : 0;
    const words = Array.from({ length: 6 }, () => WORDS[Math.floor(rand() * WORDS.length)]).filter((w) => w !== 'stolen' && w !== 'fraud');
    if (label && rand() < 0.85) words.splice(Math.floor(rand() * words.length), 0, rand() < 0.5 ? 'stolen' : 'fraud');
    texts.push(`${words.join(' ')} #${i}`);
    y.push(label);
    split.push((['train', 'train', 'calibration', 'test'] as const)[i % 4]);
    groups.push(`g${i}`);
    X.push(Array.from({ length: 6 }, (_, j) => (j < 2 ? (label ? 1 : -1) : 0) + rand() * 2 - 1));
    vec.push(Array.from({ length: 4 }, (_, j) => (j === 0 ? (label ? 0.8 : -0.8) : 0) + rand() - 0.5));
  }
  return { texts, y, split, groups, X, vec };
}
const identity = { schema: STATIC_SCHEMA, modelId: 'fixture', dimensions: 4, maxTokens: 512, tokenizerSha256: 'a'.repeat(64), weightsSha256: 'b'.repeat(64) } as const;

test('trainEdge: certified fire and dismiss thresholds, runtime scoring reproduces the training decisions', () => {
  const w = world();
  const ruleFired = w.texts.map((t) => / fraud /.test(` ${t} `));
  const r = trainEdge<'h'>({ texts: w.texts, split: w.split, groups: w.groups, static: { identity, vectors: w.vec }, heads: [{ name: 'h', y: w.y, ruleFired, maxDismissed: 0.1 }] });
  const doc = validateEdge(JSON.parse(JSON.stringify(r.edge)));
  assert.equal(edgeDigest(doc), r.edgeSha256, 'the digest survives a JSON round trip');
  const m = doc.heads.h!;
  assert.deepEqual(m.scorers, ['rules', 'lexical', 'static']);
  assert.ok(m.fire !== null && m.dismiss !== null && m.dismiss <= m.fire);
  const rows = r.rows.h!;
  // The runtime scorer gives the same decision as training did, on every row.
  w.texts.forEach((text, i) => {
    const d = scoreEdge(doc, { text, ruleFired: () => ruleFired[i], static: w.vec[i] }).h!.decision;
    assert.equal(d === 'fire', rows.fired[i], `row ${i} fire`);
    assert.equal(d === 'dismiss', rows.dismissed[i], `row ${i} dismiss`);
  });
  // The claims hold on the held-out test rows (a 95% statement; this world has plenty of margin).
  const test = w.split.map((s, i) => (s === 'test' ? i : -1)).filter((i) => i >= 0);
  const neg = test.filter((i) => !w.y[i]), pos = test.filter((i) => w.y[i]);
  assert.ok(neg.filter((i) => rows.fired[i]).length / neg.length <= 0.02, 'false alarms near the 0.5% claim');
  assert.ok(pos.filter((i) => rows.dismissed[i]).length / pos.length <= 0.15, 'dismissed positives near the 10% claim');
  assert.ok(rows.fired.filter(Boolean).length + rows.dismissed.filter(Boolean).length > 0.3 * w.texts.length, 'the edge settles a real share');
  // No dismissal when maxDismissed is 0.
  const noDismiss = trainEdge<'h'>({ texts: w.texts, split: w.split, scorers: ['lexical'], heads: [{ name: 'h', y: w.y, maxDismissed: 0 }] });
  assert.equal(noDismiss.edge.heads.h!.dismiss, null);
  assert.ok(!noDismiss.rows.h!.dismissed.some(Boolean));
});

test('trainHeads with the edge: joint certification, the edge record, pairing and round trip', () => {
  const w = world();
  const r = trainEdge<'h'>({ texts: w.texts, split: w.split, groups: w.groups, scorers: ['lexical', 'static'], static: { identity, vectors: w.vec }, heads: [{ name: 'h', y: w.y, maxDismissed: 0.08 }] });
  const rows = r.rows.h!;
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 };
  const joint = trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: { ...rows, edgeSha256: r.rungs.h![0].sha256 } }] });
  const alone = trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy }] });
  const spec = joint.heads.h!;
  assert.deepEqual(spec.edge, { model_sha256: edgeHeadDigest(r.edge, 'h') });
  assert.equal(joint.edgeChoice?.h, 0);
  // Dismissed positives are misses, fired ones caught: test recall counts the whole system.
  const test = w.split.map((s, i) => (s === 'test' ? i : -1)).filter((i) => i >= 0 && w.y[i] === 1);
  const dismissed = test.filter((i) => rows.dismissed[i]).length / test.length;
  assert.ok(joint.evaluation.h!.recall <= 1 - dismissed + 1e-12);
  // The head compensates for edge dismissals with a lower threshold than alone (or equal) - when it has
  // calibration positives left to place one. Here the edge settles every calibration positive, so the joint
  // calibrator fits negatives only and the comparison is undefined (under plain Platt both thresholds
  // used to saturate to 1.0, which hid this; Firth keeps the lone head's finite).
  const remainderPositives = w.split.filter((s, i) => s === 'calibration' && w.y[i] === 1 && !rows.fired[i] && !rows.dismissed[i]).length;
  if (remainderPositives > 0) assert.ok(spec.threshold <= alone.heads.h!.threshold + 1e-12);
  else assert.equal(remainderPositives, 0, 'fixture: the edge settles every calibration positive');
  // Rows the edge decided are left out of the round-trip probabilities (never scored at runtime).
  assert.ok(joint.testProbabilities.h!.idx.every((i) => !rows.fired[i] && !rows.dismissed[i]));
  const artifact: ClassifierArtifact<'h'> = { version: 'v1', created_at: '', embedding: { model_id: 't', dimensions: 6, normalize: false }, heads: joint.heads };
  assertRoundTrip(artifact, w.X, joint.testProbabilities);
  validateArtifact(artifact);
  assert.throws(() => validateArtifact({ ...artifact, heads: { h: { ...spec, edge: { model_sha256: 'x' } } } }), /invalid edge record/);
  assert.deepEqual(checkEdgePairing(artifact, r.edge), []);
  const other = { ...r.edge, heads: { h: { ...r.edge.heads.h!, fire: (r.edge.heads.h!.fire ?? 0) + 1 } } };
  assert.match(checkEdgePairing(artifact, other)[0], /certified with edge model/);
  assert.match(checkEdgePairing(artifact, null)[0], /has no edge/);
  const plain: ClassifierArtifact<'h'> = { ...artifact, heads: alone.heads };
  assert.match(checkEdgePairing(plain, r.edge)[0], /not certified with it/);
  // Inconsistent rows are refused.
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: { fired: rows.fired.map(() => true), dismissed: rows.fired.map(() => true), edgeSha256: r.rungs.h![0].sha256 } }] }), /both fired and dismissed/);
});

test('reserve: the threshold is certified at target + reserve, the gates stay at the target', () => {
  const w = world(3000, 9);
  const base: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9, mode: 'conformal-pac' };
  const run = (reserve?: number) => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy: { ...base, ...(reserve !== undefined ? { reserve } : {}) } as HeadPolicy }] });
  const r0 = run(), r5 = run(0.05);
  assert.ok(r5.heads.h!.threshold <= r0.heads.h!.threshold, 'a reserve lowers (or keeps) the threshold');
  assert.ok(r5.evaluation.h!.recall >= r0.evaluation.h!.recall);
  assert.throws(() => run(0.25), /reserve must be/);
});

test('validateEdge refuses malformed documents', () => {
  const w = world(1200, 4);
  const r = trainEdge<'h'>({ texts: w.texts, split: w.split, scorers: ['lexical'], heads: [{ name: 'h', y: w.y, maxDismissed: 0.1 }] });
  const doc = JSON.parse(JSON.stringify(r.edge));
  assert.throws(() => validateEdge({ ...doc, schema: 'liquidau-edge/9' }), /unsupported edge schema/);
  assert.throws(() => validateEdge({ ...doc, heads: { h: { ...doc.heads.h, blend: { coef: [1, 2], intercept: 0 } } } }), /one weight per scorer/);
  assert.throws(() => validateEdge({ ...doc, heads: { h: { ...doc.heads.h, scorers: ['static'] } } }), /static must be/);
  assert.throws(() => validateEdge({ ...doc, heads: { h: { ...doc.heads.h, lexical: { ...doc.heads.h.lexical, index: [5, 3] } } } }), /ascending/);
});

test('calibration gates judge only the rows the head scores: edge-decided rows (preset 0/1) do not fail the Cox test', () => {
  const w = world(6000, 21);
  // An edge that dismisses a tenth of the positives and a third of the negatives, and fires nothing:
  // scored as 0, those positives would look badly miscalibrated if the gates counted them.
  const dismissed = w.y.map((v, i) => (v === 1 ? i % 10 === 3 : i % 3 === 0));
  const rows = { fired: w.y.map(() => false), dismissed };
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 };
  const joint = trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: { ...rows, edgeSha256: 'e'.repeat(64) } }] });
  assert.ok(!joint.failures.some((f) => /miscalibrated|ECE/.test(f)), joint.failures.join('; '));
  // The evaluation still counts the dismissed positives as misses.
  const testPos = w.y.map((v, i) => (v === 1 && w.split[i] === 'test' ? i : -1)).filter((i) => i >= 0);
  assert.ok(joint.evaluation.h!.recall <= 1 - testPos.filter((i) => dismissed[i]).length / testPos.length + 1e-12);
});

test('the ECE gate needs enough rows left to the head: with few, a high ECE is a warning, not a failure', async () => {
  const { gateHead } = await import('../src/index.ts');
  const base = { recall: 0.9, recall_ci95: [0.8, 0.95] as [number, number], positives: 50, ece_prevalence_weighted: 0.13, reliability: [], slices: {} } as never;
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 };
  const few = gateHead(policy, { ...(base as object), calibration_rows: 29 } as never);
  assert.ok(!few.failures.some((f) => /ECE/.test(f)) && few.warnings.some((w) => /too few to judge calibration/.test(w)));
  const many = gateHead(policy, { ...(base as object), calibration_rows: 500 } as never);
  assert.ok(many.failures.some((f) => /ECE 0.130 > 0.05/.test(f)));
});

test('an edge ladder: a head that fails with the full dismissal share falls back, and assembleEdge pairs with the artifact', () => {
  const w = world(4000, 31);
  const r = trainEdge<'h'>({ texts: w.texts, split: w.split, groups: w.groups, scorers: ['lexical', 'static'], static: { identity, vectors: w.vec }, heads: [{ name: 'h', y: w.y, maxDismissed: [0.19, 0.05, 0] }] });
  assert.equal(r.rungs.h!.length, 3);
  assert.deepEqual(r.rungs.h!.map((x) => x.maxDismissed), [0.19, 0.05, 0]);
  assert.equal(r.rungs.h![2].model.dismiss, null, 'the last rung only fires');
  assert.ok(new Set(r.rungs.h!.map((x) => x.sha256)).size === 3, 'each candidate has its own digest');
  // Dismissing up to 19% of positives leaves a recall-0.8 head no room: it must fall back.
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 };
  const res = trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ name: 'h', y: w.y, prevalence: 0.2, policy, edge: edgeLadder(r, 'h') }] });
  const k = res.edgeChoice!.h!;
  assert.ok(k >= 0 && k <= 2);
  const doc = assembleEdge(r, res.edgeChoice!);
  assert.equal(doc.heads.h!.claims.max_dismissed, r.rungs.h![k].maxDismissed);
  const artifact: ClassifierArtifact<'h'> = { version: 'v1', created_at: '', embedding: { model_id: 't', dimensions: 6, normalize: false }, heads: res.heads };
  assert.deepEqual(checkEdgePairing(artifact, doc), []);
  assert.equal(res.heads.h!.edge!.model_sha256, edgeHeadDigest(doc, 'h'));
  if (k > 0) assert.ok(res.warnings.some((x) => /candidate .* chosen/.test(x)));
  // The guarantee records the edge's claims and every candidate tried.
  const g = res.heads.h!.guarantee!;
  if (g.delta !== undefined) assert.deepEqual(g.chain, { claims: 2, ladder: k + 1, joint_delta: Math.min(1, (k + 1) * 3 * g.delta) });
  if (g.delta !== undefined) assert.throws(() => validateArtifact({ ...artifact, heads: { h: { ...res.heads.h!, guarantee: { ...g, chain: { claims: 2, ladder: 0, joint_delta: 0.5 } } } } }), /guarantee chain must be/);
  else assert.equal(g.chain, undefined, 'only pac and design guarantees carry a delta, and so a chain');
});


test('an edge that decides every calibration row fails clearly, and a ladder skips it', () => {
  const w = world();
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, designRecall: 0.9 };
  // Fires every calibration positive and dismisses every calibration negative: nothing is left to calibrate on.
  const greedy = { fired: w.y.map((v, i) => v === 1 && w.split[i] === 'calibration'), dismissed: w.y.map((v, i) => v === 0 && w.split[i] === 'calibration'), edgeSha256: 'c'.repeat(64) };
  const none = { fired: w.y.map(() => false), dismissed: w.y.map(() => false), edgeSha256: 'd'.repeat(64) };
  const head = { name: 'h' as const, y: w.y, prevalence: 0.2, policy };
  assert.throws(() => trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, edge: greedy }] }), (e: Error) => e instanceof CalibrationRemainderError && /decide every calibration row/.test(e.message));
  const res = trainHeads<'h'>({ X: w.X, split: w.split, heads: [{ ...head, edge: [greedy, none] }] });
  assert.equal(res.edgeChoice?.h, 1);
  assert.ok(res.warnings.some((x) => /candidate 0 skipped/.test(x)));
});
