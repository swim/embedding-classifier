import assert from 'node:assert/strict';
import { test } from 'node:test';

import { designRiskThreshold, htTotal, seededRandom, stratifiedRatio } from '@liquidau/solvers';

import {
  applyReviews,
  capTrainingWeights,
  checkSliceAxes,
  coverageReport,
  defineAxes,
  designOf,
  designSample,
  labelQueue,
  mmrSelect,
  pickThreshold,
  ProvenanceError,
  publishPlan,
  realHardNegatives,
  reportMarkdown,
  retrieveFromSeeds,
  seedStats,
  selectForReview,
  trainHeads,
  validateProvenance,
  verifyBatch,
  type ClassifierArtifact,
  type ExampleRecord,
  type FrameItem,
  type HeadPolicy,
  type Split,
} from '../src/index.ts';

/** Simulated traffic: 8-dim embeddings, a rare true label, a rule that catches some positives, and an old model's score. */
function traffic(N: number, seed: number) {
  const rand = seededRandom(seed);
  const items: Array<FrameItem & { x: number[]; truth: 0 | 1 }> = [];
  for (let i = 0; i < N; i++) {
    const truth = rand() < 0.06 ? 1 : 0;
    const x = Array.from({ length: 8 }, (_, j) => (j < 2 ? (truth ? 1.2 : -1.2) : 0) + (rand() * 2 - 1) * 1.4);
    const score = 1 / (1 + Math.exp(-(x[0] + x[1]) * 1.5));
    const ruleFires = truth ? rand() < 0.3 : rand() < 0.01;
    items.push({ id: `m${i}`, text: `message ${i}`, group: `g${i}`, signals: { ruleFires, score, slice: i % 2 ? 'web' : 'app' }, x, truth });
  }
  return items;
}
const frame = traffic(6000, 1);
const byId = new Map(frame.map((f) => [f.id, f]));
const design = (seed = 7, total = 1600) => designSample({ frame, scoreBands: [0.95, 0.8], allocation: { total, method: 'proportional' }, scoringModel: 'old-v1', seed });
/** A perfect reviewer labels every queued record for head h. */
function label(records: ExampleRecord[], seed = 3) {
  const q = labelQueue({ items: records.map((record) => ({ record, mechanism: 'sampled' as const, heads: ['h'] })), budget: { sampled: records.length, retrieved: 0, verification: 0 }, seed });
  return applyReviews(q.key, q.review.map((r) => ({ itemId: r.itemId, labels: { h: byId.get(q.key.items[r.itemId].record.id)!.truth } })));
}

test('designSample: reproducible, floored, π = 1 strata, role minimums', () => {
  const a = design(), b = design();
  assert.equal(a.designId, b.designId);
  assert.deepEqual(a.records, b.records);
  assert.notEqual(design(8).designId, a.designId);
  assert.equal(a.records.length, 1600);
  for (const s of a.design.strata) {
    assert.ok(s.n >= Math.min(30, s.N));
    assert.ok(s.roles.calibration >= 2 && s.roles.test >= 2);
  }
  for (const r of a.records) {
    assert.equal(r.source.kind, 'sampled');
    const s = a.design.strata.find((x) => x.name === (r.source as { stratum: string }).stratum)!;
    assert.equal((r.source as { inclusionProb: number }).inclusionProb, s.roles[r.role as Split] / s.N, 'each role is its own sample');
  }
  // A stratum smaller than the floor is taken whole (π = 1).
  const rare = Array.from({ length: 12 }, (_, i): FrameItem => ({ id: `rare${i}`, text: 'r', group: `rare${i}`, signals: { ruleFires: true, score: 0.5 } }));
  const small = designSample({ frame: [...frame.slice(0, 400).map((f) => ({ ...f, signals: { ...f.signals, ruleFires: false } })), ...rare], scoreBands: [0.95], allocation: { total: 150, method: 'proportional' }, minPerStratum: 40, scoringModel: 'old-v1', seed: 1 });
  const whole = small.design.strata.find((s) => s.name.startsWith('rule|'))!;
  assert.deepEqual([whole.N, whole.n, whole.pi], [12, 12, 1]);
  assert.throws(() => designSample({ frame, scoreBands: [0.95, 0.8], allocation: { total: 20, method: 'proportional' }, scoringModel: 'm', seed: 1 }), /below the \d+ that minPerStratum 30 needs/);
  assert.throws(() => designSample({ frame: frame.slice(0, 30), scoreBands: [0.95, 0.8], allocation: { total: 30, method: 'proportional' }, minPerStratum: 4, mergeSmallStrata: false, scoringModel: 'm', seed: 1 }), /too small for 2 calibration and 2 test items/);
  const plain = frame.slice(0, 300).map((f) => ({ ...f, signals: { ...f.signals, ruleFires: false } }));
  const dup = [...plain, { ...plain[0], id: 'dup' }];
  assert.equal(designSample({ frame: dup, scoreBands: [0.8], allocation: { total: 100, method: 'proportional' }, scoringModel: 'm', seed: 1 }).design.duplicatesRemoved, 1);
});

test('labelQueue is blinded and budgeted; applyReviews recomputes inclusion probabilities and reports skips', () => {
  const { records } = design();
  const cal = records.filter((r) => r.role === 'calibration');
  const q = labelQueue({ items: cal.map((record) => ({ record, mechanism: 'sampled' as const, heads: ['h'] })), budget: { sampled: 300, retrieved: 0, verification: 0 }, seed: 5 });
  assert.equal(q.review.length, 300);
  assert.equal(q.dropped.sampled, cal.length - 300);
  for (const item of q.review) assert.deepEqual(Object.keys(item).sort(), ['heads', 'itemId', 'text'], 'no source, mechanism, score or rule firing');
  // Skip one in ten of one stratum's items.
  const reviews = q.review.filter((_, j) => j % 10 !== 0).map((r) => ({ itemId: r.itemId, labels: { h: byId.get(q.key.items[r.itemId].record.id)!.truth } }));
  const out = applyReviews(q.key, reviews);
  assert.equal(out.records.length, reviews.length);
  assert.ok(out.records.every((r) => r.labelledBy === 'human'));
  const d = designOf(out.records);
  for (const r of out.records) {
    const src = r.source as { stratum: string; stratumSize: number; designId: string; inclusionProb: number };
    const n = out.records.filter((x) => (x.source as { stratum: string }).stratum === src.stratum).length;
    assert.equal(src.inclusionProb, n / src.stratumSize, 'π = labelled / N_h');
  }
  assert.equal(d.inclusionProbs.length, out.records.length);
  assert.ok(out.warnings.some((w) => /skipped/.test(w)), 'a 10% skip rate is reported');
});

test('HT recall and prevalence from the design are unbiased across 500 draws of a fully labelled frame', () => {
  const truthPrev = frame.filter((f) => f.truth).length / frame.length;
  const fires = (f: { x: number[] }) => f.x[0] + f.x[1] > 0;
  const pos = frame.filter((f) => f.truth);
  const truthRecall = pos.filter(fires).length / pos.length;
  let prev = 0, recall = 0;
  const runs = 500;
  for (let r = 0; r < runs; r++) {
    const test = design(100 + r, 1200).records.filter((x) => x.role === 'test');
    const d = designOf(test);
    const y = test.map((x) => byId.get(x.id)!.truth);
    prev += htTotal(y, d.inclusionProbs) / htTotal(y.map(() => 1), d.inclusionProbs);
    recall += stratifiedRatio({ ...d, num: test.map((x, i) => y[i] * Number(fires(byId.get(x.id)!))), den: y }).estimate;
  }
  assert.ok(Math.abs(prev / runs - truthPrev) < 0.003, `prevalence ${prev / runs} vs ${truthPrev}`);
  assert.ok(Math.abs(recall / runs - truthRecall) < 0.01, `recall ${recall / runs} vs ${truthRecall}`);
});

function provenanceFixture(): ExampleRecord[] {
  const sampled = (id: string, role: 'calibration' | 'test' | 'train', y: 0 | 1): ExampleRecord => ({ id, text: id, group: id, role, source: { kind: 'sampled', designId: 'd', stratum: 's', inclusionProb: 0.5, stratumSize: 4 }, labels: { h: y }, labelledBy: 'human' });
  return [sampled('c1', 'calibration', 1), sampled('t1', 'test', 0), sampled('r1', 'train', 1),
    { id: 'b1', text: 'b1', group: 'gb', role: 'background', backgroundUse: 'budget', source: { kind: 'traffic' }, labels: {} }];
}

test('validateProvenance: P1-P4 and P7 throw with a code, overrides are reported, P5 drops near-duplicates', () => {
  const ok = provenanceFixture();
  assert.equal(validateProvenance(ok).records.length, 4);
  const code = (records: ExampleRecord[], opts = {}) => { try { validateProvenance(records, opts); return null; } catch (e) { return e instanceof ProvenanceError ? e.code : String(e); } };
  assert.equal(code([...ok, { ...ok[0], id: 'c2', group: 'c2', labelledBy: 'llm' }]), 'P1');
  assert.equal(code([...ok, { ...ok[0], id: 'c3', group: 'c3', source: { kind: 'traffic' } }]), 'P1');
  assert.equal(code([...ok, { ...ok[3], id: 'b2', group: 'gb2', labels: { h: 0 } }]), 'P2');
  assert.equal(code([...ok, { ...ok[3], id: 'b3', group: 'gb3', backgroundUse: undefined }]), 'P2');
  assert.equal(code([...ok, { ...ok[2], id: 'r2', group: 'c1' }]), 'P3', 'a calibration group used in training');
  assert.equal(code([...ok, { ...ok[3], id: 'b4', backgroundUse: 'veto' }]), 'P3', 'one background group in two uses');
  const retrieved: ExampleRecord = { id: 'x1', text: 'x1', group: 'x1', role: 'test', source: { kind: 'retrieved', seedIds: ['r1'], similarity: 0.8, round: 1 }, labels: { h: 1 }, labelledBy: 'human' };
  assert.equal(code([...ok, retrieved]), 'P1', 'P1 fires first for a retrieved test record');
  assert.equal(code([...ok, { ...retrieved, role: 'background', backgroundUse: 'budget', labels: {} }]), 'P2');
  assert.equal(code([...ok, { ...retrieved, role: 'stress' }]), null);
  const gen: ExampleRecord = { id: 'g1', text: 'g1', group: 'g1', role: 'train', source: { kind: 'generated', method: 'hard_negative', generator: 'gen', ruleId: 'r', batchId: 'B1' }, labels: { h: 0 }, labelledBy: 'intended' };
  assert.equal(code([...ok, gen]), 'P7');
  assert.equal(code([...ok, gen], { acceptedBatches: ['B1'] }), null);
  assert.equal(code([...ok, gen], { acceptedBatches: ['B1'], safetyCritical: ['h'] }), 'P7', 'safety-critical heads need verification');
  assert.equal(code([...ok, { ...gen, verified: true }], { safetyCritical: ['h'] }), null);
  assert.equal(code([...ok, { ...gen, role: 'test' }]), 'P1');
  const overridden = validateProvenance([...ok, gen], { overrides: ['P7'] });
  assert.deepEqual(overridden.overridden.map((o) => o.code), ['P7']);

  const near = validateProvenance([...ok, { ...gen, verified: true }], { embeddings: [[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.3], [1, 0.01]] });
  assert.deepEqual(near.dropped.map((d) => d.id), ['g1']);
  assert.ok(near.warnings.some((w) => /P5/.test(w)));
});

test('P6 caps: retrieved positives <= 50% of positive weight, generated negatives <= 30% of negative weight and per rule', () => {
  const rec = (id: string, kind: 'traffic' | 'retrieved' | 'generated', y: 0 | 1, ruleId = 'r1'): ExampleRecord => ({
    id, text: id.startsWith('p') ? `trigger ${id}` : id, group: id, role: 'train', labels: { h: y }, labelledBy: kind === 'generated' ? 'intended' : 'human', verified: true,
    source: kind === 'traffic' ? { kind } : kind === 'retrieved' ? { kind, seedIds: ['s'], similarity: 0.9, round: 1 } : { kind, method: 'hard_negative', generator: 'g', ruleId, batchId: 'b' },
  });
  const records = [rec('p1', 'traffic', 1), rec('p2', 'traffic', 1), ...['q1', 'q2', 'q3', 'q4', 'q5', 'q6'].map((id) => rec(id, 'retrieved', 1)),
    ...Array.from({ length: 10 }, (_, i) => rec(`n${i}`, 'traffic', 0)), ...Array.from({ length: 10 }, (_, i) => rec(`x${i}`, 'generated', 0))];
  const ruleMatches = (text: string) => (text.startsWith('trigger') ? ['r1'] : []);
  const { weights, summary } = capTrainingWeights(records, 'h', { ruleMatches });
  const sum = (pred: (r: ExampleRecord) => boolean) => records.reduce((s, r, i) => s + (pred(r) ? weights[i] : 0), 0);
  assert.equal(summary.retrieved_positive.weight, 2, 'retrieved 6 -> 2, half of 4');
  assert.ok(Math.abs(sum((r) => r.source.kind === 'generated') - 2) < 1e-12, 'per rule: at most the 2 traffic positives the rule matches');
  assert.ok(summary.generated_negative.weight <= (0.3 / 0.7) * 10 + 1e-12);
  assert.throws(() => capTrainingWeights(records, 'h'), /need ruleMatches/);
});

test('trainHeads on sampled records: design calibration, design threshold, HT evaluation, provenance and report', () => {
  const sampled = design(7, 3000).records;
  const { records } = label(sampled);
  const background = frame.filter((f) => !records.some((r) => r.group === f.group)).slice(0, 500);
  const bgRecords: ExampleRecord[] = background.map((f) => ({ id: f.id, text: f.text, group: f.group, role: 'background', backgroundUse: 'budget', source: { kind: 'traffic' }, labels: {} }));
  const X = records.map((r) => byId.get(r.id)!.x);
  const split = records.map((r) => r.role as Split);
  const y = records.map((r) => r.labels.h as 0 | 1);
  // linearised: exact is valid but needs far more calibration data across 6 strata (asserted below).
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, mode: 'design', designMethod: 'linearised', delta: 0.05, minPositives: 20 };
  const run = (extra: object = {}) => trainHeads({
    X, split, records, designs: [design(7, 3000).design], heads: [{ name: 'h', y, policy, ...extra }], seed: 1,
    background: { X: background.map((f) => f.x), records: bgRecords, maxRate: { h: 0.5 } },
  });
  const r = run();
  const ev = r.evaluation.h!;
  assert.deepEqual(r.failures, [], r.failures.join('; '));
  assert.equal(r.heads.h!.guarantee!.kind, 'design-approximate');
  assert.ok(ev.design && ev.design.recall.ci95[0] <= ev.recall && ev.recall <= ev.design.recall.ci95[1]);
  assert.ok(ev.design!.recall.bootstrap_ci95, 'bootstrap interval too');
  assert.equal(ev.certified.method, 'design-linearised');
  assert.ok(Math.abs(ev.design!.prevalence.estimate - 0.06) < 0.02, `prevalence ${ev.design!.prevalence.estimate}`);
  assert.equal(r.provenance!.generated, false);
  assert.equal(r.design!.designs.length, 1);

  const artifact: ClassifierArtifact<'h'> = { version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: r.heads, evaluation: r.evaluation, training: { design: r.design, provenance: r.provenance } };
  const report = reportMarkdown(artifact);
  assert.match(report, /Design-based estimates/);
  assert.match(report, /## Sampling designs/);
  assert.match(report, /## Provenance/);

  assert.throws(() => run({ prevalence: 0.06 }), /don't pass prevalence with sampled records/);
  assert.throws(() => trainHeads({ X, split, records, heads: [{ name: 'h', y, policy: { kind: 'recall', targetRecall: 0.8, mode: 'conformal-pac' } }] }), /unequal inclusion probabilities.*use mode 'design'/);
  // exact: per-stratum worst cases add up - with ~12% of each stratum in calibration, about 260
  // positives could be missed unseen against ~360 real ones, so 80% recall can't be certified.
  const infeasible = trainHeads({ X, split, records, heads: [{ name: 'h', y, policy: { kind: 'recall', targetRecall: 0.8, mode: 'design' } }] });
  assert.ok(infeasible.failures.some((f) => /^h: design: the miss-rate bound/.test(f)), infeasible.failures.join('; '));
  assert.ok(infeasible.heads.h!.threshold > 1, 'never fires rather than substitute a threshold');
  assert.throws(() => trainHeads({ X, split: split.map((s) => (s === 'test' ? 'train' : s)) as Split[], records, heads: [{ name: 'h', y, policy }] }), /has role test but split/);
});

test('weighted heuristic threshold equals the unweighted one under equal weights', () => {
  const p = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.5, 0.65];
  const y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.6, designRecall: 0.8, maxFalseAlarm: 0.2 };
  assert.equal(pickThreshold(policy, p, y, p.map(() => 3)), pickThreshold(policy, p, y));
  assert.notEqual(pickThreshold(policy, p, y, [1, 1, 1, 1, 50, 1, 1, 1, 1, 1]), pickThreshold(policy, p, y), 'a heavy low-scoring positive drags the threshold down');
});

test('retrieveFromSeeds: MMR, exclusions, dimension checks, round-2 seeds; seedStats retires weak seeds', async () => {
  const seed: ExampleRecord = { id: 's1', text: 's1', group: 's1', role: 'train', source: { kind: 'sampled', designId: 'd', stratum: 'x', inclusionProb: 1, stratumSize: 1 }, labels: { h: 1 }, labelledBy: 'human' };
  const poolRec = (id: string, extra: Partial<ExampleRecord> = {}): ExampleRecord => ({ id, text: id, group: id, role: 'train', source: { kind: 'traffic' }, labels: {}, ...extra });
  const pool = [
    { record: poolRec('a'), embedding: [1, 0.05, 0] },
    { record: poolRec('a2'), embedding: [1, 0.06, 0] }, // near-copy of a: MMR prefers b next
    { record: poolRec('b'), embedding: [1, -0.5, 0.2] },
    { record: poolRec('far'), embedding: [0, 1, 0] }, // below the floor
    { record: poolRec('labelled', { labels: { h: 0 } }), embedding: [1, 0, 0] },
    { record: poolRec('reserved', { group: 'cal-group' }), embedding: [1, 0, 0] },
    { record: poolRec('evaldup'), embedding: [0.9, 0.5, 0.5] },
  ];
  const out = await retrieveFromSeeds({ head: 'h', seeds: [{ record: seed, embedding: [1, 0, 0] }], pool, evaluation: [[0.9, 0.5, 0.5]], k: 2, mmrLambda: 0.5, round: 1, reservedGroups: ['cal-group'] });
  assert.deepEqual(out.map((r) => r.id), ['a', 'b'], 'diversified, and never labelled, reserved, below-floor or evaluation near-duplicates');
  assert.ok(out.every((r) => r.role === 'train' && r.source.kind === 'retrieved' && r.source.round === 1 && r.source.seedIds[0] === 's1'));
  const v = (x: number[]) => { const n = Math.hypot(...x); return Float64Array.from(x, (e) => e / n); };
  assert.deepEqual(mmrSelect([{ v: v([1, 0]), sim: 0.99 }, { v: v([1, 0.01]), sim: 0.98 }, { v: v([0, 1]), sim: 0.7 }], 2, 0.5), [0, 2]);
  await assert.rejects(retrieveFromSeeds({ head: 'h', seeds: [{ record: seed, embedding: [1, 0] }], pool, evaluation: [], round: 1 }), /dimensions/);
  await assert.rejects(retrieveFromSeeds({ head: 'h', seeds: [{ record: seed, embedding: [1, 0, 0] }], pool, evaluation: [], round: 2 }), /round-2 seed s1 must be a confirmed round-1 retrieval/);
  await assert.rejects(retrieveFromSeeds({ head: 'h', seeds: [{ record: { ...seed, labels: { h: 0 } }, embedding: [1, 0, 0] }], pool, evaluation: [], round: 1 }), /must be a human-labelled h positive/);
  const viaSearch = await retrieveFromSeeds({ head: 'h', seeds: [{ record: seed, embedding: [1, 0, 0] }], pool, evaluation: [], k: 1, round: 1, search: async () => [{ id: 'b', score: 0 }] });
  assert.deepEqual(viaSearch.map((r) => r.id), ['b']);

  const confirmed = (i: number, y: 0 | 1, seedId: string): ExampleRecord => ({ ...poolRec(`r${seedId}${i}`), labels: { h: y }, labelledBy: 'human', source: { kind: 'retrieved', seedIds: [seedId], similarity: 0.8, round: 1 } });
  const stats = seedStats([...Array.from({ length: 10 }, (_, i) => confirmed(i, 0, 'weak')), ...Array.from({ length: 10 }, (_, i) => confirmed(i, i < 5 ? 1 : 0, 'good'))], 'h');
  assert.deepEqual(stats.map((s) => [s.seedId, s.retire]), [['good', false], ['weak', true]]);
});

test('hard negatives: real ones exclude evaluation records; batches accept at 96/100 and reject at 95/100', () => {
  const rec = (id: string, role: ExampleRecord['role'], y: 0 | 1): ExampleRecord => ({ id, text: `trigger ${id}`, group: id, role, labels: { h: y }, labelledBy: 'human', source: { kind: 'sampled', designId: 'd', stratum: 's', inclusionProb: 0.5, stratumSize: 2 } });
  const matcher = { match: (t: string) => (t.startsWith('trigger') ? { id: 'r1' } : null) };
  assert.deepEqual(realHardNegatives([rec('a', 'train', 0), rec('b', 'calibration', 0), rec('c', 'test', 0), rec('d', 'train', 1)], matcher, 'h'), ['a']);

  const batch: ExampleRecord[] = Array.from({ length: 300 }, (_, i) => ({ id: `g${i}`, text: `g${i}`, group: `g${i}`, role: 'train', labels: { h: 0 }, labelledBy: 'intended', source: { kind: 'generated', method: 'hard_negative', generator: 'gen', ruleId: 'r1', batchId: 'B' } }));
  const picked = selectForReview(batch, { seed: 1 });
  assert.equal(picked.length, 100, 'max(20% of 300, 100)');
  const reviews = (agree: number) => picked.map((r, i) => ({ id: r.id, label: (i < agree ? 0 : 1) as 0 | 1 }));
  const ok = verifyBatch(batch, reviews(96), { head: 'h' });
  assert.equal(ok.accepted, true);
  assert.ok(ok.lower >= 0.9);
  assert.equal(ok.records.length, 300 - 4, 'disagreeing items dropped');
  assert.ok(ok.records.filter((r) => !r.verified).every((r) => r.weight === ok.lower));
  const no = verifyBatch(batch, reviews(95), { head: 'h' });
  assert.equal(no.accepted, false);
  assert.equal(no.records.length, 0, 'a rejected batch is dropped whole');
  assert.throws(() => verifyBatch(batch, reviews(96), { head: 'h', safetyCritical: true }), /all 300 items must be reviewed/);
  assert.equal(selectForReview(batch, { seed: 1, all: true }).length, 300);

  assert.match((publishPlan({ gatesPassed: true, promote: true, generated: true }) as { error: string }).error, /without acceptance evidence/);
  assert.deepEqual(publishPlan({ gatesPassed: true, promote: true, generated: true, acceptanceEvidence: { report: 'shadow-2026-10' } }), { role: 'promoted' });
  assert.deepEqual(publishPlan({ gatesPassed: true, shadowCandidate: true, generated: true }), { role: 'shadow-candidate' });
});

test('coverage report: pair gaps, slice requirements, observable slices only', () => {
  const axes = defineAxes({ channel: { values: ['app', 'web'], observable: true }, subtype: { values: ['a', 'b'], observable: false } });
  const records: ExampleRecord[] = [];
  const tags: Record<string, Record<string, string>> = {};
  for (let i = 0; i < 80; i++) {
    const id = `c${i}`;
    records.push({ id, text: id, group: id, role: 'calibration', labels: { h: i < 40 ? 1 : 0 }, labelledBy: 'human', source: { kind: 'traffic' } });
    tags[id] = { channel: i % 4 === 0 ? 'web' : 'app', subtype: 'a' };
  }
  const report = coverageReport({ head: 'h', records, tags, axes });
  assert.equal(report.values.channel.app.calibration!.positives, 30);
  assert.equal(report.values.channel.web.calibration!.positives, 10);
  assert.ok(report.gaps.some((g) => g.a === 'channel=web' && g.b === 'subtype=b' && g.positives === 0));
  assert.ok(!report.gaps.some((g) => g.a === 'channel=app' && g.b === 'subtype=a'), 'app × a has 30 positives and 30 negatives');
  const app = report.slices.find((s) => s.value === 'app')!;
  assert.deepEqual([app.needed, app.available, app.shortfall], [59, 30, 29], '⌈ln δ / ln(1 - α)⌉ = 59 at α = δ = 0.05');
  assert.ok(!report.slices.some((s) => s.axis === 'subtype'), 'non-observable axes are not slices');
  assert.throws(() => checkSliceAxes(axes, ['subtype']), /not observable/);
  assert.throws(() => coverageReport({ head: 'h', records, tags: { c0: { channel: 'fax' } }, axes }), /not a value/);
  const md = reportMarkdown({ version: 'v', created_at: '', embedding: { model_id: 'm', dimensions: 1, normalize: true }, heads: {} }, { coverage: report });
  assert.match(md, /## Coverage: h/);
});

/** Population A from the allocation study: 5% positives, and the low band holds ~13% of them at ~1%. */
function rarePositiveFrame(N: number, seed: number) {
  const rand = seededRandom(seed);
  return Array.from({ length: N }, (_, i) => {
    const truth: 0 | 1 = rand() < 0.05 ? 1 : 0;
    const u = rand();
    return { id: `a${i}`, text: '', group: `a${i}`, signals: { ruleFires: false, score: truth ? 1 - (1 - u) ** 2.5 : u ** 2.5 }, truth };
  });
}

test('expected-positives allocation: proportional shares, sized so every material stratum expects enough positives', () => {
  const prior = { 'no_rule|band0': { positives: 54, labelled: 420 }, 'no_rule|band1': { positives: 20, labelled: 400 }, 'no_rule|band2': { positives: 15, labelled: 1180 } };
  const frameA = rarePositiveFrame(30000, 11);
  const opts = (total: number, extra: object = {}) => ({ frame: frameA, scoreBands: [0.79, 0.59], scoringModel: 'm', seed: 1, allocation: { total, method: 'expected-positives' as const, prior, ...extra } });
  assert.throws(() => designSample(opts(2000)), /below the (\d+) that proportional allocation needs for every stratum holding >= 0.02 of positives to expect 10 calibration positives/);
  const required = Number(/the (\d+) that/.exec((() => { try { designSample(opts(1)); return ''; } catch (e) { return (e as Error).message; } })())![1]);
  // Band 2's share of the frame and its prior rate (15.5 / 1181) set the size: 10 / (0.25 · rate · N2/N).
  const N2 = designSample({ ...opts(required), allocation: { total: required, method: 'proportional' } }).design.strata.find((s) => s.name === 'no_rule|band2')!.N;
  assert.equal(required, Math.ceil((10 * 30000) / (0.25 * (15.5 / 1181) * N2)));
  const d = designSample(opts(required));
  for (const s of d.design.strata) assert.ok(Math.abs(s.n / s.N - required / 30000) < 0.002, `${s.name}: proportional share`);
  assert.ok(d.design.strata.every((s) => s.expectedCalibrationPositives! >= 9.5), JSON.stringify(d.design.strata));
  assert.deepEqual(d.design.allocation.minExpectedPositives, 10);
  // A stratum below minShare doesn't drive the size.
  assert.ok(Number(/the (\d+) that/.exec((() => { try { designSample(opts(1, { minShare: 0.2 })); return ''; } catch (e) { return (e as Error).message; } })())![1]) < required);
  assert.throws(() => designSample(opts(9000, { prior: { 'no_rule|band0': 0.1 } })), /prior has no rate for stratum no_rule\|band1/);
  assert.throws(() => designSample(opts(9000, { prior: { ...prior, nope: 0.1 } })), /prior names unknown stratum nope/);
  assert.throws(() => designSample(opts(9000, { prior: { ...prior, 'no_rule|band0': { positives: 5, labelled: 2 } } })), /integer 0 <= positives <= labelled/);
});

test('allocation coverage: high-score-heavy sampling under-covers; sized proportional allocation holds', () => {
  const frameA = rarePositiveFrame(30000, 11);
  const byIdA = new Map(frameA.map((f) => [f.id, f]));
  const pos = frameA.filter((f) => f.truth).map((f) => f.signals.score).sort((a, b) => a - b);
  const trueMiss = (t: number) => pos.filter((v) => v < t).length / pos.length;
  const prior = { 'no_rule|band0': { positives: 54, labelled: 420 }, 'no_rule|band1': { positives: 20, labelled: 400 }, 'no_rule|band2': { positives: 15, labelled: 1180 } };
  const run = (allocation: object, runs: number) => {
    let feasible = 0, fails = 0;
    for (let r = 0; r < runs; r++) {
      const cal = designSample({ frame: frameA, scoreBands: [0.79, 0.59], scoringModel: 'm', seed: 1000 + r, allocation: allocation as never }).records.filter((x) => x.role === 'calibration');
      const res = designRiskThreshold({ ...designOf(cal), y: cal.map((x) => byIdA.get(x.id)!.truth), scores: cal.map((x) => byIdA.get(x.id)!.signals.score), alpha: 0.1, delta: 0.05, method: 'linearised' });
      if (!res.feasible) continue;
      feasible++;
      if (trueMiss(res.threshold) > 0.1) fails++;
    }
    return { feasible, fails, runs };
  };
  const heavy = run({ total: 2000, method: 'manual', manual: { 'no_rule|band0': 1000, 'no_rule|band1': 600, 'no_rule|band2': 400 } }, 80);
  assert.ok(heavy.fails > 0.2 * heavy.runs, `high-score-heavy: ${heavy.fails} failures in ${heavy.feasible} feasible of ${heavy.runs}`);
  const sized = run({ total: 5200, method: 'expected-positives', prior }, 80);
  assert.ok(sized.feasible >= 0.9 * sized.runs, `sized: feasible ${sized.feasible}/${sized.runs}`);
  assert.ok(sized.fails <= 0.05 * sized.runs + 3 * Math.sqrt(sized.runs * 0.05 * 0.95), `sized: ${sized.fails} failures in ${sized.runs}`);
});

test('small strata are merged before sampling: neighbouring band first, then across rule firing; estimates stay unbiased', () => {
  const base = frame.slice(0, 2000).map((f) => ({ ...f, signals: { ...f.signals, ruleFires: false } }));
  // Three rule-firing items with top scores: a stratum of 3 can't give 2 calibration and 2 test items.
  const tiny = [0.999, 0.998, 0.997].map((score, i): FrameItem => ({ id: `t${i}`, text: 't', group: `t${i}`, signals: { ruleFires: true, score } }));
  const d = designSample({ frame: [...base, ...tiny], scoreBands: [0.95, 0.8], allocation: { total: 600, method: 'proportional' }, scoringModel: 'm', seed: 2 });
  assert.deepEqual(d.design.merged, [{ name: 'no_rule|band0+rule|band0', from: ['no_rule|band0', 'rule|band0'] }], 'no other rule-firing cell exists, so it joins the same band');
  assert.ok(d.design.strata.every((s) => s.roles.calibration >= 2 && s.roles.test >= 2));
  // Uniform scores; rule fires on 140 items of the middle band and the 3 top items: the small
  // rule|band0 joins its neighbouring band with the same rule firing.
  const uniform = Array.from({ length: 1000 }, (_, i): FrameItem => ({ id: `u${i}`, text: 'u', group: `u${i}`, signals: { ruleFires: i >= 830 && i < 940, score: i / 1000 } }));
  const d2 = designSample({ frame: [...uniform, ...tiny], scoreBands: [0.95, 0.8], allocation: { total: 400, method: 'proportional' }, scoringModel: 'm', seed: 2 });
  assert.deepEqual(d2.design.merged, [{ name: 'rule|band0-1', from: ['rule|band0', 'rule|band1'] }]);
  assert.throws(() => designSample({ frame: tiny, scoreBands: [0.95], allocation: { total: 3, method: 'proportional' }, scoringModel: 'm', seed: 1 }), /needs at least 8|a design needs at least/);

  // Merging is decided from frame counts only, so HT estimates remain unbiased.
  const withTiny = [...base, ...tiny.map((t) => ({ ...t, x: [0], truth: 1 as const }))] as Array<FrameItem & { truth: 0 | 1 }>;
  const truth = new Map(withTiny.map((f) => [f.id, (f as { truth?: 0 | 1 }).truth ?? byId.get(f.id)!.truth]));
  const prev = withTiny.filter((f) => truth.get(f.id)).length / withTiny.length;
  let est = 0;
  const runs = 300;
  for (let r = 0; r < runs; r++) {
    const test = designSample({ frame: withTiny, scoreBands: [0.95, 0.8], allocation: { total: 400, method: 'proportional' }, scoringModel: 'm', seed: 50 + r }).records.filter((x) => x.role === 'test');
    const dd = designOf(test);
    const y = test.map((x) => truth.get(x.id)!);
    est += htTotal(y, dd.inclusionProbs) / htTotal(y.map(() => 1), dd.inclusionProbs);
  }
  assert.ok(Math.abs(est / runs - prev) < 0.004, `prevalence ${est / runs} vs ${prev}`);
});

test('trainHeads: a precision head in design mode states a precision guarantee from the sampled calibration set', () => {
  const sampled = design(7, 3000).records;
  const { records } = label(sampled);
  const X = records.map((r) => byId.get(r.id)!.x);
  const split = records.map((r) => r.role as Split);
  const y = records.map((r) => r.labels.h as 0 | 1);
  const r = trainHeads({ X, split, records, heads: [{ name: 'h', y, policy: { kind: 'precision', targetPrecision: 0.6, mode: 'design' } }], seed: 1 });
  const g = r.heads.h!.guarantee!;
  assert.deepEqual([g.mode, g.kind, g.metric, g.method], ['design', 'design-approximate', 'precision', 'linearised'], r.failures.join('; '));
  assert.ok(Math.abs(g.alpha - 0.4) < 1e-12);
  assert.deepEqual(r.failures, [], r.failures.join('; '));
  const ev = r.evaluation.h!;
  assert.ok(ev.design!.precision!.ci95[1] >= 0.6, 'the test split does not contradict the guarantee');
  const report = reportMarkdown({ version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: r.heads, evaluation: r.evaluation });
  assert.match(report, /precision ≥ 0\.600 with probability 0\.950 \(approximate, from a stratified sample\)/);
  assert.throws(() => trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.06, policy: { kind: 'precision', targetPrecision: 0.6, mode: 'design' } }] }), /needs sampled calibration records/);
});

test('slice gate: fails a slice demonstrably below target, warns when it cannot be confirmed or is too small', async () => {
  const { evaluateHead, gateHead } = await import('../src/index.ts');
  const rand = seededRandom(4);
  const build = (recallB: number, nPos: number, extraSlice = 0) => {
    const p: number[] = [], y: number[] = [], slice: string[] = [];
    for (let i = 0; i < nPos; i++) { const b = rand() < 0.3; y.push(1); slice.push(b ? 'B' : 'A'); p.push(rand() < (b ? recallB : 0.95) ? 0.9 : 0.1); }
    for (let i = 0; i < extraSlice; i++) { y.push(1); slice.push('C'); p.push(0.9); }
    for (let i = 0; i < nPos * 5; i++) { y.push(0); slice.push(rand() < 0.3 ? 'B' : 'A'); p.push(rand() < 0.02 ? 0.9 : 0.05); }
    return evaluateHead({ p, y, w: p.map(() => 1), threshold: 0.5, slices: { src: slice } });
  };
  const policy = { kind: 'recall' as const, targetRecall: 0.9, designRecall: 0.9, sliceGate: { minPositives: 10 } };
  const bad = gateHead(policy, build(0.8, 1000), { maxEce: 1 });
  assert.ok(bad.failures.some((f) => /^slice src=B: recall 0\.\d+ is below the target 0\.9/.test(f)), bad.failures.join('; '));
  assert.ok(!bad.failures.some((f) => f.startsWith('test recall')), 'overall recall still passes - only the slice gate sees it');
  const fine = gateHead(policy, build(0.95, 1000, 4), { maxEce: 1 });
  assert.ok(!fine.failures.some((f) => f.startsWith('slice')), fine.failures.join('; '));
  assert.ok(fine.warnings.some((w) => /slice src=C: only 4\.0 positives - too few to judge/.test(w)));
  assert.ok(!gateHead({ ...policy, sliceGate: { fields: ['other'] } }, build(0.8, 1000), { maxEce: 1 }).failures.some((f) => f.startsWith('slice')), 'only the listed fields');

  // Design-based slices on a sampled test set: same gate, effective positives.
  const strata = Array.from({ length: 2000 }, (_, i) => (i % 2 ? 'h1' : 'h2'));
  const yD = strata.map((_, i) => (i % 5 === 0 ? 1 : 0));
  const sliceD = strata.map((_, i) => (i % 3 === 0 ? 'B' : 'A'));
  const pD = yD.map((v, i) => (v && (sliceD[i] === 'A' || i % 4 !== 0) ? 0.9 : 0.1));
  const evD = evaluateHead({ p: pD, y: yD, w: pD.map(() => 4), threshold: 0.5, slices: { src: sliceD }, design: { inclusionProbs: pD.map(() => 0.25), strata, stratumSizes: { h1: 4000, h2: 4000 }, replicates: 0 } });
  assert.ok(evD.design!.slices['src=B'].effective_n! > 100);
  const gD = gateHead(policy, evD, { maxEce: 1 });
  assert.ok(gD.failures.some((f) => f.startsWith('slice src=B')), gD.failures.join('; '));
});

test('calibration gate: Cox test fails overconfident rare-class probabilities that ECE lets through', async () => {
  const { evaluateHead, gateHead } = await import('../src/index.ts');
  const rand = seededRandom(7);
  const sig = (x: number) => 1 / (1 + Math.exp(-x));
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-300)) * Math.cos(2 * Math.PI * rand());
  const sample = (slope: number) => {
    const p: number[] = [], y: number[] = [];
    for (let i = 0; i < 4000; i++) { const z = -4.5 + 1.5 * gauss(); y.push(rand() < sig(z) ? 1 : 0); p.push(sig(slope * (z + 4.5) - 4.5)); }
    return evaluateHead({ p, y, w: p.map(() => 1), threshold: 0.5 });
  };
  const policy = { kind: 'precision' as const, targetPrecision: 0.01 };
  const calibrated = sample(1), overconfident = sample(1.4);
  assert.ok(calibrated.calibration_test!.p_value > 0.01);
  assert.ok(Math.abs(calibrated.calibration_test!.slope - 1) < 0.2);
  assert.ok(overconfident.ece_prevalence_weighted <= 0.05, 'ECE stays under its gate');
  assert.ok(!gateHead(policy, overconfident).failures.some((f) => f.startsWith('ECE')));
  assert.ok(gateHead(policy, overconfident, { calibrationAlpha: 0.01 }).failures.some((f) => /miscalibrated \(Cox test/.test(f)));
  assert.ok(!gateHead(policy, calibrated, { calibrationAlpha: 0.01 }).failures.some((f) => /miscalibrated/.test(f)));
});

test('sizing for several heads, without parsing errors: requiredSampleSize and priors', async () => {
  const { requiredSampleSize } = await import('../src/index.ts');
  const frameA = rarePositiveFrame(30000, 11);
  const prior = { 'no_rule|band0': { positives: 54, labelled: 420 }, 'no_rule|band1': { positives: 20, labelled: 400 }, 'no_rule|band2': { positives: 15, labelled: 1180 } };
  const rarer = { 'no_rule|band0': { positives: 20, labelled: 420 }, 'no_rule|band1': { positives: 4, labelled: 400 }, 'no_rule|band2': { positives: 9, labelled: 1180 } };
  const base = { frame: frameA, scoreBands: [0.79, 0.59], scoringModel: 'm' };
  const one = requiredSampleSize({ ...base, allocation: { prior } });
  const both = requiredSampleSize({ ...base, allocation: { priors: { a: prior, b: rarer } } });
  assert.ok(one > 0 && both > one, `${one} vs ${both}: the rarer head needs more`);
  assert.equal(both, requiredSampleSize({ ...base, allocation: { prior: rarer } }), 'the total satisfies every head');
  const d = designSample({ ...base, seed: 1, allocation: { total: both, method: 'expected-positives', priors: { a: prior, b: rarer } } });
  assert.equal(d.design.allocation.requiredTotal, both);
  assert.deepEqual(Object.keys(d.design.allocation.priorRatesByHead!), ['a', 'b']);
  assert.throws(() => designSample({ ...base, seed: 1, allocation: { total: both, method: 'expected-positives', prior, priors: { a: prior } } }), /exactly one of prior or priors/);
  // A stratum with no positives in the prior round doesn't count as material, however few were labelled.
  const zeroLow = { ...prior, 'no_rule|band2': { positives: 0, labelled: 30 } };
  const zeroLowMore = { ...prior, 'no_rule|band2': { positives: 0, labelled: 3000 } };
  assert.equal(requiredSampleSize({ ...base, allocation: { prior: zeroLow } }), requiredSampleSize({ ...base, allocation: { prior: zeroLowMore } }));
});

test('design intervals do not collapse at 0 or 1, and a disabled head gets one clear failure', async () => {
  const { evaluateHead, gateHead } = await import('../src/index.ts');
  const n = 400;
  const strata = Array.from({ length: n }, (_, i) => (i % 2 ? 'h1' : 'h2'));
  const y = strata.map((_, i) => (i % 10 === 0 ? 1 : 0));
  const p = y.map((v) => (v ? 0.9 : 0.1)); // every positive caught
  const ev = evaluateHead({ p, y, w: p.map(() => 10), threshold: 0.5, design: { inclusionProbs: p.map(() => 0.1), strata, stratumSizes: { h1: 2000, h2: 2000 }, replicates: 0 } });
  assert.equal(ev.design!.recall.estimate, 1);
  assert.equal(ev.design!.recall.se, 0);
  assert.ok(ev.design!.recall.ci95[0] < 0.95, `exact floor at ${ev.design!.recall.effective_n} effective positives: ${ev.design!.recall.ci95}`);
  assert.ok(ev.certified.recall_lower < 0.95);
  const disabled = gateHead({ kind: 'precision', targetPrecision: 0.9 }, { ...ev, threshold: 1.0000000000000002 });
  assert.deepEqual(disabled.failures, ['disabled: no threshold met its policy, so the head never fires']);
});

test('design precision: the strictest candidate is placed so calibration can fire enough', () => {
  const sampled = design(7, 3000).records;
  const { records } = label(sampled);
  const X = records.map((r) => byId.get(r.id)!.x);
  const split = records.map((r) => r.role as Split);
  const y = records.map((r) => r.labels.h as 0 | 1);
  const run = (designMinFired?: number) => trainHeads({ X, split, records, seed: 1, heads: [{ name: 'h', y, policy: { kind: 'precision', targetPrecision: 0.7, mode: 'design', designMinFired } }] });
  const r = run();
  assert.deepEqual(r.failures, [], r.failures.join('; '));
  assert.equal(r.heads.h!.guarantee!.kind, 'design-approximate');
});

test('calibration gate: an unstable recalibration fit (near-separation) warns instead of gating', async () => {
  const { evaluateHead, gateHead } = await import('../src/index.ts');
  const y = Array.from({ length: 400 }, (_, i) => (i % 20 === 0 ? 1 : 0));
  const p = y.map((v, i) => (v ? 0.999 - (i % 7) * 1e-4 : 0.001 + (i % 5) * 1e-4)); // the classes never overlap
  const ev = evaluateHead({ p, y, w: p.map(() => 1), threshold: 0.5 });
  assert.equal(ev.calibration_test!.stable, false, JSON.stringify(ev.calibration_test));
  const g = gateHead({ kind: 'precision', targetPrecision: 0.5 }, ev, { calibrationAlpha: 0.01 });
  assert.ok(!g.failures.some((f) => /miscalibrated/.test(f)));
  assert.ok(g.warnings.some((w) => /calibration test not applied: the recalibration fit is unstable/.test(w)));
});

test('combined recall with a baseline is design-weighted on a sampled test set', async () => {
  const { evaluateHead } = await import('../src/index.ts');
  // Stratum h1 is sampled at 10x the rate of h2; the baseline only catches h1 positives.
  const strata = [...Array(200).fill('h1'), ...Array(200).fill('h2')];
  const y = strata.map((_, i) => (i % 4 === 0 ? 1 : 0));
  const p = y.map(() => 0.1); // the classifier catches nothing
  const baseline = strata.map((s, i) => s === 'h1' && y[i] === 1);
  const ev = evaluateHead({ p, y, w: strata.map((s) => (s === 'h1' ? 1 : 10)), threshold: 0.5, baseline, design: { inclusionProbs: strata.map((s) => (s === 'h1' ? 1 : 0.1)), strata, stratumSizes: { h1: 200, h2: 2000 }, replicates: 0 } });
  assert.equal(ev.vs_baseline!.caught_by_baseline_only, 50, 'raw counts stay counts');
  assert.ok(Math.abs(ev.combined_recall! - 50 / 550) < 1e-12, `weighted: 50 of 550 population positives, not 50 of 100 sampled (${ev.combined_recall})`);
});

test('provenanceEmbeddings: the near-duplicate check uses real embeddings when X holds other features', () => {
  const sampled = design(7, 3000).records;
  const { records: labelledRecords } = label(sampled);
  // Add retrieved training records whose texts are unrelated to any evaluation record.
  const extra: ExampleRecord[] = frame.filter((f) => !labelledRecords.some((r) => r.group === f.group)).slice(0, 40).map((f) => ({
    id: f.id, text: f.text, group: f.group, role: 'train', labels: { h: f.truth }, labelledBy: 'human',
    source: { kind: 'retrieved', seedIds: ['s'], similarity: 0.7, round: 1 },
  }));
  const records = [...labelledRecords, ...extra];
  const embeddings = records.map((r) => byId.get(r.id)!.x);
  const score = records.map((r) => [byId.get(r.id)!.x[0] + byId.get(r.id)!.x[1]]); // one stacked feature
  const split = records.map((r) => r.role as Split);
  const y = records.map((r) => r.labels.h as 0 | 1);
  const heads = [{ name: 'h', y, policy: { kind: 'recall' as const, targetRecall: 0.8, mode: 'design' as const, designMethod: 'linearised' as const } }];
  const naive = trainHeads({ X: score, split, records, heads, seed: 1 });
  assert.ok(naive.provenance!.dropped.length >= 40, `one-column X: cosine is ±1, so ${naive.provenance!.dropped.length} records look like near-duplicates`);
  const fixed = trainHeads({ X: score, split, records, heads, seed: 1, provenanceEmbeddings: { X: embeddings } });
  const direct = validateProvenance(records, { embeddings });
  assert.deepEqual(fixed.provenance!.dropped.map((d) => d.id).sort(), direct.dropped.map((d) => d.id).sort(), 'the same decisions as on the real embeddings');
  assert.ok(fixed.provenance!.dropped.length < naive.provenance!.dropped.length);
  assert.throws(() => trainHeads({ X: score, split, records, heads, seed: 1, provenanceEmbeddings: { X: embeddings.slice(1) } }), /provenanceEmbeddings.X has/);
});
