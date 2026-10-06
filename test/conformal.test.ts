import assert from 'node:assert/strict';
import { test } from 'node:test';

import { conformalLowerThreshold, conformalUpperThreshold, minimumSamples } from '@liquidau/solvers';

import {
  conformalThreshold,
  evaluateHead,
  gateHead,
  groupScores,
  pickThreshold,
  reportMarkdown,
  trainHeads,
  validateArtifact,
  type ClassifierArtifact,
  type HeadPolicy,
  type Split,
} from '../src/index.ts';

function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}

test('groupScores: one calibration score per group, conservatively', () => {
  assert.deepEqual(groupScores([0.9, 0.1, 0.5], ['a', 'a', 'b'], Math.min).sort(), [0.1, 0.5]);
  assert.deepEqual(groupScores([0.9, 0.1, 0.5], ['a', 'a', 'b'], Math.max).sort(), [0.5, 0.9]);
  assert.deepEqual(groupScores([0.9, 0.1], undefined, Math.min), [0.9, 0.1]);
});

test('the heuristic threshold falls short far more often than δ; the PAC one does not', () => {
  // Uniform scores: a threshold t misses exactly t of positives. (Coverage of the solvers is tested there.)
  const rand = rng(42);
  const runs = 400, n = 150, alpha = 0.05, delta = 0.05;
  let pacFails = 0, heuristicFails = 0;
  for (let r = 0; r < runs; r++) {
    const scores = Array.from({ length: n }, rand);
    if (conformalLowerThreshold(scores, alpha, delta)! > alpha) pacFails++;
    if (pickThreshold({ kind: 'recall', targetRecall: 0.95, designRecall: 0.95 }, scores, scores.map(() => 1)) > alpha) heuristicFails++;
  }
  assert.ok(pacFails / runs <= delta + 3 * Math.sqrt((delta * (1 - delta)) / runs), `PAC failed in ${pacFails}/${runs} runs`);
  assert.ok(heuristicFails / runs > 0.3, `an unguarded 95% calibration-recall threshold falls short in ${heuristicFails}/${runs} runs`);
});

test('modes: auto takes the strongest guarantee the data supports and never trades one away silently', () => {
  const rand = rng(7);
  const positives = (n: number) => Array.from({ length: n }, () => 0.5 + rand() / 2);
  const negatives = Array.from({ length: 2000 }, () => rand() / 2);
  const base = { targetRecall: 0.95, heuristicAvailable: true } as const;

  const pac = conformalThreshold({ ...base, mode: 'auto', positives: positives(200) });
  assert.equal(pac.guarantee.kind, 'pac');
  assert.equal(pac.sufficiency.chosen, 'conformal-pac');
  assert.deepEqual(pac.sufficiency.feasible, ['conformal-expected', 'conformal-pac']);
  assert.deepEqual(pac.failures, []);

  const expected = conformalThreshold({ ...base, mode: 'auto', positives: positives(30) });
  assert.equal(expected.guarantee.kind, 'expected');
  assert.ok(expected.warnings.some((w) => /inconclusive: 30 calibration positive groups, 59 needed/.test(w)));

  const none = conformalThreshold({ ...base, mode: 'auto', positives: positives(10) });
  assert.equal(none.threshold, null, 'fall back to the heuristic threshold');
  assert.equal(none.sufficiency.chosen, 'heuristic');
  assert.equal(conformalThreshold({ ...base, heuristicAvailable: false, mode: 'auto', positives: positives(10) }).failures.length, 1);
  const strict = conformalThreshold({ ...base, allowHeuristic: false, mode: 'auto', positives: positives(10) });
  assert.equal(strict.guarantee.kind, 'none');
  assert.notEqual(strict.threshold, null, 'no heuristic threshold');
  assert.match(strict.failures[0], /auto: 10 calibration positive groups, 19 needed .* allowHeuristicFallback is false/);
  assert.equal(conformalThreshold({ ...base, allowHeuristic: false, mode: 'auto', positives: positives(30) }).guarantee.kind, 'expected', 'the PAC -> expected fallback is unaffected');

  const fa = { name: 'calibration false-alarm', scores: negatives, maxRate: 0.01 };
  const fine = conformalThreshold({ ...base, mode: 'conformal-pac', positives: positives(200), constraints: [fa] });
  assert.equal(fine.guarantee.kind, 'pac');
  assert.equal(fine.guarantee.false_alarm, 0.01);
  assert.ok(fine.sufficiency.needed.pac > minimumSamples(0.05, 0.05), 'δ is split between recall and the false-alarm budget');

  const overlapping = Array.from({ length: 2000 }, () => 0.4 + rand() * 0.3);
  const clash = conformalThreshold({ ...base, mode: 'auto', positives: positives(200), constraints: [{ ...fa, scores: overlapping }] });
  assert.equal(clash.guarantee.kind, 'none');
  assert.match(clash.failures[0], /auto: recall >= 0.95 needs a threshold <= .* but calibration false-alarm <= 0.01 needs >= .* choose a mode explicitly/);
  assert.ok(clash.threshold! >= conformalUpperThreshold(overlapping, 0.01, 0.025)!, 'the shipped threshold still honours the false-alarm budget');

  const short = conformalThreshold({ ...base, mode: 'conformal-pac', positives: positives(30) });
  assert.match(short.failures[0], /conformal-pac: 30 calibration positive groups, 59 needed/);
});

test('certified bounds on the test split: exact, group-aware, and the conformal recall gate checks for contradiction', () => {
  const p = [0.9, 0.8, 0.1, 0.9, 0.2, 0.1, 0.1, 0.6];
  const y = [1, 1, 1, 1, 0, 0, 0, 0];
  const ev = evaluateHead({ p, y, w: p.map(() => 1), threshold: 0.5, groups: ['a', 'a', 'b', 'c', 'n1', 'n2', 'n3', 'n4'], prevalence: 0.05 });
  assert.equal(ev.certified.positive_groups, 3);
  assert.equal(ev.certified.negative_groups, 4);
  assert.ok(ev.certified.recall_lower < 2 / 3 && 2 / 3 < ev.certified.recall_upper);
  assert.ok(ev.certified.false_alarm_upper > 0.25);
  assert.ok(ev.certified.precision_lower! > 0 && ev.certified.precision_lower! < 1);

  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.9, mode: 'conformal-pac' };
  const guaranteed = { ...ev, guarantee: { mode: 'conformal-pac' as const, kind: 'pac' as const, alpha: 0.1, delta: 0.05 } };
  assert.ok(!gateHead(policy, guaranteed).failures.some((f) => f.includes('recall')), 'recall 0.75 on 3 groups does not contradict a 90% guarantee');
  assert.ok(gateHead(policy, ev).failures.some((f) => f.startsWith('test recall 0.750 < target 0.9')), 'without a guarantee the point gate applies');
  const contradicted = { ...guaranteed, certified: { ...ev.certified, recall_upper: 0.8 } };
  assert.ok(gateHead(policy, contradicted).failures.some((f) => /contradicts the pac guarantee/.test(f)));
});

/** Two separable clusters in 8 dims with a little label noise (as in training.test.ts). */
function dataset(n: number, seed = 1) {
  const rand = rng(seed);
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

test('trainHeads: conformal modes, stored guarantee, conformal review floor and certified bounds', () => {
  const { X, y, split } = dataset(2400);
  const background = { X: X.filter((_, i) => y[i] === 0 && split[i] === 'train').slice(0, 800), maxRate: { h: 0.05 } };
  // 2% label noise puts some "positives" deep in the negative cluster: 80% is certifiable within the budget, 90% is not.
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.8, mode: 'conformal-pac', delta: 0.05 };
  const r = trainHeads({ X, split, background, reviewEpsilon: 0.02, slices: { third: X.map((_, i) => String(i % 3)) }, heads: [{ name: 'h', y, prevalence: 0.05, policy }] });
  const spec = r.heads.h!, ev = r.evaluation.h!;
  assert.equal(spec.guarantee?.kind, 'pac');
  assert.equal(spec.guarantee?.background_rate, 0.05);
  assert.deepEqual(r.failures, [], r.failures.join('; '));
  assert.equal(ev.sufficiency?.chosen, 'conformal-pac');
  assert.deepEqual(Object.keys(ev.sufficiency?.slices ?? {}), ['third=0', 'third=1', 'third=2']);
  assert.ok(ev.background_rate! <= ev.background_rate_upper!);
  assert.ok(spec.review_floor <= spec.threshold);
  assert.equal(spec.review_epsilon, 0.02);

  const artifact: ClassifierArtifact<'h'> = { version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: r.heads, evaluation: r.evaluation };
  validateArtifact(JSON.parse(JSON.stringify(artifact)));
  const report = reportMarkdown(artifact);
  assert.match(report, /Threshold mode \*\*conformal-pac\*\* \(used: conformal-pac\): guarantees recall ≥ 0.800 with probability 0.950, with background rate ≤ 0.05/);
  assert.match(report, /certified recall ≥ \(test, exact\)/);
  assert.match(report, /Review floor is conformal/);

  const bad = (guarantee: unknown) => () => validateArtifact({ ...artifact, heads: { h: { ...spec, guarantee } } });
  assert.throws(bad({ ...spec.guarantee, mode: 'heuristic' }), /mode heuristic cannot give a pac guarantee/);
  assert.throws(bad({ ...spec.guarantee, delta: undefined }), /a pac guarantee needs delta/);
  assert.throws(bad({ ...spec.guarantee, mode: 'magic' }), /unknown threshold mode magic/);

  const heuristic = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.9, designRecall: 0.95, mode: 'heuristic' } }] });
  assert.equal(heuristic.heads.h!.guarantee?.kind, 'none', 'a heuristic threshold states that it guarantees nothing');
  const c = heuristic.evaluation.h!.certified;
  assert.ok(c.recall_lower <= heuristic.evaluation.h!.recall && heuristic.evaluation.h!.recall <= c.recall_upper);

  const clash = trainHeads({ X, split, background, heads: [{ name: 'h', y, prevalence: 0.05, policy: { ...policy, targetRecall: 0.9, mode: 'auto' } }] });
  assert.ok(clash.failures.some((f) => /^h: auto: .*choose a mode explicitly/.test(f)), clash.failures.join('; '));
  assert.throws(() => trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.9, mode: 'heuristic' } }] }), /designRecall is required/);
  // The default is the strongest guarantee the data supports (here conformal: no sampled records), not the heuristic.
  const byDefault = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.9 } }] });
  assert.equal(byDefault.heads.h!.guarantee?.mode, 'auto');
  assert.notEqual(byDefault.heads.h!.guarantee?.kind, 'none');

  // Too little data for any guarantee: keep only a handful of calibration positives.
  const few = y.map((v, i) => (split[i] === 'calibration' && v === 1 && i % 120 !== 2 ? null : v));
  const tiny = (allowHeuristicFallback?: boolean) => trainHeads({ X, split, heads: [{ name: 'h', y: few, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.9, designRecall: 0.95, mode: 'auto', allowHeuristicFallback } }] });
  const allowed = tiny();
  assert.equal(allowed.evaluation.h!.sufficiency?.chosen, 'heuristic');
  assert.ok(allowed.warnings.some((w) => /^h: guarantee inconclusive/.test(w)) && !allowed.failures.some((f) => /allowHeuristicFallback/.test(f)), 'allowed by default');
  assert.ok(tiny(false).failures.some((f) => /^h: auto: .*allowHeuristicFallback is false/.test(f)));
});

test("fallback: 'heuristic' per head: a guarantee when the labels support one, a recorded heuristic threshold when they don't", () => {
  const { X, y, split } = dataset(2400);
  // A handful of calibration positives: no guarantee is possible.
  const few = y.map((v, i) => (split[i] === 'calibration' && v === 1 && i % 120 !== 2 ? null : v));
  const policy: HeadPolicy = { kind: 'recall', targetRecall: 0.9, designRecall: 0.95 };
  const strict = trainHeads({ X, split, heads: [{ name: 'h', y: few, prevalence: 0.05, policy }] });
  assert.ok(strict.failures.length > 0, 'by default a head the data cannot support fails');
  const lenient = trainHeads({ X, split, heads: [{ name: 'h', y: few, prevalence: 0.05, policy: { ...policy, fallback: 'heuristic' } }] });
  assert.deepEqual(lenient.failures.filter((f) => /auto|guarantee/.test(f)), []);
  assert.equal(lenient.heads.h!.guarantee?.kind, 'none');
  assert.equal(lenient.heads.h!.guarantee?.fallback, 'heuristic');
  // With enough positives the same policy keeps its guarantee.
  const enough = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { ...policy, fallback: 'heuristic' } }] });
  assert.notEqual(enough.heads.h!.guarantee?.kind, 'none');
  assert.equal(enough.heads.h!.guarantee?.fallback, undefined);
  // A recall fallback needs designRecall; precision without records may fall back too.
  assert.throws(() => trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.05, policy: { kind: 'recall', targetRecall: 0.9, fallback: 'heuristic' } }] }), /needs designRecall/);
  const p = trainHeads({ X, split, heads: [{ name: 'p', y, prevalence: 0.05, policy: { kind: 'precision', targetPrecision: 0.5, fallback: 'heuristic' } }] });
  assert.ok(!p.failures.some((f) => /no precision guarantee/.test(f)));
  assert.equal(p.heads.p!.guarantee?.fallback, 'heuristic');
  // The artifact records the fallback, and only with kind 'none'.
  const art = { version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 8, normalize: false }, heads: lenient.heads };
  validateArtifact(JSON.parse(JSON.stringify(art)));
  assert.throws(() => validateArtifact({ ...art, heads: { h: { ...lenient.heads.h!, guarantee: { ...lenient.heads.h!.guarantee!, kind: 'pac', delta: 0.05 } } } }), /fallback guarantee must be/);
});
