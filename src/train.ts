/**
 * Fit, calibrate, threshold, evaluate and gate every head - everything short of I/O.
 *
 *   train split        weighted L2 logistic regression per head (class-balanced, exact Newton solver),
 *                      plus any weak positives (e.g. from certified rules), weighted and capped
 *   calibration split  Platt or isotonic calibrator, weighted to the head's production prevalence,
 *                      then the threshold from the head's policy
 *   test split         evaluateHead + gateHead
 */
import { decisionFunction, fitIsotonic, fitLogistic, fitPlatt, prevalenceWeights, sigmoid } from '@liquidau/solvers';

import { calibrate, scoreEmbedding, validateArtifact, type Calibration, type ClassifierArtifact, type HeadSpec } from './artifact.ts';
import { evaluateHead, type HeadEvaluation } from './evaluate.ts';
import { gateHead } from './gates.ts';
import { budgetThreshold, pickThreshold, type HeadPolicy } from './threshold.ts';

export const SPLITS = ['train', 'calibration', 'test'] as const;
export type Split = (typeof SPLITS)[number];

export interface HeadInput<H extends string> {
  name: H;
  /** Binary target per example, aligned with X; null leaves the example out of this head. */
  y: ReadonlyArray<0 | 1 | null>;
  policy: HeadPolicy;
  /** Expected positive rate in production - calibration and precision are weighted to it. */
  prevalence: number;
  /** Whether an existing mechanism already catches each example (see evaluateHead). */
  baseline?: readonly boolean[];
  /**
   * Extra POSITIVES for the train split only, e.g. rule-miner's weakLabels(): embeddings with a
   * weight in [0, 1] each. They never reach calibration, test or the background budget, and an
   * embedding identical to a calibration, test or background row is refused as leakage.
   */
  weak?: WeakInput;
  /**
   * Cap on total weak weight as a multiple λ of the head's gold train positives (default 0.5): weak
   * weights are scaled down to fit. Class balancing is sample-weighted, so weak positives take a
   * share λ / (1 + λ) of the positive class's weight rather than adding to it.
   */
  maxWeakShare?: number;
}

export interface WeakInput {
  X: ReadonlyArray<ArrayLike<number>>;
  weights: readonly number[];
  /** Which rule produced each example, for the per-rule report. */
  rules?: readonly string[];
  /** The rule set the weak labels came from, recorded for audit. */
  source?: { rule_set_version: string; rule_set_hash: string };
}

/** What weak supervision contributed to one head - store as artifact.training.weak_labels. */
export interface WeakSummary {
  count: number;
  gold_train_positives: number;
  max_weak_share: number;
  weight_before_cap: number;
  /** After the cap: at most max_weak_share × gold_train_positives. */
  weight: number;
  /** Multiplier the cap applied to every weak weight (1 = no cap). */
  scale: number;
  by_rule?: Record<string, { count: number; weight: number }>;
  rule_set_version?: string;
  rule_set_hash?: string;
}

export interface TrainInput<H extends string> {
  X: ReadonlyArray<ArrayLike<number>>;
  split: readonly Split[];
  heads: ReadonlyArray<HeadInput<H>>;
  /** Inverse L2 strength (default 1). */
  C?: number;
  calibration?: 'platt' | 'isotonic';
  /** review_floor = threshold × reviewRatio (default 0.5). */
  reviewRatio?: number;
  maxEce?: number;
  groups?: readonly string[];
  slices?: Readonly<Record<string, readonly string[]>>;
  /**
   * Ordinary, mostly-negative traffic (e.g. everyday messages): each listed head's threshold is
   * raised until it fires on at most maxRate of it - BEFORE test evaluation, so the gates judge
   * the threshold that will ship.
   */
  background?: { X: ReadonlyArray<ArrayLike<number>>; maxRate: Partial<Record<H, number>> };
  log?: (line: string) => void;
}

export interface TrainResult<H extends string> {
  heads: Partial<Record<H, HeadSpec>>;
  evaluation: Partial<Record<H, HeadEvaluation>>;
  /** Gate failures, prefixed with the head name. Empty means every gate passed. */
  failures: string[];
  warnings: string[];
  /** Test-split probabilities per head (example indices into X), for round-trip checks. */
  testProbabilities: Partial<Record<H, { idx: number[]; p: number[] }>>;
  /** Heads trained with weak positives - store as artifact.training.weak_labels so a release can be audited. */
  weakLabels: Partial<Record<H, WeakSummary>>;
}

function checkLength(name: string, values: { length: number } | undefined, n: number): void {
  if (values !== undefined && values.length !== n) throw new Error(`${name} has ${values.length} entries but X has ${n}`);
}

function indicesBySplit(split: readonly Split[], y: ReadonlyArray<0 | 1 | null>): Record<Split, number[]> {
  const idx: Record<Split, number[]> = { train: [], calibration: [], test: [] };
  split.forEach((s, i) => {
    if (y[i] !== null) idx[s].push(i);
  });
  return idx;
}

export function trainHeads<H extends string>(input: TrainInput<H>): TrainResult<H> {
  const { X, split, C = 1, calibration: method = 'platt', reviewRatio = 0.5, maxEce, groups, slices, background, log = () => {} } = input;
  checkLength('split', split, X.length);
  split.forEach((s, i) => {
    if (!(SPLITS as readonly string[]).includes(s)) throw new Error(`split[${i}] is "${s}", expected one of ${SPLITS.join(', ')}`);
  });
  checkLength('groups', groups, X.length);
  for (const [field, values] of Object.entries(slices ?? {})) checkLength(`slices.${field}`, values, X.length);
  if (background && Object.keys(background.maxRate).length && background.X.length === 0) throw new Error('background.X must be non-empty when a maxRate is set');
  if (!(reviewRatio >= 0 && reviewRatio <= 1)) throw new Error('reviewRatio must be between 0 and 1');
  const result: TrainResult<H> = { heads: {}, evaluation: {}, failures: [], warnings: [], testProbabilities: {}, weakLabels: {} };
  const vectorKey = (x: ArrayLike<number>) => Array.from(x).join(',');
  let heldOut: Set<string> | null = null;
  const isHeldOut = (x: ArrayLike<number>) => {
    heldOut ??= new Set([...X.filter((_, i) => split[i] !== 'train'), ...(background?.X ?? [])].map(vectorKey));
    return heldOut.has(vectorKey(x));
  };

  for (const { name, y, policy, prevalence, baseline, weak, maxWeakShare = 0.5 } of input.heads) {
    checkLength(`${name}: y`, y, X.length);
    checkLength(`${name}: baseline`, baseline, X.length);
    if (!(prevalence > 0 && prevalence < 1)) throw new Error(`${name}: prevalence must be strictly between 0 and 1, got ${prevalence}`);
    if (weak) {
      checkLength(`${name}: weak.weights`, weak.weights, weak.X.length);
      checkLength(`${name}: weak.rules`, weak.rules, weak.X.length);
      if (!(maxWeakShare >= 0)) throw new Error(`${name}: maxWeakShare must be non-negative, got ${maxWeakShare}`);
      weak.X.forEach((x, j) => {
        if (x.length !== X[0]?.length) throw new Error(`${name}: weak.X[${j}] has ${x.length} dimensions but X has ${X[0]?.length}`);
        if (!(weak.weights[j] >= 0 && weak.weights[j] <= 1)) throw new Error(`${name}: weak.weights[${j}] must be in [0, 1], got ${weak.weights[j]}`);
        if (isHeldOut(x)) throw new Error(`${name}: weak.X[${j}] is a calibration, test or background example - weak labels are for the train split only`);
      });
    }
    const idx = indicesBySplit(split, y);
    for (const s of SPLITS) {
      if (new Set(idx[s].map((i) => y[i])).size < 2) throw new Error(`${name}: the ${s} split needs both positive and negative examples`);
    }
    const ySplit = (s: Split) => idx[s].map((i) => y[i] as number);
    log(`${name}: ` + SPLITS.map((s) => `${s} ${ySplit(s).reduce((a, b) => a + b, 0)}/${idx[s].length} positive`).join(', '));

    let model;
    if (weak && weak.X.length) {
      const gold = ySplit('train').reduce((a, b) => a + b, 0);
      const before = weak.weights.reduce((a, b) => a + b, 0);
      const scale = before > maxWeakShare * gold ? (maxWeakShare * gold) / before : 1;
      const w = weak.weights.map((v) => v * scale);
      model = fitLogistic([...idx.train.map((i) => X[i]), ...weak.X], [...ySplit('train'), ...w.map(() => 1)], {
        C, classWeight: 'balanced', sampleWeight: [...idx.train.map(() => 1), ...w],
      });
      const summary: WeakSummary = { count: w.length, gold_train_positives: gold, max_weak_share: maxWeakShare, weight_before_cap: before, weight: w.reduce((a, b) => a + b, 0), scale };
      if (weak.rules) {
        summary.by_rule = {};
        weak.rules.forEach((rule, j) => {
          const r = (summary.by_rule![rule] ??= { count: 0, weight: 0 });
          r.count++;
          r.weight += w[j];
        });
      }
      if (weak.source) Object.assign(summary, weak.source);
      result.weakLabels[name] = summary;
      log(`${name}: ${w.length} weak positives, weight ${summary.weight.toFixed(2)} (${scale < 1 ? `capped at ${maxWeakShare}` : 'uncapped'}) beside ${gold} gold`);
    } else {
      model = fitLogistic(idx.train.map((i) => X[i]), ySplit('train'), { C, classWeight: 'balanced' });
    }
    const score = (x: ArrayLike<number>) => calibrate(calibration, decisionFunction(model, x));

    const yCal = ySplit('calibration');
    const wCal = prevalenceWeights(yCal, prevalence);
    const calLogits = idx.calibration.map((i) => decisionFunction(model, X[i]));
    const calibration: Calibration =
      method === 'platt'
        ? { method: 'platt', ...fitPlatt(calLogits, yCal, wCal) }
        : { method: 'isotonic', ...fitIsotonic(calLogits.map(sigmoid), yCal, wCal, { yMin: 0, yMax: 1 }) };
    const pCal = calLogits.map((z) => calibrate(calibration, z));
    let threshold = pickThreshold(policy, pCal, yCal);
    if (policy.kind === 'precision' && !pCal.some((p) => p >= threshold)) {
      result.warnings.push(`${name}: no calibration example reaches the target precision ${threshold} - the head is unlikely ever to fire`);
    }
    const budget = background?.maxRate[name];
    const backgroundP = background && budget !== undefined ? background.X.map(score) : null;
    if (backgroundP && budget !== undefined) threshold = budgetThreshold(threshold, backgroundP, budget);

    const yTest = ySplit('test');
    const pTest = idx.test.map((i) => score(X[i]));
    const atTest = <T>(values: readonly T[]) => idx.test.map((i) => values[i]);
    const ev = evaluateHead({
      p: pTest,
      y: yTest,
      w: prevalenceWeights(yTest, prevalence),
      threshold,
      groups: groups && atTest(groups),
      slices: slices && Object.fromEntries(Object.entries(slices).map(([field, values]) => [field, atTest(values)])),
      baseline: baseline && atTest(baseline),
    });
    if (backgroundP) ev.background_rate = backgroundP.filter((p) => p >= threshold).length / backgroundP.length;
    const gates = gateHead(policy, ev, { maxEce });
    result.evaluation[name] = ev;
    result.failures.push(...gates.failures.map((f) => `${name}: ${f}`));
    result.warnings.push(...gates.warnings.map((w) => `${name}: ${w}`));
    result.heads[name] = { weights: model.coef, bias: model.intercept, calibration, threshold, review_floor: threshold * reviewRatio };
    result.testProbabilities[name] = { idx: idx.test, p: pTest };
  }
  return result;
}

/**
 * Serialises the artifact, re-loads it through validateArtifact and checks runtime scoring
 * reproduces the evaluated test probabilities exactly - so what was evaluated is what will run.
 */
export function assertRoundTrip<H extends string>(
  artifact: ClassifierArtifact<H>,
  X: ReadonlyArray<ArrayLike<number>>,
  testProbabilities: TrainResult<H>['testProbabilities'],
  sample = 50,
): void {
  const reloaded = validateArtifact<H>(JSON.parse(JSON.stringify(artifact)));
  for (const [head, probs] of Object.entries(testProbabilities) as Array<[H, { idx: number[]; p: number[] }]>) {
    probs.idx.slice(0, sample).forEach((i, j) => {
      const runtime = scoreEmbedding(reloaded, X[i])[head]!;
      if (runtime !== probs.p[j]) throw new Error(`${head}: runtime scoring of the saved artifact differs from evaluation (${runtime} vs ${probs.p[j]})`);
    });
  }
}
