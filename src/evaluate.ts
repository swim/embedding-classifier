/**
 * Test-split evaluation of one head: recall with a Wilson CI, false-alarm rate,
 * prevalence-weighted precision and ECE, a reliability table, recall per slice, (when a baseline
 * is supplied, e.g. existing rules) what the classifier adds on top of it, and exact certified
 * bounds for the shipped threshold whatever chose it.
 */
import { clopperPearsonUpper, coxTest, ece, kishEffectiveN, normalQuantile, stratifiedBootstrap, stratifiedRatio, weightedQuantile, wilson, type ReliabilityRow } from '@liquidau/solvers';

import type { Guarantee, Sufficiency } from './conformal.ts';

/**
 * Exact (Clopper-Pearson) bounds from the test split, which no threshold method sees - so they are
 * valid for any threshold, heuristic or conformal. Each holds with probability 1 - delta/2, so
 * recall_lower and false_alarm_upper (and precision_lower, derived from both) hold together with
 * probability 1 - delta. Groups count once: a positive group is caught for the lower bound only if
 * every member fires, and for the upper bound if any does; a negative group fires if any member does.
 */
export interface CertifiedBounds {
  delta: number;
  positive_groups: number;
  negative_groups: number;
  recall_lower: number;
  recall_upper: number;
  false_alarm_upper: number;
  /** At the head's production prevalence: π·R_L / (π·R_L + (1 - π)·FA_U). */
  precision_lower?: number;
  /** 'exact': Clopper-Pearson on an equal-probability test set. 'design-linearised': one-sided normal bounds from a stratified sample - approximate. */
  method?: 'exact' | 'design-linearised';
}

export interface DesignEstimate {
  estimate: number;
  se: number;
  /** Linearised, estimate ± 1.96·se clipped to [0, 1]. */
  ci95: [number, number];
  /** Rao-Wu bootstrap percentile interval. */
  bootstrap_ci95?: [number, number];
  /** Kish effective number of units in the estimate's denominator (e.g. positives, for recall). */
  effective_n?: number;
}

/** Design-based (Horvitz-Thompson) estimates from a stratified probability sample. Approximate. */
export interface DesignEvaluation {
  recall: DesignEstimate;
  /** At production prevalence: the design weights reproduce it. */
  precision: DesignEstimate | null;
  false_alarm_rate: DesignEstimate;
  prevalence: DesignEstimate;
  slices: Record<string, DesignEstimate>;
  /** Kish effective number of positives - what the gates' minimums count. */
  effective_positives: number;
  replicates: number;
}

export interface BaselineComparison {
  caught_by_both: number;
  caught_by_baseline_only: number;
  caught_by_classifier_only: number;
  missed_by_both: number;
}

export interface HeadEvaluation {
  /** The threshold evaluated - the one that ships, after any background budget. */
  threshold: number;
  n: number;
  /** Test examples at or above the threshold. */
  fired: number;
  positives: number;
  /** Distinct groups among test positives - paraphrases of one seed aren't independent. */
  positive_groups: number;
  recall: number;
  recall_ci95: [number, number];
  false_alarm_rate: number;
  precision_prevalence_weighted: number;
  ece_prevalence_weighted: number;
  reliability: ReliabilityRow[];
  slices: Record<string, { positives: number; recall: number; recall_ci95: [number, number] }>;
  vs_baseline?: BaselineComparison;
  /** Baseline OR classifier - what a deployment that runs both actually achieves. */
  combined_recall?: number;
  combined_recall_ci95?: [number, number];
  combined_false_alarm_rate?: number;
  /** Share of the background the head fires on at its final threshold (when a budget was applied). */
  background_rate?: number;
  /** Exact upper bound on the background firing rate (probability 1 - delta/2), valid even when the background chose the threshold. */
  background_rate_upper?: number;
  certified: CertifiedBounds;
  /** Present when the test set is a probability sample: recall, precision and false alarms above are these estimates. */
  design?: DesignEvaluation;
  /**
   * Cox's recalibration test on the test split (weights as for ECE): y ~ a + b·logit(p), H0 a = 0, b = 1.
   * Unlike ECE it has a stated error rate and stays sensitive for rare classes.
   */
  calibration_test?: { intercept: number; slope: number; lr: number; p_value: number };
  /** Recall heads: what the shipped threshold guarantees from calibration, and why. */
  guarantee?: Guarantee;
  sufficiency?: Sufficiency;
}

export interface EvaluateInput {
  /** Calibrated probabilities. */
  p: number[];
  y: number[];
  /** Per-example weights (e.g. prevalenceWeights) for precision and ECE. */
  w: number[];
  threshold: number;
  /** Group per example (default: every example is its own group). */
  groups?: readonly string[];
  /** field -> value per example; recall is reported per `field=value`. */
  slices?: Readonly<Record<string, readonly string[]>>;
  /** Whether an existing mechanism already catches each example. */
  baseline?: readonly boolean[];
  /** Certified bounds fail with probability at most delta (default 0.05). */
  delta?: number;
  /** Production prevalence, for certified.precision_lower. */
  prevalence?: number;
  /**
   * The test set's stratified design (designOf on the sampled test records), aligned with p.
   * Recall, precision, false-alarm rate and their intervals then become design-based estimates;
   * pass w = 1/π so ECE is weighted too.
   */
  design?: { inclusionProbs: readonly number[]; strata: readonly string[]; stratumSizes: Readonly<Record<string, number>>; replicates?: number; seed?: number };
}

const clopperPearsonLower = (k: number, n: number, confidence: number) => 1 - clopperPearsonUpper(n - k, n, confidence);

function designEvaluation(p: readonly number[], y: readonly number[], fired: readonly boolean[], design: NonNullable<EvaluateInput['design']>, slices: Readonly<Record<string, readonly string[]>>): DesignEvaluation {
  const { replicates = 2000, seed = 0 } = design;
  const reps = replicates > 0 ? stratifiedBootstrap({ inclusionProbs: design.inclusionProbs, strata: design.strata, replicates, seed }) : null;
  const est = (num: number[], den: number[]): DesignEstimate | null => {
    if (!den.some((d) => d !== 0)) return null;
    const r = stratifiedRatio({ ...design, num, den });
    const out: DesignEstimate = {
      estimate: r.estimate, se: r.se, ci95: [Math.max(0, r.estimate - 1.96 * r.se), Math.min(1, r.estimate + 1.96 * r.se)],
      effective_n: kishEffectiveN(den.flatMap((d, i) => (d ? [d / design.inclusionProbs[i]] : []))),
    };
    if (reps) {
      const rs = reps.flatMap((wb) => {
        let a = 0, b = 0;
        for (let i = 0; i < wb.length; i++) { a += wb[i] * num[i]; b += wb[i] * den[i]; }
        return b > 0 ? [a / b] : [];
      });
      if (rs.length) out.bootstrap_ci95 = [weightedQuantile(rs, rs.map(() => 1), 0.025), weightedQuantile(rs, rs.map(() => 1), 0.975)];
    }
    return out;
  };
  const f = fired.map(Number);
  const recall = est(y.map((v, i) => v * f[i]), [...y]);
  const falseAlarm = est(y.map((v, i) => (1 - v) * f[i]), y.map((v) => 1 - v));
  if (!recall || !falseAlarm) throw new Error('a sampled test set needs both positives and negatives');
  const sliceEstimates: Record<string, DesignEstimate> = {};
  for (const [field, values] of Object.entries(slices)) {
    for (const v of [...new Set(values)].sort()) {
      const e = est(y.map((yy, i) => (values[i] === v ? yy * f[i] : 0)), y.map((yy, i) => (values[i] === v ? yy : 0)));
      if (e) sliceEstimates[`${field}=${v}`] = e;
    }
  }
  return {
    recall,
    precision: est(y.map((v, i) => v * f[i]), f),
    false_alarm_rate: falseAlarm,
    prevalence: est([...y], p.map(() => 1))!,
    slices: sliceEstimates,
    effective_positives: kishEffectiveN(y.flatMap((v, i) => (v === 1 ? [1 / design.inclusionProbs[i]] : []))),
    replicates,
  };
}

export function evaluateHead({ p, y, w, threshold, groups, slices = {}, baseline, delta = 0.05, prevalence, design }: EvaluateInput): HeadEvaluation {
  if (!(delta > 0 && delta < 1)) throw new Error(`delta must be strictly between 0 and 1, got ${delta}`);
  const lengths: Array<[string, { length: number } | undefined]> = [['y', y], ['w', w], ['groups', groups], ['baseline', baseline], ['design.inclusionProbs', design?.inclusionProbs], ['design.strata', design?.strata], ...Object.entries(slices).map(([f, v]) => [`slices.${f}`, v] as [string, readonly string[]])];
  for (const [name, values] of lengths) {
    if (values !== undefined && values.length !== p.length) throw new Error(`${name} has ${values.length} entries but p has ${p.length}`);
  }
  const fired = p.map((v) => v >= threshold);
  const count = (pred: (i: number) => boolean) => p.reduce((n, _, i) => n + (pred(i) ? 1 : 0), 0);
  const pos = count((i) => y[i] === 1);
  const neg = count((i) => y[i] === 0);
  const tp = count((i) => fired[i] && y[i] === 1);
  const fp = count((i) => fired[i] && y[i] === 0);
  let wFired = 0;
  let wTp = 0;
  p.forEach((_, i) => {
    if (fired[i]) {
      wFired += w[i];
      wTp += w[i] * y[i];
    }
  });
  const calibration = ece(p, y, w);
  // Per group: [all members fired, any member fired], separately for each class.
  const byGroup = [new Map<string, [boolean, boolean]>(), new Map<string, [boolean, boolean]>()];
  p.forEach((_, i) => {
    if (y[i] !== 0 && y[i] !== 1) return;
    const g = groups ? groups[i] : `#${i}`;
    const prior = byGroup[y[i]].get(g) ?? [true, false];
    byGroup[y[i]].set(g, [prior[0] && fired[i], prior[1] || fired[i]]);
  });
  const tally = (cls: 0 | 1, which: 0 | 1) => [...byGroup[cls].values()].filter((v) => v[which]).length;
  const [posGroups, negGroups] = [byGroup[1].size, byGroup[0].size];
  const conf = 1 - delta / 2;
  const certified: CertifiedBounds = {
    delta, positive_groups: posGroups, negative_groups: negGroups,
    recall_lower: posGroups ? clopperPearsonLower(tally(1, 0), posGroups, conf) : 0,
    recall_upper: posGroups ? clopperPearsonUpper(tally(1, 1), posGroups, conf) : 1,
    false_alarm_upper: negGroups ? clopperPearsonUpper(tally(0, 1), negGroups, conf) : 1,
  };
  if (prevalence !== undefined) {
    const tp = prevalence * certified.recall_lower;
    const fp = (1 - prevalence) * certified.false_alarm_upper;
    certified.precision_lower = tp + fp === 0 ? 0 : tp / (tp + fp);
  }
  const result: HeadEvaluation = {
    threshold,
    n: y.length,
    fired: count((i) => fired[i]),
    positives: pos,
    positive_groups: new Set(p.flatMap((_, i) => (y[i] === 1 ? [groups ? groups[i] : `#${i}`] : []))).size,
    recall: pos ? tp / pos : NaN,
    recall_ci95: wilson(tp, pos),
    false_alarm_rate: neg ? fp / neg : NaN,
    precision_prevalence_weighted: wFired ? wTp / wFired : NaN,
    ece_prevalence_weighted: calibration.ece,
    reliability: calibration.reliability,
    slices: {},
    certified,
  };
  for (const [field, values] of Object.entries(slices)) {
    for (const v of [...new Set(values)].sort()) {
      const inSlice = (i: number) => values[i] === v && y[i] === 1;
      const n = count(inSlice);
      const k = count((i) => inSlice(i) && fired[i]);
      if (n) result.slices[`${field}=${v}`] = { positives: n, recall: k / n, recall_ci95: wilson(k, n) };
    }
  }
  if (baseline) {
    result.vs_baseline = {
      caught_by_both: count((i) => y[i] === 1 && baseline[i] && fired[i]),
      caught_by_baseline_only: count((i) => y[i] === 1 && baseline[i] && !fired[i]),
      caught_by_classifier_only: count((i) => y[i] === 1 && !baseline[i] && fired[i]),
      missed_by_both: count((i) => y[i] === 1 && !baseline[i] && !fired[i]),
    };
    const combined = count((i) => y[i] === 1 && (baseline[i] || fired[i]));
    result.combined_recall = pos ? combined / pos : NaN;
    result.combined_recall_ci95 = wilson(combined, pos);
    result.combined_false_alarm_rate = neg ? count((i) => y[i] === 0 && (baseline[i] || fired[i])) / neg : NaN;
  }
  if (y.some((v) => v === 1) && y.some((v) => v === 0)) {
    const clip = (v: number) => Math.min(1 - 1e-12, Math.max(1e-12, v));
    const c = coxTest(p.map(clip), y, w);
    result.calibration_test = { intercept: c.intercept, slope: c.slope, lr: c.lr, p_value: c.pValue };
  }
  if (design) {
    const d = designEvaluation(p, y, fired, design, slices);
    const z = normalQuantile(1 - delta / 2);
    result.design = d;
    result.recall = d.recall.estimate;
    result.recall_ci95 = d.recall.ci95;
    result.false_alarm_rate = d.false_alarm_rate.estimate;
    result.precision_prevalence_weighted = d.precision?.estimate ?? NaN;
    result.certified = {
      ...certified,
      method: 'design-linearised',
      recall_lower: Math.max(0, d.recall.estimate - z * d.recall.se),
      recall_upper: Math.min(1, d.recall.estimate + z * d.recall.se),
      false_alarm_upper: Math.min(1, d.false_alarm_rate.estimate + z * d.false_alarm_rate.se),
      precision_lower: d.precision ? Math.max(0, d.precision.estimate - z * d.precision.se) : undefined,
    };
  } else certified.method = 'exact';
  return result;
}
