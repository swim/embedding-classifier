/**
 * Test-split evaluation of one head: recall with a Wilson CI, false-alarm rate,
 * prevalence-weighted precision and ECE, a reliability table, recall per slice, and (when a
 * baseline is supplied, e.g. existing rules) what the classifier adds on top of it.
 */
import { ece, wilson, type ReliabilityRow } from '@liquidau/solvers';

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
}

export function evaluateHead({ p, y, w, threshold, groups, slices = {}, baseline }: EvaluateInput): HeadEvaluation {
  const lengths: Array<[string, { length: number } | undefined]> = [['y', y], ['w', w], ['groups', groups], ['baseline', baseline], ...Object.entries(slices).map(([f, v]) => [`slices.${f}`, v] as [string, readonly string[]])];
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
  return result;
}
