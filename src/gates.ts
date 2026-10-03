/** Release gates: an artifact is only fit to act on its decisions if every head passes. */
import { normalQuantile, wilson } from '@liquidau/solvers';

import type { HeadEvaluation } from './evaluate.ts';
import type { HeadPolicy } from './threshold.ts';

/** Below this many test examples, a gate without an explicit minimum warns that its estimate is thin. */
const MIN_SUPPORT = 30;

export function gateHead(policy: HeadPolicy, ev: HeadEvaluation, options: { maxEce?: number; calibrationAlpha?: number } = {}): { failures: string[]; warnings: string[] } {
  const { maxEce = 0.05, calibrationAlpha } = options;
  const failures: string[] = [];
  const warnings: string[] = [];
  if (policy.kind === 'recall') {
    const minPositives = policy.minPositives ?? 0;
    // A stratified test sample counts its Kish effective positives, not rows.
    const positives = ev.design ? ev.design.effective_positives : ev.positives;
    const what = ev.design ? `${positives.toFixed(1)} effective test positives` : `${positives} test positives`;
    if (policy.minPositives === undefined && positives < MIN_SUPPORT) {
      warnings.push(`recall rests on only ${what} and no minPositives gate is set`);
    }
    if (positives < minPositives) failures.push(`only ${what} (need >= ${minPositives} for a meaningful CI)`);
    if (ev.guarantee && ev.guarantee.kind !== 'none') {
      // The threshold already guarantees recall; a test sample falls below the target by chance
      // about half the time even when it holds. Fail only when the test split contradicts it.
      const upper = ev.certified?.recall_upper;
      if (!(upper >= policy.targetRecall)) {
        failures.push(`test recall ${ev.recall.toFixed(3)} contradicts the ${ev.guarantee.kind} guarantee: its upper bound ${upper?.toFixed(3)} < target ${policy.targetRecall}`);
      }
    } else if (!(ev.recall >= policy.targetRecall)) failures.push(`test recall ${ev.recall.toFixed(3)} < target ${policy.targetRecall}`);
    if (policy.minRecallLower !== undefined && !(ev.recall_ci95[0] >= policy.minRecallLower)) {
      failures.push(`recall CI lower bound ${ev.recall_ci95[0].toFixed(3)} < ${policy.minRecallLower}`);
    }
    if (ev.vs_baseline && (policy.mustBeatBaseline ?? true) && ev.vs_baseline.caught_by_classifier_only === 0) {
      failures.push('catches nothing the baseline misses - adds no value');
    }
    if (policy.sliceGate) gateSlices(policy.targetRecall, policy.sliceGate, ev, failures, warnings);
    if (policy.minPositiveGroups !== undefined && ev.positive_groups < policy.minPositiveGroups) {
      warnings.push(`test positives come from only ${ev.positive_groups} distinct group(s) - the recall CI assumes independent examples and is optimistic`);
    }
  } else if (Number.isNaN(ev.precision_prevalence_weighted)) {
    failures.push(`never fires on the test set at threshold ${ev.threshold} - model too weak or threshold too high`);
  } else {
    if (ev.guarantee && ev.guarantee.kind !== 'none' && ev.guarantee.metric === 'precision') {
      // The threshold already carries a precision guarantee: fail only when the test split contradicts it.
      const p = ev.design?.precision;
      const upper = p ? p.estimate + 1.96 * p.se : ev.precision_prevalence_weighted;
      if (!(upper >= policy.targetPrecision)) failures.push(`test precision ${ev.precision_prevalence_weighted.toFixed(3)} contradicts the ${ev.guarantee.kind} guarantee: its upper bound ${upper.toFixed(3)} < target ${policy.targetPrecision}`);
    } else if (!(ev.precision_prevalence_weighted >= policy.targetPrecision)) {
      failures.push(`prevalence-weighted precision ${ev.precision_prevalence_weighted.toFixed(3)} < target ${policy.targetPrecision}`);
    }
    if (policy.minFired !== undefined && ev.fired < policy.minFired) {
      failures.push(`fires on only ${ev.fired} test examples (need >= ${policy.minFired} for a meaningful precision)`);
    } else if (policy.minFired === undefined && ev.fired < MIN_SUPPORT) {
      warnings.push(`precision rests on only ${ev.fired} fired test examples and no minFired gate is set`);
    }
  }
  if (!(ev.ece_prevalence_weighted <= maxEce)) failures.push(`ECE ${ev.ece_prevalence_weighted.toFixed(3)} > ${maxEce}`);
  if (calibrationAlpha !== undefined && ev.calibration_test && ev.calibration_test.p_value <= calibrationAlpha) {
    const c = ev.calibration_test;
    failures.push(`probabilities are miscalibrated (Cox test p = ${c.p_value.toExponential(2)} <= ${calibrationAlpha}: intercept ${c.intercept.toFixed(3)}, slope ${c.slope.toFixed(3)}; calibrated is 0 and 1)`);
  }
  return { failures, warnings };
}

/** Per-slice recall: fail when a slice is demonstrably below target, warn when it can't be confirmed. */
function gateSlices(targetRecall: number, gate: NonNullable<Extract<HeadPolicy, { kind: 'recall' }>['sliceGate']>, ev: HeadEvaluation, failures: string[], warnings: string[]): void {
  const { target = targetRecall, minPositives = 10, fields } = gate;
  const inField = (name: string) => !fields || fields.includes(name.slice(0, name.indexOf('=')));
  // Design-based estimates when the test set is a probability sample, else counts with Wilson intervals.
  const slices = ev.design
    ? Object.entries(ev.design.slices).filter(([k]) => inField(k)).map(([k, e]) => ({ name: k, n: e.effective_n ?? 0, bounds: (z: number): [number, number] => [e.estimate - z * e.se, e.estimate + z * e.se], estimate: e.estimate }))
    : Object.entries(ev.slices).filter(([k]) => inField(k)).map(([k, s]) => ({ name: k, n: s.positives, bounds: (z: number) => wilson(Math.round(s.recall * s.positives), s.positives, z), estimate: s.recall }));
  const judged = slices.filter((s) => s.n >= minPositives);
  for (const s of slices.filter((x) => x.n < minPositives)) warnings.push(`slice ${s.name}: only ${s.n.toFixed(1)} positives - too few to judge its recall`);
  if (!judged.length) return;
  const z = normalQuantile(1 - 0.05 / (2 * judged.length));
  for (const s of judged) {
    const [lo, hi] = s.bounds(z);
    if (hi < target) failures.push(`slice ${s.name}: recall ${s.estimate.toFixed(3)} is below the target ${target} (upper bound ${hi.toFixed(3)})`);
    else if (lo < target) warnings.push(`slice ${s.name}: recall ${s.estimate.toFixed(3)} can't be confirmed at ${target} (lower bound ${lo.toFixed(3)})`);
  }
}
