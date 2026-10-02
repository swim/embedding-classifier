/** Release gates: an artifact is only fit to act on its decisions if every head passes. */
import type { HeadEvaluation } from './evaluate.ts';
import type { HeadPolicy } from './threshold.ts';

/** Below this many test examples, a gate without an explicit minimum warns that its estimate is thin. */
const MIN_SUPPORT = 30;

export function gateHead(policy: HeadPolicy, ev: HeadEvaluation, options: { maxEce?: number } = {}): { failures: string[]; warnings: string[] } {
  const { maxEce = 0.05 } = options;
  const failures: string[] = [];
  const warnings: string[] = [];
  if (policy.kind === 'recall') {
    const minPositives = policy.minPositives ?? 0;
    if (policy.minPositives === undefined && ev.positives < MIN_SUPPORT) {
      warnings.push(`recall rests on only ${ev.positives} test positives and no minPositives gate is set`);
    }
    if (ev.positives < minPositives) failures.push(`only ${ev.positives} test positives (need >= ${minPositives} for a meaningful CI)`);
    if (!(ev.recall >= policy.targetRecall)) failures.push(`test recall ${ev.recall.toFixed(3)} < target ${policy.targetRecall}`);
    if (policy.minRecallLower !== undefined && !(ev.recall_ci95[0] >= policy.minRecallLower)) {
      failures.push(`recall CI lower bound ${ev.recall_ci95[0].toFixed(3)} < ${policy.minRecallLower}`);
    }
    if (ev.vs_baseline && (policy.mustBeatBaseline ?? true) && ev.vs_baseline.caught_by_classifier_only === 0) {
      failures.push('catches nothing the baseline misses - adds no value');
    }
    if (policy.minPositiveGroups !== undefined && ev.positive_groups < policy.minPositiveGroups) {
      warnings.push(`test positives come from only ${ev.positive_groups} distinct group(s) - the recall CI assumes independent examples and is optimistic`);
    }
  } else if (Number.isNaN(ev.precision_prevalence_weighted)) {
    failures.push(`never fires on the test set at threshold ${ev.threshold} - model too weak or threshold too high`);
  } else {
    if (!(ev.precision_prevalence_weighted >= policy.targetPrecision)) {
      failures.push(`prevalence-weighted precision ${ev.precision_prevalence_weighted.toFixed(3)} < target ${policy.targetPrecision}`);
    }
    if (policy.minFired !== undefined && ev.fired < policy.minFired) {
      failures.push(`fires on only ${ev.fired} test examples (need >= ${policy.minFired} for a meaningful precision)`);
    } else if (policy.minFired === undefined && ev.fired < MIN_SUPPORT) {
      warnings.push(`precision rests on only ${ev.fired} fired test examples and no minFired gate is set`);
    }
  }
  if (!(ev.ece_prevalence_weighted <= maxEce)) failures.push(`ECE ${ev.ece_prevalence_weighted.toFixed(3)} > ${maxEce}`);
  return { failures, warnings };
}
