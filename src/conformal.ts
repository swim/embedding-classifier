/**
 * Conformal threshold modes for recall heads (class-conditional split conformal). The statistics -
 * order-statistic thresholds with PAC or expected guarantees, verified against MAPIE and crepes -
 * live in @liquidau/solvers: recall is conformalLowerThreshold over calibration positives, and a
 * false-alarm budget is conformalUpperThreshold over negatives or background text. This module
 * decides which guarantee a head gets, and why.
 *
 * The guarantees hold at any sample size; small samples only make the threshold stricter. They
 * condition on the class, so they survive a prevalence shift between calibration and production,
 * but each class's calibration examples must be exchangeable with production examples of that
 * class. A calibrated probability is a non-decreasing function of the model's logit, so they also
 * survive calibration (ties from isotonic plateaus only make them more conservative).
 */
import { conformalLowerThreshold, conformalUpperThreshold, minimumSamples } from '@liquidau/solvers';

/** `design`: designRiskThreshold on a probability-sampled calibration set (see trainHeads `records`). */
export type ThresholdMode = 'heuristic' | 'conformal-expected' | 'conformal-pac' | 'auto' | 'design';
export type GuaranteeKind = 'pac' | 'expected' | 'design-exact' | 'design-approximate' | 'none';
export const THRESHOLD_MODES: readonly ThresholdMode[] = ['heuristic', 'conformal-expected', 'conformal-pac', 'auto', 'design'];

function checkRate(name: string, v: number): void {
  if (!(v > 0 && v < 1)) throw new Error(`${name} must be strictly between 0 and 1, got ${v}`);
}

/**
 * One score per group, for calibration: paraphrases of one seed aren't exchangeable, so a group
 * counts once. `pick` = Math.min for positives (a group is caught only if every member is),
 * Math.max for negatives (it fires if any member does) - both conservative.
 */
export function groupScores(scores: readonly number[], groups: readonly string[] | undefined, pick: (a: number, b: number) => number): number[] {
  if (!groups) return [...scores];
  const by = new Map<string, number>();
  scores.forEach((s, i) => by.set(groups[i], by.has(groups[i]) ? pick(by.get(groups[i])!, s) : s));
  return [...by.values()];
}

export interface Guarantee {
  /** The mode the policy asked for. */
  mode: ThresholdMode;
  /** What the shipped threshold actually guarantees (calibration-based). */
  kind: GuaranteeKind;
  /** Miss rate guaranteed: 1 - targetRecall. */
  alpha: number;
  /** Family-wise: recall and every false-alarm budget hold together with probability 1 - delta (PAC and design guarantees). */
  delta?: number;
  /** Design guarantees: the estimator behind the bound. */
  method?: 'exact' | 'linearised' | 'bootstrap';
  /** What the guarantee is about (default 'recall'): alpha is 1 - the target recall or precision. */
  metric?: 'recall' | 'precision';
  /** Certified share of calibration negatives that may fire (from maxFalseAlarm). */
  false_alarm?: number;
  /** Certified share of background text that may fire (from the background budget). */
  background_rate?: number;
}

export interface Sufficiency {
  /** Calibration positive groups - what the recall guarantee counts. */
  positive_groups: number;
  /** Positive groups needed for each guarantee (PAC with δ split across the enforced constraints). */
  needed: { expected: number; pac: number };
  feasible: Array<'conformal-expected' | 'conformal-pac'>;
  /** The threshold method actually used. */
  chosen: Exclude<ThresholdMode, 'auto'>;
  reason?: string;
  /** Per slice (field=value): calibration positive groups and which guarantees a per-slice threshold could have. Reported, not enforced. */
  slices?: Record<string, { positive_groups: number; feasible: Array<'conformal-expected' | 'conformal-pac'> }>;
}

export interface FalseAlarmConstraint {
  /** e.g. "calibration false-alarm" or "background". */
  name: string;
  scores: readonly number[];
  maxRate: number;
}

export interface ConformalSelection {
  /** null: use the heuristic threshold (auto's fallback when the data supports no guarantee). */
  threshold: number | null;
  guarantee: Guarantee;
  sufficiency: Sufficiency;
  failures: string[];
  warnings: string[];
}

type Plan = { ok: true; threshold: number } | { ok: false; conflict: boolean; threshold: number | null; reason: string };

/**
 * Chooses a recall head's threshold under a conformal mode. The recall guarantee caps the
 * threshold from above, each false-alarm budget bounds it from below; if they cross, no threshold
 * gives both and the head fails with that reason. `auto` never trades a guarantee away silently:
 * it falls back to a weaker guarantee only for lack of data (with an "inconclusive" warning) - to
 * no guarantee at all (the heuristic threshold) only while `allowHeuristic` is true - and
 * fails when the false-alarm budget is what rules PAC out.
 */
export function conformalThreshold(input: {
  mode: Exclude<ThresholdMode, 'heuristic' | 'design'>;
  targetRecall: number;
  delta?: number;
  /** One calibrated score per calibration positive group (see groupScores). */
  positives: readonly number[];
  constraints?: readonly FalseAlarmConstraint[];
  /** Whether a heuristic threshold is available (designRecall set) as auto's last resort. */
  heuristicAvailable: boolean;
  /** auto: when the data supports no guarantee, use the heuristic threshold (true, default) or fail the head (false). */
  allowHeuristic?: boolean;
}): ConformalSelection {
  const { mode, targetRecall, delta = 0.05, positives, constraints = [], heuristicAvailable, allowHeuristic = true } = input;
  const alpha = 1 - targetRecall;
  checkRate('alpha (1 - targetRecall)', alpha);
  checkRate('delta', delta);
  for (const c of constraints) checkRate(`${c.name} maxRate`, c.maxRate);
  const perTest = delta / (1 + constraints.length);

  const plan = (kind: 'pac' | 'expected'): Plan => {
    const d = kind === 'pac' ? perTest : undefined;
    const upper = conformalLowerThreshold(positives, alpha, d);
    const lowers = constraints.map((c) => ({ c, t: conformalUpperThreshold(c.scores, c.maxRate, d) }));
    const lower = Math.max(0, ...lowers.map((l) => l.t ?? Infinity));
    const short = lowers.find((l) => l.t === null);
    if (upper === null) {
      return { ok: false, conflict: false, threshold: null, reason: `${positives.length} calibration positive groups, ${minimumSamples(alpha, d)} needed for ${kind === 'pac' ? `a PAC guarantee at δ = ${+perTest.toFixed(6)}` : 'an expected guarantee'}` };
    }
    if (short) {
      return { ok: false, conflict: false, threshold: null, reason: `${short.c.scores.length} ${short.c.name} examples, ${minimumSamples(short.c.maxRate, d)} needed to certify a rate <= ${short.c.maxRate}` };
    }
    if (lower > upper) {
      const binding = lowers.reduce((a, b) => (b.t! > a.t! ? b : a));
      return { ok: false, conflict: true, threshold: lower, reason: `recall >= ${targetRecall} needs a threshold <= ${upper} but ${binding.c.name} <= ${binding.c.maxRate} needs >= ${lower}` };
    }
    return { ok: true, threshold: upper };
  };
  const pac = plan('pac');
  const expected = plan('expected');
  const sufficiency: Sufficiency = {
    positive_groups: positives.length,
    needed: { expected: minimumSamples(alpha), pac: minimumSamples(alpha, perTest) },
    feasible: [...(expected.ok ? ['conformal-expected' as const] : []), ...(pac.ok ? ['conformal-pac' as const] : [])],
    chosen: 'heuristic',
  };
  const budgets = Object.fromEntries(constraints.map((c) => [c.name === 'background' ? 'background_rate' : 'false_alarm', c.maxRate]));
  const result = (kind: GuaranteeKind, threshold: number | null, failures: string[] = [], warnings: string[] = []): ConformalSelection => ({
    threshold,
    guarantee: { mode, kind, alpha, ...(kind === 'pac' ? { delta } : {}), ...(kind === 'none' ? {} : budgets) },
    sufficiency, failures, warnings,
  });
  const fallback = (p: Plan & { ok: false }) => p.threshold ?? (positives.length ? Math.min(...positives) : 0);

  if (mode === 'conformal-pac' || mode === 'conformal-expected') {
    const p = mode === 'conformal-pac' ? pac : expected;
    sufficiency.chosen = mode;
    if (p.ok) return result(mode === 'conformal-pac' ? 'pac' : 'expected', p.threshold);
    sufficiency.reason = p.reason;
    return result('none', fallback(p), [`${mode}: ${p.reason}`]);
  }
  // auto
  if (pac.ok) {
    sufficiency.chosen = 'conformal-pac';
    return result('pac', pac.threshold);
  }
  if (pac.conflict) {
    sufficiency.chosen = 'conformal-pac';
    sufficiency.reason = pac.reason;
    return result('none', fallback(pac), [`auto: ${pac.reason} - choose a mode explicitly rather than lose the guarantee silently`]);
  }
  const inconclusive = `guarantee inconclusive: ${pac.reason}`;
  if (expected.ok) {
    sufficiency.chosen = 'conformal-expected';
    sufficiency.reason = pac.reason;
    return result('expected', expected.threshold, [], [`${inconclusive}; using an expected (not high-probability) guarantee`]);
  }
  if (expected.conflict) {
    sufficiency.chosen = 'conformal-expected';
    sufficiency.reason = expected.reason;
    return result('none', fallback(expected), [`auto: ${expected.reason} - choose a mode explicitly rather than lose the guarantee silently`]);
  }
  sufficiency.reason = expected.reason;
  if (!allowHeuristic) return result('none', fallback(expected), [`auto: ${expected.reason}, and allowHeuristicFallback is false - no threshold with a guarantee`]);
  if (!heuristicAvailable) return result('none', fallback(expected), [`auto: ${expected.reason}, and no designRecall for a heuristic threshold`]);
  return result('none', null, [], [`${inconclusive}; ${expected.reason}; using the heuristic threshold, which guarantees nothing`]);
}
