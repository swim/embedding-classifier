/**
 * Per-head policies: how a head's threshold is chosen on the calibration split and which gates it
 * must pass on the test split.
 *
 *   recall     for heads where a miss is the costly error. The threshold is the highest that
 *              reaches `designRecall` on calibration positives, set deliberately above the
 *              gate's `targetRecall`, so a test set from the same distribution doesn't land either
 *              side of the gate at random. It is capped so that at most `maxFalseAlarm` of
 *              calibration negatives fire: without the cap, one mislabelled or very hard positive
 *              can drag the threshold to ~0 and the head fires on everything. With it, a model that
 *              can't reach the target within the budget fails its recall gate openly.
 *              That is the `heuristic` mode: the margin is chosen by hand and guarantees nothing.
 *              The conformal modes (see conformal.ts) instead pick the threshold from order
 *              statistics so production recall >= targetRecall in expectation
 *              (`conformal-expected`) or with probability 1 - delta (`conformal-pac`), with
 *              maxFalseAlarm and any background budget certified the same way; `auto` takes the
 *              strongest the data supports.
 *   precision  for heads where a false alarm is the costly error. The threshold is the target
 *              precision itself: for calibrated probabilities, messages scored >= t are on average
 *              at least t likely to be positive.
 */
import { nextUp } from '@liquidau/solvers';

import type { ThresholdMode } from './conformal.ts';

/** Re-exported from @liquidau/solvers, where it now lives. */
export { nextUp };

export type HeadPolicy =
  | {
      kind: 'recall';
      /**
       * Heuristic: the test-split gate. Conformal: the recall the threshold guarantees (α = 1 -
       * targetRecall); the test gate then fails only if the test split contradicts it.
       */
      targetRecall: number;
      /** Heuristic threshold selection target on the calibration split (>= targetRecall). Required for `heuristic`, and as `auto`'s last resort. */
      designRecall?: number;
      /** How the threshold is chosen (default `heuristic`). */
      mode?: ThresholdMode;
      /**
       * auto: when the calibration data supports no guarantee at all, use the heuristic threshold
       * with an "inconclusive" warning (true, default) or fail the head (false - e.g. for
       * safety-critical heads). Falling back from PAC to an expected guarantee is always allowed.
       */
      allowHeuristicFallback?: boolean;
      /** design mode: the bound behind the guarantee (default 'exact'; see solvers' designRiskThreshold). */
      designMethod?: 'exact' | 'linearised' | 'bootstrap';
      /**
       * design mode, linearised / bootstrap: infeasible when a sampled calibration stratum holds fewer
       * positives than this (default 5; 0 turns the guard off). See designSample's expected-positives allocation.
       */
      designMinStratumPositives?: number;
      /** Conformal modes: the guarantee fails with probability at most delta (default 0.05), split across recall and every false-alarm budget. */
      delta?: number;
      /** Highest share of calibration negatives allowed to fire (default 1: no cap). */
      maxFalseAlarm?: number;
      /** Gate: minimum test positives for the CI to mean anything (default 0). */
      minPositives?: number;
      /** Gate: lower bound of the recall 95% Wilson CI (default none). */
      minRecallLower?: number;
      /** Warning when test positives come from fewer distinct groups (default none). */
      minPositiveGroups?: number;
      /** Gate, when a baseline is supplied: must catch something the baseline misses (default true). */
      mustBeatBaseline?: boolean;
      /**
       * Per-slice recall gate on the test split (slices from trainHeads `slices`). A slice whose recall
       * upper bound is below `target` (default targetRecall) FAILS - it is demonstrably missing its
       * target; one whose lower bound is below it gets a warning (can't be confirmed); slices with
       * fewer than minPositives (effective) positives get an "insufficient data" warning. Bounds are
       * Bonferroni-corrected across the slices checked, at 95% overall.
       */
      sliceGate?: { target?: number; minPositives?: number; fields?: readonly string[] };
    }
  | {
      kind: 'precision';
      targetPrecision: number;
      /**
       * 'heuristic' (default): the threshold is targetPrecision on the calibrated probability - only as
       * good as the calibrator; in simulation a misspecified Platt fit missed the target in 95% of runs.
       * 'design': solvers' designPrecisionThreshold on sampled calibration records - the loosest
       * candidate whose precision lower bound reaches the target (candidates from training scores).
       */
      mode?: 'heuristic' | 'design';
      /** design mode: the precision bound fails with probability at most delta (default 0.05). */
      delta?: number;
      /** design mode: 'linearised' (default; approximate) or 'exact' (valid but rarely feasible: it must allow for unseen false positives in every stratum). */
      designMethod?: 'exact' | 'linearised';
      /** Gate: minimum test examples the head must fire on for its precision to mean anything (default none; warns below 30). */
      minFired?: number;
    };

/**
 * The heuristic threshold. With `w` (e.g. design weights 1/π from a stratified sample), recall and
 * the false-alarm share are weighted - unweighted shares of a stratified sample are biased. With
 * unit weights the result is exactly the unweighted one.
 */
export function pickThreshold(policy: HeadPolicy, p: ArrayLike<number>, y: ArrayLike<number>, w?: ArrayLike<number>): number {
  if (policy.kind === 'precision') return policy.targetPrecision;
  if (policy.designRecall === undefined) throw new Error('designRecall is required for a heuristic recall threshold');
  if (policy.designRecall < policy.targetRecall) throw new Error('designRecall must be >= targetRecall');
  const scored = Array.from(p, (v, i) => ({ v, w: w ? w[i] : 1, y: y[i] }));
  const positives = scored.filter((s) => s.y === 1).sort((a, b) => b.v - a.v);
  if (!positives.length) throw new Error('no positives to choose a recall threshold from');
  let recallThreshold: number;
  if (!w) recallThreshold = positives[Math.max(1, Math.ceil(policy.designRecall * positives.length)) - 1].v;
  else {
    // The highest score at which the weighted share of positives at or above it reaches designRecall.
    const total = positives.reduce((s, x) => s + x.w, 0);
    let cum = 0, j = 0;
    for (; j < positives.length; j++) { cum += positives[j].w; if (cum / total >= policy.designRecall - 1e-12) break; }
    recallThreshold = positives[Math.min(j, positives.length - 1)].v;
  }
  return Math.max(recallThreshold, falseAlarmCap(p, y, policy.maxFalseAlarm ?? 1, w));
}

/**
 * The lowest threshold at which at most a share maxFalseAlarm of the negatives (weighted by `w`
 * when given) score at or above it.
 */
export function falseAlarmCap(p: ArrayLike<number>, y: ArrayLike<number>, maxFalseAlarm: number, w?: ArrayLike<number>): number {
  const negatives = Array.from(p, (v, i) => ({ v, w: w ? w[i] : 1, y: y[i] })).filter((s) => s.y === 0).sort((a, b) => b.v - a.v);
  let allowed: number;
  if (!w) allowed = Math.floor(maxFalseAlarm * negatives.length);
  else {
    const total = negatives.reduce((s, x) => s + x.w, 0);
    let cum = 0;
    allowed = 0;
    while (allowed < negatives.length && (cum + negatives[allowed].w) / total <= maxFalseAlarm + 1e-12) cum += negatives[allowed++].w;
  }
  return allowed < negatives.length ? nextUp(negatives[allowed].v) : 0;
}

/**
 * Raises (never lowers) a threshold until at most `maxRate` of `background` scores reach it - for a
 * background of ordinary, mostly-negative traffic whose false alarms the labelled data can't show.
 */
export function budgetThreshold(threshold: number, background: ArrayLike<number>, maxRate: number): number {
  const sorted = Array.from(background).sort((a, b) => b - a);
  const allowed = Math.floor(maxRate * sorted.length);
  return allowed < sorted.length ? Math.max(threshold, nextUp(sorted[allowed])) : threshold;
}
