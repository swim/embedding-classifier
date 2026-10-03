/**
 * Drift monitoring for one deployed head: is live traffic still like the traffic its guarantees
 * were stated on? Exact binomial tests on the shares of live scores at or above the threshold and
 * the review floor - the tail where the head's decisions and guarantees live - Bonferroni-corrected
 * to `alpha` per window:
 *
 *   firing up     the share firing exceeds the certified background upper bound
 *                 (evaluation.background_rate_upper): the false-alarm guarantee itself, tested
 *   firing down   it falls below the reference's exact lower bound: messages that used to fire
 *                 no longer do - a possible recall loss
 *   review up / down   the same for the review band's floor, against the reference
 *
 * In simulation these caught bulk shifts and new high-scoring topics that a two-sample KS test on
 * the whole score distribution missed entirely (it is weak in the tail), with false alarms within
 * alpha. Reference: unlabelled real traffic scored by the same artifact (e.g. the 'budget'
 * background records). Compare like-for-like windows (same weekday and hours), and correct for the
 * number of windows you check. None of this can see a change in what labels MEAN for the same
 * text: schedule fresh sampled evaluations (designSample) for that.
 */
import { binomialCdf, clopperPearsonLower, clopperPearsonUpper, exceedanceTest } from '@liquidau/solvers';

import type { HeadSpec } from './artifact.ts';

export interface DriftCheck {
  test: 'firing-up' | 'firing-down' | 'review-up' | 'review-down';
  pValue: number;
  alert: boolean;
  detail: string;
}

export function monitorWindow(options: {
  spec: Pick<HeadSpec, 'threshold' | 'review_floor'>;
  /** Certified upper bound on the background firing rate (evaluation.background_rate_upper); else the reference's upper bound. */
  firingBound?: number;
  /** Scores of the reference traffic, from the same artifact. */
  reference: ArrayLike<number>;
  /** Scores of the live window. */
  live: ArrayLike<number>;
  /** Family-wise false-alarm rate for this window (default 0.01). */
  alpha?: number;
}): { alert: boolean; checks: DriftCheck[] } {
  const { spec, firingBound, reference, live, alpha = 0.01 } = options;
  if (!(alpha > 0 && alpha < 1)) throw new Error(`alpha must be strictly between 0 and 1, got ${alpha}`);
  const liveArr = Array.from(live), refArr = Array.from(reference);
  if (!liveArr.length || !refArr.length) throw new Error('reference and live must be non-empty');
  for (const v of [...liveArr, ...refArr]) if (!Number.isFinite(v)) throw new Error('scores must be finite');
  // Four tests; each reference bound and its test share half of a test's budget.
  const a = alpha / 4;
  const count = (xs: number[], t: number) => xs.filter((p) => p >= t).length;
  const checks: DriftCheck[] = [];
  const n = liveArr.length, m = refArr.length;
  for (const [name, t] of [['firing', spec.threshold], ['review', spec.review_floor]] as const) {
    const k = count(liveArr, t), kr = count(refArr, t);
    const upper = name === 'firing' && firingBound !== undefined ? firingBound : clopperPearsonUpper(kr, m, 1 - a / 2);
    const up = exceedanceTest(k, n, upper, name === 'firing' && firingBound !== undefined ? a : a / 2);
    checks.push({ test: `${name}-up`, pValue: up.pValue, alert: up.alert, detail: `${k}/${n} at or above ${t} vs upper bound ${upper.toFixed(4)}${name === 'firing' && firingBound !== undefined ? ' (certified)' : ` (reference ${kr}/${m})`}` });
    const lower = clopperPearsonLower(kr, m, 1 - a / 2);
    const pDown = binomialCdf(k, n, lower);
    checks.push({ test: `${name}-down`, pValue: pDown, alert: pDown <= a / 2, detail: `${k}/${n} at or above ${t} vs reference lower bound ${lower.toFixed(4)} (reference ${kr}/${m})` });
  }
  return { alert: checks.some((c) => c.alert), checks };
}
