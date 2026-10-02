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
 *   precision  for heads where a false alarm is the costly error. The threshold is the target
 *              precision itself: for calibrated probabilities, messages scored >= t are on average
 *              at least t likely to be positive.
 */
export type HeadPolicy =
  | {
      kind: 'recall';
      /** Test-split gate. */
      targetRecall: number;
      /** Threshold selection target on the calibration split (>= targetRecall). */
      designRecall: number;
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
    }
  | {
      kind: 'precision';
      targetPrecision: number;
      /** Gate: minimum test examples the head must fire on for its precision to mean anything (default none; warns below 30). */
      minFired?: number;
    };

export function pickThreshold(policy: HeadPolicy, p: ArrayLike<number>, y: ArrayLike<number>): number {
  if (policy.kind === 'precision') return policy.targetPrecision;
  if (policy.designRecall < policy.targetRecall) throw new Error('designRecall must be >= targetRecall');
  const scored = Array.from(p);
  const positives = scored.filter((_, i) => y[i] === 1).sort((a, b) => b - a);
  if (!positives.length) throw new Error('no positives to choose a recall threshold from');
  const recallThreshold = positives[Math.max(1, Math.ceil(policy.designRecall * positives.length)) - 1];
  const negatives = scored.filter((_, i) => y[i] === 0).sort((a, b) => b - a);
  const allowed = Math.floor((policy.maxFalseAlarm ?? 1) * negatives.length);
  // The lowest threshold at which at most `allowed` negatives are >= it.
  const budgetThreshold = allowed < negatives.length ? nextUp(negatives[allowed]) : 0;
  return Math.max(recallThreshold, budgetThreshold);
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

/** Smallest double strictly greater than v, so `p >= threshold` excludes v itself. */
export function nextUp(v: number): number {
  if (Number.isNaN(v) || v === Infinity) return v;
  if (v === 0) return Number.MIN_VALUE;
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, v);
  const bits = buf.getBigUint64(0);
  buf.setBigUint64(0, v > 0 ? bits + 1n : bits - 1n);
  return buf.getFloat64(0);
}
