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
/** Re-exported from @liquidau/solvers, where it now lives. */
export { nextUp };
/**
 * The heuristic threshold. With `w` (e.g. design weights 1/π from a stratified sample), recall and
 * the false-alarm share are weighted - unweighted shares of a stratified sample are biased. With
 * unit weights the result is exactly the unweighted one.
 */
export function pickThreshold(policy, p, y, w) {
    if (policy.kind === 'precision')
        return policy.targetPrecision;
    if (policy.designRecall === undefined)
        throw new Error('designRecall is required for a heuristic recall threshold');
    if (policy.designRecall < policy.targetRecall)
        throw new Error('designRecall must be >= targetRecall');
    const scored = Array.from(p, (v, i) => ({ v, w: w ? w[i] : 1, y: y[i] }));
    const positives = scored.filter((s) => s.y === 1).sort((a, b) => b.v - a.v);
    if (!positives.length)
        throw new Error('no positives to choose a recall threshold from');
    let recallThreshold;
    if (!w)
        recallThreshold = positives[Math.max(1, Math.ceil(policy.designRecall * positives.length)) - 1].v;
    else {
        // The highest score at which the weighted share of positives at or above it reaches designRecall.
        const total = positives.reduce((s, x) => s + x.w, 0);
        let cum = 0, j = 0;
        for (; j < positives.length; j++) {
            cum += positives[j].w;
            if (cum / total >= policy.designRecall - 1e-12)
                break;
        }
        recallThreshold = positives[Math.min(j, positives.length - 1)].v;
    }
    return Math.max(recallThreshold, falseAlarmCap(p, y, policy.maxFalseAlarm ?? 1, w));
}
/**
 * The lowest threshold at which at most a share maxFalseAlarm of the negatives (weighted by `w`
 * when given) score at or above it.
 */
export function falseAlarmCap(p, y, maxFalseAlarm, w) {
    const negatives = Array.from(p, (v, i) => ({ v, w: w ? w[i] : 1, y: y[i] })).filter((s) => s.y === 0).sort((a, b) => b.v - a.v);
    let allowed;
    if (!w)
        allowed = Math.floor(maxFalseAlarm * negatives.length);
    else {
        const total = negatives.reduce((s, x) => s + x.w, 0);
        let cum = 0;
        allowed = 0;
        while (allowed < negatives.length && (cum + negatives[allowed].w) / total <= maxFalseAlarm + 1e-12)
            cum += negatives[allowed++].w;
    }
    return allowed < negatives.length ? nextUp(negatives[allowed].v) : 0;
}
/**
 * Raises (never lowers) a threshold until at most `maxRate` of `background` scores reach it - for a
 * background of ordinary, mostly-negative traffic whose false alarms the labelled data can't show.
 */
export function budgetThreshold(threshold, background, maxRate) {
    const sorted = Array.from(background).sort((a, b) => b - a);
    const allowed = Math.floor(maxRate * sorted.length);
    return allowed < sorted.length ? Math.max(threshold, nextUp(sorted[allowed])) : threshold;
}
