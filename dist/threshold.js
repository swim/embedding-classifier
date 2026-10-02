export function pickThreshold(policy, p, y) {
    if (policy.kind === 'precision')
        return policy.targetPrecision;
    if (policy.designRecall < policy.targetRecall)
        throw new Error('designRecall must be >= targetRecall');
    const scored = Array.from(p);
    const positives = scored.filter((_, i) => y[i] === 1).sort((a, b) => b - a);
    if (!positives.length)
        throw new Error('no positives to choose a recall threshold from');
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
export function budgetThreshold(threshold, background, maxRate) {
    const sorted = Array.from(background).sort((a, b) => b - a);
    const allowed = Math.floor(maxRate * sorted.length);
    return allowed < sorted.length ? Math.max(threshold, nextUp(sorted[allowed])) : threshold;
}
/** Smallest double strictly greater than v, so `p >= threshold` excludes v itself. */
export function nextUp(v) {
    if (Number.isNaN(v) || v === Infinity)
        return v;
    if (v === 0)
        return Number.MIN_VALUE;
    const buf = new DataView(new ArrayBuffer(8));
    buf.setFloat64(0, v);
    const bits = buf.getBigUint64(0);
    buf.setBigUint64(0, v > 0 ? bits + 1n : bits - 1n);
    return buf.getFloat64(0);
}
