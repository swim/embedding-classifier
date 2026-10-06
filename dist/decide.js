/**
 * Policy heads that the artifact doesn't contain. decide() deliberately allows these (an artifact may
 * not ship every head yet), but a missing head silently never fires and never suppresses - so a
 * suppression rule whose guard head is absent guards nothing. Call this once at startup and refuse
 * to serve, or log, if the result isn't what you expect.
 */
export function missingPolicyHeads(artifact, policy) {
    const named = new Set([...policy.priority, ...(policy.suppress ?? []).flatMap((r) => [...r.when, ...r.heads])]);
    return [...named].filter((h) => artifact.heads[h] === undefined);
}
/**
 * Throws if a head the artifact contains has a missing or non-finite score, or one outside [0, 1]: that
 * is a scoring bug or a corrupted artifact, not a negative. `dismissed`: heads a certified dismissal rule cleared for this message (rule-miner's
 * matcher.evaluate); they need no score, never fire and never suppress, as their training counted.
 */
export function decide(artifact, scores, policy, dismissed = [], fired = null) {
    // `fired`: a certified firing rule matched this head ("rules or classifier"): it counts as at its threshold.
    const atLeast = (head, level) => {
        if (head === fired && !dismissed.includes(head))
            return true;
        const spec = artifact.heads[head];
        if (spec === undefined || dismissed.includes(head))
            return false;
        const score = scores[head];
        if (typeof score !== 'number' || !Number.isFinite(score))
            throw new Error(`head ${head} has no finite score (${score})`);
        if (score < 0 || score > 1)
            throw new Error(`head ${head} has a score outside [0, 1] (${score}): not a calibrated probability`);
        return score >= spec[level];
    };
    const suppressed = new Set();
    for (const rule of policy.suppress ?? []) {
        if (rule.when.some((h) => atLeast(h, 'review_floor')))
            rule.heads.forEach((h) => suppressed.add(h));
    }
    for (const head of policy.priority) {
        if (!suppressed.has(head) && atLeast(head, 'threshold'))
            return { head, reason: head === fired ? 'rule' : 'above_threshold' };
    }
    if (policy.priority.some((h) => atLeast(h, 'review_floor')))
        return { head: null, reason: 'near_threshold' };
    return { head: null, reason: 'none' };
}
/**
 * The rules tier's decision, without the model: settled when the rules alone fix what `decide` would
 * return whatever the scores - every head the policy names is dismissed, or a firing rule's head
 * comes after only dismissed heads in priority and nothing undismissed can suppress it. Otherwise the
 * message goes to the model tier with what the rules found, for decide(artifact, scores, policy,
 * forward.dismissed, forward.fired). Both paths give the same decision (tested), so the guarantees
 * describe the system whichever path a message takes.
 */
export function settleWithRules(policy, evaluation) {
    const named = new Set([...policy.priority, ...(policy.suppress ?? []).flatMap((r) => [...r.when, ...r.heads])]);
    const dismissed = evaluation.dismissed.filter((h) => named.has(h));
    const off = new Set(dismissed);
    const fired = evaluation.fired && named.has(evaluation.fired.label) && !off.has(evaluation.fired.label) ? evaluation.fired.label : null;
    if (fired) {
        const before = policy.priority.slice(0, Math.max(0, policy.priority.indexOf(fired)));
        const canSuppress = (policy.suppress ?? []).some((r) => r.heads.includes(fired) && r.when.some((w) => !off.has(w)));
        if (policy.priority.includes(fired) && before.every((h) => off.has(h)) && !canSuppress)
            return { settled: true, decision: { head: fired, reason: 'rule' } };
    }
    if (policy.priority.every((h) => off.has(h)))
        return { settled: true, decision: { head: null, reason: 'none' } };
    return { settled: false, forward: { fired, dismissed } };
}
