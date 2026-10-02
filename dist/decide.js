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
/** Throws if a head the artifact contains has a missing or non-finite score: that is a scoring bug, not a negative. */
export function decide(artifact, scores, policy) {
    const atLeast = (head, level) => {
        const spec = artifact.heads[head];
        if (spec === undefined)
            return false;
        const score = scores[head];
        if (typeof score !== 'number' || !Number.isFinite(score))
            throw new Error(`head ${head} has no finite score (${score})`);
        return score >= spec[level];
    };
    const suppressed = new Set();
    for (const rule of policy.suppress ?? []) {
        if (rule.when.some((h) => atLeast(h, 'review_floor')))
            rule.heads.forEach((h) => suppressed.add(h));
    }
    for (const head of policy.priority) {
        if (!suppressed.has(head) && atLeast(head, 'threshold'))
            return { head, reason: 'above_threshold' };
    }
    if (policy.priority.some((h) => atLeast(h, 'review_floor')))
        return { head: null, reason: 'near_threshold' };
    return { head: null, reason: 'none' };
}
