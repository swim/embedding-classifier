/** Locations to try, in order: shadow mode prefers a shadow candidate; enforce only loads the promoted artifact. */
export function loadOrder(mode, locations) {
    return [
        ...(mode === 'shadow' && locations.shadowCandidate != null ? [{ location: locations.shadowCandidate, role: 'shadow-candidate' }] : []),
        { location: locations.promoted, role: 'promoted' },
    ];
}
/**
 * Why `artifact` must not be served in `mode`, or null if it may. Enforcement requires recorded,
 * passed gates - checked at load time too, so an artifact copied into place by hand still can't act.
 */
export function refuseToServe(mode, artifact) {
    if (mode === 'enforce' && artifact.gates?.passed !== true)
        return 'its gates did not pass';
    return null;
}
/**
 * What publishing may do with a freshly trained artifact. Returns the role to point at it (null:
 * upload the versioned artifact only), or an error.
 *   blockedReason      set when the artifact must never be published (e.g. trained on fake embeddings)
 *   allowFailingGates  upload a failing artifact for inspection, without pointing any role at it
 */
export function publishPlan(options) {
    const { gatesPassed, promote = false, shadowCandidate = false, allowFailingGates = false, blockedReason } = options;
    if (blockedReason)
        return { error: `refusing to publish: ${blockedReason}` };
    if (promote && shadowCandidate)
        return { error: 'choose one of promote or shadow-candidate' };
    if (promote)
        return gatesPassed ? { role: 'promoted' } : { error: 'refusing to promote an artifact that failed its gates (publish it as a shadow candidate to evaluate it in shadow mode)' };
    if (shadowCandidate)
        return { role: 'shadow-candidate' };
    if (!gatesPassed && !allowFailingGates)
        return { error: 'refusing to publish: gates failed (allow failing gates to upload for inspection only, or publish as a shadow candidate)' };
    return { role: null };
}
