/**
 * The trained artifact - a self-describing JSON document - and how it's scored. Training-time
 * evaluation and runtime scoring call these same functions, so what was evaluated is exactly what
 * runs.
 *
 * Per head:  logit = w·x + b  ->  calibrated p (Platt: σ(a·logit + c) | isotonic: interp(σ(logit)))
 *            where x is the embedding, or a knn or stack head's features of it (heads.ts)
 */
import { decisionFunction, predictIsotonic, sigmoid } from '@liquidau/solvers';
import { THRESHOLD_MODES } from "./conformal.js";
import { headFeatureVector, runtimeReference, similarities } from "./heads.js";
export function calibrate(calibration, logit) {
    return calibration.method === 'platt'
        ? sigmoid(calibration.a * logit + calibration.c)
        : predictIsotonic(calibration, sigmoid(logit));
}
/** A linear head's probability (knn and stack heads need the artifact's reference: use scoreEmbedding). */
export function headProbability(spec, embedding) {
    if (spec.features)
        throw new Error(`a ${spec.features.kind} head needs the artifact's reference: score it with scoreEmbedding`);
    return calibrate(spec.calibration, decisionFunction({ coef: spec.weights, intercept: spec.bias }, embedding));
}
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v);
const allFinite = (v) => Array.isArray(v) && v.every(isFiniteNumber);
/**
 * Validates an artifact loaded from storage; throws with a specific reason if it's unusable.
 * Checks values, not just shape: a corrupted or hand-edited artifact must fail loudly here rather
 * than score NaN at runtime (which decide() would otherwise read as "no decision").
 * Pass `heads` to reject head names the caller doesn't know how to act on.
 */
export function validateArtifact(raw, options = {}) {
    const a = raw;
    if (!a || typeof a !== 'object' || typeof a.version !== 'string' || !a.embedding || !a.heads || typeof a.heads !== 'object' || Array.isArray(a.heads)) {
        throw new Error('not a classifier artifact');
    }
    const dims = a.embedding.dimensions;
    if (!Number.isInteger(dims) || dims < 1)
        throw new Error(`embedding dimensions must be a positive integer, got ${dims}`);
    if (a.reference !== undefined)
        validateReference(a.reference, dims);
    for (const [name, spec] of Object.entries(a.heads)) {
        if (options.heads && !options.heads.includes(name))
            throw new Error(`unknown head ${name}`);
        const width = spec?.features ? validateFeatures(name, spec.features, dims, a.reference) : dims;
        if (!spec || !Array.isArray(spec.weights) || spec.weights.length !== width) {
            throw new Error(`head ${name} has ${spec?.weights?.length} weights but ${spec?.features ? `its ${spec.features.kind} features have` : 'the embedding has'} ${width} dimensions`);
        }
        if (!allFinite(spec.weights) || !isFiniteNumber(spec.bias))
            throw new Error(`head ${name} has non-numeric weights or bias`);
        const cal = spec.calibration;
        if (cal?.method === 'platt') {
            if (!isFiniteNumber(cal.a) || !isFiniteNumber(cal.c))
                throw new Error(`head ${name} has non-numeric Platt parameters`);
        }
        else if (cal?.method === 'isotonic') {
            const { x, y } = cal;
            if (!allFinite(x) || !allFinite(y) || x.length === 0 || x.length !== y.length) {
                throw new Error(`head ${name} has an isotonic table that is empty, non-numeric or of mismatched length`);
            }
            for (let i = 1; i < x.length; i++)
                if (x[i] < x[i - 1])
                    throw new Error(`head ${name} has isotonic x values that are not sorted`);
        }
        else {
            throw new Error(`head ${name} has unknown calibration ${cal?.method}`);
        }
        if (!isFiniteNumber(spec.threshold) || !isFiniteNumber(spec.review_floor))
            throw new Error(`head ${name} has a non-numeric threshold or review floor`);
        // No upper bound on the threshold: a background budget can push it just past 1 ("never fire").
        if (!(spec.review_floor >= 0 && spec.review_floor <= spec.threshold)) {
            throw new Error(`head ${name} needs 0 <= review_floor <= threshold (got ${spec.review_floor}, ${spec.threshold})`);
        }
        for (const [tier, t] of Object.entries(spec.thresholds ?? {}))
            if (!isFiniteNumber(t))
                throw new Error(`head ${name} has a non-numeric ${tier} threshold`);
        if (spec.guarantee !== undefined)
            validateGuarantee(name, spec.guarantee);
        const dm = spec.dismissal;
        if (dm !== undefined && !(dm && typeof dm.rule_set === 'string' && isRate(dm.max_rate) && Number.isInteger(dm.certified) && dm.certified >= 0)) {
            throw new Error(`head ${name} has an invalid dismissal record`);
        }
        if (spec.review_epsilon !== undefined && !(isFiniteNumber(spec.review_epsilon) && spec.review_epsilon > 0 && spec.review_epsilon < 1)) {
            throw new Error(`head ${name} has a review_epsilon outside (0, 1)`);
        }
    }
    return a;
}
function validateReference(ref, dims) {
    if (!ref || typeof ref !== 'object' || (ref.encoding !== 'f32' && ref.encoding !== 'int8'))
        throw new Error('reference has an unknown encoding');
    if (ref.dims !== dims)
        throw new Error(`reference has ${ref.dims} dimensions but the embedding has ${dims}`);
    if (!Number.isInteger(ref.rows) || ref.rows < 1)
        throw new Error('reference needs at least one row');
    const width = ref.encoding === 'int8' ? 1 : 4;
    if (typeof ref.data !== 'string' || ref.data.length !== 4 * Math.ceil((ref.rows * ref.dims * width) / 3))
        throw new Error('reference data has the wrong length for its rows and dimensions');
    if (ref.encoding === 'int8' && !(allFinite(ref.scales) && ref.scales.length === ref.rows && ref.scales.every((v) => v > 0)))
        throw new Error('an int8 reference needs one positive scale per row');
    if (ref.encoding === 'f32' && ref.scales !== undefined)
        throw new Error('a float32 reference has no scales');
    if (!ref.labels || typeof ref.labels !== 'object')
        throw new Error('reference has no labels');
    for (const [head, ys] of Object.entries(ref.labels)) {
        if (!Array.isArray(ys) || ys.length !== ref.rows || !ys.every((v) => v === 0 || v === 1 || v === null))
            throw new Error(`reference labels for ${head} must be 0, 1 or null, one per row`);
    }
}
/** Checks a head's feature block and returns its width. */
function validateFeatures(name, f, dims, ref) {
    if (!f || (f.kind !== 'knn' && f.kind !== 'stack'))
        throw new Error(`head ${name} has unknown features ${f?.kind}`);
    if (!Number.isInteger(f.k) || f.k < 1)
        throw new Error(`head ${name}: k must be a positive integer`);
    const ys = ref?.labels[name];
    if (!ys)
        throw new Error(`head ${name} is a ${f.kind} head but the artifact's reference has no labels for it`);
    if (!ys.includes(1) || !ys.includes(0))
        throw new Error(`head ${name}: the reference needs both positive and negative rows`);
    if (f.kind === 'knn')
        return 1;
    if (!allFinite(f.linear?.weights) || f.linear.weights.length !== dims || !isFiniteNumber(f.linear.bias))
        throw new Error(`head ${name} has an invalid stack linear component`);
    const { mean, components } = f.pca ?? {};
    if (!allFinite(mean) || mean.length !== dims || !Array.isArray(components) || components.length < 1 || components.length > dims || !components.every((c) => allFinite(c) && c.length === dims)) {
        throw new Error(`head ${name} has an invalid stack projection`);
    }
    return 3;
}
const isRate = (v) => isFiniteNumber(v) && v > 0 && v < 1;
/** A stored guarantee must be one its mode can produce, with the parameters that make it meaningful. */
function validateGuarantee(name, g) {
    if (!g || typeof g !== 'object' || !THRESHOLD_MODES.includes(g.mode))
        throw new Error(`head ${name} has an unknown threshold mode ${g?.mode}`);
    if (!isRate(g.alpha))
        throw new Error(`head ${name}: guarantee alpha must be in (0, 1)`);
    const allowed = { heuristic: ['none'], 'conformal-expected': ['expected', 'none'], 'conformal-pac': ['pac', 'none'], auto: ['pac', 'expected', 'none'], design: ['design-exact', 'design-approximate', 'none'] };
    if (!allowed[g.mode].includes(g.kind))
        throw new Error(`head ${name}: mode ${g.mode} cannot give a ${g.kind} guarantee`);
    const withDelta = g.kind === 'pac' || g.kind === 'design-exact' || g.kind === 'design-approximate';
    if (withDelta && !isRate(g.delta))
        throw new Error(`head ${name}: a ${g.kind} guarantee needs delta in (0, 1)`);
    if (!withDelta && g.delta !== undefined)
        throw new Error(`head ${name}: only pac and design guarantees have a delta`);
    for (const k of ['false_alarm', 'background_rate']) {
        if (g[k] !== undefined && (g.kind === 'none' || !isRate(g[k])))
            throw new Error(`head ${name}: guarantee ${k} must be in (0, 1) and only with a guarantee`);
    }
}
/** Calibrated probability for every head in the artifact. */
export function scoreEmbedding(artifact, embedding) {
    if (embedding.length !== artifact.embedding.dimensions) {
        throw new Error(`embedding has ${embedding.length} dimensions but the artifact expects ${artifact.embedding.dimensions}`);
    }
    for (let i = 0; i < embedding.length; i++) {
        if (!Number.isFinite(embedding[i]))
            throw new Error(`embedding has a non-finite value at index ${i}`);
    }
    const scores = {};
    // knn and stack heads share one pass over the reference per message.
    let ref, sims;
    for (const [head, spec] of Object.entries(artifact.heads)) {
        if (!spec)
            continue;
        if (!spec.features) {
            scores[head] = headProbability(spec, embedding);
            continue;
        }
        if (!artifact.reference)
            throw new Error(`head ${head} is a ${spec.features.kind} head but the artifact has no reference`);
        ref ??= runtimeReference(artifact.reference);
        sims ??= similarities(embedding, ref.rows);
        const x = headFeatureVector(head, spec.features, embedding, ref, sims);
        scores[head] = calibrate(spec.calibration, decisionFunction({ coef: spec.weights, intercept: spec.bias }, x));
    }
    return scores;
}
/**
 * Heads certified with a different dismissal rule set than the one the rules tier runs (`ruleSet`:
 * its identity, e.g. rule-miner's ruleSetHash). Each such head's guarantee counted another rule set's
 * misses, so it doesn't describe the system: refuse to serve while this is non-empty.
 */
export function checkRuleSetPairing(artifact, ruleSet) {
    return Object.entries(artifact.heads)
        .filter(([, spec]) => spec?.dismissal && spec.dismissal.rule_set !== ruleSet)
        .map(([head, spec]) => `${head} was certified with rule set ${spec.dismissal.rule_set}, not ${ruleSet}`);
}
