/**
 * The trained artifact - a self-describing JSON document - and how it's scored. Training-time
 * evaluation and runtime scoring call these same functions, so what was evaluated is exactly what
 * runs.
 *
 * Per head:  logit = w·x + b  ->  calibrated p (Platt: σ(a·logit + c) | isotonic: interp(σ(logit)))
 */
import { decisionFunction, predictIsotonic, sigmoid } from '@liquidau/solvers';
import { THRESHOLD_MODES } from "./conformal.js";
export function calibrate(calibration, logit) {
    return calibration.method === 'platt'
        ? sigmoid(calibration.a * logit + calibration.c)
        : predictIsotonic(calibration, sigmoid(logit));
}
export function headProbability(spec, embedding) {
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
    for (const [name, spec] of Object.entries(a.heads)) {
        if (options.heads && !options.heads.includes(name))
            throw new Error(`unknown head ${name}`);
        if (!spec || !Array.isArray(spec.weights) || spec.weights.length !== dims) {
            throw new Error(`head ${name} has ${spec?.weights?.length} weights but the embedding has ${dims} dimensions`);
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
        if (spec.review_epsilon !== undefined && !(isFiniteNumber(spec.review_epsilon) && spec.review_epsilon > 0 && spec.review_epsilon < 1)) {
            throw new Error(`head ${name} has a review_epsilon outside (0, 1)`);
        }
    }
    return a;
}
const isRate = (v) => isFiniteNumber(v) && v > 0 && v < 1;
/** A stored guarantee must be one its mode can produce, with the parameters that make it meaningful. */
function validateGuarantee(name, g) {
    if (!g || typeof g !== 'object' || !THRESHOLD_MODES.includes(g.mode))
        throw new Error(`head ${name} has an unknown threshold mode ${g?.mode}`);
    if (!isRate(g.alpha))
        throw new Error(`head ${name}: guarantee alpha must be in (0, 1)`);
    const allowed = { heuristic: ['none'], 'conformal-expected': ['expected', 'none'], 'conformal-pac': ['pac', 'none'], auto: ['pac', 'expected', 'none'] };
    if (!allowed[g.mode].includes(g.kind))
        throw new Error(`head ${name}: mode ${g.mode} cannot give a ${g.kind} guarantee`);
    if (g.kind === 'pac' && !isRate(g.delta))
        throw new Error(`head ${name}: a pac guarantee needs delta in (0, 1)`);
    if (g.kind !== 'pac' && g.delta !== undefined)
        throw new Error(`head ${name}: only a pac guarantee has a delta`);
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
    for (const [head, spec] of Object.entries(artifact.heads)) {
        if (spec)
            scores[head] = headProbability(spec, embedding);
    }
    return scores;
}
