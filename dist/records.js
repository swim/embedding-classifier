export class ProvenanceError extends Error {
    code;
    ids;
    constructor(code, ids, message) {
        super(`${code}: ${message} (${ids.length} record(s): ${ids.slice(0, 5).join(', ')}${ids.length > 5 ? ', ...' : ''})`);
        this.name = 'ProvenanceError';
        this.code = code;
        this.ids = ids;
    }
}
const realKinds = new Set(['sampled', 'traffic', 'retrieved']);
export const isReal = (r) => realKinds.has(r.source.kind);
function unit(v) {
    let sq = 0;
    for (let j = 0; j < v.length; j++)
        sq += v[j] * v[j];
    const norm = Math.sqrt(sq) || 1;
    return Float64Array.from(v, (x) => x / norm);
}
const dot = (a, b) => {
    let s = 0;
    for (let j = 0; j < a.length; j++)
        s += a[j] * b[j];
    return s;
};
/** Checks P1-P5 and P7 over every record a training run uses. Throws ProvenanceError unless overridden. */
export function validateProvenance(records, options = {}) {
    const { embeddings, nearDuplicate = 0.95, safetyCritical = [], overrides = [] } = options;
    const accepted = new Set(options.acceptedBatches ?? []);
    if (embeddings && embeddings.length !== records.length)
        throw new Error(`embeddings has ${embeddings.length} entries for ${records.length} records`);
    const result = { records: [], dropped: [], warnings: [], overridden: [] };
    const violate = (code, ids, message) => {
        if (!ids.length)
            return;
        if (overrides.includes(code))
            result.overridden.push({ code, ids, message });
        else
            throw new ProvenanceError(code, ids, message);
    };
    const ids = new Set();
    for (const r of records) {
        if (ids.has(r.id))
            throw new Error(`duplicate record id ${r.id}`);
        ids.add(r.id);
    }
    violate('P1', records.filter((r) => (r.role === 'calibration' || r.role === 'test') && !(r.source.kind === 'sampled' && r.labelledBy === 'human' && r.source.inclusionProb > 0 && r.source.inclusionProb <= 1)).map((r) => r.id), 'calibration and test records must be probability-sampled and human-labelled');
    violate('P2', records.filter((r) => r.role === 'background' && !(r.source.kind === 'traffic' && Object.values(r.labels).every((v) => v === null || v === undefined) && r.backgroundUse !== undefined)).map((r) => r.id), 'background records must be unlabelled real traffic with a backgroundUse');
    const roleOf = new Map();
    const useOf = new Map();
    for (const r of records) {
        (roleOf.get(r.group) ?? roleOf.set(r.group, new Set()).get(r.group)).add(r.role);
        if (r.role === 'background')
            (useOf.get(r.group) ?? useOf.set(r.group, new Set()).get(r.group)).add(r.backgroundUse ?? '');
    }
    const split = new Set([...roleOf].filter(([, s]) => s.size > 1).map(([g]) => g));
    const splitUse = new Set([...useOf].filter(([, s]) => s.size > 1).map(([g]) => g));
    violate('P3', records.filter((r) => split.has(r.group) || splitUse.has(r.group)).map((r) => r.id), 'a group must appear in one role (and one backgroundUse) only');
    violate('P4', records.filter((r) => (r.source.kind === 'retrieved' || r.source.kind === 'generated') && r.role !== 'train' && r.role !== 'stress').map((r) => r.id), 'retrieved and generated records may be train or stress only');
    const unverifiedOk = (r) => r.source.kind === 'generated' && accepted.has(r.source.batchId) && !safetyCritical.some((h) => r.labels[h] !== null && r.labels[h] !== undefined);
    violate('P7', records.filter((r) => r.source.kind === 'generated' && r.verified !== true && !unverifiedOk(r)).map((r) => r.id), `generated records need verified === true, or an accepted batch${safetyCritical.length ? ` (and verification for safety-critical heads: ${safetyCritical.join(', ')})` : ''}`);
    const keep = records.map(() => true);
    if (embeddings) {
        const evaluation = records.flatMap((r, i) => ((r.role === 'calibration' || r.role === 'test') && embeddings[i] ? [unit(embeddings[i])] : []));
        records.forEach((r, i) => {
            if (r.role !== 'train' || (r.source.kind !== 'retrieved' && r.source.kind !== 'generated') || !embeddings[i])
                return;
            const v = unit(embeddings[i]);
            if (evaluation.length && v.length !== evaluation[0].length)
                throw new Error(`record ${r.id}: embedding has ${v.length} dimensions, evaluation records ${evaluation[0].length}`);
            const near = evaluation.find((e) => dot(e, v) >= nearDuplicate);
            if (near) {
                keep[i] = false;
                result.dropped.push({ id: r.id, code: 'P5', reason: `within cosine ${nearDuplicate} of a calibration or test record` });
            }
        });
        if (result.dropped.length)
            result.warnings.push(`P5: dropped ${result.dropped.length} ${result.dropped.length === 1 ? 'record' : 'records'} near-duplicating calibration or test records`);
    }
    result.records = records.filter((_, i) => keep[i]);
    for (const o of result.overridden)
        result.warnings.push(`${o.code} overridden for ${o.ids.length} record(s): ${o.message}`);
    return result;
}
/**
 * P6: training weights for one head's train records (record.weight, default 1), with retrieved
 * positives and generated hard negatives scaled down proportionally to their caps. Returned
 * weights align with `records`.
 */
export function capTrainingWeights(records, head, caps = {}) {
    const { maxRetrievedPositiveShare = 0.5, maxGeneratedNegativeShare = 0.3, ruleMatches } = caps;
    for (const [name, v] of [['maxRetrievedPositiveShare', maxRetrievedPositiveShare], ['maxGeneratedNegativeShare', maxGeneratedNegativeShare]]) {
        if (!(v >= 0 && v < 1))
            throw new Error(`${name} must be in [0, 1), got ${v}`);
    }
    const weights = records.map((r) => {
        const w = r.weight ?? 1;
        if (!(Number.isFinite(w) && w >= 0))
            throw new Error(`record ${r.id} has weight ${w}`);
        return w;
    });
    const label = (r) => r.labels[head];
    const sum = (pred) => records.reduce((s, r, i) => s + (pred(r) ? weights[i] : 0), 0);
    const isRetrievedPos = (r) => r.source.kind === 'retrieved' && label(r) === 1;
    const isGeneratedNeg = (r) => r.source.kind === 'generated' && label(r) === 0;
    // share = x / (x + other) <= s  <=>  x <= other · s / (1 - s)
    const capTo = (pred, otherPred, share) => {
        const x = sum(pred), limit = (sum(otherPred) * share) / (1 - share);
        const scale = x > limit ? (x ? limit / x : 1) : 1;
        records.forEach((r, i) => { if (pred(r))
            weights[i] *= scale; });
        return { weight_before: x, weight: x * scale, scale };
    };
    const retrieved_positive = capTo(isRetrievedPos, (r) => label(r) === 1 && !isRetrievedPos(r), maxRetrievedPositiveShare);
    const generated_negative = capTo(isGeneratedNeg, (r) => label(r) === 0 && !isGeneratedNeg(r), maxGeneratedNegativeShare);
    const summary = { retrieved_positive, generated_negative };
    if (records.some(isGeneratedNeg)) {
        if (!ruleMatches)
            throw new ProvenanceError('P6', records.filter(isGeneratedNeg).map((r) => r.id), 'generated hard negatives need ruleMatches for the per-rule balance cap');
        const matched = new Map();
        records.forEach((r, i) => {
            if (label(r) === 1)
                for (const rule of new Set(ruleMatches(r.text)))
                    matched.set(rule, (matched.get(rule) ?? 0) + weights[i]);
        });
        summary.by_rule = {};
        const rules = new Set(records.filter(isGeneratedNeg).map((r) => r.source.ruleId));
        for (const rule of rules) {
            const of = (r) => isGeneratedNeg(r) && r.source.ruleId === rule;
            const before = sum(of), limit = matched.get(rule) ?? 0;
            const scale = before > limit ? (before ? limit / before : 1) : 1;
            records.forEach((r, i) => { if (of(r))
                weights[i] *= scale; });
            summary.by_rule[rule] = { weight_before: before, weight: before * scale, matched_positive_weight: limit };
        }
        summary.generated_negative.weight = sum(isGeneratedNeg);
    }
    return { weights, summary };
}
