/**
 * The middle tier ('liquidau-middle/1'): a cheaper encoder's linear head that fires and dismisses
 * between the edge and the main encoder's head, per head, only where training says it pays (lab
 * stages CC and DY, 10-11 October: edge → mpnet → EmbeddingGemma).
 *
 *   model       a linear head on the middle encoder's embedding, trained by the host with trainHeads
 *               (so P5/P6 and the head's C apply); its raw logit is the middle score
 *   fire        score > fire: CUMULATIVE fires (the edge's plus the middle's) ≤ maxFalseAlarms of
 *               negatives (default 1%)
 *   dismiss     score < dismiss: CUMULATIVE dismissals ≤ maxDismissed of positives (default the edge's
 *               share + 0.25 × (1 − target))
 *
 * Both thresholds are certified on the calibration sample with the edge's decisions pinned (edge-fired
 * rows count as fired, edge-dismissed as dismissed), and the middle decides only rows the edge left.
 * The main head is then certified jointly with both tiers (HeadInput.edge semantics: decided rows score
 * 0 or 1), so the head's guarantee covers the whole pipeline.
 *
 * The per-head choice (chooseCascade, run inside trainHeads through HeadInput.cascade) uses the training
 * rows only: every tier's thresholds re-derived on them from cross-fitted scores, then false alarms at
 * target + reserve for edge → main against edge → middle → main. The cascade is kept when its cost is at
 * most 1.1 × the two-tier cost + 0.001: the rule fixed before stage DY's results. It was validated for
 * recall heads behind a single edge candidate, so that is all trainHeads accepts.
 *
 * Caveat (as for the edge): the edge, middle and head thresholds are each certified at δ on the same
 * calibration rows, chained; the joint confidence is below 1 − δ without a split or a union bound.
 */
import { decisionFunction, designRiskThreshold, fitLogistic } from '@liquidau/solvers';
import { canonicalDigest } from '@liquidau/text-preprocessing';
import { validateEmbeddingSpec } from "./artifact.js";
export const MIDDLE_SCHEMA = 'liquidau-middle/1';
/** Score and decision per head of a middle document, for one message's middle-encoder vector. */
export function scoreMiddle(doc, vector) {
    if (vector.length !== doc.embedding.dimensions)
        throw new Error(`the middle vector has ${vector.length} values, the middle encoder ${doc.embedding.dimensions}`);
    for (let j = 0; j < vector.length; j++)
        if (!Number.isFinite(vector[j]))
            throw new Error(`the middle vector has a non-finite value at ${j}`);
    const out = {};
    for (const [head, m] of Object.entries(doc.heads)) {
        const score = decisionFunction(m.model, vector);
        out[head] = { score, decision: m.fire !== null && score > m.fire ? 'fire' : m.dismiss !== null && score < m.dismiss ? 'dismiss' : null };
    }
    return out;
}
/** sha256 of the canonical middle document. */
export function middleDigest(doc) {
    return canonicalDigest(doc);
}
/** sha256 of one head's middle model (thresholds and claims included) with the document's encoder identity. */
export function middleHeadDigest(doc, head) {
    const m = doc.heads[head];
    return m ? canonicalDigest({ embedding: doc.embedding, head: m }) : null;
}
const isObject = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const rate = (v) => finite(v) && v >= 0 && v < 1;
/** Validates a parsed middle document; throws with every problem found. */
export function validateMiddle(raw) {
    if (!isObject(raw))
        throw new Error('the middle document is not an object');
    if (raw.schema !== MIDDLE_SCHEMA)
        throw new Error(`unsupported middle schema ${JSON.stringify(raw.schema)} (this library reads '${MIDDLE_SCHEMA}')`);
    const p = [];
    for (const k of Object.keys(raw))
        if (!['schema', 'embedding', 'heads'].includes(k))
            p.push(`unknown field ${k}`);
    const e = raw.embedding;
    try {
        if (!isObject(e))
            throw new Error('embedding must be an object');
        if (e.context !== undefined)
            throw new Error('embedding: a middle encoder scores the message only (no context)');
        validateEmbeddingSpec(e, 'middle embedding');
        if (typeof e.precision !== 'string' || !e.precision)
            throw new Error('middle embedding: precision is required (it is part of the encoder identity)');
    }
    catch (err) {
        p.push(err.message);
    }
    const dims = isObject(e) && Number.isInteger(e.dimensions) ? e.dimensions : -1;
    if (!isObject(raw.heads) || !Object.keys(raw.heads).length)
        p.push('heads must be a non-empty object');
    else
        for (const [head, m] of Object.entries(raw.heads)) {
            if (!isObject(m)) {
                p.push(`head ${head} is not an object`);
                continue;
            }
            for (const k of Object.keys(m))
                if (!['model', 'fire', 'dismiss', 'claims'].includes(k))
                    p.push(`head ${head}: unknown field ${k}`);
            const lm = m.model;
            if (!(isObject(lm) && Array.isArray(lm.coef) && lm.coef.length === dims && lm.coef.every(finite) && finite(lm.intercept)))
                p.push(`head ${head}: model must be a linear model over the middle embedding's ${dims} dimensions`);
            for (const k of ['fire', 'dismiss'])
                if (m[k] !== null && !finite(m[k]))
                    p.push(`head ${head}: ${k} must be a number or null`);
            if (finite(m.fire) && finite(m.dismiss) && m.dismiss > m.fire)
                p.push(`head ${head}: dismiss must not exceed fire`);
            if (m.fire === null && m.dismiss === null)
                p.push(`head ${head}: decides nothing (fire and dismiss are both null)`);
            const c = m.claims;
            if (!(isObject(c) && rate(c.max_false_alarms) && rate(c.max_dismissed) && finite(c.delta) && c.delta > 0 && c.delta < 1 && typeof c.prior_edge_sha256 === 'string' && /^[0-9a-f]{64}$/.test(c.prior_edge_sha256))) {
                p.push(`head ${head}: claims must be { max_false_alarms, max_dismissed, delta, prior_edge_sha256 }`);
            }
        }
    if (p.length)
        throw new Error(`the middle document is invalid: ${p.join('; ')}`);
    return raw;
}
// ---------- the training-row choice (lab stage DY-C, liquidau-lab src/harness/dynamic-select.ts) ----------
const CROSS_FOLDS = 5;
const PIN = 1e9;
/** The pre-registered rule: keep the cascade when its training cost ≤ 1.1 × the two-tier cost + 0.001. */
export const CASCADE_TOLERANCE = { relative: 1.1, absolute: 0.001 };
/** The middle's default extra dismissal share: 0.25 × the head's miss budget (1 − target). */
export const MIDDLE_EXTRA_DISMISSAL = 0.25;
/** The middle's default cumulative firing budget: 1% of negatives. */
export const MIDDLE_MAX_FALSE_ALARMS = 0.01;
const fnv5 = (key) => { let h = 2166136261; for (let k = 0; k < key.length; k++)
    h = Math.imul(h ^ key.charCodeAt(k), 16777619); return (h >>> 0) % CROSS_FOLDS; };
/**
 * Cross-fitted logistic decision scores (5 folds by FNV of the key, C = 1, balanced, unweighted): the
 * choice's scorer for the middle and main heads, as the lab validated the rule. It ignores the head's
 * own C, standardisation and record weights, so the choice describes a close relative of the shipped
 * head, not that head exactly. A fold whose fit rows lack a class scores 0.
 */
export function crossFitScores(X, y, keys) {
    if (X.length !== y.length || keys.length !== y.length)
        throw new Error('crossFitScores: X, y and keys must have the same length');
    const fold = keys.map(fnv5), out = new Array(X.length).fill(0);
    for (let f = 0; f < CROSS_FOLDS; f++) {
        const fit = X.map((_, k) => k).filter((k) => fold[k] !== f), held = X.map((_, k) => k).filter((k) => fold[k] === f);
        if (!held.length || new Set(fit.map((k) => y[k])).size < 2)
            continue;
        const m = fitLogistic(fit.map((k) => X[k]), fit.map((k) => y[k]), { C: 1, classWeight: 'balanced' });
        for (const k of held)
            out[k] = decisionFunction(m, X[k]);
    }
    return out;
}
/** The lowest score above which at most maxRate of the negatives fall (fire strictly above; null without negatives). */
function fireAt(scores, y, maxRate) {
    const neg = scores.filter((_, k) => y[k] === 0).sort((a, b) => b - a);
    return neg.length ? neg[Math.floor(maxRate * neg.length)] ?? null : null;
}
/** The highest score below which at most maxShare of the positives fall (dismiss strictly below; null when none may be). */
function dismissAt(scores, y, maxShare) {
    const pos = scores.filter((_, k) => y[k] === 1).sort((a, b) => a - b);
    const allowed = Math.floor(maxShare * pos.length);
    return allowed > 0 ? pos[allowed] : null;
}
/** False-alarm rate at recall ≥ target, never splitting tied scores (1 when the target is unreachable). */
export function cascadeCost(scores, y, target) {
    const order = scores.map((s, k) => [s, y[k]]).sort((a, b) => b[0] - a[0]);
    const pos = y.filter((v) => v === 1).length, neg = y.length - pos;
    if (!pos || !neg)
        return 1;
    let tp = 0, fp = 0;
    for (let k = 0; k < order.length; k++) {
        if (order[k][1])
            tp++;
        else
            fp++;
        if (k + 1 < order.length && order[k + 1][0] === order[k][0])
            continue;
        if (tp / pos >= target)
            return fp / neg;
    }
    return 1;
}
/** The per-head choice, from training rows only (lab stage DY-C's rule, reproduced exactly). */
export function chooseCascade(input) {
    const { y, edge, middle, head, budgets: b } = input;
    const n = y.length;
    if (edge.length !== n || middle.length !== n || head.length !== n)
        throw new Error('chooseCascade: y and every score array must have the same length');
    const pin = (d, v) => (d === 'fire' ? PIN : d === 'dismiss' ? -PIN : v);
    const etf = fireAt(edge, y, b.edgeFire), etd = dismissAt(edge, y, b.edgeDismiss);
    const eD = edge.map((v) => (etf !== null && v > etf ? 'fire' : etd !== null && v < etd ? 'dismiss' : null));
    const pinned = middle.map((v, k) => pin(eD[k], v));
    // A threshold that lands on a pinned row (the budget reaches past every row the edge left) decides
    // nothing, exactly as trainMiddle maps it at certification.
    const unpinned = (t) => (t !== null && Math.abs(t) < PIN / 2 ? t : null);
    const mtf = b.middleFire > b.edgeFire ? unpinned(fireAt(pinned, y, b.middleFire)) : null, mtd = unpinned(dismissAt(pinned, y, b.middleDismiss));
    const mD = middle.map((v, k) => (eD[k] ? null : mtf !== null && v > mtf ? 'fire' : mtd !== null && v < mtd ? 'dismiss' : null));
    const target = Math.min(0.99, input.targetRecall + (input.reserve ?? 0));
    const twoTier = cascadeCost(head.map((v, k) => pin(eD[k], v)), y, target);
    const cascade = cascadeCost(head.map((v, k) => pin(eD[k] ?? mD[k], v)), y, target);
    return { cascade: cascade <= CASCADE_TOLERANCE.relative * twoTier + CASCADE_TOLERANCE.absolute, costs: { twoTier, cascade } };
}
/** Certifies each head's middle thresholds behind its edge candidate. Recall heads only. */
export function trainMiddle(input) {
    const { vectors, split, groups, design, delta = 0.05 } = input;
    const n = vectors.length;
    validateEmbeddingSpec(input.embedding, 'middle embedding');
    if (input.embedding.context !== undefined)
        throw new Error('middle embedding: a middle encoder scores the message only (no context)');
    if (typeof input.embedding.precision !== 'string' || !input.embedding.precision)
        throw new Error('middle embedding: precision is required');
    if (split.length !== n)
        throw new Error(`split has ${split.length} rows, vectors ${n}`);
    if (groups && groups.length !== n)
        throw new Error(`groups has ${groups.length} rows, vectors ${n}`);
    const dims = input.embedding.dimensions;
    vectors.forEach((v, i) => { if (v.length !== dims)
        throw new Error(`vectors[${i}] has ${v.length} values, the middle embedding ${dims}`); });
    input.background?.vectors.forEach((v, i) => { if (v.length !== dims)
        throw new Error(`background.vectors[${i}] has ${v.length} values, the middle embedding ${dims}`); });
    const calRows = split.map((s, i) => (s === 'calibration' ? i : -1)).filter((i) => i >= 0);
    if (design && design.inclusionProbs.length !== calRows.length)
        throw new Error(`design has ${design.inclusionProbs.length} rows for ${calRows.length} calibration rows`);
    const calDesign = design ?? { inclusionProbs: calRows.map(() => 1e-9), strata: calRows.map(() => 'all'), stratumSizes: { all: calRows.length * 1e9 } };
    const warnings = [];
    const doc = { schema: MIDDLE_SCHEMA, embedding: { ...input.embedding }, heads: {} };
    const heads = {};
    for (const h of input.heads) {
        if (h.y.length !== n)
            throw new Error(`${h.name}: y has ${h.y.length} rows, vectors ${n}`);
        if (!(h.targetRecall > 0 && h.targetRecall < 1))
            throw new Error(`${h.name}: targetRecall must be in (0, 1)`);
        if (h.model.coef.length !== dims || !Array.from(h.model.coef).every(Number.isFinite) || !Number.isFinite(h.model.intercept))
            throw new Error(`${h.name}: model must be a finite linear model over ${dims} dimensions`);
        const er = h.edge.rows;
        if (er.fired.length !== n || er.dismissed.length !== n)
            throw new Error(`${h.name}: the edge rows must have one entry per row`);
        if (input.background && !er.background)
            throw new Error(`${h.name}: background vectors were given, but the edge candidate has no background rows`);
        const maxFA = h.maxFalseAlarms ?? MIDDLE_MAX_FALSE_ALARMS;
        const maxDis = h.maxDismissed ?? h.edge.maxDismissed + MIDDLE_EXTRA_DISMISSAL * (1 - h.targetRecall);
        if (!(maxFA >= h.edge.model.claims.max_false_alarms && maxFA < 1))
            throw new Error(`${h.name}: maxFalseAlarms (cumulative) must be at least the edge's ${h.edge.model.claims.max_false_alarms} (equal: the middle only dismisses) and below 1`);
        /** A cumulative firing budget equal to the edge's leaves the middle nothing to fire: it only dismisses. */
        const fires = maxFA > h.edge.model.claims.max_false_alarms;
        if (!(maxDis >= h.edge.maxDismissed && maxDis < 1))
            throw new Error(`${h.name}: maxDismissed (cumulative) must be at least the edge's ${h.edge.maxDismissed} and below 1`);
        const linearModel = { coef: Array.from(h.model.coef), intercept: h.model.intercept };
        const score = (x) => decisionFunction(linearModel, x);
        const prior = (i) => (er.fired[i] ? 'fire' : er.dismissed[i] ? 'dismiss' : null);
        // Certify on the calibration rows with the edge's decisions pinned: its fires count as fired, its dismissals as dismissed.
        const keep = calRows.map((i) => h.y[i] === 0 || h.y[i] === 1);
        const sub = (v) => v.filter((_, k) => keep[k]);
        const base = { inclusionProbs: sub(calDesign.inclusionProbs), strata: sub(calDesign.strata), stratumSizes: calDesign.stratumSizes };
        const yc = sub(calRows.map((i) => h.y[i]));
        const sc = sub(calRows.map((i) => { const d = prior(i); return d === 'fire' ? PIN : d === 'dismiss' ? -PIN : score(vectors[i]); }));
        let fire = null, dismiss = null;
        if (yc.includes(0) && yc.includes(1)) {
            if (fires) {
                const f = designRiskThreshold({ ...base, y: yc.map((v) => (1 - v)), scores: sc.map((v) => -v), alpha: maxFA, delta, method: 'linearised', minStratumPositives: 0 });
                if (f.feasible)
                    fire = Math.abs(f.threshold) < PIN / 2 ? -f.threshold : null;
                else
                    warnings.push(`${h.name}: no middle fire threshold keeps cumulative false alarms ≤ ${maxFA} (${f.reason})`);
            }
            const d = designRiskThreshold({ ...base, y: yc, scores: sc, alpha: maxDis, delta, method: 'linearised', minStratumPositives: 0 });
            if (d.feasible)
                dismiss = Math.abs(d.threshold) < PIN / 2 ? d.threshold : null;
            else
                warnings.push(`${h.name}: no middle dismiss threshold keeps cumulative dismissals ≤ ${maxDis} (${d.reason})`);
        }
        else
            warnings.push(`${h.name}: the calibration rows need both classes; the middle decides nothing`);
        if (fire !== null && dismiss !== null && dismiss > fire)
            dismiss = fire;
        if (fire === null && dismiss === null) {
            warnings.push(`${h.name}: the middle decides nothing behind this edge; no middle model`);
            continue;
        }
        const model = { model: linearModel, fire, dismiss, claims: { max_false_alarms: maxFA, max_dismissed: maxDis, delta, prior_edge_sha256: h.edge.sha256 } };
        const decide = (v) => (fire !== null && v > fire ? 'fire' : dismiss !== null && v < dismiss ? 'dismiss' : null);
        const all = vectors.map((x, i) => prior(i) ?? decide(score(x)));
        const bg = input.background && er.background ? input.background.vectors.map((x, j) => (er.background.fired[j] ? 'fire' : er.background.dismissed[j] ? 'dismiss' : decide(score(x)))) : undefined;
        const rows = { fired: all.map((d) => d === 'fire'), dismissed: all.map((d) => d === 'dismiss'), ...(bg ? { background: { fired: bg.map((d) => d === 'fire'), dismissed: bg.map((d) => d === 'dismiss') } } : {}) };
        // The choice's middle scores: cross-fitted on the training rows (the lab's recipe).
        const tr = split.map((s, i) => (s === 'train' && (h.y[i] === 0 || h.y[i] === 1) ? i : -1)).filter((i) => i >= 0);
        const trainScores = new Array(n).fill(NaN);
        const cf = crossFitScores(tr.map((i) => vectors[i]), tr.map((i) => h.y[i]), tr.map((i) => (groups ? groups[i] : String(i))));
        tr.forEach((i, k) => (trainScores[i] = cf[k]));
        doc.heads[h.name] = model;
        heads[h.name] = {
            model, rows, sha256: middleHeadDigest(doc, h.name), trainScores,
            budgets: { edgeFire: h.edge.model.claims.max_false_alarms, edgeDismiss: h.edge.maxDismissed, middleFire: maxFA, middleDismiss: maxDis },
        };
    }
    return { middle: doc, heads, warnings };
}
/** The middle document with only the heads that kept the cascade (trainHeads' result.cascadeChoice). */
export function assembleMiddle(result, cascadeChoice) {
    const heads = {};
    for (const [h, c] of Object.entries(cascadeChoice)) {
        if (!c?.cascade)
            continue;
        const m = result.middle.heads[h];
        if (!m)
            throw new Error(`${h} kept the cascade, but the middle result has no model for it`);
        heads[h] = m;
    }
    return Object.keys(heads).length ? { ...result.middle, heads } : null;
}
/** HeadInput.cascade for one head, from trainEdge's and trainMiddle's results (null when the middle has no model for it). */
export function cascadeInput(edge, middle, head, choice) {
    const m = middle.heads[head], scores = edge.trainScores[head];
    if (!m)
        return null;
    if (!scores)
        throw new Error(`${head}: the edge result has no training scores for it`);
    if (m.model.claims.prior_edge_sha256 !== edge.rungs[head]?.[0]?.sha256)
        throw new Error(`${head}: the middle model was certified behind another edge candidate than the edge result's first`);
    return { rows: m.rows, middleSha256: m.sha256, scores: { edge: scores, middle: m.trainScores }, budgets: m.budgets, ...(choice ? { choice } : {}) };
}
