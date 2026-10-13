/**
 * Fit, calibrate, threshold, evaluate and gate every head - everything short of I/O.
 *
 *   train split        weighted L2 logistic regression per head (class-balanced, exact Newton solver),
 *                      plus any weak positives (e.g. from certified rules), weighted and capped
 *   calibration split  Platt or isotonic calibrator, weighted to the head's production prevalence,
 *                      then the threshold from the head's policy - heuristic or conformal (see
 *                      conformal.ts) - and optionally a conformal review floor
 *   test split         evaluateHead (with exact certified bounds) + gateHead
 *
 * With `records` (one ExampleRecord per row of X), provenance rules P1-P7 are enforced first,
 * training weights come from the records with retrieved and generated data capped (P6), and the
 * calibration and test splits are probability samples: calibration is weighted by 1/π (which
 * already reproduces production prevalence, so `prevalence` must not be passed), and the test
 * metrics become design-based estimates. The `design` threshold mode then applies.
 */
import { clopperPearsonUpper, conformalLowerThreshold, conformalRank, decisionFunction, designPrecisionThreshold, designRiskThreshold, fitIsotonic, fitLogistic, fitPlatt, minimumSamples, nextUp, prevalenceWeights, sigmoid } from '@liquidau/solvers';
import { designOf } from "./design.js";
import { capTrainingWeights, validateProvenance } from "./records.js";
import { conformalThreshold, groupScores, guaranteeChain } from "./conformal.js";
import { calibrate, scoreEmbedding, validateArtifact } from "./artifact.js";
import { evaluateHead } from "./evaluate.js";
import { CASCADE_TOLERANCE, chooseCascade, crossFitScores } from "./middle.js";
import { FINER_C, FINER_SCALE, finerFeatureVector, finerProbabilities, finerRouteRates, fitFinerModel } from "./finer.js";
import { gateHead } from "./gates.js";
import { crossValidatedScores, decodeReference, encodeReference, fitFeatures, guaranteeCost, HEAD_TYPES } from "./heads.js";
import { budgetThreshold, falseAlarmCap, pickThreshold } from "./threshold.js";
export const SPLITS = ['train', 'calibration', 'test'];
/** The edge and dismissal rules decided every calibration row, leaving the head's calibrator nothing to fit. */
export class CalibrationRemainderError extends Error {
    name = 'CalibrationRemainderError';
}
function checkLength(name, values, n) {
    if (values !== undefined && values.length !== n)
        throw new Error(`${name} has ${values.length} entries but X has ${n}`);
}
function checkedLength(name, values, n) {
    checkLength(name, values, n);
    return values;
}
function indicesBySplit(split, y) {
    const idx = { train: [], calibration: [], test: [] };
    split.forEach((s, i) => {
        if (y[i] !== null)
            idx[s].push(i);
    });
    return idx;
}
export function trainHeads(input) {
    const { X, split, C = 1, autoMargin = 0.2, requireConvergence = false, calibration: method = 'platt', reviewRatio = 0.5, reviewEpsilon, maxEce, calibrationAlpha, groups, slices, background, records, seed = 0, log = () => { } } = input;
    checkLength('split', split, X.length);
    split.forEach((s, i) => {
        if (!SPLITS.includes(s))
            throw new Error(`split[${i}] is "${s}", expected one of ${SPLITS.join(', ')}`);
    });
    checkLength('groups', groups, X.length);
    for (const [field, values] of Object.entries(slices ?? {}))
        checkLength(`slices.${field}`, values, X.length);
    if (background && Object.keys(background.maxRate).length && background.X.length === 0)
        throw new Error('background.X must be non-empty when a maxRate is set');
    if (!(reviewRatio >= 0 && reviewRatio <= 1))
        throw new Error('reviewRatio must be between 0 and 1');
    if (!(autoMargin >= 0 && autoMargin < 1))
        throw new Error('autoMargin must be in [0, 1)');
    if (reviewEpsilon !== undefined && !(reviewEpsilon > 0 && reviewEpsilon < 1))
        throw new Error('reviewEpsilon must be strictly between 0 and 1');
    const result = { heads: {}, evaluation: {}, failures: [], warnings: [], testProbabilities: {}, weakLabels: {}, headChoice: {}, convergence: {}, cascadeChoice: {} };
    checkLength('foldKeys', input.foldKeys, X.length);
    const foldKey = (i) => input.foldKeys?.[i] ?? groups?.[i] ?? `#${i}`;
    // Provenance: validate every record, drop P5 near-duplicates from training, record overrides.
    const excluded = new Set();
    if (records) {
        checkLength('records', records, X.length);
        records.forEach((r, i) => {
            if (r.role !== split[i])
                throw new Error(`records[${i}] (${r.id}) has role ${r.role} but split[${i}] is ${split[i]}`);
        });
        const bgRecords = background?.records ?? [];
        if (background?.records) {
            checkLength('background.records', background.records, background.X.length);
            background.records.forEach((r, i) => {
                if (r.role !== 'background' || r.backgroundUse !== 'budget')
                    throw new Error(`background.records[${i}] (${r.id}) must have role background and backgroundUse budget`);
            });
        }
        else if (background && Object.keys(background.maxRate).length) {
            throw new Error('with records, background.records is required so the budget traffic can be checked (P2, P3)');
        }
        const safety = input.heads.filter((h) => h.safetyCritical).map((h) => h.name);
        const checked = validateProvenance([...records, ...bgRecords], { ...input.provenance, embeddings: input.provenanceEmbeddings
                ? [...checkedLength('provenanceEmbeddings.X', input.provenanceEmbeddings.X, X.length), ...(input.provenanceEmbeddings.background ?? background?.X ?? [])]
                : [...X, ...(background?.X ?? [])], safetyCritical: safety });
        const droppedIds = new Set(checked.dropped.map((d) => d.id));
        records.forEach((r, i) => { if (droppedIds.has(r.id))
            excluded.add(i); });
        result.warnings.push(...checked.warnings.map((w) => `provenance: ${w}`));
        result.failures.push(...checked.overridden.map((o) => `provenance: ${o.code} overridden for ${o.ids.length} record(s) - not releasable`));
        result.provenance = {
            dropped: checked.dropped, overridden: checked.overridden, safety_critical: safety, heads: {},
            generated: records.some((r, i) => r.source.kind === 'generated' && r.role === 'train' && !excluded.has(i)),
        };
        result.design = { designs: [...(input.designs ?? [])], heads: {} };
    }
    const vectorKey = (x) => Array.from(x).join(',');
    let heldOut = null;
    const isHeldOut = (x) => {
        heldOut ??= new Set([...X.filter((_, i) => split[i] !== 'train'), ...(background?.X ?? [])].map(vectorKey));
        return heldOut.has(vectorKey(x));
    };
    // The shared reference for knn, stack and auto heads: training rows labelled for any of them.
    // Generated records (synthetic text) are left out; retrieved ones are human-labelled and stay.
    // 'auto' heads with weak positives stay linear (weak labels are linear-only).
    const nonLinear = input.heads.filter((h) => (h.type ?? 'auto') !== 'linear' && !((h.type ?? 'auto') === 'auto' && (h.weak?.X.length || h.finer)));
    let reference = null;
    if (nonLinear.length) {
        for (const h of nonLinear) {
            const t = h.type ?? 'auto';
            if (!(t === 'auto' || HEAD_TYPES.includes(t)))
                throw new Error(`${h.name}: unknown head type ${t}`);
            checkLength(`${h.name}: y`, h.y, X.length);
        }
        const generated = new Set(records ? records.flatMap((r, i) => (r.source.kind === 'generated' ? [i] : [])) : []);
        const index = X.map((_, i) => i).filter((i) => split[i] === 'train' && !excluded.has(i) && !generated.has(i) && nonLinear.some((h) => h.y[i] === 0 || h.y[i] === 1));
        const leftOut = X.filter((_, i) => split[i] === 'train' && generated.has(i)).length;
        if (leftOut)
            result.warnings.push(`reference: ${leftOut} generated training record(s) left out of the knn reference`);
        const set = encodeReference(index.map((i) => X[i]), Object.fromEntries(nonLinear.map((h) => [h.name, index.map((i) => (h.y[i] === 0 || h.y[i] === 1 ? h.y[i] : null))])), input.referenceEncoding ?? 'f32');
        reference = { set, rows: decodeReference(set), index };
    }
    // The shared finer-label model: training rows with a finer label (not P5-dropped, not generated).
    let finerFit = null;
    if (input.heads.some((h) => h.finer)) {
        if (!input.finer)
            throw new Error(`heads ${input.heads.filter((h) => h.finer).map((h) => h.name).join(', ')} use finer, which needs TrainInput.finer`);
        checkLength('finer.labels', input.finer.labels, X.length);
        const generated = new Set(records ? records.flatMap((r, i) => (r.source.kind === 'generated' ? [i] : [])) : []);
        const rows = X.map((_, i) => i).filter((i) => split[i] === 'train' && input.finer.labels[i] != null && !excluded.has(i) && !generated.has(i));
        if (!rows.length)
            throw new Error('finer: no training row carries a finer label');
        finerFit = fitFinerModel(X, input.finer.labels, rows, foldKey, input.finer.C ?? FINER_C);
        log(`finer: ${finerFit.model.classes.length} finer labels on ${rows.length} training rows`);
        if (!finerFit.converged)
            (requireConvergence ? result.failures : result.warnings).push('finer: the finer-label model did not converge');
    }
    // standardize: per-dimension mean and sd of the embedding over X and background.X (features only).
    let standardiser = null;
    if (input.standardize && X.length) {
        const d = X[0].length, all = [...X, ...(background?.X ?? [])];
        const mean = new Float64Array(d), sd = new Float64Array(d);
        for (const x of all)
            for (let j = 0; j < d; j++)
                mean[j] += x[j] / all.length;
        for (const x of all)
            for (let j = 0; j < d; j++)
                sd[j] += (x[j] - mean[j]) ** 2 / all.length;
        for (let j = 0; j < d; j++)
            sd[j] = Math.sqrt(sd[j]) || 1;
        standardiser = { mean, sd };
    }
    /** fitLogistic, on the standardised embedding columns when `standardize` is on, folded back to raw weights. */
    const fitHead = (rows, yv, options) => {
        if (!standardiser)
            return fitLogistic(rows, yv, options);
        const { mean, sd } = standardiser, d = mean.length;
        const z = rows.map((x) => Float64Array.from(x, (v, j) => (j < d ? (v - mean[j]) / sd[j] : v)));
        const m = fitLogistic(z, yv, options);
        const coef = m.coef.map((w, j) => (j < d ? w / sd[j] : w));
        return { ...m, coef, intercept: m.intercept - m.coef.reduce((s, w, j) => s + (j < d ? (w * mean[j]) / sd[j] : 0), 0) };
    };
    const trainOne = (headIn, sink) => {
        const { name, y: yIn, policy, prevalence, baseline, weak, maxWeakShare = 0.5, type: typeIn = 'auto', dismissal, edge: edgeIn, finer, cascade } = headIn;
        /** The tiers the head is certified behind: the edge, or the edge and the middle when the cascade is kept. */
        let edge = edgeIn;
        let middleSha256 = null;
        if (finer && (typeIn === 'knn' || typeIn === 'stack'))
            throw new Error(`${name}: a finer head is linear on the embedding and its route score; type ${typeIn} is not supported`);
        if (finer && weak?.X.length)
            throw new Error(`${name}: weak positives are not supported with finer (weak examples carry no finer-label score)`);
        const type = typeIn === 'auto' && (weak?.X.length || finer) ? 'linear' : typeIn;
        if (type !== typeIn && !finer)
            sink.warnings.push(`${name}: weak positives are linear-only, so type 'auto' kept the linear head`);
        checkLength(`${name}: y`, yIn, X.length);
        checkLength(`${name}: dismissal.rows`, dismissal?.rows, X.length);
        if (dismissal?.background && background)
            checkLength(`${name}: dismissal.background`, dismissal.background, background.X.length);
        if (dismissal && !(dismissal.maxRate > 0 && dismissal.maxRate < 1))
            throw new Error(`${name}: dismissal.maxRate must be in (0, 1)`);
        checkLength(`${name}: edge.fired`, edge?.fired, X.length);
        checkLength(`${name}: edge.dismissed`, edge?.dismissed, X.length);
        if (edge?.background && background) {
            checkLength(`${name}: edge.background.fired`, edge.background.fired, background.X.length);
            checkLength(`${name}: edge.background.dismissed`, edge.background.dismissed, background.X.length);
        }
        if (edgeIn && edgeIn.fired.some((f, i) => f && edgeIn.dismissed[i]))
            throw new Error(`${name}: the edge both fired and dismissed a row`);
        if (edge && !/^[0-9a-f]{64}$/.test(edge.edgeSha256))
            throw new Error(`${name}: edge.edgeSha256 must be lowercase hex SHA-256`);
        if (cascade) {
            if (!edgeIn)
                throw new Error(`${name}: cascade needs the edge (the middle tier was validated behind it)`);
            if (policy.kind !== 'recall')
                throw new Error(`${name}: cascade supports recall heads only (precision heads were not validated)`);
            if (!/^[0-9a-f]{64}$/.test(cascade.middleSha256))
                throw new Error(`${name}: cascade.middleSha256 must be lowercase hex SHA-256`);
            checkLength(`${name}: cascade.rows.fired`, cascade.rows.fired, X.length);
            checkLength(`${name}: cascade.rows.dismissed`, cascade.rows.dismissed, X.length);
            checkLength(`${name}: cascade.scores.edge`, cascade.scores.edge, X.length);
            checkLength(`${name}: cascade.scores.middle`, cascade.scores.middle, X.length);
            if (cascade.rows.background && background) {
                checkLength(`${name}: cascade.rows.background.fired`, cascade.rows.background.fired, background.X.length);
                checkLength(`${name}: cascade.rows.background.dismissed`, cascade.rows.background.dismissed, background.X.length);
            }
            if (background && edgeIn.background && !cascade.rows.background)
                throw new Error(`${name}: the edge has background rows but cascade.rows has none`);
            for (let i = 0; i < X.length; i++) {
                if (cascade.rows.fired[i] && cascade.rows.dismissed[i])
                    throw new Error(`${name}: cascade.rows both fired and dismissed row ${i}`);
                if ((edgeIn.fired[i] && !cascade.rows.fired[i]) || (edgeIn.dismissed[i] && !cascade.rows.dismissed[i]))
                    throw new Error(`${name}: cascade.rows must keep every edge decision (row ${i})`);
            }
            const b = cascade.budgets;
            if (![b.edgeFire, b.edgeDismiss, b.middleFire, b.middleDismiss].every((v) => v >= 0 && v < 1) || b.middleFire < b.edgeFire || b.middleDismiss < b.edgeDismiss)
                throw new Error(`${name}: cascade.budgets must be shares in [0, 1), the middle's cumulative budgets at least the edge's`);
            if (cascade.choice !== undefined && cascade.choice !== 'cascade' && cascade.choice !== 'edge')
                throw new Error(`${name}: cascade.choice must be 'cascade' or 'edge'`);
        }
        if (policy.kind === 'recall' && policy.reserve !== undefined && !(policy.reserve >= 0 && policy.targetRecall + policy.reserve < 1))
            throw new Error(`${name}: reserve must be >= 0 with targetRecall + reserve < 1`);
        /** The recall the threshold is certified at: the target plus the policy's reserve for later periods. */
        const certTarget = policy.kind === 'recall' ? policy.targetRecall + (policy.reserve ?? 0) : NaN;
        /** Records the claims this head's threshold is chained to (dismissal rules, the edge, the middle tier). */
        const chained = (g) => {
            const claims = (dismissal ? 1 : 0) + (edgeIn ? 2 : 0) + (middleSha256 ? 2 : 0);
            return claims && g.delta !== undefined ? { ...g, chain: guaranteeChain(g.delta, claims, 1) } : g;
        };
        /** Rows decided before the head: dismissed (by rules or the edge) score 0, fired by the edge score 1. */
        const cleared = (i) => !!dismissal?.rows[i] || !!edge?.dismissed[i];
        const edgeFired = (i) => !!edge?.fired[i];
        const preset = (i) => (cleared(i) ? 0 : edgeFired(i) ? 1 : null);
        checkLength(`${name}: baseline`, baseline, X.length);
        if (records) {
            if (prevalence !== undefined)
                throw new Error(`${name}: don't pass prevalence with sampled records - calibration is weighted by 1/π, which already reproduces production prevalence; weighting to prevalence as well would count it twice`);
            records.forEach((r, i) => {
                if ((r.labels[name] ?? null) !== yIn[i])
                    throw new Error(`${name}: y[${i}] is ${yIn[i]} but records[${i}] (${r.id}) is labelled ${r.labels[name] ?? null}`);
            });
        }
        else if (!(prevalence !== undefined && prevalence > 0 && prevalence < 1))
            throw new Error(`${name}: prevalence must be strictly between 0 and 1, got ${prevalence}`);
        const y = excluded.size ? yIn.map((v, i) => (excluded.has(i) ? null : v)) : yIn;
        if (weak) {
            checkLength(`${name}: weak.weights`, weak.weights, weak.X.length);
            checkLength(`${name}: weak.rules`, weak.rules, weak.X.length);
            if (!(maxWeakShare >= 0))
                throw new Error(`${name}: maxWeakShare must be non-negative, got ${maxWeakShare}`);
            weak.X.forEach((x, j) => {
                if (x.length !== X[0]?.length)
                    throw new Error(`${name}: weak.X[${j}] has ${x.length} dimensions but X has ${X[0]?.length}`);
                if (!(weak.weights[j] >= 0 && weak.weights[j] <= 1))
                    throw new Error(`${name}: weak.weights[${j}] must be in [0, 1], got ${weak.weights[j]}`);
                if (isHeldOut(x))
                    throw new Error(`${name}: weak.X[${j}] is a calibration, test or background example - weak labels are for the train split only`);
            });
        }
        const idx = indicesBySplit(split, y);
        for (const s of SPLITS) {
            if (new Set(idx[s].map((i) => y[i])).size < 2)
                throw new Error(`${name}: the ${s} split needs both positive and negative examples`);
        }
        const ySplit = (s) => idx[s].map((i) => y[i]);
        log(`${name}: ` + SPLITS.map((s) => `${s} ${ySplit(s).reduce((a, b) => a + b, 0)}/${idx[s].length} positive`).join(', '));
        // Training weights: 1 per row, or the records' weights with P6 caps.
        let trainWeights = idx.train.map(() => 1);
        if (records) {
            const capped = capTrainingWeights(idx.train.map((i) => records[i]), name, input.caps);
            trainWeights = capped.weights;
            sink.provenance.heads[name] = capped.summary;
        }
        // Head type: knn and stack heads learn their layer on out-of-fold features of the training rows.
        let fitted = null;
        if (type !== 'linear') {
            if (weak && weak.X.length)
                throw new Error(`${name}: weak positives are supported for linear heads only`);
            const t = {
                rows: idx.train, y: ySplit('train'), weights: trainWeights, keys: idx.train.map(foldKey), X,
                reference: reference.rows, referenceKeys: reference.index.map(foldKey), referenceLabels: reference.set.labels[name], C, seed,
            };
            let chosen = type === 'auto' ? 'linear' : type;
            if (type === 'auto') {
                const costs = {}, fits = {};
                for (const ht of HEAD_TYPES) {
                    fits[ht] = fitFeatures(ht, t);
                    costs[ht] = guaranteeCost(crossValidatedScores(fits[ht], t), t.y, policy);
                }
                // Leave linear only for a clear cross-validated gain (autoMargin); ties go to the simpler type.
                const bar = costs.linear * (1 - autoMargin);
                chosen = HEAD_TYPES.filter((ht) => ht !== 'linear' && costs[ht] < bar - 1e-12).reduce((a, b) => (a === 'linear' || costs[b] < costs[a] - 1e-12 ? b : a), 'linear');
                fitted = chosen === 'linear' ? null : fits[chosen];
                const criterion = policy.kind === 'recall' ? `false alarms at recall ${policy.targetRecall}` : `recall lost at precision ${policy.targetPrecision}`;
                sink.headChoice[name] = { chosen, costs, criterion: `${criterion}, 3-fold cross-validated on the training rows; non-linear only below linear × ${1 - autoMargin}` };
                log(`${name}: auto chose ${chosen} (${HEAD_TYPES.map((ht) => `${ht} ${costs[ht].toFixed(4)}`).join(', ')})`);
            }
            else
                fitted = fitFeatures(chosen, t);
        }
        else if (finer) {
            // The embedding plus the route score: out-of-fold finer probabilities for the rows the finer model saw.
            const fm = finerFit;
            const labelled = idx.train.filter((i) => fm.classOf.has(i));
            const features = { kind: 'finer', scale: typeof finer === 'object' && finer.scale !== undefined ? finer.scale : FINER_SCALE,
                route_rates: finerRouteRates(fm.model.classes.length, labelled.map((i) => ({ k: fm.classOf.get(i), y: y[i] }))) };
            if (!(Number.isFinite(features.scale) && features.scale !== 0))
                throw new Error(`${name}: finer.scale must be a non-zero number`);
            const apply = (x) => finerFeatureVector(features, x, finerProbabilities(fm.model, x));
            fitted = {
                features,
                train: idx.train.map((i) => (fm.outOfFold.has(i) ? finerFeatureVector(features, X[i], fm.outOfFold.get(i)) : apply(X[i]))),
                apply,
                converged: fm.converged,
            };
            log(`${name}: finer route score from ${labelled.length} training rows with both labels`);
        }
        const trainAt = new Map(idx.train.map((i, n) => [i, n]));
        const applied = new Map();
        /** The head's input for row i of X: out-of-fold features for training rows. */
        const rowFeatures = (i) => (!fitted ? X[i] : trainAt.has(i) ? fitted.train[trainAt.get(i)] : applied.get(i) ?? applied.set(i, fitted.apply(X[i])).get(i));
        const features = (x) => (fitted ? fitted.apply(x) : x);
        // The cascade choice: training rows only (the head's own training features, cross-fitted), before any calibration.
        if (cascade && policy.kind === 'recall') {
            const criterion = `false alarms at recall ${Math.min(0.99, policy.targetRecall + (policy.reserve ?? 0))} on the training rows (every tier's thresholds re-derived there; 5-fold cross-fitted heads); cascade kept when ≤ ${CASCADE_TOLERANCE.relative} × edge → head + ${CASCADE_TOLERANCE.absolute}`;
            let record;
            if (cascade.choice)
                record = { cascade: cascade.choice === 'cascade', forced: true, criterion: 'fixed by HeadInput.cascade.choice' };
            else {
                const rows = idx.train, yTr = ySplit('train');
                const at = (v, what) => rows.map((i) => { if (!Number.isFinite(v[i]))
                    throw new Error(`${name}: cascade.scores.${what} has no finite score for training row ${i}`); return v[i]; });
                const c = chooseCascade({
                    y: yTr, edge: at(cascade.scores.edge, 'edge'), middle: at(cascade.scores.middle, 'middle'),
                    head: crossFitScores(rows.map(rowFeatures), yTr, rows.map(foldKey)),
                    budgets: cascade.budgets, targetRecall: policy.targetRecall, reserve: policy.reserve ?? 0,
                });
                record = { cascade: c.cascade, costs: c.costs, criterion };
            }
            sink.cascadeChoice[name] = record;
            if (record.cascade) {
                edge = { ...cascade.rows, edgeSha256: edgeIn.edgeSha256 };
                middleSha256 = cascade.middleSha256;
            }
            log(`${name}: cascade ${record.cascade ? 'kept' : 'not kept'}${record.costs ? ` (training false alarms: edge → head ${record.costs.twoTier.toFixed(4)}, with the middle ${record.costs.cascade.toFixed(4)})` : ' (fixed)'}`);
        }
        let model;
        if (fitted) {
            model = (finer ? fitHead : fitLogistic)(fitted.train, ySplit('train'), {
                C, classWeight: 'balanced', ...(trainWeights.some((w) => w !== 1) ? { sampleWeight: trainWeights } : {}),
            });
        }
        else if (weak && weak.X.length) {
            const gold = ySplit('train').reduce((a, b) => a + b, 0);
            const before = weak.weights.reduce((a, b) => a + b, 0);
            const scale = before > maxWeakShare * gold ? (maxWeakShare * gold) / before : 1;
            const w = weak.weights.map((v) => v * scale);
            model = fitHead([...idx.train.map((i) => X[i]), ...weak.X], [...ySplit('train'), ...w.map(() => 1)], {
                C, classWeight: 'balanced', sampleWeight: [...trainWeights, ...w],
            });
            const summary = { count: w.length, gold_train_positives: gold, max_weak_share: maxWeakShare, weight_before_cap: before, weight: w.reduce((a, b) => a + b, 0), scale };
            if (weak.rules) {
                summary.by_rule = {};
                weak.rules.forEach((rule, j) => {
                    const r = (summary.by_rule[rule] ??= { count: 0, weight: 0 });
                    r.count++;
                    r.weight += w[j];
                });
            }
            if (weak.source)
                Object.assign(summary, weak.source);
            sink.weakLabels[name] = summary;
            log(`${name}: ${w.length} weak positives, weight ${summary.weight.toFixed(2)} (${scale < 1 ? `capped at ${maxWeakShare}` : 'uncapped'}) beside ${gold} gold`);
        }
        else {
            model = fitHead(idx.train.map((i) => X[i]), ySplit('train'), {
                C, classWeight: 'balanced', ...(trainWeights.some((w) => w !== 1) ? { sampleWeight: trainWeights } : {}),
            });
        }
        const score = (x) => calibrate(calibration, decisionFunction(model, features(x)));
        const yCal = ySplit('calibration');
        // Sampled calibration: design weights 1/π (they reproduce production prevalence). Otherwise reweight to `prevalence`.
        const calDesign = records ? designOf(idx.calibration.map((i) => records[i])) : null;
        const wCal = calDesign ? calDesign.inclusionProbs.map((p) => 1 / p) : prevalenceWeights(yCal, prevalence);
        const unequal = !!calDesign && calDesign.inclusionProbs.some((p) => Math.abs(p - calDesign.inclusionProbs[0]) > 1e-12);
        const calLogits = idx.calibration.map((i) => decisionFunction(model, rowFeatures(i)));
        // The calibrator sees the rows the dismissal rules and the edge leave to the classifier.
        const kept = idx.calibration.map((i) => preset(i) === null);
        const keep = (v) => v.filter((_, j) => kept[j]);
        const keptY = keep(yCal);
        if (!keptY.length) {
            throw new CalibrationRemainderError(`${name}: the dismissal rules and the edge decide every calibration row, so the head's calibrator has nothing to fit - let the edge decide less`);
        }
        let calibrationConverged = true;
        let calibration;
        if (method === 'platt') {
            const { a, c, converged } = fitPlatt(keep(calLogits), keep(yCal), keep(wCal));
            calibration = { method: 'platt', a, c };
            calibrationConverged = converged;
        }
        else
            calibration = { method: 'isotonic', ...fitIsotonic(keep(calLogits).map(sigmoid), keep(yCal), keep(wCal), { yMin: 0, yMax: 1 }) };
        // Convergence: a fit that stopped short of its tolerance is reported (a failure with requireConvergence).
        const modelConverged = model.converged && (fitted?.converged ?? true);
        sink.convergence[name] = { model: modelConverged, calibration: calibrationConverged };
        for (const [what, ok] of [['logistic model', modelConverged], ['Platt calibrator (the calibration scores may separate the classes)', calibrationConverged]]) {
            if (!ok)
                (requireConvergence ? sink.failures : sink.warnings).push(`${name}: the ${what} did not converge`);
        }
        const pCal = calLogits.map((z, j) => preset(idx.calibration[j]) ?? calibrate(calibration, z));
        const budget = background?.maxRate[name];
        const backgroundP = background && budget !== undefined ? background.X.map((x, j) => (dismissal?.background?.[j] || edge?.background?.dismissed[j] ? 0 : edge?.background?.fired[j] ? 1 : score(x))) : null;
        const heuristic = () => {
            const t = pickThreshold(policy, pCal, yCal, calDesign ? wCal : undefined);
            return backgroundP && budget !== undefined ? budgetThreshold(t, backgroundP, budget) : t;
        };
        // Calibration scores per class, one per group: what every conformal statement counts.
        const calGroups = groups && idx.calibration.map((i) => groups[i]);
        const ofClass = (c) => (values) => values.filter((_, j) => yCal[j] === c);
        const positives = groupScores(ofClass(1)(pCal), calGroups && ofClass(1)(calGroups), Math.min);
        const delta = policy.delta ?? 0.05;
        let threshold;
        let guarantee;
        let sufficiency;
        // Every background rank a threshold could have been capped at - for a bound valid whichever applied.
        const backgroundRanks = [];
        if (backgroundP && budget !== undefined && Math.floor(budget * backgroundP.length) < backgroundP.length)
            backgroundRanks.push(Math.floor(budget * backgroundP.length));
        // Default: the strongest guarantee the data supports - 'design' with sampled calibration records,
        // else 'auto' (conformal) without the silent heuristic fallback. 'heuristic' only when chosen.
        const mode = policy.kind === 'recall' ? policy.mode ?? (calDesign ? 'design' : 'auto') : 'heuristic';
        const precisionMode = policy.kind === 'precision' ? policy.mode ?? (calDesign ? 'design' : undefined) : undefined;
        const allowHeuristic = policy.kind === 'recall'
            ? policy.allowHeuristicFallback ?? (policy.fallback !== undefined ? policy.fallback === 'heuristic' : policy.mode !== undefined)
            : false;
        if (policy.kind === 'recall' && policy.fallback === 'heuristic' && policy.designRecall === undefined)
            throw new Error(`${name}: fallback 'heuristic' needs designRecall (the heuristic threshold's target)`);
        if (unequal && (mode === 'conformal-expected' || mode === 'conformal-pac' || mode === 'auto' || reviewEpsilon !== undefined)) {
            throw new Error(`${name}: the calibration records have unequal inclusion probabilities, so they aren't exchangeable and conformal guarantees${reviewEpsilon !== undefined ? ' (including reviewEpsilon)' : ''} don't hold - use mode 'design'`);
        }
        if (policy.kind === 'recall' && mode === 'design') {
            if (!calDesign)
                throw new Error(`${name}: mode 'design' needs sampled calibration records (records)`);
            const designMethod = policy.designMethod ?? 'exact';
            const res = designRiskThreshold({ ...calDesign, y: yCal, scores: pCal, alpha: 1 - certTarget, delta, method: designMethod, seed, ...(policy.designMinStratumPositives !== undefined ? { minStratumPositives: policy.designMinStratumPositives } : {}) });
            const fails = [];
            sink.warnings.push(...(res.warnings ?? []).map((w) => `${name}: design: ${w}`));
            if (res.feasible) {
                threshold = res.threshold;
                guarantee = { mode, kind: res.guarantee, alpha: 1 - certTarget, delta, method: designMethod };
                // Budgets may only raise the threshold; raising it past the recall guarantee fails the gate.
                const caps = [];
                if (policy.maxFalseAlarm !== undefined && policy.maxFalseAlarm < 1) {
                    caps.push([`calibration false-alarm <= ${policy.maxFalseAlarm}`, falseAlarmCap(pCal, yCal, policy.maxFalseAlarm, wCal)]);
                }
                if (backgroundP && budget !== undefined)
                    caps.push([`background <= ${budget}`, budgetThreshold(res.threshold, backgroundP, budget)]);
                for (const [what, t] of caps) {
                    if (t > threshold) {
                        fails.push(`design: recall >= ${certTarget} needs a threshold <= ${res.threshold} but ${what} needs >= ${t}`);
                        threshold = t;
                        guarantee = { mode, kind: 'none', alpha: 1 - certTarget };
                    }
                }
                log(`${name}: design threshold ${threshold} (${res.guarantee}, ${designMethod}, Kish effective positives ${res.nEff.toFixed(1)})`);
            }
            else if (policy.fallback === 'heuristic') {
                // Too little evidence for the guarantee, and the policy chose a heuristic threshold for that case.
                threshold = heuristic();
                guarantee = { mode, kind: 'none', alpha: 1 - certTarget, fallback: 'heuristic' };
                sink.warnings.push(`${name}: design: ${res.reason}; heuristic threshold (fallback), no guarantee - a guarantee needs at least ${minimumSamples(1 - certTarget, delta)} effective calibration positives with no misses (have ${res.nEff.toFixed(1)})`);
            }
            else {
                fails.push(`design: ${res.reason}`);
                threshold = nextUp(1); // never fires: no threshold is substituted silently
                guarantee = { mode, kind: 'none', alpha: 1 - certTarget };
            }
            sink.failures.push(...fails.map((f) => `${name}: ${f}`));
            if (backgroundP && budget !== undefined && Math.floor(budget * backgroundP.length) < backgroundP.length)
                backgroundRanks.push(Math.floor(budget * backgroundP.length));
            sink.design.heads[name] = { calibration: 'design-weighted', threshold: `design (${designMethod})`, evaluation: 'design-linearised' };
        }
        else if (policy.kind === 'recall' && mode !== 'heuristic') {
            const constraints = [];
            if (policy.maxFalseAlarm !== undefined && policy.maxFalseAlarm < 1) {
                constraints.push({ name: 'calibration false-alarm', scores: groupScores(ofClass(0)(pCal), calGroups && ofClass(0)(calGroups), Math.max), maxRate: policy.maxFalseAlarm });
            }
            if (backgroundP && budget !== undefined)
                constraints.push({ name: 'background', scores: backgroundP, maxRate: budget });
            const sel = conformalThreshold({ mode: mode, targetRecall: certTarget, delta, positives, constraints, heuristicAvailable: policy.designRecall !== undefined, allowHeuristic });
            if (backgroundP && budget !== undefined) {
                for (const d of [undefined, delta / (1 + constraints.length)])
                    backgroundRanks.push(conformalRank(backgroundP.length, budget, d));
            }
            if (calibration.method === 'platt' && !(calibration.a > 0)) {
                sel.failures.push(`Platt slope ${calibration.a} reverses the model's score order, so no conformal guarantee holds`);
                sel.guarantee = { mode, kind: 'none', alpha: sel.guarantee.alpha };
            }
            threshold = sel.threshold ?? heuristic();
            guarantee = sel.threshold === null ? { ...sel.guarantee, kind: 'none', fallback: 'heuristic' } : sel.guarantee;
            sufficiency = sel.sufficiency;
            sufficiency.chosen = sel.threshold === null ? 'heuristic' : sufficiency.chosen;
            if (slices) {
                const perTest = delta / (1 + constraints.length);
                const needed = { expected: minimumSamples(guarantee.alpha), pac: minimumSamples(guarantee.alpha, perTest) };
                sufficiency.slices = {};
                for (const [field, values] of Object.entries(slices)) {
                    const calValues = ofClass(1)(idx.calibration.map((i) => values[i]));
                    const calPosGroups = calGroups && ofClass(1)(calGroups);
                    for (const v of [...new Set(calValues)].sort()) {
                        const n = new Set(calValues.flatMap((x, j) => (x === v ? [calPosGroups ? calPosGroups[j] : `#${j}`] : []))).size;
                        sufficiency.slices[`${field}=${v}`] = { positive_groups: n, feasible: [...(n >= needed.expected ? ['conformal-expected'] : []), ...(n >= needed.pac ? ['conformal-pac'] : [])] };
                    }
                }
            }
            sink.failures.push(...sel.failures.map((f) => `${name}: ${f}`));
            sink.warnings.push(...sel.warnings.map((w) => `${name}: ${w}`));
            log(`${name}: ${mode} threshold ${threshold} (${guarantee.kind} guarantee, ${positives.length} calibration positive groups)`);
        }
        else if (policy.kind === 'precision' && precisionMode === 'design') {
            if (!calDesign)
                throw new Error(`${name}: precision mode 'design' needs sampled calibration records (records)`);
            const designMethod = policy.designMethod ?? 'linearised';
            // Candidates fixed before calibration: training-split logits (evenly spaced ranks from the 30th
            // highest down), mapped through the calibrator so the bound is computed on the firing rule that ships.
            // Candidate scores fixed before calibration: background traffic when there is some (independent of
            // calibration, natural prevalence), else training scores - an enriched training set fires far more
            // often at the top than calibration traffic does. The r-th highest of n reference scores is expected
            // to be exceeded by ~r · n_cal / n calibration examples: start where that is about designMinFired.
            const minFired = policy.designMinFired ?? 20;
            const reference = background && background.X.length
                ? background.X.map((x) => score(x)).sort((a, b) => b - a)
                : idx.train.map((i) => calibrate(calibration, decisionFunction(model, rowFeatures(i)))).sort((a, b) => b - a);
            const start = Math.min(reference.length - 1, Math.ceil((minFired * reference.length) / Math.max(1, idx.calibration.length)) - 1);
            const ranks = Array.from({ length: 150 }, (_, k) => start + Math.floor((k * (reference.length - 1 - start)) / 149));
            const candidates = [...new Set(ranks.map((r) => reference[r]))].sort((a, b) => b - a);
            const res = designPrecisionThreshold({ ...calDesign, y: yCal, scores: pCal, candidates, targetPrecision: policy.targetPrecision, delta, method: designMethod });
            const alpha = 1 - policy.targetPrecision;
            if (res.feasible) {
                threshold = res.threshold;
                guarantee = { mode: 'design', kind: res.guarantee, alpha, delta, method: designMethod, metric: 'precision' };
                if (backgroundP && budget !== undefined) {
                    const t = budgetThreshold(threshold, backgroundP, budget);
                    if (t > threshold) {
                        sink.warnings.push(`${name}: design: the background budget raised the threshold from ${threshold} to ${t}; precision there is not covered by the guarantee`);
                        threshold = t;
                        guarantee = { mode: 'design', kind: 'none', alpha, metric: 'precision' };
                    }
                    if (Math.floor(budget * backgroundP.length) < backgroundP.length)
                        backgroundRanks.push(Math.floor(budget * backgroundP.length));
                }
                log(`${name}: design precision threshold ${threshold} (${res.guarantee}, ${designMethod}, ${res.firedEffective.toFixed(1)} effective fired)`);
            }
            else if (policy.fallback === 'heuristic') {
                // Too little evidence for the guarantee, and the policy chose a heuristic threshold for that case.
                sink.warnings.push(`${name}: design: ${res.reason}; heuristic threshold (fallback), no guarantee`);
                threshold = heuristic();
                guarantee = { mode: 'design', kind: 'none', alpha, metric: 'precision', fallback: 'heuristic' };
            }
            else {
                sink.failures.push(`${name}: design: ${res.reason}`);
                threshold = nextUp(1);
                guarantee = { mode: 'design', kind: 'none', alpha, metric: 'precision' };
            }
            sink.design.heads[name] = { calibration: 'design-weighted', threshold: `design precision (${designMethod})`, evaluation: 'design-linearised' };
        }
        else {
            threshold = heuristic();
            if (policy.kind === 'recall')
                guarantee = { mode: 'heuristic', kind: 'none', alpha: 1 - policy.targetRecall };
            if (policy.kind === 'precision' && precisionMode === undefined && policy.fallback !== 'heuristic') {
                // No guarantee is possible here, and none was waived: the head may ship only in shadow mode.
                sink.failures.push(`${name}: no precision guarantee without sampled calibration records - pass records (mode 'design'), or set mode 'heuristic' (or fallback 'heuristic') to accept an unguaranteed threshold`);
            }
            else if (policy.kind === 'precision' && precisionMode === undefined) {
                guarantee = { mode: 'heuristic', kind: 'none', alpha: 1 - policy.targetPrecision, metric: 'precision', fallback: 'heuristic' };
                sink.warnings.push(`${name}: no sampled calibration records, so no precision guarantee: heuristic threshold (fallback)`);
            }
            else
                sink.warnings.push(`${name}: mode 'heuristic' was chosen - the threshold carries no guarantee`);
        }
        if (records && !sink.design.heads[name])
            sink.design.heads[name] = { calibration: 'design-weighted', threshold: mode, evaluation: 'design-linearised' };
        if (policy.kind === 'precision' && precisionMode !== 'design' && !pCal.some((p) => p >= threshold)) {
            sink.warnings.push(`${name}: no calibration example reaches the target precision ${threshold} - the head is unlikely ever to fire`);
        }
        if (edge && threshold > 1 && edge.fired.some(Boolean))
            sink.warnings.push(`${name}: the firing budget keeps the head from firing at all, but the edge still fires`);
        if ((dismissal || edge?.dismissed.some(Boolean)) && !(threshold > 0)) {
            // Firing at 0 would mean firing on messages the rules cleared, which are never scored.
            sink.failures.push(`${name}: the target needs messages the dismissal rules cleared or the edge dismissed to fire - dismiss less`);
            threshold = nextUp(1);
            if (guarantee)
                guarantee = { mode: guarantee.mode, kind: 'none', alpha: guarantee.alpha, ...(guarantee.metric ? { metric: guarantee.metric } : {}) };
        }
        const reviewFloor = reviewEpsilon === undefined ? threshold * reviewRatio : Math.min(threshold, conformalLowerThreshold(positives, reviewEpsilon) ?? 0);
        const yTest = ySplit('test');
        const pTest = idx.test.map((i) => preset(i) ?? calibrate(calibration, decisionFunction(model, rowFeatures(i))));
        const atTest = (values) => idx.test.map((i) => values[i]);
        const testDesign = records ? designOf(idx.test.map((i) => records[i])) : null;
        const ev = evaluateHead({
            p: pTest,
            y: yTest,
            w: testDesign ? testDesign.inclusionProbs.map((p) => 1 / p) : prevalenceWeights(yTest, prevalence),
            ...(testDesign ? { design: { ...testDesign, seed } } : {}),
            threshold,
            groups: groups && atTest(groups),
            slices: slices && Object.fromEntries(Object.entries(slices).map(([field, values]) => [field, atTest(values)])),
            baseline: baseline && atTest(baseline),
            delta,
            prevalence,
            ...(dismissal || edge ? { scored: idx.test.map((i) => preset(i) === null) } : {}),
        });
        if (guarantee)
            ev.guarantee = guarantee;
        if (sufficiency)
            ev.sufficiency = sufficiency;
        if (backgroundP) {
            ev.background_rate = backgroundP.filter((p) => p >= threshold).length / backgroundP.length;
            ev.background_rate_upper = backgroundUpper(backgroundP, threshold, backgroundRanks, 1 - delta / 2);
        }
        const gates = gateHead(policy, ev, { maxEce, calibrationAlpha });
        sink.evaluation[name] = ev;
        sink.failures.push(...gates.failures.map((f) => `${name}: ${f}`));
        sink.warnings.push(...gates.warnings.map((w) => `${name}: ${w}`));
        sink.heads[name] = {
            weights: model.coef, bias: model.intercept, ...(fitted?.features ? { features: fitted.features } : {}), calibration, threshold, review_floor: reviewFloor,
            ...(dismissal ? { dismissal: { rule_set: dismissal.ruleSet, max_rate: dismissal.maxRate, certified: dismissal.certified } } : {}),
            ...(edge ? { edge: { model_sha256: edge.edgeSha256 } } : {}),
            ...(middleSha256 ? { middle: { model_sha256: middleSha256 } } : {}),
            ...(guarantee ? { guarantee: chained(guarantee) } : {}), ...(reviewEpsilon !== undefined ? { review_epsilon: reviewEpsilon } : {}),
        };
        // Round-trip checks score embeddings, so rows decided before the head (never scored at runtime) are left out.
        sink.testProbabilities[name] = { idx: idx.test.filter((i) => preset(i) === null), p: pTest.filter((_, j) => preset(idx.test[j]) === null) };
    };
    result.edgeChoice = {};
    for (const headIn of input.heads) {
        const ladder = headIn.edge === undefined ? [undefined] : Array.isArray(headIn.edge) ? [...headIn.edge] : [headIn.edge];
        if (!ladder.length)
            throw new Error(`${headIn.name}: edge is an empty list`);
        if (headIn.cascade && ladder.length > 1)
            throw new Error(`${headIn.name}: cascade needs a single edge candidate (an edge ladder with the middle tier was not validated)`);
        if (ladder.length === 1) {
            trainOne({ ...headIn, edge: ladder[0] }, result);
            if (ladder[0])
                result.edgeChoice[headIn.name] = 0;
            continue;
        }
        // An edge ladder: the first candidate with which the head passes its gates with a guarantee (else the last).
        for (let k = 0; k < ladder.length; k++) {
            const sink = { heads: {}, evaluation: {}, failures: [], warnings: [], testProbabilities: {}, weakLabels: {}, headChoice: {}, convergence: {}, cascadeChoice: {},
                ...(result.provenance ? { provenance: { ...result.provenance, heads: {} } } : {}), ...(result.design ? { design: { ...result.design, heads: {} } } : {}) };
            try {
                trainOne({ ...headIn, edge: ladder[k] }, sink);
            }
            catch (e) {
                // A candidate that decides every calibration row: try the next one (the last rethrows).
                if (!(e instanceof CalibrationRemainderError) || k === ladder.length - 1)
                    throw e;
                result.warnings.push(`${headIn.name}: edge: candidate ${k} skipped (${e.message.slice(headIn.name.length + 2)})`);
                continue;
            }
            const spec = sink.heads[headIn.name];
            const passed = !!spec && !sink.failures.length && (spec.guarantee?.kind ?? 'none') !== 'none';
            if (!passed && k < ladder.length - 1)
                continue;
            const n = headIn.name;
            for (const key of ['heads', 'evaluation', 'testProbabilities', 'weakLabels', 'headChoice', 'convergence', 'cascadeChoice'])
                if (sink[key][n] !== undefined)
                    result[key][n] = sink[key][n];
            if (sink.provenance?.heads[n] && result.provenance)
                result.provenance.heads[n] = sink.provenance.heads[n];
            if (sink.design?.heads[n] && result.design)
                result.design.heads[n] = sink.design.heads[n];
            result.failures.push(...sink.failures);
            result.warnings.push(...sink.warnings, ...(k > 0 ? [`${n}: edge: candidate ${k} chosen (the head ${passed ? 'failed its gates' : 'failed every candidate'} with ${k === 1 ? 'the first' : `the first ${k}`})`] : []));
            result.edgeChoice[n] = k;
            // The ladder tried k + 1 candidates on the same sample: the union bound grows with them.
            const g = result.heads[n]?.guarantee;
            if (g?.chain && k > 0)
                result.heads[n] = { ...result.heads[n], guarantee: { ...g, chain: guaranteeChain(g.delta, g.chain.claims, k + 1) } };
            break;
        }
    }
    if (finerFit && Object.values(result.heads).some((spec) => spec?.features?.kind === 'finer'))
        result.finer = finerFit.model;
    // Keep the reference only for heads that use it (auto may have chosen linear everywhere).
    if (reference) {
        const using = Object.entries(result.heads).filter(([, spec]) => { const k = spec.features?.kind; return k === 'knn' || k === 'stack'; }).map(([h]) => h);
        if (using.length)
            result.reference = { ...reference.set, labels: Object.fromEntries(using.map((h) => [h, reference.set.labels[h]])) };
    }
    return result;
}
/**
 * Exact upper bound on the background firing rate at `threshold`. A threshold capped by the
 * background itself sits at or above the score just past a FIXED rank r, whose firing rate is a
 * uniform order statistic - so the bound for the largest rank any cap could have used is valid
 * whichever cap applied. Otherwise the threshold never looked at the background and its count is
 * an ordinary binomial.
 */
function backgroundUpper(scores, threshold, ranks, confidence) {
    const sorted = [...scores].sort((a, b) => b - a);
    const r = Math.max(-1, ...ranks.filter((k) => k < sorted.length));
    if (r >= 0 && threshold > sorted[r])
        return clopperPearsonUpper(r, sorted.length, confidence);
    return clopperPearsonUpper(sorted.filter((p) => p >= threshold).length, sorted.length, confidence);
}
/**
 * Serialises the artifact, re-loads it through validateArtifact and checks runtime scoring
 * reproduces the evaluated test probabilities exactly - so what was evaluated is what will run.
 */
export function assertRoundTrip(artifact, X, testProbabilities, sample = 50) {
    const reloaded = validateArtifact(JSON.parse(JSON.stringify(artifact)));
    for (const [head, probs] of Object.entries(testProbabilities)) {
        probs.idx.slice(0, sample).forEach((i, j) => {
            const runtime = scoreEmbedding(reloaded, X[i])[head];
            if (runtime !== probs.p[j])
                throw new Error(`${head}: runtime scoring of the saved artifact differs from evaluation (${runtime} vs ${probs.p[j]})`);
        });
    }
}
/**
 * The artifact for a training result, with its gates taken from the result (passed exactly when no
 * gate failed), so they are never set by hand. Also records the reference, the finer model, head choices, convergence,
 * provenance, design and weak-label summaries. Serve it through validateArtifact(raw, { mode }).
 */
export function buildArtifact(result, options) {
    const training = { ...(options.training ?? {}) };
    if (options.router) {
        if (!/^[0-9a-f]{64}$/.test(options.router.ruleSetHash))
            throw new Error('router.ruleSetHash must be a lowercase hex SHA-256 (rule-miner ruleSetHash)');
        training.router = { ruleSetHash: options.router.ruleSetHash };
    }
    if (Object.keys(result.headChoice).length)
        training.head_choice = result.headChoice;
    if (Object.keys(result.cascadeChoice ?? {}).length)
        training.cascade_choice = result.cascadeChoice;
    if (Object.keys(result.convergence).length)
        training.convergence = result.convergence;
    if (Object.keys(result.weakLabels).length)
        training.weak_labels = result.weakLabels;
    if (result.provenance)
        training.provenance = result.provenance;
    if (result.design)
        training.design = result.design;
    return {
        version: options.version,
        created_at: options.createdAt ?? new Date().toISOString(),
        embedding: options.embedding,
        heads: result.heads,
        ...(result.reference ? { reference: result.reference } : {}),
        ...(result.finer ? { finer: result.finer } : {}),
        training,
        evaluation: result.evaluation,
        gates: { passed: result.failures.length === 0, failures: [...result.failures], warnings: [...result.warnings] },
    };
}
