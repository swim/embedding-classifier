/**
 * Fit, calibrate, threshold, evaluate and gate every head - everything short of I/O.
 *
 *   train split        weighted L2 logistic regression per head (class-balanced, exact Newton solver),
 *                      plus any weak positives (e.g. from certified rules), weighted and capped
 *   calibration split  Platt or isotonic calibrator, weighted to the head's production prevalence,
 *                      then the threshold from the head's policy
 *   test split         evaluateHead + gateHead
 */
import { decisionFunction, fitIsotonic, fitLogistic, fitPlatt, prevalenceWeights, sigmoid } from '@liquidau/solvers';
import { calibrate, scoreEmbedding, validateArtifact } from "./artifact.js";
import { evaluateHead } from "./evaluate.js";
import { gateHead } from "./gates.js";
import { budgetThreshold, pickThreshold } from "./threshold.js";
export const SPLITS = ['train', 'calibration', 'test'];
function checkLength(name, values, n) {
    if (values !== undefined && values.length !== n)
        throw new Error(`${name} has ${values.length} entries but X has ${n}`);
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
    const { X, split, C = 1, calibration: method = 'platt', reviewRatio = 0.5, maxEce, groups, slices, background, log = () => { } } = input;
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
    const result = { heads: {}, evaluation: {}, failures: [], warnings: [], testProbabilities: {}, weakLabels: {} };
    const vectorKey = (x) => Array.from(x).join(',');
    let heldOut = null;
    const isHeldOut = (x) => {
        heldOut ??= new Set([...X.filter((_, i) => split[i] !== 'train'), ...(background?.X ?? [])].map(vectorKey));
        return heldOut.has(vectorKey(x));
    };
    for (const { name, y, policy, prevalence, baseline, weak, maxWeakShare = 0.5 } of input.heads) {
        checkLength(`${name}: y`, y, X.length);
        checkLength(`${name}: baseline`, baseline, X.length);
        if (!(prevalence > 0 && prevalence < 1))
            throw new Error(`${name}: prevalence must be strictly between 0 and 1, got ${prevalence}`);
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
        let model;
        if (weak && weak.X.length) {
            const gold = ySplit('train').reduce((a, b) => a + b, 0);
            const before = weak.weights.reduce((a, b) => a + b, 0);
            const scale = before > maxWeakShare * gold ? (maxWeakShare * gold) / before : 1;
            const w = weak.weights.map((v) => v * scale);
            model = fitLogistic([...idx.train.map((i) => X[i]), ...weak.X], [...ySplit('train'), ...w.map(() => 1)], {
                C, classWeight: 'balanced', sampleWeight: [...idx.train.map(() => 1), ...w],
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
            result.weakLabels[name] = summary;
            log(`${name}: ${w.length} weak positives, weight ${summary.weight.toFixed(2)} (${scale < 1 ? `capped at ${maxWeakShare}` : 'uncapped'}) beside ${gold} gold`);
        }
        else {
            model = fitLogistic(idx.train.map((i) => X[i]), ySplit('train'), { C, classWeight: 'balanced' });
        }
        const score = (x) => calibrate(calibration, decisionFunction(model, x));
        const yCal = ySplit('calibration');
        const wCal = prevalenceWeights(yCal, prevalence);
        const calLogits = idx.calibration.map((i) => decisionFunction(model, X[i]));
        const calibration = method === 'platt'
            ? { method: 'platt', ...fitPlatt(calLogits, yCal, wCal) }
            : { method: 'isotonic', ...fitIsotonic(calLogits.map(sigmoid), yCal, wCal, { yMin: 0, yMax: 1 }) };
        const pCal = calLogits.map((z) => calibrate(calibration, z));
        let threshold = pickThreshold(policy, pCal, yCal);
        if (policy.kind === 'precision' && !pCal.some((p) => p >= threshold)) {
            result.warnings.push(`${name}: no calibration example reaches the target precision ${threshold} - the head is unlikely ever to fire`);
        }
        const budget = background?.maxRate[name];
        const backgroundP = background && budget !== undefined ? background.X.map(score) : null;
        if (backgroundP && budget !== undefined)
            threshold = budgetThreshold(threshold, backgroundP, budget);
        const yTest = ySplit('test');
        const pTest = idx.test.map((i) => score(X[i]));
        const atTest = (values) => idx.test.map((i) => values[i]);
        const ev = evaluateHead({
            p: pTest,
            y: yTest,
            w: prevalenceWeights(yTest, prevalence),
            threshold,
            groups: groups && atTest(groups),
            slices: slices && Object.fromEntries(Object.entries(slices).map(([field, values]) => [field, atTest(values)])),
            baseline: baseline && atTest(baseline),
        });
        if (backgroundP)
            ev.background_rate = backgroundP.filter((p) => p >= threshold).length / backgroundP.length;
        const gates = gateHead(policy, ev, { maxEce });
        result.evaluation[name] = ev;
        result.failures.push(...gates.failures.map((f) => `${name}: ${f}`));
        result.warnings.push(...gates.warnings.map((w) => `${name}: ${w}`));
        result.heads[name] = { weights: model.coef, bias: model.intercept, calibration, threshold, review_floor: threshold * reviewRatio };
        result.testProbabilities[name] = { idx: idx.test, p: pTest };
    }
    return result;
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
