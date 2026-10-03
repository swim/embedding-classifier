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
import { clopperPearsonUpper, conformalLowerThreshold, conformalRank, decisionFunction, designRiskThreshold, fitIsotonic, fitLogistic, fitPlatt, minimumSamples, nextUp, prevalenceWeights, sigmoid } from '@liquidau/solvers';
import { designOf } from "./design.js";
import { capTrainingWeights, validateProvenance } from "./records.js";
import { conformalThreshold, groupScores } from "./conformal.js";
import { calibrate, scoreEmbedding, validateArtifact } from "./artifact.js";
import { evaluateHead } from "./evaluate.js";
import { gateHead } from "./gates.js";
import { budgetThreshold, falseAlarmCap, pickThreshold } from "./threshold.js";
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
    const { X, split, C = 1, calibration: method = 'platt', reviewRatio = 0.5, reviewEpsilon, maxEce, groups, slices, background, records, seed = 0, log = () => { } } = input;
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
    if (reviewEpsilon !== undefined && !(reviewEpsilon > 0 && reviewEpsilon < 1))
        throw new Error('reviewEpsilon must be strictly between 0 and 1');
    const result = { heads: {}, evaluation: {}, failures: [], warnings: [], testProbabilities: {}, weakLabels: {} };
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
        const checked = validateProvenance([...records, ...bgRecords], { ...input.provenance, embeddings: [...X, ...(background?.X ?? [])], safetyCritical: safety });
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
    for (const { name, y: yIn, policy, prevalence, baseline, weak, maxWeakShare = 0.5 } of input.heads) {
        checkLength(`${name}: y`, yIn, X.length);
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
            result.provenance.heads[name] = capped.summary;
        }
        let model;
        if (weak && weak.X.length) {
            const gold = ySplit('train').reduce((a, b) => a + b, 0);
            const before = weak.weights.reduce((a, b) => a + b, 0);
            const scale = before > maxWeakShare * gold ? (maxWeakShare * gold) / before : 1;
            const w = weak.weights.map((v) => v * scale);
            model = fitLogistic([...idx.train.map((i) => X[i]), ...weak.X], [...ySplit('train'), ...w.map(() => 1)], {
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
            result.weakLabels[name] = summary;
            log(`${name}: ${w.length} weak positives, weight ${summary.weight.toFixed(2)} (${scale < 1 ? `capped at ${maxWeakShare}` : 'uncapped'}) beside ${gold} gold`);
        }
        else {
            model = fitLogistic(idx.train.map((i) => X[i]), ySplit('train'), {
                C, classWeight: 'balanced', ...(trainWeights.some((w) => w !== 1) ? { sampleWeight: trainWeights } : {}),
            });
        }
        const score = (x) => calibrate(calibration, decisionFunction(model, x));
        const yCal = ySplit('calibration');
        // Sampled calibration: design weights 1/π (they reproduce production prevalence). Otherwise reweight to `prevalence`.
        const calDesign = records ? designOf(idx.calibration.map((i) => records[i])) : null;
        const wCal = calDesign ? calDesign.inclusionProbs.map((p) => 1 / p) : prevalenceWeights(yCal, prevalence);
        const unequal = !!calDesign && calDesign.inclusionProbs.some((p) => Math.abs(p - calDesign.inclusionProbs[0]) > 1e-12);
        const calLogits = idx.calibration.map((i) => decisionFunction(model, X[i]));
        const calibration = method === 'platt'
            ? { method: 'platt', ...fitPlatt(calLogits, yCal, wCal) }
            : { method: 'isotonic', ...fitIsotonic(calLogits.map(sigmoid), yCal, wCal, { yMin: 0, yMax: 1 }) };
        const pCal = calLogits.map((z) => calibrate(calibration, z));
        const budget = background?.maxRate[name];
        const backgroundP = background && budget !== undefined ? background.X.map(score) : null;
        const heuristic = () => {
            const t = pickThreshold(policy, pCal, yCal, calDesign ? wCal : undefined);
            return backgroundP && budget !== undefined ? budgetThreshold(t, backgroundP, budget) : t;
        };
        // Calibration scores per class, one per group: what every conformal statement counts.
        const calGroups = groups && idx.calibration.map((i) => groups[i]);
        const ofClass = (c) => (values) => values.filter((_, j) => yCal[j] === c);
        const positives = groupScores(ofClass(1)(pCal), calGroups && ofClass(1)(calGroups), Math.min);
        const delta = policy.kind === 'recall' ? policy.delta ?? 0.05 : 0.05;
        let threshold;
        let guarantee;
        let sufficiency;
        // Every background rank a threshold could have been capped at - for a bound valid whichever applied.
        const backgroundRanks = [];
        if (backgroundP && budget !== undefined && Math.floor(budget * backgroundP.length) < backgroundP.length)
            backgroundRanks.push(Math.floor(budget * backgroundP.length));
        const mode = policy.kind === 'recall' ? policy.mode ?? 'heuristic' : 'heuristic';
        if (unequal && (mode === 'conformal-expected' || mode === 'conformal-pac' || mode === 'auto' || reviewEpsilon !== undefined)) {
            throw new Error(`${name}: the calibration records have unequal inclusion probabilities, so they aren't exchangeable and conformal guarantees${reviewEpsilon !== undefined ? ' (including reviewEpsilon)' : ''} don't hold - use mode 'design'`);
        }
        if (policy.kind === 'recall' && mode === 'design') {
            if (!calDesign)
                throw new Error(`${name}: mode 'design' needs sampled calibration records (records)`);
            const designMethod = policy.designMethod ?? 'exact';
            const res = designRiskThreshold({ ...calDesign, y: yCal, scores: pCal, alpha: 1 - policy.targetRecall, delta, method: designMethod, seed });
            const fails = [];
            if (res.feasible) {
                threshold = res.threshold;
                guarantee = { mode, kind: res.guarantee, alpha: 1 - policy.targetRecall, delta, method: designMethod };
                // Budgets may only raise the threshold; raising it past the recall guarantee fails the gate.
                const caps = [];
                if (policy.maxFalseAlarm !== undefined && policy.maxFalseAlarm < 1) {
                    caps.push([`calibration false-alarm <= ${policy.maxFalseAlarm}`, falseAlarmCap(pCal, yCal, policy.maxFalseAlarm, wCal)]);
                }
                if (backgroundP && budget !== undefined)
                    caps.push([`background <= ${budget}`, budgetThreshold(res.threshold, backgroundP, budget)]);
                for (const [what, t] of caps) {
                    if (t > threshold) {
                        fails.push(`design: recall >= ${policy.targetRecall} needs a threshold <= ${res.threshold} but ${what} needs >= ${t}`);
                        threshold = t;
                        guarantee = { mode, kind: 'none', alpha: 1 - policy.targetRecall };
                    }
                }
                log(`${name}: design threshold ${threshold} (${res.guarantee}, ${designMethod}, Kish effective positives ${res.nEff.toFixed(1)})`);
            }
            else {
                fails.push(`design: ${res.reason}`);
                threshold = nextUp(1); // never fires: no threshold is substituted silently
                guarantee = { mode, kind: 'none', alpha: 1 - policy.targetRecall };
            }
            result.failures.push(...fails.map((f) => `${name}: ${f}`));
            if (backgroundP && budget !== undefined && Math.floor(budget * backgroundP.length) < backgroundP.length)
                backgroundRanks.push(Math.floor(budget * backgroundP.length));
            result.design.heads[name] = { calibration: 'design-weighted', threshold: `design (${designMethod})`, evaluation: 'design-linearised' };
        }
        else if (policy.kind === 'recall' && mode !== 'heuristic') {
            const constraints = [];
            if (policy.maxFalseAlarm !== undefined && policy.maxFalseAlarm < 1) {
                constraints.push({ name: 'calibration false-alarm', scores: groupScores(ofClass(0)(pCal), calGroups && ofClass(0)(calGroups), Math.max), maxRate: policy.maxFalseAlarm });
            }
            if (backgroundP && budget !== undefined)
                constraints.push({ name: 'background', scores: backgroundP, maxRate: budget });
            const sel = conformalThreshold({ mode: mode, targetRecall: policy.targetRecall, delta, positives, constraints, heuristicAvailable: policy.designRecall !== undefined, allowHeuristic: policy.allowHeuristicFallback });
            if (backgroundP && budget !== undefined) {
                for (const d of [undefined, delta / (1 + constraints.length)])
                    backgroundRanks.push(conformalRank(backgroundP.length, budget, d));
            }
            if (calibration.method === 'platt' && !(calibration.a > 0)) {
                sel.failures.push(`Platt slope ${calibration.a} reverses the model's score order, so no conformal guarantee holds`);
                sel.guarantee = { mode, kind: 'none', alpha: sel.guarantee.alpha };
            }
            threshold = sel.threshold ?? heuristic();
            guarantee = sel.guarantee;
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
            result.failures.push(...sel.failures.map((f) => `${name}: ${f}`));
            result.warnings.push(...sel.warnings.map((w) => `${name}: ${w}`));
            log(`${name}: ${mode} threshold ${threshold} (${guarantee.kind} guarantee, ${positives.length} calibration positive groups)`);
        }
        else {
            threshold = heuristic();
            if (policy.kind === 'recall')
                guarantee = { mode: 'heuristic', kind: 'none', alpha: 1 - policy.targetRecall };
        }
        if (records && !result.design.heads[name])
            result.design.heads[name] = { calibration: 'design-weighted', threshold: mode, evaluation: 'design-linearised' };
        if (policy.kind === 'precision' && !pCal.some((p) => p >= threshold)) {
            result.warnings.push(`${name}: no calibration example reaches the target precision ${threshold} - the head is unlikely ever to fire`);
        }
        const reviewFloor = reviewEpsilon === undefined ? threshold * reviewRatio : Math.min(threshold, conformalLowerThreshold(positives, reviewEpsilon) ?? 0);
        const yTest = ySplit('test');
        const pTest = idx.test.map((i) => score(X[i]));
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
        });
        if (guarantee)
            ev.guarantee = guarantee;
        if (sufficiency)
            ev.sufficiency = sufficiency;
        if (backgroundP) {
            ev.background_rate = backgroundP.filter((p) => p >= threshold).length / backgroundP.length;
            ev.background_rate_upper = backgroundUpper(backgroundP, threshold, backgroundRanks, 1 - delta / 2);
        }
        const gates = gateHead(policy, ev, { maxEce });
        result.evaluation[name] = ev;
        result.failures.push(...gates.failures.map((f) => `${name}: ${f}`));
        result.warnings.push(...gates.warnings.map((w) => `${name}: ${w}`));
        result.heads[name] = {
            weights: model.coef, bias: model.intercept, calibration, threshold, review_floor: reviewFloor,
            ...(guarantee ? { guarantee } : {}), ...(reviewEpsilon !== undefined ? { review_epsilon: reviewEpsilon } : {}),
        };
        result.testProbabilities[name] = { idx: idx.test, p: pTest };
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
