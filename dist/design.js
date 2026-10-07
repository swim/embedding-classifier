/**
 * Probability samples of real traffic, and one blinded labelling queue for every human label.
 *
 *   designSample   stratify a frame by rule firing × score band (× slice), allocate, draw simple
 *                  random samples without replacement, and split each stratum's draw into
 *                  train / calibration / test BEFORE labelling. Each role is then its own stratified
 *                  sample (a random subset of a simple random sample is one): its inclusion
 *                  probability is n_{h,role} / N_h.
 *   labelQueue     sampled records, retrieval candidates and generated-item verification in one
 *                  shuffled queue. Reviewers see an opaque id, the text and the heads - never the
 *                  source, mechanism, score or rule firing, which stay in the key.
 *   applyReviews   labels from reviewers; inclusion probabilities recomputed from what was
 *                  actually labelled per stratum and role (random budget drops don't bias them;
 *                  content-related skips can - the skip rate is reported).
 *
 * Scores in the frame must come from the text alone (not from labels or outcomes) - designSample
 * can't check this; `scoringModel` records which model produced them so it can be audited.
 *
 * Allocation `expected-positives` is proportional allocation SIZED from a prior round: it checks
 * that `total` is large enough for every material stratum (one estimated to hold at least
 * `minShare` of all positives) to expect `minExpectedPositives` positives in its calibration and
 * test shares, throws with the required total if not, and otherwise allocates proportionally.
 * Proportional shares give every sampled positive the same weight, which maximises the effective
 * number of positives that design-based recall bounds rest on; in simulation, skewing samples
 * toward low-score strata instead starved the positive-rich stratum and made targets infeasible,
 * while a too-small sample of a stratum holding a real share of positives made linearised bounds
 * under-cover. Rates must come from a PRIOR, independent round, fixed before drawing.
 */
import { seededRandom, weightedQuantile } from '@liquidau/solvers';
/** A short, stable, non-cryptographic id (FNV-1a, 52 bits): reproducibility, not security. */
export function stableId(prefix, parts) {
    const text = JSON.stringify(parts);
    let h = 0xcbf29ce484222325n;
    for (let i = 0; i < text.length; i++) {
        h ^= BigInt(text.charCodeAt(i));
        h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
    }
    return `${prefix}_${(h & 0xfffffffffffffn).toString(16).padStart(13, '0')}`;
}
function shuffle(items, rand) {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
}
/** Largest-remainder rounding of non-negative targets, each within [lo, hi], summing to `total`. */
function roundWithin(targets, lo, hi, total) {
    const out = targets.map((t, h) => Math.min(hi[h], Math.max(lo[h], Math.floor(t))));
    let left = total - out.reduce((a, b) => a + b, 0);
    const order = targets.map((t, h) => [t - Math.floor(t), h]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, h]) => h);
    for (let pass = 0; left > 0 && pass < 2; pass++) {
        for (const h of order)
            if (left > 0 && out[h] < hi[h]) {
                out[h]++;
                left--;
            }
    }
    return out;
}
export function designSample(options) {
    const { frame, scoreBands, bySlice = false, mergeSmallStrata = true, allocation, minPerStratum = 30, roleSplit = { train: 0.5, calibration: 0.25, test: 0.25 }, scoringModel, seed } = options;
    if (!frame.length)
        throw new Error('the frame is empty');
    const topFactor = allocation.topFactor ?? 1;
    if (!(Number.isFinite(topFactor) && topFactor >= 1))
        throw new Error(`allocation.topFactor must be a number >= 1, got ${allocation.topFactor}`);
    if (topFactor !== 1 && allocation.method !== 'proportional')
        throw new Error(`allocation.topFactor applies to method 'proportional' only (got '${allocation.method}')`);
    if (!scoreBands.every((q, i) => q > 0 && q < 1 && (i === 0 || q < scoreBands[i - 1])))
        throw new Error('scoreBands must be descending quantiles strictly between 0 and 1');
    const splitTotal = roleSplit.train + roleSplit.calibration + roleSplit.test;
    if (!(Math.abs(splitTotal - 1) < 1e-9 && roleSplit.calibration > 0 && roleSplit.test > 0 && roleSplit.train >= 0))
        throw new Error('roleSplit must be non-negative, sum to 1, and give calibration and test a share');
    if (!(Number.isInteger(minPerStratum) && minPerStratum >= 4))
        throw new Error('minPerStratum must be an integer >= 4 (2 calibration and 2 test items per stratum)');
    for (const f of frame)
        if (!Number.isFinite(f.signals.score))
            throw new Error(`frame item ${f.id} has a non-finite score`);
    const ids = new Set();
    for (const f of frame) {
        if (ids.has(f.id))
            throw new Error(`duplicate frame id ${f.id}`);
        ids.add(f.id);
    }
    const designId = stableId('design', { ids: frame.map((f) => f.id), scoreBands, bySlice, mergeSmallStrata, allocation, minPerStratum, roleSplit, scoringModel, seed });
    const rand = seededRandom(seed);
    // 1. One item per group, chosen at random.
    const byGroup = new Map();
    for (const f of frame)
        (byGroup.get(f.group) ?? byGroup.set(f.group, []).get(f.group)).push(f);
    const items = [...byGroup.values()].map((g) => g[Math.floor(rand() * g.length)]);
    const duplicatesRemoved = frame.length - items.length;
    // 2. Strata: rule firing × score band (× slice). Band k holds scores in [cut_k, cut_{k-1}).
    const scores = items.map((f) => f.signals.score);
    const cuts = scoreBands.map((q) => weightedQuantile(scores, scores.map(() => 1), q));
    const bandOf = (s) => { let k = 0; while (k < cuts.length && s < cuts[k])
        k++; return k; };
    const bandRange = (k) => [k < cuts.length ? cuts[k] : -Infinity, k === 0 ? Infinity : cuts[k - 1]];
    const nameFor = (rule, bands, slice) => {
        const lo = Math.min(...bands), hi = Math.max(...bands);
        return `${rule ? 'rule' : 'no_rule'}|band${lo === hi ? lo : `${lo}-${hi}`}${bySlice ? `|${slice}` : ''}`;
    };
    let cells = [];
    for (const f of items) {
        const rule = f.signals.ruleFires, band = bandOf(f.signals.score), slice = bySlice ? f.signals.slice ?? '' : '';
        let c = cells.find((x) => x.rule === rule && x.bands[0] === band && x.slice === slice);
        if (!c)
            cells.push((c = { rule, bands: [band], slice, items: [], name: nameFor(rule, [band], slice), from: [nameFor(rule, [band], slice)] }));
        c.items.push(f);
    }
    // The smallest stratum that can give 2 calibration and 2 test items when taken whole.
    let minViable = 4;
    while (Math.round(minViable * roleSplit.calibration) < 2 || Math.round(minViable * roleSplit.test) < 2)
        minViable++;
    if (mergeSmallStrata) {
        const adjacent = (a, b) => Math.min(...b.bands) === Math.max(...a.bands) + 1 || Math.min(...a.bands) === Math.max(...b.bands) + 1;
        const overlap = (a, b) => a.bands.some((k) => b.bands.includes(k));
        const bySize = (a, b) => a.items.length - b.items.length || a.name.localeCompare(b.name);
        for (;;) {
            const small = cells.filter((c) => c.items.length < minViable).sort(bySize)[0];
            if (!small)
                break;
            const others = cells.filter((c) => c !== small);
            if (!others.length)
                throw new Error(`the frame has ${small.items.length} items after deduplication; a design needs at least ${minViable}`);
            const tiers = [
                (c) => c.rule === small.rule && c.slice === small.slice && adjacent(small, c), // neighbouring band
                (c) => c.rule === small.rule && overlap(small, c), // same band, another slice
                (c) => c.slice === small.slice && overlap(small, c), // same band, other rule firing
                () => true,
            ];
            const into = tiers.map((t) => others.filter(t)).find((x) => x.length).sort(bySize)[0];
            const sameKind = into.rule === small.rule && into.slice === small.slice;
            const bands = [...new Set([...into.bands, ...small.bands])].sort((a, b) => a - b);
            cells = [...others.filter((c) => c !== into), {
                    rule: into.rule, slice: into.slice, bands, items: [...into.items, ...small.items],
                    name: sameKind ? nameFor(into.rule, bands, into.slice) : `${into.name}+${small.name}`,
                    from: [...into.from, ...small.from],
                }];
        }
    }
    else {
        const small = cells.filter((c) => c.items.length < minViable);
        if (small.length)
            throw new Error(`strata ${small.map((c) => `${c.name} (N = ${c.items.length})`).join(', ')} are too small for 2 calibration and 2 test items; set mergeSmallStrata or use fewer score bands`);
    }
    const merged = cells.filter((c) => c.from.length > 1).map((c) => ({ name: c.name, from: [...c.from].sort() }));
    const strata = new Map(cells.map((c) => [c.name, c.items]));
    const names = [...strata.keys()].sort();
    const N = names.map((h) => strata.get(h).length);
    // 3. Allocate: floors first, then the rest by the chosen method, capped at N_h.
    const warnings = [];
    const minExpected = allocation.minExpectedPositives ?? 10;
    // One rate vector per head ('' for a single `prior`). `observed` judges materiality: a stratum with no
    // positives in the prior round isn't treated as holding a material share (its smoothed rate is only
    // there to keep a material stratum's size finite) - the thin-strata warnings watch for that risk.
    const rateSets = [];
    if (allocation.method === 'expected-positives') {
        if (!(minExpected > 0))
            throw new Error(`minExpectedPositives must be positive, got ${minExpected}`);
        if (!!allocation.prior === !!allocation.priors)
            throw new Error('expected-positives allocation needs exactly one of prior or priors');
        const sets = allocation.prior ? [['', allocation.prior]] : Object.entries(allocation.priors);
        for (const [head, prior] of sets) {
            const what = head ? `prior for head ${head}` : 'prior';
            for (const k of Object.keys(prior))
                if (!strata.has(k))
                    throw new Error(`${what} names unknown stratum ${k} (strata: ${names.join(', ')})`);
            const parsed = names.map((h) => {
                const v = prior[h];
                if (v === undefined)
                    throw new Error(`${what} has no rate for stratum ${h}; expected-positives allocation needs one per stratum (strata: ${names.join(', ')})`);
                if (typeof v !== 'number' && !(Number.isInteger(v.positives) && Number.isInteger(v.labelled) && v.positives >= 0 && v.labelled >= v.positives))
                    throw new Error(`${what}: ${h} must have integer 0 <= positives <= labelled`);
                const r = typeof v === 'number' ? v : (v.positives + 0.5) / (v.labelled + 1);
                if (!(r > 0 && r <= 1))
                    throw new Error(`${what}: rate for ${h} must be in (0, 1], got ${r}`);
                return { rate: r, observed: typeof v === 'number' ? v : v.labelled ? v.positives / v.labelled : 0 };
            });
            rateSets.push({ head, rates: parsed.map((x) => x.rate), observed: parsed.map((x) => x.observed) });
        }
    }
    const rates = rateSets.length === 1 && rateSets[0].head === '' ? rateSets[0].rates : null;
    // Proportional n_h = total · N_h / ΣN, so stratum h's calibration (and test) share expects
    // total · (N_h / ΣN) · share · rate_h positives: the smallest sufficient total over material strata.
    const share = Math.min(roleSplit.calibration, roleSplit.test);
    const minShare = allocation.minShare ?? 0.02;
    const frameSize = N.reduce((a, b) => a + b, 0);
    let requiredTotal = 0;
    for (const { head, rates: r, observed } of rateSets) {
        const positivesTotal = N.reduce((acc, Nh, i) => acc + Nh * observed[i], 0);
        names.forEach((h, i) => {
            if (!positivesTotal || (N[i] * observed[i]) / positivesTotal < minShare)
                return; // not a material share of this head's positives
            const need = Math.ceil((minExpected * frameSize) / (share * r[i] * N[i]));
            if (need > frameSize)
                warnings.push(`${head ? `head ${head}, ` : ''}stratum ${h}: even the whole frame expects only ${(N[i] * share * r[i]).toFixed(1)} calibration positives there, below ${minExpected}`);
            requiredTotal = Math.max(requiredTotal, Math.min(need, frameSize));
        });
    }
    if (rateSets.length && allocation.total < requiredTotal) {
        throw new Error(`allocation.total ${allocation.total} is below the ${requiredTotal} that proportional allocation needs for every stratum holding >= ${minShare} of ${rateSets.length > 1 ? "each head's" : ''} positives to expect ${minExpected} calibration positives`.replace('of  positives', 'of positives'));
    }
    const floors = N.map((Nh) => Math.min(minPerStratum, Nh));
    const minimum = floors.reduce((a, b) => a + b, 0);
    if (allocation.total < minimum) {
        throw new Error(`allocation.total ${allocation.total} is below the ${minimum} that ${rateSets.length ? `minExpectedPositives ${minExpected} and minPerStratum ${minPerStratum} need` : `minPerStratum ${minPerStratum} needs`} across ${names.length} strata`);
    }
    let n;
    if (allocation.method === 'manual') {
        // (expected-positives falls through to proportional below, sized by the check above.)
        const manual = allocation.manual ?? {};
        for (const k of Object.keys(manual))
            if (!strata.has(k))
                throw new Error(`manual allocation names unknown stratum ${k} (strata: ${names.join(', ')})`);
        n = names.map((h, i) => Math.min(N[i], Math.max(floors[i], manual[h] ?? 0)));
        const sum = n.reduce((a, b) => a + b, 0);
        if (sum > allocation.total)
            throw new Error(`manual allocation needs ${sum} items after floors, above allocation.total ${allocation.total}`);
    }
    else {
        // Proportional with floors and caps: find c with Σ clamp(c·f_h·N_h, floor_h, N_h) = total, where
        // f_h = topFactor for strata in the top score band and 1 otherwise.
        const total = Math.min(allocation.total, N.reduce((a, b) => a + b, 0));
        const inTop = new Map(cells.map((c) => [c.name, Math.min(...c.bands) === 0]));
        const f = names.map((h) => (inTop.get(h) ? topFactor : 1));
        let lo = 0, hi = 1 / Math.min(...f);
        const at = (c) => N.reduce((s, Nh, i) => s + Math.min(Nh, Math.max(floors[i], c * f[i] * Nh)), 0);
        for (let it = 0; it < 200; it++) {
            const mid = (lo + hi) / 2;
            if (at(mid) < total)
                lo = mid;
            else
                hi = mid;
        }
        n = roundWithin(N.map((Nh, i) => Math.min(Nh, Math.max(floors[i], hi * f[i] * Nh))), floors, N, total);
        if (topFactor !== 1)
            warnings.push(`allocation.topFactor ${topFactor}: valid only if signals.score (${scoringModel}) is independent of the head being certified, e.g. an offline teacher's score; oversampling by the head's own score makes recall bounds under-cover`);
        if (topFactor > 4)
            warnings.push(`allocation.topFactor ${topFactor} is above 4, the largest factor tested; check coverage before relying on it`);
    }
    // 4-5. Draw, then assign roles within each stratum, before any labelling.
    const records = [];
    const table = names.map((name, i) => {
        const drawn = shuffle([...strata.get(name)], rand).slice(0, n[i]);
        const nCal = Math.round(n[i] * roleSplit.calibration), nTest = Math.round(n[i] * roleSplit.test);
        const nTrain = n[i] - nCal - nTest;
        if (nCal < 2 || nTest < 2 || nTrain < 0) {
            throw new Error(`stratum ${name} (N = ${N[i]}) gets ${nCal} calibration and ${nTest} test items; each needs at least 2 - raise minPerStratum or use fewer score bands`);
        }
        const counts = { train: nTrain, calibration: nCal, test: nTest };
        const roles = shuffle([...Array(nCal).fill('calibration'), ...Array(nTest).fill('test'), ...Array(nTrain).fill('train')], rand);
        drawn.forEach((f, j) => {
            const role = roles[j];
            const source = { kind: 'sampled', designId, stratum: name, inclusionProb: counts[role] / N[i], stratumSize: N[i] };
            records.push({ id: f.id, text: f.text, group: f.group, role, source, labels: {} });
        });
        const bandsIn = strata.get(name).map((f) => bandOf(f.signals.score));
        const range = [bandRange(Math.max(...bandsIn))[0], bandRange(Math.min(...bandsIn))[1]];
        return {
            name, N: N[i], n: n[i], pi: n[i] / N[i], roles: counts, band: range,
            ...(rates ? { priorRate: rates[i], expectedCalibrationPositives: rates[i] * nCal } : {}),
        };
    });
    const summary = {
        designId, scoringModel, seed, duplicatesRemoved, strata: table, warnings, merged,
        allocation: {
            method: allocation.method,
            ...(topFactor !== 1 ? { topFactor } : {}),
            ...(rateSets.length ? { minExpectedPositives: minExpected, minShare, requiredTotal } : {}),
            ...(rates ? { priorRates: Object.fromEntries(names.map((h, i) => [h, rates[i]])) } : {}),
            ...(rateSets.length && !rates ? { priorRatesByHead: Object.fromEntries(rateSets.map((s) => [s.head, Object.fromEntries(names.map((h, i) => [h, s.rates[i]]))])) } : {}),
        },
    };
    const stratumOf = {};
    for (const [name, members] of strata)
        for (const f of members)
            stratumOf[f.id] = `${designId}/${name}`;
    return { designId, records, design: summary, stratumOf };
}
/**
 * The smallest `allocation.total` an expected-positives design needs: stratifies the frame exactly
 * as designSample would (including merges) and sizes it, without drawing anything you'd use.
 */
export function requiredSampleSize(options) {
    const frameSize = options.frame.length;
    const d = designSample({ ...options, seed: 0, allocation: { ...options.allocation, method: 'expected-positives', total: frameSize } });
    return d.design.allocation.requiredTotal ?? 0;
}
/**
 * The stratified design of a set of sampled records (one role, one head's labelled subset): for
 * solvers' estimators. Inclusion probabilities are recomputed as n_labelled / N_h, so items that
 * were budget-dropped or skipped are simply absent.
 */
export function designOf(records) {
    const strata = [];
    const stratumSizes = {};
    const count = new Map();
    for (const r of records) {
        if (r.source.kind !== 'sampled')
            throw new Error(`record ${r.id} is not sampled (${r.source.kind})`);
        const key = `${r.source.designId}/${r.source.stratum}`;
        if (stratumSizes[key] !== undefined && stratumSizes[key] !== r.source.stratumSize)
            throw new Error(`stratum ${key} has inconsistent stratumSize`);
        stratumSizes[key] = r.source.stratumSize;
        strata.push(key);
        count.set(key, (count.get(key) ?? 0) + 1);
    }
    return { inclusionProbs: strata.map((k) => count.get(k) / stratumSizes[k]), strata, stratumSizes };
}
/**
 * One blinded queue. Over budget, sampled items are dropped at random (inclusion probabilities are
 * recomputed later); retrieved and verification items are dropped from the end, so pass them in
 * priority order.
 */
export function labelQueue(options) {
    const { items, budget, seed } = options;
    const rand = seededRandom(seed);
    for (const it of items)
        if (!it.heads.length)
            throw new Error(`queue item ${it.record.id} has no heads to label`);
    const kept = [];
    const dropped = { sampled: 0, retrieved: 0, verification: 0 };
    for (const m of ['sampled', 'retrieved', 'verification']) {
        const of = items.filter((it) => it.mechanism === m);
        const limit = Math.max(0, Math.floor(budget[m] ?? 0));
        const take = m === 'sampled' ? shuffle([...of], rand).slice(0, limit) : of.slice(0, limit);
        dropped[m] = of.length - take.length;
        kept.push(...take);
    }
    const queueId = stableId('queue', { ids: kept.map((it) => it.record.id), seed });
    shuffle(kept, rand);
    const key = { queueId, items: {} };
    const review = kept.map((it, j) => {
        const itemId = stableId('item', [queueId, j]);
        key.items[itemId] = { record: it.record, mechanism: it.mechanism, heads: [...it.heads] };
        return { itemId, text: it.record.text, heads: [...it.heads] };
    });
    return { queueId, review, key, dropped };
}
/**
 * Applies reviewers' labels. An item counts as labelled only when every requested head came back
 * 0 or 1; anything else is a skip. Sampled records get inclusionProb = labelled_{h,role} / N_h.
 */
export function applyReviews(key, reviews) {
    const byId = new Map(reviews.map((r) => [r.itemId, r]));
    for (const r of reviews)
        if (!key.items[r.itemId])
            throw new Error(`review for unknown item ${r.itemId}`);
    const out = [];
    const skipRate = {};
    const labelledPerCell = new Map();
    const cell = (r) => (r.source.kind === 'sampled' ? `${r.source.designId}/${r.source.stratum}/${r.role}` : null);
    for (const [itemId, item] of Object.entries(key.items)) {
        const review = byId.get(itemId);
        const complete = !!review && item.heads.every((h) => review.labels[h] === 0 || review.labels[h] === 1);
        const c = cell(item.record);
        if (c) {
            const stratum = c.split('/').slice(0, 2).join('/');
            skipRate[stratum] ??= { queued: 0, labelled: 0, rate: 0 };
            skipRate[stratum].queued++;
            if (complete) {
                skipRate[stratum].labelled++;
                labelledPerCell.set(c, (labelledPerCell.get(c) ?? 0) + 1);
            }
        }
        if (!complete)
            continue;
        const labels = { ...item.record.labels };
        for (const h of item.heads)
            labels[h] = review.labels[h];
        out.push({ ...item.record, labels, labelledBy: 'human', ...(item.mechanism === 'verification' ? { verified: true } : {}) });
    }
    const records = out.map((r) => {
        const c = cell(r);
        if (!c || r.source.kind !== 'sampled')
            return r;
        return { ...r, source: { ...r.source, inclusionProb: labelledPerCell.get(c) / r.source.stratumSize } };
    });
    const warnings = [];
    for (const [s, v] of Object.entries(skipRate)) {
        v.rate = v.queued ? 1 - v.labelled / v.queued : 0;
        if (v.rate > 0.05)
            warnings.push(`stratum ${s}: ${(100 * v.rate).toFixed(1)}% of queued items were skipped - if skips relate to content, design estimates can be biased`);
    }
    for (const [c, k] of labelledPerCell) {
        // Linearised bounds pool a one-item stratum's variance with a partner stratum (solvers' collapsed
        // strata, FINDINGS O6); exact bounds need none. Still worth knowing: the pooled variance is conservative.
        if (k < 2 && !c.endsWith('/train'))
            warnings.push(`${c}: only ${k} labelled item(s); linearised bounds will pool its variance with a partner stratum (collapsed strata)`);
    }
    return { records, skipRate, warnings };
}
