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
    const { frame, scoreBands, bySlice = false, allocation, minPerStratum = 30, roleSplit = { train: 0.5, calibration: 0.25, test: 0.25 }, scoringModel, seed } = options;
    if (!frame.length)
        throw new Error('the frame is empty');
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
    const designId = stableId('design', { ids: frame.map((f) => f.id), scoreBands, bySlice, allocation, minPerStratum, roleSplit, scoringModel, seed });
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
    const nameOf = (f) => `${f.signals.ruleFires ? 'rule' : 'no_rule'}|band${bandOf(f.signals.score)}${bySlice ? `|${f.signals.slice ?? ''}` : ''}`;
    const strata = new Map();
    for (const f of items)
        (strata.get(nameOf(f)) ?? strata.set(nameOf(f), []).get(nameOf(f))).push(f);
    const names = [...strata.keys()].sort();
    const N = names.map((h) => strata.get(h).length);
    // 3. Allocate: floors first, then the rest by the chosen method, capped at N_h.
    const floors = N.map((Nh) => Math.min(minPerStratum, Nh));
    const minimum = floors.reduce((a, b) => a + b, 0);
    if (allocation.total < minimum)
        throw new Error(`allocation.total ${allocation.total} is below the ${minimum} that minPerStratum ${minPerStratum} needs across ${names.length} strata`);
    let n;
    if (allocation.method === 'manual') {
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
        // Proportional with floors and caps: find c with Σ clamp(c·N_h, floor_h, N_h) = total.
        const total = Math.min(allocation.total, N.reduce((a, b) => a + b, 0));
        let lo = 0, hi = 1;
        const at = (c) => N.reduce((s, Nh, i) => s + Math.min(Nh, Math.max(floors[i], c * Nh)), 0);
        for (let it = 0; it < 200; it++) {
            const mid = (lo + hi) / 2;
            if (at(mid) < total)
                lo = mid;
            else
                hi = mid;
        }
        n = roundWithin(N.map((Nh, i) => Math.min(Nh, Math.max(floors[i], hi * Nh))), floors, N, total);
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
        const band = bandOf(strata.get(name)[0].signals.score);
        return { name, N: N[i], n: n[i], pi: n[i] / N[i], roles: counts, band: bandRange(band) };
    });
    return { designId, records, design: { designId, scoringModel, seed, duplicatesRemoved, strata: table } };
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
        if (k < 2 && !c.endsWith('/train'))
            warnings.push(`${c}: only ${k} labelled item(s); design variance needs at least 2 per stratum`);
    }
    return { records, skipRate, warnings };
}
