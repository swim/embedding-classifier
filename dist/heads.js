/**
 * Head types beyond linear: a head may turn the embedding into a few features before its usual
 * w·x + b, calibration and threshold. The features are computed the same way at training and at
 * runtime (same functions, same stored numbers), so what was evaluated is what runs.
 *
 *   linear  the embedding itself (today's heads; no `features` block)
 *   knn     one feature: mean cosine to the k nearest labelled positives minus mean cosine to the k
 *           nearest labelled negatives, among the artifact's shared reference rows (k = 10)
 *   stack   three features: a linear head's logit, the knn score, and the knn score on 50 principal
 *           components (fitted on the head's training rows); the head's own w·x + b is the blend
 *
 * Training rows always get OUT-OF-FOLD features (3 folds by group): a row is never scored by a
 * reference or model that contains it, or a kNN score would find the row itself. Everything else
 * (calibration, test, background, runtime) is scored against the full reference.
 *
 * The reference is one block per artifact, shared by every knn and stack head: the training rows'
 * embeddings (float32, base64) and a label column per head. It is data derived from training
 * messages (embeddings can be partly inverted to text): treat the artifact accordingly.
 *
 * When each pays: knn suits clustered labels (positives that come in a few tight groups) and does
 * poorly on diffuse ones; the stack hedges between them. No type wins every label, which is why
 * type 'auto' chooses per label by cross-validation on the training rows.
 */
import { decisionFunction, fitLogistic, seededRandom } from '@liquidau/solvers';
export const HEAD_TYPES = ['linear', 'knn', 'stack'];
export const KNN_K = 10;
const PCA_DIMS = 50;
const FOLDS = 3;
// ---------- encoding (edge-safe: no Buffer; the platform's atob/btoa when present) ----------
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const g = globalThis;
function toBase64(bytes) {
    if (g.btoa) {
        let bin = '';
        for (let i = 0; i < bytes.length; i += 0x8000)
            bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        return g.btoa(bin);
    }
    let out = '';
    for (let i = 0; i < bytes.length; i += 3) {
        const a = bytes[i], b = bytes[i + 1] ?? 0, c = bytes[i + 2] ?? 0;
        const n = (a << 16) | (b << 8) | c;
        out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (i + 1 < bytes.length ? B64[(n >> 6) & 63] : '=') + (i + 2 < bytes.length ? B64[n & 63] : '=');
    }
    return out;
}
const B64_INDEX = new Map([...B64].map((ch, i) => [ch, i]));
function fromBase64(s) {
    if (s.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(s))
        throw new Error('reference data is not base64');
    if (g.atob) {
        const bin = g.atob(s);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++)
            out[i] = bin.charCodeAt(i);
        return out;
    }
    const pad = s.endsWith('==') ? 2 : s.endsWith('=') ? 1 : 0;
    const out = new Uint8Array((s.length / 4) * 3 - pad);
    let o = 0;
    for (let i = 0; i < s.length; i += 4) {
        const v = [0, 1, 2, 3].map((k) => (s[i + k] === '=' ? 0 : B64_INDEX.get(s[i + k])));
        const n = (v[0] << 18) | (v[1] << 12) | (v[2] << 6) | v[3];
        if (o < out.length)
            out[o++] = (n >> 16) & 255;
        if (o < out.length)
            out[o++] = (n >> 8) & 255;
        if (o < out.length)
            out[o++] = n & 255;
    }
    return out;
}
export function encodeReference(rows, labels, encoding = 'f32') {
    const dims = rows[0]?.length ?? 0;
    if (encoding === 'int8') {
        const q = new Int8Array(rows.length * dims), scales = [];
        rows.forEach((r, i) => {
            let m = 0;
            for (let t = 0; t < dims; t++)
                m = Math.max(m, Math.abs(r[t]));
            const scale = m / 127 || 1;
            scales.push(scale);
            for (let t = 0; t < dims; t++)
                q[i * dims + t] = Math.max(-127, Math.min(127, Math.round(r[t] / scale)));
        });
        return { encoding, rows: rows.length, dims, data: toBase64(new Uint8Array(q.buffer)), scales, labels };
    }
    const f = new Float32Array(rows.length * dims);
    rows.forEach((r, i) => { for (let t = 0; t < dims; t++)
        f[i * dims + t] = r[t]; });
    const bytes = new Uint8Array(f.length * 4);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < f.length; i++)
        view.setFloat32(i * 4, f[i], true);
    return { encoding: 'f32', rows: rows.length, dims, data: toBase64(bytes), labels };
}
/** The reference's rows as float32 (views into one buffer). */
export function decodeReference(ref) {
    const bytes = fromBase64(ref.data);
    const width = ref.encoding === 'int8' ? 1 : 4;
    if (bytes.length !== ref.rows * ref.dims * width)
        throw new Error(`reference data holds ${bytes.length} bytes, expected ${ref.rows * ref.dims * width}`);
    const flat = new Float32Array(ref.rows * ref.dims);
    if (ref.encoding === 'int8') {
        const q = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        for (let i = 0; i < ref.rows; i++) {
            const s = ref.scales[i];
            for (let t = 0; t < ref.dims; t++)
                flat[i * ref.dims + t] = q[i * ref.dims + t] * s;
        }
    }
    else {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        for (let i = 0; i < flat.length; i++)
            flat[i] = view.getFloat32(i * 4, true);
    }
    return Array.from({ length: ref.rows }, (_, i) => flat.subarray(i * ref.dims, (i + 1) * ref.dims));
}
// ---------- the shared feature functions (training and runtime) ----------
export function dot(a, b) {
    let s = 0;
    for (let k = 0; k < a.length; k++)
        s += a[k] * b[k];
    return s;
}
/**
 * Cosines (dot products) of x with every reference row: one monomorphic loop, summed in index order
 * exactly as dot() sums, so training and runtime agree to the last bit.
 */
export function similarities(x, rows) {
    const d = x.length, xd = new Float64Array(d), out = new Float64Array(rows.length);
    for (let t = 0; t < d; t++)
        xd[t] = x[t];
    for (let j = 0; j < rows.length; j++) {
        const r = rows[j];
        let s = 0;
        for (let t = 0; t < d; t++)
            s += xd[t] * r[t];
        out[j] = s;
    }
    return out;
}
/** Mean of the k largest of sims[j] over j in `which` (summed in descending order). */
function meanTop(sims, which, k) {
    const top = [];
    for (const j of which) {
        const v = sims[j];
        if (top.length < k) {
            top.push(v);
            if (top.length === k)
                top.sort((a, b) => b - a);
            continue;
        }
        if (v <= top[k - 1])
            continue;
        let p = k - 1;
        while (p > 0 && top[p - 1] < v) {
            top[p] = top[p - 1];
            p--;
        }
        top[p] = v;
    }
    if (top.length < k)
        top.sort((a, b) => b - a);
    return top.reduce((a, b) => a + b, 0) / Math.max(1, top.length);
}
/** The knn feature from similarities to every reference row. */
export function knnFeature(sims, pos, neg, k) {
    return meanTop(sims, pos, k) - meanTop(sims, neg, k);
}
/** Projection onto the components, unit-normalised (for cosine kNN). */
export function project(p, x) {
    const z = p.components.map((q) => { let s = 0; for (let t = 0; t < q.length; t++)
        s += (x[t] - p.mean[t]) * q[t]; return s; });
    const n = Math.hypot(...z) || 1;
    return z.map((v) => v / n);
}
function orthonormalise(cols) {
    for (let j = 0; j < cols.length; j++) {
        for (let i = 0; i < j; i++) {
            let d = 0;
            for (let t = 0; t < cols[j].length; t++)
                d += cols[j][t] * cols[i][t];
            for (let t = 0; t < cols[j].length; t++)
                cols[j][t] -= d * cols[i][t];
        }
        let n = 0;
        for (let t = 0; t < cols[j].length; t++)
            n += cols[j][t] ** 2;
        n = Math.sqrt(n) || 1;
        for (let t = 0; t < cols[j].length; t++)
            cols[j][t] /= n;
    }
}
/** Top-k principal components by seeded randomized subspace iteration. */
export function fitPca(rows, k, seed, iterations = 6) {
    const n = rows.length, d = rows[0].length;
    k = Math.min(k, d, n);
    const mean = new Float64Array(d);
    for (const r of rows)
        for (let t = 0; t < d; t++)
            mean[t] += r[t] / n;
    const X = rows.map((r) => Float64Array.from({ length: d }, (_, t) => r[t] - mean[t]));
    const rand = seededRandom(seed);
    const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    let Q = Array.from({ length: k }, () => Float64Array.from({ length: d }, gauss));
    orthonormalise(Q);
    for (let it = 0; it < iterations; it++) {
        const XQ = X.map((x) => Q.map((q) => { let s = 0; for (let t = 0; t < d; t++)
            s += x[t] * q[t]; return s; }));
        Q = Q.map((_, j) => { const c = new Float64Array(d); X.forEach((x, i) => { const w = XQ[i][j]; for (let t = 0; t < d; t++)
            c[t] += w * x[t]; }); return c; });
        orthonormalise(Q);
    }
    return { mean, components: Q };
}
const runtimeCache = new WeakMap();
export function runtimeReference(ref) {
    let r = runtimeCache.get(ref);
    if (!r) {
        const pos = new Map(), neg = new Map();
        for (const [head, ys] of Object.entries(ref.labels)) {
            pos.set(head, ys.flatMap((y, j) => (y === 1 ? [j] : [])));
            neg.set(head, ys.flatMap((y, j) => (y === 0 ? [j] : [])));
        }
        r = { rows: decodeReference(ref), pos, neg, projected: new Map() };
        runtimeCache.set(ref, r);
    }
    return r;
}
/** A stack head's reference rows on its principal components (computed once per head, then cached). */
export function projectedRows(head, features, ref) {
    let proj = ref.projected.get(head);
    if (!proj) {
        proj = ref.rows.map((row) => project(features.pca, row));
        ref.projected.set(head, proj);
    }
    return proj;
}
/** Features for one head at runtime; `sims` are the embedding's cosines to every reference row (computed once per message). */
export function headFeatureVector(head, features, embedding, ref, sims) {
    const pos = ref.pos.get(head), neg = ref.neg.get(head);
    if (!pos || !neg)
        throw new Error(`head ${head} has no reference labels`);
    const knn = knnFeature(sims, pos, neg, features.k);
    if (features.kind === 'knn')
        return [knn];
    const proj = projectedRows(head, features, ref);
    const z = project(features.pca, embedding);
    const simsPca = proj.map((v) => dot(z, v));
    return [decisionFunction({ coef: features.linear.weights, intercept: features.linear.bias }, embedding), knn, knnFeature(simsPca, pos, neg, features.k)];
}
// ---------- training ----------
/** Fold of a row (FNV-style hash of its key). */
export function foldOf(key) {
    return [...key].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7) % FOLDS;
}
/**
 * Fits a head type's features. Out-of-fold for the training rows: the reference rows and models of
 * the row's own fold are left out.
 */
export function fitFeatures(type, t) {
    if (type === 'linear') {
        return { train: t.rows.map((i) => Array.from(t.X[i])), apply: (x) => Array.from(x), converged: true };
    }
    const k = KNN_K;
    const refFold = t.referenceKeys.map(foldOf);
    const rowFold = t.keys.map(foldOf);
    const labelled = (keep) => ({
        pos: t.referenceLabels.flatMap((y, j) => (y === 1 && keep(j) ? [j] : [])),
        neg: t.referenceLabels.flatMap((y, j) => (y === 0 && keep(j) ? [j] : [])),
    });
    const all = labelled(() => true);
    const byFold = Array.from({ length: FOLDS }, (_, f) => labelled((j) => refFold[j] !== f));
    const simsTo = (x) => similarities(x, t.reference);
    const knnTrain = t.rows.map((i, n) => { const s = byFold[rowFold[n]]; return knnFeature(simsTo(t.X[i]), s.pos, s.neg, k); });
    if (type === 'knn') {
        return {
            features: { kind: 'knn', k },
            train: knnTrain.map((v) => [v]),
            apply: (x) => [knnFeature(simsTo(x), all.pos, all.neg, k)],
            converged: true,
        };
    }
    // stack: linear logit, knn, knn on 50 principal components; each out-of-fold for training rows.
    const fitLinear = (rows) => fitLogistic(rows.map((n) => t.X[t.rows[n]]), rows.map((n) => t.y[n]), {
        C: t.C, classWeight: 'balanced', ...(rows.some((n) => t.weights[n] !== 1) ? { sampleWeight: rows.map((n) => t.weights[n]) } : {}),
    });
    const pcaOf = (rows) => fitPca(rows.map((n) => t.X[t.rows[n]]), PCA_DIMS, t.seed);
    const knnPcaWith = (pca, keep) => {
        const proj = t.reference.map((r, j) => (keep(j) ? project(pca, r) : null));
        const s = labelled((j) => keep(j) && proj[j] !== null);
        return (x) => { const z = project(pca, x); return knnFeature(proj.map((v) => (v ? dot(z, v) : 0)), s.pos, s.neg, k); };
    };
    const linearTrain = new Array(t.rows.length), knnPcaTrain = new Array(t.rows.length);
    for (let f = 0; f < FOLDS; f++) {
        const fitRows = t.rows.map((_, n) => n).filter((n) => rowFold[n] !== f);
        const inFold = t.rows.map((_, n) => n).filter((n) => rowFold[n] === f);
        if (!inFold.length)
            continue;
        const lin = fitLinear(fitRows);
        const knnPca = knnPcaWith(pcaOf(fitRows), (j) => refFold[j] !== f);
        for (const n of inFold) {
            linearTrain[n] = decisionFunction(lin, t.X[t.rows[n]]);
            knnPcaTrain[n] = knnPca(t.X[t.rows[n]]);
        }
    }
    const lin = fitLinear(t.rows.map((_, n) => n));
    const pca = pcaOf(t.rows.map((_, n) => n));
    const features = { kind: 'stack', k, linear: { weights: [...lin.coef], bias: lin.intercept }, pca: { mean: [...pca.mean], components: pca.components.map((c) => [...c]) } };
    const knnPcaFull = knnPcaWith(features.pca, () => true);
    return {
        features,
        train: t.rows.map((_, n) => [linearTrain[n], knnTrain[n], knnPcaTrain[n]]),
        apply: (x) => [decisionFunction({ coef: features.linear.weights, intercept: features.linear.bias }, x), knnFeature(simsTo(x), all.pos, all.neg, k), knnPcaFull(x)],
        converged: lin.converged,
    };
}
/**
 * A head's guarantee cost on scores, with an oracle threshold on the given rows (lower is better):
 * false alarms at the target recall, or recall lost at the target precision.
 */
export function guaranteeCost(scores, y, policy) {
    const order = scores.map((s, i) => [s, y[i]]).sort((a, b) => b[0] - a[0]);
    const pos = y.filter((v) => v === 1).length, neg = y.length - pos;
    if (!pos || !neg)
        return 1;
    let tp = 0, fp = 0;
    if (policy.kind === 'recall') {
        for (const [, v] of order) {
            if (v)
                tp++;
            else
                fp++;
            if (tp / pos >= policy.targetRecall)
                return fp / neg;
        }
        return 1;
    }
    let best = 0;
    for (const [, v] of order) {
        if (v)
            tp++;
        else
            fp++;
        if (tp / (tp + fp) >= policy.targetPrecision)
            best = Math.max(best, tp / pos);
    }
    return 1 - best;
}
/** Out-of-fold scores of a fitted feature set's head layer on the training rows (for choosing a type). */
export function crossValidatedScores(fitted, t) {
    const rowFold = t.keys.map(foldOf);
    const out = new Array(t.rows.length);
    for (let f = 0; f < FOLDS; f++) {
        const fitRows = t.rows.map((_, n) => n).filter((n) => rowFold[n] !== f);
        const inFold = t.rows.map((_, n) => n).filter((n) => rowFold[n] === f);
        if (!inFold.length || new Set(fitRows.map((n) => t.y[n])).size < 2) {
            for (const n of inFold)
                out[n] = 0;
            continue;
        }
        const m = fitLogistic(fitRows.map((n) => fitted.train[n]), fitRows.map((n) => t.y[n]), {
            C: t.C, classWeight: 'balanced', ...(fitRows.some((n) => t.weights[n] !== 1) ? { sampleWeight: fitRows.map((n) => t.weights[n]) } : {}),
        });
        for (const n of inFold)
            out[n] = decisionFunction(m, fitted.train[n]);
    }
    return out;
}
