/**
 * The finer-label route score: a head feature for routes that are unions of a finer label, such as an
 * intent taxonomy under a "needs a person" route (lab stages SL and JC, 10 October).
 *
 *   finer model   one multinomial L2 logistic regression over the finer label, shared by the artifact's
 *                 heads: fitted on the training rows that carry a finer label, on the embedding
 *                 standardised over those rows (folded into the stored weights), C = 0.01 by default
 *   route score   s(x) = Σ_k P(k | x) · r_k, with r_k the head's training rows' route rate for finer
 *                 label k, smoothed toward their overall rate: (positives_k + p0) / (rows_k + 1)
 *   head feature  [ ...embedding, scale · logit(s(x)) ] (scale 0.25), into the head's usual w·x + b,
 *                 calibration and threshold: the guarantee is certified exactly as for any head
 *
 * Training rows the finer model was fitted on get OUT-OF-FOLD probabilities (5 folds by fold key), so
 * a row never scores itself. Everything else (calibration, test, background, runtime) is scored by the
 * stored model, the same function at training and at runtime.
 *
 * When it pays: in the lab it lowered false alarms or raised recall on Banking77 `urgent`, ABCD `access`
 * and `refund` and ToxicChat `toxic` (both encoders tested), and it never cost a guarantee. It needs the
 * finer label on some training rows: a taxonomy the labelling already records (intents, flows, product
 * lines). Without one there is nothing to fit.
 */
import { fitSoftmax, softmaxProbabilities } from '@liquidau/solvers';

/** The artifact's shared finer-label model (`artifact.finer`). */
export interface FinerModel {
  /** The finer labels, in the order of `coef` and of every finer head's route_rates. */
  classes: string[];
  /** One row of `dimensions` weights per class, on the raw embedding (the standardisation is folded in). */
  coef: number[][];
  intercept: number[];
  /** The inverse regularisation it was fitted with, and on how many training rows (for audit). */
  C: number;
  rows: number;
}

/** A finer head's feature block: its route rate per finer label, and the scale of its logit feature. */
export interface FinerFeatures {
  kind: 'finer';
  route_rates: number[];
  scale: number;
}

export const FINER_SCALE = 0.25;
export const FINER_C = 0.01;
const FOLDS = 5;
const EPS = 1e-6;

const logit = (p: number) => { const q = Math.min(1 - EPS, Math.max(EPS, p)); return Math.log(q / (1 - q)); };

/** FNV-1a of the fold key, mod the fold count: rows sharing a key share a fold. */
function foldOf(key: string): number {
  let h = 2166136261;
  for (let k = 0; k < key.length; k++) h = Math.imul(h ^ key.charCodeAt(k), 16777619);
  return (h >>> 0) % FOLDS;
}

/** P(finer label | embedding) under the stored model. */
export function finerProbabilities(model: Pick<FinerModel, 'coef' | 'intercept'>, embedding: ArrayLike<number>): number[] {
  return softmaxProbabilities(model, embedding);
}

/** A finer head's feature vector: the embedding and the scaled logit of its route score. */
export function finerFeatureVector(features: FinerFeatures, embedding: ArrayLike<number>, probabilities: readonly number[]): number[] {
  let s = 0;
  for (let k = 0; k < probabilities.length; k++) s += probabilities[k] * features.route_rates[k];
  return [...Array.from(embedding), features.scale * logit(s)];
}

export interface FittedFinerModel {
  model: FinerModel;
  /** Probabilities for each row the model was fitted on (out of fold), by row index into X. */
  outOfFold: Map<number, number[]>;
  /** Each fitted row's finer label index, by row index into X. */
  classOf: Map<number, number>;
  converged: boolean;
}

/**
 * Fits the shared finer model on `rows` (indices into X, each with a non-null label), plus one model
 * per fold for the out-of-fold probabilities. Throws with fewer than two finer labels.
 */
export function fitFinerModel(X: ReadonlyArray<ArrayLike<number>>, labels: ReadonlyArray<string | null>, rows: readonly number[], keys: (i: number) => string, C = FINER_C): FittedFinerModel {
  const classes = [...new Set(rows.map((i) => labels[i]!))].sort();
  if (classes.length < 2) throw new Error(`finer: the training rows carry ${classes.length} finer label(s); the model needs at least two`);
  const d = X[rows[0]].length;
  // Standardise over the fitted rows (the lab's recipe), then fold it into the stored weights.
  const mean = new Float64Array(d), sd = new Float64Array(d);
  for (const i of rows) for (let j = 0; j < d; j++) mean[j] += X[i][j] / rows.length;
  for (const i of rows) for (let j = 0; j < d; j++) sd[j] += (X[i][j] - mean[j]) ** 2 / rows.length;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  const z = (x: ArrayLike<number>) => Float64Array.from({ length: d }, (_, j) => (x[j] - mean[j]) / sd[j]);
  const Z = new Map(rows.map((i) => [i, z(X[i])]));
  const classOf = new Map(rows.map((i) => [i, classes.indexOf(labels[i]!)]));
  const fit = (subset: readonly number[]) => fitSoftmax(subset.map((i) => Z.get(i)!), subset.map((i) => classOf.get(i)!), classes.length, { C, maxIter: 200 });
  const full = fit(rows);
  const folded = (m: { coef: number[][]; intercept: number[] }) => ({
    coef: m.coef.map((row) => row.map((w, j) => w / sd[j])),
    intercept: m.intercept.map((b, k) => b - m.coef[k].reduce((s, w, j) => s + (w * mean[j]) / sd[j], 0)),
  });
  let converged = full.converged;
  const outOfFold = new Map<number, number[]>();
  const fold = new Map(rows.map((i) => [i, foldOf(keys(i))]));
  for (let f = 0; f < FOLDS; f++) {
    const held = rows.filter((i) => fold.get(i) === f);
    if (!held.length) continue;
    const rest = rows.filter((i) => fold.get(i) !== f);
    // A fold model needs two classes; with one, its held rows fall back to the full model's probabilities.
    if (new Set(rest.map((i) => classOf.get(i))).size < 2) continue;
    const m = fit(rest);
    converged &&= m.converged;
    for (const i of held) outOfFold.set(i, softmaxProbabilities(m, Z.get(i)!));
  }
  const model: FinerModel = { classes, ...folded(full), C, rows: rows.length };
  for (const i of rows) if (!outOfFold.has(i)) outOfFold.set(i, finerProbabilities(model, X[i]));
  return { model, outOfFold, classOf, converged };
}

/**
 * A head's route rate per finer label, from its training rows that carry both labels: smoothed toward
 * their overall rate, so a finer label with few rows stays near it.
 */
export function finerRouteRates(K: number, rows: ReadonlyArray<{ k: number; y: 0 | 1 }>): number[] {
  if (!rows.length) throw new Error('finer: no training rows carry both the finer label and this head\'s label');
  const p0 = rows.reduce((s, r) => s + r.y, 0) / rows.length;
  const pos = new Float64Array(K), n = new Float64Array(K);
  for (const r of rows) { n[r.k]++; pos[r.k] += r.y; }
  return Array.from(n, (nk, k) => (pos[k] + p0) / (nk + 1));
}

/** Throws unless `f` is a usable finer model for `dims`-dimensional embeddings. */
export function validateFinerModel(f: FinerModel, dims: number): void {
  const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
  if (!f || typeof f !== 'object' || !Array.isArray(f.classes) || f.classes.length < 2 || !f.classes.every((c) => typeof c === 'string') || new Set(f.classes).size !== f.classes.length) {
    throw new Error('finer model needs at least two distinct class names');
  }
  if (!Array.isArray(f.coef) || f.coef.length !== f.classes.length || !f.coef.every((row) => Array.isArray(row) && row.length === dims && row.every(finite))) {
    throw new Error(`finer model needs ${f.classes.length} rows of ${dims} finite weights`);
  }
  if (!Array.isArray(f.intercept) || f.intercept.length !== f.classes.length || !f.intercept.every(finite)) throw new Error('finer model needs one finite intercept per class');
  if (!(finite(f.C) && f.C > 0) || !(Number.isInteger(f.rows) && f.rows > 0)) throw new Error('finer model needs a positive C and row count');
}
