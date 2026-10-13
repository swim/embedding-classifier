/**
 * The edge tier ('liquidau-edge/1'): cheap scorers that settle confident messages before the embedding
 * model. Per head, a cross-fitted logistic blend of component scores:
 *
 *   rules     whether a certified rule for this label fired (passed in by the caller)
 *   lexical   a logistic model on hashed word and character n-grams (text-preprocessing's lexicalFeatures)
 *   static    a logistic model on a static embedding (text-preprocessing's createStaticEmbedder)
 *   vector    a logistic model on any other host-supplied vector (e.g. a small transformer's embedding)
 *
 * Two certified decisions on the calibration rows (design-based when a design is given, 95%):
 *
 *   fire      score > fire: at most maxFalseAlarms of negatives are fired (the claim rules carry)
 *   dismiss   score < dismiss: at most maxDismissed of positives are dismissed
 *
 * trainEdge returns the edge document and, per head, which rows the edge fires and dismisses: pass
 * them to trainHeads (HeadInput.edge) so the head's guarantee is certified jointly (edge fires count
 * as caught, edge dismissals as misses). In the lab (stages ET and MX, 9 October) regex + lexical +
 * static settled 34-88% of traffic at the edge with end-to-end recall within 0.02 of the model alone.
 *
 * Caveat: the edge thresholds and the joint head threshold are chosen on the same calibration rows.
 */
import { decisionFunction, designRiskThreshold, fitLogistic, seededRandom } from '@liquidau/solvers';
import { canonicalDigest, DEFAULT_LEXICAL, lexicalFeatures, lexicalSpecProblems, staticIdentityProblems, type LexicalSpec, type StaticIdentity } from '@liquidau/text-preprocessing';

import type { Split } from './train.ts';

export const EDGE_SCHEMA = 'liquidau-edge/1';
export const EDGE_SCORERS = ['rules', 'lexical', 'static', 'vector'] as const;
export type EdgeScorer = (typeof EDGE_SCORERS)[number];

export interface EdgeLinear { coef: number[]; intercept: number }

export interface EdgeHeadModel {
  /** Blend inputs, in this order (a subset of EDGE_SCORERS). */
  scorers: EdgeScorer[];
  lexical?: { bias: number; index: number[]; weight: number[] };
  static?: EdgeLinear;
  vector?: EdgeLinear;
  blend: EdgeLinear;
  /** Fire when the blended score is strictly above this (null: never). */
  fire: number | null;
  /** Dismiss when the blended score is strictly below this (null: never). */
  dismiss: number | null;
  claims: { max_false_alarms: number; max_dismissed: number; delta: number };
}

export interface EdgeDocument<H extends string = string> {
  schema: typeof EDGE_SCHEMA;
  lexical: LexicalSpec | null;
  static: StaticIdentity | null;
  vector: { model_id: string; dimensions: number } | null;
  heads: Partial<Record<H, EdgeHeadModel>>;
}

export type EdgeDecision = 'fire' | 'dismiss' | null;

export interface EdgeInputs {
  text: string;
  /** Whether a certified rule for this head fired on the text (heads with the 'rules' scorer). */
  ruleFired?: (head: string) => boolean;
  static?: ArrayLike<number>;
  vector?: ArrayLike<number>;
}

export interface EdgeHeadInput<H extends string> {
  name: H;
  y: ReadonlyArray<0 | 1 | null>;
  /** Rows (and background rows) where a certified rule for this label fired, if 'rules' is a scorer. */
  ruleFired?: readonly boolean[];
  backgroundRuleFired?: readonly boolean[];
  /** At most this share of negatives fired (default 0.005). */
  maxFalseAlarms?: number;
  /**
   * At most this share of positives dismissed (0 disables dismissal; recall heads: e.g. 0.5 × (1 − target)).
   * A list makes a ladder: one candidate per share, e.g. [0.075, 0.0375, 0], for trainHeads to try in order
   * (edgeLadder), keeping the first with which the head passes its gates.
   */
  maxDismissed: number | readonly number[];
}

export interface TrainEdgeInput<H extends string> {
  texts: readonly string[];
  split: readonly Split[];
  heads: ReadonlyArray<EdgeHeadInput<H>>;
  scorers?: readonly EdgeScorer[];
  /** Cross-fitting folds follow groups (near-duplicates stay together); default: row index. */
  groups?: readonly string[];
  /** Calibration design for the calibration rows, in row order (designOf of their records); default one unbounded stratum. */
  design?: { inclusionProbs: readonly number[]; strata: readonly string[]; stratumSizes: Readonly<Record<string, number>> };
  static?: { identity: StaticIdentity; vectors: ReadonlyArray<ArrayLike<number>> };
  vector?: { modelId: string; vectors: ReadonlyArray<ArrayLike<number>> };
  /** Unlabelled traffic the head's firing budget is measured on (trainHeads background), to mark the edge's decisions there too. */
  background?: { texts: readonly string[]; static?: ReadonlyArray<ArrayLike<number>>; vector?: ReadonlyArray<ArrayLike<number>> };
  lexical?: LexicalSpec;
  delta?: number;
  seed?: number;
}

export interface EdgeRows {
  fired: boolean[];
  dismissed: boolean[];
  background?: { fired: boolean[]; dismissed: boolean[] };
}

export interface EdgeRung {
  maxDismissed: number;
  model: EdgeHeadModel;
  rows: EdgeRows;
  /** edgeHeadDigest of this candidate: what the head's artifact record names when it is chosen. */
  sha256: string;
}

export interface TrainEdgeResult<H extends string> {
  /** The document with each head's first candidate. */
  edge: EdgeDocument<H>;
  /** sha256 of that canonical document. */
  edgeSha256: string;
  /** Each head's first candidate's rows. */
  rows: Partial<Record<H, EdgeRows>>;
  /** Every candidate per head, in ladder order. */
  rungs: Partial<Record<H, EdgeRung[]>>;
  /**
   * Out-of-fold blend scores on each head's training rows (NaN elsewhere), aligned with the rows: the
   * edge's scores for trainHeads' cascade choice (HeadInput.cascade). Rows of a fold whose fit lacked a
   * class score the blend's intercept (components 0), as in the lab.
   */
  trainScores: Partial<Record<H, number[]>>;
  warnings: string[];
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const linear = (m: EdgeLinear, x: ArrayLike<number>) => decisionFunction(m, x);

/** Dense weights per lexical model, built once: scoring costs the message's features, not the model's. */
const dense = new WeakMap<object, Float32Array>();
function lexicalScore(m: NonNullable<EdgeHeadModel['lexical']>, f: { index: Uint32Array; value: Float32Array }, bits?: number): number {
  let w = dense.get(m);
  if (!w) {
    const size = 2 ** (bits ?? 18);
    w = new Float32Array(size);
    for (let k = 0; k < m.index.length; k++) if (m.index[k] < size) w[m.index[k]] = m.weight[k];
    dense.set(m, w);
  }
  let z = m.bias;
  for (let k = 0; k < f.index.length; k++) z += w[f.index[k]] * f.value[k];
  return z;
}

function fitLexical(features: Array<{ index: Uint32Array; value: Float32Array }>, y: (0 | 1)[], seed: number, size: number): NonNullable<EdgeHeadModel['lexical']> {
  const w = new Float64Array(size), g2 = new Float64Array(size);
  let b = 0, gb = 0;
  const pos = y.filter((v) => v === 1).length, neg = y.length - pos;
  const cw = [y.length / (2 * Math.max(1, neg)), y.length / (2 * Math.max(1, pos))];
  const rand = seededRandom(seed * 7 + 1);
  const order = features.map((_, i) => i);
  const lr = 0.5, l2 = 1e-6;
  for (let e = 0; e < 8; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (const i of order) {
      const f = features[i];
      let z = b;
      for (let k = 0; k < f.index.length; k++) z += w[f.index[k]] * f.value[k];
      const g = (sigmoid(z) - y[i]) * cw[y[i]];
      for (let k = 0; k < f.index.length; k++) {
        const j = f.index[k];
        const gk = g * f.value[k] + l2 * w[j];
        g2[j] += gk * gk;
        w[j] -= (lr * gk) / Math.sqrt(g2[j] + 1e-8);
      }
      gb += g * g;
      b -= (lr * g) / Math.sqrt(gb + 1e-8);
    }
  }
  const index: number[] = [], weight: number[] = [];
  for (let j = 0; j < size; j++) if (w[j] !== 0) { index.push(j); weight.push(Math.fround(w[j])); }
  return { bias: b, index, weight };
}

function foldOf(key: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) { h ^= key.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0) % 3;
}

/** The blended score and decision for every head of an edge document. */
export function scoreEdge<H extends string>(edge: EdgeDocument<H>, input: EdgeInputs): Partial<Record<H, { score: number; decision: EdgeDecision }>> {
  const out: Partial<Record<H, { score: number; decision: EdgeDecision }>> = {};
  let features: ReturnType<typeof lexicalFeatures> | null = null;
  for (const [head, m] of Object.entries(edge.heads) as Array<[H, EdgeHeadModel]>) {
    const x: number[] = [];
    for (const s of m.scorers) {
      if (s === 'rules') x.push(input.ruleFired?.(head) ? 1 : 0);
      else if (s === 'lexical') { features ??= lexicalFeatures(input.text, edge.lexical ?? DEFAULT_LEXICAL); x.push(lexicalScore(m.lexical!, features, (edge.lexical ?? DEFAULT_LEXICAL).bits)); }
      else if (s === 'static') { if (!input.static) throw new Error(`edge head ${head} needs a static embedding`); x.push(linear(m.static!, input.static)); }
      else { if (!input.vector) throw new Error(`edge head ${head} needs a vector`); x.push(linear(m.vector!, input.vector)); }
    }
    const score = linear(m.blend, x);
    out[head] = { score, decision: m.fire !== null && score > m.fire ? 'fire' : m.dismiss !== null && score < m.dismiss ? 'dismiss' : null };
  }
  return out;
}

/** Trains and certifies the edge for each head. */
export function trainEdge<H extends string>(input: TrainEdgeInput<H>): TrainEdgeResult<H> {
  const { texts, split, groups, design, delta = 0.05, seed = 0 } = input;
  const scorers = [...(input.scorers ?? ['rules', 'lexical', 'static'])] as EdgeScorer[];
  const lexSpec = input.lexical ?? DEFAULT_LEXICAL;
  const warnings: string[] = [];
  const n = texts.length;
  if (split.length !== n) throw new Error(`split has ${split.length} rows, texts ${n}`);
  if (groups && groups.length !== n) throw new Error(`groups has ${groups.length} rows, texts ${n}`);
  for (const s of scorers) if (!EDGE_SCORERS.includes(s)) throw new Error(`unknown edge scorer ${s}`);
  if (scorers.includes('static') && !input.static) throw new Error("scorer 'static' needs static vectors");
  if (scorers.includes('vector') && !input.vector) throw new Error("scorer 'vector' needs vectors");
  if (input.static) { const p = staticIdentityProblems(input.static.identity); if (p.length) throw new Error(p.join('; ')); if (input.static.vectors.length !== n) throw new Error('static.vectors must have one row per text'); }
  if (input.vector && input.vector.vectors.length !== n) throw new Error('vector.vectors must have one row per text');
  { const p = lexicalSpecProblems(lexSpec); if (p.length) throw new Error(p.join('; ')); }
  const calRows = split.map((s, i) => (s === 'calibration' ? i : -1)).filter((i) => i >= 0);
  if (design && design.inclusionProbs.length !== calRows.length) throw new Error(`design has ${design.inclusionProbs.length} rows for ${calRows.length} calibration rows`);
  const calDesign = design ?? { inclusionProbs: calRows.map(() => 1e-9), strata: calRows.map(() => 'all'), stratumSizes: { all: calRows.length * 1e9 } };

  const lexF = scorers.includes('lexical') ? texts.map((t) => lexicalFeatures(t, lexSpec)) : null;
  const bgLexF = scorers.includes('lexical') && input.background ? input.background.texts.map((t) => lexicalFeatures(t, lexSpec)) : null;
  const size = 2 ** lexSpec.bits;
  const doc: EdgeDocument<H> = {
    schema: EDGE_SCHEMA, lexical: scorers.includes('lexical') ? { ...lexSpec } : null,
    static: scorers.includes('static') ? { ...input.static!.identity } : null,
    vector: scorers.includes('vector') ? { model_id: input.vector!.modelId, dimensions: input.vector!.vectors[0]?.length ?? 0 } : null,
    heads: {},
  };
  const rowsOut: Partial<Record<H, EdgeRows>> = {};
  const rungsOut: Partial<Record<H, EdgeRung[]>> = {};
  const trainScoresOut: Partial<Record<H, number[]>> = {};

  for (const h of input.heads) {
    if (h.y.length !== n) throw new Error(`${h.name}: y has ${h.y.length} rows, texts ${n}`);
    const maxFA = h.maxFalseAlarms ?? 0.005;
    if (!(maxFA > 0 && maxFA < 1)) throw new Error(`${h.name}: maxFalseAlarms must be in (0, 1)`);
    const shares = typeof h.maxDismissed === 'number' ? [h.maxDismissed] : [...h.maxDismissed];
    if (!shares.length || !shares.every((v) => v >= 0 && v < 1)) throw new Error(`${h.name}: maxDismissed must be in [0, 1) (or a non-empty list of such)`);
    const headScorers = scorers.filter((s) => s !== 'rules' || h.ruleFired);
    if (scorers.includes('rules') && !h.ruleFired) warnings.push(`${h.name}: no rule column, so the edge blends without rules`);
    const trainIdx = split.map((s, i) => (s === 'train' && (h.y[i] === 0 || h.y[i] === 1) ? i : -1)).filter((i) => i >= 0);
    const yTrain = trainIdx.map((i) => h.y[i] as 0 | 1);
    if (!yTrain.includes(1) || !yTrain.includes(0)) { warnings.push(`${h.name}: the training split needs both classes; no edge`); continue; }
    const fitComponents = (rows: number[]) => {
      const yy = rows.map((i) => h.y[i] as 0 | 1);
      return {
        lexical: headScorers.includes('lexical') ? fitLexical(rows.map((i) => lexF![i]), yy, seed, size) : undefined,
        static: headScorers.includes('static') ? toLinear(fitLogistic(rows.map((i) => input.static!.vectors[i]), yy, { C: 1, classWeight: 'balanced' })) : undefined,
        vector: headScorers.includes('vector') ? toLinear(fitLogistic(rows.map((i) => input.vector!.vectors[i]), yy, { C: 1, classWeight: 'balanced' })) : undefined,
      };
    };
    const componentRow = (c: ReturnType<typeof fitComponents>, i: number, bg = false): number[] => headScorers.map((s) => {
      if (s === 'rules') return (bg ? h.backgroundRuleFired?.[i] : h.ruleFired![i]) ? 1 : 0;
      if (s === 'lexical') return lexicalScore(c.lexical!, bg ? bgLexF![i] : lexF![i], lexSpec.bits);
      if (s === 'static') return linear(c.static!, bg ? input.background!.static![i] : input.static!.vectors[i]);
      return linear(c.vector!, bg ? input.background!.vector![i] : input.vector!.vectors[i]);
    });
    // The blend learns on out-of-fold component scores, so it doesn't trust components on their own training rows.
    const oof = trainIdx.map(() => [] as number[]);
    const inSample = new Set<number>();
    for (let f = 0; f < 3; f++) {
      const inFold = trainIdx.map((i, k) => ((groups ? foldOf(groups[i]) : i % 3) === f ? k : -1)).filter((k) => k >= 0);
      const rest = trainIdx.filter((i) => (groups ? foldOf(groups[i]) : i % 3) !== f);
      const yr = rest.map((i) => h.y[i]);
      if (!yr.includes(1) || !yr.includes(0)) { warnings.push(`${h.name}: a cross-fitting fold lacks a class; the blend used in-sample scores there`); }
      const c = yr.includes(1) && yr.includes(0) ? fitComponents(rest) : fitComponents(trainIdx);
      if (!(yr.includes(1) && yr.includes(0))) for (const k of inFold) inSample.add(k);
      for (const k of inFold) oof[k] = componentRow(c, trainIdx[k]);
    }
    const blend = toLinear(fitLogistic(oof, yTrain, { C: 1, classWeight: 'balanced' }));
    const trainScores = new Array<number>(n).fill(NaN);
    trainIdx.forEach((i, k) => (trainScores[i] = linear(blend, inSample.has(k) ? headScorers.map(() => 0) : oof[k])));
    trainScoresOut[h.name] = trainScores;
    const comps = fitComponents(trainIdx);
    const score = (i: number) => linear(blend, componentRow(comps, i));
    // Certify on the calibration rows.
    const calY = calRows.map((i) => h.y[i]);
    const keep = calRows.map((_, k) => calY[k] === 0 || calY[k] === 1);
    const sub = <T>(v: readonly T[]) => v.filter((_, k) => keep[k]);
    const calScores = calRows.map(score);
    const base = { inclusionProbs: sub(calDesign.inclusionProbs), strata: sub(calDesign.strata), stratumSizes: calDesign.stratumSizes };
    const yc = sub(calY) as (0 | 1)[], sc = sub(calScores);
    let fire: number | null = null;
    const dismissFor = new Map<number, number | null>();
    if (yc.includes(0) && yc.includes(1)) {
      const f = designRiskThreshold({ ...base, y: yc.map((v) => (1 - v) as 0 | 1), scores: sc.map((v) => -v), alpha: maxFA, delta, method: 'linearised', minStratumPositives: 0 });
      if (f.feasible) fire = -f.threshold; else warnings.push(`${h.name}: no edge fire threshold keeps false alarms ≤ ${maxFA} (${f.reason})`);
      for (const share of shares) {
        if (share === 0 || dismissFor.has(share)) { dismissFor.set(share, dismissFor.get(share) ?? null); continue; }
        const d = designRiskThreshold({ ...base, y: yc, scores: sc, alpha: share, delta, method: 'linearised', minStratumPositives: 0 });
        dismissFor.set(share, d.feasible ? d.threshold : null);
        if (!d.feasible) warnings.push(`${h.name}: no edge dismiss threshold keeps dismissed positives ≤ ${share} (${d.reason})`);
      }
    } else warnings.push(`${h.name}: the calibration rows need both classes; the edge decides nothing`);
    const allScores = texts.map((_, i) => score(i));
    const bgScores = input.background ? input.background.texts.map((_, i) => linear(blend, componentRow(comps, i, true))) : undefined;
    const rungs: EdgeRung[] = shares.map((share) => {
      let dismiss = dismissFor.get(share) ?? null;
      if (fire !== null && dismiss !== null && dismiss > fire) dismiss = fire;
      const model: EdgeHeadModel = {
        scorers: headScorers, ...(comps.lexical ? { lexical: comps.lexical } : {}), ...(comps.static ? { static: comps.static } : {}), ...(comps.vector ? { vector: comps.vector } : {}),
        blend, fire, dismiss, claims: { max_false_alarms: maxFA, max_dismissed: share, delta },
      };
      const decide = (v: number): EdgeDecision => (fire !== null && v > fire ? 'fire' : dismiss !== null && v < dismiss ? 'dismiss' : null);
      const all = allScores.map(decide), bg = bgScores?.map(decide);
      const rows: EdgeRows = {
        fired: all.map((d) => d === 'fire'), dismissed: all.map((d) => d === 'dismiss'),
        ...(bg ? { background: { fired: bg.map((d) => d === 'fire'), dismissed: bg.map((d) => d === 'dismiss') } } : {}),
      };
      return { maxDismissed: share, model, rows, sha256: headDigest(doc, model) };
    });
    doc.heads[h.name] = rungs[0].model;
    rowsOut[h.name] = rungs[0].rows;
    rungsOut[h.name] = rungs;
  }
  return { edge: doc, edgeSha256: edgeDigest(doc), rows: rowsOut, rungs: rungsOut, trainScores: trainScoresOut, warnings };
}

function toLinear(m: { coef: ArrayLike<number>; intercept: number }): EdgeLinear {
  return { coef: Array.from(m.coef), intercept: m.intercept };
}

/** sha256 of the canonical edge document. */
export function edgeDigest(edge: EdgeDocument): string {
  return canonicalDigest(edge);
}

function headDigest(edge: Pick<EdgeDocument, 'lexical' | 'static' | 'vector'>, model: EdgeHeadModel): string {
  return canonicalDigest({ lexical: edge.lexical, static: edge.static, vector: edge.vector, model });
}

/**
 * sha256 of one head's edge model with the document's feature specs: what the head's artifact record
 * names (HeadSpec.edge.model_sha256), so each head may use a different candidate of a ladder.
 */
export function edgeHeadDigest(edge: EdgeDocument, head: string): string | null {
  const m = (edge.heads as Record<string, EdgeHeadModel | undefined>)[head];
  return m ? headDigest(edge, m) : null;
}

/** A head's candidates as trainHeads' HeadInput.edge ladder (tried in order). */
export function edgeLadder<H extends string>(result: TrainEdgeResult<H>, head: H): Array<{ fired: boolean[]; dismissed: boolean[]; background?: { fired: boolean[]; dismissed: boolean[] }; edgeSha256: string }> {
  return (result.rungs[head] ?? []).map((r) => ({ ...r.rows, edgeSha256: r.sha256 }));
}

/**
 * The edge document with each head's chosen candidate (trainHeads' result.edgeChoice; default the first).
 * Heads left out of `heads` (e.g. heads the edge shouldn't decide) are dropped.
 */
export function assembleEdge<H extends string>(result: TrainEdgeResult<H>, chosen: Partial<Record<H, number>>, heads: readonly H[] = Object.keys(result.rungs) as H[]): EdgeDocument<H> {
  const out: EdgeDocument<H> = { ...result.edge, heads: {} };
  for (const h of heads) {
    const rungs = result.rungs[h];
    if (!rungs?.length) continue;
    const k = chosen[h] ?? 0;
    if (!(k >= 0 && k < rungs.length)) throw new Error(`${h}: no edge candidate ${k}`);
    out.heads[h] = rungs[k].model;
  }
  return out;
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const linearOk = (m: unknown, dims?: number) => isObject(m) && Array.isArray(m.coef) && m.coef.every(finite) && finite(m.intercept) && (dims === undefined || m.coef.length === dims);

/** Validates a parsed edge document; throws with every problem found. */
export function validateEdge(raw: unknown): EdgeDocument {
  const p: string[] = [];
  if (!isObject(raw)) throw new Error('the edge document is not an object');
  if (raw.schema !== EDGE_SCHEMA) throw new Error(`unsupported edge schema ${JSON.stringify(raw.schema)} (this library reads '${EDGE_SCHEMA}')`);
  for (const k of Object.keys(raw)) if (!['schema', 'lexical', 'static', 'vector', 'heads'].includes(k)) p.push(`unknown field ${k}`);
  if (raw.lexical !== null) p.push(...lexicalSpecProblems(raw.lexical).map((e) => `lexical: ${e}`));
  if (raw.static !== null) p.push(...staticIdentityProblems(raw.static).map((e) => `static: ${e}`));
  if (raw.vector !== null && !(isObject(raw.vector) && typeof raw.vector.model_id === 'string' && Number.isInteger(raw.vector.dimensions))) p.push('vector must be null or { model_id, dimensions }');
  if (!isObject(raw.heads)) p.push('heads must be an object');
  else for (const [head, m] of Object.entries(raw.heads)) {
    if (!isObject(m)) { p.push(`head ${head} is not an object`); continue; }
    const sc = m.scorers;
    if (!Array.isArray(sc) || !sc.length || !sc.every((s) => (EDGE_SCORERS as readonly string[]).includes(s as string)) || new Set(sc).size !== sc.length) { p.push(`head ${head}: scorers must be distinct names from ${EDGE_SCORERS.join(', ')}`); continue; }
    if (sc.includes('lexical')) {
      const lx = m.lexical;
      if (raw.lexical === null) p.push(`head ${head} uses lexical but the document has no lexical spec`);
      if (!(isObject(lx) && finite(lx.bias) && Array.isArray(lx.index) && Array.isArray(lx.weight) && lx.index.length === lx.weight.length && lx.weight.every(finite) && lx.index.every((v, k, a) => Number.isInteger(v) && (v as number) >= 0 && (k === 0 || (v as number) > (a[k - 1] as number))))) p.push(`head ${head}: lexical must be { bias, index (ascending), weight }`);
    }
    if (sc.includes('static') && (raw.static === null || !linearOk(m.static, (raw.static as { dimensions?: number })?.dimensions))) p.push(`head ${head}: static must be a linear model over the document's static dimensions`);
    if (sc.includes('vector') && (raw.vector === null || !linearOk(m.vector, (raw.vector as { dimensions?: number })?.dimensions))) p.push(`head ${head}: vector must be a linear model over the document's vector dimensions`);
    if (!linearOk(m.blend, sc.length)) p.push(`head ${head}: blend must be a linear model with one weight per scorer`);
    for (const k of ['fire', 'dismiss']) if (m[k] !== null && !finite(m[k])) p.push(`head ${head}: ${k} must be a number or null`);
    if (finite(m.fire) && finite(m.dismiss) && m.dismiss > m.fire) p.push(`head ${head}: dismiss must not exceed fire`);
    const c = m.claims;
    if (!(isObject(c) && finite(c.max_false_alarms) && finite(c.max_dismissed) && finite(c.delta))) p.push(`head ${head}: claims must be { max_false_alarms, max_dismissed, delta }`);
  }
  if (p.length) throw new Error(`the edge document is invalid: ${p.join('; ')}`);
  return raw as unknown as EdgeDocument;
}
