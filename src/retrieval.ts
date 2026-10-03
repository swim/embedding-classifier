/**
 * Retrieval from seeds: real positives (and real near-miss negatives) found by searching unlabelled
 * traffic near human-verified positives. Its output is TRAINING data only (P4): nearness to seeds
 * is not a probability sample, so it can never measure a model.
 *
 *   1. validate seeds: role train, human-labelled positive for the head; round-2 seeds must be
 *      confirmed round-1 retrievals
 *   2. normalise embeddings (cosine = dot product)
 *   3. per seed, the top `candidates` pool items at or above `floor` (brute force, or `search`)
 *   4. diversify to `k` per seed by maximal marginal relevance:
 *      argmax λ·sim(c, seed) - (1 - λ)·max sim(c, selected)
 *   5. merge across seeds: every seed id kept, the highest similarity kept
 *   6. exclude items already labelled for the head, items whose group is in another role, and items
 *      within cosine `nearDuplicate` of any calibration or test embedding
 *   7. return unlabelled records (source 'retrieved', role 'train') for labelQueue
 *
 * Embeddings must come from the same model and truncation as runtime; dimensions are checked.
 * Brute force suits pools up to a few hundred thousand items; pass `search` beyond that.
 */
import type { ExampleRecord } from './records.ts';

export interface Embedded {
  record: ExampleRecord;
  embedding: ArrayLike<number>;
}

export interface RetrieveOptions {
  head: string;
  seeds: readonly Embedded[];
  /** Unlabelled traffic not assigned to another role. */
  pool: readonly Embedded[];
  /** Embeddings of every calibration and test record. */
  evaluation: ReadonlyArray<ArrayLike<number>>;
  /** Per seed, before diversification (default 100). */
  candidates?: number;
  /** Per seed, after diversification (default 20). */
  k?: number;
  /** Minimum cosine (default 0.6). */
  floor?: number;
  /** Relevance weight in MMR (default 0.7). */
  mmrLambda?: number;
  round: 1 | 2;
  /** Groups already used by calibration, test, background or stress records. */
  reservedGroups?: Iterable<string>;
  /** Cosine at which a pool item counts as a near-duplicate of an evaluation record (default 0.95). */
  nearDuplicate?: number;
  /** Seeds to skip (e.g. seedStats().filter(s => s.retire)). */
  retiredSeeds?: Iterable<string>;
  /** Nearest-neighbour search over the pool by item id, for pools too large for brute force. */
  search?: (query: number[], n: number) => Promise<Array<{ id: string; score: number }>>;
}

function unit(v: ArrayLike<number>): Float64Array {
  let sq = 0;
  for (let j = 0; j < v.length; j++) sq += v[j] * v[j];
  const norm = Math.sqrt(sq);
  if (!(norm > 0) || !Number.isFinite(norm)) throw new Error('embeddings must be finite and non-zero');
  return Float64Array.from(v, (x) => x / norm);
}
const dot = (a: Float64Array, b: Float64Array) => {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += a[j] * b[j];
  return s;
};

/** Maximal marginal relevance: indices into `candidates`, in selection order. */
export function mmrSelect(candidates: ReadonlyArray<{ v: Float64Array; sim: number }>, k: number, lambda: number): number[] {
  const chosen: number[] = [];
  const maxSim = candidates.map(() => -Infinity);
  while (chosen.length < Math.min(k, candidates.length)) {
    let best = -1, bestScore = -Infinity;
    candidates.forEach((c, i) => {
      if (chosen.includes(i)) return;
      const score = lambda * c.sim - (1 - lambda) * (chosen.length ? maxSim[i] : 0);
      if (score > bestScore) { bestScore = score; best = i; }
    });
    chosen.push(best);
    candidates.forEach((c, i) => { maxSim[i] = Math.max(maxSim[i], dot(c.v, candidates[best].v)); });
  }
  return chosen;
}

export async function retrieveFromSeeds(options: RetrieveOptions): Promise<ExampleRecord[]> {
  const { head, seeds, pool, evaluation, candidates = 100, k = 20, floor = 0.6, mmrLambda = 0.7, round, nearDuplicate = 0.95, search } = options;
  if (!(mmrLambda >= 0 && mmrLambda <= 1)) throw new Error(`mmrLambda must be in [0, 1], got ${mmrLambda}`);
  if (!(Number.isInteger(k) && k > 0 && Number.isInteger(candidates) && candidates >= k)) throw new Error('need integers candidates >= k > 0');
  const retired = new Set(options.retiredSeeds ?? []);
  const reserved = new Set(options.reservedGroups ?? []);

  // 1. Seeds.
  for (const { record: s } of seeds) {
    if (s.role !== 'train' || s.labels[head] !== 1 || s.labelledBy !== 'human') throw new Error(`seed ${s.id} must be a human-labelled ${head} positive with role train`);
    if (s.source.kind === 'generated') throw new Error(`seed ${s.id} is generated; seeds must be real`);
    if (round === 2 && !(s.source.kind === 'retrieved' && s.source.round === 1)) throw new Error(`round-2 seed ${s.id} must be a confirmed round-1 retrieval`);
  }
  const active = seeds.filter((s) => !retired.has(s.record.id));

  // 2. Normalise, with one dimension throughout.
  const dims = active[0]?.embedding.length ?? pool[0]?.embedding.length;
  const check = (what: string, v: ArrayLike<number>) => { if (v.length !== dims) throw new Error(`${what} has ${v.length} dimensions, expected ${dims}`); return unit(v); };
  const seedV = active.map((s) => check(`seed ${s.record.id}`, s.embedding));
  const poolV = pool.map((p) => check(`pool item ${p.record.id}`, p.embedding));
  const evalV = evaluation.map((e, i) => check(`evaluation[${i}]`, e));
  const poolIndex = new Map(pool.map((p, i) => [p.record.id, i]));

  // 6 (first, cheaply): which pool items may be returned at all.
  const eligible = pool.map((p, i) => {
    const r = p.record;
    if (r.labels[head] !== undefined && r.labels[head] !== null) return false;
    if (r.role !== 'train' || reserved.has(r.group)) return false;
    return !evalV.some((e) => dot(e, poolV[i]) >= nearDuplicate);
  });

  // 3-5. Search, diversify, merge.
  const merged = new Map<number, { seedIds: string[]; similarity: number }>();
  for (let s = 0; s < active.length; s++) {
    let hits: Array<{ i: number; sim: number }>;
    if (search) {
      const found = await search(Array.from(seedV[s]), candidates);
      hits = found.flatMap((f) => { const i = poolIndex.get(f.id); return i === undefined ? [] : [{ i, sim: dot(seedV[s], poolV[i]) }]; });
    } else {
      hits = poolV.map((v, i) => ({ i, sim: dot(seedV[s], v) }));
    }
    hits = hits.filter((h) => h.sim >= floor && eligible[h.i]).sort((a, b) => b.sim - a.sim || a.i - b.i).slice(0, candidates);
    const picked = mmrSelect(hits.map((h) => ({ v: poolV[h.i], sim: h.sim })), k, mmrLambda).map((j) => hits[j]);
    for (const h of picked) {
      const prior = merged.get(h.i);
      if (prior) { if (!prior.seedIds.includes(active[s].record.id)) prior.seedIds.push(active[s].record.id); prior.similarity = Math.max(prior.similarity, h.sim); }
      else merged.set(h.i, { seedIds: [active[s].record.id], similarity: h.sim });
    }
  }

  // 7. Unlabelled training candidates.
  return [...merged.entries()].sort(([a], [b]) => a - b).map(([i, m]) => ({
    ...pool[i].record,
    role: 'train' as const,
    labels: { ...pool[i].record.labels },
    source: { kind: 'retrieved' as const, seedIds: m.seedIds, similarity: m.similarity, round },
  }));
}

/**
 * Per seed: how many of its retrieved neighbours were labelled for the head and how many were
 * confirmed positive. A seed is retired when its hit rate is below minHitRate after at least
 * minLabelled labels (defaults 0.1 and 10).
 */
export function seedStats(records: readonly ExampleRecord[], head: string, options: { minHitRate?: number; minLabelled?: number } = {}) {
  const { minHitRate = 0.1, minLabelled = 10 } = options;
  const stats = new Map<string, { labelled: number; confirmed: number }>();
  for (const r of records) {
    if (r.source.kind !== 'retrieved') continue;
    const label = r.labels[head];
    if (label !== 0 && label !== 1) continue;
    for (const seedId of r.source.seedIds) {
      const s = stats.get(seedId) ?? stats.set(seedId, { labelled: 0, confirmed: 0 }).get(seedId)!;
      s.labelled++;
      s.confirmed += label;
    }
  }
  return [...stats.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([seedId, s]) => {
    const hitRate = s.labelled ? s.confirmed / s.labelled : 0;
    return { seedId, labelled: s.labelled, confirmed: s.confirmed, hitRate, retire: s.labelled >= minLabelled && hitRate < minHitRate };
  });
}
