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
    search?: (query: number[], n: number) => Promise<Array<{
        id: string;
        score: number;
    }>>;
}
/** Maximal marginal relevance: indices into `candidates`, in selection order. */
export declare function mmrSelect(candidates: ReadonlyArray<{
    v: Float64Array;
    sim: number;
}>, k: number, lambda: number): number[];
export declare function retrieveFromSeeds(options: RetrieveOptions): Promise<ExampleRecord[]>;
/**
 * Per seed: how many of its retrieved neighbours were labelled for the head and how many were
 * confirmed positive. A seed is retired when its hit rate is below minHitRate after at least
 * minLabelled labels (defaults 0.1 and 10).
 */
export declare function seedStats(records: readonly ExampleRecord[], head: string, options?: {
    minHitRate?: number;
    minLabelled?: number;
}): {
    seedId: string;
    labelled: number;
    confirmed: number;
    hitRate: number;
    retire: boolean;
}[];
