/**
 * Where each example came from, and what it may be used for. The principle: generated or
 * non-randomly selected data may IMPROVE a model; only probability-sampled, human-labelled real
 * traffic may MEASURE it. validateProvenance enforces that in code:
 *
 *   P1  calibration and test records are `sampled`, human-labelled, with 0 < inclusionProb <= 1
 *   P2  background records are `traffic`, unlabelled, with a backgroundUse
 *   P3  a group appears in one role only, and background groups in one backgroundUse only
 *   P4  retrieved and generated records are `train` or `stress` only
 *   P5  with embeddings: retrieved or generated train records within cosine 0.95 of a calibration
 *       or test record are dropped (warning)
 *   P7  generated records need verified === true or an accepted batch; for safety-critical heads,
 *       verified === true only
 *
 * P1-P4 and P7 throw a ProvenanceError naming the rule and the offending ids, unless the caller
 * overrides that rule - recorded, and the release gate then fails. P6 (weight caps) is
 * capTrainingWeights.
 */
export type Role = 'train' | 'background' | 'calibration' | 'test' | 'stress';
export type BackgroundUse = 'veto' | 'certify' | 'budget';
export type Source = 
/** A probability sample of real traffic (designSample). stratumSize is N_h, so inclusion probabilities can be recomputed after non-response. */
{
    kind: 'sampled';
    designId: string;
    stratum: string;
    inclusionProb: number;
    stratumSize: number;
}
/** Real traffic, not probability-sampled. */
 | {
    kind: 'traffic';
}
/** Real traffic chosen by nearness to seeds (retrieveFromSeeds). */
 | {
    kind: 'retrieved';
    seedIds: string[];
    similarity: number;
    round: 1 | 2;
}
/** Written by a generator (hard negatives). */
 | {
    kind: 'generated';
    method: 'hard_negative';
    generator: string;
    ruleId: string;
    batchId: string;
};
export interface ExampleRecord {
    id: string;
    text: string;
    /** Paraphrase group: one item per group in any sampling frame. */
    group: string;
    role: Role;
    /** Required when role === 'background'. */
    backgroundUse?: BackgroundUse;
    source: Source;
    /** Per head; null (or absent) = not labelled for that head. */
    labels: Record<string, 0 | 1 | null>;
    labelledBy?: 'human' | 'llm' | 'rule' | 'outcome' | 'intended';
    /** A human confirmed an 'intended' label. */
    verified?: boolean;
    /** Training weight (default 1). */
    weight?: number;
}
export type ProvenanceCode = 'P1' | 'P2' | 'P3' | 'P4' | 'P5' | 'P6' | 'P7';
export declare class ProvenanceError extends Error {
    readonly code: ProvenanceCode;
    readonly ids: string[];
    constructor(code: ProvenanceCode, ids: string[], message: string);
}
export interface ProvenanceOptions {
    /** Embeddings aligned with `records`, for P5. */
    embeddings?: ReadonlyArray<ArrayLike<number> | null | undefined>;
    /** Cosine at or above which a retrieved or generated train record counts as a near-duplicate (default 0.95). */
    nearDuplicate?: number;
    /** Heads whose generated records must each be human-verified (P7). */
    safetyCritical?: readonly string[];
    /** batchIds accepted by verifyBatch (P7). */
    acceptedBatches?: Iterable<string>;
    /** Rules to report instead of enforce. Recorded; the release gate fails. P5 and P6 adjust data and can't be overridden. */
    overrides?: ReadonlyArray<'P1' | 'P2' | 'P3' | 'P4' | 'P7'>;
}
export interface ProvenanceResult {
    /** The records that remain usable (P5 near-duplicates removed). */
    records: ExampleRecord[];
    dropped: Array<{
        id: string;
        code: 'P5';
        reason: string;
    }>;
    warnings: string[];
    /** Violations reported instead of thrown, because the caller overrode the rule. */
    overridden: Array<{
        code: ProvenanceCode;
        ids: string[];
        message: string;
    }>;
}
export declare const isReal: (r: Pick<ExampleRecord, "source">) => boolean;
/** Checks P1-P5 and P7 over every record a training run uses. Throws ProvenanceError unless overridden. */
export declare function validateProvenance(records: readonly ExampleRecord[], options?: ProvenanceOptions): ProvenanceResult;
export interface WeightCaps {
    /** Highest share of the head's positive training weight from retrieved positives (default 0.5). */
    maxRetrievedPositiveShare?: number;
    /** Highest share of the head's negative training weight from generated hard negatives (default 0.3). */
    maxGeneratedNegativeShare?: number;
    /**
     * Rule ids that fire on a text. Required when generated hard negatives are present: those
     * attributed to a rule may not outweigh the head's training positives that rule matches.
     */
    ruleMatches?: (text: string) => readonly string[];
}
export interface WeightCapSummary {
    retrieved_positive: {
        weight_before: number;
        weight: number;
        scale: number;
    };
    generated_negative: {
        weight_before: number;
        weight: number;
        scale: number;
    };
    /** Generated weight per rule after the per-rule balance cap. */
    by_rule?: Record<string, {
        weight_before: number;
        weight: number;
        matched_positive_weight: number;
    }>;
}
/**
 * P6: training weights for one head's train records (record.weight, default 1), with retrieved
 * positives and generated hard negatives scaled down proportionally to their caps. Returned
 * weights align with `records`.
 */
export declare function capTrainingWeights(records: readonly ExampleRecord[], head: string, caps?: WeightCaps): {
    weights: number[];
    summary: WeightCapSummary;
};
