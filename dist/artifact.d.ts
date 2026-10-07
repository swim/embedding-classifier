import { type Guarantee } from './conformal.ts';
import { type HeadFeatures, type ReferenceSet } from './heads.ts';
export type Calibration = {
    method: 'platt';
    a: number;
    c: number;
} | {
    method: 'isotonic';
    x: number[];
    y: number[];
};
export interface HeadSpec {
    /** One per embedding dimension, or per feature for knn (1) and stack (3) heads. */
    weights: number[];
    bias: number;
    /** knn and stack heads: how the embedding becomes the head's features (absent: linear on the embedding). */
    features?: HeadFeatures;
    /**
     * Certified dismissal rules in front of this head (rule-miner): messages they clear count as misses
     * in the head's guarantee, and decide() takes them as dismissed. For audit.
     */
    dismissal?: {
        rule_set: string;
        max_rate: number;
        certified: number;
    };
    calibration: Calibration;
    /** Act at or above this calibrated probability. */
    threshold: number;
    /** Below the threshold but at or above this: worth a human look (sampling for labelling, suppression). */
    review_floor: number;
    /**
     * Named extra thresholds for tiered responses, e.g. { checkin: 0.012 } - a gentler action below
     * the main threshold. Interpreted by the application, not by decide().
     */
    thresholds?: Record<string, number>;
    /** Recall heads: how the threshold was chosen and what it guarantees from calibration, for audit. */
    guarantee?: Guarantee;
    /** When set, review_floor is the conformal floor: at most this share of positives score below it (in expectation). */
    review_epsilon?: number;
}
/** Records which embedding the heads were trained on - runtime must embed identically. */
export interface EmbeddingSpec {
    model_id: string;
    dimensions: number;
    normalize: boolean;
    /** Provider-specific input type, e.g. Cohere's "classification" - part of what the vectors mean. */
    input_type?: string;
    /**
     * Multi-layer features: the model's transformer layers (1-based, in concatenation order) whose token
     * states are pooled and concatenated, e.g. [4, 8, 12]. Absent: the model's standard sentence
     * embedding. `dimensions` is then the layers' total width.
     */
    layers?: number[];
    /** With `layers`: how each layer's token states are pooled ('mean' over the attention mask). */
    pooling?: 'mean';
    /**
     * With `layers`: whether each pooled layer is unit-normalised before concatenation. The concatenation
     * is then not unit length, whatever `normalize` says (@liquidau/router maps it to 'per-layer-unit').
     */
    layer_normalize?: boolean;
    /** The numeric precision of the model that produced the vectors, e.g. 'fp32' or 'q8': quantisation changes them. */
    precision?: string;
    /** Characters kept per text before tokenising (`truncateText`); use the same value at runtime. */
    max_chars?: number;
}
/**
 * How a runtime's embedder differs from the one the artifact's heads were trained on (empty when they
 * match). Vectors of the right width can still mean something else - another layer set, precision or
 * input type - and heads scored on them carry no guarantee, so refuse to serve while this is non-empty.
 */
export declare function checkEmbeddingSpec(artifact: Pick<ClassifierArtifact, 'embedding'>, runtime: EmbeddingSpec): string[];
/**
 * The router integration record, stored as `training.router` (buildArtifact's `router` option): the
 * semantic hash (rule-miner's ruleSetHash) of the complete rule set this artifact was trained and
 * evaluated with, firing and dismissal rules alike. @liquidau/router refuses a release whose rules
 * tier runs another rule set. An artifact without it predates the router and needs a legacy conversion.
 */
export interface RouterTraining {
    ruleSetHash: string;
}
/**
 * The artifact's `training.router.ruleSetHash`, or undefined when it has no router record. Throws if
 * the record is present but malformed: the hash must be lowercase hex SHA-256, nothing is guessed from
 * other metadata.
 */
export declare function routerRuleSetHash(artifact: Pick<ClassifierArtifact, 'training'>): string | undefined;
export interface GateResult {
    passed: boolean;
    failures: string[];
    warnings?: string[];
}
export interface ClassifierArtifact<H extends string = string> {
    version: string;
    created_at: string;
    embedding: EmbeddingSpec;
    heads: Partial<Record<H, HeadSpec>>;
    /** Training embeddings shared by knn and stack heads (data derived from training messages). */
    reference?: ReferenceSet;
    training?: Record<string, unknown>;
    evaluation?: Record<string, unknown>;
    gates?: GateResult;
}
export type Scores<H extends string = string> = Partial<Record<H, number>>;
export declare function calibrate(calibration: Calibration, logit: number): number;
/** A linear head's probability (knn and stack heads need the artifact's reference: use scoreEmbedding). */
export declare function headProbability(spec: HeadSpec, embedding: ArrayLike<number>): number;
/**
 * Validates an artifact loaded from storage; throws with a specific reason if it's unusable.
 * Checks values, not just shape: a corrupted or hand-edited artifact must fail loudly here rather
 * than score NaN at runtime (which decide() would otherwise read as "no decision").
 * Pass `heads` to reject head names the caller doesn't know how to act on, and `mode: 'enforce'` when
 * the decisions will act: the artifact's gates must then have passed (buildArtifact records them).
 */
export declare function validateArtifact<H extends string = string>(raw: unknown, options?: {
    heads?: readonly H[];
    mode?: 'shadow' | 'enforce';
}): ClassifierArtifact<H>;
/**
 * Builds the artifact's runtime scoring state now (decoded reference rows, stack heads' projected
 * rows) instead of on the first scoreEmbedding call, so a server pays it once at start-up. The state
 * is cached against the artifact's reference object: don't mutate a prepared artifact.
 */
export declare function prepareScoring<H extends string>(artifact: ClassifierArtifact<H>): void;
/** Calibrated probability for every head in the artifact. */
export declare function scoreEmbedding<H extends string>(artifact: ClassifierArtifact<H>, embedding: ArrayLike<number>): Scores<H>;
/**
 * Heads certified with a different dismissal rule set than the one the rules tier runs (`ruleSet`:
 * its identity, e.g. rule-miner's ruleSetHash). Each such head's guarantee counted another rule set's
 * misses, so it doesn't describe the system: refuse to serve while this is non-empty.
 */
export declare function checkRuleSetPairing<H extends string>(artifact: ClassifierArtifact<H>, ruleSet: string): string[];
