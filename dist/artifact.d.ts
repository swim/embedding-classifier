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
    weights: number[];
    bias: number;
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
}
/** Records which embedding the heads were trained on - runtime must embed identically. */
export interface EmbeddingSpec {
    model_id: string;
    dimensions: number;
    normalize: boolean;
    /** Provider-specific input type, e.g. Cohere's "classification" - part of what the vectors mean. */
    input_type?: string;
}
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
    training?: Record<string, unknown>;
    evaluation?: Record<string, unknown>;
    gates?: GateResult;
}
export type Scores<H extends string = string> = Partial<Record<H, number>>;
export declare function calibrate(calibration: Calibration, logit: number): number;
export declare function headProbability(spec: HeadSpec, embedding: ArrayLike<number>): number;
/**
 * Validates an artifact loaded from storage; throws with a specific reason if it's unusable.
 * Checks values, not just shape: a corrupted or hand-edited artifact must fail loudly here rather
 * than score NaN at runtime (which decide() would otherwise read as "no decision").
 * Pass `heads` to reject head names the caller doesn't know how to act on.
 */
export declare function validateArtifact<H extends string = string>(raw: unknown, options?: {
    heads?: readonly H[];
}): ClassifierArtifact<H>;
/** Calibrated probability for every head in the artifact. */
export declare function scoreEmbedding<H extends string>(artifact: ClassifierArtifact<H>, embedding: ArrayLike<number>): Scores<H>;
