import type { HeadPolicy } from './threshold.ts';
export type HeadType = 'linear' | 'knn' | 'stack';
export declare const HEAD_TYPES: readonly HeadType[];
export declare const KNN_K = 10;
export type HeadFeatures = {
    kind: 'knn';
    k: number;
} | {
    kind: 'stack';
    k: number;
    linear: {
        weights: number[];
        bias: number;
    };
    pca: {
        mean: number[];
        components: number[][];
    };
};
/** The artifact's shared reference rows. */
export interface ReferenceSet {
    /**
     * 'f32': row-major little-endian float32. 'int8': one signed byte per value with a per-row scale
     * (value = byte × scale, then rounded to float32): about 4× smaller, with the same thresholds and
     * false alarms in simulation (support-example sim-reference). Training scores with the decoded
     * reference either way, so runtime reproduces evaluation exactly.
     */
    encoding: 'f32' | 'int8';
    rows: number;
    dims: number;
    /** The values, base64. */
    data: string;
    /** int8 only: one scale per row. */
    scales?: number[];
    /** One label per row for each knn or stack head (null: the row isn't labelled for that head). */
    labels: Record<string, Array<0 | 1 | null>>;
}
export declare function encodeReference(rows: ReadonlyArray<ArrayLike<number>>, labels: Record<string, Array<0 | 1 | null>>, encoding?: 'f32' | 'int8'): ReferenceSet;
/** The reference's rows as float32 (views into one buffer). */
export declare function decodeReference(ref: ReferenceSet): Float32Array[];
export declare function dot(a: ArrayLike<number>, b: ArrayLike<number>): number;
/**
 * Cosines (dot products) of x with every reference row: one monomorphic loop, summed in index order
 * exactly as dot() sums, so training and runtime agree to the last bit.
 */
export declare function similarities(x: ArrayLike<number>, rows: readonly Float32Array[]): Float64Array;
/** The knn feature from similarities to every reference row. */
export declare function knnFeature(sims: ArrayLike<number>, pos: readonly number[], neg: readonly number[], k: number): number;
export interface Pca {
    mean: ArrayLike<number>;
    components: ReadonlyArray<ArrayLike<number>>;
}
/** Projection onto the components, unit-normalised (for cosine kNN). */
export declare function project(p: Pca, x: ArrayLike<number>): number[];
/** Top-k principal components by seeded randomized subspace iteration. */
export declare function fitPca(rows: ReadonlyArray<ArrayLike<number>>, k: number, seed: number, iterations?: number): {
    mean: Float64Array;
    components: Float64Array[];
};
interface RuntimeReference {
    rows: Float32Array[];
    pos: Map<string, number[]>;
    neg: Map<string, number[]>;
    projected: Map<string, number[][]>;
}
export declare function runtimeReference(ref: ReferenceSet): RuntimeReference;
/** Features for one head at runtime; `sims` are the embedding's cosines to every reference row (computed once per message). */
export declare function headFeatureVector(head: string, features: HeadFeatures, embedding: ArrayLike<number>, ref: RuntimeReference, sims: ArrayLike<number>): number[];
/** Fold of a row (FNV-style hash of its key), as the research's cross-fitting. */
export declare function foldOf(key: string): number;
export interface FeatureTraining {
    /** Training rows (indices into X), their labels and weights. */
    rows: number[];
    y: Array<0 | 1>;
    weights: number[];
    /** Fold key per training row. */
    keys: string[];
    X: ReadonlyArray<ArrayLike<number>>;
    /** The decoded reference rows and each one's fold key (the same keys as the training rows'). */
    reference: Float32Array[];
    referenceKeys: string[];
    /** This head's label per reference row. */
    referenceLabels: Array<0 | 1 | null>;
    C: number;
    seed: number;
}
export interface FittedFeatures {
    features?: HeadFeatures;
    /** Out-of-fold features for each training row. */
    train: number[][];
    /** Full-reference features for any other embedding. */
    apply: (x: ArrayLike<number>) => number[];
}
/**
 * Fits a head type's features. Out-of-fold for the training rows: the reference rows and models of
 * the row's own fold are left out.
 */
export declare function fitFeatures(type: HeadType, t: FeatureTraining): FittedFeatures;
/**
 * A head's guarantee cost on scores, with an oracle threshold on the given rows (lower is better):
 * false alarms at the target recall, or recall lost at the target precision.
 */
export declare function guaranteeCost(scores: readonly number[], y: ReadonlyArray<0 | 1>, policy: HeadPolicy): number;
/** Out-of-fold scores of a fitted feature set's head layer on the training rows (for choosing a type). */
export declare function crossValidatedScores(fitted: FittedFeatures, t: FeatureTraining): number[];
export {};
