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
export declare const FINER_SCALE = 0.25;
export declare const FINER_C = 0.01;
/** P(finer label | embedding) under the stored model. */
export declare function finerProbabilities(model: Pick<FinerModel, 'coef' | 'intercept'>, embedding: ArrayLike<number>): number[];
/** A finer head's feature vector: the embedding and the scaled logit of its route score. */
export declare function finerFeatureVector(features: FinerFeatures, embedding: ArrayLike<number>, probabilities: readonly number[]): number[];
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
export declare function fitFinerModel(X: ReadonlyArray<ArrayLike<number>>, labels: ReadonlyArray<string | null>, rows: readonly number[], keys: (i: number) => string, C?: number): FittedFinerModel;
/**
 * A head's route rate per finer label, from its training rows that carry both labels: smoothed toward
 * their overall rate, so a finer label with few rows stays near it.
 */
export declare function finerRouteRates(K: number, rows: ReadonlyArray<{
    k: number;
    y: 0 | 1;
}>): number[];
/** Throws unless `f` is a usable finer model for `dims`-dimensional embeddings. */
export declare function validateFinerModel(f: FinerModel, dims: number): void;
