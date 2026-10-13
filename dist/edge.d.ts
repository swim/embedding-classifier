import { type LexicalSpec, type StaticIdentity } from '@liquidau/text-preprocessing';
import type { Split } from './train.ts';
export declare const EDGE_SCHEMA = "liquidau-edge/1";
export declare const EDGE_SCORERS: readonly ["rules", "lexical", "static", "vector"];
export type EdgeScorer = (typeof EDGE_SCORERS)[number];
export interface EdgeLinear {
    coef: number[];
    intercept: number;
}
export interface EdgeHeadModel {
    /** Blend inputs, in this order (a subset of EDGE_SCORERS). */
    scorers: EdgeScorer[];
    lexical?: {
        bias: number;
        index: number[];
        weight: number[];
    };
    static?: EdgeLinear;
    vector?: EdgeLinear;
    blend: EdgeLinear;
    /** Fire when the blended score is strictly above this (null: never). */
    fire: number | null;
    /** Dismiss when the blended score is strictly below this (null: never). */
    dismiss: number | null;
    claims: {
        max_false_alarms: number;
        max_dismissed: number;
        delta: number;
    };
}
export interface EdgeDocument<H extends string = string> {
    schema: typeof EDGE_SCHEMA;
    lexical: LexicalSpec | null;
    static: StaticIdentity | null;
    vector: {
        model_id: string;
        dimensions: number;
    } | null;
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
    design?: {
        inclusionProbs: readonly number[];
        strata: readonly string[];
        stratumSizes: Readonly<Record<string, number>>;
    };
    static?: {
        identity: StaticIdentity;
        vectors: ReadonlyArray<ArrayLike<number>>;
    };
    vector?: {
        modelId: string;
        vectors: ReadonlyArray<ArrayLike<number>>;
    };
    /** Unlabelled traffic the head's firing budget is measured on (trainHeads background), to mark the edge's decisions there too. */
    background?: {
        texts: readonly string[];
        static?: ReadonlyArray<ArrayLike<number>>;
        vector?: ReadonlyArray<ArrayLike<number>>;
    };
    lexical?: LexicalSpec;
    delta?: number;
    seed?: number;
}
export interface EdgeRows {
    fired: boolean[];
    dismissed: boolean[];
    background?: {
        fired: boolean[];
        dismissed: boolean[];
    };
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
/** The blended score and decision for every head of an edge document. */
export declare function scoreEdge<H extends string>(edge: EdgeDocument<H>, input: EdgeInputs): Partial<Record<H, {
    score: number;
    decision: EdgeDecision;
}>>;
/** Trains and certifies the edge for each head. */
export declare function trainEdge<H extends string>(input: TrainEdgeInput<H>): TrainEdgeResult<H>;
/** sha256 of the canonical edge document. */
export declare function edgeDigest(edge: EdgeDocument): string;
/**
 * sha256 of one head's edge model with the document's feature specs: what the head's artifact record
 * names (HeadSpec.edge.model_sha256), so each head may use a different candidate of a ladder.
 */
export declare function edgeHeadDigest(edge: EdgeDocument, head: string): string | null;
/** A head's candidates as trainHeads' HeadInput.edge ladder (tried in order). */
export declare function edgeLadder<H extends string>(result: TrainEdgeResult<H>, head: H): Array<{
    fired: boolean[];
    dismissed: boolean[];
    background?: {
        fired: boolean[];
        dismissed: boolean[];
    };
    edgeSha256: string;
}>;
/**
 * The edge document with each head's chosen candidate (trainHeads' result.edgeChoice; default the first).
 * Heads left out of `heads` (e.g. heads the edge shouldn't decide) are dropped.
 */
export declare function assembleEdge<H extends string>(result: TrainEdgeResult<H>, chosen: Partial<Record<H, number>>, heads?: readonly H[]): EdgeDocument<H>;
/** Validates a parsed edge document; throws with every problem found. */
export declare function validateEdge(raw: unknown): EdgeDocument;
