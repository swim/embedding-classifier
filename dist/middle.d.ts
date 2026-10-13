import { type EmbeddingSpec } from './artifact.ts';
import type { EdgeDecision, EdgeLinear, EdgeRows, EdgeRung, TrainEdgeResult } from './edge.ts';
import type { CascadeInput, Split } from './train.ts';
export declare const MIDDLE_SCHEMA = "liquidau-middle/1";
export interface MiddleHeadModel {
    /** The middle head's linear model on the middle encoder's embedding (its raw logit is the score). */
    model: EdgeLinear;
    /** Fire when the score is strictly above this (null: never). */
    fire: number | null;
    /** Dismiss when the score is strictly below this (null: never). */
    dismiss: number | null;
    /** Cumulative claims (the edge's decisions included), and the edge model they were certified behind. */
    claims: {
        max_false_alarms: number;
        max_dismissed: number;
        delta: number;
        prior_edge_sha256: string;
    };
}
export interface MiddleDocument<H extends string = string> {
    schema: typeof MIDDLE_SCHEMA;
    /** The middle encoder's identity: runtime must embed exactly this way (precision required). */
    embedding: EmbeddingSpec;
    heads: Partial<Record<H, MiddleHeadModel>>;
}
/** Score and decision per head of a middle document, for one message's middle-encoder vector. */
export declare function scoreMiddle<H extends string>(doc: MiddleDocument<H>, vector: ArrayLike<number>): Partial<Record<H, {
    score: number;
    decision: EdgeDecision;
}>>;
/** sha256 of the canonical middle document. */
export declare function middleDigest(doc: MiddleDocument): string;
/** sha256 of one head's middle model (thresholds and claims included) with the document's encoder identity. */
export declare function middleHeadDigest(doc: Pick<MiddleDocument, 'embedding' | 'heads'>, head: string): string | null;
/** Validates a parsed middle document; throws with every problem found. */
export declare function validateMiddle(raw: unknown): MiddleDocument;
/** The pre-registered rule: keep the cascade when its training cost ≤ 1.1 × the two-tier cost + 0.001. */
export declare const CASCADE_TOLERANCE: {
    readonly relative: 1.1;
    readonly absolute: 0.001;
};
/** The middle's default extra dismissal share: 0.25 × the head's miss budget (1 − target). */
export declare const MIDDLE_EXTRA_DISMISSAL = 0.25;
/** The middle's default cumulative firing budget: 1% of negatives. */
export declare const MIDDLE_MAX_FALSE_ALARMS = 0.01;
/**
 * Cross-fitted logistic decision scores (5 folds by FNV of the key, C = 1, balanced, unweighted): the
 * choice's scorer for the middle and main heads, as the lab validated the rule. It ignores the head's
 * own C, standardisation and record weights, so the choice describes a close relative of the shipped
 * head, not that head exactly. A fold whose fit rows lack a class scores 0.
 */
export declare function crossFitScores(X: ReadonlyArray<ArrayLike<number>>, y: ReadonlyArray<0 | 1>, keys: readonly string[]): number[];
/** False-alarm rate at recall ≥ target, never splitting tied scores (1 when the target is unreachable). */
export declare function cascadeCost(scores: readonly number[], y: ReadonlyArray<0 | 1>, target: number): number;
export interface CascadeBudgets {
    /** The edge's fire budget (share of negatives) and dismissal share (of positives). */
    edgeFire: number;
    edgeDismiss: number;
    /** The middle's CUMULATIVE budgets (the edge's decisions included). */
    middleFire: number;
    middleDismiss: number;
}
export interface CascadeChoiceInput {
    /** Labels of the training rows, and each tier's training-row scores (edge: out-of-fold blend; middle and main: cross-fitted). */
    y: ReadonlyArray<0 | 1>;
    edge: readonly number[];
    middle: readonly number[];
    head: readonly number[];
    budgets: CascadeBudgets;
    targetRecall: number;
    reserve?: number;
}
export interface CascadeChoice {
    cascade: boolean;
    /** Training false alarms at min(0.99, target + reserve): edge → main, and edge → middle → main. */
    costs: {
        twoTier: number;
        cascade: number;
    };
}
/** The per-head choice, from training rows only (lab stage DY-C's rule, reproduced exactly). */
export declare function chooseCascade(input: CascadeChoiceInput): CascadeChoice;
export interface MiddleHeadInput<H extends string> {
    name: H;
    y: ReadonlyArray<0 | 1 | null>;
    /** The head's middle model: a linear head trained on the middle vectors (HeadSpec weights/bias of a trainHeads linear head). */
    model: {
        coef: ArrayLike<number>;
        intercept: number;
    };
    /** The recall target (the default dismissal budget is a share of 1 − target). */
    targetRecall: number;
    /** The edge candidate the middle sits behind (trainEdge's rungs[head][0]). */
    edge: EdgeRung;
    /** Cumulative firing budget (default 1% of negatives); equal to the edge's, the middle only dismisses. */
    maxFalseAlarms?: number;
    /** Cumulative dismissal budget (default the edge's share + 0.25 × (1 − target)). */
    maxDismissed?: number;
}
export interface TrainMiddleInput<H extends string> {
    embedding: EmbeddingSpec;
    /** The middle encoder's vector per row. */
    vectors: ReadonlyArray<ArrayLike<number>>;
    split: readonly Split[];
    /** Cross-fitting keys for the choice's training-row scores (default: row index). */
    groups?: readonly string[];
    /** Calibration design for the calibration rows, in row order (as trainEdge). */
    design?: {
        inclusionProbs: readonly number[];
        strata: readonly string[];
        stratumSizes: Readonly<Record<string, number>>;
    };
    /** Unlabelled traffic (trainHeads background): the middle's decisions there too, behind the edge's. */
    background?: {
        vectors: ReadonlyArray<ArrayLike<number>>;
    };
    heads: ReadonlyArray<MiddleHeadInput<H>>;
    delta?: number;
}
export interface MiddleHeadResult {
    model: MiddleHeadModel;
    /** The edge's and the middle's decisions together: pass as HeadInput.cascade.rows. */
    rows: EdgeRows;
    /** middleHeadDigest of this head's model: what the head's artifact record names when the cascade is kept. */
    sha256: string;
    /** Cross-fitted middle scores on the training rows (NaN elsewhere), aligned with the rows. */
    trainScores: number[];
    budgets: CascadeBudgets;
}
export interface TrainMiddleResult<H extends string> {
    /** The document with every trained head (assembleMiddle keeps the heads that chose the cascade). */
    middle: MiddleDocument<H>;
    heads: Partial<Record<H, MiddleHeadResult>>;
    warnings: string[];
}
/** Certifies each head's middle thresholds behind its edge candidate. Recall heads only. */
export declare function trainMiddle<H extends string>(input: TrainMiddleInput<H>): TrainMiddleResult<H>;
/** The middle document with only the heads that kept the cascade (trainHeads' result.cascadeChoice). */
export declare function assembleMiddle<H extends string>(result: TrainMiddleResult<H>, cascadeChoice: Partial<Record<H, {
    cascade: boolean;
}>>): MiddleDocument<H> | null;
/** HeadInput.cascade for one head, from trainEdge's and trainMiddle's results (null when the middle has no model for it). */
export declare function cascadeInput<H extends string>(edge: TrainEdgeResult<H>, middle: TrainMiddleResult<H>, head: H, choice?: 'cascade' | 'edge'): CascadeInput | null;
