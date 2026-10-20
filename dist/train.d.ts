import { type DesignSummary } from './design.ts';
import { type ExampleRecord, type ProvenanceCode, type ProvenanceOptions, type WeightCaps, type WeightCapSummary } from './records.ts';
import { type ClassifierArtifact, type EmbeddingSpec, type HeadSpec, type RouterTraining } from './artifact.ts';
import { type HeadEvaluation } from './evaluate.ts';
import { type CascadeBudgets } from './middle.ts';
import { type FinerModel } from './finer.ts';
import { type HeadType, type ReferenceSet } from './heads.ts';
import { type HeadPolicy } from './threshold.ts';
export declare const SPLITS: readonly ["train", "calibration", "test"];
export type Split = (typeof SPLITS)[number];
export interface HeadInput<H extends string> {
    name: H;
    /** Binary target per example, aligned with X; null leaves the example out of this head. */
    y: ReadonlyArray<0 | 1 | null>;
    policy: HeadPolicy;
    /**
     * Expected positive rate in production - calibration and precision are weighted to it. Required
     * without `records`; must be omitted with them (design weights already reproduce prevalence).
     */
    prevalence?: number;
    /** Generated records labelled for this head must each be human-verified (P7). Recorded in the provenance summary. */
    safetyCritical?: boolean;
    /** Whether an existing mechanism already catches each example (see evaluateHead). */
    baseline?: readonly boolean[];
    /**
     * The head's type (default 'auto'): 'auto' picks linear, knn or stack by cross-validated guarantee
     * cost on the training rows (recorded in result.headChoice; see autoMargin). 'knn' and 'stack' score
     * against the artifact's shared reference of training embeddings (heads.ts; buildArtifact stores it),
     * so such an artifact holds training data. Calibration, thresholds and guarantees are the same for
     * every type. Use 'linear' when X isn't an embedding (e.g. stacked scores) or the artifact must not
     * hold training embeddings. Weak positives are linear-only: with them, 'auto' stays linear.
     */
    type?: HeadType | 'auto';
    /**
     * Certified dismissal rules in front of this head (rule-miner's mineDismissals and
     * certifyDismissals): `rows` marks the rows of X a rule cleared (and `background` the rows of
     * background.X). Cleared calibration, test and background rows score 0, so a positive a rule
     * dismissed counts as a miss: the head's recall guarantee covers rules and classifier together.
     * Training is unchanged; calibration is fitted on the rows the rules leave to the classifier.
     */
    dismissal?: {
        rows: readonly boolean[];
        background?: readonly boolean[];
        ruleSet: string;
        maxRate: number;
        certified: number;
    };
    /**
     * The edge tier in front of this head (trainEdge's rows for it): edge-dismissed calibration, test
     * and background rows score 0 and edge-fired ones 1, so the head's guarantee is certified jointly:
     * a positive the edge dismissed is a miss, one it fired is caught, and its fires count against the
     * firing budget. The calibrator is fitted on the rows the edge leaves to the head. `edgeSha256` is
     * the head's edge model digest (trainEdge's rungs, edgeHeadDigest), recorded so the router can check
     * the release runs the same model. A list is a ladder (edgeLadder): the first candidate with which the
     * head passes its gates with a guarantee is kept (result.edgeChoice; assembleEdge builds the document).
     */
    edge?: EdgeInputRows | readonly EdgeInputRows[];
    /**
     * The middle tier behind the edge (middle.ts; lab stages CC and DY): chosen for this head from the
     * training rows only, before calibration. The edge's, the middle's and this head's training-row
     * scores re-derive every tier's thresholds on the training rows; the cascade is kept when its false
     * alarms at target + reserve are at most 1.1 × edge → head's + 0.001 (chooseCascade). Kept: the head
     * is certified jointly with `rows` (the edge's and the middle's decisions) and records
     * `middle: { model_sha256 }`; otherwise with the edge alone. Recall heads behind a single edge
     * candidate only (what the lab validated). cascadeInput builds this from trainEdge and trainMiddle.
     * `choice` fixes the outcome instead (decide it before training: choosing after seeing the gates
     * spends the guarantee's confidence twice).
     */
    cascade?: CascadeInput;
    /**
     * Extra POSITIVES for the train split only, e.g. rule-miner's weakLabels(): embeddings with a
     * weight in [0, 1] each. They never reach calibration, test or the background budget, and an
     * embedding identical to a calibration, test or background row is refused as leakage.
     */
    weak?: WeakInput;
    /**
     * Cap on total weak weight as a multiple λ of the head's gold train positives (default 0.5): weak
     * weights are scaled down to fit. Class balancing is sample-weighted, so weak positives take a
     * share λ / (1 + λ) of the positive class's weight rather than adding to it.
     */
    maxWeakShare?: number;
    /**
     * Add the finer-label route score (finer.ts) to this head's features: the embedding plus
     * scale · logit(Σ P(finer label | x) · route rate), certified like any head. Needs TrainInput.finer.
     * The head is linear (type 'auto' stays linear; 'knn' and 'stack' are refused). Off by default:
     * it pays where the route is a union of finer labels (lab stages SL and JC).
     */
    finer?: boolean | {
        scale?: number;
    };
}
export interface CascadeInput {
    /** The edge's and the middle's decisions together (trainMiddle's heads[h].rows). */
    rows: {
        fired: readonly boolean[];
        dismissed: readonly boolean[];
        background?: {
            fired: readonly boolean[];
            dismissed: readonly boolean[];
        };
    };
    /** middleHeadDigest of the head's middle model. */
    middleSha256: string;
    /** Training-row scores aligned with X (NaN off the training rows): the edge's out-of-fold blend, the middle's cross-fitted logit. */
    scores: {
        edge: readonly number[];
        middle: readonly number[];
    };
    budgets: CascadeBudgets;
    choice?: 'cascade' | 'edge';
}
/** A head's cascade decision (result.cascadeChoice, stored as artifact.training.cascade_choice). */
export interface CascadeChoiceRecord {
    cascade: boolean;
    /** Training false alarms at target + reserve (absent when `choice` fixed the outcome). */
    costs?: {
        twoTier: number;
        cascade: number;
    };
    forced?: true;
    criterion: string;
}
export interface EdgeInputRows {
    fired: readonly boolean[];
    dismissed: readonly boolean[];
    background?: {
        fired: readonly boolean[];
        dismissed: readonly boolean[];
    };
    edgeSha256: string;
}
export interface WeakInput {
    X: ReadonlyArray<ArrayLike<number>>;
    weights: readonly number[];
    /** Which rule produced each example, for the per-rule report. */
    rules?: readonly string[];
    /** The rule set the weak labels came from, recorded for audit. */
    source?: {
        rule_set_version: string;
        rule_set_hash: string;
    };
}
/** What weak supervision contributed to one head - store as artifact.training.weak_labels. */
export interface WeakSummary {
    count: number;
    gold_train_positives: number;
    max_weak_share: number;
    weight_before_cap: number;
    /** After the cap: at most max_weak_share × gold_train_positives. */
    weight: number;
    /** Multiplier the cap applied to every weak weight (1 = no cap). */
    scale: number;
    by_rule?: Record<string, {
        count: number;
        weight: number;
    }>;
    rule_set_version?: string;
    rule_set_hash?: string;
}
export interface TrainInput<H extends string> {
    X: ReadonlyArray<ArrayLike<number>>;
    split: readonly Split[];
    heads: ReadonlyArray<HeadInput<H>>;
    /** Inverse L2 strength (default 1). */
    C?: number;
    /**
     * Fit linear and finer heads on the embedding standardised per dimension (mean and sd over X and
     * background.X: features only, no labels), with the standardisation folded into the stored weights,
     * so the artifact and runtime are unchanged. It changes what the L2 penalty shrinks. With C = 0.01 it is
     * the lab's "recipe" (stage JC), which helped on EmbeddingGemma and was mixed on mpnet: measure before
     * adopting. Off by default. knn and stack heads are unaffected.
     */
    standardize?: boolean;
    /**
     * The finer label of each row of X (null where unknown), for heads with `finer`: one multinomial model
     * over it, fitted on the training rows that carry one (out of fold for those rows) and stored as
     * artifact.finer. C is its inverse L2 strength (default 0.01). Training rows the finer labels came
     * from need no other label: rows labelled only with a finer label still fit the shared model.
     */
    finer?: {
        labels: ReadonlyArray<string | null>;
        C?: number;
    };
    /**
     * The calibrator: Platt with Firth's bias reduction ('platt-firth', the default since stage CG: finite
     * when the calibration scores separate the classes, where plain Platt's slope runs off; stored and
     * served as Platt), plain Platt ('platt', the default before), or isotonic.
     */
    calibration?: 'platt' | 'platt-firth' | 'isotonic';
    /** review_floor = threshold × reviewRatio (default 0.5). */
    reviewRatio?: number;
    /**
     * Conformal review floor instead of reviewRatio: the floor below which at most this share of
     * positives fall, in expectation (calibration positive groups; capped at the threshold). Messages
     * below it are dismissed automatically with that stated miss rate.
     */
    reviewEpsilon?: number;
    maxEce?: number;
    /**
     * Fail a head whose test-split probabilities Cox's recalibration test rejects at this level (default
     * off). Prefer it to maxEce for rare classes: in simulation the ECE > 0.05 gate never fired at 2%
     * prevalence, even for badly miscalibrated probabilities. What it checks is narrow: whether the test
     * split's logistic recalibration matches the calibrator. It cannot see the curve's shape, so
     * probabilities that are wrong in the tail pass it as often as right ones.
     */
    calibrationAlpha?: number;
    /**
     * Which calibration gate calibrationAlpha applies: 'cox-corrected' (default since stage CK), Cox's
     * recalibration test with the Platt calibrator's own sampling error counted (for isotonic calibration,
     * plain Cox with a warning; not applied, with a warning, when a fit's covariance is singular); 'cox',
     * the plain test, which treats the fitted calibrator as exact and so refuses about 10% of correctly
     * calibrated heads at α = 0.01 - but also catches more heads whose calibrator is badly off by its own
     * sampling error, which the corrected test forgives as noise; or 'region' (stage
     * CG), the decision-region gate: fail when the observed rate and the mean probability differ by
     * demonstrably more than calibrationTolerance (default 0.05) at or above the review floor or in the top
     * decile of scores, counting the Platt calibrator's own sampling error.
     */
    calibrationGate?: 'cox' | 'cox-corrected' | 'region';
    calibrationTolerance?: number;
    /**
     * 'pooled' (opt-in; default 'none'): after the gates, refit each head's Platt calibrator on the calibration
     * and test splits together (each weighted as it was evaluated: 1/π with sampled records), and carry the
     * threshold over by rank, so every decision is unchanged and only the probabilities move. The gates, the
     * test evaluation and every threshold are still computed with the calibration-only fit; the shipped
     * probabilities therefore have no held-out check of their own (a warning says so). Lab stage CP2: on 140
     * heads the pooled probabilities were closer to the truth in the decision regions (−0.80 points; heads 5
     * points or more off fell from 23 to 8; in simulation the chance of a large error roughly halved), with the
     * best-calibrated head (ToxicChat jailbreak) slightly worse. Platt calibrators only.
     */
    calibrationRefit?: 'none' | 'pooled';
    groups?: readonly string[];
    slices?: Readonly<Record<string, readonly string[]>>;
    /**
     * Ordinary, mostly-negative traffic (e.g. everyday messages): each listed head's threshold is
     * raised until it fires on at most maxRate of it - BEFORE test evaluation, so the gates judge
     * the threshold that will ship. Real traffic holds positives at production prevalence, so
     * maxRate must sit above prevalence × the recall you want (8% for a 4% head), or the budget
     * caps recall rather than false alarms.
     */
    background?: {
        X: ReadonlyArray<ArrayLike<number>>;
        maxRate: Partial<Record<H, number>>;
        /** With `records`: one per row of background.X - unlabelled traffic with backgroundUse 'budget' (P2). */
        records?: readonly ExampleRecord[];
    };
    /**
     * One record per row of X: role must equal split, and labels[head] must equal each head's y.
     * Turns on provenance enforcement, record weights with P6 caps, and design-based calibration
     * and evaluation (see the module comment).
     */
    records?: readonly ExampleRecord[];
    /** The designSample summaries behind the sampled records - recorded in result.design. */
    designs?: readonly DesignSummary[];
    /** Overrides, accepted batches and the near-duplicate cosine for validateProvenance. */
    provenance?: Omit<ProvenanceOptions, 'embeddings' | 'safetyCritical'>;
    /** P6 caps; ruleMatches is required when generated hard negatives are present. */
    caps?: WeightCaps;
    /**
     * Embeddings for the provenance near-duplicate check (P5), aligned with X then background.X
     * (default: X and background.X). Pass them when X holds other features - stacked scores, or
     * embeddings with extra columns - since cosine similarity between those isn't about the text.
     */
    provenanceEmbeddings?: {
        X: ReadonlyArray<ArrayLike<number>>;
        background?: ReadonlyArray<ArrayLike<number>>;
    };
    /** Seed for design bootstraps and the stack head's principal components (default 0). */
    seed?: number;
    /**
     * Cross-fitting folds for knn, stack and auto heads: rows with the same key share a fold (default:
     * `groups`, else the row index). Keep near-duplicates together, or a training row's out-of-fold
     * score still finds its twin.
     */
    foldKeys?: readonly string[];
    /**
     * type 'auto': a knn or stack head is chosen only when its cross-validated cost is below the linear
     * head's × (1 − autoMargin) (default 0.2), then the cheaper of the two. Cross-validated costs on a
     * small training split are noisy; without a margin, near-ties often pick a non-linear head that does
     * worse on new traffic. A relative margin keeps the clear wins and drops the near-ties.
     */
    autoMargin?: number;
    /**
     * A fit that stopped short of its tolerance (the head's logistic model, a stack's linear component,
     * or the Platt calibrator - the likeliest, on a small calibration split the scores separate) is a
     * warning by default; true makes it a gate failure. result.convergence records every head's fits.
     */
    requireConvergence?: boolean;
    /** Encoding of the knn reference (default 'f32'): 'int8' is about 4× smaller, with essentially the same results. */
    referenceEncoding?: 'f32' | 'int8';
    log?: (line: string) => void;
}
/** What provenance enforcement did - store as artifact.training.provenance (with generated, see publishPlan). */
export interface ProvenanceSummary {
    dropped: Array<{
        id: string;
        code: 'P5';
        reason: string;
    }>;
    overridden: Array<{
        code: ProvenanceCode;
        ids: string[];
        message: string;
    }>;
    /** Any generated record was trained on: the artifact may only be a shadow candidate until acceptance evidence exists. */
    generated: boolean;
    safety_critical: string[];
    heads: Record<string, WeightCapSummary>;
}
export interface TrainResult<H extends string> {
    heads: Partial<Record<H, HeadSpec>>;
    evaluation: Partial<Record<H, HeadEvaluation>>;
    /** Gate failures, prefixed with the head name. Empty means every gate passed. */
    failures: string[];
    warnings: string[];
    /** Test-split probabilities per head (example indices into X), for round-trip checks. */
    testProbabilities: Partial<Record<H, {
        idx: number[];
        p: number[];
    }>>;
    /** Heads trained with weak positives - store as artifact.training.weak_labels so a release can be audited. */
    weakLabels: Partial<Record<H, WeakSummary>>;
    /** With records: provenance enforcement - store as artifact.training.provenance. */
    provenance?: ProvenanceSummary;
    /** Training embeddings for knn and stack heads - store as artifact.reference (scoreEmbedding needs it). */
    reference?: ReferenceSet;
    /** Whether each head's fits converged: its model (and a stack's linear component), and its Platt calibrator. */
    convergence: Partial<Record<H, {
        model: boolean;
        calibration: boolean;
    }>>;
    /** Heads trained with type 'auto': each type's cross-validated cost and the choice - store as artifact.training.head_choice. */
    headChoice: Partial<Record<H, HeadChoice>>;
    /** The finer-label model shared by finer heads - store as artifact.finer (buildArtifact does). */
    finer?: FinerModel;
    /** Heads trained with an edge: which candidate of their ladder was kept (0 for a single edge). */
    edgeChoice?: Partial<Record<H, number>>;
    /** Heads given HeadInput.cascade: whether the middle tier was kept, and why. */
    cascadeChoice: Partial<Record<H, CascadeChoiceRecord>>;
    /** With sampled records: the designs and each head's estimator - store as artifact.training.design. */
    design?: {
        designs: DesignSummary[];
        heads: Record<string, {
            calibration: 'design-weighted';
            threshold: string;
            evaluation: 'design-linearised';
        }>;
    };
    /** How the heads were calibrated and gated - store as artifact.training.calibration (buildArtifact does). */
    calibration: CalibrationRecord;
}
/**
 * The calibration choices a training run made, defaults resolved: the calibrator, and the calibration gates
 * applied (the ECE bound; the calibration test when calibrationAlpha was given). A head's own
 * `calibration.method` says only how it is served ('platt' covers both Platt fits), so a release records
 * the fit here, to show whether a stored result is comparable with a later library's defaults.
 */
export interface CalibrationRecord {
    method: 'platt' | 'platt-firth' | 'isotonic';
    max_ece: number;
    /** The calibration test, or null when calibrationAlpha was not given. */
    gate: {
        kind: 'cox' | 'cox-corrected' | 'region';
        alpha: number;
        tolerance?: number;
    } | null;
    /** Present when calibrationRefit 'pooled' was chosen: the shipped calibrators were refitted on calibration ∪ test after the gates. */
    refit?: 'pooled';
}
export interface HeadChoice {
    chosen: HeadType;
    /** Out-of-fold guarantee cost per type on the training rows (lower is better). */
    costs: Record<HeadType, number>;
    criterion: string;
}
/** The edge and dismissal rules decided every calibration row, leaving the head's calibrator nothing to fit. */
export declare class CalibrationRemainderError extends Error {
    name: string;
}
export declare function trainHeads<H extends string>(input: TrainInput<H>): TrainResult<H>;
/**
 * Serialises the artifact, re-loads it through validateArtifact and checks runtime scoring
 * reproduces the evaluated test probabilities exactly - so what was evaluated is what will run.
 */
export declare function assertRoundTrip<H extends string>(artifact: ClassifierArtifact<H>, X: ReadonlyArray<ArrayLike<number>>, testProbabilities: TrainResult<H>['testProbabilities'], sample?: number): void;
/**
 * The artifact for a training result, with its gates taken from the result (passed exactly when no
 * gate failed), so they are never set by hand. Also records the reference, the finer model, head choices, convergence,
 * provenance, design and weak-label summaries. Serve it through validateArtifact(raw, { mode }).
 */
export declare function buildArtifact<H extends string>(result: TrainResult<H>, options: {
    version: string;
    embedding: EmbeddingSpec;
    createdAt?: string;
    training?: Record<string, unknown>;
    /** For @liquidau/router: the rule set the artifact was trained and evaluated with, stored as `training.router`. */
    router?: RouterTraining;
}): ClassifierArtifact<H>;
