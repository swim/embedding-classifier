import { type DesignSummary } from './design.ts';
import { type ExampleRecord, type ProvenanceCode, type ProvenanceOptions, type WeightCaps, type WeightCapSummary } from './records.ts';
import { type ClassifierArtifact, type HeadSpec } from './artifact.ts';
import { type HeadEvaluation } from './evaluate.ts';
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
    calibration?: 'platt' | 'isotonic';
    /** review_floor = threshold × reviewRatio (default 0.5). */
    reviewRatio?: number;
    /**
     * Conformal review floor instead of reviewRatio: the floor below which at most this share of
     * positives fall, in expectation (calibration positive groups; capped at the threshold). Messages
     * below it are dismissed automatically with that stated miss rate.
     */
    reviewEpsilon?: number;
    maxEce?: number;
    groups?: readonly string[];
    slices?: Readonly<Record<string, readonly string[]>>;
    /**
     * Ordinary, mostly-negative traffic (e.g. everyday messages): each listed head's threshold is
     * raised until it fires on at most maxRate of it - BEFORE test evaluation, so the gates judge
     * the threshold that will ship.
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
    /** Seed for design bootstraps (default 0). */
    seed?: number;
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
    /** With sampled records: the designs and each head's estimator - store as artifact.training.design. */
    design?: {
        designs: DesignSummary[];
        heads: Record<string, {
            calibration: 'design-weighted';
            threshold: string;
            evaluation: 'design-linearised';
        }>;
    };
}
export declare function trainHeads<H extends string>(input: TrainInput<H>): TrainResult<H>;
/**
 * Serialises the artifact, re-loads it through validateArtifact and checks runtime scoring
 * reproduces the evaluated test probabilities exactly - so what was evaluated is what will run.
 */
export declare function assertRoundTrip<H extends string>(artifact: ClassifierArtifact<H>, X: ReadonlyArray<ArrayLike<number>>, testProbabilities: TrainResult<H>['testProbabilities'], sample?: number): void;
