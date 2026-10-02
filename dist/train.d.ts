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
    /** Expected positive rate in production - calibration and precision are weighted to it. */
    prevalence: number;
    /** Whether an existing mechanism already catches each example (see evaluateHead). */
    baseline?: readonly boolean[];
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
    };
    log?: (line: string) => void;
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
}
export declare function trainHeads<H extends string>(input: TrainInput<H>): TrainResult<H>;
/**
 * Serialises the artifact, re-loads it through validateArtifact and checks runtime scoring
 * reproduces the evaluated test probabilities exactly - so what was evaluated is what will run.
 */
export declare function assertRoundTrip<H extends string>(artifact: ClassifierArtifact<H>, X: ReadonlyArray<ArrayLike<number>>, testProbabilities: TrainResult<H>['testProbabilities'], sample?: number): void;
