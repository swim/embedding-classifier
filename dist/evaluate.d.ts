/**
 * Test-split evaluation of one head: recall with a Wilson CI, false-alarm rate,
 * prevalence-weighted precision and ECE, a reliability table, recall per slice, and (when a
 * baseline is supplied, e.g. existing rules) what the classifier adds on top of it.
 */
import { type ReliabilityRow } from '@liquidau/solvers';
export interface BaselineComparison {
    caught_by_both: number;
    caught_by_baseline_only: number;
    caught_by_classifier_only: number;
    missed_by_both: number;
}
export interface HeadEvaluation {
    /** The threshold evaluated - the one that ships, after any background budget. */
    threshold: number;
    n: number;
    /** Test examples at or above the threshold. */
    fired: number;
    positives: number;
    /** Distinct groups among test positives - paraphrases of one seed aren't independent. */
    positive_groups: number;
    recall: number;
    recall_ci95: [number, number];
    false_alarm_rate: number;
    precision_prevalence_weighted: number;
    ece_prevalence_weighted: number;
    reliability: ReliabilityRow[];
    slices: Record<string, {
        positives: number;
        recall: number;
        recall_ci95: [number, number];
    }>;
    vs_baseline?: BaselineComparison;
    /** Baseline OR classifier - what a deployment that runs both actually achieves. */
    combined_recall?: number;
    combined_recall_ci95?: [number, number];
    combined_false_alarm_rate?: number;
    /** Share of the background the head fires on at its final threshold (when a budget was applied). */
    background_rate?: number;
}
export interface EvaluateInput {
    /** Calibrated probabilities. */
    p: number[];
    y: number[];
    /** Per-example weights (e.g. prevalenceWeights) for precision and ECE. */
    w: number[];
    threshold: number;
    /** Group per example (default: every example is its own group). */
    groups?: readonly string[];
    /** field -> value per example; recall is reported per `field=value`. */
    slices?: Readonly<Record<string, readonly string[]>>;
    /** Whether an existing mechanism already catches each example. */
    baseline?: readonly boolean[];
}
export declare function evaluateHead({ p, y, w, threshold, groups, slices, baseline }: EvaluateInput): HeadEvaluation;
