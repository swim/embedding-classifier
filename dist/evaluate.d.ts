/**
 * Test-split evaluation of one head: recall with a Wilson CI, false-alarm rate,
 * prevalence-weighted precision and ECE, a reliability table, recall per slice, (when a baseline
 * is supplied, e.g. existing rules) what the classifier adds on top of it, and exact certified
 * bounds for the shipped threshold whatever chose it.
 */
import { type ReliabilityRow } from '@liquidau/solvers';
import type { Guarantee, Sufficiency } from './conformal.ts';
/**
 * Exact (Clopper-Pearson) bounds from the test split, which no threshold method sees - so they are
 * valid for any threshold, heuristic or conformal. Each holds with probability 1 - delta/2, so
 * recall_lower and false_alarm_upper (and precision_lower, derived from both) hold together with
 * probability 1 - delta. Groups count once: a positive group is caught for the lower bound only if
 * every member fires, and for the upper bound if any does; a negative group fires if any member does.
 */
export interface CertifiedBounds {
    delta: number;
    positive_groups: number;
    negative_groups: number;
    recall_lower: number;
    recall_upper: number;
    false_alarm_upper: number;
    /** At the head's production prevalence: π·R_L / (π·R_L + (1 - π)·FA_U). */
    precision_lower?: number;
    /** 'exact': Clopper-Pearson on an equal-probability test set. 'design-linearised': one-sided normal bounds from a stratified sample - approximate. */
    method?: 'exact' | 'design-linearised';
}
export interface DesignEstimate {
    estimate: number;
    se: number;
    /** Linearised, estimate ± 1.96·se clipped to [0, 1]. */
    ci95: [number, number];
    /** Rao-Wu bootstrap percentile interval. */
    bootstrap_ci95?: [number, number];
    /** Kish effective number of units in the estimate's denominator (e.g. positives, for recall). */
    effective_n?: number;
}
/** Design-based (Horvitz-Thompson) estimates from a stratified probability sample. Approximate. */
export interface DesignEvaluation {
    recall: DesignEstimate;
    /** At production prevalence: the design weights reproduce it. */
    precision: DesignEstimate | null;
    false_alarm_rate: DesignEstimate;
    prevalence: DesignEstimate;
    slices: Record<string, DesignEstimate>;
    /** Kish effective number of positives - what the gates' minimums count. */
    effective_positives: number;
    /** With a baseline: baseline OR classifier, design-weighted. */
    combined_recall?: DesignEstimate;
    combined_false_alarm_rate?: DesignEstimate;
    replicates: number;
}
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
    /** Exact upper bound on the background firing rate (probability 1 - delta/2), valid even when the background chose the threshold. */
    background_rate_upper?: number;
    certified: CertifiedBounds;
    /** Present when the test set is a probability sample: recall, precision and false alarms above are these estimates. */
    design?: DesignEvaluation;
    /**
     * Cox's recalibration test on the test split (weights as for ECE): y ~ a + b·logit(p), H0 a = 0, b = 1.
     * Unlike ECE it has a stated error rate and stays sensitive for rare classes.
     */
    calibration_test?: {
        intercept: number;
        slope: number;
        lr: number;
        p_value: number;
        /** False under separation (every positive scores above every negative), or when the fit didn't converge or ran off (|a| or |b| > 10): the p-value then means nothing. */
        stable: boolean;
    };
    /** Recall heads: what the shipped threshold guarantees from calibration, and why. */
    guarantee?: Guarantee;
    sufficiency?: Sufficiency;
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
    /** Certified bounds fail with probability at most delta (default 0.05). */
    delta?: number;
    /** Production prevalence, for certified.precision_lower. */
    prevalence?: number;
    /**
     * The test set's stratified design (designOf on the sampled test records), aligned with p.
     * Recall, precision, false-alarm rate and their intervals then become design-based estimates;
     * pass w = 1/π so ECE is weighted too.
     */
    design?: {
        inclusionProbs: readonly number[];
        strata: readonly string[];
        stratumSizes: Readonly<Record<string, number>>;
        replicates?: number;
        seed?: number;
    };
}
export declare function evaluateHead({ p, y, w, threshold, groups, slices, baseline, delta, prevalence, design }: EvaluateInput): HeadEvaluation;
