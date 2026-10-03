export type ThresholdMode = 'heuristic' | 'conformal-expected' | 'conformal-pac' | 'auto';
export type GuaranteeKind = 'pac' | 'expected' | 'none';
export declare const THRESHOLD_MODES: readonly ThresholdMode[];
/**
 * One score per group, for calibration: paraphrases of one seed aren't exchangeable, so a group
 * counts once. `pick` = Math.min for positives (a group is caught only if every member is),
 * Math.max for negatives (it fires if any member does) - both conservative.
 */
export declare function groupScores(scores: readonly number[], groups: readonly string[] | undefined, pick: (a: number, b: number) => number): number[];
export interface Guarantee {
    /** The mode the policy asked for. */
    mode: ThresholdMode;
    /** What the shipped threshold actually guarantees (calibration-based). */
    kind: GuaranteeKind;
    /** Miss rate guaranteed: 1 - targetRecall. */
    alpha: number;
    /** Family-wise: recall and every false-alarm budget hold together with probability 1 - delta (PAC only). */
    delta?: number;
    /** Certified share of calibration negatives that may fire (from maxFalseAlarm). */
    false_alarm?: number;
    /** Certified share of background text that may fire (from the background budget). */
    background_rate?: number;
}
export interface Sufficiency {
    /** Calibration positive groups - what the recall guarantee counts. */
    positive_groups: number;
    /** Positive groups needed for each guarantee (PAC with δ split across the enforced constraints). */
    needed: {
        expected: number;
        pac: number;
    };
    feasible: Array<'conformal-expected' | 'conformal-pac'>;
    /** The threshold method actually used. */
    chosen: Exclude<ThresholdMode, 'auto'>;
    reason?: string;
    /** Per slice (field=value): calibration positive groups and which guarantees a per-slice threshold could have. Reported, not enforced. */
    slices?: Record<string, {
        positive_groups: number;
        feasible: Array<'conformal-expected' | 'conformal-pac'>;
    }>;
}
export interface FalseAlarmConstraint {
    /** e.g. "calibration false-alarm" or "background". */
    name: string;
    scores: readonly number[];
    maxRate: number;
}
export interface ConformalSelection {
    /** null: use the heuristic threshold (auto's fallback when the data supports no guarantee). */
    threshold: number | null;
    guarantee: Guarantee;
    sufficiency: Sufficiency;
    failures: string[];
    warnings: string[];
}
/**
 * Chooses a recall head's threshold under a conformal mode. The recall guarantee caps the
 * threshold from above, each false-alarm budget bounds it from below; if they cross, no threshold
 * gives both and the head fails with that reason. `auto` never trades a guarantee away silently:
 * it falls back to a weaker guarantee only for lack of data (with an "inconclusive" warning) - to
 * no guarantee at all (the heuristic threshold) only while `allowHeuristic` is true - and
 * fails when the false-alarm budget is what rules PAC out.
 */
export declare function conformalThreshold(input: {
    mode: Exclude<ThresholdMode, 'heuristic'>;
    targetRecall: number;
    delta?: number;
    /** One calibrated score per calibration positive group (see groupScores). */
    positives: readonly number[];
    constraints?: readonly FalseAlarmConstraint[];
    /** Whether a heuristic threshold is available (designRecall set) as auto's last resort. */
    heuristicAvailable: boolean;
    /** auto: when the data supports no guarantee, use the heuristic threshold (true, default) or fail the head (false). */
    allowHeuristic?: boolean;
}): ConformalSelection;
