import type { ExampleRecord } from './records.ts';
/** A short, stable, non-cryptographic id (FNV-1a, 52 bits): reproducibility, not security. */
export declare function stableId(prefix: string, parts: unknown): string;
export interface FrameItem {
    id: string;
    text: string;
    group: string;
    signals: {
        ruleFires: boolean;
        score: number;
        slice?: string;
    };
}
export interface DesignStratum {
    name: string;
    /** Frame items in the stratum (after deduplication). */
    N: number;
    /** Sampled items, all roles. */
    n: number;
    /** n / N. */
    pi: number;
    /** Sampled items per role. */
    roles: {
        train: number;
        calibration: number;
        test: number;
    };
    /** Score range of the band, [lower, upper). */
    band: [number, number];
    /** expected-positives: the prior rate used and the positives the calibration share expects. */
    priorRate?: number;
    expectedCalibrationPositives?: number;
}
export interface DesignSummary {
    designId: string;
    allocation: {
        method: DesignOptions['allocation']['method'];
        minExpectedPositives?: number;
        minShare?: number;
        priorRates?: Record<string, number>;
    };
    /** Strata that could not reach minExpectedPositives even when taken whole, and similar notes. */
    warnings: string[];
    scoringModel: string;
    seed: number;
    /** Frame items removed because their group already had one. */
    duplicatesRemoved: number;
    strata: DesignStratum[];
}
export interface DesignOptions {
    frame: readonly FrameItem[];
    /** Descending quantile cut points of the frame's scores, e.g. [0.99, 0.95, 0.8]. */
    scoreBands: readonly number[];
    /** Cross strata with signals.slice (default false). */
    bySlice?: boolean;
    allocation: {
        total: number;
        method: 'proportional' | 'manual' | 'expected-positives';
        manual?: Record<string, number>;
        /**
         * expected-positives: per stratum name, the positive rate from a prior, independent round -
         * a rate in (0, 1], or counts { positives, labelled } (estimated as (positives + 0.5) / (labelled + 1),
         * so a stratum with no positives yet still gets a finite, generous size).
         */
        prior?: Record<string, number | {
            positives: number;
            labelled: number;
        }>;
        /** expected-positives: positives each stratum's calibration and test shares should expect (default 10). */
        minExpectedPositives?: number;
        /**
         * expected-positives: only strata estimated to hold at least this share of all positives get
         * the floor (default 0.02). A stratum nearly empty of positives can't move the miss rate much,
         * and proving it empty would cost most of the stratum.
         */
        minShare?: number;
    };
    /** Default 30. */
    minPerStratum?: number;
    /** Default 0.5 / 0.25 / 0.25. */
    roleSplit?: {
        train: number;
        calibration: number;
        test: number;
    };
    /** Version of the model that produced signals.score. */
    scoringModel: string;
    seed: number;
}
export declare function designSample(options: DesignOptions): {
    designId: string;
    records: ExampleRecord[];
    design: DesignSummary;
};
/**
 * The stratified design of a set of sampled records (one role, one head's labelled subset): for
 * solvers' estimators. Inclusion probabilities are recomputed as n_labelled / N_h, so items that
 * were budget-dropped or skipped are simply absent.
 */
export declare function designOf(records: ReadonlyArray<Pick<ExampleRecord, 'id' | 'source'>>): {
    inclusionProbs: number[];
    strata: string[];
    stratumSizes: Record<string, number>;
};
export type Mechanism = 'sampled' | 'retrieved' | 'verification';
export interface QueueKey {
    queueId: string;
    /** itemId -> what the reviewer must not see. Keep away from reviewers. */
    items: Record<string, {
        record: ExampleRecord;
        mechanism: Mechanism;
        heads: string[];
    }>;
}
export interface QueueItem {
    record: ExampleRecord;
    mechanism: Mechanism;
    heads: string[];
}
/**
 * One blinded queue. Over budget, sampled items are dropped at random (inclusion probabilities are
 * recomputed later); retrieved and verification items are dropped from the end, so pass them in
 * priority order.
 */
export declare function labelQueue(options: {
    items: readonly QueueItem[];
    budget: Record<Mechanism, number>;
    seed: number;
}): {
    queueId: string;
    review: Array<{
        itemId: string;
        text: string;
        heads: string[];
    }>;
    key: QueueKey;
    dropped: Record<Mechanism, number>;
};
/**
 * Applies reviewers' labels. An item counts as labelled only when every requested head came back
 * 0 or 1; anything else is a skip. Sampled records get inclusionProb = labelled_{h,role} / N_h.
 */
export declare function applyReviews(key: QueueKey, reviews: ReadonlyArray<{
    itemId: string;
    labels: Record<string, 0 | 1>;
}>): {
    records: ExampleRecord[];
    skipRate: Record<string, {
        queued: number;
        labelled: number;
        rate: number;
    }>;
    warnings: string[];
};
