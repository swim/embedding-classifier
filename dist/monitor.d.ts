import type { HeadSpec } from './artifact.ts';
export interface DriftCheck {
    test: 'firing-up' | 'firing-down' | 'review-up' | 'review-down';
    pValue: number;
    alert: boolean;
    detail: string;
}
export declare function monitorWindow(options: {
    spec: Pick<HeadSpec, 'threshold' | 'review_floor'>;
    /** Certified upper bound on the background firing rate (evaluation.background_rate_upper); else the reference's upper bound. */
    firingBound?: number;
    /** Scores of the reference traffic, from the same artifact. */
    reference: ArrayLike<number>;
    /** Scores of the live window. */
    live: ArrayLike<number>;
    /** Family-wise false-alarm rate for this window (default 0.01). */
    alpha?: number;
}): {
    alert: boolean;
    checks: DriftCheck[];
};
