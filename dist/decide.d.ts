/**
 * Turning scores into one decision. Heads are checked in priority order and the first at or above
 * its threshold wins. Suppression rules stop some heads from firing while others are in their
 * review band. That is how "never send a possibly-at-risk user a dismissive 'out of scope' reply"
 * is expressed: suppress the scope heads whenever any risk head is at or above its review floor.
 */
import type { ClassifierArtifact, Scores } from './artifact.ts';
export type DecisionReason = 'above_threshold' | 'near_threshold' | 'none';
export interface Decision<H extends string = string> {
    head: H | null;
    reason: DecisionReason;
}
export interface DecisionPolicy<H extends string = string> {
    /** Heads in priority order. Heads not listed never fire. */
    priority: readonly H[];
    /** While any `when` head is at/above its review floor, none of `heads` may fire. */
    suppress?: ReadonlyArray<{
        when: readonly H[];
        heads: readonly H[];
    }>;
}
/**
 * Policy heads that the artifact doesn't contain. decide() deliberately allows these (an artifact may
 * not ship every head yet), but a missing head silently never fires and never suppresses - so a
 * suppression rule whose guard head is absent guards nothing. Call this once at startup and refuse
 * to serve, or log, if the result isn't what you expect.
 */
export declare function missingPolicyHeads<H extends string>(artifact: ClassifierArtifact<H>, policy: DecisionPolicy<H>): H[];
/** Throws if a head the artifact contains has a missing or non-finite score: that is a scoring bug, not a negative. */
export declare function decide<H extends string>(artifact: ClassifierArtifact<H>, scores: Scores<H>, policy: DecisionPolicy<H>): Decision<H>;
