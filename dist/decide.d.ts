/**
 * Turning scores into one decision. Heads are checked in priority order and the first at or above
 * its threshold wins. Suppression rules stop some heads from firing while others are in their
 * review band. That is how "never send a possibly urgent message a dismissive 'out of scope' reply"
 * is expressed: suppress the scope heads whenever any priority head is at or above its review floor.
 */
import type { HeadSpec, Scores } from './artifact.ts';
/**
 * What decide() reads from an artifact: each head's threshold and review floor. A ClassifierArtifact
 * is one; so is a restricted projection such as @liquidau/router's offline decision model.
 */
export interface DecisionHeads<H extends string = string> {
    heads: Partial<Record<H, Pick<HeadSpec, 'threshold' | 'review_floor'>>>;
}
/** 'rule': a certified firing rule decided the head (settleWithRules, or `fired` in decide). */
export type DecisionReason = 'above_threshold' | 'near_threshold' | 'none' | 'rule';
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
export declare function missingPolicyHeads<H extends string>(artifact: DecisionHeads<H>, policy: DecisionPolicy<H>): H[];
/**
 * Throws if a head the artifact contains has a missing or non-finite score, or one outside [0, 1]: that
 * is a scoring bug or a corrupted artifact, not a negative. `dismissed`: heads a certified dismissal rule cleared for this message (rule-miner's
 * matcher.evaluate); they need no score, never fire and never suppress, as their training counted.
 */
export declare function decide<H extends string>(artifact: DecisionHeads<H>, scores: Scores<H>, policy: DecisionPolicy<H>, dismissed?: readonly H[], fired?: H | null): Decision<H>;
/** What a rule set says about one message: rule-miner's `ruleSetMatcher(set).evaluate(text)`. */
export interface RulesEvaluation {
    fired: {
        id: string;
        label: string;
    } | null;
    dismissed: readonly string[];
}
export type Settlement<H extends string> = {
    settled: true;
    decision: Decision<H>;
} | {
    settled: false;
    forward: {
        fired: H | null;
        dismissed: H[];
    };
};
/**
 * The rules tier's decision, without the model: settled when the rules alone fix what `decide` would
 * return whatever the scores - every head the policy names is dismissed, or a firing rule's head
 * comes after only dismissed heads in priority and nothing undismissed can suppress it. Otherwise the
 * message goes to the model tier with what the rules found, for decide(artifact, scores, policy,
 * forward.dismissed, forward.fired). Both paths give the same decision (tested), so the guarantees
 * describe the system whichever path a message takes.
 */
export declare function settleWithRules<H extends string>(policy: DecisionPolicy<H>, evaluation: RulesEvaluation): Settlement<H>;
