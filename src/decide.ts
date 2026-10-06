/**
 * Turning scores into one decision. Heads are checked in priority order and the first at or above
 * its threshold wins. Suppression rules stop some heads from firing while others are in their
 * review band. That is how "never send a possibly urgent message a dismissive 'out of scope' reply"
 * is expressed: suppress the scope heads whenever any priority head is at or above its review floor.
 */
import type { ClassifierArtifact, Scores } from './artifact.ts';

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
  suppress?: ReadonlyArray<{ when: readonly H[]; heads: readonly H[] }>;
}

/**
 * Policy heads that the artifact doesn't contain. decide() deliberately allows these (an artifact may
 * not ship every head yet), but a missing head silently never fires and never suppresses - so a
 * suppression rule whose guard head is absent guards nothing. Call this once at startup and refuse
 * to serve, or log, if the result isn't what you expect.
 */
export function missingPolicyHeads<H extends string>(artifact: ClassifierArtifact<H>, policy: DecisionPolicy<H>): H[] {
  const named = new Set<H>([...policy.priority, ...(policy.suppress ?? []).flatMap((r) => [...r.when, ...r.heads])]);
  return [...named].filter((h) => artifact.heads[h] === undefined);
}

/**
 * Throws if a head the artifact contains has a missing or non-finite score, or one outside [0, 1]: that
 * is a scoring bug or a corrupted artifact, not a negative. `dismissed`: heads a certified dismissal rule cleared for this message (rule-miner's
 * matcher.evaluate); they need no score, never fire and never suppress, as their training counted.
 */
export function decide<H extends string>(artifact: ClassifierArtifact<H>, scores: Scores<H>, policy: DecisionPolicy<H>, dismissed: readonly H[] = [], fired: H | null = null): Decision<H> {
  // `fired`: a certified firing rule matched this head ("rules or classifier"): it counts as at its threshold.
  const atLeast = (head: H, level: 'threshold' | 'review_floor') => {
    if (head === fired && !dismissed.includes(head)) return true;
    const spec = artifact.heads[head];
    if (spec === undefined || dismissed.includes(head)) return false;
    const score = scores[head];
    if (typeof score !== 'number' || !Number.isFinite(score)) throw new Error(`head ${head} has no finite score (${score})`);
    if (score < 0 || score > 1) throw new Error(`head ${head} has a score outside [0, 1] (${score}): not a calibrated probability`);
    return score >= spec[level];
  };
  const suppressed = new Set<H>();
  for (const rule of policy.suppress ?? []) {
    if (rule.when.some((h) => atLeast(h, 'review_floor'))) rule.heads.forEach((h) => suppressed.add(h));
  }
  for (const head of policy.priority) {
    if (!suppressed.has(head) && atLeast(head, 'threshold')) return { head, reason: head === fired ? 'rule' : 'above_threshold' };
  }
  if (policy.priority.some((h) => atLeast(h, 'review_floor'))) return { head: null, reason: 'near_threshold' };
  return { head: null, reason: 'none' };
}

/** What a rule set says about one message: rule-miner's `ruleSetMatcher(set).evaluate(text)`. */
export interface RulesEvaluation {
  fired: { id: string; label: string } | null;
  dismissed: readonly string[];
}

export type Settlement<H extends string> =
  | { settled: true; decision: Decision<H> }
  | { settled: false; forward: { fired: H | null; dismissed: H[] } };

/**
 * The rules tier's decision, without the model: settled when the rules alone fix what `decide` would
 * return whatever the scores - every head the policy names is dismissed, or a firing rule's head
 * comes after only dismissed heads in priority and nothing undismissed can suppress it. Otherwise the
 * message goes to the model tier with what the rules found, for decide(artifact, scores, policy,
 * forward.dismissed, forward.fired). Both paths give the same decision (tested), so the guarantees
 * describe the system whichever path a message takes.
 */
export function settleWithRules<H extends string>(policy: DecisionPolicy<H>, evaluation: RulesEvaluation): Settlement<H> {
  const named = new Set<H>([...policy.priority, ...(policy.suppress ?? []).flatMap((r) => [...r.when, ...r.heads])]);
  const dismissed = evaluation.dismissed.filter((h): h is H => named.has(h as H));
  const off = new Set<H>(dismissed);
  const fired = evaluation.fired && named.has(evaluation.fired.label as H) && !off.has(evaluation.fired.label as H) ? (evaluation.fired.label as H) : null;
  if (fired) {
    const before = policy.priority.slice(0, Math.max(0, policy.priority.indexOf(fired)));
    const canSuppress = (policy.suppress ?? []).some((r) => r.heads.includes(fired) && r.when.some((w) => !off.has(w)));
    if (policy.priority.includes(fired) && before.every((h) => off.has(h)) && !canSuppress) return { settled: true, decision: { head: fired, reason: 'rule' } };
  }
  if (policy.priority.every((h) => off.has(h))) return { settled: true, decision: { head: null, reason: 'none' } };
  return { settled: false, forward: { fired, dismissed } };
}

