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
  suppress?: ReadonlyArray<{ when: readonly H[]; heads: readonly H[] }>;
}

/**
 * Policy heads that the artifact doesn't contain. decide() deliberately allows these (an artifact may
 * not ship every head yet), but a missing head silently never fires and never suppresses - so a
 * suppression rule whose guard head is absent guards nothing. Call this once at startup and refuse
 * to serve, or log, if the result isn't what you expect.
 */
export function missingPolicyHeads<H extends string>(artifact: DecisionHeads<H>, policy: DecisionPolicy<H>): H[] {
  const named = new Set<H>([...policy.priority, ...(policy.suppress ?? []).flatMap((r) => [...r.when, ...r.heads])]);
  return [...named].filter((h) => artifact.heads[h] === undefined);
}

/**
 * Throws if a head the artifact contains has a missing or non-finite score, or one outside [0, 1]: that
 * is a scoring bug or a corrupted artifact, not a negative. `dismissed`: heads a certified dismissal rule cleared for this message (rule-miner's
 * matcher.evaluate); they need no score, never fire and never suppress, as their training counted.
 */
export function decide<H extends string>(artifact: DecisionHeads<H>, scores: Scores<H>, policy: DecisionPolicy<H>, dismissed: readonly H[] = [], fired: H | readonly H[] | null = null): Decision<H> {
  // `fired`: heads a certified rule (or an edge or middle tier) fired ("rules or classifier"): each counts
  // as at its threshold, as certification counted it. One head or several.
  const firedSet = new Set<H>(fired === null ? [] : typeof fired === 'string' ? [fired as H] : (fired as readonly H[]));
  const atLeast = (head: H, level: 'threshold' | 'review_floor') => {
    if (firedSet.has(head) && !dismissed.includes(head)) return true;
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
    if (!suppressed.has(head) && atLeast(head, 'threshold')) return { head, reason: firedSet.has(head) ? 'rule' : 'above_threshold' };
  }
  if (policy.priority.some((h) => atLeast(h, 'review_floor'))) return { head: null, reason: 'near_threshold' };
  return { head: null, reason: 'none' };
}

/** What a rule set says about one message: rule-miner's `ruleSetMatcher(set).evaluate(text)`. */
export interface RulesEvaluation {
  fired: { id: string; label: string } | null;
  /** Every label a firing rule matched (rule-miner's `firing`); `fired` alone when absent. */
  firing?: ReadonlyArray<{ id: string; label: string }>;
  dismissed: readonly string[];
}

export type Settlement<H extends string> =
  | { settled: true; decision: Decision<H> }
  | { settled: false; forward: { fired: H | null; firing: H[]; dismissed: H[] } };

/**
 * The rules tier's decision, without the model: settled when the rules alone fix what `decide` would
 * return whatever the scores. Every fired head counts as at its threshold (and so at its review floor),
 * a dismissed head as below both, and any other head as unknown. Walking the priority:
 *   - a head that is dismissed, or certainly suppressed (some rule naming it has a fired `when` head),
 *     can't win;
 *   - a fired head that is certainly not suppressed (every `when` head of every rule naming it is
 *     dismissed) wins;
 *   - anything else (an unknown head, or a fired one whose suppression is unknown) needs the scores.
 * When no head can win, a fired head in the priority means review ('near_threshold'), every head
 * dismissed means none, and otherwise the scores decide. Otherwise the message goes to the model tier
 * with what the rules found, for decide(artifact, scores, policy, forward.dismissed, forward.firing).
 * Both paths give the same decision (tested exhaustively in @liquidau/router), so the guarantees
 * describe the system whichever path a message takes.
 */
export function settleWithRules<H extends string>(policy: DecisionPolicy<H>, evaluation: RulesEvaluation): Settlement<H> {
  const named = new Set<H>([...policy.priority, ...(policy.suppress ?? []).flatMap((r) => [...r.when, ...r.heads])]);
  const dismissed = evaluation.dismissed.filter((h): h is H => named.has(h as H));
  const off = new Set<H>(dismissed);
  const labels = (evaluation.firing ?? (evaluation.fired ? [evaluation.fired] : [])).map((f) => f.label as H);
  const firing = [...new Set(labels)].filter((h) => named.has(h) && !off.has(h));
  const fired = evaluation.fired && firing.includes(evaluation.fired.label as H) ? (evaluation.fired.label as H) : firing[0] ?? null;
  const on = new Set<H>(firing);
  const forward = { settled: false as const, forward: { fired, firing, dismissed } };
  /** 'yes': certainly suppressed; 'no': certainly not; 'maybe': the scores decide. */
  const suppressed = (h: H): 'yes' | 'no' | 'maybe' => {
    const rules = (policy.suppress ?? []).filter((r) => r.heads.includes(h));
    if (rules.some((r) => r.when.some((w) => on.has(w)))) return 'yes';
    return rules.every((r) => r.when.every((w) => off.has(w))) ? 'no' : 'maybe';
  };
  for (const h of policy.priority) {
    if (off.has(h)) continue;
    const s = suppressed(h);
    if (s === 'yes') continue;
    if (on.has(h) && s === 'no') return { settled: true, decision: { head: h, reason: 'rule' } };
    return forward;
  }
  if (policy.priority.some((h) => on.has(h))) return { settled: true, decision: { head: null, reason: 'near_threshold' } };
  if (policy.priority.every((h) => off.has(h))) return { settled: true, decision: { head: null, reason: 'none' } };
  return forward;
}

