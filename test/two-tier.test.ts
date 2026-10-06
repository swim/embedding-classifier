import assert from 'node:assert/strict';
import { test } from 'node:test';

import { checkRuleSetPairing, decide, settleWithRules, type ClassifierArtifact, type DecisionPolicy } from '../src/index.ts';

type H = 'a' | 'b' | 'c' | 'd';
const HEADS: H[] = ['a', 'b', 'c', 'd'];
const spec = (threshold: number) => ({ weights: [1], bias: 0, calibration: { method: 'platt' as const, a: 1, c: 0 }, threshold, review_floor: threshold / 2 });
const artifact: ClassifierArtifact<H> = { version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 1, normalize: false }, heads: { a: spec(0.5), b: spec(0.6), c: spec(0.4), d: spec(0.7) } };

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}

test('settleWithRules: a settled decision is what decide returns for EVERY possible set of scores', () => {
  const rand = rng(7);
  const pick = <T>(xs: readonly T[]) => xs[Math.floor(rand() * xs.length)];
  let settledCount = 0, forwarded = 0;
  for (let trial = 0; trial < 3000; trial++) {
    const order = [...HEADS].sort(() => rand() - 0.5);
    const priority = order.slice(0, 1 + Math.floor(rand() * 4));
    const suppress = rand() < 0.5 ? [{ when: [pick(HEADS)], heads: [pick(HEADS)] }] : [];
    const policy: DecisionPolicy<H> = { priority, suppress };
    const dismissed = HEADS.filter(() => rand() < 0.5);
    const fired = rand() < 0.5 ? { id: 'r1', label: pick(HEADS) } : null;
    const s = settleWithRules(policy, { fired, dismissed });
    if (!s.settled) {
      forwarded++;
      assert.deepEqual(s.forward.dismissed, dismissed.filter((h) => priority.includes(h) || suppress.some((r) => r.when.includes(h) || r.heads.includes(h))));
      continue;
    }
    settledCount++;
    // The model tier, given what the rules found, must agree whatever the model scores.
    for (let k = 0; k < 20; k++) {
      const scores = Object.fromEntries(HEADS.map((h) => [h, rand()])) as Record<H, number>;
      const firedHead = fired && !dismissed.includes(fired.label as H) ? (fired.label as H) : null;
      assert.deepEqual(decide(artifact, scores, policy, dismissed, firedHead), s.decision, JSON.stringify({ policy, dismissed, fired, scores }));
    }
  }
  assert.ok(settledCount > 300 && forwarded > 300, `${settledCount} settled, ${forwarded} forwarded`);
});

test('settleWithRules: the shortcut cases, and messages that must go to the model', () => {
  const policy: DecisionPolicy<H> = { priority: ['a', 'b'], suppress: [{ when: ['a'], heads: ['b'] }] };
  assert.deepEqual(settleWithRules(policy, { fired: null, dismissed: ['a', 'b'] }), { settled: true, decision: { head: null, reason: 'none' } });
  assert.deepEqual(settleWithRules(policy, { fired: { id: 'x', label: 'a' }, dismissed: [] }), { settled: true, decision: { head: 'a', reason: 'rule' } });
  // b fired, but a (not dismissed) comes first and could also suppress b: only the model can tell.
  assert.deepEqual(settleWithRules(policy, { fired: { id: 'x', label: 'b' }, dismissed: [] }), { settled: false, forward: { fired: 'b', dismissed: [] } });
  // ...once a is dismissed, b's rule decides.
  assert.deepEqual(settleWithRules(policy, { fired: { id: 'x', label: 'b' }, dismissed: ['a'] }), { settled: true, decision: { head: 'b', reason: 'rule' } });
  // Only some heads dismissed: forwarded with them.
  assert.deepEqual(settleWithRules(policy, { fired: null, dismissed: ['b', 'zzz'] }), { settled: false, forward: { fired: null, dismissed: ['b'] } });
});

test('checkRuleSetPairing: heads certified with another rule set are named', () => {
  const paired: ClassifierArtifact<H> = { ...artifact, heads: { a: { ...spec(0.5), dismissal: { rule_set: 'hash-1', max_rate: 0.05, certified: 4 } }, b: spec(0.6) } };
  assert.deepEqual(checkRuleSetPairing(paired, 'hash-1'), []);
  assert.deepEqual(checkRuleSetPairing(paired, 'hash-2'), ['a was certified with rule set hash-1, not hash-2']);
});
