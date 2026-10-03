import assert from 'node:assert/strict';
import { test } from 'node:test';

import { seededRandom } from '@liquidau/solvers';

import { monitorWindow } from '../src/index.ts';

// Ordinary traffic scores u^4: threshold 0.984 fires on ~0.4%, review floor 0.9 on ~2.6%.
const make = (n: number, rand: () => number, drift: string) => Array.from({ length: n }, () => {
  if (drift === 'tail' && rand() < 0.01) return 0.985 + 0.015 * rand(); // a new topic the head fires on
  const base = rand() ** 4;
  if (drift === 'drop' && base >= 0.984) return base * 0.95; // what used to fire now scores just below
  if (drift === 'lowshift') return base < 0.5 ? base * 0.8 : base; // moves only scores no decision depends on
  return base;
});
const spec = { threshold: 0.984, review_floor: 0.9 };

test('monitorWindow: false alarms within alpha; catches new high-scoring topics and lost firing; ignores irrelevant shifts', () => {
  const rand = seededRandom(5);
  const reference = make(20000, rand, 'none');
  const firingBound = (reference.filter((p) => p >= spec.threshold).length / reference.length) * 1.2 + 0.0005;
  const rate = (drift: string, runs = 100) => {
    let alerts = 0;
    for (let r = 0; r < runs; r++) if (monitorWindow({ spec, firingBound, reference, live: make(10000, rand, drift), alpha: 0.01 }).alert) alerts++;
    return alerts / runs;
  };
  assert.ok(rate('none') <= 0.01 + 3 * Math.sqrt(0.01 * 0.99 / 100), 'false alarms');
  assert.ok(rate('tail') >= 0.95, 'a new topic above the threshold');
  assert.ok(rate('drop') >= 0.95, 'messages that used to fire no longer do');
  assert.ok(rate('lowshift') <= 0.05, 'a shift among low scores changes no decision');
  const res = monitorWindow({ spec, firingBound, reference, live: make(10000, rand, 'tail') });
  assert.deepEqual(res.checks.map((c) => c.test), ['firing-up', 'firing-down', 'review-up', 'review-down']);
  assert.match(res.checks[0].detail, /certified/);
  assert.throws(() => monitorWindow({ spec, reference: [], live: [1] }), /non-empty/);
});
