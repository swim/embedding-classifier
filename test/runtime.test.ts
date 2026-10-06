import assert from 'node:assert/strict';
import { test } from 'node:test';

import { sigmoid } from '@liquidau/solvers';

import {
  calibrate,
  decide,
  loadOrder,
  missingPolicyHeads,
  refuseToServe,
  scoreEmbedding,
  validateArtifact,
  type Calibration,
  type ClassifierArtifact,
  type DecisionPolicy,
  type HeadSpec,
} from '../src/index.ts';

type H = 'urgent' | 'spam' | 'off_topic';

const spec = (threshold: number, review_floor: number, calibration: Calibration): HeadSpec => ({
  weights: [1, -1], bias: 0, calibration, threshold, review_floor,
});

const artifact: ClassifierArtifact<H> = {
  version: 'test', created_at: '', embedding: { model_id: 'm', dimensions: 2, normalize: true },
  heads: {
    urgent: spec(0.9, 0.4, { method: 'platt', a: 1, c: 0 }),
    off_topic: spec(0.5, 0.3, { method: 'platt', a: 1, c: 0 }),
  },
};

const policy: DecisionPolicy<H> = { priority: ['urgent', 'spam', 'off_topic'], suppress: [{ when: ['urgent'], heads: ['off_topic'] }] };

test('calibration maths', () => {
  assert.equal(calibrate({ method: 'platt', a: 2, c: -1 }, 0.25), sigmoid(-0.5));
  const iso = { method: 'isotonic' as const, x: [0.2, 0.6], y: [0, 0.8] };
  assert.ok(Math.abs(calibrate(iso, 0) - 0.6) < 1e-12); // raw 0.5 -> 3/4 of the way
  assert.equal(calibrate(iso, -20), 0); // clipped below
  assert.equal(calibrate(iso, 20), 0.8); // clipped above
});

test('scoring covers exactly the artifact heads and checks dimensions', () => {
  const scores = scoreEmbedding(artifact, [3, 0]);
  assert.ok(Math.abs((scores.urgent ?? 0) - sigmoid(3)) < 1e-15);
  assert.equal(scores.spam, undefined);
  assert.throws(() => scoreEmbedding(artifact, [1, 2, 3]), /3 dimensions/);
});

test('priority and reasons', () => {
  assert.deepEqual(decide(artifact, { urgent: 0.95, off_topic: 0.99 }, policy), { head: 'urgent', reason: 'above_threshold' });
  assert.deepEqual(decide(artifact, { urgent: 0.5, off_topic: 0.1 }, policy), { head: null, reason: 'near_threshold' });
  assert.deepEqual(decide(artifact, { urgent: 0.1, off_topic: 0.1 }, policy), { head: null, reason: 'none' });
  assert.deepEqual(decide<H>(artifact, { urgent: 0.95, off_topic: 0.1 }, { priority: ['off_topic'] }), { head: null, reason: 'none' }, 'unlisted heads never fire');
});

test('suppression: a head cannot fire while a guarding head is in its review band', () => {
  assert.deepEqual(decide(artifact, { urgent: 0.5, off_topic: 0.99 }, policy), { head: null, reason: 'near_threshold' });
  assert.deepEqual(decide(artifact, { urgent: 0.1, off_topic: 0.99 }, policy), { head: 'off_topic', reason: 'above_threshold' });
  assert.deepEqual(decide(artifact, { urgent: 0.5, off_topic: 0.99 }, { priority: policy.priority }), { head: 'off_topic', reason: 'above_threshold' }, 'no rule, no suppression');
});

test('a missing or non-finite score is an error, not a negative', () => {
  assert.throws(() => decide(artifact, { urgent: NaN, off_topic: 0.1 }, policy), /urgent has no finite score/);
  assert.throws(() => decide(artifact, { urgent: 0.1 }, policy), /off_topic has no finite score/);
  assert.throws(() => scoreEmbedding(artifact, [NaN, 1]), /non-finite value at index 0/);
});

test('policy heads missing from the artifact are reported', () => {
  assert.deepEqual(missingPolicyHeads(artifact, policy), ['spam']);
  assert.deepEqual(missingPolicyHeads(artifact, { priority: ['urgent'], suppress: [{ when: ['spam'], heads: ['off_topic'] }] }), ['spam']);
  assert.deepEqual(missingPolicyHeads(artifact, { priority: ['urgent', 'off_topic'] }), []);
});

test('artifact validation checks values, not just shape', () => {
  const head = (patch: Record<string, unknown>) => validateArtifact({ ...artifact, heads: { urgent: { ...artifact.heads.urgent!, ...patch } } });
  assert.throws(() => head({ weights: ['x', 'y'] }), /non-numeric weights or bias/);
  assert.throws(() => head({ bias: undefined }), /non-numeric weights or bias/);
  assert.throws(() => head({ calibration: { method: 'platt', a: 'NaN', c: 0 } }), /Platt/);
  assert.throws(() => head({ calibration: { method: 'isotonic', x: [], y: [] } }), /isotonic table/);
  assert.throws(() => head({ calibration: { method: 'isotonic', x: [0.1, 0.2], y: [0] } }), /isotonic table/);
  assert.throws(() => head({ calibration: { method: 'isotonic', x: [0.5, 0.2], y: [0, 1] } }), /not sorted/);
  assert.equal(head({ calibration: { method: 'isotonic', x: [0.2, 0.5], y: [0, 1] } }).version, 'test');
  assert.throws(() => head({ threshold: 0.2, review_floor: 0.9 }), /review_floor <= threshold/);
  assert.throws(() => head({ review_floor: -0.1 }), /review_floor <= threshold/);
  assert.equal(head({ threshold: 1.0000000000000002 }).version, 'test', 'a budget can push the threshold just past 1');
  assert.throws(() => validateArtifact({ ...artifact, embedding: { ...artifact.embedding, dimensions: 0 }, heads: {} }), /positive integer/);
  assert.throws(() => validateArtifact({ ...artifact, heads: [artifact.heads.urgent] }), /not a classifier artifact/);
});

test('artifact validation', () => {
  assert.throws(() => validateArtifact({ ...artifact, embedding: { ...artifact.embedding, dimensions: 3 } }), /2 weights/);
  assert.throws(() => validateArtifact({ ...artifact, heads: { nonsense: artifact.heads.urgent } }, { heads: ['urgent', 'spam', 'off_topic'] }), /unknown head/);
  assert.equal(validateArtifact({ ...artifact, heads: { nonsense: artifact.heads.urgent } }).version, 'test', 'any head allowed without a list');
  assert.throws(() => validateArtifact({ ...artifact, heads: { urgent: { ...artifact.heads.urgent, threshold: null } } }), /non-numeric/);
  assert.throws(() => validateArtifact({ hello: 1 }), /not a classifier artifact/);
  assert.throws(() => validateArtifact({ ...artifact, heads: { urgent: { ...artifact.heads.urgent!, thresholds: { checkin: 'x' } } } }), /non-numeric checkin threshold/);
  assert.equal(validateArtifact({ ...artifact, heads: { urgent: { ...artifact.heads.urgent!, thresholds: { checkin: 0.2 } } } }).heads.urgent!.thresholds!.checkin, 0.2);
  assert.equal(validateArtifact(artifact).version, 'test');
});

test('lifecycle: shadow prefers a candidate, enforce only loads a promoted artifact that passed its gates', () => {
  assert.deepEqual(loadOrder('shadow', { promoted: 'p', shadowCandidate: 'c' }).map((c) => c.role), ['shadow-candidate', 'promoted']);
  assert.deepEqual(loadOrder('shadow', { promoted: 'p', shadowCandidate: null }).map((c) => c.location), ['p']);
  assert.deepEqual(loadOrder('enforce', { promoted: 'p', shadowCandidate: 'c' }).map((c) => c.location), ['p']);
  assert.equal(refuseToServe('enforce', { gates: { passed: true, failures: [] } }), null);
  assert.match(refuseToServe('enforce', { gates: { passed: false, failures: ['x'] } })!, /gates/);
  assert.match(refuseToServe('enforce', {})!, /gates/, 'no recorded gates is not a pass');
  assert.equal(refuseToServe('shadow', {}), null);
});

test('validateArtifact refuses isotonic tables that are not probabilities; decide refuses scores outside [0, 1]', async () => {
  const { validateArtifact, decide } = await import('../src/index.ts');
  const head = (x: number[], y: number[]) => ({ version: 'v', created_at: '', embedding: { model_id: 't', dimensions: 1, normalize: false }, heads: { h: { weights: [1], bias: 0, calibration: { method: 'isotonic' as const, x, y }, threshold: 0.9, review_floor: 0.5 } } });
  assert.throws(() => validateArtifact(head([0, 1], [2, 3])), /outside \[0, 1\]/);
  assert.throws(() => validateArtifact(head([0, 1], [0.6, 0.2])), /decrease/);
  assert.throws(() => validateArtifact(head([-1, 1], [0.1, 0.2])), /outside \[0, 1\]/);
  validateArtifact(head([0, 0.5, 1], [0, 0.3, 0.9]));
  assert.throws(() => decide(head([0, 1], [0, 1]), { h: 2.7 }, { priority: ['h'] }), /outside \[0, 1\]/);
});
