import assert from 'node:assert/strict';
import { test } from 'node:test';

import { assertRoundTrip, buildArtifact, checkEmbeddingSpec, featureWidth, scoreEmbedding, trainHeads, validateArtifact, type EmbeddingSpec, type Split } from '../src/index.ts';

/** Messages whose label shows only in the context (e.g. the next turn of the conversation). */
function world(n: number) {
  const X: number[][] = [], y: Array<0 | 1> = [], split: Split[] = [];
  for (let i = 0; i < n; i++) {
    const label = (i % 5 === 0 ? 1 : 0) as 0 | 1;
    const message = [Math.sin(i), Math.cos(i * 1.3), Math.sin(i * 0.7)];
    const context = [label * 2 + Math.sin(i * 2.1) * 0.5, Math.cos(i * 0.9)];
    X.push([...message, ...context]);
    y.push(label);
    split.push((['train', 'calibration', 'test'] as const)[i % 3]);
  }
  return { X, y, split };
}

const embedding: EmbeddingSpec = { model_id: 'msg', dimensions: 3, normalize: false, context: { model_id: 'ctx', dimensions: 2, normalize: false } };

test('an artifact with context features scores the message and context vectors concatenated', () => {
  const { X, y, split } = world(900);
  const result = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.2, policy: { kind: 'recall', targetRecall: 0.9 }, type: 'linear' }] });
  const artifact = buildArtifact(result, { version: '1', embedding });
  assert.equal(featureWidth(artifact.embedding), 5);
  validateArtifact(JSON.parse(JSON.stringify(artifact)));
  assertRoundTrip(artifact, X, result.testProbabilities);
  assert.throws(() => scoreEmbedding(artifact, [0, 0, 0]), /expects 5 \(the message and its context, concatenated\)/);
});

test('context specs are validated, compared and never nested', () => {
  const { X, y, split } = world(300);
  const result = trainHeads({ X, split, heads: [{ name: 'h', y, prevalence: 0.2, policy: { kind: 'recall', targetRecall: 0.9 }, type: 'linear' }] });
  const artifact = JSON.parse(JSON.stringify(buildArtifact(result, { version: '1', embedding })));
  assert.throws(() => validateArtifact({ ...artifact, embedding: { ...embedding, context: { ...embedding.context, dimensions: 0 } } }), /embedding context dimensions/);
  assert.throws(() => validateArtifact({ ...artifact, embedding: { ...embedding, context: { ...embedding.context, context: embedding.context } } }), /without context of its own/);
  assert.throws(() => validateArtifact({ ...artifact, embedding: { ...embedding, context: undefined } }), /weights but the embedding has 3/);
  assert.deepEqual(checkEmbeddingSpec(artifact, embedding), []);
  assert.equal(checkEmbeddingSpec(artifact, { ...embedding, context: { ...embedding.context!, model_id: 'other' } }).length, 1);
  assert.equal(checkEmbeddingSpec(artifact, { model_id: 'msg', dimensions: 3, normalize: false }).length, 1, 'a runtime without the context encoder is a mismatch');
});
