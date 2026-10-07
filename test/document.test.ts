import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createPipeline, fixtureTokenizer, type TextDocument } from '@liquidau/text-preprocessing';

import {
  buildDocumentArtifact, DocumentError, documentFeatureIdentity, prepareDocumentFeatures, scoreDocument, scoreDocumentEmbeddings, scoreEmbedding, trainDocumentHeads,
  validateArtifact, validateDocumentArtifact, type DocumentEncoder, type FullEncoderIdentity, type HeadPolicy, type Split,
} from '../src/index.ts';

const tokenizer = fixtureTokenizer();
const IDENTITY: FullEncoderIdentity = {
  schema: 'liquidau-encoder/1', modelId: 'fake/doc-model', revision: 'sha256:abc', dimensions: 4, precision: 'fp32', inputType: null, layers: null, pooling: 'mean',
  normalization: 'none', tokenizerRevision: 'fixture-tokenizer/1', maxChars: null, maxTokens: 64, truncation: 'none',
};
const pipeline = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 24 });

/** A deterministic fake chunk encoder: [refund?, noise, urgent?, 1]. Optional per-call delays scramble completion order. */
function fakeEncoder(options: { delay?: (call: number) => number; fail?: (call: number) => boolean; rows?: (texts: readonly string[]) => unknown } = {}): DocumentEncoder & { calls: string[][] } {
  const calls: string[][] = [];
  const vec = (t: string) => {
    let h = 7;
    for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return [/refund/i.test(t) ? 1 : 0, ((h % 1000) / 1000 - 0.5) * 0.6, /urgent/i.test(t) ? 1 : 0, 1];
  };
  return {
    identity: IDENTITY, calls,
    async embed(texts, { signal }) {
      const call = calls.push([...texts]) - 1;
      const ms = options.delay?.(call) ?? 0;
      if (ms) await new Promise((r) => setTimeout(r, ms));
      if (signal.aborted) throw new Error('aborted');
      if (options.fail?.(call)) throw Object.assign(new Error('provider 503'), { retryable: true });
      return (options.rows?.(texts) ?? texts.map(vec)) as number[][];
    },
  };
}

/** Long documents whose evidence sentence often sits past a single-input truncation point. */
function corpus(n: number): { docs: TextDocument[]; y: Array<0 | 1>; split: Split[] } {
  const docs: TextDocument[] = [], y: Array<0 | 1> = [], split: Split[] = [];
  for (let i = 0; i < n; i++) {
    const positive = i % 3 === 0;
    const filler = Array.from({ length: 2 + (i % 7) }, (_, k) => `Filler sentence number ${k} about the weather today.`);
    if (positive) filler.splice((i * 7) % filler.length, 0, 'Please refund my order.');
    docs.push({ id: `d${i}`, groupId: `g${Math.floor(i / 2)}`, text: filler.join(' ') });
    y.push(positive ? 1 : 0);
    const g = Math.floor(i / 2) % 10;
    split.push(g < 6 ? 'train' : g < 8 ? 'calibration' : 'test');
  }
  return { docs, y, split };
}

const precision: HeadPolicy = { kind: 'precision', mode: 'heuristic', targetPrecision: 0.8 };

test('pooling is the plain mean of chunk vectors, not renormalised, checked against hand-computed values', () => {
  const artifactLike = { encoderIdentity: { ...IDENTITY, normalization: 'unit' as const }, pipeline };
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.6, 0.8]];
  const tiny = { schema: 'liquidau-document-classifier/1' as const, version: 't', createdAt: '2026-10-07T00:00:00Z', ...artifactLike, featureIdentitySha256: '', classifier: {
    version: 't', created_at: '', embedding: { model_id: 'x', dimensions: 4, normalize: false },
    heads: { h: { weights: [3, 0, 0, 0], bias: -1, calibration: { method: 'platt' as const, a: 1, c: 0 }, threshold: 0.5, review_floor: 0.2 } },
  } };
  // mean = [1/3, 1/3, 0.2, 0.8667] (norm ~0.98, not 1); logit = 3 * 1/3 - 1 = 0 -> p = 0.5
  assert.equal(scoreDocumentEmbeddings(tiny, v).h, 0.5);
  assert.throws(() => scoreDocumentEmbeddings(tiny, []), (e: unknown) => e instanceof DocumentError && e.code === 'INVALID_EMBEDDING');
  assert.throws(() => scoreDocumentEmbeddings(tiny, [[2, 0, 0, 0]]), /unit length/);
  assert.throws(() => scoreDocumentEmbeddings(tiny, [[1, 0, 0]]), /width 3/);
  assert.throws(() => scoreDocumentEmbeddings(tiny, Array.from({ length: 65 }, () => v[0])), /exceed/);
});

test('training is per document, serving reproduces training features exactly, and batching never changes outputs', async () => {
  const { docs, y, split } = corpus(240);
  const encoder = fakeEncoder();
  const features = await prepareDocumentFeatures(docs, pipeline, { tokenizer, encoder });
  assert.equal(features.rows.length, docs.length, 'one row per document, whatever its chunk count');
  assert.ok(features.documents.some((d) => d.chunkCount > 2));
  assert.equal(JSON.stringify(features.documents).includes('weather'), false, 'records carry no raw text');

  const result = trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline, split, heads: [{ name: 'refund', y, prevalence: 1 / 3, policy: precision }] });
  assert.deepEqual(result.failures, [], result.failures.join('; '));
  assert.equal(result.document.documents, 240);
  const artifact = buildDocumentArtifact(result, { version: 'doc-1', createdAt: '2026-10-07T00:00:00Z', router: { ruleSetHash: 'a'.repeat(64) } });
  const stored = validateDocumentArtifact(JSON.parse(JSON.stringify(artifact)));
  assert.equal(stored.classifier.embedding.normalize, false);
  assert.equal(stored.featureIdentitySha256, documentFeatureIdentity(IDENTITY, pipeline));

  for (const i of [0, 1, 5, 99]) {
    const served = await scoreDocument(stored, docs[i].text, { tokenizer, encoder });
    assert.deepEqual(served.scores, scoreEmbedding(stored.classifier, features.rows[i]), 'training/serving parity');
    assert.equal(served.chunkCount, features.documents[i].chunkCount);
  }
  // Batch size, concurrency and completion order are operational only.
  const long = docs.find((d) => d.text.length > 300)!.text;
  const reference = (await scoreDocument(stored, long, { tokenizer, encoder })).scores;
  for (const [batchSize, concurrency] of [[1, 1], [1, 4], [2, 3], [256, 16]]) {
    const scrambled = fakeEncoder({ delay: (c) => (c % 2 ? 1 : 5) });
    assert.deepEqual((await scoreDocument(stored, long, { tokenizer, encoder: scrambled }, { batchSize, concurrency })).scores, reference, `batch ${batchSize} x ${concurrency}`);
  }
});

test('failures are explicit and no partial result is produced', async () => {
  const { docs, y, split } = corpus(120);
  const features = await prepareDocumentFeatures(docs, pipeline, { tokenizer, encoder: fakeEncoder() });
  const artifact = buildDocumentArtifact(trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline, split, heads: [{ name: 'refund', y, prevalence: 1 / 3, policy: precision }] }), { version: 'v' });
  const long = docs.find((d) => d.text.length > 300)!.text;
  const code = (c: string, retryable?: boolean) => (e: unknown) => e instanceof DocumentError && e.code === c && (retryable === undefined || e.retryable === retryable);
  await assert.rejects(scoreDocument(artifact, long, { tokenizer, encoder: fakeEncoder({ fail: (c) => c === 1 }) }, { batchSize: 1 }), code('ENCODER_FAILURE', true));
  await assert.rejects(scoreDocument(artifact, long, { tokenizer, encoder: fakeEncoder({ rows: (t) => t.slice(1).map(() => [0, 0, 0, 1]) }) }), code('INVALID_EMBEDDING'));
  await assert.rejects(scoreDocument(artifact, long, { tokenizer, encoder: fakeEncoder({ rows: (t) => t.map(() => [0, Number.NaN, 0, 1]) }) }), code('INVALID_EMBEDDING'));
  await assert.rejects(scoreDocument(artifact, '  \n ', { tokenizer, encoder: fakeEncoder() }), code('INVALID_INPUT'));
  await assert.rejects(scoreDocument(artifact, 'x'.repeat(10), { tokenizer, encoder: fakeEncoder() }, { maxInputUtf8Bytes: 5 }), code('INPUT_LIMIT_EXCEEDED'));
  await assert.rejects(scoreDocument(artifact, 'word. '.repeat(500), { tokenizer, encoder: fakeEncoder() }), code('INPUT_LIMIT_EXCEEDED'), 'more chunks than maxChunks');
  await assert.rejects(scoreDocument(artifact, long, { tokenizer: fixtureTokenizer({ revision: 'fixture-tokenizer/1', special: 3 }), encoder: fakeEncoder() }), code('IDENTITY_MISMATCH'));
  await assert.rejects(scoreDocument(artifact, long, { tokenizer, encoder: { ...fakeEncoder(), identity: { ...IDENTITY, revision: 'other' } } }), code('IDENTITY_MISMATCH'));
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 2);
  await assert.rejects(scoreDocument(artifact, long, { tokenizer, encoder: { identity: IDENTITY, embed: () => new Promise(() => {}) } }, { signal: controller.signal }), code('ABORTED'));
});

test('training input: feature identity, group partitions and head types are enforced', async () => {
  const { docs, y, split } = corpus(60);
  const features = await prepareDocumentFeatures(docs, pipeline, { tokenizer, encoder: fakeEncoder() });
  const heads = [{ name: 'refund' as const, y, prevalence: 1 / 3, policy: precision }];
  assert.throws(() => trainDocumentHeads({ features: { ...features, featureIdentitySha256: 'f'.repeat(64) }, encoderIdentity: IDENTITY, pipeline, split, heads }), /feature identity/);
  assert.throws(() => trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline: { ...pipeline, maxInputTokens: 25 }, split, heads }), /feature identity/, 'features from another pipeline');
  const leaky = [...split];
  leaky[1] = leaky[0] === 'train' ? 'test' : 'train';
  assert.throws(() => trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline, split: leaky, heads }), /group g0 spans/);
  assert.throws(() => trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline, split, heads: [{ ...heads[0], type: 'knn' as never }] }), /linear only/);
});

test('artifacts: legacy loaders refuse the envelope; every identity change is detected', async () => {
  const { docs, y, split } = corpus(120);
  const features = await prepareDocumentFeatures(docs, pipeline, { tokenizer, encoder: fakeEncoder() });
  const artifact = buildDocumentArtifact(trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline, split, heads: [{ name: 'refund', y, prevalence: 1 / 3, policy: precision }] }), { version: 'v' });
  assert.throws(() => validateArtifact(JSON.parse(JSON.stringify(artifact))), /not a classifier artifact/, 'the legacy loader rejects a document envelope');
  const edits: Array<[string, (a: Record<string, any>) => void, RegExp]> = [
    ['schema', (a) => { a.schema = 'liquidau-document-classifier/2'; }, /unsupported document artifact schema/],
    ['pipeline limit', (a) => { a.pipeline.maxChunks = 65; }, /featureIdentitySha256/],
    ['encoder revision', (a) => { a.encoderIdentity.revision = 'sha256:def'; }, /featureIdentitySha256|embedding must describe/],
    ['truncating encoder', (a) => { a.encoderIdentity.truncation = 'liquidau-truncate-utf16/1'; }, /truncat/],
    ['budget over the encoder limit', (a) => { a.encoderIdentity.maxTokens = 16; }, /exceeds the encoder/],
    ['provider-managed tokenizer', (a) => { a.encoderIdentity.tokenizerRevision = 'provider-managed'; }, /provider-managed/],
    ['a knn head', (a) => { a.classifier.heads.refund.features = { kind: 'knn', k: 1 }; }, /linear only|weights|knn head/],
    ['a sentence-embedding spec', (a) => { a.classifier.embedding.normalize = true; }, /pooled document features/],
    ['an unknown field', (a) => { a.chunking = {}; }, /unknown field chunking/],
    ['an incomplete semantic grouping', (a) => { a.pipeline.grouping = { algorithm: 'adjacent-cosine/1', threshold: 0.5 }; }, /boundaryEncoderIdentity/],
  ];
  for (const [name, edit, pattern] of edits) {
    const copy = JSON.parse(JSON.stringify(artifact));
    edit(copy);
    assert.throws(() => validateDocumentArtifact(copy), pattern, name);
  }
});

test('semantic grouping: same scorer for training and serving; vectors reused only under identical identities', async () => {
  const semantic = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 24, semantic: { threshold: 0.999, boundaryEncoderIdentity: IDENTITY, boundaryTokenizer: tokenizer.identity, boundaryMaxInputTokens: 24 } });
  const text = 'Please refund my order. The weather is fine. Please refund it now. Filler again here.';
  const shared = fakeEncoder();
  const doc = { id: 'x', groupId: 'g', text };
  const f = await prepareDocumentFeatures([doc], semantic, { tokenizer, encoder: shared, boundaryEncoder: shared, boundaryTokenizer: tokenizer });
  const sentTexts = shared.calls.flat();
  assert.equal(new Set(sentTexts).size, sentTexts.length, 'every distinct prepared input is embedded once (chunk vectors reuse identical boundary inputs)');
  assert.ok(f.documents[0].chunkCount >= 2);

  // A distinct boundary identity: nothing is shared, even for identical text.
  const boundaryId = { ...IDENTITY, modelId: 'fake/boundary' };
  const separate = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 24, semantic: { threshold: 0.999, boundaryEncoderIdentity: boundaryId, boundaryTokenizer: tokenizer.identity, boundaryMaxInputTokens: 24 } });
  const cls = fakeEncoder(), bnd = { ...fakeEncoder(), identity: boundaryId };
  await prepareDocumentFeatures([doc], separate, { tokenizer, encoder: cls, boundaryEncoder: bnd, boundaryTokenizer: tokenizer });
  assert.ok(cls.calls.flat().length >= 2 && bnd.calls.flat().length >= 4, 'both encoders were called');
  assert.notEqual(documentFeatureIdentity(IDENTITY, semantic), documentFeatureIdentity(IDENTITY, separate), 'the boundary identity is part of the feature identity');

  await assert.rejects(prepareDocumentFeatures([doc], semantic, { tokenizer, encoder: shared }), (e: unknown) => e instanceof DocumentError && e.code === 'IDENTITY_MISMATCH');
  await assert.rejects(prepareDocumentFeatures([doc], semantic, { tokenizer, encoder: shared, boundaryEncoder: { ...shared, identity: boundaryId }, boundaryTokenizer: tokenizer }), (e: unknown) => e instanceof DocumentError && e.code === 'IDENTITY_MISMATCH');
});

test('usage counts real encoder calls per stage; planning time excludes boundary embedding (review T3)', async () => {
  const semantic = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 24, semantic: { threshold: 0.999, boundaryEncoderIdentity: IDENTITY, boundaryTokenizer: tokenizer.identity, boundaryMaxInputTokens: 24 } });
  const text = 'Please refund my order. The weather is fine. Please refund it now. Filler again here. One more line. And the last.';
  const { docs, y, split } = corpus(120);
  const f = await prepareDocumentFeatures(docs, semantic, { tokenizer, encoder: fakeEncoder(), boundaryEncoder: fakeEncoder(), boundaryTokenizer: tokenizer });
  const artifact = buildDocumentArtifact(trainDocumentHeads({ features: f, encoderIdentity: IDENTITY, pipeline: semantic, split, heads: [{ name: 'refund', y, prevalence: 1 / 3, policy: precision }] }), { version: 'v' });
  for (const batchSize of [1, 3, 16]) {
    const enc = fakeEncoder();
    const r = await scoreDocument(artifact, text, { tokenizer, encoder: enc, boundaryEncoder: enc, boundaryTokenizer: tokenizer }, { batchSize });
    assert.equal(r.embedCalls + r.boundary.calls, enc.calls.length, `batch ${batchSize}: every real call counted once, memo hits not at all`);
    assert.equal(r.classification.texts + r.boundary.texts, enc.calls.flat().length);
    assert.ok(r.boundary.inputTokens > 0 && r.boundary.calls > 0);
  }
  const plain = await scoreDocument(artifact, text, { tokenizer, encoder: fakeEncoder({ delay: () => 30 }), boundaryEncoder: fakeEncoder({ delay: () => 30 }), boundaryTokenizer: tokenizer }, { batchSize: 1, concurrency: 1 });
  assert.ok(plain.boundary.ms >= 30 && plain.timings.planningMs < plain.boundary.ms, 'boundary embedding is reported separately from planning');
  // Overlapping calls count once: with concurrency, stage time is wall time, so planning time stays non-negative.
  const overlapped = await scoreDocument(artifact, text, { tokenizer, encoder: fakeEncoder({ delay: () => 30 }), boundaryEncoder: fakeEncoder({ delay: () => 30 }), boundaryTokenizer: tokenizer }, { batchSize: 1, concurrency: 4 });
  assert.ok(overlapped.timings.planningMs >= 0, `planning ${overlapped.timings.planningMs} ms`);
  assert.ok(overlapped.boundary.ms < 30 * overlapped.boundary.calls, 'concurrent calls are not summed');
});

test('a deadline holds between yields; the embedded classifier record must match the envelope (review T4, T7)', async () => {
  const { docs, y, split } = corpus(120);
  const features = await prepareDocumentFeatures(docs, pipeline, { tokenizer, encoder: fakeEncoder() });
  const artifact = buildDocumentArtifact(trainDocumentHeads({ features, encoderIdentity: IDENTITY, pipeline, split, heads: [{ name: 'refund', y, prevalence: 1 / 3, policy: precision }] }), { version: 'v' });
  const slowTok = { identity: tokenizer.identity, countInput: (t: string) => { const end = performance.now() + 15; while (performance.now() < end) { /* busy */ } return tokenizer.countInput(t); }, cutOffsets: tokenizer.cutOffsets };
  await assert.rejects(scoreDocument(artifact, docs[3].text, { tokenizer: slowTok, encoder: fakeEncoder() }, { deadline: performance.now() + 10, yieldEvery: 1_000_000 }), (e: unknown) => e instanceof DocumentError && e.code === 'DEADLINE_EXCEEDED');
  const copy = JSON.parse(JSON.stringify(artifact));
  copy.classifier.training.document.feature_identity = 'f'.repeat(64);
  assert.throws(() => validateDocumentArtifact(copy), /feature_identity/);
  const chunk = JSON.parse(JSON.stringify(artifact));
  chunk.classifier.training.document.unit = 'chunk';
  assert.throws(() => validateDocumentArtifact(chunk), /unit must be 'document'/);
});
