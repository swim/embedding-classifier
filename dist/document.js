/**
 * Document classification over chunked inputs ('liquidau-document-classifier/1').
 *
 *   text --planChunks--> exact, token-bounded chunks (text-preprocessing)
 *        --encoder-----> one validated vector per chunk (batched; order fixed by chunk index)
 *        --mean--------> ONE document feature vector: x[j] = sum_i x_i[j] / chunkCount, summed in chunk
 *                        order, never renormalised
 *        --linear head-> calibrated document probability (the ordinary artifact machinery)
 *
 * Training, calibration, thresholds and evaluation all happen on document rows, so one document is one
 * row and one weight whatever its chunk count. Chunk labels are never inferred, chunk probabilities are
 * never averaged, and only linear heads are supported. Training (prepareDocumentFeatures) and serving
 * (scoreDocument) build the vector with the same function.
 *
 * The envelope pins the full chunk-encoder identity and the pipeline; its feature identity digest binds
 * both. The embedded classifier's `embedding` describes the POOLED feature space (normalize: false),
 * not a sentence embedding. The envelope has no top-level `heads` or `embedding`, so a legacy loader
 * (validateArtifact) refuses it instead of scoring a truncated text.
 */
import { budgetedEncoderProblems, canonicalDigest, fullEncoderIdentityProblems, identityFields, planChunks, planDigest, PreprocessingError, sourceDigest, tokenizerIdentityProblems, validatePipeline, vectorProblems, } from '@liquidau/text-preprocessing';
import { scoreEmbedding, validateArtifact } from "./artifact.js";
import { buildArtifact, trainHeads } from "./train.js";
export const DOCUMENT_ARTIFACT_SCHEMA = 'liquidau-document-classifier/1';
export const DOCUMENT_FEATURES_SCHEMA = 'liquidau-document-features/1';
export const FULL_ENCODER_IDENTITY_SCHEMA = 'liquidau-encoder/1';
/**
 * The full chunk-encoder identity and its validator come from text-preprocessing (field-for-field the
 * router's EncoderIdentity; conformance fixtures in the router keep them in step).
 */
export { fullEncoderIdentityProblems };
/**
 * Why this encoder can't serve this pipeline (empty when it can): document pipelines forbid silent
 * truncation, need a known token limit at or above the chunk budget, and an exact tokenizer whose
 * revision is the encoder's.
 */
export function documentCompatibilityProblems(identity, pipeline) {
    return budgetedEncoderProblems(identity, { maxInputTokens: pipeline.maxInputTokens, tokenizerRevision: pipeline.tokenizer.revision });
}
const identityOf = identityFields;
/** The document feature identity: the full chunk-encoder identity AND the complete pipeline. */
export function documentFeatureIdentity(identity, pipeline) {
    return canonicalDigest({ schema: DOCUMENT_FEATURES_SCHEMA, encoder: identityOf(identity), pipeline: validatePipeline(pipeline) });
}
/** The embedded classifier's embedding spec: the pooled document feature space, not a sentence embedding. */
export function documentFeatureEmbedding(identity) {
    return {
        model_id: `${DOCUMENT_FEATURES_SCHEMA}:mean(${identity.modelId}@${identity.revision})`,
        dimensions: identity.dimensions,
        normalize: false,
        precision: identity.precision,
        ...(identity.inputType !== null ? { input_type: identity.inputType } : {}),
    };
}
export const DOCUMENT_ERROR_CODES = ['INVALID_INPUT', 'PREPROCESSING_FAILURE', 'INPUT_LIMIT_EXCEEDED', 'IDENTITY_MISMATCH', 'ENCODER_FAILURE', 'INVALID_EMBEDDING', 'INVALID_SCORE', 'ABORTED', 'DEADLINE_EXCEEDED'];
/** A document could not be featurised or scored. No partial result exists. */
export class DocumentError extends Error {
    name = 'DocumentError';
    code;
    /** For ENCODER_FAILURE: the encoder error's own `retryable`, when it states one. */
    retryable;
    constructor(code, message, options = {}) {
        super(message, options.cause === undefined ? undefined : { cause: options.cause });
        this.code = code;
        this.retryable = options.retryable;
    }
}
const PLAN_CODES = {
    DEADLINE_EXCEEDED: 'DEADLINE_EXCEEDED', BOUNDARY_CAPABILITY_MISSING: 'IDENTITY_MISMATCH', BOUNDARY_ENCODER_FAILURE: 'ENCODER_FAILURE', INVALID_BOUNDARY_EMBEDDING: 'INVALID_EMBEDDING',
    EMPTY_INPUT: 'INVALID_INPUT', INPUT_TOO_LARGE: 'INPUT_LIMIT_EXCEEDED', TOKEN_BUDGET_UNSATISFIABLE: 'INPUT_LIMIT_EXCEEDED', MAX_CHUNKS_EXCEEDED: 'INPUT_LIMIT_EXCEEDED',
    MAX_PLANNING_STEPS_EXCEEDED: 'INPUT_LIMIT_EXCEEDED', TOKENIZER_FAILURE: 'PREPROCESSING_FAILURE', INVALID_PIPELINE: 'IDENTITY_MISMATCH', TOKENIZER_MISMATCH: 'IDENTITY_MISMATCH', ABORTED: 'ABORTED',
};
const now = () => globalThis.performance?.now() ?? Date.now();
/** Problems with one chunk vector for an identity: width, finiteness and the declared normalisation. */
export const chunkVectorProblems = vectorProblems;
/** The mean of chunk vectors, summed in chunk-index order. Validates every input and the result. */
export function poolChunkVectors(identity, vectors, maxChunks) {
    if (!Array.isArray(vectors) || vectors.length === 0)
        throw new DocumentError('INVALID_EMBEDDING', 'a document needs at least one chunk vector');
    if (vectors.length > maxChunks)
        throw new DocumentError('INVALID_EMBEDDING', `${vectors.length} chunk vectors exceed the pipeline's ${maxChunks} chunks`);
    vectors.forEach((v, i) => { const why = chunkVectorProblems(identity, v); if (why)
        throw new DocumentError('INVALID_EMBEDDING', `chunk vector ${i}: ${why}`); });
    const out = new Array(identity.dimensions).fill(0);
    for (const v of vectors)
        for (let j = 0; j < out.length; j++)
            out[j] += v[j];
    for (let j = 0; j < out.length; j++) {
        out[j] /= vectors.length;
        if (!Number.isFinite(out[j]))
            throw new DocumentError('INVALID_EMBEDDING', `the pooled vector is non-finite at ${j}`);
    }
    return out;
}
function adapterProblems(identity, pipeline, adapters) {
    const p = [];
    if (!adapters?.encoder || typeof adapters.encoder.embed !== 'function')
        return ['adapters.encoder must implement embed(texts, { signal })'];
    if (!adapters.tokenizer || typeof adapters.tokenizer.countInput !== 'function')
        return ['adapters.tokenizer must implement the tokenizer contract'];
    const enc = fullEncoderIdentityProblems(adapters.encoder.identity);
    if (enc.length)
        p.push(...enc.map((e) => `encoder identity ${e}`));
    else if (canonicalDigest(identityOf(adapters.encoder.identity)) !== canonicalDigest(identityOf(identity)))
        p.push('the encoder does not implement the declared encoder identity');
    const tok = tokenizerIdentityProblems(adapters.tokenizer.identity);
    if (tok.length)
        p.push(...tok);
    else if (canonicalDigest(adapters.tokenizer.identity) !== canonicalDigest(pipeline.tokenizer))
        p.push("the tokenizer does not implement the pipeline's tokenizer identity");
    const g = pipeline.grouping;
    if (g.algorithm === 'adjacent-cosine/1') {
        const be = adapters.boundaryEncoder, bt = adapters.boundaryTokenizer;
        if (!be || typeof be.embed !== 'function' || !bt || typeof bt.countInput !== 'function')
            p.push('semantic grouping needs adapters.boundaryEncoder and adapters.boundaryTokenizer');
        else {
            if (fullEncoderIdentityProblems(be.identity).length || canonicalDigest(identityOf(be.identity)) !== canonicalDigest(identityOf(g.boundaryEncoderIdentity)))
                p.push("the boundary encoder does not implement the pipeline's boundary encoder identity");
            if (tokenizerIdentityProblems(bt.identity).length || canonicalDigest(bt.identity) !== canonicalDigest(g.boundaryTokenizer))
                p.push("the boundary tokenizer does not implement the pipeline's boundary tokenizer identity");
        }
    }
    return p;
}
/**
 * Vector reuse between boundary and classification embedding, scoped to one document: only when both
 * encoder identities AND input preparations are exactly equal is the same text the same input. Averaging
 * atom vectors is never substituted for embedding the joined chunk text - only identical texts are reused.
 */
function sharedEncoders(identity, pipeline, adapters) {
    const g = pipeline.grouping;
    if (g.algorithm !== 'adjacent-cosine/1' || !adapters.boundaryEncoder)
        return { classify: adapters.encoder, boundary: undefined };
    const same = canonicalDigest(identityOf(g.boundaryEncoderIdentity)) === canonicalDigest(identityOf(identity)) && canonicalDigest(g.boundaryTokenizer) === canonicalDigest(pipeline.tokenizer);
    if (!same)
        return { classify: adapters.encoder, boundary: adapters.boundaryEncoder };
    const memo = new Map();
    const remember = (enc) => ({
        identity: enc.identity,
        async embed(texts, options) {
            const missing = [...new Set(texts.filter((t) => !memo.has(t)))];
            if (missing.length) {
                const rows = await enc.embed(missing, options);
                if (!Array.isArray(rows) || rows.length !== missing.length)
                    return rows;
                missing.forEach((t, k) => memo.set(t, rows[k]));
            }
            return texts.map((t) => memo.get(t));
        },
    });
    return { classify: remember(adapters.encoder), boundary: remember(adapters.boundaryEncoder) };
}
/** One encoder vector per chunk, in chunk order, or a DocumentError. Batches may finish in any order. */
async function embedChunks(texts, identity, encoder, options) {
    const batchSize = options.batchSize ?? 16, concurrency = options.concurrency ?? 2;
    if (!(Number.isInteger(batchSize) && batchSize >= 1 && batchSize <= 256))
        throw new DocumentError('INVALID_INPUT', 'batchSize must be an integer in [1, 256]');
    if (!(Number.isInteger(concurrency) && concurrency >= 1 && concurrency <= 16))
        throw new DocumentError('INVALID_INPUT', 'concurrency must be an integer in [1, 16]');
    const controller = new AbortController();
    const outer = options.signal;
    let rejectStop;
    const stopped = new Promise((_, reject) => { rejectStop = reject; });
    stopped.catch(() => { });
    const onAbort = () => { controller.abort(); rejectStop(new DocumentError('ABORTED', 'document embedding was aborted')); };
    if (outer?.aborted)
        throw new DocumentError('ABORTED', 'document embedding was aborted');
    outer?.addEventListener('abort', onAbort, { once: true });
    const vectors = new Array(texts.length);
    const starts = Array.from({ length: Math.ceil(texts.length / batchSize) }, (_, b) => b * batchSize);
    let next = 0, calls = 0;
    const worker = async () => {
        while (next < starts.length && !controller.signal.aborted) {
            const start = starts[next++];
            const batch = texts.slice(start, start + batchSize);
            calls++;
            let rows;
            if (options.deadline !== undefined && now() > options.deadline)
                throw new DocumentError('DEADLINE_EXCEEDED', 'document embedding overran its deadline');
            try {
                const call = Promise.resolve().then(() => encoder.embed(batch, { signal: controller.signal }));
                call.catch(() => { });
                rows = await Promise.race([call, stopped]);
            }
            catch (e) {
                if (e instanceof DocumentError)
                    throw e;
                const retryable = typeof e?.retryable === 'boolean' ? e.retryable : undefined;
                throw new DocumentError('ENCODER_FAILURE', `the encoder failed: ${e?.message ?? String(e)}`, { retryable, cause: e });
            }
            if (options.deadline !== undefined && now() > options.deadline)
                throw new DocumentError('DEADLINE_EXCEEDED', 'document embedding overran its deadline');
            if (!Array.isArray(rows) || rows.length !== batch.length)
                throw new DocumentError('INVALID_EMBEDDING', `the encoder returned ${Array.isArray(rows) ? rows.length : typeof rows} rows for ${batch.length} chunks`);
            rows.forEach((r, k) => {
                const why = chunkVectorProblems(identity, r);
                if (why)
                    throw new DocumentError('INVALID_EMBEDDING', `chunk ${start + k}: ${why}`);
                vectors[start + k] = Array.from(r);
            });
        }
    };
    try {
        await Promise.all(Array.from({ length: Math.min(concurrency, starts.length) }, worker));
    }
    catch (e) {
        controller.abort();
        throw e;
    }
    finally {
        outer?.removeEventListener('abort', onAbort);
    }
    if (outer?.aborted)
        throw new DocumentError('ABORTED', 'document embedding was aborted');
    return { vectors, calls };
}
/** Counts the underlying encoder's real calls, texts, tokens and wall-clock time for one stage. */
function metered(enc, usage, tokenizer) {
    let active = 0, since = 0;
    return {
        identity: enc.identity,
        async embed(texts, options) {
            usage.calls++;
            usage.texts += texts.length;
            for (const t of texts)
                usage.inputTokens += tokenizer.countInput(t);
            if (active++ === 0)
                since = now();
            try {
                return await enc.embed(texts, options);
            }
            finally {
                if (--active === 0)
                    usage.ms += now() - since;
            }
        },
    };
}
const usage = () => ({ calls: 0, texts: 0, inputTokens: 0, ms: 0 });
/** The single feature path, shared by training (prepareDocumentFeatures) and serving (scoreDocument). */
async function documentVector(text, identity, pipeline, adapters, options) {
    const t0 = now();
    const classification = usage(), boundary = usage();
    // Meter the real encoders first, then share (memoise) on top: a memo hit never counts as a call.
    const meteredAdapters = {
        ...adapters,
        encoder: metered(adapters.encoder, classification, adapters.tokenizer),
        ...(adapters.boundaryEncoder && adapters.boundaryTokenizer ? { boundaryEncoder: metered(adapters.boundaryEncoder, boundary, adapters.boundaryTokenizer) } : {}),
    };
    const encoders = sharedEncoders(identity, pipeline, meteredAdapters);
    let plan;
    try {
        plan = await planChunks(text, pipeline, adapters.tokenizer, {
            signal: options.signal, contextCharLimit: identity.maxChars, ...(options.maxInputUtf8Bytes !== undefined ? { maxInputUtf8Bytes: options.maxInputUtf8Bytes } : {}),
            ...(options.deadline !== undefined ? { deadline: options.deadline } : {}), ...(options.yieldEvery !== undefined ? { yieldEvery: options.yieldEvery } : {}),
            ...(encoders.boundary ? { boundary: { encoder: encoders.boundary, tokenizer: adapters.boundaryTokenizer, batchSize: options.batchSize, concurrency: options.concurrency } } : {}),
        });
    }
    catch (e) {
        if (e instanceof PreprocessingError)
            throw new DocumentError(PLAN_CODES[e.code] ?? 'PREPROCESSING_FAILURE', e.message, { cause: e, ...(e.retryable !== undefined ? { retryable: e.retryable } : {}) });
        throw new DocumentError('PREPROCESSING_FAILURE', e.message, { cause: e });
    }
    const t1 = now();
    // Boundary embedding happens inside planning; report it separately, not as planning.
    const boundaryDuringPlanning = boundary.ms;
    const { vectors } = await embedChunks(plan.chunks.map((c) => text.slice(c.context.start, c.context.end)), identity, encoders.classify, options);
    const vector = poolChunkVectors(identity, vectors, pipeline.maxChunks);
    return {
        vector, plan, embedCalls: classification.calls, classification: { ...classification }, boundary: { ...boundary },
        timings: { planningMs: t1 - t0 - boundaryDuringPlanning, embeddingMs: now() - t1 },
    };
}
/** Pooled document rows for training/evaluation, built by exactly the serving path. Raw text is not retained. */
export async function prepareDocumentFeatures(documents, pipelineRaw, adapters, options = {}) {
    const pipeline = validatePipeline(pipelineRaw);
    const identity = adapters?.encoder?.identity;
    const problems = [...fullEncoderIdentityProblems(identity), ...(fullEncoderIdentityProblems(identity).length ? [] : documentCompatibilityProblems(identity, pipeline))];
    if (problems.length)
        throw new DocumentError('IDENTITY_MISMATCH', `the encoder can't serve this pipeline: ${problems.join('; ')}`);
    const ap = adapterProblems(identity, pipeline, adapters);
    if (ap.length)
        throw new DocumentError('IDENTITY_MISMATCH', ap.join('; '));
    const featureIdentitySha256 = documentFeatureIdentity(identity, pipeline);
    const ids = new Set();
    const rows = [], records = [];
    for (const d of documents) {
        if (!d || typeof d.id !== 'string' || !d.id || ids.has(d.id))
            throw new DocumentError('INVALID_INPUT', `document ids must be unique non-empty strings (got ${JSON.stringify(d?.id)})`);
        if (typeof d.groupId !== 'string' || !d.groupId)
            throw new DocumentError('INVALID_INPUT', `document ${d.id} needs a groupId`);
        ids.add(d.id);
        const v = await documentVector(d.text, identity, pipeline, adapters, options);
        rows.push(v.vector);
        records.push({
            id: d.id, groupId: d.groupId, ...(d.authorId !== undefined ? { authorId: d.authorId } : {}), sourceDigest: sourceDigest(d.text), planDigest: planDigest(v.plan),
            chunkCount: v.plan.chunks.length, inputTokens: v.plan.chunks.reduce((n, c) => n + c.inputTokens, 0), featureIdentitySha256,
        });
    }
    return { featureIdentitySha256, rows, documents: records };
}
function featureProblems(f, expected, what) {
    const p = [];
    if (!f || !Array.isArray(f.rows) || !Array.isArray(f.documents))
        return [`${what} must be prepared document features`];
    if (f.featureIdentitySha256 !== expected)
        p.push(`${what} were built under feature identity ${f.featureIdentitySha256}, not ${expected}`);
    if (f.rows.length !== f.documents.length)
        p.push(`${what} have ${f.rows.length} rows for ${f.documents.length} documents`);
    if (f.documents.some((d) => d.featureIdentitySha256 !== expected))
        p.push(`some ${what} records carry another feature identity`);
    if (new Set(f.documents.map((d) => d.id)).size !== f.documents.length)
        p.push(`${what} repeat a document id`);
    return p;
}
/**
 * Trains linear heads on document rows (one row and one weight per document, whatever its chunk
 * count), through trainHeads. Every member of a group must sit in one split. Only linear heads.
 */
export function trainDocumentHeads(input) {
    const { features, encoderIdentity, pipeline: pipelineRaw, heads, background, ...rest } = input;
    const pipeline = validatePipeline(pipelineRaw);
    const idp = fullEncoderIdentityProblems(encoderIdentity);
    if (idp.length)
        throw new Error(`invalid encoder identity: ${idp.join('; ')}`);
    const compat = documentCompatibilityProblems(encoderIdentity, pipeline);
    if (compat.length)
        throw new Error(`the encoder can't serve this pipeline: ${compat.join('; ')}`);
    const expected = documentFeatureIdentity(encoderIdentity, pipeline);
    const p = [...featureProblems(features, expected, 'features'), ...(background ? featureProblems(background.features, expected, 'background features') : [])];
    if (rest.split.length !== features.rows.length)
        p.push(`split has ${rest.split.length} entries for ${features.rows.length} documents`);
    const splitOf = new Map();
    features.documents.forEach((d, i) => {
        const s = rest.split[i], prior = splitOf.get(d.groupId);
        if (prior !== undefined && prior !== s)
            p.push(`group ${d.groupId} spans the ${prior} and ${s} splits`);
        splitOf.set(d.groupId, s);
    });
    for (const h of heads)
        if (h.type !== undefined && h.type !== 'linear')
            p.push(`head ${h.name}: document heads are linear only (got ${String(h.type)})`);
    if (p.length)
        throw new Error(`invalid document training input: ${[...new Set(p)].join('; ')}`);
    const result = trainHeads({
        ...rest,
        X: features.rows,
        groups: features.documents.map((d) => d.groupId),
        heads: heads.map((h) => ({ ...h, type: 'linear' })),
        ...(background ? { background: { X: background.features.rows, maxRate: background.maxRate, ...(background.records ? { records: background.records } : {}) } } : {}),
    });
    return { ...result, document: { featureIdentitySha256: expected, encoderIdentity: { ...encoderIdentity }, pipeline, documents: features.rows.length, groups: splitOf.size } };
}
export function buildDocumentArtifact(result, options) {
    const createdAt = options.createdAt ?? new Date().toISOString();
    const { document } = result;
    const classifier = buildArtifact(result, {
        version: options.version, createdAt, embedding: documentFeatureEmbedding(document.encoderIdentity),
        training: { ...(options.training ?? {}), document: { feature_identity: document.featureIdentitySha256, documents: document.documents, groups: document.groups, unit: 'document' } },
        ...(options.router ? { router: options.router } : {}),
    });
    return validateDocumentArtifact({ schema: DOCUMENT_ARTIFACT_SCHEMA, version: options.version, createdAt, encoderIdentity: { ...document.encoderIdentity }, pipeline: document.pipeline, featureIdentitySha256: document.featureIdentitySha256, classifier });
}
/** Validates a document artifact loaded from storage; throws with every reason it is unusable. */
export function validateDocumentArtifact(raw) {
    const a = raw;
    if (!a || typeof a !== 'object' || Array.isArray(a))
        throw new Error('not a document classifier artifact');
    if (a.schema !== DOCUMENT_ARTIFACT_SCHEMA)
        throw new Error(`unsupported document artifact schema ${JSON.stringify(a.schema)} (this library reads '${DOCUMENT_ARTIFACT_SCHEMA}')`);
    const p = [];
    for (const k of Object.keys(a))
        if (!['schema', 'version', 'createdAt', 'encoderIdentity', 'pipeline', 'featureIdentitySha256', 'classifier'].includes(k))
            p.push(`unknown field ${k}`);
    if (typeof a.version !== 'string' || !a.version)
        p.push('version must be a non-empty string');
    if (typeof a.createdAt !== 'string' || !Number.isFinite(Date.parse(a.createdAt)))
        p.push('createdAt must be an ISO 8601 date-time');
    const idp = fullEncoderIdentityProblems(a.encoderIdentity);
    p.push(...idp.map((e) => `encoderIdentity ${e}`));
    let pipeline = null;
    try {
        pipeline = validatePipeline(a.pipeline);
    }
    catch (e) {
        p.push(e.message);
    }
    const identity = a.encoderIdentity;
    if (pipeline && !idp.length) {
        p.push(...documentCompatibilityProblems(identity, pipeline));
        if (a.featureIdentitySha256 !== documentFeatureIdentity(identity, pipeline))
            p.push('featureIdentitySha256 does not match the encoder identity and pipeline');
    }
    let classifier = null;
    try {
        classifier = validateArtifact(a.classifier);
    }
    catch (e) {
        p.push(`classifier: ${e.message}`);
    }
    if (classifier) {
        if (classifier.reference !== undefined)
            p.push('document classifiers are linear only: the classifier must not carry a reference set');
        for (const [h, spec] of Object.entries(classifier.heads))
            if (spec?.features)
                p.push(`head ${h}: document heads are linear only`);
        if (!idp.length && canonicalDigest(classifier.embedding) !== canonicalDigest(documentFeatureEmbedding(identity)))
            p.push("the classifier's embedding must describe the pooled document features of the declared encoder (normalize: false)");
        // The embedded classifier's own record of its feature space must agree with the envelope.
        const rec = classifier.training?.document;
        if (rec !== undefined) {
            if (!rec || typeof rec !== 'object')
                p.push('classifier.training.document must be an object');
            else {
                if (rec.feature_identity !== a.featureIdentitySha256)
                    p.push("classifier.training.document.feature_identity is not the envelope's featureIdentitySha256");
                if (rec.unit !== 'document')
                    p.push("classifier.training.document.unit must be 'document'");
            }
        }
    }
    if (p.length)
        throw new Error(`invalid document artifact: ${p.join('; ')}`);
    return a;
}
/** True for a document envelope (by schema), so a loader can dispatch without guessing. */
export const isDocumentArtifact = (raw) => !!raw && typeof raw === 'object' && raw.schema === DOCUMENT_ARTIFACT_SCHEMA;
// ---------- scoring ----------
/**
 * Scores precomputed chunk vectors: validates a nonempty, bounded list, pools it and scores the
 * embedded model. Vectors alone can't prove provenance: callers must take them from the declared encoder.
 */
export function scoreDocumentEmbeddings(artifact, chunkVectors) {
    const pooled = poolChunkVectors(artifact.encoderIdentity, chunkVectors, artifact.pipeline.maxChunks);
    let scores;
    try {
        scores = scoreEmbedding(artifact.classifier, pooled);
    }
    catch (e) {
        throw new DocumentError('INVALID_SCORE', e.message, { cause: e });
    }
    for (const [h, v] of Object.entries(scores)) {
        if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1)
            throw new DocumentError('INVALID_SCORE', `head ${h} scored ${v}, not a probability`);
    }
    return scores;
}
/**
 * Plans, embeds, pools and scores one document with a validated artifact. Throws DocumentError; never
 * returns scores from an incomplete set of chunk vectors.
 */
export async function scoreDocument(artifact, text, adapters, options = {}) {
    const ap = adapterProblems(artifact.encoderIdentity, artifact.pipeline, adapters);
    if (ap.length)
        throw new DocumentError('IDENTITY_MISMATCH', ap.join('; '));
    const v = await documentVector(text, artifact.encoderIdentity, artifact.pipeline, adapters, options);
    const ts = now();
    let scores;
    try {
        scores = scoreEmbedding(artifact.classifier, v.vector);
    }
    catch (e) {
        throw new DocumentError('INVALID_SCORE', e.message, { cause: e });
    }
    for (const [h, s] of Object.entries(scores)) {
        if (typeof s !== 'number' || !Number.isFinite(s) || s < 0 || s > 1)
            throw new DocumentError('INVALID_SCORE', `head ${h} scored ${s}, not a probability`);
    }
    return {
        scores, chunkCount: v.plan.chunks.length, inputTokens: v.plan.chunks.reduce((n, c) => n + c.inputTokens, 0), planDigest: planDigest(v.plan),
        featureIdentitySha256: artifact.featureIdentitySha256, embedCalls: v.embedCalls, classification: v.classification, boundary: v.boundary, timings: { ...v.timings, scoringMs: now() - ts },
    };
}
