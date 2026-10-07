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
import { fullEncoderIdentityProblems, vectorProblems, type ChunkPlan, type DocumentPipeline, type EncoderAdapter, type FullEncoderIdentity, type TextDocument, type TokenizerAdapter } from '@liquidau/text-preprocessing';
import { type ClassifierArtifact, type EmbeddingSpec, type RouterTraining, type Scores } from './artifact.ts';
import { type HeadInput, type TrainInput, type TrainResult } from './train.ts';
export declare const DOCUMENT_ARTIFACT_SCHEMA = "liquidau-document-classifier/1";
export declare const DOCUMENT_FEATURES_SCHEMA = "liquidau-document-features/1";
export declare const FULL_ENCODER_IDENTITY_SCHEMA = "liquidau-encoder/1";
/**
 * The full chunk-encoder identity and its validator come from text-preprocessing (field-for-field the
 * router's EncoderIdentity; conformance fixtures in the router keep them in step).
 */
export { fullEncoderIdentityProblems, type FullEncoderIdentity };
/**
 * Why this encoder can't serve this pipeline (empty when it can): document pipelines forbid silent
 * truncation, need a known token limit at or above the chunk budget, and an exact tokenizer whose
 * revision is the encoder's.
 */
export declare function documentCompatibilityProblems(identity: FullEncoderIdentity, pipeline: DocumentPipeline): string[];
/** The document feature identity: the full chunk-encoder identity AND the complete pipeline. */
export declare function documentFeatureIdentity(identity: FullEncoderIdentity, pipeline: DocumentPipeline): string;
/** The embedded classifier's embedding spec: the pooled document feature space, not a sentence embedding. */
export declare function documentFeatureEmbedding(identity: FullEncoderIdentity): EmbeddingSpec;
/** Structurally compatible with @liquidau/router's Encoder. */
export type DocumentEncoder = EncoderAdapter;
export interface DocumentAdapters {
    tokenizer: TokenizerAdapter;
    encoder: DocumentEncoder;
    /** 'adjacent-cosine/1' pipelines only: the pipeline's boundary encoder and boundary tokenizer. */
    boundaryEncoder?: DocumentEncoder;
    boundaryTokenizer?: TokenizerAdapter;
}
export interface DocumentOptions {
    signal?: AbortSignal;
    /** Chunks per encoder call (default 16, at most 256). Operational: never changes outputs. */
    batchSize?: number;
    /** Encoder calls in flight (default 2, at most 16). Operational: never changes outputs. */
    concurrency?: number;
    /** Host limit on the document's UTF-8 size. */
    maxInputUtf8Bytes?: number;
    /**
     * An absolute deadline on performance.now()'s clock, checked at every planning step and around every
     * encoder call (DEADLINE_EXCEEDED), so it holds even when a timer-driven abort can't fire.
     */
    deadline?: number;
    /** Planner yields to the event loop every this many tokenizer calls (default 256). */
    yieldEvery?: number;
}
export declare const DOCUMENT_ERROR_CODES: readonly ["INVALID_INPUT", "PREPROCESSING_FAILURE", "INPUT_LIMIT_EXCEEDED", "IDENTITY_MISMATCH", "ENCODER_FAILURE", "INVALID_EMBEDDING", "INVALID_SCORE", "ABORTED", "DEADLINE_EXCEEDED"];
export type DocumentErrorCode = (typeof DOCUMENT_ERROR_CODES)[number];
/** A document could not be featurised or scored. No partial result exists. */
export declare class DocumentError extends Error {
    readonly name = "DocumentError";
    readonly code: DocumentErrorCode;
    /** For ENCODER_FAILURE: the encoder error's own `retryable`, when it states one. */
    readonly retryable: boolean | undefined;
    constructor(code: DocumentErrorCode, message: string, options?: {
        retryable?: boolean;
        cause?: unknown;
    });
}
/** Problems with one chunk vector for an identity: width, finiteness and the declared normalisation. */
export declare const chunkVectorProblems: typeof vectorProblems;
/** The mean of chunk vectors, summed in chunk-index order. Validates every input and the result. */
export declare function poolChunkVectors(identity: FullEncoderIdentity, vectors: ReadonlyArray<ArrayLike<number>>, maxChunks: number): number[];
/** Encoder work actually performed (memo hits are not calls), split by stage. */
export interface EncoderUsage {
    calls: number;
    texts: number;
    /** Exact input tokens of the texts actually sent, under the stage's tokenizer. */
    inputTokens: number;
    /** Wall-clock time with at least one of the stage's calls in flight (overlapping calls count once). */
    ms: number;
}
export interface DocumentVector {
    vector: number[];
    plan: ChunkPlan;
    /** Classification-stage encoder calls actually made. */
    embedCalls: number;
    classification: EncoderUsage;
    /** Semantic pipelines: boundary-stage encoder work (zero otherwise). */
    boundary: EncoderUsage;
    /** planningMs excludes boundary embedding time (reported in boundary.ms). */
    timings: {
        planningMs: number;
        embeddingMs: number;
    };
}
export interface DocumentFeatureRecord {
    id: string;
    groupId: string;
    authorId?: string;
    sourceDigest: string;
    planDigest: string;
    chunkCount: number;
    inputTokens: number;
    featureIdentitySha256: string;
}
export interface DocumentFeatures {
    featureIdentitySha256: string;
    /** One pooled row per document. */
    rows: number[][];
    documents: DocumentFeatureRecord[];
}
/** Pooled document rows for training/evaluation, built by exactly the serving path. Raw text is not retained. */
export declare function prepareDocumentFeatures(documents: readonly TextDocument[], pipelineRaw: DocumentPipeline, adapters: DocumentAdapters, options?: DocumentOptions): Promise<DocumentFeatures>;
export type DocumentHeadInput<H extends string> = Omit<HeadInput<H>, 'type'> & {
    type?: 'linear';
};
export type DocumentTrainInput<H extends string> = Omit<TrainInput<H>, 'X' | 'groups' | 'foldKeys' | 'background' | 'heads'> & {
    features: DocumentFeatures;
    encoderIdentity: FullEncoderIdentity;
    pipeline: DocumentPipeline;
    heads: ReadonlyArray<DocumentHeadInput<H>>;
    /** Background traffic for threshold budgets, featurised under the same identity. */
    background?: {
        features: DocumentFeatures;
        maxRate: Partial<Record<H, number>>;
        records?: NonNullable<TrainInput<H>['background']>['records'];
    };
};
export type DocumentTrainResult<H extends string> = TrainResult<H> & {
    document: {
        featureIdentitySha256: string;
        encoderIdentity: FullEncoderIdentity;
        pipeline: DocumentPipeline;
        documents: number;
        groups: number;
    };
};
/**
 * Trains linear heads on document rows (one row and one weight per document, whatever its chunk
 * count), through trainHeads. Every member of a group must sit in one split. Only linear heads.
 */
export declare function trainDocumentHeads<H extends string>(input: DocumentTrainInput<H>): DocumentTrainResult<H>;
export interface DocumentClassifierArtifact<H extends string = string> {
    schema: typeof DOCUMENT_ARTIFACT_SCHEMA;
    version: string;
    createdAt: string;
    /** The chunk encoder. */
    encoderIdentity: FullEncoderIdentity;
    pipeline: DocumentPipeline;
    featureIdentitySha256: string;
    /** Linear heads over pooled DOCUMENT features. */
    classifier: ClassifierArtifact<H>;
}
export declare function buildDocumentArtifact<H extends string>(result: DocumentTrainResult<H>, options: {
    version: string;
    createdAt?: string;
    training?: Record<string, unknown>;
    router?: RouterTraining;
}): DocumentClassifierArtifact<H>;
/** Validates a document artifact loaded from storage; throws with every reason it is unusable. */
export declare function validateDocumentArtifact<H extends string = string>(raw: unknown): DocumentClassifierArtifact<H>;
/** True for a document envelope (by schema), so a loader can dispatch without guessing. */
export declare const isDocumentArtifact: (raw: unknown) => boolean;
/**
 * Scores precomputed chunk vectors: validates a nonempty, bounded list, pools it and scores the
 * embedded model. Vectors alone can't prove provenance: callers must take them from the declared encoder.
 */
export declare function scoreDocumentEmbeddings<H extends string>(artifact: DocumentClassifierArtifact<H>, chunkVectors: ReadonlyArray<ArrayLike<number>>): Scores<H>;
export interface DocumentScoreResult<H extends string = string> {
    scores: Scores<H>;
    chunkCount: number;
    /** Encoder input tokens over all chunks. */
    inputTokens: number;
    planDigest: string;
    featureIdentitySha256: string;
    /** Classification-stage encoder calls actually made (memo hits excluded). */
    embedCalls: number;
    classification: EncoderUsage;
    /** Semantic pipelines: the boundary stage's encoder work. */
    boundary: EncoderUsage;
    /** planningMs excludes boundary embedding (boundary.ms). */
    timings: {
        planningMs: number;
        embeddingMs: number;
        scoringMs: number;
    };
}
/**
 * Plans, embeds, pools and scores one document with a validated artifact. Throws DocumentError; never
 * returns scores from an incomplete set of chunk vectors.
 */
export declare function scoreDocument<H extends string>(artifact: DocumentClassifierArtifact<H>, text: string, adapters: DocumentAdapters, options?: DocumentOptions): Promise<DocumentScoreResult<H>>;
