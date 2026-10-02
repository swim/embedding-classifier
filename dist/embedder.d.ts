/**
 * Truncates to at most maxChars UTF-16 code units without splitting a surrogate pair. Use the same
 * function at runtime so training and serving embed identical text.
 */
export declare function truncateText(text: string, maxChars?: number): string;
export interface CachedEmbedderOptions {
    cachePath: string;
    /** One text at a time - or pass embedBatch for providers that take many per call. */
    embed?: (text: string) => Promise<number[]>;
    /** Many texts per call (at most batchSize), vectors in the same order. */
    embedBatch?: (texts: string[]) => Promise<number[][]>;
    batchSize?: number;
    /** Texts are truncated to this before embedding and keying (keep identical to runtime). */
    maxChars?: number;
    /** Expected vector length. Checked on load and on every new embedding. */
    dimensions?: number;
    /** Parallel embed() calls (default 16 - remote embedding APIs are latency-bound). */
    concurrency?: number;
    /** Write the cache every N new embeddings, so an interrupted run keeps what it paid for (default 200). */
    checkpointEvery?: number;
    log?: (line: string) => void;
}
export declare class CachedEmbedder {
    readonly cachePath: string;
    private readonly options;
    private readonly cache;
    constructor(options: CachedEmbedderOptions);
    private checkDimensions;
    private truncate;
    private key;
    embedMany(texts: string[]): Promise<number[][]>;
    private save;
}
/**
 * Deterministic pseudo-embedding from word hashes - ONLY for exercising a pipeline without an
 * embedding provider. Never ship a model trained on these.
 */
export declare function hashEmbedding(text: string, dimensions: number): number[];
