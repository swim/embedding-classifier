import { truncateText } from './text.ts';
export { truncateText };
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
    /** 'json' (default) or 'binary' (append-only float32 records; for large caches, needs `dimensions`). See above. */
    format?: 'json' | 'binary';
    /** How long a save waits for another process's lock before failing (default 60 s). */
    lockTimeoutMs?: number;
    log?: (line: string) => void;
}
export declare class CachedEmbedder {
    readonly cachePath: string;
    private readonly options;
    private readonly cache;
    /** binary: keys embedded since the last checkpoint, in order. */
    private pending;
    constructor(options: CachedEmbedderOptions);
    private get lockTimeoutMs();
    private get binPath();
    /** Reads the records file (converting a two-file cache first), trimming a partial last record. */
    private loadBinary;
    /**
     * The earlier two-file layout (`.f32` rows, `.keys` lines) as records, or null. Keeps entries up to
     * the last complete, matched one: a key line cut off by an interrupted write is dropped.
     */
    private readLegacy;
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
