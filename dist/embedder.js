/**
 * Embedding with an on-disk cache, so re-training after a label fix doesn't re-embed (and re-pay
 * for) the whole dataset. Provider-agnostic: pass any `embed(text)` function. Node-only (uses fs),
 * hence its own entry point: `@liquidau/embedding-classifier/embedder`.
 *
 * Cache: one JSON file mapping sha256(truncated text) -> vector. Use one file per model + dimension.
 * Saves are atomic (write a temporary file, then rename), so a process reading the cache never sees
 * a half-written file. Two processes WRITING the same cache can still drop each other's newest
 * entries (last save wins) - run them one after another. Each save rewrites the whole file, so very
 * large caches (hundreds of thousands of vectors) make checkpoints slow; raise checkpointEvery.
 *
 * The key is the truncated text only, so nothing stops two models sharing a file. Pass
 * `dimensions` to make a cache written by a different-sized model fail loudly on load.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
const sha256 = (s) => createHash('sha256').update(s, 'utf8').digest();
/**
 * Truncates to at most maxChars UTF-16 code units without splitting a surrogate pair. Use the same
 * function at runtime so training and serving embed identical text.
 */
export function truncateText(text, maxChars) {
    if (maxChars === undefined || text.length <= maxChars)
        return text;
    const code = text.charCodeAt(maxChars - 1);
    return text.slice(0, code >= 0xd800 && code <= 0xdbff ? maxChars - 1 : maxChars);
}
export class CachedEmbedder {
    cachePath;
    options;
    cache;
    constructor(options) {
        if (!options.embed && !options.embedBatch)
            throw new Error('CachedEmbedder needs embed or embedBatch');
        this.options = options;
        this.cachePath = options.cachePath;
        mkdirSync(dirname(options.cachePath), { recursive: true });
        this.cache = existsSync(options.cachePath) ? JSON.parse(readFileSync(options.cachePath, 'utf8')) : {};
        if (options.dimensions !== undefined) {
            for (const vector of Object.values(this.cache))
                this.checkDimensions(vector, `cache ${options.cachePath}`);
        }
    }
    checkDimensions(vector, source) {
        const { dimensions } = this.options;
        if (dimensions !== undefined && vector.length !== dimensions) {
            throw new Error(`${source} has a ${vector.length}-dimensional vector, expected ${dimensions} (a different model?)`);
        }
    }
    truncate(text) {
        return truncateText(text, this.options.maxChars);
    }
    key(text) {
        return sha256(this.truncate(text)).toString('hex');
    }
    async embedMany(texts) {
        const { concurrency = 16, checkpointEvery = 200, log = console.log } = this.options;
        const missing = [...new Map(texts.filter((t) => !(this.key(t) in this.cache)).map((t) => [this.key(t), t])).entries()];
        if (missing.length) {
            log(`  embedding ${missing.length} new text(s) (${texts.length - missing.length} cached)`);
            const size = this.options.embedBatch ? Math.max(1, this.options.batchSize ?? 96) : 1;
            const batches = [];
            for (let i = 0; i < missing.length; i += size)
                batches.push(missing.slice(i, i + size));
            let next = 0;
            let done = 0;
            let lastCheckpoint = 0;
            const worker = async () => {
                while (next < batches.length) {
                    const batch = batches[next++];
                    const inputs = batch.map(([, text]) => this.truncate(text));
                    const vectors = this.options.embedBatch ? await this.options.embedBatch(inputs) : [await this.options.embed(inputs[0])];
                    if (vectors.length !== batch.length)
                        throw new Error(`embedBatch returned ${vectors.length} vectors for ${batch.length} texts`);
                    vectors.forEach((v) => this.checkDimensions(v, 'the embedding provider'));
                    batch.forEach(([key], k) => { this.cache[key] = vectors[k]; });
                    done += batch.length;
                    if (done - lastCheckpoint >= checkpointEvery) {
                        lastCheckpoint = done;
                        this.save();
                        log(`  embedded ${done}/${missing.length}`);
                    }
                }
            };
            try {
                await Promise.all(Array.from({ length: Math.max(1, concurrency) }, worker));
            }
            finally {
                this.save();
            }
        }
        // Copies, so a caller mutating a vector can't corrupt the cache (or another text's result).
        return texts.map((t) => [...this.cache[this.key(t)]]);
    }
    save() {
        const tmp = `${this.cachePath}.${process.pid}.tmp`;
        writeFileSync(tmp, JSON.stringify(this.cache));
        renameSync(tmp, this.cachePath);
    }
}
/**
 * Deterministic pseudo-embedding from word hashes - ONLY for exercising a pipeline without an
 * embedding provider. Never ship a model trained on these.
 */
export function hashEmbedding(text, dimensions) {
    const vec = new Float64Array(dimensions);
    for (const word of text.toLowerCase().split(/\s+/).filter(Boolean)) {
        const digest = sha256(word);
        vec[digest.readUInt32BE(0) % dimensions] += digest[4] % 2 ? 1 : -1;
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return Array.from(vec, (v) => (norm ? v / norm : v));
}
