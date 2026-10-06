/**
 * Embedding with an on-disk cache, so re-training after a label fix doesn't re-embed (and re-pay
 * for) the whole dataset. Provider-agnostic: pass any `embed(text)` function. Node-only (uses fs),
 * hence its own entry point: `@liquidau/embedding-classifier/embedder`. For training; at serving
 * time call the model directly on `truncateText(text, maxChars)` from the package root.
 *
 * Cache, format 'json' (default): one JSON file mapping sha256(truncated text) -> vector. Use one file
 * per model + dimension. Saves are atomic (write a temporary file, then rename), so a process reading
 * the cache never sees a half-written file. Two processes WRITING the same cache can still drop each
 * other's newest entries (last save wins) - run them one after another. Each save rewrites the whole
 * file, so checkpoints slow down as the cache grows, and a cache past ~512 MB (about 150,000 vectors
 * at 768 dimensions) can't be saved at all: V8 can't build the string (FINDINGS O7).
 *
 * Format 'binary': two append-only files, `${cachePath}.f32` (float32 rows) and `${cachePath}.keys`
 * (one key per line). A checkpoint appends only the new vectors, so it costs the same at any size, and
 * vectors stay float32 in memory. Requires `dimensions`. Vectors are written before their keys; on
 * load, anything after the last complete, matched entry (an interrupted append) is trimmed from both
 * files. Float32 is lossless for float32 models (local transformers.js models); providers that return
 * doubles lose about 1e-7 relative precision. One writer at a time, as with 'json'. Prefer it beyond
 * ~50,000 vectors.
 *
 * The key is the truncated text only, so nothing stops two models sharing a file. Pass
 * `dimensions` to make a cache written by a different-sized model fail loudly on load.
 */
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, statSync, truncateSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { truncateText } from "./text.js";
// Also exported from the package root, which serving code should prefer (it imports no Node built-ins).
export { truncateText };
const sha256 = (s) => createHash('sha256').update(s, 'utf8').digest();
export class CachedEmbedder {
    cachePath;
    options;
    cache;
    /** binary: keys embedded since the last checkpoint, in order. */
    pending = [];
    constructor(options) {
        if (!options.embed && !options.embedBatch)
            throw new Error('CachedEmbedder needs embed or embedBatch');
        this.options = options;
        this.cachePath = options.cachePath;
        mkdirSync(dirname(options.cachePath), { recursive: true });
        if (options.format === 'binary') {
            if (!(Number.isInteger(options.dimensions) && options.dimensions > 0))
                throw new Error("format 'binary' needs dimensions");
            this.cache = this.loadBinary(options.dimensions);
            return;
        }
        this.cache = existsSync(options.cachePath) ? JSON.parse(readFileSync(options.cachePath, 'utf8')) : {};
        if (options.dimensions !== undefined) {
            for (const vector of Object.values(this.cache))
                this.checkDimensions(vector, `cache ${options.cachePath}`);
        }
    }
    /** Reads the binary cache, trimming anything after the last complete, matched entry from both files. */
    loadBinary(d) {
        const vecPath = `${this.cachePath}.f32`, keyPath = `${this.cachePath}.keys`;
        const cache = {};
        if (!existsSync(vecPath) || !existsSync(keyPath))
            return cache;
        const keys = readFileSync(keyPath, 'utf8').split('\n').filter(Boolean);
        const rows = Math.floor(statSync(vecPath).size / (4 * d));
        const n = Math.min(keys.length, rows);
        if (statSync(vecPath).size !== n * 4 * d)
            truncateSync(vecPath, n * 4 * d);
        if (keys.length !== n)
            writeFileSync(keyPath, keys.slice(0, n).map((k) => `${k}\n`).join(''));
        const buf = readFileSync(vecPath);
        const all = new Float32Array(buf.buffer, buf.byteOffset, n * d);
        for (let k = 0; k < n; k++)
            cache[keys[k]] = all.subarray(k * d, (k + 1) * d);
        return cache;
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
                    batch.forEach(([key], k) => { this.cache[key] = vectors[k]; this.pending.push(key); });
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
        return texts.map((t) => Array.from(this.cache[this.key(t)]));
    }
    save() {
        if (this.options.format === 'binary') {
            if (!this.pending.length)
                return;
            const d = this.options.dimensions;
            const flat = new Float32Array(this.pending.length * d);
            this.pending.forEach((key, k) => flat.set(this.cache[key], k * d));
            // Vectors first, then keys: an interrupted save leaves only unmatched vectors, trimmed on load.
            appendFileSync(`${this.cachePath}.f32`, new Uint8Array(flat.buffer));
            appendFileSync(`${this.cachePath}.keys`, this.pending.map((key) => `${key}\n`).join(''));
            this.pending = [];
            return;
        }
        const tmp = `${this.cachePath}.${process.pid}.tmp`;
        writeFileSync(tmp, JSON.stringify(this.cache, (_, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v)));
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
