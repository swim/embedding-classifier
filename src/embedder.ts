/**
 * Embedding with an on-disk cache, so re-training after a label fix doesn't re-embed (and re-pay
 * for) the whole dataset. Provider-agnostic: pass any `embed(text)` function. Node-only (uses fs),
 * hence its own entry point: `@liquidau/embedding-classifier/embedder`. For training; at serving
 * time call the model directly on `truncateText(text, maxChars)` from the package root.
 *
 * Several processes may share one cache: every save takes an exclusive lock file
 * (`${cachePath}.lock`; a lock left by a crashed process is broken once its owner is gone or it is
 * older than a minute), so writers never interleave. A process doesn't see entries another process
 * adds after it loaded; it embeds those texts again and appends duplicates, which are harmless.
 *
 * Format 'json' (default): one JSON file mapping sha256(truncated text) -> vector. Each save merges
 * the file's current entries (another writer's) and replaces it atomically (a temporary file, then a
 * rename). Each save rewrites the whole file, so checkpoints slow down as the cache grows, and a cache
 * past ~512 MB (about 150,000 vectors at 768 dimensions) can't be saved at all: V8 can't build the
 * string. Use 'binary' beyond ~50,000 vectors.
 *
 * Format 'binary': one append-only file, `${cachePath}.bin`, of self-describing records - the key's
 * 32 bytes, then the vector as `dimensions` float32 values (platform byte order) - so a key can never
 * be paired with another text's vector. A checkpoint appends only the new records, so it costs the
 * same at any size. Requires `dimensions`. A partial record left by an interrupted append is trimmed
 * (under the lock) before anything else is appended. Vectors are float32 in memory too, so a run and
 * its rerun return identical values; providers that return doubles lose about 1e-7 relative precision.
 * Caches in the earlier two-file layout (`.f32` and `.keys`) are converted on first use and left in
 * place.
 *
 * The key is the truncated text only, so nothing stops two models sharing a file. Pass `dimensions`
 * to make a cache written by a different-sized model fail loudly on load.
 */
import { createHash } from 'node:crypto';
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, renameSync, statSync, truncateSync, unlinkSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

import { truncateText } from './text.ts';

// Also exported from the package root, which serving code should prefer (it imports no Node built-ins).
export { truncateText };

const sha256 = (s: string) => createHash('sha256').update(s, 'utf8').digest();
const KEY_BYTES = 32;

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

const sleepSync = (ms: number) => { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); };

/**
 * Runs `fn` holding `${path}.lock` (created exclusively). A lock whose owner process is gone, or that
 * is older than a minute (saves take well under a second), is broken.
 */
function withLock<T>(path: string, timeoutMs: number, fn: () => T): T {
  const lock = `${path}.lock`;
  const started = Date.now();
  for (;;) {
    try {
      const fd = openSync(lock, 'wx');
      writeSync(fd, `${process.pid}\n`);
      closeSync(fd);
      break;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e;
    }
    try {
      const owner = Number.parseInt(readFileSync(lock, 'utf8'), 10);
      let gone = false;
      if (owner > 0 && owner !== process.pid) {
        try { process.kill(owner, 0); } catch (e) { gone = (e as NodeJS.ErrnoException).code === 'ESRCH'; }
      }
      if (gone || Date.now() - statSync(lock).mtimeMs > 60_000) { unlinkSync(lock); continue; }
    } catch {
      continue; // released while we looked
    }
    if (Date.now() - started > timeoutMs) throw new Error(`the embedding cache ${path} is locked (${lock}); delete the lock file if no other process is writing`);
    sleepSync(5 + Math.random() * 20);
  }
  try {
    return fn();
  } finally {
    try { unlinkSync(lock); } catch { /* already gone */ }
  }
}

export class CachedEmbedder {
  readonly cachePath: string;
  private readonly options: CachedEmbedderOptions;
  private readonly cache: Record<string, ArrayLike<number>>;
  /** binary: keys embedded since the last checkpoint, in order. */
  private pending: string[] = [];

  constructor(options: CachedEmbedderOptions) {
    if (!options.embed && !options.embedBatch) throw new Error('CachedEmbedder needs embed or embedBatch');
    this.options = options;
    this.cachePath = options.cachePath;
    mkdirSync(dirname(options.cachePath), { recursive: true });
    if (options.format === 'binary') {
      if (!(Number.isInteger(options.dimensions) && options.dimensions! > 0)) throw new Error("format 'binary' needs dimensions");
      this.cache = this.loadBinary(options.dimensions!);
      return;
    }
    this.cache = existsSync(options.cachePath) ? JSON.parse(readFileSync(options.cachePath, 'utf8')) : {};
    if (options.dimensions !== undefined) {
      for (const vector of Object.values(this.cache)) this.checkDimensions(vector, `cache ${options.cachePath}`);
    }
  }

  private get lockTimeoutMs(): number {
    return this.options.lockTimeoutMs ?? 60_000;
  }

  private get binPath(): string {
    return `${this.cachePath}.bin`;
  }

  /** Reads the records file (converting a two-file cache first), trimming a partial last record. */
  private loadBinary(d: number): Record<string, ArrayLike<number>> {
    const record = KEY_BYTES + 4 * d;
    const bytes = withLock(this.binPath, this.lockTimeoutMs, () => {
      if (!existsSync(this.binPath)) {
        const legacy = this.readLegacy(d);
        if (!legacy) return null;
        writeFileSync(this.binPath, legacy);
        this.options.log?.(`  converted the embedding cache ${this.cachePath} to ${this.binPath}`);
      }
      const size = statSync(this.binPath).size;
      if (size % record) truncateSync(this.binPath, size - (size % record));
      return readFileSync(this.binPath);
    });
    const cache: Record<string, ArrayLike<number>> = {};
    if (!bytes) return cache;
    // Float32Array views need 4-byte alignment.
    const buf = bytes.byteOffset % 4 ? new Uint8Array(bytes) : bytes;
    for (let off = 0; off + record <= buf.byteLength; off += record) {
      const key = Buffer.from(buf.buffer, buf.byteOffset + off, KEY_BYTES).toString('hex');
      cache[key] = new Float32Array(buf.buffer, buf.byteOffset + off + KEY_BYTES, d);
    }
    return cache;
  }

  /**
   * The earlier two-file layout (`.f32` rows, `.keys` lines) as records, or null. Keeps entries up to
   * the last complete, matched one: a key line cut off by an interrupted write is dropped.
   */
  private readLegacy(d: number): Uint8Array | null {
    const vecPath = `${this.cachePath}.f32`, keyPath = `${this.cachePath}.keys`;
    if (!existsSync(vecPath) || !existsSync(keyPath)) return null;
    const text = readFileSync(keyPath, 'utf8');
    const lines = text.split('\n');
    if (!text.endsWith('\n')) lines.pop();
    const keys = lines.filter((k) => /^[0-9a-f]{64}$/.test(k));
    const vectors = readFileSync(vecPath);
    const n = Math.min(keys.length, Math.floor(vectors.byteLength / (4 * d)));
    const out = new Uint8Array(n * (KEY_BYTES + 4 * d));
    for (let k = 0; k < n; k++) {
      out.set(Buffer.from(keys[k], 'hex'), k * (KEY_BYTES + 4 * d));
      out.set(vectors.subarray(k * 4 * d, (k + 1) * 4 * d), k * (KEY_BYTES + 4 * d) + KEY_BYTES);
    }
    return out;
  }

  private checkDimensions(vector: ArrayLike<number>, source: string): void {
    const { dimensions } = this.options;
    if (dimensions !== undefined && vector.length !== dimensions) {
      throw new Error(`${source} has a ${vector.length}-dimensional vector, expected ${dimensions} (a different model?)`);
    }
  }

  private truncate(text: string): string {
    return truncateText(text, this.options.maxChars);
  }

  private key(text: string): string {
    return sha256(this.truncate(text)).toString('hex');
  }

  async embedMany(texts: string[]): Promise<number[][]> {
    const { concurrency = 16, checkpointEvery = 200, log = console.log } = this.options;
    const missing = [...new Map(texts.filter((t) => !(this.key(t) in this.cache)).map((t) => [this.key(t), t])).entries()];
    if (missing.length) {
      log(`  embedding ${missing.length} new text(s) (${texts.length - missing.length} cached)`);
      const size = this.options.embedBatch ? Math.max(1, this.options.batchSize ?? 96) : 1;
      const batches: Array<Array<[string, string]>> = [];
      for (let i = 0; i < missing.length; i += size) batches.push(missing.slice(i, i + size));
      const binary = this.options.format === 'binary';
      let next = 0;
      let done = 0;
      let lastCheckpoint = 0;
      const worker = async () => {
        while (next < batches.length) {
          const batch = batches[next++];
          const inputs = batch.map(([, text]) => this.truncate(text));
          const vectors = this.options.embedBatch ? await this.options.embedBatch(inputs) : [await this.options.embed!(inputs[0])];
          if (vectors.length !== batch.length) throw new Error(`embedBatch returned ${vectors.length} vectors for ${batch.length} texts`);
          vectors.forEach((v) => this.checkDimensions(v, 'the embedding provider'));
          // Binary caches hold float32 from the start, so this run returns what a rerun reads from disk.
          batch.forEach(([key], k) => { this.cache[key] = binary ? Float32Array.from(vectors[k]) : vectors[k]; this.pending.push(key); });
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
      } finally {
        this.save();
      }
    }
    // Copies, so a caller mutating a vector can't corrupt the cache (or another text's result).
    return texts.map((t) => Array.from(this.cache[this.key(t)]));
  }

  private save(): void {
    if (this.options.format === 'binary') {
      if (!this.pending.length) return;
      const d = this.options.dimensions!, record = KEY_BYTES + 4 * d;
      const out = new Uint8Array(this.pending.length * record);
      this.pending.forEach((key, k) => {
        out.set(Buffer.from(key, 'hex'), k * record);
        out.set(new Uint8Array(Float32Array.from(this.cache[key]).buffer), k * record + KEY_BYTES);
      });
      withLock(this.binPath, this.lockTimeoutMs, () => {
        // Another writer's interrupted append must not shift our records.
        const size = existsSync(this.binPath) ? statSync(this.binPath).size : 0;
        if (size % record) truncateSync(this.binPath, size - (size % record));
        appendFileSync(this.binPath, out);
      });
      this.pending = [];
      return;
    }
    withLock(this.cachePath, this.lockTimeoutMs, () => {
      // Keep entries other writers saved since we loaded; ours win where both have a key.
      if (existsSync(this.cachePath)) {
        const disk = JSON.parse(readFileSync(this.cachePath, 'utf8')) as Record<string, number[]>;
        for (const [k, v] of Object.entries(disk)) if (!(k in this.cache)) this.cache[k] = v;
      }
      const tmp = `${this.cachePath}.${process.pid}.tmp`;
      writeFileSync(tmp, JSON.stringify(this.cache, (_, v) => (ArrayBuffer.isView(v) ? Array.from(v as Float32Array) : v)));
      renameSync(tmp, this.cachePath);
    });
    this.pending = [];
  }
}

/**
 * Deterministic pseudo-embedding from word hashes - ONLY for exercising a pipeline without an
 * embedding provider. Never ship a model trained on these.
 */
export function hashEmbedding(text: string, dimensions: number): number[] {
  const vec = new Float64Array(dimensions);
  for (const word of text.toLowerCase().split(/\s+/).filter(Boolean)) {
    const digest = sha256(word);
    vec[digest.readUInt32BE(0) % dimensions] += digest[4] % 2 ? 1 : -1;
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return Array.from(vec, (v) => (norm ? v / norm : v));
}
