/** A cache writer process for the multi-writer tests: embeds `count` texts named `name` into a shared cache. */
import { CachedEmbedder, hashEmbedding } from '../../src/embedder.ts';

const [cachePath, format, name, count] = process.argv.slice(2);
const embedder = new CachedEmbedder({
  cachePath, format: format as 'json' | 'binary', dimensions: 8, batchSize: 5, concurrency: 1, checkpointEvery: 5, log: () => {},
  // Small random delays interleave the two writers' checkpoints.
  embedBatch: async (texts) => { await new Promise((r) => setTimeout(r, Math.random() * 3)); return texts.map((t) => hashEmbedding(t, 8)); },
});
await embedder.embedMany(Array.from({ length: Number(count) }, (_, i) => `${name} text number ${i}`));
