# @liquidau/embedding-classifier

Per-label classifiers on text embeddings, with thresholds you choose by the error you can accept,
release gates, and a small JSON artifact for runtime.

```
labelled data (designSample, labelQueue, retrieveFromSeeds)
        │
        ▼
trainHeads ── train ───────► logistic head per label
           ── calibration ─► calibrate ─► threshold (mode)
           ── test ────────► evaluateHead ─► gateHead
        │
        ▼
artifact.json ─► validateArtifact ─► scoreEmbedding ─► decide ─► your action
                                                     └─► monitorWindow (drift)
```

Built on [`@liquidau/solvers`](https://www.npmjs.com/package/@liquidau/solvers). Pair it with
[`@liquidau/rule-miner`](https://www.npmjs.com/package/@liquidau/rule-miner) for rules and weak labels.

## Install

```bash
npm install @liquidau/embedding-classifier
```

## Example

```ts
import { buildArtifact, decide, scoreEmbedding, trainHeads, validateArtifact, type Split } from '@liquidau/embedding-classifier';

// Toy data: 900 two-dimensional "embeddings"; urgent ones sit up and to the right.
const y = Array.from({ length: 900 }, (_, i) => (i % 10 === 0 ? 1 : 0) as 0 | 1);
const X = y.map((v, i) => [2 * v + Math.sin(i), 2 * v + Math.cos(1.7 * i)]);
const split = y.map((_, i): Split => (['train', 'calibration', 'test'] as const)[i % 3]);

const result = trainHeads({ X, split, heads: [{ name: 'urgent', y, prevalence: 0.1, policy: { kind: 'recall', targetRecall: 0.9 } }] });
console.log(result.failures, result.heads.urgent!.guarantee); // [] and a PAC recall guarantee (the default: the strongest the data supports)

const artifact = validateArtifact(buildArtifact(result, { version: '1', embedding: { model_id: 'toy', dimensions: 2, normalize: false } }), { mode: 'enforce' });
console.log(decide(artifact, scoreEmbedding(artifact, [2.1, 1.9]), { priority: ['urgent'] })); // { head: 'urgent', reason: 'above_threshold' }
```

A two-head version with a rules baseline, slices and gates is in [docs/EXAMPLE.md](docs/EXAMPLE.md).

When a route is a union of a finer label your labelling already records, such as intents under
"needs a person", pass the finer labels to `trainHeads` and set `finer: true` on the head. It adds
a certified finer-label route score to the embedding (off by default; see `finer` in
[docs/API.md](docs/API.md)).

With an expensive main encoder behind the edge, a cheaper encoder's head can fire and dismiss in
between: `trainMiddle`, with `HeadInput.cascade`. Training keeps it per head only where the training
rows say it costs no false alarms, and the head is certified behind both tiers. Off by default: it is
much cheaper, but its heads fail the release gates more often (liquidau-lab stage DZ); see
"Middle tier" in [docs/API.md](docs/API.md).

## Which threshold mode

| Mode | Use when | Guarantee |
|---|---|---|
| `heuristic` | Exploring, low stakes (the default) | None |
| `conformal-expected` | Recall heads, calibration not design-sampled | Recall on average |
| `conformal-pac` | Recall heads, enough calibration positives | Recall with probability 1 − δ (behind the edge or other tiers, 1 − `guarantee.chain.joint_delta` for the pipeline: docs/THRESHOLDS.md) |
| `auto` | Recall heads, calibration not design-sampled | Strongest conformal guarantee the data supports |
| `design` | Calibration came from `designSample` | Recall or precision, from the sample design |

**Conformal modes and `auto` refuse design-sampled calibration; precision heads take only
`heuristic` or `design`.** Details in [docs/THRESHOLDS.md](docs/THRESHOLDS.md).

## What's in it

| Area | Exports |
|---|---|
| Training | `trainHeads` (options `finer`, `standardize`, `cascade`), `assertRoundTrip` |
| Thresholds | `pickThreshold`, `conformalThreshold`, `budgetThreshold` |
| Evaluation, gates | `evaluateHead`, `gateHead`, `reportMarkdown` |
| Runtime | `validateArtifact`, `scoreEmbedding`, `decide` |
| Lifecycle, drift | `loadOrder`, `publishPlan`, `monitorWindow` |
| Labelling | `designSample`, `labelQueue`, `applyReviews` |
| Training data | `retrieveFromSeeds`, `realHardNegatives`, `verifyBatch`, `coverageReport` |
| Provenance | `ExampleRecord`, `validateProvenance` |

## Guarantees and limits

- Only an artifact that passed its gates is enforced; a failing one runs in shadow mode only.
- Threshold guarantees hold only if calibration examples resemble production traffic.
- Use the same embedding model and `truncateText` at training and at runtime.
- One linear head per label, binary.
- The package root runs on any runtime, edge included; `CachedEmbedder` (`/embedder`) is Node only.

## What is deliberately not here

The library has no knowledge of label schemas, embedding providers, storage or what an outcome
*does*. Your application supplies the targets per head, the embedding call (keep it identical
between training and runtime), the baseline, the decision policy, and what happens on a decision.

## More

- [docs/API.md](docs/API.md): every export, and why each exists.
- [docs/THRESHOLDS.md](docs/THRESHOLDS.md): modes, sufficiency, certified bounds, review floor, gates.
- [docs/DATA.md](docs/DATA.md): records and rules P1–P7, sampling, labelling, retrieval, coverage.
- [docs/EXAMPLE.md](docs/EXAMPLE.md): the full two-head example.
- [docs/SERVING.md](docs/SERVING.md): serving on serverless and edge runtimes, rules first.

## Develop

```bash
npm install
npm test            # Node 20+; @liquidau/solvers resolves to its built dist
npm run typecheck
npm run build       # dist/ (ESM + .d.ts)
npm run check:dist  # fails if the checked-in dist/ differs from a fresh build
```

## License

MIT. See [LICENSE](LICENSE).
