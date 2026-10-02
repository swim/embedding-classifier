# @liquidau/embedding-classifier

Calibrated, gated, multi-head classifiers on top of text embeddings: train in Node, ship a small
JSON artifact, and score it at runtime with the same code that evaluated it.

It is built for decisions where **you must know how often the model is wrong before letting it
act**, such as routing, triage, moderation or escalation. Each label gets its own head: a logistic
regression on the embedding, calibrated to the label's production prevalence. Its threshold comes
from an explicit recall or precision policy, and it must pass release gates before an artifact may
be enforced.

Built on [`@liquidau/solvers`](../solvers), which is verified against scikit-learn.

## Pipeline

```
embeddings ─► trainHeads ─────────────────────────────► ClassifierArtifact (JSON)
              train split:       fitLogistic (class-balanced, L2)        │
              calibration split: Platt | isotonic @ prevalence           │  validateArtifact
                                 pickThreshold(policy)                   ▼
              test split:        evaluateHead + gateHead      scoreEmbedding ─► decide(policy)
```

| Export | What |
|---|---|
| `trainHeads` | Fit, calibrate, threshold, evaluate and gate every head. Pure: you supply the embeddings, a split per example, and a binary target per head (`null` leaves an example out of a head) |
| `HeadPolicy` / `pickThreshold` | `recall`: the threshold reaches `designRecall` on calibration positives, set above the test gate's `targetRecall`, and is capped by a `maxFalseAlarm` budget. `precision`: the threshold is the target precision |
| `evaluateHead` | Recall with a Wilson 95% CI, false-alarm rate, prevalence-weighted precision and ECE, reliability table, distinct positive groups, recall per slice, and a comparison with an optional **baseline** (e.g. existing rules) including combined recall |
| `gateHead` | Release gates: minimum positives, recall, CI lower bound, beats the baseline, precision, minimum fired (`minFired`), ECE. Too few distinct groups, or fewer than 30 positives / fired examples with no explicit minimum, produces a warning |
| `background` (`trainHeads`) / `budgetThreshold` | Raise a head's threshold until it fires on at most a given share of ordinary background traffic, before test evaluation, so the gates judge the shipped threshold. Labelled test sets rarely contain enough ordinary text to show the false alarms that matter |
| `assertRoundTrip` | Serialise, reload and re-score the artifact, so what was evaluated is exactly what will run |
| `validateArtifact`, `scoreEmbedding`, `calibrate` | Runtime loading and scoring. Small and dependency-light |
| `decide` / `missingPolicyHeads` | Priority order plus suppression rules, e.g. "the scope heads can't fire while any risk head is in its review band". `decide` throws on a missing or non-finite score. Policy heads the artifact lacks never fire or suppress; check them once at startup with `missingPolicyHeads` |
| `loadOrder`, `refuseToServe`, `publishPlan` | Artifact lifecycle. A promoted artifact (must pass its gates) is enforced. A shadow candidate (may fail its gates) is only ever scored in shadow mode |
| `reportMarkdown` | A human-readable report of the gates and every head |
| `CachedEmbedder`, `truncateText`, `hashEmbedding` (from `/embedder`) | Embedding with a provider-agnostic on-disk cache (Node only): one text per call (`embed`) or batched (`embedBatch`, `batchSize`); atomic saves; optional `dimensions` check. `truncateText` is the surrogate-safe truncation the cache uses - call it at runtime too. Plus a deterministic fake embedding for pipeline tests. `EmbeddingSpec.input_type` records provider input types (e.g. Cohere `classification`) |

## Example

```ts
import { trainHeads, assertRoundTrip, decide, scoreEmbedding, validateArtifact } from '@liquidau/embedding-classifier';

const result = trainHeads({
  X: embeddings,                  // number[][]
  split,                          // ('train' | 'calibration' | 'test')[]
  groups,                         // optional: paraphrases of one seed share a group
  slices: { source: sources },    // optional: recall per source=…
  heads: [
    { name: 'urgent', y: yUrgent, prevalence: 0.005, baseline: rulesCaught,
      policy: { kind: 'recall', targetRecall: 0.95, designRecall: 0.98, maxFalseAlarm: 0.05, minPositives: 150, minRecallLower: 0.9 } },
    { name: 'off_topic', y: yOffTopic, prevalence: 0.05, policy: { kind: 'precision', targetPrecision: 0.8 } },
  ],
});
const artifact = { version, created_at, embedding: { model_id, dimensions, normalize: true },
  heads: result.heads, evaluation: result.evaluation,
  gates: { passed: result.failures.length === 0, failures: result.failures, warnings: result.warnings } };
assertRoundTrip(artifact, embeddings, result.testProbabilities);

// Runtime
const model = validateArtifact(JSON.parse(json), { heads: ['urgent', 'off_topic'] });
const decision = decide(model, scoreEmbedding(model, await embed(text)), {
  priority: ['urgent', 'off_topic'],
  suppress: [{ when: ['urgent'], heads: ['off_topic'] }],
});
// -> { head: 'urgent' | 'off_topic' | null, reason: 'above_threshold' | 'near_threshold' | 'none' }
```

`near_threshold` means at least one head is in its review band (`review_floor` ≤ p < threshold).
That band is the natural set of examples to sample for human labelling.

## What is deliberately not here

The library has no knowledge of label schemas, embedding providers, storage or what an outcome
*does*. Your application supplies the targets per head, the embedding call (keep it identical
between training and runtime), the baseline, the decision policy, and what happens on a decision.

## Develop

`@liquidau/solvers` is linked from the sibling `../solvers` directory. npm does not install a linked
package's own dependencies, so the first install is two steps:

```bash
npm run setup   # installs ../solvers' runtime deps, then this package's
npm test        # runs from source (export condition "@liquidau/source")
npm run typecheck
npm run build   # dist/ (ESM + .d.ts)
npm run check:dist  # fails if the checked-in dist/ differs from a fresh build
```
