# Full example

Two heads, a rules baseline, slices and artifact assembly. Replace the free variables with your
own data: `embeddings`, `split`, `groups`, `sources`, `yUrgent`, `yOffTopic`, `rulesCaught`,
`version`, `created_at`, `model_id`, `dimensions`, `json`, `embed` and `text`.

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
    // Without sampled calibration records a precision head can't have a guarantee: choosing
    // 'heuristic' says so explicitly (otherwise the head fails its gates).
    { name: 'off_topic', y: yOffTopic, prevalence: 0.05, policy: { kind: 'precision', targetPrecision: 0.8, mode: 'heuristic' } },
  ],
});
const artifact = buildArtifact(result, { version, embedding: { model_id, dimensions, normalize: true } }); // gates from the result
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
