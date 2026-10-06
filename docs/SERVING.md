# Serving

How to run a trained artifact in production, including on serverless and edge runtimes.

## What runs where

| Part | Runtime | Notes |
|---|---|---|
| Package root (`@liquidau/embedding-classifier`) | Any: Node, serverless functions, edge, browser | No Node built-ins; checked by a test that walks the import graph, and by bundling for the browser |
| `/embedder` (`CachedEmbedder`, `hashEmbedding`) | Node only | A disk cache for training. Don't use it to serve |
| `@liquidau/rule-miner` root, `@liquidau/solvers` | Any | Same check. `ruleSetHash` is a pure SHA-256, identical to `node:crypto`'s output |
| Training (`trainHeads`, embedding a dataset) | A batch job or container | Fits take seconds. Embedding thousands of texts on CPU takes minutes to hours, which doesn't suit request-shaped functions |

## Request path

1. **At start-up, once per instance:**
   - Load the artifact and check it with `validateArtifact`.
   - Load the rule set and check it with `validateRuleSet`, then build `ruleSetMatcher`.
   - Check the decision policy with `missingPolicyHeads`.
   - The artifact is small JSON (about 20 KB per head at 768 dimensions).
2. **Rules:** `matcher.match(text)`. Linear time, so hostile input can't stall the function.
3. **Embedding:** embed `truncateText(text, maxChars)` with the model and settings recorded in
   `artifact.embedding`. Use the same `maxChars` as training. `truncateText` is exported from the
   package root.
4. **Scoring and decision:** `scoreEmbedding`, then `decide`.
5. **Logging:** record the scores for drift checks (`monitorWindow`, run on a schedule over logged
   windows).

**Skipping the embedding when a rule decides.** Under a "rules or classifier" policy, a certified
rule's hit counts as its head firing. Apply it by lifting that head's score to its threshold before
`decide`, so priority and suppression still hold.

The embedding can be skipped only when the hit's head is **first in priority** and **no
suppression rule lists it**. Then `decide` returns that head whatever the other scores are. In any
other case, score as usual.

Messages decided this way have no scores, so embed a small random share of them (5%, say). Drift
windows then sample all traffic, not only what the rules missed.

The worked example is `support-example/src/serve.ts`. `npm run serve:check` there compares it with
the always-embed path on all 13,083 Banking77 messages under three policies: 0 mismatches, with 5%
of messages skipping the embedding where that is allowed.

## Two tiers: rules in front of the model

The deterministic rules can run in their own small function (a Lambda or an edge Worker: rule-miner
and this package's root import no Node built-ins) and send only the messages they can't settle to
the model's function.

1. **Rules tier:** `evaluation = ruleSetMatcher(set).evaluate(text)`, then
   `settleWithRules(policy, evaluation)`. Settled: answer, and forward a random share (5%, say) for
   scoring only, so drift windows see all traffic. Not settled: forward
   `{ text, fired, dismissed, ruleSet: ruleSetHash(set) }`.
2. **Model tier:** refuse if `checkRuleSetPairing(artifact, ruleSet)` is non-empty (the heads were
   certified with other dismissal rules); otherwise embed, `scoreEmbedding`, and
   `decide(artifact, scores, policy, dismissed, fired)`.

Whichever path a message takes, the decision is the same (a randomised test checks
`settleWithRules` against `decide`), so the heads' guarantees describe the system. Which path a
message takes is deterministic given the rule set. How much traffic the rules tier settles is a
measured rate, not a guarantee: with joint dismissal rules (rule-miner) it can be a large share for
clustered labels and very little for diffuse ones. Measure it on your own traffic (no labels needed).

## Where the embedding runs

| Option | Fits | Trade-off |
|---|---|---|
| Hosted embedding API | Everywhere, including edge | A network call per message (tens to hundreds of ms) and a per-token price. Text leaves your system. Train with the same API and model |
| Local ONNX model (e.g. transformers.js) | Container-style functions | all-mpnet-base-v2 is 110 MB and all-MiniLM-L6-v2 23 MB (8-bit). Cold starts load the model, and memory must hold it. Typical edge bundle and memory limits rule out the larger model |
| A long-running service | Anywhere | No cold starts. The same code as a function, loaded once |

Whichever you choose, `artifact.embedding` records the model, dimensions, normalisation and input
type the heads were trained on. Serving with anything else invalidates the guarantees.
