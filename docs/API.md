# API

Calibrated, gated, multi-head classifiers on top of text embeddings: train in Node, ship a small
JSON artifact, and score it at runtime with the same code that evaluated it.

It is built for decisions where **you must know how often the model is wrong before letting it
act**, such as routing, prioritisation, moderation or escalation. Each label gets its own head: a logistic
regression on the embedding, calibrated to the label's production prevalence. Its threshold comes
from an explicit recall or precision policy, and it must pass release gates before an artifact may
be enforced.

Built on [`@liquidau/solvers`](https://www.npmjs.com/package/@liquidau/solvers), which is verified against scikit-learn.

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
| `weak` / `maxWeakShare` (`trainHeads`) | Extra **weak positives** for the train split only (e.g. rule-miner's `weakLabels`), each weighted in [0, 1], total capped at `maxWeakShare` (λ, default 0.5) × the head's gold train positives. An embedding identical to a calibration, test or background row is refused. Class balancing is sample-weighted, so weak positives take λ / (1 + λ) of the positive class's weight instead of adding to it. `weakLabels` in the result (count, weight, cap, per rule, rule-set version and hash) belongs in `artifact.training.weak_labels`, and `reportMarkdown` shows it |
| `mode` (recall `HeadPolicy`), `conformalThreshold` | How a recall head's threshold is chosen. `heuristic` (default) is the `designRecall` margin and guarantees nothing. `conformal-expected` and `conformal-pac` pick it from order statistics of calibration positives (one per group), so production recall ≥ `targetRecall` in expectation, or with probability 1 − `delta`. `maxFalseAlarm` and any background budget are certified the same way, with δ split across them. If the limits cross, the head fails with the reason. `auto` takes the strongest guarantee the data supports. It falls back for lack of data with an *inconclusive* warning (to no guarantee at all only while `allowHeuristicFallback` is true, the default), and fails rather than drop PAC for the false-alarm budget. `evaluation.sufficiency` says why. Guarantees need each class's calibration examples to be exchangeable with production ones; prevalence may differ |
| `certified` (`evaluateHead`) | Exact Clopper–Pearson bounds from the test split, valid whatever chose the threshold: recall lower and upper, false-alarm upper, and precision lower at production prevalence. Groups count once. Plus a background-rate upper bound that stays valid when the background set the threshold. Under a conformal guarantee, the recall gate fails only when the test split contradicts it |
| `reviewEpsilon` (`trainHeads`) | A conformal review floor: at most ε of positives score below it in expectation, so automatic dismissals have a stated miss rate |
| `ExampleRecord`, `validateProvenance`, `capTrainingWeights` (`records` on `trainHeads`) | Where every example came from. Only probability-sampled, human-labelled real traffic may calibrate or test (P1). Background is unlabelled traffic with one use (P2). A group stays in one role (P3). Retrieved and generated data are training-only (P4); near-duplicates of evaluation records are dropped (P5). Retrieved positives are capped at 50% and generated hard negatives at 30% of a head's training weight, and per rule (P6). Generated records need verification or an accepted batch (P7). Violations throw `ProvenanceError`; overrides are recorded and fail the release gate |
| `designSample`, `labelQueue`, `applyReviews`, `designOf` | A stratified probability sample of a traffic frame (rule firing × score band × optional slice), split into train, calibration and test before labelling. One blinded labelling queue: reviewers see an id, the text and the heads, nothing else. Inclusion probabilities are recomputed from what was actually labelled; skip rates are reported. Scores in the frame must come from text only. Allocation `expected-positives` keeps proportional shares and throws unless `total` is large enough for every stratum holding ≥ 2% of positives to expect 10 calibration positives, using rates from a prior round. In simulation, over-sampling high-score strata (the usual way to find positives) made design-based recall bounds fail in 57% of feasible runs, while sized proportional allocation held at 3.7% for δ = 5% |
| `mode: 'design'` (recall `HeadPolicy`), design-based evaluation | With sampled records, calibration is weighted by 1/π, which already reproduces production prevalence, so `prevalence` must not be passed. The `design` mode sets the threshold with solvers' `designRiskThreshold` (`designMethod` `exact` or `linearised`). Test metrics become Horvitz–Thompson estimates with linearised and bootstrap intervals, and the gates count Kish effective positives. Conformal modes refuse unequal inclusion probabilities |
| `retrieveFromSeeds`, `seedStats` | Real candidates near verified positives (cosine floor, MMR diversity), excluding labelled, reserved and evaluation-near-duplicate items, for labelling as training data. Seeds with a hit rate below 0.1 after 10 labels are retired |
| `realHardNegatives`, `selectForReview`, `verifyBatch` | Real rule-firing negatives from training data; generated batches accepted when the Wilson lower bound on reviewer agreement is ≥ 0.9 (96/100 passes, 95/100 fails). `publishPlan` refuses to promote an artifact trained on generated data without acceptance evidence |
| `defineAxes`, `coverageReport`, `checkSliceAxes` | Where real labelled data is thin: per-value and pairwise counts, and the positive groups each observable slice needs for a zero-miss guarantee. Tags are approximate and used only in this report |
| `mergeSmallStrata` (`designSample`, default on) | Strata too small for 2 calibration and 2 test items are merged before sampling: an adjacent band with the same rule firing and slice first, then across slice, then across rule firing. The design records which cells were merged. Decided from frame counts only, so estimates stay unbiased |
| `monitorWindow` | Drift checks for a deployed head. Exact binomial tests on the live share at or above the threshold (against the certified background bound) and at or above the review floor (against a reference), in both directions, with Bonferroni across the four tests. In simulation they caught new high-scoring topics and lost firing that a KS test on the whole score distribution missed, with no false alarms in 400 windows. Label meaning can't be monitored without fresh samples |
| `mode: 'design'` (precision `HeadPolicy`) | A precision threshold with a design-based guarantee from sampled calibration records. Candidates come from training scores, so they're fixed before calibration. In simulation the heuristic precision threshold missed its target in 95% of runs when Platt was misspecified; the design threshold held it |
| `sliceGate` (recall `HeadPolicy`) | Fails a slice whose recall is demonstrably below target (upper bound below target, Bonferroni across slices), warns when it can't be confirmed or has too few positives. With a slice at 0.80 against a 0.9 target it failed 99.9% of runs, where the overall recall gate failed 29% |
| `calibration_test`, `calibrationAlpha` | Cox's recalibration test on the test split, always reported; `calibrationAlpha` makes it a gate. At 2% prevalence the ECE > 0.05 gate never fired, even for clearly miscalibrated probabilities. Cox kept a 5% false-rejection rate and caught shifted, overconfident and top-inflated probabilities |
| `stratumOf` (`designSample`) | Every frame item's stratum, keyed as `designOf` keys strata. Population-level facts (e.g. how often a deterministic rule fires in each stratum) can then sharpen design-based bounds |
| `requiredSampleSize`, `allocation.priors` | The smallest `expected-positives` total, computed without drawing. Priors for several heads give a total that satisfies every head. A stratum counts as material only if the prior round saw positives in it |
| `assertRoundTrip` | Serialise, reload and re-score the artifact, so what was evaluated is exactly what will run |
| `validateArtifact`, `scoreEmbedding`, `calibrate` | Runtime loading and scoring. Small and dependency-light |
| `decide` / `missingPolicyHeads` | Priority order plus suppression rules, e.g. "the scope heads can't fire while any priority head is in its review band". `decide` throws on a missing or non-finite score. Policy heads the artifact lacks never fire or suppress; check them once at startup with `missingPolicyHeads` |
| `loadOrder`, `refuseToServe`, `publishPlan` | Artifact lifecycle. A promoted artifact (must pass its gates) is enforced. A shadow candidate (may fail its gates) is only ever scored in shadow mode |
| `reportMarkdown` | A human-readable report of the gates and every head |
| `CachedEmbedder`, `truncateText`, `hashEmbedding` (from `/embedder`) | Embedding with a provider-agnostic on-disk cache (Node only): one text per call (`embed`) or batched (`embedBatch`, `batchSize`); atomic saves; optional `dimensions` check. `truncateText` is the surrogate-safe truncation the cache uses - call it at runtime too. Plus a deterministic fake embedding for pipeline tests. `EmbeddingSpec.input_type` records provider input types (e.g. Cohere `classification`) |

## Other exports

| Export | What |
|---|---|
| `headProbability` | One head's calibrated probability for an embedding (what `scoreEmbedding` does per head) |
| `falseAlarmCap`, `nextUp` | The lowest threshold at which at most `maxFalseAlarm` of (weighted) negatives score at or above it; the next double above a number (re-exported from solvers) |
| `SPLITS` | The three split names: `train`, `calibration`, `test` |
| `THRESHOLD_MODES`, `groupScores` | The valid `mode` values, and one score per group (min for positives, max for negatives), as the conformal modes use |
| `stableId` | A short, stable, non-cryptographic id (FNV-1a), as the design and queue ids use |
| `ProvenanceError`, `isReal` | The error `validateProvenance` throws, and whether a record's source is real traffic |
| `mmrSelect` | Maximal-marginal-relevance selection, as used by `retrieveFromSeeds` |
| Types (artifact) | `Calibration`, `ClassifierArtifact`, `EmbeddingSpec`, `GateResult`, `HeadSpec`, `Scores` |
| Types (decide) | `Decision`, `DecisionPolicy`, `DecisionReason` |
| Types (threshold) | `HeadPolicy` |
| Types (evaluate) | `BaselineComparison`, `CertifiedBounds`, `DesignEstimate`, `DesignEvaluation`, `EvaluateInput`, `HeadEvaluation` |
| Types (train) | `HeadInput`, `ProvenanceSummary`, `Split`, `TrainInput`, `TrainResult`, `WeakInput`, `WeakSummary` |
| Types (lifecycle) | `ArtifactRole`, `ServeMode` |
| Types (conformal) | `ConformalSelection`, `FalseAlarmConstraint`, `Guarantee`, `GuaranteeKind`, `Sufficiency`, `ThresholdMode` |
| Types (design) | `DesignOptions`, `DesignStratum`, `DesignSummary`, `FrameItem`, `Mechanism`, `QueueItem`, `QueueKey` |
| Types (records) | `BackgroundUse`, `ExampleRecord`, `ProvenanceCode`, `ProvenanceOptions`, `ProvenanceResult`, `Role`, `Source`, `WeightCaps`, `WeightCapSummary` |
| Types (retrieval) | `Embedded`, `RetrieveOptions` |
| Types (coverage) | `Axes`, `CoverageReport` |
| Types (monitor) | `DriftCheck` |

From `@liquidau/embedding-classifier/embedder`: `CachedEmbedder`, `truncateText`, `hashEmbedding` (see the table above).

See [THRESHOLDS.md](THRESHOLDS.md), [DATA.md](DATA.md) and [EXAMPLE.md](EXAMPLE.md).
