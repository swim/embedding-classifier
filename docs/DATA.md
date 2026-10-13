# Data: records, sampling and training data

Where examples come from and which uses each may have. The rows below are the data entries from
[API.md](API.md).

## Records and provenance (P1–P7)

| Export | What |
|---|---|
| `ExampleRecord`, `validateProvenance`, `capTrainingWeights` (`records` on `trainHeads`) | Where every example came from. Only probability-sampled, human-labelled real traffic may calibrate or test (P1). Background is unlabelled traffic with one use (P2). A group stays in one role (P3). Retrieved and generated data are training-only (P4); near-duplicates of evaluation records are dropped (P5). Retrieved positives are capped at 50% and generated hard negatives at 30% of a head's training weight, and per rule (P6). Generated records need verification or an accepted batch (P7). Violations throw `ProvenanceError`; overrides are recorded and fail the release gate |

## Sampling and labelling

| Export | What |
|---|---|
| `designSample`, `labelQueue`, `applyReviews`, `designOf` | A stratified probability sample of a traffic frame (rule firing × score band × optional slice), split into train, calibration and test before labelling. One blinded labelling queue: reviewers see an id, the text and the heads, nothing else. Inclusion probabilities are recomputed from what was actually labelled; skip rates are reported. Scores in the frame must come from text only. Allocation `expected-positives` keeps proportional shares and throws unless `total` is large enough for every stratum holding ≥ 2% of positives to expect 10 calibration positives, using rates from a prior round. In simulation, over-sampling high-score strata (the usual way to find positives) made design-based recall bounds fail in 57% of feasible runs, while sized proportional allocation held at 3.7% for δ = 5%. `allocation.topFactor` (proportional only) oversamples the top score band; it is safe only when the frame's scores come from a model independent of the head being certified, such as an offline LLM teacher run in batch (×2–×4 can substantially cut false alarms at a recall guarantee), and every factor above 1 adds a warning saying so. Allocation `neyman` sets n_h ∝ N_h·√(p_h(1 − p_h)) from a prior (a prior round, or an offline teacher's mean score per stratum, fixed before drawing), within the same floors and caps, putting labels where the label is uncertain. On real score shapes with Clef's scores as the prior it reached recall guarantees with far fewer labels than proportional or ×2 oversampling, at much lower false alarms; misses on later traffic reached 6.2% at 1,200 labels in one head, so check guarantees at large budgets |
| `mergeSmallStrata` (`designSample`, default on) | Strata too small for 2 calibration and 2 test items are merged before sampling: an adjacent band with the same rule firing and slice first, then across slice, then across rule firing. The design records which cells were merged. Decided from frame counts only, so estimates stay unbiased |
| `stratumOf` (`designSample`) | Every frame item's stratum, keyed as `designOf` keys strata. Population-level facts (e.g. how often a deterministic rule fires in each stratum) can then sharpen design-based bounds |
| `requiredSampleSize`, `allocation.priors` | The smallest `expected-positives` total, computed without drawing. Priors for several heads give a total that satisfies every head. A stratum counts as material only if the prior round saw positives in it |

**How much of the sample goes to calibration** (`roleSplit`; default train 0.5, calibration 0.25,
test 0.25). In the lab (stage SD, 11 October, EmbeddingGemma heads, a 1,800-label sample), moving from
train 0.2 / calibration 0.4 / test 0.4 to 0.1 / 0.6 / 0.3 lowered false alarms on ABCD `access` (5.10%
to 2.68%). Re-checked with a stricter rule (crashed seeds counted, effects beyond their standard error),
it helped two heads and hurt none with the finer-label score, and hurt one without it. It also has a
cost: on the Banking77 walkthrough (mpnet with the finer-label score) the smaller test split failed both
heads' calibration gates, and the release could only shadow. **Recommendation:** treat a larger
calibration share as something to try with the finer-label score. Check every gate before relying on
it, and at 0.75 heads lost guarantees. **Labelling budget:** 50% more labels, all of them to
calibration, did best on ABCD `access` (1.33%; 0.89% with the finer-label score) and helped two heads
and hurt none under the stricter rule (on ToxicChat the "+50%" exhausted the frame, so it was a census
there). Where labels can be bought, calibration is the best place to spend them. Stratifying by the
finer-label score helped nothing once the head already had the finer-label score. Neyman allocation on
those strata cost ABCD `refund` its guarantees, so it is not recommended there (RESEARCH.md, stage SD
and "Adoption re-check").

## Training data

| Export | What |
|---|---|
| `weak` / `maxWeakShare` (`trainHeads`) | Extra **weak positives** for the train split only (e.g. rule-miner's `weakLabels`), each weighted in [0, 1], total capped at `maxWeakShare` (λ, default 0.5) × the head's gold train positives. An embedding identical to a calibration, test or background row is refused. Class balancing is sample-weighted, so weak positives take λ / (1 + λ) of the positive class's weight instead of adding to it. `weakLabels` in the result (count, weight, cap, per rule, rule-set version and hash) belongs in `artifact.training.weak_labels`, and `reportMarkdown` shows it |
| `retrieveFromSeeds`, `seedStats` | Real candidates near verified positives (cosine floor, MMR diversity), excluding labelled, reserved and evaluation-near-duplicate items, for labelling as training data. Seeds with a hit rate below 0.1 after 10 labels are retired |
| `realHardNegatives`, `selectForReview`, `verifyBatch` | Real rule-firing negatives from training data; generated batches accepted when the Wilson lower bound on reviewer agreement is ≥ 0.9 (96/100 passes, 95/100 fails). `publishPlan` refuses to promote an artifact trained on generated data without acceptance evidence |

## Artifacts that contain data

A `knn` or `stack` head scores against `artifact.reference`: the training rows' embeddings
(float32) and their labels. Embeddings can be partly inverted back to text, so such an artifact is
**data derived from training messages**, not just model weights: store, share and delete it as you
would the messages. Generated records (synthetic text) are left out of the reference; retrieved
records are human-labelled and stay. A linear head's artifact holds no training rows. With the default type `auto`, any head may become knn or stack:
set `type: 'linear'` on heads whose artifact must not contain training embeddings.

## Coverage

| Export | What |
|---|---|
| `defineAxes`, `coverageReport`, `checkSliceAxes` | Where real labelled data is thin: per-value and pairwise counts, and the positive groups each observable slice needs for a zero-miss guarantee. Tags are approximate and used only in this report |
