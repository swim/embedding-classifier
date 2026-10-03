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
| `designSample`, `labelQueue`, `applyReviews`, `designOf` | A stratified probability sample of a traffic frame (rule firing × score band × optional slice), split into train, calibration and test before labelling. One blinded labelling queue: reviewers see an id, the text and the heads, nothing else. Inclusion probabilities are recomputed from what was actually labelled; skip rates are reported. Scores in the frame must come from text only. Allocation `expected-positives` keeps proportional shares and throws unless `total` is large enough for every stratum holding ≥ 2% of positives to expect 10 calibration positives, using rates from a prior round. In simulation, over-sampling high-score strata (the usual way to find positives) made design-based recall bounds fail in 57% of feasible runs, while sized proportional allocation held at 3.7% for δ = 5% |
| `mergeSmallStrata` (`designSample`, default on) | Strata too small for 2 calibration and 2 test items are merged before sampling: an adjacent band with the same rule firing and slice first, then across slice, then across rule firing. The design records which cells were merged. Decided from frame counts only, so estimates stay unbiased |
| `stratumOf` (`designSample`) | Every frame item's stratum, keyed as `designOf` keys strata. Population-level facts (e.g. how often a deterministic rule fires in each stratum) can then sharpen design-based bounds |
| `requiredSampleSize`, `allocation.priors` | The smallest `expected-positives` total, computed without drawing. Priors for several heads give a total that satisfies every head. A stratum counts as material only if the prior round saw positives in it |

## Training data

| Export | What |
|---|---|
| `weak` / `maxWeakShare` (`trainHeads`) | Extra **weak positives** for the train split only (e.g. rule-miner's `weakLabels`), each weighted in [0, 1], total capped at `maxWeakShare` (λ, default 0.5) × the head's gold train positives. An embedding identical to a calibration, test or background row is refused. Class balancing is sample-weighted, so weak positives take λ / (1 + λ) of the positive class's weight instead of adding to it. `weakLabels` in the result (count, weight, cap, per rule, rule-set version and hash) belongs in `artifact.training.weak_labels`, and `reportMarkdown` shows it |
| `retrieveFromSeeds`, `seedStats` | Real candidates near verified positives (cosine floor, MMR diversity), excluding labelled, reserved and evaluation-near-duplicate items, for labelling as training data. Seeds with a hit rate below 0.1 after 10 labels are retired |
| `realHardNegatives`, `selectForReview`, `verifyBatch` | Real rule-firing negatives from training data; generated batches accepted when the Wilson lower bound on reviewer agreement is ≥ 0.9 (96/100 passes, 95/100 fails). `publishPlan` refuses to promote an artifact trained on generated data without acceptance evidence |

## Coverage

| Export | What |
|---|---|
| `defineAxes`, `coverageReport`, `checkSliceAxes` | Where real labelled data is thin: per-value and pairwise counts, and the positive groups each observable slice needs for a zero-miss guarantee. Tags are approximate and used only in this report |
