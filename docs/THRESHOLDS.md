# Thresholds, guarantees and gates

How each head's threshold is chosen, what it guarantees, and how the result is checked. The rows
below are the threshold and gate entries from [API.md](API.md).

## Which mode

| Head kind | Mode | Use when | Guarantee |
|---|---|---|---|
| recall | `heuristic` (default) | Exploring, low stakes | None |
| recall | `conformal-expected` | Some calibration positives, equal inclusion probabilities | Recall ≥ target on average |
| recall | `conformal-pac` | Enough calibration positives, equal inclusion probabilities | Recall ≥ target with probability 1 − δ |
| recall | `auto` | Calibration was not design-sampled | The strongest conformal guarantee the data supports |
| recall | `design` | Calibration came from `designSample` | Recall ≥ target with probability 1 − δ, from the design |
| precision | `heuristic` (default) | Exploring | None |
| precision | `design` | Calibration came from `designSample` | Precision ≥ target with probability 1 − δ, from the design |

The conformal modes and `auto` refuse unequal inclusion probabilities, so design-sampled
calibration needs `design`. Precision heads accept only `heuristic` and `design`.

## Modes and sufficiency

| Export | What |
|---|---|
| `HeadPolicy` / `pickThreshold` | `recall`: the threshold reaches `designRecall` on calibration positives, set above the test gate's `targetRecall`, and is capped by a `maxFalseAlarm` budget. `precision`: the threshold is the target precision |
| `mode` (recall `HeadPolicy`), `conformalThreshold` | How a recall head's threshold is chosen. `heuristic` (default) is the `designRecall` margin and guarantees nothing. `conformal-expected` and `conformal-pac` pick it from order statistics of calibration positives (one per group), so production recall ≥ `targetRecall` in expectation, or with probability 1 − `delta`. `maxFalseAlarm` and any background budget are certified the same way, with δ split across them. If the limits cross, the head fails with the reason. `auto` takes the strongest guarantee the data supports. It falls back for lack of data with an *inconclusive* warning (to no guarantee at all only while `allowHeuristicFallback` is true, the default), and fails rather than drop PAC for the false-alarm budget. `evaluation.sufficiency` says why. Guarantees need each class's calibration examples to be exchangeable with production ones; prevalence may differ |
| `mode: 'design'` (recall `HeadPolicy`), design-based evaluation | With sampled records, calibration is weighted by 1/π, which already reproduces production prevalence, so `prevalence` must not be passed. The `design` mode sets the threshold with solvers' `designRiskThreshold` (`designMethod` `exact` or `linearised`). Test metrics become Horvitz–Thompson estimates with linearised and bootstrap intervals, and the gates count Kish effective positives. Conformal modes refuse unequal inclusion probabilities |
| `mode: 'design'` (precision `HeadPolicy`) | A precision threshold with a design-based guarantee from sampled calibration records. Candidates come from training scores, so they're fixed before calibration. In simulation the heuristic precision threshold missed its target in 95% of runs when Platt was misspecified; the design threshold held it |

## Budgets, bounds and review floor

| Export | What |
|---|---|
| `background` (`trainHeads`) / `budgetThreshold` | Raise a head's threshold until it fires on at most a given share of ordinary background traffic, before test evaluation, so the gates judge the shipped threshold. Labelled test sets rarely contain enough ordinary text to show the false alarms that matter |
| `certified` (`evaluateHead`) | Exact Clopper–Pearson bounds from the test split, valid whatever chose the threshold: recall lower and upper, false-alarm upper, and precision lower at production prevalence. Groups count once. Plus a background-rate upper bound that stays valid when the background set the threshold. Under a conformal guarantee, the recall gate fails only when the test split contradicts it |
| `reviewEpsilon` (`trainHeads`) | A conformal review floor: at most ε of positives score below it in expectation, so automatic dismissals have a stated miss rate |

## Gates and checks

| Export | What |
|---|---|
| `gateHead` | Release gates: minimum positives, recall, CI lower bound, beats the baseline, precision, minimum fired (`minFired`), ECE. Too few distinct groups, or fewer than 30 positives / fired examples with no explicit minimum, produces a warning |
| `sliceGate` (recall `HeadPolicy`) | Fails a slice whose recall is demonstrably below target (upper bound below target, Bonferroni across slices), warns when it can't be confirmed or has too few positives. With a slice at 0.80 against a 0.9 target it failed 99.9% of runs, where the overall recall gate failed 29% |
| `calibration_test`, `calibrationAlpha` | Cox's recalibration test on the test split, always reported; `calibrationAlpha` makes it a gate. At 2% prevalence the ECE > 0.05 gate never fired, even for clearly miscalibrated probabilities. Cox kept a 5% false-rejection rate and caught shifted, overconfident and top-inflated probabilities |
| `monitorWindow` | Drift checks for a deployed head. Exact binomial tests on the live share at or above the threshold (against the certified background bound) and at or above the review floor (against a reference), in both directions, with Bonferroni across the four tests. In simulation they caught new high-scoring topics and lost firing that a KS test on the whole score distribution missed, with no false alarms in 400 windows. Label meaning can't be monitored without fresh samples |
