# Thresholds, guarantees and gates

How each head's threshold is chosen, what it guarantees, and how the result is checked. The rows
below are the threshold and gate entries from [API.md](API.md).

## Which mode

| Head kind | Mode | Use when | Guarantee |
|---|---|---|---|
| recall | `heuristic` (only when set explicitly) | Exploring, low stakes | None |
| recall | `conformal-expected` | Some calibration positives, equal inclusion probabilities | Recall ≥ target on average |
| recall | `conformal-pac` | Enough calibration positives, equal inclusion probabilities | Recall ≥ target with probability 1 − δ |
| recall | `auto` (default without sampled records) | Calibration was not design-sampled | The strongest conformal guarantee the data supports; by default no heuristic fallback |
| recall | `design` (default with sampled records) | Calibration came from `designSample` | Recall ≥ target with probability 1 − δ, from the design |
| precision | `heuristic` (only when set explicitly) | Exploring | None |
| precision | `design` (default with sampled records) | Calibration came from `designSample` | Precision ≥ target with probability 1 − δ, from the design. Without sampled records and without `heuristic` set, the head fails its gates |

The conformal modes and `auto` refuse unequal inclusion probabilities, so design-sampled
calibration needs `design`. Precision heads accept only `heuristic` and `design`.

**The default is the strongest guarantee the data supports** (since 0.7): `design` with sampled
calibration records, else `auto` without the silent heuristic fallback; a head the data can't
support fails its gates, and can still be served in shadow mode. `heuristic` carries no guarantee
and must be chosen explicitly (a warning records the choice). The trade-off: a guaranteed threshold
is set conservatively, so it fires on somewhat more negatives than a heuristic one, and a head with
too few labelled positives gets no guarantee at all; a heuristic threshold always gives a number, but
its recall is never checked and can quietly fall short of the target.

**Per-head fallback (`fallback: 'heuristic'`).** For heads whose labels may not yet support a
guarantee: the head gets the guarantee when its calibration data supports one, and the heuristic
threshold when it doesn't, instead of failing its gates. The fallback is recorded in the artifact
(`guarantee.fallback: 'heuristic'`, `kind: 'none'`), with a warning saying how many effective
calibration positives a guarantee would need, so it is visible to reviewers and to monitoring.
Recall heads need `designRecall` for it. `fallback: 'fail'` (the default) keeps the strict
behaviour. The choice is per head (label), made from that head's calibration evidence; a head's
threshold applies to every message alike.

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
