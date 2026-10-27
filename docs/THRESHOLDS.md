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

**When tiers run first.** A head certified behind dismissal rules, the edge or the middle tier has its
threshold chained to their claims, which were certified on the same calibration sample. Its `delta`
holds given those earlier thresholds, not jointly with them. The guarantee records this as
`chain: { claims, ladder, joint_delta }`:
- `claims` counts the dismissal rules as 1, the edge as 2 and the middle tier as 2;
- `ladder` is how many edge candidates were tried before this one was kept;
- `joint_delta` = min(1, ladder × (claims + 1) × delta) is a union bound on any of them failing.

Read `1 − joint_delta` as the confidence of the whole pipeline. For a joint statement at a chosen level,
certify each claim at that level divided by (claims + 1), or keep a calibration sample per tier.
In testing (140 releases on a public Jira dataset) `joint_delta` was 0.15–0.20 in most releases and
0.60 in 5, where the ladder tried all three edge candidates. A calibration sample per tier makes a
single δ valid for the whole chain, but each tier then certifies on fewer positives: at about 65
positives that is worth about one point of false alarms by the headroom table below (an estimate, not
a measurement), less than the edge would lose.

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

## Recall reserve and the edge tier

**Reserve.** `HeadPolicy.reserve` (recall heads, default 0) certifies the threshold at `targetRecall +
reserve` while the gates and the stated promise stay at `targetRecall`. It is insurance against later
periods: a guarantee on one period can't vouch for the next (FINDINGS O2 in liquidau-lab). It helps
only where the later period drifts and costs false alarms everywhere, so set it per head from
evidence of drift.

**Edge tier.** With `HeadInput.edge` (trainEdge's rows), the edge's dismissed rows score 0 and its fired
rows 1 before the threshold is chosen, so the certified recall covers the edge and the head together
(a dismissed positive is a miss), and the edge's fires count against the firing budget. The edge's
own fire and dismiss thresholds are chosen on the same calibration rows, a mild optimism to keep in
mind at small samples.

**Calibration gates with rows decided before the head.** Rows the dismissal rules or the edge decided
carry preset probabilities (0 or 1) that say nothing about the calibrator, so ECE, the reliability
table and the Cox test use only the rows the head scored itself (`evaluateHead`'s `scored`;
`calibration_rows` records how many). With fewer than 100 such test rows the ECE gate warns instead
of failing: an edge that decides 99% of a head's traffic leaves too few rows to judge calibration on
(in the lab, 29 rows gave an ECE of 0.13 on a head whose guarantee and later-period promise held).

**What the calibration gate can't see.** The guarantees rest on ranks, not on the probabilities, so
calibration errors never break a promise. They do mislead whoever reads a probability, such as a
reviewer.
- **The corrected test forgives the calibrator's own bad luck.** In the lab, heads that plain Cox
  refused had a mean error of 5.1 points at the decisions, against 2.7 for the rest. The corrected test
  passes almost all of them, including one 11.4 points off.
- **No test here sees an error in the curve's shape** (too flat, too steep, a plateau) at ordinary
  sample sizes.
- **Read `calibration_region` on every head.** In the report, "calibration at the decisions" gives
  observed against predicted, and how much of the ± is the calibrator's own (`se_calibrator`). When that
  share dominates, the calibration sample is what limits the probabilities. More calibration labels help;
  more test labels don't. See DATA.md for the calibration share.

## Choosing a head's target and budget

**Two limits on one threshold.** A recall head's threshold must be low enough that the certified
recall bound reaches the target. With a background budget it must also be high enough to fire on at
most the budget's share of ordinary traffic. When the first limit sits below the second, no threshold
meets both, and the head fails its gates with the reason ("recall ≥ … needs a threshold ≤ a but
background ≤ … needs ≥ b"). The fire rate is about prevalence × recall + (1 − prevalence) × false-alarm
rate, so prevalence sets a floor: at 10% prevalence and recall 0.85, a head with 30% false alarms fires
on about 36% of traffic.

**Check a task before freezing it.** One release says only whether the head issued this time, and near
the boundary that flips between samples. Before committing to a target and budget, train on a held-out
slice of labelled history (a design replica) for the task and its stricter neighbours (the target one
step up, the budget one step down), on several seeds. A task issued on most seeds whose stricter
neighbours also mostly issue is comfortably feasible. In testing this predicted issuance on another
sample of three datasets, and promises kept six months later on Jira, including on a period held back
until the task was fixed (both later periods drifted benignly). A "no" is less reliable
than a "yes": rejected tasks still issued on some seeds (in one later period those ran at or over their
budget; in the other they didn't). The procedure is in liquidau-examples' README.

**Headroom: what certifying costs.** The certificate is a lower confidence bound, so the threshold must
reach the target with room to spare, and the room shrinks only as calibration positives grow. The
smallest observed recall that certifies each target at δ = 0.05 (one-sided Clopper–Pearson with equal
weights; design weights count Kish effective positives and need a little more):

| Calibration positives | recall ≥ 0.70 | ≥ 0.75 | ≥ 0.80 | ≥ 0.85 | ≥ 0.90 |
|---|---|---|---|---|---|
| 30 | 0.867 | 0.900 | 0.933 | 0.967 | 1.000 |
| 45 | 0.822 | 0.867 | 0.911 | 0.956 | 1.000 |
| 65 | 0.800 | 0.846 | 0.892 | 0.938 | 0.969 |
| 100 | 0.780 | 0.830 | 0.870 | 0.920 | 0.960 |
| 150 | 0.767 | 0.813 | 0.860 | 0.900 | 0.947 |
| 250 | 0.752 | 0.800 | 0.844 | 0.892 | 0.936 |
| 500 | 0.736 | 0.784 | 0.832 | 0.878 | 0.924 |

The room is paid for in false alarms. In testing, a Jira head promising recall ≥ 0.75 on about 65
effective positives finished at 0.83–0.87 on later data, and the room cost 7–10 points of false
alarms, about a quarter of them; the rest came from how separable the label is.
Half as many again calibration labels cut that cost only from 6.6 to 5.9 points. The certified bounds
held throughout: later recall fell below the test split's certified lower bound in none of 100
releases.

**A better model: take it as a stronger promise.** When better features improve the ranking, keeping
the same target lets the threshold settle closer to it and spends the room above. In testing, Jira with
the project and issue type as context broke 2 of 20 promises at recall ≥ 0.75. Raising the target
keeps the room, and the better ranking still pays for it: the same flow at recall ≥ 0.80 had fewer
false alarms than the baseline at 0.75 (25.2% against 28.2%) and kept 19 of 20 on a later period held
back until the comparison was fixed.

## Gates and checks

| Export | What |
|---|---|
| `gateHead` | Release gates: minimum positives, recall, CI lower bound, beats the baseline, precision, minimum fired (`minFired`), ECE. Too few distinct groups, or fewer than 30 positives / fired examples with no explicit minimum, produces a warning |
| `sliceGate` (recall `HeadPolicy`) | Fails a slice whose recall is demonstrably below target (upper bound below target, Bonferroni across slices), warns when it can't be confirmed or has too few positives. With a slice at 0.80 against a 0.9 target it failed 99.9% of runs, where the overall recall gate failed 29% |
| `calibration_test`, `calibrationAlpha`, `calibrationGate` | Cox's recalibration test on the test split, always reported; `calibrationAlpha` makes it a gate. At 2% prevalence the ECE > 0.05 gate never fired, even for clearly miscalibrated probabilities. `calibrationGate` picks the test the gate applies. `'cox-corrected'` is the default (stage CK). It tests whether the test split fits the same logistic curve as the shipped Platt calibrator, counting both fits' sampling error (`calibration_test.corrected`, solvers `coxTestCorrected`). Plain `'cox'` tests whether the shipped probabilities are exactly right. No fitted calibrator is, so it refused about 10-13% of well-calibrated heads in the lab (18 of 140, against 1 of 140 for the corrected test). Without a Platt calibrator (isotonic), plain Cox applies, with a warning. `'region'` fails only a demonstrable error larger than `calibrationTolerance` at the decisions. It almost never fails at ordinary sample sizes (7.4% power at 700 test rows) |
| `calibration` (`trainHeads`) | The calibrator: `'platt-firth'` (default since stage CG), `'platt'` or `'isotonic'`. Firth's bias reduction keeps the fit finite when calibration positives are separated from the negatives. Ranks, and so thresholds and guarantees, do not change |
| `calibrationRefit` | Opt-in, default `'none'`. `'pooled'` refits each head's Platt calibrator on the calibration and test splits together after the gates, each half weighted as it was evaluated (1/π with sampled records), and carries the threshold over by rank: the smallest logit the calibration-only fit sent to the threshold sets the new threshold, so every decision is unchanged and only the probabilities move. The review floor is the new threshold × `reviewRatio`. The gates, the test evaluation and every threshold still use the calibration-only fit, so the shipped probabilities have no held-out check of their own; a warning says so, and `training.calibration.refit` records it. In the lab (stage CP2, 140 heads) the pooled probabilities were closer to the truth in the decision regions by 0.80 points on average, heads 5 points or more off fell from 23 to 8, and in simulation the chance of a large error roughly halved. The best-calibrated head (ToxicChat `jailbreak`) got slightly worse. It helps most when the calibration sample drew badly, which more calibration labels also fix. Platt calibrators only |
| `training.calibration`, `CalibrationRecord` | What the run chose, defaults resolved, written into the artifact by `buildArtifact`: `{ method, max_ece, gate, refit? }`, with `gate` `{ kind, alpha, tolerance? }`, or `null` without `calibrationAlpha`, and `refit: 'pooled'` when `calibrationRefit` chose it. A head's own `calibration.method` says only how it is served (`'platt'` covers both Platt fits), so this is the record of which calibrator and gate a release was fitted under. Use it to tell whether a stored result is comparable with a later library's defaults (stage JV found the defaults had moved under stored results). |
| `calibration_region` | Observed against predicted positive rate at or above the review floor and in the top decile, always reported. Its ± combines the test sample's error with the calibrator's own share (`se_calibrator`), which the report prints separately |
| `monitorWindow` | Drift checks for a deployed head. Exact binomial tests on the live share at or above the threshold (against the certified background bound) and at or above the review floor (against a reference), in both directions, with Bonferroni across the four tests. In simulation they caught new high-scoring topics and lost firing that a KS test on the whole score distribution missed, with no false alarms in 400 windows. Label meaning can't be monitored without fresh samples |
