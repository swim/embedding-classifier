/**
 * Test-split evaluation of one head: recall with a Wilson CI, false-alarm rate,
 * prevalence-weighted precision and ECE, a reliability table, recall per slice, (when a baseline
 * is supplied, e.g. existing rules) what the classifier adds on top of it, and exact certified
 * bounds for the shipped threshold whatever chose it.
 */
import { clopperPearsonUpper, ece, wilson } from '@liquidau/solvers';
const clopperPearsonLower = (k, n, confidence) => 1 - clopperPearsonUpper(n - k, n, confidence);
export function evaluateHead({ p, y, w, threshold, groups, slices = {}, baseline, delta = 0.05, prevalence }) {
    if (!(delta > 0 && delta < 1))
        throw new Error(`delta must be strictly between 0 and 1, got ${delta}`);
    const lengths = [['y', y], ['w', w], ['groups', groups], ['baseline', baseline], ...Object.entries(slices).map(([f, v]) => [`slices.${f}`, v])];
    for (const [name, values] of lengths) {
        if (values !== undefined && values.length !== p.length)
            throw new Error(`${name} has ${values.length} entries but p has ${p.length}`);
    }
    const fired = p.map((v) => v >= threshold);
    const count = (pred) => p.reduce((n, _, i) => n + (pred(i) ? 1 : 0), 0);
    const pos = count((i) => y[i] === 1);
    const neg = count((i) => y[i] === 0);
    const tp = count((i) => fired[i] && y[i] === 1);
    const fp = count((i) => fired[i] && y[i] === 0);
    let wFired = 0;
    let wTp = 0;
    p.forEach((_, i) => {
        if (fired[i]) {
            wFired += w[i];
            wTp += w[i] * y[i];
        }
    });
    const calibration = ece(p, y, w);
    // Per group: [all members fired, any member fired], separately for each class.
    const byGroup = [new Map(), new Map()];
    p.forEach((_, i) => {
        if (y[i] !== 0 && y[i] !== 1)
            return;
        const g = groups ? groups[i] : `#${i}`;
        const prior = byGroup[y[i]].get(g) ?? [true, false];
        byGroup[y[i]].set(g, [prior[0] && fired[i], prior[1] || fired[i]]);
    });
    const tally = (cls, which) => [...byGroup[cls].values()].filter((v) => v[which]).length;
    const [posGroups, negGroups] = [byGroup[1].size, byGroup[0].size];
    const conf = 1 - delta / 2;
    const certified = {
        delta, positive_groups: posGroups, negative_groups: negGroups,
        recall_lower: posGroups ? clopperPearsonLower(tally(1, 0), posGroups, conf) : 0,
        recall_upper: posGroups ? clopperPearsonUpper(tally(1, 1), posGroups, conf) : 1,
        false_alarm_upper: negGroups ? clopperPearsonUpper(tally(0, 1), negGroups, conf) : 1,
    };
    if (prevalence !== undefined) {
        const tp = prevalence * certified.recall_lower;
        const fp = (1 - prevalence) * certified.false_alarm_upper;
        certified.precision_lower = tp + fp === 0 ? 0 : tp / (tp + fp);
    }
    const result = {
        threshold,
        n: y.length,
        fired: count((i) => fired[i]),
        positives: pos,
        positive_groups: new Set(p.flatMap((_, i) => (y[i] === 1 ? [groups ? groups[i] : `#${i}`] : []))).size,
        recall: pos ? tp / pos : NaN,
        recall_ci95: wilson(tp, pos),
        false_alarm_rate: neg ? fp / neg : NaN,
        precision_prevalence_weighted: wFired ? wTp / wFired : NaN,
        ece_prevalence_weighted: calibration.ece,
        reliability: calibration.reliability,
        slices: {},
        certified,
    };
    for (const [field, values] of Object.entries(slices)) {
        for (const v of [...new Set(values)].sort()) {
            const inSlice = (i) => values[i] === v && y[i] === 1;
            const n = count(inSlice);
            const k = count((i) => inSlice(i) && fired[i]);
            if (n)
                result.slices[`${field}=${v}`] = { positives: n, recall: k / n, recall_ci95: wilson(k, n) };
        }
    }
    if (baseline) {
        result.vs_baseline = {
            caught_by_both: count((i) => y[i] === 1 && baseline[i] && fired[i]),
            caught_by_baseline_only: count((i) => y[i] === 1 && baseline[i] && !fired[i]),
            caught_by_classifier_only: count((i) => y[i] === 1 && !baseline[i] && fired[i]),
            missed_by_both: count((i) => y[i] === 1 && !baseline[i] && !fired[i]),
        };
        const combined = count((i) => y[i] === 1 && (baseline[i] || fired[i]));
        result.combined_recall = pos ? combined / pos : NaN;
        result.combined_recall_ci95 = wilson(combined, pos);
        result.combined_false_alarm_rate = neg ? count((i) => y[i] === 0 && (baseline[i] || fired[i])) / neg : NaN;
    }
    return result;
}
