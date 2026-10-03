/**
 * Coverage report: where real, human-labelled data is thin, and how many more positive groups each
 * runtime slice needs before a per-slice guarantee is possible. Generates nothing; it directs the
 * next sampling design and retrieval round.
 *
 * Tags come from an application tagger (often an LLM) and are approximate: they are used only
 * here, never in estimates, thresholds or training weights. Only axes marked `observable` - known
 * at prediction time, such as channel, language or length - may become slices. Marking an axis
 * that describes the label (e.g. a subtype) as observable is the caller's error to avoid.
 */
import { kishEffectiveN, minimumSamples } from '@liquidau/solvers';
import { designOf } from "./design.js";
import { isReal } from "./records.js";
export function defineAxes(axes) {
    for (const [name, a] of Object.entries(axes)) {
        if (!a.values.length)
            throw new Error(`axis ${name} has no values`);
        if (new Set(a.values).size !== a.values.length)
            throw new Error(`axis ${name} has duplicate values`);
    }
    return { axes: structuredClone(axes) };
}
export function coverageReport(options) {
    const { head, tags, axes, minRealPerCell = 10, guarantee = { alpha: 0.05, delta: 0.05 } } = options;
    const records = options.records.filter((r) => isReal(r) && r.labelledBy === 'human' && (r.labels[head] === 0 || r.labels[head] === 1));
    const names = Object.keys(axes.axes);
    const tagOf = (r, axis) => {
        const v = tags[r.id]?.[axis];
        if (v !== undefined && !axes.axes[axis].values.includes(v))
            throw new Error(`record ${r.id}: ${axis} = ${v} is not a value of the axis`);
        return v;
    };
    const report = { head, values: {}, gaps: [], slices: [], untagged: records.filter((r) => !tags[r.id]).length };
    for (const axis of names) {
        report.values[axis] = {};
        for (const value of axes.axes[axis].values) {
            const byRole = {};
            for (const r of records) {
                if (tagOf(r, axis) !== value)
                    continue;
                const c = (byRole[r.role] ??= { positives: 0, negatives: 0, positive_groups: 0 });
                if (r.labels[head] === 1)
                    c.positives++;
                else
                    c.negatives++;
            }
            for (const role of Object.keys(byRole)) {
                byRole[role].positive_groups = new Set(records.filter((r) => r.role === role && r.labels[head] === 1 && tagOf(r, axis) === value).map((r) => r.group)).size;
            }
            report.values[axis][value] = byRole;
        }
    }
    // Pairs across two axes, all roles together: full cells multiply too fast to fill.
    for (let i = 0; i < names.length; i++)
        for (let j = i + 1; j < names.length; j++) {
            for (const va of axes.axes[names[i]].values)
                for (const vb of axes.axes[names[j]].values) {
                    const inCell = records.filter((r) => tagOf(r, names[i]) === va && tagOf(r, names[j]) === vb);
                    const positives = inCell.filter((r) => r.labels[head] === 1).length;
                    const negatives = inCell.length - positives;
                    if (positives < minRealPerCell || negatives < minRealPerCell)
                        report.gaps.push({ a: `${names[i]}=${va}`, b: `${names[j]}=${vb}`, positives, negatives });
                }
        }
    // Slice requirements (observable axes only): groups needed for a zero-miss guarantee.
    const needed = minimumSamples(guarantee.alpha, guarantee.delta);
    const sampledCal = options.records.filter((r) => r.role === 'calibration' && r.source.kind === 'sampled' && (r.labels[head] === 0 || r.labels[head] === 1));
    const calDesign = sampledCal.length ? designOf(sampledCal) : null;
    const piOf = new Map(sampledCal.map((r, k) => [r.id, calDesign.inclusionProbs[k]]));
    for (const axis of names.filter((a) => axes.axes[a].observable)) {
        for (const value of axes.axes[axis].values) {
            const cal = records.filter((r) => r.role === 'calibration' && r.labels[head] === 1 && tagOf(r, axis) === value);
            let available;
            if (cal.length && cal.every((r) => piOf.has(r.id))) {
                // A sampled frame has one item per group; under unequal weights count the Kish effective number.
                available = kishEffectiveN(cal.map((r) => 1 / piOf.get(r.id)));
            }
            else
                available = new Set(cal.map((r) => r.group)).size;
            report.slices.push({ axis, value, needed, available, shortfall: Math.max(0, Math.ceil(needed - available)) });
        }
    }
    return report;
}
/** Throws unless every slice axis is observable at prediction time. */
export function checkSliceAxes(axes, sliceAxes) {
    for (const a of sliceAxes) {
        const axis = axes.axes[a];
        if (!axis)
            throw new Error(`unknown axis ${a}`);
        if (!axis.observable)
            throw new Error(`axis ${a} is not observable at prediction time, so it can't be a slice`);
    }
}
