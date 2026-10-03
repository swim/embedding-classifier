// The evaluation comes from storage (typed unknown on the artifact), so every field may be missing.
const fmt = (digits) => (v) => (typeof v !== 'number' ? 'n/a' : Number.isNaN(v) ? 'nan' : v.toFixed(digits));
const f3 = fmt(3);
const f4 = fmt(4);
const ci = (c) => (Array.isArray(c) ? `${f3(c[0])}–${f3(c[1])}` : 'n/a');
export function reportMarkdown(artifact, options = {}) {
    const { title = `Model ${artifact.version}`, baselineName = 'baseline' } = options;
    const evaluation = (artifact.evaluation ?? {});
    const lines = [`# ${title}`, '', `Gates: **${artifact.gates?.passed ? 'PASSED' : 'FAILED'}**`, ''];
    for (const failure of artifact.gates?.failures ?? [])
        lines.push(`- ${failure}`);
    const warnings = artifact.gates?.warnings ?? [];
    if (warnings.length)
        lines.push('', '**Warnings**', '', ...warnings.map((w) => `- ${w}`));
    lines.push('', '> **Reading ECE for rare classes:** after reweighting to production prevalence, easy negatives carry almost all ' +
        'the weight, so ECE can look near-perfect while probabilities for the rare class are off. Check the ' +
        "reliability table's upper bins, not just the ECE.");
    for (const [head, ev] of Object.entries(evaluation)) {
        const spec = artifact.heads[head];
        if (!spec || !ev)
            continue;
        lines.push('', `## ${head}`, '', `threshold ${f4(spec.threshold)} · review floor ${f4(spec.review_floor)} · calibration ${spec.calibration?.method ?? 'n/a'}`, '', '| metric | value |', '|---|---|', `| test positives / n | ${ev.positives ?? 'n/a'} / ${ev.n ?? 'n/a'} (from ${ev.positive_groups ?? 'n/a'} distinct group(s)) |`, `| recall (95% CI) | ${f3(ev.recall)} (${ci(ev.recall_ci95)}) |`, `| false-alarm rate | ${f3(ev.false_alarm_rate)} |`, `| precision (prevalence-weighted) | ${f3(ev.precision_prevalence_weighted)} |`, `| ECE (prevalence-weighted) | ${f4(ev.ece_prevalence_weighted)} |`);
        const c = ev.certified;
        if (c) {
            lines.push(`| certified recall ≥ (test, exact) | ${f3(c.recall_lower)} over ${c.positive_groups} positive group(s) |`, `| certified false-alarm rate ≤ (test, exact) | ${f4(c.false_alarm_upper)} over ${c.negative_groups} negative group(s) |`, ...(c.precision_lower !== undefined ? [`| certified precision ≥ (at production prevalence) | ${f3(c.precision_lower)} |`] : []), ...(ev.background_rate_upper !== undefined ? [`| certified background rate ≤ | ${f4(ev.background_rate_upper)} |`] : []));
        }
        if (c)
            lines.push('', `Certified bounds hold together with probability ${f3(1 - c.delta)}, whatever chose the threshold.`);
        const g = ev.guarantee;
        if (g) {
            const what = g.kind === 'pac' ? `recall ≥ ${f3(1 - g.alpha)} with probability ${f3(1 - (g.delta ?? 0))}` : g.kind === 'expected' ? `recall ≥ ${f3(1 - g.alpha)} in expectation over calibration draws` : 'nothing (no conformal guarantee)';
            const budgets = [g.false_alarm !== undefined ? `calibration false alarms ≤ ${g.false_alarm}` : '', g.background_rate !== undefined ? `background rate ≤ ${g.background_rate}` : ''].filter(Boolean);
            lines.push('', `Threshold mode **${g.mode}**${ev.sufficiency ? ` (used: ${ev.sufficiency.chosen})` : ''}: guarantees ${what}${budgets.length ? `, with ${budgets.join(' and ')}` : ''}.`);
            const s = ev.sufficiency;
            if (s) {
                lines.push(`Calibration positive groups: ${s.positive_groups} (expected guarantee needs ${s.needed.expected}, PAC ${s.needed.pac}); feasible: ${s.feasible.join(', ') || 'none'}${s.reason ? `. ${s.reason}` : ''}.`);
                for (const [slice, v] of Object.entries(s.slices ?? {}))
                    lines.push(`- ${slice}: ${v.positive_groups} positive group(s), per-slice guarantee feasible: ${v.feasible.join(', ') || 'none'}`);
            }
        }
        if (spec.review_epsilon !== undefined)
            lines.push('', `Review floor is conformal: at most ${spec.review_epsilon} of positives score below it, in expectation.`);
        if (ev.vs_baseline) {
            lines.push(`| **recall with the ${baselineName} (${baselineName} OR classifier)** | **${f3(ev.combined_recall)} (${ci(ev.combined_recall_ci95)})** |`, `| false-alarm rate with the ${baselineName} | ${f3(ev.combined_false_alarm_rate)} |`, '', `vs ${baselineName} (test positives): ` + Object.entries(ev.vs_baseline).map(([k, v]) => `${k} ${v}`).join(', '));
        }
        const weak = (artifact.training?.weak_labels ?? {})[head];
        if (weak) {
            lines.push('', `Weak positives (train only): ${weak.count ?? 'n/a'}, weight ${f3(weak.weight)} beside ${weak.gold_train_positives ?? 'n/a'} gold ` +
                `(cap ${weak.max_weak_share ?? 'n/a'}×, scale ${f3(weak.scale)})` + (weak.rule_set_version ? ` from rule set ${weak.rule_set_version} (${String(weak.rule_set_hash).slice(0, 12)})` : ''));
            const rules = Object.entries(weak.by_rule ?? {});
            if (rules.length) {
                lines.push('', '| rule | weak examples | weight |', '|---|---|---|');
                for (const [rule, r] of rules)
                    lines.push(`| ${rule} | ${r.count} | ${f3(r.weight)} |`);
            }
        }
        lines.push('', '| slice | positives | recall (95% CI) |', '|---|---|---|');
        for (const [name, s] of Object.entries(ev.slices ?? {}))
            lines.push(`| ${name} | ${s.positives} | ${f3(s.recall)} (${ci(s.recall_ci95)}) |`);
        lines.push('', '| bin | n | mean predicted | observed |', '|---|---|---|---|');
        for (const row of ev.reliability ?? [])
            lines.push(`| ${row.bin} | ${row.n} | ${row.mean_predicted} | ${row.observed_rate} |`);
    }
    return lines.join('\n') + '\n';
}
