// The evaluation comes from storage (typed unknown on the artifact), so every field may be missing.
const fmt = (digits) => (v) => (typeof v !== 'number' ? 'n/a' : Number.isNaN(v) ? 'nan' : v.toFixed(digits));
const f3 = fmt(3);
const f4 = fmt(4);
const ci = (c) => (Array.isArray(c) ? `${f3(c[0])}–${f3(c[1])}` : 'n/a');
function coverageLines(c) {
    const out = [`## Coverage: ${c.head}`, '', 'Counts are real, human-labelled records; tags are approximate and used only here.'];
    out.push('', `**Gaps** (value pairs below the per-cell minimum): ${c.gaps.length}`, '', '| a | b | positives | negatives |', '|---|---|---|---|', ...c.gaps.map((g) => `| ${g.a} | ${g.b} | ${g.positives} | ${g.negatives} |`));
    out.push('', '**Slice requirements** (observable axes; positive groups for a zero-miss guarantee)', '', '| slice | needed | available in calibration | shortfall |', '|---|---|---|---|', ...c.slices.map((s) => `| ${s.axis}=${s.value} | ${s.needed} | ${f3(s.available)} | ${s.shortfall} |`));
    return out;
}
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
        const d = ev.design;
        if (d) {
            const fmtEst = (e) => e ? `${f3(e.estimate)} (${ci(e.ci95)}${e.bootstrap_ci95 ? `; bootstrap ${ci(e.bootstrap_ci95)}` : ''})` : 'n/a';
            lines.push('', '**Design-based estimates** (stratified probability sample; intervals are approximate)', '', '| metric | estimate (95% interval) |', '|---|---|', `| recall | ${fmtEst(d.recall)} |`, `| precision at production prevalence | ${fmtEst(d.precision)} |`, `| false-alarm rate | ${fmtEst(d.false_alarm_rate)} |`, `| prevalence | ${fmtEst(d.prevalence)} |`, `| effective positives (Kish) | ${f3(d.effective_positives)} |`, ...Object.entries(d.slices ?? {}).map(([k, e]) => `| recall, ${k} | ${fmtEst(e)} |`));
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
    const training = (artifact.training ?? {});
    if (training.design?.designs?.length) {
        lines.push('', '## Sampling designs', '', 'Calibration and test are stratified probability samples of real traffic; estimates are Horvitz-Thompson and their intervals design-based and approximate.');
        for (const d of training.design.designs) {
            lines.push('', `Design ${d.designId} (scores from ${d.scoringModel})`, '', '| stratum | N | n |', '|---|---|---|', ...d.strata.map((s) => `| ${s.name} | ${s.N} | ${s.n} |`));
        }
    }
    const prov = training.provenance;
    if (prov) {
        lines.push('', '## Provenance', '', `Generated training data: ${prov.generated ? '**yes** - shadow candidate only until acceptance evidence exists' : 'no'}. Near-duplicates dropped (P5): ${prov.dropped?.length ?? 0}.`);
        for (const o of prov.overridden ?? [])
            lines.push(`- **${o.code} overridden** for ${o.ids.length} record(s) - not releasable`);
    }
    if (options.coverage)
        lines.push('', ...coverageLines(options.coverage));
    return lines.join('\n') + '\n';
}
