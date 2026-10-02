/** A Markdown report of an artifact's gates and per-head evaluation, for humans and review. */
import type { ClassifierArtifact, HeadSpec } from './artifact.ts';
import type { HeadEvaluation } from './evaluate.ts';

// The evaluation comes from storage (typed unknown on the artifact), so every field may be missing.
const fmt = (digits: number) => (v: unknown) => (typeof v !== 'number' ? 'n/a' : Number.isNaN(v) ? 'nan' : v.toFixed(digits));
const f3 = fmt(3);
const f4 = fmt(4);
const ci = (c: unknown) => (Array.isArray(c) ? `${f3(c[0])}–${f3(c[1])}` : 'n/a');

export function reportMarkdown(artifact: ClassifierArtifact, options: { title?: string; baselineName?: string } = {}): string {
  const { title = `Model ${artifact.version}`, baselineName = 'baseline' } = options;
  const evaluation = (artifact.evaluation ?? {}) as Record<string, Partial<HeadEvaluation> | undefined>;
  const lines = [`# ${title}`, '', `Gates: **${artifact.gates?.passed ? 'PASSED' : 'FAILED'}**`, ''];
  for (const failure of artifact.gates?.failures ?? []) lines.push(`- ${failure}`);
  const warnings = artifact.gates?.warnings ?? [];
  if (warnings.length) lines.push('', '**Warnings**', '', ...warnings.map((w) => `- ${w}`));
  lines.push(
    '',
    '> **Reading ECE for rare classes:** after reweighting to production prevalence, easy negatives carry almost all ' +
      'the weight, so ECE can look near-perfect while probabilities for the rare class are off. Check the ' +
      "reliability table's upper bins, not just the ECE.",
  );
  for (const [head, ev] of Object.entries(evaluation)) {
    const spec = (artifact.heads as Record<string, HeadSpec | undefined>)[head];
    if (!spec || !ev) continue;
    lines.push(
      '', `## ${head}`, '',
      `threshold ${f4(spec.threshold)} · review floor ${f4(spec.review_floor)} · calibration ${spec.calibration?.method ?? 'n/a'}`, '',
      '| metric | value |', '|---|---|',
      `| test positives / n | ${ev.positives ?? 'n/a'} / ${ev.n ?? 'n/a'} (from ${ev.positive_groups ?? 'n/a'} distinct group(s)) |`,
      `| recall (95% CI) | ${f3(ev.recall)} (${ci(ev.recall_ci95)}) |`,
      `| false-alarm rate | ${f3(ev.false_alarm_rate)} |`,
      `| precision (prevalence-weighted) | ${f3(ev.precision_prevalence_weighted)} |`,
      `| ECE (prevalence-weighted) | ${f4(ev.ece_prevalence_weighted)} |`,
    );
    if (ev.vs_baseline) {
      lines.push(
        `| **recall with the ${baselineName} (${baselineName} OR classifier)** | **${f3(ev.combined_recall)} (${ci(ev.combined_recall_ci95)})** |`,
        `| false-alarm rate with the ${baselineName} | ${f3(ev.combined_false_alarm_rate)} |`,
        '', `vs ${baselineName} (test positives): ` + Object.entries(ev.vs_baseline).map(([k, v]) => `${k} ${v}`).join(', '),
      );
    }
    lines.push('', '| slice | positives | recall (95% CI) |', '|---|---|---|');
    for (const [name, s] of Object.entries(ev.slices ?? {})) lines.push(`| ${name} | ${s.positives} | ${f3(s.recall)} (${ci(s.recall_ci95)}) |`);
    lines.push('', '| bin | n | mean predicted | observed |', '|---|---|---|---|');
    for (const row of ev.reliability ?? []) lines.push(`| ${row.bin} | ${row.n} | ${row.mean_predicted} | ${row.observed_rate} |`);
  }
  return lines.join('\n') + '\n';
}
