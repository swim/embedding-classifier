/**
 * Fit, calibrate, threshold, evaluate and gate every head - everything short of I/O.
 *
 *   train split        weighted L2 logistic regression per head (class-balanced, exact Newton solver),
 *                      plus any weak positives (e.g. from certified rules), weighted and capped
 *   calibration split  Platt or isotonic calibrator, weighted to the head's production prevalence,
 *                      then the threshold from the head's policy - heuristic or conformal (see
 *                      conformal.ts) - and optionally a conformal review floor
 *   test split         evaluateHead (with exact certified bounds) + gateHead
 *
 * With `records` (one ExampleRecord per row of X), provenance rules P1-P7 are enforced first,
 * training weights come from the records with retrieved and generated data capped (P6), and the
 * calibration and test splits are probability samples: calibration is weighted by 1/π (which
 * already reproduces production prevalence, so `prevalence` must not be passed), and the test
 * metrics become design-based estimates. The `design` threshold mode then applies.
 */
import { clopperPearsonUpper, conformalLowerThreshold, conformalRank, decisionFunction, designPrecisionThreshold, designRiskThreshold, fitIsotonic, fitLogistic, fitPlatt, minimumSamples, nextUp, prevalenceWeights, sigmoid } from '@liquidau/solvers';

import { designOf, type DesignSummary } from './design.ts';
import { capTrainingWeights, validateProvenance, type ExampleRecord, type ProvenanceCode, type ProvenanceOptions, type WeightCaps, type WeightCapSummary } from './records.ts';
import { conformalThreshold, groupScores, type FalseAlarmConstraint, type Guarantee, type Sufficiency } from './conformal.ts';
import { calibrate, scoreEmbedding, validateArtifact, type Calibration, type ClassifierArtifact, type EmbeddingSpec, type HeadSpec, type RouterTraining } from './artifact.ts';
import { evaluateHead, type HeadEvaluation } from './evaluate.ts';
import { gateHead } from './gates.ts';
import { crossValidatedScores, decodeReference, encodeReference, fitFeatures, guaranteeCost, HEAD_TYPES, type FeatureTraining, type FittedFeatures, type HeadType, type ReferenceSet } from './heads.ts';
import { budgetThreshold, falseAlarmCap, pickThreshold, type HeadPolicy } from './threshold.ts';

export const SPLITS = ['train', 'calibration', 'test'] as const;
export type Split = (typeof SPLITS)[number];

export interface HeadInput<H extends string> {
  name: H;
  /** Binary target per example, aligned with X; null leaves the example out of this head. */
  y: ReadonlyArray<0 | 1 | null>;
  policy: HeadPolicy;
  /**
   * Expected positive rate in production - calibration and precision are weighted to it. Required
   * without `records`; must be omitted with them (design weights already reproduce prevalence).
   */
  prevalence?: number;
  /** Generated records labelled for this head must each be human-verified (P7). Recorded in the provenance summary. */
  safetyCritical?: boolean;
  /** Whether an existing mechanism already catches each example (see evaluateHead). */
  baseline?: readonly boolean[];
  /**
   * The head's type (default 'auto'): 'auto' picks linear, knn or stack by cross-validated guarantee
   * cost on the training rows (recorded in result.headChoice; see autoMargin). 'knn' and 'stack' score
   * against the artifact's shared reference of training embeddings (heads.ts; buildArtifact stores it),
   * so such an artifact holds training data. Calibration, thresholds and guarantees are the same for
   * every type. Use 'linear' when X isn't an embedding (e.g. stacked scores) or the artifact must not
   * hold training embeddings. Weak positives are linear-only: with them, 'auto' stays linear.
   */
  type?: HeadType | 'auto';
  /**
   * Certified dismissal rules in front of this head (rule-miner's mineDismissals and
   * certifyDismissals): `rows` marks the rows of X a rule cleared (and `background` the rows of
   * background.X). Cleared calibration, test and background rows score 0, so a positive a rule
   * dismissed counts as a miss: the head's recall guarantee covers rules and classifier together.
   * Training is unchanged; calibration is fitted on the rows the rules leave to the classifier.
   */
  dismissal?: { rows: readonly boolean[]; background?: readonly boolean[]; ruleSet: string; maxRate: number; certified: number };
  /**
   * Extra POSITIVES for the train split only, e.g. rule-miner's weakLabels(): embeddings with a
   * weight in [0, 1] each. They never reach calibration, test or the background budget, and an
   * embedding identical to a calibration, test or background row is refused as leakage.
   */
  weak?: WeakInput;
  /**
   * Cap on total weak weight as a multiple λ of the head's gold train positives (default 0.5): weak
   * weights are scaled down to fit. Class balancing is sample-weighted, so weak positives take a
   * share λ / (1 + λ) of the positive class's weight rather than adding to it.
   */
  maxWeakShare?: number;
}

export interface WeakInput {
  X: ReadonlyArray<ArrayLike<number>>;
  weights: readonly number[];
  /** Which rule produced each example, for the per-rule report. */
  rules?: readonly string[];
  /** The rule set the weak labels came from, recorded for audit. */
  source?: { rule_set_version: string; rule_set_hash: string };
}

/** What weak supervision contributed to one head - store as artifact.training.weak_labels. */
export interface WeakSummary {
  count: number;
  gold_train_positives: number;
  max_weak_share: number;
  weight_before_cap: number;
  /** After the cap: at most max_weak_share × gold_train_positives. */
  weight: number;
  /** Multiplier the cap applied to every weak weight (1 = no cap). */
  scale: number;
  by_rule?: Record<string, { count: number; weight: number }>;
  rule_set_version?: string;
  rule_set_hash?: string;
}

export interface TrainInput<H extends string> {
  X: ReadonlyArray<ArrayLike<number>>;
  split: readonly Split[];
  heads: ReadonlyArray<HeadInput<H>>;
  /** Inverse L2 strength (default 1). */
  C?: number;
  calibration?: 'platt' | 'isotonic';
  /** review_floor = threshold × reviewRatio (default 0.5). */
  reviewRatio?: number;
  /**
   * Conformal review floor instead of reviewRatio: the floor below which at most this share of
   * positives fall, in expectation (calibration positive groups; capped at the threshold). Messages
   * below it are dismissed automatically with that stated miss rate.
   */
  reviewEpsilon?: number;
  maxEce?: number;
  /**
   * Fail a head whose test-split probabilities Cox's recalibration test rejects at this level (default
   * off). Prefer it to maxEce for rare classes: in simulation the ECE > 0.05 gate never fired at 2%
   * prevalence, even for badly miscalibrated probabilities.
   */
  calibrationAlpha?: number;
  groups?: readonly string[];
  slices?: Readonly<Record<string, readonly string[]>>;
  /**
   * Ordinary, mostly-negative traffic (e.g. everyday messages): each listed head's threshold is
   * raised until it fires on at most maxRate of it - BEFORE test evaluation, so the gates judge
   * the threshold that will ship. Real traffic holds positives at production prevalence, so
   * maxRate must sit above prevalence × the recall you want (8% for a 4% head), or the budget
   * caps recall rather than false alarms.
   */
  background?: {
    X: ReadonlyArray<ArrayLike<number>>;
    maxRate: Partial<Record<H, number>>;
    /** With `records`: one per row of background.X - unlabelled traffic with backgroundUse 'budget' (P2). */
    records?: readonly ExampleRecord[];
  };
  /**
   * One record per row of X: role must equal split, and labels[head] must equal each head's y.
   * Turns on provenance enforcement, record weights with P6 caps, and design-based calibration
   * and evaluation (see the module comment).
   */
  records?: readonly ExampleRecord[];
  /** The designSample summaries behind the sampled records - recorded in result.design. */
  designs?: readonly DesignSummary[];
  /** Overrides, accepted batches and the near-duplicate cosine for validateProvenance. */
  provenance?: Omit<ProvenanceOptions, 'embeddings' | 'safetyCritical'>;
  /** P6 caps; ruleMatches is required when generated hard negatives are present. */
  caps?: WeightCaps;
  /**
   * Embeddings for the provenance near-duplicate check (P5), aligned with X then background.X
   * (default: X and background.X). Pass them when X holds other features - stacked scores, or
   * embeddings with extra columns - since cosine similarity between those isn't about the text.
   */
  provenanceEmbeddings?: { X: ReadonlyArray<ArrayLike<number>>; background?: ReadonlyArray<ArrayLike<number>> };
  /** Seed for design bootstraps and the stack head's principal components (default 0). */
  seed?: number;
  /**
   * Cross-fitting folds for knn, stack and auto heads: rows with the same key share a fold (default:
   * `groups`, else the row index). Keep near-duplicates together, or a training row's out-of-fold
   * score still finds its twin.
   */
  foldKeys?: readonly string[];
  /**
   * type 'auto': a knn or stack head is chosen only when its cross-validated cost is below the linear
   * head's × (1 − autoMargin) (default 0.2), then the cheaper of the two. Cross-validated costs on a
   * small training split are noisy; without a margin, near-ties often pick a non-linear head that does
   * worse on new traffic. A relative margin keeps the clear wins and drops the near-ties.
   */
  autoMargin?: number;
  /**
   * A fit that stopped short of its tolerance (the head's logistic model, a stack's linear component,
   * or the Platt calibrator - the likeliest, on a small calibration split the scores separate) is a
   * warning by default; true makes it a gate failure. result.convergence records every head's fits.
   */
  requireConvergence?: boolean;
  /** Encoding of the knn reference (default 'f32'): 'int8' is about 4× smaller, with essentially the same results. */
  referenceEncoding?: 'f32' | 'int8';
  log?: (line: string) => void;
}

/** What provenance enforcement did - store as artifact.training.provenance (with generated, see publishPlan). */
export interface ProvenanceSummary {
  dropped: Array<{ id: string; code: 'P5'; reason: string }>;
  overridden: Array<{ code: ProvenanceCode; ids: string[]; message: string }>;
  /** Any generated record was trained on: the artifact may only be a shadow candidate until acceptance evidence exists. */
  generated: boolean;
  safety_critical: string[];
  heads: Record<string, WeightCapSummary>;
}

export interface TrainResult<H extends string> {
  heads: Partial<Record<H, HeadSpec>>;
  evaluation: Partial<Record<H, HeadEvaluation>>;
  /** Gate failures, prefixed with the head name. Empty means every gate passed. */
  failures: string[];
  warnings: string[];
  /** Test-split probabilities per head (example indices into X), for round-trip checks. */
  testProbabilities: Partial<Record<H, { idx: number[]; p: number[] }>>;
  /** Heads trained with weak positives - store as artifact.training.weak_labels so a release can be audited. */
  weakLabels: Partial<Record<H, WeakSummary>>;
  /** With records: provenance enforcement - store as artifact.training.provenance. */
  provenance?: ProvenanceSummary;
  /** Training embeddings for knn and stack heads - store as artifact.reference (scoreEmbedding needs it). */
  reference?: ReferenceSet;
  /** Whether each head's fits converged: its model (and a stack's linear component), and its Platt calibrator. */
  convergence: Partial<Record<H, { model: boolean; calibration: boolean }>>;
  /** Heads trained with type 'auto': each type's cross-validated cost and the choice - store as artifact.training.head_choice. */
  headChoice: Partial<Record<H, HeadChoice>>;
  /** With sampled records: the designs and each head's estimator - store as artifact.training.design. */
  design?: { designs: DesignSummary[]; heads: Record<string, { calibration: 'design-weighted'; threshold: string; evaluation: 'design-linearised' }> };
}

export interface HeadChoice {
  chosen: HeadType;
  /** Out-of-fold guarantee cost per type on the training rows (lower is better). */
  costs: Record<HeadType, number>;
  criterion: string;
}

function checkLength(name: string, values: { length: number } | undefined, n: number): void {
  if (values !== undefined && values.length !== n) throw new Error(`${name} has ${values.length} entries but X has ${n}`);
}

function checkedLength<T>(name: string, values: readonly T[], n: number): readonly T[] {
  checkLength(name, values, n);
  return values;
}

function indicesBySplit(split: readonly Split[], y: ReadonlyArray<0 | 1 | null>): Record<Split, number[]> {
  const idx: Record<Split, number[]> = { train: [], calibration: [], test: [] };
  split.forEach((s, i) => {
    if (y[i] !== null) idx[s].push(i);
  });
  return idx;
}

export function trainHeads<H extends string>(input: TrainInput<H>): TrainResult<H> {
  const { X, split, C = 1, autoMargin = 0.2, requireConvergence = false, calibration: method = 'platt', reviewRatio = 0.5, reviewEpsilon, maxEce, calibrationAlpha, groups, slices, background, records, seed = 0, log = () => {} } = input;
  checkLength('split', split, X.length);
  split.forEach((s, i) => {
    if (!(SPLITS as readonly string[]).includes(s)) throw new Error(`split[${i}] is "${s}", expected one of ${SPLITS.join(', ')}`);
  });
  checkLength('groups', groups, X.length);
  for (const [field, values] of Object.entries(slices ?? {})) checkLength(`slices.${field}`, values, X.length);
  if (background && Object.keys(background.maxRate).length && background.X.length === 0) throw new Error('background.X must be non-empty when a maxRate is set');
  if (!(reviewRatio >= 0 && reviewRatio <= 1)) throw new Error('reviewRatio must be between 0 and 1');
  if (!(autoMargin >= 0 && autoMargin < 1)) throw new Error('autoMargin must be in [0, 1)');
  if (reviewEpsilon !== undefined && !(reviewEpsilon > 0 && reviewEpsilon < 1)) throw new Error('reviewEpsilon must be strictly between 0 and 1');
  const result: TrainResult<H> = { heads: {}, evaluation: {}, failures: [], warnings: [], testProbabilities: {}, weakLabels: {}, headChoice: {}, convergence: {} };
  checkLength('foldKeys', input.foldKeys, X.length);
  const foldKey = (i: number) => input.foldKeys?.[i] ?? groups?.[i] ?? `#${i}`;

  // Provenance: validate every record, drop P5 near-duplicates from training, record overrides.
  const excluded = new Set<number>();
  if (records) {
    checkLength('records', records, X.length);
    records.forEach((r, i) => {
      if (r.role !== split[i]) throw new Error(`records[${i}] (${r.id}) has role ${r.role} but split[${i}] is ${split[i]}`);
    });
    const bgRecords = background?.records ?? [];
    if (background?.records) {
      checkLength('background.records', background.records, background.X.length);
      background.records.forEach((r, i) => {
        if (r.role !== 'background' || r.backgroundUse !== 'budget') throw new Error(`background.records[${i}] (${r.id}) must have role background and backgroundUse budget`);
      });
    } else if (background && Object.keys(background.maxRate).length) {
      throw new Error('with records, background.records is required so the budget traffic can be checked (P2, P3)');
    }
    const safety = input.heads.filter((h) => h.safetyCritical).map((h) => h.name as string);
    const checked = validateProvenance([...records, ...bgRecords], { ...input.provenance, embeddings: input.provenanceEmbeddings
      ? [...checkedLength('provenanceEmbeddings.X', input.provenanceEmbeddings.X, X.length), ...(input.provenanceEmbeddings.background ?? background?.X ?? [])]
      : [...X, ...(background?.X ?? [])], safetyCritical: safety });
    const droppedIds = new Set(checked.dropped.map((d) => d.id));
    records.forEach((r, i) => { if (droppedIds.has(r.id)) excluded.add(i); });
    result.warnings.push(...checked.warnings.map((w) => `provenance: ${w}`));
    result.failures.push(...checked.overridden.map((o) => `provenance: ${o.code} overridden for ${o.ids.length} record(s) - not releasable`));
    result.provenance = {
      dropped: checked.dropped, overridden: checked.overridden, safety_critical: safety, heads: {},
      generated: records.some((r, i) => r.source.kind === 'generated' && r.role === 'train' && !excluded.has(i)),
    };
    result.design = { designs: [...(input.designs ?? [])], heads: {} };
  }
  const vectorKey = (x: ArrayLike<number>) => Array.from(x).join(',');
  let heldOut: Set<string> | null = null;
  const isHeldOut = (x: ArrayLike<number>) => {
    heldOut ??= new Set([...X.filter((_, i) => split[i] !== 'train'), ...(background?.X ?? [])].map(vectorKey));
    return heldOut.has(vectorKey(x));
  };

  // The shared reference for knn, stack and auto heads: training rows labelled for any of them.
  // Generated records (synthetic text) are left out; retrieved ones are human-labelled and stay.
  // 'auto' heads with weak positives stay linear (weak labels are linear-only).
  const nonLinear = input.heads.filter((h) => (h.type ?? 'auto') !== 'linear' && !((h.type ?? 'auto') === 'auto' && h.weak?.X.length));
  let reference: { set: ReferenceSet; rows: Float32Array[]; index: number[] } | null = null;
  if (nonLinear.length) {
    for (const h of nonLinear) {
      const t = h.type ?? 'auto';
      if (!(t === 'auto' || HEAD_TYPES.includes(t as HeadType))) throw new Error(`${h.name}: unknown head type ${t}`);
      checkLength(`${h.name}: y`, h.y, X.length);
    }
    const generated = new Set(records ? records.flatMap((r, i) => (r.source.kind === 'generated' ? [i] : [])) : []);
    const index = X.map((_, i) => i).filter((i) => split[i] === 'train' && !excluded.has(i) && !generated.has(i) && nonLinear.some((h) => h.y[i] === 0 || h.y[i] === 1));
    const leftOut = X.filter((_, i) => split[i] === 'train' && generated.has(i)).length;
    if (leftOut) result.warnings.push(`reference: ${leftOut} generated training record(s) left out of the knn reference`);
    const set = encodeReference(index.map((i) => X[i]), Object.fromEntries(nonLinear.map((h) => [h.name, index.map((i) => (h.y[i] === 0 || h.y[i] === 1 ? h.y[i] : null))])), input.referenceEncoding ?? 'f32');
    reference = { set, rows: decodeReference(set), index };
  }

  for (const { name, y: yIn, policy, prevalence, baseline, weak, maxWeakShare = 0.5, type: typeIn = 'auto', dismissal } of input.heads) {
    const type = typeIn === 'auto' && weak?.X.length ? 'linear' : typeIn;
    if (type !== typeIn) result.warnings.push(`${name}: weak positives are linear-only, so type 'auto' kept the linear head`);
    checkLength(`${name}: y`, yIn, X.length);
    checkLength(`${name}: dismissal.rows`, dismissal?.rows, X.length);
    if (dismissal?.background && background) checkLength(`${name}: dismissal.background`, dismissal.background, background.X.length);
    if (dismissal && !(dismissal.maxRate > 0 && dismissal.maxRate < 1)) throw new Error(`${name}: dismissal.maxRate must be in (0, 1)`);
    const cleared = (i: number) => !!dismissal?.rows[i];
    checkLength(`${name}: baseline`, baseline, X.length);
    if (records) {
      if (prevalence !== undefined) throw new Error(`${name}: don't pass prevalence with sampled records - calibration is weighted by 1/π, which already reproduces production prevalence; weighting to prevalence as well would count it twice`);
      records.forEach((r, i) => {
        if ((r.labels[name] ?? null) !== yIn[i]) throw new Error(`${name}: y[${i}] is ${yIn[i]} but records[${i}] (${r.id}) is labelled ${r.labels[name] ?? null}`);
      });
    } else if (!(prevalence !== undefined && prevalence > 0 && prevalence < 1)) throw new Error(`${name}: prevalence must be strictly between 0 and 1, got ${prevalence}`);
    const y = excluded.size ? yIn.map((v, i) => (excluded.has(i) ? null : v)) : yIn;
    if (weak) {
      checkLength(`${name}: weak.weights`, weak.weights, weak.X.length);
      checkLength(`${name}: weak.rules`, weak.rules, weak.X.length);
      if (!(maxWeakShare >= 0)) throw new Error(`${name}: maxWeakShare must be non-negative, got ${maxWeakShare}`);
      weak.X.forEach((x, j) => {
        if (x.length !== X[0]?.length) throw new Error(`${name}: weak.X[${j}] has ${x.length} dimensions but X has ${X[0]?.length}`);
        if (!(weak.weights[j] >= 0 && weak.weights[j] <= 1)) throw new Error(`${name}: weak.weights[${j}] must be in [0, 1], got ${weak.weights[j]}`);
        if (isHeldOut(x)) throw new Error(`${name}: weak.X[${j}] is a calibration, test or background example - weak labels are for the train split only`);
      });
    }
    const idx = indicesBySplit(split, y);
    for (const s of SPLITS) {
      if (new Set(idx[s].map((i) => y[i])).size < 2) throw new Error(`${name}: the ${s} split needs both positive and negative examples`);
    }
    const ySplit = (s: Split) => idx[s].map((i) => y[i] as number);
    log(`${name}: ` + SPLITS.map((s) => `${s} ${ySplit(s).reduce((a, b) => a + b, 0)}/${idx[s].length} positive`).join(', '));

    // Training weights: 1 per row, or the records' weights with P6 caps.
    let trainWeights = idx.train.map(() => 1);
    if (records) {
      const capped = capTrainingWeights(idx.train.map((i) => records[i]), name, input.caps);
      trainWeights = capped.weights;
      result.provenance!.heads[name] = capped.summary;
    }
    // Head type: knn and stack heads learn their layer on out-of-fold features of the training rows.
    let fitted: FittedFeatures | null = null;
    if (type !== 'linear') {
      if (weak && weak.X.length) throw new Error(`${name}: weak positives are supported for linear heads only`);
      const t: FeatureTraining = {
        rows: idx.train, y: ySplit('train') as Array<0 | 1>, weights: trainWeights, keys: idx.train.map(foldKey), X,
        reference: reference!.rows, referenceKeys: reference!.index.map(foldKey), referenceLabels: reference!.set.labels[name], C, seed,
      };
      let chosen: HeadType = type === 'auto' ? 'linear' : type;
      if (type === 'auto') {
        const costs = {} as Record<HeadType, number>, fits = {} as Record<HeadType, FittedFeatures>;
        for (const ht of HEAD_TYPES) { fits[ht] = fitFeatures(ht, t); costs[ht] = guaranteeCost(crossValidatedScores(fits[ht], t), t.y, policy); }
        // Leave linear only for a clear cross-validated gain (autoMargin); ties go to the simpler type.
        const bar = costs.linear * (1 - autoMargin);
        chosen = HEAD_TYPES.filter((ht) => ht !== 'linear' && costs[ht] < bar - 1e-12).reduce<HeadType>((a, b) => (a === 'linear' || costs[b] < costs[a] - 1e-12 ? b : a), 'linear');
        fitted = chosen === 'linear' ? null : fits[chosen];
        const criterion = policy.kind === 'recall' ? `false alarms at recall ${policy.targetRecall}` : `recall lost at precision ${policy.targetPrecision}`;
        result.headChoice[name] = { chosen, costs, criterion: `${criterion}, 3-fold cross-validated on the training rows; non-linear only below linear × ${1 - autoMargin}` };
        log(`${name}: auto chose ${chosen} (${HEAD_TYPES.map((ht) => `${ht} ${costs[ht].toFixed(4)}`).join(', ')})`);
      } else fitted = fitFeatures(chosen, t);
    }
    const trainAt = new Map(idx.train.map((i, n) => [i, n]));
    const applied = new Map<number, number[]>();
    /** The head's input for row i of X: out-of-fold features for training rows. */
    const rowFeatures = (i: number): ArrayLike<number> => (!fitted ? X[i] : trainAt.has(i) ? fitted.train[trainAt.get(i)!] : applied.get(i) ?? applied.set(i, fitted.apply(X[i])).get(i)!);
    const features = (x: ArrayLike<number>): ArrayLike<number> => (fitted ? fitted.apply(x) : x);

    let model;
    if (fitted) {
      model = fitLogistic(fitted.train, ySplit('train'), {
        C, classWeight: 'balanced', ...(trainWeights.some((w) => w !== 1) ? { sampleWeight: trainWeights } : {}),
      });
    } else if (weak && weak.X.length) {
      const gold = ySplit('train').reduce((a, b) => a + b, 0);
      const before = weak.weights.reduce((a, b) => a + b, 0);
      const scale = before > maxWeakShare * gold ? (maxWeakShare * gold) / before : 1;
      const w = weak.weights.map((v) => v * scale);
      model = fitLogistic([...idx.train.map((i) => X[i]), ...weak.X], [...ySplit('train'), ...w.map(() => 1)], {
        C, classWeight: 'balanced', sampleWeight: [...trainWeights, ...w],
      });
      const summary: WeakSummary = { count: w.length, gold_train_positives: gold, max_weak_share: maxWeakShare, weight_before_cap: before, weight: w.reduce((a, b) => a + b, 0), scale };
      if (weak.rules) {
        summary.by_rule = {};
        weak.rules.forEach((rule, j) => {
          const r = (summary.by_rule![rule] ??= { count: 0, weight: 0 });
          r.count++;
          r.weight += w[j];
        });
      }
      if (weak.source) Object.assign(summary, weak.source);
      result.weakLabels[name] = summary;
      log(`${name}: ${w.length} weak positives, weight ${summary.weight.toFixed(2)} (${scale < 1 ? `capped at ${maxWeakShare}` : 'uncapped'}) beside ${gold} gold`);
    } else {
      model = fitLogistic(idx.train.map((i) => X[i]), ySplit('train'), {
        C, classWeight: 'balanced', ...(trainWeights.some((w) => w !== 1) ? { sampleWeight: trainWeights } : {}),
      });
    }
    const score = (x: ArrayLike<number>) => calibrate(calibration, decisionFunction(model, features(x)));

    const yCal = ySplit('calibration');
    // Sampled calibration: design weights 1/π (they reproduce production prevalence). Otherwise reweight to `prevalence`.
    const calDesign = records ? designOf(idx.calibration.map((i) => records[i])) : null;
    const wCal = calDesign ? calDesign.inclusionProbs.map((p) => 1 / p) : prevalenceWeights(yCal, prevalence!);
    const unequal = !!calDesign && calDesign.inclusionProbs.some((p) => Math.abs(p - calDesign.inclusionProbs[0]) > 1e-12);
    const calLogits = idx.calibration.map((i) => decisionFunction(model, rowFeatures(i)));
    // The calibrator sees the rows the dismissal rules leave to the classifier; cleared rows score 0.
    const kept = idx.calibration.map((i) => !cleared(i));
    const keep = <T>(v: readonly T[]) => v.filter((_, j) => kept[j]);
    let calibrationConverged = true;
    let calibration: Calibration;
    if (method === 'platt') {
      const { a, c, converged } = fitPlatt(keep(calLogits), keep(yCal), keep(wCal));
      calibration = { method: 'platt', a, c };
      calibrationConverged = converged;
    } else calibration = { method: 'isotonic', ...fitIsotonic(keep(calLogits).map(sigmoid), keep(yCal), keep(wCal), { yMin: 0, yMax: 1 }) };
    // Convergence: a fit that stopped short of its tolerance is reported (a failure with requireConvergence).
    const modelConverged = model.converged && (fitted?.converged ?? true);
    result.convergence[name] = { model: modelConverged, calibration: calibrationConverged };
    for (const [what, ok] of [['logistic model', modelConverged], ['Platt calibrator (the calibration scores may separate the classes)', calibrationConverged]] as const) {
      if (!ok) (requireConvergence ? result.failures : result.warnings).push(`${name}: the ${what} did not converge`);
    }
    const pCal = calLogits.map((z, j) => (kept[j] ? calibrate(calibration, z) : 0));
    const budget = background?.maxRate[name];
    const backgroundP = background && budget !== undefined ? background.X.map((x, j) => (dismissal?.background?.[j] ? 0 : score(x))) : null;
    const heuristic = () => {
      const t = pickThreshold(policy, pCal, yCal, calDesign ? wCal : undefined);
      return backgroundP && budget !== undefined ? budgetThreshold(t, backgroundP, budget) : t;
    };
    // Calibration scores per class, one per group: what every conformal statement counts.
    const calGroups = groups && idx.calibration.map((i) => groups[i]);
    const ofClass = (c: number) => <T>(values: readonly T[]) => values.filter((_, j) => yCal[j] === c);
    const positives = groupScores(ofClass(1)(pCal), calGroups && ofClass(1)(calGroups), Math.min);
    const delta = policy.delta ?? 0.05;

    let threshold!: number;
    let guarantee: Guarantee | undefined;
    let sufficiency: Sufficiency | undefined;
    // Every background rank a threshold could have been capped at - for a bound valid whichever applied.
    const backgroundRanks: number[] = [];
    if (backgroundP && budget !== undefined && Math.floor(budget * backgroundP.length) < backgroundP.length) backgroundRanks.push(Math.floor(budget * backgroundP.length));
    // Default: the strongest guarantee the data supports - 'design' with sampled calibration records,
    // else 'auto' (conformal) without the silent heuristic fallback. 'heuristic' only when chosen.
    const mode = policy.kind === 'recall' ? policy.mode ?? (calDesign ? 'design' : 'auto') : 'heuristic';
    const precisionMode = policy.kind === 'precision' ? policy.mode ?? (calDesign ? 'design' : undefined) : undefined;
    const allowHeuristic = policy.kind === 'recall'
      ? policy.allowHeuristicFallback ?? (policy.fallback !== undefined ? policy.fallback === 'heuristic' : policy.mode !== undefined)
      : false;
    if (policy.kind === 'recall' && policy.fallback === 'heuristic' && policy.designRecall === undefined) throw new Error(`${name}: fallback 'heuristic' needs designRecall (the heuristic threshold's target)`);
    if (unequal && (mode === 'conformal-expected' || mode === 'conformal-pac' || mode === 'auto' || reviewEpsilon !== undefined)) {
      throw new Error(`${name}: the calibration records have unequal inclusion probabilities, so they aren't exchangeable and conformal guarantees${reviewEpsilon !== undefined ? ' (including reviewEpsilon)' : ''} don't hold - use mode 'design'`);
    }
    if (policy.kind === 'recall' && mode === 'design') {
      if (!calDesign) throw new Error(`${name}: mode 'design' needs sampled calibration records (records)`);
      const designMethod = policy.designMethod ?? 'exact';
      const res = designRiskThreshold({ ...calDesign, y: yCal, scores: pCal, alpha: 1 - policy.targetRecall, delta, method: designMethod, seed, ...(policy.designMinStratumPositives !== undefined ? { minStratumPositives: policy.designMinStratumPositives } : {}) });
      const fails: string[] = [];
      result.warnings.push(...(res.warnings ?? []).map((w) => `${name}: design: ${w}`));
      if (res.feasible) {
        threshold = res.threshold;
        guarantee = { mode, kind: res.guarantee, alpha: 1 - policy.targetRecall, delta, method: designMethod };
        // Budgets may only raise the threshold; raising it past the recall guarantee fails the gate.
        const caps: Array<[string, number]> = [];
        if (policy.maxFalseAlarm !== undefined && policy.maxFalseAlarm < 1) {
          caps.push([`calibration false-alarm <= ${policy.maxFalseAlarm}`, falseAlarmCap(pCal, yCal, policy.maxFalseAlarm, wCal)]);
        }
        if (backgroundP && budget !== undefined) caps.push([`background <= ${budget}`, budgetThreshold(res.threshold, backgroundP, budget)]);
        for (const [what, t] of caps) {
          if (t > threshold!) {
            fails.push(`design: recall >= ${policy.targetRecall} needs a threshold <= ${res.threshold} but ${what} needs >= ${t}`);
            threshold = t;
            guarantee = { mode, kind: 'none', alpha: 1 - policy.targetRecall };
          }
        }
        log(`${name}: design threshold ${threshold} (${res.guarantee}, ${designMethod}, Kish effective positives ${res.nEff.toFixed(1)})`);
      } else if (policy.fallback === 'heuristic') {
        // Too little evidence for the guarantee, and the policy chose a heuristic threshold for that case.
        threshold = heuristic();
        guarantee = { mode, kind: 'none', alpha: 1 - policy.targetRecall, fallback: 'heuristic' };
        result.warnings.push(`${name}: design: ${res.reason}; heuristic threshold (fallback), no guarantee - a guarantee needs at least ${minimumSamples(1 - policy.targetRecall, delta)} effective calibration positives with no misses (have ${res.nEff.toFixed(1)})`);
      } else {
        fails.push(`design: ${res.reason}`);
        threshold = nextUp(1); // never fires: no threshold is substituted silently
        guarantee = { mode, kind: 'none', alpha: 1 - policy.targetRecall };
      }
      result.failures.push(...fails.map((f) => `${name}: ${f}`));
      if (backgroundP && budget !== undefined && Math.floor(budget * backgroundP.length) < backgroundP.length) backgroundRanks.push(Math.floor(budget * backgroundP.length));
      result.design!.heads[name] = { calibration: 'design-weighted', threshold: `design (${designMethod})`, evaluation: 'design-linearised' };
    } else if (policy.kind === 'recall' && mode !== 'heuristic') {
      const constraints: FalseAlarmConstraint[] = [];
      if (policy.maxFalseAlarm !== undefined && policy.maxFalseAlarm < 1) {
        constraints.push({ name: 'calibration false-alarm', scores: groupScores(ofClass(0)(pCal), calGroups && ofClass(0)(calGroups), Math.max), maxRate: policy.maxFalseAlarm });
      }
      if (backgroundP && budget !== undefined) constraints.push({ name: 'background', scores: backgroundP, maxRate: budget });
      const sel = conformalThreshold({ mode: mode as 'conformal-expected' | 'conformal-pac' | 'auto', targetRecall: policy.targetRecall, delta, positives, constraints, heuristicAvailable: policy.designRecall !== undefined, allowHeuristic });
      if (backgroundP && budget !== undefined) {
        for (const d of [undefined, delta / (1 + constraints.length)]) backgroundRanks.push(conformalRank(backgroundP.length, budget, d));
      }
      if (calibration.method === 'platt' && !(calibration.a > 0)) {
        sel.failures.push(`Platt slope ${calibration.a} reverses the model's score order, so no conformal guarantee holds`);
        sel.guarantee = { mode, kind: 'none', alpha: sel.guarantee.alpha };
      }
      threshold = sel.threshold ?? heuristic();
      guarantee = sel.threshold === null ? { ...sel.guarantee, kind: 'none', fallback: 'heuristic' } : sel.guarantee;
      sufficiency = sel.sufficiency;
      sufficiency.chosen = sel.threshold === null ? 'heuristic' : sufficiency.chosen;
      if (slices) {
        const perTest = delta / (1 + constraints.length);
        const needed = { expected: minimumSamples(guarantee.alpha), pac: minimumSamples(guarantee.alpha, perTest) };
        sufficiency.slices = {};
        for (const [field, values] of Object.entries(slices)) {
          const calValues = ofClass(1)(idx.calibration.map((i) => values[i]));
          const calPosGroups = calGroups && ofClass(1)(calGroups);
          for (const v of [...new Set(calValues)].sort()) {
            const n = new Set(calValues.flatMap((x, j) => (x === v ? [calPosGroups ? calPosGroups[j] : `#${j}`] : []))).size;
            sufficiency.slices[`${field}=${v}`] = { positive_groups: n, feasible: [...(n >= needed.expected ? ['conformal-expected' as const] : []), ...(n >= needed.pac ? ['conformal-pac' as const] : [])] };
          }
        }
      }
      result.failures.push(...sel.failures.map((f) => `${name}: ${f}`));
      result.warnings.push(...sel.warnings.map((w) => `${name}: ${w}`));
      log(`${name}: ${mode} threshold ${threshold} (${guarantee.kind} guarantee, ${positives.length} calibration positive groups)`);
    } else if (policy.kind === 'precision' && precisionMode === 'design') {
      if (!calDesign) throw new Error(`${name}: precision mode 'design' needs sampled calibration records (records)`);
      const designMethod = policy.designMethod ?? 'linearised';
      // Candidates fixed before calibration: training-split logits (evenly spaced ranks from the 30th
      // highest down), mapped through the calibrator so the bound is computed on the firing rule that ships.
      // Candidate scores fixed before calibration: background traffic when there is some (independent of
      // calibration, natural prevalence), else training scores - an enriched training set fires far more
      // often at the top than calibration traffic does. The r-th highest of n reference scores is expected
      // to be exceeded by ~r · n_cal / n calibration examples: start where that is about designMinFired.
      const minFired = policy.designMinFired ?? 20;
      const reference = background && background.X.length
        ? background.X.map((x) => score(x)).sort((a, b) => b - a)
        : idx.train.map((i) => calibrate(calibration, decisionFunction(model, rowFeatures(i)))).sort((a, b) => b - a);
      const start = Math.min(reference.length - 1, Math.ceil((minFired * reference.length) / Math.max(1, idx.calibration.length)) - 1);
      const ranks = Array.from({ length: 150 }, (_, k) => start + Math.floor((k * (reference.length - 1 - start)) / 149));
      const candidates = [...new Set(ranks.map((r) => reference[r]))].sort((a, b) => b - a);
      const res = designPrecisionThreshold({ ...calDesign, y: yCal, scores: pCal, candidates, targetPrecision: policy.targetPrecision, delta, method: designMethod });
      const alpha = 1 - policy.targetPrecision;
      if (res.feasible) {
        threshold = res.threshold;
        guarantee = { mode: 'design', kind: res.guarantee, alpha, delta, method: designMethod, metric: 'precision' };
        if (backgroundP && budget !== undefined) {
          const t = budgetThreshold(threshold, backgroundP, budget);
          if (t > threshold) {
            result.warnings.push(`${name}: design: the background budget raised the threshold from ${threshold} to ${t}; precision there is not covered by the guarantee`);
            threshold = t;
            guarantee = { mode: 'design', kind: 'none', alpha, metric: 'precision' };
          }
          if (Math.floor(budget * backgroundP.length) < backgroundP.length) backgroundRanks.push(Math.floor(budget * backgroundP.length));
        }
        log(`${name}: design precision threshold ${threshold} (${res.guarantee}, ${designMethod}, ${res.firedEffective.toFixed(1)} effective fired)`);
      } else if (policy.fallback === 'heuristic') {
        // Too little evidence for the guarantee, and the policy chose a heuristic threshold for that case.
        result.warnings.push(`${name}: design: ${res.reason}; heuristic threshold (fallback), no guarantee`);
        threshold = heuristic();
        guarantee = { mode: 'design', kind: 'none', alpha, metric: 'precision', fallback: 'heuristic' };
      } else {
        result.failures.push(`${name}: design: ${res.reason}`);
        threshold = nextUp(1);
        guarantee = { mode: 'design', kind: 'none', alpha, metric: 'precision' };
      }
      result.design!.heads[name] = { calibration: 'design-weighted', threshold: `design precision (${designMethod})`, evaluation: 'design-linearised' };
    } else {
      threshold = heuristic();
      if (policy.kind === 'recall') guarantee = { mode: 'heuristic', kind: 'none', alpha: 1 - policy.targetRecall };
      if (policy.kind === 'precision' && precisionMode === undefined && policy.fallback !== 'heuristic') {
        // No guarantee is possible here, and none was waived: the head may ship only in shadow mode.
        result.failures.push(`${name}: no precision guarantee without sampled calibration records - pass records (mode 'design'), or set mode 'heuristic' (or fallback 'heuristic') to accept an unguaranteed threshold`);
      } else if (policy.kind === 'precision' && precisionMode === undefined) {
        guarantee = { mode: 'heuristic', kind: 'none', alpha: 1 - policy.targetPrecision, metric: 'precision', fallback: 'heuristic' };
        result.warnings.push(`${name}: no sampled calibration records, so no precision guarantee: heuristic threshold (fallback)`);
      } else result.warnings.push(`${name}: mode 'heuristic' was chosen - the threshold carries no guarantee`);
    }
    if (records && !result.design!.heads[name]) result.design!.heads[name] = { calibration: 'design-weighted', threshold: mode, evaluation: 'design-linearised' };
    if (policy.kind === 'precision' && precisionMode !== 'design' && !pCal.some((p) => p >= threshold)) {
      result.warnings.push(`${name}: no calibration example reaches the target precision ${threshold} - the head is unlikely ever to fire`);
    }
    if (dismissal && !(threshold > 0)) {
      // Firing at 0 would mean firing on messages the rules cleared, which are never scored.
      result.failures.push(`${name}: the target needs messages the dismissal rules cleared to fire - certify fewer dismissal rules`);
      threshold = nextUp(1);
      if (guarantee) guarantee = { mode: guarantee.mode, kind: 'none', alpha: guarantee.alpha, ...(guarantee.metric ? { metric: guarantee.metric } : {}) };
    }
    const reviewFloor = reviewEpsilon === undefined ? threshold * reviewRatio : Math.min(threshold, conformalLowerThreshold(positives, reviewEpsilon) ?? 0);

    const yTest = ySplit('test');
    const pTest = idx.test.map((i) => (cleared(i) ? 0 : calibrate(calibration, decisionFunction(model, rowFeatures(i)))));
    const atTest = <T>(values: readonly T[]) => idx.test.map((i) => values[i]);
    const testDesign = records ? designOf(idx.test.map((i) => records[i])) : null;
    const ev = evaluateHead({
      p: pTest,
      y: yTest,
      w: testDesign ? testDesign.inclusionProbs.map((p) => 1 / p) : prevalenceWeights(yTest, prevalence!),
      ...(testDesign ? { design: { ...testDesign, seed } } : {}),
      threshold,
      groups: groups && atTest(groups),
      slices: slices && Object.fromEntries(Object.entries(slices).map(([field, values]) => [field, atTest(values)])),
      baseline: baseline && atTest(baseline),
      delta,
      prevalence,
    });
    if (guarantee) ev.guarantee = guarantee;
    if (sufficiency) ev.sufficiency = sufficiency;
    if (backgroundP) {
      ev.background_rate = backgroundP.filter((p) => p >= threshold).length / backgroundP.length;
      ev.background_rate_upper = backgroundUpper(backgroundP, threshold, backgroundRanks, 1 - delta / 2);
    }
    const gates = gateHead(policy, ev, { maxEce, calibrationAlpha });
    result.evaluation[name] = ev;
    result.failures.push(...gates.failures.map((f) => `${name}: ${f}`));
    result.warnings.push(...gates.warnings.map((w) => `${name}: ${w}`));
    result.heads[name] = {
      weights: model.coef, bias: model.intercept, ...(fitted?.features ? { features: fitted.features } : {}), calibration, threshold, review_floor: reviewFloor,
      ...(dismissal ? { dismissal: { rule_set: dismissal.ruleSet, max_rate: dismissal.maxRate, certified: dismissal.certified } } : {}),
      ...(guarantee ? { guarantee } : {}), ...(reviewEpsilon !== undefined ? { review_epsilon: reviewEpsilon } : {}),
    };
    // Round-trip checks score embeddings, so rows the rules cleared (never scored at runtime) are left out.
    result.testProbabilities[name] = { idx: idx.test.filter((i) => !cleared(i)), p: pTest.filter((_, j) => !cleared(idx.test[j])) };
  }
  // Keep the reference only for heads that use it (auto may have chosen linear everywhere).
  if (reference) {
    const using = Object.entries(result.heads).filter(([, spec]) => (spec as HeadSpec).features).map(([h]) => h);
    if (using.length) result.reference = { ...reference.set, labels: Object.fromEntries(using.map((h) => [h, reference!.set.labels[h]])) };
  }
  return result;
}

/**
 * Exact upper bound on the background firing rate at `threshold`. A threshold capped by the
 * background itself sits at or above the score just past a FIXED rank r, whose firing rate is a
 * uniform order statistic - so the bound for the largest rank any cap could have used is valid
 * whichever cap applied. Otherwise the threshold never looked at the background and its count is
 * an ordinary binomial.
 */
function backgroundUpper(scores: readonly number[], threshold: number, ranks: readonly number[], confidence: number): number {
  const sorted = [...scores].sort((a, b) => b - a);
  const r = Math.max(-1, ...ranks.filter((k) => k < sorted.length));
  if (r >= 0 && threshold > sorted[r]) return clopperPearsonUpper(r, sorted.length, confidence);
  return clopperPearsonUpper(sorted.filter((p) => p >= threshold).length, sorted.length, confidence);
}

/**
 * Serialises the artifact, re-loads it through validateArtifact and checks runtime scoring
 * reproduces the evaluated test probabilities exactly - so what was evaluated is what will run.
 */
export function assertRoundTrip<H extends string>(
  artifact: ClassifierArtifact<H>,
  X: ReadonlyArray<ArrayLike<number>>,
  testProbabilities: TrainResult<H>['testProbabilities'],
  sample = 50,
): void {
  const reloaded = validateArtifact<H>(JSON.parse(JSON.stringify(artifact)));
  for (const [head, probs] of Object.entries(testProbabilities) as Array<[H, { idx: number[]; p: number[] }]>) {
    probs.idx.slice(0, sample).forEach((i, j) => {
      const runtime = scoreEmbedding(reloaded, X[i])[head]!;
      if (runtime !== probs.p[j]) throw new Error(`${head}: runtime scoring of the saved artifact differs from evaluation (${runtime} vs ${probs.p[j]})`);
    });
  }
}

/**
 * The artifact for a training result, with its gates taken from the result (passed exactly when no
 * gate failed), so they are never set by hand. Also records the reference, head choices, convergence,
 * provenance, design and weak-label summaries. Serve it through validateArtifact(raw, { mode }).
 */
export function buildArtifact<H extends string>(
  result: TrainResult<H>,
  options: {
    version: string;
    embedding: EmbeddingSpec;
    createdAt?: string;
    training?: Record<string, unknown>;
    /** For @liquidau/router: the rule set the artifact was trained and evaluated with, stored as `training.router`. */
    router?: RouterTraining;
  },
): ClassifierArtifact<H> {
  const training: Record<string, unknown> = { ...(options.training ?? {}) };
  if (options.router) {
    if (!/^[0-9a-f]{64}$/.test(options.router.ruleSetHash)) throw new Error('router.ruleSetHash must be a lowercase hex SHA-256 (rule-miner ruleSetHash)');
    training.router = { ruleSetHash: options.router.ruleSetHash };
  }
  if (Object.keys(result.headChoice).length) training.head_choice = result.headChoice;
  if (Object.keys(result.convergence).length) training.convergence = result.convergence;
  if (Object.keys(result.weakLabels).length) training.weak_labels = result.weakLabels;
  if (result.provenance) training.provenance = result.provenance;
  if (result.design) training.design = result.design;
  return {
    version: options.version,
    created_at: options.createdAt ?? new Date().toISOString(),
    embedding: options.embedding,
    heads: result.heads,
    ...(result.reference ? { reference: result.reference } : {}),
    training,
    evaluation: result.evaluation as Record<string, unknown>,
    gates: { passed: result.failures.length === 0, failures: [...result.failures], warnings: [...result.warnings] },
  };
}

