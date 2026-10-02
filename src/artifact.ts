/**
 * The trained artifact - a self-describing JSON document - and how it's scored. Training-time
 * evaluation and runtime scoring call these same functions, so what was evaluated is exactly what
 * runs.
 *
 * Per head:  logit = w·x + b  ->  calibrated p (Platt: σ(a·logit + c) | isotonic: interp(σ(logit)))
 */
import { decisionFunction, predictIsotonic, sigmoid } from '@liquidau/solvers';

export type Calibration =
  | { method: 'platt'; a: number; c: number }
  | { method: 'isotonic'; x: number[]; y: number[] };

export interface HeadSpec {
  weights: number[];
  bias: number;
  calibration: Calibration;
  /** Act at or above this calibrated probability. */
  threshold: number;
  /** Below the threshold but at or above this: worth a human look (sampling for labelling, suppression). */
  review_floor: number;
  /**
   * Named extra thresholds for tiered responses, e.g. { checkin: 0.012 } - a gentler action below
   * the main threshold. Interpreted by the application, not by decide().
   */
  thresholds?: Record<string, number>;
}

/** Records which embedding the heads were trained on - runtime must embed identically. */
export interface EmbeddingSpec {
  model_id: string;
  dimensions: number;
  normalize: boolean;
  /** Provider-specific input type, e.g. Cohere's "classification" - part of what the vectors mean. */
  input_type?: string;
}

export interface GateResult {
  passed: boolean;
  failures: string[];
  warnings?: string[];
}

export interface ClassifierArtifact<H extends string = string> {
  version: string;
  created_at: string;
  embedding: EmbeddingSpec;
  heads: Partial<Record<H, HeadSpec>>;
  training?: Record<string, unknown>;
  evaluation?: Record<string, unknown>;
  gates?: GateResult;
}

export type Scores<H extends string = string> = Partial<Record<H, number>>;

export function calibrate(calibration: Calibration, logit: number): number {
  return calibration.method === 'platt'
    ? sigmoid(calibration.a * logit + calibration.c)
    : predictIsotonic(calibration, sigmoid(logit));
}

export function headProbability(spec: HeadSpec, embedding: ArrayLike<number>): number {
  return calibrate(spec.calibration, decisionFunction({ coef: spec.weights, intercept: spec.bias }, embedding));
}

const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const allFinite = (v: unknown): v is number[] => Array.isArray(v) && v.every(isFiniteNumber);

/**
 * Validates an artifact loaded from storage; throws with a specific reason if it's unusable.
 * Checks values, not just shape: a corrupted or hand-edited artifact must fail loudly here rather
 * than score NaN at runtime (which decide() would otherwise read as "no decision").
 * Pass `heads` to reject head names the caller doesn't know how to act on.
 */
export function validateArtifact<H extends string = string>(raw: unknown, options: { heads?: readonly H[] } = {}): ClassifierArtifact<H> {
  const a = raw as ClassifierArtifact<H>;
  if (!a || typeof a !== 'object' || typeof a.version !== 'string' || !a.embedding || !a.heads || typeof a.heads !== 'object' || Array.isArray(a.heads)) {
    throw new Error('not a classifier artifact');
  }
  const dims = a.embedding.dimensions;
  if (!Number.isInteger(dims) || dims < 1) throw new Error(`embedding dimensions must be a positive integer, got ${dims}`);
  for (const [name, spec] of Object.entries(a.heads) as Array<[string, HeadSpec | undefined]>) {
    if (options.heads && !(options.heads as readonly string[]).includes(name)) throw new Error(`unknown head ${name}`);
    if (!spec || !Array.isArray(spec.weights) || spec.weights.length !== dims) {
      throw new Error(`head ${name} has ${spec?.weights?.length} weights but the embedding has ${dims} dimensions`);
    }
    if (!allFinite(spec.weights) || !isFiniteNumber(spec.bias)) throw new Error(`head ${name} has non-numeric weights or bias`);
    const cal = spec.calibration as { method?: string; a?: unknown; c?: unknown; x?: unknown; y?: unknown } | undefined;
    if (cal?.method === 'platt') {
      if (!isFiniteNumber(cal.a) || !isFiniteNumber(cal.c)) throw new Error(`head ${name} has non-numeric Platt parameters`);
    } else if (cal?.method === 'isotonic') {
      const { x, y } = cal;
      if (!allFinite(x) || !allFinite(y) || x.length === 0 || x.length !== y.length) {
        throw new Error(`head ${name} has an isotonic table that is empty, non-numeric or of mismatched length`);
      }
      for (let i = 1; i < x.length; i++) if (x[i] < x[i - 1]) throw new Error(`head ${name} has isotonic x values that are not sorted`);
    } else {
      throw new Error(`head ${name} has unknown calibration ${cal?.method}`);
    }
    if (!isFiniteNumber(spec.threshold) || !isFiniteNumber(spec.review_floor)) throw new Error(`head ${name} has a non-numeric threshold or review floor`);
    // No upper bound on the threshold: a background budget can push it just past 1 ("never fire").
    if (!(spec.review_floor >= 0 && spec.review_floor <= spec.threshold)) {
      throw new Error(`head ${name} needs 0 <= review_floor <= threshold (got ${spec.review_floor}, ${spec.threshold})`);
    }
    for (const [tier, t] of Object.entries(spec.thresholds ?? {})) if (!isFiniteNumber(t)) throw new Error(`head ${name} has a non-numeric ${tier} threshold`);
  }
  return a;
}

/** Calibrated probability for every head in the artifact. */
export function scoreEmbedding<H extends string>(artifact: ClassifierArtifact<H>, embedding: ArrayLike<number>): Scores<H> {
  if (embedding.length !== artifact.embedding.dimensions) {
    throw new Error(`embedding has ${embedding.length} dimensions but the artifact expects ${artifact.embedding.dimensions}`);
  }
  for (let i = 0; i < embedding.length; i++) {
    if (!Number.isFinite(embedding[i])) throw new Error(`embedding has a non-finite value at index ${i}`);
  }
  const scores: Scores<H> = {};
  for (const [head, spec] of Object.entries(artifact.heads) as Array<[H, HeadSpec | undefined]>) {
    if (spec) scores[head] = headProbability(spec, embedding);
  }
  return scores;
}
