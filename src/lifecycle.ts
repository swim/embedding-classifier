/**
 * Which artifact a deployment may load, and what a training run may publish.
 *
 *   promoted          the artifact a deployment acts on. Must have passed its gates.
 *   shadow-candidate  scored on live traffic in shadow mode only - it can't change any outcome, so
 *                     it may have failed its gates. This is how a not-yet-good-enough model is
 *                     evaluated on real data.
 *
 * Serving modes: `shadow` scores and records but never acts; `enforce` acts on decisions.
 */
import type { ClassifierArtifact } from './artifact.ts';

export type ServeMode = 'shadow' | 'enforce';
export type ArtifactRole = 'promoted' | 'shadow-candidate';

/** Locations to try, in order: shadow mode prefers a shadow candidate; enforce only loads the promoted artifact. */
export function loadOrder<K>(mode: ServeMode, locations: { promoted: K; shadowCandidate?: K | null }): Array<{ location: K; role: ArtifactRole }> {
  return [
    ...(mode === 'shadow' && locations.shadowCandidate != null ? [{ location: locations.shadowCandidate, role: 'shadow-candidate' as const }] : []),
    { location: locations.promoted, role: 'promoted' as const },
  ];
}

/**
 * Why `artifact` must not be served in `mode`, or null if it may. Enforcement requires recorded,
 * passed gates - checked at load time too, so an artifact copied into place by hand still can't act.
 */
export function refuseToServe(mode: ServeMode, artifact: Pick<ClassifierArtifact, 'gates'>): string | null {
  if (mode === 'enforce' && artifact.gates?.passed !== true) return 'its gates did not pass';
  return null;
}

/**
 * What publishing may do with a freshly trained artifact. Returns the role to point at it (null:
 * upload the versioned artifact only), or an error.
 *   blockedReason      set when the artifact must never be published (e.g. trained on fake embeddings)
 *   allowFailingGates  upload a failing artifact for inspection, without pointing any role at it
 */
export function publishPlan(options: {
  gatesPassed: boolean;
  promote?: boolean;
  shadowCandidate?: boolean;
  allowFailingGates?: boolean;
  blockedReason?: string;
  /** The artifact was trained on generated records (TrainResult.provenance.generated). */
  generated?: boolean;
  /** For a generated-data artifact: the shadow evaluation that met the acceptance criteria. Store it in the artifact. */
  acceptanceEvidence?: unknown;
}): { error: string } | { role: ArtifactRole | null } {
  const { gatesPassed, promote = false, shadowCandidate = false, allowFailingGates = false, blockedReason, generated = false, acceptanceEvidence } = options;
  if (blockedReason) return { error: `refusing to publish: ${blockedReason}` };
  if (promote && shadowCandidate) return { error: 'choose one of promote or shadow-candidate' };
  if (promote && generated && (acceptanceEvidence === undefined || acceptanceEvidence === null)) {
    return { error: 'refusing to promote an artifact trained on generated data without acceptance evidence from its shadow evaluation' };
  }
  if (promote) return gatesPassed ? { role: 'promoted' } : { error: 'refusing to promote an artifact that failed its gates (publish it as a shadow candidate to evaluate it in shadow mode)' };
  if (shadowCandidate) return { role: 'shadow-candidate' };
  if (!gatesPassed && !allowFailingGates) return { error: 'refusing to publish: gates failed (allow failing gates to upload for inspection only, or publish as a shadow candidate)' };
  return { role: null };
}
