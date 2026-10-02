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
export declare function loadOrder<K>(mode: ServeMode, locations: {
    promoted: K;
    shadowCandidate?: K | null;
}): Array<{
    location: K;
    role: ArtifactRole;
}>;
/**
 * Why `artifact` must not be served in `mode`, or null if it may. Enforcement requires recorded,
 * passed gates - checked at load time too, so an artifact copied into place by hand still can't act.
 */
export declare function refuseToServe(mode: ServeMode, artifact: Pick<ClassifierArtifact, 'gates'>): string | null;
/**
 * What publishing may do with a freshly trained artifact. Returns the role to point at it (null:
 * upload the versioned artifact only), or an error.
 *   blockedReason      set when the artifact must never be published (e.g. trained on fake embeddings)
 *   allowFailingGates  upload a failing artifact for inspection, without pointing any role at it
 */
export declare function publishPlan(options: {
    gatesPassed: boolean;
    promote?: boolean;
    shadowCandidate?: boolean;
    allowFailingGates?: boolean;
    blockedReason?: string;
}): {
    error: string;
} | {
    role: ArtifactRole | null;
};
