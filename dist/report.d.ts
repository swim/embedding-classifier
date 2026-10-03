/** A Markdown report of an artifact's gates and per-head evaluation, for humans and review. */
import type { ClassifierArtifact } from './artifact.ts';
import type { CoverageReport } from './coverage.ts';
export declare function reportMarkdown(artifact: ClassifierArtifact, options?: {
    title?: string;
    baselineName?: string;
    coverage?: CoverageReport;
}): string;
