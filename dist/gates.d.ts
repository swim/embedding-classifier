/** Release gates: an artifact is only fit to act on its decisions if every head passes. */
import type { HeadEvaluation } from './evaluate.ts';
import type { HeadPolicy } from './threshold.ts';
export declare function gateHead(policy: HeadPolicy, ev: HeadEvaluation, options?: {
    maxEce?: number;
}): {
    failures: string[];
    warnings: string[];
};
