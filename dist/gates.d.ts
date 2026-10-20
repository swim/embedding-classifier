import type { HeadEvaluation } from './evaluate.ts';
import type { HeadPolicy } from './threshold.ts';
export declare function gateHead(policy: HeadPolicy, ev: HeadEvaluation, options?: {
    maxEce?: number;
    calibrationAlpha?: number;
    calibrationGate?: 'cox' | 'cox-corrected' | 'region';
    calibrationTolerance?: number;
}): {
    failures: string[];
    warnings: string[];
};
