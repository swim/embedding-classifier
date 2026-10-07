import { type ExampleRecord, type Role } from './records.ts';
export interface Axes {
    readonly axes: Readonly<Record<string, {
        values: readonly string[];
        observable: boolean;
    }>>;
}
export declare function defineAxes(axes: Record<string, {
    values: string[];
    observable: boolean;
}>): Axes;
type Counts = {
    positives: number;
    negatives: number;
    positive_groups: number;
};
export interface CoverageReport {
    head: string;
    /** axis -> value -> role -> counts. */
    values: Record<string, Record<string, Partial<Record<Role, Counts>>>>;
    /** Value pairs across two axes below minRealPerCell real positives or negatives. */
    gaps: Array<{
        a: string;
        b: string;
        positives: number;
        negatives: number;
    }>;
    /** Observable axis values: positive groups a zero-miss guarantee needs, available in calibration, and the shortfall. */
    slices: Array<{
        axis: string;
        value: string;
        needed: number;
        available: number;
        shortfall: number;
    }>;
    untagged: number;
}
export declare function coverageReport(options: {
    head: string;
    records: readonly ExampleRecord[];
    tags: Readonly<Record<string, Readonly<Record<string, string>>>>;
    axes: Axes;
    minRealPerCell?: number;
    guarantee?: {
        alpha: number;
        delta: number;
    };
}): CoverageReport;
/** Throws unless every slice axis is observable at prediction time. */
export declare function checkSliceAxes(axes: Axes, sliceAxes: readonly string[]): void;
export {};
