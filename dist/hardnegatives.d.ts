import type { ExampleRecord } from './records.ts';
/** Ids of real, human-labelled train records labelled 0 for `head` that a rule fires on. */
export declare function realHardNegatives(records: readonly ExampleRecord[], matcher: {
    match(text: string): unknown;
}, head: string): string[];
/** max(share × batch, min) items, at random; every item when `all` (safety-critical heads). */
export declare function selectForReview(batch: readonly ExampleRecord[], options: {
    share?: number;
    min?: number;
    seed: number;
    all?: boolean;
}): ExampleRecord[];
/**
 * Agreement = reviewed items whose human label matches the intended one. Pass `head`, the head
 * the intended labels are for. `safetyCritical`: every item must have been reviewed.
 */
export declare function verifyBatch(batch: readonly ExampleRecord[], reviews: ReadonlyArray<{
    id: string;
    label: 0 | 1;
}>, options: {
    head: string;
    minAgreementLower?: number;
    safetyCritical?: boolean;
}): {
    batchId: string;
    accepted: boolean;
    reviewed: number;
    agreement: number;
    lower: number;
    records: ExampleRecord[];
};
