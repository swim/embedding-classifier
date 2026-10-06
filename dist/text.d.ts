/**
 * Text preparation shared by training and serving. Pure (no Node built-ins), so serving code on any
 * runtime can import it from the package root.
 */
/**
 * Truncates to at most maxChars UTF-16 code units without splitting a surrogate pair. Use the same
 * function at runtime so training and serving embed identical text.
 */
export declare function truncateText(text: string, maxChars?: number): string;
