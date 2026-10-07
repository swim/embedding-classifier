/**
 * Hard negatives: innocent messages that trigger a rule or sit near positives.
 *
 *   realHardNegatives  real, human-labelled train records a certified rule fires on, labelled 0 -
 *                      for exception mining and reports. Never calibration or test records.
 *   selectForReview    the share of a generated batch a human must check
 *   verifyBatch        accept a generated batch when the Wilson lower bound on reviewer agreement
 *                      with the intended label reaches minAgreementLower (default 0.9): agreeing
 *                      items are verified (weight 1), disagreeing ones dropped, unreviewed ones keep
 *                      verified = false at weight = the lower bound. A rejected batch is dropped whole.
 *
 * Generated records are training data only (P4), need verification or an accepted batch (P7), are
 * capped at 30% of negative training weight and per rule by the positives that rule matches (P6),
 * and make the artifact a shadow candidate until acceptance evidence exists (publishPlan).
 */
import { seededRandom, wilson } from '@liquidau/solvers';
/** Ids of real, human-labelled train records labelled 0 for `head` that a rule fires on. */
export function realHardNegatives(records, matcher, head) {
    return records.filter((r) => r.role === 'train' && (r.source.kind === 'sampled' || r.source.kind === 'traffic' || r.source.kind === 'retrieved') &&
        r.labelledBy === 'human' && r.labels[head] === 0 && matcher.match(r.text) != null).map((r) => r.id);
}
function checkBatch(batch) {
    const ids = new Set();
    let batchId = null;
    for (const r of batch) {
        if (r.source.kind !== 'generated')
            throw new Error(`record ${r.id} is not generated`);
        if (batchId !== null && r.source.batchId !== batchId)
            throw new Error(`a batch must share one batchId (${batchId}, ${r.source.batchId})`);
        batchId = r.source.batchId;
        if (ids.has(r.id))
            throw new Error(`duplicate id ${r.id} in batch`);
        ids.add(r.id);
    }
    if (batchId === null)
        throw new Error('the batch is empty');
    return batchId;
}
/** max(share × batch, min) items, at random; every item when `all` (safety-critical heads). */
export function selectForReview(batch, options) {
    const { share = 0.2, min = 100, seed, all = false } = options;
    checkBatch(batch);
    if (all)
        return [...batch];
    const n = Math.min(batch.length, Math.max(min, Math.ceil(share * batch.length)));
    const rand = seededRandom(seed);
    const items = [...batch];
    for (let i = 0; i < n; i++) {
        const j = i + Math.floor(rand() * (items.length - i));
        [items[i], items[j]] = [items[j], items[i]];
    }
    return items.slice(0, n);
}
/**
 * Agreement = reviewed items whose human label matches the intended one. Pass `head`, the head
 * the intended labels are for. `safetyCritical`: every item must have been reviewed.
 */
export function verifyBatch(batch, reviews, options) {
    const { head, minAgreementLower = 0.9, safetyCritical = false } = options;
    const batchId = checkBatch(batch);
    const byId = new Map(batch.map((r) => [r.id, r]));
    for (const rv of reviews)
        if (!byId.has(rv.id))
            throw new Error(`review for ${rv.id}, which is not in the batch`);
    for (const r of batch) {
        const intended = r.labels[head];
        if (intended !== 0 && intended !== 1)
            throw new Error(`record ${r.id} has no intended label for ${head}`);
    }
    const review = new Map(reviews.map((rv) => [rv.id, rv.label]));
    if (safetyCritical && review.size < batch.length)
        throw new Error(`safety-critical head ${head}: all ${batch.length} items must be reviewed, got ${review.size}`);
    const agree = batch.filter((r) => review.has(r.id) && review.get(r.id) === r.labels[head]).length;
    const reviewed = review.size;
    const lower = reviewed ? wilson(agree, reviewed)[0] : 0;
    const accepted = reviewed > 0 && lower >= minAgreementLower;
    const records = !accepted ? [] : batch.flatMap((r) => {
        if (!review.has(r.id))
            return safetyCritical ? [] : [{ ...r, verified: false, weight: lower }];
        return review.get(r.id) === r.labels[head] ? [{ ...r, verified: true, labelledBy: 'intended', weight: 1 }] : [];
    });
    return { batchId, accepted, reviewed, agreement: reviewed ? agree / reviewed : 0, lower, records };
}
