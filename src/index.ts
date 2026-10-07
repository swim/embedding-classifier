export { calibrate, checkEmbeddingSpec, checkRuleSetPairing, headProbability, prepareScoring, routerRuleSetHash, scoreEmbedding, validateArtifact } from './artifact.ts';
export type { Calibration, ClassifierArtifact, EmbeddingSpec, GateResult, HeadSpec, RouterTraining, Scores } from './artifact.ts';
export { decide, missingPolicyHeads, settleWithRules } from './decide.ts';
export type { Decision, DecisionHeads, DecisionPolicy, DecisionReason, RulesEvaluation, Settlement } from './decide.ts';
export { budgetThreshold, falseAlarmCap, nextUp, pickThreshold } from './threshold.ts';
export type { HeadPolicy } from './threshold.ts';
export { evaluateHead } from './evaluate.ts';
export type { BaselineComparison, CertifiedBounds, DesignEstimate, DesignEvaluation, EvaluateInput, HeadEvaluation } from './evaluate.ts';
export { gateHead } from './gates.ts';
export { assertRoundTrip, buildArtifact, SPLITS, trainHeads } from './train.ts';
export type { HeadChoice, HeadInput, ProvenanceSummary, Split, TrainInput, TrainResult, WeakInput, WeakSummary } from './train.ts';
export { loadOrder, publishPlan, refuseToServe } from './lifecycle.ts';
export type { ArtifactRole, ServeMode } from './lifecycle.ts';
export { reportMarkdown } from './report.ts';
export { conformalThreshold, groupScores, THRESHOLD_MODES } from './conformal.ts';
export type { ConformalSelection, FalseAlarmConstraint, Guarantee, GuaranteeKind, Sufficiency, ThresholdMode } from './conformal.ts';
export { applyReviews, designOf, designSample, labelQueue, requiredSampleSize, stableId } from './design.ts';
export type { DesignOptions, DesignStratum, DesignSummary, FrameItem, Mechanism, QueueItem, QueueKey } from './design.ts';
export { capTrainingWeights, isReal, ProvenanceError, validateProvenance } from './records.ts';
export type { BackgroundUse, ExampleRecord, ProvenanceCode, ProvenanceOptions, ProvenanceResult, Role, Source, WeightCaps, WeightCapSummary } from './records.ts';
export { mmrSelect, retrieveFromSeeds, seedStats } from './retrieval.ts';
export type { Embedded, RetrieveOptions } from './retrieval.ts';
export { realHardNegatives, selectForReview, verifyBatch } from './hardnegatives.ts';
export { checkSliceAxes, coverageReport, defineAxes } from './coverage.ts';
export type { Axes, CoverageReport } from './coverage.ts';
export { monitorWindow } from './monitor.ts';
export { truncateText } from './text.ts';
export { HEAD_TYPES } from './heads.ts';
export type { HeadFeatures, HeadType, ReferenceSet } from './heads.ts';
export type { DriftCheck } from './monitor.ts';
export {
  buildDocumentArtifact, chunkVectorProblems, documentCompatibilityProblems, documentFeatureEmbedding, documentFeatureIdentity, DOCUMENT_ARTIFACT_SCHEMA, DOCUMENT_ERROR_CODES,
  DOCUMENT_FEATURES_SCHEMA, DocumentError, FULL_ENCODER_IDENTITY_SCHEMA, fullEncoderIdentityProblems, isDocumentArtifact, poolChunkVectors, prepareDocumentFeatures,
  scoreDocument, scoreDocumentEmbeddings, trainDocumentHeads, validateDocumentArtifact,
} from './document.ts';
export type {
  DocumentAdapters, DocumentClassifierArtifact, DocumentEncoder, DocumentErrorCode, EncoderUsage, DocumentFeatureRecord, DocumentFeatures, DocumentHeadInput, DocumentOptions,
  DocumentScoreResult, DocumentTrainInput, DocumentTrainResult, FullEncoderIdentity,
} from './document.ts';
