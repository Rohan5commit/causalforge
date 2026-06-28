// ── Scientific Document ──
export interface ScientificDocument {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  fullText?: string;
  domain: string;
  source: "upload" | "paste" | "demo";
  ingestedAt: string;
  status: "processing" | "extracted" | "error";
}

// ── Scientific Claim ──
export interface ScientificClaim {
  id: string;
  documentId: string;
  type: "causal" | "intervention" | "outcome" | "mechanism" | "correlation";
  statement: string;
  variables: string[];
  direction: "positive" | "negative" | "neutral" | "unclear";
  confidence: number; // 0-1
  evidenceStrength: "strong" | "moderate" | "weak" | "anecdotal";
  uncertaintyLanguage: string[];
  extractedAt: string;
}

// ── Variable Node ──
export interface VariableNode {
  id: string;
  label: string;
  domain: string;
  type: "treatment" | "outcome" | "mediator" | "confounder" | "covariate";
  description: string;
  evidenceCount: number;
  uncertaintyScore: number; // 0-1, higher = more uncertain
}

// ── Causal Edge ──
export interface CausalEdge {
  id: string;
  source: string; // variable id
  target: string; // variable id
  sign: "positive" | "negative" | "unclear";
  strength: number; // 0-1
  evidenceCount: number;
  confidence: number; // 0-1
  conflictLevel: number; // 0-1, higher = more conflict
  mechanisms: string[];
  sourceClaimIds: string[];
}

// ── Contradiction Record ──
export interface ContradictionRecord {
  id: string;
  type: "direct_conflict" | "incompatible_mechanism" | "evidence_gap" | "magnitude_dispute";
  description: string;
  involvedVariables: string[];
  claimA: { id: string; statement: string; confidence: number };
  claimB: { id: string; statement: string; confidence: number };
  severity: "critical" | "major" | "minor";
  resolved: boolean;
}

// ── Intervention Scenario ──
export interface InterventionScenario {
  id: string;
  name: string;
  description: string;
  targetVariable: string;
  changeType: "increase" | "decrease" | "binary" | "continuous";
  changeValue?: number;
  unit?: string;
}

// ── Simulation Result ──
export interface SimulationResult {
  id: string;
  interventionId: string;
  scenario: InterventionScenario;
  predictedEffects: PredictedEffect[];
  overallConfidence: number;
  sensitivityToUncertainty: number;
  affectedVariableCount: number;
  runAt: string;
}

export interface PredictedEffect {
  variableId: string;
  variableLabel: string;
  direction: "positive" | "negative" | "no_change" | "uncertain";
  magnitude: number; // 0-1 normalized
  confidence: number; // 0-1
  causalPath: string[]; // intermediate variable ids
  evidenceStrength: "strong" | "moderate" | "weak";
}

// ── Evidence Link ──
export interface EvidenceLink {
  id: string;
  edgeId: string;
  claimId: string;
  documentId: string;
  documentTitle: string;
  excerpt: string;
  supporting: boolean; // true = supports edge, false = contradicts
  confidence: number;
}

// ── Experiment Recommendation ──
export interface ExperimentRecommendation {
  id: string;
  rank: number;
  name: string;
  hypothesis: string;
  variablesToMeasure: string[];
  expectedInformationGain: number; // 0-1
  resolvesUncertaintyOf: string[]; // variable or edge ids
  whyItMatters: string;
  whatWouldChangeModel: string;
  estimatedDifficulty: "low" | "medium" | "high";
  relatedContradictions: string[];
}

// ── Causal Graph (previously "World Model") ──
export interface CausalGraphSnapshot {
  id: string;
  domain: string;
  nodes: VariableNode[];
  edges: CausalEdge[];
  contradictions: ContradictionRecord[];
  interventions: InterventionScenario[];
  simulations: SimulationResult[];
  experiments: ExperimentRecommendation[];
  evidenceLinks: EvidenceLink[];
  documentCount: number;
  claimCount: number;
  createdAt: string;
  updatedAt: string;
}

// ── API Response Types ──
export interface ExtractionResult {
  claims: ScientificClaim[];
  variables: string[];
  contradictions: string[];
}

export interface SimulationRequest {
  interventionId: string;
  worldModelId: string;  // TODO: rename to causalGraphId when frontend updates
}

export interface ExperimentRankingRequest {
  worldModelId: string;  // TODO: rename to causalGraphId when frontend updates
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: EvidenceLink[];
}

// ── Demo Domain ──
export interface DemoDomain {
  id: string;
  name: string;
  description: string;
  icon: string;
  documentCount: number;
  variableCount: number;
  edgeCount: number;
}
