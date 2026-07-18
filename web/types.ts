export type ModelKind = 'instruct' | 'base';

export type FeatureKey =
  | 'crcv_mean'
  | 'crcv_max'
  | 'mean_nll'
  | 'confidence_variance_mean'
  | 'shift_variance_mean'
  | 'answer_tokens';

export interface Features {
  crcv_mean: number;
  crcv_max: number;
  mean_nll: number;
  confidence_variance_mean: number;
  shift_variance_mean: number;
  answer_tokens: number;
}

export interface Prediction {
  id: string;
  question: string;
  answers: string[];
  category: string;
  difficulty: string;
  split: 'calibration' | 'test';
  generated_answer: string;
  correct: boolean;
  is_hallucination: 0 | 1;
  token_ids: number[];
  token_pieces: string[];
  confidences: number[];
  hidden_shifts: Array<number | null>;
  features: Features;
  elapsed_seconds: number;
}

export interface ScoreMetric {
  display_name: string;
  direction: string;
  calibration_auroc: number;
  calibration_macro_f1: number;
  threshold: number;
  test_auroc: number;
  test_auroc_ci_95: [number, number];
  test_macro_f1: number;
  test_confusion: Record<string, number>;
}

export interface BenchmarkMetrics {
  label_definition: string;
  calibration_examples: number;
  calibration_hallucinations: number;
  test_examples: number;
  test_hallucinations: number;
  scores: Record<FeatureKey, ScoreMetric>;
  paired_comparison: {
    name: string;
    test_auroc_difference: number;
    test_auroc_difference_ci_95: [number, number];
  };
  runtime: Record<string, number>;
  answers_at_token_limit: number;
}

export interface TokenCalculation {
  traceIndex: number;
  tokenIndex: number;
  token: string;
  confidence: number;
  shift: number;
  coupling: number;
  window: number[] | null;
  windowMean: number | null;
  squaredDeviations: number[] | null;
  sampleVariance: number | null;
  crcv: number | null;
}

export interface LiveToken {
  tokenId: number;
  token: string;
  confidence: number;
  hiddenShift: number | null;
}

export type WorkerToMain =
  | { type: 'status'; phase: string; detail: string }
  | { type: 'progress'; file: string; loaded: number; total: number; progress: number }
  | { type: 'ready'; model: ModelKind; runtime: string; dtype: string }
  | { type: 'token'; token: LiveToken }
  | { type: 'result'; answer: string; tokens: LiveToken[]; features: Features; elapsedMs: number }
  | { type: 'error'; message: string };
