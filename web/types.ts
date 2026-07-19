export type ModelKind = 'instruct' | 'base';
export type FreshModelKind = 'qwen_instruct' | 'smollm2_instruct' | 'qwen_base';

export type FreshMethodKey =
  | 'crcv_mean'
  | 'crcv_max'
  | 'mean_nll'
  | 'confidence_variance_mean'
  | 'shift_variance_mean'
  | 'answer_tokens'
  | 'top3_token_surprise'
  | 'worst_token_surprise'
  | 'surprise_spread'
  | 'top4_token_surprise'
  | 'skip_first_top3_surprise'
  | 'surprise_shift_top3'
  | 'uncertainty_shift_top3'
  | 'token_entropy_top3'
  | 'token_entropy_mean'
  | 'token_entropy_max'
  | 'token_ambiguity_top3'
  | 'token_ambiguity_mean'
  | 'token_ambiguity_max'
  | 'hidden_cosine_mean'
  | 'hidden_cosine_max'
  | 'hidden_cosine_top3'
  | 'hidden_norm_mean'
  | 'hidden_norm_variability'
  | 'p_true'
  | 'hidden_logistic_probe'
  | 'semantic_disagreement_3'
  | 'lexical_disagreement_3'
  | 'discrete_semantic_entropy_3'
  | 'trace_logistic_8'
  | 'trace_tree_depth2';

export interface FreshSliceMetric {
  examples: number;
  incorrect: number;
  correct: number;
  auroc: number | null;
  auroc_ci_95: [number | null, number | null];
  average_precision: number | null;
  macro_f1: number;
  confusion: Record<string, number>;
  aurac: number;
  selective_accuracy: Record<'0.9' | '0.8' | '0.7', number>;
}

export interface FreshMethodMetric {
  display_name: string;
  formula: string;
  family: string;
  implementation_note: string;
  direction: string;
  threshold: number;
  calibration: { examples: number; incorrect: number; auroc: number; macro_f1: number };
  held_out: FreshSliceMetric;
  by_dataset: Record<'nq_open' | 'trivia_qa' | 'truthful_qa', FreshSliceMetric>;
  paired_vs_top3_token_surprise: null | {
    auroc_difference: number;
    auroc_difference_ci_95: [number, number];
    excludes_zero: boolean;
    reliably_improves: boolean;
  };
  incremental_runtime: {
    mean_seconds_per_question: number;
    median_seconds_per_question: number;
    total_seconds: number;
  };
}

export interface FreshComparisonMetrics {
  extension_status: string;
  label_definition: string;
  calibration_examples: number;
  held_out_examples: number;
  calibration_incorrect: number;
  held_out_incorrect: number;
  original_frozen_method_order: FreshMethodKey[];
  headline_method_order: FreshMethodKey[];
  scalar_method_order: FreshMethodKey[];
  all_method_order: FreshMethodKey[];
  trace_ensemble_features: FeatureKey[];
  methods: Record<FreshMethodKey, FreshMethodMetric>;
  posthoc_descriptive_best_non_confound: FreshMethodKey;
  posthoc_paired_intervals_above_zero: FreshMethodKey[];
  interpretation: string;
  extension_runtime: {
    trace_logistic_scoring_seconds_all_600: number;
    trace_tree_scoring_seconds_all_600: number;
  };
  runtime: {
    mean_greedy_generation_seconds: number;
    total_greedy_generation_seconds: number;
    answers_at_token_limit: number;
  };
}

export interface CrossModelReplicationRow {
  key: FreshMethodKey;
  display_name: string;
  family: string;
  qwen_auroc: number;
  qwen_auroc_ci_95: [number, number];
  smollm2_auroc: number;
  smollm2_auroc_ci_95: [number, number];
  qwen_slices_above_chance: number;
  smollm2_slices_above_chance: number;
  replicated: boolean;
  both_ci_lower_above_chance: boolean;
  worst_model_auroc: number;
}

export interface CrossModelReplication {
  status: string;
  rule: string;
  models: Record<'qwen_instruct' | 'smollm2_instruct', {
    id: string;
    parameters: string;
    held_out_examples: number;
    held_out_incorrect: number;
  }>;
  method_count: number;
  replicated_method_count: number;
  replicated_methods: FreshMethodKey[];
  both_ci_lower_above_chance: FreshMethodKey[];
  best_worst_model_method: FreshMethodKey;
  best_worst_model_auroc: number;
  rows: CrossModelReplicationRow[];
  interpretation: string;
}

export interface FreshPrediction {
  id: string;
  source_dataset: 'nq_open' | 'trivia_qa' | 'truthful_qa';
  source_index: number;
  split: 'calibration' | 'test';
  question: string;
  answers: string[];
  generated_answer: string;
  correct: boolean;
  is_hallucination: 0 | 1;
  token_ids: number[];
  token_pieces: string[];
  confidences: number[];
  hidden_shifts: Array<number | null>;
  token_margins: number[];
  token_entropies: number[];
  hidden_cosine_distances: Array<number | null>;
  hidden_norms: number[];
  features: Features;
  p_true_judgment: {
    yes_probability: number;
    no_probability: number;
    normalized_no_probability: number;
    yes_token_id: number;
    no_token_id: number;
    elapsed_seconds: number;
  };
  stochastic_answers: Array<{
    sample_index: number;
    seed: number;
    answer: string;
    elapsed_seconds: number;
  }>;
  pair_judgments: Array<{
    sample_a: number;
    sample_b: number;
    yes_probability: number;
    no_probability: number;
    normalized_no_probability: number;
    elapsed_seconds: number;
  }>;
  method_scores: Record<FreshMethodKey, number>;
  method_seconds: Record<FreshMethodKey, number>;
  lexical_disagreement_explanation: {
    pair_distances: Array<{ sample_a: number; sample_b: number; distance: number }>;
    score: number;
  };
  semantic_entropy_explanation: {
    threshold: number;
    clusters: number[][];
    cluster_sizes: number[];
    probabilities: number[];
    entropy_terms: number[];
    score: number;
  };
  trace_logistic_explanation: {
    bias: number;
    terms: Array<{
      feature: FeatureKey;
      raw_value: number;
      calibration_mean: number;
      calibration_scale: number;
      standardized_value: number;
      weight: number;
      contribution: number;
    }>;
    logit: number;
    sigmoid_risk: number;
  };
  trace_tree_path: string[];
  probe_explanation: {
    bias: number;
    top_contributions: Array<{
      dimension: number;
      raw_hidden: number;
      calibration_mean: number;
      calibration_scale: number;
      standardized_hidden: number;
      weight: number;
      contribution: number;
    }>;
    other_dimensions_contribution: number;
    logit: number;
    sigmoid_risk: number;
  };
}

export type FeatureKey =
  | 'token_entropy_top3'
  | 'token_entropy_mean'
  | 'token_entropy_max'
  | 'token_ambiguity_top3'
  | 'token_ambiguity_mean'
  | 'token_ambiguity_max'
  | 'top4_token_surprise'
  | 'skip_first_top3_surprise'
  | 'surprise_shift_top3'
  | 'uncertainty_shift_top3'
  | 'hidden_cosine_mean'
  | 'hidden_cosine_max'
  | 'hidden_cosine_top3'
  | 'hidden_norm_mean'
  | 'hidden_norm_variability'
  | 'top3_token_surprise'
  | 'worst_token_surprise'
  | 'surprise_spread'
  | 'crcv_mean'
  | 'crcv_max'
  | 'mean_nll'
  | 'confidence_variance_mean'
  | 'shift_variance_mean'
  | 'answer_tokens';

export interface Features {
  token_entropy_top3: number;
  token_entropy_mean: number;
  token_entropy_max: number;
  token_ambiguity_top3: number;
  token_ambiguity_mean: number;
  token_ambiguity_max: number;
  top4_token_surprise: number;
  skip_first_top3_surprise: number;
  surprise_shift_top3: number;
  uncertainty_shift_top3: number;
  hidden_cosine_mean: number;
  hidden_cosine_max: number;
  hidden_cosine_top3: number;
  hidden_norm_mean: number;
  hidden_norm_variability: number;
  top3_token_surprise: number;
  worst_token_surprise: number;
  surprise_spread: number;
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
  token_margins: number[];
  token_entropies: number[];
  hidden_cosine_distances: Array<number | null>;
  hidden_norms: number[];
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
  calibration_pairwise_accuracy?: number;
  test_pairwise_accuracy?: number;
}

export interface TreeRule {
  count: number;
  positives: number;
  hallucination_probability: number;
  feature?: FeatureKey;
  threshold?: number;
  left_if?: string;
  left?: TreeRule;
  right_if?: string;
  right?: TreeRule;
}

export interface TreeScoreMetric {
  calibration_auroc: number;
  calibration_macro_f1: number;
  threshold: number;
  test_auroc: number;
  test_auroc_ci_95: [number, number];
  test_macro_f1: number;
  test_confusion: Record<string, number>;
  calibration_pairwise_accuracy?: number;
  test_pairwise_accuracy?: number;
  rules: TreeRule;
}

export interface HaluEvalPrediction {
  pair_id: string;
  source_index: number;
  source_dataset: string;
  split: 'calibration' | 'test';
  knowledge: string;
  question: string;
  id: string;
  candidate_kind: 'right' | 'hallucinated';
  candidate_answer: string;
  is_hallucination: 0 | 1;
  token_ids: number[];
  token_pieces: string[];
  confidences: number[];
  hidden_shifts: Array<number | null>;
  token_margins: number[];
  token_entropies: number[];
  hidden_cosine_distances: Array<number | null>;
  hidden_norms: number[];
  features: Features;
  elapsed_seconds: number;
  tree_score: number;
  tree_path: string[];
}

export interface HaluEvalMetrics {
  scores: Record<FeatureKey, ScoreMetric>;
  selected_scalar: ScoreMetric & { score_key: FeatureKey };
  stump: TreeScoreMetric;
  depth2_tree: TreeScoreMetric;
  tree_features: FeatureKey[];
  length_diagnostics: {
    status: string;
    calibration_pairs_hallucinated_longer: number;
    test_pairs_hallucinated_longer: number;
    selected_scalar_residual: {
      status: string;
      score_key: FeatureKey;
      test_auroc: number;
      test_auroc_ci_95: [number, number];
      test_macro_f1: number;
      test_pairwise_accuracy: number;
    };
  };
}

export interface ShallowTreeResults {
  status: string;
  models: Record<ModelKind, {
    selected_scalar: ScoreMetric & { score_key: FeatureKey };
    stump: TreeScoreMetric;
    depth2_tree: TreeScoreMetric;
  }>;
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
  improvement_comparison: {
    name: string;
    baseline_score_key: FeatureKey;
    test_auroc_difference: number;
    test_auroc_difference_ci_95: [number, number];
  };
  primitive_comparison: {
    name: string;
    baseline_score_key: FeatureKey;
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
  margin: number;
  entropy: number;
  shift: number;
  hiddenCosineDistance: number;
  hiddenNorm: number;
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
  margin: number;
  entropy: number;
  hiddenShift: number | null;
  hiddenCosineDistance: number | null;
  hiddenNorm: number;
}

export type WorkerToMain =
  | { type: 'status'; phase: string; detail: string }
  | { type: 'progress'; file: string; loaded: number; total: number; progress: number }
  | { type: 'ready'; model: ModelKind; runtime: string; dtype: string }
  | { type: 'token'; token: LiveToken }
  | { type: 'result'; answer: string; tokens: LiveToken[]; features: Features; elapsedMs: number; qualityWarning: string | null }
  | { type: 'error'; message: string };
