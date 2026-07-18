import type { Features, LiveToken, Prediction, TokenCalculation } from '../types';

export const WINDOW = 5;

export function sampleStd(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

export function rollingSampleStd(values: number[], window = WINDOW): number[] {
  if (window < 2) throw new Error('window must be at least 2');
  if (values.length < 2) return [0];
  const effectiveWindow = Math.min(window, values.length);
  const result: number[] = [];
  for (let end = effectiveWindow; end <= values.length; end += 1) {
    result.push(sampleStd(values.slice(end - effectiveWindow, end)));
  }
  return result;
}

type AuxiliarySignals = {
  tokenMargins?: number[];
  tokenEntropies?: number[];
  hiddenCosineDistances?: Array<number | null>;
  hiddenNorms?: number[];
};

export function computeFeatures(confidences: number[], shifts: Array<number | null>, window = WINDOW, signals: AuxiliarySignals = {}): Features {
  const valid = confidences
    .map((confidence, index) => ({ confidence, shift: shifts[index] }))
    .filter((pair): pair is { confidence: number; shift: number } => pair.shift !== null && Number.isFinite(pair.shift));
  const validConfidences = valid.map(({ confidence }) => confidence);
  const validShifts = valid.map(({ shift }) => shift);
  const couplings = valid.map(({ confidence, shift }) => confidence * shift);
  const crcv = rollingSampleStd(couplings, window);
  const confidenceCrcv = rollingSampleStd(validConfidences, window);
  const shiftCrcv = rollingSampleStd(validShifts, window);
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const tokenSurprises = confidences.map((value) => -Math.log(Math.max(value, 1e-12)));
  const largestSurprises = [...tokenSurprises].sort((left, right) => right - left).slice(0, 3);
  const meanOrZero = (values: number[]) => values.length ? mean(values) : 0;
  const topMean = (values: number[], count = 3) => meanOrZero([...values].sort((left, right) => right - left).slice(0, count));
  const surpriseShifts = valid.map(({ confidence, shift }) => -Math.log(Math.max(confidence, 1e-12)) * shift);
  const uncertaintyShifts = valid.map(({ confidence, shift }) => (1 - confidence) * shift);
  const ambiguities = (signals.tokenMargins ?? []).map((margin) => 1 - margin);
  const entropies = signals.tokenEntropies ?? [];
  const cosineDistances = (signals.hiddenCosineDistances ?? []).filter((value): value is number => value !== null && Number.isFinite(value));
  const norms = signals.hiddenNorms ?? [];
  return {
    crcv_mean: mean(crcv),
    crcv_max: Math.max(...crcv),
    confidence_variance_mean: mean(confidenceCrcv),
    shift_variance_mean: mean(shiftCrcv),
    mean_nll: tokenSurprises.length ? mean(tokenSurprises) : 0,
    top3_token_surprise: largestSurprises.length ? mean(largestSurprises) : 0,
    top4_token_surprise: topMean(tokenSurprises, 4),
    skip_first_top3_surprise: topMean(tokenSurprises.slice(1)),
    worst_token_surprise: largestSurprises[0] ?? 0,
    surprise_spread: sampleStd(tokenSurprises),
    surprise_shift_top3: topMean(surpriseShifts),
    uncertainty_shift_top3: topMean(uncertaintyShifts),
    token_ambiguity_mean: meanOrZero(ambiguities),
    token_ambiguity_max: Math.max(...ambiguities, 0),
    token_ambiguity_top3: topMean(ambiguities),
    token_entropy_mean: meanOrZero(entropies),
    token_entropy_max: Math.max(...entropies, 0),
    token_entropy_top3: topMean(entropies),
    hidden_cosine_mean: meanOrZero(cosineDistances),
    hidden_cosine_max: Math.max(...cosineDistances, 0),
    hidden_cosine_top3: topMean(cosineDistances),
    hidden_norm_mean: meanOrZero(norms),
    hidden_norm_variability: sampleStd(norms),
    answer_tokens: confidences.length,
  };
}

export function liveFeatures(tokens: LiveToken[]): Features {
  return computeFeatures(
    tokens.map((token) => token.confidence),
    tokens.map((token) => token.hiddenShift),
    WINDOW,
    {
      tokenMargins: tokens.map((token) => token.margin),
      tokenEntropies: tokens.map((token) => token.entropy),
      hiddenCosineDistances: tokens.map((token) => token.hiddenCosineDistance),
      hiddenNorms: tokens.map((token) => token.hiddenNorm),
    },
  );
}

export function tokenCalculations(prediction: Prediction, window = WINDOW): TokenCalculation[] {
  const valid: TokenCalculation[] = [];
  prediction.hidden_shifts.forEach((shift, traceIndex) => {
    if (shift === null || !Number.isFinite(shift)) return;
    const confidence = prediction.confidences[traceIndex];
    const coupling = confidence * shift;
    valid.push({
      traceIndex,
      tokenIndex: traceIndex + 1,
      token: prediction.token_pieces[traceIndex],
      confidence,
      margin: prediction.token_margins?.[traceIndex] ?? 0,
      entropy: prediction.token_entropies?.[traceIndex] ?? 0,
      shift,
      hiddenCosineDistance: prediction.hidden_cosine_distances?.[traceIndex] ?? 0,
      hiddenNorm: prediction.hidden_norms?.[traceIndex] ?? 0,
      coupling,
      window: null,
      windowMean: null,
      squaredDeviations: null,
      sampleVariance: null,
      crcv: null,
    });
  });

  const effectiveWindow = Math.min(window, valid.length);
  valid.forEach((row, index) => {
    if (effectiveWindow < 2) {
      row.window = [row.coupling];
      row.windowMean = row.coupling;
      row.squaredDeviations = [0];
      row.sampleVariance = 0;
      row.crcv = 0;
      return;
    }
    if (index + 1 < effectiveWindow) return;
    const values = valid.slice(index + 1 - effectiveWindow, index + 1).map((item) => item.coupling);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const squares = values.map((value) => (value - mean) ** 2);
    const variance = squares.reduce((sum, value) => sum + value, 0) / (values.length - 1);
    row.window = values;
    row.windowMean = mean;
    row.squaredDeviations = squares;
    row.sampleVariance = variance;
    row.crcv = Math.sqrt(variance);
  });
  return valid;
}

export function parseJsonl(raw: string): Prediction[] {
  return raw.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as Prediction);
}
