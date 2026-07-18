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

export function computeFeatures(confidences: number[], shifts: Array<number | null>, window = WINDOW): Features {
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
  return {
    crcv_mean: mean(crcv),
    crcv_max: Math.max(...crcv),
    confidence_variance_mean: mean(confidenceCrcv),
    shift_variance_mean: mean(shiftCrcv),
    mean_nll: tokenSurprises.length ? mean(tokenSurprises) : 0,
    top3_token_surprise: largestSurprises.length ? mean(largestSurprises) : 0,
    worst_token_surprise: largestSurprises[0] ?? 0,
    surprise_spread: sampleStd(tokenSurprises),
    answer_tokens: confidences.length,
  };
}

export function liveFeatures(tokens: LiveToken[]): Features {
  return computeFeatures(
    tokens.map((token) => token.confidence),
    tokens.map((token) => token.hiddenShift),
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
      shift,
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
