import { describe, expect, it } from 'vitest';
import { computeFeatures, rollingSampleStd, sampleStd, tokenCalculations } from './metrics';

describe('browser metrics mirror the Python implementation', () => {
  it('uses sample rather than population standard deviation', () => {
    expect(sampleStd([1, 2, 3])).toBeCloseTo(1);
  });

  it('keeps complete trailing windows only', () => {
    expect(rollingSampleStd([1, 2, 3, 4], 3)).toEqual([1, 1]);
  });

  it('uses the documented zero convention for one valid shift', () => {
    const prediction = {
      hidden_shifts: [null, 1],
      confidences: [0.8, 0.7],
      token_pieces: ['A', 'B'],
    } as Parameters<typeof tokenCalculations>[0];
    expect(tokenCalculations(prediction)[0].crcv).toBe(0);
  });

  it('couples confidence and normalized hidden shift', () => {
    const result = computeFeatures([0.5, 0.5, 1], [null, 2, 1], 5);
    expect(result.crcv_mean).toBeCloseTo(0);
    expect(result.mean_nll).toBeCloseTo((Math.log(2) * 2) / 3);
    expect(result.answer_tokens).toBe(3);
  });

  it('summarizes the least-confident token tail', () => {
    const result = computeFeatures([1, 0.5, 0.25, 0.125], [null, 1, 1, 1], 3);
    expect(result.top3_token_surprise).toBeCloseTo((Math.log(2) + Math.log(4) + Math.log(8)) / 3);
    expect(result.worst_token_surprise).toBeCloseTo(Math.log(8));
    expect(result.surprise_spread).toBeGreaterThan(0);
  });
});
