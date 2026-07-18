import { describe, expect, it } from 'vitest';
import { computeFeatures, rollingSampleStd, sampleStd } from './metrics';

describe('browser metrics mirror the Python implementation', () => {
  it('uses sample rather than population standard deviation', () => {
    expect(sampleStd([1, 2, 3])).toBeCloseTo(1);
  });

  it('keeps complete trailing windows only', () => {
    expect(rollingSampleStd([1, 2, 3, 4], 3)).toEqual([1, 1]);
  });

  it('couples confidence and normalized hidden shift', () => {
    const result = computeFeatures([0.5, 0.5, 1], [null, 2, 1], 5);
    expect(result.crcv_mean).toBeCloseTo(0);
    expect(result.mean_nll).toBeCloseTo((Math.log(2) * 2) / 3);
    expect(result.answer_tokens).toBe(3);
  });
});
