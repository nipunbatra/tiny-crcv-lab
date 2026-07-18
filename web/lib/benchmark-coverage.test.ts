import { describe, expect, it } from 'vitest';
import instructMetrics from '../../outputs/qwen05b_100/metrics.json';
import baseMetrics from '../../outputs/qwen05b_base_100/metrics.json';
import haluevalMetrics from '../../outputs/halueval_qwen05b_100/metrics.json';

describe('published benchmark coverage', () => {
  it('contains the same 24 scalar methods in every evaluation context', () => {
    const instructKeys = Object.keys(instructMetrics.scores).sort();
    const baseKeys = Object.keys(baseMetrics.scores).sort();
    const haluevalKeys = Object.keys(haluevalMetrics.scores).sort();

    expect(instructKeys).toHaveLength(24);
    expect(baseKeys).toEqual(instructKeys);
    expect(haluevalKeys).toEqual(instructKeys);
  });

  it('publishes held-out results and confidence intervals for every method', () => {
    for (const metrics of [instructMetrics, baseMetrics, haluevalMetrics]) {
      expect(metrics.calibration_examples).toBe(50);
      expect(metrics.test_examples).toBe(50);
      for (const score of Object.values(metrics.scores)) {
        expect(Number.isFinite(score.test_auroc)).toBe(true);
        expect(score.test_auroc_ci_95).toHaveLength(2);
        expect(score.test_auroc_ci_95[0]).toBeLessThanOrEqual(score.test_auroc);
        expect(score.test_auroc_ci_95[1]).toBeGreaterThanOrEqual(score.test_auroc);
      }
    }
  });
});
