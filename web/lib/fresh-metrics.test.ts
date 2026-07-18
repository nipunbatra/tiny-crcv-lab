import { describe, expect, it } from 'vitest';
import instructJson from '../../outputs/fresh_qa_qwen05b_instruct/metrics.json';
import baseJson from '../../outputs/fresh_qa_qwen05b_base/metrics.json';
import type { FreshComparisonMetrics, FreshMethodKey } from '../types';

const instruct = instructJson as unknown as FreshComparisonMetrics;
const base = baseJson as unknown as FreshComparisonMetrics;
const methods: FreshMethodKey[] = [
  'top3_token_surprise',
  'p_true',
  'hidden_logistic_probe',
  'semantic_disagreement_3',
  'answer_tokens',
];

describe('fresh benchmark publishing contract', () => {
  it('publishes every frozen method for both 300-question held-out sets', () => {
    for (const metrics of [instruct, base]) {
      expect(metrics.held_out_examples).toBe(300);
      expect(Object.keys(metrics.methods)).toEqual(methods);
      expect(metrics.reliable_improvements_over_top3).toEqual([]);
    }
  });

  it('keeps the one-class Base TruthfulQA slice explicitly undefined', () => {
    const slice = base.methods.top3_token_surprise.by_dataset.truthful_qa;
    expect(slice.examples).toBe(100);
    expect(slice.correct).toBe(0);
    expect(slice.incorrect).toBe(100);
    expect(slice.auroc).toBeNull();
    expect(slice.auroc_ci_95).toEqual([null, null]);
  });

  it('retains top-3 surprise as the robust recommendation', () => {
    expect(instruct.descriptive_best_non_confound).toBe('top3_token_surprise');
    expect(instruct.methods.top3_token_surprise.held_out.auroc).toBeCloseTo(0.656, 3);
    expect(base.methods.p_true.held_out.auroc).toBeGreaterThan(
      base.methods.top3_token_surprise.held_out.auroc!,
    );
    expect(base.methods.p_true.paired_vs_top3_token_surprise?.reliably_improves).toBe(false);
  });
});
