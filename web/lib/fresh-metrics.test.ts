import { describe, expect, it } from 'vitest';
import instructJson from '../../outputs/fresh_qa_qwen05b_instruct/metrics.json';
import baseJson from '../../outputs/fresh_qa_qwen05b_base/metrics.json';
import smolJson from '../../outputs/fresh_qa_smollm2_360m_instruct/metrics.json';
import replicationJson from '../../outputs/fresh_qa_cross_model_replication.json';
import type { CrossModelReplication, FreshComparisonMetrics, FreshMethodKey } from '../types';

const instruct = instructJson as unknown as FreshComparisonMetrics;
const base = baseJson as unknown as FreshComparisonMetrics;
const smol = smolJson as unknown as FreshComparisonMetrics;
const replication = replicationJson as unknown as CrossModelReplication;
const originalMethods: FreshMethodKey[] = [
  'top3_token_surprise',
  'p_true',
  'hidden_logistic_probe',
  'semantic_disagreement_3',
  'answer_tokens',
];

describe('fresh benchmark publishing contract', () => {
  it('preserves the five frozen methods and publishes the 31-method extension', () => {
    for (const metrics of [instruct, base]) {
      expect(metrics.held_out_examples).toBe(300);
      expect(metrics.original_frozen_method_order).toEqual(originalMethods);
      expect(metrics.all_method_order).toHaveLength(31);
      expect(Object.keys(metrics.methods)).toEqual(metrics.all_method_order);
      expect(metrics.scalar_method_order).toHaveLength(24);
      expect(metrics.posthoc_paired_intervals_above_zero).toEqual([]);
      expect(metrics.extension_status).toContain('post-hoc exploratory');
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

  it('keeps the frozen recommendation separate from observed post-hoc leaders', () => {
    expect(instruct.methods.top3_token_surprise.held_out.auroc).toBeCloseTo(0.656, 3);
    expect(instruct.methods.crcv_mean.held_out.auroc).toBeCloseTo(0.634, 3);
    expect(base.methods.crcv_mean.held_out.auroc).toBeCloseTo(0.537, 3);
    expect(base.methods.p_true.held_out.auroc).toBeGreaterThan(
      base.methods.top3_token_surprise.held_out.auroc!,
    );
    expect(base.methods.p_true.paired_vs_top3_token_surprise?.reliably_improves).toBe(false);
    expect(instruct.posthoc_descriptive_best_non_confound).toBe('lexical_disagreement_3');
    expect(base.posthoc_descriptive_best_non_confound).toBe('token_ambiguity_max');
    expect(instruct.methods.lexical_disagreement_3.held_out.auroc).toBeCloseTo(0.717, 3);
    expect(base.methods.token_ambiguity_max.held_out.auroc).toBeCloseTo(0.651, 3);
  });

  it('publishes the prospectively frozen SmolLM2 replication', () => {
    expect(smol.held_out_examples).toBe(300);
    expect(smol.held_out_incorrect).toBe(253);
    expect(smol.all_method_order).toHaveLength(31);
    expect(smol.extension_status).toContain('predeclared cross-model replication');
    expect(smol.methods.hidden_logistic_probe.implementation_note).toContain('960');
    expect(smol.methods.top3_token_surprise.held_out.auroc).toBeCloseTo(0.715, 3);
    expect(smol.methods.surprise_spread.held_out.auroc).toBeCloseTo(0.731, 3);
    expect(smol.methods.worst_token_surprise.held_out.auroc).toBeCloseTo(0.745, 3);
    expect(smol.posthoc_paired_intervals_above_zero).toEqual([]);
  });

  it('applies the frozen cross-model direction rule without selecting by SmolLM2 results', () => {
    expect(replication.method_count).toBe(31);
    expect(replication.replicated_method_count).toBe(25);
    expect(replication.best_worst_model_method).toBe('surprise_spread');
    expect(replication.best_worst_model_auroc).toBeCloseTo(0.706, 3);
    expect(replication.replicated_methods).toContain('top3_token_surprise');
    expect(replication.replicated_methods).toContain('crcv_mean');
    expect(replication.replicated_methods).not.toContain('hidden_cosine_mean');
  });
});
