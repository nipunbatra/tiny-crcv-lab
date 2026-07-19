import {
  ArrowRight,
  BookOpenText,
  CheckCircle,
  Cpu,
  Flask,
  Gauge,
  MagnifyingGlass,
  Play,
  WarningCircle,
} from '@phosphor-icons/react';
import { useEffect, useMemo, useRef, useState } from 'react';
import instructRaw from '../outputs/qwen05b_100/predictions.jsonl?raw';
import baseRaw from '../outputs/qwen05b_base_100/predictions.jsonl?raw';
import instructMetricsJson from '../outputs/qwen05b_100/metrics.json';
import baseMetricsJson from '../outputs/qwen05b_base_100/metrics.json';
import haluevalRaw from '../outputs/halueval_qwen05b_100/predictions.jsonl?raw';
import haluevalMetricsJson from '../outputs/halueval_qwen05b_100/metrics.json';
import freshInstructMetricsJson from '../outputs/fresh_qa_qwen05b_instruct/metrics.json';
import freshBaseMetricsJson from '../outputs/fresh_qa_qwen05b_base/metrics.json';
import freshSmolMetricsJson from '../outputs/fresh_qa_smollm2_360m_instruct/metrics.json';
import crossModelReplicationJson from '../outputs/fresh_qa_cross_model_replication.json';
import shallowTreeJson from '../outputs/shallow_tree_results.json';
import questionTokensJson from './data/question-tokens.json';
import { FreshBenchmark, FreshSummary } from './FreshEvidence';
import { parseJsonl, rollingSampleStd, tokenCalculations } from './lib/metrics';
import type {
  BenchmarkMetrics,
  CrossModelReplication,
  FeatureKey,
  Features,
  FreshComparisonMetrics,
  FreshModelKind,
  HaluEvalMetrics,
  HaluEvalPrediction,
  LiveToken,
  ModelKind,
  Prediction,
  ShallowTreeResults,
  TreeRule,
  WorkerToMain,
} from './types';

const predictions: Record<ModelKind, Prediction[]> = {
  instruct: parseJsonl(instructRaw),
  base: parseJsonl(baseRaw),
};
const benchmarks: Record<ModelKind, BenchmarkMetrics> = {
  instruct: instructMetricsJson as unknown as BenchmarkMetrics,
  base: baseMetricsJson as unknown as BenchmarkMetrics,
};
const haluevalPredictions = parseJsonl<HaluEvalPrediction>(haluevalRaw);
const haluevalMetrics = haluevalMetricsJson as unknown as HaluEvalMetrics;
const shallowTrees = shallowTreeJson as unknown as ShallowTreeResults;
const freshMetrics: Record<FreshModelKind, FreshComparisonMetrics> = {
  qwen_instruct: freshInstructMetricsJson as unknown as FreshComparisonMetrics,
  smollm2_instruct: freshSmolMetricsJson as unknown as FreshComparisonMetrics,
  qwen_base: freshBaseMetricsJson as unknown as FreshComparisonMetrics,
};
const crossModelReplication = crossModelReplicationJson as unknown as CrossModelReplication;
const questionTokens = questionTokensJson as Record<string, Array<{ id: number; piece: string }>>;
type AppView = 'overview' | 'explore' | 'live';

const metricKeys: FeatureKey[] = [
  'token_entropy_top3',
  'skip_first_top3_surprise',
  'top3_token_surprise',
  'top4_token_surprise',
  'surprise_shift_top3',
  'uncertainty_shift_top3',
  'token_entropy_max',
  'token_ambiguity_top3',
  'token_ambiguity_max',
  'token_entropy_mean',
  'token_ambiguity_mean',
  'worst_token_surprise',
  'surprise_spread',
  'confidence_variance_mean',
  'mean_nll',
  'hidden_norm_mean',
  'crcv_max',
  'crcv_mean',
  'hidden_cosine_max',
  'hidden_cosine_top3',
  'hidden_cosine_mean',
  'hidden_norm_variability',
  'shift_variance_mean',
  'answer_tokens',
];

const metricShort: Record<FeatureKey, string> = {
  token_entropy_top3: 'Top-3 entropy',
  token_entropy_mean: 'Mean entropy',
  token_entropy_max: 'Max entropy',
  token_ambiguity_top3: 'Top-3 ambiguity',
  token_ambiguity_mean: 'Mean ambiguity',
  token_ambiguity_max: 'Max ambiguity',
  top3_token_surprise: 'Top-3 surprise',
  top4_token_surprise: 'Top-4 surprise',
  skip_first_top3_surprise: 'Top-3 after first',
  surprise_shift_top3: 'Surprise × shift',
  uncertainty_shift_top3: 'Uncertainty × shift',
  worst_token_surprise: 'Worst-token surprise',
  surprise_spread: 'Surprise spread',
  hidden_cosine_mean: 'Mean cosine change',
  hidden_cosine_max: 'Max cosine change',
  hidden_cosine_top3: 'Top-3 cosine change',
  hidden_norm_mean: 'Mean hidden norm',
  hidden_norm_variability: 'Hidden-norm var.',
  crcv_mean: 'CRCV mean',
  crcv_max: 'CRCV max',
  mean_nll: 'Token surprise',
  confidence_variance_mean: 'Confidence var.',
  shift_variance_mean: 'Hidden-shift var.',
  answer_tokens: 'Answer length',
};

type MetricDefinition = {
  formula: string;
  plain: string;
  input: string;
  group: 'distribution' | 'confidence' | 'coupled' | 'hidden' | 'control';
};

const metricDefinitions: Record<FeatureKey, MetricDefinition> = {
  token_entropy_top3: { formula: 'mean(top 3 of eₜ),  eₜ = −Σᵥpₜᵥln(pₜᵥ) / ln|V|', plain: 'Average full-vocabulary uncertainty at the three most uncertain token decisions.', input: 'the full next-token probability distribution pₜ over 151,936 tokens', group: 'distribution' },
  token_entropy_mean: { formula: 'mean(eₜ),  eₜ = −Σᵥpₜᵥln(pₜᵥ) / ln|V|', plain: 'Average normalized uncertainty of the full vocabulary distribution.', input: 'the full next-token probability distribution pₜ', group: 'distribution' },
  token_entropy_max: { formula: 'maxₜ(eₜ)', plain: 'The single most diffuse next-token decision in the answer.', input: 'normalized token entropies eₜ', group: 'distribution' },
  token_ambiguity_top3: { formula: 'mean(top 3 of aₜ),  aₜ = 1 − (p₁ₜ − p₂ₜ)', plain: 'Average ambiguity at the three decisions where the winner was least separated from the runner-up.', input: 'largest and second-largest next-token probabilities', group: 'distribution' },
  token_ambiguity_mean: { formula: 'mean(aₜ),  aₜ = 1 − (p₁ₜ − p₂ₜ)', plain: 'Average closeness of the winning token to its runner-up.', input: 'top-1 and top-2 next-token probabilities', group: 'distribution' },
  token_ambiguity_max: { formula: 'maxₜ(aₜ)', plain: 'The most ambiguous winner-versus-runner-up decision.', input: 'token ambiguity aₜ', group: 'distribution' },
  top3_token_surprise: { formula: 'T₃ = (1/k) Σ top-k(uₜ),  uₜ = −ln(cₜ),  k = min(3,n)', plain: 'Average surprise of up to the three least-confident generated tokens.', input: 'selected-token confidences cₜ', group: 'confidence' },
  top4_token_surprise: { formula: 'T₄ = (1/k) Σ top-k(uₜ),  k = min(4,n)', plain: 'The same confidence-tail idea averaged over four tokens.', input: 'selected-token surprises uₜ = −ln(cₜ)', group: 'confidence' },
  skip_first_top3_surprise: { formula: 'mean(top 3 of uₜ for t > 1)', plain: 'Top-3 surprise after removing the often-boilerplate first generated token.', input: 'token surprise and token position', group: 'confidence' },
  worst_token_surprise: { formula: 'Umax = maxₜ(−ln(cₜ))', plain: 'Surprise of the single least-confident generated token.', input: 'selected-token confidences cₜ', group: 'confidence' },
  surprise_spread: { formula: 'SDᵤ = sample-SD(−ln(c₁), …, −ln(cₙ))', plain: 'How unevenly token surprise is distributed across the answer.', input: 'selected-token confidences cₜ', group: 'confidence' },
  confidence_variance_mean: { formula: 'mean_w sample-SD(cₜ in trailing window w)', plain: 'Average local wobble in selected-token confidence.', input: 'confidences with a valid state shift; W = 5', group: 'confidence' },
  mean_nll: { formula: 'NLL = (1/n) Σₜ −ln(cₜ)', plain: 'Average surprise across every generated token.', input: 'selected-token confidences cₜ', group: 'confidence' },
  surprise_shift_top3: { formula: 'mean(top 3 of (−ln(cₜ))rₜ)', plain: 'A corrected CRCV-style coupling that grows with uncertainty instead of confidence.', input: 'token surprise and normalized hidden shift', group: 'coupled' },
  uncertainty_shift_top3: { formula: 'mean(top 3 of (1−cₜ)rₜ)', plain: 'A bounded uncertainty × hidden-movement coupling.', input: 'selected-token confidence and normalized hidden shift', group: 'coupled' },
  crcv_max: { formula: 'CRCVmax = max_w sample-SD(cₜrₜ in w)', plain: 'Largest local burst in the confidence × hidden-movement signal.', input: 'confidences cₜ and normalized shifts rₜ; W = 5', group: 'coupled' },
  crcv_mean: { formula: 'CRCVmean = mean_w sample-SD(cₜrₜ in w)', plain: 'Average local variability in confidence × hidden-state movement.', input: 'confidences cₜ and normalized shifts rₜ; W = 5', group: 'coupled' },
  hidden_cosine_mean: { formula: 'mean(dₜ),  dₜ = 1 − cos(hₜ,hₜ₋₁)', plain: 'Average directional turn in the final hidden state.', input: 'consecutive final hidden-state vectors', group: 'hidden' },
  hidden_cosine_max: { formula: 'maxₜ(dₜ)', plain: 'Largest directional turn between consecutive final states.', input: 'hidden-state cosine distances dₜ', group: 'hidden' },
  hidden_cosine_top3: { formula: 'mean(top 3 of dₜ)', plain: 'Average of the three largest final-state direction changes.', input: 'hidden-state cosine distances dₜ', group: 'hidden' },
  hidden_norm_mean: { formula: 'mean(‖hₜ‖₂ / √896)', plain: 'Average RMS magnitude of the final hidden state.', input: 'the 896-number final hidden state hₜ', group: 'hidden' },
  hidden_norm_variability: { formula: 'sample-SD(‖hₜ‖₂ / √896)', plain: 'How much final-state magnitude changes across the answer.', input: 'per-token hidden-state RMS norms', group: 'hidden' },
  shift_variance_mean: { formula: 'mean_w sample-SD(rₜ in trailing window w)', plain: 'Average local wobble in normalized hidden-state movement.', input: 'normalized final-state shifts rₜ; W = 5', group: 'hidden' },
  answer_tokens: { formula: 'L = n', plain: 'Number of generated tokens; a control for answer length.', input: 'generated token pieces', group: 'control' },
};

const format = (value: number, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const pct = (value: number) => `${Math.round(value * 100)}%`;

type WorkedExample = { inputs: string; arithmetic: string; score: number };

function workedExample(key: FeatureKey, prediction: Prediction): WorkedExample {
  const surprises = prediction.confidences.map((confidence, index) => ({
    confidence,
    surprise: -Math.log(Math.max(confidence, 1e-12)),
    token: prediction.token_pieces[index],
  }));
  const descending = [...surprises].sort((left, right) => right.surprise - left.surprise);
  const entropyItems = prediction.token_entropies.map((value, index) => ({ value, token: prediction.token_pieces[index] }));
  const ambiguityItems = prediction.token_margins.map((margin, index) => ({ value: 1 - margin, token: prediction.token_pieces[index], margin }));
  const cosineItems = prediction.hidden_cosine_distances.flatMap((value, index) => value === null ? [] : [{ value, token: prediction.token_pieces[index] }]);
  const normItems = prediction.hidden_norms.map((value, index) => ({ value, token: prediction.token_pieces[index] }));
  const stateItems = prediction.hidden_shifts.flatMap((shift, index) => shift === null ? [] : [{
    token: prediction.token_pieces[index],
    confidence: prediction.confidences[index],
    shift,
    surpriseShift: -Math.log(Math.max(prediction.confidences[index], 1e-12)) * shift,
    uncertaintyShift: (1 - prediction.confidences[index]) * shift,
  }]);
  const calculations = tokenCalculations(prediction);
  const confidenceWindows = rollingSampleStd(calculations.map((row) => row.confidence));
  const shiftWindows = rollingSampleStd(calculations.map((row) => row.shift));
  const crcvWindows = calculations.flatMap((row) => row.crcv === null ? [] : [row.crcv]);
  const numberList = (values: number[], digits = 3) => `[${values.map((value) => format(value, digits)).join(', ')}]`;
  const meanArithmetic = (values: number[]) => `mean ${numberList(values)} = ${format(values.reduce((sum, value) => sum + value, 0) / values.length, 4)}`;
  const topItems = <T extends { value: number },>(items: T[], count = 3) => [...items].sort((left, right) => right.value - left.value).slice(0, count);
  const itemInputs = (items: Array<{ token: string; value: number }>, symbol: string) => items.map((item) => `${JSON.stringify(item.token)}: ${symbol}=${format(item.value, 4)}`).join(' · ');

  if (key === 'token_entropy_top3') {
    const selected = topItems(entropyItems);
    return { inputs: itemInputs(selected, 'e'), arithmetic: meanArithmetic(selected.map((item) => item.value)), score: prediction.features[key] };
  }
  if (key === 'token_entropy_mean') {
    return { inputs: `eₜ = ${numberList(prediction.token_entropies)}`, arithmetic: meanArithmetic(prediction.token_entropies), score: prediction.features[key] };
  }
  if (key === 'token_entropy_max') {
    const selected = topItems(entropyItems, 1)[0];
    return { inputs: itemInputs([selected], 'e'), arithmetic: `max(eₜ) = ${format(selected.value, 4)}`, score: prediction.features[key] };
  }
  if (key === 'token_ambiguity_top3') {
    const selected = topItems(ambiguityItems);
    return { inputs: selected.map((item) => `${JSON.stringify(item.token)}: margin=${format(item.margin, 4)} → a=${format(item.value, 4)}`).join(' · '), arithmetic: meanArithmetic(selected.map((item) => item.value)), score: prediction.features[key] };
  }
  if (key === 'token_ambiguity_mean') {
    const values = ambiguityItems.map((item) => item.value);
    return { inputs: `aₜ = 1−margin = ${numberList(values)}`, arithmetic: meanArithmetic(values), score: prediction.features[key] };
  }
  if (key === 'token_ambiguity_max') {
    const selected = topItems(ambiguityItems, 1)[0];
    return { inputs: `${JSON.stringify(selected.token)}: 1−${format(selected.margin, 4)} = ${format(selected.value, 4)}`, arithmetic: `max(aₜ) = ${format(selected.value, 4)}`, score: prediction.features[key] };
  }

  if (key === 'top3_token_surprise') {
    const selected = descending.slice(0, 3);
    return {
      inputs: selected.map((item) => `${JSON.stringify(item.token)}: c=${format(item.confidence, 4)} → u=${format(item.surprise, 4)}`).join(' · '),
      arithmetic: `(${selected.map((item) => format(item.surprise, 4)).join(' + ')}) / ${selected.length} = ${format(prediction.features[key], 4)}`,
      score: prediction.features[key],
    };
  }
  if (key === 'top4_token_surprise') {
    const selected = descending.slice(0, 4);
    return { inputs: selected.map((item) => `${JSON.stringify(item.token)}: u=${format(item.surprise, 4)}`).join(' · '), arithmetic: meanArithmetic(selected.map((item) => item.surprise)), score: prediction.features[key] };
  }
  if (key === 'skip_first_top3_surprise') {
    const selected = [...surprises.slice(1)].sort((left, right) => right.surprise - left.surprise).slice(0, 3);
    return { inputs: `first token removed · ${selected.map((item) => `${JSON.stringify(item.token)}: u=${format(item.surprise, 4)}`).join(' · ')}`, arithmetic: meanArithmetic(selected.map((item) => item.surprise)), score: prediction.features[key] };
  }
  if (key === 'worst_token_surprise') {
    const selected = descending[0];
    return { inputs: `${JSON.stringify(selected.token)} has the lowest cₜ = ${format(selected.confidence, 5)}`, arithmetic: `−ln(${format(selected.confidence, 5)}) = ${format(selected.surprise, 4)}`, score: prediction.features[key] };
  }
  if (key === 'surprise_spread') {
    return { inputs: `uₜ = ${numberList(surprises.map((item) => item.surprise))}`, arithmetic: `sample-SD(uₜ) = ${format(prediction.features[key], 4)}`, score: prediction.features[key] };
  }
  if (key === 'surprise_shift_top3') {
    const selected = [...stateItems].sort((left, right) => right.surpriseShift - left.surpriseShift).slice(0, 3);
    return { inputs: selected.map((item) => `${JSON.stringify(item.token)}: −ln(${format(item.confidence, 3)})×${format(item.shift, 3)}=${format(item.surpriseShift, 4)}`).join(' · '), arithmetic: meanArithmetic(selected.map((item) => item.surpriseShift)), score: prediction.features[key] };
  }
  if (key === 'uncertainty_shift_top3') {
    const selected = [...stateItems].sort((left, right) => right.uncertaintyShift - left.uncertaintyShift).slice(0, 3);
    return { inputs: selected.map((item) => `${JSON.stringify(item.token)}: (1−${format(item.confidence, 3)})×${format(item.shift, 3)}=${format(item.uncertaintyShift, 4)}`).join(' · '), arithmetic: meanArithmetic(selected.map((item) => item.uncertaintyShift)), score: prediction.features[key] };
  }
  if (key === 'confidence_variance_mean') {
    return { inputs: `window SDs = ${numberList(confidenceWindows)}`, arithmetic: meanArithmetic(confidenceWindows), score: prediction.features[key] };
  }
  if (key === 'mean_nll') {
    return { inputs: `uₜ = ${numberList(surprises.map((item) => item.surprise))}`, arithmetic: meanArithmetic(surprises.map((item) => item.surprise)), score: prediction.features[key] };
  }
  if (key === 'crcv_max') {
    return { inputs: `window CRCVs = ${numberList(crcvWindows)}`, arithmetic: `max = ${format(Math.max(...crcvWindows), 4)}`, score: prediction.features[key] };
  }
  if (key === 'crcv_mean') {
    return { inputs: `window CRCVs = ${numberList(crcvWindows)}`, arithmetic: meanArithmetic(crcvWindows), score: prediction.features[key] };
  }
  if (key === 'shift_variance_mean') {
    return { inputs: `window SDs = ${numberList(shiftWindows)}`, arithmetic: meanArithmetic(shiftWindows), score: prediction.features[key] };
  }
  if (key === 'hidden_cosine_mean') {
    const values = cosineItems.map((item) => item.value);
    return { inputs: `dₜ = ${numberList(values)}`, arithmetic: meanArithmetic(values), score: prediction.features[key] };
  }
  if (key === 'hidden_cosine_max') {
    const selected = topItems(cosineItems, 1)[0];
    return { inputs: itemInputs([selected], 'd'), arithmetic: `max(dₜ) = ${format(selected.value, 4)}`, score: prediction.features[key] };
  }
  if (key === 'hidden_cosine_top3') {
    const selected = topItems(cosineItems);
    return { inputs: itemInputs(selected, 'd'), arithmetic: meanArithmetic(selected.map((item) => item.value)), score: prediction.features[key] };
  }
  if (key === 'hidden_norm_mean') {
    const values = normItems.map((item) => item.value);
    return { inputs: `RMS norms = ${numberList(values)}`, arithmetic: meanArithmetic(values), score: prediction.features[key] };
  }
  if (key === 'hidden_norm_variability') {
    const values = normItems.map((item) => item.value);
    return { inputs: `RMS norms = ${numberList(values)}`, arithmetic: `sample-SD = ${format(prediction.features[key], 4)}`, score: prediction.features[key] };
  }
  return { inputs: prediction.token_pieces.map((token) => JSON.stringify(token)).join(' · '), arithmetic: `count = ${prediction.token_pieces.length}`, score: prediction.features[key] };
}

function ModelToggle({ value, onChange }: { value: ModelKind; onChange: (value: ModelKind) => void }) {
  return (
    <div className="inline-flex border hairline bg-white p-1" role="group" aria-label="Model family">
      {(['instruct', 'base'] as ModelKind[]).map((model) => (
        <button
          key={model}
          className={`focus-ring px-4 py-2 text-sm font-semibold transition ${value === model ? 'bg-[#18211d] text-white' : 'text-[#69716d] hover:text-[#18211d]'}`}
          onClick={() => onChange(model)}
        >
          {model === 'instruct' ? '0.5B Instruct' : '0.5B Base'}
        </button>
      ))}
    </div>
  );
}

function Header({ model, onModel, view, onView }: { model: ModelKind; onModel: (value: ModelKind) => void; view: AppView; onView: (value: AppView) => void }) {
  return (
    <header className="border-b hairline bg-[#f4f1ea]/95 py-4 backdrop-blur">
      <div className="shell flex flex-wrap items-center justify-between gap-4">
        <button onClick={() => onView('overview')} className="focus-ring flex items-center gap-3 text-sm font-bold tracking-[-.02em]">
          <span className="grid h-8 w-8 place-items-center bg-[#df4c2f] text-white"><Flask size={18} weight="fill" /></span>
          TINY CRCV LAB
        </button>
        <nav className="order-3 flex w-full items-center border hairline bg-white p-1 text-sm md:order-none md:w-auto" aria-label="Main views">
          {([
            ['overview', 'Conclusion'],
            ['explore', 'Evidence'],
            ['live', 'Try it'],
          ] as Array<[AppView, string]>).map(([value, label]) => (
            <button
              key={value}
              onClick={() => onView(value)}
              aria-current={view === value ? 'page' : undefined}
              className={`focus-ring flex-1 px-4 py-2 font-medium transition md:flex-none ${view === value ? 'bg-[#18211d] text-white' : 'text-[#69716d] hover:text-[#18211d]'}`}
            >
              {label}
            </button>
          ))}
          <a
            href="./talk/s/tiny-hallucination-detector"
            className="focus-ring flex-1 px-4 py-2 text-center font-medium text-[#69716d] transition hover:text-[#18211d] md:flex-none"
          >
            Slides
          </a>
        </nav>
        {view === 'overview' ? <span className="hidden text-xs font-medium text-[#69716d] lg:block">600 questions · two model families · 31 inspectable scores</span> : <div className="flex items-center gap-3"><span className="hidden text-[10px] uppercase tracking-[.08em] text-[#69716d] xl:block">Qwen live + legacy</span><ModelToggle value={model} onChange={onModel} /></div>}
      </div>
    </header>
  );
}

function Hero({ onExplore, onRun }: { onExplore: () => void; onRun: () => void }) {
  return (
    <section id="top" className="grid-noise border-b hairline">
      <div className="shell grid gap-10 py-14 lg:grid-cols-[1.08fr_.92fr] lg:py-20">
        <div className="max-w-3xl">
          <p className="eyebrow mb-5">Cross-model result</p>
          <h1 className="display max-w-3xl text-[clamp(3rem,6.5vw,6.5rem)] font-[650]">Token uncertainty transfers.</h1>
          <p className="mt-7 max-w-2xl text-lg leading-8 text-[#505955]">
            On the same 300 held-out questions, simple surprise statistics rank wrong answers above correct ones for both Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct. The signal is useful for review order—not for declaring facts false.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <button onClick={onRun} className="focus-ring inline-flex items-center gap-2 bg-[#df4c2f] px-5 py-3 font-semibold text-white">Try one question <Cpu size={18} /></button>
            <button onClick={onExplore} className="focus-ring inline-flex items-center gap-2 border hairline bg-white px-5 py-3 font-semibold">Inspect the evidence <ArrowRight size={18} /></button>
          </div>
        </div>
        <div className="card self-end p-6 lg:p-8">
          <p className="eyebrow">Cross-model recommendation</p>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-.04em]">Surprise spread + top-3</h2>
          <p className="mono mt-4 bg-[#18211d] p-4 text-sm leading-6 text-white">spread = sample-SDₜ(−ln pₜ)<br />top-3 = mean(largest 3 of −ln pₜ)</p>
          <div className="mt-6 grid grid-cols-2 gap-px bg-[#d7d3c9]">
            <div className="bg-[#fffdf8] p-4"><p className="text-[10px] uppercase text-[#69716d]">Qwen spread AUROC</p><p className="mono mt-1 text-2xl">0.706</p></div>
            <div className="bg-[#fffdf8] p-4"><p className="text-[10px] uppercase text-[#69716d]">SmolLM2 spread AUROC</p><p className="mono mt-1 text-2xl">0.731</p></div>
          </div>
          <div className="mt-5 space-y-3 text-sm leading-6">
            <p><strong>Use:</strong> surprise spread as the lead free candidate.</p>
            <p><strong>Keep:</strong> top-3 surprise as the frozen comparison anchor.</p>
            <p><strong>Calibrate:</strong> fit the review cutoff separately for each model.</p>
          </div>
          <p className="mt-5 bg-[#fbe9e2] p-3 text-xs leading-5 text-[#6f2d20]">The labels are strict alias matches, not human factuality judgments. A high score means “review this,” not “this is false.”</p>
        </div>
      </div>
    </section>
  );
}

function BeginnerOverview({ onExplore }: { onExplore: () => void }) {
  const comparison = [
    { name: 'Top-3 token surprise', key: 'top3_token_surprise' as FeatureKey, recommended: true },
    { name: 'Top-3 entropy', key: 'token_entropy_top3' as FeatureKey },
    { name: 'Depth-2 tree', tree: true },
    { name: 'CRCV hidden-state score', key: 'crcv_mean' as FeatureKey },
    { name: 'Answer length control', key: 'answer_tokens' as FeatureKey },
  ];
  const comparisonScores = comparison.map((row) => ({
    ...row,
    instruct: row.tree ? shallowTrees.models.instruct.depth2_tree.test_auroc : benchmarks.instruct.scores[row.key!].test_auroc,
    base: row.tree ? shallowTrees.models.base.depth2_tree.test_auroc : benchmarks.base.scores[row.key!].test_auroc,
    halueval: row.tree ? haluevalMetrics.depth2_tree.test_auroc : haluevalMetrics.scores[row.key!].test_auroc,
    decision: row.recommended ? 'Most consistent simple signal.' : row.key === 'token_entropy_top3' ? 'Useful secondary diagnostic.' : row.tree ? 'No consistent gain.' : row.key === 'crcv_mean' ? 'Do not recommend.' : 'Confound check only.',
  }));
  return (
    <section className="border-b hairline bg-[#eae6dc] py-16">
      <div className="shell">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div><p className="eyebrow">The evidence at a glance</p><h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">Our 100 questions + HaluEval.</h2></div>
          <p className="max-w-xl text-sm leading-6 text-[#69716d]">AUROC: how often a wrong candidate receives a higher risk score than a correct one. 0.500 is chance; 1.000 is perfect ranking.</p>
        </div>
        <div className="card mt-7 hidden overflow-x-auto md:block">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Method</th><th className="p-4">Our generated answers<br /><span className="normal-case font-normal">Instruct · 50 held out</span></th><th className="p-4">Same questions<br /><span className="normal-case font-normal">Base · 50 held out</span></th><th className="p-4">HaluEval QA<br /><span className="normal-case font-normal">25 held-out pairs</span></th><th className="p-4">Decision</th></tr></thead>
            <tbody>{comparisonScores.map((row) => <tr className={`border-t hairline ${row.recommended ? 'bg-[#fff8f4]' : ''}`} key={row.name}><td className="p-4 font-semibold">{row.name}{row.recommended && <span className="ml-2 bg-[#df4c2f] px-2 py-1 text-[9px] uppercase text-white">recommended</span>}</td><td className="mono p-4 text-lg">{format(row.instruct)}</td><td className="mono p-4 text-lg">{format(row.base)}</td><td className="mono p-4 text-lg">{format(row.halueval)}</td><td className="p-4 text-xs leading-5 text-[#69716d]">{row.decision}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="mt-6 space-y-3 md:hidden">
          {comparisonScores.map((row) => <article className={`card p-4 ${row.recommended ? 'border-l-4 !border-l-[#df4c2f] bg-[#fff8f4]' : ''}`} key={`mobile-${row.name}`}><div className="flex items-start justify-between gap-3"><p className="font-semibold">{row.name}</p>{row.recommended && <span className="bg-[#df4c2f] px-2 py-1 text-[9px] uppercase text-white">recommended</span>}</div><div className="mt-4 grid grid-cols-3 gap-2"><div><p className="text-[9px] uppercase text-[#69716d]">Instruct</p><p className="mono mt-1 text-lg">{format(row.instruct)}</p></div><div><p className="text-[9px] uppercase text-[#69716d]">Base</p><p className="mono mt-1 text-lg">{format(row.base)}</p></div><div><p className="text-[9px] uppercase text-[#69716d]">HaluEval</p><p className="mono mt-1 text-lg">{format(row.halueval)}</p></div></div><p className="mt-3 border-t hairline pt-3 text-xs text-[#69716d]">{row.decision}</p></article>)}
        </div>
        <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_1fr]">
          <div className="bg-[#fbe9e2] p-5 text-sm leading-6 text-[#6f2d20]"><strong>Important HaluEval caveat.</strong> Its hallucinated answer was longer in all 25 held-out pairs, so even length scored 0.958. Treat the HaluEval column as a confounded stress test.</div>
          <div className="bg-[#fffdf8] p-5 text-sm leading-6"><strong>Operational recipe.</strong> Compute top-3 surprise → fit the review cutoff on labelled calibration examples → freeze it → measure once on fresh examples.</div>
        </div>
        <button onClick={onExplore} className="focus-ring mt-7 inline-flex items-center gap-2 font-semibold text-[#9e321e]">Compare all 24 methods, formulas, and intervals <ArrowRight size={18} /></button>
      </div>
    </section>
  );
}

function OverviewStep({ number, label, value, note, accent = false }: { number: string; label: string; value: string; note: string; accent?: boolean }) {
  return <div className={`overview-step ${accent ? 'is-accent' : ''}`}><span className="mono text-xs text-[#69716d]">{number.padStart(2, '0')}</span><p className="mt-4 text-sm font-semibold">{label}</p><p className="mt-2 text-balance text-lg font-semibold leading-6">{value}</p><p className="mt-3 text-xs leading-5 text-[#69716d]">{note}</p></div>;
}

function Results({ model }: { model: ModelKind }) {
  const bench = benchmarks[model];
  const [selectedMetric, setSelectedMetric] = useState<FeatureKey>('top3_token_surprise');
  const [metricFilter, setMetricFilter] = useState<'all' | MetricDefinition['group']>('all');
  const [comparisonSort, setComparisonSort] = useState<'robustness' | ModelKind | 'halueval'>('robustness');
  const comparisonRows = metricKeys.map((key) => {
    const instruct = benchmarks.instruct.scores[key].test_auroc;
    const base = benchmarks.base.scores[key].test_auroc;
    const halueval = haluevalMetrics.scores[key].test_auroc;
    return { key, instruct, base, halueval, robustness: Math.min(instruct, base, halueval) };
  }).sort((left, right) => {
    const leftValue = comparisonSort === 'robustness' ? left.robustness : left[comparisonSort];
    const rightValue = comparisonSort === 'robustness' ? right.robustness : right[comparisonSort];
    return rightValue - leftValue || metricKeys.indexOf(left.key) - metricKeys.indexOf(right.key);
  });
  const visibleLeaderboard = comparisonRows.filter(({ key }) => metricFilter === 'all' || metricDefinitions[key].group === metricFilter);
  const selectedContexts = [
    { label: 'Our 100 · Instruct', detail: '50 held-out generations', metric: benchmarks.instruct.scores[selectedMetric] },
    { label: 'Our 100 · Base', detail: 'same 50 held-out questions', metric: benchmarks.base.scores[selectedMetric] },
    { label: 'HaluEval QA', detail: '25 held-out answer pairs', metric: haluevalMetrics.scores[selectedMetric] },
  ];
  return (
    <section id="results" className="shell py-18">
      <div className="mb-8 grid gap-4 border-l-4 border-[#df4c2f] bg-[#fff8f4] p-5 md:grid-cols-[.7fr_1.3fr] md:p-6">
        <div><p className="eyebrow">Recommendation</p><p className="mt-2 text-xl font-semibold">Start with top-3 token surprise.</p></div>
        <p className="text-sm leading-6 text-[#505955]">It was the most consistent simple score across Instruct, Base, and HaluEval. Use entropy as a companion, always check answer length, and keep the tree and CRCV experimental. The tables below are the audit trail.</p>
      </div>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">All methods × all evaluation contexts</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-[-.04em]">The same {metricKeys.length} scores, side by side.</h2>
        </div>
        <p className="max-w-xl text-sm leading-6 text-[#69716d]">Every scalar formula was applied to our Instruct generations, the Base-model stress check, and HaluEval candidates. Select a row to compare its intervals, thresholded F1, formula, and worked arithmetic.</p>
      </div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Metric family filter">
          {(['all', 'distribution', 'confidence', 'coupled', 'hidden', 'control'] as const).map((group) => <button key={group} onClick={() => setMetricFilter(group)} className={`focus-ring px-3 py-2 text-xs font-semibold capitalize ${metricFilter === group ? 'bg-[#18211d] text-white' : 'border hairline bg-white'}`}>{group === 'all' ? `All ${metricKeys.length}` : group}</button>)}
        </div>
        <label className="text-xs font-semibold text-[#69716d]">Rank by<select className="focus-ring ml-2 border hairline bg-white px-3 py-2 text-[#18211d]" value={comparisonSort} onChange={(event) => setComparisonSort(event.target.value as typeof comparisonSort)}><option value="robustness">worst of all three</option><option value="instruct">Instruct AUROC</option><option value="base">Base AUROC</option><option value="halueval">HaluEval AUROC</option></select></label>
      </div>
      <div className="card overflow-x-auto">
        <table className="w-full min-w-[1040px] border-collapse text-left">
          <thead className="bg-[#eeeae1] text-xs uppercase tracking-[.08em] text-[#69716d]">
            <tr><th className="px-4 py-4">Rank</th><th className="px-4 py-4">Scalar method</th><th className="px-4 py-4">Our 100<br /><span className="normal-case font-normal">Instruct test</span></th><th className="px-4 py-4">Our 100<br /><span className="normal-case font-normal">Base test</span></th><th className="px-4 py-4">HaluEval<br /><span className="normal-case font-normal">candidate test</span></th><th className="px-4 py-4">Worst of three</th><th className="px-4 py-4">Inspect</th></tr>
          </thead>
          <tbody>
            {visibleLeaderboard.map((row) => {
              const key = row.key;
              const index = comparisonRows.findIndex((item) => item.key === key);
              return (
                <tr key={key} className={`border-t hairline align-middle ${selectedMetric === key ? 'bg-[#fff1ec]' : ''}`}>
                  <td className="mono px-4 py-5 text-[#69716d]">#{index + 1}</td>
                  <td className="px-4 py-5 font-semibold">{metricShort[key]}{key === 'top3_token_surprise' && <span className="ml-2 bg-[#df4c2f] px-2 py-1 text-[9px] uppercase text-white">recommended</span>}{key === 'crcv_mean' && <span className="ml-2 bg-[#fbe9e2] px-2 py-1 text-[9px] uppercase text-[#9e321e]">original</span>}<span className="ml-2 text-[9px] uppercase text-[#69716d]">{metricDefinitions[key].group}</span></td>
                  <td className="mono px-4 py-5 text-lg">{format(row.instruct)}</td>
                  <td className="mono px-4 py-5 text-lg">{format(row.base)}</td>
                  <td className="mono px-4 py-5 text-lg">{format(row.halueval)}</td>
                  <td className="mono px-4 py-5 text-lg font-semibold">{format(row.robustness)}</td>
                  <td className="px-4 py-5"><button onClick={() => setSelectedMetric(key)} className="focus-ring text-xs font-semibold text-[#9e321e] underline decoration-[#df4c2f]/30 underline-offset-4">formula + intervals</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 grid gap-px bg-[#d7d3c9] p-px sm:grid-cols-4">
        <div className="bg-[#eeeae1] p-4"><p className="eyebrow">Learned method</p><p className="mt-2 font-semibold">Depth-2 tree</p></div>
        <div className="bg-white p-4"><p className="text-[10px] text-[#69716d]">Instruct AUROC</p><p className="mono mt-1 text-lg">{format(shallowTrees.models.instruct.depth2_tree.test_auroc)}</p></div>
        <div className="bg-white p-4"><p className="text-[10px] text-[#69716d]">Base AUROC</p><p className="mono mt-1 text-lg">{format(shallowTrees.models.base.depth2_tree.test_auroc)}</p></div>
        <div className="bg-white p-4"><p className="text-[10px] text-[#69716d]">HaluEval AUROC</p><p className="mono mt-1 text-lg">{format(haluevalMetrics.depth2_tree.test_auroc)}</p><p className="mt-1 text-[10px] text-[#69716d]">Not scalar-ranked; it learns branches.</p></div>
      </div>
      <div className="card mt-5 overflow-hidden">
        <div className="border-b hairline bg-[#eeeae1] p-4 md:p-5"><p className="eyebrow">Selected across every context</p><div className="mt-2 flex flex-wrap items-baseline justify-between gap-3"><h3 className="text-2xl font-semibold">{metricShort[selectedMetric]}</h3><p className="mono text-xs text-[#9e321e]">{metricDefinitions[selectedMetric].formula}</p></div></div>
        <div className="grid gap-px bg-[#d7d3c9] md:grid-cols-3">{selectedContexts.map(({ label, detail, metric }) => <article className="bg-white p-5" key={label}><p className="font-semibold">{label}</p><p className="mt-1 text-xs text-[#69716d]">{detail}</p><p className="mono mt-4 text-3xl font-semibold">{format(metric.test_auroc)}</p><p className="mt-1 text-[10px] uppercase text-[#69716d]">test AUROC · 95% CI {format(metric.test_auroc_ci_95[0])}–{format(metric.test_auroc_ci_95[1])}</p><div className="mt-4 grid grid-cols-2 gap-2 border-t hairline pt-3 text-xs"><span>Macro-F1 <strong className="mono block text-sm">{format(metric.test_macro_f1)}</strong></span><span>Cal. cutoff <strong className="mono block text-sm">{format(metric.threshold, 4)}</strong></span></div>{metric.test_pairwise_accuracy !== undefined && <p className="mt-3 text-xs text-[#69716d]">Pair accuracy <strong className="mono text-[#18211d]">{format(metric.test_pairwise_accuracy)}</strong></p>}</article>)}</div>
      </div>
      <p className="mt-4 border-l-4 border-[#df4c2f] bg-[#fbe9e2] p-4 text-sm leading-6 text-[#6f2d20]">These columns are not interchangeable datasets: our 100-question run scores the model’s own generated answers; HaluEval scores supplied right versus hallucinated candidates under knowledge. HaluEval is strongly length-confounded, so compare patterns—not just its larger numbers.</p>
      <MetricGuide model={model} selected={selectedMetric} onSelect={setSelectedMetric} />
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        <Insight icon={<Gauge size={20} />} label="Entropy threshold" value={format(bench.scores.token_entropy_top3.threshold, 4)} text="Chosen only on the 50 calibration questions by maximum macro-F1." />
        <Insight icon={<WarningCircle size={20} />} label="Entropy vs surprise" value={`${bench.primitive_comparison.test_auroc_difference >= 0 ? '+' : ''}${format(bench.primitive_comparison.test_auroc_difference)}`} text={`Paired 95% interval ${format(bench.primitive_comparison.test_auroc_difference_ci_95[0])} to ${format(bench.primitive_comparison.test_auroc_difference_ci_95[1])}; the small difference is uncertain.`} />
        <Insight icon={<BookOpenText size={20} />} label="Operational label" value={`${bench.test_hallucinations}/${bench.test_examples}`} text="Wrong means no normalized accepted answer appeared. This is transparent, but imperfect." />
      </div>
      <p className="mt-5 border-l-4 border-[#df4c2f] bg-[#fbe9e2] p-4 text-sm leading-6 text-[#6f2d20]">This is an exploratory iteration. Top-3 entropy led the new primitive signals on Instruct calibration, but the held-out set has now been inspected repeatedly and the Base model did not improve. A fresh benchmark is required before treating 0.825 as confirmed.</p>
    </section>
  );
}

function MetricGuide({ model, selected, onSelect }: { model: ModelKind; selected: FeatureKey; onSelect: (key: FeatureKey) => void }) {
  const prediction = predictions[model].find((item) => item.id === 'q002') ?? predictions[model][0];
  const definition = metricDefinitions[selected];
  const example = workedExample(selected, prediction);
  return (
    <div className="card mt-5 overflow-hidden">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b hairline bg-[#eeeae1] p-4">
        <div><p className="eyebrow">One question, every metric</p><p className="mt-2 text-sm text-[#505955]">Use the same {prediction.id} token trace to compare raw inputs, substitutions, and outputs.</p></div>
        <label className="text-xs font-semibold text-[#69716d]">Featured metric<select value={selected} onChange={(event) => onSelect(event.target.value as FeatureKey)} className="focus-ring ml-2 border hairline bg-white px-3 py-2 text-[#18211d]">{metricKeys.map((key) => <option key={key} value={key}>{metricShort[key]}</option>)}</select></label>
      </div>
      <div className="grid lg:grid-cols-[.8fr_1.2fr]">
        <div className="border-b hairline p-5 lg:border-b-0 lg:border-r md:p-7">
          <div className="flex flex-wrap items-center gap-2"><p className="eyebrow">Exact definition</p><span className="bg-[#eeeae1] px-2 py-1 text-[10px] uppercase text-[#69716d]">{definition.group}</span></div>
          <h3 className="mt-3 text-2xl font-semibold">{benchmarks[model].scores[selected].display_name}</h3>
          <p className="mono mt-5 overflow-x-auto bg-[#18211d] p-4 text-sm leading-7 text-white">{definition.formula}</p>
          <p className="mt-5 leading-7 text-[#505955]">{definition.plain}</p>
          <p className="mt-4 text-sm text-[#69716d]"><strong className="text-[#18211d]">Inputs:</strong> {definition.input}</p>
        </div>
        <div className="p-5 md:p-7">
          <p className="eyebrow">Worked example · {prediction.id}</p>
          <p className="mt-2 font-semibold">{prediction.question}</p>
          <div className="mt-5 grid gap-3">
            <div className="bg-[#f4f1ea] p-4"><p className="eyebrow">Actual inputs</p><p className="mono mt-2 break-words text-sm leading-6">{example.inputs}</p></div>
            <div className="bg-[#f4f1ea] p-4"><p className="eyebrow">Substitution</p><p className="mono mt-2 break-words text-sm leading-6">{example.arithmetic}</p></div>
          </div>
          <div className="mt-4 flex items-end justify-between border-t hairline pt-4"><span className="text-sm text-[#69716d]">Saved answer-level score</span><strong className="mono text-2xl">{selected === 'answer_tokens' ? example.score : format(example.score, 6)}</strong></div>
        </div>
      </div>
      <div className="border-t hairline bg-[#f4f1ea] p-4 md:p-6">
        <div className="mb-4"><p className="eyebrow">All {metricKeys.length} calculations for {prediction.id}</p><p className="mt-2 text-sm text-[#69716d]">Expand any row. Every displayed score is reconstructed from the saved token trace below it.</p></div>
        <div className="columns-1 gap-2 lg:columns-2">
          {metricKeys.map((key) => {
            const item = workedExample(key, prediction);
            const itemDefinition = metricDefinitions[key];
            return <details key={key} className="mb-2 break-inside-avoid border hairline bg-white" open={key === selected} onToggle={(event) => { if (event.currentTarget.open) onSelect(key); }}><summary className="focus-ring flex cursor-pointer items-center justify-between gap-3 p-4"><span><span className="font-semibold">{benchmarks[model].scores[key].display_name}</span><span className="ml-2 text-[10px] uppercase text-[#69716d]">{itemDefinition.group}</span></span><strong className="mono">{key === 'answer_tokens' ? item.score : format(item.score, 5)}</strong></summary><div className="border-t hairline p-4 text-sm"><p className="mono leading-6 text-[#9e321e]">{itemDefinition.formula}</p><p className="mt-3 text-[#505955]">{itemDefinition.plain}</p><div className="mt-3 bg-[#f4f1ea] p-3"><p className="eyebrow">Inputs</p><p className="mono mt-2 break-words text-xs leading-6">{item.inputs}</p></div><div className="mt-2 bg-[#f4f1ea] p-3"><p className="eyebrow">Substitution</p><p className="mono mt-2 break-words text-xs leading-6">{item.arithmetic}</p></div></div></details>;
          })}
        </div>
      </div>
    </div>
  );
}

function Insight({ icon, label, value, text }: { icon: React.ReactNode; label: string; value: string; text: string }) {
  return <div className="card p-5"><div className="mb-5 flex items-center gap-2 text-[#df4c2f]">{icon}<span className="eyebrow">{label}</span></div><p className="metric-number text-3xl font-semibold">{value}</p><p className="mt-3 text-sm leading-6 text-[#69716d]">{text}</p></div>;
}

function TreeDiagram({ node, edge }: { node: TreeRule; edge?: string }) {
  const isLeaf = !node.feature || node.threshold === undefined;
  return (
    <div className="min-w-0 flex-1">
      {edge && <p className="mono mb-1 text-[10px] font-semibold uppercase tracking-[.08em] text-[#9e321e]">{edge}</p>}
      <div className={`border hairline p-3 ${isLeaf ? 'bg-[#eeeae1]' : 'bg-white'}`}>
        {isLeaf
          ? <><p className="eyebrow">Leaf risk</p><p className="mono mt-1 text-xl font-semibold">{pct(node.hallucination_probability)}</p></>
          : <><p className="eyebrow">Split</p><p className="mono mt-1 break-words text-xs font-semibold">{metricShort[node.feature!]} ≤ {format(node.threshold!, 4)}</p></>}
        <p className="mt-2 text-[10px] text-[#69716d]">{node.positives}/{node.count} calibration candidates labelled hallucinated</p>
      </div>
      {!isLeaf && node.left && node.right && (
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          <TreeDiagram node={node.left} edge="yes · go left" />
          <TreeDiagram node={node.right} edge="no · go right" />
        </div>
      )}
    </div>
  );
}

function CandidateAudit({ prediction, selectedMetric, onMetric }: { prediction: HaluEvalPrediction; selectedMetric: FeatureKey; onMetric: (key: FeatureKey) => void }) {
  const example = workedExample(selectedMetric, prediction as unknown as Prediction);
  const kindIsWrong = prediction.candidate_kind === 'hallucinated';
  return (
    <article className={`border-2 p-4 ${kindIsWrong ? 'border-[#df4c2f] bg-[#fff8f4]' : 'border-[#4d8a6e] bg-[#f7fbf8]'}`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className={`px-2 py-1 text-xs font-bold uppercase ${kindIsWrong ? 'bg-[#fbe9e2] text-[#9e321e]' : 'bg-[#e5f1eb] text-[#236349]'}`}>{kindIsWrong ? 'HaluEval hallucinated' : 'HaluEval right'}</span>
        <span className="mono text-xs">tree risk {pct(prediction.tree_score)}</span>
      </div>
      <p className="mt-4 leading-7">{prediction.candidate_answer}</p>
      <div className="mt-4 grid grid-cols-2 gap-px bg-[#d7d3c9]">
        {(['skip_first_top3_surprise', 'surprise_shift_top3', 'token_ambiguity_top3', 'answer_tokens'] as FeatureKey[]).map((key) => (
          <button key={key} onClick={() => onMetric(key)} className={`focus-ring bg-white p-3 text-left ${selectedMetric === key ? 'ring-2 ring-inset ring-[#df4c2f]' : ''}`}>
            <p className="text-[10px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{key === 'answer_tokens' ? prediction.features[key] : format(prediction.features[key], 4)}</p>
          </button>
        ))}
      </div>
      <div className="mt-3 bg-white p-3">
        <p className="eyebrow">{metricShort[selectedMetric]} · exact substitution</p>
        <p className="mono mt-2 break-words text-[11px] leading-5 text-[#9e321e]">{metricDefinitions[selectedMetric].formula}</p>
        <p className="mono mt-2 break-words text-[11px] leading-5">{example.inputs}</p>
        <p className="mono mt-2 break-words text-[11px] font-semibold leading-5">{example.arithmetic}</p>
      </div>
      <details className="mt-3 border hairline bg-white p-3">
        <summary className="focus-ring cursor-pointer text-xs font-semibold">All 24 scores and all raw token inputs</summary>
        <div className="mt-3 grid grid-cols-2 gap-1 sm:grid-cols-3">
          {metricKeys.map((key) => <button onClick={() => onMetric(key)} key={key} className="focus-ring bg-[#f4f1ea] p-2 text-left"><span className="block text-[9px] text-[#69716d]">{metricShort[key]}</span><strong className="mono text-[11px]">{key === 'answer_tokens' ? prediction.features[key] : format(prediction.features[key], 5)}</strong></button>)}
        </div>
        <div className="mt-3 overflow-x-auto border hairline">
          <table className="w-full min-w-[760px] text-left text-[10px]">
            <thead className="bg-[#eeeae1] text-[#69716d]"><tr><th className="p-2">t</th><th className="p-2">token</th><th className="p-2">confidence</th><th className="p-2">margin</th><th className="p-2">entropy</th><th className="p-2">hidden shift</th><th className="p-2">cosine change</th><th className="p-2">hidden RMS</th></tr></thead>
            <tbody>{prediction.token_pieces.map((token, index) => <tr className="border-t hairline" key={`${prediction.id}-raw-${index}`}><td className="mono p-2">{index + 1}</td><td className="mono p-2">{JSON.stringify(token)}</td><td className="mono p-2">{format(prediction.confidences[index], 5)}</td><td className="mono p-2">{format(prediction.token_margins[index], 5)}</td><td className="mono p-2">{format(prediction.token_entropies[index], 5)}</td><td className="mono p-2">{prediction.hidden_shifts[index] === null ? '—' : format(prediction.hidden_shifts[index]!, 5)}</td><td className="mono p-2">{prediction.hidden_cosine_distances[index] === null ? '—' : format(prediction.hidden_cosine_distances[index]!, 5)}</td><td className="mono p-2">{format(prediction.hidden_norms[index], 5)}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="mt-3 bg-[#f4f1ea] p-3"><p className="eyebrow">Tree path</p>{prediction.tree_path.map((step, index) => <p className="mono mt-1 text-[10px]" key={`${prediction.id}-path-${index}`}>{index + 1}. {step}</p>)}</div>
      </details>
    </article>
  );
}

function ExternalBenchmark() {
  const pairs = useMemo(() => {
    const grouped = new Map<string, HaluEvalPrediction[]>();
    haluevalPredictions.forEach((prediction) => grouped.set(prediction.pair_id, [...(grouped.get(prediction.pair_id) ?? []), prediction]));
    return [...grouped.entries()].map(([id, candidates]) => ({ id, candidates, split: candidates[0].split, sourceIndex: candidates[0].source_index }));
  }, []);
  const defaultPair = pairs.find((pair) => pair.split === 'test') ?? pairs[0];
  const [selectedPairId, setSelectedPairId] = useState(defaultPair.id);
  const [selectedMetric, setSelectedMetric] = useState<FeatureKey>(haluevalMetrics.selected_scalar.score_key);
  const pair = pairs.find((item) => item.id === selectedPairId) ?? defaultPair;
  const exemplar = pair.candidates[0];
  const results = [
    { name: 'Cal.-selected scalar', metric: haluevalMetrics.selected_scalar, note: metricShort[haluevalMetrics.selected_scalar.score_key] },
    { name: 'One-split stump', metric: haluevalMetrics.stump, note: 'one learned condition' },
    { name: 'Depth-2 tree', metric: haluevalMetrics.depth2_tree, note: 'at most two conditions' },
    { name: 'Length baseline', metric: haluevalMetrics.scores.answer_tokens, note: 'control, not content' },
  ];
  const residual = haluevalMetrics.length_diagnostics.selected_scalar_residual;
  const rankedScores = [...metricKeys].sort((left, right) => haluevalMetrics.scores[right].test_auroc - haluevalMetrics.scores[left].test_auroc);
  return (
    <section id="external-benchmark" className="border-y hairline bg-[#eae6dc] py-18">
      <div className="shell">
        <div className="grid gap-7 lg:grid-cols-[1.15fr_.85fr]">
          <div>
            <p className="eyebrow">External reality check</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-[-.04em]">50 real HaluEval-QA pairs, frozen first.</h2>
            <p className="mt-4 max-w-3xl leading-7 text-[#69716d]">Each pair supplies knowledge, a question, one right answer, and one intentionally hallucinated answer. We score the exact candidate tokens under the same Qwen model: 25 whole pairs calibrate thresholds and tree branches; the other 25 pairs are touched once for evaluation.</p>
          </div>
          <div className="card grid grid-cols-2 gap-px overflow-hidden bg-[#d7d3c9]">
            <OverviewStep number="1" label="Freeze" value="features + split" note="Seed and model committed before rows were fetched." />
            <OverviewStep number="2" label="Calibrate" value="25 answer pairs" note="Select scalar and every threshold using calibration only." />
            <OverviewStep number="3" label="Test" value="25 unseen pairs" note="Rank wrong above right; also report F1 at the frozen cutoff." />
            <OverviewStep number="4" label="Audit" value="100 traces" note="Every token input, score, tree path, and candidate is visible below." accent />
          </div>
        </div>

        <div className="mt-7 overflow-x-auto card">
          <table className="w-full min-w-[820px] text-left text-sm">
            <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Method</th><th className="p-4">Test AUROC</th><th className="p-4">95% CI</th><th className="p-4">Macro-F1</th><th className="p-4">Pair accuracy</th><th className="p-4">What was learned on calibration</th></tr></thead>
            <tbody>{results.map(({ name, metric, note }) => <tr className="border-t hairline" key={name}><td className="p-4 font-semibold">{name}</td><td className="mono p-4 text-lg font-semibold">{format(metric.test_auroc)}</td><td className="mono p-4 text-[#69716d]">{format(metric.test_auroc_ci_95[0])}–{format(metric.test_auroc_ci_95[1])}</td><td className="mono p-4">{format(metric.test_macro_f1)}</td><td className="mono p-4">{format(metric.test_pairwise_accuracy ?? Number.NaN)}</td><td className="p-4 text-xs text-[#69716d]">{note}{name === 'Cal.-selected scalar' ? `; cutoff ${format(metric.threshold, 4)}` : ''}</td></tr>)}</tbody>
          </table>
        </div>

        <div className="mt-5 border-l-4 border-[#df4c2f] bg-[#fbe9e2] p-5 text-sm leading-6 text-[#6f2d20]">
          <p className="font-bold">The 0.994 is real arithmetic, but not clean evidence of factual understanding.</p>
          <p className="mt-1">The hallucinated candidate is longer in {haluevalMetrics.length_diagnostics.test_pairs_hallucinated_longer}/25 held-out pairs, so even answer length reaches AUROC {format(haluevalMetrics.scores.answer_tokens.test_auroc)}. After removing a calibration-fitted log-length trend from the selected score, AUROC falls to {format(residual.test_auroc)} ({format(residual.test_auroc_ci_95[0])}–{format(residual.test_auroc_ci_95[1])}). That residual check is post-hoc, so it is a warning—not a new headline result.</p>
        </div>

        <div className="mt-8 grid gap-5 lg:grid-cols-2">
          <div className="card p-5 md:p-7">
            <p className="eyebrow">The learned depth-2 tree</p>
            <h3 className="mt-2 text-xl font-semibold">Exact branches, no black box.</h3>
            <p className="mt-3 text-sm leading-6 text-[#69716d]">The six allowed inputs were fixed in advance. Each leaf outputs its calibration-set hallucination fraction; that fraction is the tree score.</p>
            <div className="mt-5"><TreeDiagram node={haluevalMetrics.depth2_tree.rules} /></div>
          </div>
          <div className="card p-5 md:p-7">
            <p className="eyebrow">Does combining features help?</p>
            <h3 className="mt-2 text-xl font-semibold">Not consistently in these small runs.</h3>
            <div className="mt-5 space-y-4">{(['instruct', 'base'] as ModelKind[]).map((model) => {
              const run = shallowTrees.models[model];
              const delta = run.depth2_tree.test_auroc - run.selected_scalar.test_auroc;
              return <div className="border hairline p-4" key={model}><div className="flex items-center justify-between gap-3"><strong className="capitalize">Qwen {model}</strong><span className={`mono text-sm font-semibold ${delta > 0 ? 'text-[#236349]' : 'text-[#9e321e]'}`}>{delta >= 0 ? '+' : ''}{format(delta)} tree − scalar</span></div><p className="mono mt-3 text-sm">scalar {format(run.selected_scalar.test_auroc)} → depth-2 tree {format(run.depth2_tree.test_auroc)}</p><p className="mt-2 text-xs text-[#69716d]">Tree 95% CI {format(run.depth2_tree.test_auroc_ci_95[0])}–{format(run.depth2_tree.test_auroc_ci_95[1])}</p></div>;
            })}</div>
            <p className="mt-4 text-sm leading-6 text-[#69716d]">So the tree stays an experiment. With only 50 calibration examples, its extra branches can fit quirks that do not repeat.</p>
          </div>
        </div>

        <div className="mt-8">
          <p className="eyebrow">Next cheap, interpretable signals</p>
          <h3 className="mt-2 text-2xl font-semibold tracking-[-.03em]">Useful additions before any larger classifier.</h3>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-[#69716d]">These are a predeclared queue, not reported wins. Each is O(answer tokens), uses signals already produced by the forward pass, and can be printed token by token.</p>
          <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {[
              ['Low-confidence fraction', 'meanₜ 𝟙[cₜ < τ]', 'How much of the answer, rather than only its worst token, was shaky. Fit τ on calibration.'],
              ['Longest surprise burst', 'max run length where −ln(cₜ) > τ', 'Distinguishes one uncertain token from a sustained uncertain phrase.'],
              ['Surprise slope', 'OLS slope of −ln(cₜ) against token position t', 'Asks whether confidence deteriorates as the answer continues.'],
              ['Hidden acceleration', 'top-3 mean of |rₜ − rₜ₋₁|', 'Measures abrupt changes in state movement, using the shifts already captured.'],
              ['Confidence–shift correlation', 'corr(−ln(cₜ), rₜ)', 'Tests whether uncertainty and representation movement rise together.'],
              ['Greedy disagreement', 'meanₜ 𝟙[candidate token ≠ argmaxᵥ pₜᵥ]', 'For supplied-candidate benchmarks only: how often the model preferred another token.'],
            ].map(([name, formula, explanation]) => <div className="card p-4" key={name}><p className="font-semibold">{name}</p><p className="mono mt-3 bg-[#eeeae1] p-2 text-xs">{formula}</p><p className="mt-3 text-xs leading-5 text-[#69716d]">{explanation}</p></div>)}
          </div>
          <p className="mt-3 text-xs text-[#69716d]">A monotonic relabeling such as exp(entropy) may be easier to explain, but it cannot change AUROC; it is not a genuinely new ranking signal.</p>
        </div>

        <div className="card mt-8 overflow-hidden">
          <div className="flex flex-wrap items-end justify-between gap-4 border-b hairline bg-[#eeeae1] p-4 md:p-6">
            <div><p className="eyebrow">Paired candidate microscope</p><h3 className="mt-2 text-xl font-semibold">Same knowledge and question; compare every computed input.</h3></div>
            <div className="flex flex-wrap gap-2">
              <label className="text-xs font-semibold text-[#69716d]">Pair<select className="focus-ring ml-2 border hairline bg-white px-3 py-2 text-[#18211d]" value={selectedPairId} onChange={(event) => setSelectedPairId(event.target.value)}>{pairs.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.split}</option>)}</select></label>
              <label className="text-xs font-semibold text-[#69716d]">Metric<select className="focus-ring ml-2 border hairline bg-white px-3 py-2 text-[#18211d]" value={selectedMetric} onChange={(event) => setSelectedMetric(event.target.value as FeatureKey)}>{metricKeys.map((key) => <option key={key} value={key}>{metricShort[key]}</option>)}</select></label>
            </div>
          </div>
          <div className="p-4 md:p-6">
            <div className="grid gap-4 lg:grid-cols-[.75fr_1.25fr]"><div><p className="eyebrow">Question · source row {exemplar.source_index}</p><p className="mt-2 text-lg font-semibold">{exemplar.question}</p></div><div><p className="eyebrow">Supplied knowledge</p><p className="mt-2 text-sm leading-6 text-[#505955]">{exemplar.knowledge}</p></div></div>
            <div className="mt-6 grid gap-4 xl:grid-cols-2">{[...pair.candidates].sort((left, right) => left.is_hallucination - right.is_hallucination).map((candidate) => <CandidateAudit key={candidate.id} prediction={candidate} selectedMetric={selectedMetric} onMetric={setSelectedMetric} />)}</div>
          </div>
        </div>

        <details className="card mt-5 p-4 md:p-6">
          <summary className="focus-ring cursor-pointer font-semibold">Open the complete 24-score HaluEval leaderboard</summary>
          <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[680px] text-left text-sm"><thead className="text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-3">Rank</th><th className="p-3">Score</th><th className="p-3">Calibration AUROC</th><th className="p-3">Test AUROC</th><th className="p-3">Pair accuracy</th></tr></thead><tbody>{rankedScores.map((key, index) => { const metric = haluevalMetrics.scores[key]; return <tr key={key} className="border-t hairline"><td className="mono p-3">#{index + 1}</td><td className="p-3 font-semibold">{metricShort[key]}{key === haluevalMetrics.selected_scalar.score_key && <span className="ml-2 bg-[#e5f1eb] px-2 py-1 text-[9px] uppercase text-[#236349]">selected on calibration</span>}</td><td className="mono p-3">{format(metric.calibration_auroc)}</td><td className="mono p-3 font-semibold">{format(metric.test_auroc)}</td><td className="mono p-3">{format(metric.test_pairwise_accuracy ?? Number.NaN)}</td></tr>; })}</tbody></table></div>
        </details>
        <p className="mt-4 text-xs leading-5 text-[#69716d]">Source: HaluEval QA. This slice tests candidate discrimination under supplied knowledge, not whether the model’s own free-running answer is factual. The source answers are synthetic adversarial examples, and this 50-pair slice is deliberately small.</p>
      </div>
    </section>
  );
}

function QuestionLab({ model }: { model: ModelKind }) {
  const data = predictions[model];
  const [selectedId, setSelectedId] = useState('q002');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<'all' | 'wrong' | 'correct'>('all');
  const filtered = useMemo(() => data.filter((item) => {
    const matchesText = `${item.id} ${item.question} ${item.generated_answer}`.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === 'all' || (filter === 'wrong' ? !item.correct : item.correct);
    return matchesText && matchesFilter;
  }), [data, filter, query]);
  const selected = data.find((item) => item.id === selectedId) ?? filtered[0] ?? data[0];

  useEffect(() => {
    if (!data.some((item) => item.id === selectedId)) setSelectedId(data[0].id);
  }, [data, selectedId]);

  return (
    <section id="question-lab" className="border-y hairline bg-[#eae6dc] py-18">
      <div className="shell">
        <div className="mb-8">
          <p className="eyebrow">Question lab</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-[-.04em]">Audit every arithmetic step.</h2>
          <p className="mt-3 max-w-2xl text-[#69716d]">Choose any saved generation. Inspect selected-token confidence, top-two margin, full-distribution entropy, and 896-dimensional hidden-state signals—then reconstruct every published score.</p>
        </div>
        <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
          <aside className="card overflow-hidden">
            <div className="border-b hairline p-4">
              <label className="flex items-center gap-2 border hairline bg-white px-3 py-2.5">
                <MagnifyingGlass size={17} className="text-[#69716d]" />
                <input className="min-w-0 flex-1 bg-transparent text-sm outline-none" placeholder="Search 100 questions" value={query} onChange={(event) => setQuery(event.target.value)} />
              </label>
              <div className="mt-3 flex gap-2">
                {(['all', 'wrong', 'correct'] as const).map((value) => <button key={value} onClick={() => setFilter(value)} className={`focus-ring px-3 py-1.5 text-xs font-semibold capitalize ${filter === value ? 'bg-[#18211d] text-white' : 'border hairline bg-white'}`}>{value}</button>)}
              </div>
            </div>
            <div className="max-h-[680px] overflow-y-auto">
              {filtered.length === 0 ? <div className="p-8 text-center text-sm text-[#69716d]">No questions match this filter.</div> : filtered.map((item) => (
                <button key={item.id} onClick={() => setSelectedId(item.id)} className={`focus-ring block w-full border-b hairline p-4 text-left transition ${selected.id === item.id ? 'bg-[#fff8f4]' : 'hover:bg-[#f7f4ed]'}`}>
                  <div className="mb-2 flex items-center justify-between gap-2"><span className="mono text-xs text-[#69716d]">{item.id} · {item.split}</span><span className={`h-2.5 w-2.5 rounded-full ${item.correct ? 'bg-[#4d8a6e]' : 'bg-[#df4c2f]'}`} aria-label={item.correct ? 'correct' : 'wrong'} /></div>
                  <span className="line-clamp-2 text-sm font-medium leading-5">{item.question}</span>
                </button>
              ))}
            </div>
          </aside>
          <QuestionDetail prediction={selected} />
        </div>
      </div>
    </section>
  );
}

function QuestionDetail({ prediction }: { prediction: Prediction }) {
  const calculations = tokenCalculations(prediction);
  const [active, setActive] = useState<number | null>(calculations.findIndex((row) => row.crcv !== null));
  useEffect(() => setActive(calculations.findIndex((row) => row.crcv !== null)), [prediction.id]);
  const row = active === null ? null : calculations[active];
  const windowScores = calculations.flatMap((item) => item.crcv === null ? [] : [item.crcv]);
  return (
    <article className="card min-w-0 p-5 md:p-7">
      <div className="flex flex-wrap items-start justify-between gap-5 border-b hairline pb-6">
        <div className="max-w-3xl">
          <div className="mb-3 flex flex-wrap gap-2 text-xs"><span className="mono bg-[#eeeae1] px-2 py-1">{prediction.id}</span><span className="bg-[#eeeae1] px-2 py-1 capitalize">{prediction.category}</span><span className="bg-[#eeeae1] px-2 py-1 capitalize">{prediction.difficulty}</span></div>
          <h3 className="text-2xl font-semibold tracking-[-.025em]">{prediction.question}</h3>
        </div>
        <span className={`inline-flex items-center gap-2 px-3 py-2 text-sm font-bold ${prediction.correct ? 'bg-[#e5f1eb] text-[#236349]' : 'bg-[#fbe9e2] text-[#9e321e]'}`}>
          {prediction.correct ? <CheckCircle size={18} weight="fill" /> : <WarningCircle size={18} weight="fill" />}{prediction.correct ? 'accepted' : 'labelled wrong'}
        </span>
      </div>
      <div className="grid gap-4 border-b hairline py-6 md:grid-cols-2">
        <div><p className="eyebrow mb-2">Model answered</p><p className="leading-7">{prediction.generated_answer}</p></div>
        <div><p className="eyebrow mb-2">Accepted answer aliases</p><p className="leading-7">{prediction.answers.join(' · ')}</p></div>
      </div>
      <ModelJourney prediction={prediction} />
      <div className="grid grid-cols-2 gap-px bg-[#d7d3c9] sm:grid-cols-4">
        {(['token_entropy_top3', 'top3_token_surprise', 'token_ambiguity_top3', 'crcv_mean'] as FeatureKey[]).map((key) => <div className="bg-[#fffdf8] px-3 py-4" key={key}><p className="text-[11px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{format(prediction.features[key], 4)}</p></div>)}
      </div>
      <details className="border-x border-b hairline bg-[#f7f4ed] p-4"><summary className="focus-ring cursor-pointer text-sm font-semibold">Show all {metricKeys.length} scores for this answer</summary><div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">{metricKeys.map((key) => <div className="bg-white px-3 py-4" key={key}><p className="text-[11px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{key === 'answer_tokens' ? prediction.features[key] : format(prediction.features[key], 4)}</p></div>)}</div></details>
      <QuestionMetricAudit prediction={prediction} />

      <div className="mt-8 flex flex-wrap items-end justify-between gap-4">
        <div><p className="eyebrow">Calculation tape</p><h4 className="mt-1 text-xl font-semibold">c<sub>t</sub> × r<sub>t</sub>, then rolling sample deviation</h4></div>
        <p className="mono text-xs text-[#69716d]">window W = 5 · first state has no shift</p>
      </div>
      <div className="mt-4 overflow-x-auto border hairline">
        <table className="w-full min-w-[760px] border-collapse text-left text-sm">
          <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="px-3 py-3">t</th><th className="px-3 py-3">token</th><th className="px-3 py-3">confidence c<sub>t</sub></th><th className="px-3 py-3">norm. shift r<sub>t</sub></th><th className="px-3 py-3">coupling s<sub>t</sub></th><th className="px-3 py-3">CRCV<sub>t</sub></th></tr></thead>
          <tbody>{calculations.map((item, index) => <tr key={`${prediction.id}-${item.traceIndex}`} onClick={() => setActive(index)} className={`cursor-pointer border-t hairline ${active === index ? 'bg-[#fff1ec]' : 'hover:bg-[#f7f4ed]'}`}><td className="mono px-3 py-3">{item.tokenIndex}</td><td className="mono px-3 py-3 font-semibold">{JSON.stringify(item.token)}</td><td className="mono px-3 py-3">{format(item.confidence, 5)}</td><td className="mono px-3 py-3">{format(item.shift, 5)}</td><td className="mono px-3 py-3">{format(item.coupling, 5)}</td><td className="mono px-3 py-3 font-semibold text-[#9e321e]">{item.crcv === null ? 'waiting…' : format(item.crcv, 5)}</td></tr>)}</tbody>
        </table>
      </div>
      {row?.window ? <FormulaScratch row={row} /> : <div className="mt-3 border hairline bg-[#f7f4ed] p-4 text-sm text-[#69716d]">Select a row with a CRCV value. A complete trailing window needs {Math.min(5, calculations.length)} valid confidence–shift pairs.</div>}
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <div className="border hairline p-4"><p className="eyebrow">Answer-level CRCV mean</p><p className="mono mt-3 break-words text-sm">({windowScores.map((value) => format(value, 4)).join(' + ')}) / {windowScores.length}</p><p className="mono mt-3 text-xl font-semibold">= {format(prediction.features.crcv_mean, 6)}</p></div>
        <div className="border hairline p-4"><p className="eyebrow">Mean token surprise</p><p className="mono mt-3 text-sm">mean(−ln(c<sub>t</sub>)) across all {prediction.confidences.length} tokens</p><p className="mono mt-3 text-xl font-semibold">= {format(prediction.features.mean_nll, 6)}</p></div>
      </div>
    </article>
  );
}

function QuestionMetricAudit({ prediction }: { prediction: Prediction }) {
  const [selected, setSelected] = useState<FeatureKey>('top3_token_surprise');
  const definition = metricDefinitions[selected];
  const example = workedExample(selected, prediction);
  return (
    <details className="mt-4 border hairline bg-[#f7f4ed]">
      <summary className="focus-ring cursor-pointer px-4 py-3 text-sm font-semibold">Show how any score is calculated for this question</summary>
      <div className="border-t hairline p-4">
        <label className="text-xs font-semibold text-[#69716d]">Metric<select value={selected} onChange={(event) => setSelected(event.target.value as FeatureKey)} className="focus-ring ml-2 border hairline bg-white px-3 py-2 text-[#18211d]">{metricKeys.map((key) => <option key={key} value={key}>{metricShort[key]}</option>)}</select></label>
        <div className="mt-4 grid gap-3 lg:grid-cols-3">
          <div className="bg-white p-4"><p className="eyebrow">Formula</p><p className="mono mt-2 break-words text-sm leading-6">{definition.formula}</p></div>
          <div className="bg-white p-4"><p className="eyebrow">This answer’s inputs</p><p className="mono mt-2 break-words text-xs leading-6">{example.inputs}</p></div>
          <div className="bg-white p-4"><p className="eyebrow">Substitute and calculate</p><p className="mono mt-2 break-words text-sm leading-6">{example.arithmetic}</p><p className="mono mt-3 text-xl font-semibold">= {selected === 'answer_tokens' ? example.score : format(example.score, 6)}</p></div>
        </div>
      </div>
    </details>
  );
}

function ModelJourney({ prediction }: { prediction: Prediction }) {
  const calculations = tokenCalculations(prediction);
  const calculationByTrace = new Map(calculations.map((row) => [row.traceIndex, row]));
  const initialTrace = Math.min(5, Math.max(0, prediction.token_ids.length - 1));
  const [traceIndex, setTraceIndex] = useState(initialTrace);
  const [stage, setStage] = useState(0);
  useEffect(() => { setTraceIndex(Math.min(5, Math.max(0, prediction.token_ids.length - 1))); setStage(0); }, [prediction.id]);

  const promptTokens = questionTokens[prediction.id] ?? [];
  const selectedPiece = prediction.token_pieces[traceIndex];
  const confidence = prediction.confidences[traceIndex];
  const shift = prediction.hidden_shifts[traceIndex];
  const calculation = calculationByTrace.get(traceIndex);
  const coupling = shift === null ? null : confidence * shift;
  const stages = [
    { label: 'Question', value: 'text', detail: 'The factual question enters the prompt as Unicode text.' },
    { label: 'Tokenizer', value: `${promptTokens.length} question IDs`, detail: 'Qwen’s exact subword tokenizer maps text pieces to integer vocabulary IDs. This isolates the question span; the fixed prompt template contributes additional tokens.' },
    { label: 'Embedding', value: `${promptTokens.length} × 896 shown`, detail: 'Each token ID selects a learned 896-number vector. The display shows the question span; the full prompt also embeds its fixed instruction or answer-prefix tokens.' },
    { label: 'Layers 0–23', value: '24 blocks', detail: 'Each block updates the residual stream using causal self-attention, an MLP, and residual additions.' },
    { label: 'Final RMSNorm', value: 'hₜ ∈ ℝ⁸⁹⁶', detail: 'This is the exact final normalized hidden state exposed by the patched ONNX graph.' },
    { label: 'LM head', value: '151,936 logits', detail: 'The hidden state is projected against the vocabulary embedding matrix to produce one score per possible token.' },
    { label: 'Softmax', value: `cₜ = ${format(confidence, 4)}`, detail: `Softmax turns logits into probabilities; greedy decoding selects ${JSON.stringify(selectedPiece)}.` },
  ];

  return (
    <div id="model-journey" className="my-7 border-y hairline py-7">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div><p className="eyebrow">Inside one generation step</p><h4 className="mt-1 text-xl font-semibold">Question → network → metric</h4></div>
        <p className="mono text-xs text-[#69716d]">click a stage · choose a generated token</p>
      </div>

      <div className="mt-5 flex flex-wrap gap-1.5" aria-label="Generated-token step selector">
        {prediction.token_pieces.map((piece, index) => (
          <button
            key={`${prediction.id}-piece-${index}`}
            className={`focus-ring mono border px-2.5 py-1.5 text-xs ${traceIndex === index ? 'border-[#df4c2f] bg-[#df4c2f] text-white' : 'hairline bg-white hover:border-[#9d9990]'}`}
            onClick={() => setTraceIndex(index)}
            aria-pressed={traceIndex === index}
            aria-label={`Generation step ${index + 1}, token ${piece}`}
          >
            <span className="mr-1 opacity-65">{index + 1}</span>{JSON.stringify(piece)}
          </button>
        ))}
      </div>

      <div className="journey-flow mt-6" role="group" aria-label="Language model processing stages">
        {stages.map((item, index) => (
          <div className="journey-stage" key={item.label}>
            <button onClick={() => setStage(index)} aria-pressed={stage === index} className={`focus-ring journey-node ${stage === index ? 'is-active' : ''}`}>
              <span className="eyebrow">{String(index + 1).padStart(2, '0')}</span>
              <strong>{item.label}</strong>
              <span className="mono">{item.value}</span>
            </button>
            {index < stages.length - 1 && <ArrowRight className="journey-arrow" size={18} aria-hidden="true" />}
          </div>
        ))}
      </div>

      <div className="mt-4 bg-[#f7f4ed] p-4">
        <p className="text-sm leading-6"><strong>{stages[stage].label}.</strong> {stages[stage].detail}</p>
        {stage === 0 && <p className="mt-3 text-lg font-medium">“{prediction.question}”</p>}
        {stage === 1 && <div className="mt-3 flex flex-wrap gap-1.5">{promptTokens.map((token, index) => <span className="mono border hairline bg-white px-2 py-1 text-xs" key={`${token.id}-${index}`}><span className="text-[#9e321e]">{token.id}</span> {JSON.stringify(token.piece)}</span>)}</div>}
        {stage === 2 && <DimensionStrip count={Math.min(promptTokens.length, 12)} label="896 learned values per input token" />}
        {stage === 3 && <div className="mt-3 grid grid-cols-8 gap-1 sm:grid-cols-12">{Array.from({ length: 24 }, (_, index) => <span className="mono grid h-8 place-items-center bg-[#18211d] text-xs text-white" key={index}>{index}</span>)}</div>}
        {stage === 4 && <DimensionStrip count={16} label="896 final-state values used for the selected token" />}
        {stage === 5 && <div className="mt-3"><div className="flex h-8 overflow-hidden" aria-label={`Selected token probability ${format(confidence, 4)}, all other tokens ${format(1 - confidence, 4)}`}><span className="grid min-w-[5rem] place-items-center bg-[#df4c2f] px-2 text-xs font-semibold text-white" style={{ width: `${Math.max(12, confidence * 100)}%` }}>selected</span><span className="grid flex-1 place-items-center bg-[#c8c4ba] px-2 text-xs">151,935 alternatives</span></div><p className="mono mt-2 text-xs text-[#69716d]">The LM head produces raw logits; the next stage normalizes them.</p></div>}
        {stage === 6 && <p className="mono mt-3">P({JSON.stringify(selectedPiece)}) = exp(z<sub>selected</sub>) / Σ<sub>vocab</sub> exp(z) = <strong>{format(confidence, 6)}</strong></p>}
      </div>

      <div className="metric-flow mt-5">
        <FlowValue label="State movement" symbol="rₜ" value={shift === null ? 'needs hₜ₋₁' : format(shift, 5)} note="normalized L2 distance" />
        <span className="mono text-[#69716d]">×</span>
        <FlowValue label="Token confidence" symbol="cₜ" value={format(confidence, 5)} note="selected softmax probability" />
        <span className="mono text-[#69716d]">=</span>
        <FlowValue label="Coupling" symbol="sₜ" value={coupling === null ? 'waiting' : format(coupling, 5)} note="the signal entering CRCV" accent />
        <ArrowRight size={20} className="text-[#69716d]" />
        <FlowValue label="Trailing window" symbol="SD(s)" value={calculation?.crcv === null || calculation?.crcv === undefined ? 'waiting' : format(calculation.crcv, 5)} note={calculation?.window ? `[${calculation.window.map((value) => format(value, 3)).join(', ')}]` : 'up to 5 coupling values'} accent />
      </div>
      <p className="mt-4 text-xs leading-5 text-[#69716d]">The selected token is appended to the sequence and the network runs again. That repetition gives hₜ₊₁, making the next state movement—and eventually each five-step CRCV window—observable.</p>
    </div>
  );
}

function DimensionStrip({ count, label }: { count: number; label: string }) {
  return <div className="mt-3"><div className="flex gap-[3px]">{Array.from({ length: count }, (_, index) => <span key={index} className="h-7 min-w-[3px] flex-1 bg-[#c8c4ba]" />)}</div><p className="mono mt-2 text-xs text-[#69716d]">{label}</p></div>;
}

function FlowValue({ label, symbol, value, note, accent = false }: { label: string; symbol: string; value: string; note: string; accent?: boolean }) {
  return <div className={`min-w-0 flex-1 p-3 ${accent ? 'bg-[#fbe9e2]' : 'bg-[#eeeae1]'}`}><p className="text-[10px] uppercase tracking-[.07em] text-[#69716d]">{label}</p><div className="mt-1 flex items-baseline justify-between gap-2"><span className="mono text-xs">{symbol}</span><strong className="mono text-sm">{value}</strong></div><p className="mt-2 text-[10px] leading-4 text-[#69716d]">{note}</p></div>;
}

function FormulaScratch({ row }: { row: ReturnType<typeof tokenCalculations>[number] }) {
  return <div className="mt-3 bg-[#18211d] p-5 text-white"><p className="eyebrow !text-[#bfc7c2]">Selected window ending at token {row.tokenIndex}</p><div className="mt-4 grid gap-4 md:grid-cols-3"><div><p className="text-xs text-[#bfc7c2]">1 · window values</p><p className="mono mt-2 text-sm leading-6">[{row.window!.map((value) => format(value, 5)).join(', ')}]</p></div><div><p className="text-xs text-[#bfc7c2]">2 · sample variance</p><p className="mono mt-2 text-sm leading-6">{row.window!.length < 2 ? <>one value → 0 by convention</> : <>Σ(s − {format(row.windowMean!, 5)})² / {row.window!.length - 1}<br />= {format(row.sampleVariance!, 7)}</>}</p></div><div><p className="text-xs text-[#bfc7c2]">3 · square root</p><p className="mono mt-2 text-2xl font-semibold">CRCV<sub>t</sub> = {format(row.crcv!, 6)}</p></div></div></div>;
}

function LiveLab({ model }: { model: ModelKind }) {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState('Idle — no model has been downloaded.');
  const [progress, setProgress] = useState(0);
  const [tokens, setTokens] = useState<LiveToken[]>([]);
  const [result, setResult] = useState<{ answer: string; features: Features; elapsedMs: number; qualityWarning: string | null } | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [question, setQuestion] = useState('What is the capital of France?');
  const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const [runtimeChoice, setRuntimeChoice] = useState<'auto' | 'webgpu' | 'wasm'>('auto');
  const runtime = runtimeChoice === 'auto' ? (webgpu ? 'webgpu' : 'wasm') : runtimeChoice;

  useEffect(() => () => workerRef.current?.terminate(), []);

  const run = () => {
    workerRef.current?.terminate();
    const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    setBusy(true); setError(''); setTokens([]); setResult(null); setProgress(0);
    worker.onmessage = (event: MessageEvent<WorkerToMain>) => {
      const message = event.data;
      if (message.type === 'status') setStatus(`${message.phase} — ${message.detail}`);
      if (message.type === 'progress') { setProgress(message.progress); setStatus(`Downloading ${message.file}`); }
      if (message.type === 'token') setTokens((current) => [...current, message.token]);
      if (message.type === 'result') {
        setResult(message);
        setBusy(false);
        setStatus(message.qualityWarning
          ? 'Complete — generation failed the repetition check.'
          : 'Complete — signals came from the exact generation pass.');
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      }
      if (message.type === 'error') {
        setError(`${message.message}${runtime === 'webgpu' ? ' Try WASM if this GPU cannot complete the q4 graph.' : ''}`);
        setBusy(false);
        setStatus('Failed — no score was reported');
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      }
    };
    worker.postMessage({ type: 'run', model, question, runtime, maxNewTokens: 24 });
  };

  return (
    <section id="run-live" className="shell py-18">
      <div className="grid gap-8 lg:grid-cols-[.75fr_1.25fr]">
        <div>
          <p className="eyebrow">Local inference</p>
          <h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">No server. Real hidden states.</h2>
          <p className="mt-5 leading-7 text-[#69716d]">The page fetches the public quantized ONNX model, exposes the final 896-value state already inside its graph, and runs greedy generation in a worker. Confidence and hidden-state signals now come from that exact generation pass—there is no second replay. Your question and trace never leave the browser.</p>
          <div className="mt-6 border-l-4 border-[#df4c2f] bg-[#fbe9e2] p-4 text-sm leading-6 text-[#6f2d20]">
            First run is large: roughly 750 MB. WebGPU and WASM deliberately use the same q4 graph so runtime comparisons do not silently switch model precision. The patched model is cached when browser quota permits. {webgpu ? 'This browser reports WebGPU support.' : 'WebGPU is unavailable, so use WASM.'}
          </div>
          <ol className="mt-7 space-y-4 text-sm">
            <Step number="01" text="Generate greedily with all sampling and repetition penalties disabled." />
            <Step number="02" text="Capture confidence and final hidden state from the same forward call that selects each token." />
            <Step number="03" text="Verify the selected IDs, then compute every score from that single trace." />
          </ol>
        </div>
        <div className="card p-5 md:p-7">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b hairline pb-5"><div><p className="eyebrow">Browser console</p><p className="mt-1 font-semibold">Qwen2.5-0.5B {model === 'instruct' ? 'Instruct' : 'Base'} · {runtime.toUpperCase()}</p></div><span className="mono bg-[#eeeae1] px-3 py-2 text-xs">single-pass verified trace</span></div>
          <fieldset className="mt-5"><legend className="eyebrow">Runtime</legend><div className="mt-2 grid grid-cols-3 gap-2">{(['auto', 'webgpu', 'wasm'] as const).map((choice) => <button type="button" key={choice} onClick={() => setRuntimeChoice(choice)} disabled={busy || (choice === 'webgpu' && !webgpu)} className={`focus-ring border px-3 py-2 text-xs font-semibold uppercase ${runtimeChoice === choice ? 'border-[#18211d] bg-[#18211d] text-white' : 'hairline bg-white disabled:cursor-not-allowed disabled:opacity-40'}`}>{choice}</button>)}</div><p className="mt-2 text-xs text-[#69716d]">Auto currently resolves to {webgpu ? 'WebGPU' : 'WASM'}. Force WASM to compare the fallback path.</p></fieldset>
          <label className="mt-6 block"><span className="eyebrow">Question</span><textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} className="focus-ring mt-2 w-full resize-y border hairline bg-white p-3 leading-6" /></label>
          <button onClick={run} disabled={busy || !question.trim()} className="focus-ring mt-4 inline-flex w-full items-center justify-center gap-2 bg-[#df4c2f] px-5 py-3 font-bold text-white"><Play size={18} weight="fill" />{busy ? 'Working locally…' : 'Load model and calculate'}</button>
          <div className="mt-5 border hairline bg-[#f7f4ed] p-4" aria-live="polite"><div className="flex justify-between gap-3 text-xs"><span className="truncate">{status}</span><span className="mono">{progress ? pct(progress / 100) : ''}</span></div>{busy && <div className="mt-3 h-1.5 bg-[#ddd8ce]"><div className="h-full bg-[#df4c2f] transition-all" style={{ width: `${progress}%` }} /></div>}</div>
          {error && <div className="mt-4 flex gap-3 bg-[#fbe9e2] p-4 text-sm text-[#8d2e1d]"><WarningCircle className="shrink-0" size={20} /><span>{error}</span></div>}
          {!busy && !result && !error && <div className="mt-5 py-8 text-center text-sm text-[#69716d]">A live token tape will appear here after you start.</div>}
          {tokens.length > 0 && <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[900px] text-left text-xs"><thead className="text-[#69716d]"><tr><th className="pb-2">token</th><th className="pb-2">confidence</th><th className="pb-2">top-2 margin</th><th className="pb-2">entropy</th><th className="pb-2">L2 shift</th><th className="pb-2">cosine change</th><th className="pb-2">hidden RMS</th><th className="pb-2">coupling</th></tr></thead><tbody>{tokens.map((token, index) => <tr key={`${token.tokenId}-${index}`} className="border-t hairline"><td className="mono py-2 pr-4">{JSON.stringify(token.token)}</td><td className="mono py-2 pr-4">{format(token.confidence, 5)}</td><td className="mono py-2 pr-4">{format(token.margin, 5)}</td><td className="mono py-2 pr-4">{format(token.entropy, 5)}</td><td className="mono py-2 pr-4">{token.hiddenShift === null ? '—' : format(token.hiddenShift, 5)}</td><td className="mono py-2 pr-4">{token.hiddenCosineDistance === null ? '—' : format(token.hiddenCosineDistance, 5)}</td><td className="mono py-2 pr-4">{format(token.hiddenNorm, 5)}</td><td className="mono py-2">{token.hiddenShift === null ? '—' : format(token.confidence * token.hiddenShift, 5)}</td></tr>)}</tbody></table></div>}
          {result && <div className="mt-5 border-t hairline pt-5">
            <div className={`p-4 text-sm leading-6 ${result.qualityWarning ? 'bg-[#fbe9e2] text-[#8d2e1d]' : 'bg-[#e5efe8] text-[#245537]'}`}>
              <p className="font-semibold">{result.qualityWarning ? 'Generation integrity: warning' : 'Generation integrity: passed'}</p>
              <p>{result.qualityWarning ?? 'Every displayed signal came from the forward pass that selected its token; no repeated-trigram degeneration was detected.'}</p>
              {result.qualityWarning && <p className="mt-1 font-semibold">Do not interpret the scores below as factuality evidence for this answer.</p>}
            </div>
            <p className="eyebrow mt-5">Generated answer</p><p className="mt-2 leading-7">{result.answer || 'No text was generated.'}</p>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">{(['token_entropy_top3', 'top3_token_surprise', 'token_ambiguity_top3', 'crcv_mean'] as FeatureKey[]).map((key) => <div key={key} className="bg-[#eeeae1] p-3"><p className="text-[10px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{format(result.features[key], 4)}</p></div>)}</div>
            <details className="mt-4 border hairline bg-white p-4"><summary className="focus-ring cursor-pointer text-sm font-semibold">Show all {metricKeys.length} live scores and formulas</summary><div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">{metricKeys.map((key) => <div key={key} className="bg-[#eeeae1] p-3"><p className="text-[10px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{key === 'answer_tokens' ? result.features[key] : format(result.features[key], 4)}</p><p className="mono mt-2 text-[9px] leading-4 text-[#69716d]">{metricDefinitions[key].formula}</p></div>)}</div></details>
            <p className="mono mt-3 text-xs text-[#69716d]">{format(result.elapsedMs / 1000, 2)} s after model load · single-pass generation and signal capture</p>
          </div>}
        </div>
      </div>
    </section>
  );
}

function Step({ number, text }: { number: string; text: string }) { return <li className="flex items-start gap-4"><span className="mono text-[#df4c2f]">{number}</span><span>{text}</span></li>; }

function MethodNote() {
  return <section className="border-t hairline bg-[#eae6dc] py-12"><div className="shell grid gap-8 md:grid-cols-2 lg:grid-cols-4"><div><p className="eyebrow">Token uncertainty</p><p className="mono mt-3 text-sm leading-7">c<sub>t</sub> = p(top token)<br />margin = p₁ − p₂<br />entropy = −Σp ln p / ln|V|</p></div><div><p className="eyebrow">State movement</p><p className="mono mt-3 text-sm leading-7">r<sub>t</sub> = ‖h<sub>t</sub> − h<sub>t−1</sub>‖₂ / (‖h<sub>t−1</sub>‖₂ + 10⁻⁸)</p></div><div><p className="eyebrow">Coupling</p><p className="mono mt-3 text-sm leading-7">s<sub>t</sub> = c<sub>t</sub> · r<sub>t</sub></p><p className="mt-1 text-sm text-[#69716d]">Confidence multiplied by state movement.</p></div><div><p className="eyebrow">CRCV</p><p className="mt-3 text-sm leading-7">Sample standard deviation of s over each complete trailing window, W = 5.</p></div></div></section>;
}

function EarlierExperiments({ model }: { model: ModelKind }) {
  return <details className="border-y hairline bg-[#eae6dc]">
    <summary className="focus-ring shell cursor-pointer list-none py-8"><p className="eyebrow">Earlier exploratory work</p><div className="mt-2 flex flex-wrap items-center justify-between gap-4"><h2 className="text-2xl font-semibold tracking-[-.035em]">Open the original 24-score search, HaluEval stress test, and token journey.</h2><span className="mono text-sm text-[#9e321e]">show / hide</span></div><p className="mt-3 max-w-3xl text-sm leading-6 text-[#69716d]">These reused smaller splits motivated the frozen fresh comparison above. They remain fully inspectable, but they are no longer the first evidence a new visitor sees.</p></summary>
    <div className="border-t hairline bg-[#f4f1ea]"><Results model={model} /><ExternalBenchmark /><QuestionLab model={model} /><MethodNote /></div>
  </details>;
}

export default function App() {
  const [model, setModel] = useState<ModelKind>('instruct');
  const [view, setView] = useState<AppView>(() => {
    if (window.location.hash === '#live' || window.location.hash === '#run-live') return 'live';
    if (['#explore', '#results', '#question-lab', '#model-journey'].includes(window.location.hash)) return 'explore';
    return 'overview';
  });
  const changeView = (next: AppView) => {
    setView(next);
    window.history.replaceState(null, '', next === 'overview' ? window.location.pathname : `#${next}`);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  return <>
    <a href="#main-content" className="skip-link">Skip to content</a>
    <Header model={model} onModel={setModel} view={view} onView={changeView} />
    <main id="main-content">
      {view === 'overview' && <><Hero onExplore={() => changeView('explore')} onRun={() => changeView('live')} /><FreshSummary metrics={freshMetrics} replication={crossModelReplication} onExplore={() => changeView('explore')} /></>}
      {view === 'explore' && <><FreshBenchmark metrics={freshMetrics} replication={crossModelReplication} /><EarlierExperiments model={model} /></>}
      {view === 'live' && <><LiveLab model={model} /><MethodNote /></>}
    </main>
    <footer className="bg-[#18211d] py-7 text-[#aeb7b2]"><div className="shell flex flex-wrap justify-between gap-3 text-xs"><span>Tiny CRCV Lab · inspectable research prototype</span><span>600 questions × two model families + Base stress test · calibration separated from test</span></div></footer>
  </>;
}
