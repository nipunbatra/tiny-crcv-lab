import { CheckCircle, MagnifyingGlass, WarningCircle } from '@phosphor-icons/react';
import { useEffect, useMemo, useState } from 'react';
import { parseJsonl } from './lib/metrics';
import type {
  CrossModelReplication,
  FeatureKey,
  FreshComparisonMetrics,
  FreshModelKind,
  FreshMethodKey,
  FreshPrediction,
} from './types';

const overviewOrder: FreshMethodKey[] = [
  'top3_token_surprise',
  'crcv_mean',
  'surprise_spread',
  'token_entropy_top3',
  'p_true',
  'lexical_disagreement_3',
  'trace_logistic_8',
];

const datasetLabels = {
  nq_open: 'NQ-Open',
  trivia_qa: 'TriviaQA',
  truthful_qa: 'TruthfulQA',
} as const;

const freshModelConfig: Record<FreshModelKind, {
  label: string;
  shortLabel: string;
  tracePath: string;
  hiddenDimensions: number;
}> = {
  qwen_instruct: {
    label: 'Qwen2.5-0.5B Instruct',
    shortLabel: 'Qwen Instruct',
    tracePath: './data/fresh_qa_qwen05b_instruct.jsonl',
    hiddenDimensions: 896,
  },
  smollm2_instruct: {
    label: 'SmolLM2-360M Instruct',
    shortLabel: 'SmolLM2 Instruct',
    tracePath: './data/fresh_qa_smollm2_360m_instruct.jsonl',
    hiddenDimensions: 960,
  },
  qwen_base: {
    label: 'Qwen2.5-0.5B Base',
    shortLabel: 'Qwen Base',
    tracePath: './data/fresh_qa_qwen05b_base.jsonl',
    hiddenDimensions: 896,
  },
};

const traceCache: Partial<Record<FreshModelKind, FreshPrediction[]>> = {};
const format = (value: number | null | undefined, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const signed = (value: number, digits = 3) => `${value >= 0 ? '+' : ''}${format(value, digits)}`;

function methodRole(key: FreshMethodKey, metrics: FreshComparisonMetrics) {
  const special: Partial<Record<FreshMethodKey, string>> = {
    top3_token_surprise: 'Original cheap recommendation',
    crcv_mean: 'Original confidence × state-motion hypothesis',
    p_true: 'One extra self-evaluation pass',
    lexical_disagreement_3: 'Three generations; no judge',
    discrete_semantic_entropy_3: 'Three-sample budget proxy',
    semantic_disagreement_3: 'Three generations + three judgments',
    hidden_logistic_probe: 'Calibration-trained hidden-state probe',
    trace_logistic_8: 'Calibration-trained 8→1 trace model',
    trace_tree_depth2: 'Two readable calibration branches',
    answer_tokens: 'Length confound control',
  };
  return special[key] ?? metrics.methods[key].family;
}

function methodCost(key: FreshMethodKey, metrics: Record<FreshModelKind, FreshComparisonMetrics>) {
  const meanSeconds = (metrics.qwen_instruct.methods[key].incremental_runtime.mean_seconds_per_question + metrics.smollm2_instruct.methods[key].incremental_runtime.mean_seconds_per_question) / 2;
  if (key === 'p_true') return `${format(meanSeconds, 3)} s · one extra pass`;
  if (key === 'lexical_disagreement_3') return `${format(meanSeconds, 3)} s · three samples`;
  if (key === 'semantic_disagreement_3' || key === 'discrete_semantic_entropy_3') return `${format(meanSeconds, 3)} s · samples + judges`;
  if (key === 'hidden_logistic_probe') return '≈0.1 ms · saved hidden state';
  if (key === 'trace_logistic_8' || key === 'trace_tree_depth2') return '<0.01 ms · saved trace';
  return 'free from greedy trace';
}

export function FreshSummary({ metrics, replication, onExplore }: { metrics: Record<FreshModelKind, FreshComparisonMetrics>; replication: CrossModelReplication; onExplore: () => void }) {
  const qwen = metrics.qwen_instruct;
  const smol = metrics.smollm2_instruct;
  return <section className="border-b hairline bg-[#eae6dc] py-16">
    <div className="shell">
      <div className="grid gap-5 lg:grid-cols-[1fr_.72fr] lg:items-end">
        <div>
          <p className="eyebrow">Cross-model validation · same 600 questions</p>
          <h2 className="mt-2 max-w-4xl text-4xl font-semibold tracking-[-.05em]">The cheap signal survives a second model family.</h2>
        </div>
        <p className="text-sm leading-6 text-[#69716d]">We froze the full 31-score audit before generating any SmolLM2 answer. The questions were reused, so this validates model transfer—not a second untouched dataset.</p>
      </div>
      <div className="mt-7 grid gap-4 lg:grid-cols-3">
        <div className="bg-[#18211d] p-5 text-white"><p className="eyebrow !text-[#bfc7c2]">Frozen transfer rule</p><p className="mono mt-3 text-3xl font-semibold">{replication.replicated_method_count} / {replication.method_count}</p><p className="mt-2 text-sm leading-6 text-[#bfc7c2]">scores rank in the right direction on both models and at least two of three slices per model.</p></div>
        <div className="bg-[#fffdf8] p-5"><p className="eyebrow">Best shared free score</p><p className="mt-3 text-xl font-semibold">Token-surprise spread</p><p className="mt-2 text-sm leading-6 text-[#69716d]">Qwen <span className="mono">{format(qwen.methods.surprise_spread.held_out.auroc)}</span> · SmolLM2 <span className="mono">{format(smol.methods.surprise_spread.held_out.auroc)}</span>.</p></div>
        <div className="bg-[#fffdf8] p-5"><p className="eyebrow">What did not transfer</p><p className="mt-3 text-xl font-semibold">Pure hidden cosine motion</p><p className="mt-2 text-sm leading-6 text-[#69716d]">Mean cosine change falls from <span className="mono">{format(qwen.methods.hidden_cosine_mean.held_out.auroc)}</span> to <span className="mono">{format(smol.methods.hidden_cosine_mean.held_out.auroc)}</span>.</p></div>
      </div>
      <div className="card mt-5 overflow-x-auto">
        <table className="w-full min-w-[840px] text-left text-sm">
          <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Representative baseline</th><th className="p-4">Qwen Instruct</th><th className="p-4">SmolLM2 Instruct</th><th className="p-4">Online cost</th></tr></thead>
          <tbody>{overviewOrder.map((key) => <tr className="border-t hairline" key={key}>
            <td className="p-4"><p className="font-semibold">{qwen.methods[key].display_name}</p><p className="mt-1 text-xs text-[#69716d]">{methodRole(key, qwen)}</p></td>
            <td className="mono p-4 text-lg">{format(qwen.methods[key].held_out.auroc)}</td>
            <td className="mono p-4 text-lg">{format(smol.methods[key].held_out.auroc)}</td>
            <td className="p-4 text-xs leading-5 text-[#69716d]">{methodCost(key, metrics)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="mt-5 grid gap-4 lg:grid-cols-[1.25fr_.75fr]">
        <div className="border-l-4 border-[#df4c2f] bg-[#fff8f4] p-5 text-sm leading-6"><strong>Recommendation.</strong> Compute surprise spread alongside the frozen top-3 baseline. It is stronger descriptively on both models, but its paired SmolLM2 improvement interval still includes zero; treat it as the lead cheap candidate, not a proven winner.</div>
        <button onClick={onExplore} className="focus-ring flex items-center justify-between bg-[#18211d] p-5 text-left font-semibold text-white"><span>Open 31-method audit + arithmetic</span><span aria-hidden="true">→</span></button>
      </div>
    </div>
  </section>;
}

function MethodTable({ metrics, selected, onSelect }: { metrics: FreshComparisonMetrics; selected: FreshMethodKey; onSelect: (method: FreshMethodKey) => void }) {
  return <div className="card overflow-x-auto">
    <table className="w-full min-w-[1080px] text-left text-sm">
      <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Headline method</th><th className="p-4">All 300 test</th><th className="p-4">NQ</th><th className="p-4">Trivia</th><th className="p-4">Truthful</th><th className="p-4">AURAC</th><th className="p-4">Δ vs top-3</th><th className="p-4">Formula</th></tr></thead>
      <tbody>{metrics.headline_method_order.map((key) => {
        const method = metrics.methods[key];
        const paired = method.paired_vs_top3_token_surprise;
        return <tr key={key} className={`border-t hairline ${selected === key ? 'bg-[#fff1ec]' : ''}`}>
          <td className="p-4"><p className="font-semibold">{method.display_name}</p><p className="mt-1 text-xs text-[#69716d]">{methodRole(key, metrics)}</p></td>
          <td className="mono p-4"><strong>{format(method.held_out.auroc)}</strong><br /><span className="text-[10px] text-[#69716d]">{format(method.held_out.auroc_ci_95[0])}–{format(method.held_out.auroc_ci_95[1])}</span></td>
          <td className="mono p-4">{format(method.by_dataset.nq_open.auroc)}</td>
          <td className="mono p-4">{format(method.by_dataset.trivia_qa.auroc)}</td>
          <td className="mono p-4">{format(method.by_dataset.truthful_qa.auroc)}</td>
          <td className="mono p-4">{format(method.held_out.aurac)}</td>
          <td className="mono p-4">{paired ? <>{signed(paired.auroc_difference)}<br /><span className={`text-[10px] ${paired.reliably_improves ? 'text-[#236349]' : 'text-[#69716d]'}`}>{format(paired.auroc_difference_ci_95[0])} to {format(paired.auroc_difference_ci_95[1])}</span></> : 'baseline'}</td>
          <td className="p-4"><button onClick={() => onSelect(key)} className="focus-ring border hairline bg-white px-3 py-2 text-xs font-semibold">{selected === key ? 'Selected' : 'Inspect'}</button></td>
        </tr>;
      })}</tbody>
    </table>
  </div>;
}

function FullLeaderboard({ metrics, selected, onSelect }: { metrics: FreshComparisonMetrics; selected: FreshMethodKey; onSelect: (method: FreshMethodKey) => void }) {
  const ranked = [...metrics.all_method_order].sort((left, right) => (metrics.methods[right].held_out.auroc ?? -1) - (metrics.methods[left].held_out.auroc ?? -1));
  return <details className="card mt-4">
    <summary className="focus-ring cursor-pointer list-none p-5 font-semibold">All 31 methods, ranked <span className="ml-2 text-xs font-normal text-[#69716d]">24 trace scalars + 7 extra baselines</span></summary>
    <div className="overflow-x-auto border-t hairline"><table className="w-full min-w-[880px] text-left text-xs"><thead className="bg-[#eeeae1] text-[#69716d]"><tr><th className="p-3">Rank</th><th className="p-3">Method</th><th className="p-3">Family</th><th className="p-3">AUROC</th><th className="p-3">95% CI</th><th className="p-3">Macro-F1</th><th className="p-3">Inspect</th></tr></thead><tbody>{ranked.map((key, index) => {
      const method = metrics.methods[key];
      return <tr className={`border-t hairline ${selected === key ? 'bg-[#fff1ec]' : ''}`} key={key}><td className="mono p-3">{index + 1}</td><td className="p-3 font-semibold">{method.display_name}</td><td className="p-3 text-[#69716d]">{method.family}</td><td className="mono p-3">{format(method.held_out.auroc)}</td><td className="mono p-3">{format(method.held_out.auroc_ci_95[0])}–{format(method.held_out.auroc_ci_95[1])}</td><td className="mono p-3">{format(method.held_out.macro_f1)}</td><td className="p-3"><button onClick={() => onSelect(key)} className="focus-ring border hairline bg-white px-2.5 py-1.5 font-semibold">Open</button></td></tr>;
    })}</tbody></table></div>
  </details>;
}

function SelectedMethod({ metrics, methodKey }: { metrics: FreshComparisonMetrics; methodKey: FreshMethodKey }) {
  const method = metrics.methods[methodKey];
  const paired = method.paired_vs_top3_token_surprise;
  return <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_.8fr]">
    <div className="bg-[#18211d] p-5 text-white"><p className="eyebrow !text-[#bfc7c2]">Exact formula · {method.family}</p><p className="mono mt-3 text-sm leading-7">{method.formula}</p><p className="mt-4 text-xs leading-5 text-[#bfc7c2]">{method.implementation_note} Higher predicts a wrong answer. Cutoff {format(method.threshold, 5)} maximized macro-F1 on 300 calibration questions.</p></div>
    <div className="bg-[#fffdf8] p-5"><p className="eyebrow">Held-out operating point</p><div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-2"><div><p className="text-[10px] text-[#69716d]">AUROC</p><p className="mono mt-1 text-xl">{format(method.held_out.auroc)}</p></div><div><p className="text-[10px] text-[#69716d]">AURAC</p><p className="mono mt-1 text-xl">{format(method.held_out.aurac)}</p></div><div><p className="text-[10px] text-[#69716d]">Avg precision</p><p className="mono mt-1 text-xl">{format(method.held_out.average_precision)}</p></div><div><p className="text-[10px] text-[#69716d]">Macro-F1</p><p className="mono mt-1 text-xl">{format(method.held_out.macro_f1)}</p></div></div><p className={`mt-4 text-xs leading-5 ${paired?.reliably_improves ? 'text-[#236349]' : 'text-[#69716d]'}`}>{methodKey === 'top3_token_surprise' ? 'This is the frozen comparison baseline.' : paired?.reliably_improves ? 'Paired 95% interval is above zero.' : 'Paired 95% interval does not establish improvement over top-3 surprise.'}</p></div>
  </div>;
}

function AuditCard({ name, score, formula, open = false, children }: { name: string; score: number; formula: string; open?: boolean; children: React.ReactNode }) {
  return <details className="border-t hairline py-4" open={open}><summary className="focus-ring flex cursor-pointer list-none items-center justify-between gap-4"><span><strong>{name}</strong><span className="mono ml-3 text-xs text-[#69716d]">{formula}</span></span><span className="mono bg-[#eeeae1] px-3 py-2 font-semibold">{format(score, 5)}</span></summary><div className="mt-4 bg-[#f7f4ed] p-4 text-xs leading-6">{children}</div></details>;
}

function Top3Audit({ prediction }: { prediction: FreshPrediction }) {
  const selected = prediction.confidences.map((confidence, index) => ({ token: prediction.token_pieces[index], confidence, surprise: -Math.log(Math.max(confidence, 1e-12)) })).sort((a, b) => b.surprise - a.surprise).slice(0, 3);
  return <AuditCard name="Top-3 token surprise" score={prediction.method_scores.top3_token_surprise} formula="mean(top 3 of −ln confidence)" open><p>{selected.map((item) => `${JSON.stringify(item.token)}: −ln(${format(item.confidence, 5)}) = ${format(item.surprise, 5)}`).join(' · ')}</p><p className="mono mt-3">({selected.map((item) => format(item.surprise, 5)).join(' + ')}) / {selected.length} = <strong>{format(prediction.method_scores.top3_token_surprise, 5)}</strong></p></AuditCard>;
}

function sampleStd(values: number[]) {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
}

function CrcvAudit({ prediction }: { prediction: FreshPrediction }) {
  const pairs = prediction.confidences.flatMap((confidence, index) => prediction.hidden_shifts[index] === null ? [] : [{ index, token: prediction.token_pieces[index], confidence, shift: prediction.hidden_shifts[index]!, coupling: confidence * prediction.hidden_shifts[index]! }]);
  const width = Math.min(5, pairs.length);
  const windows = pairs.length < 2 ? [0] : Array.from({ length: pairs.length - width + 1 }, (_, index) => sampleStd(pairs.slice(index, index + width).map((pair) => pair.coupling)));
  const first = pairs.slice(0, width).map((pair) => pair.coupling);
  return <AuditCard name="CRCV mean" score={prediction.method_scores.crcv_mean} formula="mean windows of sample-SD(cₜ × rₜ)"><p>At each token with a previous state, multiply chosen-token confidence <span className="mono">cₜ</span> by normalized hidden-state shift <span className="mono">rₜ</span>. Then take sample SD in each complete trailing window (up to five valid pairs).</p><div className="mt-3 max-h-64 overflow-auto"><table className="w-full min-w-[680px] text-left"><thead className="text-[#69716d]"><tr><th>token</th><th>cₜ</th><th>rₜ</th><th>cₜ × rₜ</th></tr></thead><tbody>{pairs.map((pair) => <tr className="border-t hairline" key={pair.index}><td className="mono py-1.5">{pair.index + 1} · {JSON.stringify(pair.token)}</td><td className="mono">{format(pair.confidence, 5)}</td><td className="mono">{format(pair.shift, 5)}</td><td className="mono">{format(pair.coupling, 5)}</td></tr>)}</tbody></table></div>{pairs.length < 2 ? <p className="mono mt-3">Only {pairs.length} valid pair → sample SD is 0 by convention → <strong>CRCV = 0</strong>.</p> : <><p className="mono mt-3">first window = [{first.map((value) => format(value, 5)).join(', ')}] → sample-SD = {format(windows[0], 5)}</p><p className="mono mt-2">window CRCVs = [{windows.map((value) => format(value, 5)).join(', ')}]</p><p className="mono mt-2">mean = ({windows.map((value) => format(value, 5)).join(' + ')}) / {windows.length} = <strong>{format(prediction.method_scores.crcv_mean, 5)}</strong></p></>}</AuditCard>;
}

function PTrueAudit({ prediction }: { prediction: FreshPrediction }) {
  const judgment = prediction.p_true_judgment;
  return <AuditCard name="P(False self-check)" score={prediction.method_scores.p_true} formula="p(No) / (p(Yes) + p(No))"><p>The model rereads the question and greedy answer, then scores one-token <span className="mono">Yes</span> and <span className="mono">No</span> continuations.</p><p className="mono mt-3">{format(judgment.no_probability, 7)} / ({format(judgment.yes_probability, 7)} + {format(judgment.no_probability, 7)}) = <strong>{format(judgment.normalized_no_probability, 5)}</strong></p></AuditCard>;
}

function LexicalAudit({ prediction }: { prediction: FreshPrediction }) {
  const explanation = prediction.lexical_disagreement_explanation;
  return <AuditCard name="Three-answer lexical disagreement" score={explanation.score} formula="mean pairs of 1 − Jaccard(word sets)"><div className="grid gap-2 md:grid-cols-3">{prediction.stochastic_answers.map((sample) => <div className="bg-[#eeeae1] p-3" key={sample.sample_index}><p className="mono text-[10px] text-[#69716d]">sample {sample.sample_index + 1}</p><p className="mt-2 text-xs leading-5">{sample.answer || 'No text'}</p></div>)}</div><p className="mono mt-3">pair distances = [{explanation.pair_distances.map((pair) => `${pair.sample_a + 1}↔${pair.sample_b + 1}: ${format(pair.distance, 5)}`).join(', ')}]</p><p className="mono mt-2">mean = ({explanation.pair_distances.map((pair) => format(pair.distance, 5)).join(' + ')}) / 3 = <strong>{format(explanation.score, 5)}</strong></p></AuditCard>;
}

function SemanticEntropyAudit({ prediction }: { prediction: FreshPrediction }) {
  const explanation = prediction.semantic_entropy_explanation;
  return <AuditCard name="Three-sample discrete semantic entropy" score={explanation.score} formula="−Σ (n꜀/3) ln(n꜀/3)"><p>Connect answer pairs whose saved same-answer judge gives <span className="mono">P(not same) &lt; {format(explanation.threshold, 1)}</span>; connected components become meaning clusters.</p><p className="mono mt-3">clusters = {explanation.clusters.map((cluster) => `{${cluster.map((index) => index + 1).join(',')}}`).join(' · ')} · sizes = [{explanation.cluster_sizes.join(', ')}] · probabilities = [{explanation.probabilities.map((value) => format(value, 5)).join(', ')}]</p><p className="mono mt-2">entropy = {explanation.entropy_terms.map((term) => format(term, 5)).join(' + ')} = <strong>{format(explanation.score, 5)}</strong></p></AuditCard>;
}

function DisagreementAudit({ prediction }: { prediction: FreshPrediction }) {
  return <AuditCard name="Three-answer semantic disagreement" score={prediction.method_scores.semantic_disagreement_3} formula="mean of 3 pairwise P(not same)"><p className="mono">pair risks = [{prediction.pair_judgments.map((pair) => `${pair.sample_a + 1}↔${pair.sample_b + 1}: ${format(pair.normalized_no_probability, 5)}`).join(', ')}]</p><p className="mono mt-2">mean = ({prediction.pair_judgments.map((pair) => format(pair.normalized_no_probability, 5)).join(' + ')}) / 3 = <strong>{format(prediction.method_scores.semantic_disagreement_3, 5)}</strong></p></AuditCard>;
}

function TraceLogisticAudit({ prediction }: { prediction: FreshPrediction }) {
  const explanation = prediction.trace_logistic_explanation;
  return <AuditCard name="Eight-feature trace logistic" score={explanation.sigmoid_risk} formula="sigmoid(b + Σ wₖ(xₖ−μₖ)/σₖ)"><div className="overflow-x-auto"><table className="w-full min-w-[860px] text-left"><thead className="text-[#69716d]"><tr><th>feature</th><th>x</th><th>μ</th><th>σ</th><th>z</th><th>w</th><th>wz</th></tr></thead><tbody>{explanation.terms.map((term) => <tr className="border-t hairline" key={term.feature}><td className="py-2 font-semibold">{term.feature}</td><td className="mono">{format(term.raw_value, 4)}</td><td className="mono">{format(term.calibration_mean, 4)}</td><td className="mono">{format(term.calibration_scale, 4)}</td><td className="mono">{format(term.standardized_value, 4)}</td><td className="mono">{signed(term.weight, 4)}</td><td className="mono">{signed(term.contribution, 4)}</td></tr>)}</tbody></table></div><p className="mono mt-3">logit = bias {signed(explanation.bias, 5)} + contributions = {format(explanation.logit, 5)}</p><p className="mono mt-2">sigmoid({format(explanation.logit, 5)}) = <strong>{format(explanation.sigmoid_risk, 5)}</strong></p></AuditCard>;
}

function TreeAudit({ prediction }: { prediction: FreshPrediction }) {
  return <AuditCard name="Depth-2 trace tree" score={prediction.method_scores.trace_tree_depth2} formula="follow two branches → leaf wrong-rate"><ol className="space-y-2">{prediction.trace_tree_path.map((step, index) => <li className="mono" key={step}>{index + 1}. {step}</li>)}</ol><p className="mono mt-3">leaf risk = <strong>{format(prediction.method_scores.trace_tree_depth2, 5)}</strong></p></AuditCard>;
}

function ProbeAudit({ prediction, hiddenDimensions }: { prediction: FreshPrediction; hiddenDimensions: number }) {
  const probe = prediction.probe_explanation;
  return <AuditCard name="Mean-hidden linear probe" score={prediction.method_scores.hidden_logistic_probe} formula="sigmoid(b + Σ wⱼzⱼ)"><p>Average the final {hiddenDimensions}-value hidden state across answer-token decisions, standardize each dimension on calibration only, and take one weighted sum.</p><div className="mt-3 overflow-x-auto"><table className="w-full min-w-[680px] text-left"><thead className="text-[#69716d]"><tr><th>dimension</th><th>hⱼ</th><th>zⱼ</th><th>wⱼ</th><th>wⱼzⱼ</th></tr></thead><tbody>{probe.top_contributions.map((item) => <tr className="border-t hairline" key={item.dimension}><td className="mono py-2">{item.dimension}</td><td className="mono">{format(item.raw_hidden, 4)}</td><td className="mono">{format(item.standardized_hidden, 4)}</td><td className="mono">{signed(item.weight, 4)}</td><td className="mono">{signed(item.contribution, 4)}</td></tr>)}</tbody></table></div><p className="mono mt-3">logit = bias {signed(probe.bias, 5)} + shown {signed(probe.top_contributions.reduce((sum, item) => sum + item.contribution, 0), 5)} + other {hiddenDimensions - probe.top_contributions.length} dims {signed(probe.other_dimensions_contribution, 5)} = {format(probe.logit, 5)}</p><p className="mono mt-2">sigmoid = <strong>{format(probe.sigmoid_risk, 5)}</strong></p></AuditCard>;
}

function ScalarLedger({ prediction, metrics }: { prediction: FreshPrediction; metrics: FreshComparisonMetrics }) {
  return <details className="border-t hairline py-4"><summary className="focus-ring cursor-pointer list-none font-semibold">All 24 trace scalars for this answer <span className="ml-2 text-xs font-normal text-[#69716d]">value · formula · cutoff decision</span></summary><div className="mt-4 overflow-x-auto"><table className="w-full min-w-[900px] text-left text-xs"><thead className="bg-[#eeeae1] text-[#69716d]"><tr><th className="p-3">Scalar</th><th className="p-3">Value</th><th className="p-3">Exact aggregation</th><th className="p-3">Calibrated decision</th></tr></thead><tbody>{metrics.scalar_method_order.map((key) => { const score = prediction.method_scores[key]; const method = metrics.methods[key]; return <tr className="border-t hairline" key={key}><td className="p-3 font-semibold">{method.display_name}</td><td className="mono p-3">{format(score, 5)}</td><td className="mono p-3 text-[#69716d]">{method.formula}</td><td className={`p-3 font-semibold ${score >= method.threshold ? 'text-[#9e321e]' : 'text-[#236349]'}`}>{score >= method.threshold ? 'flagged' : 'not flagged'} · {format(method.threshold, 5)}</td></tr>; })}</tbody></table></div></details>;
}

function FreshQuestionInspector({ model, metrics }: { model: FreshModelKind; metrics: FreshComparisonMetrics }) {
  const [rows, setRows] = useState<FreshPrediction[]>(traceCache[model] ?? []);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [dataset, setDataset] = useState<'all' | FreshPrediction['source_dataset']>('all');
  const [selectedId, setSelectedId] = useState('fresh-trivia_qa-07851');

  useEffect(() => {
    let cancelled = false;
    setError('');
    if (traceCache[model]) { setRows(traceCache[model]!); return; }
    setRows([]);
    fetch(freshModelConfig[model].tracePath).then((response) => { if (!response.ok) throw new Error(`Trace download failed (${response.status})`); return response.text(); }).then((text) => { const loaded = parseJsonl<FreshPrediction>(text); traceCache[model] = loaded; if (!cancelled) setRows(loaded); }).catch((reason) => { if (!cancelled) setError(String(reason)); });
    return () => { cancelled = true; };
  }, [model]);

  const filtered = useMemo(() => rows.filter((row) => { const haystack = `${row.id} ${row.question} ${row.generated_answer}`.toLowerCase(); return (dataset === 'all' || row.source_dataset === dataset) && haystack.includes(query.toLowerCase()); }), [dataset, query, rows]);
  const selected = rows.find((row) => row.id === selectedId) ?? filtered[0] ?? rows[0];

  return <section className="mt-14"><div className="mb-6"><p className="eyebrow">Question-level arithmetic · {freshModelConfig[model].label}</p><h3 className="mt-2 text-3xl font-semibold tracking-[-.04em]">Follow every number for any saved answer.</h3><p className="mt-3 max-w-3xl text-sm leading-6 text-[#69716d]">Open the worked cards for token probability, CRCV, self-evaluation, sampled consistency, the eight-feature logistic, tree, and hidden probe. The ledger exposes all 24 trace scalars.</p></div><div className="grid gap-5 xl:grid-cols-[340px_1fr]">
    <aside className="card overflow-hidden"><div className="border-b hairline p-4"><label className="flex items-center gap-2 border hairline bg-white px-3 py-2.5"><MagnifyingGlass size={17} className="text-[#69716d]" /><input value={query} onChange={(event) => setQuery(event.target.value)} className="min-w-0 flex-1 bg-transparent text-sm outline-none" placeholder="Search 600 questions" /></label><select value={dataset} onChange={(event) => setDataset(event.target.value as typeof dataset)} className="focus-ring mt-3 w-full border hairline bg-white px-3 py-2 text-sm"><option value="all">All three datasets</option>{Object.entries(datasetLabels).map(([key, label]) => <option value={key} key={key}>{label}</option>)}</select></div><div className="max-h-[720px] overflow-y-auto">{rows.length === 0 && !error ? <div className="p-8 text-center text-sm text-[#69716d]">Loading saved traces…</div> : error ? <div className="p-6 text-sm text-[#9e321e]">{error}</div> : filtered.map((row) => <button key={row.id} onClick={() => setSelectedId(row.id)} className={`focus-ring block w-full border-b hairline p-4 text-left ${selected?.id === row.id ? 'bg-[#fff8f4]' : 'hover:bg-[#f7f4ed]'}`}><div className="flex items-center justify-between gap-2"><span className="mono text-[10px] text-[#69716d]">{datasetLabels[row.source_dataset]} · {row.split}</span><span className={`h-2.5 w-2.5 rounded-full ${row.correct ? 'bg-[#4d8a6e]' : 'bg-[#df4c2f]'}`} /></div><p className="mt-2 line-clamp-2 text-sm font-medium leading-5">{row.question}</p></button>)}</div></aside>
    {selected ? <article className="card min-w-0 p-5 md:p-7"><div className="flex flex-wrap items-start justify-between gap-4 border-b hairline pb-5"><div><p className="eyebrow">{datasetLabels[selected.source_dataset]} · {selected.split} · source row {selected.source_index}</p><h4 className="mt-2 text-2xl font-semibold tracking-[-.03em]">{selected.question}</h4></div><span className={`inline-flex items-center gap-2 px-3 py-2 text-sm font-semibold ${selected.correct ? 'bg-[#e5f1eb] text-[#236349]' : 'bg-[#fbe9e2] text-[#9e321e]'}`}>{selected.correct ? <CheckCircle weight="fill" /> : <WarningCircle weight="fill" />}{selected.correct ? 'alias matched' : 'no alias matched'}</span></div><div className="grid gap-4 border-b hairline py-5 md:grid-cols-2"><div><p className="eyebrow">Greedy answer</p><p className="mt-2 text-sm leading-6">{selected.generated_answer}</p></div><div><p className="eyebrow">Accepted aliases</p><p className="mt-2 text-sm leading-6">{selected.answers.join(' · ')}</p></div></div>
      <div className="overflow-x-auto border-b hairline py-4"><table className="w-full min-w-[860px] text-left text-xs"><thead className="text-[#69716d]"><tr><th className="pb-2">Headline score</th><th className="pb-2">Value</th><th className="pb-2">Calibration cutoff</th><th className="pb-2">Decision</th></tr></thead><tbody>{metrics.headline_method_order.map((key) => { const score = selected.method_scores[key]; const threshold = metrics.methods[key].threshold; return <tr className="border-t hairline" key={key}><td className="py-2 font-semibold">{metrics.methods[key].display_name}</td><td className="mono">{format(score, 5)}</td><td className="mono">{format(threshold, 5)}</td><td className={`font-semibold ${score >= threshold ? 'text-[#9e321e]' : 'text-[#236349]'}`}>{score >= threshold ? 'flagged' : 'not flagged'}</td></tr>; })}</tbody></table></div>
      <Top3Audit prediction={selected} /><CrcvAudit prediction={selected} /><PTrueAudit prediction={selected} /><LexicalAudit prediction={selected} /><SemanticEntropyAudit prediction={selected} /><DisagreementAudit prediction={selected} /><TraceLogisticAudit prediction={selected} /><TreeAudit prediction={selected} /><ProbeAudit prediction={selected} hiddenDimensions={freshModelConfig[model].hiddenDimensions} /><ScalarLedger prediction={selected} metrics={metrics} /><p className="mt-4 text-xs leading-5 text-[#69716d]">The visible label is deliberately mechanical: a normalized gold alias must occur as a whole phrase. Semantically correct paraphrases can be marked wrong; inspect them before treating this as ground truth.</p>
    </article> : <div className="card grid min-h-[360px] place-items-center text-sm text-[#69716d]">Select a question to inspect.</div>}</div></section>;
}

function LiteratureMap() {
  const rows = [
    ['P(True)', 'One self-evaluation pass', 'Evaluated as P(False)', 'Kadavath et al. (2022)', 'https://arxiv.org/abs/2207.05221'],
    ['SelfCheckGPT', 'Several sampled answers', 'Evaluated: 3-sample lexical proxy', 'Manakul et al. (2023)', 'https://aclanthology.org/2023.emnlp-main.557/'],
    ['Semantic entropy', 'Usually 10 samples + meaning clustering', 'Evaluated: explicit 3-sample proxy', 'Farquhar et al. (2024)', 'https://www.nature.com/articles/s41586-024-07421-0'],
    ['SAR', 'Token/sentence relevance passes', 'Not reconstructable from saved traces', 'Duan et al. (2024)', 'https://aclanthology.org/2024.acl-long.276/'],
    ['INSIDE / EigenScore', 'Hidden embeddings for multiple samples', 'Not saved for stochastic answers', 'Chen et al. (2024)', 'https://arxiv.org/abs/2402.03744'],
    ['Lookback Lens', 'Attention to a grounding passage', 'Task has no grounding passage', 'Chuang et al. (2024)', 'https://aclanthology.org/2024.emnlp-main.84/'],
  ];
  return <details className="card mt-5"><summary className="focus-ring cursor-pointer list-none p-5 font-semibold">Comparable literature map <span className="ml-2 text-xs font-normal text-[#69716d]">what we evaluated—and what would be fake to claim</span></summary><div className="overflow-x-auto border-t hairline"><table className="w-full min-w-[900px] text-left text-xs"><thead className="bg-[#eeeae1] text-[#69716d]"><tr><th className="p-3">Method family</th><th className="p-3">Extra input</th><th className="p-3">Status here</th><th className="p-3">Primary source</th></tr></thead><tbody>{rows.map(([name, input, status, source, href]) => <tr className="border-t hairline" key={name}><td className="p-3 font-semibold">{name}</td><td className="p-3">{input}</td><td className="p-3">{status}</td><td className="p-3"><a className="underline" href={href} target="_blank" rel="noreferrer">{source}</a></td></tr>)}</tbody></table></div><p className="p-4 text-xs leading-5 text-[#69716d]">Published headline numbers are not copied into this leaderboard: different models, prompts, labels, datasets, sample counts, and splits make them non-comparable. The deck explains those boundaries explicitly.</p></details>;
}

function CrossModelTable({ replication }: { replication: CrossModelReplication }) {
  const keys: FreshMethodKey[] = [
    'surprise_spread',
    'token_entropy_max',
    'trace_logistic_8',
    'lexical_disagreement_3',
    'worst_token_surprise',
    'top3_token_surprise',
    'crcv_mean',
    'hidden_cosine_mean',
  ];
  const rows = keys.map((key) => replication.rows.find((row) => row.key === key)!);
  return <section className="mt-7">
    <div className="grid gap-4 lg:grid-cols-[.7fr_1.3fr]">
      <div className="bg-[#18211d] p-5 text-white"><p className="eyebrow !text-[#bfc7c2]">Frozen replication rule</p><p className="mono mt-3 text-4xl font-semibold">{replication.replicated_method_count} / {replication.method_count}</p><p className="mt-3 text-sm leading-6 text-[#bfc7c2]">methods point above chance in both pooled tests and at least two of three slices for each model.</p></div>
      <div className="bg-[#fffdf8] p-5"><p className="eyebrow">Strongest shared one-pass signal</p><p className="mt-3 text-2xl font-semibold">Token-surprise spread</p><p className="mt-2 max-w-3xl text-sm leading-6 text-[#69716d]">Its worst-model AUROC is <span className="mono">{format(replication.best_worst_model_auroc)}</span>. Nineteen methods also have pooled 95% interval lower bounds above 0.500 on both models. This is model-transfer evidence on reused questions, not fresh-data confirmation.</p></div>
    </div>
    <div className="card mt-4 overflow-x-auto"><table className="w-full min-w-[920px] text-left text-sm"><thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Method</th><th className="p-4">Qwen AUROC</th><th className="p-4">SmolLM2 AUROC</th><th className="p-4">Slices &gt; .5</th><th className="p-4">Frozen rule</th></tr></thead><tbody>{rows.map((row) => <tr className="border-t hairline" key={row.key}><td className="p-4"><p className="font-semibold">{row.display_name}</p><p className="mt-1 text-xs text-[#69716d]">{row.family}</p></td><td className="mono p-4">{format(row.qwen_auroc)}<br /><span className="text-[10px] text-[#69716d]">{format(row.qwen_auroc_ci_95[0])}–{format(row.qwen_auroc_ci_95[1])}</span></td><td className="mono p-4">{format(row.smollm2_auroc)}<br /><span className="text-[10px] text-[#69716d]">{format(row.smollm2_auroc_ci_95[0])}–{format(row.smollm2_auroc_ci_95[1])}</span></td><td className="mono p-4">{row.qwen_slices_above_chance}/3 · {row.smollm2_slices_above_chance}/3</td><td className={`p-4 font-semibold ${row.replicated ? 'text-[#236349]' : 'text-[#9e321e]'}`}>{row.replicated ? 'replicates' : 'does not transfer'}</td></tr>)}</tbody></table></div>
  </section>;
}

export function FreshBenchmark({ metrics, replication }: { metrics: Record<FreshModelKind, FreshComparisonMetrics>; replication: CrossModelReplication }) {
  const [model, setModel] = useState<FreshModelKind>('qwen_instruct');
  const current = metrics[model];
  const [selected, setSelected] = useState<FreshMethodKey>('top3_token_surprise');
  const smol = model === 'smollm2_instruct';
  return <section className="shell py-16"><div><p className="eyebrow">Cross-family replication + complete audit</p><h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">Same questions. Different small model.</h2><p className="mt-3 max-w-3xl text-sm leading-6 text-[#69716d]">AUROC asks how often one wrong answer receives higher risk than one correct answer. First compare Qwen with SmolLM2; then select any checkpoint, method, or question to inspect its formula and arithmetic.</p></div>
    <CrossModelTable replication={replication} />
    <div className="mt-8 flex flex-wrap gap-2" role="group" aria-label="Saved benchmark model">
      {(Object.keys(freshModelConfig) as FreshModelKind[]).map((key) => <button key={key} onClick={() => setModel(key)} className={`focus-ring px-4 py-3 text-sm font-semibold transition ${model === key ? 'bg-[#18211d] text-white' : 'border hairline bg-white hover:bg-[#f7f4ed]'}`}>{freshModelConfig[key].shortLabel}{key === 'qwen_base' && <span className="ml-2 text-[10px] font-normal opacity-70">stress test</span>}</button>)}
    </div>
    <div className="mt-4 border-l-4 border-[#df4c2f] bg-[#fff8f4] p-4 text-sm leading-6"><strong>Evidence boundary for {freshModelConfig[model].shortLabel}.</strong> {smol ? 'All 31 methods were frozen before any SmolLM2 output or aggregate result was inspected. The questions and prior Qwen results were already known.' : 'The original five methods were frozen before Qwen held-out evaluation; the other 26 are post-hoc exploratory views.'}</div>
    <div className="mt-5 grid gap-3 sm:grid-cols-3"><div className="bg-[#eeeae1] p-4"><p className="eyebrow">Calibration</p><p className="mono mt-2 text-2xl">{current.calibration_examples}</p><p className="mt-1 text-xs text-[#69716d]">cutoffs + trained baselines only</p></div><div className="bg-[#eeeae1] p-4"><p className="eyebrow">Held-out</p><p className="mono mt-2 text-2xl">{current.held_out_examples}</p><p className="mt-1 text-xs text-[#69716d]">{current.held_out_incorrect} labelled wrong</p></div><div className="bg-[#eeeae1] p-4"><p className="eyebrow">Audit</p><p className="mono mt-2 text-2xl">31 methods</p><p className="mt-1 text-xs text-[#69716d]">12 headline · 24 trace-only scalars</p></div></div>
    {current.methods.top3_token_surprise.by_dataset.truthful_qa.correct === 0 && <div className="mt-4 bg-[#fbe9e2] p-4 text-sm leading-6 text-[#6f2d20]"><strong>Why this TruthfulQA slice is “—”.</strong> None of its 100 held-out generations contained a complete accepted answer alias, so that slice has one label class and AUROC is undefined.</div>}
    <div className="mt-5"><MethodTable metrics={current} selected={selected} onSelect={setSelected} /><SelectedMethod metrics={current} methodKey={selected} /><FullLeaderboard metrics={current} selected={selected} onSelect={setSelected} /><LiteratureMap /></div><FreshQuestionInspector model={model} metrics={current} />
  </section>;
}
