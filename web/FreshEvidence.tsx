import { CheckCircle, MagnifyingGlass, WarningCircle } from '@phosphor-icons/react';
import { useEffect, useMemo, useState } from 'react';
import { parseJsonl } from './lib/metrics';
import type {
  FreshComparisonMetrics,
  FreshMethodKey,
  FreshPrediction,
  ModelKind,
} from './types';

const methodOrder: FreshMethodKey[] = [
  'top3_token_surprise',
  'p_true',
  'hidden_logistic_probe',
  'semantic_disagreement_3',
  'answer_tokens',
];

const methodRoles: Record<FreshMethodKey, string> = {
  top3_token_surprise: 'Frozen cheap baseline',
  p_true: 'One extra model pass',
  hidden_logistic_probe: 'Calibration-trained 896→1 probe',
  semantic_disagreement_3: 'Three samples + three judgments',
  answer_tokens: 'Length confound only',
};

const datasetLabels = {
  nq_open: 'NQ-Open',
  trivia_qa: 'TriviaQA',
  truthful_qa: 'TruthfulQA',
} as const;

const traceCache: Partial<Record<ModelKind, FreshPrediction[]>> = {};
const format = (value: number | null | undefined, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const signed = (value: number, digits = 3) => `${value >= 0 ? '+' : ''}${format(value, digits)}`;

export function FreshSummary({ metrics, onExplore }: { metrics: Record<ModelKind, FreshComparisonMetrics>; onExplore: () => void }) {
  return <section className="border-b hairline bg-[#eae6dc] py-16">
    <div className="shell">
      <div className="grid gap-5 lg:grid-cols-[1fr_.72fr] lg:items-end">
        <div>
          <p className="eyebrow">Fresh confirmatory comparison</p>
          <h2 className="mt-2 max-w-4xl text-4xl font-semibold tracking-[-.05em]">Five scores. 600 public questions. Two checkpoints.</h2>
        </div>
        <p className="text-sm leading-6 text-[#69716d]">Each checkpoint answered 200 NQ-Open, 200 TriviaQA, and 200 TruthfulQA questions. Thresholds and the probe used 300 calibration questions; the AUROCs below use the untouched other 300.</p>
      </div>
      <div className="card mt-7 overflow-x-auto">
        <table className="w-full min-w-[780px] text-left text-sm">
          <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Method</th><th className="p-4">Instruct AUROC</th><th className="p-4">Base AUROC</th><th className="p-4">What it costs</th></tr></thead>
          <tbody>{methodOrder.map((key) => {
            const instruct = metrics.instruct.methods[key];
            const base = metrics.base.methods[key];
            const seconds = (instruct.incremental_runtime.mean_seconds_per_question + base.incremental_runtime.mean_seconds_per_question) / 2;
            const cost = key === 'hidden_logistic_probe'
              ? 'One 896→1 dot product; one-time calibration fit'
              : key === 'top3_token_surprise' || key === 'answer_tokens'
                ? 'Already in the generation trace'
                : `${format(seconds, 3)} extra seconds / question locally`;
            return <tr className={`border-t hairline ${key === 'answer_tokens' ? 'text-[#69716d]' : ''}`} key={key}>
              <td className="p-4"><p className="font-semibold">{instruct.display_name}</p><p className="mt-1 text-xs text-[#69716d]">{methodRoles[key]}</p></td>
              <td className="mono p-4 text-lg">{format(instruct.held_out.auroc)}</td>
              <td className="mono p-4 text-lg">{format(base.held_out.auroc)}</td>
              <td className="p-4 text-xs leading-5 text-[#69716d]">{cost}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <div className="mt-5 grid gap-4 lg:grid-cols-[1.25fr_.75fr]">
        <div className="border-l-4 border-[#df4c2f] bg-[#fff8f4] p-5 text-sm leading-6"><strong>Decision rule.</strong> A new method counts as an improvement only when its paired bootstrap AUROC-difference interval is entirely above zero versus frozen top-3 surprise. The detailed view shows that decision separately for both checkpoints.</div>
        <button onClick={onExplore} className="focus-ring flex items-center justify-between bg-[#18211d] p-5 text-left font-semibold text-white"><span>Inspect dataset slices and all 600 calculations</span><span aria-hidden="true">→</span></button>
      </div>
    </div>
  </section>;
}

function MethodTable({ metrics, selected, onSelect }: { metrics: FreshComparisonMetrics; selected: FreshMethodKey; onSelect: (method: FreshMethodKey) => void }) {
  return <div className="card overflow-x-auto">
    <table className="w-full min-w-[980px] text-left text-sm">
      <thead className="bg-[#eeeae1] text-[11px] uppercase tracking-[.07em] text-[#69716d]"><tr><th className="p-4">Method</th><th className="p-4">All 300 test</th><th className="p-4">NQ-Open</th><th className="p-4">TriviaQA</th><th className="p-4">TruthfulQA</th><th className="p-4">Δ vs top-3</th><th className="p-4">Inspect</th></tr></thead>
      <tbody>{methodOrder.map((key) => {
        const method = metrics.methods[key];
        const paired = method.paired_vs_top3_token_surprise;
        return <tr key={key} className={`border-t hairline ${selected === key ? 'bg-[#fff1ec]' : ''}`}>
          <td className="p-4"><p className="font-semibold">{method.display_name}</p><p className="mt-1 text-xs text-[#69716d]">{methodRoles[key]}</p></td>
          <td className="mono p-4"><strong>{format(method.held_out.auroc)}</strong><br /><span className="text-[10px] text-[#69716d]">{format(method.held_out.auroc_ci_95[0])}–{format(method.held_out.auroc_ci_95[1])}</span></td>
          <td className="mono p-4">{format(method.by_dataset.nq_open.auroc)}</td>
          <td className="mono p-4">{format(method.by_dataset.trivia_qa.auroc)}</td>
          <td className="mono p-4">{format(method.by_dataset.truthful_qa.auroc)}</td>
          <td className="mono p-4">{paired ? <>{signed(paired.auroc_difference)}<br /><span className={`text-[10px] ${paired.reliably_improves ? 'text-[#236349]' : 'text-[#69716d]'}`}>{format(paired.auroc_difference_ci_95[0])} to {format(paired.auroc_difference_ci_95[1])}</span></> : 'baseline'}</td>
          <td className="p-4"><button onClick={() => onSelect(key)} className="focus-ring border hairline bg-white px-3 py-2 text-xs font-semibold">{selected === key ? 'Selected' : 'Open'}</button></td>
        </tr>;
      })}</tbody>
    </table>
  </div>;
}

function SelectedMethod({ metrics, methodKey }: { metrics: FreshComparisonMetrics; methodKey: FreshMethodKey }) {
  const method = metrics.methods[methodKey];
  const paired = method.paired_vs_top3_token_surprise;
  return <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_.8fr]">
    <div className="bg-[#18211d] p-5 text-white">
      <p className="eyebrow !text-[#bfc7c2]">Exact formula</p>
      <p className="mono mt-3 text-sm leading-7">{method.formula}</p>
      <p className="mt-4 text-xs leading-5 text-[#bfc7c2]">Higher predicts a wrong answer. Classification cutoff {format(method.threshold, 5)} was chosen once on all 300 calibration questions by macro-F1, then frozen.</p>
    </div>
    <div className="bg-[#fffdf8] p-5">
      <p className="eyebrow">Held-out operating point</p>
      <div className="mt-3 grid grid-cols-3 gap-3"><div><p className="text-[10px] text-[#69716d]">AUROC</p><p className="mono mt-1 text-xl">{format(method.held_out.auroc)}</p></div><div><p className="text-[10px] text-[#69716d]">Average precision</p><p className="mono mt-1 text-xl">{format(method.held_out.average_precision)}</p></div><div><p className="text-[10px] text-[#69716d]">Macro-F1</p><p className="mono mt-1 text-xl">{format(method.held_out.macro_f1)}</p></div></div>
      <p className={`mt-4 text-xs leading-5 ${paired?.reliably_improves ? 'text-[#236349]' : 'text-[#69716d]'}`}>{methodKey === 'top3_token_surprise' ? 'This is the frozen comparison baseline.' : paired?.reliably_improves ? 'Its paired 95% interval is above zero: reliable improvement by the predeclared rule.' : 'Its paired 95% interval does not establish improvement over top-3 surprise.'}</p>
    </div>
  </div>;
}

function Top3Audit({ prediction }: { prediction: FreshPrediction }) {
  const selected = prediction.confidences.map((confidence, index) => ({ token: prediction.token_pieces[index], confidence, surprise: -Math.log(Math.max(confidence, 1e-12)) })).sort((a, b) => b.surprise - a.surprise).slice(0, 3);
  return <AuditCard name="Top-3 token surprise" score={prediction.method_scores.top3_token_surprise} formula="mean(top 3 of −ln confidence)">
    <p>{selected.map((item) => `${JSON.stringify(item.token)}: −ln(${format(item.confidence, 5)}) = ${format(item.surprise, 5)}`).join(' · ')}</p>
    <p className="mono mt-3">({selected.map((item) => format(item.surprise, 5)).join(' + ')}) / {selected.length} = <strong>{format(prediction.method_scores.top3_token_surprise, 5)}</strong></p>
  </AuditCard>;
}

function PTrueAudit({ prediction }: { prediction: FreshPrediction }) {
  const judgment = prediction.p_true_judgment;
  return <AuditCard name="P(False self-check)" score={prediction.method_scores.p_true} formula="p(No) / (p(Yes) + p(No))">
    <p>The model rereads the question and its greedy answer, then scores one-token <span className="mono">Yes</span> and <span className="mono">No</span> continuations.</p>
    <p className="mono mt-3">{format(judgment.no_probability, 7)} / ({format(judgment.yes_probability, 7)} + {format(judgment.no_probability, 7)}) = <strong>{format(judgment.normalized_no_probability, 5)}</strong></p>
  </AuditCard>;
}

function ProbeAudit({ prediction }: { prediction: FreshPrediction }) {
  const probe = prediction.probe_explanation;
  return <AuditCard name="Mean-hidden linear probe" score={prediction.method_scores.hidden_logistic_probe} formula="sigmoid(b + Σ wⱼzⱼ)">
    <p>First average the final 896-value hidden state over answer-token decisions. Standardize each dimension with calibration-only mean and SD; multiply by its learned linear weight.</p>
    <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[640px] text-left text-xs"><thead className="text-[#69716d]"><tr><th className="pb-2">dimension</th><th className="pb-2">hⱼ</th><th className="pb-2">(hⱼ−μⱼ)/σⱼ</th><th className="pb-2">wⱼ</th><th className="pb-2">wⱼzⱼ</th></tr></thead><tbody>{probe.top_contributions.map((item) => <tr className="border-t hairline" key={item.dimension}><td className="mono py-2">{item.dimension}</td><td className="mono py-2">{format(item.raw_hidden, 4)}</td><td className="mono py-2">({format(item.raw_hidden, 3)}−{format(item.calibration_mean, 3)})/{format(item.calibration_scale, 3)} = {format(item.standardized_hidden, 3)}</td><td className="mono py-2">{signed(item.weight, 4)}</td><td className="mono py-2">{signed(item.contribution, 4)}</td></tr>)}</tbody></table></div>
    <p className="mono mt-3">logit = bias {signed(probe.bias, 5)} + shown {signed(probe.top_contributions.reduce((sum, item) => sum + item.contribution, 0), 5)} + other 888 dims {signed(probe.other_dimensions_contribution, 5)} = {format(probe.logit, 5)}</p>
    <p className="mono mt-2">sigmoid({format(probe.logit, 5)}) = <strong>{format(probe.sigmoid_risk, 5)}</strong></p>
  </AuditCard>;
}

function DisagreementAudit({ prediction }: { prediction: FreshPrediction }) {
  return <AuditCard name="Three-answer disagreement" score={prediction.method_scores.semantic_disagreement_3} formula="mean of 3 pairwise P(not same answer)">
    <div className="grid gap-2 md:grid-cols-3">{prediction.stochastic_answers.map((sample) => <div className="bg-[#eeeae1] p-3" key={sample.sample_index}><p className="mono text-[10px] text-[#69716d]">sample {sample.sample_index + 1} · seed {sample.seed}</p><p className="mt-2 text-xs leading-5">{sample.answer || 'No text'}</p></div>)}</div>
    <p className="mono mt-3">pair risks = [{prediction.pair_judgments.map((pair) => `${pair.sample_a + 1}↔${pair.sample_b + 1}: ${format(pair.normalized_no_probability, 5)}`).join(', ')}]</p>
    <p className="mono mt-2">mean = ({prediction.pair_judgments.map((pair) => format(pair.normalized_no_probability, 5)).join(' + ')}) / 3 = <strong>{format(prediction.method_scores.semantic_disagreement_3, 5)}</strong></p>
  </AuditCard>;
}

function LengthAudit({ prediction }: { prediction: FreshPrediction }) {
  return <AuditCard name="Answer length control" score={prediction.method_scores.answer_tokens} formula="count(greedy answer tokens)">
    <div className="flex flex-wrap gap-1.5">{prediction.token_pieces.map((token, index) => <span className="mono border hairline bg-white px-2 py-1 text-xs" key={`${index}-${token}`}>{index + 1} · {JSON.stringify(token)}</span>)}</div>
    <p className="mono mt-3">count = <strong>{prediction.token_pieces.length}</strong></p>
  </AuditCard>;
}

function AuditCard({ name, score, formula, children }: { name: string; score: number; formula: string; children: React.ReactNode }) {
  return <details className="border-t hairline py-4" open={name === 'Top-3 token surprise'}>
    <summary className="focus-ring flex cursor-pointer list-none items-center justify-between gap-4"><span><strong>{name}</strong><span className="mono ml-3 text-xs text-[#69716d]">{formula}</span></span><span className="mono bg-[#eeeae1] px-3 py-2 font-semibold">{format(score, 5)}</span></summary>
    <div className="mt-4 bg-[#f7f4ed] p-4 text-xs leading-6">{children}</div>
  </details>;
}

function FreshQuestionInspector({ model, metrics }: { model: ModelKind; metrics: FreshComparisonMetrics }) {
  const [rows, setRows] = useState<FreshPrediction[]>(traceCache[model] ?? []);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [dataset, setDataset] = useState<'all' | FreshPrediction['source_dataset']>('all');
  const [selectedId, setSelectedId] = useState('');

  useEffect(() => {
    let cancelled = false;
    setError('');
    if (traceCache[model]) { setRows(traceCache[model]!); return; }
    setRows([]);
    fetch(`./data/fresh_qa_qwen05b_${model}.jsonl`).then((response) => {
      if (!response.ok) throw new Error(`Trace download failed (${response.status})`);
      return response.text();
    }).then((text) => {
      const loaded = parseJsonl<FreshPrediction>(text);
      traceCache[model] = loaded;
      if (!cancelled) setRows(loaded);
    }).catch((reason) => { if (!cancelled) setError(String(reason)); });
    return () => { cancelled = true; };
  }, [model]);

  const filtered = useMemo(() => rows.filter((row) => {
    const haystack = `${row.id} ${row.question} ${row.generated_answer}`.toLowerCase();
    return (dataset === 'all' || row.source_dataset === dataset) && haystack.includes(query.toLowerCase());
  }), [dataset, query, rows]);
  const selected = rows.find((row) => row.id === selectedId) ?? filtered[0] ?? rows[0];

  return <section className="mt-14">
    <div className="mb-6"><p className="eyebrow">Question-level arithmetic</p><h3 className="mt-2 text-3xl font-semibold tracking-[-.04em]">See all five scores for any question.</h3><p className="mt-3 max-w-3xl text-sm leading-6 text-[#69716d]">Nothing is summarized away: selected token probabilities, binary Yes/No probabilities, three sampled answers, pair judgments, and the strongest probe contributions are saved below.</p></div>
    <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
      <aside className="card overflow-hidden">
        <div className="border-b hairline p-4"><label className="flex items-center gap-2 border hairline bg-white px-3 py-2.5"><MagnifyingGlass size={17} className="text-[#69716d]" /><input value={query} onChange={(event) => setQuery(event.target.value)} className="min-w-0 flex-1 bg-transparent text-sm outline-none" placeholder="Search 600 questions" /></label><select value={dataset} onChange={(event) => setDataset(event.target.value as typeof dataset)} className="focus-ring mt-3 w-full border hairline bg-white px-3 py-2 text-sm"><option value="all">All three datasets</option>{Object.entries(datasetLabels).map(([key, label]) => <option value={key} key={key}>{label}</option>)}</select></div>
        <div className="max-h-[720px] overflow-y-auto">{rows.length === 0 && !error ? <div className="p-8 text-center text-sm text-[#69716d]">Loading saved traces…</div> : error ? <div className="p-6 text-sm text-[#9e321e]">{error}</div> : filtered.map((row) => <button key={row.id} onClick={() => setSelectedId(row.id)} className={`focus-ring block w-full border-b hairline p-4 text-left ${selected?.id === row.id ? 'bg-[#fff8f4]' : 'hover:bg-[#f7f4ed]'}`}><div className="flex items-center justify-between gap-2"><span className="mono text-[10px] text-[#69716d]">{datasetLabels[row.source_dataset]} · {row.split}</span><span className={`h-2.5 w-2.5 rounded-full ${row.correct ? 'bg-[#4d8a6e]' : 'bg-[#df4c2f]'}`} /></div><p className="mt-2 line-clamp-2 text-sm font-medium leading-5">{row.question}</p></button>)}</div>
      </aside>
      {selected ? <article className="card min-w-0 p-5 md:p-7">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b hairline pb-5"><div><p className="eyebrow">{datasetLabels[selected.source_dataset]} · {selected.split} · source row {selected.source_index}</p><h4 className="mt-2 text-2xl font-semibold tracking-[-.03em]">{selected.question}</h4></div><span className={`inline-flex items-center gap-2 px-3 py-2 text-sm font-semibold ${selected.correct ? 'bg-[#e5f1eb] text-[#236349]' : 'bg-[#fbe9e2] text-[#9e321e]'}`}>{selected.correct ? <CheckCircle weight="fill" /> : <WarningCircle weight="fill" />}{selected.correct ? 'alias matched' : 'no alias matched'}</span></div>
        <div className="grid gap-4 border-b hairline py-5 md:grid-cols-2"><div><p className="eyebrow">Greedy answer</p><p className="mt-2 text-sm leading-6">{selected.generated_answer}</p></div><div><p className="eyebrow">Accepted aliases</p><p className="mt-2 text-sm leading-6">{selected.answers.join(' · ')}</p></div></div>
        <div className="grid grid-cols-2 gap-px bg-[#d7d3c9] py-px sm:grid-cols-5">{methodOrder.map((key) => {
          const score = selected.method_scores[key];
          const threshold = metrics.methods[key].threshold;
          return <div className="bg-[#fffdf8] p-3" key={key}><p className="text-[10px] leading-4 text-[#69716d]">{metrics.methods[key].display_name}</p><p className="mono mt-1 font-semibold">{format(score, 4)}</p><p className={`mt-2 text-[9px] font-semibold uppercase ${score >= threshold ? 'text-[#9e321e]' : 'text-[#236349]'}`}>{score >= threshold ? 'flagged' : 'not flagged'} · cutoff {format(threshold, 3)}</p></div>;
        })}</div>
        <Top3Audit prediction={selected} />
        <PTrueAudit prediction={selected} />
        <ProbeAudit prediction={selected} />
        <DisagreementAudit prediction={selected} />
        <LengthAudit prediction={selected} />
        <p className="mt-4 text-xs leading-5 text-[#69716d]">The visible label is deliberately mechanical: a normalized gold alias must occur as a whole phrase. Semantically correct paraphrases can therefore be marked wrong; inspect them before treating this as ground truth.</p>
      </article> : <div className="card grid min-h-[360px] place-items-center text-sm text-[#69716d]">Select a question to inspect.</div>}
    </div>
  </section>;
}

export function FreshBenchmark({ model, metrics }: { model: ModelKind; metrics: Record<ModelKind, FreshComparisonMetrics> }) {
  const current = metrics[model];
  const [selected, setSelected] = useState<FreshMethodKey>('top3_token_surprise');
  return <section className="shell py-16">
    <div><p className="eyebrow">Primary evidence · frozen protocol</p><h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">Fresh public QA comparison.</h2><p className="mt-3 max-w-3xl text-sm leading-6 text-[#69716d]">AUROC asks how often one wrong answer receives a higher risk than one correct answer. 0.500 is chance. Select any method for its cutoff, interval, formula, and operating metrics.</p></div>
    <div className="mt-7 grid gap-3 sm:grid-cols-3"><div className="bg-[#eeeae1] p-4"><p className="eyebrow">Calibration</p><p className="mono mt-2 text-2xl">{current.calibration_examples}</p><p className="mt-1 text-xs text-[#69716d]">thresholds + probe only</p></div><div className="bg-[#eeeae1] p-4"><p className="eyebrow">Untouched test</p><p className="mono mt-2 text-2xl">{current.held_out_examples}</p><p className="mt-1 text-xs text-[#69716d]">{current.held_out_incorrect} labelled wrong</p></div><div className="bg-[#eeeae1] p-4"><p className="eyebrow">Datasets</p><p className="mono mt-2 text-2xl">3 × 200</p><p className="mt-1 text-xs text-[#69716d]">balanced by source, not outcome</p></div></div>
    {current.methods.top3_token_surprise.by_dataset.truthful_qa.correct === 0 && <div className="mt-4 bg-[#fbe9e2] p-4 text-sm leading-6 text-[#6f2d20]"><strong>Why Base × TruthfulQA is “—”.</strong> None of its 100 held-out generations contained a complete accepted answer alias, so that slice has one label class and AUROC is undefined. The rows remain available for audit; the pooled result still has both classes.</div>}
    <div className="mt-5"><MethodTable metrics={current} selected={selected} onSelect={setSelected} /><SelectedMethod metrics={current} methodKey={selected} /></div>
    <FreshQuestionInspector model={model} metrics={current} />
  </section>;
}
