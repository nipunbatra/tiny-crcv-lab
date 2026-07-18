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
import questionTokensJson from './data/question-tokens.json';
import { parseJsonl, tokenCalculations } from './lib/metrics';
import type {
  BenchmarkMetrics,
  FeatureKey,
  Features,
  LiveToken,
  ModelKind,
  Prediction,
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
const questionTokens = questionTokensJson as Record<string, Array<{ id: number; piece: string }>>;
type AppView = 'overview' | 'explore' | 'live';

const metricKeys: FeatureKey[] = [
  'crcv_mean',
  'crcv_max',
  'mean_nll',
  'confidence_variance_mean',
  'shift_variance_mean',
  'answer_tokens',
];

const metricShort: Record<FeatureKey, string> = {
  crcv_mean: 'CRCV mean',
  crcv_max: 'CRCV max',
  mean_nll: 'Token surprise',
  confidence_variance_mean: 'Confidence var.',
  shift_variance_mean: 'Hidden-shift var.',
  answer_tokens: 'Answer length',
};

const format = (value: number, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const pct = (value: number) => `${Math.round(value * 100)}%`;

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
            ['overview', 'Overview'],
            ['explore', 'Explore details'],
            ['live', 'Run locally'],
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
        </nav>
        <ModelToggle value={model} onChange={onModel} />
      </div>
    </header>
  );
}

function Hero({ model, onExplore, onRun }: { model: ModelKind; onExplore: () => void; onRun: () => void }) {
  const bench = benchmarks[model];
  const crcv = bench.scores.crcv_mean;
  const confidence = bench.scores.confidence_variance_mean;
  return (
    <section id="top" className="grid-noise border-b hairline">
      <div className="shell grid gap-10 py-14 lg:grid-cols-[1.15fr_.85fr] lg:py-20">
        <div className="max-w-3xl">
          <p className="eyebrow mb-5">A small, inspectable AI experiment</p>
          <h1 className="display max-w-3xl text-[clamp(3rem,6.5vw,6.5rem)] font-[650]">Can a model warn us when it is wrong?</h1>
          <p className="mt-7 max-w-2xl text-lg leading-8 text-[#505955]">
            We asked a tiny open model 100 factual questions, watched its internal signals, and checked whether those signals separated wrong answers from correct ones.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <button onClick={onExplore} className="focus-ring inline-flex items-center gap-2 bg-[#df4c2f] px-5 py-3 font-semibold text-white">See every calculation <ArrowRight size={18} /></button>
            <button onClick={onRun} className="focus-ring inline-flex items-center gap-2 border hairline bg-white px-5 py-3 font-semibold">Try it in my browser <Cpu size={18} /></button>
          </div>
        </div>
        <div className="card self-end p-6 lg:p-8">
          <p className="eyebrow">Short answer</p>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-.04em]">The new signal was weak.</h2>
          <p className="mt-4 text-base leading-7 text-[#505955]">CRCV scored {format(crcv.test_auroc)}. A simpler confidence-only signal scored {format(confidence.test_auroc)} and worked better on this small test.</p>
          <div className="mt-6 border-t hairline pt-5">
            <p className="text-sm font-semibold">How to read the score</p>
            <div className="relative mt-3 h-2 bg-[#ded9cf]"><span className="absolute left-1/2 top-[-5px] h-4 w-px bg-[#69716d]" /><span className="absolute top-0 h-2 bg-[#df4c2f]" style={{ left: '50%', width: `${Math.max(0, (crcv.test_auroc - .5) * 200)}%` }} /></div>
            <div className="mt-2 flex justify-between text-xs text-[#69716d]"><span>0.5 · chance</span><span>1.0 · perfect ranking</span></div>
          </div>
          <div className="mt-6 flex gap-8 text-sm">
            <div><span className="mono block text-xl font-semibold">100</span><span className="text-[#69716d]">questions</span></div>
            <div><span className="mono block text-xl font-semibold">50</span><span className="text-[#69716d]">held out</span></div>
            <div><span className="mono block text-xl font-semibold">{bench.test_hallucinations}</span><span className="text-[#69716d]">wrong</span></div>
          </div>
        </div>
      </div>
    </section>
  );
}

function BeginnerOverview({ model, onExplore }: { model: ModelKind; onExplore: () => void }) {
  const example = predictions[model].find((item) => item.id === 'q002') ?? predictions[model][0];
  const calculation = tokenCalculations(example).find((item) => item.crcv !== null) ?? tokenCalculations(example)[0];
  const bench = benchmarks[model];
  const scores: Array<{ key: FeatureKey; plain: string }> = [
    { key: 'confidence_variance_mean', plain: 'How much confidence jumps around' },
    { key: 'mean_nll', plain: 'How surprised the model is by its own tokens' },
    { key: 'crcv_mean', plain: 'Confidence × hidden-state movement (CRCV)' },
  ];
  return (
    <>
      <section className="shell py-16">
        <div className="grid gap-10 lg:grid-cols-[.7fr_1.3fr]">
          <div>
            <p className="eyebrow">One answer, four steps</p>
            <h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">What are we actually measuring?</h2>
            <p className="mt-5 max-w-lg text-base leading-7 text-[#69716d]">The detector does not fact-check against Wikipedia. It watches the model while it writes, then asks whether its internal behavior looks unusually unstable.</p>
          </div>
          <div className="overview-flow" aria-label="Simplified detector flow">
            <OverviewStep number="1" label="Ask" value={example.question} note="The question enters the model." />
            <ArrowRight className="overview-arrow" size={20} />
            <OverviewStep number="2" label="Watch a token" value={JSON.stringify(calculation.token)} note={`Confidence ${format(calculation.confidence, 3)} · movement ${format(calculation.shift, 3)}`} />
            <ArrowRight className="overview-arrow" size={20} />
            <OverviewStep number="3" label="Combine" value={`${format(calculation.confidence, 3)} × ${format(calculation.shift, 3)} = ${format(calculation.coupling, 3)}`} note="One confidence–movement signal." />
            <ArrowRight className="overview-arrow" size={20} />
            <OverviewStep number="4" label="Check stability" value={calculation.crcv === null ? 'Waiting for 5 tokens' : format(calculation.crcv, 3)} note="How jumpy that signal is over five tokens." accent />
          </div>
        </div>
      </section>

      <section className="border-y hairline bg-[#eae6dc] py-16">
        <div className="shell grid gap-10 lg:grid-cols-[.8fr_1.2fr]">
          <div>
            <p className="eyebrow">What the 50 unseen questions said</p>
            <h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">Simple confidence won this round.</h2>
            <p className="mt-5 max-w-lg leading-7 text-[#69716d]">Higher AUROC means a score more often ranks a wrong answer above a correct one. Chance is 0.500. This is a small experiment, so treat the ranking as a lead—not a verdict.</p>
            <button onClick={onExplore} className="focus-ring mt-7 inline-flex items-center gap-2 font-semibold text-[#9e321e]">Open the evidence and calculations <ArrowRight size={18} /></button>
          </div>
          <div className="space-y-5 self-center">
            {scores.map(({ key, plain }) => {
              const value = bench.scores[key].test_auroc;
              return <div key={key}><div className="mb-2 flex items-end justify-between gap-4"><div><p className="font-semibold">{plain}</p><p className="mt-1 text-xs text-[#69716d]">{key === 'crcv_mean' ? 'the proposed detector' : 'simpler baseline'}</p></div><span className="mono text-2xl font-semibold">{format(value)}</span></div><div className="relative h-2 bg-[#d5d0c6]"><span className="absolute left-1/2 top-[-3px] h-4 w-px bg-[#69716d]" /><span className="absolute h-2 bg-[#df4c2f]" style={{ left: '50%', width: `${Math.max(0, (value - .5) * 200)}%` }} /></div></div>;
            })}
          </div>
        </div>
      </section>
    </>
  );
}

function OverviewStep({ number, label, value, note, accent = false }: { number: string; label: string; value: string; note: string; accent?: boolean }) {
  return <div className={`overview-step ${accent ? 'is-accent' : ''}`}><span className="mono text-xs text-[#69716d]">{number.padStart(2, '0')}</span><p className="mt-4 text-sm font-semibold">{label}</p><p className="mt-2 text-balance text-lg font-semibold leading-6">{value}</p><p className="mt-3 text-xs leading-5 text-[#69716d]">{note}</p></div>;
}

function Results({ model }: { model: ModelKind }) {
  const bench = benchmarks[model];
  return (
    <section id="results" className="shell py-18">
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Experiment 01</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-[-.04em]">What survived the held-out test?</h2>
        </div>
        <p className="max-w-xl text-sm leading-6 text-[#69716d]">AUROC asks: if we draw one wrong and one correct answer, how often does the metric assign the wrong answer a higher score?</p>
      </div>
      <div className="card overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left">
          <thead className="bg-[#eeeae1] text-xs uppercase tracking-[.08em] text-[#69716d]">
            <tr><th className="px-5 py-4">Metric</th><th className="px-5 py-4">Test AUROC</th><th className="px-5 py-4">95% CI</th><th className="px-5 py-4">Macro-F1</th><th className="px-5 py-4">Reading</th></tr>
          </thead>
          <tbody>
            {metricKeys.map((key) => {
              const metric = bench.scores[key];
              const scoreWidth = `${Math.max(0, Math.min(100, metric.test_auroc * 100))}%`;
              return (
                <tr key={key} className="border-t hairline align-middle">
                  <td className="px-5 py-5 font-semibold">{metric.display_name}{key === 'crcv_mean' && <span className="ml-2 bg-[#fbe9e2] px-2 py-1 text-[10px] uppercase text-[#9e321e]">primary</span>}</td>
                  <td className="px-5 py-5"><div className="flex items-center gap-3"><span className="mono w-12 font-semibold">{format(metric.test_auroc)}</span><div className="h-1.5 w-28 bg-[#e7e2d8]"><div className="h-full bg-[#df4c2f]" style={{ width: scoreWidth }} /></div></div></td>
                  <td className="mono px-5 py-5 text-sm text-[#69716d]">{format(metric.test_auroc_ci_95[0])}–{format(metric.test_auroc_ci_95[1])}</td>
                  <td className="mono px-5 py-5">{format(metric.test_macro_f1)}</td>
                  <td className="px-5 py-5 text-sm text-[#69716d]">{metric.test_auroc > .7 ? 'useful signal here' : metric.test_auroc > .55 ? 'weak separation' : 'near chance'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        <Insight icon={<Gauge size={20} />} label="Decision threshold" value={format(bench.scores.crcv_mean.threshold, 4)} text="Chosen only on the 50 calibration questions by maximum macro-F1." />
        <Insight icon={<WarningCircle size={20} />} label="Important result" value={format(bench.paired_comparison.test_auroc_difference)} text="CRCV minus confidence-variability AUROC. Negative means the simpler baseline won." />
        <Insight icon={<BookOpenText size={20} />} label="Operational label" value={`${bench.test_hallucinations}/${bench.test_examples}`} text="Wrong means no normalized accepted answer appeared. This is transparent, but imperfect." />
      </div>
    </section>
  );
}

function Insight({ icon, label, value, text }: { icon: React.ReactNode; label: string; value: string; text: string }) {
  return <div className="card p-5"><div className="mb-5 flex items-center gap-2 text-[#df4c2f]">{icon}<span className="eyebrow">{label}</span></div><p className="metric-number text-3xl font-semibold">{value}</p><p className="mt-3 text-sm leading-6 text-[#69716d]">{text}</p></div>;
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
          <p className="mt-3 max-w-2xl text-[#69716d]">Choose any saved generation. The calculation tape reconstructs the published score from raw token probabilities and 896-dimensional hidden-state movement.</p>
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
      <div className="grid grid-cols-2 gap-px bg-[#d7d3c9] sm:grid-cols-3 xl:grid-cols-6">
        {metricKeys.map((key) => <div className="bg-[#fffdf8] px-3 py-4" key={key}><p className="text-[11px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{key === 'answer_tokens' ? prediction.features[key] : format(prediction.features[key], 4)}</p></div>)}
      </div>

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
  return <div className="mt-3 bg-[#18211d] p-5 text-white"><p className="eyebrow !text-[#bfc7c2]">Selected window ending at token {row.tokenIndex}</p><div className="mt-4 grid gap-4 md:grid-cols-3"><div><p className="text-xs text-[#bfc7c2]">1 · window values</p><p className="mono mt-2 text-sm leading-6">[{row.window!.map((value) => format(value, 5)).join(', ')}]</p></div><div><p className="text-xs text-[#bfc7c2]">2 · sample variance</p><p className="mono mt-2 text-sm leading-6">Σ(s − {format(row.windowMean!, 5)})² / {row.window!.length - 1}<br />= {format(row.sampleVariance!, 7)}</p></div><div><p className="text-xs text-[#bfc7c2]">3 · square root</p><p className="mono mt-2 text-2xl font-semibold">CRCV<sub>t</sub> = {format(row.crcv!, 6)}</p></div></div></div>;
}

function LiveLab({ model }: { model: ModelKind }) {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState('Idle — no model has been downloaded.');
  const [progress, setProgress] = useState(0);
  const [tokens, setTokens] = useState<LiveToken[]>([]);
  const [result, setResult] = useState<{ answer: string; features: Features; elapsedMs: number } | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [question, setQuestion] = useState('What is the capital of France?');
  const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator;

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
      if (message.type === 'result') { setResult(message); setBusy(false); setStatus('Complete — all arithmetic ran locally.'); }
      if (message.type === 'error') { setError(message.message); setBusy(false); setStatus('Stopped'); }
    };
    worker.postMessage({ type: 'run', model, question, runtime: webgpu ? 'webgpu' : 'wasm', maxNewTokens: 24 });
  };

  return (
    <section id="run-live" className="shell py-18">
      <div className="grid gap-8 lg:grid-cols-[.75fr_1.25fr]">
        <div>
          <p className="eyebrow">Local inference</p>
          <h2 className="mt-2 text-4xl font-semibold tracking-[-.05em]">No server. Real hidden states.</h2>
          <p className="mt-5 leading-7 text-[#69716d]">The page fetches the public quantized ONNX model, exposes the final 896-value state already inside its graph, and runs greedy generation in a worker. Your question and trace never leave the browser.</p>
          <div className="mt-6 border-l-4 border-[#df4c2f] bg-[#fbe9e2] p-4 text-sm leading-6 text-[#6f2d20]">
            First run is large: roughly {webgpu ? '483 MB (q4f16)' : '750 MB (q4)'}. The patched model is cached when browser quota permits. {webgpu ? 'WebGPU is available.' : 'WebGPU is unavailable, so the slower WASM path will be used.'}
          </div>
          <ol className="mt-7 space-y-4 text-sm">
            <Step number="01" text="Download and patch only the ONNX output declaration." />
            <Step number="02" text="At each step, read p(token) and the final hidden state." />
            <Step number="03" text="Compute normalized movement, coupling, and rolling CRCV." />
          </ol>
        </div>
        <div className="card p-5 md:p-7">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b hairline pb-5"><div><p className="eyebrow">Browser console</p><p className="mt-1 font-semibold">Qwen2.5-0.5B {model === 'instruct' ? 'Instruct' : 'Base'} · {webgpu ? 'WebGPU' : 'WASM'}</p></div><span className="mono bg-[#eeeae1] px-3 py-2 text-xs">exact CRCV</span></div>
          <label className="mt-6 block"><span className="eyebrow">Question</span><textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} className="focus-ring mt-2 w-full resize-y border hairline bg-white p-3 leading-6" /></label>
          <button onClick={run} disabled={busy || !question.trim()} className="focus-ring mt-4 inline-flex w-full items-center justify-center gap-2 bg-[#df4c2f] px-5 py-3 font-bold text-white"><Play size={18} weight="fill" />{busy ? 'Working locally…' : 'Load model and calculate'}</button>
          <div className="mt-5 border hairline bg-[#f7f4ed] p-4" aria-live="polite"><div className="flex justify-between gap-3 text-xs"><span className="truncate">{status}</span><span className="mono">{progress ? pct(progress / 100) : ''}</span></div>{busy && <div className="mt-3 h-1.5 bg-[#ddd8ce]"><div className="h-full bg-[#df4c2f] transition-all" style={{ width: `${progress}%` }} /></div>}</div>
          {error && <div className="mt-4 flex gap-3 bg-[#fbe9e2] p-4 text-sm text-[#8d2e1d]"><WarningCircle className="shrink-0" size={20} /><span>{error}</span></div>}
          {!busy && !result && !error && <div className="mt-5 py-8 text-center text-sm text-[#69716d]">A live token tape will appear here after you start.</div>}
          {tokens.length > 0 && <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[560px] text-left text-xs"><thead className="text-[#69716d]"><tr><th className="pb-2">token</th><th className="pb-2">confidence</th><th className="pb-2">hidden shift</th><th className="pb-2">coupling</th></tr></thead><tbody>{tokens.map((token, index) => <tr key={`${token.tokenId}-${index}`} className="border-t hairline"><td className="mono py-2">{JSON.stringify(token.token)}</td><td className="mono py-2">{format(token.confidence, 5)}</td><td className="mono py-2">{token.hiddenShift === null ? '—' : format(token.hiddenShift, 5)}</td><td className="mono py-2">{token.hiddenShift === null ? '—' : format(token.confidence * token.hiddenShift, 5)}</td></tr>)}</tbody></table></div>}
          {result && <div className="mt-5 border-t hairline pt-5"><p className="eyebrow">Generated answer</p><p className="mt-2 leading-7">{result.answer}</p><div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">{metricKeys.map((key) => <div key={key} className="bg-[#eeeae1] p-3"><p className="text-[10px] text-[#69716d]">{metricShort[key]}</p><p className="mono mt-1 font-semibold">{key === 'answer_tokens' ? result.features[key] : format(result.features[key], 4)}</p></div>)}</div><p className="mono mt-3 text-xs text-[#69716d]">{format(result.elapsedMs / 1000, 2)} s after model load</p></div>}
        </div>
      </div>
    </section>
  );
}

function Step({ number, text }: { number: string; text: string }) { return <li className="flex items-start gap-4"><span className="mono text-[#df4c2f]">{number}</span><span>{text}</span></li>; }

function MethodNote() {
  return <section className="border-t hairline bg-[#eae6dc] py-12"><div className="shell grid gap-8 md:grid-cols-3"><div><p className="eyebrow">State movement</p><p className="mono mt-3 text-sm leading-7">r<sub>t</sub> = ‖h<sub>t</sub> − h<sub>t−1</sub>‖₂ / (‖h<sub>t−1</sub>‖₂ + 10⁻⁸)</p></div><div><p className="eyebrow">Coupling</p><p className="mono mt-3 text-sm leading-7">s<sub>t</sub> = c<sub>t</sub> · r<sub>t</sub></p><p className="mt-1 text-sm text-[#69716d]">Confidence multiplied by state movement.</p></div><div><p className="eyebrow">CRCV</p><p className="mt-3 text-sm leading-7">Sample standard deviation of s over each complete trailing window, W = 5.</p></div></div></section>;
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
      {view === 'overview' && <><Hero model={model} onExplore={() => changeView('explore')} onRun={() => changeView('live')} /><BeginnerOverview model={model} onExplore={() => changeView('explore')} /></>}
      {view === 'explore' && <><Results model={model} /><QuestionLab model={model} /><MethodNote /></>}
      {view === 'live' && <><LiveLab model={model} /><MethodNote /></>}
    </main>
    <footer className="bg-[#18211d] py-7 text-[#aeb7b2]"><div className="shell flex flex-wrap justify-between gap-3 text-xs"><span>Tiny CRCV Lab · inspectable research prototype</span><span>100 questions · 50 calibration / 50 held out · greedy decoding</span></div></footer>
  </>;
}
