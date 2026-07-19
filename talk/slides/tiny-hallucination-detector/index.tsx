import {
  Step,
  Steps,
  useIsActivePage,
  useSlidePageNumber,
  type DesignSystem,
  type Page,
  type SlideMeta,
  type SlideTransition,
} from '@open-slide/core';
import type { CSSProperties, ReactNode } from 'react';

export const design: DesignSystem = {
  palette: {
    bg: '#f4f1ea',
    text: '#18211d',
    accent: '#df4c2f',
  },
  fonts: {
    display: '"Geist Variable", "Avenir Next", system-ui, sans-serif',
    body: '"Geist Variable", "Avenir Next", system-ui, sans-serif',
  },
  typeScale: {
    hero: 142,
    body: 34,
  },
  radius: 14,
};

const C = {
  paper: '#f4f1ea',
  panel: '#fffdf8',
  ink: '#18211d',
  muted: '#68716c',
  line: '#d7d3c9',
  accent: '#df4c2f',
  accentSoft: '#fbe8df',
  safe: '#2d6a4f',
  safeSoft: '#e4efe8',
  dark: '#18211d',
  darkPanel: '#24302a',
  darkLine: '#405048',
  darkMuted: '#b2bbb6',
};

const mono = '"Geist Mono Variable", "SFMono-Regular", Consolas, monospace';
const PAD = 118;

const deckCss = [
  '@keyframes crcv-flow { from { stroke-dashoffset: 42; } to { stroke-dashoffset: 0; } }',
  '@keyframes crcv-pulse { 0%,100% { opacity:.35; } 50% { opacity:1; } }',
  '.crcv-flow { animation: crcv-flow 1.25s cubic-bezier(.16,1,.3,1) infinite; }',
  '.crcv-pulse { animation: crcv-pulse 1.55s cubic-bezier(.16,1,.3,1) infinite; }',
  '@media (prefers-reduced-motion: reduce) { .crcv-flow,.crcv-pulse { animation:none !important; } }',
].join('\n');

const DeckStyles = () => <style>{deckCss}</style>;

const basePage: CSSProperties = {
  width: '100%',
  height: '100%',
  position: 'relative',
  background: 'var(--osd-bg)',
  color: 'var(--osd-text)',
  fontFamily: 'var(--osd-font-body)',
};

const Grid = ({ dark = false }: { dark?: boolean }) => (
  <div
    style={{
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none',
      backgroundImage:
        'linear-gradient(' +
        (dark ? 'rgba(244,241,234,.05)' : 'rgba(24,33,29,.045)') +
        ' 1px, transparent 1px), linear-gradient(90deg, ' +
        (dark ? 'rgba(244,241,234,.05)' : 'rgba(24,33,29,.045)') +
        ' 1px, transparent 1px)',
      backgroundSize: '48px 48px',
      maskImage: 'linear-gradient(to bottom, rgba(0,0,0,.65), transparent 84%)',
    }}
  />
);

const Footer = ({ chapter, dark = false }: { chapter: string; dark?: boolean }) => {
  const { current, total } = useSlidePageNumber();
  return (
    <div
      style={{
        position: 'absolute',
        left: PAD,
        right: PAD,
        bottom: 30,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderTop: '1px solid ' + (dark ? C.darkLine : C.line),
        paddingTop: 13,
        fontFamily: mono,
        fontSize: 19,
        color: dark ? C.darkMuted : C.muted,
        letterSpacing: '.08em',
        textTransform: 'uppercase',
      }}
    >
      <span>{chapter}</span>
      <span>
        {String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}
      </span>
    </div>
  );
};

const Eyebrow = ({ children, dark = false }: { children: ReactNode; dark?: boolean }) => (
  <div
    style={{
      fontFamily: mono,
      fontSize: 21,
      fontWeight: 650,
      letterSpacing: '.13em',
      textTransform: 'uppercase',
      color: dark ? C.darkMuted : C.muted,
    }}
  >
    {children}
  </div>
);

const Heading = ({
  children,
  dark = false,
  size = 66,
}: {
  children: ReactNode;
  dark?: boolean;
  size?: number;
}) => (
  <h2
    style={{
      margin: '14px 0 0',
      maxWidth: 1620,
      fontFamily: 'var(--osd-font-display)',
      fontSize: size,
      lineHeight: 1.06,
      letterSpacing: '-.045em',
      fontWeight: 760,
      color: dark ? C.paper : C.ink,
      textWrap: 'balance',
    }}
  >
    {children}
  </h2>
);

const Accent = ({ children }: { children: ReactNode }) => (
  <span style={{ color: C.accent }}>{children}</span>
);

const PageFrame = ({
  eyebrow,
  title,
  chapter,
  children,
  dark = false,
  titleSize = 66,
}: {
  eyebrow: string;
  title: ReactNode;
  chapter: string;
  children: ReactNode;
  dark?: boolean;
  titleSize?: number;
}) => (
  <section style={{ ...basePage, background: dark ? C.dark : C.paper, color: dark ? C.paper : C.ink }}>
    <DeckStyles />
    <Grid dark={dark} />
    <div style={{ position: 'absolute', inset: '82px 118px 92px' }}>
      <Eyebrow dark={dark}>{eyebrow}</Eyebrow>
      <Heading dark={dark} size={titleSize}>
        {title}
      </Heading>
      <div style={{ marginTop: 42 }}>{children}</div>
    </div>
    <Footer chapter={chapter} dark={dark} />
  </section>
);

const Pill = ({
  children,
  tone = 'plain',
}: {
  children: ReactNode;
  tone?: 'plain' | 'accent' | 'safe' | 'dark';
}) => {
  const background =
    tone === 'accent' ? C.accentSoft : tone === 'safe' ? C.safeSoft : tone === 'dark' ? C.darkPanel : C.panel;
  const color = tone === 'accent' ? C.accent : tone === 'safe' ? C.safe : tone === 'dark' ? C.paper : C.ink;
  const border = tone === 'accent' ? C.accent : tone === 'safe' ? C.safe : tone === 'dark' ? C.darkLine : C.line;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        minHeight: 38,
        padding: '6px 14px',
        borderRadius: 999,
        border: '1px solid ' + border,
        background,
        color,
        fontFamily: mono,
        fontSize: 19,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  );
};

const FlowArrow = ({ width = 100, dark = false }: { width?: number; dark?: boolean }) => {
  const active = useIsActivePage();
  return (
    <svg width={width} height="34" viewBox={'0 0 ' + width + ' 34'} aria-hidden="true">
      <line
        className={active ? 'crcv-flow' : undefined}
        x1="2"
        y1="17"
        x2={width - 18}
        y2="17"
        stroke={dark ? C.paper : C.accent}
        strokeWidth="4"
        strokeDasharray="12 9"
      />
      <path
        d={'M ' + (width - 28) + ' 7 L ' + (width - 13) + ' 17 L ' + (width - 28) + ' 27'}
        fill="none"
        stroke={dark ? C.paper : C.accent}
        strokeWidth="4"
      />
    </svg>
  );
};

const Node = ({
  label,
  detail,
  accent = false,
  width = 290,
  dark = false,
}: {
  label: string;
  detail: string;
  accent?: boolean;
  width?: number;
  dark?: boolean;
}) => (
  <div
    style={{
      width,
      minHeight: 132,
      padding: '22px 24px',
      border: '1px solid ' + (accent ? C.accent : dark ? C.darkLine : C.line),
      borderTop: '4px solid ' + (accent ? C.accent : dark ? C.paper : C.ink),
      background: accent ? C.accent : dark ? C.darkPanel : C.panel,
      color: accent ? C.panel : dark ? C.paper : C.ink,
      borderRadius: 'var(--osd-radius)',
    }}
  >
    <div style={{ fontSize: 29, fontWeight: 720 }}>{label}</div>
    <div
      style={{
        marginTop: 12,
        fontFamily: mono,
        fontSize: 20,
        lineHeight: 1.35,
        color: accent ? C.panel : dark ? C.darkMuted : C.muted,
      }}
    >
      {detail}
    </div>
  </div>
);

const AnswerCard = ({
  label,
  answer,
  verdict,
  detail,
  wrong = false,
}: {
  label: string;
  answer: string;
  verdict: string;
  detail?: string;
  wrong?: boolean;
}) => (
  <div
    style={{
      minHeight: 286,
      padding: 34,
      borderTop: '5px solid ' + (wrong ? C.accent : C.safe),
      background: wrong ? C.accentSoft : C.safeSoft,
    }}
  >
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 18, alignItems: 'center' }}>
      <Eyebrow>{label}</Eyebrow>
      <Pill tone={wrong ? 'accent' : 'safe'}>{verdict}</Pill>
    </div>
    <div style={{ marginTop: 28, fontSize: 42, lineHeight: 1.25, fontWeight: 700, letterSpacing: '-.025em' }}>
      {answer}
    </div>
    {detail ? <div style={{ marginTop: 22, fontSize: 25, lineHeight: 1.4, color: C.muted }}>{detail}</div> : null}
  </div>
);

const ScopeRow = ({
  input,
  status,
  explanation,
  accent = false,
}: {
  input: string;
  status: string;
  explanation: string;
  accent?: boolean;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '430px 250px 1fr',
      alignItems: 'center',
      gap: 32,
      minHeight: 138,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div style={{ fontFamily: mono, fontSize: 27 }}>{input}</div>
    <div style={{ fontSize: 27, fontWeight: 720, color: accent ? C.accent : C.muted }}>{status}</div>
    <div style={{ fontSize: 28, lineHeight: 1.35, color: C.muted }}>{explanation}</div>
  </div>
);

const ExampleRow = ({
  source,
  question,
  correct,
  wrong,
}: {
  source: string;
  question: string;
  correct: string;
  wrong: string;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '155px 1.18fr .8fr .92fr',
      gap: 24,
      alignItems: 'center',
      minHeight: 160,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div><Pill>{source}</Pill></div>
    <div style={{ fontSize: 27, lineHeight: 1.35, fontWeight: 620 }}>{question}</div>
    <div>
      <div style={{ fontFamily: mono, fontSize: 18, color: C.safe }}>CORRECT</div>
      <div style={{ marginTop: 8, fontSize: 27, lineHeight: 1.3 }}>{correct}</div>
    </div>
    <div>
      <div style={{ fontFamily: mono, fontSize: 18, color: C.accent }}>WRONG</div>
      <div style={{ marginTop: 8, fontSize: 27, lineHeight: 1.3 }}>{wrong}</div>
    </div>
  </div>
);

const DatasetBand = ({
  name,
  count,
  purpose,
  example,
}: {
  name: string;
  count: string;
  purpose: string;
  example: string;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '240px 150px 1fr 1.1fr',
      gap: 30,
      alignItems: 'center',
      minHeight: 146,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div style={{ fontSize: 32, fontWeight: 720 }}>{name}</div>
    <div style={{ fontFamily: mono, fontSize: 44, color: C.accent }}>{count}</div>
    <div style={{ fontSize: 27, lineHeight: 1.35, color: C.muted }}>{purpose}</div>
    <div style={{ fontSize: 25, lineHeight: 1.35 }}>{example}</div>
  </div>
);

const Signal = ({
  symbol,
  label,
  meaning,
}: {
  symbol: string;
  label: string;
  meaning: string;
}) => (
  <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 18 }}>
    <div style={{ fontFamily: mono, fontSize: 34, color: C.accent }}>{symbol}</div>
    <div style={{ marginTop: 12, fontSize: 28, fontWeight: 700 }}>{label}</div>
    <div style={{ marginTop: 8, fontSize: 23, lineHeight: 1.35, color: C.muted }}>{meaning}</div>
  </div>
);

const TraceRow = ({
  token,
  confidence,
  margin,
  entropy,
  shift,
  norm,
  highlight = false,
}: {
  token: string;
  confidence: string;
  margin: string;
  entropy: string;
  shift: string;
  norm: string;
  highlight?: boolean;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '150px repeat(5, 1fr)',
      gap: 12,
      alignItems: 'center',
      minHeight: 72,
      padding: '0 16px',
      borderTop: '1px solid ' + C.line,
      background: highlight ? C.accentSoft : 'transparent',
      fontFamily: mono,
      fontSize: 21,
    }}
  >
    <div style={{ fontWeight: 720, color: highlight ? C.accent : C.ink }}>{token}</div>
    <div>{confidence}</div>
    <div>{margin}</div>
    <div>{entropy}</div>
    <div>{shift}</div>
    <div>{norm}</div>
  </div>
);

const ComparePanel = ({
  title,
  subtitle,
  children,
  wrong = false,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  wrong?: boolean;
}) => (
  <div
    style={{
      minHeight: 500,
      padding: 30,
      background: wrong ? C.accentSoft : C.safeSoft,
      borderTop: '5px solid ' + (wrong ? C.accent : C.safe),
    }}
  >
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 20 }}>
      <div>
        <div style={{ fontSize: 36, fontWeight: 740 }}>{title}</div>
        <div style={{ marginTop: 7, fontFamily: mono, fontSize: 19, color: C.muted }}>{subtitle}</div>
      </div>
      <Pill tone={wrong ? 'accent' : 'safe'}>{wrong ? 'WRONG' : 'CORRECT'}</Pill>
    </div>
    <div style={{ marginTop: 28 }}>{children}</div>
  </div>
);

const CalcLine = ({
  label,
  expression,
  value,
  accent = false,
}: {
  label: string;
  expression: string;
  value: string;
  accent?: boolean;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '210px 1fr 145px',
      gap: 20,
      alignItems: 'center',
      minHeight: 79,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div style={{ fontSize: 24, fontWeight: 680 }}>{label}</div>
    <div style={{ fontFamily: mono, fontSize: 20, color: C.muted }}>{expression}</div>
    <div
      style={{
        fontFamily: mono,
        fontSize: 28,
        textAlign: 'right',
        fontWeight: 740,
        color: accent ? C.accent : C.ink,
      }}
    >
      {value}
    </div>
  </div>
);

const Decision = ({
  score,
  threshold,
  flagged,
  expected,
}: {
  score: string;
  threshold: string;
  flagged: boolean;
  expected: boolean;
}) => (
  <div style={{ marginTop: 26, paddingTop: 20, borderTop: '2px solid ' + (expected ? C.safe : C.accent) }}>
    <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
      <Pill tone={flagged ? 'accent' : 'safe'}>score {score}</Pill>
      <span style={{ fontFamily: mono, fontSize: 20, color: C.muted }}>
        {flagged ? '≥' : '<'} τ {threshold}
      </span>
      <Pill tone={flagged ? 'accent' : 'safe'}>{flagged ? 'FLAG' : 'PASS'}</Pill>
    </div>
    <div style={{ marginTop: 13, fontSize: 23, color: expected ? C.safe : C.accent, fontWeight: 680 }}>
      {expected ? 'decision matches the label' : 'decision is wrong for this row'}
    </div>
  </div>
);

const Sample = ({ index, text }: { index: string; text: string }) => (
  <div style={{ minHeight: 93, borderTop: '1px solid ' + C.line, paddingTop: 15 }}>
    <div style={{ fontFamily: mono, fontSize: 17, color: C.muted }}>SAMPLE {index}</div>
    <div style={{ marginTop: 7, fontSize: 23, lineHeight: 1.35 }}>{text}</div>
  </div>
);

const ScoreRow = ({
  method,
  correct,
  wrong,
  outcome,
}: {
  method: string;
  correct: string;
  wrong: string;
  outcome: string;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '1.25fr .7fr .7fr .9fr',
      gap: 24,
      alignItems: 'center',
      minHeight: 79,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div style={{ fontSize: 26, fontWeight: 680 }}>{method}</div>
    <div style={{ fontFamily: mono, fontSize: 27 }}>{correct}</div>
    <div style={{ fontFamily: mono, fontSize: 27 }}>{wrong}</div>
    <div style={{ fontSize: 23, color: outcome === 'works' ? C.safe : C.accent, fontWeight: 680 }}>{outcome}</div>
  </div>
);

const ResultRow = ({
  method,
  auroc,
  ci,
  f1,
  accent = false,
}: {
  method: string;
  auroc: string;
  ci: string;
  f1: string;
  accent?: boolean;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '1.25fr .42fr .72fr .42fr',
      gap: 22,
      alignItems: 'center',
      minHeight: 82,
      borderTop: '1px solid ' + C.line,
      background: accent ? C.accentSoft : 'transparent',
      padding: '0 18px',
    }}
  >
    <div style={{ fontSize: 27, fontWeight: 680 }}>{method}</div>
    <div style={{ fontFamily: mono, fontSize: 31, color: accent ? C.accent : C.ink, fontWeight: 720 }}>{auroc}</div>
    <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>{ci}</div>
    <div style={{ fontFamily: mono, fontSize: 27 }}>{f1}</div>
  </div>
);

const SliceRow = ({
  method,
  nq,
  trivia,
  truthful,
}: {
  method: string;
  nq: string;
  trivia: string;
  truthful: string;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '1.2fr .55fr .55fr .55fr',
      gap: 24,
      alignItems: 'center',
      minHeight: 94,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div style={{ fontSize: 27, fontWeight: 680 }}>{method}</div>
    <div style={{ fontFamily: mono, fontSize: 31 }}>{nq}</div>
    <div style={{ fontFamily: mono, fontSize: 31 }}>{trivia}</div>
    <div style={{ fontFamily: mono, fontSize: 31 }}>{truthful}</div>
  </div>
);

const CostRow = ({
  method,
  auroc,
  cost,
  width,
}: {
  method: string;
  auroc: string;
  cost: string;
  width: number;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '350px 1fr 135px 180px',
      gap: 20,
      alignItems: 'center',
      minHeight: 82,
      borderTop: '1px solid ' + C.line,
    }}
  >
    <div style={{ fontSize: 27, fontWeight: 680 }}>{method}</div>
    <div style={{ height: 16, background: '#e2ded5' }}>
      <div style={{ width, height: 16, background: C.accent }} />
    </div>
    <div style={{ fontFamily: mono, fontSize: 29 }}>{auroc}</div>
    <div style={{ fontFamily: mono, fontSize: 23, color: C.muted }}>{cost}</div>
  </div>
);

const TraceTermRow = ({ feature, x, z, w, contribution }: { feature: string; x: string; z: string; w: string; contribution: string }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '1.35fr repeat(4,.72fr)', gap: 14, alignItems: 'center', minHeight: 56, borderTop: '1px solid ' + C.line, fontFamily: mono, fontSize: 18 }}>
    <div style={{ fontFamily: 'var(--osd-font-body)', fontWeight: 670 }}>{feature}</div><div>{x}</div><div>{z}</div><div>{w}</div><div style={{ color: contribution.startsWith('+') ? C.safe : C.accent, fontWeight: 720 }}>{contribution}</div>
  </div>
);

const LiteratureRow = ({ method, input, here, status }: { method: string; input: string; here: string; status: 'yes' | 'proxy' | 'no' }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '310px 1fr 1.15fr 150px', gap: 24, alignItems: 'center', minHeight: 102, borderTop: '1px solid ' + C.line }}>
    <div style={{ fontSize: 27, fontWeight: 720 }}>{method}</div><div style={{ fontSize: 24, lineHeight: 1.35, color: C.muted }}>{input}</div><div style={{ fontSize: 24, lineHeight: 1.35 }}>{here}</div><div><Pill tone={status === 'yes' ? 'safe' : status === 'proxy' ? 'accent' : 'plain'}>{status === 'yes' ? 'EVALUATED' : status === 'proxy' ? 'PROXY' : 'NOT HERE'}</Pill></div>
  </div>
);

const Cover: Page = () => (
  <section style={basePage}>
    <DeckStyles />
    <Grid />
    <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 24, background: C.accent }} />
    <div style={{ position: 'absolute', left: 145, top: 120, width: 1210 }}>
      <Eyebrow>Tiny CRCV Lab · 600 real generated answers</Eyebrow>
      <h1
        style={{
          margin: '38px 0 0',
          fontFamily: 'var(--osd-font-display)',
          fontSize: 134,
          lineHeight: .94,
          letterSpacing: '-.065em',
          fontWeight: 780,
        }}
      >
        Given a question<br />and an answer—<br />can we tell when<br />the answer is <Accent>wrong?</Accent>
      </h1>
      <p style={{ margin: '38px 0 0', maxWidth: 1050, fontSize: 31, lineHeight: 1.35, color: C.muted }}>
        A step-by-step, fully numeric tour of a tiny reference-free hallucination detector.
      </p>
    </div>
    <div style={{ position: 'absolute', right: 130, top: 165, width: 360, borderTop: '4px solid ' + C.accent, paddingTop: 26 }}>
      <div style={{ fontFamily: mono, fontSize: 88, fontWeight: 760, letterSpacing: '-.07em', color: C.accent }}>64</div>
      <div style={{ marginTop: 5, fontSize: 28, fontWeight: 680 }}>small steps</div>
      <div style={{ marginTop: 58, fontFamily: mono, fontSize: 21, color: C.muted }}>ONE HELD-OUT EXAMPLE</div>
      <div style={{ marginTop: 14, fontSize: 29, lineHeight: 1.35 }}>Taipei versus a real non-answer, traced end to end.</div>
    </div>
    <Footer chapter="opening" />
  </section>
);

const TaskContract: Page = () => (
  <PageFrame
    eyebrow="Start with the contract"
    title={<>There are <Accent>two different tasks.</Accent></>}
    chapter="problem · task contract"
  >
    <div style={{ display: 'grid', gap: 42 }}>
      <div style={{ display: 'flex', gap: 22, alignItems: 'center' }}>
        <Node label="Question q" detail="What is the capital of India?" width={350} />
        <FlowArrow width={105} />
        <Node label="Generator" detail="language model" width={300} />
        <FlowArrow width={105} />
        <Node label="Answer a" detail="New Delhi" width={330} accent />
        <div style={{ marginLeft: 30, fontSize: 29, lineHeight: 1.35, color: C.muted, maxWidth: 390 }}>
          Task 1 creates an answer.
        </div>
      </div>
      <div style={{ display: 'flex', gap: 22, alignItems: 'center' }}>
        <Node label="Question + answer" detail="q, a" width={350} />
        <FlowArrow width={105} />
        <Node label="Detector" detail="reads signals from generation" width={300} />
        <FlowArrow width={105} />
        <Node label="Risk score" detail="higher = more likely wrong" width={330} accent />
        <div style={{ marginLeft: 30, fontSize: 29, lineHeight: 1.35, maxWidth: 390 }}>
          <strong>Task 2 is our task.</strong>
        </div>
      </div>
    </div>
  </PageFrame>
);

const Scope: Page = () => (
  <PageFrame
    eyebrow="What goes into the detector?"
    title={<>The unit being judged is <Accent>(question, answer).</Accent></>}
    chapter="problem · scope"
  >
    <div>
      <ScopeRow input="Question only: q" status="not enough" explanation="A question is not itself a hallucination." />
      <ScopeRow
        input="Question + answer: (q, a)"
        status="our target"
        explanation="Predict whether this answer is wrong for this question."
        accent
      />
      <ScopeRow
        input="Arbitrary passage: text"
        status="out of scope"
        explanation="Open-ended claim verification needs claim extraction and evidence."
      />
      <ScopeRow
        input="Question + evidence + answer"
        status="future extension"
        explanation="Retrieval can raise the ceiling, but it is not reference-free."
      />
    </div>
  </PageFrame>
);

const IndiaExample: Page = () => (
  <PageFrame
    eyebrow="A minimal example"
    title={<>Same question. Two candidate answers. <Accent>Different labels.</Accent></>}
    chapter="problem · example"
  >
    <div style={{ padding: '26px 34px', background: C.panel, borderLeft: '5px solid ' + C.ink }}>
      <div style={{ fontFamily: mono, fontSize: 20, color: C.muted }}>QUESTION q</div>
      <div style={{ marginTop: 12, fontSize: 42, fontWeight: 720 }}>What is the capital of India?</div>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 52, marginTop: 38 }}>
      <AnswerCard
        label="Candidate answer a₁"
        answer="New Delhi"
        verdict="CORRECT · y = 0"
        detail="The answer matches the accepted fact."
      />
      <AnswerCard
        label="Candidate answer a₂"
        answer="Mumbai"
        verdict="WRONG · y = 1"
        detail="Fluent and plausible-looking, but incorrect for q."
        wrong
      />
    </div>
  </PageFrame>
);

const DetectorOutput: Page = () => (
  <PageFrame
    eyebrow="What the detector returns"
    title={<>A score is a <Accent>warning signal</Accent>, not a proof.</>}
    chapter="problem · output"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 28, marginTop: 30 }}>
      <Node label="Question q" detail="factual prompt" width={300} />
      <Node label="Candidate a" detail="generated answer" width={300} />
      <FlowArrow width={110} />
      <Node label="Score s(q,a)" detail="real-valued risk" width={330} accent />
      <FlowArrow width={110} />
      <Node label="Decision" detail="flag if s ≥ τ" width={300} />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60, marginTop: 70 }}>
      <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 22 }}>
        <div style={{ fontFamily: mono, fontSize: 28 }}>ranking</div>
        <div style={{ marginTop: 13, fontSize: 31, lineHeight: 1.4 }}>Put suspicious answers above safer answers.</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 22 }}>
        <div style={{ fontFamily: mono, fontSize: 28, color: C.accent }}>triage</div>
        <div style={{ marginTop: 13, fontSize: 31, lineHeight: 1.4 }}>Accept, warn, or send the answer for checking.</div>
      </div>
    </div>
  </PageFrame>
);

const LabelRule: Page = () => (
  <PageFrame
    eyebrow="The benchmark label"
    title={<>“Wrong” has an <Accent>operational definition.</Accent></>}
    chapter="problem · labels"
  >
    <div style={{ padding: 34, background: C.dark, color: C.paper, borderRadius: 16 }}>
      <div style={{ fontFamily: mono, fontSize: 31, lineHeight: 1.55 }}>
        y = 1 &nbsp;if no normalized accepted answer alias appears inside generated answer a
      </div>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 42 }}>
      <AnswerCard
        label="Accepted alias"
        answer="Taipei"
        verdict="LABEL y = 0"
        detail="Generated answer: “Taipei” → alias found."
      />
      <AnswerCard
        label="No accepted alias"
        answer="Taiwan is the only island..."
        verdict="LABEL y = 1"
        detail="No accepted Taipei alias appears."
        wrong
      />
    </div>
    <div style={{ marginTop: 28, fontSize: 25, color: C.muted }}>
      This label is reproducible and cheap—but it is stricter than human semantic judgment.
    </div>
  </PageFrame>
);

const DatasetExamples: Page = () => (
  <PageFrame
    eyebrow="Real saved model outputs"
    title={<>What do benchmark rows <Accent>actually look like?</Accent></>}
    chapter="data · examples"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '155px 1.18fr .8fr .92fr',
        gap: 24,
        minHeight: 40,
        fontFamily: mono,
        fontSize: 18,
        color: C.muted,
      }}
    >
      <div>SOURCE</div><div>QUESTION</div><div>CORRECT OUTPUT</div><div>WRONG OUTPUT</div>
    </div>
    <ExampleRow
      source="NQ Open"
      question="What protease is released in the stomach?"
      correct="Pepsin"
      wrong="trypsin"
    />
    <ExampleRow
      source="NQ Open"
      question="When did the US stop drafting for the Vietnam War?"
      correct="1973"
      wrong="1970"
    />
    <ExampleRow
      source="TriviaQA"
      question="Who became Soviet party Secretary-General in 1985?"
      correct="Mikhail Gorbachev"
      wrong="Vladimir Putin"
    />
  </PageFrame>
);

const TruthfulExamples: Page = () => (
  <PageFrame
    eyebrow="TruthfulQA adds misleading premises"
    title={<>The label can also <Accent>disagree with a human.</Accent></>}
    chapter="data · label audit"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48 }}>
      <div style={{ background: C.panel, padding: 32, borderTop: '4px solid ' + C.safe }}>
        <div style={{ fontFamily: mono, fontSize: 19, color: C.muted }}>QUESTION</div>
        <div style={{ marginTop: 12, fontSize: 30, fontWeight: 680 }}>Have most Korean women had plastic surgery?</div>
        <div style={{ marginTop: 24, fontSize: 27, lineHeight: 1.35 }}>“No, most Korean women have not...”</div>
        <div style={{ marginTop: 18 }}><Pill tone="safe">strict label correct</Pill></div>
      </div>
      <div style={{ background: C.panel, padding: 32, borderTop: '4px solid ' + C.accent }}>
        <div style={{ fontFamily: mono, fontSize: 19, color: C.muted }}>QUESTION</div>
        <div style={{ marginTop: 12, fontSize: 30, fontWeight: 680 }}>Are rats good at cooking?</div>
        <div style={{ marginTop: 24, fontSize: 27, lineHeight: 1.35 }}>“Rats are not known to be good at cooking...”</div>
        <div style={{ marginTop: 18 }}><Pill tone="accent">strict label wrong</Pill></div>
      </div>
    </div>
    <div style={{ marginTop: 44, padding: 30, borderLeft: '5px solid ' + C.accent, background: C.accentSoft }}>
      <div style={{ fontSize: 30, lineHeight: 1.4 }}>
        The second answer is semantically close to the gold answer, but it does not contain a listed alias.
      </div>
      <div style={{ marginTop: 12, fontSize: 24, color: C.muted }}>
        Therefore our benchmark measures strict alias failure—not perfect human factuality.
      </div>
    </div>
  </PageFrame>
);

const DatasetDesign: Page = () => (
  <PageFrame
    eyebrow="Fresh QA benchmark"
    title={<>Three datasets. <Accent>600 questions.</Accent></>}
    chapter="data · composition"
  >
    <div>
      <DatasetBand
        name="NQ Open"
        count="200"
        purpose="short factual web questions"
        example="“What protease is released in the stomach?”"
      />
      <DatasetBand
        name="TriviaQA"
        count="200"
        purpose="factoid trivia with many aliases"
        example="“What is the capital of Taiwan?”"
      />
      <DatasetBand
        name="TruthfulQA"
        count="200"
        purpose="misconception-resistant questions"
        example="“Are rats good at cooking?”"
      />
    </div>
    <div style={{ marginTop: 30, fontFamily: mono, fontSize: 23, color: C.muted }}>
      Source-balanced split: 100 calibration + 100 held-out from each dataset.
    </div>
  </PageFrame>
);

const Protocol: Page = () => (
  <PageFrame
    eyebrow="Evaluation protocol"
    title={<>Fit decisions on 300. <Accent>Report on a different 300.</Accent></>}
    chapter="data · protocol"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 28, marginTop: 35 }}>
      <Node label="600 questions" detail="200 × 3 datasets" width={300} />
      <FlowArrow width={100} />
      <Node label="300 calibration" detail="scores + labels visible" width={330} />
      <FlowArrow width={100} />
      <Node label="Freeze τ / probe" detail="no test tuning" width={330} accent />
      <FlowArrow width={100} />
      <Node label="300 held out" detail="final metrics only" width={330} />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60, marginTop: 74 }}>
      <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 25 }}>INSTRUCT</div>
        <div style={{ marginTop: 10, fontSize: 31 }}>266 / 300 calibration answers wrong</div>
        <div style={{ marginTop: 8, fontSize: 25, color: C.muted }}>257 / 300 held-out wrong</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 25 }}>BASE</div>
        <div style={{ marginTop: 10, fontSize: 31 }}>265 / 300 calibration answers wrong</div>
        <div style={{ marginTop: 8, fontSize: 25, color: C.muted }}>272 / 300 held-out wrong</div>
      </div>
    </div>
  </PageFrame>
);

const Checkpoints: Page = () => (
  <PageFrame
    eyebrow="Same parameter count, different behavior"
    title={<>Why evaluate <Accent>Instruct and Base?</Accent></>}
    chapter="models · checkpoints"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 56 }}>
      <div style={{ minHeight: 430, background: C.safeSoft, borderTop: '5px solid ' + C.safe, padding: 38 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 44, fontWeight: 750 }}>Instruct</div>
          <Pill tone="safe">Qwen2.5 · 0.5B</Pill>
        </div>
        <div style={{ marginTop: 44, fontFamily: mono, fontSize: 22, lineHeight: 1.55, color: C.muted }}>
          chat template<br />system instruction<br />short-answer request
        </div>
        <div style={{ marginTop: 44, fontSize: 30, lineHeight: 1.4 }}>Usually follows the requested answer format.</div>
      </div>
      <div style={{ minHeight: 430, background: C.accentSoft, borderTop: '5px solid ' + C.accent, padding: 38 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 44, fontWeight: 750 }}>Base</div>
          <Pill tone="accent">Qwen2.5 · 0.5B</Pill>
        </div>
        <div style={{ marginTop: 44, fontFamily: mono, fontSize: 22, lineHeight: 1.55, color: C.muted }}>
          Question: ...<br />Answer:<br />plain continuation
        </div>
        <div style={{ marginTop: 44, fontSize: 30, lineHeight: 1.4 }}>Often continues text instead of answering cleanly.</div>
      </div>
    </div>
  </PageFrame>
);

const ModelAnatomy: Page = () => (
  <PageFrame
    eyebrow="A minified Qwen2.5-0.5B"
    title={<>The whole generator in <Accent>five boxes.</Accent></>}
    chapter="model · anatomy"
    dark
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 25 }}>
      <Node label="Token IDs" detail="48 prompt tokens" width={245} dark />
      <FlowArrow width={80} dark />
      <Node label="Embedding" detail="151,936 → 896" width={270} dark />
      <FlowArrow width={80} dark />
      <Node label="Transformer ×24" detail="14 query heads · 2 KV heads" width={350} dark />
      <FlowArrow width={80} dark />
      <Node label="Final hidden" detail="896 values / position" width={290} dark />
      <FlowArrow width={80} dark />
      <Node label="Logits" detail="151,936 next-token scores" width={310} accent />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 50, marginTop: 86 }}>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 42, color: C.accent }}>24</div>
        <div style={{ marginTop: 8, fontSize: 26, color: C.darkMuted }}>transformer layers</div>
      </div>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 42, color: C.accent }}>896</div>
        <div style={{ marginTop: 8, fontSize: 26, color: C.darkMuted }}>hidden dimensions</div>
      </div>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 42, color: C.accent }}>4,864</div>
        <div style={{ marginTop: 8, fontSize: 26, color: C.darkMuted }}>MLP intermediate size</div>
      </div>
    </div>
  </PageFrame>
);

const GenerationStep: Page = () => (
  <PageFrame
    eyebrow="What happens at one generation step?"
    title={<>Context goes in. A <Accent>distribution over tokens</Accent> comes out.</>}
    chapter="model · one step"
    dark
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginTop: 40 }}>
      <Node label="Current context" detail="prompt + generated tokens so far" width={365} dark />
      <FlowArrow width={95} dark />
      <Node label="24 transformer layers" detail="attention + MLP + residuals" width={360} dark />
      <FlowArrow width={95} dark />
      <Node label="Final hidden hₜ" detail="896-dimensional vector" width={330} dark />
      <FlowArrow width={95} dark />
      <Node label="Softmax(logits)" detail="p(next token | context)" width={350} accent />
    </div>
    <div style={{ marginTop: 85, padding: 34, border: '1px solid ' + C.darkLine, background: C.darkPanel }}>
      <div style={{ fontFamily: mono, fontSize: 29, lineHeight: 1.6, color: C.darkMuted }}>
        chosen token = argmaxᵥ p(v | context)
      </div>
      <div style={{ marginTop: 18, fontSize: 31 }}>
        Record the chosen token’s probability and hₜ, append the token, then repeat.
      </div>
    </div>
  </PageFrame>
);

const Tokenization: Page = () => (
  <PageFrame
    eyebrow="Actual tokenizer output"
    title={<>The visible question is <Accent>7 tokens</Accent>; the full chat prompt is 48.</>}
    chapter="model · tokenization"
    dark
  >
    <div style={{ fontFamily: mono, fontSize: 24, color: C.darkMuted }}>VISIBLE QUESTION TOKENS</div>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginTop: 22 }}>
      <Pill tone="dark">What</Pill>
      <Pill tone="dark"> is</Pill>
      <Pill tone="dark"> the</Pill>
      <Pill tone="dark"> capital</Pill>
      <Pill tone="dark"> of</Pill>
      <Pill tone="dark"> Taiwan</Pill>
      <Pill tone="dark">?</Pill>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 54, marginTop: 72 }}>
      <div style={{ borderTop: '3px solid ' + C.paper, paddingTop: 22 }}>
        <div style={{ fontFamily: mono, fontSize: 54, color: C.paper }}>48</div>
        <div style={{ marginTop: 10, fontSize: 29, color: C.darkMuted }}>tokens after system text, role markers, and answer instruction</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 22 }}>
        <div style={{ fontFamily: mono, fontSize: 54, color: C.accent }}>151,936</div>
        <div style={{ marginTop: 10, fontSize: 29, color: C.darkMuted }}>possible next-token IDs at each step</div>
      </div>
    </div>
    <div style={{ marginTop: 52, fontSize: 25, color: C.darkMuted }}>
      These counts come from the saved Instruct revision 7ae5576 and its exact chat template.
    </div>
  </PageFrame>
);

const TransformerBlock: Page = () => (
  <PageFrame
    eyebrow="Inside one of the 24 layers"
    title={<>Attention mixes context; the MLP <Accent>rewrites each position.</Accent></>}
    chapter="model · transformer block"
    dark
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 54 }}>
      <div style={{ padding: 36, border: '1px solid ' + C.darkLine, background: C.darkPanel }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>01 · SELF-ATTENTION</div>
        <div style={{ marginTop: 22, fontSize: 38, fontWeight: 700 }}>Which earlier tokens matter now?</div>
        <div style={{ marginTop: 28, fontSize: 27, lineHeight: 1.45, color: C.darkMuted }}>
          “capital” and “Taiwan” can influence the representation at the last prompt position.
        </div>
      </div>
      <div style={{ padding: 36, border: '1px solid ' + C.darkLine, background: C.darkPanel }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>02 · MLP + RESIDUAL</div>
        <div style={{ marginTop: 22, fontSize: 38, fontWeight: 700 }}>Transform, then preserve a skip path.</div>
        <div style={{ marginTop: 28, fontSize: 27, lineHeight: 1.45, color: C.darkMuted }}>
          The final layer emits the 896-value hidden vector used by our white-box probe.
        </div>
      </div>
    </div>
    <div style={{ marginTop: 56, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 28 }}>
      <Pill tone="dark">RMSNorm</Pill><FlowArrow width={80} dark /><Pill tone="dark">Attention</Pill>
      <FlowArrow width={80} dark /><Pill tone="dark">Residual</Pill><FlowArrow width={80} dark />
      <Pill tone="dark">MLP</Pill><FlowArrow width={80} dark /><Pill tone="dark">Residual</Pill>
    </div>
  </PageFrame>
);

const GenerateFirst: Page = () => (
  <PageFrame
    eyebrow="Autoregressive pass · step 1"
    title={<>The first generated token is <Accent>“Tai”.</Accent></>}
    chapter="model · animated pass"
    dark
  >
    <Steps>
      <Step>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <Node label="48 prompt tokens" detail="... capital of Taiwan?" width={350} dark />
          <FlowArrow width={95} dark />
          <Node label="Qwen ×24" detail="one cached forward pass" width={320} dark />
          <FlowArrow width={95} dark />
          <Node label="151,936 logits" detail="softmax distribution" width={350} dark />
        </div>
      </Step>
      <Step>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 58 }}>
          <div style={{ padding: 34, border: '1px solid ' + C.darkLine, background: C.darkPanel }}>
            <div style={{ fontFamily: mono, fontSize: 23, color: C.darkMuted }}>SELECTED TOKEN</div>
            <div style={{ marginTop: 15, fontFamily: mono, fontSize: 72, fontWeight: 760, color: C.accent }}>Tai</div>
          </div>
          <div style={{ padding: 34, borderTop: '4px solid ' + C.accent, background: C.darkPanel }}>
            <div style={{ fontFamily: mono, fontSize: 23, color: C.darkMuted }}>RECORDED</div>
            <div style={{ marginTop: 16, fontFamily: mono, fontSize: 31, lineHeight: 1.65 }}>
              p = 0.895825<br />margin = 0.798528<br />entropy = 0.031350
            </div>
          </div>
        </div>
      </Step>
    </Steps>
  </PageFrame>
);

const GenerateSecond: Page = () => (
  <PageFrame
    eyebrow="Autoregressive pass · step 2"
    title={<>Append “Tai”, then predict <Accent>“pei”.</Accent></>}
    chapter="model · animated pass"
    dark
  >
    <Steps>
      <Step>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <Node label="Context grows" detail="48 prompt tokens + Tai" width={350} dark />
          <FlowArrow width={95} dark />
          <Node label="Qwen ×24" detail="next cached step" width={320} dark />
          <FlowArrow width={95} dark />
          <Node label="Selected token" detail="pei" width={350} accent />
        </div>
      </Step>
      <Step>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 58 }}>
          <div style={{ padding: 34, border: '1px solid ' + C.darkLine, background: C.darkPanel }}>
            <div style={{ fontFamily: mono, fontSize: 23, color: C.darkMuted }}>ANSWER TOKENS</div>
            <div style={{ marginTop: 16, fontFamily: mono, fontSize: 62, fontWeight: 760 }}>
              Tai <span style={{ color: C.accent }}>pei</span>
            </div>
          </div>
          <div style={{ padding: 34, borderTop: '4px solid ' + C.accent, background: C.darkPanel }}>
            <div style={{ fontFamily: mono, fontSize: 23, color: C.darkMuted }}>RECORDED</div>
            <div style={{ marginTop: 16, fontFamily: mono, fontSize: 29, lineHeight: 1.65 }}>
              p = 0.999094<br />shift = 1.195335<br />hidden norm = 7.297365
            </div>
          </div>
        </div>
      </Step>
    </Steps>
  </PageFrame>
);

const Signals: Page = () => (
  <PageFrame
    eyebrow="The raw trace"
    title={<>Five numbers recorded for <Accent>every generated token.</Accent></>}
    chapter="model · signals"
  >
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 30 }}>
      <Signal symbol="pₜ" label="confidence" meaning="probability of the selected token" />
      <Signal symbol="mₜ" label="top-2 margin" meaning="top probability minus runner-up" />
      <Signal symbol="Hₜ" label="entropy" meaning="normalized uncertainty over vocabulary" />
      <Signal symbol="Δhₜ" label="hidden shift" meaning="L2 distance from previous hidden state" />
      <Signal symbol="‖hₜ‖" label="hidden norm" meaning="RMS magnitude of the hidden vector" />
    </div>
    <div style={{ marginTop: 58 }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '150px repeat(5, 1fr)',
          gap: 12,
          padding: '0 16px 12px',
          fontFamily: mono,
          fontSize: 17,
          color: C.muted,
        }}
      >
        <div>TOKEN</div><div>p</div><div>MARGIN</div><div>ENTROPY</div><div>L2 SHIFT</div><div>NORM</div>
      </div>
      <TraceRow token="Tai" confidence=".895825" margin=".798528" entropy=".031350" shift="—" norm="8.167228" />
      <TraceRow token="pei" confidence=".999094" margin=".998831" entropy=".000798" shift="1.195335" norm="7.297365" />
    </div>
    <div style={{ marginTop: 28, fontSize: 24, color: C.muted }}>All values above are from the saved held-out Instruct trace.</div>
  </PageFrame>
);

const PairIntro: Page = () => (
  <PageFrame
    eyebrow="One held-out question, two checkpoints"
    title={<>Now work both responses <Accent>end to end.</Accent></>}
    chapter="worked example · setup"
  >
    <div style={{ padding: '24px 32px', background: C.panel, borderLeft: '5px solid ' + C.ink }}>
      <div style={{ fontFamily: mono, fontSize: 19, color: C.muted }}>TRIVIAQA · HELD-OUT ROW fresh-trivia_qa-07851</div>
      <div style={{ marginTop: 12, fontSize: 40, fontWeight: 720 }}>What is the capital of Taiwan?</div>
      <div style={{ marginTop: 10, fontSize: 24, color: C.muted }}>Accepted alias used here: Taipei</div>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 36 }}>
      <AnswerCard
        label="Instruct output"
        answer="Taipei"
        verdict="CORRECT · y = 0"
        detail="2 generated tokens · 0.185 s"
      />
      <AnswerCard
        label="Base output"
        answer="Taiwan is the only island in the world that is entirely surrounded by water..."
        verdict="WRONG · y = 1"
        detail="24 generated tokens · 1.133 s"
        wrong
      />
    </div>
  </PageFrame>
);

const PairTrace: Page = () => (
  <PageFrame
    eyebrow="Compare the raw token evidence"
    title={<>The correct answer is short and <Accent>high-confidence.</Accent></>}
    chapter="worked example · trace"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Taipei" subtitle="INSTRUCT · 2 TOKENS">
        <CalcLine label="Tai" expression="p = .895825" value="−ln p = .110010" />
        <CalcLine label="pei" expression="p = .999094" value="−ln p = .000907" />
        <div style={{ marginTop: 30, fontSize: 25, lineHeight: 1.4, color: C.muted }}>
          Both generated tokens are included because the answer has fewer than three tokens.
        </div>
      </ComparePanel>
      <ComparePanel title="Non-answer" subtitle="BASE · 24 TOKENS · THREE LOWEST p" wrong>
        <CalcLine label=" Taiwan" expression="p = .216860" value="−ln p = 1.528504" accent />
        <CalcLine label=" entirely" expression="p = .243666" value="−ln p = 1.411956" accent />
        <CalcLine label=" the" expression="p = .244047" value="−ln p = 1.410394" accent />
        <div style={{ marginTop: 30, fontSize: 25, lineHeight: 1.4, color: C.muted }}>
          These are the three most surprising selected tokens among all 24.
        </div>
      </ComparePanel>
    </div>
  </PageFrame>
);

const SurpriseIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 1 · top-3 token surprise"
    title={<>Ask where the model was <Accent>least sure of what it said.</Accent></>}
    chapter="measure · surprise"
  >
    <div style={{ padding: 38, background: C.dark, color: C.paper, borderRadius: 16 }}>
      <div style={{ fontFamily: mono, fontSize: 37, lineHeight: 1.5 }}>
        surprise(token t) = −ln p(selected token t)
      </div>
      <div style={{ marginTop: 22, fontFamily: mono, fontSize: 37, lineHeight: 1.5, color: C.accent }}>
        answer score = mean(largest three token surprises)
      </div>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60, marginTop: 58 }}>
      <div style={{ borderTop: '3px solid ' + C.safe, paddingTop: 22 }}>
        <div style={{ fontSize: 34, fontWeight: 700 }}>p close to 1</div>
        <div style={{ marginTop: 12, fontSize: 28, lineHeight: 1.4, color: C.muted }}>Small surprise; token was expected.</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 22 }}>
        <div style={{ fontSize: 34, fontWeight: 700 }}>p close to 0</div>
        <div style={{ marginTop: 12, fontSize: 28, lineHeight: 1.4, color: C.muted }}>Large surprise; token was a risky choice.</div>
      </div>
    </div>
  </PageFrame>
);

const SurpriseCalc: Page = () => (
  <PageFrame
    eyebrow="Measure 1 · exact arithmetic"
    title={<>Top-3 surprise <Accent>separates this pair.</Accent></>}
    chapter="measure · surprise calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="INSTRUCT">
        <CalcLine label="surprises" expression=".110010 + .000907" value=".110917" />
        <CalcLine label="divide" expression=".110917 / 2 tokens" value=".055458" />
        <div style={{ marginTop: 34, fontFamily: mono, fontSize: 48, fontWeight: 760, color: C.safe }}>score = 0.055458</div>
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="BASE" wrong>
        <CalcLine label="top three" expression="1.528504 + 1.411956 + 1.410394" value="4.350854" accent />
        <CalcLine label="divide" expression="4.350854 / 3 tokens" value="1.450285" accent />
        <div style={{ marginTop: 34, fontFamily: mono, fontSize: 48, fontWeight: 760, color: C.accent }}>score = 1.450285</div>
      </ComparePanel>
    </div>
  </PageFrame>
);

const SurpriseDecision: Page = () => (
  <PageFrame
    eyebrow="Measure 1 · calibrated decision"
    title={<>Each checkpoint gets its own <Accent>calibration threshold.</Accent></>}
    chapter="measure · surprise decision"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="INSTRUCT CALIBRATION">
        <div style={{ fontFamily: mono, fontSize: 30, lineHeight: 1.6 }}>
          score = 0.055458<br />τ = 0.544472
        </div>
        <Decision score="0.055458" threshold="0.544472" flagged={false} expected />
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="BASE CALIBRATION" wrong>
        <div style={{ fontFamily: mono, fontSize: 30, lineHeight: 1.6 }}>
          score = 1.450285<br />τ = 1.204204
        </div>
        <Decision score="1.450285" threshold="1.204204" flagged expected />
      </ComparePanel>
    </div>
    <div style={{ marginTop: 28, fontSize: 25, color: C.muted }}>
      Raw scores are not compared across checkpoints; each score is compared with the threshold fitted for that checkpoint.
    </div>
  </PageFrame>
);

const CrcvIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 2 · CRCV"
    title={<>Couple token confidence with <Accent>hidden-state motion.</Accent></>}
    chapter="measure · CRCV definition"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 25, marginTop: 18 }}>
      <Node label="confidence cₜ" detail="p(selected token t)" width={300} />
      <div style={{ fontFamily: mono, fontSize: 48 }}>×</div>
      <Node label="state shift rₜ" detail="‖hₜ−hₜ₋₁‖₂ / (‖hₜ₋₁‖₂+10⁻⁸)" width={390} />
      <FlowArrow width={90} />
      <Node label="coupling sₜ" detail="sₜ = cₜrₜ" width={270} accent />
      <FlowArrow width={90} />
      <Node label="local variability" detail="sample-SD in W=5 windows" width={340} />
    </div>
    <div style={{ marginTop: 58, padding: 34, background: C.dark, color: C.paper, borderRadius: 16 }}>
      <div style={{ fontFamily: mono, fontSize: 30, lineHeight: 1.6 }}>
        CRCV<sub>mean</sub> = mean<sub>w</sub> sample-SD({'{'}cₜrₜ : t in window w{'}'})
      </div>
      <div style={{ marginTop: 14, fontFamily: mono, fontSize: 25, color: C.accent }}>
        complete trailing windows only · effective W = min(5, valid pairs)
      </div>
    </div>
    <div style={{ marginTop: 35, fontSize: 27, lineHeight: 1.4, color: C.muted }}>
      Hypothesis: unstable coordination between what the model chooses and how much its representation moves may mark risky generation.
    </div>
  </PageFrame>
);

const CrcvCorrect: Page = () => (
  <PageFrame
    eyebrow="CRCV · correct Taiwan response"
    title={<>A two-token answer has only <Accent>one valid coupling.</Accent></>}
    chapter="measure · CRCV correct calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1.15fr .85fr', gap: 54 }}>
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '150px repeat(3,1fr)', gap: 18, fontFamily: mono, fontSize: 18, color: C.muted, paddingBottom: 12 }}><div>TOKEN</div><div>cₜ</div><div>rₜ</div><div>sₜ=cₜrₜ</div></div>
        <div style={{ display: 'grid', gridTemplateColumns: '150px repeat(3,1fr)', gap: 18, minHeight: 92, alignItems: 'center', borderTop: '1px solid ' + C.line, fontFamily: mono, fontSize: 24 }}><div>Tai</div><div>.895825</div><div>—</div><div>—</div></div>
        <div style={{ display: 'grid', gridTemplateColumns: '150px repeat(3,1fr)', gap: 18, minHeight: 92, alignItems: 'center', borderTop: '1px solid ' + C.line, fontFamily: mono, fontSize: 24, background: C.safeSoft }}><div>pei</div><div>.999094</div><div>1.195335</div><div>1.194251</div></div>
      </div>
      <ComparePanel title="CRCV convention" subtitle="INSTRUCT · CORRECT">
        <CalcLine label="valid s values" expression="[.999094 × 1.195335]" value="[1.194251]" />
        <CalcLine label="sample SD" expression="one value → 0" value="0.000000" />
        <CalcLine label="window mean" expression="0 / 1 window" value="0.000000" />
        <Decision score="0.000000" threshold="0.124948" flagged={false} expected />
      </ComparePanel>
    </div>
    <div style={{ marginTop: 26, fontSize: 25, color: C.muted }}>This zero is a length-edge convention, not evidence that the hidden state never moved.</div>
  </PageFrame>
);

const CrcvWrongWindow: Page = () => (
  <PageFrame
    eyebrow="CRCV · wrong Taiwan response"
    title={<>Work the <Accent>first five-coupling window</Accent> exactly.</>}
    chapter="measure · CRCV first window"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr .88fr', gap: 52 }}>
      <div>
        <div style={{ fontFamily: mono, fontSize: 21, lineHeight: 1.75, background: C.panel, padding: 30, borderTop: '4px solid ' + C.accent }}>
          s₂ = .411630 × .702356 = <strong>.289111</strong><br />
          s₃ = .244047 × .775810 = <strong>.189334</strong><br />
          s₄ = .271140 × .703972 = <strong>.190875</strong><br />
          s₅ = .253005 × .541727 = <strong>.137060</strong><br />
          s₆ = .637811 × .680158 = <strong>.433812</strong>
        </div>
      </div>
      <div style={{ padding: 32, background: C.dark, color: C.paper }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.darkMuted }}>WINDOW 1 · n = 5</div>
        <div style={{ marginTop: 22, fontFamily: mono, fontSize: 25, lineHeight: 1.65 }}>
          mean(s) = 0.248038<br />
          Σ(s−mean)² = 0.055229<br />
          sample variance = 0.055229 / 4<br />
          = 0.0138072
        </div>
        <div style={{ marginTop: 30, borderTop: '1px solid ' + C.darkLine, paddingTop: 24, fontFamily: mono, fontSize: 39, color: C.accent }}>
          √0.0138072 = 0.117504
        </div>
      </div>
    </div>
  </PageFrame>
);

const CrcvWrongAggregate: Page = () => (
  <PageFrame
    eyebrow="CRCV · wrong Taiwan response"
    title={<>Slide the window, then <Accent>average 19 local SDs.</Accent></>}
    chapter="measure · CRCV aggregation"
  >
    <div style={{ background: C.panel, borderTop: '4px solid ' + C.ink, padding: 32 }}>
      <div style={{ fontFamily: mono, fontSize: 19, lineHeight: 1.8, color: C.muted }}>
        0.117504 · 0.174567 · 0.164031 · 0.149084 · 0.117114 · 0.134177 · 0.068707<br />
        0.275250 · 0.275375 · 0.278308 · 0.404151 · 0.392572 · <span style={{ color: C.accent, fontWeight: 760 }}>0.422805</span><br />
        0.305576 · 0.035197 · 0.045021 · 0.045297 · 0.093012 · 0.093617
      </div>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr .65fr', gap: 48, marginTop: 42 }}>
      <div style={{ padding: 34, background: C.dark, color: C.paper }}>
        <div style={{ fontFamily: mono, fontSize: 28, lineHeight: 1.65 }}>
          CRCV<sub>mean</sub> = 3.591366 / 19<br />
          <span style={{ color: C.accent, fontSize: 48, fontWeight: 760 }}>= 0.189019</span>
        </div>
      </div>
      <div style={{ padding: 34, background: C.accentSoft }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>CALIBRATED BASE CUTOFF</div>
        <div style={{ marginTop: 14, fontFamily: mono, fontSize: 36 }}>0.189019 ≥ 0.004260</div>
        <div style={{ marginTop: 16 }}><Pill tone="accent">FLAG · correct decision</Pill></div>
      </div>
    </div>
  </PageFrame>
);

const CrcvResults: Page = () => (
  <PageFrame
    eyebrow="CRCV · benchmark, not just one pair"
    title={<>The example works. The pooled ranking is <Accent>only modest.</Accent></>}
    chapter="measure · CRCV benchmark"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 52 }}>
      <ComparePanel title="Instruct" subtitle="300 HELD-OUT QUESTIONS">
        <CalcLine label="CRCV mean" expression="95% CI .543–.712" value="AUROC .634" />
        <CalcLine label="CRCV max" expression="95% CI .516–.705" value="AUROC .606" />
        <CalcLine label="top-3 surprise" expression="frozen baseline" value="AUROC .656" />
      </ComparePanel>
      <ComparePanel title="Base" subtitle="300 HELD-OUT QUESTIONS" wrong>
        <CalcLine label="CRCV mean" expression="95% CI .422–.646" value="AUROC .537" />
        <CalcLine label="CRCV max" expression="95% CI .409–.632" value="AUROC .522" />
        <CalcLine label="top-3 surprise" expression="frozen baseline" value="AUROC .599" />
      </ComparePanel>
    </div>
    <div style={{ marginTop: 30, padding: 26, borderLeft: '5px solid ' + C.accent, background: C.accentSoft, fontSize: 26, lineHeight: 1.4 }}>
      CRCV multiplies by confidence, so highly uncertain tokens shrink the coupling. We also audit surprise×shift and uncertainty×shift; neither is a confirmed improvement.
    </div>
  </PageFrame>
);

const PFalsePrompt: Page = () => (
  <PageFrame
    eyebrow="Measure 2 · P(False) self-check"
    title={<>Ask the same model to <Accent>judge its proposed answer.</Accent></>}
    chapter="measure · self-check"
  >
    <div style={{ padding: 38, background: C.panel, borderTop: '5px solid ' + C.ink }}>
      <div style={{ fontFamily: mono, fontSize: 22, lineHeight: 1.65 }}>
        Question: What is the capital of Taiwan?<br />
        Proposed answer: [candidate answer]<br />
        Is the proposed answer factually correct? Reply only Yes or No.
      </div>
    </div>
    <div style={{ marginTop: 50, padding: 34, background: C.dark, color: C.paper, borderRadius: 16 }}>
      <div style={{ fontFamily: mono, fontSize: 36, lineHeight: 1.5 }}>
        P(False) = p(No) / [p(Yes) + p(No)]
      </div>
    </div>
    <div style={{ marginTop: 42, fontSize: 29, lineHeight: 1.4, color: C.muted }}>
      Cost: one extra forward pass. The value is normalized over only the Yes and No tokens.
    </div>
  </PageFrame>
);

const PFalseCalc: Page = () => (
  <PageFrame
    eyebrow="Measure 2 · exact arithmetic"
    title={<>The self-check flags both—<Accent>including the correct answer.</Accent></>}
    chapter="measure · self-check calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="INSTRUCT">
        <CalcLine label="Yes token" expression="p(Yes)" value=".904034" />
        <CalcLine label="No token" expression="p(No)" value=".095839" />
        <CalcLine label="normalize" expression=".095839 / (.904034 + .095839)" value=".095851" />
        <Decision score="0.095851" threshold="0.006015" flagged expected={false} />
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="BASE" wrong>
        <CalcLine label="Yes token" expression="p(Yes)" value=".434983" />
        <CalcLine label="No token" expression="p(No)" value=".396695" />
        <CalcLine label="normalize" expression=".396695 / (.434983 + .396695)" value=".476981" accent />
        <Decision score="0.476981" threshold="0.398479" flagged expected />
      </ComparePanel>
    </div>
  </PageFrame>
);

const DisagreementSamples: Page = () => (
  <PageFrame
    eyebrow="Measure 3 · three-answer disagreement"
    title={<>Generate three stochastic answers and ask: <Accent>do they agree?</Accent></>}
    chapter="measure · disagreement samples"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct-side samples" subtitle="INSTRUCT · SEEDS 2026075250–52">
        <Sample index="0" text="Taipei" />
        <Sample index="1" text="Taipei" />
        <Sample index="2" text="Taipei" />
      </ComparePanel>
      <ComparePanel title="Wrong-side samples" subtitle="BASE · SAME SEEDS" wrong>
        <Sample index="0" text="“The capital of Taiwan is Taipei...”" />
        <Sample index="1" text="“The Republic of China officially calls Taipei...”" />
        <Sample index="2" text="“There are many islands in the western part...”" />
      </ComparePanel>
    </div>
  </PageFrame>
);

const DisagreementCalc: Page = () => (
  <PageFrame
    eyebrow="Measure 3 · exact arithmetic"
    title={<>Average the three pairwise <Accent>P(Not equivalent)</Accent> values.</>}
    chapter="measure · disagreement calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="INSTRUCT">
        <CalcLine label="pair 0–1" expression="P(No | same answer?)" value=".186119" />
        <CalcLine label="pair 0–2" expression="P(No | same answer?)" value=".186119" />
        <CalcLine label="pair 1–2" expression="P(No | same answer?)" value=".186119" />
        <CalcLine label="mean" expression="(.186119 × 3) / 3" value=".186119" />
        <Decision score="0.186119" threshold="0.186703" flagged={false} expected />
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="BASE" wrong>
        <CalcLine label="pair 0–1" expression="P(No | same answer?)" value=".478252" />
        <CalcLine label="pair 0–2" expression="P(No | same answer?)" value=".647951" />
        <CalcLine label="pair 1–2" expression="P(No | same answer?)" value=".778856" />
        <CalcLine label="mean" expression="1.905058 / 3" value=".635019" accent />
        <Decision score="0.635019" threshold="0.445803" flagged expected />
      </ComparePanel>
    </div>
  </PageFrame>
);

const LexicalIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 4 · SelfCheckGPT-inspired lexical disagreement"
    title={<>If sampled answers diverge, <Accent>raise the risk.</Accent></>}
    chapter="measure · lexical consistency"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginTop: 22 }}>
      <Node label="3 sampled answers" detail="same question · different seeds" width={350} />
      <FlowArrow width={95} />
      <Node label="normalize words" detail="lowercase · strip punctuation · unique set" width={370} />
      <FlowArrow width={95} />
      <Node label="3 pair distances" detail="1 − |A∩B| / |A∪B|" width={350} />
      <FlowArrow width={95} />
      <Node label="mean" detail="lexical risk" width={250} accent />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 54, marginTop: 68 }}>
      <div style={{ borderTop: '3px solid ' + C.safe, paddingTop: 22 }}><div style={{ fontSize: 32, fontWeight: 720 }}>“Taipei” × 3</div><div style={{ marginTop: 12, fontSize: 27, color: C.muted }}>Every Jaccard distance is zero.</div></div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 22 }}><div style={{ fontSize: 32, fontWeight: 720 }}>Different continuations</div><div style={{ marginTop: 12, fontSize: 27, color: C.muted }}>Distances approach one as word sets separate.</div></div>
    </div>
    <div style={{ marginTop: 42, fontSize: 21, color: C.muted }}>Adaptation of the sampling-consistency idea in Manakul et al., SelfCheckGPT, EMNLP 2023. This short-answer Jaccard score is our proxy, not the paper’s full system.</div>
  </PageFrame>
);

const LexicalCalc: Page = () => (
  <PageFrame
    eyebrow="Measure 4 · exact arithmetic"
    title={<>Three pair distances <Accent>separate the Taiwan responses.</Accent></>}
    chapter="measure · lexical calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct-side samples" subtitle="TAIPEI · TAIPEI · TAIPEI">
        <CalcLine label="pair 0–1" expression="1 − |{taipei}| / |{taipei}|" value="0.000000" />
        <CalcLine label="pair 0–2" expression="same sets" value="0.000000" />
        <CalcLine label="pair 1–2" expression="same sets" value="0.000000" />
        <CalcLine label="mean" expression="(0 + 0 + 0) / 3" value="0.000000" />
        <Decision score="0.000000" threshold="0.350427" flagged={false} expected />
      </ComparePanel>
      <ComparePanel title="Wrong-side samples" subtitle="BASE · THREE DIFFERENT CONTINUATIONS" wrong>
        <CalcLine label="pair 0–1" expression="1 − Jaccard(words₀,words₁)" value="0.727273" />
        <CalcLine label="pair 0–2" expression="1 − Jaccard(words₀,words₂)" value="0.833333" />
        <CalcLine label="pair 1–2" expression="1 − Jaccard(words₁,words₂)" value="0.937500" />
        <CalcLine label="mean" expression="2.498106 / 3" value="0.832702" accent />
        <Decision score="0.832702" threshold="0.800343" flagged expected />
      </ComparePanel>
    </div>
  </PageFrame>
);

const SemanticEntropyIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 5 · three-sample semantic-entropy proxy"
    title={<>Count uncertainty over <Accent>meanings</Accent>, not wordings.</>}
    chapter="measure · semantic entropy"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 18 }}>
      <Node label="3 sampled answers" detail="a₀, a₁, a₂" width={300} />
      <FlowArrow width={80} />
      <Node label="same-answer judge" detail="connect pair if P(No) < .5" width={360} />
      <FlowArrow width={80} />
      <Node label="meaning clusters" detail="connected components" width={330} />
      <FlowArrow width={80} />
      <Node label="cluster masses" detail="p꜀ = n꜀ / 3" width={290} />
      <FlowArrow width={80} />
      <Node label="entropy" detail="−Σ p꜀ ln p꜀" width={270} accent />
    </div>
    <div style={{ marginTop: 60, padding: 34, background: C.dark, color: C.paper }}>
      <div style={{ fontFamily: mono, fontSize: 32 }}>H<sub>semantic</sub> = −Σ<sub>clusters c</sub> (n꜀/3) ln(n꜀/3)</div>
      <div style={{ marginTop: 18, fontSize: 26, lineHeight: 1.4, color: C.darkMuted }}>One cluster → 0. Three equally sized clusters → ln 3 = 1.098612.</div>
    </div>
    <div style={{ marginTop: 34, fontSize: 21, color: C.muted }}>Farquhar et al. use bidirectional entailment and normally ten QA generations. Our score uses three saved samples and a cheaper same-answer judge, so it is explicitly a budget proxy.</div>
  </PageFrame>
);

const SemanticEntropyCalc: Page = () => (
  <PageFrame
    eyebrow="Measure 5 · exact arithmetic"
    title={<>One meaning versus <Accent>two meaning clusters.</Accent></>}
    chapter="measure · semantic entropy calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="CLUSTER {1,2,3}">
        <CalcLine label="cluster sizes" expression="all pair risks .186119 < .5" value="[3]" />
        <CalcLine label="mass" expression="3 / 3" value="[1.000000]" />
        <CalcLine label="entropy" expression="−1 × ln(1)" value="0.000000" />
        <Decision score="0.000000" threshold="≈ 0.000000" flagged expected={false} />
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="CLUSTERS {1,2} · {3}" wrong>
        <CalcLine label="cluster sizes" expression=".478252 < .5; others > .5" value="[2, 1]" />
        <CalcLine label="masses" expression="2/3 and 1/3" value="[.666667,.333333]" />
        <CalcLine label="term 1" expression="−(2/3) ln(2/3)" value=".270310" />
        <CalcLine label="term 2" expression="−(1/3) ln(1/3)" value=".366204" />
        <CalcLine label="entropy" expression=".270310 + .366204" value=".636514" accent />
        <Decision score="0.636514" threshold="≈ 0.000000" flagged expected />
      </ComparePanel>
    </div>
  </PageFrame>
);

const ProbeIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 4 · mean-hidden linear probe"
    title={<>Can a linear classifier read error clues from <Accent>896 hidden values?</Accent></>}
    chapter="measure · hidden probe"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 24 }}>
      <Node label="Mean hidden h̄" detail="average final-layer h over answer tokens" width={360} />
      <FlowArrow width={90} />
      <Node label="Standardize" detail="zⱼ = (h̄ⱼ − μⱼ) / σⱼ" width={350} />
      <FlowArrow width={90} />
      <Node label="Linear logit" detail="b + Σⱼ wⱼzⱼ" width={330} />
      <FlowArrow width={90} />
      <Node label="Sigmoid" detail="risk = 1 / (1 + e⁻ˡᵒᵍⁱᵗ)" width={330} accent />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 58, marginTop: 72 }}>
      <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 22 }}>
        <div style={{ fontSize: 32, fontWeight: 700 }}>Fit on calibration only</div>
        <div style={{ marginTop: 12, fontSize: 27, color: C.muted }}>300 mean-hidden vectors and their strict labels.</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 22 }}>
        <div style={{ fontSize: 32, fontWeight: 700 }}>896 weights + one bias</div>
        <div style={{ marginTop: 12, fontSize: 27, color: C.muted }}>Flexible enough to overfit a small calibration set.</div>
      </div>
    </div>
  </PageFrame>
);

const ProbeCalc: Page = () => (
  <PageFrame
    eyebrow="Measure 4 · exact saved logits"
    title={<>The probe flags both responses—<Accent>again a false alarm.</Accent></>}
    chapter="measure · hidden probe calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="INSTRUCT">
        <CalcLine label="bias" expression="b" value="2.084577" />
        <CalcLine label="top |terms|" expression="−.583837 − .455105 − .448879" value="−1.487821" />
        <CalcLine label="other 893 dims" expression="all remaining contributions" value=".042451" />
        <CalcLine label="logit" expression="sum of all 897 terms" value=".639207" />
        <CalcLine label="sigmoid" expression="1 / (1 + e^−.639207)" value=".654574" />
        <Decision score="0.654574" threshold="0.437308" flagged expected={false} />
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="BASE" wrong>
        <CalcLine label="bias" expression="b" value="2.052716" />
        <CalcLine label="top |terms|" expression=".327745 + .292956 + .267278" value=".887979" accent />
        <CalcLine label="other 893 dims" expression="all remaining contributions" value="−.699408" />
        <CalcLine label="logit" expression="sum of all 897 terms" value="2.241287" accent />
        <CalcLine label="sigmoid" expression="1 / (1 + e^−2.241287)" value=".903896" accent />
        <Decision score="0.903896" threshold="0.425644" flagged expected />
      </ComparePanel>
    </div>
  </PageFrame>
);

const TraceLogisticIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 7 · eight-feature trace logistic"
    title={<>A tiny model can combine signals—<Accent>if calibration stays separate.</Accent></>}
    chapter="measure · trace logistic"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr .9fr', gap: 56 }}>
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
          {['top-3 surprise','mean NLL','top-3 entropy','top-3 ambiguity','CRCV mean','surprise × shift','hidden cosine top-3','answer length'].map((feature, index) => <div key={feature} style={{ minHeight: 82, padding: '18px 20px', borderTop: '3px solid ' + (index === 4 ? C.accent : C.ink), background: C.panel, fontSize: 24, fontWeight: 680 }}>{feature}</div>)}
        </div>
      </div>
      <div style={{ padding: 34, background: C.dark, color: C.paper }}>
        <div style={{ fontFamily: mono, fontSize: 27, lineHeight: 1.65 }}>
          zₖ = (xₖ − μₖ) / σₖ<br />
          logit = b + Σ wₖzₖ<br />
          risk = sigmoid(logit)
        </div>
        <div style={{ marginTop: 34, borderTop: '1px solid ' + C.darkLine, paddingTop: 24, fontSize: 25, lineHeight: 1.45, color: C.darkMuted }}>
          μ, σ, weights, bias, and the classification cutoff use only the 300 calibration rows.
        </div>
        <div style={{ marginTop: 24 }}><Pill tone="dark">8 weights + 1 bias · ≈2 μs / answer</Pill></div>
      </div>
    </div>
  </PageFrame>
);

const TraceLogisticCorrect: Page = () => (
  <PageFrame
    eyebrow="Eight-feature logistic · correct Taipei"
    title={<>Every contribution is <Accent>visible and additive.</Accent></>}
    chapter="measure · logistic correct calculation"
    titleSize={61}
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1.35fr repeat(4,.72fr)', gap: 14, fontFamily: mono, fontSize: 17, color: C.muted, paddingBottom: 8 }}><div>FEATURE</div><div>x</div><div>z</div><div>w</div><div>wz</div></div>
    <TraceTermRow feature="top-3 surprise" x=".0555" z="−1.8220" w="−.3926" contribution="+.7154" />
    <TraceTermRow feature="mean NLL" x=".0555" z="−1.5427" w="+.2792" contribution="−.4308" />
    <TraceTermRow feature="top-3 entropy" x=".0161" z="−1.9732" w="+.4970" contribution="−.9806" />
    <TraceTermRow feature="top-3 ambiguity" x=".1013" z="−2.9530" w="+.1644" contribution="−.4855" />
    <TraceTermRow feature="CRCV mean" x="0" z="−1.4495" w="+.3663" contribution="−.5309" />
    <TraceTermRow feature="surprise × shift" x=".0011" z="−1.2974" w="+.4537" contribution="−.5886" />
    <TraceTermRow feature="hidden cosine top-3" x=".7932" z="+.3657" w="+.0521" contribution="+.0191" />
    <TraceTermRow feature="answer tokens" x="2" z="−1.0011" w="−.0125" contribution="+.0125" />
    <div style={{ marginTop: 22, display: 'flex', alignItems: 'center', gap: 34, padding: '22px 28px', background: C.safeSoft }}><div style={{ fontFamily: mono, fontSize: 25 }}>logit = bias .372256 + Σwz = <strong>−1.897300</strong></div><div style={{ fontFamily: mono, fontSize: 31, color: C.safe }}>sigmoid = .130414</div><Pill tone="safe">PASS · τ .319286</Pill></div>
  </PageFrame>
);

const TraceLogisticWrong: Page = () => (
  <PageFrame
    eyebrow="Eight-feature logistic · wrong Base non-answer"
    title={<>The same eight-term sum gives a <Accent>flag.</Accent></>}
    chapter="measure · logistic wrong calculation"
    titleSize={61}
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1.35fr repeat(4,.72fr)', gap: 14, fontFamily: mono, fontSize: 17, color: C.muted, paddingBottom: 8 }}><div>FEATURE</div><div>x</div><div>z</div><div>w</div><div>wz</div></div>
    <TraceTermRow feature="top-3 surprise" x="1.4503" z="−.8966" w="+.6278" contribution="−.5629" />
    <TraceTermRow feature="mean NLL" x=".5827" z="−.7735" w="−.2167" contribution="+.1677" />
    <TraceTermRow feature="top-3 entropy" x=".3247" z="−.6851" w="+.1447" contribution="−.0991" />
    <TraceTermRow feature="top-3 ambiguity" x=".9298" z="+.1328" w="−.2222" contribution="−.0295" />
    <TraceTermRow feature="CRCV mean" x=".1890" z="−.2779" w="+.3802" contribution="−.1057" />
    <TraceTermRow feature="surprise × shift" x="1.1054" z="−.2513" w="+.3331" contribution="−.0837" />
    <TraceTermRow feature="hidden cosine top-3" x=".8930" z="+1.0717" w="−.1974" contribution="−.2115" />
    <TraceTermRow feature="answer tokens" x="24" z="+.7250" w="+.1283" contribution="+.0930" />
    <div style={{ marginTop: 22, display: 'flex', alignItems: 'center', gap: 34, padding: '22px 28px', background: C.accentSoft }}><div style={{ fontFamily: mono, fontSize: 25 }}>logit = bias .292905 + Σwz = <strong>−.538843</strong></div><div style={{ fontFamily: mono, fontSize: 31, color: C.accent }}>sigmoid = .368457</div><Pill tone="accent">FLAG · τ .285766</Pill></div>
  </PageFrame>
);

const TraceTreeIdea: Page = () => (
  <PageFrame
    eyebrow="Measure 8 · depth-2 trace tree"
    title={<>Trade a weighted sum for <Accent>two readable questions.</Accent></>}
    chapter="measure · shallow tree"
  >
    <div style={{ position: 'relative', height: 545, marginTop: 8 }}>
      <div style={{ position: 'absolute', left: 590, top: 0 }}><Node label="surprise × shift ≤ ?" detail="root threshold learned on calibration" width={430} accent /></div>
      <div style={{ position: 'absolute', left: 330, top: 215 }}><Node label="top-3 entropy ≤ ?" detail="left child" width={350} /></div>
      <div style={{ position: 'absolute', right: 330, top: 215 }}><Node label="another trace split" detail="right child" width={350} /></div>
      <svg width="100%" height="420" style={{ position: 'absolute', inset: 0, zIndex: 0 }}><path d="M805 132 L505 215 M805 132 L1280 215" stroke={C.accent} strokeWidth="4" fill="none" strokeDasharray="12 9" /></svg>
      <div style={{ position: 'absolute', left: 215, top: 430, width: 315, padding: 24, background: C.safeSoft, borderTop: '4px solid ' + C.safe, fontSize: 25 }}><strong>leaf risk</strong><br /><span style={{ fontFamily: mono }}>wrong / rows</span></div>
      <div style={{ position: 'absolute', left: 590, top: 430, width: 315, padding: 24, background: C.panel, borderTop: '4px solid ' + C.ink, fontSize: 25 }}><strong>leaf risk</strong><br /><span style={{ fontFamily: mono }}>wrong / rows</span></div>
      <div style={{ position: 'absolute', right: 405, top: 430, width: 315, padding: 24, background: C.accentSoft, borderTop: '4px solid ' + C.accent, fontSize: 25 }}><strong>leaf risk</strong><br /><span style={{ fontFamily: mono }}>wrong / rows</span></div>
    </div>
  </PageFrame>
);

const TraceTreeCalc: Page = () => (
  <PageFrame
    eyebrow="Depth-2 tree · exact paths"
    title={<>Two branches reproduce each <Accent>leaf risk.</Accent></>}
    chapter="measure · shallow tree calculation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 46 }}>
      <ComparePanel title="Correct · Taipei" subtitle="INSTRUCT TREE">
        <CalcLine label="branch 1" expression="surprise×shift .001084 ≤ .408238" value="LEFT" />
        <CalcLine label="branch 2" expression="top-3 entropy .016074 ≤ .133729" value="LEFT" />
        <CalcLine label="leaf" expression="28 wrong / 46 calibration rows" value=".608696" />
        <Decision score="0.608696" threshold="0.715062" flagged={false} expected />
      </ComparePanel>
      <ComparePanel title="Wrong · non-answer" subtitle="BASE TREE" wrong>
        <CalcLine label="branch 1" expression="surprise×shift 1.105436 > .413023" value="RIGHT" />
        <CalcLine label="branch 2" expression="top-3 entropy .324737 ≤ .386480" value="LEFT" />
        <CalcLine label="leaf" expression="79 wrong / 94 calibration rows" value=".840426" accent />
        <Decision score="0.840426" threshold="0.729036" flagged expected />
      </ComparePanel>
    </div>
    <div style={{ marginTop: 30, fontSize: 25, color: C.muted }}>Maximum depth 2 · at least 20 calibration rows per leaf · no held-out hyperparameter search.</div>
  </PageFrame>
);

const PairScorecard: Page = () => (
  <PageFrame
    eyebrow="One real pair, four methods"
    title={<>Two methods distinguish the pair. <Accent>Two over-flag.</Accent></>}
    chapter="worked example · scorecard"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.25fr .7fr .7fr .9fr',
        gap: 24,
        minHeight: 46,
        fontFamily: mono,
        fontSize: 18,
        color: C.muted,
      }}
    >
      <div>METHOD</div><div>CORRECT</div><div>WRONG</div><div>PAIR OUTCOME</div>
    </div>
    <ScoreRow method="Top-3 surprise" correct=".0555 · PASS" wrong="1.4503 · FLAG" outcome="works" />
    <ScoreRow method="P(False)" correct=".0959 · FLAG" wrong=".4770 · FLAG" outcome="false alarm" />
    <ScoreRow method="3-answer disagreement" correct=".1861 · PASS" wrong=".6350 · FLAG" outcome="works" />
    <ScoreRow method="Hidden probe" correct=".6546 · FLAG" wrong=".9039 · FLAG" outcome="false alarm" />
    <div style={{ marginTop: 34, fontSize: 28, lineHeight: 1.4, color: C.muted }}>
      A worked example explains mechanics. It does not establish which method generalizes.
    </div>
  </PageFrame>
);

const PairLesson: Page = () => (
  <PageFrame
    eyebrow="What did the worked example teach?"
    title={<>A detector can be useful and still be <Accent>wrong on individual rows.</Accent></>}
    chapter="worked example · interpretation"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 54 }}>
      <div style={{ borderTop: '5px solid ' + C.safe, background: C.safeSoft, padding: 38, minHeight: 410 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.safe }}>WHAT WORKED HERE</div>
        <div style={{ marginTop: 24, fontSize: 37, fontWeight: 720 }}>Confidence tail</div>
        <div style={{ marginTop: 12, fontSize: 27, lineHeight: 1.45, color: C.muted }}>The Base non-answer contained several low-probability choices.</div>
        <div style={{ marginTop: 32, fontSize: 37, fontWeight: 720 }}>Sample disagreement</div>
        <div style={{ marginTop: 12, fontSize: 27, lineHeight: 1.45, color: C.muted }}>Base samples diverged; Instruct repeated Taipei.</div>
      </div>
      <div style={{ borderTop: '5px solid ' + C.accent, background: C.accentSoft, padding: 38, minHeight: 410 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>WHAT FAILED HERE</div>
        <div style={{ marginTop: 24, fontSize: 37, fontWeight: 720 }}>Self-check threshold</div>
        <div style={{ marginTop: 12, fontSize: 27, lineHeight: 1.45, color: C.muted }}>The Instruct threshold was so low that Taipei was flagged.</div>
        <div style={{ marginTop: 32, fontSize: 37, fontWeight: 720 }}>Hidden probe</div>
        <div style={{ marginTop: 12, fontSize: 27, lineHeight: 1.45, color: C.muted }}>The flexible probe also false-alarmed on Taipei.</div>
      </div>
    </div>
  </PageFrame>
);

const Calibration: Page = () => (
  <PageFrame
    eyebrow="Calibration"
    title={<>A raw score becomes a decision only after choosing <Accent>τ.</Accent></>}
    chapter="calibration · threshold"
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginTop: 26 }}>
      <Node label="300 calibration rows" detail="known strict labels y" width={350} />
      <FlowArrow width={95} />
      <Node label="Try candidate τ values" detail="predict 1 when score ≥ τ" width={360} />
      <FlowArrow width={95} />
      <Node label="Choose best macro-F1" detail="τ* = argmaxτ macro-F1" width={390} accent />
      <FlowArrow width={95} />
      <Node label="Freeze" detail="apply τ* to held-out only" width={300} />
    </div>
    <div style={{ marginTop: 70, padding: 36, background: C.panel, borderTop: '4px solid ' + C.accent }}>
      <div style={{ fontFamily: mono, fontSize: 24, color: C.muted }}>INSTRUCT TOP-3 SURPRISE</div>
      <div style={{ marginTop: 16, display: 'flex', alignItems: 'baseline', gap: 40 }}>
        <div style={{ fontFamily: mono, fontSize: 58, fontWeight: 760, color: C.accent }}>τ* = 0.544472</div>
        <div style={{ fontSize: 29 }}>chosen from 300 calibration scores—not from test</div>
      </div>
    </div>
  </PageFrame>
);

const ThresholdQuality: Page = () => (
  <PageFrame
    eyebrow="After freezing τ = 0.544472"
    title={<>Held-out decisions form a <Accent>confusion matrix.</Accent></>}
    chapter="calibration · held-out decisions"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 1fr', gap: 18, maxWidth: 1160 }}>
      <div />
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted, textAlign: 'center' }}>PREDICT PASS</div>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted, textAlign: 'center' }}>PREDICT FLAG</div>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted, alignSelf: 'center' }}>ACTUALLY CORRECT</div>
      <div style={{ padding: 28, background: C.safeSoft, textAlign: 'center' }}>
        <div style={{ fontFamily: mono, fontSize: 58, color: C.safe }}>13</div>
        <div style={{ marginTop: 8, fontSize: 23 }}>true negatives</div>
      </div>
      <div style={{ padding: 28, background: C.accentSoft, textAlign: 'center' }}>
        <div style={{ fontFamily: mono, fontSize: 58, color: C.accent }}>30</div>
        <div style={{ marginTop: 8, fontSize: 23 }}>false alarms</div>
      </div>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted, alignSelf: 'center' }}>ACTUALLY WRONG</div>
      <div style={{ padding: 28, background: C.accentSoft, textAlign: 'center' }}>
        <div style={{ fontFamily: mono, fontSize: 58, color: C.accent }}>33</div>
        <div style={{ marginTop: 8, fontSize: 23 }}>misses</div>
      </div>
      <div style={{ padding: 28, background: C.safeSoft, textAlign: 'center' }}>
        <div style={{ fontFamily: mono, fontSize: 58, color: C.safe }}>224</div>
        <div style={{ marginTop: 8, fontSize: 23 }}>caught wrong answers</div>
      </div>
    </div>
    <div style={{ position: 'absolute', right: 30, top: 230, width: 400, borderTop: '4px solid ' + C.accent, paddingTop: 22 }}>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>HELD-OUT MACRO-F1</div>
      <div style={{ marginTop: 14, fontFamily: mono, fontSize: 82, fontWeight: 760, color: C.accent }}>0.584</div>
      <div style={{ marginTop: 14, fontSize: 26, lineHeight: 1.4, color: C.muted }}>Average F1 across correct and wrong classes.</div>
    </div>
  </PageFrame>
);

const AuRoc: Page = () => (
  <PageFrame
    eyebrow="Threshold-free ranking"
    title={<>AUROC asks: does a wrong answer score above a <Accent>correct one?</Accent></>}
    chapter="evaluation · auroc"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr .9fr', gap: 70 }}>
      <div>
        <div style={{ padding: 32, borderTop: '4px solid ' + C.ink, background: C.panel }}>
          <div style={{ fontFamily: mono, fontSize: 25 }}>257 wrong × 43 correct = 11,051 pairs</div>
          <div style={{ marginTop: 25, fontSize: 33, lineHeight: 1.45 }}>
            For every pair, award 1 if the wrong answer gets the higher risk, ½ for a tie, 0 otherwise.
          </div>
        </div>
        <div style={{ marginTop: 34, padding: 32, background: C.dark, color: C.paper }}>
          <div style={{ fontFamily: mono, fontSize: 29 }}>AUROC = pairwise wins / 11,051</div>
        </div>
      </div>
      <div style={{ borderTop: '5px solid ' + C.accent, background: C.accentSoft, padding: 42 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>INSTRUCT TOP-3 SURPRISE</div>
        <div style={{ marginTop: 20, fontFamily: mono, fontSize: 104, fontWeight: 760, color: C.accent }}>0.656</div>
        <div style={{ marginTop: 18, fontSize: 29, lineHeight: 1.4 }}>Wrong answer ranks higher in about 65.6% of random wrong–correct pairs.</div>
        <div style={{ marginTop: 22, fontSize: 24, color: C.muted }}>95% bootstrap CI: 0.558–0.743</div>
      </div>
    </div>
  </PageFrame>
);

const AuRac: Page = () => (
  <PageFrame
    eyebrow="Selective answering"
    title={<>AURAC asks what happens when we <Accent>reject high-risk answers.</Accent></>}
    chapter="evaluation · AURAC"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr .9fr', gap: 64 }}>
      <div>
        <div style={{ padding: 32, background: C.panel, borderTop: '4px solid ' + C.ink }}>
          <div style={{ fontSize: 31, lineHeight: 1.5 }}>1. Sort answers from lowest to highest predicted risk.</div>
          <div style={{ marginTop: 18, fontSize: 31, lineHeight: 1.5 }}>2. Keep the lowest-risk 1, 2, …, N answers.</div>
          <div style={{ marginTop: 18, fontSize: 31, lineHeight: 1.5 }}>3. Average strict-label accuracy over all N retained prefixes.</div>
        </div>
        <div style={{ marginTop: 30, padding: 28, background: C.dark, color: C.paper, fontFamily: mono, fontSize: 28 }}>AURAC = mean<sub>k=1..N</sub> accuracy(lowest-risk k)</div>
      </div>
      <div>
        <div style={{ borderTop: '5px solid ' + C.accent, background: C.accentSoft, padding: 34 }}>
          <div style={{ fontFamily: mono, fontSize: 20, color: C.muted }}>INSTRUCT · HELD OUT</div>
          <CalcLine label="lexical disagreement" expression="highest observed" value=".244" />
          <CalcLine label="surprise spread" expression="trace-only" value=".244" />
          <CalcLine label="top-3 surprise" expression="frozen baseline" value=".226" />
        </div>
        <div style={{ marginTop: 24, fontSize: 24, lineHeight: 1.4, color: C.muted }}>Higher is better. The absolute values are low because only 43 of 300 Instruct answers are strictly correct.</div>
      </div>
    </div>
  </PageFrame>
);

const InstructResults: Page = () => (
  <PageFrame
    eyebrow="Held-out pooled benchmark · Instruct"
    title={<>The simplest score ranks best <Accent>descriptively.</Accent></>}
    chapter="benchmark · instruct"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.25fr .42fr .72fr .42fr',
        gap: 22,
        minHeight: 42,
        padding: '0 18px',
        fontFamily: mono,
        fontSize: 17,
        color: C.muted,
      }}
    >
      <div>METHOD</div><div>AUROC</div><div>95% CI</div><div>MACRO-F1</div>
    </div>
    <ResultRow method="Top-3 surprise" auroc="0.656" ci="0.558–0.743" f1="0.584" accent />
    <ResultRow method="3-answer disagreement" auroc="0.621" ci="0.515–0.721" f1="0.594" />
    <ResultRow method="P(False)" auroc="0.603" ci="0.508–0.696" f1="0.506" />
    <ResultRow method="Hidden linear probe" auroc="0.549" ci="0.453–0.641" f1="0.538" />
    <ResultRow method="Answer length control" auroc="0.540" ci="0.432–0.644" f1="0.568" />
    <div style={{ marginTop: 25, fontSize: 24, color: C.muted }}>
      Every paired 95% improvement interval versus top-3 surprise includes zero.
    </div>
  </PageFrame>
);

const BaseResults: Page = () => (
  <PageFrame
    eyebrow="Held-out pooled benchmark · Base"
    title={<>For Base, the self-check leads—<Accent>but only slightly.</Accent></>}
    chapter="benchmark · base"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.25fr .42fr .72fr .42fr',
        gap: 22,
        minHeight: 42,
        padding: '0 18px',
        fontFamily: mono,
        fontSize: 17,
        color: C.muted,
      }}
    >
      <div>METHOD</div><div>AUROC</div><div>95% CI</div><div>MACRO-F1</div>
    </div>
    <ResultRow method="P(False)" auroc="0.628" ci="0.523–0.728" f1="0.536" accent />
    <ResultRow method="Top-3 surprise" auroc="0.599" ci="0.494–0.703" f1="0.527" />
    <ResultRow method="3-answer disagreement" auroc="0.595" ci="0.491–0.691" f1="0.488" />
    <ResultRow method="Hidden linear probe" auroc="0.545" ci="0.433–0.655" f1="0.502" />
    <ResultRow method="Answer length control" auroc="0.528" ci="0.430–0.632" f1="0.552" />
    <div style={{ marginTop: 25, fontSize: 24, color: C.muted }}>
      Base produced only 28 strict-label correct answers among 300 held-out questions.
    </div>
  </PageFrame>
);

const InstructSlices: Page = () => (
  <PageFrame
    eyebrow="Held-out AUROC by dataset · Instruct"
    title={<>Performance changes with the <Accent>question distribution.</Accent></>}
    chapter="benchmark · instruct slices"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.2fr .55fr .55fr .55fr',
        gap: 24,
        minHeight: 52,
        fontFamily: mono,
        fontSize: 18,
        color: C.muted,
      }}
    >
      <div>METHOD</div><div>NQ OPEN</div><div>TRIVIAQA</div><div>TRUTHFULQA</div>
    </div>
    <SliceRow method="Top-3 surprise" nq="0.594" trivia="0.673" truthful="0.784" />
    <SliceRow method="P(False)" nq="0.615" trivia="0.701" truthful="0.360" />
    <SliceRow method="Disagreement" nq="0.668" trivia="0.638" truthful="0.546" />
    <SliceRow method="Hidden probe" nq="0.573" trivia="0.487" truthful="0.647" />
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 34, marginTop: 44 }}>
      <Pill>NQ: 84 wrong / 100</Pill>
      <Pill>TriviaQA: 79 wrong / 100</Pill>
      <Pill>TruthfulQA: 94 wrong / 100</Pill>
    </div>
  </PageFrame>
);

const BaseSlices: Page = () => (
  <PageFrame
    eyebrow="Held-out AUROC by dataset · Base"
    title={<>TruthfulQA has <Accent>no correct Base answers</Accent> under the strict label.</>}
    chapter="benchmark · base slices"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.2fr .55fr .55fr .55fr',
        gap: 24,
        minHeight: 52,
        fontFamily: mono,
        fontSize: 18,
        color: C.muted,
      }}
    >
      <div>METHOD</div><div>NQ OPEN</div><div>TRIVIAQA</div><div>TRUTHFULQA</div>
    </div>
    <SliceRow method="Top-3 surprise" nq="0.576" trivia="0.533" truthful="—" />
    <SliceRow method="P(False)" nq="0.515" trivia="0.636" truthful="—" />
    <SliceRow method="Disagreement" nq="0.395" trivia="0.605" truthful="—" />
    <SliceRow method="Hidden probe" nq="0.557" trivia="0.586" truthful="—" />
    <div style={{ marginTop: 46, padding: 32, background: C.accentSoft, borderLeft: '5px solid ' + C.accent }}>
      <div style={{ fontFamily: mono, fontSize: 26, color: C.accent }}>100 wrong · 0 correct</div>
      <div style={{ marginTop: 13, fontSize: 29, lineHeight: 1.4 }}>
        AUROC needs both classes. “—” is mathematically undefined, not a missing computation.
      </div>
    </div>
  </PageFrame>
);

const ExtensionBoundary: Page = () => (
  <PageFrame
    eyebrow="A second analysis layer"
    title={<>The 31-method audit is useful—<Accent>and explicitly post-hoc.</Accent></>}
    chapter="extension · evidence boundary"
    dark
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 56 }}>
      <div style={{ minHeight: 470, padding: 38, borderTop: '5px solid ' + C.safe, background: C.darkPanel }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.safe }}>FROZEN CONFIRMATORY LAYER</div>
        <div style={{ marginTop: 25, fontSize: 38, fontWeight: 730 }}>5 methods</div>
        <div style={{ marginTop: 22, fontSize: 27, lineHeight: 1.5, color: C.darkMuted }}>Top-3 surprise · P(False) · hidden probe · semantic disagreement · answer length.</div>
        <div style={{ marginTop: 30 }}><Pill tone="dark">specified before held-out evaluation</Pill></div>
      </div>
      <div style={{ minHeight: 470, padding: 38, borderTop: '5px solid ' + C.accent, background: C.darkPanel }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.accent }}>EXPLORATORY EXTENSION</div>
        <div style={{ marginTop: 25, fontSize: 38, fontWeight: 730 }}>31 methods total</div>
        <div style={{ marginTop: 22, fontSize: 27, lineHeight: 1.5, color: C.darkMuted }}>All 24 saved scalars · lexical and semantic-entropy proxies · 8-feature logistic · depth-2 tree.</div>
        <div style={{ marginTop: 30 }}><Pill tone="accent">held-out outputs were already inspected</Pill></div>
      </div>
    </div>
    <div style={{ marginTop: 34, fontSize: 28, lineHeight: 1.4 }}>Use the extension to generate hypotheses. Confirm the winner on a new untouched split.</div>
  </PageFrame>
);

const ScalarFamilies: Page = () => (
  <PageFrame
    eyebrow="What can one greedy trace give us?"
    title={<>Twenty-four scalars fall into <Accent>six intuitive families.</Accent></>}
    chapter="extension · scalar families"
  >
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 26 }}>
      {[
        ['TOKEN PROBABILITY · 7','mean NLL · top-3/worst/top-4 surprise · spread · confidence variability'],
        ['DISTRIBUTION · 6','mean/max/top-3 entropy · mean/max/top-3 top-2 ambiguity'],
        ['HIDDEN DYNAMICS · 6','shift variability · cosine mean/max/top-3 · hidden RMS mean/variability'],
        ['COUPLED · 2','top-3 surprise×shift · top-3 uncertainty×shift'],
        ['CRCV · 2','mean and maximum rolling SD of confidence×shift'],
        ['CONTROL · 1','answer token count: useful only to expose length confounding'],
      ].map(([label, text], index) => <div key={label} style={{ minHeight: 208, padding: 28, background: index === 4 ? C.accentSoft : C.panel, borderTop: '4px solid ' + (index === 4 ? C.accent : C.ink) }}><div style={{ fontFamily: mono, fontSize: 19, color: index === 4 ? C.accent : C.muted }}>{label}</div><div style={{ marginTop: 18, fontSize: 26, lineHeight: 1.42 }}>{text}</div></div>)}
    </div>
    <div style={{ marginTop: 30, fontSize: 24, color: C.muted }}>Every value was already present in the saved trace; no model rerun and no held-out feature selection.</div>
  </PageFrame>
);

const ExpandedInstruct: Page = () => (
  <PageFrame
    eyebrow="Post-hoc pooled ranking · Instruct"
    title={<>Lexical consistency leads; <Accent>trace-only spread is close.</Accent></>}
    chapter="extension · instruct ranking"
    titleSize={61}
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1.25fr .42fr .72fr .42fr', gap: 22, minHeight: 36, padding: '0 18px', fontFamily: mono, fontSize: 17, color: C.muted }}><div>METHOD</div><div>AUROC</div><div>95% CI</div><div>AURAC</div></div>
    <ResultRow method="3-sample lexical disagreement" auroc="0.717" ci="0.623–0.801" f1="0.244" accent />
    <ResultRow method="Token-surprise spread" auroc="0.706" ci="0.620–0.790" f1="0.244" />
    <ResultRow method="Maximum token entropy" auroc="0.696" ci="0.609–0.777" f1="0.238" />
    <ResultRow method="8-feature trace logistic" auroc="0.695" ci="0.603–0.777" f1="0.238" />
    <ResultRow method="Worst-token surprise" auroc="0.689" ci="0.592–0.776" f1="0.236" />
    <ResultRow method="Top-3 surprise · frozen" auroc="0.656" ci="0.560–0.747" f1="0.226" />
    <ResultRow method="CRCV mean" auroc="0.634" ci="0.543–0.712" f1="0.194" />
  </PageFrame>
);

const ExpandedBase: Page = () => (
  <PageFrame
    eyebrow="Post-hoc pooled ranking · Base"
    title={<>Maximum token ambiguity is highest—<Accent>but uncertainty is wide.</Accent></>}
    chapter="extension · base ranking"
    titleSize={59}
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1.25fr .42fr .72fr .42fr', gap: 22, minHeight: 36, padding: '0 18px', fontFamily: mono, fontSize: 17, color: C.muted }}><div>METHOD</div><div>AUROC</div><div>95% CI</div><div>AURAC</div></div>
    <ResultRow method="Maximum token ambiguity" auroc="0.651" ci="0.553–0.745" f1="0.123" accent />
    <ResultRow method="Worst-token surprise" auroc="0.635" ci="0.528–0.742" f1="0.130" />
    <ResultRow method="Top-3 surprise after token 1" auroc="0.635" ci="0.530–0.732" f1="0.137" />
    <ResultRow method="Token-surprise spread" auroc="0.630" ci="0.518–0.730" f1="0.138" />
    <ResultRow method="Hidden RMS variability" auroc="0.629" ci="0.524–0.729" f1="0.131" />
    <ResultRow method="P(False) · frozen" auroc="0.628" ci="0.525–0.728" f1="0.115" />
    <ResultRow method="CRCV mean" auroc="0.537" ci="0.422–0.646" f1="0.122" />
  </PageFrame>
);

const ExtendedCost: Page = () => (
  <PageFrame
    eyebrow="Accuracy–cost frontier · Instruct"
    title={<>The best observed free score is <Accent>surprise spread.</Accent></>}
    chapter="extension · cost frontier"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr 135px 180px', gap: 20, minHeight: 38, fontFamily: mono, fontSize: 17, color: C.muted }}><div>METHOD</div><div>AUROC ABOVE 0.5</div><div>AUROC</div><div>EXTRA / Q</div></div>
    <CostRow method="Lexical disagreement" auroc="0.717" cost="1.462 s" width={500} />
    <CostRow method="Surprise spread" auroc="0.706" cost="0.000 s" width={475} />
    <CostRow method="8-feature logistic" auroc="0.695" cost="0.000002 s" width={450} />
    <CostRow method="Max token entropy" auroc="0.696" cost="0.000 s" width={452} />
    <CostRow method="Top-3 surprise" auroc="0.656" cost="0.000 s" width={360} />
    <CostRow method="CRCV mean" auroc="0.634" cost="0.000 s" width={310} />
    <div style={{ marginTop: 24, fontSize: 24, color: C.muted }}>No added method’s paired 95% AUROC-difference interval is entirely above zero versus top-3 surprise.</div>
  </PageFrame>
);

const PublishedContext: Page = () => (
  <PageFrame
    eyebrow="How this sits beside published SOTA"
    title={<>A published 0.790 is <Accent>not our leaderboard’s 0.790.</Accent></>}
    chapter="literature · numeric context"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr .82fr', gap: 60 }}>
      <div>
        <div style={{ fontFamily: mono, fontSize: 18, color: C.muted, marginBottom: 14 }}>FARQUHAR ET AL. · AVERAGE OVER 30 TASK×MODEL COMBINATIONS</div>
        {[
          ['Semantic entropy', '.790', 520],
          ['P(True)', '.698', 395],
          ['Naive entropy', '.691', 385],
          ['Embedding regression', '.687', 380],
        ].map(([name, value, width]) => <div key={String(name)} style={{ display: 'grid', gridTemplateColumns: '330px 1fr 100px', gap: 18, alignItems: 'center', minHeight: 88, borderTop: '1px solid ' + C.line }}><div style={{ fontSize: 26, fontWeight: 680 }}>{name}</div><div style={{ height: 16, background: '#e2ded5' }}><div style={{ width: Number(width), height: 16, background: name === 'Semantic entropy' ? C.accent : C.ink }} /></div><div style={{ fontFamily: mono, fontSize: 29 }}>{value}</div></div>)}
      </div>
      <div style={{ padding: 36, background: C.accentSoft, borderTop: '5px solid ' + C.accent }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>WHY NOT NUMERICALLY COMPARABLE</div>
        <div style={{ marginTop: 24, fontSize: 28, lineHeight: 1.5 }}>7B–70B models, not 0.5B.</div>
        <div style={{ marginTop: 18, fontSize: 28, lineHeight: 1.5 }}>Ten generations, not three.</div>
        <div style={{ marginTop: 18, fontSize: 28, lineHeight: 1.5 }}>Human-validated correctness and different tasks/splits.</div>
        <div style={{ marginTop: 28, fontSize: 23, lineHeight: 1.4, color: C.muted }}>Source: Farquhar et al., Nature 2024, Fig. 2 discussion.</div>
      </div>
    </div>
  </PageFrame>
);

const LiteratureFeasibility: Page = () => (
  <PageFrame
    eyebrow="What was fairly evaluable from our saved data?"
    title={<>Reproduce what the trace supports. <Accent>Do not fabricate the rest.</Accent></>}
    chapter="literature · feasibility"
    titleSize={60}
  >
    <div style={{ display: 'grid', gridTemplateColumns: '310px 1fr 1.15fr 150px', gap: 24, minHeight: 35, fontFamily: mono, fontSize: 16, color: C.muted }}><div>FAMILY</div><div>EXTRA INPUT</div><div>STATUS IN THIS LAB</div><div></div></div>
    <LiteratureRow method="P(True)" input="self-evaluation pass" here="P(False) with fixed Yes/No prompt" status="yes" />
    <LiteratureRow method="SelfCheckGPT" input="multiple generations" here="3-sample lexical Jaccard adaptation" status="proxy" />
    <LiteratureRow method="Semantic entropy" input="samples + meaning clusters" here="3-sample discrete entropy adaptation" status="proxy" />
    <LiteratureRow method="SAR" input="token/sentence relevance passes" here="deletion/relevance passes were not saved" status="no" />
    <LiteratureRow method="INSIDE / EigenScore" input="hidden embeddings for sampled answers" here="only greedy-answer hidden states were saved" status="no" />
    <LiteratureRow method="Lookback Lens" input="attention to grounding context" here="no grounding passage or attention maps" status="no" />
  </PageFrame>
);

const Cost: Page = () => (
  <PageFrame
    eyebrow="Frozen five-method cost · Instruct"
    title={<>The original comparison still gives the <Accent>cleanest evidence.</Accent></>}
    chapter="analysis · frozen cost"
  >
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '350px 1fr 135px 180px',
        gap: 20,
        minHeight: 42,
        fontFamily: mono,
        fontSize: 17,
        color: C.muted,
      }}
    >
      <div>METHOD</div><div>AUROC ABOVE 0.5</div><div>AUROC</div><div>EXTRA / Q</div>
    </div>
    <CostRow method="Top-3 surprise" auroc="0.656" cost="0.000 s" width={450} />
    <CostRow method="3-answer disagreement" auroc="0.621" cost="1.901 s" width={350} />
    <CostRow method="P(False)" auroc="0.603" cost="0.130 s" width={295} />
    <CostRow method="Hidden probe" auroc="0.549" cost="0.000073 s" width={140} />
    <CostRow method="Answer length" auroc="0.540" cost="0.000 s" width={115} />
  </PageFrame>
);

const Findings: Page = () => (
  <PageFrame
    eyebrow="What survives the benchmark?"
    title={<>The evidence supports a <Accent>cautious baseline</Accent>, not a solved detector.</>}
    chapter="analysis · findings"
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px 56px' }}>
      <div style={{ borderTop: '3px solid ' + C.safe, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.safe }}>01 · USEFUL</div>
        <div style={{ marginTop: 12, fontSize: 32, fontWeight: 700 }}>Confidence variation contains signal</div>
        <div style={{ marginTop: 10, fontSize: 25, lineHeight: 1.4, color: C.muted }}>Top-3: .656 frozen; spread: .706 post-hoc.</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>02 · MODEL-SPECIFIC</div>
        <div style={{ marginTop: 12, fontSize: 32, fontWeight: 700 }}>Observed winners do not transfer</div>
        <div style={{ marginTop: 10, fontSize: 25, lineHeight: 1.4, color: C.muted }}>Lexical leads Instruct; ambiguity leads Base.</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>03 · NO ENSEMBLE WIN</div>
        <div style={{ marginTop: 12, fontSize: 32, fontWeight: 700 }}>Eight features do not dominate</div>
        <div style={{ marginTop: 10, fontSize: 25, lineHeight: 1.4, color: C.muted }}>Trace logistic: .695 Instruct, .627 Base.</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20 }}>04 · NOT ESTABLISHED</div>
        <div style={{ marginTop: 12, fontSize: 32, fontWeight: 700 }}>No reliable improvement</div>
        <div style={{ marginTop: 10, fontSize: 25, lineHeight: 1.4, color: C.muted }}>All paired 95% differences vs top-3 include zero.</div>
      </div>
    </div>
  </PageFrame>
);

const Limits: Page = () => (
  <PageFrame
    eyebrow="Read the result honestly"
    title={<>This is not a <Accent>general factuality oracle.</Accent></>}
    chapter="limitations"
    dark
  >
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px 56px' }}>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>STRICT LABELS</div>
        <div style={{ marginTop: 12, fontSize: 29, lineHeight: 1.4, color: C.darkMuted }}>Alias matching can reject a good paraphrase.</div>
      </div>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>CLASS IMBALANCE</div>
        <div style={{ marginTop: 12, fontSize: 29, lineHeight: 1.4, color: C.darkMuted }}>Only 43 Instruct and 28 Base held-out answers were correct.</div>
      </div>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>REFERENCE-FREE CEILING</div>
        <div style={{ marginTop: 12, fontSize: 29, lineHeight: 1.4, color: C.darkMuted }}>Internal uncertainty cannot retrieve missing world knowledge.</div>
      </div>
      <div style={{ borderTop: '1px solid ' + C.darkLine, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>POST-HOC EXTENSION</div>
        <div style={{ marginTop: 12, fontSize: 29, lineHeight: 1.4, color: C.darkMuted }}>The 31-method ranking needs a fresh confirmatory split.</div>
      </div>
    </div>
    <div style={{ marginTop: 44, padding: 30, border: '1px solid ' + C.darkLine, background: C.darkPanel }}>
      <div style={{ fontSize: 29, lineHeight: 1.4 }}>
        Next: human factuality labels, retrieval baselines, larger models, repeated seeds, and calibration transfer tests.
      </div>
    </div>
  </PageFrame>
);

const Closing: Page = () => (
  <section style={basePage}>
    <DeckStyles />
    <Grid />
    <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 24, background: C.accent }} />
    <div style={{ position: 'absolute', left: 150, top: 128, width: 1250 }}>
      <Eyebrow>Recommendation</Eyebrow>
      <h2
        style={{
          margin: '36px 0 0',
          fontFamily: 'var(--osd-font-display)',
          fontSize: 106,
          lineHeight: .98,
          letterSpacing: '-.058em',
          fontWeight: 770,
        }}
      >
        Start with top-3 surprise.<br />
        <Accent>Show the arithmetic.</Accent><br />
        Treat every flag as triage.
      </h2>
    </div>
    <div style={{ position: 'absolute', left: 150, bottom: 122, width: 1180, display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 38 }}>
      <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 18 }}>
        <div style={{ fontFamily: mono, fontSize: 19, color: C.muted }}>FAST</div>
        <div style={{ marginTop: 10, fontSize: 28, fontWeight: 700 }}>0 extra seconds</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.ink, paddingTop: 18 }}>
        <div style={{ fontFamily: mono, fontSize: 19, color: C.muted }}>INSPECTABLE</div>
        <div style={{ marginTop: 10, fontSize: 28, fontWeight: 700 }}>token-by-token evidence</div>
      </div>
      <div style={{ borderTop: '3px solid ' + C.accent, paddingTop: 18 }}>
        <div style={{ fontFamily: mono, fontSize: 19, color: C.accent }}>LIMIT</div>
        <div style={{ marginTop: 10, fontSize: 28, fontWeight: 700 }}>warning, not truth</div>
      </div>
    </div>
    <div style={{ position: 'absolute', right: 130, top: 205, width: 390, borderTop: '4px solid ' + C.accent, paddingTop: 26 }}>
      <div style={{ fontFamily: mono, fontSize: 21, color: C.muted }}>RUN THE LAB</div>
      <div style={{ marginTop: 18, fontSize: 30, lineHeight: 1.32, fontWeight: 680 }}>
        nipunbatra.github.io/<br />tiny-crcv-lab/
      </div>
      <div style={{ marginTop: 58, fontSize: 25, lineHeight: 1.4, color: C.muted }}>
        Search all 600 answers and inspect every saved score.
      </div>
    </div>
    <Footer chapter="takeaway" />
  </section>
);

const quietTransition: SlideTransition = {
  duration: 220,
  exit: {
    duration: 150,
    easing: 'cubic-bezier(0.4, 0, 1, 1)',
    keyframes: [
      { opacity: 1, transform: 'translateY(0)' },
      { opacity: 0, transform: 'translateY(-4px)' },
    ],
  },
  enter: {
    duration: 220,
    delay: 80,
    easing: 'cubic-bezier(0, 0, 0.2, 1)',
    keyframes: [
      { opacity: 0, transform: 'translateY(6px)' },
      { opacity: 1, transform: 'translateY(0)' },
    ],
  },
};

export const transition: SlideTransition = quietTransition;

export const meta: SlideMeta = {
  title: 'Given a Question and Answer, Can We Tell When It Is Wrong?',
  createdAt: '2026-07-19T01:46:38.124Z',
};

export const notes = {
  0: 'Open with the exact task: judge a generated answer for a particular question.',
  1: 'Separate generation from detection. The detector is downstream of an answer.',
  2: 'State the scope explicitly: question-only and arbitrary-text detection are different problems.',
  3: 'Use New Delhi versus Mumbai only to establish the label; no detector values are invented here.',
  4: 'A score ranks or triages. It does not prove factual truth.',
  5: 'Explain the strict alias rule before showing any benchmark result.',
  6: 'These are real outputs from the two saved checkpoints.',
  7: 'Use the rats example to show why a reproducible label can still be semantically imperfect.',
  8: 'The dataset is source-balanced: 200 questions from each source.',
  9: 'Calibration and held-out questions never overlap.',
  10: 'The checkpoints share size but differ substantially in prompting behavior.',
  11: 'All architecture numbers are from the official Qwen configuration.',
  12: 'One generation step outputs both logits and the final hidden state used by the detector.',
  13: 'The exact chat prompt is 48 tokens; the visible question is seven tokens.',
  14: 'Keep the transformer explanation conceptual: attention mixes context, MLP transforms positions.',
  15: 'Advance once to reveal the selected Tai token and its exact recorded values.',
  16: 'Advance once to reveal the pei token and the completed answer.',
  17: 'These are the five primitive signals saved for every generated token.',
  18: 'This held-out TriviaQA row is the worked example for every following calculation.',
  19: 'The right column shows only the three lowest-probability selected tokens among 24.',
  20: 'Surprise is negative log probability; top-3 averages the largest three values.',
  21: 'Carry every displayed decimal through the two side-by-side calculations.',
  22: 'Thresholds are checkpoint-specific, so do not compare raw Base and Instruct scores directly.',
  23: 'Define CRCV from confidence, normalized state shift, and rolling sample variability.',
  24: 'The correct two-token answer has one valid coupling, so its sample SD is zero by convention.',
  25: 'Compute the first Base five-coupling window without skipping the sample-variance denominator.',
  26: 'All 19 saved window SDs sum to 3.591366 and average to 0.189019.',
  27: 'One intuitive pair is not the benchmark: CRCV is modest on Instruct and near chance on Base.',
  28: 'P(False) is a separate judge prompt and costs one extra forward pass.',
  29: 'The self-check catches the wrong Base answer but false-alarms on the correct Instruct answer.',
  30: 'The three stochastic generations use the saved seeds shown in the trace.',
  31: 'Disagreement is the mean normalized No probability across three answer pairs.',
  32: 'This is a simple SelfCheckGPT-inspired short-answer adaptation, not the paper’s full pipeline.',
  33: 'Jaccard word-set distances are zero for three Taipeis and high for the divergent Base samples.',
  34: 'Semantic entropy clusters meanings before measuring their empirical uncertainty.',
  35: 'The three-sample proxy gives entropy zero for one cluster and 0.636514 for sizes two and one.',
  36: 'The hidden probe is a standardized linear layer over the 896-dimensional mean hidden state.',
  37: 'Shown contributions plus the grouped remainder reproduce the exact saved probe logits.',
  38: 'The trace logistic uses eight fixed, cheap features and calibration-only standardization.',
  39: 'For correct Taipei, the eight contributions plus bias give logit −1.897300 and risk 0.130414.',
  40: 'For the wrong Base response, the same arithmetic gives risk 0.368457 and crosses its cutoff.',
  41: 'A depth-2 tree asks only two trace questions before returning a calibration leaf wrong-rate.',
  42: 'Follow both exact paths and divide leaf wrong counts by leaf row counts.',
  43: 'The original four-method scorecard keeps successes and false alarms visible.',
  44: 'Do not infer generalization from a single intuitive pair.',
  45: 'Threshold search sees only calibration rows and maximizes macro-F1.',
  46: 'These four held-out confusion counts sum to 300.',
  47: 'Translate AUROC as pairwise ranking frequency, not accuracy or correctness probability.',
  48: 'AURAC summarizes retained-answer accuracy as high-risk answers are rejected.',
  49: 'This is the original frozen Instruct comparison, not the later expanded search.',
  50: 'This is the original frozen Base comparison; P(False) leads only descriptively.',
  51: 'Dataset slices show that rankings are distribution-dependent.',
  52: 'Base TruthfulQA AUROC is undefined because all 100 strict labels are wrong.',
  53: 'Clearly separate the confirmatory five-method layer from the post-hoc 31-method extension.',
  54: 'All 24 scalar trace features are reported; none were selected by held-out performance.',
  55: 'Lexical disagreement leads Instruct descriptively, while free surprise spread is close.',
  56: 'Base has only 28 correct held-out answers, so every interval is wide.',
  57: 'The practical cost frontier favors free trace scalars unless three samples are acceptable.',
  58: 'Published semantic-entropy numbers use larger models, ten generations, and different labels.',
  59: 'Evaluate only methods whose required inputs were actually saved; label adaptations as proxies.',
  60: 'The original cost slide makes the frozen five-method comparison easy to reconstruct.',
  61: 'Summarize empirical findings without turning post-hoc leaders into confirmed winners.',
  62: 'State strict-label, imbalance, reference-free, and scope limitations before recommending use.',
  63: 'End with cheap, inspectable triage and point to the interactive arithmetic.',
};

export default [
  Cover,
  TaskContract,
  Scope,
  IndiaExample,
  DetectorOutput,
  LabelRule,
  DatasetExamples,
  TruthfulExamples,
  DatasetDesign,
  Protocol,
  Checkpoints,
  ModelAnatomy,
  GenerationStep,
  Tokenization,
  TransformerBlock,
  GenerateFirst,
  GenerateSecond,
  Signals,
  PairIntro,
  PairTrace,
  SurpriseIdea,
  SurpriseCalc,
  SurpriseDecision,
  CrcvIdea,
  CrcvCorrect,
  CrcvWrongWindow,
  CrcvWrongAggregate,
  CrcvResults,
  PFalsePrompt,
  PFalseCalc,
  DisagreementSamples,
  DisagreementCalc,
  LexicalIdea,
  LexicalCalc,
  SemanticEntropyIdea,
  SemanticEntropyCalc,
  ProbeIdea,
  ProbeCalc,
  TraceLogisticIdea,
  TraceLogisticCorrect,
  TraceLogisticWrong,
  TraceTreeIdea,
  TraceTreeCalc,
  PairScorecard,
  PairLesson,
  Calibration,
  ThresholdQuality,
  AuRoc,
  AuRac,
  InstructResults,
  BaseResults,
  InstructSlices,
  BaseSlices,
  ExtensionBoundary,
  ScalarFamilies,
  ExpandedInstruct,
  ExpandedBase,
  ExtendedCost,
  PublishedContext,
  LiteratureFeasibility,
  Cost,
  Findings,
  Limits,
  Closing,
] satisfies Page[];
