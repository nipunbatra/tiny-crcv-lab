import {
  MorphElement,
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
    hero: 148,
    body: 36,
  },
  radius: 14,
};

const C = {
  paper: '#f4f1ea',
  panel: '#fffdf8',
  ink: '#18211d',
  muted: '#69716d',
  faint: '#a7aaa5',
  line: '#d7d3c9',
  accent: '#df4c2f',
  accentSoft: '#fbe9e2',
  inkSoft: '#e5e9e5',
  dark: '#18211d',
  darkPanel: '#232d28',
  darkLine: '#3b4741',
  darkMuted: '#aab3ae',
};

const mono = '"Geist Mono Variable", "SFMono-Regular", Consolas, monospace';
const PAD = 118;

const deckCss = `
  @keyframes crcv-flow {
    from { stroke-dashoffset: 42; }
    to { stroke-dashoffset: 0; }
  }
  @keyframes crcv-pulse {
    0%, 100% { opacity: .35; transform: scale(.92); }
    50% { opacity: 1; transform: scale(1); }
  }
  @keyframes crcv-scan {
    from { transform: translateX(-110%); }
    to { transform: translateX(520%); }
  }
  @keyframes crcv-rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .crcv-flow { animation: crcv-flow 1.3s cubic-bezier(.16,1,.3,1) infinite; }
  .crcv-pulse { animation: crcv-pulse 1.7s cubic-bezier(.16,1,.3,1) infinite; transform-origin: center; }
  .crcv-scan { animation: crcv-scan 2.1s cubic-bezier(.65,0,.35,1) infinite; }
  .crcv-rise { animation: crcv-rise .55s cubic-bezier(.16,1,.3,1) both; }
  @media (prefers-reduced-motion: reduce) {
    .crcv-flow, .crcv-pulse, .crcv-scan, .crcv-rise { animation: none !important; }
  }
`;

const DeckStyles = () => <style>{deckCss}</style>;

const basePage: CSSProperties = {
  width: '100%',
  height: '100%',
  position: 'relative',
  overflow: 'hidden',
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
      backgroundImage: `linear-gradient(${dark ? 'rgba(244,241,234,.055)' : 'rgba(24,33,29,.045)'} 1px, transparent 1px), linear-gradient(90deg, ${dark ? 'rgba(244,241,234,.055)' : 'rgba(24,33,29,.045)'} 1px, transparent 1px)`,
      backgroundSize: '48px 48px',
      maskImage: 'linear-gradient(to bottom, rgba(0,0,0,.7), transparent 86%)',
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
        bottom: 38,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderTop: `1px solid ${dark ? C.darkLine : C.line}`,
        paddingTop: 15,
        fontFamily: mono,
        fontSize: 20,
        color: dark ? C.darkMuted : C.muted,
        letterSpacing: '.08em',
        textTransform: 'uppercase',
      }}
    >
      <span>{chapter}</span>
      <span>{String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}</span>
    </div>
  );
};

const Eyebrow = ({ children, dark = false }: { children: ReactNode; dark?: boolean }) => (
  <div
    style={{
      fontFamily: mono,
      fontSize: 22,
      fontWeight: 650,
      letterSpacing: '.14em',
      textTransform: 'uppercase',
      color: dark ? C.darkMuted : C.muted,
    }}
  >
    {children}
  </div>
);

const Heading = ({ children, width = 1500, dark = false }: { children: ReactNode; width?: number; dark?: boolean }) => (
  <h2
    style={{
      margin: 0,
      maxWidth: width,
      fontFamily: 'var(--osd-font-display)',
      fontSize: 76,
      lineHeight: 1.04,
      letterSpacing: '-.048em',
      fontWeight: 760,
      color: dark ? C.paper : C.ink,
    }}
  >
    {children}
  </h2>
);

const Accent = ({ children }: { children: ReactNode }) => <span style={{ color: C.accent }}>{children}</span>;

const Pill = ({ children, tone = 'plain' }: { children: ReactNode; tone?: 'plain' | 'accent' | 'dark' }) => (
  <span
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      minHeight: 42,
      padding: '7px 16px',
      borderRadius: 999,
      border: `1px solid ${tone === 'accent' ? C.accent : tone === 'dark' ? C.darkLine : C.line}`,
      background: tone === 'accent' ? C.accentSoft : tone === 'dark' ? C.darkPanel : C.panel,
      color: tone === 'accent' ? C.accent : tone === 'dark' ? C.paper : C.ink,
      fontFamily: mono,
      fontSize: 22,
      whiteSpace: 'nowrap',
    }}
  >
    {children}
  </span>
);

const Arrow = ({ width = 90, dark = false }: { width?: number; dark?: boolean }) => (
  <svg width={width} height="30" viewBox={`0 0 ${width} 30`} aria-hidden="true">
    <line x1="2" y1="15" x2={width - 14} y2="15" stroke={dark ? C.darkMuted : C.muted} strokeWidth="2" />
    <path d={`M ${width - 22} 7 L ${width - 12} 15 L ${width - 22} 23`} fill="none" stroke={dark ? C.darkMuted : C.muted} strokeWidth="2" />
  </svg>
);

const FlowArrow = ({ width = 130, dark = false }: { width?: number; dark?: boolean }) => {
  const active = useIsActivePage();
  return (
    <svg width={width} height="38" viewBox={`0 0 ${width} 38`} aria-hidden="true">
      <line
        className={active ? 'crcv-flow' : undefined}
        x1="2"
        y1="19"
        x2={width - 18}
        y2="19"
        stroke={dark ? C.paper : C.accent}
        strokeWidth="4"
        strokeDasharray="12 9"
      />
      <path d={`M ${width - 29} 8 L ${width - 14} 19 L ${width - 29} 30`} fill="none" stroke={dark ? C.paper : C.accent} strokeWidth="4" />
    </svg>
  );
};

const Stat = ({ value, label, detail, accent = false }: { value: string; label: string; detail: string; accent?: boolean }) => (
  <div style={{ borderTop: `3px solid ${accent ? C.accent : C.ink}`, paddingTop: 20, minWidth: 250 }}>
    <div style={{ fontFamily: mono, fontSize: 66, fontWeight: 740, letterSpacing: '-.06em', color: accent ? C.accent : C.ink }}>{value}</div>
    <div style={{ marginTop: 10, fontSize: 29, fontWeight: 660 }}>{label}</div>
    <div style={{ marginTop: 7, fontSize: 23, color: C.muted }}>{detail}</div>
  </div>
);

const Claim = ({ number, title, text }: { number: string; title: string; text: string }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '95px 1fr', gap: 28, alignItems: 'start', padding: '24px 0', borderTop: `1px solid ${C.line}` }}>
    <div style={{ fontFamily: mono, fontSize: 24, color: C.accent }}>{number}</div>
    <div>
      <div style={{ fontSize: 37, fontWeight: 680, letterSpacing: '-.025em' }}>{title}</div>
      <div style={{ marginTop: 8, fontSize: 27, lineHeight: 1.42, color: C.muted }}>{text}</div>
    </div>
  </div>
);

const DatasetBand = ({ label, count, character }: { label: string; count: string; character: string }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '210px 150px 1fr', alignItems: 'center', gap: 24, minHeight: 118, borderTop: `1px solid ${C.line}` }}>
    <div style={{ fontSize: 34, fontWeight: 680 }}>{label}</div>
    <div style={{ fontFamily: mono, fontSize: 42, color: C.accent }}>{count}</div>
    <div style={{ fontSize: 27, lineHeight: 1.4, color: C.muted }}>{character}</div>
  </div>
);

const ModelTower = ({ label, prompt, emphasis = false }: { label: string; prompt: string; emphasis?: boolean }) => (
  <div style={{ flex: 1, minHeight: 230, padding: 32, borderTop: `4px solid ${emphasis ? C.accent : C.ink}`, background: emphasis ? C.accentSoft : C.panel }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ fontSize: 36, fontWeight: 720 }}>{label}</div>
      <Pill tone={emphasis ? 'accent' : 'plain'}>Qwen2.5 · 0.5B</Pill>
    </div>
    <div style={{ marginTop: 34, fontFamily: mono, fontSize: 23, lineHeight: 1.55, color: C.muted }}>{prompt}</div>
  </div>
);

const Block = ({ label, detail, accent = false, width = 250 }: { label: string; detail: string; accent?: boolean; width?: number }) => (
  <div style={{ width, minHeight: 142, padding: '24px 25px', border: `1px solid ${accent ? C.accent : C.darkLine}`, background: accent ? C.accent : C.darkPanel, color: accent ? C.panel : C.paper, borderRadius: 'var(--osd-radius)' }}>
    <div style={{ fontSize: 29, fontWeight: 700 }}>{label}</div>
    <div style={{ marginTop: 14, fontFamily: mono, fontSize: 20, lineHeight: 1.4, color: accent ? C.panel : C.darkMuted }}>{detail}</div>
  </div>
);

const QuestionMorph = ({ left, top, width, compact = false }: { left: number; top: number; width: number; compact?: boolean }) => (
  <MorphElement id="walk-question">
    <div
      style={{
        position: 'absolute',
        left,
        top,
        width,
        height: compact ? 82 : 150,
        display: 'flex',
        alignItems: 'center',
        padding: compact ? '0 30px' : '0 44px',
        border: `1px solid ${C.darkLine}`,
        background: C.darkPanel,
        borderRadius: compact ? 12 : 18,
        color: C.paper,
        fontFamily: mono,
        fontSize: compact ? 25 : 34,
        letterSpacing: '-.02em',
      }}
    >
      who is under the mask of darth vader
    </div>
  </MorphElement>
);

const TokenMorph = ({ id, text, left, top, width, accent = false }: { id: string; text: string; left: number; top: number; width: number; accent?: boolean }) => (
  <MorphElement id={id}>
    <div
      style={{
        position: 'absolute',
        left,
        top,
        width,
        height: 92,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: `1px solid ${accent ? C.accent : C.darkLine}`,
        background: accent ? C.accent : C.darkPanel,
        borderRadius: 12,
        color: C.paper,
        fontFamily: mono,
        fontSize: 31,
        fontWeight: 680,
      }}
    >
      {text}
    </div>
  </MorphElement>
);

const TraceCell = ({ token, confidence, surprise, margin, shift, norm }: { token: string; confidence: string; surprise: string; margin: string; shift: string; norm: string }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '190px repeat(5, 1fr)', alignItems: 'center', minHeight: 93, borderTop: `1px solid ${C.darkLine}`, fontFamily: mono, fontSize: 25 }}>
    <div style={{ color: C.paper, fontWeight: 680 }}>{token}</div>
    <div>{confidence}</div>
    <div>{surprise}</div>
    <div>{margin}</div>
    <div>{shift}</div>
    <div>{norm}</div>
  </div>
);

const FormulaLine = ({ label, formula, value, accent = false }: { label: string; formula: string; value: string; accent?: boolean }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr 190px', gap: 28, alignItems: 'center', minHeight: 105, borderTop: `1px solid ${C.line}` }}>
    <div style={{ fontSize: 29, fontWeight: 670 }}>{label}</div>
    <div style={{ fontFamily: mono, fontSize: 25, color: C.muted }}>{formula}</div>
    <div style={{ fontFamily: mono, textAlign: 'right', fontSize: 35, fontWeight: 720, color: accent ? C.accent : C.ink }}>{value}</div>
  </div>
);

const Threshold = ({ score, threshold, flag = true }: { score: string; threshold: string; flag?: boolean }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 18, marginTop: 32 }}>
    <Pill tone="accent">score {score}</Pill>
    <span style={{ fontFamily: mono, fontSize: 24, color: C.muted }}>{flag ? '>' : '≤'} threshold {threshold}</span>
    <Pill tone={flag ? 'accent' : 'plain'}>{flag ? 'FLAG' : 'PASS'}</Pill>
  </div>
);

const SampleAnswer = ({ index, answer }: { index: string; answer: string }) => (
  <div style={{ minHeight: 104, borderTop: `1px solid ${C.line}`, paddingTop: 19 }}>
    <div style={{ fontFamily: mono, fontSize: 19, color: C.muted }}>SAMPLE {index}</div>
    <div style={{ marginTop: 9, fontSize: 31, fontWeight: 640 }}>{answer}</div>
  </div>
);

const SplitBox = ({ label, count, detail, accent = false }: { label: string; count: string; detail: string; accent?: boolean }) => (
  <div style={{ width: 620, minHeight: 260, padding: 38, borderTop: `5px solid ${accent ? C.accent : C.ink}`, background: accent ? C.accentSoft : C.panel }}>
    <Eyebrow>{label}</Eyebrow>
    <div style={{ marginTop: 24, fontFamily: mono, fontSize: 84, fontWeight: 760, letterSpacing: '-.07em', color: accent ? C.accent : C.ink }}>{count}</div>
    <div style={{ marginTop: 10, fontSize: 28, color: C.muted }}>{detail}</div>
  </div>
);

const AuPair = ({ correctScore, wrongScore, win }: { correctScore: string; wrongScore: string; win: boolean }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '1fr 110px 1fr', alignItems: 'center', gap: 22, minHeight: 128, borderTop: `1px solid ${C.line}` }}>
    <div>
      <div style={{ fontSize: 21, color: C.muted }}>correct answer</div>
      <div style={{ fontFamily: mono, fontSize: 42 }}>{correctScore}</div>
    </div>
    <div style={{ fontFamily: mono, fontSize: 30, color: win ? C.accent : C.muted, textAlign: 'center' }}>{win ? '<' : '≥'}</div>
    <div>
      <div style={{ fontSize: 21, color: C.muted }}>wrong answer</div>
      <div style={{ fontFamily: mono, fontSize: 42, color: win ? C.accent : C.ink }}>{wrongScore}</div>
    </div>
  </div>
);

const ResultRow = ({ method, instruct, base, bestI = false, bestB = false }: { method: string; instruct: string; base: string; bestI?: boolean; bestB?: boolean }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '1.5fr .8fr .8fr', alignItems: 'center', minHeight: 92, borderTop: `1px solid ${C.line}` }}>
    <div style={{ fontSize: 30, fontWeight: 630 }}>{method}</div>
    <div style={{ fontFamily: mono, fontSize: 38, fontWeight: bestI ? 760 : 520, color: bestI ? C.accent : C.ink }}>{instruct}</div>
    <div style={{ fontFamily: mono, fontSize: 38, fontWeight: bestB ? 760 : 520, color: bestB ? C.accent : C.ink }}>{base}</div>
  </div>
);

const HeatCell = ({ label, value, strength = 0 }: { label: string; value: string; strength?: number }) => (
  <div style={{ minHeight: 104, padding: '19px 22px', borderTop: `1px solid ${C.line}`, background: strength > 0 ? `rgba(223,76,47,${strength})` : 'transparent' }}>
    <div style={{ fontSize: 20, color: C.muted }}>{label}</div>
    <div style={{ marginTop: 8, fontFamily: mono, fontSize: 34, fontWeight: 700, color: strength > .2 ? C.panel : C.ink }}>{value}</div>
  </div>
);

const CostBar = ({ label, auroc, cost, width }: { label: string; auroc: string; cost: string; width: number }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr 150px 175px', gap: 20, alignItems: 'center', minHeight: 91, borderTop: `1px solid ${C.line}` }}>
    <div style={{ fontSize: 29, fontWeight: 630 }}>{label}</div>
    <div style={{ height: 18, background: C.inkSoft, position: 'relative' }}>
      <div style={{ width, height: '100%', background: C.accent }} />
    </div>
    <div style={{ fontFamily: mono, fontSize: 31 }}>{auroc}</div>
    <div style={{ fontFamily: mono, fontSize: 25, color: C.muted }}>{cost}</div>
  </div>
);

const Cover: Page = () => (
  <section style={basePage}>
    <DeckStyles />
    <Grid />
    <div style={{ position: 'absolute', top: 0, bottom: 0, left: 0, width: 24, background: C.accent }} />
    <div style={{ position: 'absolute', left: 145, top: 132, width: 1040 }}>
      <Eyebrow>Tiny CRCV Lab · inspectable research prototype</Eyebrow>
      <h1 style={{ margin: '42px 0 0', fontFamily: 'var(--osd-font-display)', fontSize: 'var(--osd-size-hero)', lineHeight: .93, letterSpacing: '-.065em', fontWeight: 770 }}>
        Can a tiny model<br />warn us when<br />it is <Accent>wrong?</Accent>
      </h1>
      <p style={{ margin: '46px 0 0', maxWidth: 920, fontSize: 34, lineHeight: 1.35, color: C.muted }}>
        A visual tour from tokens and hidden states to calibrated hallucination-risk scores.
      </p>
    </div>
    <div style={{ position: 'absolute', right: 135, top: 152, width: 470, height: 680, borderLeft: `1px solid ${C.line}`, paddingLeft: 54 }}>
      <div style={{ fontFamily: mono, fontSize: 118, fontWeight: 760, letterSpacing: '-.08em', color: C.accent }}>0.5B</div>
      <div style={{ marginTop: 8, fontSize: 30, fontWeight: 660 }}>parameters</div>
      <div style={{ marginTop: 80, display: 'grid', gap: 34 }}>
        <Stat value="600" label="questions" detail="three QA datasets" />
        <Stat value="5" label="risk scores" detail="all arithmetic exposed" />
      </div>
    </div>
    <Footer chapter="opening" />
  </section>
);

const Motivation: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Motivation</Eyebrow>
    <Heading width={1280}>Fluency is visible. <Accent>Reliability is not.</Accent></Heading>
    <div style={{ display: 'grid', gridTemplateColumns: '1.05fr .95fr', gap: 90, marginTop: 65 }}>
      <div style={{ padding: 40, background: C.panel, borderTop: `4px solid ${C.ink}` }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>QUESTION</div>
        <div style={{ marginTop: 22, fontSize: 34, lineHeight: 1.35 }}>Who is under Darth Vader’s mask?</div>
        <div style={{ marginTop: 55, fontFamily: mono, fontSize: 22, color: C.muted }}>MODEL ANSWER</div>
        <div style={{ marginTop: 22, fontSize: 62, fontWeight: 720, letterSpacing: '-.045em' }}>Luke Skywalker</div>
        <div style={{ marginTop: 28 }}><Pill>short · grammatical · wrong</Pill></div>
      </div>
      <Steps>
        <Step><Claim number="01" title="Generation optimizes the next token" text="It does not execute an explicit fact-check before speaking." /></Step>
        <Step><Claim number="02" title="A confident surface can hide weak evidence" text="The answer alone does not tell us how the model arrived there." /></Step>
        <Step><Claim number="03" title="Can the computation leave useful clues?" text="We test cheap, reference-free signals already near the forward pass." /></Step>
      </Steps>
    </div>
    <Footer chapter="why detect risk?" />
  </section>
);

const Problem: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>The problem we actually solve</Eyebrow>
    <Heading>Estimate <Accent>risk</Accent> without pretending to know truth.</Heading>
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginTop: 92 }}>
      <Block label="Question" detail="q" width={270} />
      <Arrow />
      <Block label="Generated answer" detail="a₁ … aₙ" width={310} />
      <Arrow />
      <Block label="Side signals" detail="pₜ, entropy, hₜ, samples" width={360} />
      <Arrow />
      <Block label="Risk score" detail="s(q,a) ∈ [0,1] or ℝ" accent width={300} />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 70, marginTop: 80 }}>
      <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 24 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>TARGET IN THIS EXPERIMENT</div>
        <div style={{ marginTop: 18, fontSize: 33, lineHeight: 1.42 }}>Higher score when the answer contains no accepted answer alias.</div>
      </div>
      <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 24 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>NOT A TRUTH ORACLE</div>
        <div style={{ marginTop: 18, fontSize: 33, lineHeight: 1.42, color: C.muted }}>It can rank suspicious answers; it cannot prove a claim false.</div>
      </div>
    </div>
    <Footer chapter="problem formulation" />
  </section>
);

const DataDesign: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Dataset design</Eyebrow>
    <Heading>Three kinds of factual pressure. <Accent>One frozen split.</Accent></Heading>
    <div style={{ marginTop: 64 }}>
      <DatasetBand label="NQ-Open" count="200" character="Natural search questions; short accepted answers." />
      <DatasetBand label="TriviaQA" count="200" character="Trivia-style entities, names, dates, and aliases." />
      <DatasetBand label="TruthfulQA" count="200" character="Misconception-prone questions designed to tempt falsehoods." />
    </div>
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 30, marginTop: 52 }}>
      <SplitBox label="calibration" count="300" detail="100 per dataset · choose thresholds" />
      <div style={{ display: 'flex', alignItems: 'center' }}><Arrow width={100} /></div>
      <SplitBox label="held out" count="300" detail="100 per dataset · report once" accent />
    </div>
    <Footer chapter="600 questions" />
  </section>
);

const Models: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Two checkpoints, same architecture</Eyebrow>
    <Heading>Does instruction tuning change what the signals mean?</Heading>
    <div style={{ display: 'flex', gap: 42, marginTop: 76 }}>
      <ModelTower label="Instruct" prompt={'System: Answer factual questions directly.\nUser: Give only the short answer.'} emphasis />
      <ModelTower label="Base" prompt={'Question: {question}\nAnswer:'} />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 34, marginTop: 80 }}>
      <Stat value="24" label="layers" detail="decoder blocks" />
      <Stat value="896" label="hidden width" detail="numbers per token" />
      <Stat value="14" label="attention heads" detail="2 key/value heads" />
      <Stat value="151,936" label="vocabulary" detail="candidate tokens" />
    </div>
    <div style={{ position: 'absolute', right: PAD, bottom: 88, fontFamily: mono, fontSize: 18, color: C.muted }}>architecture: Qwen/Qwen2.5-0.5B config</div>
    <Footer chapter="the tiny language model" />
  </section>
);

const ModelAnatomy: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper, padding: `${PAD}px` }}>
    <DeckStyles />
    <Grid dark />
    <Eyebrow dark>Minified decoder-only LLM</Eyebrow>
    <Heading dark width={1460}>A token becomes a vector, then is <Accent>rewritten 24 times.</Accent></Heading>
    <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 90 }}>
      <Block label="Token IDs" detail="integers" width={220} />
      <FlowArrow width={92} dark />
      <Block label="Embedding" detail="ID → 896 values" width={270} />
      <FlowArrow width={92} dark />
      <Block label="Transformer ×24" detail="attention + MLP + residuals" accent width={360} />
      <FlowArrow width={92} dark />
      <Block label="Final hidden" detail="hₜ ∈ ℝ⁸⁹⁶" width={270} />
      <FlowArrow width={92} dark />
      <Block label="LM head" detail="151,936 logits" width={270} />
    </div>
    <div style={{ marginTop: 86, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60 }}>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 28 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>ATTENTION</div>
        <div style={{ marginTop: 15, fontSize: 31, lineHeight: 1.42, color: C.darkMuted }}>Which earlier tokens matter for this position?</div>
      </div>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 28 }}>
        <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>MLP</div>
        <div style={{ marginTop: 15, fontSize: 31, lineHeight: 1.42, color: C.darkMuted }}>How should the representation change?</div>
      </div>
    </div>
    <Footer chapter="inside qwen2.5-0.5b" dark />
  </section>
);

const WalkInput: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper }}>
    <DeckStyles />
    <Grid dark />
    <div style={{ position: 'absolute', left: PAD, top: 95 }}><Eyebrow dark>One recorded question · phase 1</Eyebrow></div>
    <QuestionMorph left={150} top={230} width={1620} />
    <div style={{ position: 'absolute', left: 230, top: 515, display: 'flex', alignItems: 'center', gap: 22 }}>
      <Block label="who" detail="token id" width={205} />
      <Block label=" is" detail="token id" width={205} />
      <Block label=" under" detail="token id" width={205} />
      <Block label=" the" detail="token id" width={205} />
      <Block label=" mask" detail="token id" width={205} />
      <Block label=" …" detail="more tokens" width={205} />
    </div>
    <div style={{ position: 'absolute', left: 480, top: 760, display: 'flex', alignItems: 'center', gap: 25 }}>
      <Pill tone="dark">text</Pill><FlowArrow dark /><Pill tone="accent">tokenizer</Pill><FlowArrow dark /><Pill tone="dark">integer sequence</Pill>
    </div>
    <Footer chapter="animated pass · tokenize" dark />
  </section>
);

const WalkModel: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper }}>
    <DeckStyles />
    <Grid dark />
    <div style={{ position: 'absolute', left: PAD, top: 68 }}><Eyebrow dark>One recorded question · phase 2</Eyebrow></div>
    <QuestionMorph left={150} top={125} width={1180} compact />
    <div style={{ position: 'absolute', left: 150, top: 270, width: 1620, height: 610, borderTop: `1px solid ${C.darkLine}`, borderBottom: `1px solid ${C.darkLine}` }}>
      <div style={{ position: 'absolute', left: 0, top: 36, width: 270 }}><Block label="Embedding" detail="each token → 896D" width={270} /></div>
      <div style={{ position: 'absolute', left: 300, top: 92 }}><FlowArrow dark /></div>
      <div style={{ position: 'absolute', left: 450, top: 4, width: 610, height: 570, border: `1px solid ${C.accent}`, borderRadius: 18, background: C.darkPanel, padding: 38 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}><div style={{ fontSize: 36, fontWeight: 720 }}>Transformer block</div><Pill tone="accent">repeat ×24</Pill></div>
        <div style={{ marginTop: 45, display: 'grid', gridTemplateColumns: '1fr 70px 1fr', alignItems: 'center', gap: 14 }}>
          <Block label="Attention" detail="mix context" width={205} />
          <Arrow width={65} dark />
          <Block label="Residual + norm" detail="retain + stabilize" width={225} />
        </div>
        <div style={{ marginTop: 30, display: 'grid', gridTemplateColumns: '1fr 70px 1fr', alignItems: 'center', gap: 14 }}>
          <Block label="MLP" detail="transform" width={205} />
          <Arrow width={65} dark />
          <Block label="Residual + norm" detail="retain + stabilize" width={225} />
        </div>
      </div>
      <div style={{ position: 'absolute', left: 1090, top: 92 }}><FlowArrow dark /></div>
      <div style={{ position: 'absolute', right: 0, top: 36 }}><Block label="Final hidden state" detail="one 896D vector per position" accent width={350} /></div>
    </div>
    <Footer chapter="animated pass · layers" dark />
  </section>
);

const WalkTokenOne: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper }}>
    <DeckStyles />
    <Grid dark />
    <div style={{ position: 'absolute', left: PAD, top: 68 }}><Eyebrow dark>One recorded question · phase 3</Eyebrow></div>
    <QuestionMorph left={150} top={125} width={1030} compact />
    <div style={{ position: 'absolute', left: 150, top: 300, width: 910 }}>
      <Heading dark width={850}>The model chooses the <Accent>next token.</Accent></Heading>
      <div style={{ marginTop: 52, fontSize: 30, lineHeight: 1.5, color: C.darkMuted }}>A linear head turns the final 896D state into one score per vocabulary token; softmax makes probabilities.</div>
    </div>
    <div style={{ position: 'absolute', right: 150, top: 280, width: 600, minHeight: 450, borderTop: `4px solid ${C.accent}`, paddingTop: 25 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 150px', minHeight: 80, alignItems: 'center', borderBottom: `1px solid ${C.darkLine}`, fontFamily: mono, fontSize: 28 }}><span>selected: Luke</span><strong style={{ color: C.accent }}>.2161</strong></div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 150px', minHeight: 80, alignItems: 'center', borderBottom: `1px solid ${C.darkLine}`, fontFamily: mono, fontSize: 28 }}><span>runner-up token</span><span>.1042</span></div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 150px', minHeight: 80, alignItems: 'center', borderBottom: `1px solid ${C.darkLine}`, fontFamily: mono, fontSize: 28 }}><span>all other tokens</span><span>.6797</span></div>
      <div style={{ marginTop: 24, fontFamily: mono, fontSize: 21, color: C.darkMuted }}>runner-up derived exactly from top-1 probability − recorded top-2 margin</div>
    </div>
    <TokenMorph id="walk-luke" text="Luke" left={1210} top={775} width={250} accent />
    <div style={{ position: 'absolute', left: 150, top: 790, fontFamily: mono, fontSize: 25, color: C.darkMuted }}>chosen-token confidence c₁ = 0.2161</div>
    <Footer chapter="animated pass · logits → softmax" dark />
  </section>
);

const WalkTokenTwo: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper }}>
    <DeckStyles />
    <Grid dark />
    <div style={{ position: 'absolute', left: PAD, top: 68 }}><Eyebrow dark>One recorded question · phase 4</Eyebrow></div>
    <QuestionMorph left={150} top={125} width={1030} compact />
    <div style={{ position: 'absolute', left: 150, top: 295, width: 830 }}>
      <Heading dark width={820}>Append. Recompute. <Accent>Repeat.</Accent></Heading>
      <div style={{ marginTop: 58, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 30 }}>
        <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 22 }}><div style={{ fontFamily: mono, fontSize: 21, color: C.darkMuted }}>NEW CONFIDENCE</div><div style={{ marginTop: 13, fontFamily: mono, fontSize: 56 }}>.9388</div></div>
        <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 22 }}><div style={{ fontFamily: mono, fontSize: 21, color: C.darkMuted }}>HIDDEN SHIFT</div><div style={{ marginTop: 13, fontFamily: mono, fontSize: 56, color: C.accent }}>1.1125</div></div>
      </div>
      <div style={{ marginTop: 54, fontSize: 30, color: C.darkMuted, lineHeight: 1.5 }}>The second token is locally confident—even though the two-token answer is factually wrong.</div>
    </div>
    <TokenMorph id="walk-luke" text="Luke" left={1110} top={370} width={250} accent />
    <TokenMorph id="walk-skywalker" text=" Skywalker" left={1390} top={370} width={360} accent />
    <div style={{ position: 'absolute', left: 1110, top: 530, width: 640, minHeight: 220, borderTop: `1px solid ${C.darkLine}`, paddingTop: 26 }}>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.darkMuted }}>FINAL ANSWER</div>
      <div style={{ marginTop: 18, fontSize: 50, lineHeight: 1.15, fontWeight: 720 }}>Luke Skywalker</div>
      <div style={{ marginTop: 28 }}><Pill tone="accent">label: incorrect</Pill></div>
    </div>
    <Footer chapter="animated pass · autoregression" dark />
  </section>
);

const WalkTrace: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper, padding: `${PAD}px` }}>
    <DeckStyles />
    <Grid dark />
    <Eyebrow dark>One recorded question · phase 5</Eyebrow>
    <div style={{ position: 'absolute', right: PAD, top: 74 }}><Pill tone="accent">raw trace</Pill></div>
    <QuestionMorph left={118} top={130} width={860} compact />
    <TokenMorph id="walk-luke" text="Luke" left={1100} top={125} width={230} accent />
    <TokenMorph id="walk-skywalker" text=" Skywalker" left={1360} top={125} width={335} accent />
    <div style={{ position: 'absolute', left: PAD, right: PAD, top: 320 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '190px repeat(5, 1fr)', alignItems: 'center', minHeight: 66, fontFamily: mono, fontSize: 20, color: C.darkMuted }}>
        <div>token</div><div>confidence cₜ</div><div>surprise −ln cₜ</div><div>top-2 margin</div><div>hidden shift rₜ</div><div>hidden RMS</div>
      </div>
      <TraceCell token="Luke" confidence=".2161" surprise="1.5319" margin=".1119" shift="—" norm="8.7570" />
      <TraceCell token=" Skywalker" confidence=".9388" surprise=".0632" margin=".9205" shift="1.1125" norm="7.8725" />
    </div>
    <div style={{ position: 'absolute', left: PAD, right: PAD, bottom: 132, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 50 }}>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 20, fontSize: 27, color: C.darkMuted }}>These are observations—not hallucination scores yet.</div>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 20, fontSize: 27 }}>Next: compress the trace into small checks.</div>
    </div>
    <Footer chapter="animated pass · collect numbers" dark />
  </section>
);

const SurpriseMetric: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Check 1 · token confidence</Eyebrow>
    <Heading>Focus on the answer’s <Accent>weakest token decisions.</Accent></Heading>
    <div style={{ display: 'grid', gridTemplateColumns: '1.12fr .88fr', gap: 85, marginTop: 76 }}>
      <div>
        <FormulaLine label="Luke" formula="−ln(0.2161)" value="1.5319" accent />
        <FormulaLine label="Skywalker" formula="−ln(0.9388)" value="0.0632" />
        <FormulaLine label="Top-3 surprise" formula="(1.5319 + 0.0632) / 2" value="0.7976" accent />
        <Threshold score="0.7976" threshold="0.5445" />
      </div>
      <div style={{ padding: 40, background: C.panel, borderTop: `4px solid ${C.accent}` }}>
        <div style={{ fontFamily: mono, fontSize: 23, color: C.accent }}>FORMULA</div>
        <div style={{ marginTop: 28, fontFamily: mono, fontSize: 36, lineHeight: 1.5 }}>T₃ = mean(top 3 of −ln cₜ)</div>
        <div style={{ marginTop: 55, fontSize: 29, lineHeight: 1.5, color: C.muted }}>Why top three? A single fragile content token can be diluted by many easy punctuation or boilerplate tokens.</div>
        <div style={{ marginTop: 46 }}><Pill>zero extra model passes</Pill></div>
      </div>
    </div>
    <Footer chapter="measure · top-3 surprise" />
  </section>
);

const PFalseMetric: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Check 2 · ask the model about itself</Eyebrow>
    <Heading>One extra pass: “Is this answer <Accent>correct?</Accent>”</Heading>
    <div style={{ marginTop: 70, padding: 34, borderTop: `4px solid ${C.ink}`, background: C.panel, fontFamily: mono, fontSize: 28, lineHeight: 1.55 }}>
      Question: who is under the mask of darth vader<br />Proposed answer: Luke Skywalker<br />Is the proposed answer correct? Answer Yes or No.
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.2fr', gap: 40, marginTop: 55 }}>
      <Stat value=".8262" label="P(Yes)" detail="raw next-token probability" />
      <Stat value=".1737" label="P(No)" detail="raw next-token probability" />
      <div style={{ borderTop: `3px solid ${C.accent}`, paddingTop: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 29, color: C.muted }}>.1737 / (.8262 + .1737)</div>
        <div style={{ marginTop: 18, fontFamily: mono, fontSize: 66, fontWeight: 760, color: C.accent }}>.1737</div>
        <div style={{ marginTop: 8, fontSize: 27 }}>normalized P(False)</div>
      </div>
    </div>
    <Threshold score="0.1737" threshold="0.0060" />
    <div style={{ position: 'absolute', right: PAD, bottom: 100, fontSize: 23, color: C.muted }}>This score ranks risk; 0.1737 is not a calibrated 17.37% error probability.</div>
    <Footer chapter="measure · self-check" />
  </section>
);

const DisagreementMetric: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Check 3 · semantic consistency</Eyebrow>
    <Heading>Sample three answers. Measure how much they <Accent>disagree.</Accent></Heading>
    <div style={{ display: 'grid', gridTemplateColumns: '1.05fr .95fr', gap: 90, marginTop: 70 }}>
      <div>
        <SampleAnswer index="A" answer="Anakin Skywalker." />
        <SampleAnswer index="B" answer="R2-D2" />
        <SampleAnswer index="C" answer="Luke Skywalker" />
      </div>
      <div>
        <FormulaLine label="A vs B" formula="P(not same answer)" value="0.3489" />
        <FormulaLine label="A vs C" formula="P(not same answer)" value="0.3916" />
        <FormulaLine label="B vs C" formula="P(not same answer)" value="0.4050" />
        <FormulaLine label="Mean" formula="(.3489 + .3916 + .4050) / 3" value="0.3818" accent />
        <Threshold score="0.3818" threshold="0.1867" />
      </div>
    </div>
    <div style={{ position: 'absolute', left: PAD, bottom: 94 }}><Pill>+3 generations · +3 pair judgments · about 1.90 s/question</Pill></div>
    <Footer chapter="measure · 3-answer disagreement" />
  </section>
);

const HiddenProbeMetric: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Check 4 · supervised internal-state probe</Eyebrow>
    <Heading>Can one linear layer decode <Accent>error risk</Accent> from h?</Heading>
    <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginTop: 74 }}>
      <Block label="Mean hidden state" detail="896 raw values" width={310} />
      <Arrow width={80} />
      <Block label="Standardize" detail="(hⱼ − μⱼ) / σⱼ" width={300} />
      <Arrow width={80} />
      <Block label="Linear score" detail="b + Σⱼ wⱼzⱼ" width={300} />
      <Arrow width={80} />
      <Block label="Sigmoid" detail="1 / (1 + e⁻ˣ)" accent width={270} />
    </div>
    <div style={{ marginTop: 66, padding: 36, background: C.panel, borderTop: `4px solid ${C.accent}` }}>
      <div style={{ fontFamily: mono, fontSize: 25, lineHeight: 1.7, color: C.muted }}>
        z = 2.085 + .508 − .458 − .416 + .411 − .395 − .394 + .363 − .361 + 2.868
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 32, marginTop: 22 }}>
        <div style={{ fontFamily: mono, fontSize: 34 }}>logit = 4.211</div>
        <Arrow />
        <div style={{ fontFamily: mono, fontSize: 76, fontWeight: 760, color: C.accent }}>risk = 0.9854</div>
      </div>
    </div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 38 }}>
      <Threshold score="0.9854" threshold="0.4373" />
      <Pill>trained only on calibration labels</Pill>
    </div>
    <Footer chapter="measure · mean-hidden probe" />
  </section>
);

const Scorecard: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>The same wrong answer, five views</Eyebrow>
    <Heading>Four checks flag it. The length control does not.</Heading>
    <div style={{ marginTop: 66 }}>
      <FormulaLine label="Top-3 surprise" formula="0 extra passes" value="0.7976 · FLAG" accent />
      <FormulaLine label="P(False)" formula="1 extra pass" value="0.1737 · FLAG" accent />
      <FormulaLine label="Hidden probe" formula="896D → sigmoid" value="0.9854 · FLAG" accent />
      <FormulaLine label="3-answer disagreement" formula="6 extra passes" value="0.3818 · FLAG" accent />
      <FormulaLine label="Answer length" formula="2 generated tokens" value="2 · PASS" />
    </div>
    <div style={{ marginTop: 42, fontSize: 31, lineHeight: 1.45, color: C.muted }}>One anecdote only shows arithmetic. Benchmarking asks whether the ranking survives hundreds of unseen questions.</div>
    <Footer chapter="from trace to checks" />
  </section>
);

const Calibration: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Calibration turns a score into a decision</Eyebrow>
    <Heading>Choose once. <Accent>Freeze before test.</Accent></Heading>
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginTop: 75 }}>
      <SplitBox label="calibration questions" count="300" detail="scores + labels visible" />
      <FlowArrow width={120} />
      <div style={{ width: 500, minHeight: 260, padding: 38, borderTop: `5px solid ${C.accent}`, background: C.accentSoft }}>
        <Eyebrow>search thresholds</Eyebrow>
        <div style={{ marginTop: 25, fontFamily: mono, fontSize: 34, lineHeight: 1.5 }}>τ* = argmaxτ macro-F1(τ)</div>
        <div style={{ marginTop: 23, fontSize: 27, color: C.muted }}>For top-3 surprise: τ* = 0.5445</div>
      </div>
      <FlowArrow width={120} />
      <div style={{ width: 300 }}><Stat value="300" label="held out" detail="threshold locked" accent /></div>
    </div>
    <div style={{ position: 'relative', height: 240, marginTop: 76, borderBottom: `2px solid ${C.ink}`, borderLeft: `1px solid ${C.line}` }}>
      <div style={{ position: 'absolute', left: '26%', bottom: 0, width: 300, height: 100, background: C.inkSoft, borderRadius: '150px 150px 0 0' }} />
      <div style={{ position: 'absolute', left: '50%', bottom: 0, width: 470, height: 190, background: C.accentSoft, borderRadius: '235px 235px 0 0' }} />
      <div style={{ position: 'absolute', left: '49%', top: 0, bottom: 0, width: 4, background: C.accent }} />
      <div style={{ position: 'absolute', left: '49%', top: -37, fontFamily: mono, fontSize: 23, color: C.accent }}>τ = .5445</div>
      <div style={{ position: 'absolute', left: '28%', bottom: 35, fontSize: 25 }}>more correct</div>
      <div style={{ position: 'absolute', left: '67%', bottom: 45, fontSize: 25, color: C.accent }}>more wrong</div>
    </div>
    <Footer chapter="calibration" />
  </section>
);

const AuRoc: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Primary benchmark metric</Eyebrow>
    <Heading>AUROC asks one clean <Accent>ranking question.</Accent></Heading>
    <div style={{ display: 'grid', gridTemplateColumns: '1.02fr .98fr', gap: 90, marginTop: 70 }}>
      <div>
        <AuPair correctScore="0.18" wrongScore="0.80" win />
        <AuPair correctScore="0.71" wrongScore="0.42" win={false} />
        <AuPair correctScore="0.33" wrongScore="0.77" win />
      </div>
      <div style={{ padding: 42, background: C.panel, borderTop: `4px solid ${C.accent}` }}>
        <div style={{ fontFamily: mono, fontSize: 25, color: C.accent }}>DEFINITION</div>
        <div style={{ marginTop: 28, fontSize: 37, lineHeight: 1.38 }}>Draw one wrong and one correct answer.</div>
        <div style={{ marginTop: 30, fontSize: 37, lineHeight: 1.38 }}>How often is the wrong answer scored higher?</div>
        <div style={{ marginTop: 50, fontFamily: mono, fontSize: 72, fontWeight: 760, color: C.accent }}>AUROC = 0.656</div>
        <div style={{ marginTop: 14, fontSize: 26, color: C.muted }}>≈ 65.6 wins per 100 random pairs</div>
      </div>
    </div>
    <div style={{ marginTop: 52, fontSize: 29, color: C.muted }}>0.5 is chance ranking. AUROC is threshold-free, but it is not a probability of factual correctness.</div>
    <Footer chapter="evaluation" />
  </section>
);

const Results: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Held-out pooled results · 300 questions per model</Eyebrow>
    <Heading>Simple wins descriptively. <Accent>Nothing wins reliably.</Accent></Heading>
    <div style={{ marginTop: 55 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr .8fr .8fr', alignItems: 'center', minHeight: 64, fontFamily: mono, fontSize: 22, color: C.muted }}><div>AUROC</div><div>INSTRUCT</div><div>BASE</div></div>
      <ResultRow method="Top-3 token surprise" instruct="0.656" base="0.599" bestI />
      <ResultRow method="P(False) self-check" instruct="0.603" base="0.628" bestB />
      <ResultRow method="3-answer disagreement" instruct="0.621" base="0.595" />
      <ResultRow method="Mean-hidden linear probe" instruct="0.549" base="0.545" />
      <ResultRow method="Answer length control" instruct="0.540" base="0.528" />
    </div>
    <div style={{ display: 'flex', gap: 24, marginTop: 36 }}><Pill tone="accent">best descriptive result</Pill><Pill>paired 95% intervals for every improvement include zero</Pill></div>
    <Footer chapter="benchmark · pooled" />
  </section>
);

const DatasetResults: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Held-out AUROC by dataset</Eyebrow>
    <Heading>The detector’s behavior is <Accent>context dependent.</Accent></Heading>
    <div style={{ display: 'grid', gridTemplateColumns: '280px repeat(3, 1fr)', columnGap: 22, marginTop: 70 }}>
      <div />
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>NQ-OPEN</div>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>TRIVIAQA</div>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>TRUTHFULQA</div>
      <div style={{ paddingTop: 30, fontSize: 29, fontWeight: 680 }}>Instruct · Top-3</div>
      <HeatCell label="84 wrong / 100" value="0.594" strength={.12} />
      <HeatCell label="79 wrong / 100" value="0.673" strength={.22} />
      <HeatCell label="94 wrong / 100" value="0.784" strength={.36} />
      <div style={{ paddingTop: 30, fontSize: 29, fontWeight: 680 }}>Instruct · P(False)</div>
      <HeatCell label="84 wrong / 100" value="0.615" strength={.15} />
      <HeatCell label="79 wrong / 100" value="0.701" strength={.27} />
      <HeatCell label="94 wrong / 100" value="0.360" />
      <div style={{ paddingTop: 30, fontSize: 29, fontWeight: 680 }}>Base · Top-3</div>
      <HeatCell label="88 wrong / 100" value="0.576" strength={.08} />
      <HeatCell label="84 wrong / 100" value="0.533" strength={.04} />
      <HeatCell label="100 wrong / 100" value="—" />
      <div style={{ paddingTop: 30, fontSize: 29, fontWeight: 680 }}>Base · P(False)</div>
      <HeatCell label="88 wrong / 100" value="0.515" strength={.02} />
      <HeatCell label="84 wrong / 100" value="0.636" strength={.17} />
      <HeatCell label="100 wrong / 100" value="—" />
    </div>
    <div style={{ marginTop: 44, fontSize: 27, color: C.muted }}>“—” means AUROC is undefined: Base produced no strict-alias correct answers on that TruthfulQA test slice.</div>
    <Footer chapter="benchmark · slices" />
  </section>
);

const CostAnalysis: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>Instruct model · quality versus incremental cost</Eyebrow>
    <Heading>More computation did not buy a <Accent>clear improvement.</Accent></Heading>
    <div style={{ marginTop: 60 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr 150px 175px', gap: 20, minHeight: 55, alignItems: 'center', fontFamily: mono, fontSize: 20, color: C.muted }}><div>METHOD</div><div>AUROC ABOVE CHANCE</div><div>AUROC</div><div>EXTRA / Q</div></div>
      <CostBar label="Top-3 surprise" auroc="0.656" cost="0.000 s" width={440} />
      <CostBar label="3-answer disagreement" auroc="0.621" cost="1.901 s" width={340} />
      <CostBar label="P(False)" auroc="0.603" cost="0.130 s" width={290} />
      <CostBar label="Hidden probe" auroc="0.549" cost="0.00007 s" width={140} />
      <CostBar label="Answer length" auroc="0.540" cost="0.000 s" width={115} />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 55, marginTop: 54 }}>
      <div style={{ borderTop: `3px solid ${C.accent}`, paddingTop: 20, fontSize: 30, lineHeight: 1.4 }}>Top-3 surprise is already emitted by greedy generation.</div>
      <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 20, fontSize: 30, lineHeight: 1.4, color: C.muted }}>Disagreement costs roughly fifteen times more than P(False).</div>
    </div>
    <Footer chapter="analysis · speed" />
  </section>
);

const Analysis: Page = () => (
  <section style={{ ...basePage, padding: `${PAD}px` }}>
    <DeckStyles />
    <Eyebrow>What the experiment teaches</Eyebrow>
    <Heading>Small signals help—but they are <Accent>not universal.</Accent></Heading>
    <div style={{ marginTop: 70 }}>
      <Claim number="01" title="Confidence tails carry useful signal" text="Instruct top-3 surprise reached 0.656 held-out AUROC at essentially zero incremental cost." />
      <Claim number="02" title="Self-judgment is checkpoint dependent" text="P(False) led descriptively for Base, but trailed top-3 surprise for Instruct." />
      <Claim number="03" title="A flexible probe can memorize calibration" text="The hidden probe scored 1.000 on calibration, then only 0.549 on held-out Instruct questions." />
      <Claim number="04" title="No reliable improvement was established" text="Every paired 95% AUROC-difference interval against top-3 surprise included zero." />
    </div>
    <Footer chapter="analysis · interpretation" />
  </section>
);

const Limits: Page = () => (
  <section style={{ ...basePage, background: C.dark, color: C.paper, padding: `${PAD}px` }}>
    <DeckStyles />
    <Grid dark />
    <Eyebrow dark>Read the number honestly</Eyebrow>
    <Heading dark>This is a compact detector benchmark—<Accent>not a SOTA factuality claim.</Accent></Heading>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60, marginTop: 80 }}>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 25 }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.accent }}>LABEL LIMIT</div>
        <div style={{ marginTop: 17, fontSize: 31, lineHeight: 1.45, color: C.darkMuted }}>Alias matching can mark a semantically correct paraphrase as wrong.</div>
      </div>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 25 }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.accent }}>CLASS IMBALANCE</div>
        <div style={{ marginTop: 17, fontSize: 31, lineHeight: 1.45, color: C.darkMuted }}>The tiny checkpoints were wrong on most held-out questions.</div>
      </div>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 25 }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.accent }}>REFERENCE-FREE CEILING</div>
        <div style={{ marginTop: 17, fontSize: 31, lineHeight: 1.45, color: C.darkMuted }}>Internal confidence cannot retrieve missing evidence from the world.</div>
      </div>
      <div style={{ borderTop: `1px solid ${C.darkLine}`, paddingTop: 25 }}>
        <div style={{ fontFamily: mono, fontSize: 21, color: C.accent }}>NEXT VALIDATION</div>
        <div style={{ marginTop: 17, fontSize: 31, lineHeight: 1.45, color: C.darkMuted }}>Human factuality labels, larger models, retrieval baselines, and repeated seeds.</div>
      </div>
    </div>
    <Footer chapter="limitations" dark />
  </section>
);

const Closing: Page = () => (
  <section style={basePage}>
    <DeckStyles />
    <Grid />
    <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 24, background: C.accent }} />
    <div style={{ position: 'absolute', left: 150, top: 130, width: 1220 }}>
      <Eyebrow>Recommendation</Eyebrow>
      <h2 style={{ margin: '38px 0 0', fontFamily: 'var(--osd-font-display)', fontSize: 108, lineHeight: .96, letterSpacing: '-.062em', fontWeight: 770 }}>
        Keep the cheap baseline.<br /><Accent>Expose every calculation.</Accent>
      </h2>
    </div>
    <div style={{ position: 'absolute', left: 150, bottom: 126, width: 1190, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 34 }}>
      <div style={{ borderTop: `2px solid ${C.ink}`, paddingTop: 18 }}><div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>01 · DEFAULT</div><div style={{ marginTop: 14, fontSize: 27, fontWeight: 680 }}>Top-3 surprise</div><div style={{ marginTop: 9, fontSize: 22, lineHeight: 1.35, color: C.muted }}>Fast, inspectable warning signal.</div></div>
      <div style={{ borderTop: `2px solid ${C.ink}`, paddingTop: 18 }}><div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>02 · OPTIONAL</div><div style={{ marginTop: 14, fontSize: 27, fontWeight: 680 }}>P(False)</div><div style={{ marginTop: 9, fontSize: 22, lineHeight: 1.35, color: C.muted }}>One-pass second opinion.</div></div>
      <div style={{ borderTop: `2px solid ${C.ink}`, paddingTop: 18 }}><div style={{ fontFamily: mono, fontSize: 20, color: C.accent }}>03 · LIMIT</div><div style={{ marginTop: 14, fontSize: 27, fontWeight: 680 }}>Triage, not truth</div><div style={{ marginTop: 9, fontSize: 22, lineHeight: 1.35, color: C.muted }}>Escalate suspicious answers.</div></div>
    </div>
    <div style={{ position: 'absolute', right: 135, top: 190, width: 390, borderTop: `4px solid ${C.accent}`, paddingTop: 28 }}>
      <div style={{ fontFamily: mono, fontSize: 22, color: C.muted }}>RUN THE LAB</div>
      <div style={{ marginTop: 24, fontSize: 32, lineHeight: 1.32, fontWeight: 660 }}>nipunbatra.github.io/<br />tiny-crcv-lab/</div>
      <div style={{ marginTop: 60, fontFamily: mono, fontSize: 22, color: C.muted }}>ALL 600 TRACES</div>
      <div style={{ marginTop: 16, fontSize: 27, lineHeight: 1.4 }}>Search any question and inspect all five scores.</div>
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

const walkMorph: SlideTransition = {
  duration: 300,
  exit: {
    duration: 220,
    easing: 'cubic-bezier(0.4, 0, 1, 1)',
    keyframes: [{ opacity: 1 }, { opacity: 0 }],
  },
  enter: {
    duration: 300,
    delay: 110,
    easing: 'cubic-bezier(0, 0, 0.2, 1)',
    keyframes: [{ opacity: 0 }, { opacity: 1 }],
  },
  morph: {
    duration: 720,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
};

WalkInput.transition = walkMorph;
WalkModel.transition = walkMorph;
WalkTokenOne.transition = walkMorph;
WalkTokenTwo.transition = walkMorph;
WalkTrace.transition = walkMorph;

export const transition: SlideTransition = quietTransition;

export const meta: SlideMeta = {
  title: 'Can a Tiny Model Warn Us When It Is Wrong?',
  createdAt: '2026-07-19T01:46:38.124Z',
};

export const notes = {
  0: 'Open with the narrow claim: this is a fast warning system, not a factuality oracle.',
  1: 'Pause on the Luke Skywalker answer. Ask the audience what in the surface text reveals the error.',
  2: 'Emphasize ranking suspicious answers, not proving truth without external evidence.',
  3: 'The split is by question: 300 calibration, 300 held out, balanced across the three datasets.',
  4: 'This is deliberately minified: embedding, repeated transformer blocks, final hidden state, vocabulary logits.',
  5: 'Advance after describing tokenization. The following pages preserve the same question through a morph sequence.',
  6: 'Attention mixes context; the MLP transforms each position; residuals carry information forward.',
  7: 'The runner-up probability is derived exactly from the selected-token probability minus the recorded top-2 margin.',
  8: 'Everything from this point is arithmetic over this raw trace or a small extra model call.',
  9: 'There are only two answer tokens, so top-3 surprise averages both. This example crosses the calibrated threshold.',
  10: 'P(False) is normalized over the Yes/No tokens. It should not be read as a calibrated probability of error.',
  11: 'The displayed probe sum groups the remaining 888 coordinates into the final aggregate contribution.',
  12: 'Threshold selection sees calibration only. The held-out set is untouched until final reporting.',
  13: 'Translate 0.656 as a ranking frequency—not 65.6% accuracy.',
  14: 'Bold values are descriptive maxima. Paired bootstrap intervals prevent us from calling either a reliable win.',
  15: 'The missing Base TruthfulQA AUROC is a one-class problem, not a software error.',
  16: 'The compute-heavy disagreement method did not outperform the free confidence-tail baseline.',
  17: 'The calibration-to-test collapse of the hidden probe is the clearest warning against flexible small-data probes.',
  18: 'State the four limits before anyone interprets this as a production factuality system.',
  19: 'End with the operational recommendation and invite the audience to inspect any of the 600 traces on the live site.',
};

export default [
  Cover,
  Motivation,
  Problem,
  DataDesign,
  ModelAnatomy,
  WalkInput,
  WalkModel,
  WalkTokenOne,
  WalkTrace,
  SurpriseMetric,
  PFalseMetric,
  HiddenProbeMetric,
  Calibration,
  AuRoc,
  Results,
  DatasetResults,
  CostAnalysis,
  Analysis,
  Limits,
  Closing,
] satisfies Page[];
