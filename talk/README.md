# Tiny CRCV Lab presentation

The 42-page Open Slide deck lives at
`slides/tiny-hallucination-detector/index.tsx`. It is built into the parent
project's `public/talk/` directory and published with the detector app.

The deck is organized as a beginner-friendly numeric walkthrough: it fixes the
task as judging a `(question, generated answer)` pair, introduces real dataset
rows, traces one held-out correct/wrong pair through the model, and expands each
detector score into its saved inputs and arithmetic before showing calibration
and benchmark results.

## Getting started

```bash
npm ci
npm run dev
```

Then open `/s/tiny-hallucination-detector` in the local Open Slide server.

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the presentation server with hot reload. |
| `npm run build` | Build the static presentation. |
| `npm run preview` | Preview the static build locally. |

Every page renders into a fixed **1920 × 1080** canvas. The source uses only
React, Open Slide primitives, inline SVG/HTML diagrams, and real experiment
outputs from the parent repository.

## Navigation

- Arrow keys / PageUp / PageDown move between pages.
- `F` enters fullscreen play mode; Esc exits.
- In play mode: Space / → next, ← prev.

The parent `npm run build` invokes this build automatically and creates static
route fallbacks so the direct GitHub Pages slide and presenter URLs survive a
browser refresh.
