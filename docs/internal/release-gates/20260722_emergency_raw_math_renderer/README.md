# v1.6.16 emergency math-renderer gate

Date: 2026-07-22 (America/Los_Angeles)

Status: `VERIFIED-LIVE_SCOPED / RELEASE CAMPAIGN STILL OPEN`

This gate covers the visible raw-TeX defect reported from the signed v1.6.15
Electron applications. It does not close the separate answer-pass pause/TPS,
multi-model cache, parser, restart, media, full-suite, or packaging gates.

## Confirmed root cause

The immutable `v1.6.15` tag sends assistant and completed-reasoning Markdown
directly to `marked.parse()`. CommonMark consumes backslashes before punctuation
but does not interpret TeX commands or dollar delimiters. Consequently:

- `\(47 \times 2 = 94\)` displays as `(47 \times 2 = 94)`;
- `$...$` and `$$...$$` remain literal dollar signs;
- `\frac`, `\approx`, and similar commands remain raw text.

There are exactly two `marked.parse()` chat surfaces in current renderer source:
`MessageBubble.tsx` and `ReasoningBox.tsx`. Both now route through the shared
`mathMarkdown.ts` renderer. The engine/API/SQLite payload is not rewritten.

## Retained change

- Add KaTeX as the renderer dependency and bundle its CSS/fonts.
- Render `\(...\)`, `\[...\]`, `$...$`, and `$$...$$` only in the Electron
  presentation layer.
- Keep plain currency such as `$43` and comparisons such as `$5<$10` literal.
- Preserve code spans/fences byte-for-byte.
- Normalize common delimiter-free TeX emitted by models into readable symbols;
  the actively streaming reasoning rail uses the same readable plain-text view
  until the completed rail is rendered with KaTeX.
- Continue sanitizing all generated HTML with DOMPurify.

## Source and focused proof

Local source checkout:
`/Users/eric/mlx/vllm-mlx-release-1.6.15`, branch
`codex/v1.6.16-release-campaign-20260722`.

Live proof checkout:
`erics-m5-max.local:/Users/eric/mlx/vllm-mlx-release-1.6.13`, same branch.

Focused proof on both checkouts:

- `panel/tests/math-markdown.test.ts`: 9 passed;
- `npm run typecheck`: passed;
- direct `electron-vite build`: passed and emitted bundled KaTeX CSS/font assets.

The full packaging command remains blocked by the intentional clean-JANG-source
gate because `/Users/eric/jang/jang-tools` currently has tracked changes. That
gate was not bypassed.

## Live Electron proof

The real v1.6.16 dev Electron renderer at CDP `127.0.0.1:9335` was opened on the
running Qwen3.6 27B MXFP4 MTP session. The retained completed turn contained
fractions, powers, multiplication, division, approximation, inline math, and
display math.

DOM evidence after the patch:

- 18 KaTeX roots;
- 10 inline math wrappers;
- 8 display math wrappers;
- zero raw `\times`, `\frac`, `\div`, `\approx`, or `\overline` commands;
- zero raw `\(...\)`, `\[...\]`, or `$$...$$` delimiters.

Visual artifacts:

- `vmlx-r16-katex-proof.png` — assistant answer with rendered fractions,
  superscripts, and display equations;
- `vmlx-r16-katex-reasoning-proof.png` — expanded reasoning rail plus rendered
  assistant answer, with no raw TeX command leakage.

The still-running signed v1.6.15 Sequoia/Tahoe applications are negative
provenance controls only. They cannot demonstrate this v1.6.16 source change.

## Raw API and separate streaming boundary

The retained timed Chat and Responses captures are:

- `qwen-chat-exact-a4.sse.jsonl`;
- `qwen-responses-exact-r2.sse.jsonl`.

They show the current exact-visible containment path keeps private math out of
visible content. They also expose a separate unresolved release blocker: after
the reasoning pass, the bounded answer pass can pause and emit the visible
answer as a single late delta, while usage/TPS blends phases. That behavior is
not classified as a renderer defect and is not closed by this gate.

## Remaining release boundary

Before v1.6.16 can be released, the campaign still requires current-source
multi-turn Electron and raw API proof across the named model families, cache
archetypes, parsers/protocols, restart/settings persistence, SSD/L2 partial
reuse and refault, media, accurate phase-local metrics, full suites, clean
bundling, signing, notarization, installed-app smoke, and public provenance.
