# MiniMax M3 Electron tool and math checkpoint

Date: 2026-07-23

Status: `FIXED+VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

Source head and runtime provenance:

- Source head for the retained renderer/API proof:
  `f34deae28b84e9e1cf0f1f1f7055127a72d16581`.
- Exact bundle:
  `/Volumes/EricsLLMDrive/JANGQ-AI/MiniMax-M3-Coder-Small`.
- Real isolated Electron profile:
  `/Users/eric/.vmlx-r17-consolidation-dev`, CDP `9335`, gateway `8090`.
- The real Sessions-card Start button launched PID `11370` on direct port
  `8008`.
- The app log resolved the project engine at
  `/Users/eric/mlx/vllm-mlx-r17-consolidation/.venv/bin/python3`.
- The launch argv selected both `minimax_m3` reasoning and tool parsers,
  Paged RAM, 64-token blocks, 1,000 maximum blocks, and 10 GB Block Disk
  Cache.

## Defect one: repeated execution of an exactly-once tool

The live M3 turn requested one `file_info(panel/package.json)` call. Before
`b482bec60`, the panel executed three calls because exactly-once retirement
was incorrectly coupled to the separate phrase `reply exactly`.

The correction:

- `requestedOnceToolNames()` treats each explicitly named `exactly once` tool
  as an execution invariant.
- The panel retires that tool schema after its first successful execution.
- A duplicate emitted in the same model pass is rejected before it reaches
  the executor.
- Other tool schemas remain available for genuinely multi-tool agentic work.

Focused remote verification:

- `panel/tests/tool-auto-continue.test.ts`: `23/23` passed.
- Combined tool-auto-continue and chat-metrics selection: `26/26` passed.
- Panel TypeScript typecheck passed.

Live Electron proof after the correction:

- One `file_info` card, one execution, one tool-result continuation, and one
  non-empty final answer.
- The visible result truthfully reported `5.2 KB`.
- Metrics: 89 output tokens, `17.4 tok/s`, 9,418 cumulative prompt tokens,
  5,440 cached tokens, `27.07s` TTFT, and `36.8s` total.

## Defect two: malformed repeated TeX opener

The third M3 turn did not suffer random dollar injection. The raw stored
assistant text contained:

```text
Path: panel/package.json — Size: 5.2 KB
$43 and inline TeX: \(\(47 \times 19 = 893 < 920 = 46 \times 20\)
```

`$43` was explicitly required as literal currency by the user prompt. The
actual malformed portion was the model-emitted duplicate `\(\(` opener with
only one closer. KaTeX correctly rejected that invalid nesting, and the prior
fallback therefore left visibly broken punctuation.

Commit `0592404d8` adds one renderer-resilience rule shared by completed
answers and the actively streaming reasoning rail:

- collapse only immediately adjacent repeated `\(`/`\)`, or `\[`/`\]`
  delimiter tokens;
- leave currency, ordinary parentheses, code spans, and code fences intact;
- keep raw API bytes unchanged.

Focused remote verification:

- `panel/tests/math-markdown.test.ts`: `17/17` passed.
- Panel TypeScript typecheck passed.

Live current-source renderer proof:

- The saved malformed third turn hot-reloaded through the exact current
  renderer.
- DOM inspection found `.math-inline .katex` with readable text
  `47×19=893<920=46×20`.
- DOM inspection found no `.math-fallback`, no raw `\times`, and no raw math
  delimiter.
- The expanded Reasoning rail also rendered the malformed stored expression
  as KaTeX; literal `$43` remained ordinary currency.

## Defect three: escaped Unicode operator inside TeX

The raw direct and gateway protocol probe asked the model to preserve
`\times`. M3 instead emitted the same malformed sequence on every route:

```text
M3-RAW-MATH
CURRENCY=$43 TEX=\(47 \× 19 = 893\)
```

Source inspection confirmed the M3 reasoning parser only partitions
`<mm:think>`/`<think>` markers and the M3 tool parser only strips its own
control envelope. Neither rewrites TeX or Unicode operators. The byte sequence
is model-owned, while the route-to-route equality proves it was not injected
or corrupted by one API adapter.

Commit `f34deae28` adds a presentation-only normalization for a stray slash
before known Unicode math glyphs. It applies inside KaTeX source and the
actively streaming reasoning text; raw API bytes remain unchanged.

Live proof:

- Direct and gateway Chat, Responses, Anthropic, and Ollama each emitted 24
  progressive content deltas with identical raw content and truthful
  terminals.
- All routes preserved `$43`, `\(`/`\)`, and the same `\×`; no route emitted
  KaTeX/HTML.
- The exact-copy diagnostic remains truthfully false because the model emitted
  `\×`, not the requested `\times`.
- A real follow-up Electron turn rendered `47×19=893` as `.math-inline .katex`
  in both answer and reasoning. DOM had no fallback, raw `\×`, or raw
  `\times`.

## Three-turn Electron evidence

1. No-tool math/currency: separate 903-character reasoning rail, 138 visible
   characters, no tool call, `19.5 tok/s`.
2. Exactly-once tool: one real `file_info` execution and one result
   continuation, `17.4 tok/s`.
3. No-tool history recall: exact prior path and size, separate 607-character
   reasoning rail, zero tools, corrected KaTeX rendering, `18.4 tok/s`.

Retained screenshots:

- `m3-ui-turn1-reasoning-math.png`
- `m3-ui-turn2-exact-once-tool.png`
- `m3-ui-turn3-history-math.png`
- `m3-ui-turn3-expanded-reasoning-math.png`
- `r17-m3-ui-escaped-unicode-math-fixed.png`

## Direct and Electron-gateway protocol evidence

- Chat and Responses, direct and gateway:
  - separate progressive reasoning and content deltas;
  - one schema-valid `file_info(panel/package.json)`;
  - one real tool-result continuation;
  - progressive exact final and completed terminal;
  - matching stream and non-stream output.
- Anthropic and Ollama, direct and gateway:
  - separate progressive reasoning;
  - one schema-valid `file_info`;
  - real tool-result continuation;
  - progressive exact final and truthful `message_stop`/`stop`.
- A separate thinking-Off matrix proved all four protocols produce the same
  exact eight-line answer in stream and non-stream modes. No reasoning delta
  appeared while explicitly Off.

Retained raw artifacts:

- `r17-m3-{direct,gateway}-openai-tools-current.json`
- `r17-m3-{direct,gateway}-anthropic-ollama-tools-current.json`
- `r17-m3-{direct,gateway}-protocol-parity-current.json`
- `r17-m3-{direct,gateway}-raw-math-current.json`
- `m3_raw_math_protocol_probe.py`

## Boundary

- The current three-turn Electron text/tool/math/history row is
  `VERIFIED-LIVE-SCOPED`.
- Current direct and gateway Chat/Responses/Anthropic/Ollama streaming,
  reasoning separation, tool continuation, raw-byte, and stream/non-stream
  parity are `VERIFIED-LIVE-SCOPED`.
- M3 exact copying of the requested command spelling remains a model-output
  diagnostic: it produced `\×` consistently instead of `\times`. The product
  does not rewrite those raw API bytes.
- M3 restart/eviction native sparse-cache proof, media, signed-app repetition,
  full suites, packaging, notarization, and publication remain open.
- Overall v1.6.17 remains `PARTIAL / NOT RELEASE-READY`.
