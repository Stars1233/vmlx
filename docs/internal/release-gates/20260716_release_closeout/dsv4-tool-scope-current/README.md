# DSV4 broad-tool continuation and progressive stream evidence

Scope: current legacy Python/Electron checkout at commit `012c1fe90` on
2026-07-17. This is a scoped agent-loop/stream proof, not a blanket DSV4 or
release pass.

## Source trace

- `vmlx_engine/api/tool_calling.py:181-200` validates DSV4's rendered prompt
  against the same explicitly requested/recent-tool scope used by the native
  encoder. Before this change, validation compared the prompt against all 33
  authorized built-ins and prepended a duplicate fallback prompt.
- `vmlx_engine/api/tool_calling.py:254-264` preserves request-bound fallback
  binding for an explicitly named current-turn tool.
- `vmlx_engine/api/tool_calling.py:582-608` accepts path-safe argument
  characters, so `panel/package.json` is not truncated to `panel`.
- `panel/src/main/ipc/chat.ts:1394-1437,1931-1942,2008-2016` maps an explicit
  current-turn no-tool directive onto standard `tool_choice: "none"` for both
  Responses and Chat Completions and suppresses the generic agent prompt.
- `panel/src/shared/toolAutoContinue.ts` owns the directive-shaped matcher.

## Live negative controls retained

- The same raw Responses history with the complete 33-tool catalog reproduced
  an unbounded literal `response` loop before the server scoping change. With
  only `file_info` authorized, the same history completed, isolating the broad
  catalog/fallback interaction rather than the Electron renderer or official
  quant artifact.
- Electron row 150, before `tool_choice: "none"` propagation, called
  `file_info` again even though the current user turn said not to call a tool.
- The first explicit-tool repair attempt truncated `panel/package.json` to
  `panel`; `dsv4-ui-explicit-fastpath-failure.png` retains that failure.

## Live positive controls

- Raw Responses with the same 33 tools completed after the repair with ten
  progressive content deltas (`S`, `IZE`, ` F`, `IVE`, ` PO`, `INT`, ` TWO`,
  ` KB`, ` D`, `ONE`), matching `response.output_text.done` text
  `SIZE FIVE POINT TWO KB DONE`, and one completed terminal. Input usage fell
  from the duplicate-fallback case (about 839 tokens) to 539 tokens.
- Fresh Electron chat `2d01790d-7d78-4714-b41f-ffed32396240`, row 153,
  executed exactly one real `file_info({"path":"panel/package.json"})`,
  persisted the matching result (`Size: 5.2 KB`), and produced
  `The Size field is 5.2 KB.`. Tool phases were generating, calling,
  executing, result, processing, and done.
- Same-chat rows 156 and 159 contained no tool call/result/status fields after
  explicit no-tool instructions and correctly retained the prior 5.2 KB
  result. Row 159 used two sentences despite a one-sentence request, so strict
  format remains `PARTIAL`.
- Long Electron row 162 emitted 470 tokens with 1,595 observed DOM mutations.
  Reasoning started painting at about 2.33s; visible content grew progressively
  from 16.732s through 27.336s. It produced all eight numbered comparisons and
  no tool status. Its requested terminal marker was not byte exact
  (`D4 STREAM COMP complete.`), so constrained exact-output reliability remains
  `PARTIAL` even though the reasoning/content stream was progressive.

Screenshots in this directory retain the corrected tool turn, three no-tool
continuations, progressive long stream, and the explicit-path negative control.

## Focused tests

- Python: `tests/test_tool_prompt_fallback.py` — 22 passed.
- Python combined: tool fallback plus DSV4 hardening — 40 passed.
- Panel: tool auto-continue plus chat UI — 162 passed.
- Panel TypeScript: `tsc --noEmit` passed.
- `git diff --check` passed before commit.
