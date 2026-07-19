# Step 3.7 JANGTQ Auto-reasoning / tool recovery

Date: 2026-07-18

Verdict: **PASS-LIVE for the scoped Step JANGTQ Auto-mode stream and explicit-tool
loop; PARTIAL for broader Step reliability/cache work.**

## Artifact and live route

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANGTQ_K`
- Weight route: JANGTQ/MXTQ codebook (`JANGTQ_K`, 2.841b), not affine JANG and
  not base MLX MXFP.
- Electron server: port 8022, PID 56622, started with the real visible Start
  control after stopping the prior process.
- Effective launch: `--reasoning-parser qwen3`, `--tool-call-parser step3p5`,
  `--enable-auto-tool-choice`, paged cache, and block-disk cache.
- Visible Chat Settings showed Thinking `Auto` selected, Thinking Off unavailable
  for this native-reasoning-only family, blank Max Tokens/model default, Responses
  wire, and built-in coding tools enabled.

## Root cause and source trace

The Step native template can emit a malformed auto-selected XML call during an
ordinary no-tool request. The schema validator correctly rejects that call, but
the old terminal path then finalized a reasoning-only assistant row with empty
visible content.

The repair is intentionally not a parser-wide or model-wide coercion:

- `vmlx_engine/server.py:1976-1993` scopes native tools-free recovery to
  `step3p7`, ordinary `tool_choice=auto` turns, and native reasoning.
- `vmlx_engine/server.py:5625` detects explicit/required/named tool intent so a
  genuine tool request remains fail-closed and cannot be converted to prose.
- Chat Completions and Responses, stream and non-stream, remove only tool schemas
  for the bounded retry; they keep Step's native open think rail and expose only
  text after the model's real close marker.
- `vmlx_engine/api/tool_calling.py:339-373` recognizes Step's real native tool
  schema on ordinary Auto turns and avoids duplicating a fallback schema.
- `tests/test_server.py:3268-3432` pins the intent gate and progressive Chat /
  Responses retry contracts.
- `tests/test_tool_format.py:1149-1197` pins native-schema reuse.

No close marker, tool call, tool result, or final answer is synthesized.

## Live Electron evidence

Screenshots:

- `step-postrel-fix-t2-result.png`
- `step-postrel-fix-t3-stream-pass.png`
- `step-postrel-fix-t4-tool-pass.png`

Persisted chat `6d438d5c-d88c-4f30-8518-52ca3afe18d0`:

1. Row 42 (`STEP-POSTREL-FIX-T2`) exercised the new recovery. The first native
   pass produced 1,216 separated reasoning characters and malformed auto-tool
   markup. The validator dropped the call, a native tools-free retry ran, and the
   final visible content was exactly `3861 STEP-POSTREL-FIX-T2-DONE`.
2. Row 45 (`STEP-POSTREL-FIX-T3`) was a same-chat no-tool recall turn. It kept
   1,248 reasoning characters separate and returned the exact four-line answer.
   A renderer MutationObserver captured progressive visible growth through
   `... MIKE N`, `... NOVEMBER OSCAR PAPA`, and the terminal marker one character
   at a time. This falsifies terminal batch insertion on the current renderer.
3. Row 48 (`STEP-POSTREL-FIX-T4`) required `file_info(panel/package.json)` exactly
   once. Electron showed two progressively changing reasoning rails around the
   real tool execution, one Info card, the real 5.2 KB result, progressive
   post-tool content, and exact `STEP-POSTREL-FIX-T4-DONE`. SQLite records one
   call/result and `512 paged+mixed_swa cached` tokens.

## Raw API evidence

See `api-stream-proof.json`.

- Responses, deterministic no-tools: 144 reasoning deltas, 9 content deltas,
  exact answer, completed terminal.
- Responses, deterministic two-tools Auto: 216 reasoning deltas, 9 content
  deltas, exact answer, completed terminal.
- Chat Completions, deterministic two-tools Auto: 255 `reasoning_content`
  deltas, 10 content deltas, exact answer, `finish_reason=stop`, then `[DONE]`.

One earlier request that omitted an explicit deterministic temperature inserted
`WASHINGTON` into a strict marker despite structurally correct 152/21
reasoning/content streaming. It is retained as a semantic reliability miss, not
hidden or reclassified as an API pass. The matched temperature-0 A/B did not
reproduce it.

## Tests

- `352 passed, 3 deselected` across `tests/test_server.py`,
  `tests/test_tool_format.py`, `tests/test_tool_prompt_fallback.py`, and
  `tests/test_tool_parsers.py`.
- `py_compile` and `git diff --check` pass.
- Repository-wide Ruff is not a clean gate: the current tree has longstanding
  lint debt. Critical-select Ruff still reports the pre-existing `_app_state`
  F821 at `server.py:11807`; this patch does not claim a full lint pass.

## Still open

- Remove or reword the visible malformed-auto-tool warning after a successful
  recovery without hiding the underlying diagnostic.
- Expand unseeded/stochastic Step soak; retain the captured `WASHINGTON` strict
  format miss.
- Complete Step cache pressure/eviction/restart media rows. The first recovery
  pass skipped a mixed-SWA prompt store under tight memory; the retry wrote six
  typed blocks. This gate does not close the full cache matrix.
- Run the same current-head Auto-mode stream matrix for every other registered
  reasoning/parser family; no family inherits this Step-only live proof.
