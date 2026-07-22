# 2026-07-22 Laguna wrapper reasoning seed proof

Status: `PARTIAL_WITH_SCOPED_FIX_PROOF`.

This gate covers the current-source fix that lets server-side reasoning seed probes inspect wrapped tokenizers. Laguna uses a tokenizer wrapper for bundle behavior; the inner tokenizer owns `apply_chat_template`. If the seed probe only checked the wrapper, thinking-on requests could be misclassified as visible content.

## Source trace

- `vmlx_engine/server.py`
  - adds `_chat_template_renderers(tokenizer)`
  - `_template_always_thinks`, `_template_starts_reasoning`, and `_engine_prompt_starts_in_reasoning` now render with the wrapper or inner tokenizer that actually supports `apply_chat_template`.
- `tests/test_streaming_reasoning.py`
  - adds `test_wrapped_tokenizer_inner_template_can_seed_reasoning` for Laguna-style `TokenizerWrapper`.

## Current-source validation

- `python -m pytest -q tests/test_streaming_reasoning.py::TestEnginePromptReasoningSeed::test_wrapped_tokenizer_inner_template_can_seed_reasoning tests/test_streaming_reasoning.py`
  - `136 passed`.
- Full panel gate already passed at this head family before this source fix: `79 passed` files, `2397 passed / 3 skipped`, and `npm run typecheck` passed.
- Full Python before the wrapper fix was down to the expected release-bundle drift: `6277 passed / 96 skipped / 92 deselected`, one failure from `verify-bundled-python.sh` requiring `bundle-python.sh`.

## Live API proof on restarted current-source Laguna PID 70727

- Health before the UI retest was observed live in the terminal as healthy with paged cache, block-disk L2, and `tq_native_enabled=true`; no health JSON is retained because the process was later stopped before artifact copy.
- Chat SSE: `chat-railwrap2-reasoning-content.sse`
  - observed separate `reasoning_content` deltas (`reasoning_len=752`) and visible content with no raw `<think>` markers in content.
  - this row hit max token length, so it proves separation but not terminal completion.
- Chat SSE: `chat-railwrap3-terminal.sse`
  - terminal `finish_reason=stop`, visible content `42 + 58 = 100`, no raw thinking markers.
- Responses SSE: `responses-clean-rail.sse`
  - terminal `response.completed`, visible text `19 + 23 = 42`, no engine error.
- Tool-enabled Responses SSE: `responses-tool-enabled.sse`
  - terminal `response.completed`, visible text `19 + 23 = 42.`, no function calls and no engine error under `tool_choice:auto` with `file_info` schema present.

## Live Electron proof

- Real Electron UI was used through CDP 9335 against the dev app and restarted Laguna via UI Stop/Start before proof.
- Chat Settings showed reasoning `Auto` selected.
- Clean retest row in `ui-rows-79-87.json`:
  - row 87 content: `LAGUNA UI CLEAN RETEST OK`
  - row 87 `reasoning_content`: `null`
  - row 87 metrics: `promptTokens=74`, `tokenCount=11`, `tokensPerSecond=75.3`, `ttft=0.46`
- Screenshot: `ui-clean-retest9.png`.

## Observed transient not counted as pass

- Row 81 in `ui-rows-79-87.json` is retained as a bad transient:
  - visible content: `The sum of 19 and 23 is 42.`
  - persisted reasoning content: `[Engine error: [full] Negative dimensions not allowed.]`
- Raw Chat/Responses and a subsequent clean Electron turn did not reproduce it, but this row remains a follow-up item for engine-loop exception provenance. Do not use row 81 as a release PASS.

## Verdict

Scoped source fix and normal UI/API retests are proven. Broader release remains blocked until bundled Python is refreshed and final full-suite/package/sign/notarize gates pass. The negative-dimension transient is documented and should be rechecked during the final Electron smoke.
