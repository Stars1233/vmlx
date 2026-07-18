# Qwen3 Reasoning Fallback Streaming Gate - 2026-07-17

Status: PARTIAL for release matrix, VERIFIED-LIVE for the scoped plain Qwen3 fallback-streaming row.

## Scope

- Model: `mlx-community/Qwen3-0.6B-8bit`
- Runtime: legacy Python/Electron repo at `/Users/eric/mlx/vllm-mlx`
- Server: `127.0.0.1:8031`, Electron dev renderer over CDP `127.0.0.1:9335`
- Purpose: verify that plain `qwen3` participates in the same bounded visible-answer fallback policy as Qwen3.5/3.6, and that answer content streams after reasoning instead of arriving only at completion.

## Source Trace

- `vmlx_engine/server.py`
  - adds `qwen3` to reasoning-answer fallback family handling
  - adds `qwen3` to fresh-context answer-pass handling
  - adds `qwen3` to automatic thinking-budget partitioning
- `panel/src/main/model-config-registry.ts`
  - exposes `supportsThinkingBudget` for the plain `qwen3` family
- Tests:
  - `tests/test_qwen3_answer_pass_policy.py`
  - `tests/test_answer_pass_families_dsv4_step37.py`
  - `tests/test_output_budget_cap.py`
  - `panel/tests/generation-defaults.test.ts`

## Live Evidence

- API proof: `/tmp/qwen3-fallback-live-auto.mjs`
  - Chat and Responses streams produced separate reasoning deltas and content deltas.
  - Auto partition without explicit `max_thinking_tokens` produced visible-answer deltas after reasoning instead of buffering the answer to completion.
- Electron proof: `/tmp/qwen3-ui-assistant-stream-sampler.cjs`
  - Fresh chat watched the assistant content node only.
  - Observed visible output increments:
    `R`, `RX0`, `RX01-S`, `RX01-SY0`, `RX01-SY02-TZ`, `RX01-SY02-TZ03`, `RX01-SY02-TZ03-UA04`, `RX01-SY02-TZ03-UA04-`, `RX01-SY02-TZ03-UA04-VB`, `RX01-SY02-TZ03-UA04-VB0`, `RX01-SY02-TZ03-UA04-VB05-`, `RX01-SY02-TZ03-UA04-VB05-W`, `RX01-SY02-TZ03-UA04-VB05-WC0`, `RX01-SY02-TZ03-UA04-VB05-WC06`.
  - Persisted DB row: exact content `RX01-SY02-TZ03-UA04-VB05-WC06`, no warning.
- Screenshots:
  - `qwen3-ui-assistant-stream-mid.png`
  - `qwen3-ui-assistant-stream-final.png`

## Focused Tests Run

```text
.venv/bin/python -m pytest tests/test_answer_pass_families_dsv4_step37.py tests/test_output_budget_cap.py tests/test_qwen3_answer_pass_policy.py -q
14 passed

cd panel
npm test -- --run tests/generation-defaults.test.ts
15 passed
npm run typecheck
completed
```

## Important Non-Passing Rows

- `Q3-UI-STREAM2` is not a pass:
  - It hit the previous 160-token output cap and produced the wrong exact marker.
  - It was a same-chat probe with prior marker history and should not be used as clean correctness evidence.
- `Q3-UI-STREAM3` is not a correctness pass:
  - It streamed visible content incrementally, but produced `Q3-UI-THIRD-PASS-DONE` instead of `Q3-UI-THIRD-TURN-DONE`, likely from same-chat marker contamination.

## Remaining Gates

- Retest Bonsai and Qwen-family tool/reasoning rows on the shared fallback behavior.
- Restart/reload Electron main before claiming the plain Qwen3 `Max Thinking Tokens` UI field is visually present from the registry change.
- Continue matrix gates for cache/TQ/VL/video/tool streaming across requested model families.
