# Qwen3.6 MXFP4-MTP gateway media and stream repair

Date: 2026-07-21

Worktree: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

## Verdict

`VERIFIED-LIVE_SCOPED` for the exact
`dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP` artifact in the current development
Electron app and the four gateway protocol surfaces. This is a base MLX
MXFP4 artifact with native MTP; it is not affine JANG and is not
JANGTQ/MXTQ.

The scoped defects closed here are:

1. Anthropic and Ollama video/audio extensions were dropped before reaching
   the model.
2. Auto reasoning could consume its first partition after beginning a visible
   answer, leaving the answer truncated instead of using the reserved answer
   pass.
3. A completed non-stream answer pass retained the first pass's `length`
   terminal in Chat and therefore in translated Ollama output.
4. A development Electron process could launch the intended console script
   but import a sibling editable checkout.

## Current live Electron proof

- The real Electron Start path loaded the exact Qwen bundle and spawned the
  engine with `PYTHONPATH=/Users/eric/mlx/vllm-mlx-release-1.6.13`; the
  Electron parent had no `PYTHONPATH`. See `automatic-dev-source-proof.txt`,
  `live-engine-source-env.txt`, and `health-auto-source-loaded.json`.
- A real file-input MP4 turn produced 90 distinct observed UI states. The
  reasoning rail appeared at 7.757 seconds, visible content began at 16.763
  seconds, and the exact two-line answer completed at 16.869 seconds:
  `BANANA8426` and `Q27-MEDIA-UI-CURRENT-DONE`. See
  `q27-media-ui-current-final.png`, `q27-media-ui-current-trace.json`,
  `ui-trace-summary.json`, and `ui-message-row.json`.
- A clean current-source follow-up persisted exact visible content
  `Q27-AUTO-SOURCE-PIN-DONE`, separate reasoning, no warning, and no tool
  payload. See `q27-auto-source-followup-final.png` and
  `q27-auto-source-followup-row.json`.

## Current raw protocol proof

`q27-media-protocol-proof.json` contains live stream and non-stream requests
through Chat Completions, Responses, Anthropic Messages, and Ollama. Every
surface returned HTTP 200, the exact requested two-line answer, separate
non-empty reasoning deltas, multiple progressive content deltas, and its
native terminal:

| Surface | Content deltas | Reasoning deltas | Stream terminal | Non-stream terminal |
|---|---:|---:|---|---|
| Chat Completions | 9 | 154 | `stop` + `[DONE]` | `stop` |
| Responses | 11 | 151 | `response.completed` | `completed` |
| Anthropic | 14 | 86 | `message_stop` | `end_turn` |
| Ollama | 11 | 150 | `stop` | `stop` |

The retained pre-fix artifacts show both distinct failures rather than hiding
them:

- `q27-media-protocol-proof-current-source-pre-streamfix.json`: partial
  visible answer at the Auto partition boundary.
- `q27-media-protocol-proof-pre-nonstream-terminal.json`: exact non-stream
  answer paired with an incorrect `length` terminal.
- `q27-media-protocol-proof-stale-source-prefx.json`: excluded sibling-source
  run that motivated the development import pin.

## Source trace

- `vmlx_engine/api/anthropic_adapter.py`: preserves image, video, and audio
  content parts instead of collapsing media-bearing messages to text.
- `vmlx_engine/api/ollama_adapter.py`: maps Ollama `images`, `videos`, and
  `audio`/`audios` extensions into typed OpenAI-compatible content parts.
- `panel/src/main/api-gateway.ts`: preserves the same Ollama extensions when
  translating through the Electron gateway.
- `vmlx_engine/server.py`: reconciles a reserved answer pass with a previously
  streamed visible prefix, emits only the missing suffix, and adopts the
  answer pass terminal in non-stream Chat output.
- `panel/src/main/sessions.ts`: pins development engine imports to the active
  checkout while leaving packaged-engine paths unchanged.
- `source-trace.txt` records exact current line excerpts and production call
  sites. The reconciliation helper has four production call sites; it is not
  dead compatibility code.

## Focused validation

- `focused-python-tests.txt`: 131 passed.
- `focused-panel-tests.txt`: 58 passed.
- `panel-typecheck.txt`: TypeScript typecheck passed.
- `git-diff-check.txt`: no whitespace errors.
- `nonstream-terminal-regression.txt`: the focused terminal regression passed
  before the complete Python slice was rerun.

## Honest remaining boundary

- This gate closes only this Qwen MXFP4 native-MTP media/stream contract. It
  does not repeat or promote other model families.
- Audio transport now has source and adapter/gateway regression coverage, but
  no audio-capable model was run live in this gate. Live audio remains open.
- The health snapshot confirms the active hybrid SSM schema and q4 storage on
  attention KV only, but this gate is not a new cold/warm/L2 restart or
  eviction proof. Those remain in their dedicated cache gates.
- The model produced 3,642 characters of separate reasoning for the exact
  source-pin prompt. Visible content and transport were correct, but Auto
  reasoning verbosity is not claimed fixed.
- Signed-app repetition, media-bearing tool continuation, cancellation/fault
  injection, alternate-media salt, and the remaining family matrix stay open.
