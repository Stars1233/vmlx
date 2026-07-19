# Ollama streaming, tool loop, and terminal parity

Date: 2026-07-19

Model: `jangq-ai/MiniMax-M2.7-Small-JANGTQ`

Artifact: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`

Runtime: real Electron-started engine on `127.0.0.1:8014`

Quantization identity: JANGTQ/MXTQ, not JANG affine and not base MLX MXFP.

## Verdict

`VERIFIED-LIVE` for this current-source M2.7 scope:

- `/api/chat` stream and non-stream;
- separate `message.thinking` and `message.content` deltas;
- one-terminal Ollama usage ordering;
- required tool emission on the terminal line;
- tool-result continuation without a second call;
- templated `/api/generate` stream and non-stream;
- exact visible output after shared think-boundary cleanup;
- progressive rendering through the real Electron app.

Overall protocol parity remains `PARTIAL`. Cancellation, client disconnect,
injected mid-stream failure/recovery, signed-app repeat, raw
`/api/generate` semantics, live multi-tool interleaving, and other model/parser
families remain separate rows.

## Issue 1: streaming exposed structural post-reasoning newlines

### Before

`chat-separator-before.raw` and `ollama-chat-stream-before.raw` each produced
the correct answer only after a visible `\n\n` prefix. This proved the issue
was upstream of the Ollama adapter: the identical direct Chat stream had the
same bytes.

### Root cause

Complete think-tag extraction strips content around `</think>`, while the
streaming base parser forwarded whitespace-only deltas after the close as
visible content. The failure therefore affected registered parsers that share
the think-tag base, not just Ollama or MiniMax. DeepSeek-R1 additionally
overrode the transition path and required the same rule at that owner.

### Fix

Commit `c1db6b745` drops whitespace only while accumulated visible content
after the close marker is still whitespace-only. Once the first visible byte
arrives, every later delta, including newlines, is preserved verbatim. Tests
cover Qwen3, DeepSeek-R1, and MiniMax-M2, both split-marker and same-delta
transitions, plus post-answer newline preservation.

### Live after

- `chat-separator-after.raw`: 1,114 reasoning characters, 12 progressive
  content deltas, exact `M27-CHAT-WHITESPACE-AFTER-DONE`, one `[DONE]`.
- `ollama-chat-stream-after.raw`: 823 thinking characters over 200 rows, 12
  progressive content rows, exact `M27-OLLAMA-STREAM-AFTER-DONE`, and one
  terminal with usage.

## Issue 2: templated `/api/generate` discarded stream usage

### Before

`ollama-generate-stream-before.raw` streamed separate thinking and exact
content, but its only terminal omitted `eval_count` and
`prompt_eval_count`. The Chat upstream sends finish, usage, and `[DONE]` as
separate events. The generate wrapper emitted the first `done:true` row and
discarded the later usage row.

### Fix

Commit `01d95b448` defers and merges templated generate terminal rows until
upstream `[DONE]`, preserving finish reason and usage in exactly one terminal.
It does not fabricate counts; only upstream usage fields are copied.

### Live after

`ollama-generate-stream-after.raw` contains 118 thinking rows, 13 progressive
response rows, exact `M27-OLLAMA-GENERATE-AFTER-DONE`, and exactly one
terminal with `eval_count=134` and `prompt_eval_count=74`.

## Ollama chat and tool-loop rows

- `ollama-chat-nonstream.json`: thinking disabled, exact
  `M27-OLLAMA-NONSTREAM-DONE`, `done_reason=stop`, and nonzero usage.
- `ollama-tool-initial.raw`: 204 thinking characters, zero visible content,
  one terminal `tool_calls` row, exactly one `file_info`, object argument
  `{"path":"panel/package.json"}`, and nonzero usage.
- `ollama-tool-followup.raw`: 94 fresh thinking characters, 17 progressive
  visible rows, exact `M27-OLLAMA-FINAL-TOOL-DONE SIZE=5.2 KB`, no second tool,
  one stop, and nonzero usage.
- `ollama-generate-nonstream.json`: exact
  `M27-OLLAMA-GENERATE-NONSTREAM-DONE`, one stop, and nonzero usage.

## Electron progressive-render verification

After a second real Electron Stop/Start loaded both patches, a fresh chat with
built-in tools enabled and an explicit no-tool directive was sampled through
CDP while generating. `electron-progress-dom.txt` records the visible DOM
growing through 13, 23, 43, 62, 81, and 108 reasoning tokens before the final
answer appeared. This rules out a reasoning-then-batched-answer observation for
the scoped turn.

`electron-progress-db.json` preserves 529 reasoning characters separately
from exact content `M27-ELECTRON-PROGRESS-AFTER2-DONE`, with no tool call and
no warning. `electron-progress-final.png` and `electron-progress-ui.txt`
retain the final visible UI.

## Tests

- `reasoning-focused-tests.txt`: 300/300 selected reasoning, streaming,
  Ollama/API, and audit tests passed; 543 unrelated rows deselected.
- `ollama-focused-tests.txt`: 36/36 selected Ollama, server, API parity, and
  audit tests passed; 696 unrelated rows deselected.

These are focused regressions. The earlier full-suite checkpoint remains the
current full-suite evidence and does not automatically cover later changes.

## Artifact map

- `chat-separator-before.raw` / `chat-separator-after.raw`
- `ollama-chat-stream-before.raw` / `ollama-chat-stream-after.raw`
- `ollama-chat-nonstream.json`
- `ollama-tool-initial.raw` / `ollama-tool-followup.raw`
- `ollama-generate-stream-before.raw` / `ollama-generate-stream-after.raw`
- `ollama-generate-nonstream.json`
- `electron-progress-dom.txt`, `electron-progress-db.json`
- `electron-progress-final.png`, `electron-progress-ui.txt`
- `reasoning-focused-tests.txt`, `ollama-focused-tests.txt`
- `health-current.json`, `commits.txt`
