# Gateway and adapter disconnect recovery — 2026-07-19

Status: **VERIFIED-LIVE (scoped)** on the working tree based on
`9eb7f9305e6d7a2f985ce91d257bcbfd0b981dff`.

This gate covers client disconnect cleanup and immediate recovery through the
real Electron gateway (`127.0.0.1:8081`) for Chat Completions, Anthropic
Messages, Ollama chat, and Ollama generate. It does not close tool-call,
reasoning-on, injected backend-failure, signed-app, or other-model rows.

## Live model and app

- Electron dev app: current source rebuilt into `panel/dist/main/index.mjs` at
  `2026-07-19 20:08:07`, CDP `127.0.0.1:9335`, user data
  `/Users/eric/.vmlx-v1611-cachefix-dev`.
- Electron startup log contains
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- Model: `jangq-ai/Laguna-M.1-JANG_2L`.
- The real Electron **Save & Restart** control replaced PID `80479` with PID
  `88506`; SQLite and `/health` then reported `running`, `model_loaded=true`,
  and scheduler `num_running=0`.
- Gateway remained on `127.0.0.1:8081` with single-model mode enabled.

Evidence: `live-electron-session.txt`, `restarted-health.json`,
`electron-ui-recovery.png`.

## Root causes and source repairs

### 1. Gateway abandoned-response propagation

`panel/src/main/api-gateway.ts` previously watched the consumed incoming
request body's `close` event and installed its response-close guard only after
backend response headers existed. Non-streaming inference does not produce
those headers until generation finishes, so a downstream client could leave
the backend request and scheduler work alive.

`abortProxyRequestOnClientClose` now attaches to the downstream
`ServerResponse` immediately after each upstream `ClientRequest` is created.
If the downstream closes before `writableEnded`, the upstream request is
destroyed. The generic proxy and the separate Ollama chat, generate, and embed
proxies all use this early guard.

### 2. Anthropic non-stream adapter receive ownership

Anthropic `stream=false` internally consumes `stream_chat_completion` to reuse
its reasoning and tool parsing, but there is no Starlette `StreamingResponse`
to own disconnect detection. Lazy `Request.is_disconnected()` polling missed
the already-consumed receive channel.

`vmlx_engine/server.py::stream_chat_completion` now creates an active receive
drain only when `request.stream is False`. Real streaming routes retain their
existing single receive owner. The non-stream adapter checks the drain every
0.25 seconds during a quiet prefill, aborts the exact scheduler request, returns
without a false terminal, and cancels the drain in `finally`.

Evidence: `source-trace.txt`.

## Reproduction before the repairs

- Chat non-stream client timeout left scheduler `num_running=1` for every
  sample across the 10-second poll. See `chat-nonstream-abort-idle.json`.
- After only the gateway repair, Anthropic non-stream still left
  `num_running=1` through 10.01 seconds. See
  `anthropic-nonstream-patched-abort-idle.json`.

These are retained failing artifacts, not passing evidence.

## Current live results

All requests below went through gateway port 8081. Intentional abort requests
used a 1.5-second client timeout while requesting a long 512-token answer.
Recovery began only after the health poll observed both scheduler counters at
zero.

| Surface | Aborted output contract | Idle after curl returned | Immediate recovery |
|---|---|---:|---|
| Chat stream | 21 progressive content deltas; no finish, usage, or `[DONE]` | 0.033 s | 11 content deltas, exact `CHAT-DISCONNECT-RECOVERY-OK`, `stop`, one usage, one `[DONE]` |
| Anthropic stream | 17 progressive text deltas; no stop/usage/message-stop | 0.034 s | 13 text deltas, exact `ANTHROPIC-DISCONNECT-RECOVERY-OK`, `end_turn`, one message-stop, nonzero usage |
| Ollama chat stream | 21 progressive content rows; no `done:true` | 0.030 s | 14 content rows, exact `OLLAMA-CHAT-DISCONNECT-RECOVERY-OK`, one `done:true/stop` row |
| Ollama generate stream | 16 progressive response rows; no `done:true` | 0.030 s | 15 response rows, exact `OLLAMA-GENERATE-DISCONNECT-RECOVERY-OK`, one `done:true/stop` row |
| Chat non-stream | zero response bytes before intentional client timeout | 0.037 s | exact `CHAT-NONSTREAM-PATCHED-RECOVERY-OK`, `stop`, nonzero usage |
| Anthropic non-stream | zero response bytes before intentional client timeout | 0.034 s | exact `ANTHROPIC-NONSTREAM-FIXED-RECOVERY-OK`, `end_turn`, nonzero usage |
| Ollama chat non-stream | zero response bytes before intentional client timeout | 0.031 s | exact `OLLAMA-CHAT-NONSTREAM-FIXED-RECOVERY-OK`, `done:true/stop`, `eval_count=17` |
| Ollama generate non-stream | zero response bytes before intentional client timeout | 0.029 s | exact `OLLAMA-GENERATE-NONSTREAM-FIXED-RECOVERY-OK`, `done:true/stop`, `eval_count=18` |

The intentionally aborted Anthropic integer stream contains `7|◊|8` in its
partial model text. It is retained in the raw evidence and is not classified as
a transport/parser pass or a completed-answer quality result. Every completed
recovery marker was exact and leak-free.

## Electron visible-output proof

After the current-source engine restart, a new real Electron chat had thinking
Off and built-in tools disabled. The prompt requested exactly
`UI-GATEWAY-DISCONNECT-FIX-OK`.

- SQLite row 728 persisted that exact 30-character content, no reasoning, no
  warnings, 14 tokens, 0.75-second TTFT, and 1.4-second total time.
- A live DOM `MutationObserver` recorded 46 changes. The visible answer advanced
  through `UI-GATEWAY-DISCONNECT-FIX-`, then `...-O`, then `...-OK` at 1497,
  1504, and 1512 ms before terminal metrics appeared. It did not arrive as one
  batched answer.

Evidence: `electron-dom-stream-probe.json`, `electron-ui-recovery.png`, and
`live-electron-session.txt`.

## Focused regressions

- Python disconnect/stream selection: **5 passed**, 704 deselected.
- Anthropic adapter suite: **63 passed**.
- Panel gateway suites: **77 passed**.
- Panel TypeScript typecheck: **PASS**.
- `git diff --check`: **PASS**.

Evidence: `python-focused-tests.txt`, `anthropic-adapter-tests.txt`,
`panel-gateway-tests.txt`, `panel-typecheck.txt`, and `git-diff-check.txt`.

## Remaining scope

- Reasoning was intentionally disabled in these transport probes; no new
  reasoning-separation claim is made here.
- No tools were supplied; automatic/required tool/result continuation remains a
  separate protocol row.
- This is one current Laguna runtime plus shared source and focused tests, not a
  claim that every model family has been re-run.
- Safe injected mid-stream backend exceptions and signed/notarized packaged-app
  repetition remain open.
- No new prefix/paged/L2/TurboQuant correctness claim is made by this gate.
