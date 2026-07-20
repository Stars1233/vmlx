# Mid-stream engine failure and immediate recovery — 2026-07-19

Status: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED` on source commit
`5f05ad72a90192f5d0d3e6b1734551d8495a380b`.

This is a post-v1.6.12 closure. The public 1.6.12 artifacts remain sealed and
do **not** contain this change. The gate proves the shared Chat Completions,
Responses, and Electron terminal-error boundary; it does not claim a model
family, reasoning parser, tool parser, cache, media, or signed-app row.

## Root cause and source trace

The production Python streamers already owned the correct terminal contract:

- `vmlx_engine/server.py::stream_chat_completion()` sends already-generated
  content, then an error chunk, authoritative partial usage, and `[DONE]`.
- `vmlx_engine/server.py::stream_responses_api()` sends already-generated
  output-text deltas, one `error`, and one `response.failed` terminal whose
  nested response contains authoritative partial usage.

The Electron client in `panel/src/main/ipc/chat.ts` threw on the first error
signal. That cancelled the reader before Chat's usage chunk or Responses'
`response.failed` terminal could be consumed. Visible bytes were recovered by
the outer catch, but persisted token metrics and terminal ownership could be
incomplete.

Commit `5f05ad72a` makes the error a pending terminal condition. The panel now:

1. keeps reading after an ordinary server error event;
2. classifies `response.failed` as a Responses terminal event;
3. extracts nested `response.error.message` through
   `panel/src/shared/chatStreamErrors.ts`;
4. consumes terminal usage before throwing through the existing partial-save
   path; and
5. retains the immediate-throw behavior for an expected backend disconnect.

The fix is shared transport code. It contains no family name, quantization
check, output rewrite, hidden retry, forced sampler, or synthetic completion.

## Production-stream-function harness

`tests/cross_matrix/live_midstream_failure_server.py` replaces only inference
with a deterministic two-delta engine. Its FastAPI endpoints call the real
`stream_chat_completion()` and `stream_responses_api()` generators. The engine
raises after the second visible delta for `FAIL` prompts and completes normally
for `RECOVER` prompts.

`panel/scripts/live-midstream-failure-proof.mjs`:

- runs literal `curl -N` probes for both APIs;
- launches the real Electron dev app with isolated user data;
- creates and starts a real remote session;
- creates fresh chats through the visible `+ Chat` control;
- sends turns through the visible textarea;
- captures the progressive partial frame before failure;
- captures the interrupted persisted terminal and error toast;
- sends an immediate same-chat recovery; and
- inspects persisted messages, metrics, and exact outbound history.

CDP used an isolated local port because 9335 was occupied by an existing SSH
tunnel. No other Mac was used for this gate.

## Raw API proof

The exact retained streams are:

- `raw-responses-fail.sse`: `RESP-PARTIAL-`, then `VISIBLE`, two negotiated
  usage events, one `error`, one `response.failed`, output tokens `2`, total
  tokens `7`, and no `response.completed`.
- `raw-responses-recover.sse`: `RESP-RECOVERY-`, then `OK`, text/item done,
  exactly one `response.completed`, and total tokens `8`.
- `raw-chat-fail.sse`: `CHAT-PARTIAL-`, then `VISIBLE`, one error chunk,
  authoritative usage after the error (`5 + 2 = 7`), then `[DONE]`.
- `raw-chat-recover.sse`: `CHAT-RECOVERY-`, then `OK`, stop, one usage-only
  chunk, then `[DONE]`.

The server stack traces in `proof-server.log` are the deliberately injected
failures. They are expected evidence, not unexplained crashes.

## Electron proof

`live-proof.json` records the committed source SHA and exact DB assertions:

| Rail | Progressive frame | Persisted interrupted row | Immediate recovery |
| --- | --- | --- | --- |
| Responses | `RESP-PARTIAL-` visible before terminal | `RESP-PARTIAL-VISIBLE\n\n[Generation interrupted]`; token count `2`, prompt `5` | exact `RESP-RECOVERY-OK` |
| Chat Completions | `CHAT-PARTIAL-` visible before terminal | `CHAT-PARTIAL-VISIBLE\n\n[Generation interrupted]`; token count `2`, prompt `5` | exact `CHAT-RECOVERY-OK` |

The recovery request bodies replayed the safe partial assistant prefix but did
not send the UI-only `[Generation interrupted]` marker back to either API.
Screenshots were visually inspected after the committed-head run:

- `electron-responses-fail-partial.png`
- `electron-responses-fail-terminal.png`
- `electron-responses-recover-terminal.png`
- `electron-chat-fail-partial.png`
- `electron-chat-fail-terminal.png`
- `electron-chat-recover-terminal.png`

## Validation

- Focused Python: 2/2 selected `midstream_exception` tests passed.
- Focused panel: 5/5 stream-error/display tests passed.
- Full Python with the same clean JANG 2.5.31 source used by v1.6.12:
  **6,185 passed, 95 skipped, 92 deselected**.
- Full panel: **2,333 passed, 3 skipped** across 75 files.
- TypeScript typecheck: PASS.
- `electron-vite build` for main/preload/renderer: PASS.
- `git diff --check`: PASS.
- ESLint: N/A; `npm run lint` is declared but this repo has no ESLint
  configuration, so ESLint exits before inspecting source. The exact output is
  retained in `panel-lint.log` rather than mislabeled as a product failure.

The first unqualified full-Python run correctly failed bundled-JANG hash parity
because it compared the released clean bundle with Eric's intentionally dirty
`~/jang/jang-tools` checkout. Re-running with the release's documented clean
source at `/Users/eric/.cache/vmlx-release/jang-clean-9081c924/jang-tools`
passed the bundled-runtime check and the complete suite. Both logs are retained.

## Scope boundary

This closes the previously explicit safe injected mid-stream engine-failure
row for dev Electron plus raw Chat/Responses. Still open elsewhere in the
matrix: gateway network-loss fault injection, Anthropic/Ollama injected-engine
failure adapters, signed-app repetition, model/parser-family stochastic soak,
and all unrelated cache/media/quality rows.
