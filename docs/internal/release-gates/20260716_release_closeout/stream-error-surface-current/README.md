# Streamed server-error propagation proof

Status: `PASS-LIVE` for the scoped Electron Chat Completions and Responses
stream-error surfaces at commit `57d5bcd0f`. Model generation correctness,
disconnect recovery, and packaged-app behavior remain separate release gates.

## Root cause and source trace

Both protocol consumers already recognized server failures inside valid SSE
JSON. The Responses path handled `error`, `response.error`, and
`response.failed`; the Chat Completions path handled a top-level `error`
object. Each path intentionally threw, but the shared per-line JSON catch then
logged and swallowed every non-syntax exception. The outer request cleanup was
never reached, so Electron persisted the pre-inserted assistant placeholder as
a false zero-token success.

`panel/src/shared/chatStreamErrors.ts` now gives intentional server events a
typed error. `panel/src/main/ipc/chat.ts:2703-2720,2812-2825` throws that type
from both wire formats, and `chat.ts:3024-3047` rethrows typed server events
and expected disconnects before retaining the tolerant malformed-line path.
The existing outer cleanup at `chat.ts:4298-4303,4362` deletes an empty
assistant placeholder and returns the actual failure to the renderer. No
model-family exception, output rewriting, prompt coercion, sampler change, or
synthetic completion was added.

Exact excerpts are in `source-trace.txt`.

## Automated proof

At commit `57d5bcd0f`:

```text
npm test -- --run \
  tests/chat-stream-errors.test.ts \
  tests/api-gateway-ollama.test.ts \
  tests/responses-stream-recovery.test.ts

3 test files passed; 45 tests passed

npm run typecheck
tsc --noEmit
```

The behavioral helper test distinguishes an intentional SSE server failure
from a malformed optional line. The source-wiring assertion covers both Chat
Completions and Responses. Existing Responses recovery and gateway/Ollama
tests remained green in the same run.

## Live Electron proof

The proof used the repo dev Electron app with the persisted 1.6.11 test
profile, CDP on 9335, and the repo `.venv/bin/vmlx-engine` detected on `PATH`.
A local scripted server on 127.0.0.1:8129 returned valid SSE rather than a
transport disconnect:

- Chat Completions: `data: {"error":{"message":"PROBE PREFILL FAILURE"}}`
- Responses: a `response.created` event followed by a `response.failed` event
  whose error message was `PROBE PREFILL FAILURE`

This is deliberately a protocol/error-surface probe, not a model-quality or
cache claim.

Before the repaired Electron main loaded, assistant row 232 persisted empty
content with `tokenCount:0`, `0.0 t/s`, and `0.0s total`; the UI rendered “No
visible response was produced.” After the repaired main loaded:

- Chat Completions prompt `[STREAM-ERROR-CHAT2]` visibly rendered `Message
  failed` and the exact server error. SQLite retained only user row 233.
- The same visible chat was switched to Responses in Chat Settings and saved.
  Prompt `[STREAM-ERROR-RESPONSES1]` visibly rendered the same failure. SQLite
  retained only user row 234.
- Neither request created an assistant row, success metrics, tool status, or a
  synthetic completion.

Screenshots are `chat-completions-error-live.png` and
`responses-error-live.png`; the before/after database rows and saved wire
override are in `live-rows.json`.

## Remaining gates

- Exercise a current real-model streamed prefill failure again when one can be
  triggered safely; the scripted probe proves the owning protocol boundary,
  while Step's original live traceback provides the motivating real failure.
- Cover mid-stream server errors after partial visible content and expected
  backend disconnect/Stop recovery separately.
- Re-run this regression in the signed packaged app before release.
- Full Python/panel suites, signing, notarization, feed publication, and public
  release remain open.
