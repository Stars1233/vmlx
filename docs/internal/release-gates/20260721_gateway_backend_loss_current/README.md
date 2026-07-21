# Current-source Electron gateway backend-loss repair

Date: 2026-07-21

Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Model under test:
`dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP`

Artifact boundary: this is base MLX `MXFP4` with native Qwen MTP, not affine
JANG and not JANGTQ/MXTQ.

## Verdict

`VERIFIED-LIVE_SCOPED` for an already-started streaming backend disappearing
behind the current Electron-owned gateway on Chat Completions, Responses,
Anthropic Messages, and Ollama chat. Each protocol now receives a native
failure event and an immediate downstream EOF instead of hanging. A subsequent
same-protocol request JIT-loads the stopped session and completes normally.

This does not close non-stream partial-response loss, active LAN/port mutation,
concurrent loss/swap soak, other model/parser families, or signed-app behavior.

## Pre-fix reproduction

The real gateway stream began progressively and Electron's visible `Stop`
button stopped its Qwen backend. After the backend disappeared, the gateway
neither emitted a terminal failure nor closed the client response. The client
eventually raised a read timeout at 36,626.17 ms. The retained artifact is:

- `q27-gateway-backend-loss-proof.json`

The same artifact also records that a later request could recover, isolating
the defect to the in-flight proxy response lifecycle rather than model reload.

## Owning source trace and dead-code cleanup

`panel/src/main/api-gateway.ts` previously ended downstream responses only from
backend `end`. Its generic, Ollama chat, Ollama generate, and Ollama embeddings
paths each had a duplicated `proxyRes.on("error")` listener. Those listeners
logged ordinary errors but treated reset/premature-close shapes as downstream
client disconnects, even when the backend was the side that vanished.

Current source adds one `guardProxyResponseLifecycle` owner with these rules:

- normal `end` or `proxyRes.complete` settles without intervention;
- downstream-client closure remains silent because the client response is no
  longer writable;
- incomplete backend `aborted`, `error`, or `close` is deduplicated;
- streaming Chat emits an OpenAI error chunk and no `[DONE]`;
- streaming Responses emits `error` then `response.failed`, never
  `response.completed`;
- streaming Anthropic emits `event: error`, never `message_stop`;
- streaming Ollama emits an error NDJSON object, never `done:true`;
- a non-stream loss before gateway headers returns 502, while a partial
  already-started HTTP response is destroyed promptly rather than being
  rewritten as a second response.

The four superseded per-route response-error listeners were removed. The
downstream close helpers were retained because they have four active production
call sites and own the opposite direction of cancellation.

## Focused validation

Current source results in this campaign turn:

- `api-gateway-ollama-behavior.test.ts`: 17/17 passed, including a real capture
  backend that destroys its socket after one partial event on all four
  protocols;
- `api-gateway-ollama.test.ts`: 39/39 passed after replacing obsolete source
  assertions for the removed duplicate listeners;
- `api-gateway-single-model.behavior.test.ts`: 25/25 passed;
- combined focused result: 81/81 passed;
- `npm run typecheck`: passed;
- `git diff --check` on the three changed panel files: clean.

No full suite was rerun for this scoped repair.

## Current Electron and raw API proof

The Electron development main process was fully relaunched from this checkout.
Startup printed:

`[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`

The real `Start` button eagerly loaded Qwen on port 8005 before the loss tests.
Health reported `model_loaded=true`, the hybrid Qwen cache schema, q4
attention-KV storage, native SSM companion state, and native MTP depth 3.

For every row below, the raw gateway stream first emitted at least three
visible content deltas. The visible Electron `Stop` button then terminated its
backend. The table records the failure boundary and the immediate
same-protocol recovery.

| Protocol | Loss result | Recovery result |
|---|---|---|
| Chat Completions | error code `backend_connection_closed` at 18,284.40 ms; no exception and no `[DONE]` | exact `GATEWAY-BACKEND-LOSS-RECOVERY-DONE`, 9 content deltas, `stop` plus `[DONE]` |
| Responses | `error` and `response.failed` at 20,562.97/20,563.30 ms; no exception and no completed terminal | exact `GATEWAY-RESPONSES-LOSS-RECOVERY-DONE`, 9 content deltas, `response.completed` |
| Anthropic | native `error` at 30,521.77 ms; no exception and no `message_stop` | exact `GATEWAY-ANTHROPIC-LOSS-RECOVERY-DONE`, 10 content deltas, `message_stop` |
| Ollama | native error object at 18,304.69 ms; no exception and no `done:true` | exact `GATEWAY-OLLAMA-LOSS-RECOVERY-DONE`, 9 content deltas, `done:true` with `done_reason=stop` |

Raw proof artifacts:

- `q27-gateway-backend-loss-postfix-proof.json`
- `q27-gateway-backend-loss-responses-postfix.json`
- `q27-gateway-backend-loss-anthropic-postfix.json`
- `q27-gateway-backend-loss-ollama-postfix.json`

Each loss was triggered by the visible app control captured in the matching
`*-ui-stop.png` screenshot.

After all four loss/recovery cycles, a fresh real Electron chat produced 153
distinct observed UI states, kept reasoning in its own rail, stored a non-empty
visible assistant message, used no tool, emitted no warning, and exact-finaled:

`GATEWAY-BACKEND-LOSS-UI-FINAL-DONE`

Evidence:

- `q27-gateway-backend-loss-ui-final-trace.json`
- `q27-gateway-backend-loss-ui-final.png`

## Remaining work

- live non-stream backend loss after response headers or partial body;
- active-request gateway LAN/port mutation and rollback;
- concurrent backend loss plus single-model swap;
- other model/parser families where translated protocol behavior differs;
- signed/notarized app repetition.
