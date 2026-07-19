# Responses cancellation, disconnect, and failed-terminal gate

Date: 2026-07-19

Status: `VERIFIED-LIVE` for explicit Responses cancellation and client-disconnect
recovery on the Electron-started MiniMax-M2.7 process. Mid-stream exception
terminal mapping is source/test verified but remains `PARTIAL` for safe live fault
injection. The overall protocol and release matrix remains `PARTIAL`.

## Artifact and launch truth

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`.
- Quant family: JANGTQ/MXTQ, not affine JANG and not base MLX MXFP.
- The real Electron Server screen stopped PID 93393 and started current-source PID
  95088 on port 8014. Health returned `healthy`, `model_loaded=true`, and the exact
  model name. The process retained paged cache, block-disk L2, and native q4
  TurboQuant arguments.
- Electron logs and the CDP screenshot are preserved in
  `electron-cancel-disconnect-logs.txt` and
  `electron-cancel-disconnect-logs.png`.

## Red reproduction

The pre-fix long Responses request was cancelled after three progressive content
deltas. The cancel endpoint returned HTTP 200, but the stream finalized partial
text `1, ` as:

- `response.output_item.done.item.status = completed`
- terminal event `response.completed`
- terminal response status `completed`

That is a shared Responses protocol failure: an explicitly aborted partial stream
was falsely reported as a successful answer. The raw pre-fix stream is
`explicit-cancel-before.raw`; its summary is
`explicit-cancel-before-summary.txt`.

Source review also found that the generic mid-stream exception branch emitted an
`error` event followed by a `response.completed` envelope whose inner status was
`failed`. Clients dispatching by event type could therefore treat a failed stream as
successful.

## Root cause and repair

Commit `ae498c70b0562541fb984309e6cf5d28e857856f` repairs the shared
`stream_responses_api` contract:

- engine `finish_reason=aborted` and a detected client disconnect are terminal
  cancellation state;
- cancelled partial message/reasoning/tool output items stay `incomplete`;
- the only terminal envelope is `response.incomplete` with
  `incomplete_details.reason=cancelled`;
- a cancelled reasoning-only stream cannot launch the bounded visible-answer retry;
- cancelled partial output is not stored in Responses history;
- a mid-stream exception emits `response.failed`, not `response.completed` with a
  contradictory inner status;
- `length` remains `response.incomplete` with `reason=max_output_tokens`.

No model output, tool argument, sampler value, prompt, or output cap is fabricated.
The fix is endpoint-global rather than MiniMax-specific.

## Regression evidence

The focused current-source command selected cancellation, abort, mid-stream,
failure, and Responses coverage from five suites. Result:

`111 passed, 741 deselected in 6.17s`

The two new regressions assert the exact aborted terminal/status/history contract
and the exact failed terminal plus partial usage contract. `git diff --check` also
passed before commit.

## Live explicit cancellation

After the Electron Stop/Start loaded commit `ae498c70b` as PID 95088, a direct
Responses request began streaming integers and was cancelled after exactly three
content deltas:

- response id: `resp_7b8a2f8c5881`
- cancel endpoint: HTTP 200
- visible partial bytes: `1, `
- output item status: `incomplete`
- only terminal type: `response.incomplete`
- terminal response status: `incomplete`
- incomplete reason: `cancelled`
- post-cancel health: `healthy`, zero active requests
- `GET /v1/responses/resp_7b8a2f8c5881`: HTTP 404, so the partial response was not
  stored as successful history

See `explicit-cancel-after.raw` and `explicit-cancel-after-summary.txt`.

## Live disconnect and recovery

A second long Responses stream was closed by the client after five progressive
content deltas:

- response id: `resp_4c882b25db62`
- engine reached zero active requests in 1.12 seconds
- the partial id returned HTTP 404 rather than a persisted success
- the immediate recovery request streamed the exact marker
  `M27-AFTER-DISCONNECT-PATCH-DONE` over 12 content deltas
- recovery emitted one `response.completed` terminal

See `disconnect-after.raw`, `disconnect-after-summary.txt`,
`disconnect-recovery.raw`, `disconnect-recovery-summary.txt`, and
`history-lookups.txt`.

## Remaining boundary

- The exception branch is proven by direct in-process stream execution and focused
  tests, but a production process was not deliberately corrupted solely to produce
  a live mid-stream engine exception. That sub-row remains `PARTIAL` until a safe,
  non-production-only fault-injection route or naturally reproducible engine error
  proves `error -> response.failed -> recovery` through the live HTTP stack.
- Chat Completions cancellation/disconnect semantics, signed-app repeat, raw
  Generate multi-tool behavior, and other model/parser families remain separate
  open rows. This gate does not close global API/protocol parity.

