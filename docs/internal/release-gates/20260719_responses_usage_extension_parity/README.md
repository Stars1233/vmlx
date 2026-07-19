# Responses usage-event parity and local telemetry extension

Date: 2026-07-19

Source cutoff: `cc42513180ce39208445a8ac6a7201feb2450558`

Model: `jangq-ai/openPangu-2.0-Flash-JANG_3M`

Verdict: `FIXED_SOURCE + VERIFIED_LIVE_SCOPED`

The overall release remains `PARTIAL_NO_1_6_12_RELEASE`. This gate closes the
specific Responses usage-event/request-shape defect and proves that the local
Electron client still receives progressive usage telemetry without exposing
the private extension to ordinary Responses clients.

## Root cause

The Responses stream accepted Chat Completions-style
`stream_options.include_usage` and emitted a custom `response.usage` event on
each generated token. The Electron client also sent that nonstandard body field
for both local and remote Responses providers.

The public Responses stream contract carries final usage on
`response.completed`; the generated OpenAI Python Responses `StreamOptions`
surface contains `include_obfuscation`, not Chat Completions'
`include_usage`. Therefore an unadvertised `response.usage` event was a public
protocol compatibility risk even though vMLX's own renderer used it for live
metrics.

## Source repair

- `vmlx_engine/server.py` emits the custom incremental event only when the
  request explicitly carries `X-vMLX-Stream-Usage: incremental`.
- Ordinary Responses streams, including requests that send the legacy body
  field, receive no custom usage events and retain terminal usage on
  `response.completed`.
- `panel/src/main/ipc/chat.ts` no longer sends
  `stream_options.include_usage` in a Responses body. It sends the private
  header only to a local vMLX engine, so remote providers receive the standard
  request shape.
- `vmlx_engine/api/models.py` accepts the standard Responses
  `include_obfuscation` stream option.
- Python and panel contracts pin the standard and private-extension paths.

## Focused and expanded tests

- `pytest -q tests/test_server.py tests/test_engine_audit.py tests/test_api_models.py -k "responses or StreamOptions"`
  -> `83 passed, 735 deselected`.
- Panel request/stream/auto-continue/error/metrics selection -> `111 passed`.
- Panel `tsc --noEmit` -> clean.

Captured output: `python-tests.txt`, `panel-tests.txt`, and `source.diff`.

## Raw Responses A/B

### Standard request, no private header

`standard.sse` and `standard-analysis.json` show:

- exact visible content `RESP-USAGE-STANDARD-DONE`;
- 383 reasoning deltas and 9 progressive content deltas;
- zero `response.usage` events;
- contiguous event sequence;
- exactly one `response.completed` terminal with status `completed`;
- terminal usage `55 input / 394 output / 449 total`;
- zero error events.

The body intentionally retained legacy `stream_options.include_usage=true` to
prove it can no longer enable the public extension accidentally.

### Explicit local extension

`extension.sse` and `extension-analysis.json` use
`X-vMLX-Stream-Usage: incremental` and show:

- exact visible content `RESP-USAGE-EXTENSION-DONE`;
- 327 reasoning deltas and 9 progressive content deltas;
- 337 private `response.usage` telemetry events;
- contiguous event sequence;
- exactly one completed terminal with final usage;
- zero error events.

## Live Electron proof

The complete Electron main process was relaunched from this source with
`VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1611-cachefix-dev`. Its log found
`/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`. The visible Sessions Start
button loaded openPangu as PID 49982 on port 8027 before the test.

The real textarea sent:

`[RESP-USAGE-ELECTRON] Think privately about whether 21 + 21 equals 42. Do not call a tool. After reasoning, reply exactly RESP-USAGE-ELECTRON-DONE and nothing else.`

Evidence:

- `electron-2s.png`: the separate reasoning rail is actively growing.
- `electron-6s.png`: reasoning is complete and visible content has already
  painted partially as `RES`; the answer was not held until completion.
- `electron-final.png`: exact non-empty `RESP-USAGE-ELECTRON-DONE`, 196 output
  tokens, 61 prompt tokens, 31.3 tok/s, 0.49 s TTFT, and 6.8 s total.
- `electron-mutations.json`: 832 DOM mutations, including character-wise
  reasoning growth and partial visible-answer states before the final.
- `electron-rows.json`: persisted visible content is exact, reasoning is 671
  characters, and both tool calls and warnings are null.
- `health.json` and `process-argv.txt`: the live model/runtime and launch
  configuration used for this gate.

The main-process trace ended with `content: 24 chars, reasoning: 671 chars,
tool calls: 0, buffered: false` and recorded server-owned live usage metrics.

## Retained boundaries

- The official OpenAI SDK was not installed in either the runtime venv or panel
  dependencies, so this gate uses literal curl-N SSE as the API consumer proof;
  no dependency was added solely for the probe.
- Remote-provider request shape is pinned by panel tests but still needs a live
  provider smoke when credentials/provider access are intentionally available.
- Signed packaged-app repetition, full protocol failure/disconnect soak, and
  the remaining model/cache/media matrix are separate release gates.
