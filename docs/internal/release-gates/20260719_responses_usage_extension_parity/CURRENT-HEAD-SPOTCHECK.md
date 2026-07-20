# Current-head Responses usage, gateway, and UI recovery spot-check

Date: 2026-07-19

Source cutoff: `76e8d6c1edba1b876a827edd04b9474d0417e840`

Model: `jangq-ai/Laguna-M.1-JANG_2L`

Verdict: `VERIFIED-LIVE_SCOPED`

This bounded spot-check confirms that later documentation-only work did not
regress the stronger usage/cancellation gates. It does not replace their raw
pre-fix controls or broaden the claim to every protocol.

## Source trace

- `vmlx_engine/server.py:19356-19369` treats `response.usage` as a private
  extension enabled only by `X-vMLX-Stream-Usage: incremental`.
- `vmlx_engine/server.py:20158-20177` emits that private event only when the
  header was negotiated; standard usage remains on the terminal response.
- `panel/src/main/ipc/chat.ts:2069-2085` sends the header only to a local vMLX
  Responses engine and does not put Chat-style usage options in a public
  Responses request body.
- `panel/src/main/ipc/chat.ts:2678-2699` consumes the negotiated local event for
  live telemetry.

Protocol reference checked for this reconciliation:
<https://platform.openai.com/docs/api-reference/responses-streaming/response/created?lang=node.js>.
The public event list carries terminal usage on `response.completed` and does
not define vMLX's private `response.usage` telemetry event.

## Direct standard stream

`direct-standard.sse` and `direct-standard-summary.txt` show:

- seven progressively timed `response.output_text.delta` events;
- exact `USAGE-STANDARD-CURRENT-OK`;
- zero `response.usage` events;
- exactly one `response.completed`;
- terminal usage `27 input / 10 output / 37 total`.

## Explicit local extension

`direct-extension.sse` and `direct-extension-summary.txt` show:

- ten explicitly negotiated private `response.usage` events;
- the same seven progressive content deltas and exact final;
- exactly one completed terminal;
- terminal usage with 23 cached `paged+tq-native` input tokens.

## Ordinary gateway client

`gateway-standard.sse` and `gateway-standard-summary.txt` show:

- eight progressive content deltas;
- exact `USAGE-GATEWAY-CURRENT-OK`;
- zero private usage events;
- one completed terminal with usage `28 / 11 / 39`.

This proves that the current gateway path did not inject the private local
header into an ordinary client request.

## Cancellation and visible Electron recovery

The stronger explicit cancellation/disconnect proof remains in
`../20260719_response_cancel_disconnect/`: it preserves incomplete terminal
state, removes partial history, reaches zero active requests, and completes an
immediate recovery stream.

The current Laguna spot-check intentionally disconnected a long gateway stream
after progressive content, observed scheduler health return to zero running and
waiting requests, and then completed an exact gateway follow-up. A real
Electron turn afterward persisted row 725 with:

- exact visible content `UI-CANCEL-RECOVERY-OK`;
- `reasoning_content=null` and no warning;
- 43 cached tokens with `paged+disk+tq-native` detail.

`laguna-cancel-recovery-ui.png` visually preserves that exact completed turn.

## Boundary

- Reasoning was disabled for this spot-check, so it does not add a new
  reasoning/content-separation claim.
- Tools were disabled / `tool_choice=none`, so it does not add a tool-loop
  claim.
- Chat Completions, Anthropic, and Ollama cancellation semantics; safe live
  mid-stream exception injection; remote-provider smoke; and signed-app repeat
  remain open.
