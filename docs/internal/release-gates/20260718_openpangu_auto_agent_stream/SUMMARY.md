# openPangu 2.0 Flash 3M Auto reasoning and agent-stream proof

Date: 2026-07-18

## Verdict

`VERIFIED-LIVE` for current-source openPangu text Auto reasoning, separate
progressive reasoning/content emission, required tool generation, real
tool-result continuation, and clean terminal behavior through Electron,
Responses, and Chat Completions. Long-context and cancellation/disconnect soak
remain `PARTIAL`, so this is not a family-wide or release-wide closure.

## Bundle and architecture truth

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/openPangu-2.0-Flash-JANG_3M`.
- Model type: `openpangu_v2`; 46 configured transformer layers with a 1:2
  DSA/SWA pattern and 512-token sliding window.
- Quantization: affine importance-quantized `JANG_3M`, actual 3.83 bits,
  3/4/8-bit tensors, asymmetric `mx.quantize`, no Hadamard rotation. This is
  not JANGTQ/MXTQ and not base MLX MXFP.
- Template: native Pangu message/tool tokens, JSON-list calls inside
  `<|tool_call_start|>...<|tool_call_end|>`, thinking on by default.
- Sampling defaults: temperature 1.0, top-p 0.8, seed 1234.

## Source trace

- `vmlx_engine/server.py:6181-6212` maps public Auto/On/Off onto the real
  openPangu template's `thinking` boolean; Auto leaves the template default.
- `vmlx_engine/server.py:8334-8351` reports the native
  `openpangu_v2_composite_v2` path-dependent cache and explicitly forbids
  generic paged blocks/TurboQuant.
- `vmlx_engine/tool_parsers/openpangu_tool_parser.py:37-52` owns the native
  JSON-list tool parser.
- The shared panel path independently handles Responses reasoning, content,
  argument, and terminal events in `panel/src/main/ipc/chat.ts:2459-2709`.

## Real Electron load and settings

The Sessions drawer Start button loaded openPangu PID 21745, stopped M3, and
left exactly one running model. Before any prompt, `/health` reported:

- `model_loaded=true`, `last_request_time=null`;
- native typed MLA + DSA indexer + rotating SWA + convolution state;
- generic TurboQuant disabled;
- generic paged/block L2 disabled;
- exact typed prompt-disk L2 enabled.

The fresh UI displayed Auto reasoning, Responses wire, tools enabled,
temperature 1.0, and top-p 0.8. The controlled proof changed only temperature
to 0 and max output to 512 while leaving Auto selected.

## Electron generation proof

- Row 207: 897 reasoning characters and 406 visible characters were stored on
  separate rails. The two-sentence answer was coherent, non-empty, complete,
  and warning-free; no tool was called.
- Same-chat row 210: exactly one real
  `file_info({"path":"panel/package.json"})` call, real 5.2 KB result, 468
  separate reasoning characters, and the requested one-sentence visible
  answer. There was no repeated call, fabricated result, or warning.
- Timed screenshots retain waiting/reasoning/content states and the final tool
  card/answer in the actual Electron dev build.

## Raw API proof

| API case | Reasoning deltas | Content/argument deltas | Terminal |
|---|---:|---:|---|
| Responses no-tool Auto | 389 | 38 content | `response.completed` |
| Responses required tool | 35 | 2 arguments | valid `file_info`, completed |
| Responses real-result follow | 124 | 151 content | completed |
| Chat no-tool Auto | 250 | 38 content | `stop`, `[DONE]` |
| Chat required tool | 35 | 2 tool deltas | `tool_calls`, `[DONE]` |
| Chat real-result follow | 185 | 145 content | `stop`, `[DONE]` |

All visible responses were non-empty and coherent. No reasoning tokens leaked
into content, no call arguments were truncated, and no stream ended without
its protocol terminal.

## Cache and MTP boundary

This architecture uses its native exact typed prompt snapshot/L2 path. It did
not receive generic TurboQuant, paged blocks, or hybrid-SSM rederive logic.
The bundle contains MTP hints in config/JANG metadata, but its name does not
declare MTP and the tensor index has no MTP tensors. Health truthfully reports
the current runtime drops those stored extra-layer heads and keeps MTP
inactive; no MTP depth was invented.

## Evidence index

- `raw-chat-responses.json`, `raw-summary.json` — complete and compact timed
  Chat/Responses traces.
- `electron-message-rows.json` — persisted ordinary/tool UI rows.
- `health-after-electron-start.json`, `health-current.json`,
  `capabilities-current.json` — runtime/cache/parser truth.
- `electron-start-clicked.png`, `electron-one-model-loaded.png` — actual UI
  load and one-model swap.
- `electron-settings-auto-default.png`,
  `electron-settings-auto-temp0-512.png` — bundle defaults and applied control.
- `electron-auto-*.png`, `electron-tool-*.png` — timed UI generation proof.

## Open boundary

Long-context coherence/cache reuse and cancellation/disconnect recovery remain
open. No fake parser output, prompt coercion, hidden sampler clamp, or generic
TQ/paged cache was added.
