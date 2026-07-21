# Qwen3.6 27B MXFP4-MTP video, streaming, and typed-cache gate

Date: 2026-07-20

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Starting source: `225188adb34e950703fdaae47b378a724a19fbc4`

## Exact artifact and runtime classification

The tested artifact is
`/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP`.
This is base MLX `MXFP4`, not affine JANG and not JANGTQ/MXTQ. Its real bundle
has a Qwen3.5-family 64-layer hybrid text graph, 333 vision tensors, 23 MTP
tensors, and no audio tensors. Sixteen layers own attention KV and 48 layers
own native SSM companion state. The artifact name declares MTP and the tensor
index contains the MTP head, so native MTP is allowed here; this does not
authorize MTP for similarly named non-MTP Qwen artifacts.

The real Electron session launched with `--is-mllm`, Qwen tool/reasoning
parsers, paged prefix cache, block-disk L2, native MTP depth 3, and the
deterministic-defaults MTP policy. Health before the first request showed the
model materialized with `last_request_time=null`, runtime scope `text+vl`, and
one active engine. The visible creation drawers showed depth 3, text+VL scope,
hybrid SSM cache, temperature 1.00, top-p 0.95, and top-k 20 from the persisted
model defaults.

## Source ownership

- `vmlx_engine/native_mtp.py` derives VL readiness from the real MTP runtime,
  vision config, and indexed vision weights.
- `vmlx_engine/mllm_batch_generator.py` owns per-request MLLM MTP acceptance,
  rollback/replay, timings, and the published `last_native_mtp` record.
- `vmlx_engine/server.py` reports `hybrid_ssm_v1`: TurboQuant applies only to
  attention KV while SSM state remains a native companion with async rederive.
- `vmlx_engine/utils/turboquant_config.py` selects q4 stored-prefix encoding
  for supported Qwen hybrid attention KV. It does not encode the SSM layers.

Exact current-source excerpts are preserved in `source-trace.txt`; bundle
sidecar/index facts are in `bundle-facts.json`.

## Live Electron cold video and streaming proof

Built-in tools were disabled through the real Chat settings drawer. A fresh
chat attached the known-marker MP4 through the actual file-input surface and
asked for two exact lines. The final was exactly:

```text
BANANA8426
Q27-MTP-VID-A1-DONE
```

The DOM observer captured 86 distinct states at 100 ms resolution. The UI
first showed a waiting state, then progressively growing reasoning, then
progressively growing visible content after the reasoning rail finalized.
SQLite row 208 contains 933 separate reasoning characters, non-empty exact
visible content, no tool calls, and no warnings. Cold TTFT was 7.68 seconds.

This request exercised native MTP rather than merely advertising it. Health
recorded 192 drafted tokens and 127 accepted tokens (66.15%), including six
depth-2 and four depth-3 accepts, with no fallback reason. Adaptive policy
finished at depth 1 after using the configured higher depths.

## Process-restart L2 and typed-cache proof

The model was stopped and started through the visible Electron controls. The
replacement process was healthy before a request with empty L1 state while L2
retained 35 q4-native attention blocks and one 2,225-token native SSM
checkpoint. A new chat reattached the same MP4 and sent the byte-identical
prompt.

The restart turn restored 2,225 of 2,226 prompt tokens as
`paged+ssm+disk+tq-native`, promoted all 35 disk blocks, restored the native
SSM companion once, and exact-finaled again. Worker telemetry reports
successful reconstruction and dequantization; TTFT fell to 0.60 seconds and
prompt throughput rose from 290.0 to 3691.5 prompt tokens/s. The UI observer
still captured 81 progressive states, so the fast prefix restore did not turn
the answer into a batched final paint.

The restart request again exercised MTP: 176 drafted / 107 accepted, including
depth-2 and depth-3 accepts, with no fallback. Health identifies q4 only at the
16 attention layers and full-precision native companion state at the other 48
layers. This is not generic all-layer TurboQuant.

## Raw Responses streaming proof

A separate omitted-max raw `/v1/responses` MP4 request produced HTTP 200 and:

- 155 progressive `response.reasoning_summary_text.delta` events;
- 16 progressive `response.output_text.delta` events;
- exact visible `BANANA8426` plus `Q27-MTP-RAW-VID1-DONE`;
- one `response.output_text.done` and one `response.completed` terminal;
- no failure/error terminal.

Content streamed from 14.837 to 15.725 seconds in prefixes such as `BAN`,
`ANA`, `8`, `4`, and `2`; it was not withheld until completion. The raw request
also used native MTP (194 drafted / 130 accepted, including deeper accepts).

## Same-chat post-video automatic tool proof

After the restart video turn, built-in tools were enabled through the real Chat
settings drawer. A no-attachment same-chat request required exactly one
`file_info(panel/package.json)` call and an exact final based on its result.
The observer captured 50 UI states spanning progressive reasoning, the tool
card, result processing, and progressive visible content. SQLite row 214
contains exactly one schema-valid call with `{"path":"panel/package.json"}`,
one real `Size: 5.2 KB` result, separate reasoning, no warning, and exact:

```text
Q27-MTP-POSTVIDEO-TOOL1-DONE SIZE=5.2 KB
```

The request did not resend the old video or hallucinate a result. The native
MTP tool-safety policy constrained this tool-bearing turn to depth 1: health
recorded 13 drafted / 6 accepted tokens, zero depth-2/depth-3 drafts, and no
fallback. This directly covers the previously risky combination of a Qwen MTP
artifact, Auto reasoning, native tool XML/JSON parsing, and continuation after
media without a reasoning-only or dropped-call final.

## Evidence map

- Settings/load: `q27mtp-create-derived-settings.png`,
  `q27mtp-create-mtp-settings-open.png`, `q27mtp-loaded-before-request.png`,
  `session-config.json`, and `live-argv.txt`.
- Cold Electron: `q27mtp-video-a1-dom-trace.json`, paired screenshots, and
  row 208 in `ui-cold-and-restart-rows.json`.
- Restart/L2 Electron: `q27mtp-video-a1-restart-dom-trace.json`, paired
  screenshots, row 211, and `health-after-raw-video.json` (which retains the
  disk-hit counters and last cache execution).
- Raw API: `q27mtp-raw-responses-video.sse`.
- Post-video tool: `q27mtp-chat-settings-before-tool.png`,
  `q27mtp-postvideo-tool-dom-trace.json`, paired screenshots,
  `ui-postvideo-tool-row.json`, and `health-after-postvideo-tool.json`.
- Source/artifact truth: `source-trace.txt`, `bundle-facts.json`, and
  `capabilities.json`.

## Verdict and remaining boundary

`VERIFIED-LIVE_SCOPED` for this exact Qwen3.6 27B MXFP4-MTP artifact: Electron
eager load, bundle-derived settings, video, separate progressive reasoning and
content, raw Responses terminals, native MTP depth use, q4 attention-KV plus
native SSM storage, and restart-from-disk restoration are directly evidenced.

Still `PARTIAL/OPEN`: image/video salt isolation beyond this identical-media
restart, raw API post-media tool continuation, Chat/Anthropic/Ollama media
requests, explicit MTP Off/depth variants, bounded eviction/fault injection,
longer video/context, 35B MoE MTP variants, Bonsai/Ornith breadth, signed-app
repeat, and the full campaign/release matrix. No public release contains this
post-v1.6.14 evidence row by implication.
