# MiniMax-M2.7 JANGTQ q4 cache and agent-stream gate

Date: 2026-07-18

Source head before this scoped commit: `2beaf0fb573c3a95bee6cf7481336b7a84b93267`

Branch: `reconcile/1.5.68`

Model: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`

## Verdict

`PASS-LIVE` for this scoped model/protocol gate on current source:

- the Electron app loaded the real model and produced non-empty, separated reasoning and visible-answer rails across three same-chat turns;
- the current post-restart process (PID 72865) executed one real `file_info(panel/package.json)` call and returned exactly `MM27-UI-CURRENT-DONE SIZE=5.2 KB`;
- raw streaming Chat Completions and Responses each emitted one schema-valid tool call, accepted the real tool result, progressively emitted the final answer, and terminated once;
- process-restart block-L2 restoration used native q4 TurboQuant storage for the full-KV cache.

This is not a whole-release verdict. The campaign-wide protocol, parser-family, media, UI, full-suite, packaging, and release rows remain governed by `CURRENT-MATRIX.md`.

## Artifact and settings truth

The tested artifact is **JANGTQ/MXTQ**, not affine JANG and not base MLX MXFP:

- `config.json`: `model_type=minimax_m2`, 62 full-attention layers, 48 query heads, 8 KV heads, no MTP, no vision/audio config;
- `jang_config.json`: `weight_format=mxtq`, `profile=JANGTQ2`, Hadamard/codebook routed experts at 2-bit, attention/dense/embed/lm-head roles at 8-bit;
- `generation_config.json`: temperature 1.0, top-p 0.95, top-k 40;
- Electron Chat Settings visibly showed reasoning `Auto`, temperature 1.00, top-p 0.95;
- current argv used `--reasoning-parser minimax_m2`, `--tool-call-parser minimax`, paged blocks of 64, 1,000 blocks, 15% cache memory, and block-disk L2.

MiniMax-M2.7 is text-only in this gate. VL belongs to the MiniMax-M3 family and is not inferred here.

## Source trace and defect

`vmlx_engine/server.py::stream_chat_completion` emitted an early Chat tool START delta containing the call id, then repeated the entire id in the final function-data delta. OpenAI-style stream accumulation concatenates string fragments, so a standards-style client reconstructed `call_abccall_abc`.

The repair preserves the START id as the sole id fragment for index 0. The later delta contains only function name/arguments. Calls after index 0 still introduce their own id/type because they did not receive an early START. The final data delta also avoids repeating the assistant role after START.

Regression coverage is in:

- `tests/test_openpangu_tool_parser.py::TestOpenPanguStreamEarlyStop`, including a client-style id concatenation assertion;
- `tests/test_server.py::test_streaming_chat_minimax_truncated_namespace_separator_is_buffered`, including the same single-id reconstruction contract.

The broad affected run is retained in `focused-parser-tests.log`: 430 passed, 3 intentionally deselected across server plus MiniMax-M3, OpenPangu, Step, Gemma, GLM, Hunyuan, Granite, DSML, native format, reasoning/tool interaction, and shared tool-parser suites.

## Live Electron evidence

- `electron-chat-settings-auto.png`: real model endpoint and Auto reasoning settings.
- `electron-multiturn-stream.png`: turns 1 and 2 have distinct reasoning rails and exact non-empty final markers. During the live CDP run, the second reasoning rail grew from 41 to 107 to 172 to 540+ characters; visible content then progressed from `MM` to the exact marker.
- `electron-tool-loop.png`: same-chat third/tool context, one visible `Info panel/package.json`, exact `SIZE=5.2 KB` final, and a resident `paged+tq-native` hit.
- `electron-current-pid72865.png`: after visible Save & Restart, current source PID 72865 executed one real tool and exact-finaled. `electron-current-row159.json` preserves the SQLite row: one call id, correct arguments, real result, non-empty visible content, separate reasoning, and no warning.

The current-source row reports 128 `paged+tq-native` cached tokens, 1.49s TTFT, and one completed tool loop. Those timing numbers are observations, not a family-wide performance guarantee.

## Raw API streaming evidence

`api-summary.json` and the four compressed `.sse.gz` files preserve the exact wire stream.

- Chat first pass: 73 reasoning deltas, zero content deltas, one `file_info` call, `finish_reason=tool_calls`, one `[DONE]`.
- Chat result pass: 46 reasoning deltas plus 14 content deltas over about 331 ms, exact `MM27-CHAT-TOOL-DONE SIZE=5.2 KB`, `finish_reason=stop`, one `[DONE]`.
- Responses first pass: one completed `function_call` item with the correct path and one completed terminal.
- Responses result pass: 55 reasoning deltas plus 15 content deltas over about 356 ms, exact `MM27-RESP-TOOL-DONE SIZE=5.2 KB`, no repeated function call, one completed terminal.

The reconstructed Chat call id is `call_369ebbcf` exactly once. The later function delta does not repeat the id/type/role fragments.

## q4 TurboQuant and L2 restart evidence

`health-after-restart.json` was captured after the visible process replacement and identical raw tool loops:

- policy: `turboquant-storage`, 4-bit key/value, `uncalibrated_full_kv_storage_tq4`;
- four requests restored 838 tokens as `paged+disk+tq-native`;
- last execution restored 192 tokens from three disk-backed blocks, `reconstructed=true`, `dequantized=false`, 186 native-TQ layer-block payloads;
- block disk store recorded 22 native-TQ hits and three native-TQ writes;
- model weights remained `turboquant_codebook` / `weight_format=mxtq` / `profile=JANGTQ2`.

This distinguishes weight quantization from KV-cache storage: the artifact is JANGTQ/MXTQ, while the compatible full-attention KV prefix payload is stored with q4 TurboQuant.

## Retained limitation

The Electron second turn expanded the rendered prompt and injected fallback tool schemas, so it did not reuse the entire prior conversation prefix. A later same-chat turn and the tool continuation did reuse resident q4-native blocks, and the identical raw loops restored q4-native blocks from disk after restart. Therefore this gate proves correct resident and L2 reuse where prompt identity matches; it does not claim every history/template transition produces a maximal prefix hit.
