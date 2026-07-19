# MiniMax M2.7 JANGTQ cache/protocol/Electron gate (2026-07-19)

Status: `VERIFIED-LIVE_SCOPED` on source cutoff `b31fdca95`.
Overall status remains `PARTIAL_NO_1_6_12_RELEASE`; this closes only the named
MiniMax M2.7 full-KV JANGTQ/MXTQ rows.

## Artifact and runtime truth

The tested artifact was loaded through the real Electron Sessions Start
control:

```text
/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ
```

Bundle files identify `MiniMaxM2ForCausalLM`, `model_type=minimax_m2`, 62
layers, 48 attention heads, and eight KV heads. `jang_config.json` identifies
`weight_format=mxtq`, profile `JANGTQ2`, two-bit routed experts, eight-bit
attention/dense/embed/head weights, and 16-bit norms/router. This is
JANGTQ/MXTQ Hadamard/codebook weight storage. It is not affine JANG and is not
base MLX MXFP.

Weight quantization is separate from prefix KV storage. Current Auto policy in
`vmlx_engine/utils/turboquant_config.py:115-184` selected q4 stored attention
KV (`uncalibrated_full_kv_storage_tq4`) for all 62 compatible full-KV layers.
Only Bonsai retains the all-q8 KV exception. Live startup logs and health
confirmed the q4 policy, 62 `TurboQuantKVCache` layers, native writes, native
hits, reconstruction, and dequantization.

## Source trace

- `vmlx_engine/paged_cache.py:572-610` owns the independent Paged-RAM,
  explicit-frugal, and disk-only policies.
- `vmlx_engine/prefix_cache.py:1938-1949,2301-2339,4308-4334` owns resident
  write-through, native/path-dependent exceptions, L2 refault, bounded L1
  promotion, and disk-only transient release.
- `vmlx_engine/server.py:8925-8988` exposes backend, resident bytes/tokens,
  disk-only, frugal, and RAM-mirror policy in health.
- `vmlx_engine/utils/turboquant_config.py:115-184` classifies full-KV q4,
  Qwen mixed q4/q8, hybrid q4, and Bonsai q8 without conflating model weights.
- `panel/src/main/sessions.ts:1559-1582` serializes the one-model stop/start
  transition.
- `vmlx_engine/engine/batched.py:835-845` starts and loads the model before the
  server reports ready; `vmlx_engine/server.py:6387-6708` owns startup loading.
- Source repair commits: `28a8acf78`, `8a93aa910`; evidence checkpoint before
  this gate: `b31fdca95`.

No prompt coercion, sampler clamp, synthetic output, fabricated cache detail,
or model-name-only special case was added for this gate.

## Real Electron loading, settings, and single-model behavior

- The real Sessions card Start action selected the exact JANGTQ artifact.
- Gateway single-model mode was visibly On and SQLite stored
  `gateway_single_model_mode=true`.
- Starting M2.7 stopped the prior LFM PID before the M2.7 process became ready;
  the final process inventory contained exactly one local `vmlx_engine.cli
  serve` process.
- The Electron main log records
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- Before any prompt, `/health` reported `model_loaded=true`,
  `last_request_time=null`, and about 38.3 GB active model memory. This proves
  eager materialization on Start rather than first-message lazy loading for
  this route.
- Final effective settings/argv agree: Paged On, Block Disk L2 On, 64-token
  blocks, 10 blocks, 15% cache memory, KV Auto, MiniMax tool parser, MiniMax M2
  reasoning parser, and the isolated proof L2 directory.

Screenshots: `m27-single-model-before-swap.png`,
`m27-eager-ready-before-prompt.png`, `m27-eager-logs-before-prompt.png`, and
`m27-settings-restored-paged-on.png`. Effective values are preserved in
`session-effective.json`, `process-argv.txt`, `settings.json`, and
`m27-health-eager.json`.

## Paged RAM and SSD hierarchy

| Gate | Live Electron/health result |
|---|---|
| Cold write-through | Row 617 exact-finaled `M27-TQ-TIER-DONE`. Health held 352 RAM tokens plus six SSD blocks / 352 SSD tokens and six native-q4 writes. |
| RAM reuse before SSD | Fresh row 620 restored 352/356 as `paged+tq-native`; TTFT fell 2.41s -> 0.37s and SSD reads remained zero. |
| Bounded eviction | Disjoint row 623 exact-finaled. The ten-block configuration recorded three L1 evictions while L2 retained both prefixes. |
| Same-process SSD fallback | Row 626 restored 352/356 as `paged+disk+tq-native`; six SSD/native-q4 hits and six L1 promotions were recorded. |
| Restart + partial SSD prefix | PID 34990 -> 35714 began with zero RAM and 12 persisted blocks. Changed-suffix row 629 restored 320/360 as `paged+disk+tq-native`, with five disk/native hits and promotions. |
| Paged Off + SSD-only partial | The real UI set Paged Off while L2 stayed On. PID 36087 reported `backend=block_disk_only`, `ram_mirror_policy=disk_only`, and zero resident bytes. Changed-suffix row 632 restored 320/361 as `block-disk+tq-native`; five SSD/native-q4 hits occurred and resident bytes stayed zero. |
| Desired hierarchy restored | UI restored Paged On. PID 36463 began with zero cold L1 state, 14 persisted SSD blocks / 779 tokens, and `ram_mirror_policy=resident`. |

This establishes the requested lookup order for the exercised full-KV family:
use the longest valid RAM block chain first when Paged RAM is On, refault the
remaining valid chain from SSD L2, and prefill only the unmatched suffix. With
Paged RAM Off, the manager directly reconstructs the longest valid SSD chain
without silently keeping a RAM mirror. If neither tier contains a valid chain,
the cold rows show full prefill and write-through.

Health snapshots and screenshots for every transition are committed in this
directory. `db-rows-summary.json` preserves the exact final markers, cache
details, tool payload, reasoning lengths, and warning absence.

## Raw API protocol and stream parity

`m27_protocol_parity.py` drove stream and non-stream requests through Chat
Completions, Responses, Anthropic Messages, and Ollama Chat against the same
Electron-started engine. `m27-protocol-parity.json` records all checks true:

- HTTP 200 for every stream and non-stream call;
- 63 progressive visible-content deltas per streaming protocol;
- no reasoning deltas when thinking was disabled;
- identical twelve-line visible content across all four protocols and their
  non-stream counterparts;
- Chat stop/usage/`[DONE]`, one Responses completed event, Anthropic
  `message_stop`, and Ollama `done:true` terminal completion.

`m27_reasoning_stream.py` enabled native reasoning and records, for each
protocol, 369 reasoning deltas followed by eight visible-content deltas, 1,359
reasoning characters, exact visible `M27-REASON-STREAM-DONE`, distinct
reasoning/content fields, no think-tag leakage, and a completed terminal.

## Tool continuation and multi-turn Electron behavior

`m27_responses_tool_loop.py` and `m27-responses-tool-loop.json` prove a raw
Responses loop:

- round one emitted exactly one `file_info` with
  `{"path":"panel/package.json"}`, two progressive argument deltas, no
  visible prose, and a terminal response ID;
- the real result reported `5.2 KB`;
- round two used `previous_response_id`, emitted no second tool call, streamed
  14 visible deltas, exact-finaled `M27-RESP-TOOL-DONE SIZE=5.2 KB`, and
  completed.

The real Electron UI independently proves:

- row 641 streamed a visible Reasoning rail before exact final
  `M27-JT-UI-REASON-A-DONE`;
- row 644 used a different prompt and different reasoning bytes, then exact
  finaled `M27-JT-UI-REASON-B-DONE`;
- row 647 visibly entered the `file_info` execution/processing state, executed
  it exactly once with the required path, and exact-finaled
  `M27-JT-UI-TOOL-DONE SIZE=5.2 KB`;
- same-chat rows 650 and 653 recalled prior facts/markers without a second tool
  call and retained `paged+tq-native` prefixes.

Screenshots preserve the in-flight reasoning rail, in-flight tool state, final
tool result, and the three-turn final state.

## Source validation

- `cache-tests.txt`: 190 selected paged/disk/TQ/hybrid/native-cache tests
  passed on the owning `8a93aa910` source repair.
- `protocol-tests.txt`: 99 API-surface/Anthropic/Ollama adapter and streaming
  tests passed on the same source cutoff.
- `b31fdca95` adds only the preceding LFM proof documents; no engine/panel
  source changed between those test logs and this M2.7 live run.
- The three committed raw drivers revalidate their own terminal/delta/tool
  invariants. Every boolean under `checks` in
  `m27-protocol-parity.json`, `m27-reasoning-stream.json`, and
  `m27-responses-tool-loop.json` is `true`.

## Honest boundary

This gate verifies the exact MiniMax M2.7 JANGTQ artifact and the current
full-KV q4 cache/protocol route. It does not close MiniMax M3 MSA/VL/video,
typed DSV4/openPangu caches, hybrid SSM/GDN companion rederive, mixed-SWA,
gateway network-loss soak, signed-app repetition, broader eager-load routes,
or the remaining family matrix. The public v1.6.11 checkpoint remains the
latest release; no v1.6.12 release claim is made here.
