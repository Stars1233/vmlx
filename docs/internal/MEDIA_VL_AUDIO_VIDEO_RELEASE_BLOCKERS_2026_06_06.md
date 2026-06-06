# Media / VL / audio / video release blockers

Date: 2026-06-06

Active worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`

Reporter credit for release notes: GitHub `@Hornsan1`

## Release rule

Do not tag, notarize, publish, or update public vMLX / mlxstudio downloads until the exact family, quant, runtime, modality, API path, cache mode, streaming mode, and UI/settings path being claimed has current proof.

Load-only, health-only, text-only, or fail-closed unsupported-route proof does not clear media support.

## Current hard answer

- MiMo V2.5 JANG_2L local Python/vMLX is not a fully working VL/audio/video model today.
- MiMo V2.5 JANG_2L local Python/vMLX is not a 40+ tok/s path today.
- Max2 docs contain a historical Swift TP4/JACCL/RDMA MiMo proof around `39.228 tok/s` decode with chat, Responses, streaming, cache reuse, L2 disk restore, and rank agreement.
- That Swift proof is stale for current release purposes until relaunched and reproved.
- Current Max2 live gateways on ports `8124` and `8125` are Qwen3.6 TP4 gateways, not MiMo.
- Current local MiMo quant endpoint on `127.0.0.1:8897` is healthy, but source-vs-quant proof is blocked because no MiMo source endpoint is currently healthy.
- MiMo source shards exist on Pod 1 nodes under `/opt/adlab/models/tp4-source/MiMo-V2.5`, but materializer status is stale/incomplete: rank0/rank2/rank3 report 93 output safetensors, rank1 reports 81, and latest old materializer logs ended `rc=143`.

## Functions that must exist before media release

| Function | Current state | Blocking work |
|---|---|---|
| Artifact capability reader | Partial. `/v1/models/{id}/capabilities.media` now separates runtime-proven modalities from preserved/unwired sidecars. | Audit every family config/jang_config/preprocessor/tokenizer/audio/video sidecar and keep advertised capability separate from runtime support. |
| Media request normalization | Partial. Text and some image/video smoke paths exist. | Normalize OpenAI image/audio/video content parts, data URLs, app-local files, safe remote URLs, and text-only fallback without losing special media tokens. |
| Processor and embedding bridge | Partial by family. Gemma4 image path has live proof; MiMo Python media bridge is unbuilt. | Wire family processors to `pixel_values`, video frames, audio features, media masks, `inputs_embeds`, rope/position metadata, and family-specific kwargs. |
| Text-after-media recovery | Partial. Gemma4 12B installed app has small-image and forced-reject recovery proof. | Repeat across every advertised media family and after success, oversized media rejection, parser error, timeout, and cancellation. |
| Prefix cache with media | Partial. Some text/cache rows pass. | Salt prefix keys by media digest, media-expanded token layout, processor config, frame sampling, audio features, and modality. |
| Paged KV cache with media | Partial. Text/native cache rows pass for some families. | Prove media-expanded prefill and decode do not reuse stale text-only states or cross-media states. |
| L2 disk cache with media | Partial. Text and some TP4 proof rows exist. | Prove store/restore for media prompts and text-after-media, including app restart where required. |
| TurboQuant KV with media | Partial by family. | Prove only compatible cache slots are compressed; mixed full/SWA, CSA/HSA, rotating, SSM, and media prefill states must not be corrupted. |
| Continuous batching with media | Open. | For one-shot media prefill families, either chunk safely or reject typed 413 before native forward; no process death. |
| Streaming parity | Partial. Gemma4 source/installed text streaming proofs exist. | Prove streaming/non-streaming parity for image, video, audio, tool calls, cancellation, and post-error recovery. |
| Tool calls after media | Open. | Real OpenAI `tool_calls`, tool-result continuation, loop stopping, and parser no-leak proof after media turns. |
| Structured JSON/XML repair | Partial. Caller-side JSON/XML repair exists; guided decoding is not proven. | Add benchmark/catalog retry-on-invalid where needed and report raw parse failure separately from repaired score. |
| UI/settings parity | Partial. Panel now gates media controls by runtime-proven capability. | Electron UI proof for model selection, max output tokens, cache toggles, media upload, media rejection, post-error text recovery, and `/health`/`cache/stats` parity. |
| Packaged/notarized parity | Open for latest source. | Rebuild, sign, notarize, install, and rerun source-vs-installed proofs only after live rows are green. |

## Family rows

| Family | Current state | Release blocker |
|---|---|---|
| Gemma4 12B MXFP4/MXFP8/JANG_4M | Text speed around 46.6 tok/s passed for JANG_4M; image small/reject recovery passed in installed app; media capability contract is honest. | Full audio/video depth, tools after media, larger media prompts, streaming media, UI/settings parity across cache modes. |
| Gemma4 26B/31B | Source smoke and installed visible/speed rows have partial proof. | Broader packaged media/tool/cache parity and long media-depth matrix. |
| Qwen3.6 35B MXFP8 MTP | `gdn_sink` dense MTP crash does not reproduce on current source/installed 1.5.56 proof. | If a user sees it on older app, it is stale packaged runtime; if on fresh app, reproduce the exact bypass route. Media/VL/MTP matrix remains open. |
| Qwen3.6 27B MXFP8/JANG_4M MTP | Local installed decode works; deterministic policy improves decode speed. Fresh Pod1 TP4 `batchfix3` run proves native MTP autodetect depth 2, rank agreement, streaming, multiturn, Responses chain cache, L2 disk cache/restore, and batch rank correctness. | TP4 remains release-open because hybrid attention-cache evidence is missing, decode speed is low, localhost can hit a stale depth-0 gateway, and media/VL proof plus UI/app policy parity remain open. |
| Step3.7 Flash JANG_2L CRACK | Text-only vMLX route is stable when metadata marks `has_vision=false`; default advertised VLM path can crash. | Runtime must fail closed for unsupported Step3p7 VLM and real Step3p7 VLM implementation remains open. Tool loops/raw dialect leaks remain model/runtime compatibility blockers. |
| MiMo V2.5 JANG_2L | Local quant endpoint healthy; text/cache narrow proof exists; source-vs-quant blocked by missing source endpoint; VL/audio sidecars are preserved but unwired in Python/vMLX. | Python media processor/embedding bridge unbuilt, source endpoint missing, long prompt/system-role stop, tool args/continuation quality, speed, restart-L2, full cache matrix. |
| LFM 2.5 | Installed UI text/cache/tools proof exists. | Any VL/audio/video advertised variants still need full capability audit and media proof. |
| Nemotron Omni | Static/source rows partial. | Omni audio/image/video processor bridge, cache safety, streaming, tool/JSON, UI and packaged proof. |
| DSV4 / JANGTQ | Text/cache smoke exists; exact long-output/code/file-generation remains red. | Do not expand claims to media until exact text/cache quality and memory-gated UI rows are resolved. |
| MiniMax / JANGTQ_K | Reporter parity/root cause still open. | Installed reporter route, cancel lifecycle, media if advertised, tool/JSON parity. |
| ZAYA VL/text | Text/VL smoke partial; reasoning-on visible output still red in one row. | Reasoning/template/media/cache/tool matrix. |

## MiMo source-vs-quant gate

Required artifact:

```text
build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json
```

Current state:

- Quant endpoint: healthy at `http://127.0.0.1:8897`, served model `mimo-v2-jang2l`.
- Source path on Max2: `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`.
- Source TP4 rank paths on Pod 1: `/opt/adlab/models/tp4-source/MiMo-V2.5/rank0..rank3`.
- Current source endpoint: missing/unhealthy.
- Current live Max2 gateways: Qwen3.6 TP4 on `8124` and `8125`, not MiMo.
- Current Pod 1 resident `TPRankWorker` processes are Qwen workers. The AdLab
  resident launch script treats live `TPRankWorker` processes as a preflight
  failure; `TP_ALLOW_EXISTING_TP_WORKER=1` intentionally does not bypass
  live/stuck workers. Relaunching MiMo source TP4 therefore requires replacing
  the current Qwen worker lane or using another pod.

Next valid actions:

1. Coordinate before disrupting current Qwen TP4 workers.
2. Relaunch the Swift MiMo TP4 source path from the documented recipe on a free port, for example `8126`.
3. Rerun the MiMo source-vs-quant first-divergence producer against source `8126` and quant `8897`.
4. Classify each failing prompt as `source_and_quant_match`, `quant_diverges_from_source`, or `source_also_fails`.
5. If source also fails, fix runtime/decode/template/cache. If only quant fails, fix the quant/model upload.

## Qwen3.6 27B MXFP8 TP4 batchfix evidence

Fresh run:

```text
run_id: qwen36-mxfp8-pod1-tp4-batchfix3-20260606T165440Z
artifact: build/current-qwen36-27b-mxfp8-tp4-batchfix3-hostip-proof-20260606.json
remote artifact: /tmp/adlab-qwen36-27b-mxfp8-tp4-batchfix3-hostip-proof-20260606.json
base_url used for valid proof: http://100.98.111.49:8124
served model: qwen36-27b-mxfp8-tp4-batchfix3
model path: /opt/adlab/models/qwen36-27b-mxfp8-mtp
```

What passed:

- Health ready: `4/4` rank targets.
- Native MTP autodetect: launcher resolved `TP_NATIVE_MTP_DEPTH=auto` to depth `2` from artifact metadata.
- Native MTP runtime proof row passed with `active_depth=2`, `runtime_active=True`, `accepted_tokens=32`, `verify_calls=11`, and `avg_committed_per_verify=2.909`.
- Chat, multiturn chat, Responses API, and Responses chain passed.
- Chat streaming and Responses streaming passed.
- Parser leak check passed.
- Loop check passed.
- Token authority stayed `rank0_send_recv_per_slot`.
- Rank agreement passed for all rows, including `batch_throughput`; `bad_rows=[]`, `mismatched_ranks=[]`.
- Batch throughput row passed the safety floor: batch size `16`, high watermark `16`, `92.600 tok/s` against the diagnostic `1 tok/s` floor.
- Cache reuse passed.
- L2 disk cache and L2 disk restore passed.
- SSM-side hybrid cache evidence passed with `cache_stats.ssm.hit_count=14` and `live_cache.ssm_hits=14` on the batch row.

What still failed:

- Overall verdict: `FAIL`.
- Failed check: `HYBRID_SSM_CACHE`.
- Failure detail: `attention=[]`, while SSM evidence was present.
- This means the run proves SSM/L2 behavior but does not prove attention-side hybrid cache hit/reuse for the Qwen TP4 route.
- Decode speed is still not production-cleared: current health showed single-row decode around `8-12 tok/s` in this run, even though batch aggregate hit `92.600 tok/s`.

Operational issue found:

- The valid proof had to use Max2 host IP `http://100.98.111.49:8124`.
- `http://127.0.0.1:8124` on Max2 was contaminated by an orphaned stale gateway serving `qwen36-27b-mxfp8-tp4-pod1-shardedvocab` with native MTP depth `0` and old rank dirs.
- The stale gateway bound `127.0.0.1:8124`; the correct `batchfix3` gateway bound `0.0.0.0:8124`.
- The AdLab proof wrapper must refuse or kill stale loopback gateways before running localhost proof, or it can test the wrong model.

Remote AdLab fixes applied during this run:

- `scripts/engine-patches/adlab-tpworker-direct-file-serve-decode-patch.py` already contains the multi-rank batch safety condition `if promptBatches.count > 1 && !group.isMultiRank`.
- `scripts/adlab-qwen36-tp4-api-proof.py` now reports `bad_rows` for rank agreement failures.
- `scripts/engine-patches/adlab-tp-hotpath-timing-patch.py` was fixed so worker sampler timing is optional/idempotent when the sampler layout has already been transformed by send/recv/direct-decode patches.
- Direct Max2 regression: `python3 scripts/test-adlab-tp-hotpath-timing-patch.py` passed.
- `scripts/adlab-qwen36-tp4-resident-api-proof.sh` now supports `TP_PROOF_BASE_URL` and validates that `/health` resolves the expected served model and current run-id rank targets before running proof.
- Direct Max2 regression: `bash -n scripts/adlab-qwen36-tp4-resident-api-proof.sh` passed.
- Direct Max2 regression: `python3 scripts/test-adlab-qwen36-tp4-resident-api-proof.py` passed.

Current classification:

- Prior batch rank-divergence is fixed/narrowed by the `batchfix3` run.
- Remaining Qwen TP4 blocker is `kernel_cache`: attention-side hybrid cache evidence is missing for TP4.
- Separate blocker is `gateway_ui`: stale localhost gateway contamination can invalidate proof traffic.
- This is not a model upload corruption finding.

## No fake fixes

- Do not fold system prompts into user prompts to hide MiMo first-token `<|im_end|>`.
- Do not mark MiMo media supported because vision/audio weights are preserved.
- Do not disable prefix cache, paged cache, L2 disk cache, TurboQuant KV, MTP, thinking, or tools and call the model release-cleared unless that disabled mode is the documented release behavior.
- Do not call Step3p7 VLM fixed because text-only metadata avoids the crash.
- Do not count JSON repair as native guided decoding.
- Do not count historical Swift TP4 proof as current vMLX packaged release proof.

## Current execution order

1. Reprove or relaunch MiMo source TP4 endpoint without losing the Qwen TP4 evidence lane.
2. Run MiMo source-vs-quant first-divergence and classify runtime versus artifact.
3. Build Python/vMLX MiMo media bridge or explicitly keep MiMo media `preserved_unwired`.
4. Close Qwen TP4 batch/rank/speed blockers.
5. Finish Gemma4 audio/video/tool/media UI matrix.
6. Reproduce Step3p7 unsupported VLM fail-closed guard and keep real VLM implementation open.
7. Run full UI/settings/cache/media matrix from source and installed app.
8. Only then rebuild, sign, notarize, tag, and publish.
