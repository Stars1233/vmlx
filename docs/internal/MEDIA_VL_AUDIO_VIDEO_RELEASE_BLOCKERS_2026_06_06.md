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
- The local MiMo JANG_2L bundle was checksum-synced in place from `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` on 2026-06-06. Rsync deleted stale local `config.json.pre-text-runtime-metadata-20260606` and replaced the mismatched `config.json`; tokenizer, preprocessor, generation config, and safetensors index already matched.
- After sync, local Python/vMLX MiMo text API proof passed health, models, capabilities, chat, math, and Responses. This is a narrow text proof only.
- After sync, local Python/vMLX MiMo image request correctly fails closed as unsupported media because the loaded runtime media contract is text-only, then text recovery succeeds. This proves no crash/recovery regression, not VL support.
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
| Qwen3.6 27B MXFP8/JANG_4M MTP | Local installed decode works; deterministic policy improves decode speed. Fresh Pod1 TP4 `batchfix3` host-IP proof now passes native MTP autodetect depth 2, rank agreement, streaming, multiturn, Responses chain cache, L2 disk cache/restore, paged-incompatible hybrid SSM/L2 attention evidence, and batch rank correctness. | TP4 remains release-open because single-row decode speed is still low, localhost can hit a stale depth-0 gateway, and media/VL proof plus UI/app policy parity remain open. |
| Step3.7 Flash JANG_2L CRACK | Current source has Step3p7 advertised-VLM fail-closed/text-only routing and loader normalization pinned by focused tests; text-only vMLX route is stable when metadata marks `has_vision=false`. | Real Step3p7 VLM/media implementation and live media proof remain open. Tool loops/raw dialect leaks remain model/runtime compatibility blockers. Older installed runtimes may still crash if they predate the fail-closed routing. |
| MiMo V2.5 JANG_2L | Local quant bundle checksum-synced from Max2; local Python/vMLX text API proof passes; image route fails closed as text-only and text recovery passes; historical Swift TP4 source proof has chat/Responses/streaming/cache/L2/rank agreement around 39 tok/s. | Python media processor/embedding bridge unbuilt, current source endpoint displaced by Qwen TP4 workers, local Python speed far below 40 tok/s, long prompt/system-role stop, tool args/continuation quality, restart-L2, full cache matrix. |
| LFM 2.5 | Installed UI text/cache/tools proof exists. | Any VL/audio/video advertised variants still need full capability audit and media proof. |
| Nemotron Omni | Static/source rows partial. | Omni audio/image/video processor bridge, cache safety, streaming, tool/JSON, UI and packaged proof. |
| DSV4 / JANGTQ | Text/cache smoke exists; exact long-output/code/file-generation remains red. | Do not expand claims to media until exact text/cache quality and memory-gated UI rows are resolved. |
| MiniMax / JANGTQ_K | Local installed-app Python live cancel probe refreshed on 2026-06-06: model loads, TurboQuant KV is active across 62 layers, q4 KV round-trip validation passes, block disk cache initializes, Responses stream starts, cancel route returns HTTP `200`, and bad text is captured in reasoning summary rather than final visible output. Reporter-machine parity/root cause still open. | Reporter parity artifact and reporter installed hash provenance remain missing; reporter session/settings/model-file parity is not proven. Media if advertised, tool/JSON parity, and broader UI proof remain open. |
| ZAYA VL/text | Text/VL smoke partial; reasoning-on visible output still red in one row. | Reasoning/template/media/cache/tool matrix. |

## MiMo source-vs-quant gate

Required artifact:

```text
build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json
```

Current state:

- Quant endpoint: relaunched after checksum sync at `http://127.0.0.1:8897`, served model `mimo-v2-jang2l`.
- Local text proof artifact: `build/current-mimo-v25-jang2l-local-sync-runtime-proof-20260606.json`.
- Local text proof verdict: `PASS`.
- Local proof rows: health, models, capabilities, chat France, chat math, and Responses basic passed with no parser leaks or loop flags.
- Local speed is not acceptable for release: the France row took `46.814s` for a short 32-token-capped answer, confirming this is not the 40+ tok/s route.
- Local synced long/tool/cache proof artifact: `build/current-mimo-v25-jang2l-synced-long-tool-cache-proof-20260606.json`.
- Local synced long/tool/cache proof verdict: `FAIL`.
- Fresh failure after Max2 checksum sync:
  - `long_prompt_recall`: HTTP `200`, 584 prompt tokens, `54.256s`, expected `FINAL alpha17 alpha63` missing, output starts `FINAL the the` then repeated punctuation.
  - `cache_repeat_1` and `cache_repeat_2`: HTTP `200`, exact `ACK-CACHE-742` passed for a short prompt.
  - `tool_required`: HTTP `200`, `48.192s`, no `tool_calls`, emitted repeated CJK/punctuation text instead of the forced `record_fact` call.
- This rules out stale local `config.json` drift as the cause of MiMo long/tool failure. Current likely boundary remains artifact/full-forward quality and/or runtime tool-template compatibility, not missing sidecars.
- Local image proof artifact: `build/current-mimo-v25-jang2l-local-sync-image-proof-20260606.json`.
- Local image proof verdict: `PASS_FAIL_CLOSED`.
- Local image result: HTTP `400` with `/v1/chat/completions received unsupported media modality image because the loaded runtime is text-only. Supported modalities: text.`
- Local post-image text recovery result: HTTP `200`, visible text `recovered`.
- JANGQ/JANG-tools doc committed and pushed: `docs/mimo-v25-python-vl-runtime-gap-20260606.md` at commit `c00fed3`.
- Exact runtime gap: `vmlx_engine.models.mllm` installs a text-only `mlx_vlm.models.mimo_v2` compatibility wrapper over `jang_tools.mimo_v2.mlx_model`; `VisionConfig` and `AudioConfig` are stubs, `visual.*` / `audio_encoder.*` / `speech_embeddings.*` are filtered at load, `pixel_values` raises `UnsupportedMediaModalityError`, and no `jang_tools/mimo_v2/mimo_v2_multimodal.py` exists locally.
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

Fresh passing run:

```text
run_id: qwen36-mxfp8-pod1-tp4-batchfix3-20260606T165440Z
artifact: build/current-qwen36-27b-mxfp8-tp4-batchfix3-hostip-proof2-20260606.json
remote artifact: /tmp/adlab-qwen36-27b-mxfp8-tp4-batchfix3-hostip-proof2-20260606.json
base_url used for valid proof: http://100.98.111.49:8124
served model: qwen36-27b-mxfp8-tp4-batchfix3
model path: /opt/adlab/models/qwen36-27b-mxfp8-mtp
verdict: PASS
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
- Hybrid SSM cache passed for the paged-incompatible TP4 topology using SSM evidence plus L2 attention evidence:
  `attention_l2.cache_stats.disk.hit_count=22`, `attention_l2.cache_stats.disk.stores=33`,
  `cache_stats.ssm.hit_count=20`, and `live_cache.ssm_hits=20`.

What still blocks release:

- Overall proof verdict: `PASS`.
- Decode speed is still not production-cleared: current health showed single-row decode around `9-10 tok/s` in this run, even though batch aggregate hit `99.044 tok/s` against the diagnostic floor.
- Media/VL routes, UI/settings parity, and installed-app packaged parity remain unproven for this Qwen TP4 lane.

Operational issue found:

- The valid proof had to use Max2 host IP `http://100.98.111.49:8124`.
- `http://127.0.0.1:8124` on Max2 was contaminated by an orphaned stale gateway serving `qwen36-27b-mxfp8-tp4-pod1-shardedvocab` with native MTP depth `0` and old rank dirs.
- The stale gateway bound `127.0.0.1:8124`; the correct `batchfix3` gateway bound `0.0.0.0:8124`.
- The AdLab proof wrapper must refuse or kill stale loopback gateways before running localhost proof, or it can test the wrong model.

Remote AdLab fixes applied during this run:

- `scripts/engine-patches/adlab-tpworker-direct-file-serve-decode-patch.py` already contains the multi-rank batch safety condition `if promptBatches.count > 1 && !group.isMultiRank`.
- `scripts/adlab-qwen36-tp4-api-proof.py` now reports `bad_rows` for rank agreement failures.
- `scripts/adlab-qwen36-tp4-api-proof.py` now treats L2 disk evidence as attention-side hybrid evidence only when the diagnostics explicitly report a hybrid, paged-incompatible topology. This avoids a fake generic disk-cache pass while allowing the actual Qwen TP4 hybrid SSM/L2 route to prove cache reuse.
- `scripts/engine-patches/adlab-tp-hotpath-timing-patch.py` was fixed so worker sampler timing is optional/idempotent when the sampler layout has already been transformed by send/recv/direct-decode patches.
- Direct Max2 regression: `python3 scripts/test-adlab-tp-hotpath-timing-patch.py` passed.
- Direct Max2 regression: `python3 scripts/test-adlab-qwen36-tp4-api-proof.py` passed.
- `scripts/adlab-qwen36-tp4-resident-api-proof.sh` now supports `TP_PROOF_BASE_URL` and validates that `/health` resolves the expected served model and current run-id rank targets before running proof.
- Direct Max2 regression: `bash -n scripts/adlab-qwen36-tp4-resident-api-proof.sh` passed.
- Direct Max2 regression: `python3 scripts/test-adlab-qwen36-tp4-resident-api-proof.py` passed.

Current classification:

- Prior batch rank-divergence is fixed/narrowed by the `batchfix3` run.
- Prior Qwen TP4 hybrid-cache proof failure is closed for the paged-incompatible SSM/L2 topology by the host-IP proof2 run.
- Separate blocker is `gateway_ui`: stale localhost gateway contamination can invalidate proof traffic.
- Separate blocker is `runtime_perf`: single-row decode speed remains below the target release bar.
- This is not a model upload corruption finding.

## No fake fixes

- Do not fold system prompts into user prompts to hide MiMo first-token `<|im_end|>`.
- Do not mark MiMo media supported because vision/audio weights are preserved.
- Do not disable prefix cache, paged cache, L2 disk cache, TurboQuant KV, MTP, thinking, or tools and call the model release-cleared unless that disabled mode is the documented release behavior.
- Do not call Step3p7 VLM fixed because current source routes advertised VLM metadata text-only/fail-closed. That closes the crash-avoidance row only; real Step3p7 VLM remains open until media forward, cache, streaming, tool, and UI rows pass.
- Do not count JSON repair as native guided decoding.
- Do not count historical Swift TP4 proof as current vMLX packaged release proof.

## Step3.7 current source audit

Focused verification:

```text
.venv/bin/python -m pytest -q tests/test_step37_vlm_runtime_audit.py tests/test_jang_loader.py -k 'step3p7 or step37'
14 passed, 47 deselected
```

What this proves:

- Current source recognizes Step3p7 advertised VLM metadata and routes text-only instead of entering the unsafe VLM path.
- Loader normalization and JANG Step3p7 remaps are pinned.
- Step3p7 VLM runtime audit artifacts/tests still distinguish missing or incomplete `mlx_vlm.models.step3p7` implementations from a complete runtime.

What this does not prove:

- No live Step3p7 image/video/audio request has passed.
- No Step3p7 media cache, streaming, tool-call, or UI/settings row is release-cleared.
- Packaged installed apps older than this source may still show the crash behavior reported by users.

## MiniMax issue179 current local refresh

Artifacts:

```text
build/current-issue179-minimax-k-model-manifest-20260606-local-refresh.json
build/current-issue179-minimax-k-responses-cancel-probe-20260606-live-refresh.json
build/current-issue179-minimax-k-root-cause-audit-after-live-refresh-20260606.json
```

What passed locally:

- MiniMax-K model manifest sidecars passed.
- Installed app bundled Python served `/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K`.
- JANGTQ loaded with 186 TurboQuant groups and 186 replaced modules.
- Runtime cache layout reported `TurboQuantKVCache` for all 62 layers.
- KV cache quantization round-trip validation passed: bits `4`, group size `64`, test shape `(1, 4, 8, 128)`.
- Block disk cache initialized with a 10GB max directory.
- Responses stream started and produced response id `resp_3572b630aaa1`.
- Cancel route returned HTTP `200` with `{"success":true,"message":"Response resp_3572b630aaa1 cancelled"}`.

What remains open:

- The controlled probe captured bad text in `raw_reasoning_text`, including Chinese system-text leakage, while `raw_content_text` was empty at cancel time.
- The root-cause audit remains `open` because reporter-machine parity/provenance is still missing:
  reporter model shard/codebook hash parity, reporter installed server hash, reporter chat/session/settings parity, reporter response active-at-cancel, and reporter raw SSE lifecycle are not proven.
- This is not enough to close issue179 or release-clear MiniMax-K; it only updates the local installed-app cancel/runtime boundary.

## Installed app parity current audit

Artifact:

```text
build/current-installed-app-runtime-parity-audit-after-minimax-live-refresh-20260606.json
```

Status: `open`.

Failed checks:

- `installed_bundled_engine_hash_parity`
- `installed_packaged_engine_source_hash_parity`

Interpretation:

- The installed `/Applications/vMLX.app` has the expected runtime guards/routes in the checked surface, but it is stale relative to current source.
- Rebuild/sign/notarize/install is required after the remaining model/runtime blockers are fixed.
- Do not rebuild and release now: current release manifest still has MiMo and MiniMax blockers before package/signing.

## Current execution order

1. Reprove or relaunch MiMo source TP4 endpoint without losing the Qwen TP4 evidence lane.
2. Run MiMo source-vs-quant first-divergence and classify runtime versus artifact.
3. Build Python/vMLX MiMo media bridge or explicitly keep MiMo media `preserved_unwired`.
4. Close Qwen TP4 single-row decode speed, stale gateway, media/VL, and UI/app parity blockers.
5. Finish Gemma4 audio/video/tool/media UI matrix.
6. Reproduce Step3p7 unsupported VLM fail-closed guard and keep real VLM implementation open.
7. Run full UI/settings/cache/media matrix from source and installed app.
8. Only then rebuild, sign, notarize, tag, and publish.
