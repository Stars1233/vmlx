# vMLX / MLXStudio Production Release Blocker Matrix - 2026-06-06

## Release state

Release status: red.

Do not package, sign, notarize, tag, upload, update appcast/downloads, or call the build production-ready while any row below is red.

Current active worktree:

`/Users/eric/mlx/vllm-mlx-finite-launch-guard`

Deprecated paths not allowed for this work:

- `/Users/eric/vmlx`
- `/Users/eric/vmlx/swift`
- ADLab/TB/RDMA transport work unless explicitly requested for that exact lane

## Required evidence for green

A family row is green only when the relevant model artifacts pass current-source and installed-app proof across:

- `/v1/chat/completions`
- `/v1/responses`
- Anthropic `/v1/messages` where supported
- Ollama chat/generate where supported
- streaming and non-streaming
- 3+ turn multiturn recall/persona/context
- tool required, tool auto, tool-result continuation, and loop stop
- JSON/XML structured output parseability and repair diagnostics
- code/syntax/whitespace-sensitive output
- max output tokens and max context behavior
- prefix cache hit/miss correctness
- paged cache where compatible
- architecture-specific native cache where required
- TurboQuant KV encode/decode/restore where valid
- L2 disk cache write, restart/restore, and hit
- cancellation, timeout, unload/reload, post-error recovery
- UI settings parity with CLI args, `/health`, `/v1/cache/stats`, capabilities, and launched process
- media/VL/audio/video rows if the runtime advertises those modalities

No source-only, load-only, health-only, or one-prompt text smoke may clear a broad row.

## Current family matrix

| Family / artifact lane | Current status | Proven current positives | Current blockers | Next proof/fix |
|---|---:|---|---|---|
| MiMo V2.5 JANG_2L | Red | Text `ACK`; paged cache hit `cached_tokens=67`; L2 block write; multiturn `blue cat`; native mixed full/SWA cache detected; generic flat TQ-KV skipped for rotating cache | Required XML tool call corrupts output; speed around 1-2 tok/s; long/system prompt quality not cleared; VL/audio/video unwired | Source-vs-quant first divergence or replacement artifact proof; then tool decode/template/runtime fix; then full cache+tool+media matrix |
| Qwen 3.6 35B MXFP8 MTP | Partial | Bundled-engine smoke passes text/cache, multiturn, reasoning, required tool, image, video, post-media text recovery; native MTP active D3; paged+SSM hit; block + SSM L2 evidence; no `gdn_sink` TypeError | Responses/Anthropic/Ollama, streaming parity, real Electron UI settings, largest-context cache, restart/L2 restore, cancellation/recovery, and 27B parity incomplete | Run missing API/UI/restart/largest-context rows and then 27B parity |
| Qwen 3.6 27B MXFP8/JANG_4M MTP | Partial | Native MTP READY D3; deterministic policy can clear decode floor for MXFP8; JANG_4M decode better than MXFP8 | MXFP8 stochastic defaults slow; PP rows low; TP4 route rank/speed evidence remains red; full UI/cache/tool matrix open | Prove deterministic policy in UI/session; PP root cause; multiturn/tool/L2 proof |
| Nemo / Nemotron Omni | Red | Some source rows exist in older matrix | Omni audio/video processor bridge, tool dialect, cache/media salt, UI proof incomplete | Build live Omni text/audio/video/tool/cache smoke |
| LFM / LFM2.5 | Partial | Source rows exist in older cross-family matrix | Full installed-app multiturn/tool/cache/UI proof incomplete | Run installed-app LFM matrix with loop stop and structured output |
| MiniMax / MiniMax-M2.7 / JANGTQ_K | Red | Public/local route drift narrowed; public DMG cancel route present | Reporter parity/root cause still open; JANGTQ/runtime/speed/tool/cancel recovery not fully cleared | Finish reporter provenance and live MiniMax model proof |
| DSV4 Flash / DeepSeek-V4-Flash-JANGTQ-K | Red | Native composite cache smoke passed; required tool row passed in focused smoke | Long-output/code/file-generation exactness red; thinking-closed rail corrupts identifiers; memory-gated UI rows open | Fix/diagnose exact-code rail and rerun long/code/file-generation proof |
| Step 3.7 Flash | Partial | Text-only guard prevents unsupported advertised-VLM crash; source/staged proof of media rejection and text recovery exists | Real VLM not implemented/proven; tool loops/raw XML and multiturn synthesis still open; installed/public parity depends on release | Live installed-app text/tool/multiturn matrix; keep VLM fail-closed until real support lands |
| Gemma 4 12B MXFP4/MXFP8/JANG_4M | Partial | JANG_4M source/installed speed around 46 tok/s; image small/reject text recovery proof exists | Full VL/audio/video/tools/cache/UI matrix incomplete; speed rows per quant not all cleared | Run per-quant media/tool/cache/UI matrix; verify image prefill guard on installed app |
| ZAYA/ZAYA-VL | Partial | ZAYA text pass; ZAYA-VL capability truth and some media/tool/cache rows pass | Reasoning/media/UI matrix needs current installed-app proof | Refresh installed-app ZAYA/ZAYA-VL live matrix |
| Hybrid SSM / nonstandard cache families | Red | Some native cache contracts exist | Async rederive/companion state/L2 restore not exhaustively proven per family | Add architecture-specific cache matrix rows and prove no plain-KV substitution |

## Cross-cutting blockers

### CACHE-001: architecture-specific cache matrix incomplete

Required:

- Standard full-KV: generic TurboQuant KV encode/decode/restore parity.
- MiMo: native mixed full/SWA rotating cache only; no flat generic TQ-KV.
- DSV4: native SWA+CSA/HCA composite cache only.
- ZAYA CCA: typed native CCA cache restore.
- Hybrid SSM: companion state plus async/rederive proof.
- MLLM/media: media-salted prefix/L2 keys and text-after-media recovery.

Current status: red.

### TOOL-001: required/auto tool calls and continuation incomplete across families

Required:

- `tool_choice=required` returns real OpenAI `tool_calls` or classified model/runtime failure.
- Tool result continuation stops and summarizes.
- No raw XML leaks, malformed JSON, repeated tool loops, or hidden reasoning leaks.

Current status: red because MiMo required tools corrupt and Step/LFM/MiniMax/Nemo matrix remains incomplete.

### MEDIA-001: VL/audio/video runtime incomplete

Required:

- Actual media tensors/masks routed through model-specific processor/forward code.
- Capabilities distinguish runtime-supported vs preserved-unwired sidecars.
- UI upload/recovery/settings proof.

Current status: red. MiMo media is preserved but unwired; Step3.7 VLM is guarded text-only; Gemma4 has partial image proof only.

### UI-001: real Electron UI matrix incomplete

Required:

- Installed app settings panel matches source capabilities.
- Cache/MTP/media/max-output/max-context controls launch correct process args.
- Chat UI recovers after media rejection, timeout, cancel, parser failure.
- Streaming and non-streaming parity.

Current status: red.

### RELEASE-001: signing/notarization/public update locked

Required:

- Current release manifest `prepackage_ready=true`.
- Current release manifest `release_ready=true`.
- Source proof green.
- Bundled Python parity green.
- Installed/staged app parity green.
- Real UI matrix green.
- Release notes credit GitHub `@Hornsan1`.

Current status: red. No release action allowed.

## Current next actions

1. Keep MiMo red; direct continuation proof now shows the current installed JANG_2L path cannot naturally continue valid XML tools even outside Chat tool parsing.
2. Obtain a corrected MiMo artifact or build a real constrained/guided XML decoder; do not synthesize fake tool calls from `tool_choice=required`.
3. Only delete old/local MiMo copies after a replacement artifact is proven better and exact paths are enumerated.
4. Add harness knobs for per-family cache/TQ/L2 diagnostics so A/B rows are reproducible instead of manual one-off commands.
5. Refresh cross-family smoke matrix after the harness can record runtime modalities and architecture-specific cache evidence.
6. Commit/push only proven source/test/docs/proof artifacts, excluding `.agents/`, local AGENTS override, vendor noise, binary cache dirs, and release outputs.

## 2026-06-06 MiMo KV-none harness update

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-kvnone-extra-args-20260606/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Finding:

- `--kv-cache-quantization none` is now reproducible through the smoke harness.
- MiMo text/cache/multiturn stay green with KV quantization disabled.
- Required tool call still corrupts into `<tool_call>` plus punctuation/fullwidth-comma garbage.
- This formally rules out q4 KV storage and generic TurboQuant KV as the root cause of the MiMo tool failure.

Matrix impact:

- `MiMo V2.5 JANG_2L` remains red.
- `CACHE-001` improves for MiMo text cache evidence but remains red for largest-context, restart/L2 restore, and cross-family rows.
- `TOOL-001` remains red.

## 2026-06-06 MiMo direct XML continuation probe

Artifact:

`build/current-mimo-v25-tool-continuation-probe-20260606`

Command boundary:

- Installed model: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Runtime: SimpleEngine, no continuous batching, prefix cache disabled, `--kv-cache-quantization none`.
- API: `/v1/completions`, not Chat Completions tool parsing.
- Prompt: real MiMo chat template plus concrete `mimo_xml_function` fallback example.

Results:

- Base prompt generated repeated `<think>` plus punctuation/fullwidth-comma garbage.
- Prompt prefilled through `<tool_call>\n<function=record_fact>\n<parameter=value>\n` generated only `blue` and then punctuation/newline garbage, not `blue-cat` and not closing XML.
- Base probe speed: 48 completion tokens in 50.42s, about `1.0 tok/s`.
- Prefilled-function probe speed: 48 completion tokens in 29.52s, about `1.6 tok/s`.

Updated classification:

- MiMo tool failure is not parser-only and not caused solely by Chat `tool_choice=required`.
- It reproduces outside tool parsing and after the runtime supplies the concrete function/parameter prefix.
- Do not implement a fake runtime tool-call synthesis as a release fix.
- Valid paths remain either a corrected MiMo artifact that naturally emits the XML tool grammar, or a real constrained/guided XML tool decoder proven end-to-end without fabricating arguments.
- MiMo speed remains red until an optimized routed-expert decode path is implemented/proven; current generic path is around 1-2 tok/s on this 106GB bundle.

## 2026-06-06 MiMo remote runtime delta and compiled-router fallback

Remote checked:

`erics-m5-max2.local:/Users/eric/jang`

Findings:

- Local bundled `jang_tools.mimo_v2.mlx_model.py` SHA-256: `7b6a0a524b486907f5e2fd2c5a122c1dc58726be1d8d6c2c1304a9aafda57f7d`.
- Remote `~/jang/jang-tools/jang_tools/mimo_v2/mlx_model.py` SHA-256: `a33ec86fc409e83feccf049c74d4c19be813ec6f5d130a6a20bf19d713d9372d`.
- Remote runtime includes a compiled single-token MiMo router and JANGTQ weighted decode hooks missing from the local bundled package.
- Remote quant contract says the coherent canonical affine JANG_2L path measured about `1.974-2.640 generation tok/s` and explicitly is not a `30 tok/s` path on the local M5 Max.
- The faster `5.626 generation tok/s` fit candidate failed arithmetic and was rejected.

vMLX source change:

- vMLX MiMo registration now installs a fallback compiled decode router when bundled `jang_tools` is stale.
- The patch is a no-op when newer `jang_tools` already exposes `run_compiled_mimo_decode_router`.
- This preserves vMLX `inputs_embeds` compatibility and does not synthesize tool calls.

Live proof:

`build/current-mimo-v25-compiled-router-live-probe-20260606`

- Server log confirms `Installed vMLX fallback compiled MiMo-V2 decode router`.
- Direct XML continuation still generated `blue` plus punctuation/newline garbage instead of `blue-cat` and closing XML.
- First short request speed was 16 completion tokens in 30.67s, about `0.5 tok/s`; this does not clear speed.

Matrix impact:

- `MIMO-TOOL-001` remains red.
- `MIMO-SPEED-001` remains red.
- Runtime stale-package integration is partially fixed, but MiMo remains blocked for release.

## 2026-06-06 Qwen3.6 35B MXFP8-MTP bundled smoke proof

Artifact:

`build/current-all-local-model-smoke-qwen35-mxfp8-mtp-tools-media-20260606/JANGQ_Qwen3.6-35B-A3B-MXFP8-MTP/result.json`

Command boundary:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`.
- Runtime Python: `panel/bundled-python/python/bin/python3.12`.
- Served with MLLM route, continuous batching, paged cache, block-disk L2, `--ssm-state-cache-mb 1024`, and default thinking off.
- Source regression check also passed: `tests/test_engine_audit.py -k "qwen35_dense_mtp_patch"` with 2 tests passed.

Result:

- Overall row: `pass`, zero failures.
- Model type: `qwen3_5_moe`.
- `is_mllm=true`, supports video, thinking, tools, and MTP.
- Native MTP: `runtime_active=true`, `effective_depth=3`, source `vmlx_mtp_tuning.json:native_mtp.best_depth`, runtime scope `text+vl`, artifact tensors present (`mtp_tensor_count=31`).
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=56`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `937`.
- `tool_required`: HTTP 200, visible content empty, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value": "blue-cat"}`.
- `vl_blue_image`: HTTP 200, visible `blue`.
- `vl_blue_image_repeat`: HTTP 200, visible `blue`.
- `vl_red_image_changed`: HTTP 200, visible `Red`.
- `vl_blue_video`: HTTP 200, visible `Blue`.
- `text_no_media_after_image`: HTTP 200, visible `NONE`.
- `text_no_media_after_video`: HTTP 200, visible `NONE`.
- Logs show repeated `MLLM native MTP path activated ... depth=3`.
- Logs show a real acceptance row on multiturn: `accepted=3/3 (100.0%)`.
- Logs show reasoning row MTP throughput: `242 tokens in 2.98s (81.3 tok/s)`.
- Logs show tool row: `27 tokens in 0.71s (37.8 tok/s)`.
- No `gdn_sink` `TypeError` appears in the smoke log.

Cache / L2 / architecture evidence:

- Native cache: `qwen3_5_moe` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV is enabled only for hybrid attention KV with reason `hybrid_attention_kv_only`.
- Live attention TQ-KV is active for attention KV layers only; SSM companion state stays native full precision.
- Storage-boundary attention KV quantization is q4, group size 64, attention layers only.
- Block disk L2 after run: 10 blocks, 431 block tokens on disk, 3 disk hits.
- SSM companion L2 after run: 9 disk entries, 687 SSM tokens on disk, about 565 MB.
- Cache totals report 1118 total L2 tokens across block and SSM stores.
- Media requests correctly skip VLM prefix cache store because media embeddings are path-dependent.

Nuance / still open:

- This is a strong bundled-engine smoke for Qwen35 MXFP8-MTP, but it is still not full release clearance.
- Missing rows: `/v1/responses`, Anthropic `/v1/messages`, Ollama, streaming/non-streaming equivalence, installed Electron UI settings, largest-context cache, restart/L2 restore, cancellation/recovery, and 27B parity.
- MTP acceptance is prompt-dependent: some short rows finish from seed/init emits with no draft cycles, while multiturn/reasoning/tool rows do exercise MTP differently.
- Keep Qwen 3.6 family status `Partial` until the missing API/UI/largest-context/restart rows are current.

## 2026-06-06 LFM2.5 MXFP4 focused source smoke

Artifact:

`build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260606/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json`

Result:

- Overall row: `pass`.
- Model type: `lfm2_moe`.
- Cache family: `hybrid_ssm`.
- Native cache: `lfm2` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV correctly disabled with reason `hybrid_ssm_state`.
- Attention KV storage quantization active: q4, group size 64, storage-boundary only.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=64`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.
- Block disk L2 after tool row: 6 blocks on disk, 265 block tokens, 265 SSM companion tokens, 3 disk hits.

Nuance / still open:

- Exact visible output is correct, but short exact prompts report high `completion_tokens` counts (`208`, `380`, `252`). This needs a follow-up parser/stop-token/accounting check before calling LFM fully clean.
- This is source-runtime no-media proof only. Installed-app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, and UI settings remain open.

## 2026-06-06 Nemotron Omni MXFP4 focused source smoke

Initial artifact:

`build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json`

Initial result:

- Text/cache/multiturn/tool passed.
- `reasoning_on` failed because the default 256-token reasoning probe stopped with hidden reasoning only and empty visible output.

Harness fix:

- Nemotron/Omni reasoning probes now receive the same 512-token budget treatment as ZAYA while retaining strict visible `FINAL=OK` validation.

Passing artifact:

`build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-after-reasoning-budget-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json`

Result:

- Overall row: `pass`.
- Model type: `nemotron_h`.
- Cache family: `hybrid_ssm`.
- Native cache: `nemotron_h` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV correctly disabled with reason `hybrid_ssm_state`.
- Attention KV storage quantization active: q4, group size 64, storage-boundary only.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=57`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `1824`, completion tokens `488`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.
- Logs show hybrid paged hit plus SSM clean companion rederive and next-fetch-ready storage.

Nuance / still open:

- Server log warns `nemotron_h: unresolved eos strings ['<|im_end|>']`; registry/tokenizer EOS mapping needs follow-up before full release clearance.
- This is source-runtime no-media proof only. Omni audio/video/image bridge, installed app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, and UI settings remain open.

## 2026-06-06 LFM2.5 MXFP8 focused source smoke

Artifact:

`build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260606/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json`

Result:

- Overall row: `pass`.
- Model type: `lfm2_moe`.
- Cache family: `hybrid_ssm`.
- Native cache: `lfm2` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV correctly disabled with reason `hybrid_ssm_state`.
- Attention KV storage quantization active: q4, group size 64, storage-boundary only.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=64`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.
- Logs show hybrid paged hit plus SSM clean companion rederive and next-fetch-ready storage.

Nuance / still open:

- Same as MXFP4: visible output is exact, but short exact rows report high completion-token counts (`156`, `229`, `128`). Follow-up needed for parser/stop-token/accounting before full LFM clearance.
- This is source-runtime no-media proof only. Installed-app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, and UI settings remain open.

## 2026-06-06 Step3.7 Flash JANG_2L CRACK focused source smoke

Initial artifact:

`build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json`

Initial result:

- Overall row passed, but the smoke harness incorrectly classified the advertised-vision Step3.7 artifact as `is_mllm=true` and passed `--is-mllm`.
- Server correctly overrode the advertised VLM route to text-only, but future proof should not depend on that override.

Harness fix:

- Step3p7 advertised media is classified text-only for the smoke launcher.
- Step3p7 cache family is classified as mixed SWA/full KV (`swa_rotating` in harness terms), not hybrid SSM.

Passing artifact after harness fix:

`build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-textonly-harness-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json`

Result:

- Overall row: `pass`.
- Served command did not include `--is-mllm`.
- Server loaded `BatchedEngine ... (mllm=False)`.
- Server log shows `tier=step3p7_advertised_vlm_text_only result=False`.
- Model type: `step3p7`.
- Runtime media: `text` only.
- Declared media: `vision`, `image`, `video` are declared but not runtime supported.
- Native cache: `step3p7` / `mixed_swa_kv_v1` / `step3p7_full_sliding_kv`.
- Storage-boundary KV quantization active: q4, group size 64.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=61`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `390`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.

Nuance / still open:

- This proves the current text-only guard and text/tool/cache route, not Step3.7 VLM.
- The bundle still advertises vision/image/video while runtime support is intentionally text-only. Model upload metadata should be clarified or vMLX-specific metadata supplied.
- Source-runtime no-media proof only. Installed-app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, real Step3.7 VLM, and UI settings remain open.

## 2026-06-06 Gemma4 12B JANG_4M focused source smoke

Initial artifact:

`build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-20260606`

Initial result:

- Both local rows structurally passed:
- `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`
- `/Users/eric/models/dealignai/Gemma-4-12B-it-JANG_4M-CRACK`
- The `tool_required` probe returned a valid OpenAI `tool_calls[0].function.name=record_fact` with arguments `{"value": "blue-cat"}`.
- The same response also leaked visible content `thought`, which is not acceptable for a tool-only response.
- The harness did not previously reject this marker leak, so the initial pass was not release-clean proof.

Harness/server fix:

- `tool_required` validation now rejects visible content alongside tool calls as `tool_visible_text_leak`.
- Chat/Responses finalization now drops only known hidden-channel marker literals (`thought`, `analysis`) when native tool calls are present.
- The fix does not force a fake tool call, does not rewrite tool arguments, and does not hide arbitrary visible text. It only removes leaked channel marker text after native tool parsing has already produced tool calls.
- Gemma4/Gemma4 unified harness classification is `swa_rotating`, matching runtime mixed full/sliding-window KV, not hybrid SSM.

Failing artifact after stricter validation:

`build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-tool-leak-check-20260606`

Failure:

- Both local rows failed only `tool_visible_text_leak`.
- Leaked visible content: `thought`.
- Native OpenAI tool call was still present and correctly parsed.

Passing artifact after server cleanup and cache-family fix:

`build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-cache-family-fix-20260606/JANGQ_gemma-4-12B-it-JANG_4M/result.json`

Result:

- Overall row: `pass`.
- Model type: `gemma4_unified`.
- Harness cache family: `swa_rotating`.
- Runtime media declares `text`, `vision`, `audio`, and `video` as runtime supported.
- Capabilities: tools supported, thinking supported, `tool_parser=gemma4`, `reasoning_parser=gemma4`.
- Native cache: `gemma4` / `mixed_swa_kv_v1` / `mixed_swa_kv`.
- Components: `full_attention_kv`, `sliding_window_kv`, `rotating_window_metadata`.
- Generic TurboQuant KV correctly disabled for this native mixed-SWA route.
- Storage-boundary KV quantization active: q4, group size 64, applies to full and sliding attention KV, preserving rotating-window metadata.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`, completion tokens `6`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=56`, `cache_detail=paged+mixed_swa`, completion tokens `6`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `402`.
- `tool_required`: HTTP 200, visible content empty, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value": "blue-cat"}`.
- Cache after run: 5 block-L2 files, 227 block tokens on disk, cache hit tokens `56`.
- Short smoke generation stats: `generation_tps=50.28720244969087`.

Nuance / still open:

- This proves the source-runtime no-media text/cache/reasoning/tool path for JANG_4M only.
- It does not yet prove MXFP4 or MXFP8 Gemma4 rows.
- It does not yet prove real image/audio/video quality, VLM image prefill guard behavior, media cache salting under live media requests, or video/audio processing.
- It does not yet prove installed-app parity, MLXStudio UI settings, max-output-token settings, streaming, Responses/Anthropic/Ollama parity, largest-context cache, restart/L2 restore, or notarized app behavior.
- Runtime reports media as supported, but that remains release-red until real VL/audio/video E2E probes pass with visible semantic proof.
