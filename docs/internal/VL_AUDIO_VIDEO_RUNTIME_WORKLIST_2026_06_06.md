# VL / audio / video runtime worklist

Date: 2026-06-06

Active worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`

Primary tracker: `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md`

Reporter credit for relevant release notes: GitHub `@Hornsan1`

## Current release boundary

Do not tag, notarize, or publish a new vMLX / mlxstudio release from this workstream until the live rows below are green. Static contracts, load-only proofs, and single-turn smoke rows are not sufficient.

Current package/proof state, refreshed 2026-06-06:

- Bundled Python was rebuilt from current source and `npm run verify-bundled` passed.
- Staged Sequoia app was rebuilt and Developer ID signed.
- `build/current-packaged-integrity-contract-after-unsupported-media-staged-app-20260606.json` is `status=pass`.
- `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json` closes the two Gemma4 26B CRACK rows for installed-app Responses visible-content/language and mixed-SWA speed.
- `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json` is `status=open` with `failed_steps=["release_regression_manifest"]`; source/package/parity/contracts pass, and five live/model rows remain.
- No release tag, notarized DMG, public download update, or installed-app replacement has been produced from this continuation.

The app can currently serve several text and some VLM paths, but full VL/audio/video support is not release-cleared across all model families. The remaining work must be classified as either:

- `runtime`: engine, processor, cache, scheduler, parser, UI/settings, or packaged-runtime bug.
- `model artifact`: uploaded metadata/sidecars/templates/weights are corrupt or inconsistent.
- `model behavior`: model emits wrong/malformed content even though runtime route is correct.

Do not hide a failure by forcing text-only, disabling cache, disabling MTP, changing parser rails, or fabricating tool calls. If a capability is not implemented, expose it honestly and keep the implementation row open.

## Required media runtime functions

Every advertised media-capable family must have these functions working through the live server path:

1. Artifact capability read

- Reads `config.json`, `generation_config.json`, `preprocessor_config.json`, tokenizer files, chat template, `jang_config.json`, audio tokenizer sidecars, MTP sidecars, and quant metadata.
- Separates advertised capability from runtime-proven capability.
- Rejects or marks unsupported metadata/runtime mismatches before generation.

2. Media request normalization

- Accepts OpenAI image/audio/video content parts.
- Accepts data URLs, local files through app tool paths, and remote URLs only where policy allows.
- Converts media to the family processor's expected tensors without losing prompt text or special media tokens.
- Keeps text-only requests on the text path.

3. Processor and embedding bridge

- Processor expands image/video/audio prompts into the exact family token layout.
- Model wrapper accepts `pixel_values`, video frames, audio features, `inputs_embeds`, media masks, rope/position metadata, and family-specific kwargs.
- Text model returns logits objects compatible with `mlx_lm` / vMLX generation.

4. Cache integration

- Prefix cache keys are salted by media digest and media token layout.
- Paged cache, L2 disk cache, native family cache, and TurboQuant KV do not reuse stale text/media states across different media.
- Text-only turn after rejected or successful media recovers cleanly.
- Path-dependent families reject unsafe partial restores instead of corrupting decode.

5. Scheduler / batching

- Continuous batching works where safe.
- Families that require one-shot media prefill either chunk safely or reject as typed 413 without killing the process.
- Max output tokens and max context limits are honored for media-expanded prompts.
- Streaming and non-streaming produce equivalent visible content.

6. Tool / JSON / XML discipline

- Tool-capable models emit real OpenAI `tool_calls` through the parser for API tool use.
- Raw XML-ish tool text must not be silently accepted as a parsed tool call.
- Structured-output benchmarks must use parse, repair, schema validation, and retry-on-invalid rows, while still reporting raw parse failure.
- JSON/XML repair belongs in caller-side benchmark/catalog tooling unless vMLX exposes a proven constrained decoding mode.

7. UI and packaged app parity

- Settings panel must expose only proven cache and media controls for the selected family.
- CLI preview, spawned server args, `/health`, `/v1/cache/stats`, and `/v1/models/{id}/capabilities` must agree.
- Installed `/Applications/vMLX.app` proof is required separately from source-server proof.

## Model-family rows

| Family / bundle | Current state | Open blockers | Required proof |
|---|---|---|---|
| Gemma4 12B JANG_4M | Source and installed text/image/video smoke and speed rows have passed; image prefill rejection recovers cleanly. | Full audio/video depth, larger media prompts, multi-turn media+tools, UI settings parity across all cache modes. | Live source and installed rows: image, video, audio, text-after-media, media-after-error, tools after media, streaming and non-streaming. |
| Gemma4 26B CRACK | Source-server smoke passed on 2026-06-06 for exact text, cache hit, multi-turn recall, reasoning, tool call, image/video color, and text-after-media. Installed-app Responses visible-content and mixed-SWA speed rows now pass in the refreshed objective digest. | Broader packaged media/tool parity remains open. | Installed full-output tails, installed speed gate, and packaged media/tool parity rows. |
| Qwen3.6 35B MXFP8 MTP | Current source and local installed 1.5.56 do not reproduce `gdn_sink` crash; 35B live MTP proof exists. | If user sees traceback on older app, classify stale packaged runtime; if on fresh app, find bypassing route. Need broader VL/video/tool rows. | Installed packaged Qwen35 MTP with chat/responses/streaming, image/video if advertised, MTP active, no `gdn_sink` crash. |
| Qwen3.6 27B JANG_4M MTP | Native MTP speed/equivalence rows pass under deterministic policy. | Broader media/tool/multi-turn matrix and packaged UI parity remain open. | Live source and installed app across cache modes, MTP on/off A/B, image/video if advertised. |
| Qwen3.6 27B MXFP8 TP4 | The stale Max2 `:8124` gateway is still false-ready for a dead `hiddenfix2` run. A fresh Pod1 resident run `qwen36-mxfp8-pod1-tp4-codexlive2-20260606T160731Z` was launched and served through Max2 `127.0.0.1:8125` with native MTP depth `2`, cache coordinator, and L2 disk enabled. Local summary: `build/current-qwen36-27b-tp4-codexlive2-strict-proof-20260606.json`; full remote proof: `/tmp/adlab-qwen36-27b-mxfp8-tp4-codexlive2-qwen36-mxfp8-pod1-tp4-codexlive2-20260606T160731Z-api-proof.json`. Follow-up inspection shows chat, multi-turn, native-MTP probe, Responses, chain, and cache rows all rank-agree; only `/adlab/tp4/batch` diverges across ranks. Remote ADLab proof reporting now lists `bad_rows`, and the direct file-serve patch template now disables the concurrent `BatchEngine` path when `group.isMultiRank` so future rebuilt workers use the safer sequential rank-zero sampler path. | Fresh route is live but strict proof remains `FAIL`: current live workers have not been rebuilt/relaunched with the safer template; existing batch row returns 502 from non-authority rank token divergence, aggregate rank0 throughput evidence reaches about `42.5 tok/s` but proof-visible `tok_per_s` is `6.582`, and hybrid SSM proof has SSM hits but no attention evidence. | Rebuild/relaunch TP4 rank workers from the updated ADLab patch, rerun strict proof, then build a genuinely collective-safe multi-rank batch hot path before claiming `40+ tok/s` batch release. Keep `:8124` stale-ready bug open separately; use `:8125` only as a diagnostic live route until strict proof passes. |
| LFM2.5 | Text/cache/tool UI rows have passed for known bundles. | VL/audio/video only if artifact advertises it; hybrid SSM path-dependent cache rows must stay architecture-aware. | Live media row for any VL-advertised LFM artifact, otherwise capability must remain text-only. |
| Step3.7 Flash JANG_2L | Text-only route is stable; VLM route remains unsupported unless implemented. | Do not advertise Step3.7 VLM in vMLX until real processor/model forward path passes. Tool dialect and loop discipline remain model-behavior blockers. | Text-only live rows plus explicit unsupported-VLM guard row, or real Step3.7 VLM implementation with image/video proof. |
| MiMo V2.5 JANG_2L | Fresh Max2 bundle copied locally and manifest verified; cache head crash fixed; optimized rerun shows paged/L2/TurboQuant telemetry. Conservative no-cache/no-TQ diagnostic can answer exact `ACK`. XML-function prompt/parser fallback placement is fixed in source. Current source also preserves `tool_choice`/tool metadata into MLLM decode and can force the required XML structural prefix. | Release-red: live required-tool still fails after the valid prefix because generated argument/closing XML is punctuation garbage; exact plain XML copying also fails; speed is around 0.3-1.4 tok/s in latest narrow probes; long-prompt coherence remains red; no release-cleared VL/audio/video path. | Fix decode quality and speed target, compare source-vs-quant first divergence, then implement processor/model bridge for image/video/audio and prove media rows. |
| Nemotron Omni | Prior rows exist but not current release-cleared. | Audio/image/video carryover and no-media prompt contamination need current-source and installed reproof. | Omni live matrix with text-only before/after media, audio/image/video, no hidden media carryover. |
| ZAYA / ZAYA1-VL | Typed CCA/path-dependent cache constraints are special. | VL rows must prove media cache salting and no unsafe partial CCA restore. | Live ZAYA-VL image/video rows with CCA cache stats and post-media recovery. |
| DSV4 Flash | Native composite cache is separate from generic TurboQuant KV; app-launch default-cache row cleared. | Long-output/code/file-generation quality and exactness remain open; media only if artifact advertises sidecars. | DSV4 long-output/code/file generation, restart-L2, exact-copy, tool loops, no generic TQ-KV substitution. |
| MiniMax M2.7 JANGTQ | K CRACK source-server smoke passed on 2026-06-06 with exact text, `paged+tq` cache hit, generic TurboQuant KV, L2 telemetry, multi-turn recall, reasoning, and required OpenAI tool call. Small JANGTQ rows also have prior proof. | Reporter/runtime mismatch and MiniMax K issue #179 parity/root cause remain open; installed-app parity still needed for the reporter class. | Installed app live rows for K/Small, reporter parity capture, parser/cache evidence, and any advertised media audit. |
| Hy3 / other JANGTQ | Not fully current-cleared in this slice. | Family-specific JANGTQ kernels, cache policy, media sidecars. | Live family smoke with advertised modalities and cache stats. |

## Known missing or incomplete runtime functions

This section is the implementation ledger. A row is not complete unless the
named function exists, routes through the live server path, and has source plus
installed-app proof where the product advertises the capability.

| Function area | Current implementation | Missing build work | Current blocker classification |
|---|---|---|---|
| Gemma4 dynamic image-prefill budget | `_resolve_vlm_image_prefill_single_buffer_limit()` scales the default with Metal working set, keeps explicit `VMLINUX_VLM_IMAGE_PREFILL_BUFFER_GB`, and typed 413 recovery exists. | Full installed-app media matrix: larger images, video, audio, tools after media, streaming/non-streaming parity. | `unknown_pending_repro` for remaining quality/media-depth rows, not the old fixed-8GB guard. |
| Gemma4 media post-error recovery | Panel removes a failed media user turn when no visible activity occurred, preventing replay of a rejected image on the next text turn. | Re-run through installed app for image reject, text-after-reject, video/audio-after-reject, and Responses/Chat parity. | `gateway_ui` until installed proof is current for every route. |
| Qwen VL/video processor bridge | Qwen video path has native processor wiring in source contracts: `videos=...`, `pixel_values_videos`, and `video_grid_thw` rows exist. | Current packaged Qwen27/35 MTP media matrix with MTP on/off, tools, streaming, and post-video text recovery. | `unknown_pending_repro` for broad media proof; stale `gdn_sink` crash is `gateway_ui` if only old app reproduces. |
| Qwen36 TP4 ADLab gateway response collection | The bounded probe SSHes through Max2, checks `/health` and `/v1/models`, posts a deterministic exact-output chat request, then snapshots every rank request/response directory through the gateway host. The probe now matches recent request IDs to expected per-rank response filenames and records exact-serve-dir `TPRankWorker` process rows via `ps eww` `TP_SERVE_DIR=` instead of trusting stale ready/response files. | Current artifact has stale ready files but zero exact-serve-dir workers for the configured `hiddenfix2` run; other TP workers exist on some ranks for different serve dirs. The gateway's readiness surface is false-positive for this route. Cache coordinator, L2, MTP, and OpenAI endpoint claims cannot be considered live-cleared for this TP4 route. | `gateway_ui` / remote runtime orchestration, not local download corruption. |
| Qwen36 TP4 multi-rank batch decode | Fresh `:8125` proof narrows rank disagreement to `/adlab/tp4/batch` only. Single-request rows all have `rank_agreement.ok=true`. Batch rank response files show 16 outputs per rank but divergent generated-token rows for ranks 1-3 while the engine advertises `token_authority=rank0_send_recv_per_slot` and `engine_path=batch_engine_no_native_mtp`. | The concurrent `BatchEngine` path is unsafe under multi-rank collectives. Remote ADLab template patch now changes that path to single-rank only (`promptBatches.count > 1 && !group.isMultiRank`) and proof output now reports `bad_rows`, but live rank workers were not rebuilt/relaunched from that patch yet. A safe high-throughput multi-rank batch engine is still unbuilt. | `distributed_runtime` / remote ADLab runtime; not model download corruption. |
| MiMo text runtime bridge | vMLX registers `mlx_vlm.models.mimo_v2` around `jang_tools.mimo_v2.mlx_model`; text/cache narrow proof exists. Missing media forward now raises typed `unsupported_media_modality` instead of a generic generation failure. | Real multimodal model module: vision encoder, audio encoder/tokenizer bridge, media projector/merger, `get_input_embeddings(pixel_values=...)`, `__call__(pixel_values=...)`, video/audio token expansion, and cache salt proof. | `runtime_dispatch` plus `model_artifact` pending source-vs-quant quality comparison. |
| Runtime media capability reflection | `/v1/models/{id}/capabilities` now keeps top-level `modalities` as runtime-proven support and adds `media.runtime_modalities`, `media.declared_modalities`, `media.preserved_modalities`, `media.unwired_modalities`, and `media.status_by_modality`. MiMo fixtures with preserved vision/image/video/audio sidecars report those modalities as `preserved_unwired` while staying top-level `["text"]`. The all-local smoke harness now prefers `media.runtime_modalities` and ignores preserved/unwired sidecars for probe scheduling. The panel registry demotes JANG text-only/unwired media stamps to text runtime, without promoting media-inclusive stamps past Qwen affine safety gates; settings video controls are allowlisted to runtime-video families. Focused tests pass. | Wire real UI smoke to assert the nested `media` contract and visible controls from the installed app. | `gateway_ui` for remaining real-UI proof; not a media-forward implementation. |
| MiMo JANG tools model forward | `/Users/eric/jang/jang-tools/jang_tools/mimo_v2/mlx_model.py` is explicitly text-only; its docstring says visual/audio towers are preserved but not wired. JANGQ/JANG tools commit `98f57a6` now makes generated MiMo metadata advertise `modalities=["text"]` and records `preserved_modalities` / `unwired_modalities` for vision/audio. | Implement `mimo_v2_multimodal.py` or equivalent in JANG tools, then make vMLX register that module instead of a text-only compatibility shell. Regenerate/reupload MiMo bundles so old metadata no longer advertises unwired media. | `runtime_dispatch`; current metadata fix is `model_artifact` prevention for future uploads, not a VL/audio/video pass. |
| MiMo long-prompt/tool quality | Cache classification and narrow text repeat work; conservative no-cache exact ACK can work. XML-function fallback placement now respects MiMo's native parser dialect, but current live `tool_choice=required` still emits no parsed OpenAI `tool_calls`. The MiMo current audit and release manifest now require a source-vs-quant first-divergence artifact before release clearance. Source exists on Max2 at `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`; producer script `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py` compares already-running source and quant endpoints and writes the required artifact. Fresh preflight artifact `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-prefix-preflight-20260607.json` is `status=missing_prerequisites`: both model paths exist, but source endpoint `erics-m5-max2.local:8126` and local quant endpoint `127.0.0.1:8897` are down. | Launch a dedicated source MiMo endpoint and local quant endpoint intentionally, then rerun the producer against real source and quant servers to prove whether long-prompt/exact-cache/speed failures are source-model behavior, quant artifact drift, or runtime decode/cache. Then prove speed target, broader loop behavior, and packaged/UI parity. | `decode_loop` or `model_artifact` still unresolved for quality/speed; parser dialect mismatch fixed in source. |
| MiMo required XML decode metadata | Current source forwards `tool_choice` and template tools through Chat/Responses, BatchedEngine, and MLLM scheduler request metadata. A MiMo-only required-tool prefix sampler is active and live-proven to emit `<tool_call>\n<function=record_fact>\n<parameter=value>` for the requested `record_fact` tool. XML-function fallback now prefers the ChatML system turn and real-tokenizer render proof shows the concrete `record_fact` / `blue-cat` example in system scope, not user scope. Full-forward logits prove the model ranks `blue`, then `-cat`, then closing XML correctly when the full sequence is provided. vMLX now patches stale MiMo SWA cache construction to use `RotatingKVCache(..., keep=0)`, matching full-forward sliding-window semantics. | Narrow live required-tool row now passes with `record_fact({"value":"blue-cat"})`, but speed remains about `0.6 tok/s`. Broader tool-result continuation, multi-turn tool loop stop, long prompt quality, L2/restart cache, UI, and media rows remain open. | Run full MiMo no-media tool/cache/L2/long-prompt smoke with keep=0, then fix speed/kernel path and implement/prove media bridge or keep capabilities text-only. |
| Step3.7 VLM bridge | Source registration/no-heavy guards exist; text-only route is stable. | Real Step3.7 image/video processor, projector, forward, media cache salt, and live proof, or an explicit unsupported-VLM product row. | `runtime_dispatch`; text-only metadata view is not a VLM fix. |
| Nemotron Omni audio/video | Prior source rows exist for text/image/video/audio but are not current release-cleared. | Current source and installed app matrix for text-before/after media, audio/image/video carryover, tools, streaming, and L2. | `unknown_pending_repro`. |
| ZAYA/ZAYA1-VL media cache | Typed CCA cache contract exists; media routing rows exist in no-heavy contracts. | Live image/video rows proving media salt plus safe CCA/path-dependent restore and post-media recovery. | `kernel_cache` if partial restore fails; otherwise `unknown_pending_repro` until live. |
| MiniMax / Hy3 / Kimi media advertised variants | Text/tool/cache rows exist for selected bundles; media proof is not current in this slice. | Per-family advertised-modality audit: processor path, special tokens, cache policy, tools after media, installed-app parity. | `unknown_pending_repro`. |
| Shared structured-output repair | Source utility now exposes post-generation JSON repair diagnostics with `raw_json_ok`, `raw_schema_ok`, `repair_needed`, `repair_actions`, parsed object, and schema validity while preserving the legacy tuple API. Repo-native adapter `bench/structured_output_repair_report.py` applies it to benchmark/catalog JSONL and writes repaired JSONL plus summary counts. | Wire any external video/catalog runner to call the adapter directly, persist raw-vs-repaired fields in benchmark result artifacts, and add retry-on-invalid rows. XML repair remains conservative extraction only. | `decode_loop` for native malformed output; caller-side repair is not guided decoding. |
| Release pointer/package parity | Source package pointers now track current 2026-06-06 artifacts and packaged integrity passes on the rebuilt staged app. | Full release still needs the seven live/model blockers closed, then notarized/stapled DMGs, install proof, public update manifest proof, and mlx.studio/vmlx.net download freshness proof. | `gateway_ui` until public release artifacts are current and verified. |

1. MiMo multimodal bridge

- `jang_tools.mimo_v2.mlx_model` is text-oriented; multimodal runtime is not release-proven.
- vMLX MiMo MLLM path must accept and correctly route image/video/audio tensors, not just load an MLLM wrapper.
- Tool parser must handle MiMo's template dialect without fabricating tool calls from malformed raw XML. Source now distinguishes MiMo `xml_function` from Qwen/Step template fallbacks, but current live proof does not parse `record_fact`; raw XML/punctuation remains the blocker.
- Current fail-closed behavior is typed: unwired MiMo vision forward raises `UnsupportedMediaModalityError` / API code `unsupported_media_modality`. This is not a pass for MiMo vision; it is only correct failure classification and recovery.

2. Step3.7 VLM bridge

- Current safe route is text-only.
- Real Step3.7 VLM requires processor, media token expansion, model forward, cache salting, and full live proof.

3. Omni audio/video normalization

- Need current-source proof that audio/video content parts do not contaminate later text-only turns.
- Need installed app proof, not only source-server proof.

4. JSON/XML structured-output repair

- Bench/catalog tooling must consume the shared JSON repair report and store raw-vs-repaired parse status. A repo-native JSONL adapter exists; external runners still need to call it.
- vMLX should not claim hard constrained JSON/schema decoding unless a runtime option is implemented and live-proven.

5. Packaged parity

- Source fixes must be rebuilt, signed, notarized, stapled, installed, and tested from `/Applications/vMLX.app` before public release claims.
- Current-source proof alone does not update mlx.studio or vmlx.net downloads.

## Required test matrix per media-capable model

For each row below, capture full output tails and server logs:

1. Load and capability reflection

- `/health`, `/v1/models`, `/v1/models/{id}/capabilities`, `/v1/cache/stats`, panel settings, CLI preview.

2. Text baseline

- Chat completions, Responses, Anthropic messages, Ollama chat/generate where supported.
- Stream and non-stream.
- Thinking off/on if supported.

3. Image

- Small image.
- Large but valid image.
- Oversized or deliberately rejected image returns typed 413 and later text works.

4. Video

- Short local fixture.
- Multi-frame prompt.
- Text-after-video and video-after-text cache recovery.

5. Audio

- Short speech fixture.
- Non-speech/noise fixture.
- Audio-after-image and text-after-audio recovery.

6. Cache modes

- No prefix cache.
- Prefix cache.
- Paged cache.
- Block disk L2 cache.
- Family-native cache.
- TurboQuant KV only where architecture permits it.
- Native MTP on/off where artifact supports it.

7. Tools and structured output

- Forced single tool call.
- Tool result continuation.
- Multi-turn tool chain.
- JSON object, JSON schema/repair benchmark, XML/tool dialect rejection.

8. UI parity

- Create session, edit settings panel, max output tokens, max prompt tokens, cache toggles, modality upload controls.
- Run same proof through installed app gateway.

## Immediate next build order

1. Build MiMo media bridge in JANG tools first (`mimo_v2_multimodal.py` or equivalent), then update vMLX registration to use the real media model instead of the text-only compatibility shell.
2. Prove or falsify MiMo model-artifact quality from the actual Max2 bundle: source-vs-quant long-prompt trace, routed expert parity, tool-argument exactness, and speed target.
3. Wire shared structured-output repair/validation into any external video/catalog benchmark layer and report raw-vs-repaired score separately.
4. Re-run live media matrix for Gemma4 12B/26B, Qwen27/35 MTP, LFM, Step3.7 text-only, MiMo, Nemotron Omni, ZAYA-VL, DSV4, MiniMax, Hy3.
5. Close cross-family multi-turn and real Electron UI live model matrix rows.
6. Only after live rows pass: rebuild app, sign, notarize, staple, install, and repeat installed-app proofs before release/tag/public download update.

## 2026-06-06 cross-family media/cache refresh

Current proof artifacts now wired into the release gates:

- `build/current-all-local-model-smoke-live-slice-tools-media-continuation-20260606/summary.json` covers Gemma4 12B, LFM2.5, MiniMax Small, Qwen27, and Step3.7 text route.
- `build/current-all-local-model-smoke-ling-hy3-nemotron-tools-media-20260606/summary.json` covers Hy3, Ling/Bailing, and Nemotron Omni.
- `build/current-all-local-model-smoke-qwen35-mxfp8-mtp-tools-media-20260606/summary.json` covers Qwen35 MXFP8 MTP including the no-`gdn_sink` source path.
- `build/current-all-local-model-smoke-zaya-text-vl-tools-media-20260606/summary.json` is intentionally red and remains a required blocker.
- `build/current-dsv4-route-mode-code-exactness-memory-preflight-cross-family-20260606.json` keeps DSV4 live exactness closed to unsafe launch until enough RAM is available.

Current blockers to build, not guard away:

- MiMo V2.5 JANG_2L needs real multimodal runtime implementation in JANG tools/vMLX before VL/audio/video can be advertised. The current text compatibility shell and typed unsupported-media error are not a VL pass.
- JANGQ/JANG tools main `98f57a6` fixed future MiMo generated metadata to keep runtime modalities text-only while preserving vision/audio sidecars as unwired. Existing uploaded/local MiMo bundles with older `modalities=["text","vision","audio"]` metadata need regeneration or metadata patching before they can be considered honest artifacts.
- Local `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L/config.json` was patched to the same honest metadata. Proof artifact: `build/current-mimo-v25-jang2l-local-metadata-truth-patch-20260606.json`. JANG verifier passes structurally on the 150-shard / 109180-tensor bundle, including preserved ViT/audio tensors and matching chat template.
- Smoke runner media scheduling now carries model `capabilities` into inventory and uses them as fallback if `/v1/models/{id}/capabilities` omits modalities. Dry-run artifact `build/current-all-local-model-smoke-mimo-v25-jang2l-after-metadata-truth-dryrun2-20260606/inventory.json` proves MiMo still looks MLLM-shaped from `vision_config`, but carries `modalities=["text"]`, so media probes must not be scheduled from metadata alone.
- MiMo text quality still needs source-vs-quant first-divergence tracing and speed proof. The release gate now requires the local source-vs-quant first-divergence artifact before MiMo can clear, and the producer script exists under `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py`. The source-side XML-function parser/template placement is fixed, but current required-tool generation still fails before parsed `record_fact`; packaged/UI parity remains open. Do not force hidden prompts, fake tool-call conversion, or cache disablement as a production fix.
- ZAYA/ZAYA1-VL needs exact-instruction, reasoning-visible-output, multi-turn recall, and red-image semantic fixes while preserving the CCA/path-dependent cache constraints.
- DSV4 now has current user-RAM-override live exactness proof. It is not memory-blocked for this narrow local slice, but it is still release-red: `thinking_closed` direct/off rails corrupt exact code identifiers while `thinking_open` rails produce exact output.
- Real Electron UI still needs the same family matrix from installed app settings through spawned server args, `/health`, cache stats, media upload, max output tokens, streaming/non-streaming, and post-error recovery.

Release sequencing remains:

1. Close MiMo, ZAYA, DSV4, MiniMax reporter parity, and real Electron UI live rows with full-output artifacts.
2. Rebuild bundled Python/app from that source.
3. Sign, notarize, staple, install, and rerun installed-app proofs.
4. Publish release/tag/downloads only after public vmlx.net and mlx.studio download freshness is proven current.

## 2026-06-06 ZAYA media/cache blocker split

Updated ZAYA evidence:

- `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json` is now the active ZAYA smoke artifact.
- ZAYA text proves typed CCA prefix/paged/L2 cache behavior and tool protocol, but reasoning-on is still unusable because output remains in reasoning with no visible final answer.
- ZAYA-VL proves typed CCA cache telemetry and blue-image handling, but text exactness, multi-turn recall, reasoning visible output, and red-image recognition are still red.

Required next ZAYA work:

1. Trace ZAYA reasoning-on prompt/template/parser behavior from rendered prompt through raw generated text and decide whether the model truly supports visible thinking mode. If it cannot close the qwen3 rail reliably, metadata/capabilities must be corrected rather than forcing hidden defaults.
2. Trace ZAYA-VL text-only prompt rendering separately from image rendering; current text cache exact prompt produces the wrong visible token while image blue works.
3. Debug red image path for ZAYA-VL: blue image and repeat pass, red image returns `white`, so this is media semantic/model-forward behavior, not attachment carryover.

## 2026-06-06 ZAYA text closure and remaining ZAYA-VL work

ZAYA text is no longer a cross-family smoke blocker after the reasoning budget refresh:

- `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json` has `ZAYA1-8B-MXFP4` status `pass`.
- Required cache, recall, reasoning, and tool surfaces are green for ZAYA text.
- Native cache remains architecture-aware: `zaya_cca_v1`, generic TurboQuant KV disabled for path-dependent CCA state, prefix/paged/block-disk L2 enabled.

ZAYA-VL remains open:

1. Text-only exact cache prompt in the VL wrapper returns the wrong token.
2. Multi-turn text recall is broken in the VL wrapper.

## 2026-06-06 DSV4 route-mode code exactness split

Current live artifacts:

- `build/current-dsv4-route-mode-code-exactness-chat-off-user-ram-override-20260606.json`
- `build/current-dsv4-route-mode-code-exactness-chat-on-user-ram-override-20260606.json`
- `build/current-dsv4-route-mode-code-exactness-ab-route-user-ram-override-20260606.json`

What is proven:

- Chat Completions with `enable_thinking=false` returns HTTP 200 and stops normally, but corrupts exact code. The first row produced `THREE.WebWebGLRenderer()` instead of `THREE.WebGLRenderer()`.
- Chat Completions with `enable_thinking=true` returns exact required code under the same model and route family.
- Responses API with `enable_thinking=false` reproduces the same `THREE.WebWebGLRenderer()` corruption, so this is not isolated to the Chat endpoint.
- Removing punctuation and using repetition-penalty mitigation did not solve the off-rail issue. `chat_off_no_punct_rep1` produced `THREE.ScScene()`.
- Responses API with `enable_thinking=true` returns exact required code.

Current classification:

- `decode_loop` / `model behavior`: the failure tracks the DSV4 assistant suffix rail, `thinking_closed` (`<|Assistant|></think>`) versus `thinking_open` (`<|Assistant|><think>`), and affects exact identifier fidelity.
- `model_artifact`: not proven yet. The same uploaded artifact can produce exact output on the thinking-open rail, so corrupt weights are not the first explanation; a source-vs-quant or higher-precision comparison is still needed.
- `runtime_dispatch`: not the current leading root cause for the suffix itself. The local model `encoding/README.md`, `encoding_dsv4.py`, and `chat_template.jinja` all declare that chat/non-think mode appends `</think>` after `<|Assistant|>`.

Required next DSV4 checks:

1. Run source or higher-precision reference generation, if available, to separate JANGTQ/model behavior from quantization artifact behavior on the official chat rail.
2. Run direct raw prompt/model-native reference generation to separate runtime prompt construction from model behavior, but keep the official chat suffix unchanged unless the artifact metadata changes.
3. Run exact-code rows across Chat, Responses, legacy completions, streaming, restart-L2, and UI-spawned installed app.
4. Do not ship a hidden force-thinking-on fix. If off-rail exact code is weak by model design, document that as a model behavior/artifact limitation; if the suffix is wrong, fix the encoder/runtime contract.

## 2026-06-06 MiMo prompt-shape and speed split

Fresh local evidence:

- `build/current-mimo-v2-jang2l-simple-conservative-cacheprompt-probe-20260606.json` failed on the long cache-style exact `ACK` prompt even without `--is-mllm`, continuous batching, prefix cache, KV quantization, or native MTP. The response was HTTP 200 with empty visible content after about `24.27s`.
- `build/current-mimo-v2-jang2l-mllm-conservative-probe-20260606.json` also failed with empty visible content. A source patch normalized MiMo text-only MLLM rich content parts to plain processor strings, but the live failure persisted, so rich-list prompt content was not the root cause.
- `build/current-mimo-v2-jang2l-prompt-shape-sweep-20260606.json` separates prompt shape from cache/runtime flags:
  - `short_user_only`: `ACK`, but slow first decode, `2` completion tokens in `23.003s`.
  - `short_system_exact`: `ACK`, `2` completion tokens in `1.792s`.
  - `long_cache_system_exact`: empty visible content, `completion_tokens=1`, `finish_reason=stop`, `prompt_tokens=60`.
  - `long_cache_no_system_exact`: `ACK`, `2` completion tokens in `1.748s`.
  - `normal_chat_short`: coherent one-sentence answer, `19` completion tokens in `11.526s`, about `1.65 tok/s`.
  - `speed_120_words`: timed out at `90s`.
- `build/current-mimo-v2-jang2l-rendered-prompt-compare-20260606.json` shows the failing separate-system prompt renders as valid ChatML and ends with the same generation prefix as the working prompts: `<|im_start|>assistant\n<think></think>`. Token counts: failing `long_cache_system_exact=60`, working folded-user `long_cache_no_system_exact=66`, working `short_system_exact=23`.
- `build/current-mimo-v2-jang2l-first-token-probe-registered-20260606.json` registers `jang_tools.mimo_v2.mlx_register` and probes direct first-token logits. The failing separate-system prompt ranks `<|im_end|>` first (`token_id=151645`, logprob `-0.8347`) and `ACK` fourth. The folded-user and short-system prompts rank `ACK` first with high confidence.
- `build/current-mimo-v2-jang2l-current-audit-after-source-preflight-refresh-20260606.json` is the active MiMo audit. It confirms stale MiMo local state is absent after deleting the stale Hugging Face `transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L` cache, while keeping MiMo release open and adding the missing source-vs-quant first-divergence proof as a named blocker.

Current interpretation:

- The immediate-empty row is triggered by the combination of the model's own first-system-message semantics plus longer exact cache-style user content. The no-system equivalent works, rendered prompts are valid, and direct logits show the model selects `<|im_end|>` as the first token, so this is not just prompt length, malformed ChatML rendering, or cache corruption.
- This cannot be closed by folding system prompts into user prompts as a hidden production behavior. If MiMo system-role handling is model-artifact-sensitive, the artifact metadata/template must be corrected or documented. If the runtime is misapplying the template/defaults, the runtime path must be fixed.
- Max2 docs confirm the promoted JANG_2L bundle was historically coherent but slow: canonical cached generation was about `1.97 tok/s` on France and `2.64 tok/s` on arithmetic. That means the current Python JANG_2L artifact is not a `40+ tok/s` release candidate. A `40 tok/s` target requires a different quant/kernel/runtime path, not a release note.

Required next MiMo build checks:

1. Render and compare the exact MiMo prompt strings/tokens for separate-system versus folded-user cases, including `enable_thinking=false`, `add_generation_prompt`, and stop-token IDs.
2. Run first-divergence next-token checks against the source or highest-quality local profile for the failing `long_cache_system_exact` prompt.
3. Verify whether the generated empty row is immediate `<|im_end|>`, `<|endoftext|>`, or another stop token and classify it as `decode_loop`, `model_artifact`, or `template/defaults`.
4. Re-test the MiMo wrapper `inputs_embeds` forwarding separately from the prompt-shape failure; do not count the rich-list normalization patch as a release fix unless live behavior improves.
5. Decide speed path explicitly: current JANG_2L Python affine path is around `1-2 tok/s`; `40+ tok/s` needs a faster quant profile, kernel path, speculative/MTP path, or a different bundle with full structural and quality proof.
3. Reasoning visible final answer is still empty.
4. Red image is misclassified as white while blue image passes.
5. Cache telemetry is present, so current evidence points at VL text/template/model-forward semantics rather than a CCA cache crash.

## 2026-06-06 ZAYA-VL no-media contract refresh

Fresh ZAYA-VL evidence:

- All-quant comparison: `build/current-all-local-model-smoke-zaya-vl-all-quants-tools-media-20260606/summary.json`.
- Focused prompt repro: `build/current-zaya-vl-mxfp4-focused-repro-20260606/summary.json`.
- Current smoke proof: `build/current-all-local-model-smoke-zaya-vl-mxfp4-after-no-media-contract-20260606/summary.json`.

Findings:

1. ZAYA1-VL-8B-MXFP4 is the healthiest local VL bundle: blue image, red image, text cache exact, recall, required tool, and no-media-after-image now pass under the family-native `ATTACHED/NONE` no-media contract.
2. The earlier no-media failure was prompt-contract incompatibility, not proven media carryover. A direct current-message attachment question returns `NONE` before and after image turns.
3. ZAYA-VL MXFP4 still fails reasoning-on: visible content is empty while the answer appears in the reasoning channel (`reasoning_chars=17`) even with larger budgets in focused repro.
4. ZAYA-VL JANGTQ4 and JANGTQ_K remain weaker than MXFP4 in smoke: JANGTQ4 adds exact-text/reasoning/no-media failures; JANGTQ_K adds exact-text, recall, reasoning, and red-image failures.
5. Cache telemetry is not the blocker for the MXFP4 no-media path: native typed CCA cache reports `zaya_cca_v1`, prefix/paged/L2 enabled, and generic TurboQuant KV disabled for path-dependent CCA state.

Next ZAYA-VL blocker is reasoning-visible compatibility, not the no-media guard. Do not fake-clear it by disabling thinking silently; trace rendered prompt, parser split, and model capability metadata first.

## 2026-06-06 MiMo sink/cache falsification refresh

Fresh MiMo diagnostics:

- Native-vs-manual sink diagnostic: `build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json`.
- Disable-sink diagnostic: `build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json`.
- Refreshed current audit: `build/current-mimo-v2-jang2l-current-audit-20260606.json`.

Findings:

1. MiMo local artifact integrity is clean: TB5 manifest matches 173 files / 113,926,313,468 bytes, stale local backups/caches are absent, structural verify passes, text-cache narrow proof passes, selected-expert SwitchGLU parity passes, and cache-prefill vs no-cache next-token top-10 logits match at 125 prompt tokens.
2. Manual sink SDPA does not clear length generation. Native sink and manual sink both fail the focused length prompts; this is not proven to be an MLX native sink-kernel-only issue.
3. Disabling SWA sink is worse: tested lengths emit repeated punctuation. Do not ship a sink-disable workaround.
4. MiMo remains release-red for local runtime quality: long-prompt coherence, exact-cache prompt-following, and speed fail. The earlier OpenAI tool protocol failure is still active in current live smoke: required `record_fact` produces no parsed tool call and raw XML/punctuation output; packaged/UI parity remains open. The remaining evidence points beyond cache store/reconstruct and beyond sink toggles toward model/runtime weight mapping, template/defaults, or quantized forward quality.
5. MiMo media remains unimplemented in Python: the bundle has vision/audio metadata and weights, but vMLX still registers the text-only compatibility shell and raises typed unsupported-media errors for pixel input. This is correct fail-closed behavior, not VL/audio support.

Next MiMo work:

1. Run source-vs-quant first-divergence against a known-good source/higher-quality MiMo profile, including rendered prompt, first bad token, router top-k, attention logits, and final-norm/lm-head logits.
2. Compare vMLX server generation against direct `jang_tools.mimo_v2` generation with identical prompt tokens and sampling to separate API/template/parser problems from model forward problems.
3. Do not fake-clear MiMo by disabling sinks, disabling tools, converting incomplete `<tool_call>` into a synthetic call, or suppressing MiMo media advertisement without implementing the actual media bridge.

## 2026-06-06 DSV4 cross-family smoke refresh

Fresh DSV4 smoke artifact:

- `build/current-all-local-model-smoke-dsv4-jangtq-k-tools-cache-20260606/summary.json`.

Findings:

1. Both matching local DSV4 rows passed the focused all-local smoke: `DeepSeek-V4-Flash-JANGTQ-K` and `DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520`.
2. Primary JANGTQ-K row proved exact `ACK` cache repeat, multi-turn recall `blue cat`, visible reasoning answer `FINAL=OK`, and required `record_fact` tool call with `{"value":"blue-cat"}`.
3. Cache repeat proved native DSV4 cache reuse: first prompt `prompt_tokens=3640`, second repeat reported `cached_tokens=3639` and `cache_detail=paged+dsv4`.
4. This clears the missing DSV4 cross-family smoke artifact only. It does not clear DSV4 long-output/code/file-generation exactness, which still requires the larger explicit code-quality suite and memory-safe host conditions.

Release boundary remains: do not use this smoke to claim DSV4 exact-code or full release readiness.

## 2026-06-06 ZAYA1-VL text-only processor normalization diagnostic

Artifacts:

- `build/current-all-local-model-smoke-zaya-vl-after-text-template-normalize-20260606/summary.json` -> `status=fail`, 5 failures.
- `build/current-all-local-model-smoke-zaya-vl-after-batched-text-template-normalize-20260606/summary.json` -> `status=fail`, 5 failures.

Narrow fix landed in source:

- Direct `MLXMultimodalLM` and batched MLLM renderer now collapse ZAYA1-VL text-only rich content lists to plain string content before processor template rendering.
- Media turns remain rich content lists; image/video processor template calls are not collapsed.
- Focused no-heavy tests pass for direct MLLM and batched ZAYA1-VL text/media template boundaries.

Observed live effect:

- `text_no_media_after_image` improved from prompt echo to a real no-image response: `I see no image file attached to this message...`.
- Tool call remains structurally correct with `record_fact({"value":"blue-cat"})`.
- ZAYA typed CCA cache hit remains active on repeat: `cached_tokens=33`, `cache_detail=paged+zaya_cca`, native cache family `zaya_cca_v1`, generic TurboQuant KV disabled.
- Media prefix store remains skipped for image prompts, as required for path-dependent media embeddings.

Still release-red:

- `text_cache_repeat_1/2` answer `green` instead of expected `blue`.
- `text_multiturn_recall` answer `color ` and misses `blue cat`.
- `reasoning_on` returns reasoning text only (`The answer is "OK."`) with empty visible content.
- `vl_red_image_changed` returns `white` for the red fixture while blue image rows pass.

Classification:

- Current ZAYA1-VL release blocker remains open.
- The fixed portion is `runtime_dispatch` / `gateway_ui` for text-only processor content shape.
- Remaining failures are `decode_loop` / `model_artifact` / deeper `runtime_dispatch` pending prompt/render trace and source-vs-artifact comparison.
- Do not mark ZAYA-VL cross-family smoke as passed from this patch.

## 2026-06-06 MiMo V2.5 JANG_2L harness/tool-cache proof

Artifact: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-harness-tighten-20260606/summary.json`

Harness fix:

- MiMo V2 now stays tool-capable even when the promoted bundle lacks `jang_config.json`; registry fallback maps it to XML tools and think XML.
- Nested MiMo `audio_tokenizer` sidecars are filtered out of the default model inventory so smoke rows do not treat processor sidecars as standalone models.
- This closes a harness blind spot only. It does not clear MiMo quality or media support.

Live proof result:

- Status remains `fail`.
- Native cache health was active: family `mimo_v2`, schema `mixed_swa_kv_v1`, subtype `mimo_v2_asymmetric_swa`, prefix cache, paged cache, block disk L2, and TurboQuant storage-boundary q4 telemetry were present.
- Multi-turn recall passed with visible `blue cat`.
- Reasoning-on passed with visible `FINAL=OK`.
- Exact cache repeat failed: first repeat was empty visible output, second repeat rambled instead of exact `ACK`, even though cached tokens were reported.
- Required tool call failed in this artifact: server generated but returned no parsed `record_fact` tool call under `tool_choice=required`. This was narrowed later to a runtime prompt/parser dialect bug and fixed in the XML-function follow-up below.

Current classification:

- `kernel_cache`: not currently proven as the primary MiMo blocker because cache telemetry is live, but exact cached decode output is still wrong and must stay under proof.
- `decode_loop` or `model_artifact`: still unresolved for exact prompt-following and speed.
- `runtime_dispatch`: still open for VL/audio/video because JANG tools MiMo forward is text-only and vMLX has no release-proven MiMo image/video/audio bridge.

Do not resolve this with fake parser injection, forced text-only metadata, or synthetic tool calls. The next proof must compare source/high-quality MiMo against the JANG_2L artifact and trace where the decoded tokens diverge.

## 2026-06-06 MiMo XML-function parser/template fix

Artifact: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-metadata-truth-20260606/summary.json`

Source/runtime change:

- vMLX no longer misclassifies explicit MiMo `xml_function` parser prompts as Qwen/Step-style native tool prompts just because they contain `<|im_start|>`, `<tools>`, and `<function=example_function_name>`.
- Fallback for MiMo XML tools now emits native XML-function instructions compatible with `xml_function` / `mimo_xml_function`, not JSON inside `<tool_call>`.
- Focused parser/fallback tests passed: `tests/test_tool_fallback_injection.py`, `tests/test_xml_function_tool_parser.py`, and selected XML/fallback/tool-format rows, `74 passed`.

Live proof result:

- `tool_required` is red in the current refresh: no parsed OpenAI function call; raw preview starts `<tool_call>` followed by punctuation/fullwidth comma garbage.
- Multi-turn recall and reasoning visible-output probes pass.
- Cache infrastructure remains active: `mixed_swa_kv_v1`, `mimo_v2_asymmetric_swa`, prefix cache, paged cache, block disk L2, and TurboQuant q4 storage-boundary telemetry.
- Smoke still fails because `text_cache_repeat_1` is empty visible output and `text_cache_repeat_2` rambles instead of exact `ACK`.
- Decode speed remains about `1.78 tok/s`, not the 40 tok/s release target.

Current MiMo boundary:

- Tool parser/runtime contract: ChatML fallback placement is fixed, but current all-local `record_fact` probe remains red.
- Release clearance: still blocked by exact-cache prompt following, long-prompt coherence, speed, packaged/UI parity, and unimplemented VL/audio/video bridge.

## 2026-06-06 ZAYA1-VL reasoning capability truth and packaged app refresh

Source/runtime change:

- ZAYA text remains reasoning-capable with `qwen3` extraction.
- Current uploaded ZAYA1-VL MXFP/JANGTQ bundles are treated as vision/tool/cache-capable but not reasoning-capable because the VLM chat template has no thinking control and live proof showed hidden-only output.
- Runtime, smoke harness, panel family registry, session comments, and release-facing docs now suppress ZAYA1-VL reasoning rather than opening a fake synthetic `<think>` rail.
- This is not a model-quality fix. It is capability truth. To restore ZAYA1-VL reasoning, the model upload must ship a real VLM thinking template or otherwise produce visible-output proof under reasoning-on.

Live proof:

- `build/current-all-local-model-smoke-zaya-vl-mxfp4-after-thinking-capability-truth-20260606/summary.json` -> pass.
- Proven lanes: exact text cache repeat, `paged+zaya_cca` cache hit, multi-turn recall, required `record_fact` tool call, blue image, no-media-after-image, repeated blue image, and changed red image.
- Reported capability truth: `supports_thinking=false`, `reasoning_parser=null`, modalities `[text, vision]`, tool parser `zaya_xml`, native cache `zaya_cca_v1` with prefix, paged cache, and block disk L2.

Packaging proof:

- Rebuilt bundled Python from current source and local `/Users/eric/jang/jang-tools`; bundled verifier passed for vMLX, JANG tools, TurboQuant kernels, Step3p7 VLM registration, Gemma4 unified registration, VL modules, and audio dependencies.
- Rebuilt the sequoia staged app at `panel/release/sequoia-app/mac-arm64/vMLX.app`; app was Developer ID signed by electron-builder.
- `build/current-packaged-integrity-contract-after-zaya-vl-thinking-capability-truth-20260606.json` -> pass.
- `build/current-regression-suite-after-zaya-vl-thinking-capability-truth-20260606.json` -> status open only because release readiness is still blocked by the five live/model requirements.

Still blocked before release:

1. Cross-family live multi-turn smoke matrix is not release-cleared.
2. MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is not release-cleared.
3. MiniMax-M2.7-JANGTQ_K reporter parity/root cause is not release-cleared.
4. Real Electron UI cross-family live model matrix is not release-cleared.
5. DSV4 long-output/code/file-generation quality is not release-cleared.

Release boundary:

- No release tag was created.
- No DMG was produced through the gated public `dist` path.
- No notarization was completed; `electron-builder --dir` explicitly skipped notarization because notarize options were not generated.
- No `mlx.studio`, `vmlx.net`, updater manifest, or public download was updated.
- Future release notes must credit GitHub `@Hornsan1` for the reported runtime/media/cache issues.

## 2026-06-06 MiMo synced long/tool/cache proof after Max2 copy

Fresh artifacts:

```text
build/current-mimo-v25-jang2l-synced-long-tool-cache-proof-20260606.json
build/current-mimo-v2-jang2l-current-audit-after-synced-long-tool-cache-proof-20260606.json
```

Result:

- MiMo local bundle integrity is clean and stale local state is absent.
- Local Python/vMLX MiMo endpoint remains behaviorally red: cache exact rows empty, long prompt empty, forced tool call missing, and speed row emitted no usable completion tokens.
- Active native/cache telemetry does not prove usable cache semantics; the cache rows must produce the required visible output and they currently do not.
- MiMo media remains unbuilt in this Python/JANG-tools path. Vision/audio/video weights may exist in the bundle, but there is no release-proven MiMo media forward bridge.

Next required proof:

1. Relaunch a real MiMo source endpoint on Max2 or another Pod without clobbering active Qwen TP4 workers.
2. Run source-vs-quant first divergence against source and local quant.
3. If source passes and quant fails, fix the Python/vMLX/JANG_2L quant/runtime decode path.
4. If source also fails, fix or re-upload the MiMo model/artifact before runtime release claims.
5. Separately implement MiMo VL/audio/video forward support; typed unsupported-media errors are honest but not a product-complete media release.

### MiMo post-proof server death addendum

Artifact: `build/current-mimo-v2-jang2l-post-proof-server-health-20260606.json`.

The local MiMo server was no longer listening on `8897` after the synced long/tool/cache proof. This strengthens the runtime blocker: MiMo Python/vMLX needs crash/process-death repro logs in addition to decode-quality/source-vs-quant analysis.

## 2026-06-06 MiMo thinking-off template fix boundary

Artifact: `build/current-mimo-v2-jang2l-thinking-off-template-fix-live-20260606.json`.

Fixed:

- MiMo text-only cache/system prompts no longer enter the native closed `<think></think>` prompt rail when `enable_thinking=false`.
- Fresh live rows changed from empty visible output to visible `ACK` under thinking-off.

Still blocked:

- Exact output remains wrong: `ACK` instead of `ACK-CACHE-742`.
- Long-prompt text row still exits the server through Metal OOM.
- Tool row needs repro after OOM is isolated.
- VL/audio/video remains unbuilt in Python/vMLX for MiMo; preserved media weights are not a wired media runtime.

### MiMo first-request long-prompt OOM

Artifact: `build/current-mimo-v2-jang2l-long-prompt-first-request-oom-20260606.json`.

Fresh patched server plus first-request long prompt still exits through Metal OOM. This rules out cache residue and keeps MiMo long-context runtime work open.

## 2026-06-06 MiMo text-only route improvement

Artifact: `build/current-mimo-v2-jang2l-text-route-live-proof-20260606.json`.

Fixed for text-only simple-engine route:

- Long-prompt OOM is cleared by routing text-only MiMo through the language model rather than `mlx_vlm.generate`.
- Required XML tool call is not parsed in the current refresh; it emits raw `<tool_call>` plus punctuation garbage.
- Cache exact rows produce visible text instead of empty/null output.

Still open before release:

- Speed is still far below target.
- Exact prompt-following still misses `ACK-CACHE-742`.
- Continuous-batching prefix/paged/L2 cache proof is not refreshed.
- MiMo source-vs-quant is still blocked by no source endpoint.
- MiMo VL/audio/video remains unwired.

## 2026-06-06 MiMo audit pointer refresh after text-route fix

Fresh audit: `build/current-mimo-v2-jang2l-current-audit-after-cb-oneshot-prefill-20260606.json`.

The audit now consumes the text-route proof as current evidence: text cache and narrow long-prompt rows have partial positives, but tool protocol is red again in current normal-engine proof. Remaining active blockers are tool protocol, exactness, speed, source-vs-quant, prefix/paged/L2 full proof, and real MiMo VL/audio/video wiring.

## 2026-06-06 MiMo CB text/cache progress, media still open

The MiMo CB one-shot prefill patch clears only text/cache correctness rows:

- `build/current-mimo-v2-jang2l-cb-cache-after-mimo-oneshot-prefill-20260606.json`
- `build/current-mimo-v2-jang2l-current-audit-after-cb-oneshot-prefill-20260606.json`

MiMo media runtime remains unbuilt. The Python path still needs a real multimodal forward that connects image/audio/video processors, media embeddings, position/rope metadata, masks, cache keys, streaming, tool calls after media, and UI settings proof. Text/cache success must not be reused as VL/audio/video proof.

## 2026-06-06 MiMo CB prompt correction, media still unbuilt

- Current source no longer rewrites MiMo `enable_thinking=false` to the thinking-on prompt rail. This fixes a text decode/template leak, not media support.
- Live text/cache artifact: `build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json`.
- Proved in that artifact: exact repeated text row, paged prefix cache hit, q8 storage quantization, block-disk L2 tokens, and native `mixed_swa_kv_v1` cache metadata.
- Still open for this media worklist: MiMo `pixel_values`, image/audio/video embeddings, media-expanded prompt masks, media cache keys, post-media recovery, streaming media output, tool calls after media, UI modality exposure, and installed-app parity. Preserved vision/audio sidecars are not runtime media support.

## 2026-06-07 MiMo tool-status correction after current normal-engine refresh

Current artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-refresh-after-qwen35-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Correction:

- Any older row saying current MiMo `record_fact` is live-proven parsed is superseded.
- Current normal Python engine proof returns text/cache/multiturn positives, but `tool_choice=required` returns HTTP 400 with zero parsed tool calls.
- The generated raw preview starts `<tool_call>` and then punctuation/fullwidth-comma garbage, without `<function=record_fact>` or `<parameter=value>blue-cat</parameter>`.
- Classification remains `decode_loop` versus `model_artifact` until source-vs-quant first divergence or a replacement artifact proves otherwise.
- Do not advertise MiMo VL/audio/video: runtime modalities remain text-only until a real multimodal forward path is implemented and proven.

## 2026-06-08 MiMo object-media live-smoke update

- Artifact: `build/current-all-local-model-smoke-mimo-v25-jangtq2-object-media-e2e-20260608`.
- Harness change: MiMo media release smoke now validates colored square-object image/video prompts instead of degenerate 16x16 solid-frame dominant-color prompts.
- Live MiMo JANGTQ2 media result: image/video media rows pass (`Blue`, repeated `Blue`, changed `Red`, video `Blue`) and text-after-media recovers with `NONE`.
- Failure count moved from `8` to `4`; remaining failures are exact literal/tool mutation rows, not image/video runtime transport rows.
- Caveat: solid full-frame dominant-color prompts remain a model-behavior caveat and must not be used to claim broad visual reasoning quality. The current pass only proves media processor/token splice/vision/video bridge/post-media recovery for object-color rows.
- Release boundary: MiMo remains unreleased/un-cleared until JANG_2L and JANGTQ2 exactness, long-prompt, cache/prefix/L2, audio waveform E2E, UI/API settings parity, and cross-family gates are current-green.

## 2026-06-08 MiMo exactness isolation update

- Artifact: `build/current-mimo-v25-jangtq2-exactness-isolation-20260608`.
- Conservative launch used MiMo JANGTQ2 only, with prefix cache disabled, paged cache/L2 absent, and `--kv-cache-quantization none`; no source-vs-quant comparison was run.
- Literal corruption still reproduced in both Chat Completions and `/v1/completions`: examples include `B7-CAT-09` -> `B7CATT-09`, tool argument `B7-CAT-09` -> `B7CAT-09`, and copy prompt `B7-CAT-09` -> `B7C9099`.
- Current classification: prefix cache, paged cache, block-disk L2, runtime KV quantization, media path, and tool parser are not the primary exactness cause. Remaining likely causes are MiMo JANGTQ2 decode/runtime kernel behavior or quant artifact quality; do not repair this by rewriting JSON/tool arguments.

## 2026-06-08 MiMo JANG_2L exactness and tool-choice update

- Artifact: `build/current-mimo-v25-jang2l-exactness-isolation-20260608`.
- JANG_2L loaded under conservative no-prefix/no-paged/no-L2/no-KV-quant settings and became healthy, but left only `6.9GB` system memory available after a `105.96GB` Metal load.
- Raw `/v1/completions` exact copy preserved `B7-CAT-09`, so JANG_2L does not show the same raw-completion literal corruption as JANGTQ2.
- Chat JSON/tool behavior remains red: chat JSON wrapped/malformed the object, and a specific `tool_choice` request produced prose instead of a parsed tool call.
- Runtime fix: specific function `tool_choice` dicts now count as required for API enforcement. Live artifact `build/current-mimo-v25-jang2l-toolchoice-dict-enforcement-20260608` returns HTTP 400 with a raw preview instead of fake HTTP 200/prose when no tool call is parsed.
- Release boundary: this prevents false pass accounting only. MiMo tool generation quality, literal exactness, speed, cache/L2, UI/API, and audio/media depth remain open.
