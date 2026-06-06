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
- `build/current-objective-proof-after-zaya-vl-no-media-refresh-20260606.json` closes the two Gemma4 26B CRACK rows for installed-app Responses visible-content/language and mixed-SWA speed.
- `build/current-regression-suite-after-zaya-vl-no-media-refresh-20260606.json` is `status=open` with `failed_steps=["release_regression_manifest"]`; source/package/parity/contracts pass, and five live/model rows remain.
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
| LFM2.5 | Text/cache/tool UI rows have passed for known bundles. | VL/audio/video only if artifact advertises it; hybrid SSM path-dependent cache rows must stay architecture-aware. | Live media row for any VL-advertised LFM artifact, otherwise capability must remain text-only. |
| Step3.7 Flash JANG_2L | Text-only route is stable; VLM route remains unsupported unless implemented. | Do not advertise Step3.7 VLM in vMLX until real processor/model forward path passes. Tool dialect and loop discipline remain model-behavior blockers. | Text-only live rows plus explicit unsupported-VLM guard row, or real Step3.7 VLM implementation with image/video proof. |
| MiMo V2.5 JANG_2L | Fresh Max2 bundle copied locally and manifest verified; cache head crash fixed; optimized rerun shows paged/L2/TurboQuant telemetry. Conservative no-cache/no-TQ diagnostic can answer exact `ACK`. | Release-red: optimized exact-cache row can be empty or rambling; required tool calls fail both optimized and conservative modes; speed is around 1-2 tok/s or worse; no release-cleared VL/audio/video path. | Implement/fix tool protocol and decode quality, prove speed target, then implement processor/model bridge for image/video/audio and prove media rows. |
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
| MiMo text runtime bridge | vMLX registers `mlx_vlm.models.mimo_v2` around `jang_tools.mimo_v2.mlx_model`; text/cache narrow proof exists. Missing media forward now raises typed `unsupported_media_modality` instead of a generic generation failure. | Real multimodal model module: vision encoder, audio encoder/tokenizer bridge, media projector/merger, `get_input_embeddings(pixel_values=...)`, `__call__(pixel_values=...)`, video/audio token expansion, and cache salt proof. | `runtime_dispatch` plus `model_artifact` pending source-vs-quant quality comparison. |
| MiMo JANG tools model forward | `/Users/eric/jang/jang-tools/jang_tools/mimo_v2/mlx_model.py` is explicitly text-only; its docstring says visual/audio towers are preserved but not wired. | Implement `mimo_v2_multimodal.py` or equivalent in JANG tools, then make vMLX register that module instead of a text-only compatibility shell. | `runtime_dispatch`; do not advertise VL/audio/video or force a parser/cache workaround. |
| MiMo long-prompt/tool quality | Cache classification and narrow text repeat work; conservative no-cache exact ACK can work. | First-divergence trace against source/higher-quality profile; tool protocol repair at model/template/runtime boundary; speed target proof. | `decode_loop` or `model_artifact` still unresolved. |
| Step3.7 VLM bridge | Source registration/no-heavy guards exist; text-only route is stable. | Real Step3.7 image/video processor, projector, forward, media cache salt, and live proof, or an explicit unsupported-VLM product row. | `runtime_dispatch`; text-only metadata view is not a VLM fix. |
| Nemotron Omni audio/video | Prior source rows exist for text/image/video/audio but are not current release-cleared. | Current source and installed app matrix for text-before/after media, audio/image/video carryover, tools, streaming, and L2. | `unknown_pending_repro`. |
| ZAYA/ZAYA1-VL media cache | Typed CCA cache contract exists; media routing rows exist in no-heavy contracts. | Live image/video rows proving media salt plus safe CCA/path-dependent restore and post-media recovery. | `kernel_cache` if partial restore fails; otherwise `unknown_pending_repro` until live. |
| MiniMax / Hy3 / Kimi media advertised variants | Text/tool/cache rows exist for selected bundles; media proof is not current in this slice. | Per-family advertised-modality audit: processor path, special tokens, cache policy, tools after media, installed-app parity. | `unknown_pending_repro`. |
| Shared structured-output repair | Register requires parse/repair/validate/retry while reporting raw parse failure. | Centralize benchmark/catalog repair for JSON/XML-ish outputs and prove raw-vs-repaired scoring rows. | `decode_loop` for native malformed output; caller-side repair is not guided decoding. |
| Release pointer/package parity | Source package pointers now track current 2026-06-06 artifacts and packaged integrity passes on the rebuilt staged app. | Full release still needs the seven live/model blockers closed, then notarized/stapled DMGs, install proof, public update manifest proof, and mlx.studio/vmlx.net download freshness proof. | `gateway_ui` until public release artifacts are current and verified. |

1. MiMo multimodal bridge

- `jang_tools.mimo_v2.mlx_model` is text-oriented; multimodal runtime is not release-proven.
- vMLX MiMo MLLM path must accept and correctly route image/video/audio tensors, not just load an MLLM wrapper.
- Tool parser must handle MiMo's template dialect without fabricating tool calls from malformed raw XML.
- Current fail-closed behavior is typed: unwired MiMo vision forward raises `UnsupportedMediaModalityError` / API code `unsupported_media_modality`. This is not a pass for MiMo vision; it is only correct failure classification and recovery.

2. Step3.7 VLM bridge

- Current safe route is text-only.
- Real Step3.7 VLM requires processor, media token expansion, model forward, cache salting, and full live proof.

3. Omni audio/video normalization

- Need current-source proof that audio/video content parts do not contaminate later text-only turns.
- Need installed app proof, not only source-server proof.

4. JSON/XML structured-output repair

- Bench/catalog tooling needs a shared parse/repair/validate/retry layer.
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
3. Add shared structured-output repair/validation to the video/catalog benchmark layer and report raw-vs-repaired score separately.
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
- MiMo text/tool quality still needs source-vs-quant first-divergence tracing, tool protocol repair at the real template/parser boundary, and speed proof. Do not force hidden prompts, fake tool-call conversion, or cache disablement as a production fix.
- ZAYA/ZAYA1-VL needs exact-instruction, reasoning-visible-output, multi-turn recall, and red-image semantic fixes while preserving the CCA/path-dependent cache constraints.
- DSV4 needs a safe-memory live host for long-output/code/file-generation and restart/L2 exactness proof.
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
