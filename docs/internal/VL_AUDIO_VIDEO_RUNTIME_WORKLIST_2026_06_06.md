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
| LFM2.5 | Text/cache/tool UI rows have passed for known bundles. | VL/audio/video only if artifact advertises it; hybrid SSM path-dependent cache rows must stay architecture-aware. | Live media row for any VL-advertised LFM artifact, otherwise capability must remain text-only. |
| Step3.7 Flash JANG_2L | Text-only route is stable; VLM route remains unsupported unless implemented. | Do not advertise Step3.7 VLM in vMLX until real processor/model forward path passes. Tool dialect and loop discipline remain model-behavior blockers. | Text-only live rows plus explicit unsupported-VLM guard row, or real Step3.7 VLM implementation with image/video proof. |
| MiMo V2.5 JANG_2L | Fresh Max2 bundle copied locally and manifest verified; cache head crash fixed; optimized rerun shows paged/L2/TurboQuant telemetry. Conservative no-cache/no-TQ diagnostic can answer exact `ACK`. XML-function prompt/parser fallback is fixed in source and live all-local smoke now emits parsed `record_fact` with `{"value":"blue-cat"}`. | Release-red: optimized exact-cache row can be empty or rambling; speed is around 1-2 tok/s; long-prompt coherence remains red; no release-cleared VL/audio/video path. | Fix decode quality and speed target, compare source-vs-quant first divergence, then implement processor/model bridge for image/video/audio and prove media rows. |
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
| MiMo long-prompt/tool quality | Cache classification and narrow text repeat work; conservative no-cache exact ACK can work. XML-function fallback now respects MiMo's native parser dialect and live `tool_choice=required` emits parsed OpenAI `tool_calls`. | First-divergence trace against source/higher-quality profile for exact-cache/long-prompt corruption; speed target proof; broader loop behavior and packaged/UI parity. | `decode_loop` or `model_artifact` still unresolved for quality/speed; parser dialect mismatch fixed in source. |
| Step3.7 VLM bridge | Source registration/no-heavy guards exist; text-only route is stable. | Real Step3.7 image/video processor, projector, forward, media cache salt, and live proof, or an explicit unsupported-VLM product row. | `runtime_dispatch`; text-only metadata view is not a VLM fix. |
| Nemotron Omni audio/video | Prior source rows exist for text/image/video/audio but are not current release-cleared. | Current source and installed app matrix for text-before/after media, audio/image/video carryover, tools, streaming, and L2. | `unknown_pending_repro`. |
| ZAYA/ZAYA1-VL media cache | Typed CCA cache contract exists; media routing rows exist in no-heavy contracts. | Live image/video rows proving media salt plus safe CCA/path-dependent restore and post-media recovery. | `kernel_cache` if partial restore fails; otherwise `unknown_pending_repro` until live. |
| MiniMax / Hy3 / Kimi media advertised variants | Text/tool/cache rows exist for selected bundles; media proof is not current in this slice. | Per-family advertised-modality audit: processor path, special tokens, cache policy, tools after media, installed-app parity. | `unknown_pending_repro`. |
| Shared structured-output repair | Register requires parse/repair/validate/retry while reporting raw parse failure. | Centralize benchmark/catalog repair for JSON/XML-ish outputs and prove raw-vs-repaired scoring rows. | `decode_loop` for native malformed output; caller-side repair is not guided decoding. |
| Release pointer/package parity | Source package pointers now track current 2026-06-06 artifacts and packaged integrity passes on the rebuilt staged app. | Full release still needs the seven live/model blockers closed, then notarized/stapled DMGs, install proof, public update manifest proof, and mlx.studio/vmlx.net download freshness proof. | `gateway_ui` until public release artifacts are current and verified. |

1. MiMo multimodal bridge

- `jang_tools.mimo_v2.mlx_model` is text-oriented; multimodal runtime is not release-proven.
- vMLX MiMo MLLM path must accept and correctly route image/video/audio tensors, not just load an MLLM wrapper.
- Tool parser must handle MiMo's template dialect without fabricating tool calls from malformed raw XML. Source now distinguishes MiMo `xml_function` from Qwen/Step template fallbacks and live-proves parsed `record_fact`; broader loop behavior still needs proof.
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
- MiMo text quality still needs source-vs-quant first-divergence tracing and speed proof. The source-side XML-function tool parser/template contract is fixed and live-proven for `record_fact`, but broader loop behavior and packaged/UI parity remain open. Do not force hidden prompts, fake tool-call conversion, or cache disablement as a production fix.
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

- `decode_loop`: the failure tracks the DSV4 assistant suffix rail, `thinking_closed` (`<|Assistant|></think>`) versus `thinking_open` (`<|Assistant|><think>`), and affects exact identifier fidelity.
- `model_artifact`: not proven yet. The same uploaded artifact can produce exact output on the thinking-open rail, so corrupt weights are not the first explanation.
- `runtime_dispatch`: not proven as the root cause yet. The encoder currently treats the `</think>` off suffix as an explicit DSV4 contract; do not change it without a source/model-card trace and broader route tests.

Required next DSV4 checks:

1. Trace DSV4 template/model documentation for whether `enable_thinking=false` should end with `</think>` or a different chat suffix.
2. Run direct raw prompt or model-native reference generation, if available, to separate runtime prompt construction from model behavior.
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
- `build/current-mimo-v2-jang2l-current-audit-after-first-token-stop-20260606.json` is the active MiMo audit. It confirms stale MiMo local state is absent after deleting the stale Hugging Face `transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L` cache, while keeping MiMo release open.

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
4. MiMo remains release-red for local runtime quality: long-prompt coherence, exact-cache prompt-following, and speed fail. The earlier OpenAI tool protocol failure is narrowed by the XML-function parser/template fix and current live smoke now emits parsed `record_fact`; this does not clear broader loop behavior or packaged/UI parity. The remaining evidence points beyond cache store/reconstruct and beyond sink toggles toward model/runtime weight mapping, template/defaults, or quantized forward quality.
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

Artifact: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-xml-function-template-fix-20260606/summary.json`

Source/runtime change:

- vMLX no longer misclassifies explicit MiMo `xml_function` parser prompts as Qwen/Step-style native tool prompts just because they contain `<|im_start|>`, `<tools>`, and `<function=example_function_name>`.
- Fallback for MiMo XML tools now emits native XML-function instructions compatible with `xml_function` / `mimo_xml_function`, not JSON inside `<tool_call>`.
- Focused parser/fallback tests passed: `tests/test_tool_fallback_injection.py`, `tests/test_xml_function_tool_parser.py`, and selected XML/fallback/tool-format rows, `74 passed`.

Live proof result:

- `tool_required` now passes with a parsed OpenAI function call: `record_fact({"value":"blue-cat"})`.
- Multi-turn recall and reasoning visible-output probes pass.
- Cache infrastructure remains active: `mixed_swa_kv_v1`, `mimo_v2_asymmetric_swa`, prefix cache, paged cache, block disk L2, and TurboQuant q4 storage-boundary telemetry.
- Smoke still fails because `text_cache_repeat_1` is empty visible output and `text_cache_repeat_2` rambles instead of exact `ACK`.
- Decode speed remains about `1.78 tok/s`, not the 40 tok/s release target.

Current MiMo boundary:

- Tool parser/runtime contract: fixed in source for the all-local `record_fact` probe.
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
