# VL / audio / video runtime worklist

Date: 2026-06-06

Active worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`

Primary tracker: `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md`

Reporter credit for relevant release notes: GitHub `@Hornsan1`

## Current release boundary

Do not tag, notarize, or publish a new vMLX / mlxstudio release from this workstream until the live rows below are green. Static contracts, load-only proofs, and single-turn smoke rows are not sufficient.

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
| Gemma4 26B CRACK | Source-server smoke passed on 2026-06-06 for exact text, cache hit, multi-turn recall, reasoning, tool call, image/video color, and text-after-media. | Installed-app Responses visible-content/language quality and mixed-SWA app-engine speed floor remain open. | Installed full-output tails, installed speed gate, and packaged media/tool parity rows. |
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

1. MiMo multimodal bridge

- `jang_tools.mimo_v2.mlx_model` is text-oriented; multimodal runtime is not release-proven.
- vMLX MiMo MLLM path must accept and correctly route image/video/audio tensors, not just load an MLLM wrapper.
- Tool parser must handle MiMo's template dialect without fabricating tool calls from malformed raw XML.

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

1. Keep Gemma4 VLM prefill guard dynamic and typed: high-memory machines should not reject a 10GB predicted buffer solely because of the old 8GB default, while explicit env overrides still work.
2. Build MiMo media bridge or prove artifact/model blocker from actual Max2 bundle configs and runtime signatures.
3. Add shared structured-output repair/validation to the video/catalog benchmark layer and report raw-vs-repaired score separately.
4. Re-run live media matrix for Gemma4 12B, Qwen27/35 MTP, LFM, Step3.7 text-only, MiMo, Nemotron Omni, ZAYA-VL, DSV4, MiniMax, Hy3.
5. Only after live rows pass: rebuild app, sign, notarize, staple, install, and repeat installed-app proofs before release/tag/public download update.
