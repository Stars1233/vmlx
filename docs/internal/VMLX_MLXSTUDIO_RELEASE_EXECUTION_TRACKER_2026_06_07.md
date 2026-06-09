# vMLX / MLXStudio release execution tracker - 2026-06-07

This is the active checklist for the Python vMLX engine and MLXStudio Electron app release objective.
It is intentionally stricter than source-only tests or narrow model smokes.

Do not use this document to justify release while any `OPEN` row remains.
Do not sign, notarize, tag, or publish downloads unless the full release checklist is green or Eric explicitly overrides the lock in the current turn.

## Current authoritative artifacts

| Surface | Artifact | Current result |
| --- | --- | --- |
| Full release checklist | `build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json` | `status=open`, `failed_count=18`; consumes the MiMo media runtime stamp-gate audit, no-source exactness classifier rows, Qwen restart/L2/video/long-context checks, and MiniMax #179 refreshed audit. The increase from 16 to 18 is intentional release honesty: current MiMo bundles fail closed as text runtime instead of over-advertising media support. |
| Objective proof digest | `build/current-objective-proof-after-mimo-media-runtime-stamp-gate-20260608.json` | status remains open; cache architecture remains PASS, while MiMo, MiniMax, full UI matrix, DSV4, and current-source proof rows remain open |
| Release regression manifest | `build/current-release-regression-manifest-after-mimo-media-runtime-stamp-gate-20260608.json` | `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; MiMo root-cause evidence now includes the media runtime stamp gate, stale-cache cleanup, no-source exactness classifier, object image/video E2E, audio waveform E2E pass, and L2 restart restore. |
| MiMo current audit | `build/current-mimo-v2-jang2l-current-audit-after-media-runtime-stamp-gate-clean-stale-20260608.json` | `status=open`, `local_release_clearance=false`; stale local state is absent; MiMo media capabilities now fail closed for current `weights_preserved_text_runtime` bundles without importing/registering the media runtime; image/video object E2E, audio waveform E2E, decode speed, and fresh-process block-disk L2 restore are green; exactness/long-prompt/CB/source-vs-quant/media-wiring remain blocked |
| MiMo no-source exactness classifier | `build/current-mimo-v2-no-source-exactness-classifier-after-media-runtime-stamp-gate-20260608.json` | `status=open`; primary classification remains `jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes`, and the classifier now separately tracks the current JANG_2L JSON sentinel as semantic exactness mismatch after runtime thinking-off forwarding. Current evidence proves JANGTQ_2 plain exact literal copy fails before tool parsing or JSON repair can matter, while local JANG_2L preserves plain literals and now passes both XML tool rows after tight-memory cache/prefill/parser repairs. JANG_2L JSON now emits parseable JSON with exact `B7-CAT-09`, but returns `count:1` instead of `count:3`; do not clear with JSON repair. Parser argument rewrite, cache/KV/L2, SwitchGLU fast path, compiled router, and TQ gather kernel are excluded as primary JANGTQ_2 causes; source-vs-quant remains missing. |
| MiMo JANG_2L runtime thinking-off proof | `build/current-mimo-v25-jang2l-json-sentinel-token-trace-20260608/summary.json`; `build/current-mimo-v25-jang2l-json-sentinel-after-enable-thinking-forward-20260608/summary.json`; `build/current-mimo-v25-jang2l-json-sentinel-toplogits-trace-scores-20260608/summary.json` | Runtime bug fixed partial. Before the forwarding fix, token trace showed the non-streaming batched MLLM request had `enable_thinking=None`, sampled `<|im_end|>` as the first token, and stopped with empty content. After forwarding `enable_thinking=false` into `MLLMScheduler.generate`, trace shows MiMo thinking/EOS processors active, first token `{\"`, final stop on `<|im_end|>`, and visible JSON output. The top-logit trace proves the remaining failure is not stop-token handling or JSON repair: the model/runtime score ranked token `1` above requested token `3` at the `count` position (`22.05` vs `20.80`), producing `{\"status\":\"ok\",\"value\":\"B7-CAT-09\",\"count\":1}` vs expected `count:3`. |
| MiniMax #179 current audit | `build/current-issue179-minimax-k-root-cause-audit-after-qwen-installed-video-wiring-20260607.json` | `status=open`; local source/installed app diagnostics and cancel route are clean, but reporter parity metadata is missing, reporter server hash drifts from local/public artifacts, and reporter-side session/log/cancel lifecycle proof is still absent |
| LFM real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json` | `status=pass`; `lfm25-mxfp4-responses-tools-cachecontrols-20260607` refresh clears LFM mixed-identity matrix partial |
| Qwen 3.6 27B MTP real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; current model identity, Responses streaming, built-in tool loop, reasoning display, image answer `Red.`, settings overrides, MTP-compatible deterministic sampling, native MTP activation, hybrid SSM cache, TurboQuant attention KV, L2 block disk, SSM companion disk, and server cache controls are proven. The same combined reasoning/tools/image row failed at `max_tokens=96`, so thinking-mode UI proofs need a realistic output budget. |
| Qwen 3.6 27B MTP installed-app current proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; installed `/Applications/vMLX.app` bundle used bundled Python `vmlx_engine 1.5.56`, Responses streaming, tool loop, reasoning display, image answer, native MTP, hybrid SSM cache, TurboQuant attention KV, block L2, SSM L2, server cache controls, settings persistence, parser/language leak checks, and installed-app UI route are proven. Functional parity is green for this slice, but packaged Metal acceleration symbols reported `nax_symbols=0` / `naxtile_symbols=0`, so installed-app speed/perf parity remains a release risk. |
| Qwen 3.6 27B MTP restart-L2 restore proof | `build/current-qwen27-mxfp4-mtp-restart-l2-restore-20260607/summary.json` | `status=pass`; phase 1 writes block L2, phase 2 fresh-process restore returns visible ack with `cached_tokens=63`, `cache_detail=paged+ssm+disk`, block disk hit, typed `hybrid_ssm_v1`, attention-only TurboQuant KV, native SSM companion policy, and native MTP active. |
| Qwen 3.6 27B MTP installed-app video proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json` | `status=pass`; installed app video needed `max_tokens=512`. The max-256 attempt decoded video but stopped reasoning-only with no visible answer. The passing max-512 proof decoded the MP4 data URL, extracted six frames, returned visible video content, and proved `video_where_supported`, reasoning display, Responses streaming/cache detail, tools, native MTP, typed `hybrid_ssm_v1`, TurboQuant attention KV, block L2, SSM L2, server cache controls, and media-safe skip of media prompt cache store. |
| Qwen 3.6 27B MTP long-context cache-tail proof | `build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json` | `status=pass`; installed bundled Python served a 31,647-input-token prompt. Cold request returned exact begin/middle/end tail markers and wrote 495 block-L2 entries (`31,646` tokens) plus SSM companion disk state (`63,262` tokens). Warm request returned exact markers again in 9.5s with `cached_tokens=31646`, `cache_detail=paged+ssm`, 1,485 block-disk hits, typed `hybrid_ssm_v1`, TurboQuant attention KV, and native MTP active. |
| Qwen 3.6 35B MXFP8 MTP real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; current model identity, Responses streaming, built-in tool loop, reasoning display, image answer, deterministic UI sampling, native MTP activation under tool-compatible D1, trained top-k 8, hybrid SSM cache, TurboQuant attention KV, block-disk L2, SSM companion disk, and server cache controls are proven |
| Qwen 3.6 35B MXFP8 MTP installed-app current proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-35b-mxfp8-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; installed `/Applications/vMLX.app` bundle used bundled Python `vmlx_engine 1.5.56`, Responses streaming, tool loop, reasoning display, image answer, native MTP, trained top-k 8, hybrid SSM cache, TurboQuant attention KV, block L2, SSM L2, server cache controls, settings persistence, parser/language leak checks, and installed-app UI route are proven. Image media was actually exercised in the corrected run (`requestedMedia=true`, `num_images_processed=1`, `vl_image` present). Functional speed was strong in this slice (`83-89 live tok/s`), but packaged Metal acceleration symbols still report `nax_symbols=0` / `naxtile_symbols=0`, so packaged acceleration parity remains a release risk. |
| Qwen 3.6 35B MXFP8 MTP restart-L2 restore proof | `build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json` | `status=pass`; bundled app Python served two fresh processes against the same block cache. Phase 1 wrote 27 block-L2 entries and SSM companion disk state. Phase 2 returned visible `ACK-QWEN35-L2`, restored `cached_tokens=1695` with `cache_detail=paged+ssm+disk`, hit 27 block-disk blocks, hit SSM companion disk once, exposed typed `hybrid_ssm_v1`, attention-only TurboQuant KV, and native MTP depth 3 active. |
| Qwen 3.6 35B MXFP8 MTP installed-app video proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json` | `status=pass`; installed app video at `max_tokens=512` decoded the MP4 data URL, extracted six frames, returned visible video content, and proved `video_where_supported`, reasoning display, Responses streaming/cache detail, tools, native MTP depth 3, trained top-k 8, typed `hybrid_ssm_v1`, TurboQuant attention KV, block L2, SSM L2, server cache controls, and media-safe skip of media prompt cache store. |
| Qwen 3.6 video UI current proofs | `docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max256-20260607-proof.json`; `docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json` | `status=pass` for both current Qwen MTP bundles; video data URL persisted, six frames decoded, `video_where_supported`, reasoning display, tool loop, native MTP, typed hybrid SSM cache, TurboQuant attention KV, block L2, SSM L2, and server cache controls are proven. Qwen35 needed `max_tokens=512`; at 256 it decoded video but stopped reasoning-only. |
| Gemma 4 12B JANG_4M real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-responses-tools-image-cachecontrols-after-media-fallback-20260607-proof.json` | `status=pass`; default optimized server launch, Responses streaming, built-in tool loop, image response `Red`, mixed-SWA paged cache hits, block-disk L2 writes, cache controls, settings persistence, and parser/language leak checks are proven |
| API/cache/Responses no-heavy contract | `build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json` | contract green, not live release proof |
| Generation defaults and startup parity contract | `build/current-generation-defaults-contract-after-do-sample-false-mimo-20260607.json` | `status=pass`; CLI/server resolver and MLXStudio startup/session settings both reflect model-owned defaults including `generation_config.do_sample=false` as greedy omitted-request sampling. JANG chat sampling metadata still overrides generic `generation_config`, explicit request/CLI overrides still win, and panel startup must not synthesize hidden `--default` sampler flags from UI/session state. |

## Open objective rows

| Row | Status | Required release evidence |
| --- | --- | --- |
| Cross-family live multi-turn smoke matrix is release-cleared | OPEN | Live server matrix across current release-critical model families with text, multi-turn, tools, tool-result continuation, raw-leak checks, JSON/code/whitespace checks, cache telemetry, and no hidden parser/template rewrites. |
| MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared | OPEN | Long prompt coherence, exact tool/JSON literal values, CB/simple parity, source-vs-quant or equivalent artifact classification, and real VL/audio/video runtime if advertised. |
| MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared | OPEN | Reporter-side parity artifact, installed/public/source hash parity, prompt reproduction or log proof, cancellation cleanup, strict native MiniMax tool dialect parsing, and no raw markup leaks. |
| Real Electron UI unblocked non-MiMo live model matrix is proven | OPEN | LFM MXFP4 dev-Electron proof refreshed on 2026-06-07 and LFM family matrix is now pass. Qwen 3.6 27B and 35B now have deterministic real-UI Responses/tools/image/video/reasoning/cache proofs with native MTP active and `long_tool_loop`, `reasoning_display`, `vl_image`, and `video_where_supported` green. Qwen27 and Qwen35 now also have installed-app image/reasoning/tools/cache and installed-app video passes, with installed-app video enforced in the full checklist. DSV4 missing/memory-gated, largest-context, cancellation cleanup, and installed-app full matrix remain open. |
| Real Electron UI cross-family live model matrix is release-cleared | OPEN | Full installed UI matrix across MiMo, Qwen MTP, Gemma4, Step3.7, LFM, MiniMax, Nemo/Nemotron Omni, DSV4, and other current release families. |
| DSV4 long-output/code/file-generation quality is release-cleared | OPEN | Memory-permitted DSV4 exact code/file/long-output proof with native SWA/CSA/HCA cache, restart L2, tool loops, and UI reflection. |

## Current MiMo V2.5 JANGTQ2 facts

Current local model path:

```text
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2
```

Current MiMo audit:

```text
build/current-mimo-v2-jang2l-current-audit-after-media-runtime-stamp-gate-clean-stale-20260608.json
```

Current green MiMo subproofs:

| Check | Status | Note |
| --- | --- | --- |
| Manifest integrity | GREEN | 115 rows, no missing/mismatched files in current audit. |
| Stale local state cleanup | GREEN | Stale backup/cache targets absent in audit. |
| Structural verify | GREEN | Current structural artifact accepted. |
| SwitchGLU selected expert parity | GREEN | Selected expert parity proof accepted. |
| Text cache narrow proof | GREEN | Narrow text cache proof accepted. |
| Cache-vs-nocache next token | GREEN | Next-token cache parity proof accepted. |
| Tool protocol structure | GREEN | OpenAI tool-call JSON structure parses for failed rows. |
| Prefix/paged/L2 cache reproved | GREEN | In-process prefix/paged/L2 cache stack proof is present, and fresh-process block-disk L2 restore now returns correct visible output with `cache_detail=paged+disk`, `cached_tokens=67`, and disk-backed hit telemetry. |
| Decode speed target | GREEN | Latest decode evidence is above the accepted near-40 tok/s floor. |
| MiMo media capability truth | GREEN | Current text-runtime bundles return runtime modalities `["text"]` and mark preserved image/video/audio as `preserved_unwired`; the capability path no longer imports/registers MiMo media runtime before the metadata gate, so stale HF dynamic-module cache is not recreated by `/capabilities`/audit. |
| API/cache/Responses no-heavy contract | GREEN | Route/telemetry/UI plumbing contract green. |
| Exactness not cache/KV caused | GREEN | Failures reproduce with KV quant, native storage quant, prefix, paged, L2, and hits disabled. |

Current MiMo blockers:

| Blocker | Classification | Required next evidence |
| --- | --- | --- |
| `mimo_long_prompt_coherence_blocked` | model/runtime decode quality or artifact behavior | Long prompt must produce coherent, complete, instruction-following output under release runtime without prompt-only hacks. |
| `mimo_jangtq2_artifact_exactness_blocked` | JANGTQ_2 plain literal copy regression; JANG_2L plain copy passes | Required plain text, tool args, and JSON exact values must preserve literals such as `blue-cat` and `B7-CAT-09`; JSON repair or parser rewriting must not rewrite semantic values to fake a pass. |
| `mimo_jang2l_json_sentinel_exactness_blocked` | JANG_2L tools survive tight memory and JSON no longer first-stops, but exact semantic values remain open | Current fixed-path proof emits parseable JSON and preserves `B7-CAT-09`, but returns `count:1` instead of requested `count:3`. Top-logit trace ranks token `1` above expected token `3` at the count position, so this is no longer the stale non-stream thinking-off forwarding bug and is not a parser, cache, or stop-token issue. It remains model/runtime exact-value quality. Do not clear it with JSON repair or prompt-only folding. |
| `mimo_xml_function_prompt_bloat_reduced` | native MiMo XML schema no longer triggers duplicate fallback injection | Current no-heavy render proof keeps MiMo tool prompt at 307 tokens without fallback text; live `JANGTQ_2` still emits structured tool calls, but literal exactness remains open. |
| `mimo_jang2l_tool_memory_pressure_fixed_partial` | 105G JANG_2L tool rows now survive tight memory | Current proof `build/current-mimo-v25-jang2l-exactness-tight64-parser-cache-skip-variant-probe-20260608/JANGQ_MiMo-V2.5-JANG_2L/result.json` passes both XML tool rows without Metal OOM. This does not release-clear JANG_2L because sentinel JSON still returns the wrong semantic count and broader long-prompt/CB/UI rows remain open. |
| `mimo_jang2l_runtime_model_type_detection_fixed_partial` | batch generator now resolves inner language-model `mimo_v2` type when the outer VLM wrapper config is blank | This prevents MiMo runtime processors from being skipped by wrapper metadata drift. Live proof after the fix shows JANG_2L tool rows still pass with exact literals, but it does not fix the isolated JSON sentinel empty-output row. |
| `mimo_cb_system_prompt_working_set_pressure_blocked` | CB route/resource behavior | Continuous-batching route must handle system/tool prompts without empty stop or working-set collapse. |
| `mimo_source_vs_quant_first_divergence_missing_or_failed` | unresolved runtime-vs-artifact boundary | User currently disallowed source-vs-quant due RAM. Need either reauthorization or an equivalent current-artifact classification that is strong enough to decide runtime fix vs model requant. |
| `mimo_audio_waveform_live_e2e` | current MiMo JANGTQ2 audio waveform E2E clear | Current proof routes raw audio through MiMo audio-code preprocessing and returns visible audio-conditioned output. Keep the fail-loud missing-payload classifier as a regression guard. |
| `mimo_media_runtime_implementation_missing` | current metadata is fail-closed text runtime | Current bundles preserve vision/audio/video sidecars and live object-media rows prove the compatibility media path, but `/capabilities` still reports those modalities as `preserved_unwired` until a media-enabled MiMo artifact stamps `mimo_v2_multimodal_runtime` and passes live media/cache/UI proof. |

MiMo cache/exactness boundary:

- Do not blame prefix cache, paged cache, block-disk L2, runtime KV quant, or cache hits as the primary literal-exactness cause.
- The no-prefix/KV-none proof reproduced the same exactness failures with those mechanisms disabled.
- The no-source exactness classifier records the current evidence as
  `model_generated_literal_mutation_after_valid_parser_structure`, not parser
  mutation.
- Do not clear MiMo with prompt folding, hidden sampling overrides, JSON repair of semantic values, forced tool argument rewrites, disabled cache, or hidden failed rows.
- If current evidence proves artifact corruption or quantization damage, tell Eric the exact requant/reupload contract instead of compensating in runtime.

## Cache and Responses release requirements

No-heavy API/cache/Responses contracts are green, but live release still requires per-family E2E.

Current no-heavy contract:

```text
build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json
```

## CLI and MLXStudio startup parity requirements

Startup parity is a release requirement, not a cosmetic UI row.

| Startup surface | Required behavior |
| --- | --- |
| CLI `vmlx serve` | Startup flags and omitted request defaults must resolve from request overrides, explicit CLI/session values, JANG chat metadata, then `generation_config.json`; no hidden sampler, parser, cache, MTP, max-output, or max-context override may be invented. |
| MLXStudio generated launch | Session create/settings/reset flows must preview and launch the same app-owned flags the server expects, including parser, reasoning, cache, MTP, max output, max context, and model-owned generation defaults. |
| `generation_config.do_sample=false` | Omitted sampling resolves to greedy defaults (`temperature=0.0`, `top_p=1.0`, `top_k=0`) unless JANG chat sampling metadata or explicit request/CLI values override it. |
| JANG chat sampling metadata | Takes precedence over generic `generation_config` and must clear generic `doSample` UI state rather than stacking conflicting defaults. |
| Additional args | User text additional args may not override app-owned startup flags for parser, reasoning, cache, MTP, model name/template, server ports, or generation defaults. |
| Release proof boundary | The no-heavy startup contract proves source/UI plumbing only. Each model family still needs live API/UI output proof showing the launched process actually used the intended parser, reasoning, cache, MTP, and token limits. |

Current green endpoint/plumbing checks in that contract:

| Endpoint or surface | Current no-heavy coverage | Release boundary |
| --- | --- | --- |
| `/v1/chat/completions` | Sampling kwargs, stream cache detail usage, output caps overriding server default. | Must still be proven with visible stream and non-stream output per live family. |
| `/v1/responses` | Sampling kwargs, `previous_response_id` history, JSON schema/text-format preservation, stream cache detail usage, output caps overriding server default. | Must still be proven per live family with tools, tool-result continuation, cancellation, structured output, no hidden-only response, and tail inspection. |
| Responses cancel route | Current-source route contract exists in the broader manifest. | Installed/public package parity still matters; stale public app route absence is not a source-engine fix. |
| Cache stats/reuse endpoints | `cache_stats_reuse_skip_telemetry` and `cache_reuse_endpoints` are green. | Endpoint green is telemetry/plumbing only; it does not prove a model family reuses cache correctly. |
| `/health` native cache block | DSV4 native cache, ZAYA typed CCA, plain attention KV, hybrid SSM partial reuse, TurboQuant KV runtime contract, and no generic TQ on hybrid SSM are covered. | Each architecture still needs live typed-cache proof with family-specific schema and restore behavior. |
| Max output vs max context | Request output caps override server default; prompt/context caps stay separate from output caps. | UI settings, CLI args, and API capabilities must agree for the shipped app. |

Every release-critical family must prove:

| Cache/API item | Required proof |
| --- | --- |
| Chat Completions | Non-stream and stream visible output, tool requests, cancellation, max output cap, max context cap. |
| Responses API | Non-stream and stream, `previous_response_id`, tool calls, tool-result continuation, cancellation, structured output, no empty hidden-only response. |
| Anthropic Messages | Output caps, tools where supported, no parser/template drift. |
| Ollama chat/generate | Output caps, streaming, sentinel behavior, no hidden server default drift. |
| Prefix cache | First miss plus second hit with `cached_tokens` and `cache_detail`. |
| Paged cache | Correct page accounting and no partial-hit corruption. |
| Block-disk L2 | Write, fresh-process restore, hit telemetry, unload/reload or restart proof. |
| TurboQuant KV | Encode/decode/restore only for standard KV families where valid; never substitute generic TQ KV for hybrid SWA/SSM/composite caches. |
| Typed native cache | SWA, SSM, DSV4 composite, MiMo asymmetric SWA, Gemma mixed SWA, and other nonstandard families must use their own schema and restore rules. |
| Largest safe context | Cold miss, warm hit, output tail inspection, cancellation cleanup, memory/resource telemetry. |

Per-family cache reuse is only release-green when the same family has all of:

| Required cache proof | Minimum evidence |
| --- | --- |
| First miss | Request starts cold or with explicit skip, no false positive cache hit. |
| Second hit | Later request reports positive `cached_tokens` and expected `cache_detail`. |
| Native schema | `/health` or capabilities expose the expected native cache family/schema. |
| L2 write | Block disk and any companion state stores positive token counts. |
| L2 restore | Fresh process, unload/reload, or restart request hits disk and returns visible correct output. |
| Media salt | Image/video/audio changes do not reuse stale text-only cache entries. |
| Cancellation cleanup | Aborted Responses/Chat stream does not leave stale scheduler/cache state that corrupts the next request. |
| UI route | MLXStudio settings and gateway route expose the same cache/parser/max-token behavior as CLI/API. |

## Tool, parser, structured output, and leak requirements

Every family must prove:

| Behavior | Required proof |
| --- | --- |
| Required tool call | Correct API-native tool call with valid function name and JSON args. |
| Auto tool call | Model chooses tool only when appropriate and does not raw-dump tool dialect. |
| Tool-result continuation | Final answer uses tool output and stops instead of looping. |
| No-tool request | No fake tool insertion or hidden tool fallback. |
| Loop stop | Repeated mock outputs do not cause infinite command/tool loops. |
| Hidden reasoning leak | Thinking/reasoning stays in the correct channel or remains suppressed when disabled. |
| Raw markup leak | No raw XML, DSML, MiniMax, Qwen, or template tags in visible assistant text unless explicitly requested. |
| JSON/XML/code exactness | Valid parse plus exact semantic values; repair can fix syntax, not fabricate correct values. |
| Whitespace-sensitive output | Exact code and whitespace rows checked. |

## Media/VL/audio/video release requirements

Every advertised media-capable family must prove:

| Media item | Required proof |
| --- | --- |
| OpenAI content parts | Images/audio/video accepted through API content parts. |
| MLXStudio app files | Drag/drop or file picker reaches the same server path as API. |
| Data URLs and safe URLs | Supported where intended, rejected safely where unsupported. |
| Media accounting | Media-expanded prompt accounting prevents unsafe OOM while allowing reasonable inputs on 128GB systems. |
| Media-salted cache | Text cache does not reuse across changed image/video/audio content. |
| Post-media recovery | Text request after media failure or rejection returns visible output without rolling back conversation manually. |
| Media plus tools | Tool requests after media prefill work or fail with a typed, recoverable error. |
| Unwired modalities | If runtime is text-only, metadata and UI must say preserved/unwired, not advertise working media. |

## Per-family release checklist

| Family | Current status | Must still prove or fix |
| --- | --- | --- |
| MiMo V2.5 JANGTQ2 | OPEN | Exact literals, long coherence, CB pressure, artifact/runtime classification, real VL/audio/video, UI parity. |
| Qwen 3.6 27B MTP | PARTIAL/GREEN UI TOOLS+IMAGE+VIDEO+REASONING+MTP | Current deterministic real UI proofs confirm model identity, Responses streaming, built-in tool loop, reasoning display, image answer, video answer, settings overrides, native MTP activation under tool-compatible D1, cache-hit telemetry, hybrid SSM cache, TurboQuant attention KV, block-disk L2, SSM companion L2, server cache controls, and no parser/language leak. Installed-app image/reasoning/tools/cache parity, installed-app video, restart-L2 restore, and 31k-token long-context cache-tail proof are now functionally green and enforced in the full checklist, but cancellation cleanup matrix and packaged acceleration/speed parity are still missing. |
| Qwen 3.6 35B MTP | PARTIAL/GREEN UI TOOLS+IMAGE+VIDEO+REASONING+MTP | Current source, dev-Electron, and installed-app proofs confirm MTP autodetect, `gdn_sink` compatibility, Responses streaming, built-in tool loop, reasoning display, image answer, video answer, deterministic UI sampling, native MTP activation under tool-compatible D1, trained top-k 8, hybrid SSM cache, TurboQuant attention KV, block-disk L2, SSM companion L2, server cache controls, and no parser/language leak. Installed-app image/reasoning/tools/cache parity, installed-app video, and restart-L2 restore are now functionally green and enforced in the full checklist, but 30k-token largest-context tail proof, cancellation cleanup matrix, and packaged acceleration symbol parity are still missing. |
| Nemo/Nemotron Omni | OPEN for media/full matrix | RADIO/vision/audio/video bridge, media cache, tools, structured output, UI. |
| LFM | Needs current full matrix proof | Text/tools/multiturn, JSON/XML/code exactness, cache/L2, API parity, UI. |
| MiniMax/JANGTQ_K | OPEN | Current local source/installed app diagnostics and cancel route are clean, but Issue179 remains open because reporter-machine parity metadata is missing, reporter installed/public/local server hashes drift, reporter session/log/cancel lifecycle proof is absent, and no concrete prompt currently reproduces the reported wrong-language/numeric screenshot shape locally. Do not clear from local-only smoke. |
| DSV4 Flash | OPEN | Native SWA/CSA/HCA cache, exact code/file/long output, restart L2, tool loops, UI, sufficient memory. |
| Step 3.7 Flash | PARTIAL/GREEN SOURCE VLM ROUTING | Text-only stability, honest VLM boundary, tools, loop stop, multiturn, API/UI parity. Do not fake media by metadata override. |
| Gemma 4 12B MXFP4/MXFP8/JANG_4M | PARTIAL/GREEN UI TOOLS+IMAGE+CACHE FOR JANG_4M | Current JANG_4M default-launch Electron proof confirms Responses streaming, built-in tool loop, image answer, server cache controls, settings persistence, `paged+mixed_swa` text/tool cache hits, and block-disk L2 writes. The media turn uses a scoped simple MLLM fallback because optimized batched Gemma4 media prefill produced corrupted visible output; do not claim optimized batched media-cache parity is fixed. MXFP4/MXFP8, audio/video, installed-app parity, and full matrix remain open. |
| ZAYA/hybrid/SSM families | OPEN inside cross-family matrix | Typed cache, partial-hit rejection, async rederive or clean prompt-boundary store, media if advertised, UI. |

## Release sequence lock

Only after all objective rows are green:

1. Re-run current release checklist and objective proof.
2. Re-run current regression suite under expected-open policy with no unexpected open rows.
3. Verify current branch/main/push state explicitly.
4. Rebuild packaged Python and MLXStudio app from current source.
5. Verify bundled Python parity.
6. Run installed-app live UI/API/cache/model matrix.
7. Developer ID sign.
8. Notarize and staple.
9. Verify notarized app/DMG.
10. Tag and publish release only after release notes include GitHub `@Hornsan1` credit for reported runtime/media/cache issues.
11. Update public downloads/appcast only after the notarized artifact and release manifest prove green.

## Immediate next blocker choices

Pick one, do not drift:

1. MiMo quality/media blocker: decide runtime vs artifact without source-vs-quant unless Eric reauthorizes RAM, then implement the real fix or give exact requant contract.
2. MiniMax #179 blocker: reproduce reporter parity/root cause and fix runtime/parser/cancel path if it is not stale public/install drift.
3. Real Electron UI matrix: run current installed-app UI/API/cache settings proof for non-MiMo models first, then full cross-family matrix.
4. DSV4 blocker: wait for sufficient memory or use a guarded host, then prove exact code/file/long-output quality with native composite cache.

## Agent control-plane update - 2026-06-07

`AGENTS.md` now contains a hard current-objective execution contract for this
release. Future agents must keep the whole Python engine plus MLXStudio app
release map active, including CLI startup, MLXStudio startup, Chat Completions,
Responses, tools, structured output, cache, TurboQuant KV, media, installed app,
per-family blockers, and signing/notarization lock.

This does not make any model family green by itself. It prevents future work
from drifting into a smaller proof such as source-only tests, one model smoke,
upload chores, or deprecated workspace notes.

Focused validation:

- `.venv/bin/python -m py_compile tests/cross_matrix/run_full_release_objective_checklist.py tests/cross_matrix/release_regression_manifest.py tests/cross_matrix/summarize_objective_proof.py` passed.
- `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py` passed (`2 passed`).
- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py` passed (`3 passed`) and pins AGENTS release-objective surfaces, per-family rows, no-fake-fix rules, and release-lock boundaries.
- `tests/test_agents_release_control_plane.py` is now wired into the current regression suite focused pytest gate and source-hash list.
- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py tests/test_current_regression_suite.py -k "agents_release_control_plane or focused_pytest_gate_sources"` passed (`4 passed`, `73 deselected`).
- The release manifest focused-gate source-hash expectation now explicitly requires `tests/test_agents_release_control_plane.py`.
- `.venv/bin/python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py -k "focused_pytest_gate_source_hashes or current_suite_source_hash_files"` passed (`1 passed`, `386 deselected`).
- `tests/test_mimo_v2_no_source_exactness_classifier.py` is now wired into the current regression suite focused pytest gate and source-hash list, and the release manifest source-hash expectation.
- `.venv/bin/python -m pytest -q tests/test_mimo_v2_no_source_exactness_classifier.py tests/test_current_regression_suite.py tests/test_release_regression_manifest.py -k "mimo_v2_no_source_exactness_classifier or focused_pytest_gate_source_hashes or focused_pytest_gate_sources"` passed (`4 passed`, `385 deselected`).
- The release manifest MiMo root-cause validator now consumes `build/current-mimo-v2-no-source-exactness-classifier-after-audio-expanded-token-l2-restart-20260608.json`.
- `.venv/bin/python -m pytest -q tests/test_release_regression_manifest.py -k "mimo_v2_root_cause or current_mimo_v2_proof_artifact_constants or current_proof_sweep_tracks_mimo"` passed (`5 passed`, `308 deselected`).
- `.venv/bin/python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-after-mimo-audio-expanded-token-l2-restart-20260608.json` regenerated the manifest and correctly remained `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- `.venv/bin/python tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json` regenerated the checklist and correctly remained `status=open`, `failed_count=14`.
- The full checklist now consumes `build/current-mimo-v2-no-source-exactness-classifier-after-audio-expanded-token-l2-restart-20260608.json` in the MiMo group.
- `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py` passed (`2 passed`).
- `.venv/bin/python tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json` regenerated the checklist and correctly remained `status=open`, `failed_count=14`.
- `tests/cross_matrix/run_full_release_objective_checklist.py` default output and `tests/cross_matrix/run_current_regression_suite.py` full-checklist command now point at `build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json`.
- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py tests/test_current_regression_suite.py -k "agents_release_control_plane or full_release_objective_checklist"` passed (`5 passed`, `72 deselected`).

## Manifest/checklist sync update - 2026-06-07

The full checklist now consumes `build/current-release-regression-manifest-after-mimo-audio-expanded-token-l2-restart-20260608.json` as its release-manifest input.

Fresh generated checklist:

```text
build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json
```

Result remains `status=open`, `failed_count=14`.

Focused validation:

- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py tests/test_current_regression_suite.py tests/test_full_release_objective_checklist.py tests/test_release_regression_manifest.py -k "agents_release_control_plane or full_release_objective_checklist or mimo_v2_root_cause"` passed (`10 passed`, `382 deselected`).
- Compile validation passed for touched checklist/current-suite/manifest files.

Release boundary is unchanged: no model load, no source-vs-quant, no package build, no signing/notarization/tag/download update.

## Objective proof refresh - 2026-06-07

Refreshed no-heavy contract artifacts in place for current source hashes:

- `build/current-tool-call-contract-after-current-mimo-proof-20260607.json`
- `build/current-max-output-context-contract-after-current-mimo-proof-20260607.json`
- `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`
- `build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json`
- `build/current-cache-architecture-contract-after-mimo-tq-kv-boundary-20260607.json`
- `build/current-model-family-detection-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-parser-registry-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-model-artifact-format-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-generation-defaults-contract-after-do-sample-false-mimo-20260607.json`
- `build/current-native-mtp-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-vl-media-cache-contract-after-mimo-capability-snapshot-fix-20260607.json`

Fresh objective proof:

```text
build/current-objective-proof-after-mimo-audio-expanded-token-l2-restart-20260608.json
```

Current digest result is 15 PASS / 11 OPEN. The remaining open rows are:

- Cross-family live multi-turn smoke matrix is release-cleared.
- MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared.
- Real Electron UI unblocked non-MiMo live model matrix is proven.
- Real Electron UI cross-family live model matrix is release-cleared.
- DSV4 long-output/code/file-generation quality is release-cleared.

Focused validation passed:

- `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py tests/test_current_regression_suite.py tests/test_release_regression_manifest.py -k "objective_proof_digest or current_regression_suite_keeps_declared_known_blockers_open or source_hash_list_matches_release_manifest or release_manifest_pointer_matches_current_suite"` -> `108 passed`, `385 deselected`.
- Compile validation passed for objective/current-suite/release-manifest files.

## Startup parity guardrail refresh - 2026-06-07

- Updated local-only `AGENTS.md` to make CLI `vmlx serve` startup and MLXStudio generated startup independent release gates.
- A CLI proof does not clear UI-generated launch/settings/session behavior; a MLXStudio proof does not clear CLI/API startup, request override, `/health`, or capabilities reflection.
- Current no-heavy guardrail test passed: `tests/test_agents_release_control_plane.py` -> `3 passed`.
- Regenerated current manifest/checklist/objective artifacts after the source-hash change; release remains locked with `prepackage_ready=false`, `release_ready=false`, checklist `status=open`, `failed_count=14`, and objective proof 15 PASS / 11 OPEN.

## Generation defaults startup-parity gate refresh - 2026-06-07

- Added `panel_cli_startup_contract` to `tests/cross_matrix/run_generation_defaults_contract.py`.
- The generation-defaults contract now publishes `generation_defaults_family_matrix.cli_mlxstudio_startup_parity` with checks for CLI flag registration, MLXStudio preview/runtime parity, and startup-surface independence.
- Regenerated artifact: `build/current-generation-defaults-contract-after-do-sample-false-mimo-20260607.json`.
- Artifact result: `status=pass`; `panel_generation_defaults` 27 passed, `engine_generation_defaults` 60 passed, `local_generation_metadata_audit` 5 passed, `panel_cli_startup_contract` 9 passed; `missing_markers=[]`.
- Focused validation: `tests/test_generation_defaults_contract.py`, `tests/test_panel_cli_flag_contract.py`, current-suite/release-manifest/objective proof selectors -> `128 passed`, `368 deselected`.
- Regenerated current release manifest, full checklist, and objective proof. Release remains locked: `prepackage_ready=false`, `release_ready=false`, checklist `status=open`, `failed_count=14`, objective proof 15 PASS / 11 OPEN.

## Real-UI unblocked non-MiMo classifier refresh - 2026-06-07

- Fixed `tests/cross_matrix/release_regression_manifest.py` DSV4 real-UI memory preflight validation to accept the current canonical DSV4 path `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K` as well as the legacy HeadBF16 probe suffix.
- DSV4 real-UI memory preflight is now classified as a resource blocker, not a missing unblocked-family proof: status `pass`, launch decision `do_not_launch`, reason `insufficient_memory`.
- Added exact allowed variant handling for Qwen36 real-UI matrix: the family can pass with exactly `Qwen3.6-27B-JANG_4M-MTP` and `Qwen3.6-35B-A3B-MXFP8-MTP`; unexpected or missing variants still remain partial.
- Updated test fixtures for current Qwen36 and LFM25 20260607 proof filenames so synthetic matrix tests cover the same Responses/tools/cache/media surfaces as the current proof rows.
- Regenerated current manifest/checklist/objective artifacts. `Real Electron UI unblocked non-MiMo live model matrix is proven` is now PASS in objective proof.
- Current release state remains blocked: manifest `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; checklist `status=open`, `failed_count=14`.
- Remaining objective OPEN rows: cross-family live multi-turn smoke matrix, MiMo V2.5 runtime/tool/long-prompt quality, MiniMax reporter parity/root cause, Real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- Focused validation: real-UI matrix/current-suite/objective selectors -> `143 passed`, `350 deselected`; py_compile passed for edited Python files.

## Cross-family live smoke matrix stale ZAYA aggregation refresh - 2026-06-07

- Fixed `tests/cross_matrix/summarize_objective_proof.py` to use current filtered ZAYA text smoke proof `build/current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json` instead of the older combined ZAYA text+VL failure artifact.
- Current objective proof now classifies cross-family live smoke as non-MiMo green and MiMo-only deferred: `non_mimo_status=pass`, `non_mimo_missing_required_family_keys=[]`, `non_mimo_not_pass_artifacts=[]`, `missing_required_family_keys=[mimo_v2]`, `release_boundary=non_mimo_live_smoke_clear_mimo_v2_deferred`.
- This does not clear the cross-family smoke requirement because MiMo V2.5 remains a required family and still fails exact tool/JSON literal rows.
- Regenerated current objective proof, release manifest, and full checklist. Release remains locked: manifest `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; checklist `status=open`, `failed_count=14`.
- Focused validation: objective/current-suite/manifest selectors -> `108 passed`, `385 deselected`; py_compile passed for edited objective proof files.

## MiMo local JANG_2L/JANGTQ_2 metadata honesty contract - 2026-06-07

- Patched local `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L/config.json` so it no longer advertises unwired media runtime.
- Backup: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L/config.json.pre-text-runtime-metadata-20260607`.
- Patch proof: `build/current-mimo-v25-jang2l-local-config-text-runtime-patch-20260607.json`.
- Added no-heavy contract `tests/cross_matrix/run_mimo_v2_local_bundle_metadata_contract.py` covering both local MiMo bundles:
  - `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`
  - `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`
- Contract requires runtime modalities `['text']`, preserved/unwired modalities `['vision','audio']`, `multimodal_status='weights_preserved_text_runtime'`, `runtime.multimodal_mode='weights_preserved_text_runtime'`, and preserved `vision_config`, `audio_config`, `preprocessor_config.json`, and `audio_tokenizer/` sidecars.
- Generated proof: `build/current-mimo-v2-local-bundle-metadata-contract-20260607.json`, `status=pass`; `jangtq2` pass; `jang2l` pass.
- Wired the contract into `tests/cross_matrix/run_current_regression_suite.py` as `mimo_v2_local_bundle_metadata_contract`; the step passed inside the current suite.
- Focused validation passed: `tests/test_mimo_v2_local_bundle_metadata_contract.py` plus current-suite/release-manifest source-hash selectors -> `7 passed`, `383 deselected`.
- Full current suite was run and remains `status=open`. New MiMo metadata step passed, but failed steps remain `packaged_integrity_contracts`, `focused_regression_pytest`, `release_regression_manifest`, and `release_gate_skip_app`.
- Release boundary: this is metadata honesty only. MiMo VL/audio/video runtime remains unwired until real `mimo_v2_multimodal.py` or equivalent forward path, media embedding bridge, and media-aware cache/L2 proof exist.

## 2026-06-08 MiMo safe-headroom cache proof refresh

| Item | Artifact / proof | Status |
| --- | --- | --- |
| MiMo mixed-SWA explicit cap regression | `tests/test_mllm_scheduler_cache.py::TestMLLMMixedSWACleanStorePolicy::test_tight_memory_mixed_swa_skips_clean_prompt_above_configured_cap` | PASS after scheduler fix; explicit cap wins over safe-headroom unless force env is set. |
| Focused scheduler proof | `.venv/bin/python -m pytest -q tests/test_mllm_scheduler_cache.py -k 'tight_memory_mixed_swa_skips_clean_prompt_above_configured_cap or mixed_swa_tight_memory_store_uses_clean_prefill_when_headroom_is_safe or tight_memory_mixed_swa_force_env_overrides_cap'` | PASS; 3 passed. |
| Cache architecture contract | `build/current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json` | PASS; `cache_family_pytest` 426 passed, no failed/missing markers. |
| API/cache contract | `build/current-noheavy-api-cache-contract-after-dsv4-preflight-refresh-20260608.json` | PASS with current scheduler/test hashes. |
| VL/media cache contract | `build/current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json` | PASS with current scheduler/test hashes. |
| Objective proof digest | `build/current-objective-proof-after-mimo-safe-headroom-contract-refresh-20260608.json` | Cache architecture, generation defaults/Native MTP/VL media, current-source API/cache, and real UI unblocked non-MiMo rows are now PASS. Release remains OPEN on live/model-quality rows. |

Remaining release blockers: cross-family live multi-turn smoke, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity, real Electron UI full cross-family matrix, and DSV4 long-output/code/file-generation quality. No signing/notarization/tag/download update from this proof.

## 2026-06-09 MiMo/N2 runtime-cache-parser refresh

Scope: active Python/Electron worktree only. No deprecated `/Users/eric/vmlx`, ADLab, Max2, Swift, source-vs-quant, or heavy model launch.

Artifacts refreshed:

- `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json` - status `pass`.
- `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json` - status `pass`.
- `build/current-model-family-detection-contract-after-mimo-n2-runtime-refresh-20260609.json` - status `pass`.
- `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json` - decision `do_not_launch`; indexed payload `118.73GB`, free+speculative memory `94.26GiB`.
- `docs/internal/release-gates/20260608_190041/SUMMARY.md` - skip-app release gate: package/type/bundled import checks pass; objective digest remains fail.

Current classification:

- Built/plumbed: prefix/paged/block-L2 API contracts, hybrid SSM companion cache contracts, generic TurboQuant skip for hybrid SSM, MiMo asymmetric SWA cache status, model-family detection, panel session launch wiring, parser/reasoning CLI startup surfaces, bundled JANGTQ kernels, and MiMo registration.
- N2 JANGTQ2: narrow local proof remains green for text/tool/Responses/cache/small-image and the later clean repeat-image row is pass.
- N2 JANG_1L: not locally proven; blocked by memory preflight, not cleared.
- MiMo JANGTQ_2: runtime/cache/speed path is active but exact literal and tool-argument fidelity fails; not cleared.
- MiMo JANG_2L: not release-cleared; exactness/media/L2 restart rows remain open or missing.

Release boundary: do not sign, notarize, tag, or update downloads from this state. The blocker is no longer missing no-heavy plumbing; it is live model/output/resource/UI proof.

MiMo exactness refinement:

- Existing live proof separates into two blockers.
- No-cache deterministic rows already normalize or mutate literals (`BLUE-CAT` -> `BLUE CAT`, `CERULEAN-472` -> `CERULEAN472`), so this portion is not a detokenizer/parser/cache-only bug.
- The repeated cache row changes from `MIMO-CUTER-17` to `MIMO-CUTR-17` with `cached_tokens=31`, `cache_detail=paged`, and `temperature=0`; this is a live MiMo mixed-SWA paged-cache semantic-fidelity blocker.
- Do not clear MiMo cache from shape/unit tests alone. Next proof needs live no-cache vs paged-hit token trace or a cache-disabled A/B on the same exact prompt to isolate runtime cache from artifact/logit quality.

## 2026-06-09 MiMo lossless auto-cache policy

Change:

- `vmlx_engine/cli.py`: omitted `--kv-cache-quantization` now resolves MiMo V2 asymmetric mixed-SWA to stored-cache `none` instead of q4/q8. Prefix cache, paged cache, and block-disk L2 remain enabled; only lossy storage-boundary quantization is removed from the default path.
- `vmlx_engine/mllm_scheduler.py`: defensive guard disables auto q4/q8 stored-cache quantization for mixed-SWA VLM caches when callers bypass CLI detection.
- `tests/test_turboquant_cache_contract.py`: added MiMo regression while keeping Qwen3.5/N2 hybrid selective live TQ plus stored q4 behavior covered.

Proof:

- Focused pytest passed: MiMo auto lossless row, Qwen3.5 hybrid stored q4 row, and plain Qwen MoE auto-TQ row.
- `build/current-noheavy-api-cache-contract-after-mimo-lossless-auto-kv-20260609.json` - `pass`.
- `build/current-cache-architecture-contract-after-mimo-lossless-auto-kv-20260609.json` - `pass`.
- `build/current-model-family-detection-contract-after-mimo-lossless-auto-kv-20260609.json` - `pass`.
- Bundled Python rebuilt from current source and `panel/npm run verify-bundled` passed.
- `docs/internal/release-gates/20260608_190852/SUMMARY.md` - package/type/bundled import checks pass; objective digest remains fail.

Boundary:

- This fixes the unsafe default policy that allowed lossy q4/q8 stored cache on MiMo mixed-SWA without semantic parity proof.
- This does not yet clear MiMo release. A live repeat-cache exactness run under the new lossless default is still required.
- Explicit MiMo q4/q8 stored-cache quantization remains diagnostic-only until family-specific semantic parity is proven.

## 2026-06-09 MiMo lossless-cache live proof

Artifact:

- `build/current-mimo-v2-jangtq2-cb-cache-lossless-auto-live-20260609.json`

Result:

- Server loaded `MiMo-V2.5-JANGTQ_2` and exited cleanly after the probe.
- Native cache: `mixed_swa_kv_v1`, `cache_subtype=mimo_v2_asymmetric_swa`.
- Prefix cache, paged cache, and block-disk L2 were enabled.
- Stored-cache quantization was disabled: `storage_quantization.enabled=false`, `bits=null`.
- Repeated prompt hit paged cache: `cached_tokens=46`, `cache_detail=paged`.
- Block L2 wrote `78` tokens.
- Repeat output was stable across uncached and cached runs: both returned `ACKCB-742`.
- No hidden `<think>` tags and no 503/working-set rejection.
- Final health generation throughput reported about `58.8 tok/s` for the short run.

Classification:

- The previous MiMo repeat-cache drift under lossy q4/q8 storage is no longer the active default-path blocker under the new lossless policy.
- MiMo is still not release-cleared because literal exactness remains failed before and after cache: expected `ACK-CB-742`, actual `ACKCB-742`.
- Current audit pointer updated to this artifact; refreshed audit is `build/current-mimo-v2-jang2l-current-audit-after-lossless-cache-live-20260609.json`.

### 2026-06-09 MiMo JANGTQ2 token-trace exactness classification
- Proof: `build/current-mimo-v2-jangtq2-cb-cache-lossless-token-trace-live-20260609.json`.
- Server trace: `build/current-mimo-v2-jangtq2-cb-cache-lossless-token-trace-live-20260609.server.log`.
- Status: run passed for process stability, visible output, native cache telemetry, warm cache hit, and L2 write; exact literal row remains failed.
- Cache evidence: `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, prefix enabled, paged enabled, block-disk L2 enabled, storage quantization disabled, warm cache hit `cached_tokens=46` with `cache_detail=paged`, L2 `78` tokens.
- Token evidence: cold and warm trace generated `ACK`, `CB`, `-`, `7`, `4`, `2`, then stop. Visible output is still `ACKCB-742`; the hyphen before the digits is preserved, while the delimiter between `ACK` and `CB` was not generated.
- Classification: not a prefix/paged/L2/cache-quantization issue and not a parser rewrite. Cache/storage is cleared for this row, but the remaining exactness failure is at token choice / prompt-template / artifact behavior around the second token, not proven text assembly.

### 2026-06-09 MiMo no-source exactness classifier refresh
- Source updated: `tests/cross_matrix/run_mimo_v2_no_source_exactness_classifier.py` now accepts current list-shaped exactness probe artifacts and does not crash when a stale all-local smoke artifact is absent.
- Audit pointer updated: `tests/cross_matrix/run_mimo_v2_jang2l_current_audit.py` now uses `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`.
- Live exactness proof: `build/current-mimo-v25-jangtq2-exactness-variant-probe-live-after-lossless-token-trace-20260609/result.json`.
- Classifier proof: `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`, `status=open`, classification `jangtq2_plain_literal_copy_fails_before_parser_or_json_repair`.
- Current audit proof: `build/current-mimo-v2-jang2l-current-audit-after-lossless-token-trace-classifier-20260609.json`, `status=open`.
- Key failures: completions `blue-cat -> blue cat`, completions/chat `B7-CAT-09 -> B7 CAT-09`, JSON `B7-CAT-09 -> B7CAT-09`, tool args `B7-CAT-09 -> B7CAT-09`.
- Boundary: tool parser/protocol works in this run, but exact argument literals are wrong. This is not JSON repair, not raw XML fallback, not cache quantization, and not prefix/paged/L2 reuse. Source-vs-quant remains skipped by user RAM policy, so the remaining action is either runtime/logit-path root cause with no-source diagnostics or a model rebuild contract if artifact quality is confirmed as the cause.

### 2026-06-09 N2 no-heavy runtime/cache/parser status refresh
- API/cache proof: `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`.
- Cache architecture proof: `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`.
- Family detection proof: `build/current-model-family-detection-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`.
- N2 JANG_1L memory preflight: `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`, decision `do_not_launch`.
- Covered by no-heavy contracts: Chat/Responses sampling kwargs, max-output/max-context separation, JSON schema preservation, streaming cache-detail usage, Responses `previous_response_id`, cache stats/reuse endpoints, TurboQuant KV runtime contract, TurboQuant disk roundtrip, hybrid/native cache matrix, parser registration, CLI parser choices, panel launch policy, and JANG/JANGTQ/MXFP row distinctions.
- JANG_1L live boundary: payload is `118.73GB`; preflight host is `128GB`; free+speculative memory was `94.26GiB`; required extra headroom is `20GiB`. No live load was attempted, and this row remains release-open until tested with sufficient headroom or a smaller runtime strategy.

### 2026-06-09 N2 explicit autodetect/startup policy row
- Source coverage added: Python model registry, MLXStudio panel detector, and no-heavy family detection contract now include `n2_pro_qwen35_moe_hybrid_vl_policy`.
- Proof: `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`, `status=pass`, `missing_rows=[]`, `n2_pro_qwen35_moe_hybrid_vl_policy=true`.
- Policy pinned: N2 Pro JANG_1L-style metadata resolves as Qwen3.5-MoE hybrid cache with Qwen tool parser, Qwen3 reasoning parser, thinking support, and paged-cache requirement. Because this affine JANG VL-shaped bundle lacks indexed MTP/VL-ready tensors, Python routes it text-only and panel marks `forceTextOnly=true` until real VL support is implemented and live-proven.
- Boundary: this is autodetect/startup/UI-policy coverage only. N2 live runtime, N2 JANG_1L memory-safe load, VL/audio/video, L2 restart restore, installed-app UI proof, and release clearance remain open.

### 2026-06-09 gateway stale-port and standby wake routing contract
- Source fix: `panel/src/main/api-gateway.ts` now treats only active local sessions (`running`, `loading`, `standby`) as gateway-port conflicts. Stopped/error local sessions and remote sessions can keep stale DB ports after restart/sleep without blocking gateway startup.
- Test added: `panel/tests/api-gateway-single-model.behavior.test.ts` -> `allows gateway startup on ports used only by stopped or remote saved sessions`.
- Existing wake path covered: `auto-switches to a standby model by waking it before direct OpenAI streaming`.
- Release gate updated: `tests/cross_matrix/run_noheavy_api_cache_contract.py` now runs `panel_gateway_contracts` and records `gateway_stale_port_startup` plus `gateway_standby_wake_routing`.
- Proof: `build/current-noheavy-api-cache-contract-after-gateway-stale-port-20260609.json`, `status=pass`, `missing_markers=[]`, `gateway_stale_port_startup=true`, `gateway_standby_wake_routing=true`, `panel_gateway_contracts rc=0 passed=2`.
- Boundary: source/panel no-heavy proof only. Installed-app parity, live model routing, model cache/media rows, notarization, and public release remain blocked by the broader objective gates.
