# Cross-Model Runtime Issue Register and Proof Tracker

Date: 2026-06-05

Purpose: never lose track of model/runtime/config/VL/audio/tool/cache regressions, proof requirements, and release status. This is an itemized tracker. Do not use a broad green check to close a narrow row unless the row's exact proof exists.

Related narrative document: `docs/internal/CROSS_MODEL_RUNTIME_FAILURE_CLASSES_2026_06_05.md`

Current known release state:

- Gemma4 26B installed-app closure, 2026-06-06: `/Applications/vMLX.app` bundled Python proved Responses visible content at `build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260606.json` with `status=pass`, HTTP 200, 599 visible chars, 1550 reasoning chars, 451 completion tokens, mixed-SWA native cache active, and generic TurboQuant KV off. The installed-app mixed-SWA speed floor passed at `build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-20260606.json`: cold wall decode `90.619 tok/s`, cache-hit wall decode `104.426 tok/s`, internal generation about `107.8 tok/s`, `paged+mixed_swa` cached tokens on repeat, and `mixed_swa_kv_v1` with generic TQ-KV off. Objective digest `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json` now marks both Gemma4 26B CRACK rows `PASS`.
- Package/proof pointer refresh, 2026-06-06: `panel/bundled-python` was rebuilt from current source, `npm run verify-bundled` passed, and the staged Sequoia app under `panel/release/sequoia-app/mac-arm64/vMLX.app` was rebuilt with Developer ID signing after keychain repair. Canonical packaged integrity now passes at `build/current-packaged-integrity-contract-after-dsv4-smoke-refresh-20260606.json`. Release-gate/current-suite/public-audit defaults were refreshed away from stale 2026-06-04 objective digest artifacts and now target `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json`. Current suite `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json` is `status=open` with only `release_regression_manifest` failed and five remaining live/model blockers: cross-family live multi-turn smoke matrix, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality. No release tag, notarized DMG, public download update, or installed-app replacement has been performed from this continuation.
- Continuation VL/audio/video implementation pass, 2026-06-06: the media worklist now contains an explicit implementation ledger for each unbuilt function area. Current source already has a dynamic VLM image-prefill buffer resolver and typed 413 recovery path, so a user still seeing a fixed `8.0GB` default on a 128GB machine must be separated into stale installed app, explicit env override, or packaged-runtime drift before blaming model weights. MiMo remains the clearest unbuilt media runtime: `jang_tools.mimo_v2.mlx_model.py` is explicitly text-only and says visual/audio towers are preserved but not wired; vMLX therefore must not advertise MiMo VL/audio/video until JANG tools grows a real multimodal forward module and live proofs pass. Follow-up source change added typed `UnsupportedMediaModalityError` / API code `unsupported_media_modality` for unwired family media forward paths so MiMo vision failures are classified as unsupported runtime capability instead of generic server failure. Classifications: Gemma4 prefill guard follow-up is `gateway_ui` if stale app-only, MiMo media is `runtime_dispatch`, MiMo long-prompt/tool/speed remains `decode_loop` or `model_artifact` pending source-vs-quant proof.
- Continuation correction, 2026-06-06: current source heads are vMLX `a5bc1152` and JANG/JANGQ `9532380`. `panel/bundled-python` was refreshed from current local source and `npm run verify-bundled` passed, including critical `vmlx_engine` source hash parity, critical `jang_tools` source hash parity, MiMo import registration, Step3p7 VLM registration, Gemma4 unified registration, TurboQuant kernels, and Gemma4 vision list-coercion proof. A raw `npx electron-builder --mac` rebuild produced signed `release/vMLX-1.5.56-arm64.dmg` / zip artifacts, but electron-builder skipped notarization and the canonical release gate still checks `panel/release/sequoia-app/mac-arm64/vMLX.app`, which remains hash-stale in `build/current-packaged-integrity-contract-after-rebuilt-app-20260606.json`. Do not claim current public downloads, `/Applications/vMLX.app`, notarized DMG, or staged release app contain the post-`a5bc1152` state until `panel/scripts/build-release-dmgs.sh` can pass its prepackage manifest and a notarized/stapled release artifact is produced.
- Continuation MiMo correction, 2026-06-06: stale local MiMo state was cleaned by removing `/Users/eric/.cache/huggingface/modules/transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L`; `build/current-mimo-v2-jang2l-current-audit-after-stale-cleanup-20260606.json` now reports `stale_local_state_absent=true`, `manifest_integrity=true`, `text_cache_narrow=true`, `switchglu_selected_expert_parity=true`, and `cache_vs_nocache_next_token=true`. The row remains `status=open` because `long_prompt_coherence=false` and `tool_protocol=false`. MiMo V2.5 JANG_2L is not VL-ready, not audio/video-ready, and not `40+ tok/s` proven; Max2 docs for the promoted local bundle explicitly record short cached decode around `2 tok/s`, with VL/audio runtime deferred. The `MiMo-V2.5-JANGTQ_2-candidate` must not be treated as a quality fix until structural, text, tool, long-prompt, speed, and VL proofs pass.
- Continuation MiMo JANGTQ capacity note, 2026-06-06: `erics-m5-max2.local` has `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2-candidate` at `82G`, but the local machine had only `57GiB` free after cleanup. A local rsync attempt reached shard 50, then failed with `No space left on device`; the incomplete `55G` partial directory was deleted. Do not mark the candidate imported locally. Use Max2-side structural/runtime checks or free/attach at least another 30GiB before retrying local import.
- Continuation no-heavy contract refresh, 2026-06-06: `build/current-noheavy-api-cache-contract-after-bundled-refresh-continuation-20260606.json`, `build/current-cache-architecture-contract-after-bundled-refresh-continuation-20260606.json`, and `build/current-vl-media-cache-contract-after-bundled-refresh-continuation-20260606.json` all pass. This refresh covers API route/cache contracts, scheduler/native cache family classification, MLLM/TurboQuant cache contracts, DSV4 DSML tool contracts, Responses history, panel cache launch policy, and VL/media cache/settings/family detection. These are still no-heavy/static/panel-contract proofs; they do not close live multi-turn model, MiMo VL/speed, or notarized installed-app rows.
- Continuation MiMo live VL/tool proof, 2026-06-06: `build/current-all-local-model-smoke-mimo-v25-jang2l-media-tools-continuation-20260606/summary.json` is `status=fail` with `14` failures. The server loaded MiMo as MLLM at ~`106103MB` active memory and exposed native mixed full/SWA cache, paged prefix cache, block-disk L2, rotating metadata, and TurboQuant/storage quantization telemetry. Cache behavior partially worked: `text_cache_repeat_2` reported `cached_tokens=54`, `cache_detail=paged`, and cache stats showed `disk_hits=3`. Runtime quality remained blocked: first cache repeat was empty, second did not return exact `ACK`, tool-required request emitted no tool call, post-media text replies were empty, and image/video requests failed with `ValueError: MiMo-V2.5 JANG_2L vision input is not wired in this Python runtime.` Classification: `runtime_dispatch` + `decode_loop` + `model_artifact` pending source-vs-quant comparison; not a successful VL proof.
- Continuation MiMo capability-surface fix, 2026-06-06: current source now keeps MiMo V2 runtime modalities honest as `["text"]` until the real multimodal forward path is implemented. Focused proof: `tests/test_engine_audit.py -k 'mimo_v2_runtime_modalities_stay_text_only or mimo_v2_capabilities_do_not_advertise_unwired_vl or zaya_vl_runtime_modalities_do_not_infer_video_from_vision or qwen_vl_runtime_modalities_keep_explicit_video'` passed `4` selected tests. This is a fail-closed truth-surface fix only; it does not complete MiMo VL/audio/video.
- Continuation VL/audio/video implementation checklist, 2026-06-06: every model family must be tracked separately through `(1)` artifact sidecars/tokens/templates present, `(2)` processor/media expansion wired, `(3)` model forward accepts image/video/audio embeddings, `(4)` cache salt/reuse correct for media-expanded prompts, `(5)` post-media text recovery works, `(6)` streaming and non-streaming return visible content, `(7)` tool calls and multi-turn context survive media turns, `(8)` large-context cache hit/L2/TurboQuant behavior is proven, and `(9)` UI settings/API gateway expose only proven modalities. Current explicit missing functions/blockers: MiMo `vmlx_engine.models.mllm.Model.__call__` and `get_input_embeddings` reject `pixel_values`; `jang_tools.mimo_v2.mlx_model` is documented text-only and says visual/audio towers live in future `mimo_v2_multimodal.py`; MiMo audio/video forward path is absent; MiMo JANGTQ candidate is remote-only due local disk capacity; Step3p7 VLM has source registration/no-heavy guard proof but still needs live media proof; Gemma4 12B has narrow image/recovery proof but not full video/audio/quality clearance; Qwen VL/MTP, Nemo/Nemotron Omni, LFM, MiniMax, DSV4, and hybrid SSM families need per-family live multi-turn media/tool/cache proof before any release claim.
- Continuation proof-pointer refresh, 2026-06-06: after the MiMo modality truth-surface patch, all impacted no-heavy/static contracts were rerun and proof pointers were moved to current artifacts. Passing artifacts: `build/current-tool-call-contract-after-mimo-modality-truth-20260606.json`, `build/current-max-output-context-contract-after-mimo-modality-truth-20260606.json`, `build/current-cache-architecture-contract-after-mimo-modality-truth-20260606.json`, `build/current-noheavy-api-cache-contract-after-mimo-modality-truth-20260606.json`, `build/current-parser-registry-contract-after-mimo-modality-truth-20260606.json`, `build/current-model-artifact-format-contract-after-mimo-modality-truth-20260606.json`, `build/current-model-family-detection-contract-after-mimo-modality-truth-20260606.json`, `build/current-generation-defaults-contract-after-mimo-modality-truth-20260606.json`, `build/current-native-mtp-contract-after-mimo-modality-truth-20260606.json`, and `build/current-vl-media-cache-contract-after-bundled-refresh-continuation-20260606.json`. `build/current-objective-proof-after-final-pointer-refresh-20260606.json` now shows all no-heavy/static/API/cache/tool/defaults/parser rows as `PASS`; release remains blocked only by true live/model rows.
- vMLX 1.5.56 DMG hotfix shipped, signed, notarized, stapled, public updater/download current.
- `jjang-ai/vmlx` main after 1.5.56 includes structured JSON repair, DSV4 completions rail fix, MiMo JANG_2L load/MLLM adapter work, current Qwen35 dense MTP `gdn_sink` signature contract, DSV4 restart-L2 restore, Qwen27 JANG_4M installed-app speed/MTP proof-pointer refresh, and active MiMo release-blocker tracking.
- No post-1.5.56 tag/DMG/notarization/public release has been produced from these post-release source/main fixes; treat them as source/main plus local installed/staged proof only until a new notarized public package exists.
- PyPI is not current: PyPI latest remains `1.5.49`; `1.5.56` upload blocked by PyPI trusted-publisher/API-token config.
- Full cross-family runtime matrix remains open. Do not claim all model families production-cleared.
- Superseded current regression suite proof after MiMo current-audit tracking was `build/current-regression-suite-after-mimo-current-audit-20260606.json`: it remained open with seven live/model rows, including the two Gemma4 26B rows that are now closed by `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json`.
- MiniMax #117/#179 proof boundary: current root-cause audit is `open`, memory-preflight artifact exists and did not launch the huge model, and live Responses cancel/reporter parity proof is still absent. This must stay open; do not classify screenshot/output corruption as model artifact or runtime until reporter parity proof exists.
- DSV4 default-cache tool loop boundary: `build/current-dsv4-default-cache-tool-loop/result.json` was run live with native prefix+paged+block-disk L2 enabled and `status=review`. Runtime/tool/cache checks pass: DSML tools executed `list_directory -> write_file -> write_file`, final answer was `DONE`, cached tokens were seen with `paged+dsv4`, native cache was `native_composite`, and generic TurboQuant KV stayed off. The remaining review cause is generated code exactness (`THREE.ScScene()` and `THREE.BBoxGeometry()`), so this is tracked under DSV4 code/file-generation quality, not as a default-cache/tool-loop runtime failure.
- DSV4 rows now proven from current artifacts: app-launch default native prefix/paged/L2 wiring is proven by `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json` plus `build/current-dsv4-default-cache-tool-loop/result.json`; native SWA+CSA/HCA composite cache is proven; same-process TTFT/cache-hit is proven from `build/current-dsv4-responses-cache-gate-20260606.json`; one-tool stop and multi-tool default-cache loop are proven; restart-L2 block-disk restore is proven from `build/current-dsv4-responses-restart-l2-gate-20260606.json`. Still open: DSV4 exact code/file generation quality.
- DSV4 one-tool-after-result row is now proven from `build/current-dsv4-responses-one-tool-stop-20260606.json`: round 1 emitted exactly one structured `list_directory` call, round 2 used `previous_response_id`, kept `tools=TOOLS` with `tool_choice=auto`, emitted no function calls, and returned exactly `DONE` with native prefix+paged+block-disk L2 enabled.
- DSV4 restart-L2 row is now fixed in source and live-proven. Current artifact `build/current-dsv4-responses-restart-l2-gate-20260606.json` is `status=pass`: native cache/prefix/paged/L2 were enabled, generic TurboQuant KV stayed off, disk write before restart passed, restart L2 disk hit passed, `restart_dsv4_cache_hit=true`, same cache dir/fresh nonce/same terminal prompt/server restart checks passed, and visible output was `STORED`. Classification: fixed `kernel_cache` runtime issue, not model artifact corruption.
- Superseded packaged release signing/parity row from earlier 2026-06-06: fresh Developer ID signing was repaired and the older proof artifact `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json` passed for that slice. Current package status is now tracked by `build/current-packaged-integrity-contract-after-unsupported-media-staged-app-20260606.json`; use the current artifact above for release decisions.
- Superseded Post-MiMo/DSV4 packaging refresh from earlier 2026-06-06: the older package/proof gates used `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json`, `build/current-staged-app-runtime-parity-audit-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json`, and `build/current-regression-suite-after-dsv4-restart-l2-fix-20260606.json`. Current package status is now tracked by `build/current-packaged-integrity-contract-after-unsupported-media-staged-app-20260606.json` and `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json`. Do not claim the installed app or public download contains the latest continuation until a new app is installed/notarized and installed parity is rerun.
- MiMo is explicitly back in scope as of 2026-06-06. Eric requested: delete all past local MiMo model copies on this machine because they are bad; HTTP-download the MiMo JANG_2L artifact referenced in `erics-m5-max2.local:~/jang` docs; then implement/fix the new MiMo JANG_2L runtime path with real live proof. Do not reuse older local MiMo artifacts as evidence.
- Gemma4 12B JANG_4M installed-app VLM recovery has two narrow live proofs from 2026-06-06. `build/current-gemma4-12b-installed-image-prefill-recovery-20260606.json` proves a small image request succeeds and the next text turn returns `text recovery ok` with native Gemma4 mixed SWA KV, prefix/paged/block-disk cache, and q8 attention KV enabled. `build/current-gemma4-12b-installed-image-prefill-forced-reject-recovery-20260606.json` proves a deliberately tiny image-prefill budget rejects a long image prompt with HTTP 413 and the next text turn returns `text recovery after reject ok`. This is installed-app/runtime recovery evidence, not full Gemma4 VL quality clearance and not video/audio clearance.
- JANGQ/JANG tools remote and local checkout state as of 2026-06-06: `jjang-ai/jangq` `origin/main` and `/Users/eric/jang` local `main` are both at `d1316c3` (`Add MiMo V2 JANG runtime support`). The prior local equivalent head `821f639` is preserved on backup branch `codex-backup/local-main-before-origin-sync-20260605214651`. The three untracked local files that overlapped remote additions were moved to `/Users/eric/.codex/tmp/jang-untracked-overlap-backup-20260605214620/` before rebasing; unrelated untracked files in `/Users/eric/jang` remain untouched.
- Cross-model no-heavy routing/cache contracts refreshed after Step3p7/Qwen/MiMo subtype follow-up: `build/current-model-family-detection-contract-after-step37-refresh-20260606.json` `status=pass`; `build/current-vl-media-cache-contract-after-step37-refresh-20260606.json` `status=pass`; `build/current-noheavy-api-cache-contract-after-step37-refresh-20260606.json` `status=pass`. These cover engine/panel family detection, VL/media serialization/cache salting, panel VLM settings, API route contracts, scheduler cache contracts, MLLM cache contracts, DSV4 DSML tools, and Responses history. They are no-heavy/static/contract evidence, not live full-output media/model proof.
- Cross-model no-heavy runtime policy contracts refreshed after the Gemma4 VLM recovery proof: `build/current-vl-media-cache-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-noheavy-api-cache-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-cache-architecture-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-step37-crash-falsification-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-issue175-admin-sleep-probe-after-gemma4-vlm-recovery-20260606.json` `status=pass`. These prove static/no-heavy route contracts, panel/cache policy, Step3.7 crash-class guardrails, and admin sleep behavior, but they do not replace live full-output model-family proofs.
- MiMo V2.5 JANG_2L local artifact was refreshed in this run from `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` to `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`. The stale local `107G` copy was deleted first. Transfer used `rsync` over the direct `en9` route. Current structural proof `build/current-mimo-jang2l-local-structural-verify-20260606.json` is `status=pass`: 173 files, 113,926,313,468 bytes, 150 top-level safetensor shards, 109,180 indexed tensors, audio tokenizer present, and chat template matched embedded.
- MiMo V2.5 JANG_2L text/cache live proof is now narrow-pass on current source: `build/current-mimo-jang2l-live-text-cache-smoke-20260606.json` returned exact `cache ok` on the first Chat Completions request and exact `cache ok` on the repeat with `cached_tokens=28`, `cache_detail=paged`, q8 KV storage, paged prefix cache, and block-disk L2 enabled. This proves the asymmetric full/SWA cache path survives a same-process paged hit; it does not clear speed, tools, VL/audio/video, or restart-L2.
- MiMo V2.5 JANG_2L cache classification bug fixed after current audit: the registry already declared `cache_subtype="mimo_v2_asymmetric_swa"`, but scheduler/native-cache detection only recognized `mixed_swa_kv` and `step3p7_full_sliding_kv`. Source now treats MiMo as a mixed full/SWA KV cache family for LLM scheduler, MLLM scheduler, and native-cache health reporting. Focused contract proof: `tests/test_cache_bypass.py -k 'mimo_v2_asymmetric_swa or mixed_attention_helper'` passed `7` selected tests; `tests/test_engine_audit.py -k 'mimo_v2_registry_subtype or native_cache_status_reports_mimo_v2_asymmetric_swa or step37_registry_subtype_marks_scheduler_mixed_attention or native_cache_status_reports_step37_full_sliding_kv'` passed `4` selected tests.
- MiMo V2.5 JANG_2L tool behavior is still blocked. Classification: `decode_loop` / `model_artifact` pending source-vs-quant/template comparison, not a parser crash. Artifact `build/current-mimo-jang2l-live-tool-smoke-20260606.json` returned HTTP 200 with a structured OpenAI `tool_calls` item, but the generated arguments were wrong (`{"city": ": "}` instead of `{"city":"Paris"}`). Continuation artifact `build/current-mimo-jang2l-live-tool-continuation-smoke-20260606.json` stopped with no visible content after a tool result. This proves runtime/parser structure improved versus raw XML leakage, but MiMo tool-use quality is not release-cleared.
- Release gates now include MiMo as an active named blocker. Objective digest row `MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared` is `OPEN` from current local artifacts. The release ledger emits `mimo_v2_jang2l_runtime_quality_open` when the MiMo component is missing/open/failing. Current passing sub-evidence: structural verify, narrow text/cache proof, and selected-expert SwitchGLU parity. Current blocking sub-evidence: direct length sweep corrupts at longer prompts and tool dialect/protocol proof fails.
- MiMo V2.5 JANG_2L Responses API short text proof is only partial: `build/current-mimo-jang2l-live-responses-smoke-20260606.json` returned HTTP 200 and coherent `response ok`, but the prompt requested exact `responses ok`. Treat as API route alive, not exact instruction-following clearance.
- API/tool/reasoning/gateway policy contracts refreshed after the MiMo tool blocker: `build/current-tool-call-contract-after-mimo-tool-blocker-20260606.json` `status=pass`; `build/current-api-surface-contract-after-mimo-tool-blocker-20260606.json` `status=pass`; `build/current-reasoning-template-contract-after-mimo-tool-blocker-20260606.json` `status=pass`; `build/current-panel-tool-security-contract-after-mimo-tool-blocker-20260606.json` `status=pass`; `build/current-mcp-policy-contract-after-mimo-tool-blocker-20260606.json` `status=pass`; `build/current-release-surface-contract-after-mimo-tool-blocker-20260606.json` `status=pass`. Direct cancellation pytest also passed (`tests/test_cancellation.py`, 6 selected). Streaming remains open: `build/current-streaming-detokenizer-pytest-after-mimo-tool-blocker-20260606.json` is `status=skipped` with 13 skipped tests, so do not count it as streaming release proof.
- Source-level streaming API proof refreshed: `build/current-gemma4-12b-live-streaming-api-proof-20260606.json` is `status=pass` using Gemma4 12B JANG_4M on current source with continuous batching, paged prefix cache, block-disk L2, and q8 KV enabled. The `/v1/chat/completions` streaming request returned HTTP 200, three SSE chunks, visible text `stream ok`, and terminal `data: [DONE]`. This closes a current-source streaming/API row only; installed-app/notarized streaming parity still needs its own proof.
- Installed-app streaming API proof refreshed: `build/current-gemma4-12b-installed-app-streaming-api-proof-20260606.json` is `status=pass` using `/Applications/vMLX.app` bundled runtime with Gemma4 12B JANG_4M, continuous batching, paged prefix cache, block-disk L2, and q8 KV enabled. The streaming Chat Completions request returned HTTP 200, four content chunks forming `installed stream ok`, and terminal `data: [DONE]`. This proves the current installed app streaming path, but it does not package the post-1.5.56 MiMo cache fix.

## Status Legend

- `[ ]` Not started or no current proof.
- `[~]` Partial proof exists; scope is narrower than row.
- `[x]` Current proof exists for this exact row.
- `[!]` Known failure or blocker.
- `[D]` Deferred by explicit release exception; still open.

## Golden Rule

A model/runtime/config path is production-cleared only when the exact model family, quant/runtime, modality, API path, cache mode, streaming mode, and lifecycle path being claimed have current proof. Load-only, health-only, or one text smoke does not clear image/video/audio/tool/cache behavior.

## No Fake Fix / No Hidden Force Contract

Every row in this register must classify the root cause before it can move to `[x]`.

- Runtime/decode-loop/kernel/cache/parser incompatibilities must be fixed in the runtime path that fails, not hidden by forcing the feature off.
- Model artifact issues must be called out as model-side metadata/config/upload issues with the exact bad fields and the exact corrected artifact or metadata view.
- Fail-closed unsupported-route guards are allowed only when the user-visible result is an explicit unsupported-runtime rejection, post-error recovery is proven, and the real implementation row stays open.
- Disabling native MTP, prefix cache, paged cache, L2 disk cache, TurboQuant KV, VL, audio, video, thinking, or tool parsing does not count as a pass unless that disabled mode is the documented product behavior for that exact release row.
- Sampling/default/parser overrides are not fixes unless traced from `generation_config.json`, `jang_config.json`, tokenizer/chat-template metadata, model registry, UI request assembly, and server effective params.
- Postprocessing can repair structured output for downstream storage, but it does not prove native guided decoding or tool-call protocol correctness.
- A skipped proof, memory-gated proof, dry run, stale installed-app proof, or source-overlay proof must remain `[~]`, `[!]`, or `[D]`; it cannot close a packaged/notarized/installed release row.

Required classification labels for every new issue row:

- `model_artifact`: bad or overbroad model metadata, missing sidecar, corrupt upload, bad chat template, or wrong model-owned defaults.
- `runtime_dispatch`: wrong family/router/modality dispatch, unsafe MLLM/omni path, or unsupported advertised capability.
- `decode_loop`: stop-condition, thinking/template, tool-loop, streaming finalization, max-token, or visible-output failure.
- `kernel_cache`: Metal kernel, quantized matmul, TurboQuant/JANGTQ, MTP, KV/paged/L2/SWA/HSA/CSA cache, or memory-layout failure.
- `gateway_ui`: UI/settings/API gateway mismatch, stale installed app, stale public download/update manifest, or packaging/notarization drift.
- `unknown_pending_repro`: not classifiable yet; must stay open until a reproducible proof separates model artifact from runtime.

## Master Failure Classes

### CM-001 Unsupported advertised modality routes into unsafe runtime

Status: `[!]` Known concrete repro on Step3p7 CRACK. Must audit every family.

Symptom:

- Model metadata advertises vision/audio/video support.
- vMLX routes into MLLM/VLM/omni.
- Family-specific modality runtime is absent, incomplete, or unsafe.
- Server can crash or silently disconnect mid-request.

Concrete evidence:

- Step-3.7 Flash JANG_2L CRACK has `vision_config` and `jang_config.architecture.has_vision=true`.
- vMLX 1.5.55 classified it as `MLLM=True`.
- Unsupported MLLM path died after `MLLM.chat()` / chat template application.
- Text-only metadata view with `has_vision=false` loaded as `MLLM=False` and was stable.

Required checks:

- [ ] Print modality-detection source for every load: config, jang_config, registry, sidecar, CLI override.
- [ ] Assert unsupported modality routes fail closed before native forward.
- [ ] Require explicit override to enter known-unsupported modality path.
- [ ] Verify text-only workaround still works after guard.
- [ ] Verify error recovery: next text request works after rejected media/modality request.
- [ ] Add release-gate row for metadata-advertised unsupported modality.

Families to audit:

- [!] Step3p7: concrete failure exists; needs vMLX guard.
- [~] Gemma4 unified: 1.5.56 image hotfix verified for 12B JANG_4M/MXFP4/MXFP8, but audio/video full proof still open.
- [ ] Qwen VL/MTP: audit metadata modality vs runtime support.
- [ ] LFM: audit text and any VL-advertised variants.
- [ ] Nemotron Omni: audit audio/image/video dispatch and fallback.
- [D] DSV4: text/cache quality still deferred; audit media sidecars if present.
- [ ] Zaya/MiMo/Kimi VLM: audit advertised modality vs parser/runtime support.

Proof artifacts required per family:

```text
model_path
config modality fields
jang_config architecture fields
vMLX classification log
load mode: LLM or MLLM or omni
request route
one accepted supported modality request
one rejected unsupported modality request
post-error health
post-error text generation
server alive after all probes
```

### CM-002 Native crash without Python traceback

Status: `[!]` Known concrete repro on unsupported Step3p7 MLLM path.

Symptom:

- Client sees disconnected/no response or connection refused.
- Logs end at last Python stage.
- No Python traceback, so diagnosis is slow.

Required checks:

- [ ] Parent-process crash sentinel records last request id, model path, family, modality, cache mode, thinking mode, parser IDs.
- [ ] Last-stage breadcrumb before Metal/native forward.
- [ ] Crash reproducer uses safe prompt too, not only unsafe/adversarial prompt.
- [ ] Test unsupported route returns controlled 4xx/5xx instead of process death.

Architectures/runtime paths to probe:

- [!] Step3p7 unsupported MLLM.
- [~] Gemma4 image prefill: 1.5.56 now fails closed for oversized prefill and recovers.
- [ ] Video prefill/contact-sheet paths.
- [ ] Audio/omni dispatch.
- [ ] Qwen MTP/VL gated-delta paths.
- [ ] TurboQuant/JANGTQ kernels.
- [ ] DSV4 composite cache/state paths.

### CM-003 Tool-call dialect ambiguity

Status: `[!]` Known behavior failure on Step Flash CRACK eval. Cross-family open.

Symptom:

- Model receives API tool schemas.
- Sometimes emits structured `tool_calls`.
- Sometimes emits visible raw tool text like `<tool_call><function=...>`.
- Harness/client does not execute tool.

Concrete Step evidence:

- `A01`, `A17`, and several terminal-tool tasks emitted valid tool calls.
- `A20` emitted literal `<tool_call><function=write_file>...` text.
- `P11` emitted literal `<tool_call><function=terminal>...` text.

Required checks:

- [ ] For each parser family, run same tool prompt under Chat, Responses, Ollama, Anthropic adapter if applicable.
- [ ] Verify native API `tool_calls` object exists, not just visible raw tool text.
- [ ] Detect raw known dialect in visible text when tools were supplied.
- [ ] Convert raw dialect to tool_calls only when unambiguous and parser-supported, otherwise flag `raw_tool_dialect_leak`.
- [ ] Streaming path: verify streamed tool deltas and final structured tool calls.
- [ ] Multi-turn path: tool result is consumed and final answer is produced.

Parser families to audit:

- [!] Step3p5 / Step3p7 parser.
- [ ] Qwen parser.
- [ ] Gemma3/Gemma4 parser.
- [D] DSV4 DSML parser.
- [ ] Zaya XML parser.
- [ ] MiniMax parser.
- [ ] Nemotron parser.
- [ ] Hunyuan parser.
- [ ] Kimi parser.
- [ ] Generic fallback injection.

### CM-004 Tool loop after mock/tool output

Status: `[!]` Known behavior failure on Step Flash CRACK eval. Cross-family open.

Symptom:

- Model makes reasonable first diagnostic call.
- Receives enough tool output to answer.
- Repeats same/near-same command until turn budget exhausted.
- No final answer or weak synthesis.

Concrete Step tasks affected:

- `A14`
- `A16`
- `H09`
- `H12`
- `P09`

Required metrics:

- [ ] `tool_repeat_count`
- [ ] `same_tool_same_args_count`
- [ ] `near_duplicate_tool_call_count`
- [ ] `turn_budget_exhausted`
- [ ] `final_answer_after_tool_output`
- [ ] `used_tool_output_in_final_answer`
- [ ] `stopped_after_sufficient_observation`

Required checks:

- [ ] Repeated `id`, `whoami`, `top`, `systemctl`, `journalctl`, SSH, package install, file read loops.
- [ ] Mock permission error then final diagnostic.
- [ ] Mock OOM/service log then final diagnosis.
- [ ] File-read comparison then final synthesis.
- [ ] Tool-call budget pressure: model must stop, not continue blindly.

Families to audit:

- [!] Step3p7 CRACK text-only.
- [ ] Step3p5/Step3p7 non-CRACK.
- [ ] Qwen 27B/35B MTP and non-MTP.
- [ ] Gemma4 12B/27B/31B quants.
- [ ] LFM 8B/other.
- [D] DSV4.
- [ ] Nemotron Omni.
- [ ] Zaya/MiMo/Kimi.

### CM-005 Thinking/template mismatch

Status: `[!]` Known warning on Step Flash CRACK. Cross-family open.

Symptom:

- Runtime requests `enable_thinking=false`.
- Template still injects `<think>` or family-specific reasoning markers.
- Parser may hide leakage, but behavior and tool use can still drift.

Concrete Step evidence:

```text
Template for models/Step-3.7-Flash-JANG_2L-CRACK always injects <think> (ignores enable_thinking=False)
```

Required checks:

- [ ] Rendered prompt inspection under thinking off.
- [ ] Raw output inspection under thinking off.
- [ ] Visible output inspection after parser cleanup.
- [ ] Tool-call output under thinking off.
- [ ] Same prompt under thinking on/off; compare behavior.
- [ ] Verify defaults come from model/runtime metadata, not hidden agent knobs.

Families to audit:

- [!] Step3p5/Step3p7.
- [~] Gemma4: 1.5.56 set default visible-answer mode for Gemma4 unified; broader proof open.
- [ ] Qwen reasoning templates.
- [D] DSV4 thinking/direct/max rails.
- [ ] Zaya/MiMo XML thinking templates.
- [ ] GPT-OSS/MiniMax/Nemotron reasoning parsers.

### CM-006 Structured JSON/XML parse and repair reliability

Status: `[~]` Partial fix landed on main in `fa9f455b`; retry/guided decoding still open.

Problem:

- Models can understand task but emit malformed JSON/XML.
- Benchmark/database pipelines should not store broken structured output.

Concrete malformed JSON example:

```json
"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"
```

Fix landed:

- JSON repair in `parse_json_output` path.
- Handles fences, trailing commas, Python literals, prose, obvious missing closers, schema string-to-array coercion, adjacent string fragments.
- Conservative XML extraction helper added.
- Chat Completions route-level test proves canonical JSON output under `response_format=json_schema`.

Verified:

```text
.venv/bin/python -m pytest -q tests/test_structured_output.py
=> 38 passed, 2 skipped

.venv/bin/python -m pytest -q tests/test_server.py tests/test_api_models.py tests/test_ollama_adapter.py
=> 208 passed, 3 deselected

.venv/bin/python -m pytest -q tests/test_xml_function_tool_parser.py tests/test_tool_parsers.py -k 'xml or json or structured or schema'
=> 16 passed, 75 deselected
```

Still open:

- [ ] Retry model with "fix this JSON/XML only" when repair fails.
- [ ] Benchmark runner integration: score repaired object while tracking raw parse failure.
- [ ] Public docs: repair/validation vs hard constrained decoding.
- [ ] Investigate runtime-guided JSON/schema grammar support.
- [ ] Streaming structured output repair/canonicalization policy.

### CM-007 Cache/config/runtime interaction regressions

Status: `[~]` Some 1.5.56 hotfix paths verified; full matrix open.

Risk:

- A model works only with narrow flags, or cache/MTP/TurboQuant combinations break quality/stability.

Known relevant examples:

- Gemma4 12B JANG needed conservative runtime flags in earlier notes.
- Qwen35 MXFP8 MTP had reported packaged `gdn_sink` crash in an older app. Current main has source and freshly bundled-runtime compatibility proof: dense GatedDeltaNet, dense DecoderLayer, VLM GatedDeltaNet, VLM DecoderLayer, and VLM Model all accept `gdn_sink`. Full live packaged speed/equivalence proof remains open.
- Step Flash CRACK text-only launch used no continuous batching, no prefix cache, no KV quant, no native MTP.
- DSV4 cache/runtime has architecture-specific composite cache requirements; generic TurboQuant KV is not a drop-in substitute.
- DSV4 block-disk L2 restart restore currently fails closed for disk-backed terminal `DeepseekV4Cache` state. Disk writes and disk hits are observable, but the runtime does not yet safely execute cached DSV4 composite state after process restart.

Required matrix per model:

- [ ] continuous batching on/off.
- [ ] prefix cache on/off.
- [ ] paged cache on/off where applicable.
- [ ] L2 disk cache hit/miss/restore.
- [ ] KV quantization none/default/TurboQuant where applicable.
- [ ] native MTP off/on/depth variants.
- [ ] max output tokens low/high.
- [ ] top_k/top_p/temp defaults vs explicit.
- [ ] soft sleep and JIT wake.
- [ ] recovery after rejected request.

### CM-008 VL/image/video/audio request recovery

Status: `[~]` Gemma4 image prefill hotfix verified in 1.5.56; broader matrix open.

Risk:

- A failed media turn remains in UI/database history and poisons later text prompts.
- Oversized media prefill kills server or logs scary tracebacks.
- Unsupported video/audio route silently downgrades or crashes.

1.5.56 proof:

- Gemma4 12B JANG_4M image request visible output.
- Gemma4 MXFP4/MXFP8 image request visible output.
- Oversized image prefill rejects with clean 413 and no traceback.
- Fresh text request after 413 returns visible response.
- Panel rollback removes failed media user message when there was no visible activity.

Still open:

- [ ] Step3p7 VLM unsupported route guard.
- [ ] Step video processing proof.
- [ ] Qwen VL video/image proof with MTP variants.
- [ ] Nemotron Omni audio/image/video proof.
- [ ] LFM VL proof if applicable.
- [ ] UI settings panel media/cache/max-token combination proof.
- [ ] Streaming media request proof.

### CM-009 Public release surface drift

Status: `[~]` 1.5.56 public surface fixed; PyPI still stale.

Known issue:

- Public sites and updater feeds can stay old after GitHub release.

1.5.56 proof:

- `mlx.studio/update/latest.json` returns `1.5.56`, final hashes, no-store/DYNAMIC.
- `mlx.studio/download/` links to 1.5.56 Sequoia/Tahoe and contains no 1.5.55 refs.
- `vmlx.net/update/latest.json` redirects/returns 1.5.56.
- `vmlx.net/download/` returns 1.5.56 links.
- GitHub releases `jjang-ai/vmlx` and `jjang-ai/mlxstudio` have all four assets.

Still open:

- [!] PyPI latest is `1.5.49`; `1.5.56` absent.
- [!] Local twine upload got HTTP 403.
- [!] GitHub trusted publishing failed with `invalid-publisher` for both main and tag OIDC claims.
- [ ] Fix PyPI project trusted publisher or add valid `PYPI_API_TOKEN` secret.
- [ ] Re-run Publish PyPI workflow and verify `https://pypi.org/pypi/vmlx/1.5.56/json`.

### CM-010 Release-gate proof quality

Status: `[D]` Current umbrella manifest still red for broad rows.

Risk:

- Agents claim production-ready from partial checks.

Known current state:

- Packaged 1.5.56 installed-app gate passed for Gemma4 JANG_4M hotfix path.
- Umbrella release manifest still has deferred/open rows for DSV4 long-output/code exactness and full cross-family Electron UI matrix.

Required policy:

- [ ] Do not mark `production-cleared` from `/health` or load-only.
- [ ] Do not clear a model family from one model size/quant.
- [ ] Do not clear tool use from one first-turn tool call.
- [ ] Do not clear VL from text-only stability.
- [ ] Do not clear cache from cache disabled.
- [ ] Do not clear streaming from non-streaming.
- [ ] Do not clear app/UI from source server only.

## Per-Family Proof Matrix

### Gemma4

Current status: `[~]` 12B image hotfix path verified; full family open.

Proofs present:

- [x] 12B JANG_4M image/text/Responses/multi-turn/cache/recovery source proof.
- [x] 12B MXFP4 image source proof.
- [x] 12B MXFP8 image source proof.
- [x] Packaged installed 1.5.56 Gemma4 JANG_4M API/cache/sleep proof.
- [x] 2026-06-06 source speed gate for `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` passed: artifact `build/current-gemma4-12b-speed-gate-20260606-live.json`, target `45.0 tok/s`, default median `46.695 tok/s`, temp0/topk0 median `47.171 tok/s`, temp1/topk0 median `46.391 tok/s`, temp1/topk64 median `46.601 tok/s`, `top_k_is_primary_cause=false`, speed-row cache hits `0`, native cache `gemma4` / `mixed_swa_kv_v1`.
- [x] 2026-06-06 installed-app packaged speed gate passed through `/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12`: artifact `build/current-gemma4-12b-speed-gate-installed-app-20260606-live.json`, target `45.0 tok/s`, default median `46.665 tok/s`, temp0/topk0 median `47.006 tok/s`, temp1/topk0 median `45.81 tok/s`, temp1/topk64 median `41.934 tok/s`, `top_k_is_primary_cause=false`, speed-row cache hits `0`, native cache `gemma4` / `mixed_swa_kv_v1`. Nuance: topk64 median missed the 45 tok/s target in the installed-app row, but the configured top-k primary-regression threshold did not fire.

Still needed:

- [ ] 27B and 31B if shipped/supported.
- [ ] Video path.
- [ ] Audio path if advertised.
- [ ] Full UI settings panel proof.
- [ ] Structured JSON/schema live proof after repair.
- [ ] Tool-call proof.
- [ ] MTP variants if present.

### Step / Step3p7

Current status: `[~]` Text-only CRACK view is stable, Step3p7 VLM/source-runtime registration and crash-falsification contracts pass, but full live Step3.7 VLM media proof remains open.

Proofs present from review:

- [x] Step Flash CRACK text-only view stable under vMLX 1.5.55.
- [x] Fast eval completed 18/18, 0 failed requests.
- [x] Runtime crash reproduced in unsupported MLLM route.

Still needed:

- [x] Current source Step3p7 MLLM detection/source-runtime/crash-falsification contracts passed on 2026-06-06: `tests/test_step3p7_mllm_detection_guard.py`, `tests/test_step37_crash_falsification_contract.py`, and `tests/test_step37_vlm_runtime_audit.py` -> `14 passed`.
- [~] Reproduce on local vMLX HEAD: no-heavy guard/falsification contracts pass; full live Step3.7 media request proof remains open.
- [ ] Test Step 3.7 Flash JANG_2L non-CRACK and CRACK metadata variants.
- [ ] Tool dialect leak tests.
- [ ] Tool-loop eval rows.
- [ ] Thinking template render tests.
- [ ] Video processing/cache proof.

### Qwen

Current status: `[~]` Current named Qwen speed/MTP release rows are proven from installed-app/live artifacts; Qwen VL image/video proof and Qwen27 MXFP8 model-owned stochastic/PP review remain open.

Proofs present:

- [x] Qwen3.6 35B MXFP8 MTP source Chat/Responses/160-token decode no `gdn_sink` crash.
- [x] MTP accepted-token logging observed.
- [x] Fresh bundled Python probe reports `gdn_sink` accepted on dense GatedDeltaNet, dense DecoderLayer, VLM GatedDeltaNet, VLM DecoderLayer, and VLM Model.
- [x] 2026-06-06 focused source regression passed: `.venv/bin/python -m pytest -q tests/test_engine_audit.py -k qwen35_dense_mtp_patch_accepts_gdn_sink_kwarg` -> `1 passed, 503 deselected`.
- [x] 2026-06-06 local installed-app signature check: `/Applications/vMLX.app` reports `1.5.56`, and its bundled `vmlx_engine/patches/mlx_lm_mtp/qwen35_model.py` has `gdn_sink` in the patched dense GatedDeltaNet and DecoderLayer signatures.
- [x] 2026-06-06 real source-server Qwen35 MXFP8 MTP repro passed: artifact `build/current-qwen35-mxfp8-mtp-gdn-sink-live-qwen3parser-20260606.json`. Model `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP` loaded as `native_mtp_vl_artifact`, native MTP reported `READY D3`, short exact returned HTTP 200 with `qwen mtp ok`, 160-token decode returned HTTP 200, no `gdn_sink` crash, and server logged `77.4 tok/s` for the 160-token row.
- [x] 2026-06-06 real source-server Qwen27 MXFP8 MTP repro passed: artifact `build/current-qwen27-mxfp8-mtp-live-20260606.json`. Model `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP` loaded as `native_mtp_vl_artifact`, native MTP reported `READY D3`, short exact returned HTTP 200 with `qwen27 mtp ok`, 160-token decode returned HTTP 200, no `gdn_sink` crash, and server logged `18.0 tok/s` for the 160-token row. This clears only source no-crash/functionality, not speed.
- [x] 2026-06-06 installed-app packaged Qwen35 MXFP8 MTP repro passed: artifact `build/current-qwen35-mxfp8-mtp-installed-app-live-20260606.json`. Bundled Python `/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12` loaded the model as `native_mtp_vl_artifact`, native MTP reported `READY D3`, short exact returned HTTP 200 with `qwen35 installed mtp ok`, 160-token decode returned HTTP 200, no `gdn_sink` crash, and server logged `77.3 tok/s`.
- [x] 2026-06-06 installed-app packaged Qwen27 MXFP8 MTP repro passed for no-crash/functionality: artifact `build/current-qwen27-mxfp8-mtp-installed-app-live-20260606.json`. Bundled Python loaded the model as `native_mtp_vl_artifact`, native MTP reported `READY D3`, short exact returned HTTP 200 with `qwen27 installed mtp ok`, 160-token decode returned HTTP 200, no `gdn_sink` crash, and server logged `18.0 tok/s`. This still leaves Qwen27 speed/equivalence open.
- [!] 2026-06-06 formal installed-app decode-speed gate for Qwen27 MXFP8 MTP is `review`: artifact `build/current-decode-speed-live-qwen27-mxfp8-mtp-installed-app-20260606.json`. Bundle/model-owned stochastic sampling row (`generation_config`: `temperature=1.0`, `top_k=20`, `top_p=0.95`) decoded at `13.14 tok/s` vs expected `25.0`; greedy/topk0 decoded at `27.02 tok/s`; PP rows were below expected `600 tok/s` (`541.20`, `561.67`, `482.14`). Health showed native MTP active, affine quantized matmul active, M5 Max Metal N/A active, hybrid SSM cache active, and attention KV q4 storage active.
- [~] 2026-06-06 formal installed-app decode-speed gate for Qwen27 JANG_4M MTP is `review`: artifact `build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-20260606.json`. Bundle sampling decoded at `29.71 tok/s`, greedy/topk0 at `43.28 tok/s`, but one PP row was below expected (`574.34`). Same dense Qwen27 architecture, same native-MTP/cache route; this contrasts with the MXFP8-specific stochastic/default slowdown.
- [~] 2026-06-06 Qwen27 MXFP8 MTP deterministic native-MTP policy gate: artifact `build/current-decode-speed-live-qwen27-mxfp8-mtp-installed-app-deterministic-policy-20260606.json`, launched with `--native-mtp-sampling-policy deterministic-defaults`. Decode cleared the `25 tok/s` floor (`25.80 tok/s` bundle row, `25.75 tok/s` greedy row), but PP stayed below expected (`516.32`, `545.62`, `492.82`). Health recorded `request_policy=deterministic-defaults`. This means the default decode speed gap is tied to stochastic model-owned generation defaults/direct-CLI compatible-only behavior, not missing MTP or missing weight kernels. PP remains a runtime/performance gap.
- [x] 2026-06-06 no-heavy native-MTP contract passed: artifact `build/current-native-mtp-contract-qwen27-speed-boundary-20260606.json`; engine native-MTP contracts `123` passed, panel native-MTP controls `16` passed, panel native-MTP detection `8` passed. This verifies the app/session settings expose native-MTP controls and deterministic default policy wiring; direct CLI still requires the explicit policy flag unless model metadata changes.
- [x] 2026-06-06 Qwen27 JANG_4M source speed gate passed after deterministic PP harness fix: artifact `build/current-decode-speed-live-qwen27-jang4m-source-20260606.json`, `status=pass`, bundle decode `22.72 tok/s`, PP rows `816.95/903.98/767.56 tok/s`.
- [x] 2026-06-06 Qwen27 JANG_4M installed-app packaged speed gate passed after deterministic PP harness fix: artifact `build/current-decode-speed-live-qwen27-jang4m-installed-app-deterministic-pp-20260606.json`, `status=pass`, bundle decode `27.70 tok/s`, PP rows `800.46/907.23/795.09 tok/s`.
- [x] 2026-06-06 Qwen27 JANG_4M MTP installed-app deterministic PP/speed gate passed: artifact `build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-deterministic-pp-20260606.json`, launched with `--native-mtp-sampling-policy deterministic-defaults`, `status=pass`, bundle decode `45.68 tok/s`, PP rows `664.67/723.00/718.56 tok/s`.
- [x] 2026-06-06 Qwen27 JANG_4M installed-app native-MTP A/B equivalence passed: artifact `build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606/result.json`. Same artifact baseline no-MTP decode was `29.71 tok/s`; native MTP D3 decode was `50.65 tok/s`; speedup `1.70x`; `output_equivalence.all_content_equal=true`; `all_full_text_equal=true`; MTP acceptance rate `0.956`.
- [x] 2026-06-06 decode-speed PP harness fixed: `run_pp_case()` now uses deterministic `temperature=0`, `top_p=1`, `top_k=0`, matching the coherency/greedy rows. This prevents prompt-processing gates from measuring model-owned stochastic decode overhead instead of prefill throughput.
- [!] Reporter/user traceback `TypeError: _patch_gated_delta_net.<locals>.__call__() got an unexpected keyword argument 'gdn_sink'` is therefore not explained by current source or the local 1.5.56 bundle signature. If the reporter was on 1.5.55 or an older app, classify as stale packaged runtime. If it reproduces on a fresh 1.5.56+ app, open a separate live-model route bug because another Qwen patch path is bypassing the fixed signature.

Still needed:

- [!] Qwen27 MXFP8 MTP stochastic/default speed and PP proof. Packaged no-crash proof exists and deterministic policy clears decode, but model-owned stochastic defaults are slow and PP remains below target.
- [ ] Qwen VL image/video proof.
- [ ] Tool dialect and loop proof.
- [ ] Structured JSON repair benchmark proof.

### LFM

Current status: `[ ]` Needs current matrix.

Required:

- [ ] MXFP4 text.
- [ ] MXFP8 text.
- [ ] Any VL-advertised variants.
- [ ] Tool/JSON/schema.
- [ ] Cache/sleep/UI path.

### DSV4

Current status: `[D]` Runtime/cache exactness still deferred.

Known:

- [x] Main now includes DSV4 `/v1/completions` chat rail fix test restored in `fa9f455b`.
- [x] Native SWA/CSA/HSA composite cache verified in current live default-cache artifact; generic TurboQuant KV stayed off.
- [x] Responses same-process cache hit verified with `paged+dsv4`, cached-token accounting, and TTFT/wall-latency comparison.
- [x] Responses one-tool stop after tool result verified while tools remained available on the final turn.
- [~] Restart block-disk L2 is fail-closed, not release-cleared: disk-backed DSV4 terminal restore is rejected after live proof showed a Metal timeout on cached decode.
- [D] Long-output/code exactness remains open.
- [D] Full real UI DSV4 proof remains open.

Required:

- [x] Native SWA/CSA/HSA composite cache verification.
- [x] Same-process Responses cache hit/TTFT proof.
- [x] One-tool-after-result stop proof with tools still available.
- [ ] Safe DSV4 block-disk L2 restore after server restart with `paged+dsv4` usage detail.
- [ ] Long output full-tail read.
- [ ] Code/file-generation exactness.
- [~] Tool loops and DSML parser proof: current multi-tool runtime loop passed, exact code/file-generation quality remains open.
- [ ] Memory preflight on 128 GB/large model paths.

### Nemotron Omni / audio-video models

Current status: `[ ]` Needs matrix.

Required:

- [ ] Text-only smoke.
- [ ] Audio input.
- [ ] Audio output if supported.
- [ ] Image input.
- [ ] Video input.
- [ ] Unsupported modality failure closed.
- [ ] Cache/sleep/gateway proof.

### JANG / JANGTQ / MXFP cross-cutting

Current status: `[~]` Some hotfix paths verified; full matrix open.

Required:

- [ ] JANG_K non-TQ text/tool/cache proof.
- [ ] JANG_4M text/VL/tool proof.
- [ ] JANG_2L text/VL/tool proof.
- [ ] JANGTQ MoE codebook routing proof.
- [ ] MXFP4/MXFP8 dense and VL proof.
- [ ] Metadata mixed-precision/config consistency proof.
- [ ] TurboQuant kernel import and runtime proof.

### MiMo

Current status: `[~]` Re-entered active scope on 2026-06-06. Existing local MiMo models are considered bad unless replaced from the Max2-documented JANG_2L source. Local stale-artifact cleanup, Max2 intake, and structural verification are complete. Text/cache live proof is narrow-pass; tools, speed, VL/audio/video, restart-L2, and packaged/UI proof remain open.

2026-06-06 intake evidence:

- Removed stale local MiMo directory in this run: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` (`107G`).
- Previously removed stale HF MiMo cache directory: `/Users/eric/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro`.
- Max2 docs located the promoted bundle at `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Source used: `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` over `rsync` on the direct `en9` route.
- Downloaded local path: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Downloaded artifact count: `173` files, including `150` `model-*-of-00150.safetensors` shards.
- Local file bytes: `113926313468`; indexed payload: `113.25 GB`; weight map count: `109180`.
- Config facts: `model_type=mimo_v2`, `architectures=["MiMoV2ForCausalLM"]`, `num_hidden_layers=48`, `hidden_size=4096`, `num_attention_heads=64`, full-attention KV heads `4`, SWA KV heads `8`, `sliding_window=128`, `attention_value_scale=0.707`, `vision_config` present, `audio_config` present, `mtp_config` absent.
- Runtime metadata is embedded in `config.json`; this bundle does not include `jang_config.json`.
- `generation_config.json`: `do_sample=false`, `temperature=1.0`, `top_p=0.95`, `max_new_tokens=2048`, EOS `[151643,151645,151672]`.
- `tokenizer_config.json` embeds the same `chat_template.jinja`; tool/reasoning parser fields are not top-level tokenizer fields, so vMLX must use config/model registry metadata.
- Current local `/Users/eric/jang/jang-tools` MiMo verifier passed against the downloaded artifact: config OK, `109180` tensors across `150` shards, routed expert `.weight` count `36096`, BF16 passthrough embed/head/audio/vision checks OK, audio tokenizer present, chat template matches embedded.
- Packaged vMLX imports `jang_tools.mimo_v2.mlx_register`; after that registration, `mlx_lm.models.mimo_v2` imports and exposes `Model`/`ModelArgs`.
- Runtime gap found: generic `load_model_with_fallback` did not explicitly register MiMo before `mlx_lm.load`; this is a vMLX load-path integration issue, not evidence of download corruption.

2026-06-06 source-runtime live smoke:

- Artifacts: `build/current-mimo-jang2l-live-text-cache-smoke-20260606.json`, `build/current-mimo-jang2l-live-tool-smoke-20260606.json`, `build/current-mimo-jang2l-live-tool-continuation-smoke-20260606.json`, and `build/current-mimo-jang2l-live-responses-smoke-20260606.json`.
- Server command used source runtime with `--tool-call-parser xml_function`, `--reasoning-parser think_xml`, `--kv-cache-quantization q8`, paged prefix cache, and block-disk L2. This is cache route evidence, not release clearance.
- Source runtime live load now passes.
- Observed cache layout: `model_type=mimo_v2`, `48` layers, full attention layers use `KVCache`, SWA layers use `RotatingKVCache`; logs show full/SWA interleave ending at layer `47:KVCache`.
- Exact text/cache check passed: HTTP 200, output `cache ok`; repeated request returned `cache ok` with `cached_tokens=28`, `cache_detail=paged`.
- Tool call check is open: structured `tool_calls` emitted, but generated arguments were wrong and continuation after tool output had no visible content.
- Responses route is alive but not exact: output `response ok` for requested `responses ok`.
- No current proof in this pass clears MiMo multi-turn recall, streaming, tool continuation, audio, video, VL, restart-L2, installed-app parity, or release speed.

2026-06-06 MiMo tool-dialect follow-up:

- Artifact: `build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json`.
- Current fallback-injection behavior still fails tools: HTTP 200, content `<tool_call>`, `tool_calls=null`, finish `stop`, `3` completion tokens.
- Native-template/no-fallback experiment was explicitly tested and not shipped as a fix: fallback warnings disappeared, but tool request generated `128` tokens of garbled text and hit `finish_reason=length`.
- Native-template/no-fallback with `enable_thinking=true` also failed with garbled output and `finish_reason=length`.
- Plain-text XML control without API tools failed formatting: model emitted markdown-fenced XML, dropped the opening `<tool_call>`, and produced only `<function=get_weather>... </tool_call>`.
- `tool_choice=required` correctly returned HTTP 400 because no API `tool_calls` were produced. This enforcement is correct; it does not fix the model/tool dialect failure.
- Do not silently infer a tool call from a bare `<tool_call>` marker, do not force-disable MiMo tools and call it fixed, and do not mark MiMo tool support cleared from text/streaming success.
- Direct `mlx_lm` comparison reproduces the same failure class outside vMLX server scheduling: native MiMo tool prompt produced the same garbled `者...` sequence as the vMLX no-fallback experiment, and fallback-injected prompt produced only `<tool_call>`. This strongly indicates model/template/artifact behavior for tool turns unless another known-good MiMo profile proves complete XML tool output.
- Direct non-tool prompt-length sweep shows broader coherence failure: `60` prompt tokens produced `</think>length ok<think>` marker leakage, `93` prompt tokens produced `length ok`, but `148`, `214`, `280`, and `347` prompt tokens produced corrupt/gibberish output. Artifact: `build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json`.
- Sink A/B probe shows the prompt-length corruption is not isolated to native MLX `sinks=`: at `303` prompt tokens, native MLX `sinks=`, manual source-equivalent sink softmax, and sink-disabled variants all failed with punctuation/repetition/CJK output. Artifact: `build/current-mimo-v2-jang2l-sink-above-swa-probe-20260606.json`. Do not disable sinks as a fake fix.
- Manual sink fallback boolean-mask bug fixed in JANG tools after that probe: `_sdpa_with_sink` now converts boolean masks to additive `0/-inf` before adding logits. Focused JANG test `tests/test_mimo_v2_mlx_runtime.py` passed `3` tests. Real-model rerun artifact `build/current-mimo-v2-jang2l-sink-above-swa-probe-after-bool-mask-fix-20260606.json` still does not clear MiMo: native `sinks=` aborted with Metal GPU timeout, manual sink remained corrupt/repetitive, and sink-disabled remained corrupt punctuation.
- Similar cache-subtype coverage added after the MiMo miss: no-heavy contracts now explicitly cover `mimo_v2_asymmetric_swa`, `nemotron_h_ssm_attention`, and `lfm2_moe_hybrid_ssm` native-cache reporting. Full cache architecture contract artifact `build/current-cache-architecture-contract-after-hybrid-subtype-coverage-20260606.json` is `status=pass` with `failed=[]`, `missing_markers=[]`, `missing_api_checks=[]`, `missing_api_command_markers=[]`, and `missing_panel_markers=[]`.
- Max2 docs already frame this boundary: short MiMo JANG_2L smokes can look coherent, but no-cache/cached deeper prompts showed gibberish/layer drift and open proof targets. Current local evidence matches that; do not classify MiMo as production-cleared from short text smokes.

Required cleanup/intake:

- [x] Inventory local MiMo model directories on this machine.
- [x] Delete all past local MiMo model copies after recording paths removed.
- [x] SSH or otherwise access `erics-m5-max2.local:~/jang` docs and locate the HTTP download source for the MiMo JANG_2L artifact.
- [x] Download the documented MiMo JANG_2L artifact over HTTP, not by silently reusing stale local copies.
- [x] Verify artifact integrity: config, tokenizer/chat template, sidecars, quant metadata, shard count, and expected JANG_2L precision. `jang_config.json` is absent by design for this bundle path; Max2 current verifier passed locally from a temp copy.
- [ ] Sync local JANG MiMo verifier/docs from Max2 or otherwise make local verifier respect BF16 bookends and `routed_expert_bit_plan.layer_overrides`.

Required runtime work:

- [x] Implement/fix MiMo JANG_2L model-family detection without directory-name regex. Registry/classifier and live source runtime detect `model_type=mimo_v2`.
- [~] Route MiMo JANG_2L through the correct loader/runtime, not generic fallback if architecture-specific code is required. Source MLLM text path loads and answers after sidecar-aware quantization fix; packaged parity still open.
- [x] Route `mimo_v2_asymmetric_swa` into the same mixed full/SWA KV scheduler/native-cache contract used by Gemma/Step mixed-SWA families. This fixes the engine classification bug; it does not clear MiMo long-prompt/tool quality by itself.
- [~] Verify cache policy: prefix, paged, L2 disk, and any hybrid/SSM/architecture-specific state handling. Source logs prove hybrid `KVCache`/`RotatingKVCache` layout only; cache hits, restart, and L2 disk are still open.
- [ ] Verify TurboQuant/JANG kernels or explicitly classify unsupported kernel paths.
- [ ] Verify thinking/template/parser behavior from model-owned metadata.
- [!] Verify prompt-length coherence. Direct `mlx_lm` sweep passes around `93` prompt tokens but corrupts by `148` prompt tokens and above; this is a release blocker independent of tools.
- [!] Verify tool-call protocol and loop behavior. Current tool probes fail across fallback, native-template, thinking-enabled, plain XML control, and `tool_choice=required`; no API `tool_calls` produced.
- [ ] Verify Chat Completions, Responses, Anthropic, and Ollama surfaces where supported.
- [ ] Verify streaming and non-streaming full visible outputs, including tail review.
- [ ] Verify sleep/wake/unload/reload lifecycle.
- [ ] Verify panel settings reflection and launch flags.

Proof required before closing:

```text
removed_bad_local_paths
download_url
downloaded_artifact_path
artifact_hash_or_shard_manifest
config/jang_config/tokenizer/template summary
vMLX load classification
live server commands
API surface results
cache/scheduler telemetry
tool/multi-turn outputs
UI/settings evidence
post-error recovery
release-gate artifact path
```

## Release Notes / Reporter Credit

- [!] Future release notes and public acknowledgements must credit GitHub `@Hornsan1` for reporting many of the recent runtime/model/UI/API issues covered by this register. This is a release-process requirement from 2026-06-06 onward; do not ship the next release notes without this credit.

## Release-Surface Blockers

### CM-REL-001 Fresh Developer ID signing blocked while stale app verifies

Status: `[x]` Repaired on 2026-06-06 for the staged Sequoia app path. Keep as a regression row.

Classification: `gateway_ui`.

Symptom:

- Existing staged `vMLX.app` can pass signature verification.
- Fresh signing of a copied binary with the configured Developer ID identity fails with `errSecInternalComponent`.
- Keychain inspection also reports non-interactive user-access denial.
- A release gate that accepts the stale signed app as proof can falsely hide that the current source/bundled runtime cannot be rebuilt, signed, notarized, or shipped.

Concrete evidence:

- Direct signing probe against a copied bundled `.so` failed with `errSecInternalComponent`.
- `security find-identity -v -p codesigning` sees Developer ID identities, including `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`.
- `security show-keychain-info` for signing keychains reported `User interaction is not allowed`.
- `npm run build && npm run package` failed during Electron signing on bundled scipy extension signing, consistent with signer/keychain access rather than a specific model/runtime file.
- Packaged integrity now reports release blocker `packaged_app_developer_id_signing_blocked` instead of treating the old staged app as sufficient proof.
- Repair evidence: unlocking `vmlx-build.keychain-db` and restoring the codesign partition list made fresh Developer ID signing pass on a copied bundled scipy `.so`.
- Repair evidence: `npx electron-builder --mac --dir --config.directories.output=release/sequoia-app` rebuilt and signed `panel/release/sequoia-app/mac-arm64/vMLX.app`.
- Earlier repair evidence: `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json` is `status=pass`; `package_signing_preflight.status=pass`; `staged_app_engine_hash_parity=true`; `staged_app_engine_source_hash_parity=true`. Current package evidence for release decisions is `build/current-packaged-integrity-contract-after-unsupported-media-staged-app-20260606.json`.

Required checks before closing:

- [x] Fresh Developer ID signing probe passes on a copied binary in the current non-interactive release environment.
- [x] `npx electron-builder --mac --dir --config.directories.output=release/sequoia-app` produces a new staged `vMLX.app` from current source and bundled Python.
- [x] New staged app passes strict Developer ID signature verification.
- [x] Staged app engine hash parity and engine source hash parity pass against current source.
- [ ] Notarization submit, wait, staple, and `stapler validate` pass for the newly built artifact.
- [ ] Public updater/download manifests point to the newly notarized artifact only after the above checks pass.

Do not close with:

- Existing old `panel/release/.../vMLX.app` signature verification alone.
- Ad-hoc signing.
- Unsigned app copying.
- Disabling hardened runtime.
- Skipping the signing preflight.
- Treating source-server tests as packaged-app proof.

## Required Evidence Template Per Run

## MiMo V2.5 JANG_2L Follow-up Evidence - 2026-06-06

Current local bundle:

- `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`
- `model_type=mimo_v2`
- `attention_projection_layout=fused_qkv`
- `hidden_size=4096`, `layers=48`, `full/SWA pattern=9/39`
- full attention KV heads `4`, SWA KV heads `8`
- q/k head dim `192`, v head dim `128`
- SWA window `128`, SWA sink bias present
- `attention_value_scale=0.707`
- `routed_expert_bits={gate_proj:3, up_proj:2, down_proj:2}`
- `routed_expert_group_size=128`
- `jang_config.json` is absent; runtime facts are currently embedded in `config.json`

New evidence:

- Direct `mlx_lm` with `JANG_MIMO_DISABLE_SINK=1` did not fix coherence. All tested prompt lengths produced punctuation-heavy corrupt output. Artifact: `build/current-mimo-v2-jang2l-direct-length-sweep-sinkoff-20260606.json`.
- Direct `mlx_lm` next-token prefill with `cache=None` and with `model.make_cache()` produced identical top-10 logits for a 125-token prompt. Artifact: `build/current-mimo-v2-jang2l-cache-vs-nocache-next-token-20260606.json`.
- This rules out the simple "prefill cache changes logits" explanation for that prompt. It does not rule out deeper forward/runtime math or quantization quality.
- RoPE convention check matched the bundled PyTorch reference: MLX `mx.fast.rope(..., traditional=False)` matches source-style `rotate_half` within float tolerance.
- `SwitchGLU` activation order matches the source MoE formula: MLX calls `silu(gate) * up`.
- Quantized `SwitchGLU` selected-expert parity passed against manual selected-expert dequantized math for layer 1. Max absolute diff was `0.0007556`, mean absolute diff was `0.0000971`. Artifact: `build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json`.
- Local JANG runtime patch changed SWA sink attention to use MLX native `scaled_dot_product_attention(..., sinks=attention_sink_bias)` by default, matching the Max2 dirty runtime direction while retaining manual sink only behind explicit `JANG_MIMO_MANUAL_SINK_SDPA`.
- Native `sinks=` improved short/medium failure shape from punctuation/CJK to English prompt-copying for 92/125-token prompts, but it did not clear coherence. At 152+ prompt tokens output still degraded into CJK/punctuation. Artifact: `build/current-mimo-v2-jang2l-direct-length-sweep-native-sinks-20260606.json`.
- vMLX live MLLM route initially failed with HTTP 500 because `mlx_vlm.generate` called MiMo's language model with `inputs_embeds=...`, but MiMo's MLX `Model.__call__` accepted only raw `input_ids`.
- vMLX source fix now adapts the dynamically registered MiMo MLLM `language_model` so `mlx_vlm` can pass `inputs_embeds` and receive an object with `.logits`. Direct raw-logits calls without `inputs_embeds` are preserved through the adapter.
- Current vMLX source adapter proof returned HTTP 200 and exact short output `mimo runtime ok`.
- The same live source server still failed long-output quality: a 152-token prompt returned visible `<think><think><think>`. Artifact: `build/current-mimo-v2-jang2l-vmlx-source-adapter-smoke2-20260606.json`; log: `build/current-mimo-v2-jang2l-vmlx-source-adapter-smoke2-20260606.log`.
- Focused vMLX regression coverage for MiMo registration, load-side sanitization/quantization, and both old/new MLLM `inputs_embeds`/`.logits` contracts passed: `6 passed, 498 deselected`.

Max2 contrast evidence:

- `erics-m5-max2.local` has a separate MiMo TP4 Swift/JACCL proof under `~/adlab/docs/mimo-v25-tp4-live-proof.md`.
- That proof is not the local Python `JANG_2L` bundle. It uses Swift `TPRankWorker`, TP4 source shards, `TP_QUANTIZE_EXPERTS=1`, `TP_MIMO_ROUTED_EXPERT_BITS=4`, group size `64`, cache coordinator, and L2 disk cache.
- Max2 proof reports chat, multi-turn, Responses, streaming, cache reuse, L2 disk restore, rank agreement, and `39.2284 tok/s` decode throughput passing.
- Therefore "MiMo architecture is inherently impossible" is false. The local blocker is specific to the Python/MLX local `JANG_2L` runtime/profile/artifact path.

Current classification:

- `runtime/server parser only`: unlikely. Direct `mlx_lm` reproduces prompt-length and tool failures outside vMLX server.
- `MLLM call-shape incompatibility`: confirmed and fixed in vMLX's dynamic MiMo MLLM adapter; this fixes HTTP 500 for short text on the MLLM route, not long-output quality.
- `simple cache/prefix bug`: unlikely for the tested prefill row. Cache and no-cache logits matched.
- `SWA sink-only bug`: unlikely. Sink-off made output worse, not better.
- `RoPE convention bug`: unlikely. Numeric convention check matched reference.
- `MoE activation-order bug`: unlikely. `SwitchGLU` order matches reference.
- `selected-expert quantized SwitchGLU kernel bug`: unlikely for the tested layer and selected experts; manual dequant parity passed.
- `quantized routed-expert quality, low-bit profile, or deeper full-forward mismatch`: still plausible and now the leading local hypothesis.
- `model upload corrupt`: not proven. Structural verifier passes, but quality evidence is red; compare against source or a higher-quality MiMo profile before public claims.

Required next checks:

- [x] Run a valid quantized `SwitchGLU` parity check against manual dequantized selected-expert math for actual MiMo layer weights.
- [x] Fix local MiMo MLX language-model call contract for vMLX MLLM `inputs_embeds` route and prove short source-server text no longer 500s.
- [ ] Compare local Python `JANG_2L` against a higher-quality local/Max2 MiMo profile if disk allows, especially 4-bit routed-expert or source-shard path.
- [ ] Add a source-vs-quant first-divergence probe for the first MoE layer that can run without loading the full 294 GB source into local memory.
- [ ] Keep MiMo out of release-clear claims until long prompt, tools, cache, and API rows pass through the actual vMLX source/packaged runtime.

## Adjacent MLLM Language-Model Interface Checks - 2026-06-06

Status: `[~]` Focused source contracts passed for Step/Zaya/MiMo call-shape compatibility; live packaged model-family proof remains open.

Proof:

- Step3.7 focused source test passed: `.venv/bin/python -m pytest -q tests/test_step37_mlx_vlm_runtime.py::test_step37_language_model_returns_logits_object_for_mlx_vlm_generate` -> passed.
- Zaya focused source test passed: `.venv/bin/python -m pytest -q tests/test_zaya_runtime.py::test_zaya1_vl_language_model_accepts_mlx_vlm_inputs_embeds_keyword` -> passed.
- MiMo focused source contracts passed earlier in the current pass: `6 passed, 498 deselected` for registration, load-side quantization/sanitization, and MLLM `inputs_embeds`/`.logits` adapter behavior.

Release interpretation:

- These are no-heavy source interface contracts only. They prevent obvious `mlx_vlm.generate` call-shape regressions, but do not clear media quality, long-output decode quality, cache behavior, packaged-app parity, or speed.
- If a family advertises MLLM/VL, it still needs live supported-media and unsupported-media recovery rows before release claims.

Do not close with:

- Short exact text smoke only.
- Disabling sink.
- Forcing parser fallback.
- Claiming Max2 TP4 Swift proof clears local Python `JANG_2L`.
- Re-uploading the same local `JANG_2L` as fixed without forward-quality proof.

Every model-family proof should record:

```json
{
  "date": "YYYY-MM-DD",
  "vmlx_version": "",
  "commit": "",
  "app_or_source": "source|packaged-app",
  "model_path": "",
  "model_family": "",
  "quant_runtime": "",
  "metadata": {
    "config_model_type": "",
    "architectures": [],
    "has_vision_config": false,
    "has_audio_config": false,
    "jang_has_vision": null,
    "jang_has_audio": null,
    "tool_parser": "",
    "reasoning_parser": "",
    "supports_thinking": null,
    "think_in_template": null
  },
  "route": {
    "api": "chat|responses|completions|anthropic|ollama|ui",
    "stream": false,
    "modality": "text|image|video|audio|mixed",
    "cache_mode": "",
    "mtp": "off|on|depth=N"
  },
  "checks": {
    "load_classification_logged": false,
    "visible_text_ok": false,
    "structured_json_or_xml_ok": false,
    "tool_calls_structured_not_raw_text": false,
    "multi_turn_recall_ok": false,
    "cache_hit_or_disabled_proven": false,
    "unsupported_modality_fails_closed": false,
    "post_error_recovery_ok": false,
    "soft_sleep_wake_ok": false,
    "server_survived": false
  },
  "metrics": {
    "eval_tok_s": null,
    "latency_p50_s": null,
    "latency_p95_s": null,
    "latency_p99_s": null,
    "tool_repeat_count": null,
    "same_tool_same_args_count": null,
    "turn_budget_exhausted": null
  },
  "artifacts": {
    "server_log": "",
    "summary_json": "",
    "release_gate_summary": ""
  },
  "verdict": "pass|partial|fail|blocked"
}
```

## Immediate Next Work Queue

1. `[!]` Build vMLX guard for Step3p7 advertised vision when Step3p7 VLM runtime is unsupported.
2. `[!]` Add release-gate metadata/runtime modality mismatch audit.
3. `[!]` Add raw tool dialect leak detector or parser repair for known XML-ish tool text.
4. `[!]` Add loop-control eval metrics and rows.
5. `[~]` Finish structured output retry/guided decoding work after `fa9f455b` repair layer.
6. `[!]` Fix PyPI trusted publisher or token so Python package release is not stale.
7. `[D]` Re-run DSV4 long-output/code exactness and real UI proof when memory/model state allows.
8. `[~]` Complete DMG notarization/stapling/public manifest work after live/model objective rows are green; staged Sequoia app signing/parity is fixed, notarized DMG is not.
9. `[!]` MiMo cleanup/intake/runtime: delete bad local MiMo models, download documented MiMo JANG_2L from Max2 docs, then implement/fix and live-prove MiMo JANG_2L runtime.
10. `[ ]` Execute per-family matrix for Gemma4, Qwen, LFM, Step, DSV4, Nemotron, Zaya/MiMo/Kimi, JANG/JANGTQ/MXFP.

## Non-Negotiable Release Notes

- Do not say "all good" for a model family until this register has a proof row for that family and route.
- Do not say "VL works" if only text-only worked.
- Do not say "tools work" if raw tool text leaked in any path without a parser/repair/flag.
- Do not say "cache works" if the successful run disabled cache.
- Do not say "packaged app works" from source-server proof.
- Do not say "PyPI released" until PyPI has the exact version JSON.

## GitHub Tracking Issues

- Master cross-model matrix: https://github.com/jjang-ai/vmlx/issues/188
- Unsupported advertised modality/runtime guard: https://github.com/jjang-ai/vmlx/issues/189
- Raw tool dialect leaks, tool loops, thinking-template mismatch: https://github.com/jjang-ai/vmlx/issues/190
- Structured JSON/XML repair follow-up: https://github.com/jjang-ai/vmlx/issues/187

- MiMo current audit `build/current-mimo-v2-jang2l-current-audit-20260606.json` proves the canonical local bundle matches the TB5/HTTP manifest (`173/173` files, 113.93GB payload by manifest, zero missing/mismatch) and stale MiMo backup/cache directories were removed. It does not release-clear MiMo: long-prompt coherence and API tool protocol remain blocked without fake parser injection, forced fallback, or cache disabling.

## 2026-06-06 VL/audio/video runtime worklist

Durable media-runtime worklist added:

- `docs/internal/VL_AUDIO_VIDEO_RUNTIME_WORKLIST_2026_06_06.md`

This is the active checklist for the remaining VL/audio/video release blockers. It lists the required runtime functions, per-family blocker state, missing implementation areas, and live proof rows needed before a public release/tag/notarized build can be claimed.

## 2026-06-06 live smoke continuation: Gemma26, MiniMax K, MiMo

Fresh source-server smoke:

- Combined source run: `build/current-all-local-model-smoke-gemma26-minimaxk-tools-media-continuation-20260606/summary.json`
- Split Gemma26 proof: `build/current-all-local-model-smoke-gemma26-jang4m-tools-media-continuation-20260606/summary.json`
- Split MiniMax K proof: `build/current-all-local-model-smoke-minimaxk-tools-continuation-20260606/summary.json`

Results:

- Gemma4 26B JANG_4M CRACK source smoke passed: exact `ACK`, paged mixed-SWA cache hit, multi-turn recall, reasoning-on final answer, required OpenAI tool call, image color, video color, and text-after-media recovery all returned HTTP 200. Health reported native `gemma4/mixed_swa_kv_v1`, full/sliding KV, rotating metadata, and storage quantization.
- MiniMax-M2.7-JANGTQ_K CRACK source smoke passed: exact `ACK`, paged cache hit with `cache_detail=paged+tq`, generic TurboQuant KV enabled for plain attention KV, block-disk L2 present, multi-turn recall, reasoning-on final answer, and required OpenAI `record_fact` tool call all returned HTTP 200.
- These source smokes update cross-family live evidence, but they do not close Gemma4 26B installed-app Responses visible-content, Gemma4 26B installed mixed-SWA speed floor, or MiniMax K reporter parity/root-cause rows.

Fresh MiMo rerun:

- Optimized rerun: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-media-rerun-20260606/summary.json`
- Conservative diagnostic: `build/current-mimo-conservative-diagnostic-20260606/summary.json`

MiMo classification:

- Current source no longer shows the old MiMo cache-head shape crash. The optimized run loaded MiMo as MLLM, exposed `mimo_v2/mixed_swa_kv_v1` with `mimo_v2_asymmetric_swa`, paged prefix cache, block-disk L2, and TurboQuant/storage quantization telemetry.
- MiMo remains release-red: first exact-cache request produced empty visible output, cached repeat hit `cached_tokens=54` / `cache_detail=paged` but rambled instead of exact `ACK`, and `tool_choice=required` returned a typed 400 because the model produced no parsed tool calls.
- Conservative simple-mode diagnostic with no prefix cache and no KV quantization proved this is not only paged cache/TurboQuant corruption: exact `ACK` passes, but required tool calls fail with no tool calls both thinking-on and thinking-off, each around 96 seconds. MiMo is blocked by tool protocol/output quality and speed, with VL/audio/video still not implemented/cleared.

Proof-pointer refresh:

- `tests/cross_matrix/summarize_objective_proof.py`, current-suite defaults, release-manifest proof pointers, and exact-pointer tests now use the 2026-06-06 Gemma26/MiniMax/MiMo artifacts.
- Focused proof-pointer regression: `254 passed, 270 deselected`.
- Current suite: `build/current-regression-suite-after-gemma26-minimaxk-mimo-rerun-final-20260606.json` is `status=open`. All no-heavy/static/API/cache/tool/defaults/parser/model-family/VL-media/focused pytest gates pass. Remaining failed steps are `packaged_integrity_contracts`, `release_regression_manifest`, and `release_gate_skip_app`.
- Remaining open objective rows: Gemma4 26B Responses visible-content/language quality, Gemma4 26B mixed-SWA app-engine speed, cross-family live multi-turn smoke matrix, MiMo runtime/tool/long-prompt quality, MiniMax K reporter parity/root cause, real Electron UI cross-family matrix, and DSV4 long-output/code/file-generation quality.

## 2026-06-06 cross-family smoke proof refresh

Fresh proof-pointer refresh after the cross-family continuation:

- Objective digest: `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json`.
- Release manifest: `build/current-release-regression-manifest-after-dsv4-smoke-refresh-20260606.json` -> `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- Current suite: `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.
- Focused regression pytest inside the suite passed after pointer refresh; no source/static/package gate is masking the remaining live/model blockers.

Current cross-family source-smoke state:

- Covered/pass in current artifacts: Gemma4, Hy3, LFM2.5, Ling/Bailing, MiniMax Small/K text-cache/tool rows, Nemotron Omni, Qwen3.6, and Step3.7 text route.
- Qwen35 MXFP8 MTP current source smoke passed without the `gdn_sink` crash and with image/video/tool/cache rows in `build/current-all-local-model-smoke-qwen35-mxfp8-mtp-tools-media-20260606/summary.json`.
- Ling/Hy3/Nemotron current source smoke passed in `build/current-all-local-model-smoke-ling-hy3-nemotron-tools-media-20260606/summary.json`.
- Existing live slice remains the current proof for Gemma4 12B, LFM2.5, MiniMax Small, Qwen27, and Step3.7: `build/current-all-local-model-smoke-live-slice-tools-media-continuation-20260606/summary.json`.

Remaining release blockers from the current objective digest:

- Cross-family live multi-turn smoke matrix is still open because DSV4 is memory-preflight blocked, MiMo remains red, and ZAYA text/VL quality rows are red.
- MiMo V2.5 JANG_2L runtime/tool/long-prompt quality remains open. Do not paper over this with fake parser injection, cache disablement, or hidden prompt/default forcing.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause remains open.
- Real Electron UI cross-family live model matrix remains open.
- DSV4 long-output/code/file-generation quality remains open; latest local exactness preflight refused launch with `available_gb=107.91` vs `required_available_gb=120.0`.

ZAYA classification from current source smoke:

- `build/current-all-local-model-smoke-zaya-text-vl-tools-media-20260606/summary.json` is a real fail, not a cache crash.
- ZAYA text: CCA/paged cache and required tool call work, but exact `ACK` and visible reasoning output fail.
- ZAYA-VL: CCA/paged cache, tool call, blue image, and text-after-image work, but exact `ACK`, multi-turn recall, reasoning visible output, and red-image recognition fail.
- Classification: model-route/artifact quality and media semantics until source-vs-artifact comparison proves otherwise; cache telemetry itself is not the primary failure.

Release note requirement:

- Credit GitHub `@Hornsan1` in the next vMLX/mlxstudio release notes for reporting many of the runtime/media/cache issues that shaped this release hardening.

## 2026-06-06 ZAYA cache-contract refresh

Fresh source-server ZAYA rerun after making the all-local smoke cache probe family-aware:

- Harness change: ZAYA cache-repeat probes now use a native one-word color contract with `expected_content=blue` instead of a generic system-role `ACK` prompt that ZAYA ignores/repeats. Generic families still use exact `ACK`.
- Fresh artifact: `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json`.
- ZAYA text result: still `probe_failed`, but narrowed to reasoning-on only. Exact text cache repeat now returns `blue`, repeat request reports `cached_tokens=39` and `cache_detail=paged+zaya_cca`; multi-turn recall returns `blue cat`; required `record_fact` tool call passes.
- ZAYA text cache health: native cache reports `family=zaya`, `schema=zaya_cca_v1`, `cache_type=typed_cca`, components `standard_kv`, `cca_conv_state`, `cca_prev_hidden`, `moe_no_state_slots`; generic TurboQuant KV remains disabled because CCA state is path-dependent; prefix, paged, and block-disk L2 are true; block disk wrote 12 blocks / 611 tokens in the text row.
- ZAYA text remaining blocker: `reasoning_on` produces reasoning-only empty visible output after 256 completion tokens. This is a reasoning/template/decode behavior blocker, not a cache/TurboQuant/L2 failure.
- ZAYA-VL result: still `probe_failed`; cache hit telemetry exists (`cached_tokens=33`, `cache_detail=paged+zaya_cca`) and blue image/tool rows pass, but text cache exact answer is wrong (`green`), multi-turn recall returns `color `, reasoning is empty-visible, and red image is classified as `white`.
- Current digest: `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json`.
- Current suite: `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.

Do not mark ZAYA or ZAYA-VL release-cleared from this. The patch only separates cache proof from behavior failures so the next fix can target reasoning/template semantics and VL color/recall quality without confusing it with CCA cache corruption.

## 2026-06-06 ZAYA text reasoning budget closure

Fresh focused repro and smoke rerun proved ZAYA text thinking support was not fake; the prior smoke budget was too low:

- Repro artifact: `build/current-zaya-text-reasoning-budget-repro-20260606/summary.json`.
- At `max_tokens=256`, ZAYA text stops with `finish_reason=length`, `reasoning_chars=1101`, empty visible content.
- At `max_tokens=512`, `1024`, and `1536`, the same prompt stops normally and emits visible `FINAL=OK`; completion length was 364 tokens in the focused repro.
- Harness fix: ZAYA reasoning probes now use at least 512 max tokens while keeping strict visible `FINAL=OK` validation.
- Fresh smoke artifact: `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json`.
- ZAYA text is now pass in live smoke: exact cache repeat `blue`, repeat `cached_tokens=39` / `cache_detail=paged+zaya_cca`, multi-turn recall `blue cat`, reasoning visible `FINAL=OK`, and required `record_fact` tool call pass.
- Digest fix: mixed pass/fail smoke artifacts are now evaluated per family. The combined ZAYA artifact can clear `zaya_text` while keeping `zaya_vl` red.
- Current objective digest: `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json` now covers `zaya_text` and leaves `dsv4`, `mimo_v2`, and `zaya_vl` as the cross-family smoke blockers.
- Current suite: `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.

ZAYA-VL remains a real blocker:

- Text cache exactness returns `green` instead of expected `blue`.
- Multi-turn recall returns `color ` instead of `blue cat`.
- Reasoning-on still emits empty visible output.
- Red image returns `white`; blue image and repeated blue image pass.
- Tool call and typed CCA cache telemetry pass.

Do not mark ZAYA-VL release-cleared. Next work should trace the ZAYA-VL text template/language wrapper separately from the image path and inspect why red image embeddings classify as white while blue works.

## 2026-06-06 ZAYA-VL no-media contract refresh

- Live ZAYA-VL MXFP4 smoke after prompt-contract correction: `build/current-all-local-model-smoke-zaya-vl-mxfp4-after-no-media-contract-20260606/summary.json` -> `status=fail` with exactly one failure, `reasoning_on` empty visible output and `reasoning_chars=17`.
- Focused repro: `build/current-zaya-vl-mxfp4-focused-repro-20260606/summary.json` shows direct `ATTACHED/NONE` no-media prompts return `NONE` before and after image turns, while generic no-picture wording can produce apologetic/image-context text. This falsifies the prior no-media carryover hypothesis for MXFP4.
- All-quant comparison: `build/current-all-local-model-smoke-zaya-vl-all-quants-tools-media-20260606/summary.json` shows MXFP4 is currently healthier than JANGTQ4/K. JANGTQ4/K failures should be treated as quant/artifact/runtime-compatibility blockers until isolated by rendered prompt and raw-generation probes.
- Current objective digest: `build/current-objective-proof-after-dsv4-smoke-refresh-20260606.json`.
- Current suite: `build/current-regression-suite-after-dsv4-smoke-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.
- Release status remains blocked. No tag, notarization, DMG publication, `mlx.studio`, or `vmlx.net` download update was performed from this continuation.

## 2026-06-06 MiMo sink/cache falsification refresh

- New diagnostics: `build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json` and `build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json`.
- Refreshed audit: `build/current-mimo-v2-jang2l-current-audit-20260606.json` -> `status=open`, `local_release_clearance=false`.
- Cleared/falsified sub-hypotheses: local artifact manifest matches, stale local state absent, structural verify passes, narrow text cache passes, selected-expert SwitchGLU parity passes, and cache-prefill vs no-cache next-token top-10 logits match. Manual sink SDPA and disabling SWA sink do not clear generation quality.
- Remaining MiMo blockers: long-prompt coherence and tool protocol. The current blocker should not be described as cache-only or sink-kernel-only.
- MiMo VL/audio/video remains not implemented in Python; typed unsupported-media errors are fail-closed recovery only.

## 2026-06-06 DSV4 cross-family smoke refresh

- Fresh smoke artifact: `build/current-all-local-model-smoke-dsv4-jangtq-k-tools-cache-20260606/summary.json` -> `status=pass`, `row_count=2`, `completed=2`, `failed=0`.
- Primary row `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K` passed exact cache repeat, multi-turn recall, reasoning visible output, and required tool call.
- Cache proof: repeat prompt reported `cached_tokens=3639`, `cache_detail=paged+dsv4`, and cache stats mentioned TurboQuant, SSM/composite, and rotating cache components.
- Boundary: this is cross-family smoke coverage only. DSV4 long-output/code/file-generation quality remains open and still needs the exactness suite on a memory-safe host.
