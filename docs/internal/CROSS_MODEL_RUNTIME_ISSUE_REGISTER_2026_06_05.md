# Cross-Model Runtime Issue Register and Proof Tracker

Date: 2026-06-05

Purpose: never lose track of model/runtime/config/VL/audio/tool/cache regressions, proof requirements, and release status. This is an itemized tracker. Do not use a broad green check to close a narrow row unless the row's exact proof exists.

Related narrative document: `docs/internal/CROSS_MODEL_RUNTIME_FAILURE_CLASSES_2026_06_05.md`

Current known release state:

- vMLX 1.5.56 DMG hotfix shipped, signed, notarized, stapled, public updater/download current.
- `jjang-ai/vmlx` main after 1.5.56 includes structured JSON repair, DSV4 completions rail fix, MiMo JANG_2L load/MLLM adapter work, current Qwen35 dense MTP `gdn_sink` signature contract, DSV4 restart-L2 restore, Qwen27 JANG_4M installed-app speed/MTP proof-pointer refresh, and active MiMo release-blocker tracking.
- No post-1.5.56 tag/DMG/notarization/public release has been produced from these post-release source/main fixes; treat them as source/main plus local installed/staged proof only until a new notarized public package exists.
- PyPI is not current: PyPI latest remains `1.5.49`; `1.5.56` upload blocked by PyPI trusted-publisher/API-token config.
- Full cross-family runtime matrix remains open. Do not claim all model families production-cleared.
- Current regression suite proof after MiMo active-scope tracking is `build/current-regression-suite-after-mimo-active-scope-20260606.json`: `status=open`, `failed_steps=["release_regression_manifest"]`. Source/package/parity/contracts and focused pytest passed; the release manifest correctly remains red because 8 live/model objective rows are still open: Ling/Bailing multilingual output quality, Gemma4 26B CRACK Responses visible-content/language quality, Gemma4 26B CRACK mixed-SWA app-engine speed, cross-family live multi-turn smoke matrix, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- MiniMax #117/#179 proof boundary: current root-cause audit is `open`, memory-preflight artifact exists and did not launch the huge model, and live Responses cancel/reporter parity proof is still absent. This must stay open; do not classify screenshot/output corruption as model artifact or runtime until reporter parity proof exists.
- DSV4 default-cache tool loop boundary: `build/current-dsv4-default-cache-tool-loop/result.json` was run live with native prefix+paged+block-disk L2 enabled and `status=review`. Runtime/tool/cache checks pass: DSML tools executed `list_directory -> write_file -> write_file`, final answer was `DONE`, cached tokens were seen with `paged+dsv4`, native cache was `native_composite`, and generic TurboQuant KV stayed off. The remaining review cause is generated code exactness (`THREE.ScScene()` and `THREE.BBoxGeometry()`), so this is tracked under DSV4 code/file-generation quality, not as a default-cache/tool-loop runtime failure.
- DSV4 rows now proven from current artifacts: app-launch default native prefix/paged/L2 wiring is proven by `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json` plus `build/current-dsv4-default-cache-tool-loop/result.json`; native SWA+CSA/HCA composite cache is proven; same-process TTFT/cache-hit is proven from `build/current-dsv4-responses-cache-gate-20260606.json`; one-tool stop and multi-tool default-cache loop are proven; restart-L2 block-disk restore is proven from `build/current-dsv4-responses-restart-l2-gate-20260606.json`. Still open: DSV4 exact code/file generation quality.
- DSV4 one-tool-after-result row is now proven from `build/current-dsv4-responses-one-tool-stop-20260606.json`: round 1 emitted exactly one structured `list_directory` call, round 2 used `previous_response_id`, kept `tools=TOOLS` with `tool_choice=auto`, emitted no function calls, and returned exactly `DONE` with native prefix+paged+block-disk L2 enabled.
- DSV4 restart-L2 row is now fixed in source and live-proven. Current artifact `build/current-dsv4-responses-restart-l2-gate-20260606.json` is `status=pass`: native cache/prefix/paged/L2 were enabled, generic TurboQuant KV stayed off, disk write before restart passed, restart L2 disk hit passed, `restart_dsv4_cache_hit=true`, same cache dir/fresh nonce/same terminal prompt/server restart checks passed, and visible output was `STORED`. Classification: fixed `kernel_cache` runtime issue, not model artifact corruption.
- Packaged release signing/parity row was repaired on 2026-06-06. Fresh Developer ID signing now works non-interactively, `panel/release/sequoia-app/mac-arm64/vMLX.app` was rebuilt from current source, and `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json` is `status=pass` with staged app engine hash parity, source hash parity, hardened runtime, and signature preflight all green. This does not mean the public release is clear: DMG notarization was not produced in this pass, and live/model objective rows still block release readiness.
- Post-MiMo/DSV4 packaging refresh on 2026-06-06 rebuilt `panel/bundled-python`, staged `panel/release/sequoia-app/mac-arm64/vMLX.app`, and reran the canonical package/proof gates. `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json` is `status=pass`; `build/current-staged-app-runtime-parity-audit-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json` is `status=pass`; and the full current suite artifact `build/current-regression-suite-after-dsv4-restart-l2-fix-20260606.json` is `status=open` with only the expected release-manifest failure. A separate current installed-app audit `build/current-installed-app-runtime-parity-audit-after-mimo-cache-fix-20260606.json` is still `status=open` against `/Applications/vMLX.app` because the installed app runtime hashes drift from source in `server.py`, `mllm_scheduler.py`, `prefix_cache.py`, and other runtime files. Do not claim the installed app or public download contains the post-1.5.56 MiMo/DSV4 fixes until a new app is installed/notarized and installed parity is rerun.
- MiMo is explicitly back in scope as of 2026-06-06. Eric requested: delete all past local MiMo model copies on this machine because they are bad; HTTP-download the MiMo JANG_2L artifact referenced in `erics-m5-max2.local:~/jang` docs; then implement/fix the new MiMo JANG_2L runtime path with real live proof. Do not reuse older local MiMo artifacts as evidence.
- Gemma4 12B JANG_4M installed-app VLM recovery has two narrow live proofs from 2026-06-06. `build/current-gemma4-12b-installed-image-prefill-recovery-20260606.json` proves a small image request succeeds and the next text turn returns `text recovery ok` with native Gemma4 mixed SWA KV, prefix/paged/block-disk cache, and q8 attention KV enabled. `build/current-gemma4-12b-installed-image-prefill-forced-reject-recovery-20260606.json` proves a deliberately tiny image-prefill budget rejects a long image prompt with HTTP 413 and the next text turn returns `text recovery after reject ok`. This is installed-app/runtime recovery evidence, not full Gemma4 VL quality clearance and not video/audio clearance.
- JANGQ/JANG tools remote and local checkout state as of 2026-06-06: `jjang-ai/jangq` `origin/main` and `/Users/eric/jang` local `main` are both at `d1316c3` (`Add MiMo V2 JANG runtime support`). The prior local equivalent head `821f639` is preserved on backup branch `codex-backup/local-main-before-origin-sync-20260605214651`. The three untracked local files that overlapped remote additions were moved to `/Users/eric/.codex/tmp/jang-untracked-overlap-backup-20260605214620/` before rebasing; unrelated untracked files in `/Users/eric/jang` remain untouched.
- Cross-model no-heavy runtime policy contracts refreshed after the Gemma4 VLM recovery proof: `build/current-vl-media-cache-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-noheavy-api-cache-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-cache-architecture-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-step37-crash-falsification-contract-after-gemma4-vlm-recovery-20260606.json` `status=pass`; `build/current-issue175-admin-sleep-probe-after-gemma4-vlm-recovery-20260606.json` `status=pass`. These prove static/no-heavy route contracts, panel/cache policy, Step3.7 crash-class guardrails, and admin sleep behavior, but they do not replace live full-output model-family proofs.
- MiMo V2.5 JANG_2L local artifact was refreshed in this run from `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` to `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`. The stale local `107G` copy was deleted first. Transfer used `rsync` over the direct `en9` route. Current structural proof `build/current-mimo-jang2l-local-structural-verify-20260606.json` is `status=pass`: 173 files, 113,926,313,468 bytes, 150 top-level safetensor shards, 109,180 indexed tensors, audio tokenizer present, and chat template matched embedded.
- MiMo V2.5 JANG_2L text/cache live proof is now narrow-pass on current source: `build/current-mimo-jang2l-live-text-cache-smoke-20260606.json` returned exact `cache ok` on the first Chat Completions request and exact `cache ok` on the repeat with `cached_tokens=28`, `cache_detail=paged`, q8 KV storage, paged prefix cache, and block-disk L2 enabled. This proves the asymmetric full/SWA cache path survives a same-process paged hit; it does not clear speed, tools, VL/audio/video, or restart-L2.
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

Current status: `[!]` Text-only stable for CRACK view; unsupported VLM route crash class open.

Proofs present from review:

- [x] Step Flash CRACK text-only view stable under vMLX 1.5.55.
- [x] Fast eval completed 18/18, 0 failed requests.
- [x] Runtime crash reproduced in unsupported MLLM route.

Still needed:

- [ ] Implement vMLX unsupported Step3p7 VLM guard.
- [ ] Reproduce on local vMLX 1.5.56/HEAD.
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
- Repair evidence: `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json` is `status=pass`; `package_signing_preflight.status=pass`; `staged_app_engine_hash_parity=true`; `staged_app_engine_source_hash_parity=true`.

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
