# Local vMLX / MLXStudio Release Blocker Ledger - 2026-06-06

Scope: local vMLX Python engine and MLXStudio/panel release path only. No adlab, Max2, TP4, RDMA, or Thunderbolt work belongs in this lane unless explicitly requested.

## Current release state

- Worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- HEAD: `d93dd2be`
- Version: `1.5.56`
- Source manifest: `build/current-release-regression-manifest-after-bundled-refresh-20260606.json`
- Manifest status: `fail`
- Prepackage ready: `False`
- Release ready: `False`
- Current objective proof: `build/current-objective-proof-after-bundled-refresh-20260606.json`
- Current regression suite: `build/current-regression-suite-after-bundled-refresh-20260606.json`
- Current packaged integrity proof: `build/current-packaged-integrity-contract-after-bundled-refresh-20260606.json`
- Current hard release rule: packaging, staging, signing, notarization, release tags, public uploads, and download updates are locked until the runtime/model/UI/cache blockers below are green, unless Eric explicitly unlocks release packaging in the current turn.
- Important packaging nuance: `electron-builder --dir` signs this app because the panel package configuration has Developer ID signing enabled. It must not be used as a harmless staged-parity check while blockers are open.

## Current blocker-driven operating contract

- Every action must reduce a named blocker in this ledger.
- Do not chase package/sign/notary rows while runtime/model/UI/cache blockers remain open.
- Do not call proof-pointer cleanup release progress unless it removes stale evidence and exposes the real blocker state.
- Do not commit local `AGENTS.md`, `.agents/`, `vmlx_engine/vendor/`, staged app output, release output, or generated signing artifacts.
- Do not claim production readiness from source-only, import-only, load-only, health-only, text-only, fail-closed, or narrow unit-test proof.
- Do not call Step3p7 VLM fixed because advertised VLM now routes text-only; that is a crash-avoidance row only.
- Do not call JSON repair native guided decoding.
- Do not call preserved media sidecars runtime media support.
- Do not disable prefix cache, paged cache, L2 cache, TurboQuant KV, MTP, thinking, tools, media, or continuous batching and call the family release-cleared unless that disabled mode is the documented release behavior.

## Current top-level open blockers from `after-bundled-refresh`

- Cross-family live multi-turn smoke matrix is not release-cleared.
- MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is not release-cleared.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause is not release-cleared.
- Real Electron UI cross-family live model matrix is not release-cleared.
- DSV4 long-output/code/file-generation quality is not release-cleared.
- Packaged integrity remains red only as a release gate because staged app parity and current-objective dry gate are not release-cleared; do not address this by signing/staging while runtime blockers remain open.

## Explicit release blockers from current manifest

- `mimo_v2_jang2l_runtime_quality_open`: `open`
  Evidence: `build/current-mimo-jang2l-local-structural-verify-20260606.json,build/current-mimo-jang2l-live-text-cache-smoke-20260606.json,build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json,build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json,build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json,build/current-mimo-v2-jang2l-current-audit-after-native-thinking-off-20260606.json,build/current-mimo-v25-jang2l-local-metadata-truth-patch-20260606.json,build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json,build/current-mimo-batched-thinking-off-decode-policy-source-20260606.json,build/current-mimo-v2-jang2l-cb-cache-after-batched-policy-live-20260606.json,build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json`
  Next proof: Pass current local MiMo JANG_2L tool protocol/continuation, long/system prompt behavior without working-set rejection, source-vs-quant classification, speed target, and media honesty/API/UI proof before including MiMo in a production release.
- `issue179_minimax_k_root_cause_audit`: `open`
  Evidence: `build/current-issue179-minimax-k-root-cause-audit-after-public-v1556-scan-20260606.json`
  Next proof: Obtain reporter installed app bundle hash provenance matching a public/local vMLX server.py route proof, or refresh reporter parity metadata against a known public DMG; do not rerun broad cache/model probes until this provenance gap changes.

## Open/non-pass sweep rows

- `component_ok`: `None`; artifact `None`
- `regression_suite`: `open`; artifact `build/current-regression-suite-after-noheavy-pointer-refresh-20260606.json`
- `live_smoke_summaries`: `fail`; artifact `None`
- `live_tool_smoke_summaries`: `fail`; artifact `None`
- `mimo_v2_jang2l_root_cause`: `open`; artifact `None`
  Failures: `["mimo_source_vs_quant_first_divergence_missing"]`
- `diagnostic_live_smoke_summaries`: `fail`; artifact `None`
- `issue175_179_release_boundary_audit`: `open`; artifact `build/current-issue175-179-release-boundary-audit-after-public-v1556-scan-20260606.json`
  Failures: `["unexpected_issue_slice:175", "unexpected_issue_slice:176", "unexpected_issue_slice:177", "failed_issue_checks:175", "missing_issue_check:175:installed_app_live_memory_stress_proven", "missing_issue_check:175:installed_app_memory_clear_runtime_proven", "failed_issue_checks:176", "missing_issue_check:176:installed_app_live_memory_pressure_proven", "missing_issue_check:176:installed_app_promoted_block_cleanup_proven", "failed_issue_checks:177", "missing_issue_check:177:installed_app_cache_selection_telemetry_proven", "missing_issue_check:177:installed_live_ttft_and_cold_paged_tq_proven"]`
- `issue175_177_installed_runtime_audit`: `missing`; artifact `build/current-issue175-177-installed-runtime-audit-20260602-v1554-installed-tahoe.json`
- `issue175_177_live_runtime_audit`: `missing`; artifact `build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json`
- `issue179_minimax_k_root_cause_audit`: `open`; artifact `build/current-issue179-minimax-k-root-cause-audit-after-public-v1556-scan-20260606.json`
  Failures: `["missing_proven:local_installed_responses_cancel_live_probe", "missing_proven:local_reporter_prompt_reproduction_clean", "missing_proven:public_v1549_tahoe_dmg_has_responses_cancel_route", "missing_proven:reporter_cancel_404_after_econnreset_same_response_id_proven", "missing_proven:reporter_cancel_404_after_stream_abort_order_proven", "missing_proven:reporter_log_has_abort_before_visible_content", "missing_proven:reporter_log_has_responses_cancel_404", "missing_proven:reporter_log_installed_app_bundled_python_seen", "missing_proven:reporter_log_launch_parser_cache_flags_seen", "missing_prove`
- `public_app_issue_audit`: `open`; artifact `build/current-public-app-issue-audit-after-public-v1556-scan-20260606.json`
  Failures: `["unexpected_issue_slice:115", "failed_issue_checks:115", "missing_issue_check:115:gemma4_current_installed_ui_speed_gate_passes", "missing_issue_check:115:gemma4_cold_wall_includes_ttft_tracked", "unexpected_issue_slice:119", "failed_issue_checks:119"]`
- `real_ui_live_model_matrix`: `open`; artifact `None`
- `release_blocker_ledger`: `open`; artifact `None`
  Failures: `[{"details": {"prompt_length_coherence_blocked": true, "release_boundary": "local artifact/runtime has narrow text-cache proof but fails long-prompt coherence, tool protocol, and/or local source-vs-quant first-divergence proof; do not release-clear MiMo", "root_cause_candidate": "mimo_v2_jang2l_quantized_profile_or_full_forward_quality_pending_source_vs_quant", "status": "open", "structural_verify_passed": true, "switchglu_selected_expert_parity_passed": true, "text_cache_narrow_pass": true, "tool_protocol_blocked": true}, "evidence": "build/current-mimo-jang2l-local-structural-verify-20260606`

## Objective scope status

### `mimo_v2_5`

- Status: `open`
- Proved: local structural/integrity proof passed
- Proved: narrow text cache proof passed
- Proved: selected SwitchGLU expert parity passed
- Proved: text-only runtime modality honesty/fail-closed media proof exists
- Proved: continuous-batching exact cache prompt following was reproved with native thinking-off prompt rendering, paged prefix cache, q8 storage, and block-disk L2 counters
- Not proven/failed: tool protocol/continuation not cleared
- Not proven/failed: long/system prompt behavior is not cleared because the third live CB row still hit Metal working-set pressure reject at 98.5% of 107.5GB
- Not proven/failed: decode speed below target
- Not proven/failed: source-vs-quant first divergence missing
- Not proven/failed: real vision/audio/video forward path not wired/proven
- Artifact: `build/current-mimo-v2-jang2l-current-audit-after-native-thinking-off-20260606.json`
- Artifact: `build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json`
- Artifact: `build/current-mimo-v25-jang2l-local-metadata-truth-patch-20260606.json`
- Artifact: `build/current-mimo-v25-jang2l-local-sync-image-proof-20260606.json`
- Artifact: `build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json`

### `qwen36_27b_35b_mtp`

- Status: `contract_proven_not_full_e2e_release_proven`
- Proved: noheavy model family/artifact format contracts include qwen36 MTP/VL/MXFP/JANG rows
- Proved: native MTP policy/autodetect tests are indexed in current artifacts
- Not proven/failed: current full multi-turn live/UI E2E matrix is not release-pass in latest manifest
- Not proven/failed: reported qwen35b MTP gdn_sink regression is not represented here as a passed live installed-app proof

### `gemma4_12b`

- Status: `partial_contracts_open_public_app_issue`
- Proved: Gemma4 mixed SWA/KV cache contract rows pass in cache architecture matrix
- Not proven/failed: public app issue audit has missing Gemma4 installed UI speed checks

### `dsv4_flash`

- Status: `cache_tool_pass_but_code_exactness_not_broadly_clear`
- Proved: default native composite prefix/paged/L2 cache pass
- Proved: Responses cache and restart L2 pass
- Proved: multi-tool DSV4 loop final answer pass
- Not proven/failed: objective proof still records code identifier corruption in DSV4 generated code/tool content diagnostics

### `minimax`

- Status: `release_blocker_open`
- Not proven/failed: issue179 MiniMax K reporter/root-cause provenance remains open in release blockers

### `step37_flash_lfm_nemotron_omni_zaya_hybrid`

- Status: `contract_rows_present_but_full_live_release_matrix_not_clear`
- Proved: cache architecture/model family noheavy contracts include Step3.7, LFM2 hybrid, Nemotron hybrid, ZAYA typed CCA rows
- Not proven/failed: latest release manifest still has live smoke/tool/real UI rows open or failing, so no full production claim

### `mlxstudio_app_release`

- Status: `not_ready`
- Proved: version metadata is 1.5.56 in pyproject, panel package, and vmlx_engine
- Not proven/failed: prepackage_ready=false
- Not proven/failed: release_ready=false
- Not proven/failed: installed-app issue audits missing/open
- Not proven/failed: no current signed/notarized 1.5.56 release artifact produced from this ledger state

## Next local-only actions

- Do not touch adlab/Max2/TP4/RDMA/TB for this objective.
- Resolve or explicitly defer MiMo release blocker before claiming production release.
- Resolve MiniMax issue179 reporter provenance or mark release boundary honestly.
- Refresh missing/open installed-app/public-app audits locally.
- Only after release manifest prepackage_ready/release_ready passes, build/sign/notarize MLXStudio/vMLX 1.5.56 artifacts.

## 2026-06-06 parser repair update

- Added schema-gated `xml_function` parser repair for MiMo-style output that contains a complete `<function=...><parameter=...></function></tool_call>` block but dropped the opening `<tool_call>` wrapper.
- This is not fake tool injection: repair only runs when the request tool schema contains the function name. Unknown functions and bare incomplete `<tool_call>` still fail closed.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/tool_parsers/xml_function_tool_parser.py tests/test_xml_function_tool_parser.py`
  - `.venv/bin/python -m pytest -q tests/test_xml_function_tool_parser.py` -> `10 passed`
  - `.venv/bin/python tests/cross_matrix/run_tool_call_contract.py --out build/current-tool-call-contract-after-xml-function-repair-20260606.json` -> `status=pass`
- Release boundary unchanged: MiMo live tool protocol, long-prompt coherence/OOM, decode speed, source-vs-quant classification, and real VL/audio/video wiring remain open.

## 2026-06-06 manifest pointer refresh after parser repair

- Updated current proof pointers from `current-tool-call-contract-after-mimo-tool-blocker-20260606.json` to `current-tool-call-contract-after-xml-function-repair-20260606.json` in release/objective/current-suite runners.
- Verification:
  - py_compile for touched proof scripts/tests passed.
  - `.venv/bin/python -m pytest -q tests/test_release_regression_manifest.py -k 'tool_calls_with_runner_artifact or source_hashes_all_referenced_code_files or source_hash_list_matches_current_suite_runner'` -> `3 passed`.
  - `.venv/bin/python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-after-xml-function-repair-20260606.json` wrote a valid manifest and exited nonzero because the release is still not ready.
- Current release state after pointer refresh remains: `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.

## 2026-06-06 live MiMo tool probe after XML parser repair

- Artifact: `build/current-mimo-live-xml-repair-tool-probe-20260606.json`.
- Local-only server: `127.0.0.1:8897`, model `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`, simple engine, `--tool-call-parser xml_function`, `--reasoning-parser think_xml`, `--default-enable-thinking false`, `--kv-cache-quantization none`.
- Result: `status=fail`.
- `tool_choice=auto`: HTTP 200 after about 80s, `finish_reason=stop`, punctuation/CJK-style garbage content, no `tool_calls`.
- `tool_choice=required`: HTTP 400 after about 59s because no tool call was produced.
- Post-tool text recovery failed in the same probe, so the live runtime remains unstable after this row.
- Classification: the schema-gated XML parser repair is valid and covered by noheavy tests, but it does not clear MiMo live tool protocol. Remaining likely blocker is MiMo local quant/runtime generation quality, not parser strictness alone.

## 2026-06-06 live MiMo probe with auto tool choice enabled

- Artifact: `build/current-mimo-live-auto-tool-enabled-probe-20260606.json`.
- Local-only server: `127.0.0.1:8897`, simple engine, `--enable-auto-tool-choice`, `--tool-call-parser xml_function`, `--reasoning-parser think_xml`, `--default-enable-thinking false`, `--kv-cache-quantization none`.
- Server log confirmed: `Tool calling: ENABLED (parser: xml_function)`.
- Result: `status=fail`.
- Baseline exact text before tool request failed: returned `The user said` with `finish_reason=length` instead of `READY`.
- Auto tool request failed: punctuation/CJK-style garbage, no `tool_calls`.
- Post-tool exact text also failed: returned `The user said`.
- Classification update: MiMo live failure is broader than parser strictness or missing `--enable-auto-tool-choice`; this local bundle/runtime path is failing baseline exact instruction following at about 0.5-1.5 tok/s in simple/no-KV-quant mode.

## 2026-06-06 MiMo MLLM `inputs_embeds` interface fix

- Artifact: `build/current-mimo-v2-mllm-inputs-embeds-interface-fix-20260606.json`.
- Fixed a concrete runtime-interface bug: the registered `mlx_vlm.models.mimo_v2.Model.__call__` accepted `inputs_embeds` through `**kwargs` but dropped it before calling the language model.
- The wrapper now forwards `inputs_embeds`, `cache`, `mask`, and remaining kwargs to the MiMo language model for MLLM-style calls.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/models/mllm.py tests/test_mimo_v2_mllm_runtime_registration.py` -> pass.
  - `.venv/bin/python -m pytest -q tests/test_mimo_v2_mllm_runtime_registration.py` -> `1 passed`.
- Release boundary unchanged: this is required MLLM/VL interface plumbing, but it does not wire MiMo vision/audio/video towers and does not clear current live text/tool generation quality failures.

## 2026-06-06 release manifest after MiMo MLLM interface fix

- Artifact: `build/current-release-regression-manifest-after-mimo-mllm-inputs-embeds-fix-20260606.json`.
- Result: manifest runner exited nonzero because release is still blocked.
- Current release state remains: `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- MiMo current audit now includes `mllm_inputs_embeds_interface=true`, but still blocks on long-prompt coherence, tool protocol, decode speed, source-vs-quant proof, and VL/audio/video media wiring.

## 2026-06-06 local-only MiMo source-vs-quant preflight refresh

- Artifact: `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json`.
- Artifact: `build/current-mimo-v2-jang2l-tool-source-preflight-20260606.json`.
- Updated current MiMo source-vs-quant and tool-source preflight evidence to local-only scope.
- Refreshed after the SimpleEngine MiMo decode fixes: local quant bundle `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` exists and the local quant endpoint was healthy on `127.0.0.1:8897` during preflight; remote source path `erics-m5-max2.local:/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5` exists.
- Remaining source-vs-quant prerequisite blocker: source endpoint `http://erics-m5-max2.local:8126` is not listening, so source/quant prompt rows have not executed and model-artifact versus runtime/decode-loop classification remains unresolved.
- Remote source-path evidence is preflight-only. It is not source-vs-quant clearance because no source endpoint rows executed.
- Release manifest after this refresh: `build/current-release-regression-manifest-after-mimo-local-only-preflight-20260606.json`.
- Current release state remains: `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.

## 2026-06-06 Qwen dense MTP `gdn_sink` runtime fix

- Artifact: `build/current-qwen35-dense-mtp-gdn-sink-fix-20260606.json`.
- Fixed dense `mlx_lm` Qwen MTP adapter so `gdn_sink` is not just accepted at `GatedDeltaNet`/`DecoderLayer`, but captured and propagated through `Qwen3_5TextModel`, `TextModel`, and outer `Model`.
- This addresses the reported crash class: `_patch_gated_delta_net.<locals>.__call__() got an unexpected keyword argument 'gdn_sink'` and prevents dense MTP from silently dropping the sink capture path.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/patches/mlx_lm_mtp/qwen35_model.py tests/test_engine_audit.py` -> pass.
  - `.venv/bin/python -m pytest -q tests/test_engine_audit.py -k 'qwen35_dense_mtp_patch'` -> `2 passed`.
  - `.venv/bin/python -m pytest -q tests/test_native_mtp_autodetect.py -k 'qwen36_vlm_runtime_patch_installs_native_mtp or gated_delta'` -> `3 passed`.
- Release boundary unchanged: this is a source runtime fix, not a full installed-app/live Qwen 27B/35B MTP E2E clearance.

## 2026-06-06 VLM image prefill high-memory guard proof

- Artifact: `build/current-vlm-image-prefill-high-memory-guard-proof-20260606.json`.
- Source behavior is memory-aware: without explicit `VMLINUX_VLM_IMAGE_PREFILL_BUFFER_GB`, the single-buffer guard scales with Metal working set and preserves an 8GB floor.
- The Gemma-class high-memory case modeled in source uses `seq_len=12953`, `num_attention_heads=32`, `predicted_attention=10.0GB`, `max_working_set=96GB`; source decision is `should_reject=false`.
- Explicit `VMLINUX_VLM_IMAGE_PREFILL_BUFFER_GB=8` still preserves the old 8GB cap intentionally.
- Verification: `.venv/bin/python -m pytest -q tests/test_vl_video_regression.py -k 'vlm_image_prefill_default_single_buffer_guard_scales_on_high_memory or vlm_image_prefill_explicit_single_buffer_guard_preserves_old_limit or vmlx156_simple_mllm_guard_uses_media_expanded_input_ids or simple_mllm_guard'` -> `4 passed`.
- Release boundary: this proves the source guard policy, not installed-app parity or full Gemma 4 VL quality. A signed/notarized app must be rebuilt from current source for affected users to stop seeing old fixed-8GB behavior.

## 2026-06-06 Step3.7 advertised VLM text-only routing proof

- Artifact: `build/current-step37-advertised-vlm-text-only-routing-proof-20260606.json`.
- Current source behavior: Step3p7 bundles that advertise VLM through `vision_config` and/or `jang_config.architecture.has_vision=true` route text-only by default to avoid unsupported MLLM crashes.
- `force_mllm=True` is also overridden for this unsafe Step3p7 advertised-VLM path.
- Verification:
  - `.venv/bin/python -m pytest -q tests/test_step3p7_mllm_detection_guard.py tests/test_step37_vlm_runtime_audit.py` -> `10 passed`.
  - `.venv/bin/python -m pytest -q tests/test_engine_audit.py -k 'step3p7 and mllm'` -> `1 passed`.
- Release boundary: source text-runtime guard is covered; real Step3.7 image/video VLM runtime and installed-app parity are not release-cleared.

## 2026-06-06 structured-output JSON repair proof

- Artifact: `build/current-structured-output-json-repair-proof-20260606.json`.
- Current source includes post-generation JSON repair and schema normalization in `vmlx_engine/api/tool_calling.py`, plus benchmark reporting in `bench/structured_output_repair_report.py`.
- Covered cases include markdown fences, trailing commas, Python literals, missing closing braces, prose around JSON, adjacent string fragments for schema array fields, scalar-to-array schema coercion, and raw-vs-repaired diagnostics.
- The Qwen-style malformed case `{"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"}` repairs to `{"visible_text": ["CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"]}` when the schema declares `visible_text` as an array.
- Verification: `.venv/bin/python -m pytest -q tests/test_structured_output.py tests/test_structured_output_repair_report.py` -> `44 passed, 2 skipped`.
- Release boundary: this is post-generation repair/validation, not hard constrained JSON decoding.

## 2026-06-06 MiMo SimpleEngine non-MLLM prompt-shape fix

- Artifact: `build/current-mimo-simple-engine-llm-template-fix-20260606.json`.
- Fixed a concrete prompt-rendering gap in `vmlx_engine/engine/simple.py`: the non-MLLM SimpleEngine text path for `mimo_v2` now renders the native MiMo template with `enable_thinking=true` when the API request asks `enable_thinking=false`, matching the existing MLLM and batched MiMo text-only plain-prefix contract.
- Root issue: MiMo's native `enable_thinking=false` template emits `<think></think>` after the assistant prefix. Current MiMo artifacts already showed that closed rail can produce `<|im_end|>` first-token stop, empty visible output, or garbage on long/cache/tool rows.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/engine/simple.py tests/test_mllm_message_serialization.py` -> pass.
  - `.venv/bin/python -m pytest -q tests/test_mllm_message_serialization.py -k 'mimo_v2_thinking_false_uses_plain_template_prefix or simple_engine_routes_mimo_text_only_chat_through_language_model or simple_engine_mimo_llm_path_uses_plain_template_prefix or batched_engine_mimo_text_only_uses_plain_template_prefix'` -> `4 passed, 67 deselected`.
- Release boundary unchanged: this is a source regression fix for one prompt-rendering gap. It does not clear MiMo live long-prompt coherence, tool protocol, source-vs-quant classification, decode speed, cache/L2, or VL/audio/video runtime support.

Superseded by the later native-thinking-off correction below. Do not carry the
plain-prefix assumption forward as a release contract.

## 2026-06-06 MiMo SimpleEngine thinking-off decode policy fix

- Artifact: `build/current-mimo-simple-thinking-off-decode-fix-live-red-20260606.json`.
- Fixed a second concrete SimpleEngine MiMo text-only gap: when the effective API rail is `enable_thinking=false`, the decode loop now suppresses the native MiMo `<think>` and `</think>` token IDs at the logits boundary.
- This is not synthetic output injection and does not fold prompts. It enforces the requested thinking-off API contract after the prompt-rendering fix switched MiMo to the stable plain assistant prefix.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/engine/simple.py tests/test_mllm_message_serialization.py` -> pass.
  - `.venv/bin/python -m pytest -q tests/test_mllm_message_serialization.py -k 'mimo_v2_thinking_false_uses_plain_template_prefix or simple_engine_routes_mimo_text_only_chat_through_language_model or simple_engine_mimo_text_only_generate_suppresses_think_tags or simple_engine_mimo_llm_path_uses_plain_template_prefix or batched_engine_mimo_text_only_uses_plain_template_prefix'` -> `5 passed, 67 deselected`.
- Live source-server proof on `127.0.0.1:8898`, conservative SimpleEngine/no-continuous-batching/no KV quant:
  - `exact_ack`: passed short visible answer, returned `ACK-MIMO-742`, no `<think>` tag leak.
  - `longish_no_think_explicit_system`: failed, empty visible output with `completion_tokens=0`.
  - `sentinel_no_explicit_system`: failed exact instruction following, produced explanatory text instead of the requested sentinel.
  - `sentinel_explicit_system_thinking_true`: failed, empty visible output with `completion_tokens=40`.
- Classification: the visible `<think>` leak is improved in the SimpleEngine MiMo text-only path, but MiMo remains release-red. Current remaining blockers are exact instruction following, explicit-system visible output, tool protocol, source-vs-quant first divergence, decode speed, cache/L2 matrix, and VL/audio/video runtime wiring.
- Release boundary unchanged: do not package, sign, notarize, tag, upload, or update downloads from this state.

## 2026-06-06 MiMo SimpleEngine first-token EOS decode policy fix

- Artifact: `build/current-mimo-simple-thinking-off-first-token-eos-live-red-20260606.json`.
- Extended the SimpleEngine MiMo thinking-off logits policy so the MiMo EOS marker is suppressed only on the first generated token. This targets the proven `failing_system_long` first-token `<|im_end|>` stop without globally disabling natural EOS later in the answer.
- This is still a runtime decode policy, not prompt folding and not synthetic output. The forbidden workaround remains forbidden: do not fold system prompts into user prompts to hide this bug.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/engine/simple.py tests/test_mllm_message_serialization.py` -> pass.
  - `.venv/bin/python -m pytest -q tests/test_mllm_message_serialization.py -k 'mimo_v2_thinking_false_uses_plain_template_prefix or simple_engine_routes_mimo_text_only_chat_through_language_model or simple_engine_mimo_text_only_generate_suppresses_think_tags or simple_engine_mimo_llm_path_uses_plain_template_prefix or batched_engine_mimo_text_only_uses_plain_template_prefix'` -> `5 passed, 67 deselected`.
- Live source-server proof on `127.0.0.1:8898`, conservative SimpleEngine/no-continuous-batching/no KV quant:
  - `exact_ack`: passed short visible answer, returned `ACK-MIMO-742`, no `<think>` tag leak.
  - `longish_no_think_explicit_system`: improved from empty output to visible output, but still failed instruction following and hit `finish_reason=length`.
  - `first_token_probe_shape_ack`: improved from first-token stop to starting with `ACK`, but still continued with extra text: `ACKThe user wants...`.
  - `sentinel_no_explicit_system`: still failed exact instruction following.
  - `sentinel_explicit_system_thinking_true`: still produced empty visible output.
- Classification: first-token stop is improved in the conservative SimpleEngine MiMo path. MiMo remains release-red because exact instruction following, long-output coherence, speed, tools, cache/L2, source-vs-quant classification, and VL/audio/video runtime wiring remain open.
- Release boundary unchanged: do not package, sign, notarize, tag, upload, or update downloads from this state.

## 2026-06-06 MiMo batched/MLLM thinking-off decode policy source fix

- Artifact: `build/current-mimo-batched-thinking-off-decode-policy-source-20260606.json`.
- Extended the MiMo thinking-off decode policy into the continuous-batching/cache route by threading effective `enable_thinking` through `BatchedEngine -> MLLMScheduler -> MLLMBatchRequest` and applying MiMo-only logits processors inside `MLLMBatchGenerator._make_request_sampler`.
- Policy: suppress native `<think>` and `</think>` token IDs whenever MiMo API thinking is off; suppress `<|im_end|>` only before the first generated token, using `request.output_tokens` rather than prompt-token length.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/mllm_batch_generator.py vmlx_engine/mllm_scheduler.py vmlx_engine/engine/batched.py tests/test_mllm_continuous_batching.py` -> pass.
  - `.venv/bin/python -m pytest -q tests/test_mllm_continuous_batching.py -k 'mimo_batched_thinking_off_sampler_suppresses_tags_and_first_eos or mimo_batched_thinking_on_sampler_does_not_suppress_eos'` -> `2 passed, 37 deselected`.
- Release boundary unchanged: this is source regression coverage only. It does not clear live MiMo continuous-batching cache/L2 rows, exact instruction following, tool protocol, speed target, source-vs-quant classification, or VL/audio/video runtime support.

## 2026-06-06 MiMo native thinking-off prompt correction

- Source fix: `vmlx_engine/engine/simple.py`, `vmlx_engine/engine/batched.py`, and `vmlx_engine/models/mllm.py` no longer rewrite MiMo `enable_thinking=false` requests to the thinking-on prompt rail.
- Reason: live CB proof showed the old rewrite leaked reasoning-style text into visible content even when thinking tags were suppressed at logits/output boundaries.
- Source proof: `build/current-mimo-batched-thinking-off-decode-policy-source-20260606.json` remains the sampler-policy source proof; focused tests now assert native thinking-off prompt preservation.
- Failed pre-fix live proof: `build/current-mimo-v2-jang2l-cb-cache-after-batched-policy-live-20260606.json` had paged/L2 cache alive but returned visible reasoning text for exact prompts and hit a 503 on the third row.
- Current live proof: `build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json`.
- Live improvement: `exact_repeat_1` and `exact_repeat_2` both returned exactly `ACK-CB-742`, no visible think tags, finish `stop`, repeat row had `cached_tokens=33`, `cache_detail=paged`, q8 storage quantization, native `mixed_swa_kv_v1`, and `l2_tokens_on_disk=33`.
- Still release-red: decode speed was about `0.4 tok/s` cold and `1.5 tok/s` on cache hit, and the third system row returned HTTP 503 from Metal working-set pressure (`98.5%` of `107.5GB`, threshold `98.0%`).
- Classification: prompt-mode rewrite was a runtime decode/template bug and is fixed in source; remaining MiMo blockers are speed, memory pressure/recovery, tool protocol, source-vs-quant classification, and real VL/audio/video wiring.

## 2026-06-06 MiMo current audit after native thinking-off proof

- Artifact: `build/current-mimo-v2-jang2l-current-audit-after-native-thinking-off-20260606.json`.
- Audit pointer update: MiMo current-audit runners and release/objective proof consumers now target the native-thinking-off audit artifact instead of `after-mllm-inputs-embeds-fix`.
- Cleanup: removed stale `/Users/eric/.cache/huggingface/modules/transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L`; current audit reports `stale_local_state_absent=true`.
- Newly green in current audit: `exact_cache_prompt_following=true`, `prefix_paged_l2_cache_reproved=true`, `system_prompt_first_token_stop=true`, structural/text-cache/SwitchGLU/cache-vs-nocache/inputs-embeds remain green.
- Remaining MiMo blockers from current audit: `mimo_long_prompt_coherence_blocked`, `mimo_tool_protocol_blocked`, `mimo_decode_speed_below_release_target`, `mimo_cb_system_prompt_working_set_pressure_blocked`, `mimo_source_vs_quant_first_divergence_missing_or_failed`, and `mimo_vl_audio_video_unwired`.

## 2026-06-06 Current suite and release manifest after MiMo native thinking-off audit

- Current regression suite: `build/current-regression-suite-after-mimo-native-thinking-off-audit-20260606.json`.
- Suite status: `open`.
- Suite failed steps: `packaged_integrity_contracts`, `focused_regression_pytest`, `release_regression_manifest`, `release_gate_skip_app`.
- Suite open requirements: cross-family live multi-turn smoke matrix, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- Release manifest: `build/current-release-regression-manifest-after-mimo-native-thinking-off-audit-20260606.json`.
- Release manifest status: `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- Release action: do not package, sign, notarize, or publish MLXStudio/vMLX from this state.

## 2026-06-06 Qwen 3.6 GatedDelta MTP `gdn_sink` regression proof

- Artifact: `build/current-qwen36-mtp-gdn-sink-source-proof-20260606.json`.
- Classification: runtime source compatibility issue, not model-download corruption.
- Source state: current source includes commit `525ccedf Fix Qwen dense MTP gdn_sink propagation`.
- Focused proof: `py_compile` passed for Qwen MTP patch/test files; focused pytest passed `3 passed, 587 deselected` for dense `gdn_sink` signature/propagation and VL Qwen3.5/3.6 pre-load activation.
- Covered: dense GatedDeltaNet accepts `gdn_sink`; dense decoder/backbone/text/outer path propagates it; VL Qwen language runtime exposes `n_confirmed` and `gdn_sink` across patched runtime methods.
- Still not release-cleared: live 27B/35B model-server native-MTP generation, packaged `/Applications/vMLX.app` parity, long multi-turn tool/caching stress, signing, notarization, and public download update.
