# Local vMLX / MLXStudio Release Blocker Ledger - 2026-06-06

Scope: local vMLX Python engine and MLXStudio/panel release path only. No adlab, Max2, TP4, RDMA, or Thunderbolt work belongs in this lane unless explicitly requested.

## Current release state

- Worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- HEAD: `d93dd2be`
- Version: `1.5.56`
- Source manifest: `build/current-release-regression-manifest-after-noheavy-pointer-refresh-20260606.json`
- Manifest status: `fail`
- Prepackage ready: `False`
- Release ready: `False`

## Explicit release blockers from current manifest

- `mimo_v2_jang2l_runtime_quality_open`: `open`
  Evidence: `build/current-mimo-jang2l-local-structural-verify-20260606.json,build/current-mimo-jang2l-live-text-cache-smoke-20260606.json,build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json,build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json,build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json,build/current-mimo-v2-jang2l-current-audit-after-mllm-inputs-embeds-fix-20260606.json,build/current-mimo-v25-jang2l-local-metadata-truth-patch-20260606.json,build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json`
  Next proof: Pass current local MiMo JANG_2L long-prompt coherence, tool protocol/continuation, cache, and API proof before including MiMo in a production release.
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
- Proved: continuous-batching exact cache prompt following was reproved with paged/L2 counters
- Not proven/failed: tool protocol/continuation not cleared
- Not proven/failed: long-prompt coherence/OOM not cleared
- Not proven/failed: decode speed below target
- Not proven/failed: source-vs-quant first divergence missing
- Not proven/failed: real vision/audio/video forward path not wired/proven
- Artifact: `build/current-mimo-v2-jang2l-current-audit-after-mllm-inputs-embeds-fix-20260606.json`
- Artifact: `build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json`
- Artifact: `build/current-mimo-v25-jang2l-local-metadata-truth-patch-20260606.json`
- Artifact: `build/current-mimo-v25-jang2l-local-sync-image-proof-20260606.json`

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
- Local source bundle `/Users/eric/models/MiMo-V2.5-source` is missing; local quant bundle `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` exists.
- No remote/adlab/Max2 endpoint is part of the current proof boundary.
- Release manifest after this refresh: `build/current-release-regression-manifest-after-mimo-local-only-preflight-20260606.json`.
- Current release state remains: `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
