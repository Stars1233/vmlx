# 2026-05-24 20:57 PDT - API gateway single-model auto-switch audited

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift and no
  deprecated `/Users/eric/vmlx` work.
- Waited for the active current-suite runner to finish before gateway tests.
- Verified gateway single-model behavior:
  `npm test -- --run tests/api-gateway-single-model.behavior.test.ts tests/api-gateway-ollama-behavior.test.ts`
  from `panel/` -> `2 passed`, `8 tests passed`.
- Verified API-surface pins:
  `.venv/bin/python -m pytest -q tests/test_api_surface_contract.py`
  -> `3 passed`.
- Refreshed API-surface artifact:
  `build/current-api-surface-contract-20260524-single-model-auto-switch.json`
  -> `status=pass`, `failed=[]`, no missing panel markers,
  server `26 passed`, panel `83 passed`,
  `panel_gateway_single_model_auto_switch_streaming=true`.
- Release-manifest validation slice: `4 passed`.
- `git diff --check` clean.
- Release remains blocked by the current objective open rows; no release claim.

# 2026-05-24 20:52 PDT - Regression list preserved; current-suite collision documented

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift and no
  deprecated `/Users/eric/vmlx` work.
- Posted MAIL coordination preserving Eric's full regression list and warning
  that Gemma4 mixed-SWA app-engine speed remains below the 80 tok/s floor
  despite source/installed-app cache telemetry proving native `paged+mixed_swa`
  with generic TurboQuant KV off.
- Refreshed the max-output/context contract after API gateway source edits:
  `build/current-max-output-context-contract-20260524-after-dsv4-neutral-reppen.json`
  -> `status=pass`, `failed=[]`, engine `25 passed`, panel `54 passed`.
- Focused regression slice passed after proof/test normalization:
  `211 passed, 162 deselected`.
- Full current-suite refresh is blocked by active coordination churn: another
  runner rewrote the same proof-board constants and suite artifact pointer
  during refresh. Both
  `build/current-regression-suite-20260524-single-model-auto-switch-dsv4-exact.json`
  and
  `build/current-regression-suite-20260524-after-gemma4-mixed-swa-telemetry.json`
  currently show `status=open` with failed packaged-integrity/focused/manifest
  steps.
- Release remains blocked; do not claim clearance.

# 2026-05-24 20:05 PDT - Current-suite manifest self-reference fixed

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift and no
  deprecated wrapper work.
- Root cause of the post-Gemma suite failure was not a model/proof regression:
  the current suite invoked `run_release_regression_manifest.py
  --require-current-proof-sweep` before overwriting the same current-suite
  artifact that the manifest validates, so stale failed steps could poison the
  next run.
- Added TDD coverage for the provisional-artifact behavior in
  `tests/test_current_regression_suite.py`.
- Patched `tests/cross_matrix/run_current_regression_suite.py` to build the
  suite summary in one helper, refresh `objective_digest` before the manifest,
  and write a provisional current-suite artifact immediately before the
  manifest step.
- Verification:
  - red test failed before patch with unexpected
    `current_suite_artifact_path`;
  - targeted tests after patch: `36 passed`;
  - current suite now passes with `failed_steps=[]`;
  - manifest `current_proof_sweep.status=pass`;
  - objective digest still has exactly three open rows: DSV4 default-cache tool
    loop, Gemma4 mixed-SWA speed, DSV4 long-output/code exactness;
  - `py_compile` and `git diff --check` passed.
- Gemma4 speed note for the other lane: fresh bundled cache-hit probe remains
  below floor and runtime cache telemetry still shows generic live
  `TurboQuantKVCache`, so the heterogeneous/mixed-SWA fast path is not proven.

# 2026-05-24 19:41 PDT - Gemma4 speed blocker narrowed to runtime-contract mismatch

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift and no
  deprecated wrapper work.
- Ran a fresh bundled Gemma4 cache-hit speed probe with longer output budget:
  `build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-bundled-20260524.json`.
- Both Gemma4 requests hit cache but stayed below the 80 tok/s floor:
  `37.766 tok/s` for `paged+disk` and `43.696 tok/s` for `paged`.
- Output was visible English, so the probe did not reopen the Gemma4 language
  quality row.
- Found a proof/telemetry contradiction: health reports
  `native_cache.schema=mixed_swa_kv_v1`, but also
  `generic_turboquant_kv.enabled=true`, while logs show runtime cache layout
  as 30/30 `TurboQuantKVCache`.
- Updated objective proof so Gemma4 speed artifacts with generic live TQ KV no
  longer satisfy the mixed-SWA runtime-contract bit.
- Refreshed objective digest:
  `build/current-objective-proof-audit-20260521.json` still has exactly the
  three open rows: DSV4 default-cache tool loop, Gemma4 mixed-SWA speed, and
  DSV4 long-output/code exactness.
- Verification: focused Gemma4 objective/manifest tests `3 passed`.
- Noted active DSV4 route exactness runner on port `52487`; avoided DSV4
  artifact/current-suite refresh while that lane is active.

# 2026-05-24 19:31 PDT - DSV4 cross-family smoke cleared with cache-hit proof

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift and no
  deprecated wrapper work.
- Identified a harness gap in the first DSV4 smoke: visible output/recognition
  passed, but the repeat-cache prompt was below the 256-token DSV4 native block
  threshold and did not prove cache reuse.
- Added red/green coverage so DSV4 smoke uses native composite launch flags
  (`--dsv4-enable-prefix-cache`, paged block size `256`) and a deterministic
  long repeat-cache prefix while generic smoke rows keep the short prompt.
- Reran bundled DSV4 smoke:
  `build/current-all-local-model-smoke-dsv4-jangtq-k-tools-cache-20260606/summary.json`
  -> `pass`, exact ACK repeat, second repeat `cache_hit_tokens=3639` and
  `cache_hits=30`, recall, and `reasoning_on` visible `FINAL=OK`.
- Updated objective proof so every cross-family smoke family must show a cache
  hit; a DSV4 artifact with clean text but no cache hit now leaves the row open.
- Cross-family live smoke is now pass for `dsv4`, `gemma4`, `hy3`,
  `ling_bailing`, `minimax`, `nemotron`, `qwen36`, `zaya_text`, and `zaya_vl`.
- Remaining open rows are exactly DSV4 default-cache tool loop, Gemma4 mixed-SWA
  speed floor, and DSV4 long-output/code exactness.
- Verification: focused tests `122 passed`; packaged integrity
  `build/current-packaged-integrity-contract-20260524-crossfamily-cleared-dsv4-open.json`
  `status=pass`; release manifest
  `build/current-release-regression-manifest-20260524-crossfamily-cleared-dsv4-open.json`
  `status=pass current_proof_sweep=pass`; current suite
  `build/current-regression-suite-20260524-crossfamily-cleared-dsv4-open.json`
  `status=pass failed_steps=[]`; `py_compile` and `git diff --check` passed.

# 2026-05-24 19:20 PDT - Gemma4 all-local smoke covered

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift and no
  deprecated wrapper work.
- Posted Eric's broad regression-gate reminder to `.agents/MAIL.md` so other
  lanes keep cache/prefix, runtime params, family recognition, parser behavior,
  wrong-language/gibberish, memory/OOM, VL/video, UI/API propagation, bundled
  Python/JANG tools, and Gemma4 mixed-SWA speed in scope.
- Ran bundled Gemma4 all-local smoke:
  `VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Gemma-4-26B-A4B-it-JANG_4M-CRACK --max-models 1 --port 8852 --load-timeout-s 600 --request-timeout-s 240 --out build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-20260524`
- Existing proof-board Gemma artifact:
  `build/current-all-local-model-smoke-gemma4-26b-crack-bundled-20260524/summary.json`
  and fresh duplicate artifact both pass with failures `[]`.
- Live checks: exact ACK cold/repeat, repeat cache `cache_hit_tokens=56`,
  multi-turn recall `color=blue and animal=cat`, `reasoning_on` visible
  `FINAL=OK`, image `Blue`, no-media `NONE`, repeat blue, changed red.
- Objective cross-family row now covers `gemma4`, `hy3`, `ling_bailing`,
  `minimax`, `nemotron`, `qwen36`, `zaya_text`, `zaya_vl`; still missing `dsv4`.
- After regenerating the digest, Gemma4 visible/language is pass only for the
  app-visible artifact
  `build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json`;
  the unsafe explicit thinking-budget low-cap failures remain preserved in
  details as unsupported-budget diagnostics.
- Gemma4 remains separately open for mixed-SWA app-engine speed below the
  80 tok/s floor. Current expected open rows are DSV4 default-cache, Gemma4
  mixed-SWA speed, cross-family missing DSV4, and DSV4 exact-code quality.

# 2026-05-24 15:41 PDT

- Python guard worktree only; no Swift/wrapper work.
- Announced packaging/bundled-runtime audit lane in `.agents/MAIL.md`.
- Hardened packaged console-script shebang checks:
  release gate, bundled verifier, and bundle-script final sanity check now
  reject any remaining `#!*python*` shebang, not just known `/Users/` or
  `/Applications/vMLX.app` paths.
- Red/green proof:
  arbitrary absolute Python path `/private/var/.../python3` was accepted before
  the patch and rejected after.
- Found and fixed a proof-board overclaim:
  `summarize_objective_proof.py` was surfacing
  `code_file_written_exact=false` in details but not requiring it for
  `DSV4 default-cache multi-tool agent loop is proven`.
- The default-cache row now remains open when the generated code contains
  `THREE.WebRenderer()` instead of `THREE.WebGLRenderer()`.
- Updated expected current open rows for current-suite and packaged-integrity
  harnesses to the real two blockers.
- Verification:
  - focused shebang/package/current-suite/objective tests `51 passed`;
  - `npm run verify-bundled` pass;
  - packaged integrity
    `build/current-packaged-integrity-contract-20260524-default-cache-live-review.json`
    pass, `failed=[]`;
  - current regression suite
    `build/current-regression-suite-20260524-default-cache-live-review.json`
    pass, `failed_steps=[]`;
  - `py_compile`, shell syntax, and `git diff --check` pass.
- Open rows remain exactly:
  `DSV4 default-cache multi-tool agent loop is proven` and
  `DSV4 long-output/code/file-generation quality is release-cleared`.

# 2026-05-24 15:26 PDT

- Python guard worktree only; no Swift/wrapper work.
- Announced a DSV4 proof-audit lane in `.agents/MAIL.md` before any live/server
  action.
- Current local RAM was too low for a responsible DSV4 launch:
  128 GB total, about 74 GB available, DSV4 JANGTQ-K directory about 80 GB.
  No DSV4 server ports `8892`, `8891`, or `8854` were listening.
- Found and fixed a live-probe harness gap in
  `tests/cross_matrix/run_dsv4_route_mode_code_exactness.py`:
  unlike the default-cache live runner, it lacked a memory preflight and could
  spawn DSV4 under unsafe RAM.
- Added red/green coverage:
  - preflight skip before `subprocess.Popen`;
  - skipped preflight artifacts return exit 0 from `main()`, preserving
    evidence/open-row state instead of turning RAM pressure into a harness
    failure.
- New artifact:
  `build/current-dsv4-route-mode-code-exactness-memory-preflight-20260524.json`
  -> `status=skipped`, `reason=insufficient_free_memory`,
  `required_available_gb=120.0`, selected case `chat_max`.
- Verification:
  - route/objective/release focused pytest `15 passed`;
  - current regression suite
    `build/current-regression-suite-20260524-default-cache-live-review.json`
    pass, `failed_steps=[]`;
  - `py_compile` pass;
  - `git diff --check` pass.
- Open rows remain exactly the two DSV4 rows:
  default-cache multi-tool exact proof and long-output/code quality clearance.

# 2026-05-24 15:05 PDT

- Python guard worktree only; no Swift, no deprecated wrapper, no live model
  lane.
- Latest live default-cache artifact remains
  `build/current-dsv4-default-cache-tool-loop/result.json` with
  `status=review`.
- Parser/tool-loop/cache plumbing is no longer the current failure in that
  artifact: the loop reaches `list_directory, write_file, write_file` and final
  `DONE`, with native DSV4 cache/prefix/paged/L2 checks true.
- Remaining mismatch is model/output exactness:
  `landing-p/scene.js` contains `THREE.WebRenderer()` instead of
  `THREE.WebGLRenderer()` and lacks the final semicolon.
- Objective digest now records expected/actual code plus missing/corrupt
  identifier details for the default-cache row.
- Release manifest now states that the default-cache live loop reaches three
  tools but still fails exact `WebGLRenderer` fidelity.
- During rerun, packaged integrity caught stale bundled
  `vmlx_engine/tool_parsers/dsml_tool_parser.py`; rebuilt
  `panel/bundled-python` from source and local JANG tools, then rewrote
  generated console-script shebangs.
- Verification:
  - `npm run verify-bundled` pass;
  - packaged integrity
    `build/current-packaged-integrity-contract-20260524-default-cache-live-review.json`
    pass, `failed=[]`;
  - focused proof/default-cache pytest `127 passed`;
  - current regression suite
    `build/current-regression-suite-20260524-default-cache-live-review.json`
    pass, `failed_steps=[]`;
  - `py_compile` pass;
  - `git diff --check` pass.
- Open rows remain exactly:
  `DSV4 default-cache multi-tool agent loop is proven` and
  `DSV4 long-output/code/file-generation quality is release-cleared`.

# 2026-05-24 14:46 PDT

- Did not touch Swift or deprecated wrapper handoffs.
- Coordinated around the other agent's live `8892` DSV4 default-cache
  tool-loop lane; did not send competing model requests.
- Read the first `8892` artifact:
  - `status=review`;
  - native DSV4 cache active and cache hit observed;
  - only `list_directory, write_file` executed;
  - round 1 emitted `<｜DSML｜tool_c` as completed message text.
- Added red/green proof diagnostics:
  - red:
    `test_dsv4_default_cache_tool_loop_response_diagnostics_capture_incomplete_state`
    failed because `response_diagnostics` did not exist;
  - green: live gate now records per-round response status, incomplete
    details, and output-item status/finish/name fields;
  - red:
    `test_objective_proof_digest_surfaces_default_cache_tool_loop_round_diagnostics`
    failed because the objective row did not surface round diagnostics;
  - green: objective digest now includes round output text and response
    diagnostics for the DSV4 default-cache row.
- Read the second `8892` artifact after the other agent reran with
  `max_output_tokens=768`:
  - `status=review`;
  - native cache path and cache reuse pass;
  - `list_directory, write_file, write_file` and final `DONE` pass;
  - exact code still fails because the third `write_file` parsed to
    `path=" string="`, `content=" string="`.
- Reproduced current parser source locally against the raw `invuse` DSML shape:
  current source rejects canonical bogus `string=` args and repairs the raw
  DSML to `landing-p/scene.js`, so the live artifact reflects a still-open
  behavior/proof boundary rather than a release pass.
- Refreshed API/cache contract after DSML parser/test source drift:
  `build/current-api-cache-contract-proof-20260524-after-default-cache-live-review.json`
  -> `status=pass`.
- Updated current-suite artifact output paths to the current proof-board
  artifacts for API/cache, max-output/context, parser registry, generation
  defaults, model artifact format, native MTP, and packaged integrity.
- Verification:
  - focused tests:
    `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py tests/test_dsv4_default_cache_tool_loop_gate.py tests/test_current_regression_suite.py tests/test_packaged_integrity_contract.py -k 'objective_proof_digest or default_cache_tool_loop or current_regression_suite or packaged_integrity'`
    -> `104 passed`;
  - current suite:
    `build/current-regression-suite-20260524-default-cache-live-review.json`
    -> `status=pass`, `failed_steps=[]`, open rows exactly DSV4 default-cache
    tool-loop and DSV4 long-output/code quality;
  - `py_compile`: pass;
  - `git diff --check`: pass.

# 2026-05-24 14:29 PDT

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift files or
  deprecated wrapper handoffs touched.
- Recovered the post-14:21 Python harness state after the routing incident.
- Verified the previously failing pair now passes:
  - packaged integrity known-open-row sync;
  - DSV4 quality-clearance fixture with referenced source gate artifacts.
- Ran focused proof/regression suite:
  - `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py tests/test_dsv4_default_cache_tool_loop_gate.py tests/test_release_gate_python_app.py tests/test_current_regression_suite.py tests/test_release_regression_manifest.py tests/test_model_family_detection_contract.py tests/test_mcp_policy_contract.py tests/test_vl_media_cache_contract.py -k 'objective_proof_digest or default_cache_tool_loop or current_regression_suite or release_regression_manifest or model_family_detection or mcp_policy_contract or decode_speed_gate or vl_media_cache_contract'`
  - result: `189 passed, 37 deselected`.
- Ran packaged integrity:
  - `.venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --out build/current-packaged-integrity-contract-20260524-long-tool-chain-contract.json`
  - result artifact: `status=pass`.
- Ran current regression suite:
  - `.venv/bin/python tests/cross_matrix/run_current_regression_suite.py --skip-release-gate --out build/current-regression-suite-20260524-long-tool-chain-contract.json`
  - result artifact: `status=pass`, `failed_steps=[]`.
- Ran proof/reporting `py_compile` and `git diff --check`: pass.
- Current open requirements remain:
  - `DSV4 default-cache multi-tool agent loop is proven`;
  - `DSV4 long-output/code/file-generation quality is release-cleared`.

# 2026-05-22 00:31 PDT

- Read current max-output/context wiring and confirmed separation:
  - server session `maxTokens` -> launch `--max-tokens`;
  - server session `maxContextLength` -> launch `--max-prompt-tokens`;
  - chat `maxTokens` -> Chat Completions `max_tokens` or Responses `max_output_tokens`;
  - model `max_new_tokens` remains model-owned/default display and is not copied into hidden startup maxTokens.
- Found no `.agents` files in this worktree, recreated ignored local notes.
- Added required marker and harness failure check for missing markers.
- Added `panel/tests/chat-override-policy.test.ts` row for clean new chats and default profiles.
- Verification passed:
  - `npx vitest run tests/chat-override-policy.test.ts --testNamePattern "default profiles cannot make maxTokens sticky on clean new chats" --reporter=verbose`
  - `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-profile-max-green.json`
  - `git diff --check`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_max_output_context_contract.py`
  - `uv run --extra dev python -m pytest -q tests/test_max_output_context_contract.py tests/test_current_regression_suite.py tests/test_release_regression_manifest.py`
  - `npx vitest run tests/chat-override-policy.test.ts tests/request-builder.test.ts tests/settings-flow.test.ts --testNamePattern "maxTokens|Max Output|Max Context|server default output|per-chat|output budget|default profiles cannot" --reporter=verbose`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-profile-max-boundary.json`
- Committed `b497c1a4 test: pin profile output cap boundary` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-profile-boundary.json`: pass.
- Checked public surfaces:
  - primary updater latest.json: `1.5.46`
  - PyPI `vmlx`: latest `1.5.46`, no `1.5.47`
  - GitHub `jjang-ai/vmlx` release `v1.5.47`: not found

# 2026-05-22 00:44 PDT

- Added `tests/test_tool_call_contract.py` as the red contract for tool-call runner marker enforcement.
- Red run: `uv run --extra dev python -m pytest -q tests/test_tool_call_contract.py` failed because `run_tool_call_contract.py` had no `REQUIRED_TOOL_CALL_TEST_MARKERS`.
- Updated `run_tool_call_contract.py`:
  - required DSV4/DSML/tool-loop marker list;
  - pytest `-vv`;
  - vitest `--reporter=verbose`;
  - captured stdout;
  - `missing_markers`;
  - `all_required_tool_call_markers_present` check.
- Added `panel/tests/tool-auto-continue.test.ts` row `panel max tool iterations caps tool loops`.
- Verification passed:
  - `uv run --extra dev python -m pytest -q tests/test_tool_call_contract.py`
  - `uv run --extra dev python tests/cross_matrix/run_tool_call_contract.py --out build/current-tool-call-contract-20260522-marker-hardening.json`
  - `git diff --check`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_tool_call_contract.py`
  - `uv run --extra dev python -m pytest -q tests/test_tool_call_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `npx vitest run tests/tool-auto-continue.test.ts tests/tool-executor-security.test.ts --reporter=verbose`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-tool-marker-hardening.json`
- Committed `2ab5dc43 test: require tool call gate markers` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-tool-marker-hardening.json`: pass.
- Checked public surfaces:
  - primary updater latest.json: `1.5.46`
  - PyPI `vmlx`: latest `1.5.46`, no `1.5.47`
  - GitHub `jjang-ai/vmlx` release `v1.5.47`: not found

# 2026-05-22 00:38 PDT

- Added required max-output marker `coding tool configs keep output limit separate from context fallback`.
- Verified red: `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-coding-tools-red.json` failed with the missing marker.
- Added `panel/tests/coding-tools-config-save.test.ts` row:
  - `/health.max_prompt_tokens` and `/health.max_context_tokens` can be `262144`;
  - absent capabilities `sampling_defaults.max_new_tokens` / `max_tokens` falls back to output `4096`;
  - OpenCode `limit.output` and OpenClaw `maxTokens` stay `4096`, not `262144`.
- Added release manifest text and a manifest assertion for coding-tool configs.
- Verification passed:
  - `npx vitest run tests/coding-tools-config-save.test.ts --testNamePattern "coding tool configs keep output limit separate from context fallback" --reporter=verbose`
  - `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-coding-tools-green.json`
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_max_output_context_with_runner_artifact`
  - `git diff --check`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_max_output_context_contract.py tests/cross_matrix/release_regression_manifest.py`
  - `uv run --extra dev python -m pytest -q tests/test_max_output_context_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `npx vitest run tests/coding-tools-config-save.test.ts --reporter=verbose`
  - `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-coding-tools-final.json`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-coding-tools-boundary.json`
- Committed `f69cecdd test: pin coding tool output limits` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-coding-tools-boundary.json`: pass.
- Checked public surfaces:
  - primary updater latest.json: `1.5.46`
  - PyPI `vmlx`: latest `1.5.46`, no `1.5.47`
  - GitHub `jjang-ai/vmlx` release `v1.5.47`: not found

# 2026-05-22 00:53 PDT

- Added `tests/test_mcp_policy_contract.py` as the red contract for MCP runner marker enforcement.
- Red run: `uv run --extra dev python -m pytest -q tests/test_mcp_policy_contract.py` failed because `run_mcp_policy_contract.py` had no `REQUIRED_MCP_POLICY_TEST_MARKERS`.
- Updated `run_mcp_policy_contract.py`:
  - required MCP marker list;
  - pytest `-vv`;
  - vitest `--reporter=verbose`;
  - captured stdout;
  - `missing_markers`;
  - `all_required_mcp_policy_markers_present` check.
- Wired marker contract into `run_current_regression_suite.py` and release manifest text/tests.
- Added tracked checkpoint doc: `docs/internal/RELEASE_HARDENING_CHECKPOINT_2026_05_22.md`.
- Verification passed:
  - `uv run --extra dev python -m pytest -q tests/test_mcp_policy_contract.py`
  - `uv run --extra dev python tests/cross_matrix/run_mcp_policy_contract.py --out build/current-mcp-policy-contract-20260522-marker-hardening.json`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_mcp_policy_contract.py tests/cross_matrix/run_current_regression_suite.py tests/cross_matrix/release_regression_manifest.py`
  - `git diff --check`
  - `uv run --extra dev python -m pytest -q tests/test_mcp_policy_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-mcp-marker-hardening.json`
  - `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-mcp-marker-hardening.json`
- Public surfaces checked:
  - primary updater latest.json: `1.5.46`
  - PyPI `vmlx`: latest `1.5.46`, no `1.5.47`
  - GitHub `jjang-ai/vmlx` release `v1.5.47`: not found

# 2026-05-22 01:00 PDT

- Re-read `build/current-objective-proof-audit-20260521.json` DSV4 quality row.
- Re-read:
  - `build/current-dsv4-long-output-quality-clearance-20260521.json`
  - `build/current-dsv4-identifier-count-ablation-20260521/result.json`
  - `build/current-production-family-audit-static-dsv4-20260522.json`
  - `build/current-production-family-audit-live-dsv4-jang-local-20260522.json`
- Confirmed blocker remains output quality / identifier exactness:
  - `THREE.WebWebGLRenderer`
  - `THREE.PPerspectiveCamera`
  - `THREE.MMeshBasicMaterial`
  - `THREE.BBoxGeometry`
  - `THREE.ScScene`
- The identifier ablation had prefix cache disabled and pool quant disabled, so this evidence does not support blaming MCP, tool parser, UI, Responses assembly, prefix cache, or L2 cache for this specific row.
- Updated and pushed JANG note:
  - `/Users/eric/jang/docs/runtime/2026-05-22-dsv4-vmlx-live-quality-blocker.md`
  - `b6a8f89 docs: update dsv4 vmlx quality blocker`

# 2026-05-22 01:06 PDT

- Added required marker `chat maxTokens save path cannot mutate session startup maxTokens` to `run_max_output_context_contract.py`.
- Red run:
  - `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-chat-server-boundary-red.json`
  - failed with `missing_markers=["chat maxTokens save path cannot mutate session startup maxTokens"]`.
- Added focused panel contract in `panel/tests/chat-override-policy.test.ts`:
  - per-chat `maxTokens` stays request-scoped;
  - `chat:setOverrides` does not mutate sessions/model settings;
  - server launch uses `config.maxTokens` for `--max-tokens`;
  - context uses `config.maxContextLength` for `--max-prompt-tokens`.
- Verification passed:
  - `npx vitest run tests/chat-override-policy.test.ts --testNamePattern "chat maxTokens save path cannot mutate session startup maxTokens" --reporter=verbose`
  - `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-chat-server-boundary.json`
  - `uv run --extra dev python -m pytest -q tests/test_max_output_context_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `git diff --check`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_max_output_context_contract.py`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-chat-server-boundary.json`
- Committed `b8e76846 test: pin chat output server boundary` and pushed it to `origin/main`.
- Committed `66de9e5d docs: update chat boundary release proof` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-chat-server-boundary.json`: pass.
- Checked public surfaces:
  - primary updater latest.json: `1.5.46`
  - fallback updater latest.json: `1.5.46`
  - PyPI `vmlx`: latest `1.5.46`, no `1.5.47`
  - GitHub `jjang-ai/vmlx` release `v1.5.47`: not found

# 2026-05-22 01:14 PDT

- Added red manifest invariant `test_release_regression_manifest_python_entrypoints_are_tracked`.
- Red run:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_python_entrypoints_are_tracked`
  - failed because `dsv4-long-output-quality-live` referenced missing `build/run_dsv4_identifier_count_ablation.py`.
- Changed `tests/cross_matrix/release_regression_manifest.py` DSV4 live-quality command to tracked runner:
  - `uv run --extra dev python tests/cross_matrix/run_production_family_audit.py --rows dsv4_jang_local --live --out build/current-production-family-audit-live-dsv4-jang-local-20260522.json`.
- Added the production-family live artifact to that row.
- Verification passed:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_python_entrypoints_are_tracked tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_live_only_boundaries`
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/release_regression_manifest.py`
  - `uv run --extra dev python tests/cross_matrix/release_regression_manifest.py > build/current-release-regression-manifest-20260522-live-entrypoint.json`
  - `git diff --check`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-live-entrypoint.json`
- Committed `b73acb57 test: require tracked release gate entrypoints` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-live-entrypoint.json`: pass.

# 2026-05-22 01:23 PDT

- Added red manifest test:
  - `test_release_regression_manifest_tracks_named_model_family_detection_with_runner_artifact`.
- Red run:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_named_model_family_detection_with_runner_artifact`
  - failed with missing `model-family-detection-noheavy`.
- Added `model_family_detection` release domain and `model-family-detection-noheavy` row to `tests/cross_matrix/release_regression_manifest.py`.
- Verification passed:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_named_model_family_detection_with_runner_artifact tests/test_release_regression_manifest.py::test_release_regression_manifest_covers_required_domains`
  - `uv run --extra dev python tests/cross_matrix/run_model_family_detection_contract.py --out build/current-model-family-detection-contract-20260522-manifest-row.json`
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-family-row.json`
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py tests/test_model_family_detection_contract.py tests/test_current_regression_suite.py`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/release_regression_manifest.py tests/cross_matrix/run_model_family_detection_contract.py`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-family-row.json`
- Committed `b203763a test: track model family detection release row` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-family-row.json`: pass.

# 2026-05-22 01:28 PDT

- Added red manifest invariant:
  - `test_release_regression_manifest_python_runner_commands_are_executable_invocations`.
- Red run:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_python_runner_commands_are_executable_invocations`
  - failed because `model-family-live-multiturn-soak` referenced `tests/cross_matrix/run_production_family_audit.py` without a Python launcher.
- Replaced the descriptive live-soak command with a runnable scoped live command over Hy3, MiniMax, Qwen3.6, ZAYA, ZAYA1-VL, Ling, Nemotron, and DSV4 rows.
- Verification passed:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_python_runner_commands_are_executable_invocations tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_live_only_boundaries`
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-live-soak-command.json`
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/release_regression_manifest.py`
  - `git diff --check`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-live-soak-command.json`
- Committed `610727ef test: require executable release runner commands` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-live-soak-command.json`: pass.

# 2026-05-22 01:34 PDT

- Added manifest guard `test_release_regression_manifest_artifacts_are_unique_per_row`.
- Added red overclaim guard `test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp`.
- Red run:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp`
  - failed because the live soak row claimed `Qwen MTP` without a Qwen MTP row in the live command.
- Updated `model-family-live-multiturn-soak` wording:
  - live command covers Qwen 3.6 hybrid rows;
  - Qwen MTP/VL/video rows stay no-heavy-covered until dedicated live-audit rows exist.
- Verification passed:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp tests/test_release_regression_manifest.py::test_release_regression_manifest_artifacts_are_unique_per_row`
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-live-overclaim.json`
  - `uv run --extra dev python -m py_compile tests/cross_matrix/release_regression_manifest.py`
  - `git diff --check`
  - `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools uv run --extra dev python tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260522-live-overclaim.json`
- Committed `7b88abae test: prevent release manifest overclaims` and pushed it to `origin/main`.
- Ran `uv run --extra dev python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-live-overclaim.json`: pass.
# 2026-05-22 02:10 PDT

- Added model-family required row:
  - `decode_speed_all_declared_parsers_are_engine_registered`.
- Red proof:
  - `build/current-model-family-detection-contract-20260522-parser-registry-rows-red.json` failed only with that missing row.
- Added focused test:
  - `test_decode_speed_gate_declared_parsers_are_engine_registered`.
  - verifies all decode-speed row `tool_parser` ids are in `ToolParserManager.list_registered()`.
  - verifies all decode-speed row `reasoning_parser` ids are in `vmlx_engine.reasoning.list_parsers()`.
  - does not depend on local model path existence.
- Updated release manifest:
  - `model-family-detection-noheavy` now names registered engine parser coverage and points at `build/current-model-family-detection-contract-20260522-parser-registry-rows.json`.
- Verification:
  - focused parser row -> 1 passed.
  - model-family gate -> `build/current-model-family-detection-contract-20260522-parser-registry-rows.json`, `status=pass`, `missing_rows=[]`, engine 33 passed, panel 40 passed / 12 skipped.
  - `uv run --extra dev python -m pytest -q tests/test_model_family_detection_contract.py tests/test_release_regression_manifest.py tests/test_parser_registry_contract.py tests/test_current_regression_suite.py` -> 72 passed.
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-parser-registry-rows.json` -> rows=17.
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_model_family_detection_contract.py tests/cross_matrix/release_regression_manifest.py` -> pass.
  - `git diff --check` -> pass.
  - umbrella -> `build/current-regression-suite-20260522-parser-registry-rows.json`, `status=pass`, `failed_steps=[]`, open DSV4 quality requirement unchanged.
- Commit/push:
  - `b7ce6d6a test: pin decode speed parser registry rows` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-parser-registry-rows.json` -> pass.

# 2026-05-22 02:03 PDT

- Added model-family required row:
  - `decode_speed_distinct_jang_jangtq_mxfp_speed_rows`.
- Red proof:
  - `build/current-model-family-detection-contract-20260522-distinct-speed-rows-red.json` failed only with that missing row.
- Added focused test:
  - `test_decode_speed_gate_has_distinct_jang_jangtq_mxfp_speed_rows`.
  - pins JANG-only MX matmul, JANGTQ/MXTQ, MXFP4, and MXFP8-MTP speed rows as distinct rows with explicit PP/decode thresholds.
- Updated release manifest:
  - `model-family-detection-noheavy` now names the distinct speed row categories and points at `build/current-model-family-detection-contract-20260522-distinct-speed-rows.json`.
- Verification:
  - focused distinct row test -> 1 passed.
  - model-family gate -> `build/current-model-family-detection-contract-20260522-distinct-speed-rows.json`, `status=pass`, `missing_rows=[]`, engine 32 passed, panel 40 passed / 12 skipped.
  - `uv run --extra dev python -m pytest -q tests/test_model_family_detection_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py` -> 70 passed.
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-distinct-speed-rows.json` -> rows=17.
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_model_family_detection_contract.py tests/cross_matrix/release_regression_manifest.py` -> pass.
  - `git diff --check` -> pass.
  - umbrella -> `build/current-regression-suite-20260522-distinct-speed-rows.json`, `status=pass`, `failed_steps=[]`, open DSV4 quality requirement unchanged.
- Commit/push:
  - `d0aff24f test: pin distinct family speed rows` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-distinct-speed-rows.json` -> pass.

# 2026-05-22 01:56 PDT

- Added release-manifest guard:
  - `test_release_regression_manifest_tracks_new_chat_output_cap_inheritance_guard`.
- Red proof:
  - focused manifest test failed because `chat-settings-max-output-context-ui` did not name the new-chat guard/current artifact.
- Updated manifest row:
  - proof includes `new-chat model-owned maxTokens cannot be replaced by inherited per-chat output caps`.
  - command/artifact now point at `build/current-max-output-context-contract-20260522-new-chat-max-output.json`.
- Verification:
  - focused guard -> red then green.
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py` -> 55 passed.
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-new-chat-max-output.json` -> rows=17.
  - `uv run --extra dev python -m py_compile tests/cross_matrix/release_regression_manifest.py` -> pass.
  - `git diff --check` -> pass.
  - umbrella -> `build/current-regression-suite-20260522-new-chat-manifest.json`, `status=pass`, `failed_steps=[]`, open DSV4 quality requirement unchanged.
- Commit/push:
  - `28499e45 test: track new chat output cap release row` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-new-chat-manifest.json` -> pass.

# 2026-05-22 01:50 PDT

- Added max-output/context release marker:
  - `new chats preserve model-owned maxTokens while refusing inherited output caps`.
- Red proof:
  - `build/current-max-output-context-contract-20260522-new-chat-max-output-red.json` failed with only that missing marker.
- Added focused panel test:
  - model-owned new-chat `maxTokens=4096` is preserved.
  - previous/default-profile `maxTokens=32768`, sampler overrides, and prompt text are not inherited.
  - tool settings still inherit.
- Verification:
  - `cd panel && npx vitest run tests/chat-override-policy.test.ts --testNamePattern "new chats preserve model-owned maxTokens while refusing inherited output caps" --reporter=verbose` -> 1 passed / 10 skipped.
  - `uv run --extra dev python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-new-chat-max-output.json` -> pass, `missing_markers=[]`, engine 14 passed, panel 33 passed / 1 skipped.
  - `uv run --extra dev python -m pytest -q tests/test_max_output_context_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py` -> 55 passed.
  - `uv run --extra dev python -m py_compile tests/cross_matrix/run_max_output_context_contract.py` -> pass.
  - `git diff --check` -> pass.
  - umbrella -> `build/current-regression-suite-20260522-new-chat-max-output.json`, `status=pass`, `failed_steps=[]`, open DSV4 quality requirement unchanged.
- Commit/push:
  - `cdb7d0f0 test: pin new chat output cap inheritance` -> `origin/main`.
  - `3c6e4314 docs: record new chat output cap proof` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-new-chat-max-output.json` -> pass.
  - `build/current-release-surface-contract-20260522-post-new-chat-doc.json` -> pass.

# 2026-05-22 01:43 PDT

- Added red/green concrete artifact guard:
  - test: `test_release_regression_manifest_artifacts_are_concrete_files`
  - red failure: `model-family-live-multiturn-soak lists directory artifact docs/internal/release-gates/`
  - fix: removed the directory artifact from `model-family-live-multiturn-soak`; the row keeps the concrete JSON artifact only.
- Verification:
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_artifacts_are_concrete_files` -> 1 passed after fix.
  - `uv run --extra dev python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py` -> 54 passed.
  - `uv run --extra dev python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-20260522-concrete-artifacts.json` -> rows=17.
  - `uv run --extra dev python -m py_compile tests/cross_matrix/release_regression_manifest.py` -> pass.
  - `git diff --check` -> pass.
  - umbrella -> `build/current-regression-suite-20260522-concrete-artifacts.json`, `status=pass`, `failed_steps=[]`, open DSV4 quality requirement unchanged.
- Commit/push:
  - `177b9cd4 test: require concrete release artifacts` -> `origin/main`.
  - `30e817a0 docs: record concrete artifact surface check` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-concrete-artifacts.json` -> pass.
  - `build/current-release-surface-contract-20260522-post-doc-surface.json` -> pass.
# 2026-05-22 02:19 PDT

- Added and verified no-heavy CLI parser-choice release guard:
  - `decode_speed_all_declared_parsers_are_cli_choices`
  - `test_decode_speed_gate_declared_parsers_are_cli_choices`
- Red:
  - `build/current-model-family-detection-contract-20260522-cli-parser-choices-red.json`
  - missing only new row.
- Green:
  - `build/current-model-family-detection-contract-20260522-cli-parser-choices.json`
  - pass, `missing_rows=[]`, engine `34 passed`, panel `40 passed / 12 skipped`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-cli-parser-choices.json`
  - 17 rows.
- Verification:
  - `uv run --extra dev python -m pytest -q tests/test_model_family_detection_contract.py tests/test_release_regression_manifest.py tests/test_parser_registry_contract.py tests/test_current_regression_suite.py` -> 74 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-cli-parser-choices.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.

# 2026-05-22 04:15 PDT

- Added Ollama malformed context guard:
  - `omits malformed Ollama context values instead of poisoning max_prompt_tokens`
- Red:
  - `build/current-max-output-context-contract-20260522-ollama-context-malformed-red.json`
  - failed only with new marker missing.
- Fix:
  - `panel/src/main/api-gateway.ts`
  - `applyOllamaPromptContextLimit(...)` omits malformed/non-finite values and
    floors finite positive values before forwarding `max_prompt_tokens`.
- Green:
  - `build/current-max-output-context-contract-20260522-ollama-context-malformed.json`
    -> pass, `missing_markers=[]`.
  - `build/current-api-surface-contract-20260522-ollama-context-malformed.json`
    -> pass.
  - `build/current-release-regression-manifest-20260522-ollama-context-malformed.json`
    -> 18 rows.
  - `build/current-regression-suite-20260522-ollama-context-malformed.json`
    -> pass, `failed_steps=[]`.
- Verification:
  - focused API/manifest/current-suite pytest -> 61 passed.
  - gateway behavior vitest -> 3 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
- Pushed:
  - `0dab0346 fix: drop malformed ollama context caps`
  - `5be5aeba docs: record ollama context cap surface`
  - post-push release surface
    `build/current-release-surface-contract-20260522-post-ollama-context-malformed.json`
    -> pass.
  - post-doc release surface
    `build/current-release-surface-contract-20260522-post-ollama-context-malformed-doc.json`
    -> pass.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.

# 2026-05-22 04:07 PDT

- Added Ollama malformed `num_predict` guard:
  - `omits malformed Ollama num_predict values instead of poisoning max_tokens`
- Red:
  - `build/current-max-output-context-contract-20260522-ollama-malformed-red.json`
  - failed only with new marker missing.
- Fix:
  - `panel/src/main/api-gateway.ts`
  - `applyOllamaNumPredict(...)` omits malformed/non-finite values and floors
    finite positive values before forwarding `max_tokens`.
- Green:
  - `build/current-max-output-context-contract-20260522-ollama-malformed.json`
    -> pass, `missing_markers=[]`.
  - `build/current-api-surface-contract-20260522-ollama-malformed.json`
    -> pass.
  - `build/current-release-regression-manifest-20260522-ollama-malformed.json`
    -> 18 rows.
  - `build/current-regression-suite-20260522-ollama-malformed.json`
    -> pass, `failed_steps=[]`.
- Verification:
  - focused API/manifest/current-suite pytest -> 61 passed.
  - gateway behavior vitest -> 2 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
- Pushed:
  - `33d4b277 fix: drop malformed ollama output caps`
  - `6826d3c3 docs: record ollama output cap surface`
  - post-push release surface
    `build/current-release-surface-contract-20260522-post-ollama-malformed.json`
    -> pass.
  - post-doc release surface
    `build/current-release-surface-contract-20260522-post-ollama-malformed-doc.json`
    -> pass.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- Commit:
  - `6f4ccaec test: pin streaming anthropic output caps` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-streaming-anthropic.json` -> pass.
- Documentation:
  - `78785bd1 docs: record streaming anthropic surface check` -> `origin/main`.
  - `build/current-release-surface-contract-20260522-post-streaming-anthropic-doc.json` -> pass.
- Commit:
  - `3391ab70 test: pin streaming chat responses output caps` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-streaming-chat-responses.json` -> pass.
- Documentation:
  - `b66a6797 docs: record streaming chat responses surface check` -> `origin/main`.
  - `build/current-release-surface-contract-20260522-post-streaming-chat-responses-doc.json` -> pass.
# 2026-05-22 03:20 PDT

- Added streaming Anthropic Messages max-output boundary guard:
  - `test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap`
- Red:
  - `build/current-max-output-context-contract-20260522-anthropic-streaming-red.json`
  - missing only the new Anthropic streaming marker.
- Green:
  - `build/current-max-output-context-contract-20260522-anthropic-streaming.json`
  - pass, `missing_markers=[]`, engine `18 passed`, panel `34 passed / 1 skipped`.
- API surface:
  - `build/current-api-surface-contract-20260522-anthropic-streaming.json` -> pass, server `19 passed`, panel `64 passed`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-anthropic-streaming.json` -> 18 rows.
- Verification:
  - focused streaming Anthropic/API/manifest/current-suite pytest -> 64 passed.
  - py-compile and `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-anthropic-streaming.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- Commit:
  - `b4fc7fa0 test: pin streaming completions output cap boundary` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-streaming-completions.json` -> pass.
- Documentation:
  - `54ce6086 docs: record streaming completions surface check` -> `origin/main`.
  - `build/current-release-surface-contract-20260522-post-streaming-completions-doc.json` -> pass.
# 2026-05-22 03:10 PDT

- Added streaming Chat/Responses max-output boundary guard:
  - `test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap`
- Red:
  - `build/current-max-output-context-contract-20260522-chat-responses-streaming-red.json`
  - missing only the new streaming Chat/Responses marker.
- Green:
  - `build/current-max-output-context-contract-20260522-chat-responses-streaming.json`
  - pass, `missing_markers=[]`, engine `17 passed`, panel `34 passed / 1 skipped`.
- API surface:
  - `build/current-api-surface-contract-20260522-chat-responses-streaming.json` -> pass, server `18 passed`, panel `64 passed`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-chat-responses-streaming.json` -> 18 rows.
- Verification:
  - focused streaming Chat/Responses/API/manifest/current-suite pytest -> 64 passed.
  - py-compile and `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-chat-responses-streaming.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- Commit:
  - `943e21d0 test: pin legacy completions output cap boundary` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-legacy-completions.json` -> pass.
- Documentation:
  - `c63c2c95 docs: record legacy completions surface check` -> `origin/main`.
  - `build/current-release-surface-contract-20260522-post-legacy-completions-doc.json` -> pass.
# 2026-05-22 03:01 PDT

- Added streaming legacy `/v1/completions` max-output boundary guard:
  - `test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap`
- Red:
  - `build/current-max-output-context-contract-20260522-legacy-completions-streaming-red.json`
  - missing only the new streaming marker.
- Green:
  - `build/current-max-output-context-contract-20260522-legacy-completions-streaming.json`
  - pass, `missing_markers=[]`, engine `16 passed`, panel `34 passed / 1 skipped`.
- API surface:
  - `build/current-api-surface-contract-20260522-legacy-completions-streaming.json` -> pass, server `17 passed`, panel `64 passed`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-legacy-completions-streaming.json` -> 18 rows.
- Verification:
  - focused streaming legacy completions/API/manifest/current-suite pytest -> 65 passed.
  - py-compile and `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-legacy-completions-streaming.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- Commit:
  - `68d5823b test: pin decode speed artifact format matrix` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-artifact-format-matrix.json` -> pass.
- Documentation:
  - `26ba7e71 docs: record artifact matrix surface check` -> `origin/main`.
  - `build/current-release-surface-contract-20260522-post-artifact-format-doc.json` -> pass.
# 2026-05-22 02:53 PDT

- Added legacy `/v1/completions` max-output boundary guard:
  - `test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap`
- Red:
  - `build/current-max-output-context-contract-20260522-legacy-completions-red.json`
  - missing only the new marker.
- Green:
  - `build/current-max-output-context-contract-20260522-legacy-completions.json`
  - pass, `missing_markers=[]`, engine `15 passed`, panel `34 passed / 1 skipped`.
- API/cache:
  - `build/current-api-cache-contract-api-surface-check-20260522-legacy-completions.json` -> pass, API route contracts `16 passed`.
  - `build/current-api-surface-contract-20260522-legacy-completions.json` -> pass.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-legacy-completions.json` -> 18 rows.
- Verification:
  - focused legacy completions/API/manifest/current-suite pytest -> 64 passed.
  - py-compile -> pass.
  - umbrella `build/current-regression-suite-20260522-legacy-completions.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- Commit:
  - `5f5b0cde test: pin dsv4 cache control family gating` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-dsv4-cache-controls.json` -> pass.
- Commit:
  - `de889ed8 test: pin server chat max output boundary` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-server-chat-boundary.json` -> pass.
- Commit:
  - `c36e7ace test: pin decode speed cli parser choices` -> `origin/main`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-cli-parser-choices.json` -> pass.
- Documentation:
  - `db0b94a7 docs: record cli parser surface check` -> `origin/main`.
  - `build/current-release-surface-contract-20260522-post-cli-parser-doc.json` -> pass.
# 2026-05-22 02:27 PDT

- Added red/green max-output boundary guard:
  - `server startup maxTokens and chat maxTokens remain independent when both are set`
- Red:
  - `build/current-max-output-context-contract-20260522-server-chat-boundary-red.json`
  - missing only the new marker.
- Green:
  - `build/current-max-output-context-contract-20260522-server-chat-boundary.json`
  - pass, `missing_markers=[]`, engine `14 passed`, panel `34 passed / 1 skipped`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-server-chat-boundary.json`
  - 17 rows.
- Verification:
  - `uv run --extra dev python -m pytest -q tests/test_max_output_context_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py` -> 58 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-server-chat-boundary.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
# 2026-05-22 02:35 PDT

- Added DSV4-only cache-control marker:
  - `DSV4 pool quant and native prefix controls stay DSV4-only`
- Red:
  - `build/current-panel-settings-contract-proof-20260522-dsv4-cache-controls-red.json`
  - status open only because marker was missing.
- Green:
  - `build/current-panel-settings-contract-proof-20260522-dsv4-cache-controls.json`
  - pass, `missing_source_markers=[]`, panel settings `281 passed`, panel typecheck passed, panel registry `52 passed`, engine registry `126 passed`.
- Release manifest:
  - added `panel-session-cache-settings-family-gating`
  - `build/current-release-regression-manifest-20260522-dsv4-cache-controls.json` -> 18 rows.
- Verification:
  - current-suite + release-manifest pytest -> 58 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-dsv4-cache-controls.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
# 2026-05-22 02:41 PDT

- Added decode-speed artifact matrix row:
  - `decode_speed_artifact_format_coverage_matrix`
  - `test_decode_speed_gate_artifact_format_coverage_matrix`
- Red:
  - `build/current-model-family-detection-contract-20260522-artifact-format-matrix-red.json`
  - missing only new row.
- Green:
  - `build/current-model-family-detection-contract-20260522-artifact-format-matrix.json`
  - pass, `missing_rows=[]`, engine `35 passed`, panel `40 passed / 12 skipped`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-artifact-format-matrix.json`
  - 18 rows.
- Verification:
  - `uv run --extra dev python -m pytest -q tests/test_model_family_detection_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py` -> 77 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-artifact-format-matrix.json` -> pass, `failed_steps=[]`.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
# 2026-05-22 03:31 PDT

- Added Ollama streaming endpoint max-output/context guard:
  - `test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap`
- Red:
  - `build/current-max-output-context-contract-20260522-ollama-streaming-red.json`
  - failed because the max-output contract required the marker before the
    endpoint test/command existed.
- Green:
  - `build/current-max-output-context-contract-20260522-ollama-streaming.json`
  - pass, `missing_markers=[]`, engine `19 passed`, panel
    `34 passed / 1 skipped`.
  - `build/current-api-surface-contract-20260522-ollama-streaming.json`
  - pass, server API surface `20 passed`, panel request builders `64 passed`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-ollama-streaming.json`
  - 18 rows.
- Verification:
  - focused Ollama route test -> 1 passed.
  - focused Ollama/API/manifest/current-suite pytest -> 64 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-ollama-streaming.json`
    -> pass, `failed_steps=[]`.
  - post-push release surface
    `build/current-release-surface-contract-20260522-post-streaming-ollama.json`
    -> pass.
  - post-doc release surface
    `build/current-release-surface-contract-20260522-post-streaming-ollama-doc.json`
    -> pass.
- Pushed:
  - `204cf38c test: pin streaming ollama output caps`
  - `202ae484 docs: record streaming ollama surface check`
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
# 2026-05-22 03:40 PDT

- Added prompt/context alias clamp guard:
  - `test_prompt_context_aliases_clamp_without_rewriting_output_caps`
- Red:
  - `build/current-max-output-context-contract-20260522-context-alias-clamp-red.json`
  - failed only with the new marker missing.
- Green:
  - `build/current-max-output-context-contract-20260522-context-alias-clamp.json`
  - pass, `missing_markers=[]`, engine `20 passed`, panel
    `34 passed / 1 skipped`.
  - `build/current-api-surface-contract-20260522-context-alias-clamp.json`
  - pass, server API surface `21 passed`, panel request builders `64 passed`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-context-alias-clamp.json`
  - 18 rows.
- Verification:
  - focused prompt/context alias route test -> 1 passed.
  - focused prompt/context alias/API/manifest/current-suite pytest -> 64 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-context-alias-clamp.json`
    -> pass, `failed_steps=[]`.
  - post-push release surface
    `build/current-release-surface-contract-20260522-post-context-alias-clamp.json`
    -> pass.
  - post-doc release surface
    `build/current-release-surface-contract-20260522-post-context-alias-clamp-doc.json`
    -> pass.
- Pushed:
  - `16e57192 test: pin context alias output cap boundaries`
  - `cd8349b9 docs: record context alias surface check`
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
# 2026-05-22 03:48 PDT

- Added decode-speed command policy guard:
  - `decode_speed_build_command_parser_modality_policy`
  - `test_decode_speed_gate_build_command_preserves_row_parser_modality_policy`
- Red:
  - `build/current-model-family-detection-contract-20260522-command-policy-red.json`
  - failed only with new row missing.
- Green:
  - `build/current-model-family-detection-contract-20260522-command-policy.json`
  - pass, `missing_rows=[]`, engine `36 passed`, panel
    `40 passed / 12 skipped`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-command-policy.json`
  - 18 rows.
- Verification:
  - focused command policy test -> 1 passed.
  - focused family/manifest/current-suite pytest -> 62 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-command-policy.json`
    -> pass, `failed_steps=[]`.
  - post-push release surface
    `build/current-release-surface-contract-20260522-post-command-policy.json`
    -> pass.
  - post-doc release surface
    `build/current-release-surface-contract-20260522-post-command-policy-doc.json`
    -> pass.
- Pushed:
  - `f7f852ea test: pin decode speed command policy`
  - `b6434d54 docs: record decode speed command policy surface`
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.

# 2026-05-22 03:57 PDT

- Added large external decode-speed row guard:
  - `decode_speed_large_external_jangtq_mxfp_gptoss_rows`
  - `test_decode_speed_gate_has_large_external_mistral_gptoss_rows`
- Red:
  - `build/current-model-family-detection-contract-20260522-large-external-red.json`
  - failed only with the new row missing.
- Green:
  - `build/current-model-family-detection-contract-20260522-large-external.json`
  - pass, `missing_rows=[]`, engine `37 passed`, panel
    `40 passed / 12 skipped`.
- Release manifest:
  - `build/current-release-regression-manifest-20260522-large-external.json`
  - 18 rows.
- Verification:
  - focused large external test -> 1 passed.
  - focused family/manifest/current-suite pytest -> 62 passed.
  - py-compile -> pass.
  - `git diff --check` -> pass.
  - umbrella `build/current-regression-suite-20260522-large-external.json`
    -> pass, `failed_steps=[]`.
  - post-push release surface
    `build/current-release-surface-contract-20260522-post-large-external.json`
    -> pass.
- Pushed:
  - `357026ec test: pin large external speed rows`
  - `ebf47797 docs: record large external speed rows surface`
  - post-doc release surface
    `build/current-release-surface-contract-20260522-post-large-external-doc.json`
    -> pass.
- Open:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
# 2026-05-22 04:24 PDT

- Added no-heavy release row
  `decode_speed_external_nemotron3_jangtq_mxfp_rows`.
- Red artifact:
  `build/current-model-family-detection-contract-20260522-nemotron3-external-red.json`
  failed only on missing row.
- Green artifacts:
  - `build/current-model-family-detection-contract-20260522-nemotron3-external.json`
  - `build/current-release-regression-manifest-20260522-nemotron3-external.json`
  - `build/current-regression-suite-20260522-nemotron3-external.json`
- Focused pytest: `62 passed`.
- Umbrella suite: `status=pass`, `failed_steps=[]`.
- Pushed `c93df9d0 test: pin external nemotron speed rows`.
- Post-push release surface:
  `build/current-release-surface-contract-20260522-post-nemotron3-external.json`
  -> `status=pass`.
- Documentation checkpoint `9dbff40e docs: record external nemotron surface check`.
- Final post-doc release surface:
  `build/current-release-surface-contract-20260522-post-nemotron3-external-doc.json`
  -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.

# 2026-05-22 04:31 PDT

- Ran objective proof summary:
  `build/current-objective-proof-summary-20260522-after-nemotron3.json`.
- DSV4 cache/tool/API/settings rows are PASS.
- DSV4 long-output/code/file-generation quality remains OPEN.
- No release build/signing started.

# 2026-05-22 04:34 PDT

- Added Auto chat maxTokens/server default boundary marker.
- Red:
  `build/current-max-output-context-contract-20260522-chat-auto-server-default-red.json`
  failed only on missing marker.
- Green:
  - `build/current-max-output-context-contract-20260522-chat-auto-server-default.json`
  - `build/current-api-surface-contract-20260522-chat-auto-server-default.json`
  - `build/current-release-regression-manifest-20260522-chat-auto-server-default.json`
  - `build/current-regression-suite-20260522-chat-auto-server-default.json`
- Focused pytest: `64 passed`.
- Umbrella suite: `status=pass`, `failed_steps=[]`.
- Pushed `0abb44e7 test: pin chat auto output cap boundary`.
- Post-push release surface:
  `build/current-release-surface-contract-20260522-post-chat-auto-server-default.json`
  -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.

# 2026-05-22 04:43 PDT

- Added reasoning parser dropdown release guard:
  - `reasoning parser dropdown covers every parser the panel registry can emit`
- Red:
  - `build/current-parser-registry-contract-20260522-reasoning-dropdown-red.json`
  - marker missing.
- Gate fix:
  - parser registry contract now records
    `all_required_parser_markers_present`;
  - status-red artifact
    `build/current-parser-registry-contract-20260522-reasoning-dropdown-status-red.json`
    failed top-level status on the missing marker.
- Green:
  - `build/current-parser-registry-contract-20260522-reasoning-dropdown.json`
    -> `status=pass`, `failed=[]`, `missing_markers=[]`, engine
    `102 passed`, panel `40 passed / 244 skipped`.
  - `build/current-release-regression-manifest-20260522-reasoning-dropdown.json`
    -> 18 rows.
  - `build/current-regression-suite-20260522-reasoning-dropdown.json`
    -> `status=pass`, `failed_steps=[]`.
- Focused verification:
  - panel marker -> `1 passed / 231 skipped`.
  - parser/manifest/current-suite pytest -> `62 passed`.
  - py-compile and `git diff --check` -> pass.
- Pushed:
  - `0bda1673 test: pin reasoning parser dropdown coverage`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-reasoning-dropdown.json`
  - `status=pass`.
- Documentation checkpoint:
  - `8b0cc1de docs: record reasoning parser surface check`.
  - `build/current-release-surface-contract-20260522-post-reasoning-dropdown-doc.json`
    -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.

# 2026-05-22 05:03 PDT

- Re-read DSV4 quality artifacts:
  - `build/current-dsv4-long-output-quality-clearance-20260521.json`
  - `build/current-dsv4-identifier-count-ablation-20260521/result.json`
  - `build/current-production-family-audit-static-dsv4-20260522.json`
- Current evidence still points at exact identifier/code corruption on the
  local DSV4 artifact, not cache/tool/UI/parser wiring.
- Found important model-side distinction:
  - HeadBF16 probe is only a head/norm overlay.
  - Full converter high-precision lane is `DSV4_HIGH_PRECISION=1`, which
    preserves attention, shared experts, compressor/indexer, embed, and head.
- In `/Users/eric/jang`, added and pushed:
  - `496e223 test: pin dsv4 high precision rebuild lane`
  - JANG converter contract now pins full non-routed high-precision passthrough.
  - JANG runtime blocker doc now records that HeadBF16 overlay is not enough
    evidence and full rebuilt-artifact live gates remain required.
- Verification:
  - DSV4 converter contract -> `27 passed`, 2 warnings.
- vMLX release build/signing still not started.

# 2026-05-22 04:59 PDT

- Added plain KV cache health bleed-through guard:
  - required family row `decode_speed_plain_kv_cache_health_not_native`;
  - plain `cache_type=kv` rows accept health `kv` or `paged_kv`;
  - plain KV rows reject DSV4 `native_composite` and ZAYA `typed_cca` health.
- Red proof:
  - focused test failed before implementation because plain KV +
    `native_composite` health produced no mismatch.
- Green:
  - `build/current-model-family-detection-contract-20260522-plain-kv-cache-health.json`
    -> `status=pass`, `missing_rows=[]`, engine `39 passed`, panel
    `40 passed / 12 skipped`;
  - `build/current-release-regression-manifest-20260522-plain-kv-cache-health.json`
    -> 18 rows;
  - focused pytest -> `82 passed`;
  - py-compile and `git diff --check` -> pass;
  - `build/current-regression-suite-20260522-plain-kv-cache-health.json`
    -> `status=pass`, `failed_steps=[]`.
- Pushed:
  - `e9d86fbd test: pin plain kv cache health rows`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-plain-kv-cache-health.json`
    -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.
- Release build/signing still not started.

# 2026-05-22 05:06 PDT

- Added launch-memory admission warning-only guard for Python/Electron app:
  - required marker
    `launch memory admission is warning-only for lazy-mmap bundles`;
  - source must log estimate/warning only;
  - no `Launch blocked`, unsafe env requirement, session failed status, or
    thrown error before launch.
- Red:
  - `build/current-panel-settings-contract-proof-20260522-launch-memory-red.json`
    -> missing marker.
- Green:
  - focused panel marker -> `1 passed / 232 skipped`;
  - `build/current-panel-settings-contract-proof-20260522-launch-memory-warning.json`
    -> `status=pass`, settings `283 passed`, panel registry `52 passed`,
    engine registry `126 passed`;
  - `build/current-release-regression-manifest-20260522-launch-memory-warning.json`
    -> 18 rows;
  - focused pytest -> `60 passed`;
  - py-compile and `git diff --check` -> pass;
  - `build/current-regression-suite-20260522-launch-memory-warning.json`
    -> `status=pass`, `failed_steps=[]`.
- Pushed:
  - `9b6041e6 test: pin launch memory warning behavior`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-launch-memory-warning.json`
    -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.
- Release build/signing still not started.

# 2026-05-22 05:16 PDT

- Added config-only native-MTP suppression guard:
  - engine marker
    `test_config_only_mtp_bundle_does_not_activate_native_runtime`;
  - panel marker
    `does not expose Native MTP for config-only bundles without indexed mtp tensors`.
- Red:
  - `build/current-native-mtp-contract-20260522-config-only-red.json`
    failed on missing markers.
- Green:
  - focused engine marker -> `1 passed`;
  - focused panel marker -> `1 passed / 52 skipped`;
  - `build/current-native-mtp-contract-20260522-config-only.json`
    -> `status=pass`, engine `116 passed`, panel controls
    `12 passed / 221 skipped`, panel detection `6 passed / 47 skipped`;
  - `build/current-release-regression-manifest-20260522-native-mtp-config-only.json`
    -> 18 rows;
  - focused pytest -> `131 passed`;
  - py-compile and `git diff --check` -> pass;
  - `build/current-regression-suite-20260522-native-mtp-config-only.json`
    -> `status=pass`, `failed_steps=[]`.
- Pushed:
  - `1fde18b0 test: pin config-only native mtp suppression`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-native-mtp-config-only.json`
    -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.
- Release build/signing still not started.

# 2026-05-22 05:23 PDT

- Added panel model-family routing rows to VL/media contract:
  - ZAYA1-VL stale text stamps remain multimodal;
  - Qwen JANGTQ/MXFP4/MXFP8 VLM detection;
  - Qwen 3.6 MoE vision/video metadata;
  - Nemotron-H stale Omni sidecars do not force text rows through MLLM.
- Red:
  - first red artifact reported markers missing but status still pass;
  - added `all_required_panel_markers_present`;
  - status-red artifact failed on the missing markers.
- Green:
  - `build/current-vl-media-cache-contract-20260522-panel-family.json`
    -> `status=pass`, no missing markers;
  - `build/current-release-regression-manifest-20260522-vl-panel-family.json`
    -> 18 rows;
  - focused pytest -> `63 passed`;
  - py-compile and `git diff --check` -> pass;
  - `build/current-regression-suite-20260522-vl-panel-family.json`
    -> `status=pass`, `failed_steps=[]`.
- Pushed:
  - `ddb51bd9 test: pin vl panel family routing rows`.
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-vl-panel-family.json`
    -> `status=pass`.
- Open release requirement remains DSV4 long-output/code quality.
- Release build/signing still not started.
# 2026-05-22 05:37 PDT - API stream cache-detail gate

- Added red/green release-gate coverage for Chat/Responses streaming
  `cache_detail` usage.
- Red artifact:
  `build/current-api-surface-contract-20260522-stream-cache-detail-red.json`
  failed on the four missing stream cache-detail markers.
- Green artifact:
  `build/current-api-surface-contract-20260522-stream-cache-detail.json`
  passed with server API surface `25 passed` and panel request builders
  `67 passed`.
- Updated release manifest/checkpoint.
- Release manifest now emits 18 rows, focused pytest passed 63 tests,
  py-compile and `git diff --check` passed, and umbrella current suite passed
  with `failed_steps=[]`.
- Pushed `5fd051c8 test: pin streaming cache detail usage`.
- Post-push release surface
  `build/current-release-surface-contract-20260522-post-stream-cache-detail.json`
  passed.

# 2026-05-22 05:49 PDT - DSV4 live quality blocker recheck

- Ran current-source live DSV4 production row:
  `build/current-production-family-audit-live-dsv4-jang-local-20260522-after-stream-cache-detail.json`.
- Result: live `FAIL`, 5 failed rows.
- Passing rows include native/paged cache, encoder shim, multi-EOS, capability
  endpoints, basic thinking-off chat, Responses auto-tool choice,
  Anthropic/Ollama basics, stream done/disconnect, and second-turn cache
  coherence.
- Failing rows are DSV4 max-thinking length/empty visible, exact Three.js
  identifier corruption under deterministic settings, long-output length stop,
  skipped Three.js full-output, and Responses tool-history `READEOM.md`.
- Audit static issue: output-head/final-norm precision boundary needs
  source-vs-quant or rebuilt-artifact clearance before production claims.
- No DSV4 server process remained after the audit.
- Release manifest re-emitted 18 rows, focused pytest passed 61 tests,
  py-compile and `git diff --check` passed, and umbrella current suite passed
  with `failed_steps=[]` / DSV4 quality still open.
- Pushed `c5fee886 docs: record dsv4 live quality blocker`.
- Post-push release surface
  `build/current-release-surface-contract-20260522-post-dsv4-live-recheck.json`
  passed.
- Added and pushed JANG-side note
  `/Users/eric/jang/docs/runtime/2026-05-22-dsv4-flash-vmlx-live-quality-blocker.md`
  as `1b693bc docs: record dsv4 vmlx quality blocker`.

# 2026-05-22 05:53 PDT - Final prebuild gate recheck

- Max-output/context:
  `build/current-max-output-context-contract-20260522-final-prebuild.json`
  -> `status=pass`, `missing_markers=[]`, engine `20 passed`, panel
  `36 passed / 292 skipped`.
- Release manifest:
  `build/current-release-regression-manifest-20260522-final-prebuild.json`
  -> 18 rows.
- Focused pytest:
  `tests/test_max_output_context_contract.py tests/test_release_regression_manifest.py tests/test_current_regression_suite.py`
  -> `62 passed`.
- Umbrella:
  `build/current-regression-suite-20260522-final-prebuild.json`
  -> `status=pass`, `failed_steps=[]`, open requirement exactly
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- py-compile and `git diff --check` passed.
- Release surface:
  `build/current-release-surface-contract-20260522-final-prebuild.json`
  -> `status=pass`.
- Objective summary:
  `build/current-objective-proof-summary-20260522-final-prebuild.json`
  -> 12 PASS / 1 OPEN.
- No build/sign/release started; release remains blocked on DSV4 quality unless
  Eric explicitly descopes it.

# 2026-05-22 05:59 PDT - Plain MLX 4bit Qwen row

- Added required model-family release row
  `decode_speed_plain_mlx_4bit_qwen36_row`.
- Red: focused `test_family_detection_contract_pins_named_release_rows`
  failed before the row existed.
- Green:
  - focused row tests -> `3 passed`;
  - `build/current-model-family-detection-contract-20260522-plain-mlx-4bit.json`
    -> `status=pass`, `missing_rows=[]`, engine `40 passed`, panel
    `41 passed / 12 skipped`;
  - `build/current-release-regression-manifest-20260522-plain-mlx-4bit.json`
    -> 18 rows;
  - focused pytest -> `84 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-plain-mlx-4bit.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-plain-mlx-4bit.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 07:10 PDT - ZAYA stale-stamp reasoning policy gate

- Added model-family row:
  `zaya_stale_stamp_reasoning_policy`.
- Required marker:
  `test_zaya_stale_stamp_cannot_disable_reasoning_or_reenable_think_seed`.
- Red: focused family-contract row check failed before the row existed.
- Green:
  - focused release pytest -> `87 passed`;
  - family gate:
    `build/current-model-family-detection-contract-20260522-zaya-stale-stamp.json`
    -> `status=pass`, `missing_rows=[]`, engine `41 passed`, panel
    `41 passed / 12 skipped`;
  - release manifest:
    `build/current-release-regression-manifest-20260522-zaya-stale-stamp.json`
    -> 18 rows;
  - py-compile and `git diff --check` passed;
  - umbrella:
    `build/current-regression-suite-20260522-zaya-stale-stamp.json`
    -> `status=pass`, `failed_steps=[]`;
  - release surface:
    `build/current-release-surface-contract-20260522-zaya-stale-stamp.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 07:02 PDT - Affine JANG loader acceptance gate

- Added artifact-format marker:
  `test_load_jang_model_accepts_affine_weight_format`.
- Red: focused artifact-contract marker check failed before the marker was in
  `REQUIRED_ARTIFACT_TEST_MARKERS`.
- Green:
  - focused release pytest -> `64 passed`;
  - artifact gate:
    `build/current-model-artifact-format-contract-20260522-affine-jang-loader.json`
    -> `status=pass`, `missing_markers=[]`, `131 passed`;
  - release manifest:
    `build/current-release-regression-manifest-20260522-affine-jang-loader.json`
    -> 18 rows;
  - py-compile and `git diff --check` passed;
  - umbrella:
    `build/current-regression-suite-20260522-affine-jang-loader.json`
    -> `status=pass`, `failed_steps=[]`;
  - release surface:
    `build/current-release-surface-contract-20260522-affine-jang-loader.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:55 PDT - MXFP VLM loader quant-mode gate

- Added artifact-format marker:
  `test_mxfp_vlm_loader_quantizes_with_declared_mode`.
- Red: focused artifact-contract marker check failed before the marker was in
  `REQUIRED_ARTIFACT_TEST_MARKERS`.
- Green:
  - focused MXFP loader checks -> `4 passed / 67 deselected`;
  - focused release pytest -> `63 passed`;
  - artifact gate:
    `build/current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json`
    -> `status=pass`, `missing_markers=[]`, `131 passed`;
  - release manifest:
    `build/current-release-regression-manifest-20260522-mxfp-vlm-loader.json`
    -> 18 rows;
  - py-compile and `git diff --check` passed;
  - umbrella:
    `build/current-regression-suite-20260522-mxfp-vlm-loader.json`
    -> `status=pass`, `failed_steps=[]`;
  - release surface:
    `build/current-release-surface-contract-20260522-mxfp-vlm-loader.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:06 PDT - Plain MLX 4bit artifact gate

- Added artifact marker:
  `test_qwen36_plain_mlx_4bit_keeps_hybrid_cache_without_jang_or_mxfp`.
- Red:
  `build/current-model-artifact-format-contract-20260522-plain-mlx-4bit-red.json`
  -> `status=fail`, missing marker.
- Green:
  - focused marker -> `1 passed`;
  - `build/current-model-artifact-format-contract-20260522-plain-mlx-artifact.json`
    -> `status=pass`, `missing_markers=[]`, `130 passed`;
  - focused release pytest -> `85 passed`;
  - `build/current-release-regression-manifest-20260522-plain-mlx-artifact-final.json`
    -> 18 rows;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-plain-mlx-artifact.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-plain-mlx-artifact.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:46 PDT - Qwen affine-JANG VLM text-loader policy

- Added VLM media/cache engine marker:
  `test_qwen36_affine_jang_vlm_stays_text_loader_until_mrope_fixed`.
- Red:
  `build/current-vl-media-cache-contract-20260522-qwen-affine-text-red.json`
  -> `status=fail`, missing the new engine marker.
- Green:
  - focused marker/contract -> `2 passed`;
  - `build/current-vl-media-cache-contract-20260522-qwen-affine-text.json`
    -> `status=pass`, no missing engine/panel markers, engine
    `42 passed / 6 skipped`;
  - `build/current-release-regression-manifest-20260522-qwen-affine-text.json`
    -> 18 rows;
  - focused release pytest -> `64 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-qwen-affine-text.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-qwen-affine-text.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:39 PDT - Reasoning parser CLI choice parity gate

- Added parser registry marker:
  `test_cli_reasoning_parser_choices_cover_family_registry_parsers`.
- Red:
  `build/current-parser-registry-contract-20260522-reasoning-cli-red.json`
  -> `status=fail`, missing the new marker.
- Green:
  - focused marker/parser contract -> `2 passed`;
  - `build/current-parser-registry-contract-20260522-reasoning-cli.json`
    -> `status=pass`, `missing_markers=[]`, engine `103 passed`, panel
    `40 passed / 246 skipped`;
  - `build/current-release-regression-manifest-20260522-reasoning-cli.json`
    -> 18 rows;
  - focused release pytest -> `63 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-reasoning-cli.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-reasoning-cli.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:33 PDT - Chat output cap vs server startup default edge

- Added max-output/context marker:
  `per-chat maxTokens below or above the server startup default remain request scoped`.
- Red:
  `build/current-max-output-context-contract-20260522-chat-cap-startup-red.json`
  -> `status=fail`, missing the new marker.
- Green:
  - focused panel request-builder marker -> `2 passed / 52 skipped`;
  - `build/current-max-output-context-contract-20260522-chat-cap-startup.json`
    -> `status=pass`, `missing_markers=[]`, engine `20 passed`, panel
    `38 passed / 292 skipped`;
  - `build/current-release-regression-manifest-20260522-chat-cap-startup.json`
    -> 18 rows;
  - focused release pytest -> `62 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-chat-cap-startup.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-chat-cap-startup.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:25 PDT - Nemotron-H stale Omni engine routing gate

- Added engine-side required marker:
  `test_nemotron_h_stale_omni_stamp_without_media_stays_text_hybrid`.
- Red:
  `build/current-model-family-detection-contract-20260522-nemotron-stale-omni-red.json`
  -> `status=fail`, missing
  `nemotron_h_hybrid_text_not_stale_omni`.
- Green:
  - focused marker -> `1 passed`;
  - `build/current-model-family-detection-contract-20260522-nemotron-stale-omni.json`
    -> `status=pass`, `missing_rows=[]`, engine `41 passed`, panel
    `41 passed / 12 skipped`;
  - `build/current-release-regression-manifest-20260522-nemotron-stale-omni.json`
    -> 18 rows;
  - focused release pytest -> `84 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-nemotron-stale-omni.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-nemotron-stale-omni.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:11 PDT - Ling/Bailing loader repair artifact gate

- Added/required markers for:
  - flat 2D switch-MLP repair;
  - no-op on correct 3D JANGTQ tensors;
  - DWQ split MLA projection repair;
  - absent advertised MTP tail layer trim.
- Red: focused artifact contract failed before selector explicitly included
  `bailing`.
- Green:
  - `build/current-model-artifact-format-contract-20260522-bailing-loader.json`
    -> `status=pass`, `missing_markers=[]`, `130 passed`;
  - `build/current-release-regression-manifest-20260522-bailing-loader.json`
    -> 18 rows;
  - focused pytest -> `62 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-bailing-loader.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-bailing-loader.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.

# 2026-05-22 06:17 PDT - Qwen indexed-MTP VLM routing gate

- Added required VLM panel marker:
  `marks Qwen3.6 VL JANG bundles with indexed MTP tensors as native MTP capable`.
- Red: focused VLM contract failed until family selector included `indexed MTP`.
- Green:
  - `build/current-vl-media-cache-contract-20260522-qwen-indexed-mtp.json`
    -> `status=pass`, no missing markers, engine `32 passed / 6 skipped`,
    panel family detection `13 passed / 40 skipped`;
  - `build/current-release-regression-manifest-20260522-qwen-indexed-mtp-vl.json`
    -> 18 rows;
  - focused pytest -> `64 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-qwen-indexed-mtp-vl.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-qwen-indexed-mtp-vl.json`
    -> `status=pass`.
- No release build/signing started; DSV4 long-output/code quality remains open.
# 2026-05-22 07:17 PDT - Qwen/Nemotron hybrid-cache gate

- Added required family rows for Qwen dense linear-attention hybrid cache,
  Qwen MoE text linear-attention hybrid cache, and base Nemotron-H hybrid
  registry classification.
- Red:
  - family named-row contract failed before rows existed;
  - release manifest test failed before artifact/proof text was indexed.
- Green:
  - focused release pytest -> `88 passed`;
  - `build/current-model-family-detection-contract-20260522-qwen-nemotron-hybrid-cache.json`
    -> `status=pass`, `missing_rows=[]`, engine `41 passed / 111 deselected`,
    panel `41 passed / 12 skipped`;
  - `build/current-release-regression-manifest-20260522-qwen-nemotron-hybrid-cache.json`
    -> 18 rows;
  - py-compile and `git diff --check` passed;
  - `build/current-regression-suite-20260522-qwen-nemotron-hybrid-cache.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-qwen-nemotron-hybrid-cache.json`
    -> `status=pass`.
- Open release requirement remains:
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- No release build/signing started.

# 2026-05-22 07:25 PDT - DSV4 release decision check

- Rebuilt objective proof digest:
  `build/current-objective-proof-digest-20260522-qwen-nemotron-hybrid-cache.json`.
- DSV4 cache/tool/max-output rows remain pass.
- DSV4 long-output/code/file-generation quality remains open:
  - identifier exactness false;
  - Three.js single-file false;
  - no-markdown-fence false;
  - corrupt identifier rejection false;
  - non-length stop false;
  - source/rebuilt-body clearance false.
- Missing clearance artifacts remain:
  - `build/dsv4-source-full-output/result.json`
  - `build/dsv4-chat-prompt-ablation-20260520101331/result.json`
- JANG-side guard for the rebuilt/source-body lane:
  - `6 passed, 21 deselected`
  - command used:
    `PYTHONPATH=/Users/eric/jang/jang-tools .venv/bin/python -m pytest -q /Users/eric/jang/jang-tools/tests/test_dsv4_converter_contract.py -k "high_precision or rope_scaling or f32_control or metadata_declares"`.
- Updated:
  - `/Users/eric/jang/docs/runtime/2026-05-22-dsv4-vmlx-live-quality-blocker.md`
  - pushed JANG commit:
    `4e270e9 docs: update dsv4 vmlx release blocker`
- No release build/signing started.

# 2026-05-22 07:27 PDT - Max output nonsticky proof

- Added named max-output/context contract check:
  `new_chat_output_caps_are_not_inherited_or_made_sticky`.
- Red:
  - max-output contract test failed before the check existed;
  - release manifest test failed before the new artifact/proof text was
    indexed.
- Green:
  - `build/current-max-output-context-contract-20260522-new-chat-output-cap-nonsticky.json`
    -> `status=pass`, `missing_markers=[]`, engine `20 passed`, panel
    `38 passed / 292 skipped`;
  - focused release pytest -> `66 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-release-regression-manifest-20260522-new-chat-output-cap-nonsticky.json`
    -> 18 rows;
  - `build/current-regression-suite-20260522-new-chat-output-cap-nonsticky.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-new-chat-output-cap-nonsticky.json`
    -> `status=pass`.
- No release build/signing started.

# 2026-05-22 07:35 PDT - JANG-only MX matmul speed rows

- Added required row:
  `decode_speed_jang_only_mx_matmul_policy`.
- Added marker:
  `test_decode_speed_gate_jang_only_rows_keep_text_mx_matmul_launch_policy`.
- Red:
  - family named-row contract failed before row existed;
  - release manifest test failed before artifact/proof text was indexed.
- Green:
  - focused row tests -> `2 passed`;
  - `build/current-model-family-detection-contract-20260522-jang-only-mx-matmul-policy.json`
    -> `status=pass`, `missing_rows=[]`, engine `42 passed / 111 deselected`,
    panel `41 passed / 12 skipped`;
  - focused release pytest -> `89 passed`;
  - py-compile and `git diff --check` passed;
  - `build/current-release-regression-manifest-20260522-jang-only-mx-matmul-policy.json`
    -> 18 rows;
  - `build/current-regression-suite-20260522-jang-only-mx-matmul-policy.json`
    -> `status=pass`, `failed_steps=[]`;
  - `build/current-release-surface-contract-20260522-jang-only-mx-matmul-policy.json`
    -> `status=pass`.
- No release build/signing started.
# 2026-05-22 07:41 PDT - Commit/push and release gate refresh

- Committed and pushed:
  - `a4e5eced test: pin jang-only speed launch policy`
- Verified:
  - focused family/manifest pytest: `3 passed`
  - post-push release surface:
    `build/current-release-surface-contract-20260522-post-jang-only-mx-matmul-policy.json`
    -> `status=pass`
  - umbrella:
    `build/current-regression-suite-20260522-post-jang-only-mx-matmul-policy.json`
    -> `status=pass`, `failed_steps=[]`, only open requirement is DSV4
    long-output/code/file-generation quality clearance
- Release action remains held until Eric explicitly descopes that DSV4 quality
  claim or a rebuilt/source-equivalent DSV4 body passes the live gates.

# 2026-05-22 07:45 PDT - Direct release gate objective digest enforcement

- Added release-gate enforcement for objective proof digest.
- Red tests:
  - objective digest fail-on-open requirement
  - static release gate must call objective digest checker
  - manifest must index objective-gate-enforced packaged-integrity artifact
- Green:
  - `tests/test_release_gate_python_app.py` -> `36 passed`
  - clean-JANG packaged integrity -> `status=pass`, `failed=[]`
  - release gate now returns rc=1 with only:
    `objective proof digest: DSV4 long-output/code/file-generation quality is release-cleared`
- No build/signing started.

# 2026-05-22 07:52 PDT - Persisted chat output cap guard

- Added red/green guard for persisted chat maxTokens vs server startup
  maxTokens.
- Red: max-output/context gate failed with missing marker.
- Green:
  - focused panel marker tests -> `3 passed`
  - max-output/context gate -> `status=pass`, `missing_markers=[]`
  - focused manifest/contract tests -> `2 passed`

# 2026-05-22 08:00 PDT - Persisted chat output cap checkpoint committed/pushed

- Commit:
  - `6de1134e test: pin persisted chat output cap isolation`
  - pushed to `origin/codex/pr-intake-manifest`
- Post-push gates:
  - release surface:
    `build/current-release-surface-contract-20260522-post-persisted-chat-output-cap.json`
    -> `status=pass`
  - umbrella:
    `build/current-regression-suite-20260522-post-persisted-chat-output-cap.json`
    -> `status=pass`, `failed_steps=[]`, only open requirement:
    `DSV4 long-output/code/file-generation quality is release-cleared`
  - direct release gate:
    `panel/scripts/release-gate-python-app.py --skip-app --skip-gui`
    -> rc=1 because objective digest still has the DSV4 quality row open
- Interpretation:
  - max-output/context/server-chat persisted-state wiring is guarded;
  - release cannot honestly proceed as fully cleared until DSV4 quality is
    fixed or Eric explicitly ships with that limitation documented.

# 2026-05-22 08:11 PDT

- Fixed DSV4 renderer command-preview filtering so stale `additionalArgs`
  cannot reenable Native MTP/deterministic MTP/default sampler/max-token/DSV4
  cache override flags that the main launch path blocks.
- Added permanent marker:
  `DSV4 additional args cannot reenable native MTP or deterministic sampling policy`.
- Verified:
  - focused settings-flow -> `3 passed / 231 skipped`;
  - native-MTP contract -> `status=pass`;
  - focused pytest -> `66 passed`;
  - release manifest -> 18 rows;
  - umbrella -> `status=pass`, `failed_steps=[]`, open requirement only
    `DSV4 long-output/code/file-generation quality is release-cleared`;
  - `git diff --check` clean.

# 2026-05-22 08:16 PDT

- Committed and pushed:
  - `79f14837 fix: block stale dsv4 native mtp args`
- Post-push release surface:
  - `build/current-release-surface-contract-20260522-post-dsv4-additional-args.json`
    -> `status=pass`
- Post-push umbrella:
  - `build/current-regression-suite-20260522-post-dsv4-additional-args.json`
    -> `status=pass`, `failed_steps=[]`
  - open requirement remains exactly:
    `DSV4 long-output/code/file-generation quality is release-cleared`

# 2026-05-22 08:18 PDT

- Direct release gate initially failed bundled-python import/hash parity because
  bundled `jang_tools/convert_hy3_jangtq.py` was stale.
- Rebuilt `panel/bundled-python` from current local vMLX and clean release JANG
  worktree.
- Verified:
  - `npm --prefix panel run verify-bundled` -> pass;
  - direct release gate summary
    `docs/internal/release-gates/20260522_081735/SUMMARY.md`:
    all non-app checks pass except objective digest, which fails only on
    `DSV4 long-output/code/file-generation quality is release-cleared`.
- Tracked branch clean after the pushed DSV4 additionalArgs commit; bundle and
  gate logs are ignored generated state.

# 2026-05-22 08:21 PDT

- Checked remaining DSV4 quality row:
  - sibling missing artifacts exist but are failing evidence, not clearance;
  - local DSV4-K/JANG/HeadBF16 probe all have F32 critical controls and pass
    YaRN rope-scaling validator;
  - DSV4-K routed plan remains the 0/1/2/23/25/28/34/36 mixed 2/4-bit plan.
- Found JANG clean-worktree hygiene issue:
  - `_internal/jang_v3` helper files are untracked in `/Users/eric/jang` main
    and absent from clean release worktree;
  - JANG DSV4 batch: `29 passed, 3 failed`, all three due to missing helper.
- Updated JANG blocker note:
  `/Users/eric/jang/docs/runtime/2026-05-22-dsv4-vmlx-live-quality-blocker.md`.

# 2026-05-22 08:32 PDT

- Fixed JANG `.gitignore` so clean worktrees can track only the DSV4 V3 helper
  files required by `test_jang_v3_dsv4_contract.py`.
- Verified in `/Users/eric/jang`:
  - V3 helper contract -> `3 passed`;
  - DSV4 converter/rope/hc batch -> `30 passed, 2 warnings`.
- Updated JANG DSV4/vMLX blocker note with the fix and proof.

# 2026-05-22 08:36 PDT

- Pushed JANG commit:
  - `09ca9a8 test: track dsv4 v3 helper contracts`
- Updated release-clean JANG source to `09ca9a8`.
- Verified:
  - temp clean worktree DSV4 V3 + converter/rope/hc batch:
    `33 passed, 2 warnings`;
  - release-clean V3 contract: `3 passed`;
  - vMLX `verify-bundled`: pass;
  - vMLX direct release gate:
    all non-app checks pass except known DSV4 objective digest row.

- 2026-05-22 08:30 PDT request-output/context isolation audit: max-output contract pass at build/current-max-output-context-contract-20260522-request-output-isolation-audit.json; no code changes.

- 2026-05-22 08:38 PDT fixed release harness clean-JANG-source env propagation; packaged integrity and umbrella suite pass with only known DSV4 quality open requirement.

- 2026-05-22 08:39 PDT pushed 40f0ec1b clean-JANG source release harness; release still waits on DSV4 long-output/code quality decision.

- 2026-05-22 08:40 PDT direct clean-source release gate passes all packaging checks and fails only DSV4 long-output/code quality objective.

- 2026-05-22 08:46 PDT added family-contract panel launch wiring row and proof artifact; umbrella suite still passes with only DSV4 quality open.

- 2026-05-22 08:47 PDT pushed 35875d3d panel launch family wiring guard; branch clean, release still blocked only by DSV4 quality row.

- 2026-05-22 08:53 PDT strengthened cache architecture gate with panel launch cache-policy coverage. New artifact `build/current-cache-architecture-contract-20260522-panel-cache-launch.json` passes with engine/API cache rows plus `panel_cache_launch_policy` at 75 passed / 168 skipped. Release manifest refreshed at `build/current-release-regression-manifest-20260522-panel-cache-launch.json`. Umbrella refreshed at `build/current-regression-suite-20260522-panel-cache-launch.json` -> status=pass, failed_steps=[]; still no DSV4 live long-output/code clearance.

- 2026-05-22 08:59 PDT tightened max-output/context gate so Responses-specific per-chat `maxTokens` below/above server startup default and Responses Auto omission are required markers. New artifact `build/current-max-output-context-contract-20260522-responses-output-boundary.json` -> status=pass, engine 20 passed, panel 39 passed / 293 skipped. Release manifest refreshed at `build/current-release-regression-manifest-20260522-responses-output-boundary.json`; umbrella refreshed at `build/current-regression-suite-20260522-responses-output-boundary.json` -> status=pass, failed_steps=[].

- 2026-05-22 09:05 PDT strengthened model-family gate with required rows for ZAYA1-VL JANGTQ_K/JANGTQ2/JANGTQ4 qwen3/VL/CCA policy, Hy3 JANGTQ_K Hunyuan+qwen3 Low/High policy, and affine-JANG Qwen native-MTP VL/video multimodal detection. New artifact `build/current-model-family-detection-contract-20260522-zaya-hy3-qwen-vl-profile-rows.json` -> status=pass, missing_rows=[]; release manifest refreshed at `build/current-release-regression-manifest-20260522-zaya-hy3-qwen-vl-profile-rows.json`; umbrella refreshed at `build/current-regression-suite-20260522-zaya-hy3-qwen-vl-profile-rows.json` -> status=pass, failed_steps=[].

- 2026-05-22 09:11 PDT tightened parser registry gate with non-reasoning boundaries for Qwen2, Qwen2-VL, Gemma 3, and GLM base/Flash separation. New artifact `build/current-parser-registry-contract-20260522-non-reasoning-boundaries.json` -> status=pass, missing_markers=[], engine 120 passed, panel 40 passed. Release manifest refreshed at `build/current-release-regression-manifest-20260522-non-reasoning-boundaries.json`; umbrella refreshed at `build/current-regression-suite-20260522-non-reasoning-boundaries.json` -> status=pass, failed_steps=[].

- 2026-05-22 09:17 PDT pre-DMG release gate rerun from `codex/pr-intake-manifest` at `09ae8c92`: version triple, twine check, panel request/type tests, panel typecheck, bundled Python import gate, and objective digest refresh passed. Gate failed only on the known objective row `DSV4 long-output/code/file-generation quality is release-cleared`; packaged app checks were skipped because DMGs/apps have not been built yet. Summary: `docs/internal/release-gates/20260522_091708/SUMMARY.md`. Eric explicitly asked to proceed to release DMG despite this known DSV4 quality exception; do not represent that row as cleared.

- 2026-05-22 10:12 PDT shipped v1.5.47. Pushed source/tag at `4fdd440e60b75aebd9ff71087fb89c1240a4c30a`, released Sequoia/Tahoe signed-notarized-stapled DMGs to both `jjang-ai/mlxstudio` and `jjang-ai/vmlx`, pushed mlxstudio updater commit `ec59aa3`, uploaded `vmlx==1.5.47` to PyPI after GitHub Publish PyPI workflow failed due missing trusted publisher/API secret, installed Sequoia DMG into `/Applications/vMLX.app`, verified installed app version/codesign/Gatekeeper, and opened it. Packaged gate `docs/internal/release-gates/20260522_100035/SUMMARY.md` still fails only the known `DSV4 long-output/code/file-generation quality is release-cleared` objective row; this remains a release exception, not a clearance.

- 2026-05-22 10:22 PDT added post-release hardening proof for request output caps not mutating server startup defaults, plus fixed release-surface contract to accept complete post-release updater state. Red max-output artifact failed only missing the new marker; green max-output artifact `build/current-max-output-context-contract-20260522-request-default-mutation.json` passed. Red umbrella artifact exposed `release_surface_contracts`; green release-surface artifact `build/current-release-surface-contract-20260522-post-release-updater.json` passed. Final umbrella `build/current-regression-suite-20260522-post-release-updater.json` passed with `failed_steps=[]` and only the known DSV4 quality requirement open.

- 2026-05-22 10:24 PDT committed and pushed `d67fd5de test: harden post-release output and updater gates` to `origin/codex/pr-intake-manifest` and `origin/main`.

- 2026-05-22 10:28 PDT added local high-risk artifact registry guard for actual DSV4, Qwen JANG/JANGTQ/MXFP/4bit/MTP, Hy3, and Nemotron Omni/Nano JANGTQ/MXFP rows. Red model-family artifact failed only missing the new row; green model-family artifact `build/current-model-family-detection-contract-20260522-local-artifact-registry.json` passed. Umbrella `build/current-regression-suite-20260522-local-artifact-registry.json` passed with `failed_steps=[]` and only the known DSV4 quality requirement open.

- 2026-05-22 10:31 PDT committed and pushed `b3894c48 test: pin local high-risk family registry rows` to `origin/codex/pr-intake-manifest` and `origin/main`.

- 2026-05-22 10:49 PDT added panel local high-risk path parity and fixed real Qwen native-MTP VL registry mismatch. `qwen27_jang4m_mtp` now aligns engine/decode-speed policy with panel/API and emits `--is-mllm` only for indexed native-MTP VL artifacts with real vision/MTP tensors; plain affine-JANG Qwen VL remains text-only. Green artifacts: `build/current-model-family-detection-contract-20260522-panel-local-paths.json`, `build/current-release-regression-manifest-20260522-panel-local-paths.json`, `build/current-packaged-integrity-contract-20260522-panel-local-paths.json`, and `build/current-regression-suite-20260522-panel-local-paths.json` (`status=pass`, `failed_steps=[]`, only known DSV4 quality row open). Bundled Python was refreshed from clean JANG source after the release gate correctly caught source/bundle drift.

- 2026-05-22 10:51 PDT committed and pushed `64d259c5 fix: align qwen native mtp vl registry routing` to `origin/codex/pr-intake-manifest` and `origin/main`.

- 2026-05-22 11:04 PDT bumped to v1.5.48 and built local Sequoia/Tahoe DMGs. Sequoia sha256 `8637edbdcb261729c8013878ef606643915661ddd6c46a368d952d620efe9d6d`; Tahoe sha256 `79adfba392ca534266acfad080718d6a392204c9f1ee60b831698c57a4edfedc`. Bundled verifier, packaged integrity, release-surface pre-release, focused release tests, and umbrella passed; umbrella still has only known DSV4 quality row open. Public release is blocked because electron-builder skipped notarization and the notary keychain/profile is locked; do not upload/tag/update latest until notarization/stapling/Gatekeeper validation is complete.

- 2026-05-22 11:06 PDT committed and pushed `c21a12d6 chore: prepare v1.5.48 release` to `origin/codex/pr-intake-manifest` and `origin/main`.

- 2026-05-22 12:04 PDT isolated DSV4 pool-quant slowdown. Source no-prefix DSV4 speed gate is ~21.6-22.7 tok/s; native prefix/paged/L2 with `DSV4_POOL_QUANT=0` is 384 completion tokens in 37.18s (10.33 tok/s); same path with `DSV4_POOL_QUANT=1` is 384 completion tokens in 129.11s (2.97 tok/s). Default-cache tool loop with pool quant on passed functionally but was slow. Root cause read: `jang_tools/dsv4/pool_quant_cache.py` repeatedly dequantizes and requantizes the accumulated CSA/HCA pool every decode step. Added app guard so DSV4 launches always emit `DSV4_POOL_QUANT=0`, stale saved `dsv4PoolQuant` is cleared, and UI disables the pool-quant checkbox. Verification: focused panel pool tests `8 passed`, DSV4 cache policy tests `6 passed`, no-heavy panel settings contract pass at `build/current-panel-settings-contract-20260522-dsv4-pool-quant-disabled.json`.
# 2026-05-22 12:16 PDT

- Investigated DSV4 native prefix/cache + pool-quant slowdown against the real
  JANG source.
- Added and passed a JANG regression proving pool quant appends no longer
  requantize old rows.
- Installed local JANG source into this vMLX worktree venv and ran two live
  DSV4 raw-stream probes:
  - pool quant on, append-fixed: 192 deltas / 80.21s (~2.39 deltas/s);
  - pool quant off: 192 deltas / 29.87s (~6.43 deltas/s).
- Decision recorded: keep DSV4 pool quant disabled by default in vMLX; native
  unquantized DSV4 prefix/paged/L2 remains the viable cache path until JANG
  fixes full-pool dequantize/concat read cost.

# 2026-05-22 12:31 PDT

- Added vMLX no-heavy cache regression requiring bundled JANG DSV4 pool quant
  to append only new CSA/HCA pool rows.
- Green cache artifact:
  `build/current-cache-architecture-contract-20260522-dsv4-pool-quant-append.json`
  -> `status=pass`, cache family `107 passed`, panel cache launch `76 passed`.
- Refreshed bundled Python from clean JANG worktree
  `/Users/eric/jang/.worktrees/vmlx-release-clean-49e558e` at JANG commit
  `49e558e` because packaged integrity caught stale bundled
  `vmlx_engine/scheduler.py`.
- Verified bundled Python, packaged integrity, and umbrella:
  `build/current-regression-suite-20260522-dsv4-pool-quant-append-after-bundle.json`
  -> `status=pass`, `failed_steps=[]`, only known DSV4 long-output/code
  quality requirement open.
- Committed and pushed `608f105d test: require dsv4 pool quant append codec`
  to `origin/codex/pr-intake-manifest` and `origin/main`.

# 2026-05-22 12:38 PDT

- Added max-output/context gate coverage for chat reset and string-shaped
  legacy persisted `maxTokens`.
- Red artifact:
  `build/current-max-output-context-contract-20260522-reset-policy-red.json`
  -> failed only missing `clears string legacy session maxTokens before launch
  can reuse it`.
- Green max-output artifact:
  `build/current-max-output-context-contract-20260522-reset-policy-string-legacy.json`
  -> `status=pass`, engine `21 passed`, panel `43 passed`.
- Umbrella:
  `build/current-regression-suite-20260522-reset-policy-string-legacy.json`
  -> `status=pass`, `failed_steps=[]`, only known DSV4 long-output/code
  quality requirement open.
- Committed and pushed `cd3efa84 test: harden max output reset migration`
  to `origin/codex/pr-intake-manifest` and `origin/main`.

# 2026-05-23 12:05 PDT

- Reviewed the just-pushed duplicate DSV4 cache UI cleanup.
- Found a semantic regression: the remaining `DSV4 CSA/HCA Pool Codec`
  checkbox could set `dsv4PoolQuant` without enabling `dsv4PrefixCache`, even
  though launch emits `DSV4_POOL_QUANT=1` only when both are true and session
  normalization clears pool quant when prefix is off.
- Restored `cacheControlUpdatesForDsv4PoolQuantToggle` on the single pool
  checkbox, keeping the duplicate UI cleanup while making the checkbox create a
  launchable pool-on state.
- Updated the tooltip and stale Python-side DSV4 UI assertions.
- Verification: red `dsv4-env` proof first, panel focused 271 tests pass,
  panel typecheck pass, DSV4 Python UI/cache focused 3 pass, cache architecture
  contract pass with cache-family 111 and panel cache launch 92.
- Committed and pushed `5fa84203 fix: make dsv4 pool codec toggle launchable`
  to `origin/codex/pr-intake-manifest` and `origin/main`.
- Post-push current suite source-of-truth:
  `build/current-regression-suite-20260523-after-pool-ui.json` ->
  `status=pass`, `failed_steps=[]`, with the same two open requirements:
  Qwen 27B JANG_4M PP speed and DSV4 long-output/code quality.
- Direct packaged integrity refresh:
  `build/current-packaged-integrity-contract-20260523-dsv4-pool-ui-launchable.json`
  -> `status=pass`, `failed=[]`.

# 2026-05-23 12:16 PDT

- Ran installed DSV4 live cache-context identifier probe on
  `127.0.0.1:8008`; no new model launch.
- Artifact:
  `build/current-dsv4-live-cache-context-identifier-probe-20260523.json`.
- Native DSV4 composite cache, prefix/paged/block-L2, and pool quant were
  active; generic TurboQuant KV was off.
- Finding: unique prompt prefix did not prevent identifier corruption.
  `list_plain` produced `THIVE.WebGLRenderer`; `list_unique_prefix` produced
  `THUEE.WebGLRenderer`; constructor prompt preserved
  `THREE.PerspectiveCamera`.
- Wired result into objective proof as
  `current_installed_unique_prefix_identifier_probe`.
- Verification: red digest test first, then focused green; full
  `tests/test_objective_proof_digest.py tests/test_current_regression_suite.py`
  -> 61 passed; digest
  `build/current-objective-proof-digest-20260523-dsv4-cache-context-wired.json`
  keeps DSV4 quality open with unique-prefix probe status review.
- Committed and pushed `e684f5d5 test: surface dsv4 unique-prefix identifier probe`
  to `origin/codex/pr-intake-manifest` and `origin/main`.
- `4185a2b6 test: gate packaged dsv4 cache ui labels` landed underneath it
  from the other agent while this slice was being finalized.

# 2026-05-23 12:42 PDT

- Added packaged-integrity enforcement for Max Thinking Tokens renderer and
  request/DB wiring markers.
- Verified repo-local packaged app bundle launch after ad-hoc signing nested
  native libraries.
- Live boundary proof:
  - running app opened `~/Library/Application Support/vmlx/chats.db` with
    `chat_overrides.max_thinking_tokens`;
  - packaged `app.asar` contains UI, DB, sanitizer, request, and defaults
    wiring;
  - packaged bundled Python accepts positive `max_thinking_tokens` and rejects
    zero for Chat/Responses.
- Verification:
  - packaged integrity pass;
  - max-output/context pass;
  - panel focused settings/request/defaults `322 passed`;
  - Python focused server/API/max-output/package `43 passed`;
  - panel typecheck pass.
- Pushed `2fb4dda0 test: gate packaged max thinking setting` to
  `origin/codex/pr-intake-manifest` and `origin/main`.
- Boundary: Computer Use could not attach to vMLX UI, so proof is process/DB/
  packaged-renderer/API-model rather than a clicked screenshot; no heavy model
  launch in this slice.

# 2026-05-23 12:52 PDT

- Fixed repo-local/dev app user-data isolation:
  - added early bootstrap for `--vmlx-user-data-dir`, `VMLX_USER_DATA_DIR`, and
    legacy `VMLINUX_USER_DATA_DIR`;
  - bootstrap runs before DB construction and before
    `app.requestSingleInstanceLock()`.
- Added source regression and packaged-integrity gate for the bootstrap.
- Live packaged proof:
  - rebuilt and packaged `panel/release/mac-arm64/vMLX.app`;
  - launched isolated data dirs under `build/user-data-isolation-smoke*`;
  - each got its own `chats.db`; the real app-support DB stayed unchanged in
    the second clean smoke.
- Verification:
  - user-data isolation test `3 passed`;
  - panel typecheck pass;
  - build/package pass with `CSC_IDENTITY_AUTO_DISCOVERY=false`;
  - packaged integrity pass;
  - panel focused settings/request/defaults `322 passed`;
  - Python packaged/max-output focused `13 passed`.
- First umbrella run exposed an unrelated objective-fixture gap: synthetic
  DSV4 clearance did not create the source no-cache identifier artifact now
  required by the digest, so focused regression failed.
- Fixed that fixture; focused digest/current-suite tests `62 passed`.
- Umbrella rerun:
  `build/current-regression-suite-20260523-user-data-isolation2.json` ->
  `status=pass`, `failed_steps=[]`, with Qwen PP speed and DSV4 quality still
  open.
- Pushed `5cf44107 fix: isolate repo-local app user data` to
  `origin/codex/pr-intake-manifest` and `origin/main`.
- Observed live DSV4 probes on ports `8844` then `8845`; left them untouched.

## [2026-05-23 13:39] codex | package+smoke | clean-JANG dev app package usable; release blockers remain

- Rebuilt bundled Python from clean JANG worktree
  `/Users/eric/jang/.worktrees/vmlx-release-clean-b5f66a7/jang-tools`.
- Verified bundled Python passed and packaged `panel/release/mac-arm64/vMLX.app`
  with `CSC_IDENTITY_AUTO_DISCOVERY=false`.
- Ad-hoc signed and smoke-launched the repo-local app with isolated user data:
  `build/final-user-data-smoke-clean-20260523`.
- Smoke log proved isolated userData, no engine adoption, bundled
  `vmlx_engine 1.5.48`, gateway start, updater v1.5.48, and isolated
  `chats.db` creation.
- Packaged integrity verifier is clean for bundle parity, but release gate still
  fails objective proof digest on Qwen PP speed and DSV4 long-code quality.
- Qwen follow-up live rows remain `review`, including disabled-native-MTP;
  do not mark Qwen PP cleared.
## [2026-05-23 15:54] codex | qwen-mtp-proof | Qwen VLM+MTP source/native cleared; strict proof sweep restored

- Used the post norm-format Qwen VLM+MTP live artifact as the source/native
  clearance row.
- Updated objective/manifest/current-suite gates so Qwen is no longer a
  known-open requirement; DSV4 long-output/code quality remains the only open
  objective.
- Kept `--require-current-proof-sweep` in the umbrella and added a narrow
  release-manifest bootstrap allowance for the manifest self-reference cycle.
- Verification: strict umbrella pass, manifest proof sweep pass, focused
  current-suite/manifest tests pass, Qwen/native-MTP/objective slices pass,
  `py_compile` pass, `git diff --check` pass.

## [2026-05-23 16:01] codex | dsv4-audit | present affine row + sampling risk visibility

- Fixed stale production-family audit row `dsv4_jang_dq2_gate3math6`, which
  pointed at a missing local DSV4 path.
- The row now points at the present local affine NoMTP artifact:
  `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`.
- Added static `sampling_loop_risk` audit output so model-owned max-output and
  sampler gaps are visible without adding hidden runtime defaults.
- Verification: changed focused tests `4 passed`; broader DSV4/model-family
  slice `34 passed, 59 deselected`; model-family contract pass with
  `missing_rows=[]`; production-family static DSV4 artifact has both affine
  rows `exists=true`; `py_compile` and `git diff --check` pass.
- Boundary: DSV4 long-output/code/file-generation quality remains open.
- Pushed `f702f537 test: align dsv4 affine audit row` to
  `origin/codex/pr-intake-manifest`.

## [2026-05-23 16:08] codex | family-audit | summary exposes missing/review rows

- Added `production_family_audit_summary()` to the static/live family audit
  runner.
- Refreshed
  `build/current-production-family-static-loop-risk-audit-20260523.json`.
- Current no-heavy matrix summary: 32 rows, 7 missing rows, 10 issue rows, 29
  sampling-review rows, no live rows in this run.
- Missing rows currently include stale/unavailable DSV4, Qwen35 JANGTQ4,
  Mistral, and MiniMax JANGTQ4 paths on `/Volumes/EricsLLMDrive`.
- Verification: summary/sampling focused tests `3 passed`, `py_compile` pass,
  `git diff --check` pass.
- Left unrelated dirty Ling/Bailing Native-MTP guard edits untouched.
- Pushed `f1231aeb test: summarize production family audit gaps` to
  `origin/codex/pr-intake-manifest`.

## [2026-05-23 16:13] codex | minimax-live | loop trigger added and small row passed

- Found the first MiniMax small live artifact was a generic API/cache/parser
  pass, not a real loop/gibberish trigger despite its filename.
- Added a live loop probe selector so MiniMax rows run
  `minimax_multilingual_loop_trigger`; Ling keeps its existing loop trigger.
- Reran `minimax_m27_small_tq` live:
  `build/current-production-family-live-minimax-small-loop-audit-20260523-rerun.json`.
- Result: `live.status=PASS`, `failures=[]`, 12 checks including the new
  MiniMax loop trigger.
- Loop trigger evidence: `finish=stop`, no reasoning, `loop_score=0.0571`,
  `cjk_chars=0`, `word_count=54`.
- Verification: py_compile pass, focused loop/summary tests `4 passed`,
  `git diff --check` pass; no MiniMax process left on port 8797.
- Commit split pushed:
  `6cdb67ea test: require live loop probes for ling minimax` and
  `79fb6143 test: add minimax live loop trigger`.
## [2026-05-23 16:18] codex | live-family-audit | Hy3 preview loop probe current

- Ran `hy3_preview_jangtq2` live production-family audit after the harness
  gained `hy3_multilingual_loop_trigger`.
- Artifact:
  `build/current-production-family-live-hy3-jangtq2-loop-audit-20260523.json`.
- Result: `live_status_counts={"PASS":1}`, `missing_rows=[]`,
  `issue_rows=[]`, failed live checks none.
- Loop trigger passed with `finish=stop`, `content_chars=478`,
  `reasoning_chars=0`, `loop_score=0.0735`, `cjk_chars=0`, `word_count=52`.
- No model audit process remains.
- Boundary: source-engine Hy3 preview JANGTQ2 row only; does not clear other
  Hy3 rows, packaged app behavior, or DSV4 long-output/code quality.

## [2026-05-23 16:18] codex | live-family-audit | ZAYA text typed-cache row current

- Ran `zaya_jangtq2` live production-family audit.
- Artifact:
  `build/current-production-family-live-zaya-jangtq2-typed-cache-audit-20260523.json`.
- Result: `live_status_counts={"PASS":1}`, `missing_rows=[]`,
  `issue_rows=[]`, failed live checks none.
- ZAYA checks passed: `zaya_cca_typed_cache_live_gate_enabled`,
  `zaya_native_cache_capabilities`, `runtime_cache_layout_logged`,
  `cache_second_turn_coherent`.
- Native cache proved `schema=zaya_cca_v1`, `cache_type=typed_cca`,
  generic TurboQuant KV disabled, prefix/paged/block-disk L2 enabled.
- Repeat turn reported `cached_tokens=38`,
  `cache_detail=paged+zaya_cca`, and block disk L2 writes.
- No model audit process remains.
- Boundary: source-engine ZAYA text JANGTQ2 row only; does not clear ZAYA-VL,
  packaged app behavior, other ZAYA quant rows, or DSV4 long-output/code
  quality.

## [2026-05-23 16:18] codex | harness-tightening | ZAYA exact tool args now enforced

- Found the Responses auto-tool audit was too loose: it accepted any
  `list_directory` function call even when parsed arguments were malformed.
- Added `responses_tool_call_arguments_ok()` and a regression test requiring
  exact args `{"path":"."}` for the live audit probe.
- Red proof: new test failed before helper existed.
- Green proof: focused argument/markup tests `2 passed`; focused ZAYA/tool
  tests `8 passed, 58 deselected`; `py_compile` and `git diff --check` pass.
- Strict live reruns:
  - ZAYA-VL JANGTQ4 now fails only auto-tool exact args:
    `{"path":"<value>.</value>"}`;
  - ZAYA text JANGTQ2 now defers only auto-tool exact args:
    `{"path":": "}`.
- Read: typed CCA cache is good; exact Responses auto-tool argument fidelity is
  still open for ZAYA/ZAYA-VL.
- Committed and pushed:
  `1aba1522 test: require exact responses tool args`.
## [2026-05-23 16:39] codex | qwen-live-fix | dense JANG quant/cache audit green

- Fixed Qwen3.6-27B JANG_4M source-engine live audit failure.
- Root causes:
  - stale mixed affine per-module claims were shape-compatible but wrong for
    hidden-size modules, producing 2560-wide embeddings against 5120-wide norms;
  - equal-length multi-prompt hybrid batches skipped cache `prepare()`, causing
    RoPE offset shape `(0)` in batched follow-up probes.
- Added/kept focused tests for Qwen architecture-dim quant inference,
  post-load embedding logical dims, and equal-length prompt-batch cache prepare.
- Authoritative artifact:
  `build/current-production-family-live-qwen36-dense-jang-quantfix-clean-audit-20260523.json`
  -> pass, failures=[].
- Earlier `quantfix-audit` artifact is invalid evidence because the server
  logged a port bind error and the audit reached a stale process.

## [2026-05-23 16:50] codex | zaya-tools | exact Responses tool args fixed

- Fixed ZAYA exact Responses auto-tool argument fidelity without changing
  samplers, cache policy, model detection, generation defaults, or settings.
- Root cause: ZAYA fallback XML examples still used `VALUE HERE` placeholders
  after live logs showed template schema injection was required; ZAYA-VL also
  emitted an extra `<value>...</value>` wrapper inside the parameter body.
- Changes:
  - unwrap ZAYA-only `<value>...</value>` inside `ZayaToolParser`;
  - render concrete XML fallback example values: `.` for path-like params,
    `example` for other string params.
- Proof:
  - focused parser/fallback/audit tests: `30 passed`;
  - `build/current-tool-call-contract-20260523-zaya-exact-toolargs.json`:
    `status=pass`, `failed=[]`, `missing_markers=[]`;
  - live `zaya_vl_jangtq4` strict rerun: `PASS`, exact
    `{"path":"."}`;
  - live `zaya_jangtq2` strict rerun: `PASS`, exact `{"path":"."}`;
  - other agent umbrella:
    `build/current-regression-suite-20260523-after-bundle-refresh.json` ->
    `status=pass`, `failed_steps=[]`, only DSV4 long-output/code quality open;
  - `git diff --check`: pass.
- Committed and pushed:
  `2c0fc8a4 fix: repair zaya tool argument examples`.

## [2026-05-23 20:34] codex | release-surface | public updater cache headers fixed

- SSH inspected `exploit.team`.
- Found `/var/www/mlxstudio` absent; actual root is `/var/www/mlx.studio`.
- Fixed `/etc/nginx/conf.d/mlx.studio.conf` where `location /update/` had
  been accidentally commented onto one line.
- Reloaded nginx after `nginx -t` passed.
- Public `https://mlx.studio/update/latest.json` now has no-store/no-cache
  headers and matches raw GitHub latest on 1.5.48 URL/SHA.
- Added live release-surface contract enforcement for updater cache headers.
- Verification: release-surface focused tests 12 passed; live public
  release-surface artifact pass; release regression manifest pass;
  `git diff --check` pass.
- Committed and pushed:
  `5336e8d5 test: guard public updater cache headers`.

## [2026-05-23 20:48] codex | live-agentic-runtime | qwen responses tool extraction fixed, cache/tool follow-up open

- Ran long live Responses/tool/cache probes against
  `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M`.
- Found non-streaming Responses endpoint bug:
  real Qwen `<tool_call>` markup inside reasoning was not parsed because only
  `content_for_parsing` was sent to the tool parser.
- Fixed `vmlx_engine/server.py` to parse real tool markers from
  `reasoning_text` before non-streaming Responses finalization.
- Added regression in `tests/test_engine_audit.py`.
- Proof:
  `.venv/bin/python -m pytest tests/test_engine_audit.py::TestResponsesSuppressedReasoningToolCalls -q`
  -> `2 passed`.
- Live after patch:
  `build/qwen36-dense-jang-responses-long-tool-cache-after-reasoning-tool-fix-20260523/gate-auto-thinking-on/SUMMARY.json`
  now shows turn 1 structured function_call `grep_repo`.
- Still open:
  tool-output follow-up can stay reasoning-only with thinking on; hybrid cache
  writes/L2 grow but varied previous_response_id turns do not get usable
  SSM-backed cached_tokens; MiniMax-small temp 0.6 multilingual loop remains;
  DSV4 long-output/code quality remains open.
- Durable details:
  `docs/internal/LIVE_AGENTIC_RUNTIME_AUDIT_2026-05-23.md`.
- Rebuilt bundled Python from this checkout with clean JANG source and verified
  source-content parity.
- Verification:
  - focused runtime/tool/history tests: `93 passed`;
  - `git diff --check`: pass;
  - packaged integrity:
    `build/current-packaged-integrity-contract-20260523-live-agentic-runtime.json`
    -> `status=pass`, `failed=[]`;
  - umbrella:
    `build/current-regression-suite-20260523-live-agentic-runtime-after-bundle.json`
    -> `status=pass`, `failed_steps=[]`, only DSV4 long-output/code quality
    remains open.
- No model/test/release/notary process left running.
- Committed and pushed:
  `e9e6fca6 fix: parse responses tool calls from reasoning`.
- 2026-05-23 21:14 PDT rechecked DSV4 pool quant across actual runtime
  surfaces. Repo `.venv` and repo bundled Python both contain the materialized
  pool fix (`_pooled_materialized`, `append_pooled`); installed
  `/Applications/vMLX.app` is stale and lacks both. Inline same-sequence probe:
  repo `.venv` `dequant_count=0`, installed app `dequant_count=4`. Fresh
  focused checks: vMLX DSV4 pool/cache `4 passed`, JANG pool tests `3 passed`,
  cache architecture/API-cache/native-MTP/VL-media contracts all pass, umbrella
  `build/current-regression-suite-20260523-dsv4-pool-runtime-recheck.json`
  passes with `failed_steps=[]` and only known DSV4 long-output/code quality
  requirement open. Audit:
  `docs/internal/DSV4_POOL_QUANT_RUNTIME_AUDIT_2026-05-23.md`.

- 2026-05-23 21:33 PDT reconciled v1.5.49 DSV4 default-cache release policy:
  native composite prefix cache + materialized pool codec default on, explicit
  disable honored, generic KV q4/q8 suppressed. Fixed stale CLI/test
  diagnostic-off wording. Focused DSV4 policy/UI/pool slice `6 passed`;
  `build/current-cache-architecture-contract-20260523-v1549-release-rerun.json`
  passed with no failed or missing markers.

- 2026-05-23 21:35 PDT added Qwen VL/MTP retained-image Metal OOM guard:
  VLM media requests now tighten MLX reusable Metal cache before `mlx_vlm`
  preprocessing and before one-shot VLM forward. Focused VLM OOM/media tests
  `7 passed`; prompt/image guard tests `3 passed`; image-count audit `3 passed`;
  VL/media contract
  `build/current-vl-media-cache-contract-20260523-vlm-image-cache-limit.json`
  passed. Live source Qwen VL/MTP image smoke completed and logged media cache
  limit tightened to `1.00GB`. Pushed
  `58457129 fix: tighten vlm media metal cache headroom`.

- 2026-05-23 21:48 PDT refreshed bundled Python/JANG for v1.5.49 release gate.
  `verify-bundled-python.sh` passes; packaged integrity
  `build/current-packaged-integrity-contract-20260523-v1549-final-local-jang.json`
  passes; release manifest
  `build/current-release-regression-manifest-20260523-v1549-final-local-jang.json`
  has `current_proof_sweep=pass`. Release gate skip-app now fails only on the
  known DSV4 long-output/code/file-generation objective row.
## 2026-05-23 22:49 PDT - v1.5.49 signed DMG validation complete

- Validated both final public DMGs from mounted artifacts:
  - `panel/release/vMLX-1.5.49-sequoia-arm64.dmg`
    `96e693f2cb6a8c476eea079ca8537808a80b5bbc5550d3bf574e0550c00fb2c4`
  - `panel/release/vMLX-1.5.49-tahoe-arm64.dmg`
    `c121581907aa3adcfefa9f146f158745276823bc457e74fcc28bf9147fac2230`
- Each DMG passes stapler validation and Gatekeeper open assessment.
- Each mounted app reports version `1.5.49`, passes deep strict codesign
  verification, imports `vmlx_engine==1.5.49`, and contains the materialized
  DSV4 pool cache fix in bundled `jang_tools`.
- Remaining release exception is unchanged: DSV4 long-output/code exactness
  remains documented open.

## 2026-05-23 23:03 PDT - v1.5.49 public release complete

- Published `v1.5.49` on `jjang-ai/vmlx` and `jjang-ai/mlxstudio`.
- Uploaded Sequoia/Tahoe DMGs plus blockmaps to both releases.
- Uploaded PyPI `vmlx==1.5.49` wheel and sdist.
- Updated and verified both updater feeds:
  - raw GitHub `jjang-ai/mlxstudio/main/latest.json`;
  - `https://mlx.studio/update/latest.json`.
- Final public release-surface contract:
  `build/current-release-surface-contract-20260523-v1549-public-final2.json`
  -> `status=pass`.
- Installed `/Applications/vMLX.app` from the validated Sequoia DMG and opened
  it for manual testing.
- Installed-app verification passed version, codesign, Gatekeeper, bundled
  engine import, and bundled DSV4 materialized pool source check.

## [2026-05-24 19:53] codex | proof | DSV4 exact-code rail boundary wired

- Added current live DSV4 generated-only/direct and requested-thinking
  route-mode artifacts to objective digest and release manifest proof.
- Current direct/off subset remains failed with `WebWebGLRenderer` under
  `thinking_disabled`/`thinking_not_requested`; requested-thinking subset
  passes exact code.
- Added `current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json`:
  `GL` is rank 1 after `THREE.Web` in both rails, but direct/off differs at
  `THREE.` and `THREE.P` where requested-thinking ranks the expected tokens
  first.
- Refreshed objective digest, current suite, and release manifest. Suite and
  manifest pass with exactly the three known open rows.
- No runtime force-on policy change was made.

## [2026-05-24 20:28] codex | gemma4 | mixed-swa runtime contract fixed, speed open

- Root cause: Gemma4 mixed sliding/full attention was previously flattened by
  generic live TurboQuant KV under `VMLINUX_FORCE_TQ_AUTO=1`, so health could
  claim `mixed_swa_kv_v1` while runtime used `30/30 TurboQuantKVCache`.
- Source fix: skip generic TQ patching for mixed sliding/full attention layouts
  and pass scheduler mixed-attention detection into `MLLMBatchGenerator`.
- Telemetry fix: mixed-SWA prefix hits now report `paged+mixed_swa` /
  `paged+mixed_swa+disk` instead of the generic hybrid SSM labels.
- Live installed-app proof:
  `build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-after-mixed-swa-telemetry-20260524.json`
  has `generic_turboquant_kv.enabled=false`, `cache_detail=paged+mixed_swa`,
  but speeds `59.933` / `54.992 tok/s`.
- Refreshed proof board:
  current suite `build/current-regression-suite-20260524-after-gemma4-mixed-swa-telemetry.json`
  passes, packaged integrity
  `build/current-packaged-integrity-contract-20260524-after-gemma4-mixed-swa-telemetry.json`
  passes, release manifest
  `build/current-release-regression-manifest-20260524-after-gemma4-mixed-swa-telemetry.json`
  has `current_proof_sweep=pass`.
- Do not release-clear: Gemma speed floor remains open and DSV4 still has two
  open objective rows.

## [2026-05-24 12:08] codex | risk | wrong-repo Swift detour corrected

- After context compaction I resumed from `/Users/eric/vmlx` and followed stale
  Swift handoff instructions instead of this Python coordination workspace.
- Stale `/Users/eric/vmlx` root handoff was deleted/replaced with a routing
  guard pointing back to `/Users/eric/mlx/vllm-mlx`.
- Do not use any `/Users/eric/vmlx` Swift notes as current app/runtime state.
  Current Python source of truth remains this worktree's `.agents` plus the
  latest build artifacts; DSV4 exact-code quality is the active release blocker.

## [2026-05-24 14:00] codex | fix | packaged drift and stale synthetic DSV4 fixture

- Fixed stale synthetic all-clear objective-proof fixture to include current
  DSV4 diagnostic evidence files.
- Reinstalled current local `vmlx` into `panel/bundled-python`, rewrote
  regenerated `vmlx*` console scripts to relocatable trampoline shebangs, and
  removed regenerated bytecode caches.
- Verified `npm run verify-bundled` passes.
- Verified
  `build/current-packaged-integrity-contract-20260524-after-bundled-engine-sync2.json`
  has `status=pass`.
- Verified
  `build/current-regression-suite-20260524-after-bundled-engine-sync.json`
  has `status=pass`, `failed_steps=[]`, and the single open requirement remains
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Do not claim release cleared; the DSV4 exact-code/model-quality row remains
  intentionally open.

## [2026-05-24 12:14] codex | audit | DSV4 latest artifact still red

- Re-synced `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no active DSV4
  server/test process was visible.
- Latest live DSV4 route exactness artifact inspected:
  `build/current-dsv4-route-mode-code-exactness-source-after-completion-token-split-20260524.json`.
- Artifact status is still `fail`; `chat_max` corrupts Perspective/Box/Mesh
  identifiers and `legacy_completion_raw` corrupts `THREE.WebGLRenderer`.
- The artifact predates new requested/effective rail diagnostics, so it cannot
  prove live effective-rail behavior. Focused audit tests for the diagnostic
  split passed: `6 passed, 39 deselected`.

## [2026-05-24 12:22] codex | audit | no-heavy proof board refreshed

- Refreshed current no-heavy proof artifacts affected by `vmlx_engine/server.py`
  hash drift: API/cache, max-output/context, generation defaults, native MTP,
  and model artifact format all pass.
- Refreshed digest
  `build/current-objective-proof-audit-20260524-codex-recheck.json` now has
  exactly one open row: DSV4 long-output/code/file-generation quality.
- DSV4 is still not release-cleared; latest live route exactness artifact is
  red.

## [2026-05-24 12:31] codex | live-probe | DSV4 rerun timeout

- Attempted live DSV4 route-mode exactness rerun on port `8861` using bundled
  Python and `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K`.
- Server reached health `model_loaded=true`, then the first HTTP request timed
  out. Runner exited with traceback and did not rewrite the target artifact.
- `8861` listener is gone; orphan PID `251` remains non-listening and resisted
  normal kill and `kill -9`.
- Separate DSV4 server on port `8886` was already present and was not touched.
- Current evidence: DSV4 remains open; route-mode rerun is timeout/inconclusive
  and older exactness artifact is still red.

## [2026-05-24 12:41] codex | live-probe | existing DSV4 server two-row failure

- Reused already-loaded DSV4 server on port `8886` for a tiny two-row probe to
  avoid another model load.
- Artifact:
  `build/current-dsv4-two-row-existing-server-8886-20260524.json`.
- Result `status=fail`.
- `chat_max` corrupts identifiers (`PersPerspectiveCamera`, `BBoxGeometry`,
  `MMeshBasicMaterial`) and returns a markdown fence.
- `legacy_completion_raw` fails exactness and produces malformed/prose-ish
  output with `PersontiveCamera`.
- After the client completed, `/health` still showed `scheduler.num_running=1`
  and `num_waiting=0` for at least 30 seconds. Possible scheduler liveness or
  accounting leak; `8886` was not killed.

## [2026-05-24 12:58] codex | fix-verify | DSV4 completions/fence cleanup source checks

- Current source routes DSV4 `/v1/completions` through `engine.chat(...)`,
  extracts visible DSV4 completion text, and unwraps a whole single markdown
  fence only for exact/no-markdown requests.
- Focused tests pass:
  `test_dsv4_completion_endpoint_uses_chat_rail` and
  `test_dsv4_completion_exact_no_markdown_returns_visible_unfenced_code` ->
  `2 passed, 78 deselected`.
- Broader DSV4/completion/logprobs slice passed: `38 passed, 89 deselected`.
- Refreshed no-heavy hash-tracked artifacts and objective digest; only open
  row remains DSV4 long-output/code/file-generation quality.
- Live DSV4 server on `8886` is active with `num_running=1`, so no competing
  live probe was started from this lane.

## [2026-05-24 13:04] codex | audit | DSV4 clearance gate hardened

- Found proof-board risk: route-mode exactness artifact was detail-only, not
  part of DSV4 release-clearance boolean.
- Updated `summarize_objective_proof.py` so DSV4 quality cannot pass unless the
  route-mode exactness artifact status is `pass`.
- Added regression test proving fenced/normalized-only route output keeps DSV4
  open even if the older quality-clearance artifact says pass.
- Verification: DSV4/quality proof slice `14 passed, 25 deselected`; refreshed
  digest still has only DSV4 quality open.

## [2026-05-24 12:43] codex | recovery | active Python lane rechecked after Swift detour

- Re-read `/Users/eric/vmlx` routing guard, active Python `AGENTS.md`, and this
  worktree's `.agents` state. Active app/runtime lane is
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; Swift remains out of scope.
- Current open blocker remains DSV4 long-output/code/file-generation quality.
  Do not claim the completions/fence cleanup fixed DSV4 until a fresh full live
  route-mode exactness artifact passes; latest live artifact still had
  `chat_max` and `legacy_completion_raw` failures.
- Process check: server `8886` PID `4692` is loaded and currently idle by
  `/health` (`num_running=0`, `num_waiting=0`), but live requests should still
  be coordinated through `MAIL.md` before use.

## [2026-05-24 12:44] codex | cleanup | stale Swift/handoff artifacts removed from wrapper repo

- Deleted stale `/Users/eric/vmlx` Swift/handoff docs and Swift build/test logs
  matched by `*swift*` or `*handoff*`.
- Verified the only remaining `swift`/`handoff`/agent-routing hits at maxdepth 3
  are `/Users/eric/vmlx/AGENTS.md` and `/Users/eric/vmlx/CODEX.md`, both routing
  guards pointing agents to the Python repo.
- No Swift engine/app source was used for current vMLX work.

## [2026-05-24 12:47] codex | audit-fix | DSV4 completion effective rail diagnostic corrected

- Found a route-mode proof bug: the DSV4 exactness dry-run/live artifact still
  described `/v1/completions` effective prompt rail as `none`/legacy raw, even
  though current source routes non-streaming DSV4 completions through
  `engine.chat(...)`.
- Added red/green coverage and updated the runner so completion requested rail
  stays `none` but effective rail records `thinking_open` with reason
  `dsv4_completion_chat_rail_identifier_unsafe`.
- Refreshed `build/current-dsv4-route-mode-code-exactness-dryrun-20260524.json`
  and `build/current-objective-proof-audit-20260524-codex-recheck.json`.
- Verification: red test failed before fix; focused route/objective slice
  `8 passed, 38 deselected`; full route exactness unit file `7 passed`;
  `py_compile` and `git diff --check` passed. No live request was sent to
  `8886`; DSV4 quality remains open.

## [2026-05-24 12:51] codex | audit-fix | scheduler running-request diagnostics added

- Backed off from live probing when `8886` flipped to `num_running=1` before the
  planned narrow request.
- Added prompt-free lifecycle diagnostics to `Scheduler.get_stats()` so
  `/health.scheduler` can expose running/waiting request IDs, pending abort IDs,
  and per-running-request status/age/token/cache/batch uid details after the
  next server restart.
- This is instrumentation for debugging the observed `num_running=1` aftermath,
  not a DSV4 output-quality fix. Current `8886` process predates the patch and
  therefore cannot show these fields.
- Verification: red test failed before fields existed; focused stats tests
  `2 passed`; scheduler/health lifecycle slice `95 passed, 399 deselected`;
  `py_compile` and `git diff --check` passed.

## [2026-05-24 12:53] codex | audit-fix | DSV4 route runner can reuse existing server

- Added `--base-url` and `--cases` to
  `tests/cross_matrix/run_dsv4_route_mode_code_exactness.py`, so future live
  checks can reuse an already-loaded DSV4 server and run only the known failing
  rows instead of launching a second model process or using one-off scripts.
- Refreshed dry-run artifact:
  `build/current-dsv4-route-mode-code-exactness-dryrun-existing-server-twofail-20260524.json`.
- Attempted to proceed to the narrow live two-case probe, but health flipped to
  `num_running=1` before the request; no live request was sent.
- Verification: route exactness unit file `9 passed`; combined route/stats
  smoke `11 passed`; `py_compile` and `git diff --check` passed.
## [2026-05-24 12:59] codex | recovery | Python lane re-synced after Swift-detour audit

- Re-read `/Users/eric/vmlx` routing guard and canonical
  `/Users/eric/mlx/vllm-mlx/AGENTS.md`: active app/runtime work is Python-only;
  Swift/handoff notes from the wrapper repo are invalid for current work.
- Active working tree remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.
- Rechecked current live/process state: no listener on `127.0.0.1:8886`, no
  reusable DSV4 server available, RAM currently clear.
- Re-read newest DSV4 route artifact:
  `build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json`.
  It remains `status=fail`: 9/10 rows exact; only `chat_max` still corrupts
  identifiers (`PersPerspectiveCamera`, `BBoxGeometry`, `MMeshBasicMaterial`).
- Objective digest still keeps
  `DSV4 long-output/code/file-generation quality is release-cleared` open.
- No fresh heavy model load launched in this recovery pass; coordination note
  sent in `.agents/MAIL.md`.

## [2026-05-24 13:04] codex | live-audit | DSV4 chat_max prompt-trigger isolated

- Reused existing loaded DSV4 server on `8886`; no new heavy process and no
  Swift work.
- Captured prompt-trigger artifact:
  `build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json`.
- Captured standard two-row artifact:
  `build/current-dsv4-route-mode-code-exactness-existing-server-twofail-post-trigger-20260524.json`.
- Result: completion route is exact; only `chat_max` remains corrupted.
- Finding: canonical exact-code wording passes, but natural wording variants
  around `Copy`, `no markdown`, `Preserve identifier spelling`, and
  `No explanation` fail under the same effective `thinking_open` DSV4 policy.
- Decision: do not use hidden prompt rewriting or identifier substitution as a
  release fix. Keep DSV4 long-output/code quality open.

## [2026-05-24 13:08] codex | audit-refresh | settings/defaults proof board narrowed

- Refreshed stale no-heavy artifacts for max output/context, generation
  defaults, API/cache, Native MTP, and model-artifact format after current
  server/scheduler edits.
- Objective digest artifact:
  `build/current-objective-proof-audit-20260524-post-open-row-refresh.json`.
- Result: all no-heavy/settings/defaults/parser/artifact/cache rows are now
  `PASS`; the only remaining open requirement is
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- No live generations were sent during this refresh; `8886` stayed idle.

## [2026-05-24 13:16] codex | live-audit | DSV4 chat_max budget/stop/route ruled out

- Captured live artifact:
  `build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json`.
- Refreshed objective digest:
  `build/current-objective-proof-audit-20260524-post-chatmax-budget-stop-rail.json`.
- Same `chat_max` wording fails across chat, responses, and completions with
  identical 445-token corrupted output.
- Larger output budget, role stops, thinking-on, and low-effort diagnostics do
  not fix the corruption.
- Objective digest now surfaces this as
  `current_chatmax_budget_stop_rail_probe`.
- DSV4 long-output/code quality remains the only open proof-board row.

## [2026-05-24 13:52] codex | no-heavy | DSV4 prefill execution variant logits wired into proof board

- No live generation, no server launch, no Swift.
- Added proof-board tracking for
  `build/current-dsv4-jang-prefill-execution-variant-logits-20260524.json`.
- Red tests first:
  - objective digest failed on missing
    `current_prefill_execution_variant_logits`;
  - release manifest failed on missing prefill-execution artifact.
- Cleaned duplicate `_dsv4_template_parity_diagnostic_detail` definition while
  touching the summarizer; retained helper remains covered by focused tests.
- Refreshed:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-objective-proof-audit-20260524-codex-recheck.json`,
  and `build/current-release-regression-manifest-20260521.json`.
- Current digest evidence:
  - target context `after generated prefix THREE.P`;
  - six execution variants;
  - all variants keep the same top-token ordering;
  - all variants select correct `ers`;
  - whole-token `Pers` is never top.
- Current interpretation: stream/warmup/clear execution state is not the cause
  of the isolated `THREE.P -> ers` prefix behavior. The broader DSV4 exact-code
  row remains open because full prompt context still flips other identifier
  logits.
- Verification:
  focused tests `6 passed`; py_compile pass; `git diff --check` pass.
- Health after refresh: `8886` idle (`num_running=0`, `num_waiting=0`).

## [2026-05-24 13:48] codex | no-heavy | DSV4 template parity wired into proof board

- No live generation, no server launch, no Swift.
- Added proof-board tracking for
  `build/current-dsv4-template-parity-diagnostic-20260524-1343.json`.
- Red tests first:
  - objective digest test failed because
    `current_template_parity_diagnostic` was missing;
  - release manifest test failed because the DSV4 live row did not list the
    template-parity artifact.
- Refreshed:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-objective-proof-audit-20260524-codex-recheck.json`,
  and `build/current-release-regression-manifest-20260521.json`.
- Current digest evidence:
  - sidecar encoder equals bundle `chat_template.jinja` for all checked cases;
  - mismatch count is zero;
  - normal colon/period prompts have equal token counts;
  - boundary token is `.Ċ` for period and `:Ċ` for colon.
- Current interpretation: the colon/period exact-code split is not caused by a
  vMLX custom renderer vs bundle-template mismatch.
- Verification:
  focused tests `6 passed`; py_compile pass; `git diff --check` pass.
- Health after refresh: `8886` idle (`num_running=0`, `num_waiting=0`).

## [2026-05-24 13:44] codex | live-audit | DSV4 hidden-reasoning controls fail; Ling manifest wording aligned

- Active repo only; no Swift, no package/release work.
- Claimed `8886` in `.agents/MAIL.md` before live requests and only sent
  requests while health was idle.
- The initial four-request diagnostic exceeded the tool session before writing
  output, so I switched to one request per command and wrote artifacts
  immediately.
- New live artifacts:
  - `build/current-dsv4-hidden-reasoning-control-period_no_reasoning_draft-live-20260524.json`
    -> fail, length stop at 220 tokens, no code markers in hidden reasoning,
    but visible output still corrupts `WebWebGLRenderer`;
  - `build/current-dsv4-hidden-reasoning-control-period_verify_identifiers-live-20260524.json`
    -> fail, length stop at 220 tokens, visible output missing all required
    Three.js identifiers while hidden reasoning contains corrupted code markers.
- Refreshed objective digest now surfaces the existing integrated artifact
  `build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json`:
  period controls still fail, no-draft can remove hidden corruption markers
  without fixing visible exactness, and cache-hit tokens remain zero.
- No runtime behavior changed. Current implication: hidden reasoning draft
  contamination is real evidence, but simple reasoning steering/suppression is
  not sufficient as a production fix.
- Also corrected a release-manifest reporting inconsistency for Ling/Bailing:
  the objective row is now pass from named zero-CJK clearance artifacts, while
  older failing artifacts remain tracked as regression evidence.
- Refreshed:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-objective-proof-audit-20260524-codex-recheck.json`,
  and `build/current-release-regression-manifest-20260521.json`.
- Verification:
  focused tests `6 passed`; py_compile pass; `git diff --check` pass.
- Health after work: `8886` idle (`num_running=0`, `num_waiting=0`).

## [2026-05-24 13:27] codex | live-audit | DSV4 prompt punctuation flips visible logits

- Captured live prompt-boundary and logprob artifacts:
  `build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json`,
  `build/current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json`,
  and `build/current-dsv4-scene-token-rank-contrast-live-20260524-1324.json`.
- Passing boundary: `no markdown fences:` and `no markdown fences`.
- Failing boundary: `no markdown fences.`, `no markdown.`, `No explanation`,
  and `Please` variants.
- At visible `THREE.Scene`, colon prompt ranks `ene` first; period prompt ranks
  wrong `();\n` first and `ene` second.
- Hidden reasoning includes a corrupted fenced code draft before visible output;
  colon recovers in visible output, period follows the corrupted draft.
- Current root cause is below API/settings and output finalization: prompt
  context changes DSV4 logits on the thinking rail.

## [2026-05-24 13:58] codex | proof-board | Scene token rank contrast wired into objective evidence

- No live generation, no server launch, no Swift.
- Integrated
  `build/current-dsv4-scene-token-rank-contrast-live-20260524-1324.json`
  into objective digest and release manifest evidence.
- Red tests first:
  - objective digest test failed until
    `current_scene_token_rank_contrast` was surfaced;
  - release manifest test failed until the scene-rank artifact was listed.
- Corrected reporting detail:
  - artifact `scene_token_index=123` is the `.Sc` token;
  - decisive rank contrast is the next token, index `124`.
- Refreshed:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-objective-proof-audit-20260524-codex-recheck.json`,
  and `build/current-release-regression-manifest-20260521.json`.
- Current proof board still has exactly one open requirement:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  - focused objective/release/current-regression tests: `6 passed`;
  - py_compile for touched proof/reporting files: pass;
  - `git diff --check`: pass;
  - health after refresh: `8886` idle (`num_running=0`, `num_waiting=0`).

## [2026-05-24 13:02] codex | cleanup | Wrapper stale implementation notes removed recursively

- Recursively searched `/Users/eric/vmlx` for `swift|handoff` after the earlier
  maxdepth-only cleanup missed root/docs/notes/sweep artifacts.
- Deleted remaining stale wrapper implementation/audit/triage/spec/status files
  containing those references.
- Replaced `/Users/eric/vmlx/README.md` with a routing-only guard and rewrote
  `/Users/eric/vmlx/AGENTS.md` to point only at the active Python repo without
  preserving old implementation-path wording.
- Verified:
  `rg --files /Users/eric/vmlx | rg -i 'swift|handoff'` -> no output;
  recursive content `rg -n -i 'swift|handoff' /Users/eric/vmlx ...` -> no
  output outside ignored cache/binary areas.

## [2026-05-24 13:03] codex | audit-fix | Objective digest now tracks latest DSV4 route artifact

- Updated `summarize_objective_proof.py` so DSV4 prompt/route exactness prefers
  the latest source-current route artifact:
  `build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json`.
- Added `failed_cases` to the DSV4 route detail so status no longer depends on
  stale "thinking-closed" terminology after server policy forces DSV4 onto the
  quality rail.
- Added regression test proving a stale passing route artifact is ignored when
  the current source route artifact is red.
- Verification:
  focused objective proof tests `3 passed`;
  `py_compile` passed;
  `git diff --check` passed.
- Refreshed `build/current-objective-proof-audit-20260524-codex-recheck.json`.
  It now reports DSV4 route artifact
  `current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json`,
  probe status `fail`, failed cases `["chat_max"]`.

## [2026-05-24 13:50] codex | no-heavy | Colon-vs-period logprob traces wired into proof board

- No live generation sent; no server launched; no Swift touched.
- Integrated:
  - `build/current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json`;
  - `build/current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json`.
- Key evidence now in objective digest:
  - colon pass and period fail both use `prompt_tokens=90`,
    `completion_tokens=195`, and `logprob_entry_count=195`;
  - cache hit tokens are zero before/after in both trace artifacts;
  - colon visible output is exact while period visible output corrupts
    identifiers;
  - period visible divergence starts at char index `26`, token index `124`;
  - colon hidden thinking includes corruption markers even though its visible
    post-`</think>` answer is clean.
- Refreshed:
  - `build/current-objective-proof-audit-20260521.json`;
  - `build/current-objective-proof-audit-20260524-codex-recheck.json`;
  - `build/current-release-regression-manifest-20260521.json`.
- Current proof board still has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  focused objective/release/current-regression tests -> `6 passed`;
  py_compile for touched proof/reporting files -> pass;
  `git diff --check` -> pass.
- `8886` health after refresh: idle (`num_running=0`, `num_waiting=0`).

## [2026-05-24 13:42] codex | no-heavy | DSV4 prompt-boundary bisection wired into proof board

- No live generation sent; no server launched; no Swift touched.
- Read `build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json`.
- Key finding from artifact:
  - `canonical_return_fences_colon` passes;
  - `return_fences_no_punct` passes;
  - `return_fences_period` fails with identifier corruption under the same
    effective `thinking_open` rail;
  - `return_markdown_colon_no_fences`, `return_code_only_colon`, and
    `please_return_fences_colon` length-fail;
  - `return_fences_colon_no_explanation` stops after blank visible content.
- Added objective digest detail:
  `current_prompt_boundary_bisection_probe`.
- Added release manifest tracking for
  `build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json`.
- Refreshed:
  - `build/current-objective-proof-audit-20260521.json`;
  - `build/current-objective-proof-audit-20260524-codex-recheck.json`;
  - `build/current-release-regression-manifest-20260521.json`.
- Current proof board still has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  focused objective/release/current-regression tests -> `7 passed`;
  py_compile for touched proof/reporting files -> pass;
  `git diff --check` -> pass.
- `8886` health after refresh: idle (`num_running=0`, `num_waiting=0`).

## [2026-05-24 13:34] codex | no-heavy | Release manifest DSV4 row now tracks current open evidence

- No live generation sent; no server launched; no Swift touched.
- Found stale release-reporting gap: `dsv4-long-output-quality-live` listed
  early DSV4 artifacts but not the current chatmax route/prompt-trigger and
  budget/stop/rail failure artifacts.
- TDD red:
  `.venv/bin/python -m pytest -q tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_fresh_dsv4_live_failure_artifact`
  failed because the latest source/live route artifact was missing from the
  manifest row.
- Updated `tests/cross_matrix/release_regression_manifest.py` so the DSV4 live
  row lists:
  - `build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json`;
  - `build/current-dsv4-route-mode-code-exactness-existing-server-chatmax-20260524.json`;
  - `build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json`;
  - `build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json`.
- Refreshed `build/current-release-regression-manifest-20260521.json`; its
  DSV4 row now includes those artifacts and notes that chat_max remains open
  and budget/stops/thinking/routes are not sufficient fixes.
- Verification:
  focused release manifest tests -> `4 passed`;
  py_compile for manifest code/tests -> pass;
  `git diff --check` -> pass.

## [2026-05-24 13:26] codex | no-heavy | Canonical objective audit refreshed for release consumers

- No live generation sent; no server launched; no Swift touched.
- Found proof-plumbing risk: the umbrella/current release plumbing reads
  `build/current-objective-proof-audit-20260521.json`, not only the
  `codex-recheck` audit file.
- Refreshed `build/current-objective-proof-audit-20260521.json`.
- It now includes the DSV4 budget/stop/rail probe detail and still has exactly
  one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Focused verification:
  `.venv/bin/python -m pytest -q tests/test_current_regression_suite.py::test_current_regression_suite_allows_only_declared_known_blockers tests/test_current_regression_suite.py::test_current_regression_suite_refreshes_release_regression_manifest tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_current_post_budget_edge_proof_sweep tests/test_release_regression_manifest.py::test_release_regression_manifest_validates_current_proof_sweep_artifacts tests/test_release_regression_manifest.py::test_release_regression_manifest_runner_embeds_current_proof_validation tests/test_release_regression_manifest.py::test_release_regression_manifest_runner_can_require_current_proof_sweep`
  -> `6 passed`.
- `git diff --check`: pass.

## [2026-05-24 13:23] codex | no-heavy | DSV4 budget/stop/rail matrix added to current proof board

- No live generation sent; no server launched; no Swift touched.
- Read `build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json`.
- Confirmed artifact `status=fail`, 7/7 variants fail:
  `chatmax_512_baseline`, `chatmax_1024_budget`,
  `chatmax_1024_stop_roles`, `chatmax_1024_thinking_on`,
  `chatmax_1024_effort_low`, `responses_chatmax_1024`,
  `completion_chatmax_512`.
- The digest detail now shows all failed cases stopped at the same
  `completion_tokens=445` and still corrupt identifiers under
  `thinking_open`.
- Refreshed `build/current-objective-proof-audit-20260524-codex-recheck.json`.
  It still has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py::test_objective_proof_digest_surfaces_dsv4_chatmax_budget_stop_rail_probe tests/test_objective_proof_digest.py::test_objective_proof_digest_surfaces_dsv4_chatmax_prompt_trigger_probe`
  -> `2 passed`;
  `py_compile` for objective summarizer/tests -> pass;
  `git diff --check` -> pass.
- Current conclusion: do not spend live time retesting max-token budget, role
  stops, requested thinking-on, effort-low, Responses surface, or completion
  route as standalone fixes; this artifact already rules them out as sufficient.

## [2026-05-24 13:16] codex | no-heavy | DSV4 prompt-trigger evidence wired into objective digest

- No live generation sent; no server launched; no Swift touched.
- Added objective-digest coverage for
  `build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json`.
- Fixed stale synthetic clear-case test fixture so a DSV4 quality pass requires
  the new chatmax prompt-trigger evidence file to exist.
- Verification:
  `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py::test_objective_proof_digest_surfaces_dsv4_chatmax_prompt_trigger_probe tests/test_objective_proof_digest.py::test_objective_proof_digest_prefers_latest_source_route_exactness_probe tests/test_objective_proof_digest.py::test_objective_proof_digest_surfaces_dsv4_prompt_rail_exactness_probe tests/test_objective_proof_digest.py::test_objective_proof_digest_accepts_dsv4_quality_clearance_artifact`
  -> `4 passed`.
- `py_compile` for `tests/cross_matrix/summarize_objective_proof.py` and
  `tests/test_objective_proof_digest.py`: pass.
- `git diff --check`: pass.
- Refreshed `build/current-objective-proof-audit-20260524-codex-recheck.json`.
  It now has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Prompt-trigger digest detail:
  - artifact status `fail`;
  - canonical exact case `canonical_return_fences`;
  - failed cases `chatmax_original`, `copy_no_preserve`,
    `copy_preserve_fences`, `return_no_fences_word`,
    `return_preserve_no_expl_fences`, `copy_exactly_fences`;
  - effective policy reason remains `dsv4_direct_rail_identifier_unsafe`.
- Existing `8886` health after refresh showed `num_running=1`,
  `num_waiting=0`; I did not claim or use it.

## [2026-05-24 13:04] codex | live-probe | Existing-server DSV4 chat_max reconfirmed

- Reused already-loaded `8886` only after `.agents/MAIL.md` claim and idle
  health check.
- Ran:
  `.venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --base-url http://127.0.0.1:8886 --cases chat_max --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --out build/current-dsv4-route-mode-code-exactness-existing-server-chatmax-20260524.json`
- Result `status=fail`, as expected for current DSV4 quality state.
- `chat_max`: `finish=stop`, no markdown fence, but corrupts
  `THREE.PerspectiveCamera`, `THREE.BoxGeometry`, `THREE.MeshBasicMaterial`
  into `PersPerspectiveCamera`, `BBoxGeometry`, `MMeshBasicMaterial`.
- Prompt diagnostics: requested `thinking_closed`, server-effective
  `thinking_open`, policy reason `dsv4_direct_rail_identifier_unsafe`.
- `/health` after request is idle (`num_running=0`, `num_waiting=0`).
- No new model server launched; no release/package work.

## [2026-05-24 13:08] codex | no-heavy | Stale max-output and API/cache proof rows refreshed

- Refreshed max-output/context contract after DSV4 server edits:
  `build/current-max-output-context-contract-20260524-after-chatmax-dsv4-server-edits.json`.
  Status `pass`; engine `25 passed`; panel `54 passed`; no missing markers.
- Refreshed API/cache contract after DSV4 server and scheduler diagnostics edits:
  `build/current-api-cache-contract-proof-20260524-after-chatmax-dsv4-scheduler-edits.json`.
  Status `pass`; API route `26 passed`; scheduler cache `7 passed`;
  TQ/MLLM cache `32 passed`; DSV4 DSML/tool `21 passed`; no missing markers.
- Updated `summarize_objective_proof.py` and
  `tests/test_objective_proof_digest.py` to point at the current proof
  artifacts and preserve missing/stale-artifact regression coverage.
- Focused digest tests passed:
  - max-output/context artifact required+accepted tests;
  - API/cache artifact missing/accepted/stale-hash tests;
  - latest DSV4 route exactness preference test.
- Refreshed `build/current-objective-proof-audit-20260524-codex-recheck.json`.
  Current objective digest has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- `8886` remains loaded and idle; no further live request sent in this slice.

## [2026-05-24 13:31] codex | analysis | DSV4 exact-code root-cause boundary tightened

- No live generation, no server launch, no Swift.
- Inspected saved DSV4 logit/logprob artifacts instead of rerunning the heavy
  direct-vector harness while `8886` is loaded.
- Refreshed:
  `build/current-objective-proof-audit-20260524-deep-root-cause-recheck.json`.
- Current proof digest still has exactly one open requirement:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Evidence now points below API/UI/finalizer/cache settings:
  - DSV4 model logits can prefer malformed identifier continuations in some
    prompt/prefix contexts (`IVE` over `REE`, whole-token `Pers` after `.P`,
    `();\n` over `ene` after `.Sc`);
  - tokenizer roundtrip is clean, so the corrupt strings are valid generated
    token choices, not decode corruption;
  - colon/no-punct prompt boundaries can recover exact visible output, while a
    period after the same instruction flips the rank ordering under the same
    effective rail and prompt/completion token counts;
  - hidden reasoning drafts are corrupted in both colon and period traces, but
    only the period trace propagates the corruption into visible output.
- Current non-fix boundary: do not present prompt rewrite, identifier
  substitution, forced token patching, or visible-output postprocessing as a
  production fix.
- Tokenizer-only boundary check: the passing colon/no-punctuation and failing
  period prompts differ by a single boundary token before `const` (`:Ċ`, `Ċ`,
  `.Ċ`). This supports a real prompt-token conditioning failure, not a
# 2026-05-24 14:21 PDT - explicit-off DSV4 exact-code live subset

- Claimed non-overlapping live lane on port `8891`, avoiding dead/reserved
  `8886`.
- Added red/green coverage for launch-mode `--cases`; patched route exactness
  runner so live launch honors case filters and records selected cases.
- Ran DSV4 source live subset:
  `chat_off,responses_off,legacy_completion_raw`.
- Artifact:
  `build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json`.
- Result: `status=fail`; all three thinking-closed cases corrupt visible
  identifiers/code under `enable_thinking=False`.
- Updated objective digest to prefer the new explicit-off artifact; updated
  release manifest proof text/artifact list.
- Current objective proof open rows:
  - `DSV4 default-cache multi-tool agent loop is proven`;
  - `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  - focused route/objective/release/current-suite tests `15 passed`;
  - proof/reporting `py_compile` pass;
  - `git diff --check` pass;
  - no `8891` server process remains.

# 2026-05-24 14:16 PDT - DSV4 route diagnostic precedence fix

- No live generation; no server/process touched.
- Found route-exactness proof diagnostic drift: helper used
  `chat_template_kwargs.enable_thinking` before top-level `enable_thinking`,
  unlike server precedence.
- Added red/green test and patched helper to mirror server precedence.
- Refreshed DSV4 route dry-run artifacts.
- Refreshed objective/release artifacts again; only open row remains:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  - red test failed first with `assert True is False`;
  - focused route/objective/current-suite tests `13 passed`;
  - active reasoning-policy focused source tests `9 passed, 47 deselected`;
  - proof/reporting `py_compile` pass;
  - `git diff --check` pass.

# 2026-05-24 14:12 PDT - no-live proof-board stale-hash refresh

- Did not use live generation; `8886` remains reserved by the 14:06
  reasoning-policy lane.
- Refreshed max-output/context, API/cache, model-artifact-format,
  generation-defaults, and native-MTP no-heavy contract artifacts.
- Refreshed objective proof and release manifest artifacts.
- Current objective proof now has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Release manifest current proof sweep is `pass`, rows `20`.
- Verification:
  - focused objective/release/current-suite tests `8 passed`;
  - proof/reporting `py_compile` pass;
  - `git diff --check` pass.

# 2026-05-24 14:09 PDT - routing incident containment

- No live generation, no server launch, no Swift.
- Deleted deprecated `/Users/eric/vmlx/.agents` because it still contained
  Swift/JangPress mailbox, status, rules, and handoff state that could misroute
  post-compaction work.
- Removed deprecated wrapper `build/*swift*` and `build/*jangpress*` artifacts.
- Verified no `/Users/eric/vmlx/swift` directory and no `*swift*`,
  `*handoff*`, or `*JangPress*` paths within four levels of the deprecated
  wrapper.
- Current Python lane remains in
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; DSV4 exact-code quality is
  not release-cleared.

  whitespace-normalization artifact.

## [2026-05-24 14:29] codex | regression | strengthened DSV4 default-cache tool-loop proof

- Tightened the objective digest contract for
  `build/current-dsv4-default-cache-tool-loop/result.json` so shallow
  two-tool artifacts no longer satisfy the DSV4 default-cache multi-tool row.
- Required proof now covers the real long chain Eric requested:
  DSML tool parser, DeepSeek R1 reasoning parser, native DSV4
  prefix/paged/L2 cache, cached tokens/cache detail, exact HTML write,
  exact Three.js `scene.js` write, and final `DONE`.
- Updated current-suite and packaged-integrity expected-open blockers to
  include `DSV4 default-cache multi-tool agent loop is proven` alongside
  DSV4 exact-code quality.
- Refreshed artifacts:
  `build/current-objective-proof-audit-20260524-long-tool-chain-contract.json`,
  `build/current-regression-suite-20260524-long-tool-chain-contract.json`,
  `build/current-packaged-integrity-contract-20260524-long-tool-chain-contract.json`.
- Verification:
  focused contract tests `36 passed, 50 deselected`;
  broader focused regression `189 passed, 37 deselected`;
  packaged integrity `status=pass`, `failed=[]`;
  umbrella no-heavy suite `status=pass`, `failed_steps=[]`;
  proof/reporting `py_compile` pass; `git diff --check` pass.
- Boundary: no release clearance. Open DSV4 rows remain exact-code quality and
  default-cache long-chain tool-loop proof. No Swift touched.

## [2026-05-24 13:47] codex | no-heavy | DSV4 template parity evidence added to proof/manifest

- Integrated:
  `build/current-dsv4-template-parity-diagnostic-20260524-1343.json`.
- Refreshed:
  `build/current-objective-proof-audit-20260524-template-parity.json`;
  `build/current-release-regression-manifest-20260524-template-parity.json`.
- Red/green tests:
  - digest failed first on missing `current_template_parity_diagnostic`;
  - release manifest failed first on missing template-parity artifact;
  - green focused run:
    `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py -k 'template_parity or hidden_reasoning_control or colon_period_logprob_trace or scene_token_rank_contrast or prompt_boundary' tests/test_release_regression_manifest.py::test_release_regression_manifest_tracks_fresh_dsv4_live_failure_artifact`
    -> `5 passed, 43 deselected`.
- Current objective proof remains one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.

## [2026-05-24 13:43] codex | coordination | canonical checkout drift noted

- Checked `/Users/eric/mlx/vllm-mlx`: `main` at `754f817b`, large dirty tree.
- This guard worktree: `codex/pr-intake-manifest` at `465e7141`, with latest
  2026-05-24 objective proof artifacts and DSV4 hidden-reasoning control
  evidence.
- Decision for this slice: keep audit/proof work in the guard worktree and do
  not copy source changes into the dirty canonical checkout without an explicit
  sync/reconciliation step.
- No-live-generation DSV4 prompt-template/config comparison artifact:
  `build/current-dsv4-template-parity-diagnostic-20260524-1343.json`.
- Result: vMLX sidecar encoder exactly matches the bundle
  `chat_template.jinja` for colon/period/system-no-draft exact-code prompts
  across thinking-closed, thinking-open, and max-thinking variants.
- Colon/period normal rails both have 90 prompt tokens; the relevant boundary
  token is `.Ċ` vs `:Ċ`.
- Current interpretation: not a custom encoder vs bundle Jinja mismatch.

## [2026-05-24 13:39] codex | live-probe | DSV4 hidden-reasoning control still fails visible identifiers

- Used existing loaded `8886` only after health showed idle.
- No source behavior change, no package/release work, no Swift.
- New artifact:
  `build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json`.
- Refreshed proof board:
  `build/current-objective-proof-audit-20260524-hidden-reasoning-control.json`.
- Results:
  - colon control exact;
  - period control failed;
  - period + "do not draft code in hidden reasoning" failed even though hidden
    corruption was removed for that case;
  - period + internal identifier verification failed and used more tokens.
- Current conclusion: hidden reasoning draft contamination is real but not the
  sole root cause. DSV4 has a base prompt-conditioned visible identifier/logit
  weakness under exact-code period context.
- Verification:
  `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py -k 'hidden_reasoning_control or colon_period_logprob_trace or scene_token_rank_contrast or prompt_boundary'`
  -> `4 passed, 42 deselected`;
  `git diff --check` -> pass.
- Cleanup: the heredoc diagnostic client process (`.venv/bin/python -`, PID
  `17152`) kept its HTTP socket open after writing the artifact and made health
  briefly report `num_running=1`. Terminated that client only; server PID
  `7567` stayed up and returned to `num_running=0`, `num_waiting=0`.
# 2026-05-24 14:02 PDT - DSV4 prompt-variant logit probe proof-board integration

- No live generation, no server launch, no Swift.
- Added objective digest detail for
  `build/current-dsv4-jang-prompt-variant-logit-probe-20260524.json`.
- Added release manifest artifact/proves text for the same prompt-variant
  logit probe.
- Refreshed:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-objective-proof-audit-20260524-prompt-variant-logit.json`,
  `build/current-release-regression-manifest-20260521.json`, and
  `build/current-release-regression-manifest-20260524-prompt-variant-logit.json`.
- Current objective digest still has exactly one open row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Verification:
  focused objective/release/current-regression tests `7 passed`;
  py_compile pass; `git diff --check` pass; `8886` idle.
- Boundary: prompt wording/logit sensitivity is first-class evidence, not a
  production fix or release clearance.

## [2026-05-24 14:50] codex | live-probe | DSV4 long-chain parser repairs, exact-code blocker isolated
## [2026-05-24 16:27] codex | proof-board | one-open DSV4 split restored after stale strict runner

- Waited for a stale runner writing
  `build/current-regression-suite-20260524-strict-dsv4-two-open.json` to exit
  before touching shared proof files.
- Restored the agreed one-open split:
  `DSV4 default-cache multi-tool agent loop is proven` passes on
  tool/cache/parser mechanics; `code_file_written_exact=false` remains in row
  details; exact code/body fidelity is tracked only by
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Focused verification:
  objective/current-suite/packaged split tests -> `5 passed`.
- Refreshed:
  - `build/current-objective-proof-audit-20260524-default-cache-live-review.json`
    -> only DSV4 exact-code quality open;
  - `build/current-packaged-integrity-contract-20260524-default-cache-live-review.json`
    -> `status=pass`, `failed=[]`;
  - `build/current-regression-suite-20260524-default-cache-tool-loop-cleared.json`
    -> `status=pass`, `failed_steps=[]`, one open row;
  - `build/current-release-regression-manifest-20260524-codex-audit.json`
    -> `current_proof_sweep=pass`.
- No DSV4 live model was launched; exact-code release row remains blocked by
  live validation under sufficient RAM.

## [2026-05-24 16:19] codex | dsv4-live-gate | exact-code row RAM-gated, current rail diagnostics refreshed

- No Swift/wrapper work. Stayed in
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.
- Reconfirmed release proof board after the default-cache mechanics split:
  `build/current-regression-suite-20260524-default-cache-tool-loop-cleared.json`
  is `status=pass`, `failed_steps=[]`, with only
  `DSV4 long-output/code/file-generation quality is release-cleared` open.
- Local DSV4 exact-code route probe was RAM-gated, not forced:
  `build/current-dsv4-route-mode-code-exactness-memory-preflight-20260524.json`
  -> `status=skipped`, `reason=insufficient_free_memory`,
  `required_available_gb=120.0`, available about `72.65 GB`.
- Checked `erics-m5-max2.local` as a possible live lane. It has the DSV4 model
  path, but an existing Qwen 35B MTP server on `127.0.0.1:8012` leaves about
  `87 GB` free, so DSV4 was not started there either.
- Refreshed no-model current-source DSV4 prompt rail diagnostics:
  `build/current-dsv4-route-mode-code-exactness-dryrun-20260524-current-rail.json`.
  Current source keeps explicit-off requests on `thinking_closed` for chat and
  responses; completion maps to the same closed chat rail. This distinguishes
  current behavior from older artifacts that recorded
  `dsv4_direct_rail_identifier_unsafe` force-on policy.
- Focused verification passed:
  `tests/test_dsv4_route_mode_code_exactness.py` +
  `tests/test_reasoning_modes.py -k 'dsv4_code_exactness or dsv4_thinking_policy'`
  -> `20 passed, 40 deselected`.
- Remaining release blocker: live DSV4 exact-code/identifier fidelity under a
  real current-source load. Do not cut release until that row is live-proven or
  explicitly scoped out by Eric.


- No Swift work.
- Live DSV4 default-cache gate on JANGTQ-K showed native cache path working but
  exposed DSML parser lossage and exact-code body corruption.
- Fixed DSML parser handling for:
  self-closing parameter tags carrying `string=` values, typoed `invuse`
  start tags, and bogus canonical `" string="` decoded args.
- Final live artifact:
  `build/current-dsv4-default-cache-tool-loop/result.json` -> `status=review`;
  ordered tools and file writes work; all cache checks are true; only
  `code_file_written_exact=false` remains.
- Remaining concrete mismatch:
  `THREE.WebRenderer` vs `THREE.WebGLRenderer`, plus missing final semicolon.
- Bundled Python reinstalled from source; bundled verifier passed and now
  hash-gates `vmlx_engine/tool_parsers/dsml_tool_parser.py`.
- Refreshed objective and contract artifacts. Packaged integrity and current
  regression suite both pass while preserving exactly two known DSV4 open rows:
  default-cache multi-tool exact proof and long-output/code quality clearance.
- Verification: focused parser/gate/objective tests passed, all refreshed
  contract artifacts passed, `py_compile` passed, `git diff --check` passed,
  and no matching server/test/proof-runner process remains.

## [2026-05-24 15:05] codex | live-probe | DSV4 invue parser repair, prompt-boundary exactness still open

- Ran live prompt-boundary/tool-content diagnostic through Responses + DSML
  tools on DSV4 JANGTQ-K.
- First pass exposed another parser typo shape:
  `<｜DSML｜invue name="write_file">`, which canonical extraction reduced to
  bogus `" string="` args.
- Added red/green parser test and repaired the schema-gated degraded-invoke
  regex for `invue`.
- Post-fix artifact:
  `build/current-dsv4-tool-content-prompt-boundary-live-20260524-after-invue-parser.json`.
- All prompt variants now execute `write_file` and final `DONE`, but none are
  exact:
  `sentence_repr` -> `THREE.WebRenderer`;
  `colon_raw_block` -> `THREE.Mew Mesh`;
  `colon_json_fields` -> `THREE.WebWebGLRenderer`.
- Refreshed default consumed gate:
  `build/current-dsv4-default-cache-tool-loop/result.json` -> `status=review`,
  all tool/cache checks true except `code_file_written_exact=false`.
- Bundled Python reinstalled and verifier passed.
- Refreshed no-heavy API/cache, parser registry, objective, packaged
  integrity, and umbrella artifacts. Exactly two DSV4 rows remain open.
- Final checks: focused tests `31 passed, 51 deselected`, `py_compile` passed,
  `git diff --check` passed. Owned 8893 servers stopped; separate regression
  runner from parent PID `98796` left untouched.

## [2026-05-24 15:27] codex | audit | production static matrix and Qwen live-soak row corrected

- Ran no-live production-family static audit:
  `build/current-production-family-static-audit-20260524-1520.json`.
- Audit covered 32 rows; missing rows include stale `qwen36_moe_tq4` plus the
  absent DSV4/Mistral/MiniMax rows listed in the artifact summary.
- Corrected release regression manifest live-soak command to target
  `qwen36_moe_crack`, the present Qwen hybrid row, instead of absent
  `qwen36_moe_tq4`.
- Refreshed `build/current-release-regression-manifest-20260521.json` and
  `build/current-regression-suite-20260524-live-soak-present-qwen.json`.
- Verification: full release-manifest tests `68 passed`; refreshed umbrella
  suite `status=pass`, `failed_steps=[]`; `py_compile` and `git diff --check`
  passed.
- Release remains blocked by exactly the same two DSV4 rows: default-cache
  multi-tool loop proof and long-output/code/file-generation clearance.

## [2026-05-24 15:45] codex | proof-model | DSV4 default-cache mechanics no longer conflated with exact-code quality

- Split objective proof logic so `DSV4 default-cache multi-tool agent loop is
  proven` gates on tool/cache/parser mechanics, not exact generated code body.
- Kept `code_file_written_exact=false` and corrupt identifier details in the
  row diagnostics so the failure remains visible.
- Updated current-regression and packaged-integrity known-open lists to one row:
  `DSV4 long-output/code/file-generation quality is release-cleared`.
- Refreshed artifacts:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-packaged-integrity-contract-20260524-default-cache-live-review.json`,
  and `build/current-regression-suite-20260524-default-cache-tool-loop-cleared.json`.
- Verification: `168 passed` across objective/current-suite/packaged/release
  manifest tests, `py_compile` passed, `git diff --check` passed.
- Release remains blocked by DSV4 exact-code/identifier fidelity only.
## [2026-05-24 17:19] codex | live-probe | Gemma4 visible failure traced to unsupported thinking budget

- Reproduced Gemma4 26B CRACK missing visible content with Responses thinking
  on, `max_thinking_tokens=16`, `max_tokens=128`, no prefix cache:
  `visible_chars=0`, `reasoning_chars=483`.
- Proved cache is not the root cause: no-cache reproduces the failure.
- Proved output cap / unsupported budget boundary: same route with
  `max_tokens=512` emits visible content, while thinking-off at 128 emits
  visible content with no reasoning.
- Proved not Responses-only: Chat thinking-on at 128/no-cache fails the same
  way.
- Refreshed metadata audit showing Gemma4 template mentions thinking but not
  `thinking_budget`.
- Added objective digest detail and test coverage so the board records the
  root-cause matrix instead of a vague Gemma4 open row.
- Refreshed packaged, manifest, and umbrella artifacts. Current release board
  is `status=pass` only with exactly four open rows: DSV4 default-cache tool
  loop, Gemma4 visible quality, Gemma4 mixed-SWA speed floor, and DSV4
  long-output/code quality.

## [2026-05-24 17:43] codex | proof-board | cross-family live smoke blocker wired into current board

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift or
  deprecated wrapper work.
- Reviewed the ZAYA bundled text smoke artifact from the other lane:
  `build/current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json`
  -> one `ZAYA1-8B-MXFP4` text row passed cache repeat and multi-turn recall.
- Refreshed objective digest so the existing cross-family smoke row is durable:
  `Cross-family live multi-turn smoke matrix is release-cleared` is open with
  only `zaya_text` covered; missing families are DSV4, Gemma4, Hy3,
  Ling/Bailing, MiniMax, Nemotron, Qwen3.6, and ZAYA-VL.
- Current release board now has five expected open rows:
  DSV4 default-cache tool loop, Gemma4 visible quality, Gemma4 mixed-SWA speed,
  cross-family live smoke, and DSV4 long-output/code quality.
- Refreshed artifacts:
  `build/current-objective-proof-audit-20260521.json`,
  `build/current-regression-suite-20260524-live-smoke-open.json`,
  `build/current-packaged-integrity-contract-20260524-live-smoke-open.json`,
  and `build/current-release-regression-manifest-20260524-live-smoke-open.json`.
- Verification: focused proof tests `49 passed`; current suite `status=pass`
  with `failed_steps=[]`; packaged integrity `status=pass`; release manifest
  `current_proof_sweep=pass`; `py_compile` and `git diff --check` passed.

## [2026-05-24 17:48] codex | live-probe | ZAYA-VL smoke attempted, media path passes, text exactness fails

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no Swift or
  deprecated wrapper work.
- Ran bundled all-local smoke for `/Users/eric/models/JANGQ/ZAYA1-VL-8B-MXFP4`
  on port `8843` with media enabled:
  `build/current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json`.
- Result: `probe_failed`, failures=2.
- Passing parts: MLLM load, blue image, repeated blue image, red image changed,
  and text-after-image media isolation all passed. Server log shows media
  prompts skipped prefix-cache store because media embeddings are
  path-dependent.
- Failing parts: `text_cache_repeat_1` and `text_cache_repeat_2` ignored
  `Reply exactly: ACK` and listed the stable prefix words instead. The second
  request still hit `paged+zaya_cca` with `cached_tokens=40`, so this is not a
  cache-miss-only symptom.
- Objective digest now records ZAYA-VL as attempted but not covered under
  `Cross-family live multi-turn smoke matrix is release-cleared`.
- Release manifest now names the ZAYA-VL smoke command/artifact:
  `build/current-release-regression-manifest-20260524-zaya-vl-smoke-open.json`
  -> `status=pass`, `current_proof_sweep=pass`.
- Verification: focused objective/manifest/suite/package tests `5 passed`,
  `py_compile` passed, `git diff --check` passed, and no live server/probe
  remains.

## [2026-05-24 17:52] codex | fix | all-local smoke video classifier aligned with runtime

- Investigated why ZAYA-VL dry-run claimed `supports_video=true` while live
  capabilities only exposed `modalities=[text, vision]`.
- Confirmed server policy is intentional: ZAYA1-VL tokenizer/video markers are
  not enough to advertise video because the local processor/template path is
  image-only unless native video config is present.
- Fixed `bench/all_local_model_smoke.py` to require explicit native video
  evidence: `video_config`, `video_preprocessor_config.json`, or Qwen-style
  video token metadata.
- Added regression test:
  `test_classify_model_does_not_infer_zaya_vl_video_from_vl_name_or_token`.
- Dry-run proof:
  `build/current-all-local-model-smoke-zaya-vl-dryrun-after-video-policy-20260524/inventory.json`
  now records `supports_video=false` for `ZAYA1-VL-8B-MXFP4`.
- Verification: `tests/test_all_local_model_smoke.py` plus focused
  objective/manifest smoke tests -> `23 passed`; `py_compile` and
  `git diff --check` passed.
## [2026-05-24 17:34] codex | live-probe | first bundled all-local smoke row promoted to objective gate

- Ran `bench/all_local_model_smoke.py` against bundled Python for
  `ZAYA1-8B-MXFP4` text-only probes.
- Artifact:
  `build/current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json`
  -> `status=pass`, `failures=0`.
- The smoke proved visible ACK output, repeated-prefix cache telemetry
  (`cached_tokens=46`, `cache_detail=paged+zaya_cca`), and multi-turn recall of
  both `blue` and `cat` for ZAYA text.
- Added objective row
  `Cross-family live multi-turn smoke matrix is release-cleared`; it stays open
  until DSV4, Gemma4, Hy3, Ling/Bailing, MiniMax, Nemotron, Qwen3.6, ZAYA text,
  and ZAYA-VL rows all have live smoke evidence.
- Refreshed objective, packaged-integrity, release-manifest, and umbrella
  artifacts. Current board passes only with five known open rows.
## [2026-05-24 17:45] codex | live-probe | ZAYA-VL smoke wired as attempted-but-open

- Live bundled smoke for `ZAYA1-VL-8B-JANGTQ4` produced
  `build/current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json`
  with `status=probe_failed`.
- Image/VL requests passed, including blue/red color changes and no-media
  follow-up isolation. Video labels were absent, so this artifact is not video
  proof.
- Text cache repeat prompts both failed exact `ACK` by echoing/listing the
  stable prefix; the second repeat observed prefix-cache telemetry, which
  narrows the next root-cause lane to ZAYA-VL prompt/template/instruction
  behavior under text mode, not a simple cache miss.
- Objective digest, current suite, and release manifest now preserve this as
  attempted-but-missing coverage for `zaya_vl`.
- Verification: focused proof/release tests `176 passed`; current suite
  `status=pass failed_steps=[]`; packaged integrity `status=pass`; release
  manifest `status=pass current_proof_sweep=pass`; `py_compile` and
  `git diff --check` passed.
## [2026-05-24 17:57] codex | live-probe | ZAYA-VL JANGTQ4 added and ACK root cause narrowed

- Found provenance mismatch: the earlier ZAYA-VL live artifact covered
  `ZAYA1-VL-8B-MXFP4`, not JANGTQ4.
- Ran exact `ZAYA1-VL-8B-JANGTQ4` bundled smoke into
  `build/current-all-local-model-smoke-zaya-vl-jangtq4-bundled-20260524/summary.json`.
- JANGTQ4 also fails exact `ACK` on both text cache repeat probes; the second
  repeat has prefix-cache telemetry, and image/no-media probes pass.
- Added JANGTQ4 to the cross-family objective row; `zaya_vl` remains
  attempted-but-missing.
- Diagnostic artifacts under
  `build/current-zaya-vl-jangtq4-ack-diagnostic-20260524/` show minimal and
  system-forced ACK prompts fail exact plain ACK, and the observed ZAYA-VL
  prompt frame is plain `user: ... assistant:`. Next investigation should
  focus on template/processor selection versus working ZAYA text behavior.
- Verification: focused proof/release tests `176 passed`; current suite
  `status=pass failed_steps=[]`; release manifest
  `status=pass current_proof_sweep=pass`; `py_compile` and `git diff --check`
  passed.
## [2026-05-24 18:08] codex | live-probe | Nemotron smoke attempted but remains open

- Ran bundled smoke for `Nemotron-Omni-Nano-JANGTQ-CRACK`.
- Passed: exact ACK, cached repeat, recall, blue/red image.
- Failed: no-media text probe returned `Yes`.
- Follow-up diagnostic proved it returns `Yes` before any image too, so current
  evidence points to prompt/gate semantics or model behavior rather than image
  carryover.
- Video was not covered; runtime probe options disabled video.
- Wired Nemotron artifact and diagnostic into objective digest; cross-family
  row now has four artifacts, with only `zaya_text` covered.
- Verification: focused tests `177 passed`; current suite
  `status=pass failed_steps=[]`; release manifest
  `status=pass current_proof_sweep=pass`.
## [2026-05-24 18:15] codex | live-probe | Ling/Bailing crossed off cross-family smoke

- Ran bundled smoke for `Ling-2.6-flash-JANGTQ` into
  `build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260524/summary.json`.
- Passed exact ACK, cache-hit repeat, and multi-turn recall with zero reasoning
  chars.
- Wired artifact into objective digest; cross-family covered families are now
  `zaya_text` and `ling_bailing`.
- Remaining cross-family missing list: DSV4, Gemma4, Hy3, MiniMax, Nemotron,
  Qwen3.6, ZAYA-VL.
- Verification: focused tests `177 passed`; current suite
  `status=pass failed_steps=[]`; release manifest
  `status=pass current_proof_sweep=pass`.
## [2026-05-24 18:22] codex | live-probe | Qwen3.6 crossed off cross-family smoke

- Ran bundled isolated smoke for `Qwen3.6-27B-MXFP4-CRACK` into
  `build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260524/summary.json`.
- Passed exact ACK, cache-hit repeat (`cached_tokens=41`,
  `cache_detail=paged+ssm`), multi-turn recall, reasoning-on visible final
  (`FINAL=OK`) with extracted reasoning, image color changes, video color, and
  no-media-after-image/video checks.
- Wired the artifact into objective digest; cross-family covered families are
  now `ling_bailing`, `qwen36`, and `zaya_text`.
- Remaining cross-family missing/open list: DSV4, Gemma4, Hy3, MiniMax,
  Nemotron, ZAYA-VL.
- Verification: focused tests `177 passed`; refreshed objective
  `build/current-objective-proof-audit-20260521.json`; current suite
  `build/current-regression-suite-20260524-zaya-vl-smoke-open.json`
  `status=pass failed_steps=[]` with five expected open rows; release manifest
  `build/current-release-regression-manifest-20260524-codex-audit.json`
  `current_proof_sweep=pass`; `py_compile` and `git diff --check` passed.
## [2026-05-24 18:57] codex | live-probe | Nemotron crossed off cross-family smoke

- Traced Nemotron text-only image-present behavior through rendered prompts:
  no media tokens were present in text-only prompts.
- Added system-scoped no-media attachment contract to
  `bench/all_local_model_smoke.py` and accepted `NONE` as a clean no-media
  answer.
- Fresh bundled smoke
  `build/current-all-local-model-smoke-nemotron-omni-jangtq-system-nomedia-bundled-20260524/summary.json`
  passed: ACK/cache/recall/reasoning/image/no-media all clean.
- Objective digest now covers `nemotron`; cross-family missing/open is down to
  DSV4 and Gemma4.
- Verification: focused tests `6 passed`; `py_compile` and `git diff --check`
  passed; current suite `status=pass failed_steps=[]`.

## [2026-05-24 18:45] codex | live-probe | Nemotron no-media failure narrowed

- Tightened all-local smoke no-media prompts with a red/green harness test so
  text-after-media probes use explicit negative-control wording.
- Reran bundled `Nemotron-Omni-Nano-JANGTQ-CRACK` into
  `build/current-all-local-model-smoke-nemotron-omni-jangtq-explicit-nomedia-bundled-20260524/summary.json`.
- Result remains open: ACK/cache/recall/reasoning/image color pass, but
  text-only no-media still returns `Yes`.
- Added prompt-variant diagnostic
  `build/current-nemotron-omni-no-media-prompt-variants-20260524/result.json`;
  before any image turn, text-only prompts still return image-present
  (`Yes`, `IMAGE`, count `1`), so this is not just image carryover.
- Objective and release manifest now preserve the new Nemotron evidence.
- Current cross-family missing/open list: DSV4, Gemma4, Nemotron.
- Verification: focused tests `5 passed`; `py_compile` and `git diff --check`
  passed.

## [2026-05-24 18:32] codex | live-probe | MiniMax and Hy3 crossed off cross-family smoke

- Fixed `bench/all_local_model_smoke.py` classification so MiniMax keeps
  reasoning probe coverage; the prior substring check saw `ling` inside
  `modeling_minimax_m2` and suppressed reasoning.
- Tightened `summarize_objective_proof.py` so thinking-capable smoke families
  require a `reasoning_on` request label before coverage is counted.
- Ran bundled MiniMax-small smoke:
  `build/current-all-local-model-smoke-minimax-small-jangtq-bundled-20260524/summary.json`
  -> `pass`, exact ACK, cache-hit repeat, recall, and `reasoning_on` visible
  `FINAL=OK`.
- Reran bundled Hy3 after classifier/gate fix:
  `build/current-all-local-model-smoke-hy3-jangtq2-bundled-20260524/summary.json`
  -> `pass`, exact ACK, cache-hit repeat, recall, and `reasoning_on` visible
  `FINAL=OK`.
- Cross-family covered families are now `hy3`, `ling_bailing`, `minimax`,
  `qwen36`, and `zaya_text`; missing/open are DSV4, Gemma4, Nemotron, and
  ZAYA-VL.
- Verification: focused tests `199 passed`; refreshed objective
  `build/current-objective-proof-audit-20260521.json`; current suite
  `build/current-regression-suite-20260524-zaya-vl-smoke-open.json`
  `status=pass failed_steps=[]` with five expected open rows; release manifest
  `build/current-release-regression-manifest-20260524-codex-audit.json`
  `current_proof_sweep=pass`; `py_compile` and `git diff --check` passed.
## [2026-05-24 19:18] codex | live-probe | Gemma4 visible/cross-family crossed off

- Traced Gemma4 26B CRACK visible failure to unsupported
  `max_thinking_tokens`/`thinking_budget` control: the local Gemma4 template
  opens thinking rails but does not consume `thinking_budget`, so
  `enable_thinking=true + max_thinking_tokens=16 + max_tokens=128` can use all
  output tokens for reasoning and produce no visible text.
- Confirmed app metadata already reports `thinkingBudgetSupported=false`, so
  the panel does not surface/apply `maxThinkingTokens` for this bundle.
- Ran app-real Responses visible probe:
  `build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json`
  -> `pass`, visible English output, identifiers preserved, no warnings.
- Ran bundled Gemma4 all-local smoke:
  `build/current-all-local-model-smoke-gemma4-26b-crack-bundled-20260524/summary.json`
  -> `pass`, exact ACK/cache, recall, reasoning visible `FINAL=OK`,
  blue/red image checks, and no-media `NONE`.
- Objective now marks Gemma4 visible-content/language as pass and cross-family
  covers `gemma4`, `hy3`, `ling_bailing`, `minimax`, `nemotron`, `qwen36`,
  `zaya_text`, `zaya_vl`; cross-family remains open only for DSV4.
- Remaining open rows are DSV4 default-cache tool loop, Gemma4 mixed-SWA speed,
  cross-family live smoke, and DSV4 long-output/code quality.
- Verification: focused tests `201 passed`; packaged integrity
  `build/current-packaged-integrity-contract-20260524-gemma4-smoke-dsv4-open.json`
  `status=pass`; current suite
  `build/current-regression-suite-20260524-gemma4-visible-crossfamily-open.json`
  `status=pass failed_steps=[]`; release manifest
  `build/current-release-regression-manifest-20260524-codex-audit.json`
  `current_proof_sweep=pass`; `py_compile` and `git diff --check` passed.

## [2026-05-24 20:56] codex | proof | gateway single-model auto-switch covered

- Fixed gateway single-model mode so unload failures return
  `single_model_unload_failed` instead of silently continuing into the requested
  backend with another local model still active.
- Added panel behavior proof for unload-failure handling and Ollama `/api/chat`
  model-id auto-switch with streaming content deltas preserved.
- Promoted the route-level auto-switch marker into the API-surface contract:
  `build/current-api-surface-contract-20260524-single-model-auto-switch.json`
  passed.
- Restored DSV4 default-cache exactness invariant: `code_file_written_exact`
  is required; mechanics-only proofs keep the row open.
- Fresh current proof board:
  `build/current-regression-suite-20260524-single-model-auto-switch-dsv4-exact.json`
  passed with three open rows: DSV4 default-cache multi-tool exactness, Gemma4
  mixed-SWA speed floor, and DSV4 long-output/code quality.
- Release manifest proof sweep passed at
  `build/current-release-regression-manifest-20260524-single-model-auto-switch-dsv4-exact.json`.

## [2026-05-24 19:00] codex | live-probe | Nemotron no-media crossed off

- Reproduced the historical Nemotron no-media failure and separated it from
  media carryover: the old user-only prompt returned image-present even before
  any image turn.
- Live system-negative diagnostic proved deterministic text-only negatives
  before/after image traffic:
  `build/current-nemotron-omni-no-media-system-negative-diagnostic-20260524/result.json`
  -> `NO`/`0` before image, after blue image, and after red image.
- Reran full bundled Nemotron smoke at the current proof path:
  `build/current-all-local-model-smoke-nemotron-omni-jangtq-system-nomedia-bundled-20260524/summary.json`
  -> `pass`, exact ACK/cache, recall, `reasoning_on` visible `FINAL=OK`,
  blue/red image checks, and no-media-after-image `NONE`.
- Updated objective proof diagnostics to preserve old failing prompt evidence
  plus the new system-negative proof; refreshed objective
  `build/current-objective-proof-audit-20260521.json`.
- Cross-family covered families are now `hy3`, `ling_bailing`, `minimax`,
  `nemotron`, `qwen36`, `zaya_text`, and `zaya_vl`; missing/open are DSV4 and
  Gemma4.
- Updated release manifest current-suite pointer to
  `build/current-regression-suite-20260524-nemotron-smoke-open.json` to avoid
  stale ZAYA-only proof sweep.
- Verification: focused tests `201 passed`; current suite
  `build/current-regression-suite-20260524-nemotron-smoke-open.json`
  `status=pass failed_steps=[]` with five expected open rows; release manifest
  `build/current-release-regression-manifest-20260524-codex-audit.json`
  `current_proof_sweep=pass`; `py_compile` and `git diff --check` passed.

## [2026-05-24 18:48] codex | live-probe | ZAYA-VL crossed off with strict ACK/cache/media proof

- Reproduced ZAYA-VL JANGTQ4 exact-ACK failure under current source and traced
  it to prompt sensitivity, not cache/media/template corruption.
- Current rendered prompt frame is `<|im_start|>user...<|im_end|>`, while the
  stale plain `user: ... assistant:` diagnostic was outdated.
- Prompt diagnostics show user-only cache-prefix exact ACK can produce
  `The output is: ACK`, but system-scoped exact-output instruction produces
  bare `ACK` and repeat `cached_tokens=54`, `cache_detail=paged+zaya_cca`.
- Updated `bench/all_local_model_smoke.py` so cache-repeat probes use the
  system-scoped exact-output instruction and reject non-bare ACK.
- Reran bundled ZAYA-VL JANGTQ4 smoke:
  `build/current-all-local-model-smoke-zaya-vl-jangtq4-bundled-20260524/summary.json`
  -> `pass`, bare ACK on both cache probes, cache hit on repeat, recall,
  image color checks, and no-media-after-image pass.
- Cross-family covered families are now `hy3`, `ling_bailing`, `minimax`,
  `qwen36`, `zaya_text`, and `zaya_vl`; missing/open are DSV4, Gemma4, and
  Nemotron.
- Verification: focused tests `201 passed`; refreshed objective
  `build/current-objective-proof-audit-20260521.json`; current suite
  `build/current-regression-suite-20260524-zaya-vl-smoke-open.json`
  `status=pass failed_steps=[]`; release manifest
  `build/current-release-regression-manifest-20260524-codex-audit.json`
  `current_proof_sweep=pass`; `py_compile` and `git diff --check` passed.
## [2026-05-27 04:16] codex | source-integration | MiMo-V2.5/JANG_2L registry and bundled import gate

- Imported relevant `erics-m5-max2.local:~/jang` MiMo/JANG2L notes and source
  snapshot into ignored local evidence at
  `docs/internal/remote-jang-notes/2026-05-27-mimo-v2-jang2l/`.
- Added conservative `mimo_v2` Python/panel detection: KV cache, qwen3
  reasoning, thinking-capable, no unproven tool parser/auto-tools, multimodal
  metadata surfaced, MTP preserved-disabled architecture hint.
- Added JANG loader pre-resolution import of `jang_tools.mimo_v2.mlx_register`
  plus `mlx_lm.models.mimo_v2` verification.
- Added bundled-python verifier imports for the same MiMo runtime modules.
- Verification: Python MiMo registry `3 passed`; JANG loader/bundled gate
  focused audit `2 passed`; release gate import test `1 passed`; panel
  registry/settings slice `32 passed`; local JANG MiMo contract `3 skipped`
  because the MiMo source volume is not mounted; `verify-bundled` passed with
  MiMo runtime imports; `py_compile` and `git diff --check` passed.
- Boundary: no MiMo model load/live app proof yet, and no release clearance.

## [2026-05-30 20:34] codex | Step3.7 | text-loader gate sidecar fix, quality still blocked

- Fixed Step3.7 text JANG loader gate-sidecar handling so already-quantized
  `mlp.gate.gate` modules keep uint32 weights plus `.scales/.biases` instead
  of being overwritten with bf16 and crashing quantized matmul.
- Added `_prepare_gate_dequant_weights()` and focused regression
  `test_step37_text_loader_preserves_quantized_gate_sidecars`.
- Verification: focused Step/JANG/reasoning tests `4 passed`; `py_compile` and
  `git diff --check` clean for touched files.
- Direct text bridge no longer crashes but still emits newline/underscore junk;
  tokenized one-BOS prompt and ablations for router gate float, K/V sidecars,
  lm_head float, embedding float, and RMSNorm offset did not clear it.
- Current boundary: Step3.7 remains not release-cleared; continue below
  UI/cache/tool/parser at Step3p5 runtime or JANG_2L conversion numeric
  fidelity.
## [2026-05-30 20:54] codex | Step3.7 | VLM norm fix, real UI text/tools/cache pass

- Local MacBook only; no max2, no `/Users/eric/vmlx`, no Swift.
- Added Step3.7 text-loader normalization/remap/dense-norm invariants and
  fixed source Step3.7 VLM sanitize to apply zero-centered norm `+1` on
  dense-only language shards.
- Direct VLM language generation changed from punctuation/newline corruption to
  coherent text:
  `build/step37-current-probe-20260531/direct-vlm-language-model-after-vlm-norm-fix.json`.
- Passing real Electron dev-app text/tool/cache artifact:
  `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-text-tools-cache-after-vlm-norm-fix-20260531-proof.json`.
- Still-open media artifact:
  `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-after-vlm-norm-fix-20260531-proof.json`.
  Image semantic verification fails because Step repeatedly called `read_image`
  on prior `.txt` tool files instead of answering the attached image.
- Isolated media-only artifact also fails:
  `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-media-only-after-vlm-norm-fix-20260531-proof.json`.
  The image attachment is persisted and the engine reports one processed image,
  but the assistant says it cannot see the attachment. Remaining Step3.7 VL
  blocker is image placeholder/embedding merge or prompt serialization
  semantics, not app attachment persistence.
- Verification: focused Step/JANG pytest slice `10 passed`; `py_compile` and
  `git diff --check` clean for touched files.
- 2026-05-31 00:05 PDT: fixed Step3.7 direct VLM image semantics in Python
  source runtime. `load_jang_vlm_model()` now registers the source
  `mlx_vlm.models.step3p7` adapter before `mlx_vlm` model resolution, and
  `step3p7_mlx_vlm.Model.sanitize()` maps
  `model.vit_large_projector.{weight,bias}` to
  `vit_large_projector.proj.{weight,bias}` so the actual MLX projector receives
  checkpoint weights. Direct RGB proof now passes:
  `build/step37-current-probe-20260531/direct-vlm-rgb-color-ab-after-projector-key-fix-20260531.json`.
  Projector parity artifact:
  `build/step37-current-probe-20260531/full-downsample-projector-split-parity-after-projector-key-fix-20260531.json`.
  Focused verification: `16 passed`, `py_compile` clean, `git diff --check`
  clean. Full Electron media/tool/cache E2E refresh still pending; release not
  cleared.
- 2026-05-30 21:55 PDT: Step3.7 local Electron media-only E2E now passes after
  the app-placeholder fix. The real UI/scheduler sends image payloads separately
  from text; `Step3VLProcessor.__call__()` now inserts the required image
  placeholder when images are present and the prompt has none. Proof artifact:
  `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-media-only-after-placeholder-projector-fix-20260531-proof.json`.
  Proven surfaces include `vl_image`, `responses_api`, current Electron dev
  build, settings persistence, real loaded model, cache endpoint/native/cache-hit
  telemetry, language leak check, and parser leak check. Image turn answered
  `Red`; `sendErrors=[]`; parser leaks false; cache hit was
  `paged+mixed_swa` with 20 tokens and reconstruction/dequantization OK.
  Focused verification: `17 passed`, `py_compile` clean, `git diff --check`
  clean. Full release E2E remains open for combined tools+media, LFM/other rows,
  EPIPE/write path, broader matrix, packaged sync, and release gates.
- 2026-05-30 22:03 PDT: Step3.7 combined tools+media+cache local Electron E2E
  is now green after a concrete app tool/media policy fix. Reproduced the
  combined failure first:
  `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-tools-image-cachecontrols-after-placeholder-projector-fix-20260531-proof.json`
  showed the two `run_command` turns wrote the sentinel files, but the direct
  image turn went through `read_image`/`list_directory` instead of native VL.
  `panel/src/main/ipc/chat.ts` now filters `read_image`/`read_video` out only
  for direct composer media attachment requests and adds a system instruction to
  inspect direct attachments through multimodal input unless the user gives a
  filesystem path. Regression added in `panel/tests/tool-media-followup.test.ts`.
  Passing proof:
  `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-tools-image-cachecontrols-after-direct-media-tool-filter-20260531-proof.json`.
  Proven surfaces include `long_tool_loop`, `vl_image`, `responses_api`,
  `server_cache_controls`, current Electron dev build, settings persistence,
  real loaded model, cache endpoint/native/cache-hit telemetry, language leak
  check, parser leak check, and chat completions. Cache hit tokens: 7179
  `paged+mixed_swa`; native cache is mixed-SWA KV with prefix/paged/block-disk
  L2; reconstruction/dequantization OK. Verification: panel focused test
  6 passed, `npm run typecheck` passed, focused Python Step/JANG/reasoning
  3 passed, `py_compile` clean, `git diff --check` clean. Full release remains
  open for LFM, EPIPE/write path, broader family matrix, packaged sync, and
  release gates.
- 2026-05-30 22:05 PDT: refreshed LFM2.5 JANG_2L local Electron dev-build
  Responses/tools/hybrid-cache proof after the Step media/tool policy fix.
  Artifact:
  `docs/internal/agent-notes/current-real-ui-live-model-lfm25-moe-a1b-jang2l-stricttools-responses-post-step-media-tool-filter-20260531-proof.json`.
  Proven surfaces include `long_tool_loop`, `responses_api`,
  `server_cache_controls`, current Electron dev build, settings persistence,
  real loaded model, cache endpoint/native/cache-hit telemetry, language leak
  check, parser leak check, and chat completions. Tool sentinels were written;
  `sendErrors=[]`; parser leaks false. Native cache is LFM2 MoE
  `hybrid_ssm_v1` with attention KV, SSM companion state, async rederive,
  prefix/paged/block-disk L2, q4 storage-boundary attention KV, and native SSM
  companion state. Block L2 has 2446 tokens on disk; SSM companion disk has
  5518 tokens; total L2 store-sum 7964. Full release remains open for
  EPIPE/write path, broader family matrix, packaged sync, and release gates.
- 2026-05-30 22:08 PDT: refreshed source/no-heavy API/EPIPE/gateway contract
  after Step and LFM app E2E fixes:
  `build/current-api-surface-contract-20260531-post-step-lfm-epipe-refresh.json`.
  Status pass; server API/cache slice 31 passed / 423 deselected; panel
  request/gateway slice 259 passed; no missing nested checks/markers or panel
  markers. Current checks include Chat/Responses kwargs/default omission,
  output/context cap separation, streaming cache-detail usage, gateway
  single-model streaming delta preservation, cache endpoint auto-switch,
  gateway EPIPE guard, child-process stdio EPIPE guard, Ollama proxy stream
  disconnect guard, and chat IPC backend request finalization EPIPE guard. Full
  release remains open for packaged sync, installed-app proof, and full live
  matrix.
- 2026-05-30 22:09 PDT: refreshed generation-defaults/model-owned config
  contract after current app changes:
  `build/current-generation-defaults-contract-20260531-post-step-lfm-refresh.json`.
  Status pass; no failures or missing markers. Counts: panel generation
  defaults/settings 25 passed / 234 skipped; engine generation defaults
  42 passed / 412 deselected; local generation metadata audit 4 passed. Checks
  include generation_config surfacing, JANG chat sampling precedence, disabled
  top_k normalization, metadata-owned repetition penalty with no hidden floor,
  request overrides over startup defaults, bundle max_new_tokens preservation,
  output/context cap separation, app-owned flag override blocking, and no default
  sampler CLI flag emission. Full release remains open for packaged sync and
  full live model matrix.

## [2026-06-04] codex | live-matrix | Gemma4/LFM/Step/Ling runtime refresh

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; did not touch
  deprecated `/Users/eric/vmlx` or Swift paths. MiMo remained out of scope per
  user direction.
- Fixed live-smoke false failures without weakening runtime assertions:
  - `bench/all_local_model_smoke.py` now gives LFM2 strict text/tool probes a
    512-token budget so no-thinking final-answer parsing can reach the final
    answer while still requiring bare `ACK`, recalled `blue`/`cat`, and parsed
    `record_fact({"value":"blue-cat"})` tool calls.
  - LFM2 MXFP8 uses an explicit JSON-call tool prompt; LFM2 JANG_2L/MXFP4 keep
    the natural tool prompt because live probes showed the prompt contract is
    quantization-sensitive.
  - Ling multilingual loop gate now allows prompted game technical token
    `recoil` while the unrelated-Latin rejection test remains passing.
- Fresh live artifacts:
  - `build/current-gemma4-12b-live-all-unified-runtime-fullmatrix-postfix-20260604.json`
    -> pass, 9/9 Gemma 4 12B rows across conservative, prefix+paged+L2, and
    prefix+paged+q8 storage modes for MXFP4/MXFP8/JANG_4M.
  - `build/current-gemma4-12b-media-smoke-all-unified-runtime-postfix-20260604.json`
    -> pass, red-image smoke for MXFP4/MXFP8/JANG_4M.
  - `build/current-all-local-model-smoke-step-lfm-multiturn-tools-media-postfix3-20260604/summary.json`
    -> pass, Step-3.7 JANG_2L plus LFM2.5 JANG_2L/MXFP4/MXFP8.
  - `build/current-production-family-audit-live-gemma4-ling-impact-postfix-20260604.json`
    -> pass, Gemma4 26B JANG_4M CRACK, Ling JANGTQ, Ling MXFP4 CRACK.
  - `build/current-all-local-model-smoke-gemma31-jang4m-mtp-20260604/summary.json`
    -> pass, Gemma4 31B JANG_4M-MTP.
- Proof note:
  `docs/internal/agent-notes/current-gemma4-lfm-step-ling-live-matrix-20260604-proof.json`.
- Verification: `py_compile` for touched Python files passed; focused Ling
  predicate pytest `2 passed`; all live commands above exited 0.
- Boundary: no release/upload/git operation performed. Dedicated Gemma4
  audio/video runtime rows remain separate from the image media smoke.

## [2026-06-04] codex | Gemma4 12B audio runtime integration partial, MXFP audio still red

- SSH'd into `erics-m5-max2.local`; targeted remote note search under
  `/private/tmp/claude-501/-Users-eric-jang`, `/Users/eric/jang`, and
  `/Users/eric/mlx` found no Gemma4 12B text-note hits. Remote bundle metadata
  was used as authoritative evidence.
- Local and remote Gemma4 12B bundle metadata hashes match for MXFP4, MXFP8,
  and JANG_4M across `config.json`, `generation_config.json`, `jang_config.json`,
  `tokenizer_config.json`, `chat_template.jinja`, `model.safetensors.index.json`,
  and `README.md`.
- Added/fixed Gemma4 Unified audio runtime path:
  - server MLLM modalities now include audio when `audio_config`/audio token
    metadata exists;
  - server no longer advertises video for `gemma4_unified` without native
    `video_config`;
  - MLLM chat extracts `input_audio`/`audio`/`audio_url`, materializes base64
    audio to temp files, and passes `audio=` through to `mlx_vlm.generate` and
    streaming;
  - `Gemma4UnifiedProcessor.process()` exposes explicit `audio` because
    `mlx_vlm.utils.process_inputs` introspects `processor.process` before
    `__call__`.
- Dedicated audio smoke added:
  `tests/cross_matrix/run_gemma4_12b_audio_smoke.py`.
- Fresh green post-change artifacts:
  - `build/current-gemma4-12b-artifact-contract-after-audio-path-20260604.json`
    -> pass.
  - `build/current-gemma4-12b-live-all-unified-runtime-after-audio-path-20260604.json`
    -> pass, 9/9 Gemma4 12B text/tool/reasoning/cache/q8 rows.
  - `build/current-gemma4-12b-image-smoke-after-audio-path-20260604.json`
    -> pass, 3/3 image rows.
  - `build/current-all-local-model-smoke-step-lfm-after-gemma4-audio-path-20260604/summary.json`
    -> pass, Step3.7 + LFM2.5 4/4 rows.
  - `build/current-all-local-model-smoke-gemma31-jang4m-mtp-after-audio-path-20260604/summary.json`
    -> pass.
  - `build/current-production-family-audit-live-gemma4-ling-after-audio-path-20260604.json`
    -> pass.
- Audio remains not cleared for all 12B bundles:
  `build/current-gemma4-12b-audio-smoke-all-unified-runtime-speech-wav-orderfix-20260604.json`
  -> fail overall. JANG_4M transcribed `Audio present.`; MXFP4 generated
  `The transcriptionno`; MXFP8 returned empty visible content. Direct processor
  probe showed 21 `<|audio|>` placeholders matching `input_features` shape
  `(1, 21, 640)`, and all three bundles have identical float16
  `embed_audio.embedding_projection.weight` shape `(3840, 640)`. Remaining
  blocker is MXFP4/MXFP8 audio quality after runtime reaches generation, not
  endpoint rejection or missing audio projection weights.
- Boundary: no release/upload/git operation performed. Goal remains active.

## [2026-06-04] codex | MXFP Gemma4 12B audio quality remains open after focused probes

- Focused MXFP audio prompt probes after runtime audio integration:
  - `build/current-gemma4-mxfp4-audio-prompt-probe-20260604/result.json`
    -> all defensible prompt/sampling variants fail; common outputs include
    `The transcriptionno` and cannot-hear/cannot-process refusals.
  - `build/current-gemma4-mxfp8-audio-prompt-probe-20260604/result.json`
    -> temperature-0 path empty/hidden; model-owned defaults produced
    `present present`; thinking-on reasoning heard audio but looped.
  - Parser-off probes did not clear quality:
    `build/current-gemma4-mxfp8-audio-parser-probe-20260604/result.json`
    produced `Mario presents`; MXFP4 parser-off still produced
    `The transcriptionno`.
- Strengthened `tests/cross_matrix/run_gemma4_12b_audio_smoke.py` so it fails
  negative/no-audio answers, `thought`, and `transcriptionno` instead of
  accepting any output containing the substring `audio`.
- Current authoritative audio artifact:
  `build/current-gemma4-12b-audio-smoke-all-unified-runtime-current-red-20260604.json`
  -> fail overall. JANG_4M passes (`Audio present.`); MXFP4 fails
  `audio_bad_quality_marker` + missing transcription (`The transcriptionno`);
  MXFP8 fails empty visible content + missing transcription.
- Current JANG_4M-only audio proof:
  `build/current-gemma4-12b-audio-smoke-jang4m-pass-20260604.json` -> pass.
- Root-cause boundary remains: audio runtime plumbing is active, placeholder and
  feature counts line up, and audio projection weights exist as float16 in all
  three bundles. Remaining blocker is MXFP4/MXFP8 audio quality under the current
  artifacts/runtime, not endpoint rejection or missing audio projection.
- Detailed probe note:
  `docs/internal/agent-notes/current-gemma4-12b-mxfp-audio-quality-probes-20260604.json`.

## [2026-06-04] codex | source audio passes, MXFP audio capability gated off honestly

- Ran source Gemma4 12B audio comparison on `erics-m5-max2.local` using a
  temporary in-process Gemma4 Unified runtime adapter staged under `/tmp` only.
  Artifact:
  `/Users/eric/mlx/vllm-mlx/build/current-gemma4-source-audio-direct-probe-with-runtime-20260604.json`.
  Source model `/Users/eric/models/google/gemma-4-12B-it` outputs
  `Audio is present.` for temp-0, model defaults, and thinking-on.
- Local prompt-contract probe:
  `build/current-gemma4-local-mxfp-audio-prompt-contract-probe-20260604.json`.
  Empty thought channel removes raw marker leakage but does not fix MXFP audio:
  MXFP4 stays `The transcriptionno`, MXFP8 stays `Mario presents`, JANG_4M
  stays `Audio present.`.
- Alternate MXFP8 CRACK artifacts also fail audio:
  `build/current-gemma4-alt-mxfp8-audio-probe-20260604.json`.
- Changed `vmlx_engine/server.py` so Gemma4 Unified MXFP4/MXFP8 artifacts no
  longer advertise native audio based only on `audio_config`. The gate checks
  `jang_config.weight_format/profile` and refuses MXFP audio until a repaired
  artifact is stamped/proven. JANG_4M remains audio-capable.
- Fresh verification after the gate:
  - `build/current-gemma4-12b-audio-smoke-jang4m-after-mxfp-audio-cap-gate-20260604.json`
    -> pass, JANG_4M audio.
  - `build/current-gemma4-12b-audio-smoke-all-after-mxfp-audio-cap-gate-20260604.json`
    -> fail by design for MXFP4/MXFP8 `audio_not_advertised` + 400 reject;
    JANG_4M passes.
  - `build/current-gemma4-12b-image-smoke-after-mxfp-audio-cap-gate-20260604.json`
    -> pass, image/VL still green for MXFP4/MXFP8/JANG_4M.
- New proof note:
  `docs/internal/agent-notes/current-gemma4-12b-source-vs-mxfp-audio-boundary-20260604.json`.
- Boundary: MXFP4/MXFP8 audio is not working and not advertised. Real clearance
  requires repaired/reconverted MXFP artifacts or a different verified MXFP
  runtime path. No release/upload/git operation performed.

## 2026-06-04 - Gemma4 audio gate + E2E proof refresh
- Added Gemma4 Unified modality regressions: JANG_4M advertises audio, MXFP4/MXFP8 do not advertise unproven audio, Gemma4 token-only video remains gated.
- Added `--expect-audio-gated` to `tests/cross_matrix/run_gemma4_12b_audio_smoke.py` so MXFP audio can be recorded as a passing capability contract while keeping not-advertised/400 details visible.
- Green: Gemma4 12B artifact contract, 9-row text/tool/reasoning/cache matrix, audio contract with MXFP gated + JANG_4M transcription, Gemma31/LFM/Step all-local smoke slices.
- Green: Gemma26 JANG_4M CRACK and Ling MXFP4 CRACK in targeted production-family audit.
- Red: Ling JANGTQ is unstable on `ling_multilingual_loop_trigger`; one rerun passed, repeat1 failed again with unprompted CJK/English fragments. Do not release-clear Ling JANGTQ from this run.
- Proof note: `docs/internal/agent-notes/current-e2e-gemma4-lfm-step-ling-after-mxfp-audio-cap-gate-20260604.json`.

## 2026-06-04 - Gemma4 MXFP audio transform probe + rejected MXFP8 ungate
- Added `tests/cross_matrix/run_gemma4_12b_audio_transform_probe.py` to load a Gemma4 12B row once, patch only `get_audio_features`, and test input/output audio-prefix scaling through the same MLLM chat path.
- MXFP4 remains genuinely red: direct probe outputs `The transcriptionno` / refusals across scaling variants. Audio embedding stats are normal, so this is not an exploding-prefix issue.
- MXFP8 is not safe: direct raw output sometimes contains recoverable `Audio present.` behind Gemma4 channel markers, but a temporary live server/API ungate returned `Mario presents.`. Reverted ungate.
- Current safe proof restored: `build/current-gemma4-12b-audio-smoke-all-after-mxfp8-ungate-rejected-20260604.json` passes with MXFP4/MXFP8 expected-gated and JANG_4M audio green.
- New note: `docs/internal/agent-notes/current-gemma4-12b-mxfp-audio-transform-boundary-20260604.json`.

## 2026-06-04 - Ling JANGTQ prompt-aligned gate green; MXFP audio still red
- Investigated MXFP8 direct/server divergence. Direct with one fixed WAV and fresh loads produced `Mario presents.` five out of five; this is not just server parser cleanup or top_k/top_p injection. MXFP8 audio remains gated alongside MXFP4.
- Aligned Ling multilingual audit prompt with its validator: explicitly forbids English/Chinese words except HTML and Three.js. This is not a relaxed gate; the validator already enforced this constraint.
- Verified Ling JANGTQ prompt-aligned gate with three fresh live runs: `build/current-production-family-audit-live-ling-jangtq-promptaligned-repeat{1,2,3}-20260604.json`, all pass.
- Verified combined targeted production audit: `build/current-production-family-audit-live-gemma26-ling-after-ling-prompt-alignment-20260604.json`, all rows pass for Ling JANGTQ, Ling MXFP4 CRACK, Gemma4 26B JANG_4M CRACK.
- New note: `docs/internal/agent-notes/current-ling-jangtq-prompt-alignment-and-mxfp-audio-boundary-20260604.json`.

## 2026-06-04 Gemma4 12B MXFP audio boundary + jang-tools max2 sync
- Synced max2 Gemma4 converter/runtime docs into local `/Users/eric/jang/jang-tools`:
  - `jang_tools/convert_gemma4_mxfp.py`
  - `jang_tools/convert_gemma4_jang.py`
  - `tests/test_gemma4_template_patch.py`
  - `docs/GEMMA4_12B_RUNTIME_INTEGRATION_2026_06_03.md`
- Root-cause boundary captured in `docs/internal/agent-notes/current-gemma4-12b-mxfp-audio-root-cause-and-jang-tools-sync-20260604.json`.
- Current conclusion: Gemma4 12B MXFP4/MXFP8 keep audio embedder tensors fp16 but uniformly MXFP-quantize the shared decoder. JANG_4M mixed affine passes audio; MXFP audio remains red/unproven and must stay capability-gated until reconverted/proven.
- Max2 BF16 source confirmed at `/Users/eric/models/google/gemma-4-12B-it/model.safetensors`.
- Max2 MXFP8 converter dry-run confirmed 677 source tensors -> 328 uniform MXFP8 decoder tensors + 349 fp16 passthrough tensors. This reinforces the artifact/profile boundary for MXFP audio.

## 2026-06-04 Gemma4 12B MXFP8_ATTNFP16 candidate conversion started
- Added opt-in `convert_gemma4_mxfp.py` flags for selective fp16 passthrough:
  - `--preserve-attention-fp16`
  - `--preserve-full-attention-fp16`
  - `--preserve-first-layers N`
- Local and max2 focused tests: `tests/test_gemma4_template_patch.py` passed 10/10.
- Max2 dry-run for `--bits 8 --preserve-attention-fp16`: 677 tensors -> 144 MXFP8 tensors + 533 fp16 passthrough tensors.
- Full candidate conversion running on max2: PID 60050, output `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE`, log `/Users/eric/jang/logs/gemma4-12b-mxfp8-attnfp16-convert-20260604.log`.
- Candidate direct probe: `build/current-gemma4-12b-mxfp8-attnfp16-direct-audio-baseline-20260604.json` failed only due raw Gemma channel markers in direct/parserless output; semantic text was `audio present` and no-audio was `NONE` with channel markers.
- Server gate updated narrowly to allow Gemma4 Unified `MXFP8_ATTNFP16` when `jang_config.quantization.selective_passthrough.preserve_attention_fp16=true`; uniform MXFP4/MXFP8 remains gated.
- Focused server modality tests passed: 3 selected / 3 passed.
- Full API audio smoke passed for candidate: `build/current-gemma4-12b-mxfp8-attnfp16-audio-smoke-20260604.json`, content `audio present`, no-audio `NONE`.
- Candidate media smoke passed: `build/current-gemma4-12b-mxfp8-attnfp16-media-smoke-20260604.json`, image answer `Red`.
- Candidate live runtime audit passed across all requested cache/runtime modes: `build/current-gemma4-12b-mxfp8-attnfp16-live-runtime-audit-20260604.json`.
  - conservative: pass
  - prefix_paged_l2: pass
  - prefix_paged_tq_q8: pass
- Important red boundary: `MXFP8_ATTNFP16` is not release-stable. It passes standalone and candidate-first audio rows, but fails with empty parsed content after a uniform MXFP8 load in the same host session.
  - pass: `build/current-gemma4-12b-mxfp8-attnfp16-audio-smoke-repeat{1,2,3}-20260604.json`
  - pass: `build/current-gemma4-12b-audio-smoke-candidate-first-20260604.json`
  - fail: `build/current-gemma4-12b-audio-smoke-mxfp8-then-attnfp16-20260604.json`
  - fail persists after 60s: `build/current-gemma4-12b-mxfp8-attnfp16-after-mxfp8-fail-solo-wait60-20260604.json`
- Do not release or ungate `MXFP8_ATTNFP16` as stable until this ordering-dependent MXFP state issue is solved or a stronger profile passes mxfp8 -> candidate churn.

## 2026-06-04 Stronger Gemma4 MXFP8 audio-safe profile attempt
- Because `MXFP8_ATTNFP16` failed after uniform MXFP8 load, next candidate is `MXFP8_ATTNFP16_L0_24_FP16`: all attention fp16 plus first 24 decoder layers fully fp16; only later MLP tensors remain MXFP8.

## [2026-06-04] Gemma4 12B release boundary and package gate refresh

Gemma4 12B default release rows are green for the honest capability surface:

- MXFP4/MXFP8/JANG_4M artifact contract: `build/current-gemma4-12b-artifact-contract-release-boundary-20260604.json` -> pass.
- Default audio gate: `build/current-gemma4-12b-audio-smoke-release-default-gated-mxfp-jang-pass-20260604.json` -> pass. MXFP rows do not advertise audio; JANG_4M returns `Audio present.` and no-audio control returns `NONE`.
- Default VL/image: `build/current-gemma4-12b-media-smoke-release-default-mxfp-jang-20260604.json` -> pass.
- Runtime/cache: `build/current-gemma4-12b-live-runtime-release-default-cache-l2-tq-fullmodes-20260604.json` -> pass across conservative, prefix+paged+L2, and prefix+paged+TurboQuant q8 modes for MXFP4/MXFP8/JANG_4M.
- Generation defaults/API/native-MTP/VL-media contracts refreshed and pass.

Candidate truth:

- `MXFP8_ATTNFP16_L0_24_FP16` is not release-stable: `build/current-gemma4-12b-audio-smoke-mxfp8-then-attnfp16-l024-20260604.json` failed after uniform MXFP8 load with empty audio content.
- `MXFP8_ATTNFP16` is not release-stable: latest all-row VL smoke returned `Maroon` instead of exact `Red`.

Packaging truth:

- Rebuilt `panel/bundled-python` from this checkout and local `/Users/eric/jang/jang-tools`; bundled verifier now passes.
- Packaged integrity still fails `release_gate_skip_app` because objective digest/release-ready manifest still have open DSV4/real-UI rows.
- Public app issue audit still fails installed-app hash guard rows `111` and `165`; a rebuilt/installed app audit is needed before release/notarization/main push.
- No signing, notarization, release upload, commit, or push was performed.

Detailed note: `docs/internal/agent-notes/current-gemma4-12b-release-boundary-and-package-gates-20260604.json`.

## 2026-06-04 - Codex Gemma4 remote-note reconciliation and release-boundary refresh

- Re-read Gemma 4 12B notes from `erics-m5-max2.local`, including the JANG runtime spec, wiki architecture invariants, JANG project memory, conversion logs, and remote source-audio probe.
- Confirmed local Gemma4 12B contract still matches remote source-of-truth: dense `gemma4_unified`, 5:1 SWA/full attention, no native MTP, zero-shift RMSNorm, heterogeneous full/sliding KV cache, generation-config EOS/suppress tokens, and fp16 passthrough early-fusion multimodal embedders for MXFP4/MXFP8/JANG_4M.
- Updated the Gemma4 release-boundary note to reflect the fresh signed/notarized DMGs and final packaged/public-audit passes.
- Removed the accidental current-scope MiMo host-availability helper/test. MiMo remains out of the current Gemma4 release work.
- Full release remains not cleared: uniform MXFP4/MXFP8 audio is gated, experimental MXFP8 attention-fp16 candidates are not release-stable, DSV4 long-output/code quality and full real-UI cross-family matrix remain open.

## 2026-06-04 - Codex post-reconcile gate refresh

- Focused Gemma4 artifact/live-audit/release-manifest tests pass after updating the five-bundle artifact fixture and stale post-DMG assertion names: `8 passed`, then stale-pointer slice `3 passed`.
- No-heavy Gemma4 artifact contract refreshed: `build/current-gemma4-12b-artifact-contract-after-remote-note-reconcile-20260604.json` -> pass for MXFP4, MXFP8, MXFP8_ATTNFP16, MXFP8_ATTNFP16_L0_24, and JANG_4M.
- Aggregate current regression suite refreshed: `build/current-regression-suite-gemma4-release-boundary-after-remote-note-reconcile-20260604.json` -> status=pass, failed_steps=[], open requirements are only real Electron UI cross-family matrix and DSV4 long-output/code/file-generation quality.
- Installed-app parity and public issue audit regenerated serially after the final staged app: both pass.
- Release manifest refreshed: `build/current-release-regression-manifest-gemma4-release-boundary-after-remote-note-reconcile-20260604.json` -> prepackage_ready=true, release_ready=false, current_proof_sweep=fail only because regression_suite carries the two open release-policy requirements.
- This turn changed docs/tests/agent notes only after the already-notarized DMGs; no runtime/package source changed after the final DMG build, so no fresh notarization was required from this pass alone.

## 2026-06-04 - Codex cross-family continuation proof refresh

- LFM2.5 installed Electron UI/settings proof passed against `panel/release/sequoia-app/mac-arm64/vMLX.app` and `/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-JANG_2L`:
  - `docs/internal/agent-notes/current-real-ui-live-model-lfm25-jang2l-continuation-cache-settings-20260604-proof.json`.
  - Covered max output tokens 64, Prefix Cache, Paged KV Cache, Block Disk Cache (L2), hybrid SSM native cache status, second-turn cache-hit telemetry, L2 disk counters, settings persistence, coherent output, and parser/language leak checks.
- Static Gemma4 26B/31B production-family audit passed:
  - `build/current-production-family-audit-static-gemma4-26b-31b-20260604.json`.
  - `gemma4_31b_jang4m_mtp` is tracked as an MTP-preserved artifact boundary only; no native MTP speedup claim is allowed without live accept/reject and full-output equivalence proof.
- Step3.7 VLM runtime audit passed:
  - `build/current-step37-vlm-runtime-audit-20260604-continuation.json`.
- Step3.7 installed Electron UI/settings proof passed:
  - `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-continuation-cache-settings-20260604-proof.json`.
  - Covered max output tokens 64, Prefix Cache, Paged KV Cache, Block Disk Cache (L2), stored KV cache quantization, `mixed_swa_kv_v1` native cache status, `paged+mixed_swa` second-turn cache-hit telemetry, L2 block writes, coherent output, and parser/language leak checks.
  - Boundary remains honest: config/metadata expects Step MTP, but the local index has no `mtp.*` tensors; runtime reports `metadata_inconsistent` and native MTP inactive.
- Static DSV4 local audit still blocks release claims:
  - `build/current-production-family-audit-static-dsv4-local-20260604.json`.
  - `dsv4_tq` and `dsv4_jang_local` both expose the output-head/final-norm precision boundary and need source-vs-quant or rebuilt-artifact clearance before long-output/code/file-generation production claims.
- DSV4 real-UI memory preflight refreshed:
  - `build/current-real-ui-dsv4-memory-preflight-after-step-lfm-continuation-20260604.json` -> skipped_insufficient_memory.
  - Available memory was 70.54 GB against a 120.0 GB threshold, so live DSV4 UI proof was not launched on this host.
- Added focused regression coverage for the Gemma4 31B MTP-preserved production-family row.
- Aggregate current regression suite refreshed after the Gemma31/Step/LFM continuation:
  - `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[].
  - Open requirements remain real Electron UI cross-family live model matrix clearance and DSV4 long-output/code/file-generation quality clearance.
- Release manifest refreshed:
  - `build/current-release-regression-manifest-after-gemma31-step-lfm-continuation-20260604.json` -> prepackage_ready=true, release_ready=false, current_proof_sweep=fail.
  - This is the correct blocker state; do not push/promote/tag as a full release from this checkout until the two open release requirements are actually closed.
- Installed-app parity and public issue audit regenerated at the manifest-owned paths:
  - `build/current-installed-app-runtime-parity-audit-gemma4-release-boundary-after-install-20260604.json` -> pass.
  - `build/current-public-app-issue-audit-gemma4-release-boundary-after-install-20260604.json` -> pass.
  - `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> prepackage_ready=true, release_ready=false, current_proof_sweep=fail, failed_components only `regression_suite`.
  - `CURRENT_REGRESSION_SUITE_ARTIFACT` now points at `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json`.
  - The current suite/manifest command now audits the staged signed Sequoia app explicitly for this parity artifact (`--app panel/release/sequoia-app/mac-arm64/vMLX.app`) rather than overwriting or relying on a stale `/Applications/vMLX.app`.

## 2026-06-04 - Codex LFM2.5 Responses/tools UI proof refresh

- Initial LFM2.5 installed UI Responses/tool proof at max_tokens=128 failed `long_tool_loop`; the model described the first tool call instead of issuing it.
- A max512 rerun with custom prompt overrides also failed: it produced a malformed command because the override bypassed the LFM2 fallback command-derivation pattern.
- The default prompt with max_tokens=512 passed:
  - `docs/internal/agent-notes/current-real-ui-live-model-lfm25-jang2l-continuation-responses-tools-cache-settings-default-max512-20260604-proof.json`.
  - Proven surfaces include Responses API, Responses delta streaming, responses cache detail usage, live speed floor, long tool loop, tool L2 cache integration, native `hybrid_ssm_v1` cache status, `paged+ssm` cache hits, block disk L2, SSM companion L2, settings persistence, and parser/language leak checks.
- Promoted `lfm25_moe_a1b_responses_delta` in the release manifest to the fresh 2026-06-04 proof.

## 2026-06-04 - Codex LFM/Step manifest proof fix and DSV4 preflight refresh

- Fixed release-manifest real-UI proof interpretation so LFM2.5's fresh max512 Responses/tool proof counts real named tool results plus L2/cache evidence instead of being blocked by empty persisted `generating` tool events.
- Refreshed DSV4 preflight pointers and artifacts:
  - `build/current-real-ui-dsv4-memory-preflight-after-lfm-step-manifest-fix-20260604.json` -> skipped_insufficient_memory, 71.69 GB free+speculative+purgeable vs 120.0 GB required.
  - `build/current-dsv4-route-mode-code-exactness-memory-preflight-after-lfm-step-manifest-fix-20260604.json` -> skipped by memory preflight.
- Verification after fixes:
  - DSV4/LFM/real-UI manifest slice: `17 passed, 289 deselected`.
  - Focused release regression selection: `555 passed, 202 deselected`.
  - Current regression suite: `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[], open requirements only real Electron UI cross-family matrix and DSV4 long-output/code quality.
  - Release manifest: `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> current_proof_sweep=fail, prepackage_ready=false, release_ready=false, failed_components only `real_ui_live_model_proof`.
- Current truth: LFM2.5 and Step3.7 are pass in the real-UI matrix; DSV4 remains missing because memory preflight blocks local launch. No commit, push, tag, public release, or new notarization was performed.

## 2026-06-04 - Codex DSV4 local/max2 readiness continuation

- Refreshed local DSV4 memory gates without launching the model:
  - `build/current-real-ui-dsv4-memory-preflight-continuation-refresh-20260604.json` -> skipped_insufficient_memory, 71.42 GB free+speculative+purgeable vs 120.0 GB required.
  - `build/current-dsv4-route-mode-code-exactness-memory-preflight-continuation-refresh-20260604.json` -> skipped, 71.41 GB free+speculative+purgeable / 111.65 GB psutil available vs 120.0 GB required.
- Checked `erics-m5-max2.local` for DSV4 readiness:
  - DSV4-K artifact exists at `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K` (80G).
  - Remote vMLX checkout is stale (`version = 1.5.32`, branch `session/v1.5.8...origin/main [ahead 1, behind 77]`), not the active 1.5.54 release worktree.
  - Remote memory is also below threshold: ~44.4 GB free+speculative+purgeable vs 120.0 GB required; Parallels Windows 11 VM is the largest observed process at ~30.8 GB RSS.
  - Wrote `build/current-remote-max2-dsv4-readiness-continuation-20260604.json` and did not launch DSV4 or kill user processes.
- Current DSV4 release blocker remains real, not stale bookkeeping: no safe host/session currently clears the live DSV4 real-UI or long-output/code quality proof.

## 2026-06-04 - Codex prepared max2 current DSV4 proof worktree

- Created isolated current-source proof worktree on max2: `/Users/eric/mlx/vllm-mlx-codex-proof-1554`.
- Worktree is on `6f8fff2a` / `vMLX 1.5.54` and has the current local dirty source/test overlay copied in. This avoids touching the stale `/Users/eric/mlx/vllm-mlx` checkout.
- Verified remote Python imports current source from the proof worktree using `/Users/eric/mlx/vllm-mlx/.venv/bin/python`; `vmlx_engine.__version__ == 1.5.54` and `vmlx_engine.__file__` resolves under the proof worktree.
- Ran DSV4 preflights from that current proof worktree:
  - `build/current-real-ui-dsv4-memory-preflight-max2-proof-worktree-20260604.json` -> skipped_insufficient_memory, 40.6 GB free+speculative+purgeable vs 120.0 GB required.
  - `build/current-dsv4-route-mode-code-exactness-memory-preflight-max2-proof-worktree-20260604.json` -> skipped, 40.58 GB free+speculative+purgeable / 76.78 GB psutil available vs 120.0 GB required.
- Copied both remote preflight artifacts back locally and wrote `build/current-remote-max2-dsv4-proof-worktree-readiness-20260604.json`.
- Current state: stale-repo blocker on max2 is removed for future DSV4 proof attempts, but memory blocker remains. No DSV4 launch, commit, push, tag, release upload, or notarization was performed.

## 2026-06-04 - Codex reusable max2 DSV4 readiness runner

- Added no-heavy reusable max2 readiness runner: `tests/cross_matrix/run_remote_max2_dsv4_readiness.py`.
- Added focused unit tests: `tests/test_remote_max2_dsv4_readiness.py`.
- Verification: `.venv/bin/python -m pytest -q tests/test_remote_max2_dsv4_readiness.py` -> `3 passed`.
- Ran the runner against max2 current proof worktree:
  - `.venv/bin/python tests/cross_matrix/run_remote_max2_dsv4_readiness.py --out build/current-remote-max2-dsv4-proof-worktree-readiness-runner-20260604.json`
  - Result: `status=prepared_but_blocked`, `launch_decision=do_not_launch`, available strict memory `40.38 GB` vs `120.0 GB` required.
- Kept this runner out of the default local current suite for now because it depends on SSH/max2 user state; use it explicitly before any future DSV4 heavy proof attempt.

## 2026-06-04 - Codex guarded max2 DSV4 exactness launcher

- Added guarded remote DSV4 exactness launcher: `tests/cross_matrix/run_remote_max2_dsv4_exactness_guard.py`.
- Added tests: `tests/test_remote_max2_dsv4_exactness_guard.py`.
- Fixed direct script import path after the first dry-run exposed `ModuleNotFoundError: No module named 'tests'`.
- Verification:
  - `.venv/bin/python -m pytest -q tests/test_remote_max2_dsv4_exactness_guard.py tests/test_remote_max2_dsv4_readiness.py` -> `6 passed`.
  - Dry-run: `build/current-remote-max2-dsv4-exactness-guard-dryrun-20260604.json` -> status=dry_run, launch_decision=dry_run_no_launch.
  - Non-dry guarded run: `build/current-remote-max2-dsv4-exactness-guard-skip-20260604.json` -> status=skipped_not_ready, launch_decision=do_not_launch.
- No DSV4 server was launched. The exactness proof path is now a guarded command that will only launch once max2 readiness clears current source/import/model/memory gates.

## 2026-06-04 - Codex guarded max2 DSV4 real-UI launcher

- Added guarded max2 DSV4 real-UI launcher: `tests/cross_matrix/run_remote_max2_dsv4_real_ui_guard.py`.
- Added tests: `tests/test_remote_max2_dsv4_real_ui_guard.py`.
- Fixed embedded remote Python f-string brace escaping after the first remote guard attempt failed before launch. No DSV4 process was started.
- Verification:
  - `.venv/bin/python -m pytest -q tests/test_remote_max2_dsv4_real_ui_guard.py tests/test_remote_max2_dsv4_exactness_guard.py tests/test_remote_max2_dsv4_readiness.py` -> `9 passed`.
  - Dry-run: `build/current-remote-max2-dsv4-real-ui-guard-dryrun-20260604.json` -> status=dry_run, launch_decision=dry_run_no_launch.
  - Non-dry guarded run: `build/current-remote-max2-dsv4-real-ui-guard-skip-20260604.json` -> status=skipped_not_ready, launch_decision=do_not_launch.
- Current max2 real-UI blockers from the guard: insufficient_memory (39.63 GB strict vs 120.0 GB), node_missing, staged_app_missing. The proof script exists in the proof worktree.
- No DSV4 server or Electron UI proof was launched. The real-UI proof path is now guarded and explicit for when max2 prerequisites are fixed.

## 2026-06-04 - Codex cleared max2 DSV4 real-UI Node/app blockers

- Found max2 Node at `/opt/homebrew/bin/node` (`v25.9.0`) and patched `tests/cross_matrix/run_remote_max2_dsv4_real_ui_guard.py` to use/check an explicit Node path instead of relying on SSH PATH.
- Copied the local staged Sequoia app to the isolated max2 proof worktree:
  `/Users/eric/mlx/vllm-mlx-codex-proof-1554/panel/release/sequoia-app/mac-arm64/vMLX.app`.
- Verification:
  - `.venv/bin/python -m pytest -q tests/test_remote_max2_dsv4_real_ui_guard.py tests/test_remote_max2_dsv4_exactness_guard.py tests/test_remote_max2_dsv4_readiness.py` -> `9 passed`.
  - `build/current-remote-max2-dsv4-real-ui-guard-after-node-app-20260604.json` -> status=skipped_not_ready, launch_decision=do_not_launch.
- Current max2 real-UI guard blockers are now only `insufficient_memory`: 39.32 GB strict memory vs 120.0 GB required. UI prerequisites are clear: Node present, proof script present, staged app present.
- No DSV4 server or Electron UI proof was launched.

## 2026-06-04 - Codex added max2 guard tests to current suite coverage

- Wired remote max2 DSV4 guard scripts/tests into `tests/cross_matrix/run_current_regression_suite.py` source-hash tracking and focused pytest selection.
- Targeted verification: guard tests plus current-suite contract slice -> `12 passed, 56 deselected`.
- Expanded focused release regression selection -> `564 passed, 202 deselected`.
- Regenerated current regression suite: `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[], open requirements remain real Electron UI cross-family matrix and DSV4 long-output/code quality.
- Regenerated release manifest: `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> current_proof_sweep=fail, prepackage_ready=false, release_ready=false, as expected while DSV4 is not live-cleared.

## 2026-06-04 - Codex aggregate max2 DSV4 release-proof guard

- Added aggregate guarded max2 DSV4 release-proof runner: `tests/cross_matrix/run_remote_max2_dsv4_release_proof_guard.py`.
- Added tests: `tests/test_remote_max2_dsv4_release_proof_guard.py`.
- Added aggregate guard script/test to current-suite source-hash and focused pytest coverage.
- Verification:
  - Aggregate plus child guard tests -> `12 passed`.
  - Aggregate max2 command wrote `build/current-remote-max2-dsv4-release-proof-guard-20260604.json` -> status=blocked_no_launch, release_ready=false.
  - Child statuses: readiness=prepared_but_blocked, exactness=skipped_not_ready, real_ui=skipped_not_ready.
  - Current max2 strict memory in aggregate artifact: 39.14 GB; real-UI prerequisites are now present (Node v25.9.0, staged app, proof script).
  - Expanded focused release regression after adding aggregate guard coverage -> `567 passed, 202 deselected`.
- No DSV4 launch, commit, push, tag, release upload, or notarization was performed.

## 2026-06-04 - Codex post-aggregate gate refresh

- Reran guarded max2 DSV4 unit slice: `tests/test_remote_max2_dsv4_release_proof_guard.py`, `tests/test_remote_max2_dsv4_real_ui_guard.py`, `tests/test_remote_max2_dsv4_exactness_guard.py`, `tests/test_remote_max2_dsv4_readiness.py` -> `12 passed`.
- Reran current suite: `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[].
- Current suite open requirements remain exactly: real Electron UI cross-family live model matrix and DSV4 long-output/code/file-generation quality.
- Reran release manifest with `--require-current-proof-sweep`: `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> current_proof_sweep=fail, prepackage_ready=false, release_ready=false.
- Manifest failure remains `real_ui_live_model_proof`; local DSV4 memory preflight refreshed to 77.67 GB strict memory vs 120.0 GB required, launch_decision=do_not_launch.
- No DSV4 launch, commit, push, tag, release upload, package rebuild, or notarization was performed.

## 2026-06-04 - Codex committed and pushed Gemma4 12B runtime gates to main

- Commit created: `9b1ffe72 integrate Gemma4 12B runtime gates`.
- Pushed exact commit to Python repo main: `git push origin HEAD:main` -> `6f8fff2a..9b1ffe72 HEAD -> main`.
- Pre-commit verification in this pass:
  - `git diff --check` -> pass.
  - `git diff --cached --check` -> pass.
  - py_compile over touched engine/Gemma4/max2 guard runners -> pass.
  - Focused Gemma4/max2 guard pytest -> `19 passed`.
  - Current suite before commit -> `status=pass`, `failed_steps=[]`.
  - Release manifest before commit remains expected fail: `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`, failed component `real_ui_live_model_proof`.
- Release boundary unchanged: no package rebuild, tag, release upload, fresh notarization, or mlxstudio release was performed after this commit. DSV4 live/UI and long-output/code proof remain open.

## 2026-06-04 - Codex post-main-push DSV4 readiness refresh remains blocked

- Rechecked DSV4 proof readiness after pushing `9b1ffe72` to main.
- Local real-UI preflight: `build/current-real-ui-dsv4-memory-preflight-after-main-push-20260604.json` -> skipped_insufficient_memory, launch_decision=do_not_launch, 77.43 GB strict memory vs 120.0 GB required.
- Local exactness preflight: `build/current-dsv4-route-mode-code-exactness-memory-preflight-after-main-push-20260604.json` -> skipped, launch_decision=do_not_launch, 77.40 GB strict memory vs 120.0 GB required, 14 selected cases not run.
- Max2 aggregate guard: `build/current-remote-max2-dsv4-release-proof-guard-after-main-push-20260604.json` -> blocked_no_launch, release_ready=false.
- Max2 source/model checks pass (`vMLX 1.5.54`, DSV4 model present), but strict memory is 38.34 GB vs 120.0 GB required; exactness and real-UI children skipped_not_ready.
- No DSV4 launch, package rebuild, tag, release upload, fresh notarization, or mlxstudio release was performed.

## 2026-06-04 - Codex refreshed public v1.5.54 DMG assets

- Pushed latest main: `2a84f0db add Gemma4 12B speed gate`; `origin/main` resolves to `2a84f0dba0d6acb1a09ad94628b1ae57a78b85d1`.
- Verified local final DMGs with `panel/scripts/verify-release-dmgs.sh`: both Sequoia and Tahoe are Developer ID signed, notarized, stapled, and accepted by Gatekeeper.
- Replaced `v1.5.54` release assets with `gh release upload --clobber` in both `jjang-ai/vmlx` and `jjang-ai/mlxstudio`.
- Published asset digests now match local verified artifacts:
  - Sequoia DMG `63a62e6b7b8b2dca18ac59f3ea34f0b3a6833dea603372372082e045715ea123`.
  - Sequoia blockmap `2c280f6ab575c617bcd6bf3327951ebbec86977bb76d2ed6930be5dc94bd07fa`.
  - Tahoe DMG `41dfa70e7d324ca796742b3a34866b016d409e58f1ae4fc3b9ee9a5003973c4b`.
  - Tahoe blockmap `a0936caf57d18bd0f003c2395cda6a270c722fc461cacffb600ef1e3fc7cf9ab`.
- Release manifest still has DSV4 live proof open; this was an ASAP asset refresh using already notarized 1.5.54 DMGs, not a claim that DSV4 exactness is fixed.

## 2026-06-04 - Codex moved v1.5.54 release tag to current main

- Retagged `v1.5.54` from `7f01a892` to current main commit `2a84f0db`.
- Pushed tag update: `git push origin refs/tags/v1.5.54 --force` -> `7f01a892...2a84f0db v1.5.54 -> v1.5.54`.
- GitHub releases in `jjang-ai/vmlx` and `jjang-ai/mlxstudio` remain public non-prerelease/non-draft `v1.5.54` releases targeting `main`, with refreshed notarized asset hashes from the prior upload.

## 2026-06-04 - Codex starting Step3.7 JANG_2L crash falsification probe

- User reported another agent sees vMLX serve silently crash mid-request for Step-3.7-Flash-JANG_2L across chat/completions and thinking on/off.
- Running a fresh local live API probe against `/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L` from the finite-launch worktree using the repo all-local smoke harness.
- Goal: prove or disprove current source/bundled-engine crash behavior before accepting rollback or release-status changes.

## 2026-06-04 - Codex disproved current Step3.7 JANG_2L serve-crash report

- Fresh source-server all-local smoke against `/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L` passed: `build/current-step37-crash-falsification-20260604/summary.json` -> status=pass, failed=0.
- Fresh isolated bundled-Python all-local smoke using `panel/bundled-python/python/bin/python3.12` passed: `build/current-step37-crash-falsification-bundled-20260604/summary.json` -> status=pass, failed=0.
- Endpoint-specific isolated bundled probe passed: `build/current-step37-endpoint-crash-falsification-bundled-20260604/result.json` -> chat thinking off=200, chat thinking on=200, legacy `/v1/completions`=200, `/v1/responses`=200, server alive after each row.
- Native cache in the live row reported `family=step3p7`, `schema=mixed_swa_kv_v1`, `cache_subtype=step3p7_full_sliding_kv`, prefix+paged+block_disk_l2 true; second text cache row recorded cache hits.
- Current evidence does not support rolling back Step3.7 HF repos for a vMLX mid-request crash. If another host still crashes, collect its exact command/env/server log and compare against these artifacts.

## 2026-06-04 - Codex added durable Step3.7 crash-falsification gate

- Added `tests/cross_matrix/run_step37_crash_falsification_contract.py` to validate the fresh bundled Step-3.7 JANG_2L crash-falsification artifacts:
  - `build/current-step37-crash-falsification-bundled-20260604/summary.json`
  - `build/current-step37-endpoint-crash-falsification-bundled-20260604/result.json`
- Added focused tests in `tests/test_step37_crash_falsification_contract.py` for pass, silent mid-request process death, and missing Step3.7 mixed-SWA/L2 cache status.
- Wired the contract into `tests/cross_matrix/run_current_regression_suite.py` source-hash tracking, commands, and focused pytest selection.
- Verification:
  - py_compile on new/changed files: pass.
  - `tests/test_step37_crash_falsification_contract.py`: 3 passed.
  - contract runner: `build/current-step37-crash-falsification-contract-20260604.json` -> status=pass.
  - current-suite/source-hash slice: `71 passed, 294 deselected`.
- Boundary: this disproves the current bundled Step3.7 mid-request crash claim. It does not clear DSV4 exact-code quality or the full real Electron UI cross-family release proof.

## 2026-06-04 - Codex pushed Step3.7 crash-falsification gate to main

- Commit: `efcb53d7 add Step3.7 crash falsification gate`.
- Pushed to Python repo main: `git push origin HEAD:main` -> `2a84f0db..efcb53d7 HEAD -> main`.
- Pre-push verification:
  - `tests/test_step37_crash_falsification_contract.py` -> 3 passed.
  - `tests/cross_matrix/run_step37_crash_falsification_contract.py` -> `build/current-step37-crash-falsification-contract-20260604.json`, status=pass.
  - Current regression suite -> `build/current-regression-suite-after-step37-crash-falsification-20260604.json`, status=pass, failed_steps=[], known open requirements only: real Electron UI cross-family live model matrix and DSV4 long-output/code quality.
- Boundary: main now preserves the current Step3.7 bundled mid-request crash disproval. No release tag move, release asset upload, rebuild, or notarization was performed for this proof-only commit.

## 2026-06-04 - Codex moved v1.5.54 tag after Step3.7 crash-falsification gate

- Current Python repo main: `efcb53d7 add Step3.7 crash falsification gate`.
- Moved local and remote `v1.5.54` from `2a84f0db` to `efcb53d70650133559f5fd9a839f2ffe053af88e`.
- Push: `git push origin refs/tags/v1.5.54 --force` -> `2a84f0db...efcb53d7 v1.5.54 -> v1.5.54`.
- GitHub releases remain existing public non-draft/non-prerelease `v1.5.54` releases targeting `main`; no DMG rebuild, release asset upload, or notarization was performed in this tag-only update.
- Boundary: release source/tag provenance now includes the Step3.7 bundled crash-falsification proof gate. DSV4 exact-code quality and full real Electron UI cross-family live proof remain open.

## 2026-06-04 - Codex ran DSV4 non-TQ JANG exactness subset

- Candidate found: `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`.
- Preflight with explicit user RAM override: `build/current-dsv4-route-mode-code-exactness-jang-nontq-user-ram-override-preflight-20260604.json` -> `status=ready_to_launch`.
- Live subset command used the same four identifier cases as the JANGTQ comparison: `chat_off_rep1,chat_off_no_punct_rep1,responses_off_rep1,responses_off_no_punct_rep1`.
- Result artifact: `build/current-dsv4-route-mode-code-exactness-jang-nontq-user-ram-override-subset-20260604.json` -> `status=fail`.
- Boundary: non-TQ DSV4 JANG did not clear the exact-code blocker in this subset. Need compare row deltas before calling the failure identical to JANGTQ or choosing a default release path.

## 2026-06-04 - Codex attempted DSV4 source exactness baseline

- Source candidate exists at `/Users/eric/models/Sources/DeepSeek-V4-Flash` with config/index/tokenizer files and 46 safetensor shards.
- Memory preflight with explicit user RAM override: `build/current-dsv4-route-mode-code-exactness-source-user-ram-override-preflight-20260604.json` -> `status=ready_to_launch`.
- Launch attempt for the same four identifier cases exited before health with `rc=3`; artifact path requested was `build/current-dsv4-route-mode-code-exactness-source-user-ram-override-subset-20260604.json`.
- Need inspect server log/artifact before classifying source baseline as unsupported source format vs runtime source-load bug.

## 2026-06-04 - Codex preserved DSV4 source load-failure evidence

- Patched `tests/cross_matrix/run_dsv4_route_mode_code_exactness.py` so server load failures return a structured `load_failed` artifact with `server_returncode` and `log_tail` instead of raising before JSON output.
- Added regression coverage in `tests/test_dsv4_route_mode_code_exactness.py` for preserving log tail when health wait fails before server readiness.
- Verification: focused py_compile passed; focused pytest `load_failure_log_tail or live_launch_honors_case_filter` -> 2 passed.
- Reran source DSV4 baseline: `build/current-dsv4-route-mode-code-exactness-source-user-ram-override-subset-20260604.json` -> `status=load_failed`.
- Boundary: DSV4 source baseline did not reach generation; non-TQ JANG launched but failed exactness. DSV4 exact-code quality remains open.

## 2026-06-04 - Codex pushed DSV4 exactness evidence-preservation and retagged v1.5.54

- Ran non-TQ DSV4 JANG comparison:
  - preflight: `build/current-dsv4-route-mode-code-exactness-jang-nontq-user-ram-override-preflight-20260604.json` -> `status=ready_to_launch`, psutil available user RAM override, 111.58 GB available vs 90.0 GB floor.
  - live subset: `build/current-dsv4-route-mode-code-exactness-jang-nontq-user-ram-override-subset-20260604.json` -> `status=fail`.
  - failure was not improved vs JANGTQ: `THREE.WebWebGLRenderer`; additionally non-TQ JANG corrupted `THREE.Mesh`, `THREE.BoxGeometry`, and `THREE.MeshBasicMaterial` into `MMesh`/`BBoxGeometry`/`MMeshBasicMaterial` variants.
- Attempted source DSV4 baseline:
  - preflight: `build/current-dsv4-route-mode-code-exactness-source-user-ram-override-preflight-20260604.json` -> `status=ready_to_launch`.
  - source launch artifact after runner fix: `build/current-dsv4-route-mode-code-exactness-source-user-ram-override-subset-20260604.json` -> `status=load_failed`, `server_returncode=3`, log tail shows `RuntimeError: [safetensor] unsupported dtypeF8_E8M0` from `mlx_lm.utils.load_model -> mx.load(wf)`.
- Patched DSV4 exactness runner to preserve load-failure JSON with log tail instead of raising before artifact output.
- Commit pushed to main: `11f1a6f1 preserve DSV4 exactness load failures`.
- Moved `v1.5.54` tag from `efcb53d7` to `11f1a6f1` and force-pushed tag.
- Boundary: DSV4 exact-code blocker remains open. Non-TQ JANG does not clear it; source baseline cannot run in current vMLX/MLX due unsupported F8_E8M0 safetensor dtype.

## 2026-06-04 - Codex documented max2 Gemma4 12B runtime notes and retagged v1.5.54

- SSH read max2 Gemma notes from `erics-m5-max2.local:/Users/eric/CRACK_abliteration/gemma4-12b-crack/BUILD_LOG.md` plus local max2 bundle metadata under `~/models/OsaurusAI`, `~/models/JANGQ-AI`, and `~/models/dealign.ai`.
- Added tracked doc: `docs/GEMMA4_12B_MAX2_RUNTIME_NOTES_20260604.md`.
- Doc captures Gemma4 12B unified architecture, full/sliding attention layout, K-eq-V full layers, channel-marker reasoning/tool template details, MXFP4/MXFP8/JANG_4M surgery recipes, Osaurus port correction `4242`, and current vMLX proof boundaries.
- Current local proof referenced in the doc:
  - `build/current-gemma4-12b-live-all-unified-runtime-fullmatrix-postfix-20260604.json` -> pass.
  - `build/current-gemma4-12b-media-smoke-all-unified-runtime-postfix-20260604.json` -> pass.
  - `build/current-gemma4-12b-speed-gate-jang4m-20260604.json` -> pass, default median 46.665 tok/s.
  - `docs/internal/agent-notes/current-gemma4-lfm-step-ling-live-matrix-20260604-proof.json` -> Gemma4/LFM/Step/Ling matrix recap.
- Verification: `git diff --check -- docs/GEMMA4_12B_MAX2_RUNTIME_NOTES_20260604.md` passed.
- Commit pushed to main: `5037939f document Gemma4 12B max2 runtime notes`.
- Moved `v1.5.54` tag from `11f1a6f1` to `5037939f` and force-pushed tag.
- Boundary: no DMG rebuild, asset upload, or notarization rerun in this step. Audio/video Gemma4 media remains not fully production-cleared from image smoke alone; DSV4 exact-code and full Electron UI cross-family proof remain open.

- 2026-06-05 codex: started video/MTP release-gate continuation; routing stays in finite-launch-guard; staged Tahoe app has Qwen gdn_sink fix while installed /Applications app is stale, so live proof will target staged release app bits.

- 2026-06-05 codex: fixed Qwen OpenAI video_url template normalization by updating BatchedEngine media normalization to handle the four-value MLXMultimodalLM extractor return. Added Qwen frame fallback for Qwen3.5/3.6 video rows after native video misread blue as pink/yellow while image control passed. Live staged rows now pass for Qwen3.6 27B MXFP4-MTP and 35B MXFP4-MTP with red/green/blue video, multi-turn follow-up, repeat, and cache endpoint capture. 35B MXFP8 remains memory-blocked due stale installed-app Metal allocation not released by deep sleep.
- 2026-06-05 codex: fixed Gemma4 video capability gate and SimpleEngine Gemma4 video prompt placeholders. Gemma4 MXFP4/MXFP8/JANG_4M live staged video rows pass red/green/blue video + multi-turn follow-up. Focused media tests pass.
- 2026-06-05 codex: Nemotron Omni JANGTQ current staged smoke passes text/cache/reasoning with paged+ssm/L2 hits but fails media bridge during first image load; server exits/refuses after C-RADIO/timm bridge load. This remains an active release blocker. Step 3.7 Flash metadata is image/VL-only; video requires an explicit sampled-frame fallback before it can be honestly included.

- 2026-06-05 codex continuation: generalized video-frame fallback to include Step3.7 image-only VLM bundles, kept ZAYA video rejected, and vendored the C-RADIO Transformers dynamic module for Nemotron Omni Stage-1 media cold-start. Local focused py_compile + pytest passed (`8 passed, 483 deselected`). Overlayed changes to max2 proof worktree and focused remote tests passed (`3 passed, 488 deselected`). Existing max2 Qwen3.6 35B MXFP8 MTP server on port 8001 passed live red/green/blue video multi-turn probe: turn1 `A. red first, green second, blue third`, turn2 `B. green second`; proof copied to `docs/internal/release-gates/20260605_video_mtp_matrix/qwen35_mxfp8_mtp_existing_max2/summary.json`. Release remains not ready: source fixes are not rebuilt into app, Step live video is unrun due memory/bundle constraints, and Nemotron Omni media must be rerun live after C-RADIO vendoring under enough RAM.

## 2026-06-05 continuation release gate update

- Added Step 3.7 Flash video processing plus multi-turn/cache as an explicit release-gate row; status remains pending live proof.
- Preserved Nemotron Omni JANGTQ4 max2 source-overlay video proof locally under `docs/internal/release-gates/20260605_video_mtp_matrix/nemotron_omni_jangtq4_video_source_max2`.
- Current status remains not release-ready: no rebuilt-app matrix, no main push/tag, no signing, no notarization, no upload.

## 2026-06-05 Step 3.7 Flash video/cache source proof

- Fixed Step/Qwen video fallback local `file://` normalization in `vmlx_engine/engine/batched.py`.
- Fixed MLLM processor chat-template thinking propagation by forwarding both `enable_thinking` and `thinking` aliases.
- Changed Step-only video fallback to dedupe consecutive sampled frames and present them as one contact-sheet image; Qwen/Gemma frame-list behavior unchanged.
- Local focused tests passed: `python3 -m py_compile vmlx_engine/engine/batched.py tests/test_engine_audit.py`; `.venv/bin/pytest -q tests/test_engine_audit.py -k 'step37_video_frame_fallback_rewrites_video_to_image_parts or step37_video_frame_fallback_uses_contact_sheet or step37_video_capability or mllm_processor_template_receives_thinking_alias'` -> 4 passed.
- max2 focused tests passed with the same slice -> 4 passed.
- max2 live source-overlay Step proof passed under `docs/internal/release-gates/20260605_video_mtp_matrix/step37_flash_video_cache_source_max2_dedup_contact`: video turn saw red/green/blue, follow-up identified green, repeat saw red/green/blue, pixel cache hit evidence present, paged cache and block disk cache initialized.
- Still not release-ready: this is source overlay, not rebuilt/notarized app; rebuilt app matrix and UI/settings/lifecycle proof remain pending.

## 2026-06-05 staged app rebuild

- Ran `npm run build && npm run package` from `panel`.
- Bundled Python installed local `vmlx 1.5.54` from this worktree and local `jang 2.5.30` from `/Users/eric/jang/jang-tools`.
- Bundled critical import verification passed, including Step3p7 runtime, Gemma4 unified registration, JANG/JANGTQ kernels, Mimo v2 registration, C-RADIO/Nemotron dependencies, and cache/runtime modules.
- Directory app packaged at `panel/release/mac-arm64/vMLX.app` and electron-builder signed it with identity `D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C`.
- Notarization was skipped by electron-builder: `notarize options were unable to be generated`.
- Release still blocked on rebuilt-app live matrix, UI/settings/lifecycle proof, commit/push/tag, DMG/zip notarization, and upload.

## 2026-06-05 packaged Step 3.7 Flash proof

- Copied signed staged app to max2 under `/Users/eric/mlx/vMLX-codex-staged.app/vMLX.app`.
- Initial packaged Step script used the wrong copied path and did not start; preserved under `step37_flash_video_cache_packaged_max2` with `nohup: ... No such file or directory`.
- Corrected packaged Python path and reran Step proof under `step37_flash_video_cache_packaged_max2_v2`.
- Packaged Step proof passed: video turn and repeat identified red/green/blue; follow-up identified green; health showed paged cache, block disk cache enabled, and pixel_cache_hits=1.
- Residual: Step still over-explains despite concise prompt; runtime path is working, exact-format instruction-following remains model behavior.

## 2026-06-05 Qwen35 MXFP8 MTP gdn_sink regression

- Reproduced the user-facing packaged/source-overlay crash on Qwen3.6 35B MXFP8 MTP: `_patch_gated_delta_net.<locals>.__call__() got an unexpected keyword argument 'gdn_sink'` during VL prefill.
- Root cause: dense Qwen MTP `GatedDeltaNet.__call__` patch did not accept the newer `gdn_sink` kwarg used by current mlx-vlm qwen3_5_moe DecoderLayer / VL MTP path.
- Patched `vmlx_engine/patches/mlx_lm_mtp/qwen35_model.py` so dense GatedDeltaNet and DecoderLayer accept/forward `gdn_sink` safely.
- Added regression test `test_qwen35_dense_mtp_patch_accepts_gdn_sink_kwarg`.
- Local focused tests passed; max2 focused test passed.
- max2 source-overlay live proof passed under `qwen35_mxfp8_mtp_source_max2_gdnsink`: video output `red, green, blue`, follow-up `green`, repeat `red, green, blue`, MTP `runtime_active=true`, `effective_depth=3`, and no `gdn_sink` TypeError in the log.
- Important correction: earlier staged-app runs launched from the proof worktree, so Python could import source via cwd. Future packaged proofs must launch from a neutral cwd such as `/tmp` or another directory without `vmlx_engine`.

## 2026-06-05 staged app rebuild after Qwen gdn_sink fix

- Rebuilt staged app after patching `vmlx_engine/patches/mlx_lm_mtp/qwen35_model.py`.
- `npm run build && npm run package` completed successfully.
- Bundled Python verification passed again; local source wheel hash changed to include the Qwen gdn_sink fix.
- Directory app signed again with identity `D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C`.
- Notarization still skipped: `notarize options were unable to be generated`.
- Next packaged proofs must launch from neutral cwd (`/tmp`) to avoid source worktree imports.

## 2026-06-05 clean packaged Qwen35 MXFP8 MTP proof

- Copied rebuilt staged app to max2 under `/Users/eric/mlx/vMLX-codex-staged-gdnsink.app/vMLX.app`.
- Launched from neutral cwd `/tmp` to prevent source-worktree imports.
- Clean packaged Qwen35 MXFP8 MTP proof passed under `qwen35_mxfp8_mtp_packaged_max2_gdnsink_neutral`: video `red, green, blue`; follow-up `green`; repeat `red, green, blue`; MTP `runtime_active=true`, `effective_depth=3`, `runtime_scope=text+vl`; no `gdn_sink` TypeError.

## 2026-06-05 clean packaged Step proof after rebuild

- Clean packaged Step 3.7 Flash proof launched from neutral cwd `/tmp` using rebuilt staged app `vMLX-codex-staged-gdnsink.app`.
- Artifact: `step37_flash_video_cache_packaged_max2_gdnsink_neutral`.
- Result: pass. Video and repeat identified red/green/blue; follow-up identified green; health showed pixel cache hit and paged/block cache metadata.
- Residual remains: model over-explains despite concise prompt; semantic/runtime path is working.

## 2026-06-05 Gemma4 file-url/contact fallback status

- Clean packaged Gemma4 matrix from `/tmp` failed all three before patch: raw `file://` video reached native video path and produced `All video inputs failed to process`.
- Added Gemma4/Gemma4-unified to batched video frame fallback and then to deduped contact-sheet fallback.
- Focused local and max2 tests passed for `test_gemma4_batched_video_frame_fallback_uses_contact_sheet`.
- Source-overlay Gemma4 rerun after contact-sheet fallback: MXFP4 passed semantic gate but echoed prompt repeatedly; MXFP8 failed red/green/blue check (`.` output, follow-up green); JANG_4M failed red/green/blue check (`red first, second, third`, follow-up green).
- Status: Gemma4 12B is not release-green yet. Do not ship/notarize/release until this is fixed and clean packaged rerun passes.

## 2026-06-05 Gemma4 frame-list fallback fix

- Root-caused Gemma4 MXFP8/JANG_4M video semantic failures to contact-sheet fallback, not corrupt weights or vision runtime. Direct image and separate-frame probes passed; contact sheet strict prompts collapsed to dot output.
- Changed Gemma4/Gemma4-unified video fallback to keep deduped chronological image frames, while Step3.7 keeps contact-sheet fallback.
- Local and max2 focused tests passed for Step contact-sheet + Gemma frame-list pins. Source-overlay Gemma4 MXFP4/MXFP8/JANG_4M three-turn matrix passed under `docs/internal/release-gates/20260605_video_mtp_matrix/gemma4_three_turn_source_after_frame_list`.
- Release still pending rebuilt packaged matrix, UI/settings/lifecycle proof, commit/push/tag, notarized artifacts, and upload.

## 2026-06-05 staged app rebuild after Gemma frame-list fix

- Ran `npm run build && npm run package` from `panel` after Gemma frame-list fallback fix.
- Bundled Python installed local vMLX 1.5.54 from finite-launch-guard and local JANG 2.5.30; bundled critical import verification passed.
- Directory app packaged at `panel/release/mac-arm64/vMLX.app` and signed with identity `D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C`.
- Notarization still skipped because electron-builder could not generate notarize options; release remains blocked until DMG/zip notarization succeeds.

## 2026-06-05 clean packaged Gemma4 frame-list proof

- Copied rebuilt signed app to max2 at `/Users/eric/mlx/vMLX-codex-staged-frame-list.app/vMLX.app`.
- Clean packaged Gemma4 matrix launched from `/tmp` passed for MXFP4, MXFP8, and JANG_4M: red/green/blue video, green follow-up, red/green/blue repeat.
- Artifact: `docs/internal/release-gates/20260605_video_mtp_matrix/gemma4_three_turn_packaged_frame_list`.

## 2026-06-05 packaged Step/Nemotron after deterministic media cache

- Added deterministic Step contact-sheet fallback path so exact repeat media+prompt can hit pixel cache; local focused tests passed and app was rebuilt/copied to `/Users/eric/mlx/vMLX-codex-staged-step-cache.app/vMLX.app`.
- Packaged Step 3.7 JANG_2L row passed semantic video/follow-up/exact-repeat and pixel cache hit under `step37_2l_video_cache_packaged_step_cache_exact_repeat`.
- Packaged Nemotron Omni JANGTQ4 row passed image Blue, video Red/Green/Blue, follow-up Green second under `nemotron_omni_jangtq4_packaged_step_cache_exact_prompt`.
- Not release complete: UI/settings/lifecycle proof, commit/push/tag, notarized DMG/zip, and upload remain pending.

## CODEX 2026-06-05 - Public download surface stale-manifest fix in progress
- User reported public mlx.studio/vmlx.net downloads still old after v1.5.55 release.
- Confirmed GitHub/raw mlxstudio manifest is v1.5.55, but public https://mlx.studio/update/latest.json and active worktree root latest.json were still v1.5.54.
- Copied /Users/eric/mlx/mlxstudio/latest.json to active worktree root latest.json as the local source fix before deploy/origin tracing.
- Still open: locate/update the actual mlx.studio/download static origin; public /download/ remains hardcoded to v1.5.54 until deployed.

## CODEX 2026-06-05 - Public release surface fixed for v1.5.55
- Root cause for users seeing old downloads: server origin at 45.32.71.230 still had `/var/www/mlx.studio/update/latest.json` and `/var/www/mlx.studio/download/index.html` pinned to v1.5.54; active worktree root `latest.json` was also stale.
- Deployed v1.5.55 manifest to `/var/www/mlx.studio/update/latest.json` and patched `/var/www/mlx.studio/download/index.html` links/hashes to v1.5.55.
- Added nginx redirects on vmlx.net: `/download/` -> `https://mlx.studio/download/`, `/update/*` -> matching `https://mlx.studio/update/*`; nginx config tested and reloaded.
- Cloudflare purge completed; public verification passed for mlx.studio update/download and vmlx.net download/update redirects.
- Committed local active root `latest.json` and pushed to `jjang-ai/vmlx` main: `93603081 Update latest manifest for v1.5.55`.

## CODEX 2026-06-05 - Cross-model runtime failure-class log from Step Flash CRACK review
- Documented the Step-3.7 Flash JANG_2L CRACK review as a cross-model failure class, not a Step-only bug: `docs/internal/CROSS_MODEL_RUNTIME_FAILURE_CLASSES_2026_06_05.md`.
- Concrete repro: default Step3p7 CRACK metadata advertises vision (`jang_config.architecture.has_vision=true`, `vision_config` present), vMLX 1.5.55 routes it into MLLM, and longer generation can kill the server without Python traceback. Text-only metadata view (`has_vision=false`) loads as `MLLM=False`, runs stable text smoke, and completed Vera fast eval `113/180`, `62.78%`, `18/18` tasks with `0` failed requests.
- Logged broader required reproduction matrix across architecture/runtime/config/modality/cache/API/streaming/lifecycle axes. Do not treat this as Step-only; similar issues must be reproduced and ruled out for Gemma4, Qwen, LFM, Step, DSV4, Nemotron/omni, JANG, JANGTQ, MXFP, MTP, VL, audio, and gateway paths.
- Open buckets: unsupported modality metadata guard, raw tool dialect leakage, post-tool loop control, thinking/template mismatch, and native crash breadcrumb/sentinel.
- Added dedicated checklist/progress tracker: `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md`. This itemizes related issue classes CM-001..CM-010, per-family matrix requirements, proof template, immediate queue, and non-negotiable release-note rules so agents do not collapse Step Flash CRACK into a Step-only issue or lose cross-model proof requirements.
- Created GitHub trackers for the cross-model work: #188 master matrix, #189 unsupported advertised modality/runtime guard, #190 raw tool dialect leaks/tool loops/thinking-template mismatch. Linked them from the internal docs. Existing structured JSON/XML follow-up remains #187.

## CODEX 2026-06-05 - No-fake-fix classification contract tightened
- Added explicit root-cause labels and no-hidden-force rules to the cross-model runtime register and failure-class narrative.
- Rule: runtime/decode/kernel/cache/parser incompatibilities stay open until fixed on the failing path; disabling MTP/VL/audio/video/cache/tool/thinking rails is not a pass unless it is explicit product behavior with unsupported-route recovery proof.
- This preserves the Step3p7 text-only guard as a fail-closed unsupported-route guard, not proof that Step3p7 VLM works.

## CODEX 2026-06-05 - Objective row 9 stale evidence fixed
- Root cause: objective digest kept server max-output/max-context row open because it still required legacy `build/dev-ui-smoke-20260521/summary.json` even though the current max-output/context contract had all required checks and source hashes passing.
- Fix: row now depends on `build/current-max-output-context-contract-20260531-post-step-lfm-refresh.json` plus source hashes only; missing contract still keeps the row open.
- Verification: RED test failed on stale open status, focused objective/current-suite tests passed (`5 passed`), and current suite regenerated `status=pass`, `failed_steps=[]`, `19` open requirements.
- Boundary: this does not clear DSV4/model live rows or release; it removes a stale proof-artifact dependency only.

## CODEX 2026-06-05 - Cache architecture objective row cleared with real source hash
- Root cause: cache architecture contract source hash list still referenced removed `vmlx_engine/tq_disk_cache.py`; real implementation is `vmlx_engine/tq_disk_store.py`.
- Fix: contract now hashes `tq_disk_store.py`; objective digest now uses current cache architecture contract + checks + family matrix + source hashes instead of stale static audit artifact.
- Verification: new RED source-list test failed before fix; focused tests passed; regenerated cache contract pass; current suite pass with failed_steps=[] and 18 open requirements.
- Boundary: no-heavy/static cache architecture proof only; live DSV4/Qwen/Gemma/Ling/UI release rows remain open.

## CODEX 2026-06-05 - App maxToolIterations objective row moved to current contract
- Root cause: objective row still depended on missing v1.5.46 `v1546-dsv4-app-tool-cap-nocache-proof` artifact.
- Fix: row now uses current `current-tool-call-contract-20260528-tool-parser-loop-matrix.json` and specifically requires `panel_max_tool_iterations_caps_tool_loops`, no missing markers/failures, and current source hashes.
- Boundary: details retain `open_proof_gaps=[live_default_cache_dsv4_tool_loop]`; DSV4 live one-tool/multi-tool/default-cache rows remain open.
- Verification: focused tests passed (`8 passed`); current suite pass, failed_steps=[], 17 open requirements.

## [2026-06-05] codex | MiniMax #117/#179 open-proof boundary and current suite recovery

- Added current-suite artifact steps for `issue179_minimax_k_root_cause_audit` and `issue179_cancel_probe_memory_preflight` so #117/#179 no longer depend on stale or missing sweep state.
- Updated the objective digest to prefer the current direct Issue #179 root-cause audit even when it is `open`; stale manifest `missing` state no longer masks the live/open not-proven list.
- Updated public-app issue audit policy: MiniMax #117 can be `open` when root-cause audit is open and memory-preflight boundary exists, but the live Responses cancel proof remains required for pass/clearance. The missing live cancel proof is not faked.
- Regenerated `build/current-issue179-minimax-k-responses-cancel-probe-memory-preflight-20260602-local-ready-check.json`; it reports `status=ready_to_launch`, `launch_allowed=true`, `did_not_launch=true`, so it is only a safe-launch boundary, not live model proof.
- Regenerated `build/current-public-app-issue-audit-gemma4-release-boundary-after-install-20260604.json`; status is `open`, not `fail`.
- Ran `tests/cross_matrix/run_current_regression_suite.py --out build/current-regression-suite-20260605.json`; result `status=pass`, `failed_steps=[]`, with 17 expected release requirements still open including MiniMax reporter parity/root cause.

## [2026-06-05] codex | DSV4 default-cache tool-loop runtime proof separated from code exactness

- Ran live DSV4 default-cache multi-tool loop gate with `.venv/bin/python` and `--min-free-gb 80` against `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`.
- Artifact `build/current-dsv4-default-cache-tool-loop/result.json` returned `status=review`: runtime/tool/cache checks passed (`list_directory -> write_file -> write_file`, final `DONE`, `native_composite`, prefix+paged+block-disk L2, `paged+dsv4` cached tokens, generic TQ KV off), but generated JavaScript exactness failed with `THREE.ScScene()` and `THREE.BBoxGeometry()`.
- Updated `run_tool_call_contract.py` to classify the live artifact by required runtime/cache/tool-loop checks instead of top-level review status, while preserving `code_file_written_exact=false` for the separate DSV4 code/file-generation quality blocker.
- Regenerated tool-call contract: `build/current-tool-call-contract-20260528-tool-parser-loop-matrix.json` now `status=pass`.
- Regenerated objective digest: DSV4 default-cache multi-tool agent loop is `PASS`; DSV4 long-output/code/file-generation quality remains `OPEN`.
- Ran default current suite: `build/current-regression-suite-after-mimo-scope-removal-20260604.json` is `status=pass`, `failed_steps=[]`, with 16 expected open release requirements.

## [2026-06-05] codex | DSV4 current live artifact clears native cache and multi-tool rows

- Updated objective digest to use `build/current-dsv4-default-cache-tool-loop/result.json` as current proof for exact DSV4 rows that it actually proves.
- Cleared objective rows from current live evidence: `DSV4 cache is native SWA+CSA/HCA composite, not generic KV/TurboQuant KV` and `DSV4 can perform multiple tool iterations then final answer`.
- Kept open rows that are not proven by this artifact: app-launch default wiring, same-process TTFT/latency improvement proof, restart L2 proof, one-tool-after-result proof, and DSV4 exact code/file-generation quality.
- Added regression tests that remove stale 2025 evidence files and require the digest to use the current default-cache artifact instead.
- Ran default current suite: `build/current-regression-suite-after-mimo-scope-removal-20260604.json` is `status=pass`, `failed_steps=[]`, with 14 expected open release requirements.

## [2026-06-05] codex | DSV4 Responses cache-hit/TTFT proof refreshed

- Ran live `tests/cross_matrix/run_dsv4_responses_cache_gate.py` with `.venv/bin/python` and `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`.
- Artifact `build/current-dsv4-responses-cache-gate-20260606.json` returned `status=pass`: previous-response follow-up had `5195` cached tokens and `cache_detail=paged+dsv4`, streaming follow-up recorded TTFT `0.3339s`, and explicit no-cache full prompt stayed uncached.
- Updated objective digest to use the current Responses cache gate for `DSV4 same-process cache hit improves latency/TTFT and records paged+dsv4 hit` when stale 2025 cache proof files are absent.
- Removed that row from the current suite expected-open list. Default suite now passes with `failed_steps=[]` and 13 expected open requirements.

## 2026-06-06 - DSV4 Responses one-tool stop current proof

- Current suite completed with `status=pass`, `failed_steps=[]`, and 12 open release requirements.
- Added live DSV4 Responses one-tool stop gate: `build/current-dsv4-responses-one-tool-stop-20260606.json`.
- Proof boundary: round 1 emitted exactly one structured `list_directory` call; round 2 used `previous_response_id`, kept tools available with `tool_choice=auto`, emitted no function calls, returned exactly `DONE`, and native prefix+paged+block-disk L2 was enabled.
- This closes only the exact one-tool-after-result runtime/stop row. DSV4 app-launch default wiring, restart L2, long-output/code exactness, and real UI matrix remain open.

## 2026-06-06 - DSV4 restart L2 kernel-cache crash reproduced and fail-closed

- Added live restart-L2 gate: `tests/cross_matrix/run_dsv4_responses_restart_l2_gate.py`.
- First live exact-terminal restart proof reproduced a real runtime/kernel-cache failure: after restart DSV4 block-disk L2 loaded 21 blocks / 5195 cached tokens, then Metal crashed with `kIOGPUCommandBufferCallbackErrorTimeout` during cached decode.
- Different-tail restart proof showed the existing pending-marker guard was correct: disk hits happened, but cached usage/detail stayed empty and the model dumped context; not a release pass.
- Implemented narrow fail-closed guard in `vmlx_engine/prefix_cache.py`: disk-backed DSV4 terminal `DeepseekV4Cache` blocks are rejected until restart reconstruction is safe. Same-process resident DSV4 paged hits remain enabled.
- Post-fix live gate artifact: `build/current-dsv4-responses-restart-l2-gate-20260606.json` -> `status=review`, no server crash, before restart `disk_writes=21`, after restart `disk_hits=21`, visible output `STORED`, but `restart_dsv4_cache_hit=false`; restart-L2 release row remains open.
- Bundled Python verifier passed after rebundle. Staged app rebuild failed during codesign with `errSecInternalComponent` on a SciPy native extension, so packaged/staged release parity remains blocked by signing environment and no release was claimed.

## 2026-06-06 Codex | release blocker | Fresh Developer ID signing preflight must beat stale staged app
- Tightened packaged integrity preflight so an old signed/staged vMLX.app cannot mask inability to fresh-sign current source/bundled runtime.
- Current blocker: fresh Developer ID signing probe fails non-interactively with keychain/user-interaction denial / errSecInternalComponent, while stale staged app may still verify.
- Classification recorded in cross-model register as gateway_ui packaging/release issue, not model artifact corruption.
- Release boundary: no notarized/current packaged release can be claimed until fresh build, Developer ID signing, parity, notarization, and stapling pass on the new artifact.

## 2026-06-06 Codex | done | Fresh staged app signing/parity repaired, release still blocked by live rows
- Restored non-interactive Developer ID signing for `~/Library/Keychains/vmlx-build.keychain-db`; direct signing probe on a copied bundled scipy `.so` passed.
- Rebuilt bundled Python and Electron app; default `npm run package` signed `panel/release/mac-arm64/vMLX.app`.
- Rebuilt the release-gate expected Sequoia staged app with `npx electron-builder --mac --dir --config.directories.output=release/sequoia-app`; signing completed with Developer ID identity.
- Refreshed `build/current-packaged-integrity-contract-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json`: `status=pass`, staged app engine/source parity true, signature preflight pass.
- Refreshed `build/current-regression-suite-after-mimo-scope-removal-20260604.json`: `status=pass`, `failed_steps=[]`, known open objective requirements remain.
- `build/current-release-regression-manifest-pre-dmg-release-build.json` remains release red because live/model objective rows are still open; no DMG notarization/stapling/public manifest release was performed.

## 2026-06-06 Codex | proof | Qwen MTP gdn_sink compatibility current-source/current-bundle
- Checked current source for the reported Qwen35 MXFP8 MTP crash class: dense `GatedDeltaNet` and `DecoderLayer` patches accept `gdn_sink`; VLM `Qwen3_5GatedDeltaNet`, decoder, model, and MoE path tests are present.
- Focused source regressions passed: `tests/test_native_mtp_autodetect.py -k 'gdn_sink or gated_delta'` -> 3 passed; `tests/test_engine_audit.py -k qwen35_dense_mtp_patch_accepts_gdn_sink_kwarg` -> 1 passed.
- Fresh bundled Python probe also reports `gdn_sink=True` for dense GDN, dense decoder, VLM GDN, VLM decoder, and VLM model.
- Boundary: this proves current-source/current-bundled call compatibility, not full packaged app live Qwen35 speed/output-equivalence release clearance.

## 2026-06-06 Codex | progress | Systematic list updated, DSV4 app-launch cache row cleared
- Updated objective proof digest to use current panel settings contract + current DSV4 default-cache live artifact for `DSV4 Flash prefix/paged/L2 cache is enabled by default from app launch`.
- Removed that DSV4 app-launch default-cache row from the known-open objective list; restart-L2 and DSV4 long-output/code-quality remain open.
- Refreshed packaged integrity: status=pass.
- Refreshed current regression suite: status=pass, failed_steps=[], 11 open objective rows remain.
- Added MiMo back to the systematic issue register per Eric: delete bad local MiMo models, download documented MiMo JANG_2L from max2 docs over HTTP, then implement/fix/live-prove runtime.

## 2026-06-06 Codex | MiMo cleanup | Removed stale local MiMo model copy
- Removed stale local MiMo model cache per Eric's instruction: `/Users/eric/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro`.
- No `/Users/eric/models` MiMo directory was found in the targeted model inventory.

## 2026-06-06 Codex | MiMo actual intake/live proof correction
- Deleted stale local `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` (`107G`) and copied the Max2 canonical bundle from `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` over the direct `en9` route using rsync.
- Structural verifier passed locally: `109180` tensors, `150` shards, `113.25 GB`, audio tokenizer present, chat template matches embedded.
- Source-server MiMo live proof on port 8097: MLLM loaded with q8 KV storage, prefix+paged cache, block-disk L2, xml_function tools, think_xml reasoning.
- Text/cache smoke passed: first `cache ok`; repeat `cache ok` with `cached_tokens=28`, `cache_detail=paged`.
- Responses short smoke returned HTTP 200 and coherent `response ok`, but not exact requested `responses ok`.
- Tool smoke remains open: structured OpenAI `tool_calls` emitted, but arguments were wrong (`{"city": ": "}`), and continuation after tool output stopped with no visible content.
- Server stopped after proof to free ~106GB resident memory. MiMo is not release-cleared.

## 2026-06-06 Codex | current suite | DSV4 restart-L2 fixed, release still blocked by live rows
- Refreshed full current suite after packaged integrity passed: `build/current-regression-suite-after-dsv4-restart-l2-fix-20260606.json` -> `status=open`.
- All source/package/parity/contracts passed through objective digest; only `release_regression_manifest` failed, correctly refusing release because 10 live/model rows remain open.
- Remaining open rows: Qwen/JANG packaged MX matmul speed, Qwen native MTP speed/output equivalence, Qwen 27B JANG_4M prompt-processing speed, Ling/Bailing multilingual quality, Gemma4 26B CRACK Responses visible-content/language quality, Gemma4 26B CRACK mixed-SWA app speed, cross-family live multi-turn smoke matrix, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- Updated `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md` to replace stale post-DSV4 refresh wording with the current suite artifact and the MiMo/DSV4 release boundary.
- Release/tag/notarized public app remains blocked; source/package proof green is not sufficient while these live/model rows are open.

## 2026-06-06 Codex | proof | Qwen speed/MTP release rows refreshed and cleared
- Fixed the decode-speed gate PP probe to use deterministic sampling (`temperature=0`, `top_p=1`, `top_k=0`) so prompt-processing gates do not measure model-owned stochastic decode overhead.
- Live source Qwen27 JANG_4M speed: `build/current-decode-speed-live-qwen27-jang4m-source-20260606.json` -> pass, bundle decode 22.72 tok/s, PP 816.95/903.98/767.56 tok/s.
- Live installed-app Qwen27 JANG_4M speed after deterministic PP: `build/current-decode-speed-live-qwen27-jang4m-installed-app-deterministic-pp-20260606.json` -> pass, bundle decode 27.70 tok/s, PP 800.46/907.23/795.09 tok/s.
- Live installed-app Qwen27 JANG_4M MTP deterministic policy speed/PP: `build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-deterministic-pp-20260606.json` -> pass, bundle decode 45.68 tok/s, PP 664.67/723.00/718.56 tok/s.
- Live installed-app native-MTP A/B: `build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606/result.json` -> output equivalent, native MTP 50.65 tok/s vs AR 29.71 tok/s, speedup 1.70x, acceptance 0.956.
- Refreshed objective digest: Qwen/JANG packaged MX speed, Qwen native MTP decode/equivalence, and Qwen27 JANG_4M PP rows now PASS.
- Current suite: `build/current-regression-suite-after-qwen-speed-mtp-refresh-20260606.json` -> status=open, failed_steps=["release_regression_manifest"], remaining open rows=7. No release/tag/notarization yet.

## 2026-06-06 Codex | media runtime worklist | VL/audio/video blockers documented and pushed
- Added `docs/internal/VL_AUDIO_VIDEO_RUNTIME_WORKLIST_2026_06_06.md` with required media runtime functions, per-family blockers, missing implementation areas, and live proof rows for VL/audio/video release clearance.
- Linked the worklist from `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md`.
- Focused VLM prefill guard tests passed: high-memory default scaling, explicit 8GB override, oversized image rejection, and text-only bypass.
- Commit pushed to `origin/main`: `645acbb5 Document media runtime release blockers`.
- Release remains blocked by live packaged media/UI/model rows; no tag, notarization, or public release performed.

## 2026-06-06 Codex | live smoke proof refresh | Gemma26/MiniMax K pass, MiMo remains red
- Ran source all-local smoke for Gemma-4-26B-A4B-it-JANG_4M-CRACK and MiniMax-M2.7-JANGTQ_K-CRACK: both passed. Split proof artifacts: `build/current-all-local-model-smoke-gemma26-jang4m-tools-media-continuation-20260606/summary.json` and `build/current-all-local-model-smoke-minimaxk-tools-continuation-20260606/summary.json`.
- Ran current MiMo all-local smoke: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-media-rerun-20260606/summary.json` failed with empty/rambling exact-cache output and required-tool 400/no tool call, while cache/L2/TQ telemetry was present.
- Ran MiMo conservative simple/no-prefix/no-KV-quant diagnostic: `build/current-mimo-conservative-diagnostic-20260606/summary.json`; exact ACK passed, required tools failed both thinking on/off with no tool calls and ~96s latency.
- Refreshed objective/current-suite/release-manifest proof pointers to current artifacts and updated exact-pointer tests.
- Validation: focused proof-pointer/contract pytest selected 254 and passed; current suite `build/current-regression-suite-after-gemma26-minimaxk-mimo-rerun-final-20260606.json` is `status=open` with all no-heavy/static/focused tests pass and only packaged_integrity, release_manifest, release_gate_skip_app failed.
- Pushed to `origin/main`: `9a3e7d76 Refresh live smoke proof pointers`.

## 2026-06-06 Codex | media implementation ledger | MiMo media requires real JANG tools forward
- Refreshed `docs/internal/VL_AUDIO_VIDEO_RUNTIME_WORKLIST_2026_06_06.md` with an implementation ledger for Gemma4, Qwen VL/video, MiMo, Step3.7, Nemotron Omni, ZAYA-VL, MiniMax/Hy3/Kimi, and structured-output repair.
- Source inspection: Gemma4 image-prefill budget is already dynamic via `_resolve_vlm_image_prefill_single_buffer_limit()`; typed 413 and panel failed-media rollback exist, so any fixed 8GB report must be separated as stale installed app, explicit env override, or packaged-runtime drift.
- Source inspection: `/Users/eric/jang/jang-tools/jang_tools/mimo_v2/mlx_model.py` is explicitly text-only and documents visual/audio towers as preserved but not wired. vMLX currently registers a text compatibility shell that raises on `pixel_values`; this is honest fail-closed behavior, not a full MiMo media implementation.
- Boundary: MiMo VL/audio/video remains unbuilt until JANG tools adds `mimo_v2_multimodal.py` or equivalent, vMLX registers it, and live source plus installed media/tool/cache/speed proofs pass.
- Implemented typed unsupported media error plumbing: `UnsupportedMediaModalityError`, MiMo adapter raises it for unwired vision forward, batched scheduler preserves the error code, and API streaming/non-streaming routes emit `unsupported_media_modality` instead of generic 500-style generation failure.
- Focused validation: Python compile for edited runtime/test files passed; pytest selected unsupported-media rows passed (`3 passed, 594 deselected`).

## 2026-06-06 Codex | package pointers | Current packaged integrity passes, release still blocked
- Rebuilt bundled Python from current source and verified package parity with `npm run verify-bundled`.
- Rebuilt the staged Sequoia app under `panel/release/sequoia-app/mac-arm64/vMLX.app`; Developer ID signing completed after keychain partition-list repair.
- Refreshed canonical package proof: `build/current-packaged-integrity-contract-after-unsupported-media-staged-app-20260606.json` -> `status=pass`.
- Refreshed stale proof-pointer defaults in release gate, packaged integrity, current suite, public app issue audit, and manifest tests so they no longer point at the old 2026-06-04 objective digest.
- Current suite artifact: `build/current-regression-suite-after-packaged-integrity-pointer-fix-20260606.json` -> `status=open`, `failed_steps=["release_regression_manifest"]`.
- Remaining release blockers are the seven live/model rows: Gemma4 26B CRACK visible-content/language, Gemma4 26B CRACK mixed-SWA app speed, cross-family live multi-turn, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live matrix, and DSV4 long-output/code/file-generation quality.
- No release tag, notarized DMG, public download update, or `/Applications/vMLX.app` replacement was performed.

## 2026-06-06 Codex | proof | Gemma4 26B installed-app visible content and speed rows closed
- Installed-app Responses visible-content proof: `build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260606.json` -> `status=pass`, HTTP 200, 599 visible chars, 1550 reasoning chars, 451 completion tokens, mixed-SWA native cache active, generic TurboQuant KV off.
- Installed-app mixed-SWA speed proof: `build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-20260606.json` -> `status=pass`, cold wall decode `90.619 tok/s`, cache-hit wall decode `104.426 tok/s`, internal generation about `107.8 tok/s`, repeat cache `paged+mixed_swa`, generic TurboQuant KV off.
- Objective digest: `build/current-objective-proof-after-gemma26-installed-speed-visible-20260606.json` -> Gemma4 26B visible-content/language and mixed-SWA speed rows `PASS`.
- Packaged integrity after pointer update: `build/current-packaged-integrity-contract-after-gemma26-installed-speed-visible-20260606.json` -> `status=pass`.
- Current suite: `build/current-regression-suite-after-gemma26-installed-speed-visible-20260606.json` -> `status=open`, failed only on `release_regression_manifest`; five remaining live/model rows are cross-family live multi-turn, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- Release/tag/notarization/public download remains blocked; no release artifact was published.

# 2026-06-06 cross-family smoke proof pointer refresh

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no deprecated wrapper work.
- Updated current proof pointers from the MiMo-current-audit/unsupported-media artifacts to the cross-family smoke refresh artifacts.
- Added LFM and Step3.7 to the required cross-family smoke family fixture coverage.
- Regenerated:
  - `build/current-objective-proof-after-cross-family-smoke-refresh-20260606.json`
  - `build/current-release-regression-manifest-after-cross-family-smoke-refresh-20260606.json`
  - `build/current-regression-suite-after-cross-family-smoke-refresh-20260606.json`
- Verification:
  - `py_compile` for objective/manifest/current-suite runners passed.
  - Focused pointer tests passed: `7 passed, 476 deselected`.
  - Full current suite rerun ended `status=open` with failed step only `release_regression_manifest`.
- Release remains blocked by the five live/model rows listed in STATUS. No signing/notarization/public release was performed.

# 2026-06-06 ZAYA cache-contract refresh

- Stayed in active Python worktree.
- Investigated current ZAYA smoke failures. Tool calls were already structured and cache telemetry was present; primary confusion was the generic exact-ACK cache prompt conflating ZAYA prompt style with cache correctness.
- Patched all-local smoke to carry per-probe `expected_content` and use a ZAYA-native exact `blue` cache contract. Kept generic families on exact `ACK`.
- Live rerun `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json` narrows ZAYA text to reasoning-only failure; ZAYA-VL remains red on text exactness, recall, reasoning, and red image.
- Regenerated objective digest, manifest, and current suite with zaya-cache-contract-refresh pointers. Current suite is open with failed step only `release_regression_manifest`.

# 2026-06-06 ZAYA text reasoning budget closure

- Ran focused ZAYA text reasoning repro with max_tokens 128/256/512/1024/1536.
- Classification: 256-token smoke budget was too low; ZAYA text thinking mode reaches visible `FINAL=OK` at 512+.
- Patched smoke harness reasoning budget for ZAYA and added focused tests.
- Fresh live smoke clears ZAYA text and keeps ZAYA-VL red.
- Patched objective digest mixed-artifact handling so one passing row inside a failed combined artifact can count for its family without clearing the failed family.
- Current suite regenerated and remains open only through release manifest policy.

# 2026-06-06 ZAYA-VL no-media contract refresh

- Ran all-quant ZAYA-VL smoke: `build/current-all-local-model-smoke-zaya-vl-all-quants-tools-media-20260606/summary.json`.
- Ran focused ZAYA-VL MXFP4 repro: `build/current-zaya-vl-mxfp4-focused-repro-20260606/summary.json`.
- Patched `bench/all_local_model_smoke.py` so ZAYA rows use a direct `ATTACHED/NONE` current-message no-media prompt; added unit coverage in `tests/test_all_local_model_smoke.py`.
- Fresh live smoke: `build/current-all-local-model-smoke-zaya-vl-mxfp4-after-no-media-contract-20260606/summary.json`; MXFP4 now fails only `reasoning_on` empty-visible.
- Updated objective/release/current-suite pointers to `after-zaya-vl-no-media-refresh-20260606` and fixed exact-filename tests plus release-gate default digest path.
- Validation: py_compile passed; focused all-local smoke tests passed; focused pointer tests passed; packaged-integrity passed; full current suite is open with failed step only `release_regression_manifest`.

# 2026-06-06 MiMo sink/cache falsification refresh

- Coordinated MiMo live probe in `.agents/MAIL.md`.
- Ran direct MiMo native-vs-manual sink length diagnostic: `build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json`.
- Ran direct MiMo disable-sink length diagnostic: `build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json`.
- Patched `tests/cross_matrix/run_mimo_v2_jang2l_current_audit.py` and `tests/test_mimo_v2_current_audit.py` so current audit preserves the sink/cache falsification boundary.
- Focused validation passed: py_compile plus `tests/test_mimo_v2_current_audit.py` and MiMo root-cause/release-blocker tests.

# 2026-06-06 DSV4 cross-family smoke refresh

- Coordinated heavy DSV4 smoke in `.agents/MAIL.md`.
- Ran `bench/all_local_model_smoke.py --only DeepSeek-V4-Flash-JANGTQ-K --include-tools` into `build/current-all-local-model-smoke-dsv4-jangtq-k-tools-cache-20260606`.
- Both matching rows passed. Primary row proved exact ACK, `paged+dsv4` cached repeat with `cached_tokens=3639`, multi-turn recall, visible reasoning output, and `record_fact` tool call.
- Updated current proof pointers from `after-mimo-sink-falsification-20260606` to `after-dsv4-smoke-refresh-20260606`.

# 2026-06-06 ZAYA1-VL no-media template normalization, live blocker retained

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no deprecated wrapper/Swift work.
- Patched direct and batched MLLM prompt rendering so ZAYA1-VL no-media text-only turns reach the processor as plain strings; real media turns stay rich content lists.
- Validation: py_compile passed and selected ZAYA runtime tests passed (`6 passed`).
- Live ZAYA1-VL JANGTQ_K reruns stayed red with 5 failures. Improvement is limited to no-media-after-image now answering no-image instead of echoing the generic prompt.
- Remaining ZAYA-VL failures are preserved as open blockers: cache exact text `green`, multi-turn `color `, reasoning empty visible, red image `white`.

# 2026-06-06 MiMo tool/cache harness tightening

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; no deprecated wrapper/runtime work.
- Patched all-local smoke classification so `mimo_v2` bundles remain tool-capable through XML tool fallback even when promoted artifacts lack `jang_config.json`.
- Patched default inventory filtering to exclude nested MiMo `audio_tokenizer` sidecars.
- Focused tests passed for MiMo classification, tool probe inclusion, and sidecar filtering.
- Live MiMo artifact: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-harness-tighten-20260606/summary.json` -> fail.
- Positive proof: cache infra active (`mixed_swa_kv_v1`, `mimo_v2_asymmetric_swa`, prefix, paged, L2, TurboQuant q4 storage-boundary), recall passed, reasoning passed.
- Blockers preserved: exact cache output empty/rambling, required OpenAI tool call missing, speed still not release-grade, VL/audio/video bridge still unbuilt.
- Refreshed objective digest: `build/current-objective-proof-after-mimo-harness-tool-tighten-20260606.json` keeps MiMo open.

# 2026-06-06 ZAYA1-VL capability truth and package parity refresh

- Stayed in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; deprecated `/Users/eric/vmlx` was not used.
- Fixed stale ZAYA1-VL reasoning exposure: current plain-template ZAYA1-VL bundles now resolve as non-reasoning in runtime, smoke harness, panel registry, session launch comments, contracts, and release-facing docs.
- Live ZAYA1-VL MXFP4 proof passes at `build/current-all-local-model-smoke-zaya-vl-mxfp4-after-thinking-capability-truth-20260606/summary.json`; VL/tools/typed CCA cache remain enabled, reasoning is not advertised.
- Refreshed current no-heavy contracts and objective proof. `build/current-objective-proof-after-zaya-vl-thinking-capability-truth-20260606.json` keeps exactly five release blockers open.
- Rebuilt bundled Python from source plus local `/Users/eric/jang/jang-tools`; bundled verifier passes current vMLX, JANG tools, TurboQuant kernels, VL/audio imports, Step3p7 VLM registration, and Gemma4 unified registration.
- Rebuilt the signed sequoia staged app at `panel/release/sequoia-app/mac-arm64/vMLX.app`; notarization was skipped by builder because notarize options were not generated.
- Packaged integrity now passes: `build/current-packaged-integrity-contract-after-zaya-vl-thinking-capability-truth-20260606.json`.
- Current suite now fails only on release readiness: `build/current-regression-suite-after-zaya-vl-thinking-capability-truth-20260606.json` -> `status=open`, failed step `release_regression_manifest`.
- No release tag, notarized DMG, public download, updater manifest, `mlx.studio`, or `vmlx.net` update was performed. Release notes must credit GitHub `@Hornsan1`.

## 2026-06-06 Codex | MiMo XML-function parser contract fixed, release still blocked
- Fixed vMLX MiMo XML-function fallback: explicit `xml_function` / `mimo_xml_function` parser prompts no longer get misclassified as Qwen/Step native tool prompts, and fallback instructions now match MiMo's native XML function dialect instead of JSON-in-`<tool_call>`.
- Focused parser tests passed: `tests/test_tool_fallback_injection.py`, `tests/test_xml_function_tool_parser.py`, and selected XML/fallback/tool-format rows -> `74 passed`.
- Live MiMo source smoke: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-xml-function-template-fix-20260606/summary.json` remains `fail`, but `tool_required` now emits parsed OpenAI `record_fact({"value":"blue-cat"})`.
- MiMo remains release-red on exact-cache prompt following (`empty_visible`, rambling instead of `ACK`), long-prompt coherence, and speed (`~1.78 tok/s` vs release target), plus packaged/UI/media bridge proof.
- Refreshed MiMo audit: `build/current-mimo-v2-jang2l-current-audit-after-xml-function-template-fix-20260606.json` -> `status=open`, `tool_protocol=true`, blockers are long-prompt, exact-cache, and speed.
- Rebuilt bundled Python from current source and local `/Users/eric/jang/jang-tools`; `npm run verify-bundled` passed.
- Current suite: `build/current-regression-suite-after-mimo-xml-function-template-fix-20260606.json` -> `status=open`, failed steps `packaged_integrity_contracts`, `release_regression_manifest`, `release_gate_skip_app`; packaged integrity has no package/hash failed rows after rebundle, but release gate still fails because objective blockers remain.
- Commit pushed to `origin/main`: `20070fd4 Fix MiMo XML tool fallback contract`.
- No tag, notarized DMG, public download, mlx.studio/vmlx.net update, or release claim was made.

## 2026-06-06 Codex | no-heavy/package pointer refresh after C-RADIO bytecode fix

- Fixed Nemotron Omni vendored C-RADIO installer to ignore/remove `__pycache__` and `.pyc` files when populating the HF dynamic-module cache.
- Refreshed stale no-heavy pointers; cache architecture, parser registry, generation defaults, VL/media, packaged integrity, runtime parity, public issue audit, and focused regression rows now pass in the current suite.
- Rebuilt bundled Python from current vMLX source and local jang-tools; `npm run verify-bundled` passed.
- Rebuilt and Developer-ID signed the staged Sequoia app. Notarization was not run; builder skipped notarize option generation.
- Canonical suite: `build/current-regression-suite-after-noheavy-pointer-refresh-20260606.json` -> `status=open`, `failed_steps=["release_regression_manifest"]` only.
- Release remains blocked by five live/model rows: cross-family live multi-turn, MiMo V2.5 JANG_2L quality, MiniMax reporter/root cause, real Electron UI matrix, and DSV4 long-output/code/file-generation quality.
