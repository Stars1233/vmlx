# STATUS - 2026-06-06 post-1.5.56 runtime/debug continuation

Active worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

Latest continuation:
- Source commit `76e22794` (`Classify unwired media runtime errors`) was pushed to `origin/main`.
- Source commit `fe61db17` (`Refresh package proof pointers`) was pushed to `origin/main`.
- Bundled Python was rebuilt from current source; `npm run verify-bundled` passed with `vmlx_engine` and `jang_tools` parity.
- Staged Sequoia app was rebuilt and Developer ID signed after keychain repair.
- Packaged integrity now passes with `build/current-packaged-integrity-contract-after-unsupported-media-staged-app-20260606.json`.
- Gemma4 26B CRACK installed-app Responses visible-content and mixed-SWA speed rows are now closed in `build/current-objective-proof-after-gemma26-installed-speed-visible-20260606.json`.
- Packaged integrity refresh after that pointer update passes with `build/current-packaged-integrity-contract-after-gemma26-installed-speed-visible-20260606.json`.
- Current suite now records source/package/parity/contracts as passing and remains open only because release-manifest refuses five live/model rows: cross-family live multi-turn, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live matrix, and DSV4 long-output/code/file-generation quality. Artifact: `build/current-regression-suite-after-gemma26-installed-speed-visible-20260606.json`.
- Pointer defaults were refreshed so release gate, packaged integrity, current suite, public issue audit, and manifest tests use current 2026-06-06 artifacts instead of stale 2026-06-04 objective digests.
- Full public release is still blocked. No release tag, notarized DMG, mlx.studio/vmlx.net publication, or installed-app replacement has been performed from this continuation.

Current pushed source state:
- `origin/main` / `origin/HEAD`: `dc3e7205` (`Fix MiMo MLLM language model adapter`).
- Post-1.5.56 source/main includes MiMo JANG_2L load/MLLM adapter fixes and current Qwen35 dense MTP `gdn_sink` signature coverage.
- No post-1.5.56 release tag, DMG rebuild, notarization, or public release has been produced from `dc3e7205`.
- Local `/Applications/vMLX.app` reports `1.5.56`; this is not evidence that post-1.5.56 main commits are publicly released.

Fresh focused checks from this continuation:
- Qwen35 dense MTP source contract: `.venv/bin/python -m pytest -q tests/test_engine_audit.py -k qwen35_dense_mtp_patch_accepts_gdn_sink_kwarg` -> `1 passed, 503 deselected`.
- Local installed 1.5.56 bundled signature probe: `vmlx_engine/patches/mlx_lm_mtp/qwen35_model.py` contains `gdn_sink` in the patched dense GatedDeltaNet and DecoderLayer signatures.
- Real source-server Qwen35 MXFP8 MTP repro: `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP` loaded as native-MTP VL artifact, native MTP `READY D3`, short exact and 160-token decode both HTTP 200, no `gdn_sink` crash, 160-token row logged `77.4 tok/s`. Artifact: `build/current-qwen35-mxfp8-mtp-gdn-sink-live-qwen3parser-20260606.json`.
- Real source-server Qwen27 MXFP8 MTP repro: `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP` loaded as native-MTP VL artifact, native MTP `READY D3`, short exact and 160-token decode both HTTP 200, no `gdn_sink` crash, 160-token row logged `18.0 tok/s`. Artifact: `build/current-qwen27-mxfp8-mtp-live-20260606.json`. This is no-crash/functionality proof only, not speed clearance.
- Gemma4 12B JANG_4M source speed gate: `build/current-gemma4-12b-speed-gate-20260606-live.json` -> pass, default median `46.695 tok/s` against `45.0`, no top-k primary regression, native cache `gemma4` / `mixed_swa_kv_v1`, speed-row cache hits `0`. Packaged speed parity still open.
- Gemma4 12B JANG_4M installed-app speed gate: `build/current-gemma4-12b-speed-gate-installed-app-20260606-live.json` -> pass, default median `46.665 tok/s` against `45.0`, native cache `gemma4` / `mixed_swa_kv_v1`, speed-row cache hits `0`. Nuance: installed-app `temp1_topk64` median was `41.934 tok/s`; top-k primary-regression threshold did not fire because the median delta versus topk0 stayed below threshold.
- Packaged Qwen35 MXFP8 MTP repro: `build/current-qwen35-mxfp8-mtp-installed-app-live-20260606.json` -> bundled `/Applications/vMLX.app` Python, native MTP `READY D3`, short exact and 160-token decode HTTP 200, no `gdn_sink` crash, 160-token row `77.3 tok/s`.
- Packaged Qwen27 MXFP8 MTP repro: `build/current-qwen27-mxfp8-mtp-installed-app-live-20260606.json` -> bundled `/Applications/vMLX.app` Python, native MTP `READY D3`, short exact and 160-token decode HTTP 200, no `gdn_sink` crash, 160-token row `18.0 tok/s`; speed/equivalence remains open.
- Formal Qwen27 MXFP8 MTP installed decode-speed gate: `build/current-decode-speed-live-qwen27-mxfp8-mtp-installed-app-20260606.json` -> `status=review`, stochastic bundle row `13.14 tok/s` below `25`, greedy/topk0 `27.02 tok/s`, PP below `600` at `541.20`, `561.67`, `482.14`. Runtime health shows native MTP, affine quantized matmul, M5 Max Metal N/A, hybrid SSM cache, and q4 attention KV storage active.
- Formal Qwen27 JANG_4M MTP installed decode-speed gate: `build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-20260606.json` -> `status=review`, bundle decode `29.71 tok/s`, greedy/topk0 `43.28 tok/s`, one PP row low at `574.34`.
- Qwen27 MXFP8 MTP deterministic-policy gate: `build/current-decode-speed-live-qwen27-mxfp8-mtp-installed-app-deterministic-policy-20260606.json` -> decode clears floor (`25.80 tok/s`) when launched with `--native-mtp-sampling-policy deterministic-defaults`, but PP remains low (`516.32`, `545.62`, `492.82`). Classification: stochastic model-owned defaults/direct CLI compatible-only are slow; app/session deterministic policy is wired; PP still needs work.
- Native-MTP no-heavy policy/UI contract: `build/current-native-mtp-contract-qwen27-speed-boundary-20260606.json` -> pass, engine contracts `123`, panel controls `16`, panel detection `8`.
- Gemma4 12B JANG_4M installed-app VLM small-image recovery: `build/current-gemma4-12b-installed-image-prefill-recovery-20260606.json` -> image request HTTP 200 with `Red color`, next text request HTTP 200 with `text recovery ok`, native Gemma4 mixed SWA KV plus prefix/paged/block-disk cache and q8 attention KV active.
- Gemma4 12B JANG_4M installed-app VLM forced prefill rejection recovery: `build/current-gemma4-12b-installed-image-prefill-forced-reject-recovery-20260606.json` -> deliberate tiny `VMLINUX_VLM_IMAGE_PREFILL_BUFFER_GB=0.000001` long image prompt rejected with HTTP 413, next text request HTTP 200 with `text recovery after reject ok`. This proves controlled rejection/post-error recovery for this route, not full VL/video/audio quality.
- No-heavy VL/media cache contract refreshed: `build/current-vl-media-cache-contract-after-gemma4-vlm-recovery-20260606.json` -> pass; engine `45 passed/5 skipped`, panel follow-up `13 passed`, panel VLM settings `12 passed`, panel family detection `14 passed`.
- No-heavy API/cache contract refreshed: `build/current-noheavy-api-cache-contract-after-gemma4-vlm-recovery-20260606.json` -> pass; API route `35 passed`, scheduler cache `8 passed`, TQ/MLLM cache `32 passed`, DSV4 DSML tools `21 passed`, Responses history `3 passed`.
- Cache architecture contract refreshed: `build/current-cache-architecture-contract-after-gemma4-vlm-recovery-20260606.json` -> pass; cache-family pytest `401 passed`, panel cache launch policy `102 passed`.
- Step3.7 crash-class falsification contract refreshed: `build/current-step37-crash-falsification-contract-after-gemma4-vlm-recovery-20260606.json` -> pass.
- Admin sleep probe refreshed: `build/current-issue175-admin-sleep-probe-after-gemma4-vlm-recovery-20260606.json` -> pass.
- MiMo local artifact refresh: stale `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` 106G copy was deleted, then the promoted max2 bundle was copied from `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` using TB-routed HTTP endpoint `169.254.232.233:8765`. Manifest proof `build/current-mimo-jang2l-local-manifest-verify-20260606.json` -> pass, 173 files, 113,926,313,468 bytes, no missing/mismatched files.
- MiMo no-cache/simple live text proof: `build/current-mimo-jang2l-live-text-smoke-20260606.json` -> pass, loaded as MLLM, returned exact `mimo ok`, but speed was poor (`4` completion tokens in `27.909s`, logged `0.1 tok/s`). Functionality proof only.
- MiMo cache-stack repro before fix: `build/current-mimo-jang2l-live-cache-stack-smoke-20260606.json` -> review/fail. First request returned `cache ok`; second identical request hit paged cache and failed HTTP 500 with KV shape mismatch `(1,4,25,192)` vs `(1,8,5,192)`. Root cause was runtime cache extraction slicing SWA rotating layers from legal 8 KV heads down to primary 4 heads.
- MiMo cache-stack after fix: `build/current-mimo-jang2l-live-cache-stack-after-head-fix-20260606.json` -> pass. Continuous batching + paged prefix cache + block-disk L2 + q8 KV; first request returned `cache ok`, second request returned `cache ok` with `cached_tokens=25`, `cache_detail=paged`, HTTP 200.
- MiMo tool/cache proof: `build/current-mimo-jang2l-live-tool-cache-smoke-20260606.json` -> review. Warmup exact text and paged cache hit worked, but subsequent tool requests were rejected by Metal working-set guard at 98.6% occupancy. This did not test model tool generation.
- MiMo tool-only proof: `build/current-mimo-jang2l-live-tool-only-smoke-20260606.json` -> review. Simple MLLM mode with `--enable-auto-tool-choice`, `xml_function`, and `think_xml`; forced `lookup_city_code` request returned HTTP 200 but emitted malformed raw `<tool_call>` plus repeated punctuation/fullwidth-comma garbage until length, with zero parsed tool calls. Classification: MiMo tool behavior/model-output quality blocker, not cache crash and not parser fabrication.
- Post-fix cache architecture contract: `build/current-cache-architecture-contract-after-mimo-head-fix-20260606.json` -> pass, cache-family pytest `401 passed`, panel cache launch policy `102 passed`.
- API/tool/reasoning/gateway contracts after MiMo tool blocker:
  - `build/current-tool-call-contract-after-mimo-tool-blocker-20260606.json` -> pass.
  - `build/current-api-surface-contract-after-mimo-tool-blocker-20260606.json` -> pass.
  - `build/current-reasoning-template-contract-after-mimo-tool-blocker-20260606.json` -> pass.
  - `build/current-panel-tool-security-contract-after-mimo-tool-blocker-20260606.json` -> pass.
  - `build/current-mcp-policy-contract-after-mimo-tool-blocker-20260606.json` -> pass.
  - `build/current-release-surface-contract-after-mimo-tool-blocker-20260606.json` -> pass.
  - Direct cancellation pytest selected 6 tests and passed.
  - `build/current-streaming-detokenizer-pytest-after-mimo-tool-blocker-20260606.json` -> skipped, 13 streaming detokenizer tests skipped; streaming proof remains open.
- Source-level streaming API proof: `build/current-gemma4-12b-live-streaming-api-proof-20260606.json` -> pass. Gemma4 12B JANG_4M on current source with continuous batching, paged prefix cache, block-disk L2, q8 KV; `/v1/chat/completions` streaming returned HTTP 200, three SSE chunks, visible `stream ok`, and final `data: [DONE]`. Installed-app streaming parity remains separate/open.
- Installed-app streaming API proof: `build/current-gemma4-12b-installed-app-streaming-api-proof-20260606.json` -> pass. `/Applications/vMLX.app` bundled runtime with Gemma4 12B JANG_4M, continuous batching, paged prefix cache, block-disk L2, q8 KV; streaming returned HTTP 200, visible `installed stream ok`, and final `data: [DONE]`. This does not prove the post-1.5.56 MiMo cache fix is packaged.
- Step3.7 MLLM language-model output contract passed: `tests/test_step37_mlx_vlm_runtime.py::test_step37_language_model_returns_logits_object_for_mlx_vlm_generate`.
- Zaya MLLM `inputs_embeds` contract passed: `tests/test_zaya_runtime.py::test_zaya1_vl_language_model_accepts_mlx_vlm_inputs_embeds_keyword`.
- Media implementation ledger refreshed: `docs/internal/VL_AUDIO_VIDEO_RUNTIME_WORKLIST_2026_06_06.md` now itemizes concrete missing function areas for Gemma4, Qwen VL/video, MiMo, Step3.7, Nemotron Omni, ZAYA-VL, MiniMax/Hy3/Kimi, and structured-output repair.
- Current source inspection: Gemma4 image-prefill budget is already dynamic via `_resolve_vlm_image_prefill_single_buffer_limit()` and no longer a fixed 8GB default in this worktree. A fixed 8GB report on a 128GB machine should be checked as stale app, explicit env override, or packaged-runtime drift.
- Current source inspection/fix: MiMo media is not implemented by the vMLX adapter. `/Users/eric/jang/jang-tools/jang_tools/mimo_v2/mlx_model.py` is explicitly text-only; vMLX now raises typed `UnsupportedMediaModalityError` / API code `unsupported_media_modality` on unwired `pixel_values` instead of a generic generation failure. MiMo VL/audio/video still needs a real JANG tools multimodal forward module before vMLX can truthfully enable it.
- Focused validation for typed unsupported media: py_compile passed for edited runtime/tests; pytest selected rows passed (`3 passed, 594 deselected`).

Current blocker classification:
- Qwen user traceback `unexpected keyword argument 'gdn_sink'`: not explained by current source or local installed 1.5.56 signature. If reported on 1.5.55/older app, classify as stale packaged runtime. If reproduced on fresh 1.5.56+, treat as a separate live route bypassing the fixed dense patch and reproduce with actual Qwen35 MXFP8 MTP.
- MiMo is back in scope. Source short MLLM text no longer 500s, but long-output/tool quality remains red and must not be release-cleared.
- MiMo VL/audio/video is unbuilt, not merely untested. Classification is `runtime_dispatch` for missing media forward plus `decode_loop`/`model_artifact` for text/tool/speed quality until source-vs-quant proof separates them.
- JANGQ/JANG tools MiMo V2 support was pushed separately to `jjang-ai/jangq` main at `d1316c3a1bcd7f4eeef899ffbc1ced729cfa0780`; local `/Users/eric/jang` main is now aligned to `d1316c3`. The old local equivalent head is preserved on `codex-backup/local-main-before-origin-sync-20260605214651`, and the three overlapping untracked files are backed up at `/Users/eric/.codex/tmp/jang-untracked-overlap-backup-20260605214620/`.
- Full release remains blocked by live packaged model-family matrix, Qwen35/27 MTP live proof, MiMo long/tool/cache proof, DSV4 restart-L2/exact-code rows, Gemma4 audio/video rows, and UI/settings/cache release parity.

Primary systematic tracker: `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md`.

# STATUS - 2026-06-04 Gemma4 12B release boundary

Active worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

Latest gate refresh after LFM/Step manifest proof fix:
- Focused release regression pytest passed: `555 passed, 202 deselected`.
- Aggregate current regression suite passed under expected-open policy:
  - `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[].
  - Open requirements: real Electron UI cross-family live model matrix and DSV4 long-output/code/file-generation quality.
- Release manifest regenerated:
  - `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> current_proof_sweep=fail, prepackage_ready=false, release_ready=false.
  - Failed component is now only `real_ui_live_model_proof`; release blocker ledger is pass.
  - Real UI matrix: LFM2.5 pass, Step3.7 pass, DSV4 missing/partial because the live UI proof is memory-gated.
- DSV4 memory preflights refreshed:
  - `build/current-real-ui-dsv4-memory-preflight-after-lfm-step-manifest-fix-20260604.json` -> skipped_insufficient_memory, 71.69 GB free+speculative+purgeable vs 120.0 GB required.
  - `build/current-dsv4-route-mode-code-exactness-memory-preflight-after-lfm-step-manifest-fix-20260604.json` -> skipped by memory preflight.
- DSV4 continuation host readiness:
  - `build/current-real-ui-dsv4-memory-preflight-continuation-refresh-20260604.json` -> skipped_insufficient_memory, 71.42 GB free+speculative+purgeable vs 120.0 GB required.
  - `build/current-dsv4-route-mode-code-exactness-memory-preflight-continuation-refresh-20260604.json` -> skipped, 71.41 GB free+speculative+purgeable / 111.65 GB psutil available vs 120.0 GB required.
  - `build/current-remote-max2-dsv4-readiness-continuation-20260604.json` -> max2 is not currently usable for DSV4 release proof: remote repo is stale at vMLX 1.5.32, free+speculative+purgeable is ~44.4 GB vs 120.0 GB required, and Parallels Windows 11 is using ~30.8 GB RSS. No DSV4 launch and no user-process kill was performed.
- max2 stale-repo blocker partially removed for future runs:
  - Created `/Users/eric/mlx/vllm-mlx-codex-proof-1554` on max2 from `origin/codex/pr-intake-manifest` at `6f8fff2a` / vMLX 1.5.54 and overlaid the current local dirty source/test changes.
  - Verified `/Users/eric/mlx/vllm-mlx/.venv/bin/python` imports `vmlx_engine` from `/Users/eric/mlx/vllm-mlx-codex-proof-1554/vmlx_engine/__init__.py` with `__version__ == 1.5.54`.
  - Remote current-source DSV4 preflights still block on memory:
    - `build/current-real-ui-dsv4-memory-preflight-max2-proof-worktree-20260604.json` -> skipped_insufficient_memory, 40.6 GB free+speculative+purgeable vs 120.0 GB required.
    - `build/current-dsv4-route-mode-code-exactness-memory-preflight-max2-proof-worktree-20260604.json` -> skipped, 40.58 GB free+speculative+purgeable / 76.78 GB psutil available vs 120.0 GB required.
  - Local summary: `build/current-remote-max2-dsv4-proof-worktree-readiness-20260604.json` -> prepared_but_memory_blocked.
- Reusable max2 DSV4 readiness runner added:
  - Script: `tests/cross_matrix/run_remote_max2_dsv4_readiness.py`.
  - Test: `tests/test_remote_max2_dsv4_readiness.py` -> `3 passed`.
  - Runner command: `.venv/bin/python tests/cross_matrix/run_remote_max2_dsv4_readiness.py --out build/current-remote-max2-dsv4-proof-worktree-readiness-runner-20260604.json`.
  - Result: `build/current-remote-max2-dsv4-proof-worktree-readiness-runner-20260604.json` -> prepared_but_blocked, launch_decision=do_not_launch, 40.38 GB free+speculative+purgeable vs 120.0 GB required.
  - Kept out of the default local current suite because it depends on SSH/max2 user state; use explicitly before future DSV4 heavy proof attempts.
- Guarded max2 DSV4 exactness launcher added:
  - Script: `tests/cross_matrix/run_remote_max2_dsv4_exactness_guard.py`.
  - Test: `tests/test_remote_max2_dsv4_exactness_guard.py`; combined exactness/readiness tests -> `6 passed`.
  - Dry-run artifact: `build/current-remote-max2-dsv4-exactness-guard-dryrun-20260604.json` -> status=dry_run, launch_decision=dry_run_no_launch.
  - Non-dry guarded artifact: `build/current-remote-max2-dsv4-exactness-guard-skip-20260604.json` -> status=skipped_not_ready, launch_decision=do_not_launch.
  - This is now the safe entrypoint for max2 DSV4 exactness: it refuses to launch unless current source/import/model/memory gates pass.
- Guarded max2 DSV4 real-UI launcher added:
  - Script: `tests/cross_matrix/run_remote_max2_dsv4_real_ui_guard.py`.
  - Test: `tests/test_remote_max2_dsv4_real_ui_guard.py`; combined real-UI/exactness/readiness tests -> `9 passed`.
  - Dry-run artifact: `build/current-remote-max2-dsv4-real-ui-guard-dryrun-20260604.json` -> status=dry_run, launch_decision=dry_run_no_launch.
  - Non-dry guarded artifact: `build/current-remote-max2-dsv4-real-ui-guard-skip-20260604.json` -> status=skipped_not_ready, launch_decision=do_not_launch.
  - Current max2 real-UI blockers: insufficient_memory (39.63 GB strict vs 120.0 GB), node_missing, staged_app_missing. The proof script exists in the proof worktree.
  - This is now the safe entrypoint for max2 DSV4 real-UI proof: it refuses to launch unless current source/import/model/memory, Node, proof script, and staged app gates pass.
- max2 DSV4 real-UI prerequisites reduced to memory only:
  - Found Node at `/opt/homebrew/bin/node` (`v25.9.0`) and updated the real-UI guard to use/check that explicit path instead of SSH PATH.
  - Copied staged Sequoia app to `/Users/eric/mlx/vllm-mlx-codex-proof-1554/panel/release/sequoia-app/mac-arm64/vMLX.app`.
  - Verification: combined real-UI/exactness/readiness tests still `9 passed`.
  - `build/current-remote-max2-dsv4-real-ui-guard-after-node-app-20260604.json` -> skipped_not_ready, launch_decision=do_not_launch, blockers only `insufficient_memory`; strict memory 39.32 GB vs 120.0 GB required. Node, proof script, and staged app are present.
- max2 guard tests are now part of current-suite coverage:
  - `tests/cross_matrix/run_current_regression_suite.py` tracks the remote max2 DSV4 readiness/exactness/real-UI guard scripts in source hashes and includes their unit tests in the focused pytest selection.
  - Targeted guard/suite-contract slice -> `12 passed, 56 deselected`.
  - Expanded focused release regression selection -> `564 passed, 202 deselected`.
  - Current suite regenerated: `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[].
  - Release manifest regenerated: `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> current_proof_sweep=fail, prepackage_ready=false, release_ready=false because DSV4 remains live-uncleared.
- aggregate max2 DSV4 release-proof guard is now the safe top-level entrypoint:
  - Script: `tests/cross_matrix/run_remote_max2_dsv4_release_proof_guard.py`.
  - Test: `tests/test_remote_max2_dsv4_release_proof_guard.py`.
  - Aggregate plus child guard tests -> `12 passed`.
  - Guard command wrote `build/current-remote-max2-dsv4-release-proof-guard-20260604.json` -> status=blocked_no_launch, release_ready=false.
  - Child states: readiness=prepared_but_blocked, exactness=skipped_not_ready, real_ui=skipped_not_ready.
  - Current max2 blocker is only DSV4 memory for launch/proof: strict memory 39.14 GB vs 120.0 GB required; Node v25.9.0, proof script, staged app, current source import, and 80G DSV4 model are present.
  - Expanded focused release regression after aggregate coverage -> `567 passed, 202 deselected`.
- post-aggregate gate refresh:
  - Guard tests rerun -> `12 passed`.
  - `.venv/bin/python tests/cross_matrix/run_current_regression_suite.py` regenerated `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> status=pass, failed_steps=[].
  - Open requirements remain exactly: real Electron UI cross-family live model matrix and DSV4 long-output/code/file-generation quality.
  - `.venv/bin/python tests/cross_matrix/run_release_regression_manifest.py --require-current-proof-sweep` regenerated `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` and exited 1 by design: current_proof_sweep=fail, prepackage_ready=false, release_ready=false.
  - Current manifest failure is still real_ui_live_model_proof; local DSV4 preflight refreshed to 77.67 GB strict memory vs 120.0 GB required, launch_decision=do_not_launch.
- Main integration push completed:
  - Commit: `9b1ffe72 integrate Gemma4 12B runtime gates`.
  - Remote verification: `origin/main` resolves to `9b1ffe7296526df740e00774403c74ef7c3727d1`.
  - Local branch still tracks `origin/codex/pr-intake-manifest`, so it shows ahead of that branch by 1; the intended main push is done.
  - No tag, release upload, package rebuild, fresh notarization, or mlxstudio release was performed after this commit because the release manifest remains `release_ready=false`.
- post-main-push DSV4 readiness refresh:
  - Local real-UI preflight: `build/current-real-ui-dsv4-memory-preflight-after-main-push-20260604.json` -> skipped_insufficient_memory, launch_decision=do_not_launch, 77.43 GB strict memory vs 120.0 GB required.
  - Local exactness preflight: `build/current-dsv4-route-mode-code-exactness-memory-preflight-after-main-push-20260604.json` -> skipped, launch_decision=do_not_launch, 77.40 GB strict memory vs 120.0 GB required, 14 selected cases not run.
  - Max2 aggregate guard: `build/current-remote-max2-dsv4-release-proof-guard-after-main-push-20260604.json` -> blocked_no_launch, release_ready=false.
  - Max2 current source/model checks still pass (`vMLX 1.5.54`, DSV4 model present), but strict memory is 38.34 GB vs 120.0 GB required; exactness and real-UI children skipped_not_ready.
  - Current status: no safe DSV4 launch path remains on local or max2 until memory is freed / a suitable host is provided.

Gemma4 12B default release surface after final DMG rebuild:
- MXFP4/MXFP8/JANG_4M artifact contract passed.
- MXFP4/MXFP8/JANG_4M vision passed.
- MXFP4/MXFP8/JANG_4M runtime passed across conservative, prefix+paged+L2, and prefix+paged+TurboQuant q8 modes.
- Default audio capability is honest: MXFP4/MXFP8 do not advertise audio; JANG_4M advertises audio and passed speech fixture transcription.
- Generation defaults, API surface/cache endpoint, native-MTP, and VL-media contracts pass with refreshed current artifacts.
- Fresh DMGs were rebuilt after the Gemma4 fixes, signed, notarized, stapled, and verified:
  - Sequoia SHA256 `63a62e6b7b8b2dca18ac59f3ea34f0b3a6833dea603372372082e045715ea123`.
  - Tahoe SHA256 `41dfa70e7d324ca796742b3a34866b016d409e58f1ae4fc3b9ee9a5003973c4b`.
- Packaged integrity, installed-app parity, public issue audit, and current regression suite pass after regenerating final artifacts.
- Post remote-note reconciliation artifacts:
  - `build/current-gemma4-12b-artifact-contract-after-remote-note-reconcile-20260604.json` -> pass.
  - `build/current-regression-suite-gemma4-release-boundary-after-remote-note-reconcile-20260604.json` -> pass, failed_steps=[].
  - `build/current-release-regression-manifest-gemma4-release-boundary-after-remote-note-reconcile-20260604.json` -> current_proof_sweep=fail, prepackage_ready=true, release_ready=false, only regression_suite policy failure.
- Continuation gate refresh artifacts:
  - `build/current-regression-suite-after-gemma31-step-lfm-continuation-20260604.json` -> pass, failed_steps=[].
  - `build/current-release-regression-manifest-after-gemma31-step-lfm-continuation-20260604.json` -> current_proof_sweep=fail, prepackage_ready=true, release_ready=false.
  - `build/current-release-regression-manifest-after-installed-public-refresh-20260604.json` -> current_proof_sweep=fail, prepackage_ready=true, release_ready=false; failed_components only `regression_suite`.

Still not releasable as a full app/public release:
- Uniform MXFP4/MXFP8 audio remains red/gated.
- `MXFP8_ATTNFP16` and `MXFP8_ATTNFP16_L0_24_FP16` are experimental and not release-stable.
- Release manifest is still `release_ready=false` because real Electron UI cross-family and DSV4 long-output/code/file-generation quality rows remain open.
- LFM2.5 JANG_2L has a fresh installed Electron UI/settings proof:
  - `docs/internal/agent-notes/current-real-ui-live-model-lfm25-jang2l-continuation-cache-settings-20260604-proof.json` -> pass.
  - Covered max output tokens 64, Prefix Cache, Paged KV Cache, Block Disk Cache (L2), hybrid SSM native cache status, cache-hit telemetry, L2 disk counters, settings persistence, and coherent live output.
  - `docs/internal/agent-notes/current-real-ui-live-model-lfm25-jang2l-continuation-responses-tools-cache-settings-default-max512-20260604-proof.json` -> pass.
  - Covered installed UI Responses API/delta streaming, two-turn `run_command` long tool loop, tool L2 cache integration, `paged+ssm` cache hits, block disk L2, SSM companion L2, max output tokens 512, and parser/language leak checks.
- Production-family static refreshes:
  - `build/current-production-family-audit-static-gemma4-26b-31b-20260604.json` -> pass for `gemma4_crack` and `gemma4_31b_jang4m_mtp`.
  - `build/current-step37-vlm-runtime-audit-20260604-continuation.json` -> pass / audit does not block release.
  - `docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-continuation-cache-settings-20260604-proof.json` -> installed UI/settings/cache proof pass for Step3.7 JANG_2L, including `paged+mixed_swa` cache-hit telemetry and L2 block writes.
  - `build/current-production-family-audit-static-dsv4-local-20260604.json` -> DSV4 rows exist, but both still expose the output-head/final-norm precision boundary and remain release blockers for long-output/code/file-generation claims.
  - `build/current-real-ui-dsv4-memory-preflight-after-step-lfm-continuation-20260604.json` -> skipped_insufficient_memory, 70.54 GB available vs 120.0 GB required; do not launch live DSV4 UI proof on this host until this clears.
- No commit, push, tag, release upload, or main promotion has been done.
- MiMo is out of the current list; the accidental current-scope MiMo host-availability helper/test were removed.

Primary note: `docs/internal/agent-notes/current-gemma4-12b-release-boundary-and-package-gates-20260604.json`.

## CODEX

- now: active MiMo recovery lane after Eric's correction; stale local MiMo JANG_2L was removed, Max2 canonical `MiMo-V2.5-JANG_2L` copy is in progress from `erics-m5-max2.local` over direct `en9` route using rsync.
- current transfer: source `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` on Max2 -> local same path; 176 items / 150 shards; observed 140-160 MB/s.
- already confirmed from Max2 docs: MiMo Python/JANG_2L nuance is asymmetric full/SWA KV heads, partial RoPE, SWA sinks, V scale, sigmoid+bias top-k, XML tools, think XML; Swift TP4 proof reached ~39 tok/s with cache/L2/API proof, but Python local runtime still needs live proof.
- current blocker: copy must finish before local live serve/smoke/root-cause can prove artifact-vs-runtime.
- do not claim release-ready: DSV4 restart-L2 and live model/UI rows remain open; no DMG/notarized/public release has been made.


## CODEX 2026-06-06 current suite after DSV4 restart-L2
- current suite artifact: `build/current-regression-suite-after-dsv4-restart-l2-fix-20260606.json` -> `status=open`, `failed_steps=["release_regression_manifest"]`.
- source/package/parity/contracts passed, including packaged integrity and staged app runtime parity.
- release remains blocked by 10 live/model objective rows; do not tag/notarize/publish as full release yet.
- MiMo is in scope: structural proof and text/cache same-process proof pass; tool args/continuation, exact Responses behavior, speed, VL/audio/video, and restart-L2 remain uncleared.

## CODEX 2026-06-06 Qwen speed/MTP refresh
- Qwen rows closed in objective digest: packaged MX matmul speed, native MTP decode/equivalence, Qwen27 JANG_4M prompt-processing speed.
- Current suite artifact: `build/current-regression-suite-after-qwen-speed-mtp-refresh-20260606.json` -> `status=open`, `failed_steps=["release_regression_manifest"]`.
- Remaining release blockers: Ling/Bailing multilingual quality, Gemma4 26B CRACK visible-content/language quality, Gemma4 26B CRACK mixed-SWA app speed, cross-family live multi-turn matrix, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, DSV4 long-output/code/file-generation quality.

## CODEX 2026-06-06 cross-family smoke pointer refresh

- Active worktree remains `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; deprecated `/Users/eric/vmlx` was not used for runtime work.
- Refreshed cross-family smoke proof pointers to current 2026-06-06 artifacts, including LFM and Step3.7 as required family keys.
- Generated objective digest: `build/current-objective-proof-after-cross-family-smoke-refresh-20260606.json`.
- Generated release manifest: `build/current-release-regression-manifest-after-cross-family-smoke-refresh-20260606.json` -> current proof sweep fails by design, `prepackage_ready=false`, `release_ready=false`.
- Generated current suite: `build/current-regression-suite-after-cross-family-smoke-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.
- Focused regression pytest in the suite now passes; package/runtime/static gates are not the remaining blocker.
- Current open objective rows: cross-family live multi-turn smoke matrix, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- Current cross-family source coverage is green for Gemma4, Hy3, LFM2.5, Ling/Bailing, MiniMax Small/K source rows, Nemotron Omni, Qwen3.6, and Step3.7 text route. ZAYA text/VL and MiMo remain red; DSV4 remains memory-preflight blocked.
- No tag, notarized DMG, public download update, or release claim was made.
- Next release notes must credit GitHub `@Hornsan1` for reporting many of these runtime/media/cache issues.

## CODEX 2026-06-06 ZAYA cache-contract refresh

- Patched `bench/all_local_model_smoke.py` so ZAYA cache-repeat probes use a family-native exact one-word color contract instead of generic system-role `ACK`; generic families still use exact `ACK`.
- Fresh ZAYA artifact: `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json`.
- ZAYA text is narrowed: text cache exact output, paged `zaya_cca` hit, multi-turn recall, and required tool call pass; reasoning-on remains red with reasoning-only empty visible output.
- ZAYA-VL remains red: text cache exact output, multi-turn recall, reasoning visible output, and red-image recognition fail; blue image and tool call pass; CCA cache hit telemetry exists.
- Current objective digest: `build/current-objective-proof-after-zaya-vl-no-media-refresh-20260606.json`.
- Current suite: `build/current-regression-suite-after-zaya-vl-no-media-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.
- Release remains blocked by cross-family smoke, MiMo, MiniMax reporter parity, real Electron UI matrix, and DSV4 long-output/code quality. No tag/notarized/public release was made.

## CODEX 2026-06-06 ZAYA text reasoning budget closure

- Focused ZAYA text reasoning repro: `build/current-zaya-text-reasoning-budget-repro-20260606/summary.json` proves 256-token thinking-on stops before visible answer, while 512+ reaches visible `FINAL=OK`.
- Patched all-local smoke to give ZAYA reasoning probes at least 512 tokens, with strict visible final-answer validation unchanged.
- Fresh smoke: `build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json` -> ZAYA text pass, ZAYA-VL still red.
- Patched objective digest to evaluate mixed artifacts per family, so ZAYA text can be covered while ZAYA-VL remains a blocker.
- Current digest: `build/current-objective-proof-after-zaya-vl-no-media-refresh-20260606.json` covers `zaya_text`; cross-family smoke still misses `dsv4`, `mimo_v2`, and `zaya_vl`.
- Current suite: `build/current-regression-suite-after-zaya-vl-no-media-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.
- No release tag/notarization/public download update was made.

## CODEX 2026-06-06 ZAYA-VL no-media contract refresh

- Patched all-local smoke so only ZAYA/ZAYA-VL no-media followups use the family-native `ATTACHED/NONE` current-message attachment contract. Generic VLM families keep the stronger system-scoped `answer exactly NONE/IMAGE/VIDEO` contract.
- Fresh MXFP4 proof: `build/current-all-local-model-smoke-zaya-vl-mxfp4-after-no-media-contract-20260606/summary.json` -> one remaining failure: `reasoning_on` empty visible output with `reasoning_chars=17`.
- Prior no-media carryover is falsified for MXFP4 by `build/current-zaya-vl-mxfp4-focused-repro-20260606/summary.json`; direct current-message attachment prompts return `NONE` before and after image turns.
- Current objective digest: `build/current-objective-proof-after-zaya-vl-no-media-refresh-20260606.json`.
- Current suite: `build/current-regression-suite-after-zaya-vl-no-media-refresh-20260606.json` -> `status=open`, failed step only `release_regression_manifest`.
- Remaining blockers: cross-family live multi-turn smoke, MiMo V2.5 JANG_2L runtime/tool/long-prompt, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family matrix, DSV4 long-output/code/file-generation. ZAYA-VL reasoning remains a live smoke blocker inside the cross-family row.
