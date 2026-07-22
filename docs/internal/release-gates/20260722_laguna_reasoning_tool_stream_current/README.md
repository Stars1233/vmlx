# Laguna S-2.1 current-source reasoning/tool streaming checkpoint

Date: 2026-07-21/22 local.  Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13` on `erics-m5-max.local`.  Branch: `codex/postrelease-ui-drawers-20260720`.

## Scope

This gate is narrow: Laguna S-2.1 JANG_2L, current source, Electron-launched server, Chat Completions streaming, reasoning rail separation, tool-call continuation, bundle-derived sampler defaults, and live cache detail.  It is not a full release matrix and does not close packaging/notarization.

## Source changes covered

- `vmlx_engine/engine/batched.py`: renders chat templates through the wrapper/inner tokenizer object that actually owns `apply_chat_template`; this is required for Laguna tokenizer wrappers whose inner tokenizer carries the native Poolside/Laguna template.
- `vmlx_engine/server.py`: Chat/Responses streaming now treats structured engine errors as errors instead of assistant reasoning/content, Chat streaming no longer emits speculative empty `tool_calls` START deltas before a parser-valid function call exists, and resolved `enable_thinking` is mirrored into `chat_template_kwargs` across Chat, Responses, and Anthropic routes so tokenizer/template wrappers see the same effective reasoning state as the top-level engine kwarg.
- `vmlx_engine/engine_core.py`, `vmlx_engine/engine/base.py`, `vmlx_engine/output_collector.py`: preserve structured engine-loop failures as error fields instead of generated text.
- Tests pin the above in `tests/test_streaming_reasoning.py`, `tests/test_batching.py`, and `tests/test_server.py`.

## Live process proof

- `live-processes.txt`: single active `vmlx-engine serve` process for `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L`, PID 93364 in the final post-fix capture; earlier screenshots used PID 90393 before the final no-START rerun.
- `live-process-env.txt`: Electron-managed process had `PYTHONPATH=/Users/eric/mlx/vllm-mlx-release-1.6.13`, `PWD=/Users/eric/mlx/vllm-mlx-release-1.6.13/panel`, and `VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1613-responsive-dev`.

## Bundle/default proof

`bundle-defaults-summary.json` records the real bundle defaults used for this row:

- `generation_config.json`: `temperature=1.0`, `top_p=1.0`, `top_k=20`, `min_p=0.0`, `do_sample=true`, `reasoning_parser=poolside_v1`, `tool_call_parser=poolside_v1`.
- `jang_config.json`: `cache_subtype=kv`; JANG_2L mixed affine quantization with routed experts 2/2/3-bit and attention 8-bit.

The Electron chat settings drawer screenshot/text (`ui/laguna-chat-settings-drawer.*`) showed Top K `20`, matching this bundle. This observation is specific to this Laguna S-2.1 bundle; it does not prove global settings parity for all models.

## API proof

### Pre-fix classifier

`api_pre_fix_compare/summary.json` preserves the failure shape: post-tool continuation produced correct final content, but also emitted a phantom empty streaming `tool_calls` delta before content. This is the harness-facing bug fixed here.

### Post-fix proof

`api_post_fix_no_phantom_final/summary.json` after Electron Save & Restart on current source/PID 93364:

- Round 1 required tool call: `tool_delta_count=1`, `finish_reasons=["tool_calls"]`, assembled call `file_info({"path":"panel/package.json"})`.
- Round 2 tool-result continuation: visible content `LAG-S21-API-TOOL-NOSTART-FINAL-DONE SIZE=5.2 KB`, `tool_delta_count=0`, `finish_reasons=["stop"]`, `done_seen=true`.
- Cache detail on the live API rows included `paged+disk+tq-native` on the first round and `paged+tq-native` on continuation.

`postcommit_0b267_pid93786/api/summary.json` adds a clean-restart proof after
the final source commit `0b267fda724729649ad862aedbc179817b866f4d`:

- Source mtime: `2026-07-21 23:51:57 -0700`; Electron-launched PID `93786`
  started afterward at `2026-07-21 23:52:34 -0700`.
- Post-tool continuation marker `LAG-S21-CHAT-POSTCOMMIT-T` returned
  `LAG-S21-CHAT-POSTCOMMIT-T-DONE VALUE=BETA-84` with
  `tool_delta_count=0`, `finish_reasons=["stop"]`, and no inline
  think/tool marker leakage.
- Required-tool control still preserved valid OpenAI SDK assembly:
  `tool_delta_count=2`, first delta contained id/type plus empty function
  start, second delta contained `lookup_code({"key":"alpha"})`, and terminal
  `finish_reasons=["tool_calls"]`.
- Health on the same process reported q4 TurboQuant storage:
  `stored_prefix_quantization="turboquant-q4"` and
  `auto_policy="mixed_swa_full_attention_kv_storage_tq4"`.

`api_reasoning_prompt_compare/summary.json` documents variable Laguna reasoning behavior with `enable_thinking=true`:

- Simple arithmetic Auto/On-style prompts can emit no reasoning rail and stream content directly.
- A stronger prompt requiring at least five private sentences emitted `reasoning_len=703`, `reasoning_delta_count=248`, then progressive content deltas, no marker leakage, and terminal stop.

Interpretation: for Laguna S-2.1, Auto/On permits reasoning and the parser separates it if emitted. It does not force a non-empty reasoning rail on every easy prompt. That is model behavior, not an inline-reasoning leak, as long as emitted reasoning remains separate and visible content has no raw think/tool markers.

## Electron visual proof

- `ui/laguna-ui-force-rail-j-chat.png` and `ui/laguna-ui-force-rail-j-chat-text.txt`: Electron chat row for `[LAG-S21-UI-FORCE-RAIL-J]` shows `Reasoning 394 chars`, visible answer `888 + 111 = 999` plus marker, and metrics `206 tokens`, `38.1 t/s`, `1178 prompt (1050 paged+disk+tq-native cached)`, `0.76s TTFT`, `6.2s total`.
- `selected-ui-db-rows.json`: persisted UI rows include row 156 with non-empty `reasoning_content`, clean visible content, and cache detail `paged+disk+tq-native`; row 138 shows tool execution + continuation with final `LAG-S21-UI-REASON-TOOL-D-DONE SIZE=5.2 KB`.

## Cache proof level

`health-after-final-api.json` after the final UI/API pass showed:

- scheduler cache hits/misses present, `tokens_saved=327`, `backend_mode=paged`, `disk_hits=2`, `disk_promotion_hits=2`.
- block disk cache had `blocks_on_disk=1763`, `l2_block_tokens_on_disk=106025`, `disk_writes=4`, `tq_native_writes=4`, `tq_native_hits=2`, `tq_native_enabled=true`.

This proves current live Laguna reuse through paged + disk + TQ-native on this session. It does not prove disk-only-with-paged-off partial prefix restore; that remains a separate open gate.

## Tests

`focused-tests.log`: 16 passed:

- Laguna/wrapped-template and reasoning seed tests.
- structured engine error tests for batching/output collector/Chat/Responses.
- Chat post-tool phantom `tool_calls` regression.
- Responses tool-call argument buffering regression.

`release_bundle_refresh/reasoning-template-kwargs-tests.log`: 141 passed:

- `tests/test_streaming_reasoning.py::TestEnableThinkingTriState`
- `tests/test_reasoning_modes.py`
- `tests/test_reasoning_tool_interaction.py`

## Bundled Python / release-prep proof

After the server reasoning/template-kwargs change, `panel/scripts/bundle-python.sh`
was rerun from this checkout using the clean JANG `origin/main` source at
`/Users/eric/jang` commit `801209c13c189ebb8fb4d1596748a336f568da38`. The
script installed local `vmlx` 1.6.14 and local `jang` 2.5.31 into the
untracked release artifact directory `panel/bundled-python/`.

`release_bundle_refresh/verify-bundled-python.log`: `panel/scripts/verify-bundled-python.sh` passed after the refresh:

- bundled `vmlx_engine` version matches `panel/package.json` (`1.6.14`);
- bundled critical `vmlx_engine` files match current source hashes;
- bundled critical `jang_tools` files match the clean JANG source checkout;
- critical MLX, MLX-VLM, audio, JANG, DSV4/Kimi/Step/Gemma runtime imports pass.

`release_bundle_refresh/targeted-release-failures-rerun.log`: 28 previously
failing release/audit tests passed after running with
`PATH=/Users/eric/.local/node/bin:$PATH` and the refreshed bundle.

`release_bundle_refresh/full-python-pytest-with-nodepath.log`: the complete
Python suite passed with the corrected node PATH after the 1.6.14 bundle
refresh and before the subsequent 1.6.15 version bump:

```text
6290 passed, 96 skipped, 92 deselected, 2 warnings in 268.94s
```

`release_bundle_refresh/verify-bundled-python-1.6.15.log`: after bumping
`panel/package.json`, `panel/package-lock.json`, `pyproject.toml`, and
`vmlx_engine/__init__.py` to `1.6.15`, `panel/scripts/bundle-python.sh` was
rerun and `panel/scripts/verify-bundled-python.sh` passed with bundled
`vmlx_engine 1.6.15`.

`release_bundle_refresh/version-bump-release-tests.log`: 73 release/version
tests passed after the 1.6.15 bump and rebundling:

```text
tests/test_release_gate_python_app.py
tests/test_vl_video_regression.py::TestBundledPythonVerifyScript::test_verify_script_passes_against_current_bundle
tests/test_installed_app_runtime_parity_audit.py
tests/test_public_app_issue_audit.py

73 passed in 81.07s
```

The earlier unqualified full Python-suite invocation captured in
`/tmp/vmlx-full-pytest-20260722-235509.log` failed 12 tests because `node`/`npx`
were absent from PATH, then the bundled verifier exposed stale bundled Python
content. The node/npx portion was environment setup noise, but the stale bundle
was a real release blocker until `bundle-python.sh` was rerun.

## Remaining release blockers not closed by this gate

- Current PID 4279 addendum (2026-07-22 00:07-00:18 local): after the
  `enable_thinking` template-kwargs mirror patch, the real Electron Start
  button launched PID 4279 with `PYTHONPATH` pointed at this checkout. The
  process argv kept `--reasoning-parser deepseek_r1`, `--tool-call-parser
  glm47`, paged cache, block-disk L2, and 15% cache RAM. Health artifact:
  `current_pid4279_after_template_mirror/health_pid4279_after_template_mirror.json`.
- Source/test trace for this addendum: `vmlx_engine/server.py` mirrors resolved
  thinking into `chat_template_kwargs` at the Anthropic, Chat Completions,
  Responses, streaming Chat, and streaming Responses handoff sites. Focused
  validation rerun on this source passed:
  `tests/test_streaming_reasoning.py -k "laguna_forwards_reasoning_on_to_engine
  or stamped_think_template_seeds_without_renderer or
  stream_chat_forwards_effective_thinking_to_engine_kwargs or
  stream_responses_forwards_effective_thinking_to_engine_kwargs"` (4/4) and
  `tests/test_server.py -k "post_tool_false_marker or
  tool_call_arguments_survive_buffering or reasoning_tool_call_keeps_arguments
  or required_empty_xml_tool_call_is_rejected or
  streaming_chat_minimax_truncated_namespace_emits_only_tool_call"` (5/5).
- Raw Chat API proof on PID 4279:
  `current_pid4279_after_template_mirror/reasoning_rail_model_default_summary.json`
  shows 120 `reasoning_content` deltas followed by 30 content deltas for
  `[LAG-S21-API-THINK-RAIL-V]`, with no inline think/private marker leakage
  and terminal `stop`. Required-tool proof:
  `chat_tool_w_after_template_mirror_summary.json` shows round 1 emitted
  exactly one `file_info({"path":"panel/package.json"})` call and round 2
  streamed `LAG-S21-CHAT-TOOL-W-DONE SIZE=5.2 KB` with zero phantom tool
  deltas.
- Raw Responses API proof on PID 4279:
  `responses_think_rail_v_model_default.raw.jsonl` and the same summary show
  448 `response.reasoning_summary_text.delta` events followed by 31
  `response.output_text.delta` events for the reasoning row, with no inline
  marker leakage and `response.completed`. Required-tool proof:
  `responses_tool_x_after_template_mirror_summary.json` shows a completed
  `function_call` item for `file_info({"path":"panel/package.json"})`, then a
  `function_call_output` continuation streaming
  `LAG-S21-RESP-TOOL-X-DONE SIZE=5.2 KB` with no repeated call.
- Current PID 4279 Anthropic/Ollama streaming supplement (2026-07-22 local):
  `anthropic_tool_ab_summary.json` shows `/v1/messages` emitted exactly one
  `file_info({"path":"panel/package.json"})` `tool_use` with
  `stop_reason=tool_use`, then the real `tool_result` continuation returned
  exact visible `LAG-S21-ANTH-TOOL-AB-DONE SIZE=5.2 KB` with
  `stop_reason=end_turn`, no inline think/tool marker leakage, and no visible
  content before the tool. `ollama_tool_ac_summary.json` shows `/api/chat`
  emitted exactly one `file_info({"path":"panel/package.json"})` tool call
  with `done_reason=tool_calls`, then the real tool-result continuation
  returned exact visible `LAG-S21-OLLAMA-TOOL-AC-DONE SIZE=5.2 KB` with
  `done_reason=stop`, `done=true`, zero repeated tool calls, and no inline
  think/tool marker leakage. Raw event artifacts:
  `anthropic_tool_ab_round1.events.jsonl`,
  `anthropic_tool_ab_round2.events.jsonl`, `ollama_tool_ac_round1.jsonl`, and
  `ollama_tool_ac_round2.jsonl`.
- Electron controls on the same PID are intentionally not hidden. Fresh
  Electron row 165 (`[LAG-S21-UI-FRESH-Z]`) answered exactly with
  `reasoning_content=null`; this is an empty think-rail/easy-prompt behavior,
  not a parser proof. Fresh Electron row 168 (`[LAG-S21-UI-FRESH-AA]`) emitted
  visible step-by-step text with `reasoning_content=null` despite
  `enable_thinking=true`; the in-app log shows the request used `/v1/responses`
  with `thinking_mode="reasoning"`, `reasoning_effort="medium"`, and no
  history. A raw replay of the same UI-shaped request in
  `responses_raw_ui_shape_ab_summary.json` did not reproduce the visible-step
  leak, so this remains a stochastic/model-output negative control rather than
  a proven renderer/request-off bug. Artifact:
  `current_pid4279_after_template_mirror/electron_rows_162_165_168.json`;
  screenshot: `laguna-ui-fresh-aa-visible-negative.png`.
- Therefore this addendum upgrades the raw API reasoning/tool-loop evidence for
  Laguna S-2.1 on PID 4279, but it does not close the global
  reasoning-content protocol gate or guarantee every Electron prompt produces a
  non-empty reasoning rail.

- Current 1.6.15 release prepackage check (2026-07-22 local):
  `current_pid4279_after_template_mirror/verify-bundled-python-current-1b905.log`
  passed, confirming the bundled `vmlx_engine` version matches
  `panel/package.json` (`1.6.15`), critical bundled `vmlx_engine` files match
  source hashes, critical bundled `jang_tools` files match source hashes, and
  the required MLX/VLM/audio/JANG runtime imports load from the bundled Python
  tree.
- The actual prepackage manifest gate was rerun first from HEAD
  `1b905cfbcad93877d17f255ea08df35651616fea` and again after the later panel
  build proof/source HEAD `841ff7ebd2f10f26d9a9d8cf9ecbe7ef190f2553`, both
  with the correct shared venv:
  `PYTHON=/Users/eric/mlx/vllm-mlx/.venv/bin/python npm run
  release:prepackage`. Logs:
  `current_pid4279_after_template_mirror/release-prepackage-current-1b905.log`
  and
  `current_pid4279_after_template_mirror/release-prepackage-current-841ff.log`.
  The current rerun wrote
  `build/current-release-regression-manifest-pre-panel-dist.json` and reported
  `current_proof_sweep=fail`, `prepackage_ready=false`, and
  `release_ready=false`. Failed/missing manifest components include broad API
  surface, cache architecture, model-family/artifact detection, parser
  registry, reasoning template, tool-call loop, native MTP, VL/media,
  packaged-integrity, release-surface, live-smoke, real Electron full-model
  matrix, and DSV4 freshness rows. The `841ff7ebd` rerun is the current
  release-stop evidence for 1.6.15.
- Current panel/build proof after the later Laguna/API commits (2026-07-22
  local): `panel-typecheck-current-9f18.log` records `npm run typecheck`
  passing on current 1.6.15 source. `panel-full-tests-current-9f18.log`
  records the complete panel Vitest suite passing: `79 passed` test files,
  `2403 passed`, `3 skipped`. `panel-build-current-9f18-rerun-correct-jang.log`
  records `npm run build` passing after pointing the release bundle script at
  the actual clean JANG package source
  `VMLX_JANG_TOOLS_SOURCE=/Users/eric/jang/jang-tools`; the build installed
  local `vmlx 1.6.15` and local `jang 2.5.31`, rewrote 96 console-script
  shebangs to the relocatable sibling-Python trampoline, passed bundled import
  verification, and completed `electron-vite build`. The first production
  build attempt used the wrong source path (`/Users/eric/jang` instead of the
  `jang-tools` package directory) and failed before shebang rewrite; it was an
  invocation error, not counted as a product pass.

- Full Python suite was not rerun end-to-end after the later 1.6.15 version bump; targeted release/version tests were. Current full panel suite, panel production build, bundled-Python verification, and TypeScript typecheck have now been rerun and passed in this gate.
- Full protocol matrix remains open: non-stream Chat/Responses and
  cancellation/disconnect/recovery across representative models remain open;
  this gate now adds current-source Laguna streaming Anthropic/Ollama
  one-tool continuations but does not generalize them cross-family.
- Full settings parity remains open for all models; this gate only checks Laguna S-2.1 generation defaults observed in UI.
- Disk-only L2/paged-off partial-prefix restore remains unproven here.
- Broader model family matrix remains open: DSV4 Flash typed composite cache, MiniMax M3 sparse/lightning cache, Gemma, Qwen/Bonsai/Ornith, Step, Nemotron/Omni/audio/video, M2.7, LFM, openPangu.
- Packaging, signing, notarization, v1.6.15 git tag/GitHub release,
  latest.json/feed updates, and install smoke were not performed in this gate.

Verdict for this gate: PARTIAL scoped closure. Laguna S-2.1 current-source
reasoning separation, post-tool phantom-call streaming, explicit
`enable_thinking` propagation, and local bundled-Python freshness are
source/test/live-proven for the named scope. Electron/API proof used
Electron-launched PIDs 93364, 93786, and 4279 plus raw API. Current 1.6.15
prepackage manifest evidence is negative (`prepackage_ready=false`,
`release_ready=false`), so overall release remains BLOCKED until the remaining
release gates above are closed or explicitly deferred with a release-risk note.
