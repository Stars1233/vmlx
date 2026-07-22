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

The unqualified full Python-suite invocation captured in
`/tmp/vmlx-full-pytest-20260722-235509.log` failed 12 tests because `node`/`npx`
were absent from PATH, then the bundled verifier exposed stale bundled Python
content. The node/npx portion was environment setup noise, but the stale bundle
was a real release blocker until `bundle-python.sh` was rerun. A complete
end-to-end Python suite rerun with the corrected PATH was not performed in this
gate.

## Remaining release blockers not closed by this gate

- Full Python suite with corrected node PATH, full panel suite, panel build, and typecheck were not rerun end-to-end in this gate.
- Full protocol matrix remains open: non-stream Chat/Responses, Anthropic, Ollama, cancellation/disconnect/recovery across representative models.
- Full settings parity remains open for all models; this gate only checks Laguna S-2.1 generation defaults observed in UI.
- Disk-only L2/paged-off partial-prefix restore remains unproven here.
- Broader model family matrix remains open: DSV4 Flash typed composite cache, MiniMax M3 sparse/lightning cache, Gemma, Qwen/Bonsai/Ornith, Step, Nemotron/Omni/audio/video, M2.7, LFM, openPangu.
- Packaging, signing, notarization, version bump, latest.json/feed updates, and install smoke were not performed in this gate.

Verdict for this gate: PARTIAL scoped closure. Laguna S-2.1 current-source reasoning separation, post-tool phantom-call streaming, explicit enable_thinking propagation, and local bundled-Python freshness are source/test/live-proven for the named scope. Electron/API proof used Electron-launched PIDs 93364 and 93786 plus raw API. Overall release remains BLOCKED until the remaining release gates above are closed or explicitly deferred with a release-risk note.
