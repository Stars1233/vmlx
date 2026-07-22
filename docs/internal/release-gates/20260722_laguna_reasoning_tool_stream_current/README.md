# Laguna S-2.1 current-source reasoning/tool streaming checkpoint

Date: 2026-07-21/22 local.  Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13` on `erics-m5-max.local`.  Branch: `codex/postrelease-ui-drawers-20260720`.

## Scope

This gate is narrow: Laguna S-2.1 JANG_2L, current source, Electron-launched server, Chat Completions streaming, reasoning rail separation, tool-call continuation, bundle-derived sampler defaults, and live cache detail.  It is not a full release matrix and does not close packaging/notarization.

## Source changes covered

- `vmlx_engine/engine/batched.py`: renders chat templates through the wrapper/inner tokenizer object that actually owns `apply_chat_template`; this is required for Laguna tokenizer wrappers whose inner tokenizer carries the native Poolside/Laguna template.
- `vmlx_engine/server.py`: Chat/Responses streaming now treats structured engine errors as errors instead of assistant reasoning/content, and Chat streaming no longer emits speculative empty `tool_calls` START deltas before a parser-valid function call exists.
- `vmlx_engine/engine_core.py`, `vmlx_engine/engine/base.py`, `vmlx_engine/output_collector.py`: preserve structured engine-loop failures as error fields instead of generated text.
- Tests pin the above in `tests/test_streaming_reasoning.py`, `tests/test_batching.py`, and `tests/test_server.py`.

## Live process proof

- `live-processes.txt`: single active `vmlx-engine serve` process for `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L`, PID 90393 at capture time.
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

`api_post_fix_no_phantom/summary.json` after Electron Save & Restart on current source:

- Round 1 required tool call: `tool_delta_count=1`, `finish_reasons=["tool_calls"]`, assembled call `file_info({"path":"panel/package.json"})`.
- Round 2 tool-result continuation: visible content `LAG-S21-API-TOOL-NOPHANTOM-DONE SIZE=5.2 KB`, `tool_delta_count=0`, `finish_reasons=["stop"]`, `done_seen=true`.
- Cache detail on the live API rows included `paged+disk+tq-native` on the first round and `paged+tq-native` on continuation.

`api_reasoning_prompt_compare/summary.json` documents variable Laguna reasoning behavior with `enable_thinking=true`:

- Simple arithmetic Auto/On-style prompts can emit no reasoning rail and stream content directly.
- A stronger prompt requiring at least five private sentences emitted `reasoning_len=703`, `reasoning_delta_count=248`, then progressive content deltas, no marker leakage, and terminal stop.

Interpretation: for Laguna S-2.1, Auto/On permits reasoning and the parser separates it if emitted. It does not force a non-empty reasoning rail on every easy prompt. That is model behavior, not an inline-reasoning leak, as long as emitted reasoning remains separate and visible content has no raw think/tool markers.

## Electron visual proof

- `ui/laguna-ui-force-rail-j-chat.png` and `ui/laguna-ui-force-rail-j-chat-text.txt`: Electron chat row for `[LAG-S21-UI-FORCE-RAIL-J]` shows `Reasoning 394 chars`, visible answer `888 + 111 = 999` plus marker, and metrics `206 tokens`, `38.1 t/s`, `1178 prompt (1050 paged+disk+tq-native cached)`, `0.76s TTFT`, `6.2s total`.
- `selected-ui-db-rows.json`: persisted UI rows include row 156 with non-empty `reasoning_content`, clean visible content, and cache detail `paged+disk+tq-native`; row 138 shows tool execution + continuation with final `LAG-S21-UI-REASON-TOOL-D-DONE SIZE=5.2 KB`.

## Cache proof level

`health-after-ui-api.json` after the UI/API pass showed:

- scheduler cache hits/misses present, `tokens_saved=1944`, `backend_mode=paged`, `disk_hits=25`, `disk_promotion_hits=25`.
- block disk cache had `blocks_on_disk=1725`, `l2_block_tokens_on_disk=104283`, `disk_writes=38`, `tq_native_writes=38`, `tq_native_hits=25`, `tq_native_enabled=true`.

This proves current live Laguna reuse through paged + disk + TQ-native on this session. It does not prove disk-only-with-paged-off partial prefix restore; that remains a separate open gate.

## Tests

`focused-tests.log`: 16 passed:

- Laguna/wrapped-template and reasoning seed tests.
- structured engine error tests for batching/output collector/Chat/Responses.
- Chat post-tool phantom `tool_calls` regression.
- Responses tool-call argument buffering regression.

## Remaining release blockers not closed by this gate

- Full Python and panel suites/build/typecheck were not rerun in this gate.
- Full protocol matrix remains open: non-stream Chat/Responses, Anthropic, Ollama, cancellation/disconnect/recovery across representative models.
- Full settings parity remains open for all models; this gate only checks Laguna S-2.1 generation defaults observed in UI.
- Disk-only L2/paged-off partial-prefix restore remains unproven here.
- Broader model family matrix remains open: DSV4 Flash typed composite cache, MiniMax M3 sparse/lightning cache, Gemma, Qwen/Bonsai/Ornith, Step, Nemotron/Omni/audio/video, M2.7, LFM, openPangu.
- Packaging, signing, notarization, version bump, latest.json/feed updates, and install smoke were not performed in this gate.

Verdict for this gate: PARTIAL scoped closure. Laguna S-2.1 current-source reasoning separation and post-tool phantom-call streaming are live-proven on Electron-launched PID 90393 and raw API. Overall release remains BLOCKED until the remaining release gates above are closed or explicitly deferred with a release-risk note.
