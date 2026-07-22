# 2026-07-21 Reasoning Protocol Global Gate

Status: PARTIAL checkpoint. This closes one scoped streaming/protocol defect class on current source; it does not close the full release matrix.

Remote checkout:

- Host: `erics-m5-max.local`
- Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Branch: `codex/postrelease-ui-drawers-20260720`
- Commit: `2f3c36017 fix(streaming): keep reasoning on protocol rails`

## Scope closed by this checkpoint

This checkpoint fixes and proves the following scoped behavior:

- Streaming Chat Completions and Responses no longer fall back to exposing synthetic inline `<think>` text when the global reasoning parser is missing but the concrete registry/bundle names a parser.
- The base `<think>` reasoning parser handles repeated reasoning blocks, so `reasoning -> visible/tool -> reasoning -> final` does not leak the later `<think>...</think>` block as visible content.
- The Electron active tool loop preserves the current reasoning segment in follow-up request history before the tool call item, for both Responses-style and Chat Completions-style continuations.
- A current Laguna Electron UI row can complete a required built-in tool call with visible final text and without reasoning-only finalization.

## Source trace

- `vmlx_engine/server.py:3971-4035` adds per-request parser construction from the active configured parser, registry parser, or standard `<think>` fallback.
- `vmlx_engine/server.py:18332-18340` uses that parser for Chat Completions streaming.
- `vmlx_engine/server.py:20477-20485` uses that parser for Responses streaming.
- `vmlx_engine/reasoning/think_parser.py:116-144` parses repeated complete `<think>...</think>` blocks into reasoning while preserving visible content separately.
- `vmlx_engine/reasoning/think_parser.py:231-270` routes later streamed explicit think blocks back to reasoning instead of visible content.
- `panel/src/main/ipc/chat.ts:1737-1745` tracks current reasoning segments for the active turn.
- `panel/src/main/ipc/chat.ts:3327-3360` pushes current reasoning before active tool calls in Responses and Chat follow-up histories.

## Regression tests run

Remote command:

```sh
cd /Users/eric/mlx/vllm-mlx-release-1.6.13
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest tests/test_reasoning_parser.py tests/test_server.py -q -k "repeated_reasoning_blocks or registry_reasoning_parser_when_global_missing or invalid_minimax_xml_keeps_only_visible_prefix"
cd panel
npm test -- --run tests/tool-status-responsiveness.test.ts tests/reasoning-display.test.ts tests/tool-history-replay.test.ts
npm run typecheck
```

Result:

- Python: 5 selected tests passed.
- Panel: 3 files passed, 128 tests passed.
- TypeScript: `tsc --noEmit` passed.

## Live Electron proof

Current dev app/log state:

- CDP target: `127.0.0.1:9335`
- User data dir: `/Users/eric/.vmlx-v1613-responsive-dev`
- Dev log contains `[STARTUP] Using vMLX userData override: /Users/eric/.vmlx-v1613-responsive-dev`
- Dev log contains `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`

Current live model:

- UI session: `jangq-ai/Laguna-S-2.1-JANG_4M`
- Backend: `127.0.0.1:8009`
- PID: `30020`
- Launch argv included `--tool-call-parser glm47`, `--enable-auto-tool-choice`, `--reasoning-parser deepseek_r1`, `--use-paged-cache`, `--enable-block-disk-cache`, `--block-disk-cache-max-gb 10`, `--enable-jit`.

Electron no-tool row:

- Prompt row: SQLite row `393`
- Assistant row: SQLite row `395`
- Visible content: `LAG-GLOBAL-UI1-DONE`
- `reasoning_content`: `null`
- Warnings: `null`
- Metrics: `9 tokens`, `72.6 t/s`, `107.5 pp/s`, `72 prompt`, `0.67s TTFT`, `0.8s total`
- Screenshot: `/tmp/lag-global-ui1.png`

Electron tool row:

- Prompt row: SQLite row `402`
- Assistant row: SQLite row `404`
- Visible content: `LAG-GLOBAL-UI2-DONE SIZE=5.2 KB`
- `reasoning_content`: `null`
- Tool status: exactly one `file_info` call, result size `5.2 KB`, then `done`
- Warnings: `null`
- Metrics: `39 tokens`, `43.1 t/s`, `142.8 pp/s`, `415 prompt (128 paged+disk+tq-native cached)`, `1.36s TTFT`, `3.1s total`
- Screenshot: `/tmp/lag-global-ui2-clean.png`

## Raw API protocol proof

Artifact summary: `/tmp/laguna-protocol-rails-1784686222-summary.json`

Rows:

- `/v1/chat/completions` no-tool stream: exact content `PROTO-CHAT-NOTOOL-DONE`, terminal event present, no inline `<think>`.
- `/v1/chat/completions` tool stream: structured `lookup_code` tool call with `{"key": "alpha"}`, terminal `finish_reason="tool_calls"`, no inline `<think>`.
- `/v1/responses` no-tool stream: exact content `PROTO-RESP-NOTOOL-DONE`, `response.completed` present, no inline `<think>`.
- `/v1/responses` tool stream: `response.output_item.done` function call `lookup_code` with `{"key": "beta"}`, `response.completed` present, no inline `<think>`.
- `/v1/messages` no-tool stream: exact content `PROTO-ANTH-NOTOOL-DONE`, `message_stop` present, no inline `<think>`.
- `/v1/messages` tool stream: Anthropic `tool_use` block `lookup_code` plus `input_json_delta` `{"key": "gamma"}`, `message_stop` present, no inline `<think>`.
- `/api/chat` no-tool stream: exact content `PROTO-OLLAMA-NOTOOL-DONE`, final `done` present, no inline `<think>`.

## Important open issues observed or not closed

These remain PARTIAL or OPEN and must not be described as release-closed by this checkpoint:

1. Laguna produced unrelated Git guidance under a contaminated/stale multi-turn UI history for prompt rows `396/399` with assistant rows `398/401`. Reasoning and content were separated, but coherence was wrong. This needs a separate history/cache/prompt-contamination investigation.
2. The live Laguna rows in this checkpoint did not emit reasoning deltas (`reasoning_count=0`). They prove no inline leakage on these prompts, not live reasoning-delta behavior for all reasoning families. Qwen/Bonsai/Gemma/MiniMax rows still need current-source live proof.
3. Chat Completions live token/s can still use client-side `tokenCount++` per emitted chunk when the server does not send per-chunk usage (`panel/src/main/ipc/chat.ts:2475-2487`). Final metrics may be acceptable when terminal usage exists, but live rolling TPS remains a separate truthfulness gate.
4. Responses tool streams still emit vMLX heartbeat events such as `response.heartbeat` with `tool_call_generating=true`. That may be acceptable as an extension, but SDK compatibility needs a dedicated coding-harness soak.
5. Anthropic parallel/interleaved multi-tool streaming was not proven here. The observed single-tool row is clean.
6. Ollama tool calling was not proven here; only no-tool content streaming was exercised.
7. Full raw API continuation after tool result to final answer was not proven in this checkpoint. Electron built-in tool continuation was proven for one Laguna row.
8. Cache scope here is a hit observation only: Electron row `404` showed `paged+disk+tq-native cached`. SSD-only block-disk reuse, partial L2 match, restart restore, eviction, and architecture-specific cache rows remain separate gates.
9. One-model-only unload/swap, gateway port/LAN recovery, media audio/video/VL, DSV4 Flash native composite cache, MiniMax M3 sparse indexer, and Laguna speed/default sampling parity remain outside this scoped fix.

## Next recommended gates

1. Run a current-source live Qwen/Bonsai/Gemma/MiniMax reasoning-delta row that actually emits reasoning, then verify raw Chat/Responses/UI separation and non-inline content.
2. Fix or explicitly relabel live rolling TPS for Chat when only chunk counts are available.
3. Add a raw API tool-result continuation harness for Chat, Responses, Anthropic, and Ollama.
4. Investigate the Laguna contaminated-history coherence row before any broad Laguna release claim.
5. Continue the cache matrix: paged on, paged off plus SSD-only L2, partial block match, restart restore, eviction, and TQ-native hit accounting.
