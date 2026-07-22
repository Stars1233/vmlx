# Laguna S2.1 current reasoning/API/UI supplement — 2026-07-22

Status: `PARTIAL_NO_RELEASE`.

This supplement records current-source evidence for the Laguna S2.1 JANG_2L
reasoning/content/tool rail work after the 2026-07-21 cache gate. It does not
replace the broader matrix and does not authorize a release by itself.

## Checkout and runtime

- Host: `erics-m5-max.local`
- Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Branch: `codex/postrelease-ui-drawers-20260720`
- Source HEAD during latest proof: `ffbc19dd8`
- Latest committed source gate: `ffbc19dd8 fix(api): suppress phantom post-tool tool deltas` (`vmlx_engine/server.py` + `tests/test_server.py`), validated by focused tests and the live PID below.
- Electron profile: `/Users/eric/.vmlx-v1613-responsive-dev`
- CDP: `127.0.0.1:9335`
- Live model: `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L`
- Live backend during latest proof: `127.0.0.1:8018`, PID `90393`
  (started Tue Jul 21 23:38:22 2026)

## Bundle-grounded defaults

The real bundle reports:

- `config.json`: `model_type="laguna"`, `architectures=["LagunaForCausalLM"]`
- `generation_config.json`: temperature `1.0`, top-p `1.0`, top-k `20`,
  `do_sample=true`
- `jang_config.json`: chat reasoning supported, parser `deepseek_r1`,
  `default_enabled=true`, `template_kwargs_defaults.enable_thinking=true`
- `jang_config.json`: JANG affine mixed profile `JANG_2L`

The live server argv included `--tool-call-parser glm47`,
`--reasoning-parser deepseek_r1`, `--use-paged-cache`,
`--enable-block-disk-cache`, and `--stream-interval 1`.

## Source trace in this working set

- `panel/src/main/ipc/chat.ts`
  - Auto local reasoning sessions send `enable_thinking=true` and
    `chat_template_kwargs.enable_thinking=true` when a model has a reasoning
    parser.
  - Current reasoning segments are replayed before active tool calls in both
    Responses-style and Chat Completions-style continuations.
  - Strict no-tool/private-reasoning prompts can omit built-in tool schemas
    without disabling reasoning.
- `panel/src/shared/toolAutoContinue.ts`
  - Separates explicit tool requests from private calculation/reasoning prompts
    so text-only reasoning probes are not polluted by the built-in tool catalog.
- `panel/src/main/sessions.ts`
  - Guards stale child-process exits and PID liveness checks during one-model
    swaps/re-adoption so a healthy replacement is not marked down by an old PID.
- `vmlx_engine/server.py`
  - Builds per-request reasoning parsers from active parser/registry/template
    state and emits reasoning deltas separately from visible content deltas.
  - Current source gate under validation removes speculative empty OpenAI
    `tool_calls` deltas during native tool buffering. A tool call is advertised
    only after final parsing returns a schema-valid call, preventing optional
    post-tool native marker residue from becoming a phantom empty tool call.

## Live Electron evidence

Artifacts:

- `lag-s21-ui-think-rail-c.png`
- `lag-s21-ui-reason-tool-d.png`
- `laguna-ui-rows-135-138.json`
- `artifacts/laguna-ui-postpatch-rail-f.png`
- `artifacts/laguna-ui-auto-rail-h.png`
- `artifacts/laguna-ui-on-rail-i.png`

Fresh Electron Auto reasoning row:

- Prompt marker: `LAG-S21-UI-THINK-RAIL-C`
- DB row: `135`
- Visible content:
  `2468 + 1357 = 3825` plus `LAG-S21-UI-THINK-RAIL-C-DONE`
- `reasoning_content`: 1,011 chars persisted separately
- `reasoning_segments_json`: one reasoning segment persisted
- Warnings: `null`
- Metrics: `403 tokens`, `115 prompt`, `6.34s TTFT`, `13.9s total`
- UI screenshot visibly shows a `Reasoning` rail, then visible answer text.
- No inline `<think>`/`</think>` appeared in visible content.

Electron reasoning-history plus required-tool continuation:

- Prompt marker: `LAG-S21-UI-REASON-TOOL-D`
- DB row: `138`
- Visible content:
  `LAG-S21-UI-REASON-TOOL-D-DONE SIZE=5.2 KB`
- `reasoning_content`: `Let me call the file_info tool ...`
- Tool calls: exactly one `file_info` call with `{"path":"panel/package.json"}`
- Tool result: `Size: 5.2 KB`
- Warnings: `null`
- UI screenshot shows the reasoning rail and final visible answer.

Retained current long-chat negatives:

- `LAG-S21-UI-POSTPATCH-RAIL-F` in the already-long visible chat produced a
  correct non-empty visible answer, but `reasoning_content` length was `0`.
  Metrics: `30 tokens`, `800 prompt`, `114 paged+tq-native cached`, `0.88s
  TTFT`.
- `LAG-S21-UI-AUTO-RAIL-H` in the same long chat had Auto selected in the UI
  and produced a correct non-empty visible answer, but `reasoning_content`
  length was `0`. Metrics: `29 tokens`, `926 prompt`, `799
  paged+disk+tq-native cached`, `0.76s TTFT`.
- `LAG-S21-UI-ON-RAIL-I` in the same long chat had the visible UI control set
  to On and also produced a direct non-empty answer with `reasoning_content`
  length `0`. Metrics: `29 tokens`, `1051 prompt`, `925 paged+tq-native
  cached`, `0.66s TTFT`.
- These rows are not inline-thinking leaks and not empty-answer failures, but
  they are retained as Laguna/history/model direct-rail behavior. Do not cite
  them as proof that Auto/On always paints a reasoning rail.

Fresh Electron-main IPC Auto reasoning proof:

- Chat was created through the live renderer IPC (`window.api.chat.create`) and
  sent through the same Electron main `chat:sendMessage` path as the UI.
- Prompt marker: `LAG-S21-ELECTRON-IPC-FRESH-K`
- Visible content:
  `700 + 89 = 789` plus `LAG-S21-ELECTRON-IPC-FRESH-K-DONE`
- `reasoningContent`: 143 chars persisted separately
- Metrics: `97 tokens`, `112 prompt`, `0.41s TTFT`, `3.0s total`

Fresh Electron-main IPC built-in tool proof:

- Prompt marker: `LAG-S21-ELECTRON-IPC-FRESH-TOOL-L`
- Visible content:
  `LAG-S21-ELECTRON-IPC-FRESH-TOOL-L-DONE SIZE=5.2 KB`
- Tool calls: exactly one `file_info` call with
  `{"path":"panel/package.json"}`
- Tool result: `Size: 5.2 KB`
- Metrics: `60 tokens`, `439 prompt`, `128 paged+disk+tq-native cached`,
  `1.04s TTFT`, `2.7s total`
- This tool row did not emit reasoning; it proves the tool loop and final
  content path, not interleaved reasoning for this exact Laguna tool prompt.

## Raw API evidence

Artifacts:

- `artifacts/laguna-s21-current-raw-sse.json`
- `artifacts/laguna-raw-chat-tool-m-current.json`
- `laguna-current-raw-reasoning-rail.json`
- `laguna-current-chat-reasoning-B.json`
- `laguna-api-tool-loop-e.json`
- `laguna-chat-tool-first-rerun.json`
- `laguna-chat-tool-continuation-simple2.json`
- `laguna-anthropic-ollama-tool-f.json`
- `laguna-anthropic-final-rerun-g.json`

Raw `/v1/responses` reasoning stream:

- Request used explicit `enable_thinking=true`.
- Emitted progressive `response.reasoning_summary_text.delta` events.
- Then emitted progressive `response.output_text.delta` events.
- Final content:
  `2468 + 1357 = 3825` plus `LAG-S21-RAW-THINK-A-DONE`
- `content_has_inline_think=false`
- Terminal event: `response.completed`

Latest raw `/v1/responses` reasoning stream on PID `90393`:

- Request marker: `LAG-S21-RAW-RESP-RAIL-E`
- Request used explicit `enable_thinking=true` and
  `chat_template_kwargs.enable_thinking=true`.
- Emitted 263 progressive `response.reasoning_summary_text.delta` events,
  then 27 progressive `response.output_text.delta` events.
- Final content:
  `271 + 382 = 653` plus `LAG-S21-RAW-RESP-RAIL-E-DONE`
- Clean reasoning delta length: 638 chars
- `content_has_inline_think=false`
- Terminal event: `response.completed`

Raw `/v1/chat/completions` reasoning stream:

- Request used explicit `enable_thinking=true`.
- Emitted `delta.reasoning_content` separately from `delta.content`.
- Final content:
  `2468 + 1357 = 3825 LAG-S21-CHAT-THINK-B-DONE`
- `reasoning_len=1422`
- `content_has_inline_think=false`
- Terminal `finish_reason="stop"`

Latest raw `/v1/chat/completions` reasoning stream:

- Request marker: `LAG-S21-RAW-CHAT-RAIL-E`
- Request used explicit `enable_thinking=true` and
  `chat_template_kwargs.enable_thinking=true`.
- Emitted `delta.reasoning_content` before visible content deltas.
- Final content:
  `314 + 159 = 473` plus `LAG-S21-RAW-CHAT-RAIL-E-DONE`
- `content_has_inline_think=false`
- Terminal `finish_reason="stop"`

Raw `/v1/responses` tool loop:

- First request emitted one `function_call` for `lookup_code` with
  `{"key":"beta"}`.
- Follow-up with `function_call_output` emitted final visible content:
  `LAG-S21-API-RESP-TOOL-E-DONE VALUE=BETA-84`
- No repeat tool call and no inline thinking marker in final content.

Raw `/v1/chat/completions` tool loop:

- Simple tool request emitted one `lookup_code` tool call with
  `{"key":"alpha"}` and `finish_reason="tool_calls"`.
- Follow-up with a real `tool` role result emitted final visible content:
  `LAG-S21-CHAT-TOOL-SIMPLE2-DONE VALUE=ALPHA-42`
- No repeat tool call and no inline thinking marker in final content.

Retained negative: an over-specified Chat prompt combining
`tool_choice="required"` with an exact after-tool final contract produced an
empty tool-call shell and `finish="length"`. The simpler API-shaped tool
contract above passed; this negative remains a prompt-shape/model/tool-choice
edge to keep in the broader coding-harness soak.

Latest raw Chat tool negative/control:

- Request marker: `LAG-S21-RAW-CHAT-TOOL-M`
- Live restarted PID `90393` returned visible text
  `{call_file_info_of_panel_package_json}` and `finish_reason="stop"` rather
  than a schema-valid tool call.
- No empty-name `tool_calls` delta was emitted.
- This is a parser/model prompt-shape negative, not a valid tool-loop pass.
  It supports the phantom-delta guard by showing the server did not invent a
  tool call from non-schema text.

Raw `/v1/messages` Anthropic-style tool loop:

- First request emitted one `tool_use` block for `lookup_code` with
  `{"key":"gamma"}` and reached `message_stop`.
- The first follow-up returned only `GAMMA-126`; that exact-format miss is
  retained in `laguna-anthropic-ollama-tool-f.json`.
- A stricter follow-up rerun emitted final visible text exactly:
  `LAG-S21-ANTH-TOOL-G-DONE VALUE=GAMMA-126`
- No repeat `tool_use` block appeared in the passing follow-up.

Raw `/api/chat` Ollama-style tool loop:

- First request emitted one `lookup_code` tool call with `{"key":"delta"}` and
  terminal `done_reason="tool_calls"`.
- Follow-up with a `tool` role result emitted progressive content chunks and
  final visible text:
  `LAG-S21-OLLAMA-TOOL-F-DONE VALUE=DELTA-168`
- No repeat tool call appeared and `content_has_inline_think=false`.

## Current q4/cache telemetry

Artifact:

- `laguna-health-current-8018.json`

Live health reported:

- `kv_cache_quantization.enabled=true`
- `kv_cache_quantization.mode="turboquant-storage"`
- `kv_cache_quantization.bits=4`
- `kv_cache_quantization.stored_prefix_quantization="turboquant-q4"`
- `kv_cache_quantization.auto_policy="mixed_swa_full_attention_kv_storage_tq4"`
- `block_disk_cache.tq_native_enabled=true`
- `block_disk_cache.tq_native_writes=65`
- `block_disk_cache.tq_native_hits=4`
- `scheduler.cache_hit_tokens_by_detail` includes `paged+disk+tq-native` and
  `paged+tq-native`

This is current live q4/L2 telemetry for the active paged-on session. The
older 2026-07-21 Laguna gate contains the scoped paged-off SSD-only partial
prefix row. This supplement reran the focused current-source disk-only partial
prefix tests but did not rerun the full UI paged-off SSD-only live row.

Latest live PID `90393` health also reported:

- `block_disk_cache.tq_native_enabled=true`
- `block_disk_cache.tq_native_writes=34`
- `block_disk_cache.tq_native_hits=25`
- `block_disk_cache.total_tokens_on_disk=104082`
- `block_disk_cache.disk_size_gb=5.172`

Focused current-source cache validation:

- `pytest tests/test_paged_cache.py -k "disk_only_store_and_restart_restore_exact_partial_prefix or restart_restores_short_partial_prefix_for_longer_prompt or fetch_prefers_exact_partial_prefix_over_shorter_block_hit or extending_partial_prefix_realigns_durable_block_chain" -q`
- Result: `4 passed, 47 deselected`

Focused current-source tool-buffering validation:

- `pytest tests/test_server.py -k "post_tool_false_marker or tool_call_arguments_survive_buffering or reasoning_tool_call_keeps_arguments or required_empty_xml_tool_call_is_rejected" -q`
- Result: `4 passed, 131 deselected`

## Verdict

Scoped current-source PASS:

- Laguna Auto reasoning can produce a real Electron `Reasoning` rail with
  separate `reasoning_content`.
- Raw Chat Completions and Responses can emit reasoning on reasoning rails and
  visible answers on content rails without inline `<think>` leakage.
- Current Electron and raw Chat/Responses/Anthropic/Ollama can complete a
  one-tool continuation without repeating the tool after a result, within the
  prompt-shape caveats above.
- Active Laguna q4 TurboQuant storage and block-disk L2 telemetry are visible.
- Current source rejects speculative/invalid post-tool native marker residue
  without emitting phantom empty Chat Completions `tool_calls` deltas.

Still `PARTIAL_NO_RELEASE`:

- Full Python and panel suites were not rerun for this supplement.
- The full UI paged-off SSD-only partial prefix proof was not rerun in this
  supplement; the 2026-07-21 Laguna gate remains the live source for that row.
- Cross-chat/cross-session SSD partial-prefix matching, eviction breadth,
  signed-app proof, and notarized package gates remain outside this row.
- `jangtools` is not synced yet; current audit found split dirty/clean branches
  and version drift across `2.5.31`, `2.5.32`, and `2.5.33`.
- Current long-chat UI Auto/On rows can still legitimately produce no reasoning
  rail while emitting a correct visible answer; fresh Electron-main IPC and raw
  API prove the separated rail can work, not that every prompt/history forces a
  reasoning rail.
