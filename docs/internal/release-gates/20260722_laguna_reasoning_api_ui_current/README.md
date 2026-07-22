# Laguna S2.1 current reasoning/API/UI supplement — 2026-07-22

Status: `PARTIAL_NO_RELEASE`.

This supplement records current-source evidence for the Laguna S2.1 JANG_2L
reasoning/content/tool rail work after the 2026-07-21 cache gate. It does not
replace the broader matrix and does not authorize a release by itself.

## Checkout and runtime

- Host: `erics-m5-max.local`
- Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Branch: `codex/postrelease-ui-drawers-20260720`
- Source HEAD during proof: `a22321300`
- Electron profile: `/Users/eric/.vmlx-v1613-responsive-dev`
- CDP: `127.0.0.1:9335`
- Live model: `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L`
- Live backend: `127.0.0.1:8018`, PID `85920`

## Bundle-grounded defaults

The real bundle reports:

- `config.json`: `model_type="laguna"`, `architectures=["LagunaForCausalLM"]`
- `generation_config.json`: temperature `1.0`, top-p `1.0`, top-k `20`,
  `do_sample=true`, `reasoning_parser="poolside_v1"`
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

## Live Electron evidence

Artifacts:

- `lag-s21-ui-think-rail-c.png`
- `lag-s21-ui-reason-tool-d.png`
- `laguna-ui-rows-135-138.json`

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

## Raw API evidence

Artifacts:

- `laguna-current-raw-reasoning-rail.json`
- `laguna-current-chat-reasoning-B.json`
- `laguna-api-tool-loop-e.json`
- `laguna-chat-tool-first-rerun.json`
- `laguna-chat-tool-continuation-simple2.json`

Raw `/v1/responses` reasoning stream:

- Request used explicit `enable_thinking=true`.
- Emitted progressive `response.reasoning_summary_text.delta` events.
- Then emitted progressive `response.output_text.delta` events.
- Final content:
  `2468 + 1357 = 3825` plus `LAG-S21-RAW-THINK-A-DONE`
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
prefix row. This supplement did not rerun the paged-off SSD-only row.

## Verdict

Scoped current-source PASS:

- Laguna Auto reasoning can produce a real Electron `Reasoning` rail with
  separate `reasoning_content`.
- Raw Chat Completions and Responses can emit reasoning on reasoning rails and
  visible answers on content rails without inline `<think>` leakage.
- Current Electron and raw Chat/Responses can complete a one-tool continuation
  without repeating the tool after a result.
- Active Laguna q4 TurboQuant storage and block-disk L2 telemetry are visible.

Still `PARTIAL_NO_RELEASE`:

- Full Python and panel suites were not rerun for this supplement.
- Anthropic/Ollama current-source tool-result continuations were not rerun in
  this supplement.
- Paged-off SSD-only partial prefix proof was not rerun in this supplement.
- Cross-chat/cross-session SSD partial-prefix matching, eviction breadth,
  signed-app proof, and notarized package gates remain outside this row.
- `jangtools` is not synced yet; current audit found split dirty/clean branches
  and version drift across `2.5.31`, `2.5.32`, and `2.5.33`.
