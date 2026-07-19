# Qwen3.6 35B JANGTQ current-source stream/cache/tool gate

Date: 2026-07-19

Source cutoff: `54222003d` (`reconcile/1.5.68`, pushed to
`origin/codex/live-electron-gates-20260715`)

Verdict: `PASS-LIVE_SCOPED_TEXT_TOOL_STREAM_RESTART_L2_PARTIAL_PREFIX`.
This is not a family-wide, media, strict-sampling, or release verdict.

## Artifact and route truth

- Bundle: `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`.
- `config.json`: outer `qwen3_5_moe`, inner `qwen3_5_moe_text`, 40 layers,
  ten `full_attention` layers and thirty `linear_attention` companion layers,
  `weight_format=mxtq`, `mxtq_bits=2`.
- `jang_config.json`: `profile=JANGTQ2`, `weight_format=mxtq`, routed experts
  2-bit, attention/linear-attention/shared/embed/head 8-bit. This is
  JANGTQ/MXTQ (Hadamard/codebook), not affine JANG and not base MLX MXFP.
- The bundle name, JANG sidecar, and tensor index do not declare MTP. The nested
  architecture hint is therefore inactive; health reports `artifact_available=false`
  and `runtime_active=false`.
- Vision weights/config exist, but this loaded route reports
  `vl_runtime_available=false`. This gate is text-only; image/video remains OPEN.

Source trace:

- `vmlx_engine/utils/hybrid_tq_cache.py:57-107` classifies Qwen from nested
  model type plus the real mixed attention/companion layout, never the name.
- `vmlx_engine/utils/turboquant_config.py:107-201` assigns non-Bonsai Qwen
  hybrid attention slots the q4 native storage policy; companion state is not
  assigned fake KV/TQ slots.
- `vmlx_engine/cli.py:1005-1035` disables only the second, generic
  `QuantizedKVCache` wrapper. Commit `87e11c5ee` corrects the misleading startup
  message: native architecture-selected attention-TQ storage remains enabled.
- `vmlx_engine/server.py:19610-19741` emits Responses reasoning-summary and
  output-text deltas separately while shielding tool markup.
- `panel/src/main/ipc/chat.ts:2503-2570` consumes reasoning and visible deltas
  separately and retains the terminal-text fallback only when no text delta was
  seen.
- `panel/src/shared/jangQuantization.ts:13-65` reads the current sidecar's
  top-level `profile=JANGTQ2` as well as nested legacy profiles.
- `panel/src/main/model-config-registry.ts:947-953` exports that bundle-grounded
  label through `detectConfig`; `SessionCard.tsx:116-151` uses the real bundle
  basename for its immediate fallback, so a provider directory such as
  `jangq-ai/` cannot relabel an MXFP child bundle.

## Focused tests

- Python hybrid/TQ/cache and policy selection: 103 passed, 609 deselected.
- Panel reasoning display, tool-history replay, and chat override policy:
  127 passed across three files.
- Quant-label detector/card regression: 94 passed across three files, followed
  by clean `tsc --noEmit`.
- No full-suite/build inference is made from these focused tests.

## Electron proof

The model was stopped and started through the real Sessions UI multiple times;
the tested PIDs include 22225, 23858, and current-source PID 24686. The server
drawer shows the project engine command
`/Users/eric/mlx/vllm-mlx/.venv/bin/python3 -m vmlx_engine.cli serve ...`.

Current-source row 437:

- New chat inherited `builtin_tools_enabled=1` and working directory
  `/Users/eric/mlx/vllm-mlx`.
- Exactly one `file_info({"path":"panel/package.json"})` was generated and
  executed.
- The real tool result reported `Size: 5.2 KB`.
- Visible final was exactly
  `Q35-JT-CURRENT-SOURCE-TOOL-DONE SIZE=5.2 KB`.
- Reasoning remained in the separate rail; visible content was non-empty; no
  warning was persisted.
- Metrics: 549 prompt tokens, 256 `paged+ssm+disk` cached, 0.28s TTFT, 1.8s
  total.

Three repeated Electron tool turns had distinct reasoning hashes and lengths
(367, 253, and 235 characters), so this row does not exhibit stale reasoning
replay.

Controls retained rather than hidden:

- A fresh chat created after a tools-Off diagnostic inherited
  `builtin_tools_enabled=0`; request logs truthfully reported `has_tools=false`.
  The model did not call a tool and hallucinated a hypothetical/blank size. This
  is a configuration control, not a parser pass.
- The current no-tool Auto turn had separate, non-empty reasoning and content
  and a clean terminal, but expanded/echoed the requested markers instead of
  obeying byte-exact formatting. Strict sampled instruction reliability remains
  PARTIAL.

Post-label-fix current-source UI/runtime proof:

- A complete Electron-main relaunch used
  `VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1611-cachefix-dev` and logged
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- The real Sessions `Start` button launched PID 26427. The Sessions card and
  active chat header both displayed `JANGTQ2 (2b)`.
- Visual controls stayed distinct: Bonsai 1-bit displayed
  `JANG_AFFINE_1BIT (1.1128b)`; DSV4/Gemma basename fallbacks remained JANG;
  base Nemotron MXFP4 showed no JANG badge; and the MXFP4 child under the
  `jangq-ai/` provider directory also showed no false JANG badge. The excluded
  Mistral MXFP4 model was not loaded or generation-tested.
- Fresh Electron row 440 executed exactly one real
  `file_info({"path":"panel/package.json"})`, returned a non-empty visible
  answer, restored 3,904 `paged+ssm+disk` tokens, and persisted no warning.
  It misspelled the requested marker as `Q35-JT-L-LABEL-LIVE-DONE`; therefore
  the agent/tool loop passes but strict sampled formatting remains PARTIAL.

Visual and DB evidence:

- `q35jt-current-source-tool-pass.png`
- `session-card-quant-labels-final-v3.png`
- `session-card-qwen-started.png`
- `q35jt-active-header-final.png`
- `q35jt-label-live-tool-partial.png`
- `q35jt-label-live-tool-row.json`
- `health-after-label-live-turn.json`
- `q35jt-truthfix-live-log.png`
- `electron-current-source-tool-db-rows.json`
- `electron-db-rows.json`
- `electron-chat-overrides.json`

## Current-source API tool and streaming proof

Responses:

- Required-tool request emitted 297 reasoning-summary deltas, two incremental
  argument deltas, one schema-valid `file_info` call, and one completed terminal.
- Real-result continuation emitted 256 reasoning deltas followed by 18 visible
  deltas containing exactly
  `Q35-JT-CURRENT-API-RESP-DONE SIZE=5.2 KB`, then one
  `response.completed` with terminal usage and no warning.

Chat Completions:

- Required-tool request emitted one `file_info` with exact arguments,
  `finish_reason=tool_calls`, and terminal usage.
- Real-result continuation emitted 256 reasoning deltas followed by 18 content
  deltas containing exactly
  `Q35-JT-CURRENT-API-CHAT-DONE SIZE=5.2 KB`, then `finish_reason=stop`, usage,
  and `[DONE]`.

Longer timed controls immediately before the comment/log-only source change
proved the deltas were not flushed as one final batch:

- Chat: 256 reasoning chunks from 0.283s to 2.849s, then 107 content chunks
  from 3.100s to 4.225s (1.124s content span), terminal at 4.239s.
- Responses: 256 reasoning chunks from 0.268s to 2.802s, then 91 content chunks
  from 3.058s to 4.021s (0.963s content span), terminal at 4.045s.

Raw current-source traces:

- `current-source-responses-tool.sse`
- `current-source-responses-follow.sse`
- `current-source-chat-tool.sse`
- `current-source-chat-follow.sse`

The earlier full raw/timed controls are retained as
`q35jt-responses-*.sse` and `q35jt-chat-*.sse`.

## Exact restart and partial-prefix L2 proof

Exact Electron replay after real Stop/Start:

- Before replay: all in-process scheduler, native-TQ hit, and SSM-disk hit
  counters were zero; 418 prior disk blocks remained.
- After replay: 435 tokens restored as `paged+ssm+disk`; seven block-disk hits
  were seven native-TQ hits; two SSM companion disk hits were recorded.
- The tool executed once and exact-finaled; TTFT improved from the earlier
  0.38s row to 0.09s.

Changed-suffix partial-prefix replay after another real Stop/Start:

- Seed prompt: 2,587 input tokens, exact `Q35-JT-L2-SEED-A`, 2.653s.
- The changed-suffix request shared the long prefix but required a different
  exact answer. It restored 2,560 tokens from 40 disk-backed blocks, leaving 27
  prompt tokens uncached, and returned exact `Q35-JT-L2-PARTIAL-B` in 0.476s.
- Current source repeated the changed suffix as
  `Q35-JT-L2-PARTIAL-C`: 2,560 cached tokens, exact output, 0.492s.
- Live log: `Paged cache hit ... 40 blocks, 2560 tokens`, `SSM disk HIT ...
  states=30 complete=True`, then `VLM HYBRID cache HIT ... 34 remaining`.
- After reuse, health recorded 40 block-disk hits, all 40 native-TQ hits, plus
  one SSM-disk hit. No unsafe KV-only reuse was accepted.

Evidence:

- `health-after-restart-before-replay.json`
- `health-after-restart-replay.json`
- `health-partial-after-restart-before-reuse.json`
- `health-partial-after-reuse.json`
- `health-current-source-final.json`
- `q35jt-restart-disk-replay-pass.png`
- `q35jt-restart-disk-replay-logs.png`
- `q35jt-partial-api-disk-logs.png`

## Remaining issues

- `Q35-STRICT-SAMPLED-RELIABILITY`: PARTIAL. Retain the no-tool marker
  expansion and historical stochastic malformed/long tool candidates; no
  synthetic dedup, guessed argument, hidden sampler clamp, or prompt coercion
  was added.
- `Q35-VL-ROUTE`: OPEN. The artifact advertises vision, but this loaded route
  reports `vl_runtime_available=false`; no media claim is made.
- `Q35-SESSION-CARD-QUANT-LABEL`: PASS-LIVE at `54222003d`. The bundle-grounded
  card and active header both say `JANGTQ2 (2b)`; affine JANG and base MXFP
  controls remain distinct. Provider directory names no longer contaminate the
  fallback label.
- Full suites, build, bundled-Python refresh, signed/notarized packaging, and
  the broader model/protocol/settings/media/gateway matrix remain outside this
  scoped pass.
