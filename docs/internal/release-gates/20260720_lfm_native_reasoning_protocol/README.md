# LFM2.5 native reasoning and required-tool protocol gate

Date: 2026-07-20

Status: `PARTIAL_LIVE`

This gate covers the post-v1.6.13 remote Electron/runtime worktree at
`/Users/eric/mlx/vllm-mlx-release-1.6.13` on `erics-m5-max.local`. The tested
branch is `codex/postrelease-ui-drawers-20260720`; the source base before this
gate was `5c3a3b9ed`. The commit containing this file is the scoped fix commit.

## Artifact identity and source trace

Tested bundle:

`/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`

- `config.json` reports `model_type=lfm2_moe`, 24 layers, and base MLX
  `mxfp4` quantization. There is no `jang_config.json`; this is not affine
  JANG and not JANGTQ/MXTQ.
- Layer types contain six `full_attention` slots (2/6/10/14/18/21) and 18
  convolution/SSM companions.
- `chat_template.jinja` owns Liquid's native reasoning and Python-call tool
  syntax. It contains no `enable_thinking` branch.
- The bundle README says the generation prompt ends at
  `<|im_start|>assistant\n`, reasoning may appear in `<think>...</think>`, and
  the runtime must not inject a synthetic `<think>` prefill.

Current source therefore removes the LFM-only synthetic empty-think sentinel,
advertises native reasoning (`supportsThinking=true`) without a fabricated
thinking-off rail (`supportsInstructMode=false`), and makes Auto/On the only UI
choices for this family. Explicit Off is rejected consistently instead of
silently changing the prompt contract.

## Live Electron evidence

The Electron renderer ran from this checkout through CDP `127.0.0.1:9335`
with user data `/Users/eric/.vmlx-v1613-responsive-dev`. A visible Save &
Restart changed the LFM engine process to PID 25504; its environment included
`PYTHONPATH=/Users/eric/mlx/vllm-mlx-release-1.6.13`.

No-tool Auto turn:

- reasoning rail grew progressively from 52 to 327 to 657 to 686 characters;
- visible content began before terminal completion;
- final visible content was exactly `LFM-UI-AUTO-CURRENT-DONE`;
- persisted metrics: 224 tokens, 211.3 tok/s, 4,643 prompt tokens, 0.47 s
  TTFT, 1.6 s total.

This is a scoped PASS for native Auto reasoning and UI output emission.

Required built-in tool turn:

- prompt required one `file_info` call with `path=panel/package.json`;
- the model emitted malformed native arguments parsed as `{"path": ": "}`;
- the real executor returned `Path not found: : `;
- the final content leaked faux tool JSON and replayed the previous
  `LFM-UI-AUTO-CURRENT-DONE` marker;
- the UI showed `Info : failed`.

This is `FAIL_LIVE` for this MXFP4 artifact's required-tool reliability and
multi-turn marker fidelity. It does not invalidate the older JANG_2L artifact
proof; artifact and prompt scope must remain explicit.

Screenshots:

- `lfm-current-settings.png`
- `lfm-current-logs.png`
- `lfm-current-server-before-restart.png`
- `lfm-current-tool-failure.png`

## API/protocol evidence

Auto-mode direct and gateway captures are preserved in
`lfm-auto-direct-current.json` and `lfm-auto-gateway-current.json`.

Explicit Off returned a clear 400 for direct Chat Completions, Responses,
Anthropic Messages, and Ollama Generate. The gateway preserved the same clear
400 for Chat, Responses, and Anthropic, but gateway Ollama reduced it to
`{"error":"Backend request failed"}`. That gateway error-parity row remains
OPEN.

For `tool_choice=required`, the current model again produced no schema-valid
call. Current source now terminates truthfully:

- incremental `error.code=tool_calls_required`;
- final event `response.failed`;
- final response `status=failed` with the same structured error;
- no contradictory `response.completed`;
- the failed result is not stored as successful Responses history.

The current raw proof is `lfm-tool-ab-results.json`. This is a PASS for the
global Responses terminal contract and a FAIL for the model/tool row.

## Cache evidence observed during these turns

The live required-tool turn reused 128 tokens as
`paged+ssm+disk+tq-native`. Health reported a hybrid cache with six q4
TurboQuant attention-KV layers and native SSM companions, bounded paged RAM,
block-L2 writes/hits, and one q4-native disk hit. The stronger cold/warm,
eviction, disk-only, partial-prefix, and process-restart proofs remain in:

- `docs/internal/release-gates/20260719_lfm_native_tq4/`
- `docs/internal/release-gates/20260719_paged_ram_ssd_hierarchy/`

This gate does not re-promote those broader rows beyond their documented
scope.

## Validation

- 13/13 selected LFM template, reasoning-mode, registry, prompt-fallback, and
  parser tests passed.
- The exact Responses required-empty-tool regression passed.
- 88/88 selected panel registry/history tests passed.
- Panel TypeScript typecheck passed.
- `git diff --check` passed for the scoped source/test files.

## Verdict

- Native LFM Auto reasoning and progressive UI output: `PASS_LIVE_SCOPED`.
- Synthetic thinking-off removal and capability parity: `PASS_SOURCE_TESTED`
  plus live UI/API evidence.
- Required-tool Responses terminal semantics: `PASS_LIVE_SCOPED`.
- LFM2.5 MXFP4 required built-in tool execution/final fidelity:
  `FAIL_LIVE`.
- Gateway Ollama error-detail parity: `OPEN`.
- Overall LFM family/tool/protocol campaign: `PARTIAL_LIVE`, not release-ready
  by itself.
