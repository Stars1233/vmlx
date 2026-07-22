# Qwen3.6 35B JANGTQ agentic protocol matrix

Date: 2026-07-22 (America/Los_Angeles)

Current status: `PARTIAL / ELECTRON TWO-TOOL, HISTORY, AND DIRECT ANTHROPIC
STREAM ROW VERIFIED-LIVE`.

The retained Electron row below ran against source/evidence head
`aa97a531b8c8193be38a6fe8e7f766f0e31499c1` on the M5 Max proof host. The
four-protocol direct/gateway stream/non-stream and cancellation matrix is still
open; this file must not be read as closing the global `R16-AGENTIC-HARNESS`
gate.

## Exact artifact and runtime identity

- Bundle:
  `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`
- `config.json`: `model_type=qwen3_5_moe`,
  `weight_format=mxtq`, top-level routed storage `bits=2`, `mxtq_bits=2`.
- `jang_config.json`: profile `JANGTQ2`, method `affine+mxtq`, 8-bit
  attention/linear-attention/shared-expert/embed/head and 2-bit routed expert.
  This is JANGTQ/MXTQ Hadamard/codebook storage, not affine JANG and not base
  MLX MXFP.
- `generation_config.json`: temperature `1.0`, top-p `0.95`, top-k `20`,
  sampling enabled.
- Launch argv: qwen tool parser, qwen3 reasoning parser, continuous batching,
  64-token in-memory paged blocks, 1,000 blocks, 15% RAM ceiling, and Block
  Disk Cache L2.
- The Electron Sessions **Start** control created PID `20167` on port `8007`.
  Before any chat request, `/health` reported `model_loaded=true` and about
  11.1 GB active model memory. `r16-devapp.log` lines 49-50 contain the required
  `[Engine Manager] Found in PATH: .../.venv/bin/vmlx-engine` evidence.

The startup log identifies 40 hybrid layers: 10 `TurboQuantKVCache` attention
layers and 30 native SSM/linear-attention companion layers. Stored-prefix policy
is q4 for the eligible attention KV component; the SSM companion remains native
full precision. This does not convert the JANGTQ model weights into affine JANG.

## Anthropic combined-result ordering and exact-tool finalization

The first coding-harness-shaped Anthropic continuation exposed two shared
integration defects rather than a model-family quantization defect:

1. `_convert_user_message` flattened a single Anthropic user block containing
   `tool_result` plus follow-up text as user text first and tool result second.
   That broke the required OpenAI-compatible adjacency between an assistant
   function call and its result. The adapter now emits the prior tool result
   first, followed by the new user instruction.
2. Chat streaming's exact-one native-tool buffer could expose pre-tool prose or
   a rejected Qwen `<parameter=...>` suffix after final parsing. An exact-one
   turn now suppresses that prose and keeps invalid native-control residue
   hidden while returning the existing truthful required-tool error.

The source regressions are
`TestRequestValidationEdgeCases::test_mixed_text_and_tool_result_in_user_message`
and
`TestOpenAILogprobsFormatting::test_streaming_chat_exact_once_hides_pretool_prose_and_orphan_markup`.
Together with the reusable harness contracts in
`tests/test_agentic_protocol_matrix.py`, the current focused run passed 44/44.
No alias was added for `_run_command`: source inspection found no prompt or
parser path that emits that name, and the live log classified it as an extra
model-emitted unavailable candidate while retaining a separate valid
`run_command` call. Silently rewriting arbitrary leading underscores would be
an unsafe fake repair.

The authoritative natural-prompt live artifact is
`r16-agentic-anthropic-stream-natural-4096.json`. Direct port `8007` completed:

1. exactly one `file_info(path="panel/package.json")` call and real 5.2 KB
   result;
2. 594 private-reasoning characters followed by exactly one
   `run_command(command="pwd")` call and real result;
3. 1,511 private-reasoning characters followed by 85 progressively streamed
   visible characters and exactly
   `AGENTIC-ANTHROPIC-STREAM-DONE SIZE=5.2 KB
   PWD=/Users/eric/mlx/vllm-mlx-release-1.6.13`.

Every direct terminal was balanced and truthful, tool turns exposed no prose,
and no private reasoning or native tool control text appeared in visible
content. The Electron gateway completed round one but its round-two generation
produced 463 private-reasoning characters and no second tool call, so it
returned the truthful required-tool error. That gateway row remains `PARTIAL`;
the successful direct row does not promote the complete protocol matrix.

The three earlier artifacts are diagnostic controls, not parity passes:

- `r16-agentic-anthropic-stream-postfix.json` records the 1,024-token
  post-fix boundary;
- `r16-agentic-anthropic-stream-postfix-4096.json` used base-specific prompt
  labels and therefore cannot prove direct/gateway parity;
- `r16-agentic-anthropic-stream-identical-4096.json` used the same synthetic
  diagnostic prefix on both paths and showed the prompt itself could change
  native call formation.

## Electron settings and two-tool loop

The live Chat Settings drawer showed the bundle-derived values:

- Thinking `Auto`;
- temperature `1.00`;
- top-p `0.95`;
- top-k `20`;
- repetition penalty `1.00`;
- Built-in Coding Tools enabled with working directory
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`.

Prompt marker `R16-Q35-UI-TWO-TOOL` required this exact order:

1. `file_info(path="panel/package.json")`;
2. only after the real result, `run_command(command="pwd")`;
3. only after that result, one exact visible synthesis.

The first live poll caught the intermediate `file_info` result while the UI
still displayed `Processing tool results...`. The later frame showed exactly
one `file_info`, exactly one `run_command`, and:

```text
R16-Q35-UI-TWO-TOOL-DONE SIZE=5.2 KB PWD=/Users/eric/mlx/vllm-mlx-release-1.6.13
```

SQLite row 204 preserves three separate post-prompt/post-tool reasoning
segments, two distinct tool-call IDs, both real tool results, the exact final
content, and no warnings. The aggregate UI row reported 359 completion tokens,
5,336 prompt tokens, 0.87 s TTFT, and 11.2 s wall time. Aggregate row metrics
are not used as proof that each agent step had one continuous decode rate.

## Follow-up history and resident cache reuse

The next user turn explicitly prohibited tools and asked the model to recall
both preceding real results. It emitted no tool call and exactly:

```text
R16-Q35-UI-HISTORY-DONE SIZE=5.2 KB PWD=/Users/eric/mlx/vllm-mlx-release-1.6.13
```

SQLite row 207 has a distinct 300-character reasoning segment, non-empty exact
visible content, no tool-call history for that turn, and no warnings. Its
metrics report 4,036 restored `paged+ssm+tq-native` prompt tokens and 0.56 s
TTFT. Post-turn health agrees: one scheduler hit, 4,036 tokens saved, 85
q4-TQ-native Block L2 writes, native SSM companion records, and no eviction.
This proves the same-process resident reuse subrow only; it does not replace the
required SSD-only, eviction, or process-restart gates.

## Retained evidence

- `r16-qwen-chat-settings-current.png` — real Chat Settings drawer.
- `r16-q35-ui-two-tool-progress.png` — completed two-tool Electron row.
- `r16-q35-ui-history-final.png` — exact no-tool history continuation.
- `r16-q35-ui-two-tool-db.json` — persisted two-tool row and tool history.
- `r16-q35-ui-history-db.json` — persisted follow-up row.
- `r16-q35-health-after-ui.json` — cache/TQ/SSM state after tool loop.
- `r16-q35-health-after-history.json` — resident reuse counters.
- `r16-q35-ui-two-tool-logs.txt` — startup, resolved sampler, three Responses
  iterations, TQ Block L2 writes, and native SSM companion logs.
- `r16-devapp.log` — Electron dev-app launcher/path evidence.
- `r16-agentic-anthropic-stream-natural-4096.json` — current natural-prompt
  direct pass and gateway round-two partial boundary with timestamped,
  privacy-safe hashes/counts and real tool-result hashes.
- `r16-agentic-anthropic-stream-postfix.json`,
  `r16-agentic-anthropic-stream-postfix-4096.json`, and
  `r16-agentic-anthropic-stream-identical-4096.json` — retained diagnostic
  controls, explicitly not parity passes.

## Still open before global closure

- direct and gateway Chat/Responses/Anthropic/Ollama; the direct streamed
  Anthropic natural two-tool row is closed, while its gateway counterpart is
  still partial at the second-tool continuation;
- stream and non-stream for all four protocols;
- no-tool, auto, required, explicit function choice, and the same two-tool
  real-result continuation on the raw wire;
- malformed/missing/unknown/repeated tool-call negative controls;
- cancellation/disconnect during post-tool continuation, zero-active-request
  recovery, and no false success terminal;
- normalized timestamped direct/gateway comparison;
- disk-only partial match, eviction, and process-restart TQ/SSM restoration.
