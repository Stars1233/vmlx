# Qwen3.6 35B JANGTQ tool parser and reasoning/tool rail gate

Date: 2026-07-21

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Artifact: `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`

## Scope and artifact identity

This is a scoped JANGTQ/MXTQ row for
`dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`. It is not affine JANG and it is not
base MLX/MXFP. The live engine argv in the current Electron run includes:

```text
--tool-call-parser qwen --enable-auto-tool-choice --reasoning-parser qwen3
--use-paged-cache --enable-block-disk-cache --stream-interval 1
```

Health in the same run reported q4 TurboQuant storage policy for the compatible
attention-KV component:

```json
{"mode":"turboquant-storage","bits":4,"stored_prefix_quantization":"turboquant-q4","auto_policy":"qwen_hybrid_attention_kv_storage_tq4"}
```

This gate does not close all Qwen JANGTQ output-format behavior. One earlier
strict prompt still showed model-visible formatting drift when built-in tools
were not enabled in the current chat. The closed defect is narrower: a
schema-valid off-template Qwen tool-call shape is now parsed as a real function
call, fake model-written tool results are not trusted, and Chat/Responses plus
the Electron tool loop can complete.

## Source repair

- `vmlx_engine/tool_parsers/qwen_tool_parser.py:77-104` adds two
  schema-gated observed live shapes:
  - markdown header:
    `# Calling file_info` plus `# path=panel/package.json`;
  - labeled-line header:
    `[Q36-JT-UI-TOOL2]`, `file_info`, `path: panel/package.json`.
- `vmlx_engine/tool_parsers/qwen_tool_parser.py:162-224` centralizes
  single-required-string schema validation so unadvertised tools, wrong
  parameter names, and non-string required arguments are rejected.
- `vmlx_engine/tool_parsers/qwen_tool_parser.py:341-357` routes the new shapes
  through final extraction as `tool_calls` with no visible content.
- `vmlx_engine/tool_parsers/qwen_tool_parser.py:383-407` detects the same
  shapes in streaming extraction and passes the active request schema into
  final parsing.
- `vmlx_engine/tool_parsers/qwen_tool_parser.py:462-474` includes the new
  shapes in exact-once early-stop truncation so post-call fake result text is
  discarded.
- `vmlx_engine/server.py:18384-18417` makes Chat Completions buffer explicit
  exact-once tool-selection turns from token one, matching the existing
  Responses exact-once behavior. This prevents off-template call scaffolding
  from streaming as visible content before final parsing converts it into a
  structured tool call.
- `vmlx_engine/server.py:18788-18824` still allows safe reasoning deltas during
  that exact-once selection turn, but trims at the first unparsed tool marker
  so tool-call text does not leak through the reasoning rail.

No fake tool result, prompt coercion, hidden sampler clamp, or synthetic tool
execution was added. The parser accepts only request-schema-valid calls.

## Focused tests

Remote commands:

```sh
cd /Users/eric/mlx/vllm-mlx-release-1.6.13
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest tests/test_tool_parsers.py -q -k "QwenToolParser"
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest tests/test_tool_parsers.py tests/test_server.py -q -k "QwenToolParser or exact_once_buffers_schema_labeled_qwen_call"
```

Results:

```text
17 passed, 88 deselected
20 passed, 219 deselected
```

The added coverage includes markdown call parsing, schema rejection, streaming
request-schema use, labeled-line parsing after a prompt marker, and exact-once
truncation after the argument line. The server regression pins the Chat
Completions exact-once buffering contract for the labeled Qwen shape.

## Live Electron proof

The current-source development Electron app was running on CDP `127.0.0.1:9335`
with user data `/Users/eric/.vmlx-v1613-responsive-dev`. Qwen was stopped and
restarted through the real UI `Start` button after the parser source changed.
The new live PID was `36630` and its argv had the parser/cache flags listed
above.

The current chat settings had built-in coding tools enabled before the decisive
turn. Earlier in the same current profile, with the tool path unavailable to the
model, Qwen wrote a fake `file_info` result and SQLite had
`tool_calls_json=null`; that row remains a negative control.

The decisive Electron row is:

- prompt row: SQLite row `7`;
- assistant row: SQLite row `9`;
- visible content: `Q36-JT-UI-TOOL4-DONE SIZE=5.2 KB`;
- reasoning rail: non-empty and separate;
- tool status: exactly one `file_info` call, followed by one real tool result;
- terminal: no warning;
- metrics: `206` tokens, `110.2` t/s, `1783` prompt tokens,
  `867 paged+ssm+tq-native cached`, `0.76s` TTFT, `3.3s` total.

Artifacts retained in this directory:

- `q36-jt-ui-tool4-current-profile.png` — visible Electron row with
  `Info panel/package.json` and exact final text.
- `q36-jt-ui-tool4-current-profile-row.json` — current SQLite row evidence.

## Raw API proof

Artifact retained:

- `q36-jt-api-tool-parser-summary.json`
- `q36-jt-responses-reasoning-summary-event-proof.json`

Summary:

- Chat Completions first turn streamed a structured `file_info` call with
  arguments `{"path": "panel/package.json"}` and `finish_reason="tool_calls"`;
  continuation with a real tool-output message streamed visible content
  `Q36JTAPI1-DONE SIZE=5.2 KB` with `finish_reason="stop"`.
- Responses first turn streamed one `response.output_item.done` function call
  named `file_info` with arguments `{"path": "panel/package.json"}` and
  `response.completed`; continuation with `function_call_output` streamed
  visible content `Q36JTRESP1-DONE SIZE=5.2 KB` and `response.completed`.
- Neither raw path emitted inline `<think>` in the reasoning or visible text
  captured by the probe.
- A separate Responses reasoning probe confirmed the current event names and
  progressive rails for this artifact: `256`
  `response.reasoning_summary_text.delta` events, `110`
  `response.output_text.delta` events, and `response.completed`.

## Remaining open rows

- This does not certify all Qwen exact-format compliance. The model can still
  produce verbose, explanatory, or bracketed visible text on some strict labels.
  In the raw reasoning probe, protocol rails were separate and progressive, but
  the model still placed explanatory math text in visible content before the
  final marker. That is recorded as model-visible style behavior, not inline
  `<think>` parser leakage.
- This does not close Anthropic/Ollama for this exact JANGTQ artifact; those
  protocol families remain broader matrix rows.
- This does not close SSD-only, partial-prefix, eviction, restart, or media
  rows for this artifact.
- This does not close the global metrics/EOS/fallback-accounting issue; that
  remains under separate audit.
