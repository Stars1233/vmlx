# LFM2.5 reasoning and tool streaming proof

Date: 2026-07-21 (America/Los_Angeles)

Scope: non-Laguna Python/API lane only. The shared Electron instance was
running another agent's Laguna session, so this lane deliberately did not
operate, restart, or stop that UI/model. Electron verification is therefore
still required before a release-wide claim.

## Source and bundle trace

- vMLX checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Source HEAD when the live retest started: `ad931a45f`
- Bundle:
  `/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`
- Registry/runtime configuration exposed family `lfm2`, tool parser `lfm2`,
  reasoning parser `qwen3`, thinking support with no native thinking-off mode,
  and text-only modalities.
- Live scheduler startup reported 24 state layers: 6
  `TurboQuantKVCache` attention layers plus 18 native SSM/array layers. Stored
  attention prefix state used native q4 TurboQuant; SSM companion state stayed
  full precision and was re-derived.

## Defect and fix

The exact natural request:

`Use file_info exactly once to inspect panel/package.json.`

previously left the fallback LFM example as
`file_info(path='VALUE_HERE')`. The live model copied that example and emitted
the wrong schema-valid argument. `_derive_lfm2_request_param_value` only bound
quoted values or values following the literal parameter name.

The fix in `vmlx_engine/api/tool_calling.py` recognizes an explicit file tool
request followed by a file-oriented verb (`inspect`, `examine`, `stat`, `read`,
or `check`) and binds the following path only for path-like parameters. It does
not guess from unrelated prose. The new regression test is
`test_lfm2_natural_file_info_request_binds_path_without_path_keyword`.

Focused source verification:

- `tests/test_tool_prompt_fallback.py`: **25 passed**

## Raw API streaming results

All probes used the live model on `127.0.0.1:8051`, source loaded through
`PYTHONPATH` from the checkout above, `--stream-interval 1`, and raw SSE/NDJSON
arrival timestamps. Counts below are wire events, not reconstructed final
messages.

### Chat Completions, reasoning then answer

- 73 `delta.reasoning_content` chunks, 0.060757s through 0.357253s.
- 6 `delta.content` chunks, 0.369568s through 0.389938s.
- Visible answer: `The product is 323.`
- Terminal: `finish_reason=stop` at 0.396666s.
- No raw `<think>` markers and no tool calls.

### Responses, reasoning then progressive answer

- 120 `response.reasoning_summary_text.delta` events, 0.065793s through
  0.553788s.
- 5 `response.output_text.delta` events, 0.566306s through 0.582387s.
- Visible answer: `Spring Summer Autumn Winter`.
- `response.completed` at 0.588981s.

### Chat Completions, exact tool and continuation

- Initial turn streamed 81 reasoning chunks.
- The completed tool call was exactly
  `file_info({"path":"panel/package.json"})`; terminal reason was
  `tool_calls`.
- The tool-result continuation streamed 76 reasoning chunks followed by 11
  content chunks.
- Final visible answer: `LFM-TOOL-CONTINUATION-PASS`.
- No second tool call; terminal reason was `stop`.

### Responses, exact tool and continuation

- With `max_output_tokens=2048`, the initial turn streamed 178 reasoning
  deltas and two function-argument deltas.
- Completed arguments were exactly
  `{"path":"panel/package.json"}`.
- Terminal was `response.completed`; usage reported 239 input, 416 output,
  and 235 cached input tokens with cache detail `paged+ssm+tq-native`.
- The function-output continuation streamed 72 reasoning deltas followed by
  10 output-text deltas.
- Final visible answer: `LFM-RESPONSES-TOOL-PASS`.
- No repeated tool call; terminal was `response.completed` with cache detail
  `paged+ssm+tq-native` (162 cached of 166 input tokens).

### Honest output-budget boundary

The same Responses tool request with `max_output_tokens=1024` still emitted a
complete, schema-valid tool call, but its terminal was honestly
`response.incomplete` with reason `max_output_tokens`. Raising the explicit
request budget to 2048 produced `response.completed`. This row is retained so
the lower-budget result is not misreported as a transport/parser success.

### Explicit thinking Off

`enable_thinking=false` returned HTTP 400 with the advertised contract:
LFM2 has no native thinking-off/instruct mode and accepts Auto/On only.

## Verdict

`PASS` for this LFM2.5 API scope: separate progressive reasoning/content
events, Chat and Responses tool emission, exact argument binding, tool-result
continuation, no repeated call, and honest terminal status were live-proven.

`PARTIAL` for the release: no Electron visual proof was run in this lane; no
Anthropic/Ollama LFM row was run; no restart-from-disk cache proof was run; and
the separate family-wide source audit found other non-LFM blockers that must be
closed before packaging.
