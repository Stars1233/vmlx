# Qwen 3.6 35B JANGTQ Responses post-tool continuation

Status: `PASS-LIVE` for this scoped current-source row at commit `2fbf38d19`.
Global release status remains `PARTIAL_NO_RELEASE` until the full release gates
and remaining matrix rows are closed.

## Artifact identity

- Bundle: `dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`
- Weight format: `mxtq`; profile `JANGTQ2`; routed expert weights use the
  JANGTQ/TurboQuant codebook path. This is not JANG affine and is not base MLX
  MXFP.
- Cache quantization is a separate axis. This run intentionally retained the
  session's explicit cache-off settings, so it proves the Responses/tool
  continuation path only and makes no cache-hit claim.

## Root cause and source trace

`vmlx_engine.server._coerce_orphan_tool_messages_for_template` cleared active
tool-call IDs when a request-scoped `system` instruction appeared between the
stored assistant function call and the current `function_call_output`. The
valid tool result was therefore rewritten as ordinary user prose only on the
`previous_response_id` path. The current-source fix preserves active tool-call
IDs across `system`/`developer` messages, keeps request-scoped instructions out
of stored history, normalizes stored function arguments, and reconciles
suppressed tool markup monotonically without reviving raw XML.

Relevant source/tests:

- `vmlx_engine/server.py`
- `tests/test_responses_history.py`
- `tests/test_responses_multimodal_history.py`
- `tests/test_qwen3_answer_pass_policy.py`
- `tests/test_tool_format.py`
- `tests/test_server.py`

Focused current-source test result: 163 passed.

## Raw API proof

`q35-final-source-stream-nonstream.json` records one required `file_info` call,
then reuses its real `function_call_output` through `previous_response_id`.
Three streaming continuations each emitted 38 reasoning deltas and 22 content
deltas, with content appearing from approximately 0.65 to 0.92 seconds. Three
non-streaming controls completed with the same real 5.2 KB result. Every row
had one terminal completion, no repeated function call, no tool-control markup
in visible content, and the final marker `Q35-FINAL-SOURCE-DONE`.

## Electron proof

The real dev Electron app at CDP 9335 was stopped and started from its visible
controls after the final source was installed. The engine reloaded the bundle
on port 8029. A fresh chat then:

1. called built-in `file_info(panel/package.json)` exactly once;
2. persisted one matching call/result (`call_d27f5c2d`), reporting 5.2 KB;
3. streamed a non-empty answer ending `Q35-UI-FINAL-TOOL-DONE`;
4. answered a same-chat no-tool follow-up with 5.2 KB and
   `Q35-UI-FINAL-RECALL-DONE`.

The database artifacts retain content, reasoning, metrics, native tool call,
and real tool result. Screenshots retain waiting/mid-stream and final UI states.

## Artifacts

- `q35-final-source-stream-nonstream.json`
- `q35-final-source-stream-nonstream.out`
- `q35-ui-final-tool-db.json`
- `q35-ui-final-recall-db.json`
- `q35-ui-final-tool-mid.png`
- `q35-ui-final-tool-done.png`
- `q35-ui-final-recall-mid.png`
- `q35-ui-final-recall-done.png`
- `q35-final-health.json`
