# Bonsai native Qwen tool-stream evidence

Status: `PASS-LIVE` for the scoped literal `file_info` exact-once flow;
`PARTIAL` for the broader Qwen tool catalog and release matrix.

## Source trace

- Commit `f993e36b8` preserves a Qwen bundle's native `<tools>` schema and
  trained XML/result framing for ordinary auto-tool turns instead of replacing
  it with a second synthetic system contract.
- `vmlx_engine/api/tool_calling.py` now prevents the generic Step scaffold
  detector from also claiming an already-classified Qwen prompt. Required-tool
  API requests and post-tool continuations retain their stricter guarded paths.
- `panel/src/shared/toolAutoContinue.ts` recognizes the observed exact-once
  wording `after its result`, so the first tool result enters the existing
  tools-off, thinking-off direct-answer follow-up. Ordinary multi-tool prompts
  are unchanged.

Focused checks at this source:

- 136/136 tool prompt/format tests passed.
- Four Qwen exact-once/reasoning-stream server and engine-audit tests passed.
- 18/18 panel tool-auto-continue tests passed.
- Panel TypeScript typecheck passed.

## Live API evidence

On `jangq-ai/Bonsai-27b-1bit-JANG`, port 8030, with the real Qwen parser and
reasoning parser:

- Raw Chat selection emitted 95 incremental reasoning deltas, then one valid
  `file_info({"path":"panel/package.json"})` and
  `finish_reason=tool_calls` at 3.009s. The pre-fix matched branch had emitted
  227 reasoning deltas, malformed its output, and ended at `length`.
- Raw Responses selection emitted 117 reasoning deltas from 0.394s to 2.745s
  and one completed `file_info` function item. Supplying its real function
  output in the continuation produced 14 content deltas from 0.431s to 0.669s,
  exact `B1-RESPONSES-TOOL-TEMPLATE1-DONE`, one `response.completed`, and no
  warning.

## Live Electron evidence

The dev Electron app was relaunched from current source using
`/Users/eric/.vmlx-v1611-cachefix-dev` and CDP 9335. Bonsai was started with the
visible `Load Model` control; PID 34438 launched with `--tool-call-parser qwen`,
`--reasoning-parser qwen3`, paged cache, Block Disk L2, and Auto KV cache.

- Persisted assistant row 54 has exact content
  `B1-ELECTRON-TOOL-TEMPLATE2-DONE`, one OpenAI `file_info` call, one matching
  result, no warning, and no second execution.
- Renderer/preload listeners observed 61 progressively timed reasoning events
  and 13 progressively timed content events. Content arrived from 6.830s to
  7.078s; total time was 7.1s.
- An identical new-chat repeat restored 157 of 158 prompt tokens as
  `paged+ssm`; TTFT moved from 0.46s to 0.20s. It retained one tool call and
  progressive 59 reasoning / 13 content events.
- A visible Electron Stop/Load replaced PID 34438 with PID 34884 without
  clearing L2. The identical request restored 157 of 158 prompt tokens as
  `paged+ssm+disk`, streamed 55 reasoning / 13 content events, executed one
  tool, and exact-finaled. Disk-restore TTFT was 1.26s.
- `b1-electron-tool-template2.png` visibly shows the Bonsai model header, one
  Info result, the exact final marker, and the completed metrics.

The earlier failed row is retained: before the wording detector was fixed, the
same live form executed `file_info` twice and emitted one terminal content blob.
That row is evidence for the harness root cause, not counted as a pass.
