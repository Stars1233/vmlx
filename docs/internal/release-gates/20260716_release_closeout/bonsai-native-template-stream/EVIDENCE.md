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

## Current multi-tool continuation repair

Status: `PASS-LIVE` for the explicit two-tool Bonsai/Qwen Responses contract;
`PARTIAL` for unconstrained catalog selection and cross-model live coverage.

Source trace:

- Panel commit `a1a6591b9` recognizes explicit `call <name> exactly once`
  sequences, retires each completed schema, and uses a bounded rolling decode
  sample for final multi-iteration TPS instead of pairing cumulative tokens
  with only the final tail.
- Shared server commit `3d32b944b` separates a terminal Qwen tool-result
  continuation from a client-narrowed or explicitly requested remaining-tool
  continuation. The second case receives only the uncompleted schema instead
  of the contradictory `Do not emit another <tool_call>` instruction.
- `tests/test_tool_prompt_fallback.py` retains the single-tool duplicate guard
  and adds the remaining-tool prompt contract. The complete prompt/format set
  passed 139/139 at this commit (`b1-qwen-tool-tests.log`).

Retained red controls:

- Persisted row 309 executed only `file_info`, then the panel's old singular
  direct-answer classifier removed `run_command` from the next request.
- Row 315 used the repaired TPS accounting (52.8 t/s) but repeated completed
  tools until the four-iteration ceiling and produced no final answer.
- The pre-server-fix raw Responses replay produced a valid first `file_info`,
  then spent 768 tokens on a truncated native tool marker because the shared
  Qwen fallback called every post-result request terminal. These failures are
  root-cause controls, not passes.

Current live proof:

- A real `/usr/bin/curl -N` Responses harness executed one
  `file_info(panel/package.json)`, replayed its real output, executed one
  `run_command({"command":"pwd"})`, replayed that output, and then sent a
  tools-off final request. The final marker arrived in nine separately timed
  `response.output_text.delta` events from 0.9265s through 1.0741s, matched
  `response.output_text.done`, and ended once as `response.completed` with no
  warnings (`b1-api-multi-current.json`).
- The dev Electron model was visibly stopped and started so Python reloaded
  commit `3d32b944b` without clearing L2. Fresh row 321 then executed exactly
  one `file_info` followed by exactly one `run_command`, persisted both real
  results, and displayed exact `B1-CURRENT-MULTI7-DONE`. It reported 464 output
  tokens, 52.8 t/s, 0.56s TTFT, and 11.9s total. The 3s/7s/final screenshots
  retain the progressive UI states.
- Current health after the row distinguishes Bonsai's cache modes truthfully:
  16 attention-KV layers use q8 TurboQuant at the paged/L2 storage boundary;
  48 SSM/GDN companion layers remain native. Mid-request packing is disabled
  (`compress_after=0`) and is not claimed as memory reduction. The process
  recorded 10 native-TQ writes, 12 native-TQ hits, one `paged+ssm+disk`
  execution with `dequantized=true`, and a real SSM disk hit. A separate
  changed-prefix KV-only candidate safely full-prefilled because its SSM
  fingerprint did not match.

Remaining limits:

- Bonsai still produces verbose/repetitive native reasoning on some exact-tool
  prompts, and one post-repair row attempted a retired duplicate schema that
  the server correctly dropped. Keep broader catalog reliability `PARTIAL`.
- This proves the shared server change on Bonsai only. Other Qwen artifacts and
  non-Qwen parser families need their own live rows before broad classification.
