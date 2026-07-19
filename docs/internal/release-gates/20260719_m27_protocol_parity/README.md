# MiniMax M2.7 protocol parity and effective no-tool prompt state

Date: 2026-07-19

Status: `VERIFIED-LIVE` for the scoped current-source Electron, Chat
Completions, and Responses rows below. Overall protocol/release status remains
`PARTIAL`: Anthropic, Ollama, cancellation/disconnect/mid-stream recovery, and
the signed-app repeat were not exercised in this gate.

## Artifact identity and runtime

- Model:
  `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`
- Bundle/runtime class: JANGTQ/MXTQ TurboQuant codebook weights. This is not
  affine JANG and not base MLX MXFP.
- Current source/fix: `ffb9ed7db fix(stream): honor effective no-tool prompt state`
- Electron engine after the real Stop/Start: PID 88616 on `127.0.0.1:8014`.
- Effective cache/runtime argv is visible in `electron-server-panel.png` and
  `electron-logs-after-fix.png`: paged 64-token blocks, four-block limit,
  block-disk L2, q4 native-TQ KV storage, `minimax_m2` reasoning parser, and
  `minimax` tool parser.
- `health-after-fix.json` reports 62 live TurboQuant KV layers, 1,816 L2 block
  tokens, five native-TQ disk hits, and six L1 evictions. This is full-KV
  MiniMax M2.7; it is text-only and has no VL row.
- The retained current logs do not contain the historical literal
  `[Engine Manager] Found in PATH` line. They do show the exact
  `.../.venv/bin/python3 -B -s -m vmlx_engine.cli serve ...` launch command.
  The literal-line sub-check is therefore `PARTIAL`, not inferred.

## Defect and root cause

The pre-fix raw Chat post-tool continuation retained public tool schemas while
setting `tool_choice="none"` and `enable_thinking=false`. The renderer correctly
removed the tools, but four Chat/Responses streaming/non-streaming parser and
answer-policy paths re-read `request.tools` instead of the effective prompt
tool set. MiniMax was seeded as if a tool/thinking prompt still existed and a
plain direct answer was classified as suppressed reasoning.

`chat-tool-followup-before.raw` is the live red control: it has no visible
content, emits a misleading `stop`, then a reasoning-only warning, then
`length`, and finally `[DONE]`.

The shared helper at `vmlx_engine/server.py:3398` now treats
`tool_choice="none"` as no available schemas and otherwise reads the attached
effective tool set. It is used by Chat and Responses, streaming and
non-streaming paths (`server.py:13731,13866,13974,16177,16305,16431,17312,
17424,19039,19153`). This is an endpoint contract repair, not a MiniMax output
rewrite, fabricated tool call, sampler clamp, or hidden thinking retry.

Regression coverage is in `tests/test_server.py:4707-4806` and includes the
actual post-tool shape. `focused-tests.txt` records 244 passed with three
intentional deselections across `test_server.py` and
`test_streaming_reasoning.py`.

## Live Electron proof

The real Electron UI Start/Stop controls loaded the current engine. The fresh
three-turn chat is preserved in screenshots and
`electron-three-turn-db.json`:

1. Auto reasoning: 6,502 reasoning characters were stored separately and the
   exact visible answer was three lines ending `M27-TERMINAL-COMPLETE`.
2. Required tool: exactly one real
   `file_info({"path":"panel/package.json"})` call returned `5.2 KB`; the UI
   progressed through reasoning, call, result, processing, and the exact final
   `M27-PROTO-UI-TOOL-DONE SIZE=5.2 KB`. It restored 227
   `paged+disk+tq-native` tokens and stored no warning.
3. Same-chat recall: no second tool call, distinct reasoning, and exact visible
   `M27-PROTO-UI-RECALL-DONE PATH=panel/package.json SIZE=5.2 KB`.

The recall turn performed a full prefill because the explicit no-tool request
changed its tool-schema prompt shape. It is history/tool-result proof, not a
cache-hit claim.

## Raw API proof

- Chat stream (`chat-stream.raw`): 187 reasoning deltas, nine progressive
  content deltas, exact `M27-CHAT-SSE-DONE`, one stop, one terminal usage, and
  one `[DONE]`.
- Chat non-stream (`chat-nonstream.json`): non-empty exact content with separate
  reasoning.
- Responses stream (`responses-stream.raw`): 75 reasoning deltas, nine content
  deltas, matching `response.output_text.done`, and one `response.completed`.
- Responses non-stream (`responses-nonstream.json`): non-empty exact content
  with separate reasoning.
- Responses required tool (`responses-tool-initial.raw`) produced one valid
  `file_info` call with split argument deltas. The continuation
  (`responses-tool-followup.raw`) emitted 18 progressive content deltas and an
  exact final grounded in the real 5.2 KB result.
- Chat required tool (`chat-tool-initial.raw`) produced the valid call. After
  the source fix and a real Electron engine restart, the identical retained-
  schema continuation (`chat-tool-followup-after.raw`) emitted 18 progressive
  content deltas, exact `M27-CHAT-TOOL-CONTINUE-DONE SIZE=5.2 KB`, one stop,
  one terminal usage event, and one `[DONE]`. It restored 173
  `paged+disk+tq-native` tokens.
- The repaired Responses retained-schema shape is independently preserved in
  `responses-tool-retained-initial.json` and
  `responses-tool-retained-followup.raw`: 19 progressive content deltas and
  one `response.completed`.

## Gate boundary

| Row | Status | Evidence |
| --- | --- | --- |
| Electron real UI load/settings | PASS with literal PATH-log sub-check PARTIAL | Server/Cache/Logs screenshots, health, argv in logs |
| Electron Auto reasoning/content/terminal | PASS | `electron-auto-final.png`, DB rows 352/354 |
| Electron real tool/result/final | PASS | `electron-tool-final.png`, DB rows 355/357 |
| Electron same-chat recall/no replay | PASS | `electron-recall-final.png`, DB rows 358/360 |
| Chat stream/non-stream | PASS | `chat-stream.raw`, `chat-nonstream.json` |
| Chat tool/result continuation | PASS after live red control | `chat-tool-followup-before.raw`, `chat-tool-followup-after.raw` |
| Responses stream/non-stream | PASS | `responses-stream.raw`, `responses-nonstream.json` |
| Responses tool/result continuation | PASS | retained Responses tool artifacts |
| Anthropic Messages | OPEN | No current-source run in this gate |
| Ollama chat/generate | OPEN | No current-source run in this gate |
| cancellation/disconnect/mid-stream recovery | OPEN | No current-source fault injection in this gate |
| signed packaged-app repeat | OPEN | This used the current Electron dev build |

`SHA256SUMS` binds the retained raw streams, JSON, screenshots, and test output.
