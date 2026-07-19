# Chat Completions terminal-usage parity proof

Date: 2026-07-19
Source fix: `5358842b2ec337c0e69fcc81c97a81d078827970`
Live model: `jangq-ai/openPangu-2.0-Flash-JANG_3M`
Scoped status: **VERIFIED-LIVE**
Overall release status: **PARTIAL**

## Defect and owning layer

The shared Chat Completions generator intentionally attached a growing,
non-null usage object to every reasoning/content delta for real-time UI
metrics. A live pre-fix stream carried 317 non-null usage objects. This
violated the OpenAI Chat streaming contract: ordinary chunks carry
`usage:null`, and one additional choices-empty chunk immediately before
`[DONE]` carries the total request usage.

This was a global server serialization/finalization issue. It was not a
Qwen, Bonsai, openPangu, JANG/JANGTQ, parser, cache, or model-output defect.
`chat-before-analysis.json` preserves the measured failing shape; its raw SSE
is retained in the preceding prompt-disk evidence gate.

## Source repair

Commit `5358842b2` makes the Chat stream contract explicit:

- `_dump_chat_chunk` emits `usage:null` on every ordinary chunk when
  `stream_options.include_usage=true`;
- growing engine counters are no longer exposed on reasoning, content,
  whitespace, or tool-buffering chunks;
- exactly one choices-empty terminal chunk carries the authoritative total;
- the generic terminal-finish guard detects a usage-only tail and inserts a
  missing `finish_reason=stop` before that tail, preserving the required
  `finish -> usage -> [DONE]` order;
- error and tool-call early-return paths retain one terminal partial/final
  usage chunk;
- Electron continues deriving live TPS from actual progressive deltas and
  consumes authoritative prompt/cache/completion totals at stream end.

`source-trace.txt` records the implementation and test locations.

## Raw Chat Completions after Electron process replacement: PASS

The real Electron Start/Stop controls replaced the openPangu server process
before the current-source request. The streamed Auto-reasoning request then
produced:

- 388 ordinary JSON chunks with `usage:null`;
- zero ordinary chunks missing the usage field;
- exactly one non-null usage chunk, with `choices:[]`;
- a `finish_reason=stop` chunk at index 387;
- the usage-only chunk at index 388;
- `[DONE]` after the usage chunk;
- 1,330 separate reasoning characters and exact visible content
  `OP-CHAT-USAGE-ORDER-DONE STATE-49059`.

`chat-after.sse`, `chat-after-analysis.json`, and the health/restart snapshots
preserve the complete wire proof. The request was a disk-cache miss after
the bounded 10 GB directory evicted the original base record, so this row
does not make a new cache-hit claim; the prior cache gate separately proved
1,431 disk-cached tokens with the same endpoint.

## Electron Chat-wire progressive no-tool turn: PASS

The current chat override was first verified in SQLite as
`wire_api=completions`, and the settings UI showed
`http://127.0.0.1:8027/v1/chat/completions`. Built-in tools were enabled, but
the prompt explicitly required no tool.

The six retained frames from `electron-progress-01.png` through
`electron-progress-11-final.png` show the reasoning rail grow from 61 to
1,370 characters while token/TPS metrics changed progressively. The final
frame and DB row contain exactly `CHAT-USAGE-UI3-DONE`, separate reasoning,
no tool call, no warning, and final metrics:

- 414 output tokens;
- 364 prompt tokens;
- 290 memory-cached tokens;
- 25.3 tokens/s;
- 0.87s TTFT;
- 17.3s total.

This proves that removing per-delta usage objects did not batch or freeze the
Electron Chat-wire rendering path. `electron-chat-settings.png`,
`electron-chat-overrides.json`, and `electron-rows.json` ground the wire and
persistence claims.

## Electron Chat-wire exact-once tool loop: PASS

With the same Chat Completions wire and built-in tools enabled, openPangu:

- emitted one `file_info` call;
- supplied exactly `{"path":"panel/package.json"}`;
- executed the tool once;
- received `Size: 5.2 KB`;
- continued after the tool result;
- returned exactly `CHAT-USAGE-TOOL-DONE SIZE=5.2 KB`;
- persisted separate reasoning, one OAI tool call, one matching tool result,
  coherent final metrics, and no warnings.

Evidence: `electron-tool-loop.png`, `electron-rows.json`, and
`electron-tool-health.json`.

## Tests

- Python stream/parser/cache telemetry selection: 666/666 passed.
- Panel request-builder and tool-status selection: 84/84 passed.
- Panel TypeScript typecheck: passed.

The Python selection includes normal Chat usage shape, MiniMax-M3 bounded
answer-pass usage, Qwen answer-pass policy, Anthropic's Chat adapter,
terminal-finish ordering, tool paths, and the stale object-construction
fixtures surfaced by the broad run.

## Non-claims and remaining work

- This shared fix is source-level for every Chat Completions model route, but
  only openPangu received fresh live Electron + raw API proof in this gate.
  Cross-family current-source live coverage remains required by the master
  matrix.
- Responses, Anthropic, and Ollama were not all re-run live in this scoped
  gate. Existing focused tests passed; protocol matrix closure remains a
  separate required row.
- This does not close paged RAM eviction/block-disk refault, media, gateway,
  settings, full-suite/build, packaging, signing, notarization, or release
  gates.
- No release or publication claim is made.
