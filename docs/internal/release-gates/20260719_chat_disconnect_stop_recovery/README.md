# Chat disconnect and Electron user-stop recovery gate

Date: 2026-07-19

Status: `VERIFIED-LIVE` for raw Chat Completions client disconnect plus immediate
recovery and real Electron prefill/mid-content user stop plus same-chat recovery on
current source `576e12733`. Overall protocol/release status remains `PARTIAL`.

## Raw Chat Completions disconnect

The live request used the Electron-started MiniMax-M2.7 process on port 8014 with
thinking explicitly disabled and a long direct counting response. The client
closed the HTTP stream after five progressive content deltas:

- chat id: `chatcmpl-00a861ba`;
- five visible content deltas were received before close;
- the engine returned to zero active requests in 1.078 seconds;
- no success terminal was consumed or inferred after client close.

The immediate fresh Chat request then emitted:

- exact content `M27-CHAT-AFTER-DISCONNECT-DONE`;
- 12 progressive content deltas;
- exactly one `finish_reason=stop`;
- one usage chunk with 58 prompt, 13 completion, and 71 total tokens;
- one `[DONE]` marker.

See `chat-disconnect.raw`, `chat-disconnect-summary.txt`,
`chat-disconnect-recovery.raw`, and `chat-disconnect-recovery-summary.txt`.

## Real Electron user stop

Two distinct visible stop axes were exercised in fresh chats:

1. With Auto/tool defaults still producing a long prefill, the real
   `Stop generating (Esc)` control was clicked while the UI said
   `Waiting for model response...`. The engine returned idle and SQLite retained
   only the user row (373), with no false empty assistant success.
2. With thinking Off, built-in tools Off, and Max Tokens visibly set to 4096, the
   UI progressively painted integers through 76. The same real control was then
   clicked. SQLite row 376 retains only the actual partial content plus
   `[Generation interrupted]`, 228 generated tokens, real timing metrics, no
   reasoning/tool call, and no warning. It does not pretend the 1-through-2000
   task completed.

The immediate same-chat follow-up streamed and stored exact content
`M27-ELECTRON-AFTER-STOP2-DONE` in row 379, with no warning. The engine was healthy
and idle. The temporary test settings were then restored to Auto reasoning,
built-in tools On, and blank Max Tokens.

See `electron-stop-recovery.png`, `electron-stop-recovery-ui.txt`, and
`electron-rows.json`.

## Source and regression ownership

No runtime source change was needed for this row. Existing shared ownership is:

- `panel/src/main/ipc/chat.ts` strips the display-only interruption marker from
  subsequent API history while preserving partial assistant content;
- the aborted request cleanup removes an empty assistant placeholder and stores an
  explicit interruption only when visible partial content exists;
- engine/API request cleanup returns the scheduler to idle after the HTTP consumer
  closes.

Current-source focused results:

- Python cancel/abort selection: `7 passed, 118 deselected in 1.76s`;
- panel interruption/error/recovery selection: `3 files passed, 368 tests passed`.

## Boundary

This closes Chat client-disconnect and Electron user-stop/recovery for the loaded
M2.7 artifact. It does not close safe live mid-stream engine-exception injection,
signed-app repeat, every model/parser family, gateway network loss, or prolonged
multi-session soak.

