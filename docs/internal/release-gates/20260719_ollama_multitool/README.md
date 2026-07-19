# Ollama and Electron two-tool continuation gate

Date: 2026-07-19

Status: `VERIFIED-LIVE` for two simultaneous Ollama tool calls, both real
results, streamed post-tool continuation, and the matching real Electron built-in
tool loop on MiniMax-M2.7. Overall protocol and release status remains `PARTIAL`.

## Artifact and process truth

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`.
- Quant family: JANGTQ/MXTQ codebook, not affine JANG and not base MLX MXFP.
- The process was PID 95088, loaded by the real Electron Server Stop/Start action
  on current source. The active endpoint was port 8014.
- The session used Auto reasoning, Responses wire mode, built-in coding tools On,
  working directory `/Users/eric/mlx/vllm-mlx`, and the default ten tool rounds.

## Raw Ollama agent loop

The first `/api/chat` stream was instructed to call exactly two tools before
answering. It emitted:

- 19 separate thinking rows, 100 thinking characters;
- exactly one terminal with `done_reason=tool_calls`;
- exactly two schema-valid calls in the same terminal:
  - `file_info({"path":"panel/package.json"})`
  - `run_command({"command":"pwd"})`
- object-valued arguments on the Ollama wire, not encoded JSON strings.

The harness executed only those exact requested operations. It returned the real
5.2 KB file metadata and `/Users/eric/mlx/vllm-mlx` working directory as separate
Ollama `role=tool` messages. The next stream emitted:

- 43 fresh thinking rows, 119 thinking characters;
- 30 progressive visible content rows;
- exact content
  `M27-OLLAMA-MULTI1-DONE SIZE=5.2 KB PWD=/Users/eric/mlx/vllm-mlx`;
- no repeated tool call;
- exactly one `done_reason=stop` terminal.

See `ollama-multitool.raw` and `ollama-multitool-summary.txt`.

## Real Electron agent loop

The same two-tool contract was run in a fresh Electron chat, not only through the
raw endpoint. The visible UI showed two separate reasoning rails, `Info` for
`panel/package.json`, `$ pwd`, and the exact final marker. SQLite row 372 records:

- two distinct call ids and exactly one call for each expected tool;
- exact object arguments for both calls;
- both real results associated with their matching ids;
- non-empty fresh reasoning before and after tool execution;
- exact non-empty final content;
- `warnings_json=null`;
- 192 cached tokens with detail `paged+disk+tq-native`;
- 8,090 prompt tokens, 288 generated tokens, 35.0 tok/s, 26.85s TTFT, and
  63.2s total.

The engine was healthy with zero active requests after completion. See
`electron-row.json`, `electron-final-ui.txt`, and `electron-final.png`.

## Regression coverage

Commit `1b35d7a9bfaad4675d4825d462ee112919fb4cf8` adds adapter tests that
pin both sides of the live contract:

- two assistant tool calls plus two named Ollama tool-result messages survive
  request conversion intact;
- a two-call OpenAI terminal becomes one Ollama `done_reason=tool_calls` row with
  object arguments and retained usage.

Focused result: `31 passed, 119 deselected in 1.80s`.

## Boundary

This closes the M2.7 raw Ollama plus Electron two-tool child row. It does not prove
every parser family, signed-app behavior, cancellation on Ollama/Chat, media tool
turns, or repeated long-loop soak. Those remain separate `PARTIAL` rows.

