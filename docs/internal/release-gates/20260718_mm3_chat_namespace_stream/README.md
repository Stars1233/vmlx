# MiniMax-M3 Chat namespace streaming gate (2026-07-18)

Status: **VERIFIED-LIVE for this scoped Chat/Responses namespace and tool-loop
contract.** This does not close the broader M3 larger-video, ambiguous-glyph,
or REAP32 rows.

## Artifact and launch truth

- Electron-selected bundle:
  `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M3-Coder-Small`
- Bundle configuration read before the run:
  `model_type=minimax_m3_vl`,
  `architectures=[MiniMaxM3SparseForConditionalGeneration]`, and JANG
  `method=jang-affine-mixed`, profile `JANG_2L`. This is affine JANG, not
  JANGTQ/MXTQ.
- The real Electron Start/Server flow owned the process. The final visible
  `Save & Restart` replaced PID `67052` with PID `67856` and the Server panel
  reported `Restarted with new settings.`
- Spawned arguments included `--tool-call-parser minimax_m3`,
  `--enable-auto-tool-choice`, `--reasoning-parser minimax_m3`,
  `--use-paged-cache`, and `--enable-block-disk-cache` on port 8017.
- Post-restart `/health` was healthy with M3 loaded. The persisted block-L2
  store contained 194 blocks / 12,004 tokens before the final proof. Generic
  KV TurboQuant remains inapplicable to M3's native MSA/index cache tuple.

## Reproduced defect and source trace

The natural raw Chat tool request produced a valid
`file_info({"path":"panel/package.json"})`, but exposed one visible content
delta containing `]<]minimax[>` before the tool call.

The complete MiniMax namespace separator is `]<]minimax[>[`. The live output
stopped one character before its final `[`. Two boundaries owned the leak:

1. `vmlx_engine/tool_parsers/minimax_m3_tool_parser.py` removed only the
   complete separator, so final parsing retained the one-character-truncated
   control token as content.
2. `vmlx_engine/server.py` did not treat the namespace separator as a native
   tool-stream marker. Its first characters could therefore stream before the
   later `<tool_call>` switched Chat into buffering.

The repair strips only the complete separator and its observed terminal
one-character truncation in the M3 parser. The shared stream marker table now
recognizes the complete separator; existing partial-prefix logic withholds its
first one to three ambiguous bytes and switches to buffering at the fourth.
The display-residue cleaner removes the complete or terminally truncated
separator. Ordinary `MiniMax`/`minimax` prose is intentionally unchanged.

## Tests

- Focused exact regression: 4 passed.
- Wider M3/parser/server slice: **46 passed, 109 deselected**.
- `py_compile` passed for the parser, server, and both changed test files.
- The server regression emits the separator one character per engine chunk,
  requires zero visible content, requires the OpenAI START and populated
  argument deltas to reuse one call ID, and requires
  `finish_reason=tool_calls` plus `[DONE]`.

## Current-source raw Chat proof

First pass (`chat-tool-first-pass.sse.gz`):

- 24 separate reasoning deltas.
- 0 content deltas; no namespace/control-token leak.
- OpenAI-compatible empty START delta and populated delta shared
  `call_83e4497b`.
- Populated call was exactly
  `file_info({"path":"panel/package.json"})`.
- `finish_reason=tool_calls`, then `[DONE]`.
- 415/420 prompt tokens restored as `paged+disk`.

Real tool-result continuation (`chat-tool-continuation.sse.gz`):

- 32 separate reasoning deltas and 20 progressive content deltas.
- Visible answer: `The file panel/package.json has a size of 5.2 KB` (with
  Markdown formatting in the raw text).
- No repeated tool, no namespace marker, `finish_reason=stop`, and `[DONE]`.
- 415 cached prompt tokens were reported as `paged`.

## Current-source raw Responses proof

First pass (`responses-tool-first-pass.sse.gz`):

- 28 reasoning deltas, zero visible content, no namespace leak.
- One completed `function_call`, `call_id=call_5ec89b11`, name `file_info`,
  arguments `{"path":"panel/package.json"}`.
- Exactly one `response.completed`, no error event.

`previous_response_id` tool-result continuation
(`responses-tool-continuation.sse.gz`):

- 67 reasoning deltas and 75 progressive content deltas.
- No repeated function call and no namespace leak.
- The answer reports 5.2 KB and the stream contains exactly one
  `response.completed`, with no error event.

## Current-source Electron proof

The final fresh Electron chat used the built-in tool and persisted one exact
call and one real result:

- `file_info({"path":"panel/package.json"})`
- tool result size `5.2 KB`
- exact final visible content `MM3-NS-UI-FIX-DONE SIZE=5.2 KB`
- `warnings_json=null`
- 4,147 cached prompt tokens, `cacheDetail=paged`
- the DOM observer recorded the final answer growing through `M`, `MM`,
  `MM3`, ... to the complete marker; it did not arrive as one terminal blob.

The screenshot is preserved as `electron-tool-loop.png`. The two Reasoning
sections in the UI correspond to the pre-tool and post-tool generation
iterations; the database stores their combined reasoning separately from the
single visible final answer.

## Retained negative control

The first post-parser-only restart still emitted one visible
`]<]minimax[>` content delta before a valid tool call. That falsified the
parser-only repair and led to the stream-boundary fix above. It is intentionally
recorded here rather than reclassified as model behavior.
