# MiniMax M2.7 Anthropic stream and tool parity

Date: 2026-07-19

Model: `jangq-ai/MiniMax-M2.7-Small-JANGTQ`

Artifact: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`

Runtime: real Electron-started engine on `127.0.0.1:8014`

Quantization identity: JANGTQ/MXTQ (Hadamard/codebook route), not JANG affine
and not base MLX MXFP. Health retains 62 `TurboQuantKVCache` attention layers
with q4 native-TQ storage.

## Verdict

`VERIFIED-LIVE` for the scoped current-source Anthropic rows below:

- streaming thinking and visible content separation;
- non-stream visible answer and usage;
- required named tool emission;
- MiniMax orphaned native-tool opener recovery;
- tool-result continuation with retained schemas and `tool_choice:none`;
- the shared parser through a real Electron built-in tool loop.

Overall protocol parity remains `PARTIAL`. Ollama, cancellation, client
disconnect, injected mid-stream failure/recovery, signed-app repeat, and other
model/parser families are not inherited from this result.

## Issue 1: split Chat tool name became an invalid Anthropic block

### Before

`tool-empty-name-before.raw` contains a `tool_use` content block whose name was
empty even though its JSON input was valid. `chat-split-name-control.raw`
shows why: the upstream Chat stream introduced the tool id/type first, then
delivered `name=file_info` and arguments in a later delta.

### Root cause

Anthropic requires the tool name in `content_block_start`, but
`vmlx_engine/api/anthropic_adapter.py` opened the block as soon as it saw the
id. It did not buffer split Chat deltas until id and name were both known.

### Fix

Commit `c707bb61a` buffers each tool index until id and name are present, then
opens the Anthropic block and emits the accumulated argument delta.

### Live after

`tool-name-buffer-after.raw` contains one `file_info` tool block with a
non-empty name and exact `{"path":"panel/package.json"}` input, no error,
and one terminal `message_stop`.

## Issue 2: valid MiniMax invoke leaked when its outer opener was consumed

### Before

`orphan-opener-before.raw` contains a complete
`<invoke name="file_info">...</invoke></minimax:tool_call>` sequence but no
opening `<minimax:tool_call>`. The parser exposed the invoke as visible text
and the required-tool gate rejected the turn.

### Root cause

The M2.7 tokenizer/stream can consume the outer opening namespace token while
leaving a complete invoke plus the matching outer close. The MiniMax parser's
initial marker gate required the opener and never reached its native invoke
parser.

### Fix

Commit `d7f74b982` restores the opener only for the narrow unambiguous shape:
a complete invoke must precede an orphan `</minimax:tool_call>`. A standalone
visible invoke example without the outer close remains ordinary content; the
negative regression test pins that boundary. No tool name or argument is
synthesized.

### Live after

`tool-route-after.raw` emits one `tool_use` named `file_info`; its
`input_json_delta` is exactly `{"path": "panel/package.json"}`. It has no
error and exactly one terminal `message_stop`.

## Issue 3: Anthropic retained disabled schemas in the generation prompt

### Before

`tool-none-followup-before.raw` completed but prefixed the required exact
answer with a visible meta-explanation. The matched direct Chat control in
`tool-none-chat-control.raw` returned only the requested marker.

### Root cause

The `/v1/messages` route unconditionally converted and rendered public
`request.tools`, even when `tool_choice:{"type":"none"}` mapped to Chat
`tool_choice=none`. Parser state considered tools disabled, but the generation
prompt still contained the schema. Chat and Responses already used an
effective prompt-tool set; Anthropic did not.

### Fix

Commit `4a53f16e1` centralizes the effective public prompt-tool selection for
this route. `none` renders no schemas; a named choice renders only the named
schema. The same effective set drives reasoning/tool availability and DSV4
policy. Public history remains intact.

### Live after

`tool-none-followup-after.raw` contains exactly
`M27-ANTHROPIC-TOOL-DONE SIZE=5.2 KB` over 17 progressive `text_delta` events,
zero thinking deltas, zero tool blocks, no error, one `end_turn`, and one
`message_stop`.

## Ordinary Anthropic output rows

- `anthropic-stream.raw`: 205 progressive thinking deltas, 12 text deltas,
  exact `M27-ANTHROPIC-STREAM-DONE`, thinking/text block separation, and one
  terminal stop.
- `anthropic-nonstream.json`: thinking disabled, exact
  `M27-ANTHROPIC-NONSTREAM-DONE`, `end_turn`, and nonzero usage.

## Electron verification

After a real Electron Stop/Start loaded the patched Python source, a fresh
Electron chat executed exactly one built-in `file_info` call with
`panel/package.json`, consumed the real 5.2 KB result, and rendered exactly
`M27-ELECTRON-TOOL-ORPHAN-AFTER-DONE SIZE=5.2 KB`.

`electron-tool-db.json` preserves one OAI call/result, separate reasoning and
visible content, no warning, and metrics reporting 128
`paged+disk+tq-native` cached tokens. `electron-tool-final.png` and
`electron-tool-ui.txt` preserve the visible UI result.

## Regression validation

`focused-tests.txt` records 119/119 selected MiniMax, Anthropic, server,
reasoning-stream, API-surface, and audit tests passing; 894 unrelated tests
were deselected. This is a focused gate, not a replacement for the already
preserved full-suite checkpoint.

## Artifact map

- `tool-empty-name-before.raw` / `tool-name-buffer-after.raw`
- `chat-split-name-control.raw`
- `orphan-opener-before.raw` / `orphan-opener-after.raw`
- `tool-none-followup-before.raw` / `tool-none-followup-after.raw`
- `tool-none-chat-control.raw`
- `tool-route-after.raw`
- `anthropic-stream.raw` / `anthropic-nonstream.json`
- `electron-tool-db.json`, `electron-tool-final.png`, `electron-tool-ui.txt`
- `health-current.json`, `focused-tests.txt`, `commits.txt`
