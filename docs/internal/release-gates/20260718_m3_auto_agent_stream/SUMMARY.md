# MiniMax-M3 Auto reasoning and agent-stream proof

Date: 2026-07-18

## Verdict

`VERIFIED-LIVE` for MiniMax-M3 text Auto reasoning, separate progressive
reasoning/content emission, required tool generation, real tool-result
continuation, and clean terminal behavior through Electron, Responses, and
Chat Completions. The overall M3 family and release remain `PARTIAL` because
larger media, OCR exactness, terminal delay, and REAP32 headroom are separate
open rows.

## Bundle-grounded runtime contract

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M3-Coder-Small`.
- `config.json`: `model_type=minimax_m3_vl`, architecture
  `MiniMaxM3SparseForConditionalGeneration`, real vision configuration.
- `jang_config.json`: affine mixed `JANG_2L`, REAP 70/128 experts,
  `gqa+msa_sparse`, `kv+msa_index_dual`, tools/thinking/vision advertised.
  This is not JANGTQ/MXTQ and not base MLX MXFP.
- `generation_config.json`: sampling defaults temperature 1.0 and top-p 0.95.
- Session config: reasoning parser `minimax_m3`, tool parser `minimax_m3`,
  Auto cache policy, prefix/paged/block-disk enabled, generic TQ incompatible.

## Source trace

- `vmlx_engine/server.py:6144-6178` maps the public Auto/On/Off contract to
  M3's real `adaptive`/`enabled`/`disabled` template vocabulary.
- `vmlx_engine/server.py:19047-19124` derives the M3 reasoning seed from the
  rendered prompt and prevents internal reasoning from entering visible
  content.
- `panel/src/main/ipc/chat.ts:2503-2555` handles Responses reasoning and output
  deltas as separate rails; `:2611` handles progressive function arguments;
  terminal dispatch is handled around `:2459` and `:2709`.

## Real Electron proof

The real Sessions drawer Start button loaded M3 PID 19963 and automatically
stopped DSV4. The saved one-model screenshot, SQLite session rows, and process
argv show exactly one running model. Before the first prompt, health reported:

- `model_loaded=true`, `last_request_time=null`;
- native cache schema `minimax_m3_msa_v1`;
- dense KV layers 0-2 and sparse MSA/index layers 3-59;
- prefix/paged/block-disk enabled;
- generic TurboQuant KV disabled for the native MSA tuple.

The fresh chat visibly showed Auto reasoning, Responses wire, and built-in
tools enabled. It initially reflected bundle defaults temperature 1.0 and
top-p 0.95. The controlled stream proof applied temperature 0 and max 512
while leaving Auto selected.

### Ordinary Auto turn

Electron row 201 stored 2,065 reasoning characters separately from 541 visible
characters. Timed screenshots show:

- 1 second: waiting for the model;
- 3 seconds: 42 reasoning characters visible;
- 6 seconds: 413 reasoning characters visible;
- later: visible content begins while the reasoning rail stays separate.

The model refused the synthetic requested `M3-*` marker as prompt injection,
but the response was coherent, non-empty, separately streamed, and complete.

### Same-chat real tool turn

Electron row 204 called `file_info({"path":"panel/package.json"})` exactly
once, received the real 5.2 KB metadata, and stored no warning. During the
turn, reasoning grew progressively to 1,851 characters; visible content then
grew from 178 to 394 characters before finalization. The request reused 9,004
tokens as `paged+disk` and did not repeat the tool or fabricate its result.

## Raw API proof

`raw-chat-responses.json` preserves every timed SSE event; `raw-summary.json`
is the compact review surface.

| API case | Reasoning deltas | Content/argument deltas | Terminal |
|---|---:|---:|---|
| Responses no-tool Auto | 262 | 46 content | `response.completed` |
| Responses required tool | 19 | 2 arguments | valid `file_info`, completed |
| Responses real-result follow | 15 | 119 content | completed |
| Chat no-tool Auto | 304 | 51 content | `stop`, `[DONE]` |
| Chat required tool | 19 | 2 tool deltas | `tool_calls`, `[DONE]` |
| Chat real-result follow | 37 | 125 content | `stop`, `[DONE]` |

Both protocols returned coherent non-empty visible answers after reasoning.
No reasoning marker leaked into content, no answer was emitted as one delayed
batch, no tool call was truncated, and no stream ended without its protocol
terminal.

## Evidence index

- `raw-chat-responses.json`, `raw-summary.json` — complete and compact raw
  Chat/Responses SSE proof.
- `electron-message-rows.json` — persisted ordinary/tool Electron rows.
- `health-after-electron-start.json`, `health-current.json`,
  `capabilities-current.json` — runtime/cache/parser truth.
- `electron-start-clicked.png`, `electron-one-model-loaded.png` — real UI load
  and one-model swap.
- `electron-settings-auto-default.png`,
  `electron-settings-auto-temp0-512.png` — visible settings state.
- `electron-auto-*.png`, `electron-tool-*.png` — timed progressive UI proof.

## Retained boundary

The artifact's refusal to reproduce synthetic marker tokens is retained as
model policy/strict-format behavior. No prompt coercion, hidden sampler clamp,
fabricated tool arguments, or output rewrite was added. Larger-video, digit
OCR, terminal-delay, and REAP32 memory-headroom gates remain open.
