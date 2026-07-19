# Prompt-disk N-1 payload-prefix index proof

Date: 2026-07-19  
Source fix: `a96a44559f0c57dfe38e2b2f056f9e5e8f53fa7c`  
Model: `jangq-ai/openPangu-2.0-Flash-JANG_3M`  
Cache-fix status: **VERIFIED-LIVE**  
Overall release status: **PARTIAL**

This gate closes the openPangu Auto-reasoning prompt-disk miss without a
model-specific prompt rewrite. It also records a separate Chat Completions
usage-stream protocol failure that remains open.

## Root cause

Prompt-disk records use the full rendered N-token prompt as their lookup key,
but the serialized cache payload deliberately owns only the first N-1 tokens.
The default Auto-reasoning replay rendered the base prompt at 1,432 tokens and
the follow-up at 1,495 tokens. Their common prefix was exactly 1,431 tokens,
which is the entire persisted payload:

- base boundary token 1,432: `<think>` (`148905`);
- replayed-history boundary token 1,432: `</think>` (`148906`);
- all 1,431 payload-owned tokens before that boundary were identical.

The old lookup compared only the full N-token key and therefore rejected an
otherwise exact reusable payload. `prompt-boundary-analysis.jsonl` preserves
the token-level comparison. This was a shared prompt-disk indexing defect, not
an openPangu quant, parser, or reasoning failure.

## Source repair and safety boundary

Commit `a96a44559` adds a nullable `payload_prefix_hash` to the SQLite index
and records the hash of `tokens[:-1]` for standard and TQ-native prompt stores.
Longest-prefix lookup now permits a different token at the N boundary only
when the complete N-1 payload hash matches. It loads the existing record by
its stored full-key hash, reports N-1 cached tokens, and re-feeds the current
boundary token plus the unmatched tail.

Safety properties:

- existing exact N-token lookup remains unchanged;
- any divergence inside the payload-owned N-1 prefix is a miss;
- raw prompt tokens are not stored in SQLite;
- legacy rows with a null payload hash remain exact-only until a caller with
  the original prompt opportunistically backfills them or they are rewritten;
- all ordinary record validation and worker-owned MLX loading still run
  through the existing fetch implementation.

`source-trace.txt` records the implementation and regression-test locations.
`focused-python-tests.txt` records 84/84 current-source focused tests passing,
including changed-sentinel reuse, earlier-divergence rejection, MiniMax typed
paths, and TQ-native disk paths.

## Model and UI configuration

The real Electron dev app on CDP 9335 loaded
`openPangu-2.0-Flash-JANG_3M` through the session Start control. It is a JANG
affine `JANG_3M` artifact, not JANGTQ/MXTQ and not base MLX MXFP. The live
settings were:

- Prefix cache On;
- Paged cache Off;
- block-disk L2 Off;
- prompt-disk cache On;
- reasoning Auto;
- generic TurboQuant KV Off because openPangu owns a typed composite cache.

`electron-cache-config.png`, `electron-chat-settings.png`, and
`process-argv.txt` prove UI/argv parity. The process used
`--no-paged-cache --enable-disk-cache` and the dedicated directory
`live-proof-openpangu-auto-nminus1-20260719`. Health and logs identify 46
`OpenPanguV2LayerCache` layers and the native composite schema containing MLA
latent KV, DSA indexer state, rotating SWA state, and path-dependent causal
convolution state. Generic paged, block-disk, and TQ restore paths remained at
zero.

## Electron Auto reasoning after process replacement: PASS

The base turn produced a fresh, separate 585-character reasoning rail and the
exact visible answer `OP-TYPED-BASE-DONE`, then wrote the 1,432-token record.
Electron Stop/Start replaced the server process. The longer same-chat turn:

- restored 1,431/1,495 prompt tokens from disk;
- recomputed the current boundary token and unmatched tail;
- produced a different 1,141-character reasoning rail, so no byte-identical
  stale reasoning replay occurred;
- returned exactly `OP-TYPED-PARTIAL-DONE STATE-49059`;
- completed without warnings at 0.68s TTFT.

The log records `Disk cache N-1 payload prefix hit` and explicitly says the
current boundary token is re-fed. Evidence is in `electron-rows.json`,
`electron-auto-base.png`, `electron-auto-partial.png`,
`electron-auto-partial-logs.png`, and the Electron health snapshots.

## Raw Responses after independent Electron restart: PASS

After another UI Stop/Start, a streamed Responses request replayed the same
base history and requested the retained fact. It:

- restored 1,431/1,495 input tokens with `cache_detail=disk`;
- emitted 262 non-empty reasoning-summary deltas separately from 15
  progressive output-text deltas;
- returned exactly `OP-API-AUTO-N1-DONE STATE-49059`;
- emitted reasoning done, output-text done, output-item done, and
  `response.completed` with status `completed`.

See `responses.sse`, `responses-analysis.json`, and the Responses health
snapshots.

## Raw Chat: cache/stream PASS, usage protocol FAIL

After another UI Stop/Start, Chat Completions independently:

- restored 1,431/1,497 prompt tokens with `cache_detail=disk`;
- emitted 300 non-empty `reasoning_content` deltas separately from 16
  progressive `content` deltas;
- returned exactly `OP-CHAT-AUTO-N1-DONE STATE-49059`;
- emitted `finish_reason=stop` and `[DONE]`.

However, `stream_options.include_usage=true` attached a non-null usage object
to 317 intermediate chunks, with completion counts growing from 1 to 318.
OpenAI-compatible streaming requires ordinary chunks to omit/null usage and a
single final usage chunk before `[DONE]`. This is a distinct shared protocol
defect, not a cache restore failure. `chat.sse` and `chat-analysis.json`
preserve it, and `source-trace.txt` points to the intentional per-chunk usage
emission in `vmlx_engine/server.py`.

## Current verdict and remaining gates

- `OPENPANGU-AUTO-REASONING-PROMPT-DISK-PARTIAL`: **VERIFIED-LIVE** for
  Electron, Responses, and Chat cache/content/reasoning/terminal behavior.
- `CHAT-STREAM-INCLUDE-USAGE-PARITY`: **OPEN / FAIL** because usage is emitted
  on 317 intermediate chunks.
- The parent cache campaign remains **PARTIAL**. A paged-compatible model still
  must prove block-aligned partial RAM reuse, forced eviction, block-disk L2
  refault, and restart restore. openPangu cannot close that row because its
  native path-dependent composite state intentionally disables generic paged
  blocks.
- Tool calls were not requested in this cache-specific gate; tool-loop parity
  remains governed by the separate parser/family matrix.
- No release, packaging, signing, notarization, or publication claim follows
  from this scoped closure.
