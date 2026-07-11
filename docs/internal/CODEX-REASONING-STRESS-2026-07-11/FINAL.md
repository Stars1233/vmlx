# FINAL — reasoning / streaming / sampler stress

## Verdict

**NO-SHIP** for the reasoning/streaming change set.

There are two release blockers after source fixes: the answer-pass can be starved to zero by the first reasoning pass, and the fixed Ollama/Hy3 streaming policy has not been live-retested because this lane was forbidden from restarting the 105GB engine. The live pre-fix process also failed Ollama warm greedy byte determinism.

## Live provenance

- Server: existing `http://127.0.0.1:8010`; no model start/restart.
- Model: `jangq-ai/Hy3-JANG_2K-MTP`, family `hy_v3`, batched engine, native MTP D1 active.
- Source at intake: `edb0fac4246f176138250d49983da6eac8603bdc`, equal to `origin/main`.
- Reasoning capability: `supports_thinking=true`, efforts `low/high`.
- Cache truth: TQ objects active, live encode disabled, `compress_after=0`, stored prefix q4, no resident-memory reduction claimed.
- Caps: 128 tokens for off/auto; 384 for on. No reported usage exceeded its requested cap. Ollama on-stream omitted usage and therefore fails the cap-observability requirement.

## Graded API matrix

`deltas` lists the three non-empty content-delta counts. Non-stream rows use `—`.

| Route | Mode | Path | Grade | Usage | deltas | Result |
|---|---|---:|---:|---:|---:|---|
| Chat Completions | off | non-stream | PASS | 28/31/32 | — | 3 turns, visible, no reasoning/tag leak |
| Chat Completions | off | stream | PASS | 24/32/33 | 23/31/32 | incremental and terminal usage present |
| Chat Completions | auto | non-stream | PASS | 27/34/38 | — | 3 turns, visible |
| Chat Completions | auto | stream | PASS | 27/28/35 | 26/27/34 | incremental and visible |
| Chat Completions | on | non-stream | FAIL | 202/307/384 | — | turn 3 hit cap; content was exactly `TURN1: ORBIT-`, no `FINAL-CHECK` |
| Chat Completions | on | stream | FAIL | 282/196/384 | 7/9/0 | turn 3 content exactly empty; 384 reasoning tokens; finish `length` |
| Responses | off | non-stream | PASS | 24/31/32 | — | 3-turn `previous_response_id` chain passed |
| Responses | off | stream | PASS | 26/31/31 | 25/30/30 | incremental, usage present |
| Responses | auto | non-stream | PASS | 24/31/33 | — | 3-turn chain passed |
| Responses | auto | stream | PASS | 24/32/31 | 23/31/30 | incremental, usage present |
| Responses | on | non-stream | FAIL | 305/384/384 | — | turn 2 content exactly empty; turn 3 exposed truncated/garbled visible text and no reasoning item |
| Responses | on | stream | FAIL | 186/316/384 | 13/17/0 | turn 3 content exactly empty; status `incomplete` |
| Anthropic Messages | off | non-stream | PASS | 30/35/36 | — | 3 turns, visible |
| Anthropic Messages | off | stream | PASS | 26/31/31 | 25/30/30 | incremental reasoning/content adapter path |
| Anthropic Messages | auto | non-stream | PASS | 28/34/25 | — | 3 turns, visible |
| Anthropic Messages | auto | stream | PASS | 24/32/31 | 23/31/30 | incremental, usage present |
| Anthropic Messages | on | non-stream | FAIL | 213/324/384 | — | turn 3 content exactly empty; finish `max_tokens` |
| Anthropic Messages | on | stream | FAIL | 344/206/384 | 8/11/1 | turn 3 was one diagnostic blob, not incremental answer content |
| Ollama Chat | off | non-stream | PASS | 26/34/31 | — | 3 turns, visible |
| Ollama Chat | off | stream | PASS | 28/33/36 | 27/32/35 | incremental with terminal usage |
| Ollama Chat | auto | non-stream | PASS | 30/35/37 | — | 3 turns, visible |
| Ollama Chat | auto | stream | PASS | 29/32/37 | 28/31/36 | incremental with terminal usage |
| Ollama Chat | on | non-stream | PASS | 202/276/249 | — | native non-stream delegation was correct |
| Ollama Chat | on | stream | FAIL | missing/missing/missing | 1/1/1 | final answers were misrouted to `message.thinking`; visible content was a notice |

Summary: **17/24 sequence rows passed**. All 16 off/auto rows passed. Reasoning-on passed only Ollama non-stream; 7/8 reasoning-on stream/non-stream rows failed at least one required assertion.

## Exact failure transcripts and usage

The complete exact request bodies, reconstructed content, reasoning text, individual deltas, timestamps, finish state, and usage for every failed row are in [`api-stress-failures.json`](api-stress-failures.json). The full wire-event capture is in [`api-stress.json`](api-stress.json).

Key exact visible outputs:

- Chat on, non-stream turn 3: `"TURN1: ORBIT-"`; usage `384/384`; finish `length`.
- Chat on, stream turn 3: `""`; usage `384/384`; content deltas `0`; finish `length`.
- Responses on, non-stream turn 2: `""`; usage `384/384`; status reported `completed`.
- Responses on, non-stream turn 3: `"\" and then nothing? Wait, the prompt says:\n   User: \"Recall the code and compute 7+5. Briefly answer with TURN2.\"\n   Model: (the model output was just empty/whitespace? Actually in the provided context, the assistant message after \"Recall the code...\" is empty/blank in the user turn? Let me re-read the prompt:\n   \"User: Remember code OR淋-731..."`; usage `384/384`; reasoning item missing; status `completed`.
- Responses on, stream turn 3: `""`; usage `384/384`; content deltas `0`; status `incomplete`.
- Anthropic on, non-stream turn 3: `""`; usage `384/384`; finish `max_tokens`.
- Anthropic on, stream turn 3: `"\n\n[vMLX notice] This response produced reasoning only (no visible message, no tool calls). The reasoning was preserved separately, but the visible answer is empty. Consider raising max_output_tokens or sending enable_thinking=false for the final synthesis turn."`; usage `384/384`; one content delta; finish `max_tokens`.
- Ollama on, stream turns 1–3 used that same exact notice as visible content. The exact `message.thinking` strings were respectively `"TURN1 — ORBIT-731 acknowledged."`, `"TURN2 — ORBIT-731 recalled; 7+5 = 12."`, and `"TURN1: acknowledged code ORBIT-731.\nTURN2: recalled code, computed 7+5=12.\nRemembered code: ORBIT-731.\nFINAL-CHECK."`. Terminal usage was absent on all three; finish was `length`.

## Greedy warm determinism

| Route | Grade | measured outputs | usage |
|---|---:|---|---:|
| Chat Completions | PASS | `"DET-731"` / `"DET-731"` | 4 / 4 |
| Responses | PASS | `"DET-731"` / `"DET-731"` | 4 / 4 |
| Anthropic Messages | PASS | `"DET-731"` / `"DET-731"` | 4 / 4 |
| Ollama Chat | FAIL | `"DET-731"` / `"DET-731."` | 4 / 5 |

## Source fixes made

1. Corrected raw-logit processor ordering and normalize-once behavior.
2. Kept seeded sampler state request-local by disabling shared sampling for seeded batches.
3. Prevented the legacy chat reasoning fallback from emitting before an armed answer pass.
4. Preserved visible prefixes around orphan think-close markers.
5. Applied Hy3 reasoning dialect normalization to Ollama streaming.

Unit verification after these fixes: **432 passed**. Live post-fix verification remains mandatory for Ollama streaming and a within-cap answer-pass reservation design remains open.

## Artifact integrity

- `api-stress.json`: SHA-256 `bbf54becdd8b75a3c64ecf7e553e349036ef399112bb75628f4ace8e50530167`
- `api-stress-failures.json`: SHA-256 `2935dc20a6bd8d2404e645d3b2fa8ba3af2e67cd4e38a0a8dd13ce431fe4500a`
