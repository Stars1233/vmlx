# ZAYA typed CCA cache and terminal live gate

Date: 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` for typed exact RAM reuse, process-restart SSD
restore, terminal delivery, eager Electron loading, and current UI output.
Changed-suffix partial reuse is `SAFE-REJECT` because the available boundary
lacks complete path-dependent CCA state. The overall release remains blocked.

## Current source and artifact

- Source: `1aa5f8e4994b0af3df63c86a16b48e5c4bb3cd3b`
- Branch: `reconcile/1.5.68`
- Push target: `origin/codex/live-electron-gates-20260715`
- Artifact: `/Volumes/EricsLLMDrive/jangq-ai/Zaya-8B-JANG_4M`
- Bundle truth: `model_type=zaya`, `weight_format=jang`, profile `JANG_4M`.
  This is affine JANG, not JANGTQ/MXTQ and not base MLX MXFP.
- Cache truth: native `zaya_cca_v1` carrying standard KV, CCA convolution
  state, previous hidden state, and no-state MoE slots. Generic TurboQuant KV
  is disabled because the companion state is path-dependent.

## Source trace

- `vmlx_engine/scheduler.py:632-642` disables generic q4/q8 KV
  quantization for typed ZAYA CCA.
- `vmlx_engine/scheduler.py:766-778` documents the typed paged and L2
  contract and its generic-TQ boundary.
- `vmlx_engine/scheduler.py:845-856` routes prefix-only ZAYA configurations
  to typed paged storage; a generic SSD-only KV block is not sufficient to
  reconstruct CCA state.
- `vmlx_engine/prefix_cache.py:1649-1664` rejects a paged/L2 chain with KV
  pages but no terminal CCA `conv_state/prev_hs` payload.
- The shared post-terminal deferred materializer remains in
  `vmlx_engine/scheduler.py`; this current live run exercised the ZAYA branch
  after the generalized terminal-first change.

## Raw Responses proof

`zaya_typed_hierarchy.py` sent a 919-token typed prompt cold, repeated it
exactly, then changed only the suffix.

| run | HTTP | content deltas | last content -> completed | cache result |
| --- | ---: | ---: | ---: | --- |
| cold A | 200 | 39 | 0.0161 s | miss, typed blocks written |
| exact warm A | 200 | 39 | 0.0156 s | 919 tokens saved from typed L1 |
| changed suffix B | 200 | 26 | 0.0159 s | safe clean prefill |

Final pre-restart health recorded one request hit, 919 saved tokens, 15 typed
block writes, zero native-TQ writes/hits, and schema `zaya_cca_v1`. The raw
long answers were coherent code-like text but did not follow the exact-marker
instruction, so raw strict-format fidelity remains `PARTIAL`.

## Electron proof

- ZAYA was started from the real Sessions UI under gateway single-model mode.
  Before any prompt, PID 50901 reported `model_loaded=true` and
  `last_request_time=null`, proving eager materialization.
- A fresh Electron chat used temperature 0, max 128, Thinking Off, Responses,
  and tools Off. Its visible answer grew from `ZAYA-U` to exact
  `ZAYA-UI-FIRST-DONE`; SQLite persisted non-empty content, no reasoning, no
  tool call, no warning, and a 0.3-second terminal.
- A 537-token UI turn returned exact `ZAYA-UI-L2-DONE`. The first same-process
  pass reported 30 `paged+zaya_cca` tokens from prior history.
- Real Electron `Save & Restart` replaced PID 50901 with 52039. Before the
  first post-restart request, health reported loaded model, null request time,
  2,101 L2 tokens, and zero L1 tokens.
- UI Regenerate then restored 529/537 tokens as
  `paged+zaya_cca+disk`, promoted nine SSD blocks, exact-finaled in 0.4 seconds,
  and retained zero generic/native TurboQuant activity.

Both committed PNGs were opened and visually inspected: the header shows the
expected ZAYA artifact and PID, the short response is exact and non-empty, and
the post-restart row visibly shows `529 paged+zaya_cca+disk cached`.

## Validation and boundary

- Focused current-source validation: 49 passed, 1 skipped across ZAYA runtime,
  engine audit, and family-detection contracts.
- The skip is pre-existing in the broad model-family selection; all ten
  selected ZAYA numerical/runtime contracts and all five selected engine
  policy/detail contracts passed.
- ZAYA cannot truthfully support generic Paged-Off + SSD-only KV reuse today:
  disabling paged storage is auto-promoted to the typed paged path because a
  disk block without terminal CCA companion state is incomplete. This is an
  architecture-specific safety exception, not a reason to weaken the global
  Paged-Off/L2 requirement for compatible full-KV and hybrid-companion models.
- Media, tool continuation, four-protocol breadth, cancellation/disconnect,
  eviction pressure, signed-app repetition, and alternate ZAYA variants remain
  separate open rows.
