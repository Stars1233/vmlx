# DSV4 v8 typed composite cache: Electron and Responses proof

Date: 2026-07-18
Source base: `ec7682171606399096bfdd9e2e6e61df2d54815f` plus the scoped diff

## Verdict

`VERIFIED-LIVE` for the deterministic DSV4 native composite prefix-cache and
one-tool streaming loop. Overall release status remains `PARTIAL`: stochastic
strict-format behavior and broader long quality/performance are separate open
rows.

## Artifact and architecture truth

- Bundle: `dealignai/DeepSeek-V4-Flash-JANG-CRACK`.
- Quant format: affine JANG, not JANGTQ/MXTQ and not base MLX MXFP.
- Runtime cache: 43-layer native DSV4 composite with local SWA and compressed
  CSA/HCA pools. `PoolQuantizedV4Cache` is its native incremental q4 pool
  codec; it is not generic TurboQuant KV.
- Current `/health::native_cache` reports schema `deepseek_v4_v8`, pool quant
  enabled, prefix/paged/block-disk enabled, and generic TurboQuant KV forced
  off. The disk store recorded zero `tq_native_writes` and zero
  `tq_native_hits`.

## Root cause and source trace

1. `vmlx_engine/utils/dsv4_batch_generator.py:360-375` now retains
   `PoolQuantizedV4Cache` when taking a clean prompt-boundary snapshot.
2. `vmlx_engine/prefix_cache.py:3864-3935` reconstructs the stored class rather
   than silently downgrading it to `DeepseekV4Cache`.
3. `vmlx_engine/scheduler.py:4725-4747` records the native pool-quant flag.
4. `vmlx_engine/utils/dsv4_batch_generator.py:821-863` uses the safe default
   that realizes cache-hit tail prefill before allocator clearing. The removed
   `realize_before_clear=False` overrides could leave lazy cache side effects
   incomplete.
5. The persisted namespace moved from v7 to v8 so incompatible blocks created
   after the prior silent class downgrade cannot replay.

## Source tests

The current scoped suite completed with `813 passed, 1 skipped` across DSV4
generation, paged/L2 reconstruction, cache-bypass, hardening, engine-audit,
and cross-matrix coverage. New assertions pin prompt-snapshot type,
scheduler metadata, paged reconstruction type, and cache-hit realization.

## Live Electron proof

- Pool-codec resident A/B: rows 189 and 192 produced byte-identical reasoning
  and `DSV4-UI-CACHE-POOLFIX3-DONE SIZE=5.2 KB`, each with exactly one real
  `file_info({"path":"panel/package.json"})`. Row 192 restored 340 tokens as
  `paged+dsv4`.
- v8 disk cold/restart: row 195 wrote two v8 blocks covering 338 tokens. The
  app was stopped and restarted through the real Electron UI without clearing
  L2. Row 198 restored 338 tokens as `paged+dsv4+disk`, repeated the exact
  reasoning and final, and executed exactly one real tool. Health recorded two
  disk hits and zero generic TQ activity.
- The saved screenshots show the real Electron chat, current model, settings,
  tool result, and cold/resident/restart outcomes. `electron-message-rows.json`
  preserves the corresponding SQLite records.

## Raw Responses streaming proof

`raw-responses-sse.json` and its compact summary preserve timed SSE events:

- cold: 78 reasoning deltas, two argument deltas, one completed terminal;
- warm: the same normalized call, 311 cached tokens as `paged+dsv4`, first
  delta 0.220657 seconds;
- skip-prefix control: the same normalized call without cache reuse;
- real tool-result continuation: 15 progressive content deltas, matching
  output-text done, exact `DSV4-RAW-V8-DONE SIZE=5.2 KB`, and completed status.

Cold, warm, and skip-control normalized outputs are equal. There is no
reasoning-only final, fabricated result, incomplete native call, or batched
post-tool answer in this deterministic proof.

## Evidence index

- `raw-responses-sse.json` — complete timed Responses event capture.
- `raw-responses-summary.json` — compact event/output summary.
- `electron-message-rows.json` — persisted cold/resident/restart rows.
- `health-before-restart.json`, `health-after-restart.json`,
  `health-current.json`, `capabilities-current.json` — cache/runtime truth.
- `electron-poolfix-cold.png`, `electron-poolfix-warm.png`,
  `electron-v8-cold.png`, `electron-v8-disk-restore.png` — live chat proof.
- `electron-settings-cold.png`, `electron-settings-restart.png` — applied UI
  generation settings.

## Open boundary

This proof does not erase earlier stochastic temperature-0.6 exact-format or
long-output quality/performance failures. Those remain `PARTIAL`; no bundle
sampling mutation or synthetic output repair was introduced.
