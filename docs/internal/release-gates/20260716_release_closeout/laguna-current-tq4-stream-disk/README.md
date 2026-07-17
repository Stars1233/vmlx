# Laguna current-head reasoning, streaming, and TQ4 restart proof

Date: 2026-07-17

Verdict: `PASS-LIVE` for this scoped Electron/API/restart sequence; `PARTIAL`
for strict raw byte formatting, long/eviction coverage, and restart latency.

## Source trace

- `vmlx_engine/utils/turboquant_config.py:149-181` selects storage-only TQ4
  for non-Bonsai full-KV and selective-attention Auto policies; Bonsai retains
  its explicit TQ8 policy.
- `vmlx_engine/scheduler.py:5928-5984` reconstructs TQ-native paged/L2 blocks
  on the model worker and restores the live `TurboQuantKVCache` class without
  re-encoding the stored values.
- `vmlx_engine/server.py:8416-8433` reports Laguna as plain `paged_kv` with
  attention-KV, prefix, paged, and block-disk tiers.
- Panel commit `e9af64474` uses the effective
  `turboquant_kv_cache.storage_*` fields for the Cache/Perf `Attention KV L2`
  card and hides the contradictory disabled legacy card while TQ storage owns
  the prefix.

## Live Electron sequence

The dev Electron app was attached over CDP at `127.0.0.1:9335`. Laguna ran
from `/Volumes/EricsLLMDrive/jangq-ai/Laguna-M.1-JANG_2L` with Responses wire,
reasoning Auto, built-in tools enabled, no max-token override, paged cache,
64-token blocks, 1,000 maximum blocks, and block-disk L2.

The identical prompt was used in three fresh chats:

`[LAG-TQ8-COLD1] Compute 17 + 28. Use reasoning mode, then reply with exactly TQ8-COLD1=45 and no other visible text.`

The historical marker name is retained for exact cache-key continuity; current
health proves the active storage codec is TQ4.

| DB row | Process/tier | Prompt/cache | TTFT / total | Result |
|---|---|---:|---:|---|
| 327 | PID 64431 first current run | 4,178 / no hit | 7.87s / 14.9s | exact visible content; separated reasoning; no tool payload; no warning |
| 330 | PID 64431 resident replay | 4,178 / 4,174 `paged+tq-native` | 2.50s / 6.2s | exact visible content; no tool payload; no warning |
| 333 | PID 65648 after visible Stop/Start, no L2 clear | 4,178 / 4,174 `paged+disk+tq-native` | 5.22s / 8.9s | exact visible content; no tool payload; no warning |

This current three-row sequence did not reproduce historical row 2022's
unsolicited `ask_user`. That old failure remains a reliability control rather
than being erased by this scoped pass.

## Raw Responses streaming

`/usr/bin/curl -N` sent a separate reasoning-on Responses request. Reasoning
deltas arrived continuously from +1.002s through +20.918s. Nine distinct
`response.output_text.delta` events then arrived from +20.999s through
+21.326s, followed by one `response.completed` at +21.384s. The content was
`\nLAG-API-CURRENT=45\n`; progressive streaming passed, but the surrounding
newlines keep raw byte-exact formatting `PARTIAL`.

## Cache and UI telemetry after restart

- `cache_detail=paged+disk+tq-native`, `cached_tokens=4174`, 66 blocks.
- Reconstruction succeeded in 1.193189s; live TQ rewrap succeeded in
  0.000369s.
- `stored_prefix_quantization=turboquant-q4`, K q4, V q4,
  `storage_encode_enabled=true`, `compress_after=0`.
- Block L2 reported 66 disk hits and 66 TQ-native hits for the restart request.
- The live Cache and Perf panels both visibly displayed
  `Attention KV L2 turboquant-q4 (K q4 / V q4)` after the panel repair.

The 5.22s restart TTFT is still materially slower than the 2.50s resident
replay despite about 1.19s measured worker reconstruction. No broad latency,
long-context, or eviction claim is made from this sequence.

## Validation

- `npm test -- --run tests/settings-flow.test.ts`: 283/283 passed.
- `npm run typecheck`: passed.
- Live screenshots in this directory capture cold/resident/restart output and
  the corrected Cache/Perf telemetry.
