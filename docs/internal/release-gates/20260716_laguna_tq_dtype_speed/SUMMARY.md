# Laguna TurboQuant dtype and warm-cache speed gate

Date: 2026-07-16

Status: scoped cache-speed `PASS`; Laguna overall `PARTIAL`; global release
`PARTIAL_NO_RELEASE`.

## Root cause and source trace

The real `jang_tools.turboquant.pipeline.decode_keys/decode_values` surface
returns float32. `TurboQuantKVCache.compress()` retained that decoded dtype in
its joined live buffer, while vMLX TQ block and prompt records omitted the
model's original KV dtype. A 49-token, 70-layer Laguna restore therefore held
28,098,560 bytes of float32 KV and decoded at about 8 tok/s, versus about
25 tok/s cold.

Current source repairs every owning surface:

- `/Users/eric/jang/jang-tools/jang_tools/turboquant/cache.py` records the
  incoming key/value dtype and casts decoded buffers back before building the
  joined attention cache.
- `vmlx_engine/tq_disk_store.py` records and restores key/value dtype for
  positional TQ blocks, native prompt records, and nested CacheList records.
- `vmlx_engine/block_disk_store.py` carries those dtype strings through the
  safetensors metadata round trip.
- `vmlx_engine/cache_record_validator.py` rejects missing or unsupported TQ
  dtype metadata before restore.
- `vmlx_engine/prefix_cache.py` adds `tq_storage_schema=dtype_v1` only to TQ
  model keys, preventing legacy dtype-less L2 records from entering the new
  restore path without invalidating non-TQ caches.
- `vmlx_engine/scheduler.py` rewraps reconstructed full-KV layers with the
  model's native TQ template.
- `vmlx_engine/mllm_batch_generator.py` preserves dtype metadata when it
  performs that rewrap.

## Live Electron proof

Real bundle: `/Volumes/EricsLLMDrive/jangq-ai/Laguna-M.1-JANG_2L`

Real app/session: Electron dev build over CDP `127.0.0.1:9335`, model server
`127.0.0.1:8015`.

Prompt (fresh UI chats, thinking Off, temperature 0, top-p 1, min-p 0,
max-output 512, built-in tools Off):

`[LAG-DTYPE-COLD-1] Output the integers 1 through 100 in ascending order,
separated by single spaces. Output only the integers and spaces; no punctuation
or explanation.`

All accepted rows returned exactly `1 2 ... 100`, with no reasoning payload.

| Arm | Electron evidence | Cache | Decode | TTFT | Total |
| --- | --- | --- | ---: | ---: | ---: |
| New dtype namespace cold | `laguna-dtype-cold-pass.png` | none | 25.0 tok/s | 0.84 s | 12.8 s |
| Same-process warm | `laguna-dtype-warm-pass.png` | 49 paged+tq-native | 21.2 tok/s | 0.87 s | 14.7 s |
| UI Save & Restart, PID 7811 -> 9056 | `laguna-dtype-l2-pass.png` | 49 paged+disk+tq-native | 24.6 tok/s | 2.53 s | 14.6 s |
| Final current-source restart, PID 9056 -> 10474 | `laguna-dtype-l2-current-source.png` | 49 paged+disk+tq-native | 25.1 tok/s | 3.16 s | 14.9 s |

Electron Logs after restart show:

`Re-wrapped 70 KV layers as TurboQuant objects: encoded=0,
resident_before=14049280, resident_after=14049280, delta=+0 bytes`

14,049,280 bytes is exactly the 2-byte KV footprint for 49 tokens, 8 KV
heads, 128 head dimension, keys+values, and 70 layers. The prior float32
restore occupied twice that amount. Health also reported one native TQ disk
hit, a successful 70-layer rewrap, and 0.183291 s reconstruction.

Screenshots:

- `laguna-dtype-cold-pass.png`
- `laguna-dtype-warm-pass.png`
- `laguna-dtype-l2-pass.png`
- `laguna-dtype-l2-logs.png`
- `laguna-dtype-l2-current-source.png`
- `laguna-dtype-l2-current-source-logs.png`

## Tests

- JANG TurboQuant cache: 24 passed, including float16 and bfloat16 compression
  dtype preservation.
- vMLX TQ/paged/batching group: 115 passed, 2 intentionally deselected by the
  file-level collection configuration.
- vMLX block/prefix compatibility group: 46 passed.

## Still red / not claimed

- Laguna reasoning-on is not closed. A fresh Electron chat using the bundle's
  Auto/default reasoning generated repetitive meta-reasoning about spacing and
  was manually interrupted after 726 tokens. This is a visible runtime failure,
  not counted as a parser/model pass.
- The server advertises `--enable-jit`, but Logs state that live compile is
  skipped when TurboQuant KV objects are active. A controlled UI `Auto` versus
  explicit `None` cache/JIT comparison remains open.
- Tool parsing, multi-turn interleaved reasoning, long-context stability, and
  exact settings persistence remain open for Laguna.
- No package version bump, PyPI upload, app release, signing, notarization,
  tag, or feed update is authorized by this scoped proof. Release stays locked.
