# HY3 q4 TurboQuant decoder startup warmup - 2026-07-19

Status: VERIFIED-LIVE for the tested HY3 full-KV q4 stored-prefix path. This
is a scoped cache-latency closure, not a release claim.

## Root cause and source trace

- `vmlx_engine/tq_disk_store.py` previously retained only 32 decoder pairs,
  while the live 80-layer HY3 cache uses 80 distinct per-layer seeds. The
  sequential restore therefore thrashed the decoder LRU.
- The first persisted-prefix restore also became the first real invocation of
  the packed TurboQuant q4 encode/decode kernels. The isolated record loaded in
  milliseconds, but the live model worker paid kernel materialization inside
  `reconstruct_cache()`.
- This change retains 256 decoder pairs and materializes each exact
  bundle-derived `(key_dim, value_dim, key_bits, value_bits, seed)` codec on
  the pinned model worker before readiness. The probe uses the model's actual
  `num_key_value_heads`; it does not alter prompts, KV payloads, sampling, or
  generated output.
- `vmlx_engine/engine/batched.py` runs the warmup on the loader executor after
  scheduler construction and before `engine.start()`.
- `vmlx_engine/scheduler.py` records and logs the warmup result.

## Current-source tests

- `tests/test_tq_paged_block_cache.py` verifies distinct seeded codecs are
  retained and the real codec probes use the requested token/head shape.
- `pytest tests/test_tq_paged_block_cache.py tests/test_batching.py -q`:
  89 passed, 2 deselected.

## Live Electron proof

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP`.
- The model was stopped and started with the real Electron controls on CDP
  9335. The in-app Logs view reported:
  `configs=80 arrays=640 bytes=10583040 codec_probes=80 probe_tokens=64 probe_heads=8 seconds=6.484`.
- Before the real-codec warmup, an exact 36-token `paged+disk+tq-native`
  replay reported `reconstruction_seconds=9.688565` and Electron TTFT 9.88s.
- On the current source, the same exact prompt restored the same 36 disk/TQ
  tokens with `reconstruction_seconds=0.951356`, TTFT 1.14s, no warnings, and
  exact visible content `HY3-INIT-CONTROL-DONE`.
- A second current-source fresh-chat replay restored 82 disk/TQ tokens across
  two blocks and 160 TQ layer-blocks with
  `reconstruction_seconds=1.044455`, TTFT 1.21s, no warnings, separate
  reasoning, and the exact requested three-line visible answer.

## Artifacts

- `hy3-tq-before-warm-health.json`
- `hy3-tq-current-36-health.json`
- `hy3-tq-current-36-row.json`
- `hy3-tq-current-36-electron.png`
- `hy3-tq-current-82-health.json`
- `hy3-tq-current-82-row.json`
- `hy3-tq-current-82-electron.png`

Remaining classification: first disk/TQ restore is now bounded near one
second for these tested tail shapes, rather than the prior approximately ten
seconds. This evidence does not generalize to hybrid SSM rederive, q8 Bonsai,
or other model families; those retain their own matrix rows.
