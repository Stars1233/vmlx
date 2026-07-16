# Bonsai hybrid SSM L2 restore closeout — 2026-07-16

Status: `SCOPED_1BIT_EXACT_RESTART_PASS_PARTIAL_PREFIX_REPAIR_OPEN_RELEASE_LOCKED`.

This row covers the current `jangq-ai/Bonsai-27b-1bit-JANG` Electron dev
session only. It does not clear Bonsai ternary, other hybrid families, broad
sampling reliability, long context, or the release.

## Source trace

- `vmlx_engine/cli.py` no longer disables SSM disk restore solely because the
  registry classifies a model as Qwen3.5 hybrid. The disk store's explicit
  `VMLX_DISABLE_SSM_DISK_RESTORE` opt-out remains authoritative.
- `vmlx_engine/utils/ssm_companion_disk_store.py` already rejects stale record
  versions and runtime fingerprints and preserves `ArraysCache.cache`,
  `lengths`, and `left_padding` through safetensors.
- `vmlx_engine/utils/ssm_companion_cache.py::fetch_longest_prefix` now probes
  the scheduler-selected exact checkpoint boundary through `fetch()` before
  scanning the process-local length index. A fresh process can therefore
  discover a matching SSM L2 checkpoint and backfill L1.
- The Qwen/Bonsai cache remains typed: 16 full-attention KV slots use native
  storage-boundary TQ8 in Auto; 48 GatedDelta/SSM companion slots remain full
  precision.

## Live Electron evidence

All rows used the real Electron dev app, visible Chat/Server controls, built-in
`file_info`, Responses history, temperature 0 for the deterministic restart
comparison, and UI Save & Restart process replacement.

- `B1-SSML2-DET1` cold row 2049 completed one tool and exact final. After PID
  replacement, row 2052 restored 160 tokens as `paged+ssm+disk`, reconstructed
  in 0.100 seconds, executed one tool, and returned exact
  `B1-SSML2-DET1-DONE`.
- `B1-SSML2-DET2` cold row 2061 completed one tool and exact final. After a
  second independent PID replacement, row 2064 restored 168 tokens as
  `paged+ssm+disk`, reconstructed in 0.100 seconds, executed one tool, and
  returned exact `B1-SSML2-DET2-DONE`.
- DET2 restart health recorded four native-TQ block hits, two SSM disk hits,
  zero SSM disk misses, `restore_enabled=true`, and 360 aggregate
  `paged+ssm+disk` hit tokens across the two agent iterations.
- Same-process/default-temperature cache-on reliability was repeated in two
  fresh chats: rows 2037/2040/2043/2046 each made one real tool call and exact
  final. Health recorded 26 RAM block hits, five disk block hits, 23 native-TQ
  writes, and four native-TQ hits.
- The earlier row 2028 cache-on reasoning loop remains retained. Because two
  subsequent default-temperature cache-on chats passed, it is a real sampling
  reliability outlier rather than a deterministic cache-corruption proof.

## Remaining partial boundary

- A longer continuation after restart selected only a 64-token KV prefix for
  which no matching SSM checkpoint existed. The runtime correctly reported
  `no_ssm_companion_state`, released the unusable KV resume, full-prefilled,
  and still returned exact text.
- That fallback wrote a complete 64-token SSM repair checkpoint to L2. The
  source regression proves a fresh cache instance can restore a
  scheduler-selected disk boundary without a pre-populated L1 length index.
  The subsequent fresh prompt was coherent and the SSM disk hit counter moved,
  but its final row did not expose cached-token detail; therefore broad
  partial-prefix acceleration remains `PARTIAL`, not claimed green.
- Temperature was changed to 0 only for the deterministic cold/restart
  comparison. It is a test control, not a hidden runtime default or sampler
  fix.

## Verification

- `tests/test_engine_audit.py`: 581 passed.
- Hybrid cache focused suite: 166 passed with two existing warnings.
- Python compile and scoped `git diff --check`: passed.
- Evidence files in this directory include the DB rows, health snapshots,
  current argv, visible screenshots, and test outputs named above.

Release remains locked pending ternary/current-family parity, long-context and
eviction gates, protocol/settings matrix, full build, signing, and notarization.
