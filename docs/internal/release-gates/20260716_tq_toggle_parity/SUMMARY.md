# TurboQuant UI/CLI/L2 toggle parity — 2026-07-16

Scoped verdict: **PASS** for Bonsai/Qwen3.5 hybrid `Auto -> None -> Auto`
configuration and persisted-cache behavior. The wider release remains **PARTIAL**;
the Bonsai high-temperature pre-tool reasoning-verbosity row and other model-family
gates remain open.

## Source trace

- `vmlx_engine/block_disk_store.py`
  - derives native-TQ permission from the CLI-owned `VMLX_DISABLE_TQ_KV` flag;
  - rejects TQ-native reads/writes when the session explicitly selects a non-TQ
    codec;
  - evicts an incompatible TQ record during `has_block()` so its single hash slot
    can be rebuilt as a standard record by the Off run;
  - exposes `tq_native_enabled` in block-L2 stats.
- `vmlx_engine/disk_cache.py`
  - applies the same permission to prompt-cache TQ reads and writes;
  - evicts incompatible native-TQ prompt records instead of silently decoding
    them under `None`;
  - exposes `tq_native_enabled` in prompt-disk stats.
- `tests/test_tq_paged_block_cache.py` and `tests/test_tq_disk_cache.py`
  cover explicit constructor gating, CLI-environment derivation, incompatible
  record eviction, and zero native-TQ hits in Off mode.

## Live Electron evidence

Model: `jangq-ai/Bonsai-27b-1bit-JANG` on `127.0.0.1:8030`, controlled through
the current Electron dev build over CDP.

### UI/defaults

- `b1-cache-settings-auto-ui.png` shows paged KV enabled, block disk cache (L2)
  enabled, and `AUTO` / engine-selected native cache.
- `b1-tq-off-selected-ui.png` shows the user-visible `None (disable stored quant
  + live TQ-KV)` selection.
- `b1-tq-auto-restored-selected-ui.png` shows Auto restored before the final
  restart.

### Explicit Off

- The restarted process used `--kv-cache-quantization none`.
- Live telemetry reported:
  - `turboquant_kv_cache.enabled=false`;
  - `native_cache.attention_kv_storage_quantization.enabled=false`;
  - `block_disk_cache.tq_native_enabled=false`;
  - zero TQ-native hits and writes.
- Electron row 1818 returned exact `B1-TQ-OFF-REBUILD2-PASS`.
- The same Off process recorded a standard block-disk hit and 64 tokens saved
  while native-TQ hits/writes remained zero, proving L2 was still functional.
- `b1-tq-off-live3-pass.png` and `live-message-rows.json` preserve UI/DB output
  evidence from the current-source Off lane.

### Auto restored

- `auto-argv.txt` shows final current-source PID 71147 with no explicit
  `--kv-cache-quantization`
  override.
- `auto-session.json` records `kvCacheQuantization: auto`, paged cache enabled,
  and block disk L2 enabled.
- `auto-cache-stats.json` reports:
  - native TurboQuant enabled with storage policy
    `qwen_hybrid_attention_kv_storage_tq8`;
  - q8 applied to the 16 attention-KV layers only;
  - all 48 SSM/GatedDelta companion layers remain native;
  - async clean-prefill rederive remains the companion-state policy;
  - 16 disk hits, 14 TQ-native hits, one TQ-native write, and 1,024 tokens saved.
- Electron row 1824 returned exact `B1-TQ-AUTO-FINAL1-PASS` after the final
  fail-closed cleanup-race change.
- `b1-tq-auto-final1-pass.png` is the final current-source Electron visual
  result; `b1-tq-auto-live1-pass.png` preserves the immediately preceding Auto
  proof.

## Regression suite

`pytest-630.txt`: 630 passed across the two TQ disk suites, TurboQuant policy
contracts, hybrid live-TQ contracts, and the full engine audit.

## Remaining Bonsai finding

The pre-tool verbosity is not caused directly by TQ: it reproduced with TQ Off
(row 1803: one real `file_info`, exact final text, but 1,415 output tokens).
A controlled Responses diagnostic at bundle-declared sampling defaults
(`temperature=1.0`, `top_p=0.95`) used 942 output tokens / 3,604 reasoning
characters before one valid call, while explicit `temperature=0.0` used 174
output tokens / 561 reasoning characters. The bundle's own
`generation_config.json` declares temperature 1.0, top-p 0.95, top-k 20, and
sampling enabled. No hidden clamp was added. This remains **PARTIAL** pending the
1-bit-versus-ternary and parser/template A/B.
