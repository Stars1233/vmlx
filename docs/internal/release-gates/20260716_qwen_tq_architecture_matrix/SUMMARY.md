# Qwen cache architecture / TurboQuant live matrix

Date: 2026-07-16

Checkout: `reconcile/1.5.68` at pre-commit base
`757c6e30e007baf3afcad36d7e1615188a98a080`.

Scoped verdict: **PASS** for Qwen full-KV Auto/None selection, persisted block
representation isolation, process-restart restoration, coherent exact-once tool
continuation, and current health truth. Overall release verdict remains
**PARTIAL / NO RELEASE** because the remaining model, protocol, settings, media,
signing, and notarization rows have not all closed.

## Root cause and repair

The first live Qwen3-0.6B Auto run reported active TurboQuant objects but wrote
full-precision block records. The prompt-boundary truncation path rebuilt every
positional cache as `KVCache`, erasing `TurboQuantKVCache` identity, key/value
bits, seed, and codec policy before paged-block extraction.

The first native Q3 storage repair exposed a second, more serious red row:
restoring 896 paged tokens from Q3 records corrupted generated text and could
stall a real tool turn. Q3 is therefore retained as failed evidence, not called
working. Uncalibrated Qwen Auto now uses a correctness-first TQ8 storage codec
for actual attention-KV slots, keeps live transition disabled, and leaves
cumulative-only Qwen variants without invented KV/TQ slots. A calibrated
bundle-owned `turboquant` config remains authoritative.

Auto and explicit None previously both used `quant=none` in persistent scope
keys. Prompt and paged block namespaces now also include native-TQ enabled
state, key/value bits, transition boundary, and policy so one representation
cannot shadow the other after an Electron restart.

## Current source trace

- `vmlx_engine/utils/turboquant_config.py:78-130` classifies the Qwen cache
  architecture and applies TQ8 only to real full-KV or hybrid attention slots.
- `vmlx_engine/scheduler.py:951-985` and `1196-1208` include the effective
  native-TQ policy in block and prompt persistent namespaces.
- `vmlx_engine/scheduler.py:4162-4224` preserves `TurboQuantKVCache` identity
  and policy through prompt-boundary truncation using the full decoded state.
- `vmlx_engine/mllm_scheduler.py:488-493`, `554-566`, and `746-767` perform
  early VLM policy capture and use the same persistent namespace contract.
- `tests/test_tq_paged_block_cache.py:31-62` pins truncation identity, bits,
  seed, sink tokens, offset, and tensor shape.
- `tests/test_hybrid_live_tq_kv.py:100-143` pins full-KV TQ8 and cumulative-only
  non-fabrication behavior.
- `tests/test_engine_audit.py:5376-5404` pins block and prompt namespace parity
  in both schedulers.

## Preserved red live rows

- DB row 1908: exact final marker but **zero** tool calls because the chat was
  not fresh; this is not credited as a tool pass.
- DB row 1911: fresh real-tool attempt stalled at `Generating tool call...`
  and was interrupted after 113.4 seconds.
- DB row 1914: Q3 paged hit restored 896 tokens and emitted corrupt tool-like
  gibberish (`</tool-name...`); interrupted after 44.5 seconds.
- DB row 1917: first TQ8 cold strict-text row was coherent English but failed
  the exact-output instruction. It is not credited as strict-format passing.

The red UI stall is preserved in `qwen3-fullkv-tool-stall-red.png`. The first
native-write screenshot is retained for storage-path history only; it is not a
tool/coherence pass.

## Current live Electron proof

All UI actions used the running dev Electron app over CDP, with DB rows checked
after completion and health checked from the restarted model process.

1. TQ8 warm row 1920 returned exact `Q3KV-Q8-TOOL1-DONE`, executed exactly one
   real `file_info(panel/package.json)`, and restored 896 paged tokens.
2. After a visible Electron process restart, row 1923 returned exact
   `Q3KV-Q8-RESTART-DONE`, executed exactly one real `file_info(README.md)`, and
   restored 896 `paged+disk` tokens. Health recorded 14 native-TQ hits.
3. The UI selected Stored Cache Quantization `None` and Save & Restart launched
   PID 88783 with explicit `--kv-cache-quantization none`. Health reported
   `tq_native_enabled=false` in a separate initially-empty namespace.
4. None row 1926 returned exact `Q3KV-NONE-TOOL1-DONE`, executed one real
   `file_info(panel/package.json)`, and wrote three standard `dtype=kv` blocks.
   Two 64-token files were 14,686,026 bytes and contained `layer_*_keys` /
   `layer_*_values`, with no native-TQ tensors.
5. The UI restored `Auto` and Save & Restart launched PID 89139 with no explicit
   cache-quantization flag. Health reported `turboquant-q8`, key/value bits 8,
   `qwen_full_kv_storage_tq8`, and reopened 25 persisted blocks.
6. Auto-restored row 1929 returned exact `Q3KV-AUTO-RESTORE-DONE`, executed one
   real `file_info(README.md)`, and restored 896 `paged+disk` tokens. Final
   health recorded 14 native-TQ hits, three native-TQ writes, 30 blocks, and
   1,574 persisted tokens.

Artifacts:

- `qwen3-fullkv-q8-hit-tool-pass.png`
- `qwen3-fullkv-q8-restart-disk-pass.png`
- `qwen3-fullkv-ui-none-selected.png`
- `qwen3-fullkv-none-tool-pass.png`
- `qwen3-fullkv-ui-auto-restored.png`
- `qwen3-fullkv-auto-restored-disk-tool-pass.png`
- `q3-none-health.json`, `q3-none-after-tool-health.json`
- `q3-auto-restored-health.json`, `q3-auto-restored-after-tool-health.json`
- `none-process-argv.txt`, `auto-process-argv.txt`

## Tests

- Full affected cache/policy/campaign set: **41 passed**.
- Full `tests/test_engine_audit.py`: **581 passed**.
- Cross-file Qwen/TurboQuant/hybrid/audit selection: **212 passed**, 410
  deselected.
- Python compilation and `git diff --check` passed for all seven changed source
  and test files.
- The repository-wide Ruff baseline is not clean (496 existing findings across
  the selected large legacy files); it is not represented as a passing gate.

Final user-facing state: Qwen3-0.6B is running in `Auto`, reasoning parser
`qwen3`, tool parser `qwen`, paged cache enabled, block-disk cache enabled, and
the current model process is PID 89139. Release action remains locked.
