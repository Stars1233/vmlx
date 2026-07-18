# Gemma 4 mixed-SWA TQ4 cache/settings evidence — 2026-07-16

Status: `PASS-LIVE` for scoped cache topology, Auto/None parity, tool
continuation, process restart, bounded eviction, and L2 reload. Coherent
constrained long output remains `PARTIAL`; the release remains locked.

## Source trace

- `vmlx_engine/utils/jang_loader.py:1670-1688` classifies mixed sliding/full
  attention and lets normal Auto enter the selective path.
- `vmlx_engine/utils/jang_loader.py:1793-1804` requires the detected layout to
  map one-to-one to native cache slots and fails closed otherwise.
- `vmlx_engine/utils/jang_loader.py:1861-1871` applies the mixed-SWA policy
  only after the native cache layout is known.
- `vmlx_engine/utils/turboquant_config.py:142-173` sets q4, names only full
  attention slots as critical, uses `compress_after=0`, and stamps
  `mixed_swa_full_attention_kv_storage_tq4`.
- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx:368-431`
  distinguishes mixed SWA from SSM/GLA and reports explicit live-TQ off state.
- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx:1076-1103`
  renders the architecture hint and live codec label/badge.

Code commits:

- `3385cb019` — mixed-SWA selective TQ4 Auto policy and regression tests.
- `ba68f8fba` — settings copy/status parity and regression tests.

## Focused verification

- Python mixed-SWA/TurboQuant/cache selection: 153/153 passed.
- Panel settings flow: 281/281 passed.
- Panel TypeScript typecheck passed.

## Live Electron proof

The isolated dev Electron profile was
`/Users/eric/.vmlx-v1611-cachefix-dev`, CDP `127.0.0.1:9335`. The model was
`jangq-ai/gemma-4-12B-it-qat-JANG_4M` on port 8009.

- Auto rows 2425 and 2428 executed exactly one requested `file_info` call and
  exact finals. Auto process restart row 2431 restored 704 tokens as
  `paged+mixed_swa+disk` and block L2 recorded 44 native-TQ hits.
- The UI set `max_blocks=16`; process PID 14826 launched with the same value.
  Distinct Electron chats raised scheduler and total L1 evictions to 38.
  Post-eviction row 2464 replayed the older ALPHA prompt, restored 704 tokens
  as `paged+mixed_swa+disk`, executed exactly one
  `file_info(pyproject.toml)`, and returned only `G4-EV3-ALPHA-DONE`.
- The UI restored 1,000 blocks and selected None. PID 15388 launched with
  `--kv-cache-quantization none`; row 2467 executed one real
  `file_info(panel/package.json)` and returned only `G4-NONE1-DONE`.
  Health then showed three ordinary block-disk writes,
  `tq_native_writes=0`, `tq_native_hits=0`, and
  `tq_native_enabled=false`.
- The UI restored Auto. PID 15797 launched with 1,000 blocks and no explicit
  quantization override. Final row 2470 restored 704 tokens as
  `paged+mixed_swa+disk`, executed one real
  `file_info(vmlx_engine/server.py)`, and returned only
  `G4-AUTO-FINAL1-DONE`. Health showed three native-TQ writes, eleven
  native-TQ hits, and `tq_native_enabled=true`.
- The hot-reloaded current Electron drawer visibly showed
  `TQ4 full-attention KV + native rotating SWA` and `MIXED AUTO`.

## Screenshots

- `gemma4-maxblocks16-setting.png` — bounded L1 setting before restart.
- `gemma4-eviction-l2-replay.png` — correct post-eviction L2 tool replay.
- `gemma4-explicit-none-setting.png` — explicit None UI state.
- `gemma4-explicit-none-live.png` — correct tool completion under None.
- `gemma4-auto-final-live.png` — final correct Auto tool completion.
- `gemma4-mixed-auto-ui-parity.png` — final mixed-SWA label and Auto badge.

## Still open

- Coherent constrained long-output/strict-marker reliability under the final
  Auto configuration.
- This scoped PASS does not clear HY3 depth-3/cache, Qwen 27 MTP reliability,
  Laguna latency/unsolicited-tool, DSV4 long quality, M3 exact OCR, or the
  broader protocol/release matrix.
