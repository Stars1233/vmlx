# 2026-07-22 cache partial SSD proof — Bonsai/Laguna/Gemma 4

Status: `PARTIAL / RELEASE-CRITICAL`.

Source checkout: `/Users/eric/mlx/vllm-mlx-r16-reasoning-p0` at
`4b6ec9378613872e5714a81a469775b8b1f51a92` before the local UI-policy test
addition.

Live Electron checkout: `/Users/eric/mlx/vllm-mlx-r16-reasoning-p0-live` on
`erics-m5-max.local`, started through the real Electron session manager /
preload IPC. This is the same app path as the Start button; no SQLite session
mutation was used for model start/stop.

## What this gate proves

This gate specifically answers the emergency question: if a user turns
**In-Memory Paged Cache (RAM)** on or off while **Block Disk Cache (SSD / L2)**
is enabled, can the runtime still discover reusable partial prefix blocks from
SSD across changed suffixes and process restarts?

It does **not** close global cache hierarchy:

- Gemma 4 was not run in this pass because the current Electron userData has no
  existing Gemma session.
- Low-capacity `Max Cache Blocks` / `Block Cache Max (GB)` eviction and refault
  were not live-run here.
- DSV4 composite, MiniMax M3 sparse/MSA, openPangu prompt disk, CCA/ZAYA, and
  media-salted cache rows remain separate gates.

## Bonsai 27B ternary JANG affine

Model:
`/Volumes/EricsLLMDrive/dealignai/Bonsai-27b-Ternary-JANG-CRACK`.

Electron/API behavior was also checked while this model was loaded:

- UI turn 1: Auto reasoning rail was separate, visible output exactly
  `BONSAI-R16-UI-T1-DONE`.
- UI turn 2: multi-turn recall passed, visible output exactly
  `BONSAI-R16-UI-T2-DONE PREV=BONSAI-R16-UI-T1-DONE`, and the prompt reported
  `61 paged+ssm+tq-native cached`.
- UI turn 3: the real built-in `file_info` card was emitted once for
  `panel/package.json`, final text exactly
  `BONSAI-R16-UI-T3-DONE SIZE=5.2 KB`.
- Gateway Chat/Responses API: with sufficient thinking budget, reasoning deltas
  were separate and visible content was exactly
  `BONSAI-R16-API-CHAT-B-DONE` and `BONSAI-R16-API-RESP-B-DONE`.
- Gateway Chat required-tool row emitted tool calls with `finish_reason:
  tool_calls`; full API-side result-continuation remains a broader
  `R16-AGENTIC-HARNESS` row.

Durable API artifact:
`../20260722_bonsai_r16_ui_api/bonsai-api-gateway-proof.json`.

### Bonsai Paged-On + SSD/L2

Paged RAM was restored on before this row. Block Disk Cache remained on.

| Artifact | Request | Result |
|---|---|---|
| `bonsai_paged_on_store/summary.json` | same-process warm A | `9781` cached tokens, `paged+ssm+tq-native`, exact marker |
| `bonsai_paged_on_store/summary.json` | changed-tail partial B | `9728` cached tokens, `paged+ssm+tq-native`, exact marker |
| `bonsai_paged_on_probe_after_ui_restart/summary.json` | restart A | `9781` cached tokens, `paged+ssm+disk+tq-native`, `disk_hit=true`, exact marker |
| `bonsai_paged_on_suffix_c_partial_after_restart/summary.json` | never-stored suffix C after restart | `9728` cached tokens, `paged+ssm+disk+tq-native`, `disk_hit=true`, exact marker |

Interpretation: with Paged Cache on, same-process reuse can hit the RAM tier,
and after restart a new changed suffix can restore the shared prefix from SSD
and promote into the paged tier. Bonsai’s cache detail keeps the expected
hybrid SSM and TQ-native identity.

### Bonsai Paged-Off + SSD/L2 disk-only

The session was updated through Electron `sessions.update` plus stop/start to
`usePagedCache=false`, `enableBlockDiskCache=true`. Pre-probe health reported:

- `backend_mode=block_disk_only`
- `paged_ram_enabled=false`
- `disk_only=true`
- `ram_tokens_cached=0`
- Bonsai native cache:
  `attention_kv_storage_quantization.bits=8`,
  `auto_policy=bonsai_hybrid_attention_kv_storage_tq8`,
  `ssm_policy=native_companion_state`.

| Artifact | Request | Result |
|---|---|---|
| `bonsai_paged_off_disk_only_store/summary.json` | same-process warm A | `9781` cached tokens, `block-disk+ssm+tq-native`, `disk_hit=true`, exact marker |
| `bonsai_paged_off_disk_only_store/summary.json` | changed-tail partial B | `9728` cached tokens, `block-disk+ssm+tq-native`, `disk_hit=true`, exact marker |
| `bonsai_paged_off_suffix_d_partial_after_restart/summary.json` | never-stored suffix D after restart | `9728` cached tokens, `block-disk+ssm+tq-native`, `disk_hit=true`, `ram_tokens_cached=0`, exact marker |

Interpretation: with Paged Cache off, Bonsai still finds partial common prefix
blocks from SSD, reconstructs transiently, keeps zero paged-RAM residency, and
preserves the q8 attention-KV plus native SSM companion-state policy.

The session was restored to `usePagedCache=true`, `enableBlockDiskCache=true`
after this proof.

## Laguna S-2.1 JANG_4M

Model:
`/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_4M`.

The existing Electron session is intentionally configured Paged-Off +
Block Disk L2. Pre-probe health reported:

- `backend_mode=block_disk_only`
- `paged_ram_enabled=false`
- `disk_only=true`
- `ram_tokens_cached=0`
- native cache `mixed_swa_kv_v1`
- `storage_quantization.bits=4`
- `applies_to=full_attention_kv_only`
- `sliding_window_policy=native_rotating_kv_state`
- `restore_policy=decode_full_attention_tq_and_restore_rotating_state`

| Artifact | Request | Result |
|---|---|---|
| `laguna_jang4m_paged_off_disk_only_store/summary.json` | same-process warm A | `6454` cached tokens, `block-disk+tq-native`, exact marker |
| `laguna_jang4m_paged_off_disk_only_store/summary.json` | changed-tail partial B | `6400` cached tokens, `block-disk+tq-native`, exact marker |
| `laguna_jang4m_paged_off_suffix_c_partial/summary.json` | never-stored suffix C after Electron restart | `6400` cached tokens, `block-disk+tq-native`, `ram_tokens_cached=0`, exact marker |

Interpretation: Laguna JANG_4M can use SSD-only partial prefix blocks with
native mixed-SWA state preserved and q4 storage only on eligible full-attention
KV. The retained `last_cache_execution` reports `disk_blocks` and
`cache_detail=block-disk+tq-native`; this family does not currently expose a
separate boolean `disk_hit` in the same shape as Bonsai.

## UI control policy

The intended user-facing behavior is:

- **Block Disk Cache (SSD / L2)** is the real content-addressed SSD block cache
  used above. It remains visible and clickable whether **In-Memory Paged Cache
  (RAM)** is on or off, as long as continuous batching is on.
- Legacy **Enable Disk Cache** is the old whole-prompt disk format. It is not a
  `.16` release blocker and may be disabled when it conflicts with Paged RAM or
  Block Disk L2. The UI text must continue to tell users to use **Block Disk
  Cache (SSD / L2)** for persistent SSD block reuse.

Source policy:

- `panel/src/shared/cacheControlPolicy.ts` keeps
  `blockDiskCacheDisabled = batchingOff`; it is not disabled by Paged-On or
  Paged-Off.
- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` renders
  the separate **Block Disk Cache (SSD / L2)** checkbox in the In-Memory Paged
  Cache section and labels the old control as legacy disk cache.

Focused policy test addition:
`panel/tests/cache-control-policy.test.ts` now pins that Block Disk L2 remains
available for both Paged-On and Paged-Off states, while legacy disk is only
available when both RAM paged cache and block SSD/L2 are off.

Focused verification after syncing the patched test to the live remote checkout:

```text
cd /Users/eric/mlx/vllm-mlx-r16-reasoning-p0-live
PATH=/opt/homebrew/bin:$PATH npm --prefix panel test -- --run tests/cache-control-policy.test.ts
tests/cache-control-policy.test.ts (18 tests) passed
```

## Still open before release

- Gemma 4 current-source Electron/API/cache run.
- Low-limit `Max Cache Blocks` eviction/refault live proof.
- Low-limit `Block Cache Max (GB)` disk eviction/refault live proof.
- Cross-family cache archetypes: standard KV, M3 sparse/MSA, DSV4 composite,
  openPangu native prompt disk, CCA/ZAYA, media-salted VL/audio/video.
- Full agentic tool-result continuation through all four gateway protocols.
