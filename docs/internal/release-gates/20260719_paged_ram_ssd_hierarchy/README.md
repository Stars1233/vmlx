# Paged RAM + block-disk L2 hierarchy gate (2026-07-19)

Status: `FIXED_SOURCE + PASS-LIVE_SCOPED` at pushed commit `8a93aa910`.
Overall release status remains `PARTIAL_NO_1_6_12_RELEASE`.

## Release blocker reduced

`cache/storage`: Paged RAM and block-disk L2 were not independent tiers. With
both UI toggles On, `BlockAwarePrefixCache.store_cache()` defaulted
`VMLX_PAGED_FRUGAL` On merely because a disk store existed, skipped ordinary
`block.cache_data`, and later discarded successful L2 promotions. Live LFM
health therefore reported indexed paged tokens but zero resident payloads; an
identical same-process replay used SSD rather than RAM.

## Source repair

- `vmlx_engine/paged_cache.py` now owns one explicit policy:
  - Paged On defaults to `ram_mirror_policy=resident` even with L2 enabled.
  - Paged Off + Block L2 forces `ram_mirror_policy=disk_only` and zero
    persistent KV payloads.
  - `VMLX_PAGED_FRUGAL=1` remains an explicit diagnostic/low-RAM override.
  - health/cache stats expose `paged_frugal` and `ram_mirror_policy`.
- `vmlx_engine/prefix_cache.py` keeps successful Paged-On L2 promotions as
  normal evictable L1 entries. Disk-only/explicit generic frugal restores are
  transient. DSV4, ZAYA CCA, and rotating-SWA path-dependent payloads retain
  their native-residency protection.
- `vmlx_engine/server.py` exposes the effective policy in `/health`.
- Regression tests pin resident default, explicit frugal, disk-only, RAM hit
  without an SSD read, LRU spill, SSD refault/promotion, and truthful stats.

No prompt coercion, sampler change, output rewrite, cache-detail fabrication,
or family-specific LFM branch was added.

## Live Electron proof

Artifact: external-drive `dealignai/LFM2.5-8B-A1B-MXFP4-CRACK` (base MXFP4,
not affine JANG and not JANGTQ/MXTQ). Real MLXStudio settings and Save & Restart
launched the project `.venv/bin/vmlx-engine` on port 8016.

Effective argv after restoring the desired defaults:

```text
--use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 10
--enable-block-disk-cache
--block-disk-cache-dir /Users/eric/.cache/vmlx-engine/live-proof-lfm-paged-tier-fixed-20260719
--cache-memory-percent 0.15
```

The engine materialized before a prompt: each UI restart reached
`model_loaded=true`, `last_request_time=null`. The current idle process after
the proof is Paged On / L2 On and reports `ram_mirror_policy=resident`.

| Row | Result | Current live evidence |
|---|---|---|
| Cold write-through | PASS | Electron row 599 exact-finaled `LFM-TIER-FIX-COLD-DONE`; prompt 310, cached 0. Health: 306 RAM tokens, 1,057,536 resident bytes, 306 SSD block tokens, five disk writes. |
| Same-process RAM hit | PASS | Row 602 restored 306/310 as `paged+ssm`; SSD reads stayed zero and L1 bytes/tokens were unchanged. |
| Bounded eviction | PASS | Disjoint row 605 was a cold 333-token prompt. The 9-usable-block pool recorded two evictions while L2 retained both prefixes. |
| SSD fallback + L1 promotion | PASS | Row 608 restored 306/310 as `paged+ssm+disk`; actual SSD reads advanced 0 -> 5 and the output exact-finaled. Successful L2 blocks remained in the bounded RAM tier. |
| Process restart + partial SSD prefix | PASS | PID 31958 -> 32602 started with zero RAM tokens and 635 SSD tokens. Changed-suffix row 611 restored 256/312 as `paged+ssm+disk`, exact-finaled, and promoted the hit into L1. |
| Paged Off + SSD-only partial | PASS | UI turned Paged Off while keeping Block L2 On. PID 33370 reported `block_disk_only`, `paged_frugal=true`, `ram_mirror_policy=disk_only`, zero RAM. Row 614 restored 256/311 as `block-disk+ssm`, exact-finaled, and remained at zero resident bytes. |
| Desired state restored | PASS | UI turned Paged On again. PID 33679 reached READY before a request with `paged_ram_enabled=true`, `ram_mirror_policy=resident`, zero cold L1 tokens, and 1,033 persisted L2 tokens. |

Every Electron turn above had non-empty visible content, no reasoning-only
finalization, no tool call, no warning, and no truncation. Screenshots and DB
rows are preserved in this directory.

## Raw protocol parity

The same live LFM engine served stream and non-stream requests through Chat
Completions, Responses, Anthropic Messages, and Ollama Chat.

| Protocol | Stream evidence | Terminal | Cache |
|---|---|---|---|
| Chat | 145 progressive content deltas, zero reasoning deltas | `stop`, usage, `[DONE]` | 295/299 `paged+ssm` |
| Responses | 145 progressive content deltas, zero reasoning deltas | one `response.completed` | 295/299 `paged+ssm` |
| Anthropic | 145 progressive `text_delta` events | one `message_stop` | shared engine hit; protocol usage 299 input / 146 output |
| Ollama | 145 progressive message-content objects | one `done:true`, reason `stop` | shared engine hit; 299 prompt / 146 eval |

All four stream-visible byte strings were identical. Each non-stream response
was non-empty and byte-identical to its matching stream. LFM added an unwanted
explanatory paragraph after the requested twelve lines on every protocol, so
strict-format quality is `PARTIAL_MODEL_OUTPUT`; transport, delta emission, and
terminal finalization pass. An initial explicit 96-token Responses probe ended
in the expected incomplete/length boundary; 192 tokens completed normally.

## Validation

- 190/190 selected paged/disk/TQ/hybrid/native-cache Python tests passed.
- 99/99 API-surface/Anthropic/Ollama adapter and streaming tests passed.
- The earlier 189/190 run exposed one stale source assertion expecting the old
  `hybrid paged HIT` diagnostic. The worker supports paged and disk-only
  backends, so the test now pins the existing production text
  `hybrid block-cache HIT`.
- Live UI screenshots, pre/post health, exact DB rows, current argv/session
  config, and raw protocol events are committed here.

## Retained boundaries

- This LFM run had `tq_native_enabled=false` and zero TQ-native block writes.
  It proves storage-tier selection, partial reuse, eviction, and restart—not
  LFM TurboQuant eligibility/encode/decode. That classification remains open.
- The protocol run did not request tools or reasoning. Cross-protocol
  required/auto/no-tool continuations remain in the broader model/parser
  matrix.
- Signed packaged-app repetition, fault-injected SSD failures, full suites,
  build, and release/notarization remain open.
