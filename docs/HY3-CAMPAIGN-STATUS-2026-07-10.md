# Hy3 (hy_v3) Campaign Status — 2026-07-10

Living status doc for the Hy3-JANG_2L/2K validation, loop root-cause, fix and
release lane. Updated as arms complete.

## Verdicts so far

### Loop root cause: sampler tail, NOT weights, NOT engine params
- Bare weights clean 5/5 seeds at temp 0.9; teacher-forced loop entry shows the
  bare model assigns ~0 probability to the loop token — the serving-stack
  suspicion was investigated arm by arm.
- top_p sweep: loop rate monotone in top_p (3/6 @ 1.0, 2/6 @ 0.95, 0/6 @ 0.9,
  0/6 with min_p 0.05 or top_k 40) → the server honors sampling params; the
  2-bit routed tail is genuinely heavy and untruncated sampling dips into the
  degenerate mass.
- TurboQuant exonerated for decode numerics: live TQ cache stores/reads float
  (see "TQ encode" below). q4 stored-prefix restore is the only lossy KV path
  (known F1, cold-vs-warm first-token class, task #45).

### Fix shipped (bundle + converter)
- `jang_config.chat.sampling_defaults = {temperature 0.9, top_p 0.9, min_p 0.05}`
  stamped into Hy3-JANG_2L and Hy3-JANG_2K; converter now stamps audited
  defaults instead of the vendor's raw 0.9/1.0/-1 (jang `b07e0dc`, pinning test).
- Defense-in-depth proven live: bundle min_p rides along even when a request
  explicitly forces top_p 1.0 (unset request params resolve from bundle).

### Loop-rate scoreboard (10 x 700-token runs each, loop = same-token run >= 20)
| arm | rate |
|---|---|
| 2L p0.95 (pre-fix, other harness) | 2/6 |
| 2L p1.0 (pre-fix) | 3/6 |
| 2L p0.95 post-stamp | 0/10 |
| 2L p1.0 post-stamp | 0/10 |
| 2L kv-quant none p1.0 | 0/10 |
| 2K p0.95 | 0/10 |
| 2K p1.0 | 0/10 |

### Bundles
| bundle | size | routed bits | MTP | loop gate |
|---|---|---|---|---|
| Hy3-JANG_2L | 89.5 GB | 2/2/2 | dropped | 0/30 post-stamp |
| Hy3-JANG_2L-withmtp-bak | 93.3 GB | 2/2/2 | 42 tensors | phase-3 depth proof in flight |
| Hy3-JANG_2K | 101.4 GB | 2/2/3 | dropped | 0/20 |
| Hy3-JANG_2K-MTP | converting | 2/2/3 | preserve-affine8 | queued |

### Engine facts established
- Gen-config resolution: all six API surfaces share one resolver stack
  (request > CLI/session > jang chat defaults > generation_config > family);
  VLM and LLM branches consume the same resolved kwargs dict. Live-proven:
  bare requests stochastic at bundle temp, explicit max_tokens exact,
  4096 fallback, temp0 warm runs byte-equal.
- Multi-eos dialect fix (`47365762f`): stop set now installs for
  variant-suffixed tokenizers (Hy3 `:opensource`); role-flip guard live
  (`eos_token_ids={120025, 120006, 120007}`).
- **TQ encode is inert engine-wide**: `TurboQuantConfig` has no
  `compress_after` param; 2000 tokens through the exact Hy3 auto cache →
  `_compressed_tokens=0`, K/V bf16. Logs/capabilities advertise 3-bit encode
  that never engages. Task #78: wire + per-family coherence gates + honest
  status reporting until then.
- `seed` is accepted only by image endpoints; text chat/completions silently
  ignore it. Wire per-request seed for reproducible debugging (follow-up).
- MTP request gate today is greedy-only (`temperature=0, repetition_penalty=1.0`)
  because native verify is exact-match. Stochastic speculative verify + the
  min_p/top_p floors + the 3-bit down backbone are the three levers that could
  flip the earlier "MTP nets -3%" measurement; 2K-MTP chain measures it.

## Cache-subsystem matrix (Hy3-JANG_2K) — in flight
- M1 default (memory-aware prefix + TQ objects): **ALL PASS** — temp0 warm
  stability, warm-vs-warm byte-equal, pollution guard, recall; 6 live cache
  hits; 0 "not isolatable"; 0 tracebacks. Paged cache confirmed default OFF
  (memory-aware selected).
- M2 `--use-paged-cache`: running, first checks passing.
- M3 paged + L2 block disk (`--enable-block-disk-cache`) incl. server restart
  to prove disk reload: queued.

## Phase 3 — queued behind cache matrix
- Arm E: withmtp-bak MTP autodetect + depths 1/2/3 byte-equal-vs-baseline
  greedy + timing per depth.
- Arm F: family regressions (Qwen3.6-27B hybrid, Zaya, gemma as discovered):
  full battery + streaming-leak + tool round-trip + unresolved-eos audit
  (dialect fix touches every family's stop-set path).

## 2K-MTP chain — queued behind phase 3
- Convert preserve-affine8 (first bundle born with the audited-defaults
  converter), then MTP-off baseline vs d1/d2/d3 byte-equal + tok/s, then
  loop gate at p0.95.

## Release v1.6.6 — staged pending publish word
- Ships: hy_v3 runtime, tag dialect, multi-eos dialect fix, native MTP runtime.
- Gates green: full pytest zero new failures (bundled-python drift resolved by
  re-bundle from the clean jang hy3 worktree — includes `jang_tools.hy3.*`);
  panel vitest 2176 pass / 3 documented stale.
- Release commit `9bc8a3694`; DMGs building; notarize → staple → verify next.
- UI↔engine parity dual-reviewer pass planned on the built app (CDP :9333).
