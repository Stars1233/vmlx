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

### Bundles (Eric 2026-07-10: keep ONE — the 2K with working MTP)
| bundle | size | routed bits | MTP | loop gate | disposition |
|---|---|---|---|---|---|
| Hy3-JANG_2L | 89.5 GB | 2/2/2 | dropped | 0/30 post-stamp | **DELETED** (superseded by 2K) |
| Hy3-JANG_2L-withmtp-bak | 93.3 GB | 2/2/2 | 42 tensors | — | delete after Arm E depth proof |
| Hy3-JANG_2K | 101.4 GB | 2/2/3 | dropped | 0/20 | delete after 2K-MTP passes gates |
| Hy3-JANG_2K-MTP | converting | 2/2/3 | preserve-affine8 | queued | **KEEPER** (pending gates) |

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

## Cache-subsystem matrix (Hy3-JANG_2K) — COMPLETE
- M1 default (memory-aware prefix + TQ objects): **ALL PASS** — temp0 warm
  stability (runs 2-4 byte-equal), warm-vs-warm byte-equal, pollution guard,
  recall; 6 live cache hits; 0 "not isolatable"; 0 tracebacks. **Paged cache
  confirmed default OFF** (memory-aware auto-selected).
- M2 `--use-paged-cache`: **ALL PASS** — paged path active (22 block lines,
  11 hits), determinism identical to M1.
- M3 paged + L2 block disk: storage + restart-reload **PROVEN** (49MB
  content-addressed block safetensors; post-restart same prompt byte-identical
  from disk). **One FAIL (filed as F21, task #79)**: consecutive warm greedy
  requests diverged by a trailing period ('Paris.' vs 'Paris') — suspect
  memory-block vs lossy disk-block restore mismatch, F1-class. Opt-in path
  (requires paged + explicit L2 flag); not a v1.6.6 blocker.

## MTP — RESOLVED: depth-1 default, +10-14% measured win
Two stale blocks had made every earlier "depth test" silently inert:
withmtp-bak's bundle-declared stamp, and a legacy engine-side JANG_2K profile
block from the 2026-05-17 six-variant gate on the Hy3 PREVIEW. Forced
measurement on the final post-train Hy3-JANG_2K-MTP (greedy 600-tok x3/arm):

| config | tok/s | acceptance | vs baseline |
|---|---|---|---|
| no MTP | 27.8 | — | — |
| depth 1 | 30.6 | ~full | **+10%** |
| depth 2 | 26.3 | 24.5% | -6% |
| depth 3 | 19.1 | 1.9% | -31% |

Fixes landed (all pushed):
- Engine: legacy JANG_2K profile block DELETED (`2b770a3fd`); bundle-declared
  measured stamps (+ FORCE override) are the only block source. 91 MTP tests
  pass; 3 rewritten to pin the new contract.
- Bundle: `vmlx_mtp_tuning.json` best_depth=1 stamped into Hy3-JANG_2K-MTP.
- Converter: hy3 preserve builds stamp the sidecar automatically (jang
  `3cace35`).
- Live default-config proof: plain serve, no env/flags →
  `runtime_active: True, effective_depth: 1
  (vmlx_mtp_tuning.json:native_mtp.best_depth)`, 33.2/34.1/28.2 tok/s.

Baseline-vs-MTP byte divergence is the inherent MoE multi-row verify routing
fork (chunked-prefill class; divergent continuations verified coherent by
inspection) — the MoE MTP gate is coherence + self-consistency, not cross-arm
byte equality.

## Bundle end state (single keeper)
`Hy3-JANG_2K-MTP` (105.3 GB, routed 2/2/3, MTP depth-1 active by default,
audited sampling stamp) is the ONLY Hy3 bundle on the drive. 2L,
2L-withmtp-bak and 2K deleted after their gates were superseded (~167 GB
reclaimed).

## Arm F — family regressions (re-running after zsh `path` variable bug)
Qwen3.6-27B hybrid, Zaya, gemma: full battery + streaming-leak + tool
round-trip + unresolved-eos audit (dialect fix touches every family's
stop-set path).

## Release v1.6.6 — staged pending publish word
- Ships: hy_v3 runtime, tag dialect, multi-eos dialect fix, native MTP runtime.
- Gates green: full pytest zero new failures (bundled-python drift resolved by
  re-bundle from the clean jang hy3 worktree — includes `jang_tools.hy3.*`);
  panel vitest 2176 pass / 3 documented stale.
- Release commit `9bc8a3694`; DMGs building; notarize → staple → verify next.
- UI↔engine parity dual-reviewer pass planned on the built app (CDP :9333).
