# DSV4 long-context snapshot budget and current-source proof

Date: 2026-07-20  
Host: `erics-m5-max.local`  
Repository: `/Users/eric/mlx/vllm-mlx-release-1.6.13`  
Branch: `codex/postrelease-ui-drawers-20260720`  
Starting source: `f5558c1ef5d48d9611b09b46e7c6c0064328823d`

## Scope and artifact truth

The tested artifact is
`/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`.
It is affine JANG, not JANGTQ/MXTQ. The live 43-layer cache is the
architecture-owned DSV4 composite: two local-SWA KV layers, 21 CSA compressed
pool layers (ratio 4), and 20 HCA compressed pool layers (ratio 128), with SWA
window 128. Generic TurboQuant is intentionally disabled on this route; the
separate DSV4 native pool codec remained enabled for the passing cache rows.

## Root cause and source repair

Two independent memory-lifetime problems were found.

1. `vmlx_engine/utils/dsv4_batch_generator.py` deep-copied the complete nested
   typed prompt cache before decode without checking its size against the active
   RAM/disk cache budget or current Metal headroom. Current source estimates the
   nested cache first, rejects an unsafe copy, and records estimate, headroom,
   and skip telemetry. `vmlx_engine/scheduler.py` now supplies the budget before
   the DSV4 early-return path and exposes that telemetry through cache health.
2. The DSV4 JANG model left the lazy CSA/HCA attention graphs from all 43 layers
   alive until final logits. The query-by-compressed-pool graphs exhausted the
   Metal working set even though the persistent typed cache fit. Current JANG
   source materializes the hidden state after every decoder layer for multi-token
   prefill at or above 256 tokens. This preserves model math while bounding
   transient graph lifetime; the behavior can be explicitly disabled with
   `DSV4_LAYERWISE_PREFILL=0` for numeric diagnosis.

The Electron settings path had a separate parity defect: DSV4 suppressed both
batch-size controls and `--prefill-step-size`. The batch-size flags remain
suppressed because `DSV4BatchGenerator` is single-batch, but the user-selected
prefill step now appears in both preview and the actual engine argv.

## Automated validation

- `tests/test_dsv4_paged_cache.py`: 67 passed.
- Focused DSV4 snapshot-admission selection: 12 passed.
- `python -m py_compile` for the changed Python engine files: passed.
- Panel `tests/settings-flow.test.ts`: 286 passed.
- Panel TypeScript typecheck: passed.
- JANG DSV4 overlap/pool selection: 7 passed.

These are focused suites, not the full repository checkpoint suites.

## Live Electron and API results

| Gate | Result | Current evidence |
|---|---|---|
| Real UI Start and effective argv | **PASS-LIVE scoped** | The current Electron main process found the venv engine, the real Start control launched DSV4, and `dsv4-live-argv.txt` records `--prefill-step-size 512`. Only one model process remained under Single Model mode. |
| 23,477-token prefill memory survival | **PASS-LIVE scoped** | `dsv4-long-layerwise-real-step512-summary.json` and screenshot show the current source surviving the exact 85,720-byte prompt. TTFT was 89.94 s; peak active memory was about 104.9 GB instead of the previous Metal OOM near 107.5 GB. Live logs reported layerwise prefill materialization and an approximately 394 MB prompt snapshot estimate with positive headroom. |
| 23,477-token Auto output correctness | **FAIL** | SQLite row 224 in `electron-assistant-rows.json` has 23,560 reasoning characters, 4,144 output tokens, a length warning, and wrong visible content. The memory repair does not promote this quality row. |
| 7,875-token Auto Electron output | **FAIL / PARTIAL transport** | `dsv4-medium-ui-auto-summary.json` records 1,228 progressive UI changes and separate reasoning, so it was not a batch-at-end renderer failure. The reasoning looped incoherently until the real Stop control interrupted at 2,288 output tokens. |
| 7,875-token Instruct Electron output | **PASS-LIVE scoped** | `dsv4-medium-ui-instruct.json` records 18 progressive UI states and exact non-empty three-line content containing both hidden secrets and `DSV4-MEDIUM-UI-INSTRUCT-DONE`. SQLite row 230 has 32 output tokens, no warning, and no reasoning leak. |
| Raw Responses reasoning/content/terminal | **PASS-LIVE scoped** | `dsv4-medium-raw-stream-ab.json`: Instruct emitted 31 progressive content deltas; Thinking emitted 512 separate reasoning deltas plus 29 content deltas. Both emitted exactly one completed terminal and correct visible content. |
| Exact warm and restart-from-disk composite reuse | **PASS-LIVE scoped** | `dsv4-medium-raw-stream-disk.json` records deterministic identical output. After real UI Stop/Start, 7,874 tokens restored as `paged+dsv4+disk`; the immediate resident repeat restored 7,874 as `paged+dsv4`. `dsv4-health-after-deterministic-cache.json` records 31 disk promotions, two scheduler hits, 15,748 saved tokens, exact 43-layer typed state, native pool codec On, and generic TQ Off. |
| DSV4 sampling defaults in real drawer | **PASS-LIVE scoped** | Exact bundle `jang_config.json` defaults are temperature 0.6, top-p 0.95, repetition 1.0, max 4096. `dsv4-sampling-defaults-live.png` shows sliders `0.6 / 0.95 / top-k Off / min-p Off / 1.0` and max placeholder 4096. `sampling-defaults-live.json` also captures session-default/API detection parity for eight existing model sessions; only DSV4 was visually reopened in this gate. |
| Current-loader direct model A/B | **FAIL / PARTIAL root cause** | Follow-up gate `../20260721_dsv4_direct_quality_boundary/` bypassed Electron, API adapters, parsers, scheduler, paged RAM, and block L2. All three 7,879-token direct runs (bundle sampling, greedy pool-on, greedy pool-off) failed to close reasoning or reach the marker within 512 tokens. This narrows the defect below transport/cache/sampling/pool-codec layers but does not distinguish the official artifact from the shared DSV4 architecture implementation. |
| Independent architecture reference | **BLOCKED** | `dsv4-direct-jang-reference-loader-fail.txt` records the legacy standalone JANG loader rejecting the official bundle because it lacks its old top-level `format` field. No artifact blame or independent-reference quality conclusion is made. |
| Current-head short Auto stream boundary | **PASS-LIVE scoped** | Follow-up Electron row 284 progressively painted separate reasoning and exact visible content over 40 DOM states; raw Responses emitted 44 reasoning-summary and 12 content deltas plus one completed terminal. This confirms the shared stream boundary for short prompts without promoting the retained long-quality row. |

## Current verdict

The long-prefill memory lifetime, UI prefill-step argv parity, short/raw
streaming separation, and deterministic resident/SSD typed-cache restore are
current-source **PASS-LIVE scoped** with the evidence above. DSV4
Auto-reasoning quality on the repeated-record long-context prompts is
**FAIL/PARTIAL**, and the direct-model follow-up proves the symptom exists below
the API/cache layers. Artifact-versus-shared-architecture attribution remains
unresolved. The broader DSV4 quality, stochastic, gateway-protocol,
fault-injection, and signed-app matrix remains open. This gate is not a
release-readiness claim.

## Evidence inventory

- `dsv4-after-crash-view.png`
- `dsv4-long-ui1-dom-trace.json`
- `dsv4-long-postfix-fresh.json`
- `dsv4-long-pool-on-step512.json`
- `dsv4-long-layerwise-step512.json`
- `dsv4-long-layerwise-real-step512-final.png`
- `dsv4-long-layerwise-real-step512-summary.json`
- `dsv4-medium-ui-auto-final.png`
- `dsv4-medium-ui-auto-summary.json`
- `dsv4-medium-ui-instruct-final.png`
- `dsv4-medium-ui-instruct.json`
- `dsv4-medium-raw-stream-ab.json`
- `dsv4-medium-raw-stream-disk.json`
- `dsv4-health-after-deterministic-cache.json`
- `dsv4-live-argv.txt`
- `dsv4-sampling-defaults-live.png`
- `sampling-defaults-live.json`
- `electron-assistant-rows.json`
- `dsv4-direct-jang-reference-loader-fail.txt`
