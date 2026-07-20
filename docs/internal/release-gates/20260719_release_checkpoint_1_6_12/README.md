# vMLX 1.6.12 release checkpoint — 2026-07-19

Status: **IN PROGRESS — NOT YET RELEASED**

This is the canonical checkpoint for the requested full public release. It
records exactly what is complete, what is under current validation, what must
finish before publication, and what is intentionally deferred to the next
campaign. A scoped live gate does not automatically close a family-wide row.

## Source and synchronization truth

- Working repository: `/Users/eric/mlx/vllm-mlx` on
  `erics-m5-max.local`, branch `reconcile/1.5.68`.
- Push target: `origin/codex/live-electron-gates-20260715`.
- Current pre-bump source head:
  `c4ec592b3ed8c2c8b165c92cdb19ff81dbc81c16`.
- GitHub branch and clean second-Mac worktree matched that SHA before the
  `1.6.12` version commit; final release SHA is pending.
- Clean synchronized worktree on the second Mac:
  `/Users/eric/mlx/vllm-mlx-live-electron-gates`, detached at the same SHA,
  with zero changes.
- The second Mac's original `/Users/eric/mlx/vllm-mlx` `main` worktree has 150
  pre-existing changes. It is deliberately untouched; synchronization uses
  the clean linked worktree so user-owned state is not overwritten.
- `.agents/STATUS.md` and `.agents/LOG.md` are local coordination files and are
  intentionally not staged or published.

## Current live Electron state

- Dev Electron uses CDP `127.0.0.1:9335` and user data
  `/Users/eric/.vmlx-v1611-cachefix-dev`.
- The current main-process log contains
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- The last current-source live model is
  `jangq-ai/Laguna-M.1-JANG_2L`, PID `88506`, port `8015`.
- It was restarted through the real Electron **Save & Restart** control and
  then returned through the real moon control to `standby/soft`.
- SQLite reports `status=standby`, `standby_depth=soft`, PID `88506`.
  `/health` reports `standby_soft`, `model_loaded=true`, and scheduler
  `num_waiting=0`, `num_running=0`.

## Completed and committed in the immediate checkpoint

### Gateway downstream disconnect ownership

- Commit: `5e83f2775737c557395262b0092d508412744927`
  (`fix(gateway): abort abandoned nonstream requests`).
- Root cause 1: the Electron gateway watched the consumed incoming request
  body and installed downstream-close handling only after upstream headers.
  Headerless non-stream inference could therefore remain scheduled after the
  client disappeared.
- Root cause 2: Anthropic `stream=false` internally consumed the shared Chat
  stream without a Starlette streaming response owning the ASGI receive
  channel, so the already-consumed disconnect was missed.
- Source repair: bind downstream `ServerResponse` close before upstream
  headers, destroy abandoned upstream requests, and use an active receive drain
  only for non-stream adapter consumption.
- Live proof through the real gateway covers stream and non-stream Chat,
  Anthropic Messages, Ollama chat, and Ollama generate. Abandoned requests
  returned idle in `0.029-0.037 s`, emitted no false terminal, and immediate
  recoveries completed exactly with truthful terminal and usage.
- Real Electron visible recovery grew
  `UI-GATEWAY-DISCONNECT-FIX-` -> `...-O` -> `...-OK` before metrics; SQLite
  retained exact content and no reasoning/warnings.
- Evidence:
  `../20260719_gateway_disconnect_recovery/`.

### q4 memory-prefix storage robustness

- Commit: `4b3d6951c444c22ce6d273350aef69e056ae1f22`
  (`fix(cache): tolerate uninitialized paged cache state`).
- The clean full suite exposed a reproducible failure in
  `test_memory_prefix_q4_storage_is_stream_independent_numpy_and_restores`.
- Root cause: `_packed_cpu` directly dereferenced `self.block_aware_cache`.
  Runtime construction initializes the field, but a minimal scheduler or an
  early serialization boundary without the field fell into the broad
  quantization-failure fallback and stored full unquantized KV.
- Repair: a missing field is treated exactly like Paged cache Off using
  `getattr(..., None)`. The existing contract now passes and proves independent
  NumPy q4 payload storage plus dequantized shape restoration.
- This is source plus focused contract proof. It does not create a new live
  family/cache-performance claim; existing live q4/TQ/L2 rows remain the
  runtime evidence.

## Previously completed current-campaign gates retained for this release

The master matrix and linked evidence remain authoritative. Important retained
scoped closures include:

- current dev Electron Start-before-request materialization across DSV4,
  Laguna, Step, openPangu, Gemma mixed-SWA, HY3 native MTP D1, and MiniMax M2.7;
- repeated real one-model Start swaps with stop-before-start and exactly one
  engine process;
- cache defaults/settings parity for the exercised families, including explicit
  Off behavior, Auto policy, q4 native-TQ labels, parser None, Min-P zero, LAN
  rollback, port conflict recovery, and single-model mode;
- architecture-specific cache paths: DSV4 native composite/pool codec with no
  generic TQ, openPangu typed MLA/DSA/SWA/conv state with no generic TQ,
  Gemma mixed-SWA, hybrid SSM/GDN rederive paths, q4 native-TQ KV components,
  paged memory, block-disk L2, partial-prefix reuse, eviction, and restart
  restore on the specifically named evidence rows;
- current model/parser streaming rows for reasoning/content separation,
  progressive visible output, terminal/usage ordering, real tool calls and
  tool-result continuations on Chat/Responses/Anthropic/Ollama where named;
- MiniMax M3 image and larger-video routes, Step image/video/media-keyed cache,
  Nemotron Omni media rows, and other scoped VL/audio rows linked from the
  matrix;
- responsive minimum-width Electron chrome and the five-locale surfaces that
  have direct DOM/screenshot proof;
- stale model-path classification, repoint, and removal UX;
- soft sleep, deep sleep, Wake, and repeated model-swap lifecycle gates.

These are not generalized to untested artifacts or parser families. See:

- `docs/internal/ISSUE-LEDGER.md`
- `../20260716_release_closeout/CURRENT-MATRIX.md`
- `../20260719_current_reconciliation/README.md`

## Current clean-head validation

The current run is isolated in a clean linked worktree at the pushed commit and
uses the documented clean JANG source
`/Users/eric/.cache/vmlx-release/jang-clean-9081c924/jang-tools`.

| Gate | Current status | Evidence/result |
| --- | --- | --- |
| Complete panel suite | PASS on `5e83f2775` | 2,332 passed, 3 skipped; `full-panel.log` |
| Complete Python suite, first run | FAIL on `5e83f2775` | 6,161 passed, 102 skipped, 92 deselected, 8 failed in 303.36 s |
| Focused q4 reproduction | FAIL before repair | missing `block_aware_cache` caused unquantized fallback |
| Focused q4 contract after repair | PASS on `4b3d6951c` | 1 passed; q4 NumPy storage and restore shape verified |
| Eight-failure triage union | PASS after repairs | all 8 previously failing tests passed together in 15.75 s |
| Clean source rerun on `320b1eef0` | FAIL, then triaged | 6,166 passed; 3 generated-proof audit failures |
| Clean source rerun on `2b28a82af` | HARNESS PARTIAL | 6,167 passed; two public-app audit failures remained because the Python-only worktree lacked `panel/node_modules` for ASAR extraction |
| Public-app audit after proof-boundary repair | PASS focused | full file 6/6; absent generated proof is OPEN, present stale/mismatched proof remains FAIL |
| Complete Python rerun | PENDING | must run from a clean worktree at the final pre-bump source head |
| Complete panel rerun | PENDING | required on final release-bump head if source changes |
| TypeScript typecheck | PENDING current wrapper | runs after full suite wrapper completes |
| Production bundle/build | PENDING | must run after version reconciliation and final source gates |

Current logs live under
`docs/internal/release-gates/20260719_post_disconnect_full_gates/` and will be
copied/linked into this checkpoint after finalization.

The eight failures split into one real runtime defect, three stale source
contracts, and four release-audit failures caused by tracked generated
`build/` proof files from June being resurrected by a clean checkout. The q4
runtime defect is repaired in `4b3d6951c`. The three contracts now point to
the current owning code and truthful log wording. The four audit contracts now
target current bounded RAM-tier tests instead of treating old generated output
as source truth. All previously tracked `build/` artifacts are being removed
from the Git index while locally regenerated copies remain on disk under the
existing `build/` ignore rule. This prevents a clean release checkout from
mistaking historical proof output for current packaged-runtime evidence.

## Release-critical work still required before publication

1. Fix every release-relevant failure without hiding, deselecting, or weakening
   the contract; commit/push each scoped repair.
2. Rerun the complete Python suite on the final pushed source.
3. Rerun the complete panel suite and TypeScript typecheck on the final pushed
   source.
4. Reconcile all source/package/feed versions to the selected `1.6.12`
   checkpoint only after verifying the repository's current version surfaces.
5. Commit and push the version/release-note/feed changes; synchronize both Macs
   and GitHub to the exact release SHA.
6. Run `panel/scripts/bundle-python.sh` first because Python engine source
   changed, then verify the bundled runtime and build the production app.
7. Build both Sequoia and Tahoe distribution artifacts.
8. Sign, notarize, staple, and verify every app/DMG artifact; retain notarization
   request IDs, signature assessment, package hashes, and install-smoke logs.
9. Install-smoke both OS-targeted builds without replacing or confusing the
    current dev Electron state.
10. Create/push the release tag, GitHub release, assets, `latest.json`, and feed
    updates; verify public URLs, hashes, version truth, and downloadable assets.
11. Update the master ledger/matrix with the final release SHA and exact
    pass/fail/deferred boundary, then pause.

## Explicitly deferred after this release checkpoint

The user requested a useful checkpoint now and continuation later. The
following remain documented and do not silently become PASS because packaging
succeeds:

- signed-app repetition of the broad live model matrix beyond install smoke;
- every remaining parser-family cross-model Electron tool row, including MiMo
  where current proof is absent;
- safe live injected engine exception after progressive deltas (client socket
  loss is now proved separately);
- long/stochastic reliability and latency/quality soak retained for Laguna,
  Bonsai, Step, DSV4, M3, openPangu, LFM, Nemotron, and other matrix families;
- larger or additional UI media attachment/reuse/salt/audio combinations not
  already named as live-proven;
- media-salt axes, Omni audio variants, and broader post-media text/tool turns
  still marked partial by the matrix;
- remaining transient/secondary/destructive modals, native sheets, and full
  accessibility/minimum-width sweep;
- 512K/long openPangu soak and other very-long-context performance work;
- DSV4 controlled reference-vs-vMLX sampling-quality A/B and any subsequent
  copied-bundle generation-config proposal;
- lower-priority missing-path session cleanup beyond the already proved
  repoint/remove workflow;
- a fresh comprehensive signed-app gateway/model-swap soak after publication.

## Release stop conditions

Do not publish if any of these is true:

- final complete Python or panel suite fails;
- typecheck or production build fails;
- bundled Python source/hash/import verification fails;
- Sequoia or Tahoe artifact is unsigned, unnotarized, unstapled, or fails
  install smoke;
- local, remote-box, GitHub branch, tag, feed, and packaged version disagree;
- a release-critical live smoke exposes empty output, reasoning-only final,
  batched/non-progressive content, false tool success, cache corruption, or a
  stuck active scheduler.

## Final release record (to fill before stopping)

- Final source SHA: `PENDING`
- Version/tag: `PENDING`
- Python suite: `PENDING`
- Panel suite: `PENDING`
- Typecheck/build: `PENDING`
- Sequoia artifact/hash: `PENDING`
- Tahoe artifact/hash: `PENDING`
- Codesign/Gatekeeper: `PENDING`
- Notarization request IDs/status: `PENDING`
- Staple validation: `PENDING`
- Install smoke: `PENDING`
- GitHub release/assets: `PENDING`
- Feed/latest version truth: `PENDING`
- Deferred rows preserved: `YES — see above and master matrix`
