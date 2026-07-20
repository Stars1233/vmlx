# vMLX 1.6.12 release checkpoint — 2026-07-19

Status: **PUBLIC CHECKPOINT RELEASED — BROADER DEFERRED ROWS REMAIN PARTIAL**

This is the canonical checkpoint for the requested full public release. It
records exactly what is complete, what is under current validation, what must
finish before publication, and what is intentionally deferred to the next
campaign. A scoped live gate does not automatically close a family-wide row.

## Source and synchronization truth

- Runtime/source checkpoint:
  `6de1096eca0ea2d5516ad64d6e79da98f3ae20a2`.
- Push target: `origin/codex/live-electron-gates-20260715`.
- Clean release worktree:
  `/Users/eric/mlx/vllm-mlx-live-electron-gates`.
- Live-model worktree on `erics-m5-max.local`:
  `/Users/eric/mlx/vllm-mlx`, branch `reconcile/1.5.68`; it and the GitHub
  branch matched the runtime/source checkpoint before release-metadata work.
- Unrelated local coordination and harness files on the remote workbox remain
  deliberately unstaged and were not overwritten.
- `.agents/STATUS.md` and `.agents/LOG.md` are local coordination files and are
  intentionally not staged or published.

## Exact installed-app live proof

Both post-staple DMGs were mounted and copied to isolated app paths without
replacing `/Applications/vMLX.app`:

- `/Applications/vMLX-1.6.12-Sequoia-Checkpoint.app`
- `/Applications/vMLX-1.6.12-Tahoe-Checkpoint.app`

The exact app path was verified from each Electron page URL. Each app was
driven through its real **Start** and **Stop** controls over its own CDP target.
Both logs report `installed=1.6.12`, `source=1.6.12`, and a bundled Python
engine at `1.6.12`. The selected Gemma 4 bundle launched with model-derived
`gemma4` tool and reasoning parsers, auto tool choice, VLM routing, paged cache,
block-disk L2, and the configured cache limits in the actual child-process
argv.

Sequoia evidence root:
`/Users/eric/.cache/vmlx-release/1.6.12/installed-smoke/final-6de1096ec`.

- UI turn 1: separate 1,783-character reasoning rail, non-empty three-sentence
  visible answer, and terminal marker `REL1612-FINAL-SEQ-T1-DONE`.
- UI turn 2: exactly one real `file_info(panel/package.json)` call and exact
  visible final `REL1612-FINAL-SEQ-T2-DONE SIZE=5.2 KB`.
- UI turn 3: no tool, distinct 463-character reasoning rail, and exact
  multi-turn recall `REL1612-FINAL-SEQ-T3-DONE CODEWORD=NEBULA-612 SIZE=5.2 KB`.
- Raw Responses: 116 reasoning deltas, 16 content deltas, exact visible output,
  `response.completed`, and usage. Raw Chat Completions: 200 reasoning chunks,
  15 content chunks, `finish_reason=stop`, usage, and `[DONE]`.
- Warm cache usage reported 100 cached tokens with
  `paged+mixed_swa+tq-native`; health recorded one cache-hit request, 100 hit
  tokens, native-TQ reconstruction/dequantization, disk writes, and L2 blocks.
- Two deliberately undersized/over-complex probes truthfully terminated
  `response.incomplete` at `max_output_tokens`; they are retained as negative
  evidence and are not counted as completion passes.

Tahoe used the same evidence root:

- real Start completed without an error toast and exposed the actual bundled
  engine PID/argv;
- UI turn `REL1612-FINAL-TAHOE-T1` showed a separate 173-character reasoning
  rail and exact non-empty visible output `42` plus the completion marker;
- raw Responses emitted 147 reasoning deltas, 17 content deltas, exact output
  `42\nREL1612-FINAL-TAHOE-RESP-DONE`, `response.completed`, and usage
  `73/199/272`;
- the real Stop control returned the app to `Start` / `Model is not running`.

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
| Complete Python suite | PASS on `6de1096ec` | 6,186 passed, 185 skipped in 271.90 s; `~/.cache/vmlx-release/logs-6de1096ec/full-python-after-bundle.log` |
| Complete panel suite | PASS on `6de1096ec` | 75 files; 2,332 passed, 3 skipped; `~/.cache/vmlx-release/logs-6de1096ec/panel-full.log` |
| TypeScript typecheck | PASS on `6de1096ec` | `~/.cache/vmlx-release/logs-6de1096ec/panel-typecheck.log` |
| Bundled runtime compatibility | PASS | engine 1.6.12 and clean JANG 2.5.31 source; `verify-bundled-python-compat.log` |
| Production Sequoia/Tahoe build | PASS | `build-release-dmgs-all.log` |
| Fresh Apple notarization | PASS | Sequoia `8b4a213b-a856-4659-8aa9-146ba211c163`; Tahoe `4fb3b188-5c57-4eb2-a909-85a917ee31b4`; both Accepted |
| Staple, signature, Gatekeeper | PASS | staple validation succeeded; both apps/DMGs accepted as Notarized Developer ID |
| Exact installed-app smoke | PASS | Sequoia UI/API/tool/cache and Tahoe UI/API evidence described above |

Current build/notary logs live under
`/Users/eric/.cache/vmlx-release/logs-6de1096ec/`; exact installed-app evidence
lives under the evidence root named above. Source regressions and earlier
campaign evidence remain linked from the master ledger and matrix.

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

## Public surface verification

- Source release:
  `https://github.com/jjang-ai/vmlx/releases/tag/v1.6.12`; annotated tag target
  `15da63f6d5d3323ca09c1c7cb1ab99251a2163d6`.
- DMG release:
  `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.12`; updater-repo
  tag target `4221cc58a82443567778db3ef1c60fd2486b1cba`.
- GitHub reports all four assets uploaded with the exact local sizes and
  SHA-256 digests. Blockmap SHA-256 values are
  `201988a4d54b454cf6513805e044442d481995dd09cf82f2ef8f14cca5dc9639`
  (Sequoia) and
  `8395cff9765f2e5a210d480b51fa360ae02011719e51cf9950e1f11b7aa54a0a`
  (Tahoe).
- PyPI `vmlx==1.6.12` is public. Wheel: 1,698,741 bytes, SHA-256
  `bd225af3b929173976c7e772bb96ce01542b723df26bb331aee23d08838ebc20`.
  Sdist: 2,698,733 bytes, SHA-256
  `a7b76485a28d77be946a887c3d402b0d3244af073ad6e745f3c273e759c82d9f`.
- Homebrew commit `1f9410e` publishes cask version `1.6.12` with the exact
  Sequoia hash. `brew style Casks/mlxstudio.rb` inspected one file with no
  offenses.
- Both `vmlx/main/latest.json` and `mlxstudio/main/latest.json` serve 1.6.12
  with the exact Sequoia/Tahoe hashes.
- `https://mlx.studio/update/latest.json` was deployed to the actual origin
  with a timestamped 1.6.11 backup and publicly serves 1.6.12 with the same
  hashes.
- The repository PyPI workflow run
  `https://github.com/jjang-ai/vmlx/actions/runs/29722123933` built and checked
  the package but failed OIDC with `invalid-publisher` because PyPI lacks a
  publisher matching the current GitHub claims. The exact checked artifacts
  were then published with the existing authenticated `.pypirc` on the trusted
  live-model Mac. Repairing the trusted-publisher configuration remains a
  post-release operational item; it did not change the published artifacts.

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

## Final release record

- Runtime/source checkpoint: `6de1096eca0ea2d5516ad64d6e79da98f3ae20a2`
- Version/tag: `1.6.12` / `v1.6.12`; tag target
  `15da63f6d5d3323ca09c1c7cb1ab99251a2163d6`
- Python suite: `6,186 passed, 185 skipped`
- Panel suite: `2,332 passed, 3 skipped`
- Typecheck/build/bundled runtime: PASS; see exact logs above
- Sequoia DMG: 509,134,318 bytes; SHA-256
  `704d87edf168a73d4ca2d94e8cb6190ca593ada71bca181bf369c84ea13ae421`
- Tahoe DMG: 525,182,991 bytes; SHA-256
  `81b9205a722282cc1eec75713c18dec3efc34ed76e3bcaf6587147e0ce372c49`
- Codesign/Gatekeeper: both apps and DMGs accepted as Notarized Developer ID,
  signed by `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`
- Notarization: Sequoia `8b4a213b-a856-4659-8aa9-146ba211c163` Accepted;
  Tahoe `4fb3b188-5c57-4eb2-a909-85a917ee31b4` Accepted
- Staple validation: PASS for both DMGs
- Install smoke: PASS for both isolated exact post-staple apps; see evidence
  root above
- GitHub releases/assets: public; exact URLs and hashes above
- Feed/PyPI/Homebrew public truth: public 1.6.12; exact commits/hashes above
- Deferred rows preserved: `YES — see above and master matrix`

## Post-release follow-up index

The v1.6.12 artifacts and tag remain immutable. After publication, source
commit `5f05ad72a` closed the dev-Electron/raw-Chat/raw-Responses safe injected
mid-stream failure row with current committed-source live evidence. It is not
part of 1.6.12 and requires a future release checkpoint to ship. Evidence:
`../20260719_midstream_failure_recovery/README.md`.
