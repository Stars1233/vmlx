# vMLX 1.6.14 public checkpoint and retained matrix — 2026-07-20

Status: `PUBLIC_1_6_14_RELEASED_BROADER_MATRIX_PARTIAL`.

This is the current additive closeout view over `docs/internal/ISSUE-LEDGER.md`,
`.agents/STATUS.md`, the July 15–16 live proof directories, the shared wiki
production gate, and the current branch. Older contradictory rows remain in
their original ledgers for provenance; the newest source-plus-live row wins and
superseded conclusions are called out here.

The public v1.6.14 checkpoint is tagged at `e1776a485`, signed/notarized for
Sequoia and Tahoe, installed-smoked through the real Electron UI, published to
GitHub/PyPI/Homebrew/updater feeds, and independently re-read. Canonical proof:
`../20260720_release_checkpoint_1_6_14/README.md`. This changes only the release
checkpoint status; every explicit family/protocol/cache/media/gateway/stress
PARTIAL or OPEN row below remains retained.

### 2026-07-20 LFM native-reasoning and gateway-error override

- Current remote post-release head is pushed source `1d04d49b3` on
  `codex/postrelease-ui-drawers-20260720`. Public v1.6.13 remains sealed at
  tagged source `2f509f79d`; neither post-release fix is part of that public
  artifact.
- `24330de7e` removes the synthetic LFM-only `<think>` sentinel after the real
  `LFM2.5-8B-A1B-MXFP4-CRACK` bundle proved it is base MLX MXFP4 (not affine
  JANG or JANGTQ/MXTQ), has six attention plus 18 SSM/conv layers, owns native
  reasoning in its template, and explicitly forbids synthetic prefill.
- A current-source Electron Auto/no-tool turn streamed a separate reasoning
  rail, progressively painted content, and exact-finaled. The required
  `file_info(panel/package.json)` Electron turn remains `FAIL-LIVE` for this
  artifact: it parsed `path=": "`, failed execution, leaked faux JSON, and
  replayed the prior marker. No artifact blame, forced tool output, sampler
  clamp, or hidden answer rewrite was added.
- The shared Responses terminal bug is closed: unmet `tool_choice=required`
  now emits `tool_calls_required`, terminates only with `response.failed`, and
  is not persisted as successful history. Raw current-source API proof
  reproduces the model miss and the truthful terminal.
- `1d04d49b3` teaches the Ollama gateway error translator to preserve FastAPI
  `detail` messages and backend status. After a full current-source Electron
  relaunch and real UI Start of LFM PID 26730, direct/gateway Chat, Responses,
  Anthropic, and Ollama all returned the same explicit 400 family message.
- Focused validation: 14 selected Python tests, 88 LFM panel tests, 53 gateway
  tests, panel typecheck, and diff checks. Evidence:
  `docs/internal/release-gates/20260720_lfm_native_reasoning_protocol/`.
- Overall status remains `PARTIAL`: the LFM MXFP4 native required-tool row,
  remaining family/parser/media/cache/settings/network-loss/stress rows, and
  full-suite/build repetition at the eventual next cutoff are still open.

### 2026-07-20 Anthropic/Ollama mid-stream failure override

- Pushed source `d811270ad` closes the native terminal contract for injected
  mid-stream engine failures on Anthropic Messages, Ollama `/api/chat`,
  templated `/api/generate`, and raw `/api/generate`. Public v1.6.12 remains
  sealed and does not contain this post-release fix.
- Ollama previously discarded the upstream structured error and synthesized a
  false `done:true` from the later `[DONE]`. All three converters now emit the
  official native `{"error":"..."}` NDJSON row, while each route clears any
  deferred success/tool terminal and suppresses later success synthesis.
- Literal `curl -N --no-buffer` failure/recovery pairs against production
  handlers streamed two visible chunks before each error. No failing Ollama
  stream contained `done:true`; every immediate recovery completed with one
  truthful terminal and usage. Anthropic likewise ended on `event:error`
  without `message_stop`, then recovered to `message_stop` with usage.
- Focused current validation is 30 adapter/route tests plus the existing
  Chat/Responses mid-stream and Ollama output-cap selections. Evidence:
  `docs/internal/release-gates/20260720_anthropic_ollama_midstream_failure/`.
- This supersedes only the older `Anthropic/Ollama injected failures` OPEN
  wording. Gateway network-loss injection, signed-app repetition, broader
  parser/model agentic loops, and unrelated family/cache/media/stress rows
  remain `PARTIAL`/`OPEN`.

### 2026-07-19 post-release mid-stream failure/recovery override

- Pushed source `5f05ad72a` closes the shared safe injected engine-failure
  boundary for raw Chat Completions, raw Responses, and the real Electron dev
  app. The public v1.6.12 artifacts remain sealed and do not include this
  post-release change.
- Both production Python stream generators already emitted progressive text
  plus terminal error/usage truth. Electron cancelled the reader at the first
  error event, before Chat's usage chunk or Responses' `response.failed`
  terminal. The client now defers the ordinary server error until the terminal,
  consumes authoritative partial usage, recognizes nested failed-response
  errors, and retains immediate expected-disconnect handling.
- Literal curl-N probes proved error/terminal/usage ordering and clean immediate
  recovery on both APIs. Current committed-source Electron frames visibly show
  partial text before failure, exact interrupted persistence with 2 output / 5
  prompt tokens, error UI, and exact same-chat recovery on both protocols.
  Outbound recovery history keeps the safe partial prefix and strips the
  UI-only interruption marker.
- Full current validation is 6,185 Python passed / 95 skipped / 92 deselected,
  2,333 panel passed / 3 skipped, typecheck PASS, and Electron production build
  PASS. Evidence:
  `docs/internal/release-gates/20260719_midstream_failure_recovery/`.
- This supersedes only the older `safe live mid-stream exception injection`
  OPEN wording. Gateway network-loss injection, Anthropic/Ollama injected
  failures, signed-app repetition, and other model/parser/stress rows remain
  `PARTIAL`/`OPEN`.

### 2026-07-19 MiniMax M2.7 JANGTQ hierarchy/protocol override

- Current cutoff `b31fdca95` was exercised with the exact external-drive
  `jangq-ai/MiniMax-M2.7-Small-JANGTQ` artifact through the real Electron
  Sessions Start control. Bundle truth is `minimax_m2`, full KV, 62 layers,
  `weight_format=mxtq`, and profile `JANGTQ2`; it is not affine JANG or base
  MXFP. Auto prefix storage selected q4 native TurboQuant for all 62 attention
  KV layers. Bonsai remains the separate q8 exception.
- Single-model mode visibly stopped the prior LFM process and left one local
  engine. Before any prompt, health reported `model_loaded=true`,
  `last_request_time=null`, and about 38.3 GB active model memory; the Electron
  main log again resolved the project `.venv/bin/vmlx-engine`. This closes
  eager materialization only for this route.
- Electron rows 617/620/623/626 prove cold q4 write-through, 352-token RAM
  reuse without an SSD read, bounded eviction, and 352-token SSD refault with
  L1 promotion. PID restart row 629 restored a 320/360 partial chain as
  `paged+disk+tq-native`. With the real UI setting Paged Off / L2 On, row 632
  restored 320/361 directly as `block-disk+tq-native` while health stayed at
  zero resident bytes. Paged On / L2 On was restored afterward.
- Raw stream and non-stream Chat, Responses, Anthropic, and Ollama calls
  emitted identical non-empty progressive content and their native terminal
  events. A separate native-reasoning run emitted 369 reasoning deltas plus
  eight visible deltas on all four protocols with no rail leakage. Raw
  Responses executed one `file_info`, continued from its real result via
  `previous_response_id`, and exact-finaled.
- Electron rows 641/644 prove non-empty exact visible answers after distinct
  non-byte-identical reasoning rails. Row 647 executed exactly one real
  `file_info(panel/package.json)` and exact-finaled with `5.2 KB`; same-chat
  rows 650/653 recalled it without another tool call.
- Verdict: `VERIFIED-LIVE_SCOPED` for this M2.7 JANGTQ full-KV q4 hierarchy,
  disk-only partial-prefix, four-protocol stream, reasoning, tool, and UI
  multi-turn row. Remaining families, typed/hybrid/mixed caches, gateway soak,
  broader eager routes, signed-app repetition, and the overall release remain
  `PARTIAL`/`OPEN`. Evidence:
  `docs/internal/release-gates/20260719_minimax_m27_tq_hierarchy_protocol/`.

### 2026-07-19 Qwen3.6 35B JANGTQ current-source override

- Pushed source `54222003d` has a new current-source Electron/API/cache/UI proof.
  A fresh Electron tools-on chat executed one real `file_info` and exact-finaled;
  current Responses and Chat each emitted a valid tool call and streamed the
  real-result continuation as 256 reasoning plus 18 content deltas to clean
  terminals with usage.
- A real process restart restored the exact Electron tool prefix from seven q4
  native-TQ disk blocks plus two SSM disk hits. A separate changed-suffix
  2,587-token request restored 2,560 tokens from 40 disk blocks, all 40 native-TQ
  hits, plus a complete 30-layer SSM checkpoint, then returned the changed exact
  answer in 0.476s. Current source repeated it in 0.492s.
- Commit `87e11c5ee` corrects the startup wording: generic
  `QuantizedKVCache=none` avoids a second codec; native architecture-selected
  q4 attention-TQ remains enabled at prefix/paged/L2 boundaries. It does not
  change or duplicate the codec.
- Commit `54222003d` closes the Sessions-card quant-label defect. A complete
  Electron-main relaunch and real Sessions Start showed `JANGTQ2 (2b)` on both
  the card and active header. Affine JANG controls stayed explicit; base MXFP
  controls stayed unlabeled, including a child beneath provider directory
  `jangq-ai/`. Fresh row 440 completed one real tool loop with non-empty content
  and a 3,904-token disk-backed hybrid hit, but misspelled its requested marker.
- Retained PARTIAL/OPEN rows: strict sampled formatting/repeated-tool
  reliability and advertised VL (`vl_runtime_available=false` here). Evidence:
  `docs/internal/release-gates/20260719_qwen35_jangtq_current/`.

### 2026-07-19 HY3 native-MTP D1 current-source override

- Source cutoff `0e09ce789` loaded the real affine `Hy3-JANG_2K-MTP` bundle
  through the Electron Sessions Start control. Single-model mode stopped Qwen
  PID 26427 before HY3 PID 27632 became ready; health was model-loaded with no
  request and argv selected Hunyuan tools, qwen3 reasoning, Auto q4 stored
  prefixes, and native MTP depth 1.
- Four same-chat Electron turns exact-finaled with distinct reasoning. One turn
  executed exactly one real `file_info`; later no-tool turns recalled both
  facts. Browser mutation evidence proves post-reasoning/tool visible text grew
  character-by-character rather than terminal-batching.
- Literal curl-N Responses and Chat no-tool, required-tool, and real-result
  continuations all kept reasoning/content/arguments separate, exact-finaled,
  and terminated cleanly. Chat emitted zero intermediate non-null usage objects,
  one choices-empty usage chunk, then one `[DONE]`.
- The first current process recorded 1,194 drafted/497 accepted D1 tokens.
  Electron replaced PID 27632 with 29852; the next turn restored 4,655/4,872
  tokens as `paged+disk+tq-native`, then recorded 87 drafted/35 accepted tokens.
  Health counted 73 real disk reads, all 73 native-TQ hits, with zero resident
  L1 bytes. Current focused validation is 318/318.
- Verdict: `VERIFIED-LIVE_SCOPED`; retained `PARTIAL` rows are long/stochastic
  reliability and a fresh current-source MTP-Off performance A/B. Evidence:
  `docs/internal/release-gates/20260719_current_hy3_mtp/`.

### 2026-07-19 current-head reconciliation override

- Current audited/pushed head is `54222003d`. The public v1.6.11 checkpoint
  remains released, signed, and notarized; this post-release head is not a
  v1.6.12 candidate yet.
- The earlier full-suite checkpoint remains valid for its source, but is stale
  relative to later shared protocol/parser/loader fixes. Complete Python/panel,
  typecheck, bundle verification, and production build must run again at the
  chosen cutoff.
- Paged-Off SSD reuse is now live-proven rather than assumed: M2.7 restored a
  q4 TQ-native partial prompt prefix without paged blocks, and openPangu restored
  its typed N-1 payload under Auto/Off reasoning. Paged-On M2.7 separately proves
  a 64+64+50 partial chain, bounded eviction, same-process L2 refault, and
  process-restart disk-only restore. Evidence: `20260719_nonpaged_prompt_disk_partial/`,
  `20260719_prompt_disk_payload_prefix_index/`, and
  `20260719_m27_paged_l2_partial_refault/`.
- Shared Chat usage, Anthropic, Ollama, Responses cancellation/disconnect, Chat
  disconnect, Electron stop/recovery, and simultaneous multi-tool rows have
  current scoped proof under their July 19 evidence directories. Protocol parity
  remains `PARTIAL` for live injected mid-stream failure, network-loss/gateway
  soak, signed-app repeats, and remaining parser/model families.
- Mistral Medium 3.5 JANGTQ2 is newly `BLOCKED_CURRENT_ARTIFACT_RUNTIME`:
  strict 616/616 hydration succeeds, legacy prefill stalls, and both original and
  FP32 MPP NAX diagnostics emit newline-only tokens. The unsafe dense-Mistral auto
  exception was withdrawn by `fad7356d4`; no artifact blame or hidden output fix
  was added. Evidence: `20260719_mistral35_jangtq_prefill/`.
- Canonical current worklist and evidence classification:
  `docs/internal/release-gates/20260719_current_reconciliation/README.md`.

This override supersedes older OPEN/PARTIAL child rows only where the named
current evidence explicitly closes them. It does not convert the overall matrix
to a release pass.

### 2026-07-19 Responses cancellation/disconnect override

- Commit `ae498c70b` repairs the shared Responses terminal contract. A pre-fix
  explicit cancel returned HTTP 200 after three content deltas but falsely emitted
  a completed output item and `response.completed`. Aborted or detected-disconnect
  streams now retain incomplete item state, emit only `response.incomplete` with
  `reason=cancelled`, skip answer retry/history persistence, and return the engine
  to idle. Mid-stream exceptions emit `response.failed` rather than a completed
  envelope with failed inner status.
- Focused validation is 111 passed with 741 deselected. After a real Electron
  Stop/Start, PID 95088 produced the corrected cancel terminal, zero active
  requests, and no stored response. A separate client disconnect after five
  deltas reached idle in 1.12 seconds; immediate recovery streamed 12 content
  deltas to an exact marker and completed once.
- Evidence:
  `docs/internal/release-gates/20260719_response_cancel_disconnect/`.
- This advances explicit Responses cancellation and client-disconnect recovery to
  `PASS-LIVE scoped`. API/protocol parity remains `PARTIAL`: safe live mid-stream
  fault injection, Chat cancellation/disconnect, signed-app repeat, raw Generate
  multi-tool, and other parser/model family rows remain open.

### 2026-07-19 Ollama/Electron multi-tool override

- Current-source M2.7 emitted two schema-valid Ollama calls in one terminal, both
  real results were returned as named tool messages, and the follow-up streamed 30
  exact visible deltas with one stop and no repeated tool. A fresh Electron chat
  independently executed the same `file_info` and `run_command` calls exactly once
  each, exact-finaled, retained separate reasoning, stored no warning, and returned
  the engine to idle.
- Commit `1b35d7a9b` pins two-call object-argument conversion and two-result history;
  31 selected adapter/protocol tests pass. Evidence:
  `docs/internal/release-gates/20260719_ollama_multitool/`.
- This advances M2.7 Ollama/Electron multi-tool to `PASS-LIVE scoped`. Other model
  and parser families, signed-app repeat, cancellation, media tools, and long-loop
  soak remain open.

### 2026-07-19 Chat disconnect/Electron stop override

- A current-source Chat stream was closed after five progressive content deltas;
  the Electron-started engine returned idle in 1.078s and an immediate recovery
  streamed 12 exact content deltas, one stop, usage, and `[DONE]`.
- Real Electron user-stop is live-proven both during prefill and after visible
  partial content. Prefill stop persisted no false assistant. Mid-content stop
  preserved only real partial bytes plus `[Generation interrupted]`; same-chat
  recovery exact-finaled and health stayed idle. Test controls were restored.
- Existing focused coverage passed 7 Python selections and 368 panel tests.
  Evidence:
  `docs/internal/release-gates/20260719_chat_disconnect_stop_recovery/`.
- Chat client-disconnect and Electron user-stop/recovery advance to `PASS-LIVE
  scoped`. Safe live engine-exception injection, signed app, gateway network loss,
  other model/parser families, and soak remain `PARTIAL`.

## Release truth

### 2026-07-19 current-source full-suite and bundle-safety override

- Current pushed code head `92935ada5` closes the source/full-suite/build
  checkpoint without closing the retained live-runtime matrix. The final
  isolated Python run passed **6,125**, skipped **96**, and deselected **92**;
  full panel passed **2,312** with three skips; TypeScript typecheck passed;
  bundled source/hash/import verification passed; and the clean-JANG
  production build passed.
- Full-suite discovery `fb9689968` updates stale test doubles/contracts without
  changing runtime behavior. Release fix `92935ada5` stops
  `bundle-python.sh` from deleting the repository's tracked `build/` proof
  tree: it now removes only setuptools-owned scratch subdirectories. A real
  clean-JANG bundle rebuild retained a sentinel in `build/` and printed
  `PRESERVE_PROBE_PASS`.
- The regenerated canonical current-regression orchestrator remains honestly
  `status=open`: MiMo bundle absence, staged packaged-integrity/signing drift
  for the post-release head, DSV4/Qwen/Gemma/cross-family live requirements,
  and other retained rows are not converted into passes by full unit suites.
- This supersedes the historical `Full tests/build | OPEN` row for the
  current-source unit/build gate only. It also supersedes the historical
  `Packaging/public release | BLOCKED | Public truth remains 1.6.10` row:
  v1.6.11 is public/signed/notarized, while the current post-release head is
  not packaged or released as a newer version.
- Evidence:
  `docs/internal/release-gates/20260719_full_suite_checkpoint/`.

### 2026-07-19 MiniMax M2.7 effective no-tool protocol override

- Current source `ffb9ed7db` fixes a shared Chat/Responses prompt-state bug:
  retained public schemas with `tool_choice=none` no longer seed parsers or
  answer policy as though tools were rendered. The pre-fix live Chat
  continuation had no visible content and contradictory stop/length terminals.
  After a real Electron Stop/Start, the identical request emitted 18
  progressive content deltas, exact-finaled, emitted one stop, one terminal
  usage event, and one `[DONE]`; 173 tokens restored as
  `paged+disk+tq-native`. Retained-schema Responses independently emitted 19
  content deltas and completed once.
- The real Electron UI also passed an Auto reasoning/content/terminal turn,
  exactly one `file_info(panel/package.json)` call/result/exact-final, and a
  no-second-tool same-chat recall. Raw Chat and Responses passed stream,
  non-stream, required tool, and result continuation. Focused regression is
  244 passed with three intentional deselections.
- Evidence:
  `docs/internal/release-gates/20260719_m27_protocol_parity/`.
- This advances the scoped Chat/Responses row only. API/protocol parity stays
  `PARTIAL` until Anthropic, Ollama, cancellation/disconnect/mid-stream
  recovery, and signed-app repeat have current-source live evidence.

### 2026-07-18 current override

- vMLX `1.6.11` is a public, signed, notarized checkpoint. The package was
  built from engine/source commit
  `95b2caa956c592a9caa706f2a790dcd5664721b7`; the final annotated tag,
  `origin/main`, `origin/codex/live-electron-gates-20260715`, and this release
  evidence head all resolve to
  `df244c4a858df3894fa3911b270d6d1b175966d6`.
- Public surfaces are live: the `jjang-ai/vmlx` source release,
  `jjang-ai/mlxstudio` Sequoia/Tahoe DMG release, PyPI `vmlx==1.6.11`, the
  raw/site updater feeds, and the Homebrew cask. The final current-head live
  release contract reports `status=pass` with no failed checks in
  `docs/internal/release-gates/20260718_v1_6_11_release/`.
- Both final DMGs passed signature, notarization-ticket, staple, Gatekeeper,
  and installed-app smoke verification. The installed Sequoia and Tahoe apps
  each loaded Gemma 4 through the real Electron Start action. Sequoia also
  proved three UI turns, one real tool continuation, separate reasoning and
  content SSE, and a disk-restored mixed-SWA prefix; Tahoe independently
  proved UI and Responses streaming.
- This release checkpoint does **not** close the retained rows below. Family
  reliability/latency/eviction, larger media and Omni audio, protocol/gateway
  soak, broader eager loading, locale/narrow-window UI, and stale-path UX stay
  post-release `PARTIAL`/`OPEN` until current-source live evidence closes each
  one.

### 2026-07-18 post-release source/meta-audit override

- Current pushed post-release source head is
  `db07a6fc1b319c65bc8d47e72f18c3e46a8bbb5b` on
  `origin/codex/live-electron-gates-20260715`.
- A shared ANSI-safe count parser now owns pytest/Vitest proof counts for 15
  cross-matrix runners; every consumer hashes that source. The clean-checkout
  orchestrator bootstrap and the stale Native-MTP JANG_2K marker were repaired.
- Focused source verification is **222 passed, 1 skipped**. The canonical
  no-heavy sweep's 658-test focused sub-suite is **656 passed, 1 skipped, 232
  deselected**. Full panel verification is **2311 passed, 3 skipped**, with
  typecheck and a clean-JANG production build passing.
- Overall post-release campaign status remains `PARTIAL/OPEN`. The canonical
  sweep still names missing MiMo bundles, signing preflight, release-manifest
  readiness, current DSV4 live tool/cache evidence, and absent historical
  live speed/quality/cross-family artifacts. No live model ran in this
  meta-audit, so no runtime row inherits a pass.
- Evidence:
  `docs/internal/release-gates/20260718_cross_matrix_count_parser/`.

### 2026-07-18 post-release Qwen 3.6 JANGTQ Auto-stream override

- The 35B artifact is bundle-grounded as `weight_format=mxtq`,
  `profile=JANGTQ2`, and `codec=turboquant_codebook`. It is JANGTQ/MXTQ, not
  affine JANG or base MLX MXFP. Its separate hybrid cache policy stores only
  attention KV as TurboQuant q4 and persists native SSM companion state.
- The current post-release source repairs an ordinary Qwen Auto request that
  carries a tool catalog but explicitly requests no tool: Qwen may now reserve
  the bounded visible-answer share instead of allowing hidden reasoning to
  consume the full output cap. Required, named, and explicit-tool requests
  remain unpartitioned and fail closed.
- Final-source Electron PID 63899 row 126 emitted separate reasoning then exact
  non-empty three-line content. Same-chat row 129 executed one real
  `file_info({"path":"panel/package.json"})`, exact-finaled 5.2 KB, and
  restored 325 tokens as `paged+ssm+disk`. Raw Responses emitted 237 reasoning
  and 14 timed content deltas before one completed terminal. The affected
  focused matrix passed 101 tests. Raw Chat emitted 1,024 reasoning plus 353
  content deltas for ordinary Auto, one valid explicit tool call, then 152
  reasoning plus 17 content deltas for the exact post-result continuation.
  Each Chat phase ended with the correct finish reason and one `[DONE]`; the
  ordinary control retained a native strict-format miss.
- Explicit native-tool reliability remains `PARTIAL-STOCHASTIC`: row 63 and a
  distinct raw prompt emitted `file_info` without required `path` and were
  safely rejected; repeated-history row 69 reused an old result without a new
  call; row 72 needed 52,343 reasoning characters before its eventual valid
  call. In contrast, 11/11 fresh Electron tools, 3/3 fresh Auto-to-tool pairs,
  the final-source pair, and 12/12 exact raw prompt repeats were valid. No
  synthetic tool call, guessed argument, hidden sampler clamp, or forced
  thinking-off retry was introduced.
- Evidence:
  `docs/internal/release-gates/20260718_qwen35_jangtq_auto_partition/`.

The historical pre-release statements immediately below are retained as
provenance and are superseded only for release/version/public-surface truth by
the current override above.

- Working branch: `reconcile/1.5.68`; current scoped code head `45c64f85e`;
  typed-settings,
  non-MTP architecture-hint, paged resident-accounting, typed hybrid-companion
  ownership, and v8 cache-namespace repairs plus their focused tests are pushed
  to the closeout branch described below.
- Push target: `origin/codex/live-electron-gates-20260715`.
- At scoped code head `45c64f85e`, the branch is 153 commits ahead of
  `origin/main` and zero behind. Matrix-only commits may follow the scoped code
  head.
- Source versions are `1.6.11` in `pyproject.toml`,
  `vmlx_engine/__init__.py`, and `panel/package.json`.
- Public GitHub app release, PyPI, and `mlxstudio/latest.json` are all 1.6.10.
- Version-truth reconciliation (2026-07-18, live-verified): the PUBLIC feed
  `jjang-ai/mlxstudio` `latest.json` serves 1.6.10 (curl raw, today). The
  repo-TRACKED `latest.json` on this branch still reads 1.5.67: it is a
  release-process artifact only rewritten by the publish chain at release time
  on the releasing branch, never a live pointer — this branch forked from the
  1.5.68 reconcile lineage and has not run a publish. Source versions 1.6.11
  are the unreleased candidate. No feed/publish change is made or allowed
  without a separate explicit PUBLISH.
- The Laguna parser-default migration is committed and pushed as `7b45676ce`.
  Current Electron main launched PID 32806 with `--tool-call-parser glm47`,
  and the session is stamped migration version 1.
- No package, version bump, tag, signing, notarization, feed update, PyPI
  upload, or GitHub release is allowed until the red rows below close.

## Latest current-head override — HY3 bounded TQ4 L2 and agent streaming

Status: `SCOPED_HY3_LONG_TQ4_CACHE_API_ELECTRON_PASS_STOCHASTIC_FORMAT_AND_BROADER_RELEASE_GATES_OPEN`.

- Retained failure control: PID 77153 aborted on the second 8K-class TQ cache
  pass. The current crash report records `SIGABRT` on
  `com.Metal.CompletionQueueDispatch`; the kernel reported 400,000 leaked
  IOGPU resources. A first direct-encoder attempt still retained every lazy
  page/layer encode graph and reached `[metal::malloc] Resource limit (499000)`
  with zero disk writes.
- Commit `45c64f85e` fixes the shared storage boundary rather than special
  casing HY3. `encode_tq_block` now uses the immutable TurboQuant encoder pair
  directly and never calls live-cache `compress()`. Native-TQ pages are
  evaluated/serialized in `extract -> write -> extract -> write` order, so
  Metal resources are bounded by one complete page instead of the whole
  prompt. Focused verification passes 15/15 TQ page tests plus 35/35 adjacent
  TurboQuant/prefix/terminal-cleanup tests.
- Live PID 80838 stored a new 9,061-token q4 prefix as all 142 native-TQ
  pages. The identical request reached first content in 23.073s cold, 5.802s
  on same-process `paged+tq-native` reuse, and 10.763s after a visible
  Electron Stop/Start as `paged+disk+tq-native`. Every run streamed ten
  content deltas, returned exact `HY3-TQ-TTFT-D=583`, and ended with one
  completed terminal. Restart health reported 9,061 cached tokens, 142 disk
  hits, and 142 native-TQ hits. No second crash report appeared.
- Electron Auto-thinking row 369 emitted 759 reasoning and 76 content IPC
  stream events, then exact six-line `HY3-UI-STREAM1-DONE`. Same-chat tool row
  372 executed one `file_info` but retained a stochastic 0.90 draft/correction
  format miss. That failure is not hidden. A raw deterministic Responses
  continuation remained exact on a 457-token `paged+tq-native` hit. Same-chat
  Electron row 375 at explicit temperature 0 then executed exactly one
  `file_info(vmlx_engine/tq_disk_store.py)`, streamed the answer from 1 to 135
  characters, persisted exact six numbered lines, and reused 5,629
  `paged+tq-native` tokens. This discriminates the retained row-372 model
  sampling miss from global SSE/parser failure or deterministic cache drift.
- Evidence:
  `docs/internal/release-gates/20260716_release_closeout/hy3-tq-bounded-current/`.
- This closes only the current HY3 long TQ4 storage/resource, matched
  cold/RAM/disk reuse, MTP-D1, Responses stream, and same-chat tool-loop row.
  The broader family/media/settings/protocol/full-test/release matrix remains
  `PARTIAL_NO_RELEASE`.

## Current blockers

| Area | Status | Current evidence | Required closeout |
|---|---|---|---|
| Laguna parser migration | PASS-LIVE / COMMITTED | Electron UI/DB/argv migrated to `glm47`; rows 1992/1995 each executed one `file_info` and exact final text; 94 parser/migration tests and panel typecheck passed | Keep as a release regression row |
| Laguna reasoning | PASS-LIVE current scoped repeat + post-reasoning streaming / PARTIAL reliability+raw strict format | Retain the old TQ3 loop and row-2022 unsolicited `ask_user` controls. On current head, fresh Electron rows 327/330/333 each separated reasoning from exact `TQ8-COLD1=45`, persisted no tool payload or warning, and included a no-clear PID 64431→65648 restart. Raw `curl -N` Responses reasoning streamed from +1.002s through +20.918s; nine content deltas followed from +20.999s through +21.326s before one completed terminal. Direct API retained leading/trailing newlines. After shared renderer commit `a7b34bc4a` and a visible single-model Bonsai→Laguna swap, raw row `LAG-RAW-STREAM2` emitted 201 reasoning and 86 timed content deltas. Electron row 366 then painted 369 visible-content increments over 4.208 seconds after reasoning, ended with `LAG-UI-STREAM2-DONE`, restored 4,096 `paged+disk+tq-native` tokens, and persisted no warning. The model added an introductory sentence, retained as a strict-format miss. | Long reasoning/agent soak and strict byte-format closeout; keep the historical variability red without sampler coercion or synthetic think tags |
| Laguna cache/perf | PASS-LIVE correctness / PARTIAL latency+long eviction | Current Auto is storage-only TurboQuant q4, not the stale TQ8 row. Electron cold/warm/restart rows used the same 4,178-token prompt; warm restored 4,174 `paged+tq-native` tokens at 2.50s TTFT and restart restored 4,174 `paged+disk+tq-native` tokens at 5.22s. Health records q4 K/V, 66 TQ-native disk hits, successful rewrap, and 1.19-1.25s reconstruction. Commit `e9af64474` makes Cache/Perf visibly report that TQ q4 storage instead of contradictory disabled legacy labels | Optimize/accept the still-slower restart TTFT with a measured release budget; add long-context and bounded-eviction proof |
| Bonsai hybrid restart | PASS-LIVE exact boundary / PARTIAL partial prefix | Two independent 1-bit PID replacements restored 160/168 tokens and ternary restored 153 tokens as `paged+ssm+disk`, with native-TQ plus SSM disk hits, ~0.10s reconstruction, one tool, and exact finals. A longer 64-token KV prefix without a companion safely full-prefilled and wrote the missing checkpoint | Repeat partial-prefix repair as a measured hit, then long-context/eviction |
| Bonsai current-HEAD regression | PASS-LIVE agent correctness + restart + eviction + scoped Electron streaming / PARTIAL soak | Current Electron PID 83540 launched 1-bit with `--tool-call-parser qwen`, `--reasoning-parser qwen3`, paged cache, Block Disk L2, and Auto. Rows 2286/2289 prove same-chat two-turn exact one-tool/final behavior. Fresh exact old repro row 2292 produced one real `file_info(panel/package.json)` and exact `B1-UI-TOOL3-DONE`; screenshot `bonsai-b1-ui-tool3-current-pass.png` captured the live UI. Health after the rows shows `tq_native_enabled=true`, 17 native-TQ L2 writes, four block-disk hits, SSM companion disk stores, and only one local serve process for Bonsai. UI restart produced PID 84219; rows 2294/2298 again finalized with one real tool and exact visible markers. The changed-prefix row correctly recorded `hybrid_kv_without_ssm_hits=1` / `reason=no_ssm_companion_state`; source trace `mllm_batch_generator.py:5814-5863` releases that KV-only candidate and full-prefills rather than using an unsafe hybrid prefix. A second UI restart to PID 84984 replayed the exact row-2298 prompt; row 2301 restored 154 tokens as `paged+ssm+disk`, `disk_hit=true`, `reconstructed=true`, `dequantized=true`, `tq_native_hits=3`, and no `hybrid_kv_without_ssm`, while again producing one real tool and exact final. UI-applied four-block PID 85595 (`--max-cache-blocks 4`) rows 2304/2306 each executed one real tool and exact finals; health showed `max_blocks=4`, `free_blocks=0`, `l1_evictions=9`, `l2_block_tokens_on_disk=21303`, `l2_ssm_tokens_on_disk=18482`, `tq_native_enabled=true`, and SSM companion evictions. UI restored 1,000 blocks on PID 85909. Earlier 1-bit loop row 2028 remains retained. Current repro row 2349 now proves the agent loop does finish: one real `file_info(panel/package.json)`, exact `B1-UI-TOOL3-DONE`, `cachedTokens=158`, `cacheDetail=paged+ssm+disk`, TQ8 native hits, and SSM companion disk hits. It also retains the reliability issue: 4,222 chars / 1,101 tokens of native reasoning before the tool call, visible as a large Reasoning panel. Commit `a7b34bc4a` fixes the shared Electron paint boundary: the pre-fix coherent row 360 collapsed post-reasoning content into one terminal paint even though a raw Responses probe emitted 406 reasoning and 46 timed content deltas. After a true Electron-main restart, row 363 exact-finaled while the DOM recorded 173 distinct visible-content mutations over 1.998 seconds; it restored 216 `paged+ssm+disk` tokens. Health then showed four native-TQ q8 disk hits and one SSM companion disk hit. | Regress the shared paint fix on non-Bonsai families; long-context/output soak without forcing sampler defaults; keep long pre-tool reasoning under UI/API observation and determine whether a non-fake request policy can reduce it without disabling reasoning globally |
| Bonsai native Qwen tool stream | PASS-LIVE explicit one/two-tool contracts / PARTIAL broader catalog | Earlier commit `f993e36b8` preserves Qwen's native schema and proves progressive exact-once `file_info` through API/Electron plus RAM/restart cache hits. Commit `a1a6591b9` repairs the panel's explicit multi-tool schema retirement and final TPS accounting; commit `3d32b944b` repairs the shared Qwen Responses continuation so a real result is terminal only when no explicitly requested/client-narrowed tool remains. All 139 prompt/format tests pass. A raw `/usr/bin/curl -N` three-round harness now executes exactly one `file_info`, exactly one `run_command(pwd)`, then streams exact `B1-API-MULTI1-DONE` over nine timed content deltas with one completed terminal and no warning. After a visible Electron Stop/Start, fresh row 321 executes those same two tools once each and visibly exact-finals `B1-CURRENT-MULTI7-DONE` at the corrected 52.8 t/s. Health records q8 native-TQ storage-boundary writes/hits, `paged+ssm+disk` dequantization, and native SSM disk restore; `compress_after=0` means no mid-request resident-memory compression is claimed. Rows 309/315 and the pre-fix 768-token raw continuation remain retained red controls. | Sample the shared continuation on other Qwen artifacts; broaden unconstrained catalog/repeated-tool reliability and keep verbose/repeated native reasoning under soak |
| Bonsai media-keyed hybrid cache | PASS-LIVE image + video-A cache / PARTIAL cross-video exactness+catalog | The real 1-bit `Qwen3_5ForConditionalGeneration` artifact advertises vision config plus image/video tokens. Image A cold returned exact `Q27-EXACTONCE-ELECTRON2-DONE` in 14 progressive paints; identical A restored 4,963/4,964 tokens as `paged+ssm` and reduced TTFT from 21.33s to 0.69s. Same-shape marker image B was a zero-cache miss and returned exact `B1-MEDIA-B-DONE`; return-A restored A at 0.66s. PID 34884→36409 then restored A as `paged+ssm+disk` at 1.64s with 78 native-TQ disk hits and one SSM disk hit. Video A cold returned exact `FRAME START 2468` / `FRAME END 9753` in 15 progressive paints; identical A restored 2,933/2,934 `paged+ssm` tokens and reduced TTFT from 8.19s to 0.66s. Alternate videos were zero-cache misses and never leaked A, but Bonsai abbreviated their visible digits (`ALT START 1` / `ALT END 86`, then `START` / `END`), retained as OCR quality misses. Return-A was exact at 0.64s. Visible PID 36409→37342 restored A as `paged+ssm+disk` at 1.72s with 46 native-TQ disk hits and one SSM hit. Real `curl -N` Responses requests reused the Electron image and video prefixes and emitted 14/15 timed `response.output_text.delta` events plus one completed terminal each. Current focused media/scheduler/SSM tests pass 219 with six intentional skips. | Improve/characterize alternate-video OCR exactness without fake postprocessing; cover other advertised Qwen3.5/Bonsai artifacts. Auto-thinking image row remains retained after 1,024 reasoning paints and a truncated marker. |
| Step 3.7 media-keyed mixed-SWA cache + native reasoning contract | PASS-LIVE current image/video/cache/API scope / PARTIAL stochastic reliability+cold latency+strict content+PID telemetry | Current gate `20260719_current_step37_jangtq/` fixes a real MLX/NumPy metadata boundary: `array([0])` was falsely reduced to `[]`, dropping the base image while 169 placeholders remained. Electron A/A/B/A rows prove a 4,290-token resident hit, same-shape media-salt miss/no A leak, and return-A reuse. A real four-second distinct video returned exact `VIDEO-B-8264`. After visible Stop/Start with zero L1 tokens, image A restored all 4,290 tokens as `paged+mixed_swa+disk` with 68 block-disk and 68 q4 native-TQ hits at 1.71s TTFT. Literal Chat streamed 46 reasoning plus 42 content deltas and clean stop/usage/DONE; Responses streamed 73 reasoning-summary plus six exact content deltas and completed with 223 cached tokens. The shared MLLM source now promotes lazy worker-side L2 refaults into cache detail; live rows 473/476 both report `paged+mixed_swa+disk`, and row 476 records `disk_hit=true`, `disk_blocks=68` on an immediate same-process repeat. Expanded tests pass 513 with two intentional deselections. Retained failures are explicit: cold image/video latency, model-native post-think self-correction in some content, missing PID in the post-restart UI header, larger-video and unseeded stochastic soak. Earlier seed controls remain provenance. | Fix/retest the restarted PID surface; enlarge video/stochastic/long soak without parser hiding, sampler coercion, or synthetic output cleanup; preserve JANGTQ/MXTQ classification separately from affine JANG and MLX MXFP |
| Mistral Medium 3.5 | DEFERRED BY USER | Prior text load/cache observations are retained, but the user explicitly excluded further Mistral MXFP4 testing from this closeout run | Do not spend this campaign on Mistral MXFP4; it is not used to claim a current release pass |
| DSV4 CRACK | PASS-LIVE eager/cache/settings/eviction/stream/tool-scope/reasoning-history tiers / PARTIAL strict long quality+perf | Commit `1e15c94bd` makes DSV4 start materialize stored parameters before a first prompt; live health before any request reported `last_request_time=null` and about 99.7 GB active model memory. Native composite cache and DSML separation pass, including 3,244/3,245-token RAM and `paged+dsv4+disk` restores. Commit `012c1fe90` aligns broad-catalog fallback validation with native DSV4 tool scoping, preserves explicit path binding, and maps explicit no-tool turns to standard `tool_choice:none`. Commit `35b444ce3` preserves Responses reasoning items and replays panel tool history in per-iteration reasoning/call/result/final order. After a true Electron-main restart, row 351 coherently recalled `panel/package.json` and `5.2 KB` from the repaired history; warm row 354 restored 274 `paged+dsv4` tokens at 1.28s TTFT and visibly painted progressive reasoning before the final answer. A raw seeded replay completed coherently 5/5 with progressive reasoning/content, but only 1/5 met the requested byte-exact marker. Earlier rows 153/156/159/162 and their strict-format misses remain retained. | Diagnose/retest constrained exact-output and long factual reliability without sampler coercion; quiet speed; exact JANGTQ bundle only if locally available |
| MiniMax-M3 | PASS-LIVE scoped tools/image+video/cache/terminal transport / PARTIAL exact OCR+REAP | Commit `8df1bfe86` keys M3 media prefixes from original content, salts BlockAware RAM/L2 fetch/store, rejects unsalted alternatives, and reuses only a prefix strictly past every media-token position. The first implementation's retained row 243 exposed a real same-shape A→B collision (746 cached tokens and A answer leakage); raw-media forwarding owns the fix. Post-fix Electron image rows 246/249/252/255/258 prove exact A cold, 746-token RAM hit, same-shape B zero-hit/no-A-leak, return-A hit, and exact `paged+disk` restore. Deterministic MP4 rows 267/270/273 prove exact `BANANA8426` cold, 1,690-token RAM, and restart/L2 restores. Distinct two-frame video B row 276 was a zero-cache miss with no A leak; after an explicitly excluded bad-settings row, return-A row 282 was exact with 1,690 `paged` tokens at 1.31s. Raw Responses reused 1,690 tokens and progressively emitted `BAN`/`ANA`/`842`/`6` before matching done and one completed terminal. Native MSA stays unquantized generic-TQ because its index-key tuple is not plain KV. Focused rerun passes 207/207. | 14-second/28-frame video now passes real Electron attachment, 1,701-token paged+disk restore, progressive raw Responses, and a 0.0415s last-content-to-done terminal gap after moving clean M3 rederive behind terminal dispatch. Exact digit OCR/ordering remains partial; live REAP32 only without host-reboot risk. Evidence: `20260719_m3_terminal_dispatch_large_video/` |
| MiniMax-M3 Chat namespace/tool stream | PASS-LIVE current-source scoped | `20260718_mm3_chat_namespace_stream/` retains the pre-fix live leak and the current-source gate. The M3 parser strips the complete `]<]minimax[>[` separator plus its observed one-character terminal truncation, while Chat buffers its prefix before any byte is visible. A true Electron Save & Restart replaced PID 67052 with 67856. Raw Chat produced 24 reasoning deltas, zero leaked content, one valid tool call and a 20-content-delta result continuation; Responses produced one valid call and a 75-content-delta continuation. Fresh Electron row 138 executed one `file_info`, progressively painted, exact-finaled, and stored no warning. Focused M3/server regression is 46/46. | Keep this as a parser/protocol regression row; broader M3 media/REAP issues remain in the family row above |
| MiniMax-M3 Auto reasoning + real tool continuation | PASS-LIVE current-head scoped | Electron Start loaded PID 19963 and left DSV4 stopped, proving the one-model swap. Bundle-grounded settings were Auto reasoning, Responses wire, tools enabled, temperature 1.0/top-p 0.95; controlled temperature 0/max 512 retained Auto. Row 201 visibly progressed from waiting to 42 then 413 reasoning characters before content appeared; SQLite kept 2,065 reasoning and 541 visible characters separate. Same-chat row 204 executed one real `file_info(panel/package.json)`, reused 9,004 `paged+disk` tokens, and progressively grew visible content 178→394 characters with no warning. Raw Responses no-tool/tool/follow emitted 262/46 reasoning-content, 19 reasoning plus two argument, and 15/119 reasoning-content deltas, each with a completed terminal. Chat emitted 304/51, 19 plus two tool deltas, and 37/125; call/follow finish reasons were `tool_calls`/`stop`, and every stream ended `[DONE]`. | Synthetic marker exactness was refused by model policy, while transport/parser/content remained coherent. Broader media quality, exact OCR, and REAP32 remain separate partial rows; the reproduced terminal delay is closed in `20260719_m3_terminal_dispatch_large_video/`. Evidence: `20260718_m3_auto_agent_stream/`. |
| openPangu 2.0 Flash 3M Auto reasoning + native tool protocol | PASS-LIVE current-head scoped | The real Electron Start button replaced M3 with PID 21745. Bundle/UI defaults were Auto reasoning, Responses/tools enabled, temperature 1.0/top-p 0.8; controlled temperature 0/max 512 retained Auto. Health before the first request identified native schema `openpangu_v2_composite_v2`: MLA latent KV, DSA indexer, rotating SWA, and path-dependent convolution state with exact typed prompt-disk L2; generic TQ, paged blocks, and block L2 were off. Electron row 207 kept 897 reasoning and 406 visible characters separate. Row 210 executed one real `file_info(panel/package.json)` and returned a coherent one-sentence 5.2 KB result with no warning. Raw Responses no-tool/tool/follow emitted 389/38 reasoning-content, 35 reasoning plus two arguments, and 124/151 reasoning-content deltas, all completed. Chat emitted 250/38, 35 plus two tool deltas, and 185/145; call/follow ended `tool_calls`/`stop` plus `[DONE]`. | Long-context and cancellation/disconnect soak remain partial. Evidence: `20260718_openpangu_auto_agent_stream/`. |
| openPangu 43,980-token native snapshot admission | PASS-LIVE guard / PARTIAL reuse+fallback latency | Before: exact answer but 22,090.8 MB typed snapshot copied then rejected, 186.35s TTFT, 236.0 pp/s, 138,814.3 MB peak. After the global single-active pre-copy guard: exact answer, separate reasoning/content, 103.20s TTFT, 426.2 pp/s, 115,551.8 MB peak, no false RAM/L2 entry. Raw Responses emitted 256 progressive reasoning deltas and 23 progressive content deltas plus `response.completed`; a tight 256-token cap caused a second full prefill because reasoning consumed the first pass and the native boundary exceeded both backend caps. Health exposes estimate/limit/skip counters. | Very-large native boundaries remain deliberately uncached; optimize or accept the two-prefill tight-cap policy without enabling generic TQ. 524,288 tokens and cancellation/disconnect remain unproven. Evidence: `20260718_openpangu_long_snapshot_guard/`. |
| openPangu | PASS-LIVE scoped / PARTIAL long-context+protocol | Source policy `_apply_openpangu_cache_policy` forces paged/block/TQ off and preserves typed MLA KV, DSA indexer, rotating-SWA metadata, causal-conv state, 128 sinks, and mHC runtime. Electron-loaded 3M PID 86212/86842/87268 launched with `--no-paged-cache --enable-disk-cache`, no KV quantization, and no block L2; Bonsai was unloaded by the single-model swap. Rows 2310/2313 prove same-chat exact one-tool finals; row 2313 hit 152 memory tokens. PID 87268 exact first-turn replay row 2322 restored 152 tokens from prompt Disk L2 (`cacheDetail=disk`, TTFT 0.18s), executed one real tool, and returned exact final. Health reports `native_path_dependent_composite`, schema `openpangu_v2_composite_v2`, `generic_turboquant_kv.enabled=false`, paged false, prompt disk L2 true | 512K/long-context soak, full protocol matrix, and broader openPangu bit-variant coverage; MTP remains detection-only/unwired for this family |
| Cross-model post-tool | PARTIAL | Many named families pass exact one-tool/final rows | MiMo and every remaining configured parser family need current Electron rows |
| Settings parity | PARTIAL | Cache defaults, Auto/None, typed-setting restart, selective-TQ Cache/Perf labeling, explicit Tool Parser None, explicit per-chat Min-P zero, gateway LAN/port persistence, and single-model swap now have scoped source-plus-live proof. Commit `d49f500a3` preserves slider zero in SQLite and both wire builders; clean current-source Electron PID 8935 displayed Min P `0.00`, DB stored `min_p=0.0`, and live `[CHAT_DIAG]` serialized `"min_p":0`. Commit `4e13b19a7` keeps parser None literal. Current-source Electron PID 9909 also exercised gateway conflict rollback, LAN rebind/restore, and the session-manager single-model swap below. Commit `e9af64474` additionally makes both live Cache and Perf panels display Laguna's effective `turboquant-q4 (K q4 / V q4)` prefix storage and suppress the contradictory disabled legacy card. | Complete remaining UI/DB/preview/argv/health rows across model-derived defaults and cache controls; retain parser None, Min-P zero, gateway rollback, single-model swap, and effective TQ storage labels as regression rows |
| API/protocol parity | PARTIAL | Responses emits the standards-matching `response.incomplete` terminal for length caps and cancellations, and the Electron client consumes completed/incomplete final text, usage, warnings, and status symmetrically (`a36a5ea66`, `ae498c70b`). The explicit-cancel gate retained incomplete item state, emitted only `response.incomplete(reason=cancelled)`, removed partial history, reached zero active requests, and recovered immediately after a five-delta client disconnect. Standard Responses usage is terminal on `response.completed`; the private `response.usage` telemetry event is header-gated to the local Electron client (`cc4251318`). A current-head Laguna spot-check at `76e8d6c1e` emitted seven direct and eight gateway progressive content deltas, exact finals, one completed terminal each, and zero private usage events; explicit local negotiation emitted ten private usage events plus the same standard terminal usage. Current Bonsai gateway controls also streamed across Chat Completions, Responses, Anthropic, and Ollama. A controlled Responses `tool_choice:none` result continuation completed once but repeated native tool markup to the cap on another run, so tool-result synthesis remains variable. Commit `57d5bcd0f` fixes the shared Electron SSE error boundary: live Chat Completions and Responses probes visibly surface `PROBE PREFILL FAILURE` and persist no false assistant success row. Current Electron-gateway Chat, Anthropic Messages, Ollama chat, and Ollama generate now have scoped live stream and non-stream downstream-disconnect proof: abandoned requests returned idle in 0.029-0.037 seconds, emitted no false terminal, and immediate recoveries completed exactly with truthful terminal and usage. | Stable auto tool/result continuation through each protocol, explicit cancellation semantics beyond downstream socket loss, safe live mid-stream exception injection, signed-app repeat, and strict Responses formatting. Evidence: `../20260719_response_cancel_disconnect/`, `../20260719_responses_usage_extension_parity/`, `../20260719_gateway_disconnect_recovery/`. |
| Gateway lifecycle | PASS-LIVE lifecycle+repeated swap+disconnect recovery / PARTIAL agent protocols | Commit `e76cc5451` makes restart transactional: a rejected port change restores the prior listener and rethrows the original error. Through the current Electron API page, changing running gateway `127.0.0.1:8081` to DSV4's occupied `8012` first reproduced the old stopped-listener bug, then the fixed build rejected the conflict while health and SQLite remained running on 8081. LAN UI enable rebound to `0.0.0.0:8081`, displayed routable `192.168.1.110`, served `/health` over that LAN address, and rebound to localhost when disabled. With Single model mode enabled, the visible Bonsai Start control stopped DSV4 PID 10013 and launched Bonsai PID 10495; UI, SQLite, process listing, gateway discovery, and `[SESSIONS]` lifecycle log all showed exactly one running engine. The current repeated Start soak then completed two MiniMax M2.7/Laguna round trips with PID sequence `70292 -> 78868 -> 79430 -> 80033 -> 80479`, one engine after every transition, prior endpoints down, and both families eagerly loaded before any request. Commit `a0aa81a94` waits for usage and `[DONE]` before emitting Ollama's empty-message terminal, preventing cumulative thinking duplication and premature loss of `eval_count`; current Electron PID 12046 / Bonsai PID 12114 proved the live stream above. The current-source gateway now cancels headerless upstream work as soon as the downstream response closes; live streaming and non-streaming Chat, Anthropic, Ollama chat, and Ollama generate aborts returned Laguna idle in 0.029-0.037 seconds and immediate exact recovery requests completed normally. | Agentic tool/result continuation per protocol, safe injected error recovery, broader model/parser coverage, and signed-app lifecycle repetition. Evidence: `../20260719_one_model_swap_soak/`, `../20260719_gateway_disconnect_recovery/`. |
| Full tests/build | PASS-CURRENT-SOURCE / PARTIAL next signed-package repeat | The complete clean-JANG Python suite passes 6,153 tests with 96 explicit skips and 92 deselections. After the 2026-07-20 accessibility follow-through, the complete panel suite passes 2,340 tests with three skips; TypeScript typecheck and the direct main/preload/renderer `electron-vite build` pass. The released v1.6.13 production build previously used clean JANG source and verified bundled Python critical hashes/imports. The current umbrella `npm run build` correctly refuses the dirty primary JANG checkout; the guard was not bypassed. Evidence: `20260719_minwidth_locale_drawers/`, `../20260720_minwidth_drawer_followthrough/`, and `../20260720_minwidth_accessibility_followthrough/`. | Preserve these logs; use a clean pinned JANG source before the next bundled-Python build and repeat the signed/notarized package/install-smoke gate only at the next selected checkpoint |
| Eager session materialization | PASS-LIVE scoped across DSV4, Laguna, Step, openPangu, Gemma, HY3, and MiniMax M2.7 / PARTIAL remaining loader classes | DSV4 retains its `skip_params_eval=False` proof. The real Electron Start control also loaded Laguna PID 70292 with `last_request_time=null` and 82,631.3 MB active memory before a request. Existing current-source gates independently prove the same Start-before-request boundary for Step (`model_loaded=true`, `last_request_time=null`, single-model Gemma→Step swap), openPangu (single-model Step→openPangu swap), Gemma mixed-SWA (post-restart health with `last_request_time=null` and 9,681.9 MB active memory), HY3 native-MTP D1, and full-KV MiniMax M2.7 JANGTQ. The repeated current-source Start soak additionally swapped M2.7/Laguna twice, preloaded both without a request, and retained exactly one engine. These rows cover dedicated JANGTQ, affine JANG, base MLLM, typed openPangu, mixed-SWA, MTP, and full-KV loader representatives without another generation loop. | Keep the named routes as regressions. Inventory only loader classes still lacking current Start-before-request evidence; retain signed-app repetition as open. Evidence: `../20260719_laguna_eager_current/`, `../20260719_step37_mixed_swa_disk_only_ui/`, `../20260719_openpangu_current_disk_restore/`, `../20260719_gemma_mixed_swa_disk_only_ui/`, `../20260719_current_hy3_mtp/`, `../20260719_minimax_m27_tq_hierarchy_protocol/`, and `../20260719_one_model_swap_soak/`. |
| Responsive Electron chrome | PASS-LIVE current-source scoped toolbar/drawers/native Korean confirm/base keyboard / PARTIAL transient+secondary-modals+screen-reader+next signed repeat | Existing 600 px source proof covered five locales but did not falsify fixed-width drawers. The signed v1.6.13 Sequoia control retained here clipped lifecycle/settings controls beyond 600 px, and the current-source Korean Server drawer measured `x=216..600` inside an `x=260..600` main pane, losing its first 44 px under `overflow:hidden`. The shared repair makes toolbar controls wrap/shrink, bounds Chat/Server wrappers to the visible pane, and routes the observed Chat Settings agent controls through the catalogs. Current-source Electron at 600x760 now measures the Server drawer and wrapper exactly `x=260..600`, Chat Settings `x=280..600`, and document width 600/600. The real Clear All action opened a localized Korean native confirm sheet and was dismissed with all three chats retained. Theme/voice icon controls now expose localized title/aria names; live theme clicks traversed dark/light/system and restored dark, while 25 unique base-Chat Tab stops were named and fully within the viewport with zero failures. Complete panel suite passes 2,340 tests with three skips; typecheck and direct production Electron source build pass. Evidence: `../20260720_minwidth_drawer_followthrough/` and `../20260720_minwidth_accessibility_followthrough/`. | Force remaining wait/empty/image and secondary custom-modal states, perform drawer/modal-specific keyboard plus full screen-reader semantics, and repeat these repaired surfaces in the next signed/notarized app |
| Packaging/public release | PASS-PUBLIC v1.6.13 checkpoint / current post-release UI follow-through unshipped | Public v1.6.13 Sequoia/Tahoe DMGs are signed, notarized, stapled, Gatekeeper-verified, install-smoked, and published across GitHub/PyPI/feeds/Homebrew as recorded in the v1.6.13 addendum below. The responsive/i18n changes after `f8bc773a8` are current-source only and are not part of those public artifacts. | Preserve the immutable v1.6.13 evidence; only package/publish a newer checkpoint after its retained release blockers are explicitly accepted or closed |

### DSV4 eager materialization and broad-tool continuation — current source

- Commit `35b444ce3` repairs a shared Responses history contract rather than a
  DSV4-only output rewrite. The server now accepts and preserves Responses
  reasoning items; the panel reconstructs persisted assistant history in
  per-iteration reasoning, function-call, function-result, and final-output
  order. A restarted Electron main sent the repaired wire shape and the server
  reconstructed seven ordered messages. Row 351 coherently recalled the real
  prior tool result; warm row 354 restored 274 `paged+dsv4` tokens at 1.28s
  TTFT while reasoning painted progressively. The raw five-seed replay was
  coherent and terminal-complete 5/5, but strict marker matching was 1/5, so
  this does not close constrained-format or long-context reliability.
  Evidence is in `dsv4-reasoning-history-current/`.

- Commit `1e15c94bd` changes only the DSV4 JANGTQ load route to evaluate stored
  quantized parameters during session start (`skip_params_eval=False`). In the
  Electron UI, Stop then Start completed before any prompt. Current health at
  that boundary reported `model_loaded=true`, `last_request_time=null`, and
  about 99.7 GB active model memory. This is materialization evidence, not a
  synthetic warmup or prompt-cache hit. Focused eager tests pass 18/18.
- A 3,245-token DSV4 cache control cold-prefilled, then restored 3,244 tokens
  from resident `paged+dsv4` blocks at 1.50s TTFT. Visible process restart
  without clearing L2 restored the same 3,244-token boundary as
  `paged+dsv4+disk`. A changed TULIP prompt did not leak the ORCHID control.
  Raw Responses produced progressive output and a matching terminal.
- Before commit `012c1fe90`, the same DSV4 history completed with only
  `file_info` authorized but looped literal `response` when all 33 built-ins
  were authorized. Source trace showed fallback validation comparing the
  native scoped prompt against the entire catalog and injecting a second
  prompt. The repair validates against the same explicit/recent-tool scope,
  retains request-bound explicit argument fallback, and accepts slash-bearing
  paths. No artifact, sampler, or output-rewrite exception was added.
- Current raw Responses with all 33 tools emitted ten progressive content
  deltas and exact done text `SIZE FIVE POINT TWO KB DONE`. Fresh Electron row
  153 executed one real `file_info(panel/package.json)` and completed with the
  real 5.2 KB result. Rows 156/159 explicitly disabled tools and emitted no
  tool status/call/result fields. The panel sends standard `tool_choice:none`
  in both Responses and Chat Completions for those directive-shaped turns.
- Electron row 162 retained 1,595 DOM mutations: reasoning began painting at
  about 2.33s and visible content grew from 16.732s through 27.336s. It did not
  freeze then batch. It did miss the requested exact marker, and row 159 used
  two sentences, so DSV4 constrained-format/long reliability remains
  `PARTIAL`. Negative controls are intentionally retained in
  `dsv4-tool-scope-current/`.
- Focused validation: 40/40 combined Python fallback/DSV4 hardening tests,
  162/162 affected panel tests, panel typecheck, and `git diff --check`.

### Shared terminal dispatch before cache persistence — current source

- The freeze-then-batch symptom was reproduced below Electron on an exact
  4,629-token Qwen 27 MTP request. Before the repair, the resident
  `paged+ssm` hit emitted its first eleven visible deltas from 0.9990s through
  2.4198s, then withheld the final delta until 10.6316s while synchronous
  paged/TQ/SSM persistence ran.
- Source trace found the same ordering defect in both async schedulers:
  `EngineCore._engine_loop` and `MLLMScheduler._process_loop` called terminal
  cache cleanup before the consumer could flush the terminal output. Commit
  `aa6a3d2ef` defers cleanup only for the async engine paths, dispatches and
  yields first, then performs cache persistence on the same model worker before
  the next scheduler step. Direct synchronous `step()` callers retain the old
  cleanup-before-return default. No parser, sampler, prompt, or model-family
  special case was added.
- The matched post-fix long-prefix row emitted all twelve cold deltas from
  7.7420s-9.2557s and all twelve resident-hit deltas from
  1.0256s-2.5516s. It restored 4,628 `paged+ssm` tokens for a 7.549x matched
  first-content improvement. After visible Electron restart without clearing
  L2, the same request restored 4,628 `paged+ssm+disk` tokens and streamed all
  deltas from 0.9171s-2.4300s.
- Raw current-source Chat turn 1 emitted 334 reasoning / 10 content deltas;
  same-conversation turn 2 emitted 350 / 18, recalled its codeword, and reused
  52 `paged+ssm` tokens. Responses emitted 321 reasoning / 11 content deltas,
  matching `output_text.done`, and one `response.completed`. Electron emitted
  118 reasoning / 11 content paints, executed exactly one real
  `file_info(vmlx_engine/scheduler.py)`, persisted a matching OpenAI call/result,
  and exact-finaled. The failed tools-disabled setup row is retained separately.
- Focused terminal-order tests pass 4/4; the affected suite passed 50/50. A
  wider cache/batching slice passed 260/261, with only the already-retained
  unrelated source-string assertion
  `test_streaming_tool_detection_requires_request_tools` failing. Evidence:
  `qwen27-mtp-stream-cache-current/`.

### Terminal persistence admission and externally final stream contracts — current source

- Commit `016d661ca` closes the race left after terminal output was deliberately
  dispatched before slow paged/TQ/typed-companion persistence. Both async
  schedulers now clear a terminal-cleanup event before dispatch, reopen it in a
  `finally` after cache persistence, and wait on that event before admitting a
  new request. Source trace: `vmlx_engine/engine_core.py:100-104,189-195,292-305,472-475`
  and `vmlx_engine/mllm_scheduler.py:882-886,4026-4053,4074-4079,4180-4184`.
  This preserves progressive terminal emission without allowing an immediate
  identical request to select a partially persisted prefix.
- MiniMax M2.7 live control: an identical 3,756-token request moved from cold
  first content at 23.6206s to a full 3,752-token resident
  `paged+tq-native` hit at 7.4957s. After Electron process restart without
  clearing L2, the full 3,752-token prefix restored as
  `paged+disk+tq-native` at 1.4364s. Logs show the full store completed before
  the next prefix selection. Raw Chat/Responses and the Electron tool row all
  completed with progressive reasoning/content; Electron executed one
  `file_info(vmlx_engine/scheduler.py)` and exact-finaled. Two leading newlines
  in raw strict-format controls remain retained as `PARTIAL`.
- Qwen 3.6 27B MTP live control: an identical 4,623-token request moved from
  cold first content at 14.8964s to a full 4,622-token resident `paged+ssm`
  hit at 1.0513s (14.17x). Process restart restored all 4,622 tokens as
  `paged+ssm+disk` at 0.9008s. Health records 292 q4 native-TQ hits, 73 block
  disk hits, one native SSM companion disk hit, and zero unsafe
  KV-without-companion reuse.
- The first current Qwen raw stream exposed a second shared defect: Chat sent
  public finish reasons `["length", "stop"]` because the internal
  reasoning-only first pass leaked its `length` terminal before the bounded
  visible-answer continuation. Commit `9618e2e46` adds the state-based
  `server.py::_main_pass_finish_reason` in the shared Chat path. The matched
  post-fix stream emitted only `stop` on both Chat turns, with 79/7 and 79/12
  reasoning/content deltas, while Responses emitted 74/7 and one completed
  terminal.
- The first post-terminal-fix Electron Qwen exact-once row is deliberately
  retained as a failure: it executed two identical
  `file_info(panel/package.json)` calls. Cache-on and `skip_prefix_cache=true`
  raw A/B controls each generated one call, isolating the defect from TQ/SSM
  prefix restore. Source trace showed the Qwen exact-once stream-stop parser
  reused an eight-chunk generic grace window, allowing a second call before
  natural EOS. Commit `9618e2e46` scopes zero grace to explicit Qwen
  exact-once requests and preserves the completed truncated candidate for both
  Chat and Responses final parsing; ordinary Qwen multi-call behavior retains
  the generic grace window.
- After restart on the pushed head, Electron emitted 279 reasoning and ten
  progressive content updates, executed exactly one
  `file_info(panel/package.json)`, persisted one matching OpenAI call/result,
  and returned exact `Q27-EXACTONCE-ELECTRON2-DONE`. The row restored 128
  `paged+ssm+disk` tokens. This is a scoped exact-once PASS, not a blanket
  parser-family pass.
- Current validation at pushed head: 131/131 selected
  `test_terminal_dispatch_before_cache_cleanup.py`,
  `test_answer_pass_streaming.py`, and `test_server.py` tests pass, with three
  intentional deselections. Evidence, including pre-fix failures and post-fix
  Electron screenshots: `stream-cache-admission-current/` and
  `qwen-terminal-exactonce-current/`.

### Shared Electron post-reasoning paint — Laguna cross-family regression

- The shared renderer repair is now live-proven outside Bonsai/Qwen. The
  visible Server UI stopped Bonsai PID 75463 and started Laguna PID 76348 with
  Single model mode retaining exactly one active session. Health was ready
  before a prompt with `last_request_time=null`.
- A raw Laguna Responses request emitted 201 reasoning deltas, then 86 content
  deltas from 9.695s through 13.068s, then exactly one completed terminal.
  A fresh Electron chat on the same running engine recorded one final
  reasoning snapshot followed by 369 incremental visible-content mutations
  from 9.871s through 14.079s. Row 366 ended with
  `LAG-UI-STREAM2-DONE`, restored 4,096 `paged+disk+tq-native` tokens, and
  persisted no warning. Health recorded 64 native-TQ q4 disk hits. The model
  added an unrequested introductory sentence, so strict-format reliability is
  still `PARTIAL`; this row closes stream delivery and paint only. Evidence:
  `laguna-post-reasoning-stream-current/`.

### Bonsai post-dispatch q8 hybrid regression — current source

- Commit `a7b34bc4a` fixes a shared Electron streaming boundary, not a Bonsai
  output rewrite. Before the change, current row 360 persisted a coherent
  992-token response but the renderer observed only one terminal content
  paint after reasoning. The same engine/request path over raw Responses
  emitted 406 timed reasoning deltas followed by 46 timed content deltas, so
  the server and API stream were not the batching owner. Source trace found
  `panel/src/main/ipc/chat.ts::streamSSE` draining every buffered SSE line and
  sending all renderer IPC messages in one main-process turn; React then
  coalesced those updates with completion, while
  `MessageBubble.tsx::useTypewriter` snapped the terminal target.
- The repair yields the main process after every visible content delta and
  drains a just-finished renderer backlog rather than snapping it. After a
  true Electron-main restart, row 363 returned its coherent exact final and
  the DOM observer recorded 173 distinct `.prose` mutations from the first
  visible character to the final marker over 1.998 seconds. That turn restored
  216 `paged+ssm+disk` tokens; health recorded four native-TQ q8 disk hits and
  one SSM companion disk hit. Affected panel verification passed 301/301 plus
  typecheck. This closes the live Bonsai Electron paint row only; other model
  families remain explicit regression rows. Evidence:
  `bonsai-post-reasoning-stream-current/`.

- The exact family that exposed reasoning-then-batched-answer behavior now
  passes the shared scheduler repair on raw and Electron surfaces. Chat turn 1
  emitted 363 reasoning / 14 content deltas; same-conversation turn 2 emitted
  512 / 22, recalled its codeword, and reused 46 `paged+ssm` tokens. Responses
  emitted 512 / 11, matching done text, and one completed terminal. The first
  Chat marker retained two leading newlines, so strict byte-format reliability
  remains `PARTIAL` even though all three semantic markers completed.
- Current Electron emitted 51 reasoning and 12 content paints, executed one
  real `file_info(panel/package.json)`, persisted a matching call/result, and
  exact-finaled. The visible answer arrived as distinct ~20ms paints rather
  than a terminal blob.
- On one identical 4,631-token prompt, cold first content was 40.3291s and the
  4,630-token q8-attention/native-SSM RAM restore was 0.6969s (57.869x). Visible
  restart PID 94843 to 95400 restored the same prefix as `paged+ssm+disk` in
  1.5169s; the following RAM hit was 0.6716s. Health records 292 native-TQ q8
  hits, one SSM disk hit, and zero unsafe KV-without-SSM reuse. Evidence:
  `bonsai-postdispatch-current/`.

### Cross-model tool-setting inheritance — current source

- Live pre-fix inspection showed a Qwen chat with built-in tools and workspace
  saved, followed by a fresh Bonsai chat with tools unchecked and no working
  directory. `chat:create` searched only same-model siblings and stopped at the
  newest sibling even when that row had no overrides, contradicting the visible
  “last chat” contract.
- Commit `d9cef0b0c` scans recent chats across model switches, skips the newly
  inserted/override-less rows, and feeds the newest actual override through the
  existing tool/workspace-only allow-list. It does not make sampling, prompt,
  output/thinking caps, or reasoning mode sticky. The starred profile remains
  higher priority. Affected validation passed 299/299 plus typecheck.
- After full Electron-main restart, a visible Bonsai-to-Qwen switch and fresh
  chat inherited built-in tools, allowed categories, and
  `/Users/eric/mlx/vllm-mlx`. SQLite kept all model-owned generation/reasoning
  fields NULL. That inherited chat executed one real `file_info(README.md)`,
  emitted 298 reasoning / 10 content paints, persisted the call/result, and
  exact-finaled. Evidence: `cross-model-tool-inheritance-current/`.

### TQ prompt-L2 stream ownership and TTFT — current source

- Commit `db2d6d5fb` fixes a shared prompt-disk ownership defect. Safetensors
  load, TQ decode, and cache-class reconstruction now run on the same
  single-worker executor that loaded and runs the model. A controlled direct
  restore before this fix failed with `There is no Stream(gpu, 0) in current
  thread`; the identical post-fix Hy3 request completed exactly.
- The same commit permits legitimate long-context TQ packed vectors past the
  generic safetensors axis guard only when both the native-TQ metadata marker
  and an exact packed-field name are present. Decoded shapes, tensor bytes,
  file bytes, offsets, layer count, dtype, and runtime fingerprint remain
  independently validated.
- Prompt TQ layers now share decoder/codebook state and batch compatible packed
  layouts without merging layer boundaries. TQ prompt hits skip synchronous
  decoded-to-plain paged backfill; the worker restores the model's native live
  TQ cache class, and the normal completion path writes native-TQ paged blocks.
  This applies by cache layout/codec, not by a Hy3 name check, and keeps scalar
  fallback for mixed layouts.
- Source validation: 75/75 validator, prompt-disk, and paged-TQ tests plus 4/4
  scheduler ownership/direct-restore tests pass. A wider 211-test cache slice
  has one unrelated pre-existing source-string assertion failure in
  `test_streaming_tool_detection_requires_request_tools`; it is retained and
  is not counted as a pass for this change.
- Raw streamed Hy3 controls used one identical 3,737-token prompt. A clean cold
  pass reached first content in 6.3663s. Worker-owned prompt L2 restored 3,733
  tokens exactly with 11 content deltas in 7.2035-7.3279s; the following native
  paged hit produced 11 deltas in 5.9512-6.0415s. Functionality and stream
  ownership pass, but disk TTFT is slower than the matched cold reference and
  remains `PARTIAL` rather than a performance pass.
- Current Electron PID 72531 launched Hy3 through the visible Start control
  with Prefix/Paged/Block-Disk enabled, Auto TQ, and the stale manual
  `--enable-disk-cache` Additional Argument cleared. The current app row made
  exactly one real `file_info(panel/package.json)` call, emitted 38 incremental
  reasoning and 11 incremental content events, returned exact
  `HY3-ELECTRON-OWNER1-DONE`, and displayed 1,472
  `paged+disk+tq-native` cached tokens with 1.11s request TTFT. Screenshot and
  probe data are under `hy3-tq-ownerload/`.
- Nemotron hybrid/SSM provides the counterexample needed to avoid calling TQ
  globally slow. With thinking disabled, one exact 4,638-token prompt reached
  first content in 7.5035s cold. The identical resident request restored 4,631
  tokens as `paged+ssm+tq-native` in 0.4953s (15.149x), with 0.2048s worker
  reconstruction. After Electron `Save & Restart` without clearing L2, the
  first identical request restored 4,631 `paged+ssm+disk+tq-native` tokens in
  0.3950s and the following RAM hit was 0.4817s. Both remained exact and
  streamed 13 answer deltas. Evidence: `nemotron-stream-cache/nemo-tqfair1*.json`.
- Laguna plain-attention KV used the same TQ4 storage boundary on an exact
  4,635-token prompt. Cold first content was 13.7515s; 4,631 resident
  `paged+tq-native` tokens reduced it to 2.6702s (5.15x). The no-clear restart
  restored `paged+disk+tq-native` in 4.3248s, followed by a 2.6243s RAM hit.
  Its 1.286-1.391s reconstruction is materially slower than Nemotron and
  remains a performance target even though all four outputs were exact and
  incremental. Evidence: `laguna-stream-cache/laguna-tqfair1*.json`.
- MiniMax M2.7 full-attention KV provides a second strong TQ4 control. On an
  exact 3,760-token prompt, cold first content was 28.8343s; the resident
  request restored 3,756 tokens as `paged+tq-native` in 1.7701s (16.29x), with
  1.2560s worker reconstruction. Visible Electron `Save & Restart` changed the
  engine PID from 82493 to 86185 without clearing L2. The first identical
  request then restored all 3,756 tokens as `paged+disk+tq-native` in 1.4197s;
  the following RAM hit used `paged+tq-native`. Post-run health records 59
  disk-block hits for that first request, 236 native-TQ hits total, TQ4 key and
  value storage, and zero hybrid-companion fallbacks. Evidence:
  `mm27-current-stream-cache/mm27-tqfair1*.json` and
  `mm27-health-after-l2.json`.
- Bonsai 1-bit provides the hybrid q8 control. On an exact 4,623-token prompt,
  cold first content was 5.0758s; the resident request restored 4,622 tokens as
  `paged+ssm` in 0.6784s (7.482x), with q8 storage applied only to 16 attention
  KV layers while 48 companion layers remained native. Visible Electron
  restart changed PID 87231 to 87712; the first identical request restored all
  4,622 tokens as `paged+ssm+disk` in 1.5259s with a real SSM checkpoint hit.
  Health recorded 98 native-TQ writes, 225 native-TQ hits, 13 SSM disk stores,
  one SSM disk hit, and zero unsafe KV-without-SSM reuse. A controlled UI None
  restart launched PID 88154 with `--kv-cache-quantization none`, completed an
  exact streamed tool turn, wrote 34 raw blocks with zero TQ writes/hits, then
  Auto was restored on PID 88434. The original q8+SSM prefix still restored as
  `paged+ssm+disk` in 1.5499s, proving codec-namespace isolation across the None
  detour. Evidence: `bonsai-shared-q8-current/`.
- Qwen 3.6 35B provides the non-MTP hybrid q4 control. Raw Chat turn 1 emitted
  379 reasoning plus 14 content deltas; turn 2 emitted 448 plus 23, recalled
  the exact codeword, and reused 46 `paged+ssm` tokens. Current Electron turns
  emitted 56/12 and 149/14 reasoning/content events, executed distinct
  `file_info(panel/package.json)` and `file_info(README.md)` calls exactly once,
  and exact-finaled. On an exact 4,625-token prompt, cold first content was
  8.8514s; 4,624 resident `paged+ssm` tokens reduced it to 0.5543s (15.969x).
  Visible Save & Restart changed PID 88980 to 89919; the first identical request
  restored 4,624 `paged+ssm+disk` tokens in 0.4785s and the next RAM hit took
  0.5596s. Health records 292 q4 native-TQ hits including 73 disk hits plus one
  real SSM companion disk hit and zero unsafe KV-without-companion reuse. One
  Responses sample repeated its requested marker in the live deltas; three
  saved full-harness repeats were exact and source trace found no terminal/API
  replay. The miss remains a sampled reliability red; no output deduplication
  was added. Evidence: `qwen35-current-stream-cache/`.
- Verdict: source correctness, API streaming, model-worker ownership, Electron
  agent loops, and real RAM/L2 TQ TTFT speedups are `PASS-LIVE` for Nemotron,
  Laguna, MiniMax M2.7, and Bonsai 1-bit. Hy3 remains `PARTIAL`: its matched
  prompt-L2 request was slower than cold and its resident gain was only about 1.05x. TQ
  performance is therefore cache-family specific, not globally green; no
  cache-hit or release-ready claim may hide the matched cold comparisons.

### Qwen 3.6 27B post-tool progressive streaming — current source

- Commit `b33d80589` fixes two owning-layer defects without synthesizing model
  output. Native tool buffering now requires a distinctive partial marker
  prefix and releases ambiguous one-to-three-character suffixes when they no
  longer match. Explicit/Auto Qwen reasoning partitioning is enforced on the
  post-tool continuation while the initial tool-selection turn remains native.
- Current-source validation passed 832 tests with three deliberate deselections
  across server, streaming-reasoning, engine-audit, output-budget, and answer
  family suites. The nine focused marker/tool/Qwen continuation tests also pass.
- Direct Responses SSE emitted 153 reasoning deltas and 113 visible-content
  deltas, one matching `output_text.done`, exactly one completed terminal, no
  incomplete terminal, no heartbeat/tool re-entry, and exact final marker
  `Q27-API-POST-TOOL-PROGRESSIVE-FIX-DONE` under explicit
  `max_thinking_tokens=256`.
- Current Electron row 2606 executed exactly one real
  `file_info(panel/package.json)` call. The visible answer grew while Stop was
  active from 11 to 73 to 252 to 496 to 805 characters, then exact-finaled
  `Q27-ELECTRON-TOOLSTREAM-FIX7-DONE`. It reused 512 tokens as
  `paged+ssm+disk`, persisted one matching call/result pair, and recorded no
  warning. Screenshots are in the Qwen evidence directory.
- This closes the Qwen MXFP8 post-tool reasoning/content streaming defect on
  the current head. It does not reclassify Bonsai or any other parser family;
  each still requires its own post-`b33d80589` regression row.

### Bonsai multi-turn argument and parser-off recheck — current source

- The earlier one-turn proof was insufficient. Same-chat row 2352 requested
  `README.md` but executed the stale prior argument `panel/package.json`; its
  exact final marker does not make that turn correct. The red row is retained
  as native/stream reliability evidence.
- Source trace found a separate explicit-off contract bug:
  `_parse_tool_calls_with_parser()` auto-detected a model parser even after the
  UI launched literal `--tool-call-parser none`, and both streaming paths still
  armed native marker buffering. Commit `4e13b19a7` gates final parsing and
  Chat/Responses streaming on `_tool_call_parser_disabled_explicitly`.
- Live Electron parser-off PID 99835 persisted row 2358 as raw model text with
  no `tool_calls_oai_json` or tool result. The next parser-off turn generated
  3,701 reasoning tokens until visibly stopped; parser-off is therefore an
  actual opt-out, not a hidden fallback or proposed Bonsai workaround.
- Electron restored production `qwen` on PID 864. Same-chat rows 2364, 2367,
  and 2370 executed exactly `panel/package.json`, `README.md`, and
  `pyproject.toml` respectively, once each, with exact finals. Row 2373 then
  restored 258 tokens as `paged+ssm` and again executed the requested path
  exactly. Screenshot: `/tmp/bonsai-qwen-3turn-current.png`.
- The same chat then completed four more correct single-tool turns: rows 2376,
  2379, 2382, and 2385 used `tests/test_server.py`, `vmlx_engine/server.py`,
  `panel/src/main/ipc/chat.ts`, and `vmlx_engine/engine/batched.py`. This gives
  eight consecutive correct turns after the retained row-2352 failure, not a
  claim that the earlier failure disappeared.
- Bonsai remains `PARTIAL` because row 2352 and the earlier 4,222-character
  reasoning turn prove variability. No sampler clamp, prompt coercion, hidden
  reasoning disable, or argument rewriting was added.

### Shared reasoning-to-content streaming — current source

- Two independent server buffers caused the visible freeze. The bounded direct
  answer pass used a static 48-character tail and inherited full-pass buffering
  from a broad family set. Separately, the main Chat reasoning-parser path put
  every content delta from any answer-pass-capable family into
  `deferred_reasoning_visible_content`, then emitted one terminal blob. The
  Responses main path did not have that second buffer, which is why Step could
  stream through Responses while batching through Chat on the same process.
- Commit `3fe331b8e` replaces the direct-pass family allowlist with
  `server.py::_answer_pass_safe_visible_raw`: ordinary prose is immediately
  safe, while only an unresolved leading Gemma channel, split close-think token,
  or re-opened reasoning rail is withheld. The only retained full-pass family
  is `deepseek_v4`, backed by its deterministic live planning re-entry; Step,
  MiniMax, Qwen, Gemma, Laguna, and Hy3 no longer inherit that DSV4 assumption.
- The same commit removes `deferred_reasoning_visible_content` from
  `stream_chat_completion`. Once the reasoning parser exposes content, Chat now
  emits it progressively like Responses. If a cap is reached after visible
  content, the client receives the honest streamed prefix and
  `finish_reason=length`; the bounded replacement pass runs only when the first
  pass produced no content, so it cannot duplicate unretractable output.
- Focused validation is 239/239 across answer-pass family/marker handling,
  terminal finish, streaming reasoning, reasoning/tool interaction, Gemma4
  no-leak, and DeepSeek-R1 no-leak tests. The marker tests split `</think>` over
  multiple chunks and verify that a re-opened think rail never becomes content.
- Live Step raw proof after Electron `Save & Restart` loaded PID 76317:
  `STEP37-SHARED3` Chat turn 1 emitted 244 reasoning plus 14 content deltas;
  turn 2 emitted 110 reasoning plus 24 content deltas, recalled the exact
  codeword, and reused 49 `paged+mixed_swa` tokens; Responses emitted 111
  reasoning plus 12 content deltas. All three assembled exact markers and ended
  cleanly (`stop` / `response.completed`). Evidence:
  `step37-streaming/step37-shared3.json`.
- Live Electron Step tool proof used the real renderer/preload stream:
  53 progressively timed reasoning updates, 11 progressively timed content
  updates, exactly one `file_info(panel/package.json)`, exact
  `STEP37-ELECTRON-STREAM1-DONE`, and `finishReason=stop`. The screenshot visibly
  shows the Step session header, reasoning rail, one Info result, exact final,
  and metrics. Evidence: `step37-streaming/step37-electron-stream1.{json,png}`.
- Bonsai current-source `B1-SHARED4` emitted 262/13 reasoning/content deltas on
  Chat turn 1, 768/20 on cached multi-turn recall, and 287/11 on Responses.
  Turn 2 exhausted its first 1,024-token reasoning pass, then progressively
  emitted the bounded answer instead of freezing and batching it; all three
  outputs were exact. Electron turn 1 emitted 139 timed reasoning and 11 timed
  content updates before one real `file_info(panel/package.json)` and exact
  final. The second turn in the same chat emitted 83/13 updates, called
  `file_info(README.md)` exactly once, and exact-finaled. The controlled None
  turn emitted 52/10 updates, executed one real `file_info(pyproject.toml)`,
  and exact-finaled. Evidence: `bonsai-shared-q8-current/`.
- Nemotron current raw proof emitted 50/14 reasoning/content deltas on Chat
  turn 1, 256/23 on the cached multi-turn recall, and 43/13 on Responses; every
  terminal marker was exact and turn 2 used 53 `paged+ssm+tq-native` tokens.
  Electron emitted 265 timed reasoning updates and 14 timed content updates,
  executed one `file_info`, and returned exact `NEMO-ELECTRON-STREAM1-DONE`.
  Evidence: `nemotron-stream-cache/nemo-shared1.json` and
  `nemo-electron-stream1.{json,png}`.
- Laguna current raw proof also streamed progressively and exactly: Chat turn 1
  emitted 512 reasoning plus 12 answer deltas through the bounded direct pass;
  cached turn 2 emitted 324/23 with 51 `paged+tq-native` tokens; Responses
  emitted 346/12 and one completed terminal. Electron emitted 33 timed reasoning
  and 13 timed content updates, executed one `file_info`, and exact-finaled.
  This is a streaming pass but a reasoning/performance partial: raw turn 1 took
  48.1s and the Electron tool loop 41.4s. Evidence under
  `laguna-stream-cache/`.
- Commit `f16c51d18` removes the last DSV4 family-name full-pass buffer. The
  shared dynamic state already withholds DSV4's known `<thinking>...` variant,
  split over arbitrary chunks, and hides a terminal unclosed rail. At
  `max_tokens=64`, the live DSV4 history probe exhausted reasoning on both Chat
  turns yet streamed 13 and 24 answer deltas; the second fallback consumed the
  first fallback through history, recalled the codeword exactly, and did not
  leak `<thinking>` or `+DERIV`. Responses streamed 12 answer deltas and exact
  completion. Turn 1 abbreviated `CHAT` to `CH`, so strict-format quality is
  still partial even though the stream contract passed.
- DSV4 Electron tool behavior remains variable and is recorded, not hidden.
  `DSV4-ELECTRON-DYN1` emitted an incomplete DSML suffix; the parser hid it and
  surfaced a schema-validity warning with zero fake tool executions. The
  immediate fresh-chat `DYN2` restored 1,372 `paged+dsv4` tokens, executed one
  real `file_info`, emitted 11 timed content updates, and exact-finaled. Evidence:
  `dsv4-dynamic-streaming/`.
- HY3 current-source raw proof emitted 301/12 reasoning/content deltas on Chat
  turn 1, 399/21 on cached multi-turn recall, and 43/11 on Responses; all three
  exact-finaled, and turn 2 reused 51 `paged+tq-native` tokens. Electron emitted
  31 timed reasoning and 11 timed content updates, executed one `file_info`,
  exact-finaled, and displayed 3,964 `paged+disk+tq-native` cached tokens.
  Evidence: `hy3-current-streaming/`.
- MiniMax M2.7 current-source raw proof emitted 194/13 reasoning/content deltas
  on Chat turn 1, 292/21 on cached multi-turn recall, and 239/11 on Responses;
  all three exact-finaled and turn 2 reused 72 `paged+tq-native` tokens. The
  visible Electron tool row emitted 38 timed reasoning updates and 10 timed
  content updates, executed exactly one `file_info(panel/package.json)`, and
  exact-finaled. Evidence: `mm27-current-stream-cache/mm27-shared1.json` and
  `mm27-electron-stream1.{json,png}`.
- Verdict: source contract and live Step, Bonsai, Nemotron, Laguna, DSV4, HY3,
  and MiniMax M2.7 Chat, Responses, Electron, multi-turn, and one-tool loops are
  `PASS-LIVE` for progressive emission. Cross-family release status remains
  `PARTIAL` because DSV4 produced one malformed native tool turn and one
  strict-format miss, Laguna reasoning latency is excessive, Hy3 TQ TTFT is
  still poor, and displayed TPS still blends reasoning with any bounded
  answer-pass phase.

### Responses terminal-event correctness — current source

- Official Responses streaming semantics define `response.incomplete` as the
  terminal event for an incomplete response. Source previously emitted
  `response.completed` with an inner `status=incomplete`; commit `a36a5ea66`
  now makes the SSE event name and payload type match the final status, while
  the Electron parser handles completed/incomplete terminal events uniformly.
- Tests: 135/135 affected Python (`test_responses_history.py` plus
  `test_server.py`), 50/50 affected panel tests, and panel typecheck.
- Current-source server PID 2658 was restarted from the visible Electron Server
  dialog. Direct Responses emitted one correct streamed `README.md` call with
  split argument deltas. Its result continuation completed once with seven
  output-text deltas and exact `B1-RESP2-DONE`; a repeat instead consumed 1,024
  tokens, leaked repeated native tool markup under `tool_choice:none`, and now
  truthfully terminated as `response.incomplete`. That variability is retained
  as a Bonsai/model-template blocker rather than hidden by cleanup.
- Electron Chat Settings then set Max Tokens to 32. Row 2388 visibly preserved
  122 reasoning characters and the partial answer, reported exactly 32 output
  tokens, and did not fabricate `TERM-UI1-END`. Screenshot:
  `/tmp/bonsai-response-incomplete-ui.png`. Max Tokens was restored blank
  (model default) and the system prompt remained blank.

## Architecture-specific cache truth

| Architecture | Production cache contract | Current status |
|---|---|---|
| Plain full attention KV | Paged/prompt cache; uncalibrated Auto uses storage-only TQ4, while Bonsai-specific policies use TQ8; codec fields are part of the persisted namespace | Qwen full-KV and Laguna scoped pass; broader family regression matrix open |
| Qwen/Bonsai hybrid GDN/SSM | Eligible slots come from the real layer graph, not a family-name constant. Qwen 35B has 10 attention KV plus 30 companion layers; tested Bonsai bundles have 16 attention KV plus 48 companion layers. Only attention KV is TQ encoded; companion state remains native with clean boundary capture/rederive plus fingerprinted SSM L2 | Qwen 35B and two 1-bit plus one ternary Bonsai restart restores pass with native TQ8 + SSM disk; current Bonsai 1-bit PID 83540/84219/84984/85595 writes native-TQ L2 blocks and SSM companion disk records while preserving exact multi-turn tool behavior. Changed-prefix native-TQ candidates without an SSM checkpoint safely full-prefill; exact-prefix replay restores cleanly as `paged+ssm+disk`; forced four-block capacity evicts L1 while keeping L2 block+SSM stores intact. Broad long-context coverage remains partial |
| Other hybrid SSM/GLA | Architecture allow-list plus native companion state and async clean-prefill rederive | Nemotron-H current-source Auto/None, L1/L2, and forced-eviction rows pass with exactly six attention slots TQ-eligible and native Mamba companion state; per-family proof remains required and no name-only inference is allowed |
| Gemma 4 mixed rotating SWA | Rotating SWA state remains native; only compatible full-attention KV may be TQ encoded. Prefix lookup, resident paged blocks, L2 disk promotion, companion-state restore/rederive, and bounded eviction must agree on one valid boundary | Parser/tool-loop fix is current-source PASS: raw Responses trace proved the model generated a valid `<|tool_call>` by token 20 then hallucinated client-owned `<|tool_response>`/answer text; source now opts Gemma into completed-call stream stop and truncates at the regex-parseable native call boundary. Focused parser tests pass 13/13. Direct multi-turn Responses proof dropped from 97 output tokens / 82 heartbeats to 28 output tokens / 20 heartbeats and emitted one `file_info({"path":"README.md"})`. Live Electron same-chat rows 2265/2268 each executed one real `file_info` and exact finals; row 2268 reused 218 memory tokens and completed in 3.4s. Restored Auto/paged/L2 rows 2271/2274/2277 then proved `paged+mixed_swa+disk`, resident `paged+mixed_swa`, and post-restart `paged+mixed_swa+disk` exact tool continuations. UI-constrained four-block rows 2280/2283 stayed exact while L1 evictions reached 9 and both rows restored 192 tokens as `paged+mixed_swa+disk`; normal 1,000 blocks were restored on PID 82981. Commit `cc1562a2b` additionally stores Gemma's captured media-conditioned 48-layer boundary instead of re-prefilling text after tensor release. Current Electron image A/B/A rows isolated same-shape media keys, restored 304/305 tokens from RAM and then `paged+mixed_swa+disk`, and remained exact. Real MP4 rows restored 303/304 tokens from RAM and disk and remained exact; raw Responses emitted 89 timed reasoning plus six content deltas before completion. None A/B recheck, alternate-video isolation, audio, and long-output cache proof remain PARTIAL. Evidence: `gemma4-media-cache-current/`. |
| DSV4 Flash | Native `deepseek_v4_v8` SWA + CSA/HCA composite and pool codec; never generic TQ KV | Current-source deterministic cache/tool tier is PASS-LIVE. The v8 schema preserves `PoolQuantizedV4Cache` through prompt snapshot, paged reconstruction, and L2 restore, and realizes cache-hit tail prefill before allocator clearing. Electron rows 189/192 produced byte-identical reasoning/content, exactly one real `file_info`, and a 340-token `paged+dsv4` hit. Rows 195/198 repeated that result across a visible process restart with 338 `paged+dsv4+disk` tokens, two disk hits, and zero generic TQ writes/hits. Raw Responses cold/warm/skip-control outputs were equal; each tool pass streamed 78 reasoning deltas, two argument deltas, and a completed terminal, then the tool-result continuation streamed 15 content deltas and exact-finaled. Strict stochastic formatting and broader long quality/performance remain red. Evidence: `20260718_dsv4_v8_typed_cache_electron/`. |
| MiniMax-M3 | Native `minimax_m3_msa_v1`, dense KV 0–2 plus sparse MSA/index state 3–59; generic TQ off | Cache/restart scoped pass |
| openPangu 2.0 Flash | Native typed MLA + DSA/SWA + mHC + 128-sink composite; generic paged/block/TQ off | Current 3M Electron rows pass scoped tools, same-chat memory hit, process-restart prompt Disk L2 hit, and single-model swap; long-context/protocol soak remains partial |
| ZAYA/CCA | Typed CCA state; generic TQ off until typed parity exists | Historical live proof; current release regression row still required |
| VLM/video/audio | Architecture cache plus canonical media salt and real media payload | Qwen 3.6 27B image transport and Qwen video-frame fallback/cache are scoped PASS-LIVE: real pixels/MP4, media-keyed RAM hit, cross-media isolation, bypass, block+SSM disk restore, progressive Chat streaming, and visible Electron persistence. Gemma 4 image/video is separately scoped PASS-LIVE at `cc1562a2b`: same-shape image A/B isolation, image and MP4 resident hits, process-restart `paged+mixed_swa+disk` restores, native-TQ block hits only on compatible full-attention lanes, visible Electron output, and progressive raw Responses reasoning/content. Step 3.7 is scoped PASS-LIVE at `c305b18b5`: its real NumPy/MLX prefill crash is repaired, image A/B/A isolates media keys, image and MP4 prefixes restore from RAM/L2, and raw Responses streams both rails. MiniMax-M3 is scoped PASS-LIVE at `8df1bfe86`: image A/B/A and video A/B/A isolation, image and MP4 resident/restart-L2 hits, native MSA tuple preservation, exact video-A Electron output, and progressive raw Responses content. Other advertised-family media matrices, Gemma audio, larger-video rows, and broader catalogs remain open. Evidence: `qwen27-media-cache-current/`, `gemma4-media-cache-current/`, `step37-media-cache-current/`, and `mm3-media-cache-current/`. |

## Mandatory current-source architecture rows

These rows are release requirements, not load-only smoke tests. Each model
must be loaded and operated through the current Electron dev build. A PASS
requires a source trace plus persisted live artifacts for: cold generation,
same-chat multi-turn continuation, a real tool result followed by a complete
visible answer, process restart/L2 restore, cache accounting, and eviction or
bounded-capacity behavior. API-only evidence is secondary and cannot replace
the Electron row.

| Model / family | Cache and runtime invariant | Required live proof | Status |
|---|---|---|---|
| Qwen 3.6 35B MXFP/JANG (name has no `MTP`) | Hybrid layout is derived from the real 10-attention/30-companion layer graph. TQ encode/decode applies only to eligible attention KV; GDN/SSM companions remain native and are cleanly rederived/restored. This artifact is not assigned an MTP gate. | Cold + two-turn + tool continuation, RAM hit, restart/L2 hit, and forced eviction/reload with coherent output. | PASS-LIVE current stream/cache tiers / PARTIAL sampled reliability: raw Chat produced exact two-turn reasoning/content streams and a 46-token `paged+ssm` hit. Current Electron produced two same-chat, distinct one-tool turns with progressive content and exact finals. One exact 4,625-token prompt improved from 8.8514s cold to 0.5543s on a 4,624-token RAM restore (15.969x); visible process restart restored the same prefix as `paged+ssm+disk` in 0.4785s. Health proves q4 native-TQ block hits plus a real SSM disk checkpoint and zero unsafe KV-only reuse. Earlier four-block eviction rows remain valid. One Responses sample repeated its marker in the model deltas; three saved repeats did not. Source trace found no API-side replay, so no synthetic dedup was added and strict sampled reliability remains red. Evidence: `qwen35-current-stream-cache/`. |
| Qwen 3.6 27B `...-MTP` | The same hybrid cache invariant applies, and MTP is eligible because the actual model/bundle name says `MTP`. Compatible non-Bonsai attention KV uses TQ4; native SSM/GDN companion state remains full precision and independently restored/rederived. Media prompts add a canonical pixel/video-derived side-key; the N-1 cache must be produced while the media tensors remain live. | MTP depth 1 and 3 launch/health, real draft/accepted counters, cold + two-turn + tool continuation, RAM hit, restart/L2 hit, forced eviction/reload, and media A/B isolation with coherent streaming output. | PASS-LIVE agent loop+Auto-TQ4 cache+terminal-stream+image/video-media tiers / PARTIAL long-reasoning and broader-variant reliability: prior current-source rows prove exact distinct tools, q4 plus native-SSM RAM/restart/forced-eviction behavior, truthful UI settings, D1 tool policy, genuine D3 draft/accept counters, and media image/video isolation. Commit `3d32b944b` is now independently live-proven on this artifact: raw curl executed exactly one `file_info`, one `run_command(pwd)`, then streamed an exact final over eight timed deltas; Electron row 324 visibly repeated that exact two-tool/final sequence with both real results and no warning. Identical replay restored 388/206 tokens as `paged+ssm`; visible PID 63193 to 63864 replacement restored them as `paged+ssm+disk`, with 11 native-TQ q4 hits, two SSM-disk hits, and `disk_hit/reconstructed/dequantized=true`. MTP remained active at configured depth 3, while this short final exercised only D1 acceptance and is not counted as a fresh D3 speed proof. Earlier image/video media proof remains in `qwen27-media-cache-current/`; text/terminal/MTP proof is in `qwen27-mtp-stream-cache-current/`. Long native reasoning latency, larger-context cancellation, and broader variant reliability remain open. |
| Step 3.7 Flash JANGTQ_K | This artifact is mixed attention: 12 full-attention slots may use stored-prefix TQ4, 33 rotating-SWA slots remain native, and both lanes share one media-keyed boundary. Its name and tensor index do not declare MTP, so the nested architecture hint must not activate MTP. | Live source model load, image/video prefill, image A/B isolation, resident and restart/L2 media hits, progressive Responses, exact Electron output, and truthful native-reasoning-only UI/API contract. | PASS-LIVE scoped image+video/cache/stream/mode parity / PARTIAL stochastic reliability+cold-store latency+strict raw formatting+alternate video: commit `c305b18b5` normalizes the live NumPy media tensors before MLX vision operations and admits config-derived Step to the captured 45-layer N-1 boundary. Image A restored 2,202/2,203 tokens from RAM and `paged+mixed_swa+disk`; same-shape B was a zero-hit exact miss and return-A restored A. The MP4 route decoded and sampled a real frame, then restored 372/373 tokens from RAM and disk. Commit `8b0e23dc1` hides unsupported Off, advertises low/medium/high, rejects API Off with HTTP 400, and preserves the official open think rail. Deterministic cold/RAM/restart-disk Responses runs produced the same exact answer with 11 progressive content deltas, but one temperature-0.6 disk-hit run looped for 1,024 reasoning deltas and ended incomplete. The raw leading newline and doubled cold-store TTFT also remain partial. Evidence: `step37-media-cache-current/` and `step37-reasoning-mode-current/`. |
| HY3 MTP | The exact `Hy3-JANG_2K-MTP` bundle declares one MTP layer and 42 MTP tensors, so the runtime must use depth 1 rather than inventing a depth-3 gate. HY3's plain attention KV may use the family-scoped TQ4 stored-prefix codec; live decode stays native, and MTP batch split/verify must own independent cache copies. | Depth-1 draft/accepted counters, same-chat multi-turn tool loop, process-restart L2 restore, forced eviction/reload, explicit None A/B, and coherent long/streaming output. | PASS-LIVE cache+settings+restart+eviction / PARTIAL reliability+long: commit `ab5d01e04` selects HY3 full-KV TQ4 and installs an independent `TurboQuantKVCache.__deepcopy__`, fixing the live `cannot pickle 'mlx.core.Dtype' object` scheduler retry loop; 19 focused TQ tests and 178 native-MTP tests passed. Commit `5e6a1f8a1` reports `Native HY3 KV + TQ4 stored prefixes / TQ4 AUTO` in settings; 282 settings tests and typecheck passed, and the label is visible in Electron. PID 22265 row 2483 survived UI process restart, restored 3,272 tokens as `paged+disk+tq-native`, executed one `file_info`, and exact-finaled. UI-applied four-block PID 23635 produced rows 2488/2489, 11 L1 evictions, five TQ-native L2 writes, and 18 TQ-native hits; older-prefix row 2492 restored the bounded 192 tokens as `paged+disk+tq-native`, executed exactly one tool, and returned exact `HY3-Q4-T1R-DONE`. Explicit UI None launched PIDs 26444/27461 with `--kv-cache-quantization none`; cold row 2495 wrote 54 raw blocks with zero TQ activity, and restart row 2498 restored 3,258 tokens as `paged+disk`, made one real tool call, and exact-finaled while TQ writes/hits remained zero. UI restored Auto, 1,000 blocks, and TQ-native enablement on PID 28473. Same-chat rows 2474/2480 exact-finaled, but row 2477 emitted `HY3-Q4-T2-D-DE-DONE`; strict-format reliability therefore remains PARTIAL. |
| MiniMax M2.7 | Ordinary KV attention may use calibrated or correctness-safe TQ storage; parser/reasoning rails must survive multi-turn tool continuation. | Auto and None UI/argv/health A/B, two-turn tool loop, RAM/L2 restore, eviction, long visible answer, and streaming rail continuity. | PASS-LIVE current source: rows 2187/2190 prove cold plus same-chat two-tool continuation and a 173-token resident `paged+tq-native` hit. PID 63682 row 2193 restored 173/177 as `paged+disk+tq-native`. None mode PID 64194 launched with explicit `--kv-cache-quantization none`, wrote raw `dtype=kv` blocks, and PID 64579 row 2199 restored 161/165 as `paged+disk` with zero TQ activity. Commit `af7815f1a` repairs fetched-block ref ownership; under the UI-applied four-block ceiling, PID 65838 rows 2208/2211 completed exact tool loops, returned all three usable blocks to the free queue, and raised L1 evictions from 3 to 9. Normal 1,000-block Auto was restored on PID 66306 and row 2214 repeated the exact 173-token disk hit. Electron row 2217 produced a coherent 582-token reasoning/content answer with the exact terminal marker. A direct Responses stream with a 1,024-token budget emitted 711 reasoning deltas, 48 content deltas, matching text-done, and `response.completed(status=completed)` with its exact marker. The controlled 512-token cap correctly reported `status=incomplete` instead of pretending completion. |
| ZAYA / CCA | Typed CCA state owns its cache. Generic TQ is forbidden unless a typed CCA codec has source and live parity. | Typed cold/warm/restart/eviction rows plus multi-turn tool and reasoning/content stream. | BLOCKED current generic row: the external drive contains only the `AppleScript-8B-JANG_4M` single-tool specialist, which the user excluded from this campaign. This is a missing-artifact gate, not a runtime failure. |
| Nemotron hybrid | Eligible attention KV may be TQ encoded; non-KV hybrid state remains native and is async clean-prefill rederived/restored. Family selection must come from config/layers, not a name match. | Auto/None A/B, cold + two-turn + tool continuation, L2 restart, eviction, long output, no reasoning leak. | PASS-LIVE cache/settings/tools/API / PARTIAL repeated long reasoning: rows 2223/2226 were exact cold and same-chat one-tool turns, with 162 tokens restored as `paged+ssm+tq-native`. PID 74652 row 2229 restored 192 tokens as `paged+ssm+disk+tq-native`. UI-applied four-block PID 75038 rows 2235/2238 stayed exact while evictions rose 3 to 9 and three usable blocks returned free. Explicit None PIDs 75398/75644 rows 2241/2244 wrote and restored raw `paged+ssm+disk` blocks with zero TQ activity. Auto/1,000 blocks is restored on PID 75939. Electron row 2247 completed a coherent marked answer but repeated 2,962 tokens of native reasoning before the real `</think>`; retained as reliability PARTIAL. Direct Responses emitted 424 reasoning deltas, 30 content deltas, matching done events, and `response.completed`. Focused source tests pass 25/25. |
| Gemma 4 rotating SWA | TQ applies only to compatible full-attention KV. Rotating SWA cache remains native, and a prefix hit is valid only when both lanes share a restorable boundary; otherwise safely rederive/full-prefill. Media prefixes additionally require a canonical image/video side-key and must capture the boundary while vision tensors remain live. | Auto/None UI/argv/health A/B, cold + two-turn + tool continuation, resident paged hit, L2 restart promotion, forced eviction/reload, true-miss fallback, image/video A/B isolation, and coherent long output. | PASS-LIVE cache/settings/tools/eviction/restart+scoped image/video / PARTIAL long-output+audio+alternate-video: commit `3385cb019` makes UI/CLI Auto select q4 only for full-attention slots, keeps rotating-SWA slots native, and fails closed on layout mismatch; 153 focused tests passed. Auto rows 2425/2428/2431 and final row 2470 each made one real tool call and exact final; row 2431 survived process replacement with 704 `paged+mixed_swa+disk` tokens and 44 native-TQ disk hits. A visible 16-block pressure run produced 38 L1 evictions; post-eviction row 2464 restored 704 tokens from TQ-native L2 and exact-finaled. Explicit None PID 15388 launched with `--kv-cache-quantization none`; row 2467 exact-finaled while ordinary disk writes rose to three and TQ writes/hits stayed zero. UI Auto/1,000 blocks is restored on PID 15797; row 2470 restored 704 `paged+mixed_swa+disk` tokens with three TQ-native writes and eleven hits. Commit `ba68f8fba` fixes the settings drawer to show `TQ4 full-attention KV + native rotating SWA / MIXED AUTO` instead of an SSM/GLA label; 281 settings tests and typecheck passed, and the label is visible in Electron. Commit `cc1562a2b` enables Gemma's captured media-conditioned N-1 boundary by default while preserving explicit off. Image A/B/A rows proved zero-hit cross-image isolation, 304-token resident reuse, and process-restart `paged+mixed_swa+disk`; real MP4 rows proved 303-token resident and disk reuse. All scoped finals were exact, and raw video Responses emitted 89 reasoning plus six content deltas progressively. The two full media/scheduler files pass 167 with six intentional skips. Coherent constrained long-output, audio, and a distinct-content video B remain open. Evidence: `docs/internal/release-gates/20260716_gemma4_mixed_swa_tq4/` and `gemma4-media-cache-current/`. |
| DSV4 Flash | Native DSA/SWA/CSA/HCA composite and pool codec only; never generic TQ KV. | Composite cache health, eager load, cold/warm/restart/eviction, multi-turn agent loop, reasoning/content stream continuity and coherent constrained output. | PASS-LIVE eager/cache/settings/eviction/stream/tool-scope tiers / PARTIAL strict stochastic long quality+perf. The current v8 repair preserves the native pool-quantized class across snapshots, paged blocks, and disk reconstruction and removes an unsafe lazy cache-hit tail before `mx.clear_cache`. Electron rows 189/192 exact-finaled one real tool on cold and 340-token resident restore; rows 195/198 exact-finaled across a real UI Stop/Start with 338 `paged+dsv4+disk` tokens and byte-identical reasoning/content. Raw Responses cold/warm/skip-control normalized to the same function call; reasoning, arguments, post-tool content, and terminal events all streamed separately. Health identifies 43 native composite layers, pool quant enabled, generic TQ forced off, and v8 schema. Focused current-source validation is 813 passed / 1 skipped. Earlier eager/eviction/tool-scope evidence remains retained; stochastic exact-format and long quality/performance failures remain open. Evidence: `dsv4-eager-current/`, `dsv4-cache-current/`, `dsv4-tool-scope-current/`, `20260718_dsv4_v8_typed_cache_electron/`. |
| MiniMax M3 / openPangu | Native typed architecture cache only; generic TQ remains off. | openPangu 3M current rows pass scoped tools/restart prompt L2, and current-head Auto now passes real Electron load/swap, progressive reasoning/content, required tool, post-tool continuation, and raw Chat/Responses terminals in `20260718_openpangu_auto_agent_stream/`. MiniMax-M3 image and MP4 prefixes pass content-keyed RAM/restart-L2 proof at `8df1bfe86`, and its current text Auto protocol gate is in `20260718_m3_auto_agent_stream/`. Larger-video transport and the M3 terminal delay are now scoped live-proven in `20260719_m3_terminal_dispatch_large_video/`; REAP32 headroom, exact digit OCR/order, openPangu long context, and cancellation/disconnect soak remain partial. | PARTIAL |

Current Step session-lifecycle supplement: the missing post-restart PID was a
shared Electron event/state defect, not a model-runtime failure. Local
`session:ready` now transports the real PID and the shared chat context clears
it on Stop. A full Electron-main relaunch plus visible Start/Stop/Start showed
38968 -> no PID -> 39507, with DB/`ps` parity and exactly one engine. The
focused panel selection passes 174/174 plus typecheck. Evidence:
`20260719_current_step37_jangtq/`.

Current Qwen 27 settings-parity evidence: the Electron number field published
`3` before blur, SQLite persisted `nativeMtpDepth=3` with override enabled, PID
52719 launched with `--native-mtp-depth 3`, `/health::mtp.effective_depth` was
`3`, and `qwen36-27-mtp-d3-settings-parity.png` visibly records the current
model, PID, Server Settings drawer, and depth. This proves the settings
round-trip only; it does not clear the reasoning-loop or cache-behavior rows.

Current Qwen 35 source-plus-live evidence: PID 55959 was the only active local
engine after the Electron single-model swap. Rows 2139 and 2142 each made one
schema-valid real `file_info` call and returned exact visible finals in the
same chat. Fresh repeat row 2145 restored 152/153 prompt tokens as
`paged+ssm`; Electron Save & Restart produced PID 56619 and row 2148 restored
the same boundary as `paged+ssm+disk`. After the non-MTP telemetry repair,
another Electron restart produced PID 57270 and row 2157 again restored
152/153 tokens from disk, executed one real tool, and returned exact final
text. Current health records seven native-TQ attention-block hits and two
native SSM companion-disk hits. The UI visibly showed Prefix Cache on, required
Paged KV on, 64-token blocks, 1,000 blocks, 15% L1 memory, Block Disk L2 on,
and Stored Cache Quantization Auto. Commit `b0b21ed12` now reports the nested
Qwen architecture field as an inactive hint (`mtp_declared=false`,
`status=not_configured`, no issues) because this bundle name, JANG sidecar,
and tensor index do not declare MTP; 96 focused MTP tests pass. Commit
`7bb34fa0d` then fixed the owning paged-cache accounting bug: disk promotion
released the arrays but left phantom resident bytes, and a reused block could
inherit `keep_resident`. Electron Save & Restart produced PID 58213; row 2160
again restored 152/153 `paged+ssm+disk` tokens, made one real tool call, and
returned exact final text. Health and the visible Cache Management drawer now
show 152 indexed tokens but 0 resident bytes, seven native-TQ block hits, and
two SSM-disk hits. The repair passes 595/595 audit/byte-budget tests and 177/177
paged/disk/TQ/hybrid cache tests. Screenshots are stored as
`qwen36-35b-*.png` and `qwen35-*-postfix.png` in this evidence directory.
Commits `df945f065`, `133d8c8e9`, and `7cb89185c` then moved generic hybrid
cumulative state to the typed companion store, fixed the NumPy disk-writer
branch found by live safetensor inspection, and invalidated malformed v7 files.
The corrected v8 directory contains eight files whose tags are exactly 10
`turboquant_kv` plus 30 `skip`; terminal partial files fell from roughly 64 MB
in malformed v7 to 30 KB/295 KB in v8. Rows 2169/2172/2175 prove cold,
same-process, and process-restart tiers. The Electron UI then set Max Cache
Blocks to four and restarted PID 61405: rows 2178/2181 each restored 154/155
`paged+ssm+disk` tokens, executed one real tool, and returned exact final text.
The visible Cache drawer recorded nine L1 evictions and a safe full-prefill when
192 KV-only tokens had no matching SSM companion. The UI restored 1,000 blocks,
PID 61919 launched with that argv, and row 2184 repeated the exact disk hit.
Python hybrid/cache/scheduler tests pass 784/784; panel settings tests pass
278/278 plus typecheck. Only the stricter long-format/reliability row remains
partial for this Qwen artifact.

## Gateway and single-model lifecycle — current source

- The pre-fix Electron UI reproduction changed a running gateway from 8081 to
  DSV4's occupied 8012 and left the gateway stopped. Source trace found
  `ApiGateway.restart()` stopped the old listener before validating the new
  bind. Commit `e76cc5451` remembers the prior host/port, attempts the new
  listener, restores the prior listener on failure, and rethrows the original
  bind error. A real-listener regression test covers this rollback; the three
  focused gateway suites pass 75/75 plus panel typecheck.
- In the clean current-source Electron instance, the same API-page port edit
  was rejected while `/health` and SQLite stayed at running
  `127.0.0.1:8081`. Screenshot:
  `/tmp/gateway-port-conflict-restored.png` (the retained pre-fix screenshot is
  `/tmp/gateway-port-conflict-prefx.png`).
- The LAN toggle then rebound the listener to `0.0.0.0:8081`, displayed
  `192.168.1.110:8081`, and a request to
  `http://192.168.1.110:8081/health` returned gateway health with
  `single_model_mode=true`. Disabling LAN rebound to localhost without losing
  the listener. Screenshot: `/tmp/gateway-lan-enabled.png`.
- With Single model mode still enabled, the visible Server-page Start button
  for `jangq-ai/Bonsai-27b-1bit-JANG` exercised
  `SessionManager.startSession()`. The lifecycle log records DSV4 session
  `a6810958-...` being stopped before Bonsai `5fd14571-...` started. The after
  screenshot shows Bonsai PID 10495 as the sole Active session and DSV4 as
  Stopped; SQLite, `ps`, and gateway discovery independently showed exactly one
  local engine. Screenshots: `/tmp/single-model-before-dsv4.png` and
  `/tmp/single-model-after-bonsai.png`.
- This row proves listener rollback/rebinding and one-engine session ownership.
  It does not by itself prove cross-protocol streaming or Bonsai output/cache
  correctness; those remain separate rows.
- Basic gateway streaming controls then used that same UI-proven listener and
  Bonsai backend. OpenAI Chat Completions streamed 233 events, 857 reasoning
  characters, exact `OAI-GW1-DONE`, `finish_reason=stop`, and usage. Responses
  streamed 151 reasoning and seven content deltas, matching output-text done,
  and one completed terminal; its two leading newlines remain a strict-format
  miss. Anthropic streamed 793 thinking characters, exact `ANT-GW1-DONE`, and
  one `message_stop` with no error event.
- The first Ollama `think:true` run exposed two gateway translation defects:
  its final object repeated all already-streamed thinking, and it ended at the
  finish-reason chunk before the later usage event. Commit `a0aa81a94` emits
  the terminal only at `[DONE]`/backend end, carries the later usage, and leaves
  the terminal message empty as Ollama's streaming contract requires. Focused
  gateway tests pass 76/76 plus typecheck. After a clean Electron main-process
  restart, Bonsai PID 12114 streamed 193 thinking deltas / 728 characters once,
  exact `OLL-GW4-DONE`, and one empty-message terminal with
  `eval_count=202`, `prompt_eval_count=18`, and `done_reason=stop`.
- An earlier Ollama thinking run ended after native reasoning with no visible
  content, while `think:false`, a direct-backend explicit-thinking control,
  and the final current-source Ollama run all produced exact markers. That miss
  is retained under Bonsai reliability; no synthetic continuation was added.

## Non-negotiable correctness invariants

- No prompt coercion, hidden sampler clamps, forced thinking tags, synthetic
  tool output, invented continuation, or arbitrary output cap may be used to
  make a gate appear green. Fix the layer that owns the defect.
- Assign MTP gates only to actual model/bundle names containing `MTP`. Do not
  infer MTP eligibility from a Qwen, HY, Nemotron, or other family name alone.
- Treat official JANGQ/dealignai quantized models as trusted artifacts. If a
  live row loops, truncates, or emits incoherently, investigate vMLX
  architecture dispatch, quantized layer utilization, cache state,
  sampling/template behavior, parsers, streaming, and UI/API parity; do not
  attribute the defect to the official quantized model.
- Cache keys and persisted records must cover model/runtime fingerprint,
  architecture codec, quantization parameters, original KV dtype, media salt,
  MTP mode/depth where relevant, and every state needed for exact restore.
- Stored-cache TurboQuant bit width is a release-gated policy, not a naming
  assumption. Bonsai is the current TQ8 exception. Compatible non-Bonsai hybrid
  attention KV uses TQ4 only while native SSM/GDN companion state is separately
  restored or safely rederived; Qwen 27/35 current-source RAM and restart rows
  provide that source-plus-live proof. For other compatible non-composite KV
  families, Q4 is the target Auto storage width only after source classification excludes typed
  composite caches such as DSV4, MiniMax-M3 MSA, openPangu MLA/DSA/SWA/mHC, and
  ZAYA/CCA, and live Electron restart evidence proves correct encode/decode.
- Prefix reuse is explicitly three-tiered: use a valid resident L1/paged block
  first, otherwise promote a matching L2 disk record, and only full-prefill
  when neither tier is usable. Hybrid hits are valid only when the attention
  KV boundary and its companion state are both restorable; a partial component
  must trigger safe rederive or full-prefill rather than a false hit.
- Multi-turn means at least two user turns in the same chat. Agentic proof
  additionally requires a schema-valid tool call, a real tool result, and a
  complete post-tool answer. A one-turn exact marker is insufficient.
- Streaming proof must persist and compare reasoning deltas, visible content
  deltas, tool-call argument deltas, tool result continuation, finish reason,
  and final assembled text. The stream must not silently end in an incomplete
  tool call, unfinished reasoning rail, or missing visible answer.
- Every cache row records configured capacity, resident blocks/bytes,
  hits/misses/writes, TQ encode/decode counters where allowed, companion-state
  rederive/restore counters for hybrids, disk reconstruction time, and an
  eviction followed by a correct reload or safe full-prefill fallback.

## Closed rows that must remain regression-gated

- Bonsai and Qwen cache architecture selection uses nested model type plus
  actual layer layout; it does not classify from a name containing `qwen`.
- Bonsai UI Auto stores TQ8 only for attention KV. UI None launches with
  `--kv-cache-quantization none` and cannot decode stale native-TQ records.
- Exact-once Qwen/Bonsai requests stop after one schema-valid required tool
  without disabling general multi-tool/interleaved behavior.
- HY3 MTP depth 1 is the bundle-declared runtime: health reports one configured
  layer, one indexed layer, 42 MTP tensors, `runtime_active=true`, and
  `effective_depth=1`. The earlier controlled warm median improved 21.234247s
  to 16.081931s with 180/414 draft tokens accepted. Current TQ4 restart and
  eviction rows additionally prove batch-copy ownership and L2 restore; the
  retained T2 strict-marker miss keeps broad reliability PARTIAL.
- Gateway LAN display selects a routable address rather than APIPA.
- Single-model mode visibly stops the old model and leaves one local server.
- Laguna/JANG and vMLX preserve original float16/bfloat16 KV dtype through
  TQ encode, disk persistence, decode, and native cache rewrap.
- Uncalibrated Auto TQ no longer silently assigns 3-bit storage to ordinary
  full-KV families. The correctness-first default is TQ8, while calibrated
  bundle policy remains authoritative. Every codec field participates in the
  persisted cache namespace so old TQ3 blocks cannot replay after upgrade.

## Execution order

1. MiniMax M2.7 is closed current-source across Auto/None, multi-turn tools,
   RAM/L2, eviction, long visible output, and direct streaming. Generic ZAYA
   is blocked because only the excluded AppleScript specialist is present.
   Nemotron cache/settings/tools/API rows are closed, with repeated long
   reasoning retained as PARTIAL. Gemma 4 cache/settings/tool/eviction rows are
   now closed; run its coherent constrained long-output row with the remaining
   reliability matrix rather than reopening its cache tier.
2. Keep HY3 cache/settings/restart/eviction functionally regression-gated at
   its bundle-declared depth 1, but leave Q4 disk/paged TTFT `PARTIAL` until the
   matched cold comparison improves; close its remaining long reliability row.
   Close DSV4 long
   quality/performance and the remaining M3/Pangu long/media boundaries. Do
   not test Mistral MXFP4 in this campaign per the user's explicit instruction.
3. Re-prove Bonsai forced eviction/repair boundaries, retaining its recorded
   sampling miss as reliability evidence; keep Qwen 35B's long-format miss in
   the reliability ledger without reopening its now-closed cache tier row.
4. Close the remaining Laguna unsolicited-tool/long-context/latency rows and
   run the complete settings and protocol matrix through Electron/gateway.
5. Run focused and full tests, audit the dirty tree, commit/push only scoped
   files, and merge/integrate the closeout branch deliberately.
6. Treat public v1.6.11 as the completed signed/notarized checkpoint. Build,
   sign, notarize, staple, verify, install-smoke, and publish a newer version
   only after its explicitly selected release blockers are closed or accepted;
   do not rerun or republish v1.6.11 as if it were still pending.

## 2026-07-18 evidence-preservation addendum (audit-driven; citations repointed off /tmp)

A commit-level audit (43/43 cited commits verified present) found eight rows
citing evidence that existed only untracked in-tree or in /tmp. All of it is
now force-added and committed. Mapping for rows whose citations named /tmp:

- `/tmp/gateway-port-conflict-prefx.png`, `/tmp/gateway-port-conflict-restored.png`,
  `/tmp/gateway-lan-enabled.png`, `/tmp/single-model-before-dsv4.png`,
  `/tmp/single-model-after-bonsai.png` -> `settings-current-head/` (tracked).
- `/tmp/bonsai-qwen-3turn-current.png`, `/tmp/bonsai-response-incomplete-ui.png`
  -> `bonsai-current-head/` (tracked).
- `bonsai-b1-ui-tool3-current-pass.png` -> `bonsai-1bit-current/` (tracked).
- Qwen 3.6 35B row screenshots `qwen36-35b-*.png` -> tracked at top level.
- Gemma 4 mandatory row gate dir `../20260716_gemma4_mixed_swa_tq4/` -> fully
  tracked (was 100% untracked).
- openPangu row: `openpangu3m-current/` (tracked) is the artifact set for the
  prose PIDs 86212/86842/87268 rows; row stays scoped-PASS with sub-axes
  PARTIAL as written.
- Full-suite gate: `full-suite-20260718/` holds the 2026-07-18 full panel
  vitest log (2302 passed / 3 skipped / 0 failed — FULL suite) and round-1
  eviction probe JSON; full pytest + typecheck logs land there when complete.
- `.agents/LOG.md` is now tracked (was gitignored working-tree only).

Also recorded: the ledger 07-18 12:0x DSV4 effort-none artifact-blame entry is
RETRACTED by the 15:0x entry (jang_config.chat.sampling_defaults declares
temp 0.6 / top_p 0.95 / rep-pen 1.05 and the engine resolves them; cause
reopened, same-artifact A/B pending). The matrix DSV4 row remains PARTIAL for
long/strict quality; no artifact regeneration is recommended or authorized.
Scoped-head note: code commits after `45c64f85e` through `937fd7639` and this
addendum supersede the pinned head; the newest source-plus-live row wins.

## 2026-07-18 post-release Step Auto reasoning/tool recovery

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Step 3.7 JANGTQ Auto no-tool recovery | PASS-LIVE scoped | Step-only native tools-free retry is guarded by explicit tool intent and preserves the qwen3 rail. Electron rows 42/45 returned non-empty exact finals; row 45 visibly grew character-by-character. Raw deterministic Responses A/B emitted 144/9 deltas without tools and 216/9 with two Auto tools; Chat emitted 255/10 and clean stop/DONE. | Successful recovery still exposes a diagnostic warning on row 42; first-pass tight-memory cache store skipped; retain stochastic/default-sampling miss. |
| Step explicit tool + post-tool stream | PASS-LIVE scoped | Electron row 48 executed one real `file_info(panel/package.json)`, rendered two separate progressive reasoning rails and progressive post-tool content, exact-finaled, and reused 512 `paged+mixed_swa` tokens. | Broader protocol cancellation/disconnect and media/eviction rows remain separate. |
| Auto reasoning stream by registered parser/family | OPEN/PARTIAL | Registry inventory identifies eight reasoning parsers: `deepseek_r1`, `gemma4`, `minimax_m2`, `minimax_m3`, `mistral`, `openai_gptoss`, `qwen3`, `think_xml`. Existing historical rows are not promoted automatically to current-head proof. | For each locally configured family: current Electron and raw Chat/Responses no-tool Auto, required tool, post-tool continuation, progressive reasoning/content, clean terminal, distinct multi-turn reasoning, and no parser-marker leakage. |

Evidence: `../20260718_step_auto_reasoning_tool_recovery/`.

## 2026-07-18 MiniMax-M2.7 current-source Chat id and q4/L2 checkpoint

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Chat tool-call delta assembly | PASS-LIVE scoped | `stream_chat_completion` now introduces index 0's id only in the early START delta; the function-data delta omits repeated id/type/role fragments. The raw M2.7 Chat stream reconstructed one clean id, one valid `file_info(path)`, `tool_calls`, and one DONE. Chat result synthesis emitted 46 reasoning and 14 progressive content deltas; Responses emitted one call then 55 reasoning and 15 progressive content deltas with one completed terminal. Broad affected tests: 430 passed / 3 deselected. | Multi-call raw accumulation remains covered by source contracts but should remain in the campaign-wide protocol soak; non-stream behavior is unchanged by this patch. |
| MiniMax-M2.7 Electron Auto/tool current PID | PASS-LIVE scoped | Visible Save & Restart produced PID 72865. Fresh Electron row 159 kept reasoning separate, executed exactly one real `file_info(panel/package.json)`, returned exact non-empty `MM27-UI-CURRENT-DONE SIZE=5.2 KB`, stored no warning, and displayed 128 `paged+tq-native` cached tokens. | This adds current-head Auto/required-tool/post-tool proof for `minimax_m2`; it does not promote the other seven registered reasoning-parser families. |
| MiniMax-M2.7 q4 resident and restart-L2 | PASS-LIVE scoped | Bundle truth is JANGTQ/MXTQ (`weight_format=mxtq`, `profile=JANGTQ2`), distinct from affine JANG and base MLX MXFP. After the visible restart, four identical raw requests restored 838 tokens as `paged+disk+tq-native`; health recorded 22 native-TQ L2 hits, three native-TQ writes, and no dequantization in the last 192-token reconstruction. | Tool-schema/history rendering prevented a maximal second-turn prefix hit. Matching later/resumed boundaries are proven; universal maximal history reuse is not claimed. |
| MiniMax-M2.7 partial block + bounded eviction/refault | PASS-LIVE current source | Current Electron created a fresh L2 directory under a four-block ceiling. A 178-token cold base wrote exact q4 native-TQ blocks of 64+64+50 tokens; same-chat pressure evicted L1, a new chat refaulted the old partial-terminal chain, and UI process replacement restored it from zero L1 tokens. Every scoped visible final was exact and warning-free. | Evidence: `20260719_m27_paged_l2_partial_refault/`. This is the full-KV M2.7 child row, not a substitute for hybrid/mixed/native typed family rows. |
| Frugal worker L2 source reporting | PASS-LIVE current source | Commit `97a84fed5` records successful disk payload reads performed during worker reconstruction, closing a live under-report where later frugal hits said `paged+tq-native` while L2 hit counters rose. Patched Electron, raw Responses, and raw Chat all reported `paged+disk+tq-native`; raw rails stayed progressive and Chat terminal usage order stayed finish -> usage -> `[DONE]`. | Keep this assertion in every retained family run; configured L2 alone must never be labeled a disk hit. |

Evidence: `../20260718_minimax_m27_tq4_agent_stream/`.

## 2026-07-19 MiniMax-M2.7 Anthropic protocol repair

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Anthropic ordinary stream/non-stream | PASS-LIVE scoped | Current Electron-started M2.7 emitted 205 separate thinking deltas plus 12 exact visible-content deltas and one stop. Thinking-disabled non-stream returned exact content, `end_turn`, and nonzero usage. | Other model/parser families do not inherit this row. |
| Anthropic split tool name | PASS-LIVE scoped | `c707bb61a` buffers split Chat tool deltas until id and name exist. Live before had an empty tool name; live after emitted one named `file_info`, exact path JSON, no error, and one stop. | Multi-tool interleaving remains a source-test row until a live multi-call model run is retained. |
| MiniMax orphaned native-tool opener | PASS-LIVE scoped | `d7f74b982` narrowly recovers a complete invoke followed by an orphan outer close. Live Anthropic and real Electron each executed exactly one valid `file_info`; Electron exact-finaled with no warning and 128 `paged+disk+tq-native` tokens. | Keep negative coverage against promoting arbitrary visible XML. |
| Anthropic post-tool `tool_choice:none` | PASS-LIVE scoped | `4a53f16e1` renders only the effective prompt tool set. Current live follow-up emitted exact content over 17 progressive deltas, no reasoning/tool leakage, one `end_turn`, and one `message_stop`. | Overall protocol row remains PARTIAL pending Ollama and failure/disconnect/recovery soak. |

Evidence: `../20260719_anthropic_tool_parity/`.

## 2026-07-19 MiniMax-M2.7 Ollama and shared reasoning stream repair

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Shared think-close stream separator | PASS-LIVE scoped | `c1db6b745` repairs the shared Qwen3/DeepSeek-R1/MiniMax-M2 think-tag boundary. Direct Chat and Ollama live before each exposed `\n\n`; live after streamed exact content with separate reasoning. Real Electron DOM grew across six sampled reasoning stages before exact visible output. | Other registered reasoning formats need their own current-family live rows; no universal model claim. |
| Ollama `/api/chat` stream/non-stream | PASS-LIVE scoped | M2.7 current source emitted 200 thinking rows plus 12 exact content rows and one usage terminal; non-stream thinking-off was exact with nonzero usage. | Disconnect/error recovery and signed-app repeat remain open. |
| Ollama tool/result continuation | PASS-LIVE scoped | Required-tool stream placed one object-argument `file_info` only on the single `done:true/tool_calls` row. Result continuation emitted fresh separate thinking, 17 exact content rows, no second tool, one stop, and usage. | Live multi-tool interleaving and other parser families remain open. |
| Templated Ollama `/api/generate` | PASS-LIVE scoped | `01d95b448` defers split Chat finish/usage events. Live stream now has 118 thinking rows, 13 exact response rows, one terminal with eval/prompt counts; non-stream is exact. | Raw-mode semantics and failure recovery remain open. |

Evidence: `../20260719_ollama_stream_tool_parity/`.

### Bonsai partial-prefix and rejected reasoning-tool suffix - current source

- Bundle/config truth: Qwen3.5 `JANG_AFFINE_1BIT`, not JANGTQ/MXTQ; 16
  full-attention KV plus 48 native companion lanes.
- Electron Auto/q8 stored attention KV only. A 6,336-token sibling prefix hit
  twice as `paged+ssm` and after process replacement as `paged+ssm+disk`;
  health recorded native-TQ q8 block and SSM-disk restoration.
- Commit `359ce6b2b` fixes the shared Responses finalizer so an incomplete,
  rejected tool suffix on the reasoning rail cannot become visible output or
  suppress the bounded tools-free answer pass. Neighboring validation is
  147 passed / 3 deselected.
- Current-source raw Responses completed one valid tool call, then streamed
  185 reasoning plus 18 progressive content deltas to the exact post-tool
  final. Electron row 385 independently performed one real call and visibly
  painted the final over multiple DOM mutations with no warning.
- Verdict: scoped `PASS-LIVE`; retain cross-parser, stochastic long-output,
  media, forced-eviction repeat, and signed-app rows as `PARTIAL`.
  Evidence: `../20260719_bonsai_partial_prefix_responses/`.

### MiniMax-M3 current-head Auto/tool recheck and bundle drift

- PID 2277 was loaded by the real Electron Sessions-card Start action; single
  model mode stopped Bonsai. Bundle/health confirm affine `JANG_2L`, native
  MSA/index cache, and generic TurboQuant KV Off.
- Electron Auto no-tool and same-chat real-tool rows both retained separate
  reasoning, non-empty progressively painted content, clean results, and no
  warning. The tool row reused 8,980 tokens as `paged+disk`.
- Raw Responses and Chat each completed Auto no-tool, one required tool, and
  post-result continuation with separate progressive rails and correct
  terminals. Focused source/runtime validation is 759 passed / 46 skipped /
  one packaging-verifier deselection.
- Scoped M3 text Auto/tool stream: `PASS-LIVE`. Larger media/OCR/delay/REAP32/
  signed-app remain `PARTIAL`.
- Packaging: `BLOCKED`. Source `server.py` hash differs from bundled-python
  after `359ce6b2b`; run `bundle-python.sh` at release cutoff, then rerun the
  verifier without deselection.
- Evidence: `../20260719_m3_current_postfinalizer/`.
## 2026-07-19 Gemma 4 current-head parser/stream addendum

| Row | Status | Current evidence | Remaining |
|---|---|---|---|
| Gemma4 Auto reasoning + no-tool stream | PASS-LIVE at adequate cap / PARTIAL default verbosity | Real Electron row 394 produced non-empty coherent content separately from 15,629 reasoning characters and painted progressively. Raw Responses at 4,096 emitted 356 reasoning + 44 content deltas and completed; Chat emitted 428 + 30 and stopped cleanly. The 512 controls truthfully ended incomplete/length. | Default temp-1.0 short-prompt reasoning used 3,322 output tokens; classify/improve without hidden sampling or output coercion. |
| Gemma4 required tool + continuation | PASS-LIVE current | Electron row 397 made one real `file_info`, returned 5.2 KB, no warning, and restored 7,168 `paged+mixed_swa+disk` tokens. Both raw APIs emitted one schema-valid call and progressively streamed the real-result follow-up with clean terminals. | Signed packaged-app repeat after bundle refresh. |
| Gemma4 mixed-SWA storage truth | PASS-LIVE scoped | Health identifies native live rotating caches, generic live TQ Off, q4 storage-boundary encoding for full+sliding KV, preserved rotating metadata, 56 scheduler disk hits, and 239 native-TQ L2 hits. Focused selection is 361 passed. | Retain alternate/larger media and audio-family rows separately. |

Evidence: `../20260719_gemma4_current_parser_stream/`.

### DSV4 current-source Auto/parser/restart addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Electron Start and Auto reasoning UI | PASS-LIVE | Commit `4e723f311` declares DSV4 thinking/instruct capabilities and renders an explicit Auto state. A full Electron main restart logged the project `vmlx-engine`; the real Sessions Start action loaded PID 8882 before any prompt. The reopened settings drawer visibly selected Auto, with Instruct/Reasoning/Max unselected. | Signed-app repeat after bundled-Python refresh. |
| Electron one-tool continuation | PASS-LIVE | Fresh row 406 retained two separate reasoning rails, one exact `file_info(panel/package.json)`, the real 5.2 KB result, non-empty visible content, and no warning. The final screenshot shows the reasoning cards, tool card, and answer. | Broader stochastic/long exact-format quality remains separate. |
| Responses/Chat transport | PASS-LIVE controlled / PARTIAL quality | Identical no-tool and tool-schema prompts normalized byte-identically across Responses and Chat with separated reasoning/content. Both tool/result continuations streamed arguments, post-tool content, and clean terminals. | Both endpoints mutated one synthetic marker identically; same-artifact reference-runtime A/B is still required before assigning cause. |
| Native composite restart/L2 | PASS-LIVE | After visible process replacement, health reported `engine_path=dsv4`, two disk hits, 3,173 L2 block tokens, and zero generic TQ writes/hits. The tested bundle is affine JANG and uses native 43-layer SWA+CSA/HCA pool cache, not JANGTQ/MXFP or generic TQ. | Retain long reliability/perf and stress/eviction breadth as recorded. |

Focused validation is 329 Python + 100 panel tests and panel typecheck. Evidence:
`../20260719_dsv4_current_parser_auto_stream/`. Overall release remains blocked by
stale bundled Python and the other open matrix rows.

### Laguna current stream/cache addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Electron load, reasoning/content paint, and tool history | PASS-LIVE controlled | Real Sessions Start loaded PID 13706 on 8015. Row 409 separated reasoning from non-empty content. Row 412 executed one real `file_info` and painted the final answer incrementally. After two UI restarts, row 415 recalled 5.2 KB without a new tool and restored 4,980 `paged+disk+tq-native` tokens. | Longer agent/tool soak and signed packaged-app repeat. |
| Chat/Responses stream and non-stream | PASS-LIVE controlled | Responses emitted 234 reasoning/43 content deltas for no-tool, one exact call, 130/10 continuation deltas, and completed terminals. Chat kept reasoning/content separate, emitted one exact call, stopped cleanly, and produced one DONE per stream. Non-stream variants returned visible content. | Broader parser-family/protocol matrix remains campaign-level work. |
| UI/DB/argv/health cache controls | PASS-LIVE | Auto/1000 produced q4 stored-prefix TQ, paged cache, and block L2. Explicit None/max-four produced `--kv-cache-quantization none --max-cache-blocks 4`, visible TQ Off/256-token capacity, raw storage, and zero TQ objects/hits/writes. Auto was restored by another real Save & Restart. | Retain as regression row across remaining families. |
| Bounded eviction and partial L2 refault | PASS-LIVE with TQ Off | Four blocks forced ten evictions and ten disk writes. The oldest 4,538-token prompt refaulted 192 tokens from three blocks as `paged+disk` and repeated exact answer `166`. | Larger/longer stress breadth remains open. |
| q4 greedy determinism | PASS stable warm / PARTIAL cold equivalence | Four q4 restores are byte-identical to each other. Bypass-cold and explicit-None raw restores match cold byte-for-byte. q4 restored output differs from the cold full-precision output. | Decide whether lossy q4 cold-byte equivalence is required; do not mask with sampling/prompt/output coercion. |
| Laguna performance/reliability | PARTIAL | Controlled output is coherent and progressive, but natural decode remains about 23.8 tok/s and restart recall TTFT is 5.10 s. | Reference comparison, performance work/acceptance budget, and long-agent reliability. |

Validation: 411 Python passed / 1 skipped, 771 panel passed, and panel typecheck
passed. Evidence:
`../20260719_laguna_current_stream_tq_determinism_eviction/`. Release remains
blocked by stale bundled Python and all other explicit open rows.

### Prompt-disk immediate-Stop/first-turn role addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Terminal cache cleanup on Stop | PASS-LIVE scoped | `7a146eefb` makes text and MLLM stop paths wait for an in-flight terminal cleanup before cancellation. Focused race tests force the losing branch. Electron immediate Stop retained the just-finished typed snapshot before application shutdown. | Signed-app repeat after final bundled-Python refresh. |
| One-message role-aware L2 eviction | PASS-LIVE | Single user/system turns now receive real segment boundaries. At the full 10 GB ceiling, the new 1,322-token entry remained `cache_type=user` and an older 1,582-token LRU entry was evicted. | Broader stress breadth across compatible generic-TQ families remains open. |
| openPangu paged-Off SSD partial prefix | PASS-LIVE current | After UI process replacement, Electron restored 1,321/1,395 tokens from disk with zero resident L1 bytes and exact progressive content. Detached Responses and Chat after independent UI restarts restored the same 1,321-token prefix and exact-finaled. | Generic paged blocks, block-L2 refault, and generic TQ are architecture-incompatible/N/A here and remain assigned elsewhere. |

### openPangu current-head exact prompt-L2 restart recheck - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Electron eager one-model load | PASS-LIVE current | Real Start stopped Step, left only openPangu PID 65893, and health before a request reported loaded with `last_request_time=null`. Argv used `--no-paged-cache --enable-disk-cache`, with no generic TQ/block-L2. | Repeated-swap/port-conflict/LAN soak remains campaign-level. |
| Exact memory then process-restart SSD restore | PASS-LIVE current | Cold stored 817 typed tokens; same-process exact restored 817 as memory. UI Stop/Start produced PID 66691 with zero pre-request memory/L1 entries and 6,502 prompt-L2 SSD tokens; first exact Electron turn restored 817 as disk and exact-finaled with separate progressive reasoning/content. | 512K and failure/cancellation/disconnect soak remain open. |
| Raw Responses forward-prefix | PASS cache/transport, PARTIAL strict output | Cold/exact/forward emitted 307/307/512 reasoning deltas, 6/6/12 content deltas, one text-done and completed each. Forward reused 592 tokens but emitted both deliberately conflicting A/B markers. | Retain strict B-only fidelity as partial; do not coerce output or blame the official affine bundle. |
| Generic block/TQ cache | N/A architecture-safe | Health/source expose typed MLA latent KV, DSA indexer, rotating SWA, and path-dependent convolution state; generic block partial reuse and TQ remain unsupported. | Prove generic SSD-only and RAM-to-SSD hierarchy on compatible families. |

Evidence: `../20260719_openpangu_current_disk_restore/`.
| Responses `response.usage` extension parity | PASS-LIVE scoped | `cc4251318` explicitly gates the private event behind `X-vMLX-Stream-Usage: incremental`; ordinary Responses clients receive authoritative usage only on `response.completed`. The retained openPangu A/B had zero private events on the standard stream and 337 when negotiated. A current-head Laguna direct/gateway spot-check at `76e8d6c1e` again produced zero private events and one completed terminal with usage; the negotiated direct stream produced ten private events plus terminal usage. | Live remote-provider smoke and signed packaged-app repeat remain open. Evidence: `../20260719_responses_usage_extension_parity/`. |

Validation: 119/119 focused tests. Evidence:
`../20260719_prompt_disk_stop_role_durability/`. Overall release remains
blocked by the explicit open matrix rows, full suites/build, bundled-Python
refresh, signing/notarization, and publication gates.

### Responses usage event parity addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Standard Responses stream usage | PASS-LIVE scoped | `cc4251318` gates the private usage event independently of Chat-style body input. Literal curl-N emitted 383 separate reasoning deltas, nine progressive content deltas, zero `response.usage` events, one completed terminal with final usage, exact content, contiguous sequence numbers, and no errors. | Signed packaged-app repeat and failure/disconnect soak remain open. |
| Local vMLX incremental usage extension | PASS-LIVE scoped | `X-vMLX-Stream-Usage: incremental` explicitly produced 337 private usage events while preserving exact progressive content, a single completed terminal, final usage, and sequence correctness. | This private event is not claimed as part of the public Responses protocol. |
| Electron local Responses request and paint | PASS-LIVE scoped | The panel no longer sends Chat's `include_usage` field in Responses bodies and adds the private header only for local engines. Full Electron relaunch plus visible Start loaded PID 49982; the fresh turn painted separate reasoning and a partial visible `RES` before exact completion. DB content was non-empty with no tool call or warning. | Live remote-provider request smoke is pending intentional provider access. |
| Official SDK parser probe | N/A for this gate | Neither the runtime venv nor panel dependencies contained the OpenAI SDK. Literal current-source curl-N SSE is the retained API consumer proof; no test-only dependency was installed. | Optional SDK smoke may be added when the release dependency set intentionally includes one. |

Validation: 83/83 selected Python tests, 111/111 selected panel tests, and clean
panel typecheck. Evidence:
`../20260719_responses_usage_extension_parity/`. Overall release remains
`PARTIAL_NO_1_6_12_RELEASE`.

### Block-disk-only partial-prefix addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Generic full-KV Block L2 with Paged Off | PASS-LIVE scoped | MiniMax-M2.7 JANGTQ loaded through the real Electron Start action with Paged Off, Block L2 On, and q4 TQ prefix storage. An identical fresh chat after process replacement restored 192 SSD tokens and exact-finaled. A raw long Chat repeat restored 192/846 tokens as `block-disk+tq-native`. | Repeat across other compatible full-KV families. Hybrid/native/mixed-SWA families retain their architecture-specific rows. |
| UI/DB/argv/health capacity parity | PASS-LIVE | DB persisted Paged Off / Prefix On / Block L2 On. Real argv used `--no-paged-cache`. UI, corrected launch log, and health agree on three usable 64-token blocks / 192 tokens from four configured blocks with block 0 reserved; idle utilization is zero. | Retain as a regression row for later settings/model-swap sweeps. |
| SSD durability and zero persistent RAM payload | PASS-LIVE scoped | Source waits for file/SQLite visibility before dropping the only payload. Final health reported 14 durable blocks / 753 tokens, actual SSD and TQ-native hits, and zero resident KV bytes. | Fault-injected SSD write failure and signed-app repeat remain broader resilience gates. |
| Chat/Responses output emission | PASS-LIVE scoped | Timed Chat emitted 330 reasoning and 15 content deltas, stop, usage, and DONE. Responses emitted 512 reasoning-summary and 16 content deltas, output-text done, and one completed terminal. Exact content was non-empty on both. | Other model/parser families and long soak remain campaign-level work. |
| Synthetic tools-enabled replay | PARTIAL model-choice observation | Cache and loop mechanics completed: two schema-valid write/read tools, two results, 576 aggregate SSD tokens, exact final marker, no warning. The model unnecessarily interpreted `DISKONLY` as a file task and took 113.3s. | Do not use this ambiguous prompt as a strict no-tool quality row; raw no-tool streams passed separately. |

Validation: 13 focused Python passes, 299 panel passes, clean typecheck, and a
passing aggregate cache contract (454 cache-family plus 115 panel-policy
selections). Evidence: `../20260719_block_disk_only_partial/`. Overall release
remains `PARTIAL_NO_1_6_12_RELEASE`.

### Qwen3.6 JANGTQ hybrid disk-only addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Hybrid Block L2 with Paged RAM Off | PASS-LIVE scoped | Real Electron settings, DB, preview, and argv agree on Paged Off / Block L2 On. Restart row 575 restored 5,165/5,166 tokens and changed-suffix row 578 restored 5,120/5,175 as `block-disk+ssm+tq-native`; health records q4 native-TQ attention-KV hits, typed SSM companion disk hits, and zero paged-KV resident bytes. | Paged-On RAM-to-SSD hierarchy, explicit TQ Off, fault injection, signed-app repeat. |
| Disk-only cache telemetry | FIXED_SOURCE + PASS-LIVE | A live warm row incorrectly said `paged+ssm+disk`. Current source identifies the disk-only manager and appends `tq-native` only from reconstruction counters; the restart and partial UI rows show the corrected detail. | Retain pre-fix screenshot and regression tests. |
| Chat/Responses reasoning-content streams | PASS-LIVE stream / PARTIAL strict format | Chat emitted 256 reasoning plus 169 content deltas and stopped; Responses emitted 256 reasoning-summary plus 151 content deltas and completed. Both included extra math prose before the requested lines. | Ollama/Anthropic and strict model-format breadth. |
| Auto one-tool continuation | PASS-LIVE | Electron row 581 made exactly one `file_info(panel/package.json)` call, received 5.2 KB, and exact-finaled with separate reasoning and no warning. | Longer/stochastic agent soak and signed app. |

Validation: 911 expanded Python tests, 304 panel tests, and panel typecheck.
Evidence: `../20260719_qwen35_hybrid_diskonly/`. Overall release remains
`PARTIAL_NO_1_6_12_RELEASE`.

### Paged RAM + block-disk L2 hierarchy addendum - 2026-07-19

| Row | Status | Current source + live evidence | Remaining |
|---|---|---|---|
| Paged On RAM-first + L2 fallback | PASS-LIVE scoped | `8a93aa910` removes the implicit frugal default. Electron LFM cold stored 306 tokens in RAM+SSD; exact replay was 306 `paged+ssm` with zero disk reads. After bounded eviction it was 306 `paged+ssm+disk` and promoted back to L1. | Repeat on additional compatible full-KV/TQ families; signed app. |
| Restart partial block reuse | PASS-LIVE scoped | After PID 31958 -> 32602, L1 was empty and L2 retained 635 tokens. A changed suffix restored 256/312 as `paged+ssm+disk` and exact-finaled. | Fault injection and larger-context soak. |
| Paged Off SSD-only regression | PASS-LIVE scoped | Real UI Paged Off + Block L2 On restarted as `block_disk_only`; a changed suffix restored 256/311 as `block-disk+ssm` while RAM tokens/bytes stayed zero. UI then restored Paged On. | Typed/TQ breadth remains per-family. |
| Chat/Responses/Anthropic/Ollama stream parity | PASS transport / PARTIAL strict format | Each stream emitted 145 progressive content deltas and one native terminal; stream/non-stream outputs matched byte-for-byte. Chat/Responses reported 295/299 `paged+ssm`. LFM added unwanted explanation on every protocol. | Tool/reasoning continuations, cancellation/failure soak, other parsers. |
| LFM stored TQ | PASS-LIVE scoped at `748929fe3` | Base-MXFP4 LFM now derives 64-wide heads and applies q4 native TQ only to six attention-KV slots. Electron cold/warm wrote nine q4-native blocks; a real Stop/Start began with zero L1 tokens and restored all nine SSD blocks (`tq_native_hits=9`) plus one typed SSM companion entry, saving 576/1,204 tokens. | Paged-Off native-TQ SSD-only, explicit Off, larger-context eviction, four-protocol/tool/cancel breadth, signed app. |
| LFM Paged-Off SSD + explicit TQ None | PASS-LIVE scoped at post-v1.6.14 source `d23a4a37f` | Current source replaces every codec-incompatible q4 row before ordinary writes, rolls back rejected-hit credit/refs, keeps idle hybrid SSM rederive live, and retargets the companion from the 712-token cacheable prompt to the actual 576-token stored boundary. Expanded cache/scheduler tests pass 326/326. Raw Chat, raw Responses, and real Electron exact-finaled with separate reasoning/content. After real process replacement, the first Electron Responses turn restored nine ordinary SSD blocks plus one 576-token SSM disk checkpoint as `block-disk+ssm`; health retained zero L1 resident bytes and zero TQ-native writes/hits. | LFM required tools remain failing in the separate native-reasoning gate. Anthropic/Ollama under this exact setting, tool-result/cancel/failure breadth, larger eviction/fault injection, other families, signed app, and restoring the UI's steady-state Auto/Paged policy remain open. Evidence: `../20260720_lfm_diskonly_tq_off_truth/`. |

Validation: 190/190 selected cache-family tests plus 99/99 protocol/adapter
tests. Evidence:
`../20260719_paged_ram_ssd_hierarchy/`. Overall release remains
`PARTIAL_NO_1_6_12_RELEASE`.

LFM native-TQ addendum evidence:
`../20260719_lfm_native_tq4/`. The 64-token post-restart Responses probe is
`PARTIAL` for strict output because it correctly ended `response.incomplete`;
the cache restore row is independently live-proven.

## 2026-07-19 stale local model-path recovery

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Missing local model path | PASS-LIVE scoped dev Electron | Current source classifies filesystem truth without mutating SQLite and renders a missing-path card with no Start action. A disposable absent-path session visibly appeared under `MISSING MODEL (1)` with Repoint/Remove only. The real native-chooser Repoint moved it to a valid disposable path and ordinary INACTIVE Start/Delete state; the real Remove/Delete actions cleaned both fixture runs. Active Laguna PID 70292 and health remained unchanged. Focused tests pass 8/8 and typecheck passes. | Repeat in the signed app; the disposable config-only target was intentionally never started. Evidence: `../20260719_stale_path_recovery_live/`. |

## 2026-07-19 Laguna soft-sleep/wake lifecycle soak

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Same-process soft sleep/Wake | PASS-LIVE scoped 3-cycle soak | The real Electron moon and Wake controls completed three consecutive cycles on Laguna PID 70292. Each wake produced DB `running` and health `healthy/model_loaded=true`; each sleep produced DB `standby/soft` and health `standby_soft/model_loaded=true`. The final process list had exactly one engine, and the log retained all six transitions. | Deep sleep and repeated cross-model swaps are covered by the later gates; signed-app repeat remains open. Evidence: `../20260719_laguna_soft_sleep_soak/`. |

## 2026-07-19 Laguna deep-sleep unload lifecycle

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Deep sleep and same-PID Wake | PASS-LIVE scoped | The real power-settings UI set Light/Deep to `0`/`1`, after which idle Laguna automatically entered deep sleep. Electron visibly rendered `Deep Sleep`; DB stored `standby/deep`; health reported `standby_deep/model_loaded=false`; PID 70292 remained alive. Visible Wake reloaded the model in place. UI defaults were restored to `10`/`30`, and final state is `standby/soft/model_loaded=true` with one engine. | Repeat another loader class and signed-app lifecycle; repeated cross-model swaps are covered by the later gate. Evidence: `../20260719_laguna_deep_sleep_ui/`. |

## 2026-07-19 repeated one-model Electron Start swap soak

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Two MiniMax M2.7/Laguna round trips | PASS-LIVE scoped dev Electron | Real dashboard Start controls produced PID sequence `70292 -> 78868 -> 79430 -> 80033 -> 80479`. Every Start stopped the prior session and endpoint first; SQLite and `ps` showed exactly one engine after every transition. Health before any request reported both replacement models loaded with `last_request_time=null`. The Electron main log preserved the venv engine PATH and all four stop-before-start events. The final real moon control restored Laguna PID 80479 to `standby/soft`. | No generation ran; streaming/tool/cache-hit claims stay in their dedicated gates. Repeat in the signed app. Evidence: `../20260719_one_model_swap_soak/`. |

## v1.6.12 signed checkpoint addendum - 2026-07-19

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Full source/package gates | PASS on runtime source `6de1096ec` | Python 6,186 passed / 185 skipped; panel 2,332 passed / 3 skipped; typecheck, bundled engine/JANG compatibility, and both production builds passed. | Public surface publication/verification is recorded in `../20260719_release_checkpoint_1_6_12/README.md`. |
| Sequoia signed installed app | PASS-LIVE checkpoint | Exact post-staple app loaded Gemma 4 through real Start, completed three distinct reasoning/content turns, exactly one real `file_info`, cross-turn recall, raw Responses/Chat streaming, and a 100-token `paged+mixed_swa+tq-native` cache hit; real Stop returned it to inactive. | This is checkpoint smoke, not repetition of every family in the signed build. |
| Tahoe signed installed app | PASS-LIVE checkpoint | Exact post-staple app loaded through real Start; Electron showed separate reasoning and exact non-empty content. Raw Responses emitted 147 reasoning and 17 content deltas, completed terminal, and usage; real Stop returned the model inactive. | This is checkpoint smoke, not the broad signed-app family matrix. |
| Signing/notarization | PASS | Both DMGs/apps passed strict codesign and Gatekeeper; notary IDs `8b4a213b-a856-4659-8aa9-146ba211c163` and `4fb3b188-5c57-4eb2-a909-85a917ee31b4` were Accepted; both tickets stapled and validated. | None for these exact artifacts. |
| Deferred campaign rows | PARTIAL/OPEN preserved | The canonical checkpoint lists remaining parser, failure-injection, long/stochastic/media, accessibility, 512K, quality-A/B, and broader signed-app soak rows. | Resume only after the requested release pause. |
| Public v1.6.12 surfaces | PASS-PUBLIC | Source and four-asset DMG releases are public; GitHub digests equal local artifacts; PyPI wheel/sdist, Homebrew cask, both GitHub manifests, and the `mlx.studio` edge feed all serve 1.6.12. | Repair the GitHub-to-PyPI trusted-publisher mapping; authenticated publication succeeded for this release. Canonical evidence: `../20260719_release_checkpoint_1_6_12/README.md`. |

## v1.6.13 signed checkpoint addendum - 2026-07-20

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Full source/package gates | PASS at tagged source `2f509f79d` | Python 6,185 passed / 99 skipped / 92 deselected; panel 2,336 passed / 3 skipped; typecheck, pinned bundled-runtime import/source verification, and both production builds passed. | Public publication evidence is recorded in `../20260720_release_checkpoint_1_6_13/README.md`. |
| Packaged child path isolation | PASS-LIVE | The pre-fix installed Sequoia app aborted because launch cwd `/Users/eric` exposed `/Users/eric/mlx` beside bundled MLX. `5fae65d38` adds `PYTHONSAFEPATH=1` plus stable engine cwd. Final Sequoia and Tahoe session logs show each exact installed bundled Python command and clean model startup. | Keep the regression test in the full panel gate. |
| Sequoia signed installed app | PASS-LIVE scoped | On `erics-m5-max.local`, the exact stapled DMG loaded affine Gemma 4 through real Start. A clean three-turn Electron chat separated reasoning/content, executed one real `file_info`, and recalled history. Raw Responses/Chat completed progressive streams with terminal usage. After UI Stop/Start, an identical fresh prompt restored 3,359 tokens as `paged+mixed_swa+disk+tq-native`, with 53 disk/q4-native hits. | Gemma strict marker-only compliance is PARTIAL; broader signed-app family repetition remains OPEN. |
| Tahoe signed installed app | PASS-LIVE scoped | Independent Tahoe DMG passed strict signing/Gatekeeper, loaded through real Start, completed a coherent Electron reasoning/content turn, and raw Chat emitted 141 reasoning + 24 content deltas, stop, usage, and DONE. Real Stop shut it down. | Broader Tahoe-native family/media repetition remains OPEN. |
| Signing/notarization | PASS | Sequoia `bc4293f5-02f8-4f28-9cd3-d7bf51031f51` and Tahoe `4dbf39a0-d2ec-43a8-a126-ca24f3cdc3d0` are Accepted; tickets stapled/validated; DMG SHA-256 are `21cf069c...490f2` and `a244eedb...0b887`. | None for these exact artifacts. |
| Truncation control | PASS truthful / PARTIAL strict format | A 512-token Responses probe ended `response.incomplete` with `max_output_tokens`; the retained 2,048-token rerun emitted 120 reasoning + 93 content deltas and completed. Gemma sometimes adds explanation before requested markers. | Do not hide with output rewrites or sampler coercion; retain strict-format reliability as PARTIAL. |
| Deferred campaign rows | PARTIAL/OPEN preserved | This checkpoint does not promote the retained family, parser, media/audio, gateway/network-loss, accessibility, long/stochastic, 512K, or reference-A/B rows. | Resume only after the requested release pause. |
| Public v1.6.13 surfaces | PASS-PUBLIC | Source tag `v1.6.13` peels to `2f509f79d`; the public DMG release exposes four exact-digest assets; updater tag/main are `07c402d`; PyPI serves the exact tagged wheel/sdist; Homebrew main is `0b0f54c`; both GitHub manifests and `mlx.studio` serve 1.6.13 with the exact Sequoia/Tahoe hashes. | Repair the GitHub-to-PyPI trusted-publisher mapping. Authenticated publication succeeded from `erics-m5-max.local`; broader matrix rows remain PARTIAL/OPEN. |
## 2026-07-20 post-release Gemma 4 1120-media and recovery addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Gemma composer order + 1120 image budget | PASS-LIVE transport / PARTIAL OCR | On `erics-m5-max.local`, the real Electron Start button loaded the affine `JANG_4M` Gemma bundle. Commit `7687f237b` changed only Gemma composer turns to `[image_url, text, audio]` ordering. Commit `a0abd7ab3` exposed the bundle-supported 70/140/280/560/1120 image budgets, forwarded 1120 through Responses, and salted media caches by budget. The UI request log showed `[image_url,text]` plus `image_token_budget=1120`; prompt size rose 328 -> 1,144. Electron A5 completed with separate reasoning/content and no loop, but read `jangq-ai` as `jiang-ai`. | Exact small-text OCR remains PARTIAL. Run a controlled same-artifact reference A/B before assigning fault to the affine quant; never rewrite output. Evidence: `../20260720_gemma4_media_stream_cache/`. |
| Gemma mixed-SWA resident/L2 reuse at 1120 | PASS-LIVE cache | Identical A6 restored 1,137/1,138 tokens as `paged+mixed_swa+tq-native` and reduced first reasoning 1.44s -> 0.32s. A visible Electron Stop/Start cleared L1; A7 then restored the same 1,137 tokens as `paged+mixed_swa+disk+tq-native`. Health recorded 18 disk promotions and 18 native-TQ hits; runtime layout retained 40 rotating-SWA layers and TQ4 only on eight full-attention layers. | Bounded eviction and audio remain open at the 1120 budget. |
| Gemma reasoning-only fallback stream | PASS-LIVE current source (`1b89e1118`) | The retained pre-fix disk row promoted 18 blocks correctly but a length-capped first pass entered Gemma's fallback, leaked literal `thought`, and ended incomplete. Source trace found the shared control-prefix guard omitted degraded `thought\n`, while the fallback appended a reasoning-only assistant turn that the real Gemma template cannot render without tool calls. Current source buffers the degraded marker and reruns fresh original context. Forced A8 (`max_thinking_tokens=64`) emitted 57 reasoning events, then 30 progressive content events, no marker leak, and `response.completed`. Expanded focused validation is 57 passed with two intentional skips. | Keep broader family/parser protocol and signed-app repetition separate; OCR exactness is not promoted by this streaming fix. |
| Gemma media salt + post-media continuation | PASS-LIVE scoped | Same-size A/B/A answered `2`/`1`/`2`; B had no cached-token claim, return-A restored 1,097 tokens from RAM, and post-Electron-restart return-A restored 1,097 as `paged+mixed_swa+disk+tq-native` with 18 disk/native-TQ hits. The original Electron image chat then exact-recalled its marker and completed exactly one real `file_info` tool call/result/final-answer loop. | Audio, bounded eviction, non-advertised-video rejection, strict OCR reference A/B, and signed-app repetition remain open. |

## 2026-07-20 post-release Nemotron Omni audio/media-state addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Nemotron Omni audio UI/API transport | PASS-LIVE scoped | On `erics-m5-max.local`, the real Electron `Start` button eagerly loaded the JANGTQ/MXTQ Omni bundle before any request (`last_request_time=null`, 9,348.1 MB active). The clean UI attached a real WAV and produced separate reasoning plus non-empty content on three turns. Raw Chat emitted 146 reasoning and 24 content deltas, stop, usage, and DONE; Responses emitted 160 reasoning-summary and 26 text deltas plus matching done/item/completed events. | Strict stochastic formatting is PARTIAL: one early UI answer added a closing fence, and clean turn 2 omitted the requested hyphen before turn 3 exact-finaled `BLUE-6813`. Broader signed-app audio repetition remains open. |
| Omni media-salt isolation | PASS-LIVE scoped | Pre-fix identical-text orange->blue replay returned stale `MARKER=ORANGE-4729`. Current source salts each user turn with media identity, resets on mismatch, and rehydrates the latest prior media after reset. Post-fix blue replay progressively returned exact `MARKER=BLUE-6813` with stop/usage/DONE. | Add same-shape image/video media-salt controls for this Omni artifact. |
| Electron post-audio continuation | PASS-LIVE scoped | The panel now preserves historical media only for a bundle-grounded `nemotron-h` route with `config_omni.json`. A fresh UI follow-up with no attachment logged `preserveHistoricalMediaForOmni:true`, retained the first `input_audio`, reached `[MEDIA_DIAG]`, and the dispatcher logged `continuing conversation (prefix matches)`. The unseen marker was recalled over two no-attachment turns; all reasoning hashes differed. | Ordinary scheduler `paged+ssm+tq-native` counters must not be misreported as persisted Omni media state. The architecture-owned process-restart row is closed separately by the post-v1.6.14 addendum below. Evidence: `../20260720_nemotron_omni_audio/`. |
| Post-fix source/package validation | PASS scoped | Focused Omni/multimodal Python tests pass 25. Full panel passes 77 files / 2,346 tests with 3 skipped; typecheck passes. Full Python reached 6,202 pass / 96 skip / 92 deselect and failed only the intentional bundle-drift gate; after rebuilding from the clean detached JANG checkout, the complete bundled-runtime verifier and the formerly failing test pass. | Rerun both production DMG builds only when selecting the next release checkpoint; these post-v1.6.13 changes are not publicly shipped. |

## v1.6.14 public signed checkpoint addendum - 2026-07-20

| Gate | Status | Current source + live evidence | Remaining boundary |
|---|---|---|---|
| Exact source and complete suites | PASS at tagged source `e1776a485` | Annotated `v1.6.14` peels to `e1776a485e8a85f3957b79030e12f4c312eda04b`. Full Python: 6,203 passed / 96 skipped / 92 deselected. Full panel: 77 files / 2,346 passed / 3 skipped. Typecheck, production compile, bundled engine 1.6.14, and clean JANG 2.5.31 verification passed. | Focused/scoped live rows outside this checkpoint retain their own status. |
| Sequoia signed installed app | PASS-LIVE scoped | Exact stapled app loaded the real affine Gemma 4 JANG_4M bundle through Electron `Launch Session`. Two prompt-distinct turns retained separate reasoning and non-empty content with multi-turn recall. A later correctly configured turn executed exactly one real `file_info`, consumed the 5.2 KB result, and exact-finaled. Raw Responses emitted 337 reasoning and 50 content deltas with done/item/completed/usage. Real Stop closed the engine. | UI2 needed 72.8 s / 2,782 tokens despite progressive paint; strict-format/latency remains PARTIAL. Broad signed-family repeat remains OPEN. |
| Tahoe signed installed app | PASS-LIVE scoped | Independent Tahoe app loaded, then completed a literal UI Stop/Start. Its fresh identical prompt restored 73 tokens as `paged+mixed_swa+disk+tq-native` across process/profile/app variant with 0.29 s TTFT. Raw Chat emitted 185 reasoning plus 18 content deltas, stop, and DONE. Real Stop closed the engine. | Broader Tahoe-native family/media and longer soak remain OPEN. |
| Signed artifact integrity | PASS | Sequoia/Tahoe DMGs passed `hdiutil verify`, strict Developer ID signing, Apple notarization, stapling, and Gatekeeper. SHA-256 are `345fd1ec...8022c` and `d77b49ed...a739d`; exact submission IDs and full hashes are in `../20260720_release_checkpoint_1_6_14/README.md`. | None for these exact artifacts. |
| Public v1.6.14 surfaces | PASS-PUBLIC | Source and DMG releases are public/non-draft; four GitHub assets match exact sizes/digests. PyPI wheel/sdist match locally built hashes. Homebrew main is `47a691a2`. Both GitHub manifests and the custom origin are byte-identical at version 1.6.14. | Public checkpoint is complete; broader campaign remains PARTIAL. |
| Negative controls and retained quality rows | PARTIAL retained | Tools-Off raw markup and unset-working-directory structured error are explicitly excluded from tool PASS. Gemma strict output/OCR and short-prompt reasoning economy remain PARTIAL. Omni process-restart/L2 media-session persistence was OPEN at tagged source and is closed only by the later post-release addendum below. | No hidden output rewrite, sampler coercion, or false cache promotion. Continue later from the retained matrix. |
| Deferred campaign rows | PARTIAL/OPEN preserved | This checkpoint does not promote remaining family/parser/cache/media/gateway/network-loss/accessibility/long-context/stochastic/reference-A/B rows. | Resume only after the requested post-release pause. Canonical checkpoint: `../20260720_release_checkpoint_1_6_14/README.md`. |

## 2026-07-20 post-release Nemotron Omni session-L2 addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Architecture-owned process-restart restore | PASS-LIVE scoped post-v1.6.14 | Current source persists the exact Stage-1 Omni conversation/media signature and history with q4 only on attention KV and native `ArraysCache` for SSM. A real Electron seed exact-finaled `SEEDED`; real Stop/Start created fresh PID 82724; the next UI turn progressively painted separate reasoning and exact `FIR-9928`. Health recorded `hits=1`, 0.000317 s restore, q4/native codecs, and no error. | Latest exact-prefix snapshot only. Multi-snapshot LRU, partial-prefix reuse inside this architecture-owned file, bounded eviction, image/video restart controls, and signed-app repetition remain OPEN. Evidence: `../20260720_nemotron_omni_session_l2/`. |
| Raw Chat stream after independent restart | PASS-LIVE scoped | A second real UI Stop/Start preceded raw `curl -N` Chat Completions. The stream emitted 129 reasoning deltas and 13 content deltas, exact `blue6813 FIR-9928`, one stop, one usage, and one DONE. Health again recorded a q4-KV/native-SSM session hit with no error. | Responses/Anthropic process-restart repetition remains part of broader protocol breadth; this row proves Chat plus Electron. |
| Explicit L2 Off and final On | PASS-LIVE negative control | The real Server Settings UI disabled Block Disk Cache and applied Save & Restart. Health showed `enabled=false` while the old file still existed; the next exact `OFF-PATH-ACTIVE` turn left hits/stores at zero. The same UI restored L2 On and applied another restart; final screenshots/health preserve On. | None for the toggle contract on this exact route; wider family Off/On parity remains in their own rows. |

## 2026-07-20 post-release Gemma direct-audio and sampling-settings addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Gemma direct-audio math and capability | PASS-SOURCE+LIVE scoped | The real affine `JANG_4M` bundle declares audio and contains `embed_audio.*`. Current source requires both facts before advertising direct audio. A same-artifact A/B traced the decode collapse to a processor 2-D padding mask incorrectly forwarded into causal language attention: bad-wrapper max logit error `18.86328125`; corrected wrapper/direct error `0.0`. Focused tests pass 7 capability plus 9 scheduler/media rows. | Signed-app repetition and non-advertised-video rejection remain separate. Evidence: `../20260720_gemma4_audio_mask_cache/`. |
| Gemma Electron/API audio output | PASS-LIVE Thinking Off and transport / PARTIAL Auto quality | Real Electron visibly attached the WAV and produced non-empty exact-content transcription in 1.0 s. Raw Responses explicit None and Auto streamed 21 content deltas plus done/completed. Auto no longer numeric-loops, but two UI Auto turns overthought and one denied the attachment. | Investigate Auto-thinking economy/quality without hidden coercion. |
| Gemma audio resident/restart cache | PASS-LIVE scoped | Resident request restored 218/219 as `paged+mixed_swa+tq-native`; after real Electron Save & Restart the first request restored 218/219 as `paged+mixed_swa+disk+tq-native`. Health recorded four disk/native-TQ hits and zero writes. | Bounded eviction, broader protocols, and signed-app repetition. |
| Model-derived sampling sliders | PASS-LIVE affine Gemma scoped / PARTIAL family breadth | Gemma bundle `generation_config.json` declares temp 1.0/top-p .95/top-k 64; persisted session detection and the real drawer show the same. The shared defect was explicit `top_k=0` being stored but dropped by Electron Chat/Responses and both Ollama translations, which reactivated bundle 64. Current source forwards zero, Reset clears SQL overrides to inheritance, and the real drawer transitions Off -> Reset 64. Electron payload/engine logs prove zero disables top-k and Reset re-resolves 64. Raw Responses, Chat, and Ollama each produced nine progressive content deltas, exact finals, and valid terminals. The hidden engine-only Ling/Bailing top-k 20 fallback was removed (source/test only; no active artifact found). Panel 519 tests + typecheck, engine 51 sampling tests, and generation-default matrix pass. Evidence: `../20260720_sampling_defaults_ui_runtime/`. | Repeat the visual/payload/runtime chain on JANGTQ/MXTQ, base MLX/MXFP, DSV4/M3 typed routes, and a bundle with non-neutral repetition penalty. Trace the v1.6.14 app vs PATH engine 1.6.12 version-string mismatch. |

## 2026-07-20 current-source DSV4/M3 typed-cache addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| DSV4 composite cache/eager/settings | PASS-LIVE scoped | Real Electron creation and Launch Session eagerly loaded the CRACK artifact before a request, stopped the prior engine, and visibly matched bundle sampling plus native-cache controls. Exact warm restored 1,722/1,723 tokens as `paged+dsv4`; real Save & Restart restored the same as `paged+dsv4+disk`. Explicit pool-codec Off changed health and retained nine progressive content deltas plus exact completed output. | Nonterminal partial requests correctly recompute without terminal CSA/HCA state. Longer constrained-output quality/perf and broader agentic protocols remain PARTIAL. Evidence: `../20260720_dsv4_m3_current_typed_cache/`. |
| MiniMax-M3 native MSA RAM/L2 | PASS-LIVE scoped | Real Electron Launch Session replaced DSV4 and eagerly loaded before a request. Health shows dense KV 0-2 plus MSA sparse cache 3-59. Exact warm restored 1,495/1,500 tokens; resident partial restored 1,472/1,512. After real UI process replacement and empty L1, a never-stored suffix restored 1,472/1,514 from SSD as `paged+disk` with 23 disk hits/promotions, then stored the new tail. | Representative current-head image/video media is refreshed in the later capability-truth row; broader OCR/Auto quality and REAP32 remain PARTIAL/risk-blocked. |
| MiniMax-M3 TQ truth and two-turn tool stream | PASS-SOURCE+LIVE scoped | The M3 CLI now sets `VMLX_DISABLE_TQ_KV=1`, aligning disk admission/health with the loader's native MSA no-TQ contract. After Electron restart, health reported `tq_native_enabled=false`. UI turn 1 had separate non-empty reasoning/content; same-chat turn 2 had different reasoning, exactly one schema-valid `file_info(panel/package.json)`, one result, exact `M3-HEAD-TOOL-DONE SIZE=5.2 KB`, 256 `paged+disk` cached tokens, and no warning/zero-tool card. 132 focused tests pass. | Gateway-wide Chat/Responses/Anthropic/Ollama, media, version truth, and signed-app repetition stay separate. |

## 2026-07-20 current-source gateway agentic/ownership addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Electron gateway surface | PASS-LIVE scoped | The real current dev Electron API drawer visibly showed `localhost:8088`, one model, LAN Off/local-only, and Single Model On. Gateway health agreed and only M3 was running. | LAN enable/rollback, port conflict, stale installed listener 8081, and signed-app repeat. Evidence: `../20260720_gateway_agentic_ownership_current/`. |
| Chat/Responses/Anthropic/Ollama stream and non-stream | PASS-LIVE scoped | All eight request cells returned 200 and identical exact eight-line output. Each stream emitted 40 progressive content deltas and its native terminal; reasoning was empty under explicit disable. | Current cancellation/network-loss/failure injection through the gateway and additional model/parser families. |
| Four-protocol Auto reasoning | PASS-LIVE transport / PARTIAL economy | Every protocol emitted 512 separate reasoning deltas, 12 progressive content deltas, exact `M3-GATEWAY-REASON-DONE VALUE=45`, and its native terminal. | All four exhausted the 512-token reasoning budget; economy/latency and longer stochastic use remain PARTIAL. |
| Four-protocol real tool-result continuation | PASS-LIVE scoped | Each route emitted one exact `file_info(panel/package.json)` call, consumed the actual 5.2 KB result, and progressively exact-finaled without a second call. Chat/Responses also passed stream/non-stream controls. | Media-bearing tools, other parser families, simultaneous/concurrent tools, and longer soak. |
| One-model gateway auto-swap | PASS-LIVE scoped | M3 -> DSV4 -> M3 requests through 8088 stopped the old backend, eagerly loaded only the requested backend, left exactly one engine process, progressively exact-finaled, and updated health. | Repeated longer soak, crash/port/LAN rollback, and signed app. |
| Qwen3.6 MXFP4-MTP four-protocol reasoning | PASS-LIVE scoped | With the real API drawer showing Qwen as the only running backend and Single Model On, Chat, Responses, Anthropic, and Ollama each emitted 460-469 separate reasoning deltas, 12 progressive content deltas, the exact final, and its native terminal. One Qwen process remained after all four requests. | Non-stream parity for this exact artifact, gateway media/tool continuations, cancel/failure injection, other parsers, and signed app remain OPEN. Evidence: `../20260720_gateway_agentic_ownership_current/`. |

Focused validation: 92 Python adapter passes plus 87 panel gateway/session
passes with three skips. This scoped closure does not promote the overall
family/media/network/release matrix.

## 2026-07-20 MiniMax-M3 media capability-truth addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| MTP versus media telemetry | PASS-SOURCE+LIVE scoped | `/health.mtp.vl_runtime_available=false` belongs to inactive MTP on this non-MTP artifact; it is not the media capability. The owning `/v1/capabilities` endpoint reports text/vision/video runtime support, 907 vision tensors, no unwired media lanes, and the native MSA cache layout. | Keep the fields distinct in future audits; do not promote REAP or MTP from this row. Evidence: `../20260720_m3_media_capability_truth_current/`. |
| Current Electron image | PASS-LIVE scoped | Fresh real file-input attachment with Thinking Off, tools Off, temperature 0, and Responses wire visibly rendered the marker image and exact-finaled `MAGNOLIA CACHE DONE`. Persisted content was non-empty with null reasoning and no warning/tool. | Auto-thinking/stochastic and broader OCR catalog remain PARTIAL. |
| Current raw Responses video | PASS-LIVE scoped | Real MP4 `input_video` exact-finaled `BANANA8426` over four progressive content deltas, zero reasoning under explicit Off, one text-done, and one completed event. Last content to completion was 0.039767 s. | Larger-video/media-isolation/L2 evidence remains in the prior dedicated gates; signed-app repeat and REAP remain OPEN. |

## 2026-07-20 Gemma native-video versus frame-fallback addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Artifact/runtime video truth | PASS-SOURCE+LIVE scoped | The affine 12B sidecars explicitly stamp native video false; current source now honors that in `_bundle_declares_native_video` and excludes video from `declared_modalities`, while the exact token + vision config + video processor activates the already-source-owned sampled-image-frame bridge. Current `/v1/capabilities` reports runtime video without falsely calling it native artifact video. | Apply the same bundle-grounded audit to each E2B/E4B/26B/31B variant; do not infer family-wide audio/video. Evidence: `../20260720_gemma4_video_capability_bridge_current/`. |
| Electron load/settings/stream | PASS-LIVE transport | Real dropdown + Start materialized PID 20930 before any request. The visible drawer matched Auto, temperature 1.00, top-p 0.95, top-k 64, repetition 1.00. A truly fresh MP4 chat visibly streamed growing content prefixes, kept reasoning separate/empty under Off, and finalized with one non-empty message and no warning/tool/control leak. | Signed-app repetition remains OPEN. |
| Blind Gemma visual quality | FAIL/PARTIAL root cause | The fresh Electron final was `FRANCMASSONIC`, not visible marker `BANANA8426`. Timed raw direct-PNG and MP4-frame A/B also missed the same pixels while emitting progressive content plus one text-done/one completed terminal. A same-chat exact answer after the marker already appeared in history is excluded. | Run a controlled same-artifact reference-runtime A/B before assigning the failure to shared vMLX preprocessing or artifact/runtime quality. No output rewrite or prompt answer injection. |
| Automated coverage | PASS | 39 focused capability/modality rows and 739 expanded engine/multimodal/Gemma/scheduler rows passed. | Full repo suite remains a separate checkpoint gate. |

## 2026-07-20 Nemotron Omni current image-session and omitted-cap addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Electron eager load and exact cache type | PASS-LIVE scoped | Real Start loaded the MXTQ/JANGTQ2 Omni artifact before any request (`last_request_time=null`) and left one engine. Health reports q4 TurboQuant only on six attention-KV layers, native/full-precision SSM companion state, and async rederive. The visible drawer matches Auto, temperature .60, top-p .95, top-k Off, repetition 1.00. | Signed-app repetition and broader one-model swap soak remain separate. Evidence: `../20260720_nemotron_omni_media_cache_current/`. |
| Same-image session reuse and distinct-image salt | PASS-LIVE scoped in-process | Image A exact-finaled `vMLX` through progressive character paints and stored q4-KV/native-SSM state. A no-attachment follow-up exact-recalled it at 0.37 s TTFT; owning logs show `prefix matches` and zero new images. Fresh image B logged reset and returned unseen `BANANA8426` without leaking A. | Image/video process-restart controls, latest-only replacement, bounded eviction, and multi-snapshot/partial-prefix design remain PARTIAL/OPEN. |
| Omni omitted-Max Auto finalization | PASS-SOURCE+LIVE scoped | Pre-fix fresh Auto stopped at 256 reasoning-only tokens with empty visible content although bundle/server resolved 16,384. Current Chat/Responses/Anthropic pass effective max/temp/top-p to the Omni bridge. Patched UI Auto emitted separate reasoning and exact content in 107 tokens; omitted-max raw Responses emitted 257 reasoning + 24 content deltas and one completed event, while Chat emitted 351 + 23, stop, usage, and DONE. | Media-bearing Anthropic/Ollama repetition and full-suite checkpoint remain separate; no hidden cap, sampler coercion, or output rewrite was added. |
| Video stream and process-restart architecture L2 | PASS-LIVE cache/stream; PARTIAL exact format | Fresh Electron MP4 produced 33 visible states, separate 785-character reasoning, exact unseen `BANANA8426`, and one final row. Real Stop/Start replaced PID 23620 with 24498; pre-request health was loaded with `last_request_time=null`. A no-attachment same-chat follow-up restored the 51,847,398-byte q4-attention-KV/native-SSM snapshot in 0.000328 s (`hits=1`, `misses=0`) and streamed recall. Raw omitted-max Responses MP4 emitted 447 reasoning + 25 content deltas, one text-done, and one completed event. | Electron follow-up appended an unrequested Python block after the correct two lines. One-file `latest.safetensors` replacement behavior, bounded/multi-snapshot eviction, and different-history partial-prefix matching remain OPEN. |

## 2026-07-20 Gemma 4 26B-A4B MoE media-capability addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Token-only audio capability truth | PASS-SOURCE+LIVE scoped | Exact bundle is a 30-layer, 128-expert top-8 affine JANG_4M MoE with vision, `audio_config=null`, and no audio tower. Pre-fix runtime falsely promoted its reserved `audio_token_id`; current Gemma 4 detection fails closed. After real UI Stop/Start, capabilities report text/vision/video and `audio=not_advertised`; raw Responses WAV returns HTTP 400. Fifteen focused tests pass. | Other audio-capable Gemma variants keep their weight/config-gated path. Repeat signed-app and broader catalog independently. Evidence: `../20260720_gemma4_moe_media_capability_current/`. |
| MoE MP4 Electron/API stream | PASS-LIVE scoped | Real `Launch Session` stopped Nemotron and eagerly loaded one Gemma process with model-owned sampling and mixed-SWA controls visible. Fresh MP4 produced 16 progressive UI states, separate reasoning, exact unseen `BANANA8426`, and exact final. Owning logs show six-frame decode and Gemma sampled-frame fallback. Raw omitted-max Responses emitted 100 reasoning + 18 content deltas, one text-done, and one completed event. | Alternate-video salt, longer stochastic quality, native-video artifacts, and signed-app repeat remain OPEN. |
| MoE video process-restart L2 | PASS-LIVE scoped | Cold row stored six 327-token TQ-native mixed-SWA blocks. Real Stop/Start emptied L1. Fresh identical media/prompt restored 327/328 as `paged+mixed_swa+disk+tq-native`, with six disk promotions/native-TQ hits and 0.33 s TTFT; content remained exact. | Same-chat no-attachment history was exact but had zero hits and is excluded. Bounded eviction and partial-prefix media histories remain OPEN. |
| MoE post-video automatic tool turn | PASS-LIVE Electron scoped | In the same real chat after the exact MP4 row, the visible Chat drawer enabled built-in tools. The next no-attachment turn emitted 16 progressive states, kept 253 characters of reasoning separate, executed exactly one schema-valid `file_info(panel/package.json)`, consumed the real 5.2 KB result, and progressively exact-finaled in 1.8 s with no warning. `CHAT_DIAG` records zero current attachments and `has_tools=true`, so the old video bytes were not resent or treated as the new request. | Raw API post-media continuation and additional Gemma variants remain separate. Evidence: `../20260720_gemma4_moe_media_capability_current/`. |

## 2026-07-20 Qwen3.6 27B MXFP4-MTP video/cache addendum

| Gate | Status | Current-source evidence | Remaining |
|---|---|---|---|
| Exact artifact/MTP classification | PASS-SOURCE+LIVE scoped | The real artifact is base MLX MXFP4, not affine JANG or JANGTQ/MXTQ. Its index contains 333 vision and 23 MTP tensors; the 64-layer hybrid graph owns 16 attention-KV and 48 native SSM layers. Real Electron launch showed text+VL native MTP depth 3 and materialized one process before any request. | Do not generalize MTP to non-MTP names or artifacts without indexed MTP tensors. 35B MoE and other quant variants remain separate. Evidence: `../20260720_qwen36_27b_mxfp4_mtp_video_current/`. |
| Electron video stream and native MTP execution | PASS-LIVE scoped | A real file-input MP4 turn produced 86 observed UI states, separate progressively growing reasoning, progressive visible content, exact `BANANA8426` / marker, no warning/tool, and 7.68 s cold TTFT. Per-request telemetry records 192 drafts / 127 accepts, including depth-2 and depth-3 accepts. | Post-video tools, alternate-media salt, longer media/context, explicit MTP policy variants, and signed-app repeat remain OPEN. |
| Process-restart q4 attention-KV + native SSM L2 | PASS-LIVE scoped | After real Stop/Start left L1 empty, a new identical media/prompt turn restored 2,225/2,226 tokens from 35 q4-native SSD attention blocks plus one native SSM checkpoint as `paged+ssm+disk+tq-native`. It exact-finaled over 81 UI states at 0.60 s TTFT; restart MTP telemetry again includes depth-2/depth-3 accepts. | Partial-prefix changed-media/history, eviction, and SSD failure injection remain OPEN. |
| Raw Responses video streaming | PASS-LIVE scoped | Omitted-max MP4 Responses emitted 155 reasoning deltas, 16 progressive content deltas, exact content, one text-done, and one completed terminal. MTP recorded 194 drafts / 130 accepts. | Chat/Anthropic/Ollama media parity and cancellation/failure recovery remain OPEN. |
| Same-chat post-video automatic tool | PASS-LIVE Electron scoped | The real Chat drawer enabled built-in tools after the restart MP4 turn. A no-attachment follow-up produced 50 observed states, separate reasoning, exactly one schema-valid `file_info(panel/package.json)` call, the real 5.2 KB result, and exact visible content with no warning. Tool-safe native MTP correctly used depth 1 only: 13 drafts / 6 accepts and zero deeper drafts. | Raw API post-media continuation, multi-tool soak, and other Qwen parser/artifact variants remain OPEN. |
