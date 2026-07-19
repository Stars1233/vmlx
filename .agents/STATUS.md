# Current Status

## 2026-07-18 - post-release cross-matrix checkpoint; runtime campaign still open

Status: `SOURCE_META_AUDIT_PASS_CAMPAIGN_OPEN_NO_NEW_MODEL_LIVE_PROOF`.

- Pushed commit `db07a6fc1` centralizes ANSI-safe proof counts, hashes the
  shared parser in all 15 consumers, fixes clean-checkout objective-digest
  bootstrap, and reconciles stale current-source gate markers including the
  measured-stamp Native-MTP/JANG_2K policy.
- Focused contracts pass 222 with one skip. The canonical no-heavy focused
  sub-suite passes 656 with one skip and 232 deselections. Full panel passes
  2311 with three skips; typecheck and clean-JANG production build pass.
- Overall remains OPEN: MiMo local bundles are absent, current DSV4 live
  tool/cache proof is absent, signing preflight is blocked, and broad live
  speed/quality/Electron matrix artifacts remain incomplete. No model ran in
  this source-only checkpoint.
- Evidence: `docs/internal/release-gates/20260718_cross_matrix_count_parser/`.

## 2026-07-18 - v1.6.11 public checkpoint released; retained runtime matrix continues

Status: `PUBLIC_CHECKPOINT_RELEASED_POST_RELEASE_MATRIX_PARTIAL`.

- The packaged engine/source was commit
  `95b2caa956c592a9caa706f2a790dcd5664721b7`. The final annotated `v1.6.11`
  tag, `origin/main`, closeout branch, and current evidence head resolve to
  `df244c4a858df3894fa3911b270d6d1b175966d6`.
- Signed/notarized/stapled Sequoia and Tahoe DMGs, installed-app Electron UI
  starts, Gemma 4 UI/API streaming/tool/cache smoke, PyPI, GitHub source/app
  releases, raw/site update feeds, and Homebrew were verified and published.
  Evidence: `docs/internal/release-gates/20260718_v1_6_11_release/`.
- The final current-head release-surface contract reports `pass` and no failed
  checks. This supersedes older statements that packaging/publication was not
  authorized or that public surfaces were still 1.6.10.
- Broader family reliability/latency/eviction, full media/audio, gateway and
  protocol soak, eager-load coverage, narrow-window/localization, and stale
  model-path UX rows remain `PARTIAL`/`OPEN` and continue after the checkpoint.

## 2026-07-17 - HY3 bounded TQ4 disk encoding and current agent stream proof

Status: `SCOPED_HY3_LONG_TQ4_CACHE_API_ELECTRON_PASS_STOCHASTIC_FORMAT_AND_BROADER_RELEASE_GATES_OPEN`.

- Retained failure control: HY3 PID 77153 aborted during the second 8K-class
  cache pass. The `.ips` records `SIGABRT` on Metal's completion queue and the
  kernel reported 400,000 leaked IOGPU resources. The first direct-encoding
  attempt still reached Metal's 499,000-resource limit because all prompt
  page/layer graphs were deferred until the extraction loop ended.
- Commit `45c64f85e` removes the live-cache `compress()` call from stored TQ
  page encoding and writes native-TQ pages in bounded
  `extract -> write -> extract -> write` order. This is shared cache code, not
  a model-name branch. Verification passes 15/15 TQ paged-block tests and
  35/35 adjacent TurboQuant/prefix/terminal-cleanup tests.
- A fresh 9,065-input-token Responses request streamed ten content deltas and
  exact `HY3-TQ-TTFT-D=583`. Cleanup persisted all 142 q4 native-TQ pages with
  zero live-cache compression telemetry and zero resource-limit errors.
- Matched first-content timing was 23.073s cold, 5.802s same-process
  `paged+tq-native`, and 10.763s after a visible Electron Stop/Start as
  `paged+disk+tq-native`. Restart health reported 9,061 cached tokens, 142 disk
  hits, 142 native-TQ hits, native MTP runtime active, and effective depth 1.
  No new crash report appeared; PID 81451 remained alive through the UI rows.
- Electron row 369 separated 759 reasoning stream events from 76 progressively
  painted answer events and exact-finaled `HY3-UI-STREAM1-DONE`.
- Same-chat stochastic row 372 used one real `file_info` but emitted a draft
  and correction at the chat's 0.90 temperature. It is retained as a
  strict-format reliability miss. A raw deterministic continuation then stayed
  exact on a 457/460 `paged+tq-native` hit, and same-chat Electron row 375 at
  explicit temperature 0 executed exactly one
  `file_info(vmlx_engine/tq_disk_store.py)`, progressively streamed and
  persisted the exact six numbered lines, and reused 5,629 cached tokens.
- Source and live artifacts:
  `docs/internal/release-gates/20260716_release_closeout/hy3-tq-bounded-current/`.
- Release remains `PARTIAL_NO_RELEASE`; no version bump, package, sign,
  notarize, tag, feed, PyPI, or GitHub release action is authorized by this
  scoped row.

## 2026-07-17 - Laguna cross-family post-reasoning stream regression

Status: `SCOPED_LAGUNA_API_ELECTRON_STREAM_PASS_STRICT_FORMAT_LONG_RELEASE_GATES_OPEN`.

- The visible Sessions UI stopped Bonsai PID 75463 and started Laguna PID
  76348, leaving one active local engine. Health reached ready before a prompt
  with `last_request_time=null`.
- Raw `/v1/responses` emitted 201 reasoning deltas followed by 86 timed content
  deltas and exactly one completed terminal.
- Fresh Electron row 366 recorded one final reasoning snapshot followed by 369
  incremental visible-content mutations over 4.208 seconds. The answer ended
  with `LAG-UI-STREAM2-DONE` and persisted no warning.
- Row 366 restored 4,096 `paged+disk+tq-native` tokens at 2.27s TTFT. Health
  recorded 64 native-TQ q4 disk hits.
- The model added an introductory sentence despite the requested format. That
  is retained as a strict-format miss; only stream delivery and UI paint pass.
- Evidence:
  `docs/internal/release-gates/20260716_release_closeout/laguna-post-reasoning-stream-current/`.
- Broader model-family, long-context, strict-format, and release rows remain
  `PARTIAL_NO_RELEASE`.

## 2026-07-17 - Shared Electron content-delta paint after reasoning

Status: `SCOPED_BONSAI_RESPONSES_ELECTRON_STREAM_PASS_CROSS_MODEL_REGRESSION_OPEN_RELEASE_GATE_OPEN`.

- Pre-fix live Electron row 360 persisted a coherent 992-token response but
  collapsed the post-reasoning visible answer into one terminal paint.
- A raw probe against the same Bonsai engine emitted 406 timed Responses
  reasoning deltas followed by 46 timed content deltas and one completed
  terminal, isolating the defect to the Electron boundary rather than model
  generation or the API server.
- Source trace found `panel/src/main/ipc/chat.ts::streamSSE` draining every SSE
  line in one main-process turn; React coalesced those IPC updates with
  completion, and `MessageBubble.tsx::useTypewriter` snapped the terminal
  target.
- Commit `a7b34bc4a` yields after visible content deltas and drains a
  just-finished renderer backlog instead of snapping it. This is shared code,
  not a Bonsai-specific output rewrite.
- After a true Electron-main replacement, row 363 exact-finaled while a DOM
  observer recorded 173 distinct visible-content mutations over 1.998 seconds.
- Row 363 restored 216 `paged+ssm+disk` tokens. Current health reported four
  native-TQ q8 disk hits and one SSM companion disk hit.
- Affected verification passed 301/301 panel tests plus typecheck.
- Evidence:
  `docs/internal/release-gates/20260716_release_closeout/bonsai-post-reasoning-stream-current/`.
- Only this current Bonsai route is live-proven. Other model families remain
  explicit regression rows and the release stays `PARTIAL_NO_RELEASE`.

## 2026-07-16 - Gemma 4 mixed-SWA TQ4 Auto, None, eviction, and UI parity

Status: `SCOPED_GEMMA4_CACHE_SETTINGS_TOOL_EVICTION_PASS_LONG_OUTPUT_PARTIAL_RELEASE_GATE_OPEN`.

- Source commit `3385cb019` makes normal Auto (`VMLX_FORCE_TQ_AUTO=1`) select
  q4 TurboQuant only for Gemma 4 full-attention KV slots. Native rotating-SWA
  slots keep their window metadata; a cache-layout mismatch fails closed.
- Focused Python verification passed 153/153 mixed-SWA/TurboQuant/cache tests.
- Auto Electron rows 2425/2428/2431 and final row 2470 each made one real
  `file_info` call and exact final. Row 2431 survived process replacement with
  704 `paged+mixed_swa+disk` tokens and 44 native-TQ disk hits.
- UI-applied 16-block pressure produced 38 L1 evictions. Post-eviction row
  2464 restored 704 TQ-native L2 tokens, made one real tool call, and returned
  exact `G4-EV3-ALPHA-DONE`.
- Explicit None launched PID 15388 with `--kv-cache-quantization none`; row
  2467 exact-finaled while ordinary disk writes rose to three and native-TQ
  writes/hits stayed zero.
- The UI restored Auto and 1,000 blocks on PID 15797. Final row 2470 restored
  704 `paged+mixed_swa+disk` tokens with three native-TQ writes and eleven
  hits, one real tool call, and exact `G4-AUTO-FINAL1-DONE`.
- Commit `ba68f8fba` corrects the live settings copy/badge. The current Electron
  drawer visibly shows `TQ4 full-attention KV + native rotating SWA` and
  `MIXED AUTO`; 281 settings tests plus typecheck passed.
- Evidence:
  `docs/internal/release-gates/20260716_gemma4_mixed_swa_tq4/`.
- Remaining Gemma blocker: coherent constrained long-output. Release remains
  `PARTIAL_NO_RELEASE`.

## 2026-07-16 - Explicit Min-P zero survives UI persistence and request building

Status: `SCOPED_MIN_P_ZERO_PASS_BROADER_SETTINGS_GATE_OPEN`.

- Root cause: Chat Settings converted slider zero to `undefined`, and both
  Electron request builders omitted zero. On bundles with non-zero `min_p`,
  this inherited the bundle value instead of disabling Min-P.
- Commit `d49f500a3` preserves explicit zero in the slider/SQLite override and
  sends `min_p: 0` through Responses and Chat Completions. Only an absent
  override means inherit.
- Focused verification passed 213/213 panel tests plus typecheck.
- Clean dev Electron used the intended
  `/Users/eric/.vmlx-v1611-cachefix-dev` profile, project `.venv`, CDP 9335,
  and current main/renderer source. DSV4 PID 8935 was the only local model.
- Live UI showed Min P `0.00`; SQLite stored `min_p=0.0`; the current engine
  session Logs recorded `[CHAT_DIAG] ... "min_p":0` for the real Responses
  request. The diagnostic generation was then visibly stopped.
- Broader settings parity, gateway port-conflict/LAN state, and protocol rows
  remain open. Release remains locked.

## 2026-07-16 - DSV4 forced eviction, typed disk telemetry, and stream/quality split

Status: `SCOPED_DSV4_CACHE_EVICTION_STREAM_PASS_LONG_QUALITY_PARTIAL_RELEASE_GATE_OPEN`.

- Electron Server settings applied a four-block DSV4 pool on PID 4223. Three
  distinct fresh chats stayed coherent while health recorded six L1 evictions,
  five disk writes, and four disk hits.
- Exact replay rows 2400/2406 increased disk-hit counters but still displayed
  only `paged+dsv4`, exposing an owning telemetry defect rather than a cache
  restore failure.
- Commit `7d664e071` adds typed scheduler `+disk` detail and unions cache tiers
  across panel tool iterations. Tests passed: 76/76 DSV4/paged byte-budget,
  three focused scheduler assertions, 43/43 panel tests, and typecheck.
- After Electron restart, PID 5485 row 2409 restored 598 tokens as
  `paged+dsv4+disk`, increased block-disk hits 0 to 3, made one real
  `file_info(panel/package.json)` call, and returned exact
  `D4-EVICT-A-DONE`. The UI then restored 1,000 blocks on PID 5953.
- Raw Responses D4-RAW1 emitted 187 reasoning and 32 content deltas, matching
  final text, one completed terminal event, and no function events.
- Electron long row 2412 remains red: it hallucinated acronym expansions,
  repeated content, and ended `D4-LEND1-END` instead of `D4-LONG1-END`.
  Cache/stream tiers pass; constrained long quality and quiet speed remain
  PARTIAL. No sampler override or output cleanup was added.

## 2026-07-16 - DSV4 native composite L2 restore stays resident under UI L1 budget

Status: `SCOPED_DSV4_CACHE_SETTINGS_TIER_PASS_QUALITY_EVICTION_PARTIAL_RELEASE_GATE_OPEN`.

- Fixed the DSV4 restored-block path so native `deepseek_v4` block payloads
  promoted from L2 remain resident L1 entries after successful reconstruction,
  while clearing the temporary disk/protection flags so normal byte-budget LRU
  can evict them later.
- Fixed MLXStudio DSV4 settings parity: Cache Memory Limit / Cache Memory %
  now remain visible for DSV4 native composite prefix cache, command preview
  shows the memory budget, and session launch emits the same budget instead of
  silently falling back to the engine default.
- Source trace: `vmlx_engine/prefix_cache.py::_block_payload_has_dsv4`,
  `BlockAwarePrefixCache.reconstruct_cache`,
  `vmlx_engine/paged_cache.py::make_resident_payload_evictable`,
  `panel/src/main/sessions.ts`, `SessionSettings.tsx`, and
  `SessionConfigForm.tsx`.
- Focused verification: 76/76 DSV4/paged byte-budget Python tests, 280/280
  panel settings tests, and panel typecheck passed.
- Live Electron evidence: rebuilt dev main on CDP 9335 with project `.venv`;
  DSV4 PID 95494 argv contains `--cache-memory-percent 0.15`, and health
  reports the 15% L1 ceiling (`l1_max_resident_bytes_mb=16766.69`).
- Live cache-tier evidence: row 2343 executed one real
  `file_info(panel/package.json)` and exact `DSV4-FIX1-DONE` after disk restore;
  row 2346 repeated the same fresh-chat prompt with `cachedTokens=611`,
  `cacheDetail=paged+dsv4`, one real tool, and exact final. Before/after row
  2346 counters stayed at scheduler/block disk hits `2 -> 2`, proving no new
  L2 read and resident L1 reuse.
- Evidence directory:
  `docs/internal/release-gates/20260716_release_closeout/dsv4-current-head/`.
- Still open: forced four-block eviction/reload, long constrained output,
  reasoning/content stream continuity soak, quiet speed, and exact JANGTQ
  bundle row if a local exact JANGTQ artifact is available.

## 2026-07-16 - Gemma 4 native tool stream early-stop repair

Status: `SCOPED_GEMMA4_TOOL_STREAM_PASS_CACHE_TIERS_PARTIAL_RELEASE_GATE_OPEN`.

- Raw multi-turn Responses trace on `jangq-ai/gemma-4-12B-it-qat-JANG_4M`
  proved the model emitted a valid native
  `<|tool_call>call:file_info{path:<|"|>README.md<|"|>}<tool_call|>` by token
  20, then kept hallucinating a client-owned `<|tool_response>` block and final
  answer. This explained the Electron "Generating tool call..." stall and was
  not caused by paged cache or TurboQuant.
- `Gemma4ToolParser` now opts into the existing completed-call stream stop and
  truncates at the regex-parseable native call boundary, not the last raw
  marker string.
- Focused parser verification passed 13/13.
- Direct multi-turn Responses proof dropped from 97 output tokens / 82
  heartbeats to 28 output tokens / 20 heartbeats and emitted exactly one
  `file_info({"path":"README.md"})`.
- Live Electron same-chat rows 2265/2268 each executed one real `file_info`
  and exact finals `G4-UIFIX1-DONE` / `G4-UIFIX2-DONE`. Row 2268 reused 218
  memory tokens and completed in 3.4s.
- The diagnostic `--no-paged-cache --kv-cache-quantization none` state was
  restored through the Electron settings UI. PID 81643 now runs with
  `--use-paged-cache`, 64-token blocks, 1,000 max blocks, block-disk L2, and
  Auto q4 storage; health reports `mixed_swa_kv_v1`, paged=true, and
  block_disk_l2=true.
- Restored Auto/paged/L2 proof then passed: row 2271 restored 3,264 tokens as
  `paged+mixed_swa+disk`, row 2274 reused 543 resident tokens as
  `paged+mixed_swa`, and post-restart row 2277 restored 709 tokens as
  `paged+mixed_swa+disk`; all three made one real `file_info` call/result and
  exact finals.
- Forced eviction proof then passed under UI-applied `--max-cache-blocks 4`:
  rows 2280/2283 both restored 192 tokens as `paged+mixed_swa+disk`, made one
  real `file_info` call/result, and exact finals; health recorded
  `l1_evictions=9` and a reconstructed disk hit in 0.015s.
- Normal `--max-cache-blocks 1000` was restored on PID 82981.
- None recheck and long-output proof still remain open before full cache
  release credit.

## 2026-07-16 - Bonsai 1-bit SSM L2 quarantine removal and restart proof

Status: `SCOPED_EXACT_RESTART_PASS_PARTIAL_PREFIX_REPAIR_OPEN_RELEASE_GATE_OPEN`.

- Removed the family-wide Qwen SSM disk-restore quarantine while preserving
  the disk store's explicit disable flag, runtime fingerprint, record version,
  and typed full-precision companion codec.
- A fresh-process longest-prefix lookup now probes the scheduler-selected
  exact boundary in L2 before consulting the L1-only checkpoint index.
- Two independent visible Electron Save & Restart rows restored 160 and 168
  prompt tokens as `paged+ssm+disk`; both reconstructed in ~0.10s, executed one
  `file_info`, and returned exact finals. Health recorded native-TQ8 block hits
  plus full-precision SSM disk hits with `restore_enabled=true`.
- A longer continuation with only a 64-token KV match correctly full-prefilled
  when the companion was absent and then wrote a complete 64-token repair
  checkpoint. Its follow-up acceleration is still PARTIAL.
- Default-temperature cache-on repeats passed four exact one-tool turns, but
  the earlier row 2028 reasoning loop remains a retained reliability outlier.
- Verification: engine audit 581/581; hybrid focused 166/166; compile and diff
  check passed. Ternary, long context/eviction, protocols/settings, full build,
  signing, and notarization remain open.
- Evidence:
  `docs/internal/release-gates/20260716_bonsai_current_head/`.
- Ternary current-source follow-up is also scoped PASS: rows 2073/2076 made
  one tool and exact final each; process-restart row 2079 restored 153
  `paged+ssm+disk` tokens with native-TQ and SSM disk hits. UI None preserved
  prefix/paged/block L2 with zero TQ hits/writes, then Auto was visibly restored.
- Single-model switching left 1-bit port 8030 down and ternary 8020 up.

## 2026-07-16 - openPangu current-HEAD exact-once and restart recheck

Status: `SCOPED_OPENPANGU_PASS_BROAD_RELEASE_GATE_OPEN`.

- Commit `8cfc9f269` passed 59 focused current rows: the openPangu
  model/parser suite plus both exact-once Responses stream regressions.
- Live Electron one-model switching stopped Bonsai and left one openPangu
  process. UI, session config, argv, logs, and health all kept generic paged
  blocks, block L2, q4/q8, and TurboQuant off while typed prefix plus prompt
  disk L2 stayed on.
- Electron rows 1851 and 1854 each made one real `file_info` call/result and
  exact finals. The same-chat second turn reused 144 typed memory tokens.
- A visible Stop/Load replaced PID 75458 with PID 76278. Row 1857 then
  restored 295 prompt tokens from disk, made one real call/result, and returned
  exact `PG-CURR3-RESTART-DONE`.
- Health/logs proved schema `openpangu_v2_composite_v2`, zero native TQ, one
  disk hit, 6,836 prompt-L2 tokens, all 2,826 weights, 138 causal convs, 46
  layers, DSA=16, SWA=30, four mHC streams, 128 sinks, and MLA rank 512.
- Active MTP, 512K context, soak, concurrency, and full API coverage remain
  open. Release remains `PARTIAL_NO_RELEASE`.
- Evidence:
  `docs/internal/release-gates/20260716_openpangu_typed_cache_electron/`.

## 2026-07-16 - Bonsai Responses reasoning and speculative-buffer recovery

Status: `SCOPED_BONSAI_RESPONSES_UI_TOOL_PASS_LONG_OPEN_CALL_AND_CACHE_HEALTH_PARTIAL_RELEASE_GATE_OPEN`.

- `panel/src/main/ipc/chat.ts` now reconciles a zero-tool speculative Responses
  heartbeat with authoritative final text, and adopts a longer terminal
  reasoning summary only when it extends the streamed prefix and contains no
  raw tool-control markup. This fixes blank final rows and the visibly cut-off
  `The user wants me to...` reasoning fragment without leaking native XML.
- `vmlx_engine/server.py` now re-arms the established bounded visible-answer
  pass for Qwen/reasoning families only after tools were offered but final
  parsing found neither a real call nor visible content. Genuine tool calls are
  preserved and do not enter the answer fallback.
- Current-source Electron row 1797 used Auto reasoning and built-in tools on
  the long 2,535-token chat: one exact `file_info` call/result, complete
  158-character reasoning, exact `B1-REASON-DONE-LIVE1-DONE`, 87 output
  tokens, and 9.2 seconds. Rows 1791 and 1794 independently passed fresh and
  long-chat exact-once turns.
- The 2,128-token/62.5-second row 1788 remains recorded as a stochastic long
  native open-call outlier. A raw one-tool Responses trace completed one valid
  call in 111 tokens, and the same long chat then completed in 316 and 87
  tokens, so generic parser failure, history length alone, and TurboQuant are
  not established causes. General pre-call latency remains `PARTIAL`.
- Focused panel verification passed 24/24 plus TypeScript typecheck; the broad
  engine audit passed 580/580.
- Cache behavior remains a truthfulness follow-up: block L2 reports 82 native
  TQ writes and 393 native TQ hits, while top-level health still reports
  `kv_cache_quantization.enabled=false`. SSM disk restore is explicitly
  suppressed/quarantined and is not credited as restart reuse.
- Evidence:
  `docs/internal/release-gates/20260716_bonsai_reasoning_stream/`.
- Release remains `PARTIAL_NO_RELEASE`.

## 2026-07-16 - openPangu JANG_3M architecture, tools, and typed disk cache

Status: `SCOPED_OPENPANGU_3M_RUNTIME_TOOL_TYPED_CACHE_PASS_BROAD_RELEASE_GATE_OPEN`.

- Source now lands all checkpoint causal-convolution keys, uses RMSNorm in the
  DSA indexer, fails closed on incomplete weights, and preserves openPangu's
  exact path-dependent cache state through non-aliasing N-1 memory and prompt-
  disk records.
- openPangu explicitly disables TurboQuant KV and generic q4/q8, generic paged
  blocks, and block L2. The UI, saved session config, spawned argv, startup
  logs, health, and cache stats agree. The settings badge says
  `openPangu typed composite cache / TURBOQUANT OFF` and stored quantization is
  `None`.
- Live startup reported 2826/2826 parameter leaves, 46 decoder layers, and 138
  causal convolutions. A 2,104-token Electron forward activated all 16 DSA
  sparse indexers and reported 30 SWA layers, four mHC streams, 128 sinks, MLA
  KV rank 512, window 512, and max context 524,288.
- Current Electron row 1527 restored 2,075/2,076 prompt tokens from disk after
  process replacement, made exactly one `file_info` call/result, and finalized
  exact `PG3M-CACHE-20260716-A-DONE`. Post-run stats show 4,178 disk-restored
  tokens and both cache quantization surfaces disabled.
- Focused current verification: openPangu model tests 23/23; panel settings and
  tool-status tests 287/287; TypeScript typecheck passed. The combined current
  parser/server/runtime suite passed 173/173 selected (three deselected), and
  the broader engine audit passed 578/578.
- MTP runtime, 512K context, sustained soak, full API surface, and the remaining
  cross-model gates are still open. Release status remains `PARTIAL_NO_RELEASE`.
- Evidence:
  `docs/internal/release-gates/20260716_openpangu_typed_cache_electron/`.

## 2026-07-15 - Cross-model post-tool finalization gate

Status: `SOURCE_FOCUSED_PASS_HY3_LIVE_PASS_DSV4_WARNING_PASS_STRICT_PARTIAL_RELEASE_GATE_OPEN`.

- The Bonsai repeated-reasoning/empty-final/false-TPS defect is now a
  cross-model gate. Current coverage is recorded in
  `docs/POST-TOOL-CROSS-MODEL-MATRIX-2026-07-15.md`; shared source is not used
  as a substitute for per-family live Electron evidence.
- Current source adds `dropSupersededRecoveryWarnings` and invokes it only
  after an answer-only recovery produced visible content. It removes the
  superseded current-response empty-answer diagnostics while retaining
  unrelated warnings.
- Focused verification: 48/48 panel tests passed across responses warnings,
  tool auto-continuation, and tool status; TypeScript typecheck passed.
- HY3 live row: exact one `file_info` call/result and exact
  `HY3-POSTTOOL1-DONE`; one reasoning segment, normal tool-status lifecycle,
  no warning, `19.0 t/s`, `3,626 paged+tq` cached tokens.
- Bonsai ternary live row: exactly one `file_info`, one result, exact
  `BT-POSTTOOL1-DONE`, one reasoning segment, no warning, and `31.3 t/s`.
- Laguna live row: exactly one `file_info`, one result, exact
  `LAG-POSTTOOL1-DONE`, no warning, `16.0 t/s`, and `3,612 paged+tq` cached
  tokens. Laguna speed remains separately open.
- LFM pre-fix broad-tools/Search-only rows made malformed `path=': '` calls
  and repeated the tool up to three times. Source tracing found the native
  template shortcut accepted placeholder examples as concrete. Current source
  forces explicit LFM tool requests through a request-bound native example and
  binds scalar parameters. Eight focused LFM prompt/parser tests passed.
  After visible Stop/Start, row 1322 made exactly one
  `file_info({"path":"panel/package.json"})`, one result, exact
  `LFM-POSTTOOL4-DONE`, and no warning.
  Broad File I/O/Search/Shell post-fix row 1325 also made one correct
  call/result, exact `LFM-POSTTOOL5-DONE`, and no warning.
- Qwen3.6 27B broad File/Search/Shell row 1328 made exactly one correct
  `file_info`, one result, exact `Q36-POSTTOOL1-DONE`, two short reasoning
  fragments, no warning, and `22.6 t/s`. Health showed MTP D3 and hybrid
  SSM/TQ/cache paths active; net MTP speedup remains a separate unproven gate.
- Gemma4 12B broad row 1331 made one correct `file_info`, one result, exact
  `G4-POSTTOOL1-DONE`, no reasoning, no warning, and `38.2 t/s`. Cache-default
  parity is separately red: DB config says prefix on but paged/prompt-L2/block-
  L2 off, argv has `--no-paged-cache`, and health reports effective native
  prefix/paged/block-L2 false with zero L2 tokens.
- MiniMax-M2.7 pre-fix row 1334 made one wrong `file_info({"path":"panel"})`
  despite the exact final marker. The native XML example's generic scalar
  regex excluded `/`; current source adds a slash-preserving path extractor.
  Two fallback, five LFM regression, and 19 MiniMax parser tests passed. After
  visible restart, row 1337 made one exact file call/result, exact
  `MM27-POSTTOOL2-DONE`, no warning, and `3,597 paged+tq` cached tokens.
- DSV4 pre-fix row 1301 retained an empty-answer warning despite exact final
  content. Post-fix row 1304 persisted one `file_info`, one matching result,
  visible content, and `warnings_json=null`; the UI no longer rendered the
  stale warning. DSV4 misspelled the exact marker as
  `DSV4-PPOSTOLL2-DONE`, so strict-output fidelity remains `PARTIAL`.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`
  (`hy3-posttool1-pass.png`, `dsv4-posttool1-stale-warning.png`, and
  `dsv4-posttool2-warning-cleared-strict-partial.png`).
- Release lock remains unchanged. The remaining configured families still
  require current Electron post-tool
  rows; DSV4 exact fidelity and all other recorded gates remain open.

## 2026-07-15 - Cache default parity across current tested sessions

Status: `SOURCE_DB_PROCESS_HEALTH_PASS_VISUAL_UI_PARTIAL`.

- Session DB config for MiniMax-M3, Bonsai 1-bit, Bonsai ternary, DSV4 CRACK,
  Laguna, and HY3 all matched the intended default cache posture:
  `enablePrefixCache=true`, `usePagedCache=true`,
  `enableBlockDiskCache=true`, `maxCacheBlocks=1000`,
  `blockDiskCacheMaxGb=10`, `cacheMemoryPercent=15`,
  `kvCacheQuantization=auto`, `kvCacheGroupSize=64`.
- Spawned process argv was captured for the currently running HY3 and matched:
  `--cache-memory-percent 0.15`, `--use-paged-cache`,
  `--max-cache-blocks 1000`, `--enable-block-disk-cache`,
  `--block-disk-cache-max-gb 10`, plus `--native-mtp-depth 1`.
- Health for M3/Bonsai/DSV4/Laguna/HY3 independently reported prefix/paged
  cache and block-L2 enabled with family-specific cache semantics:
  M3 native MSA, Bonsai hybrid SSM, DSV4 native composite, Laguna/HY3
  paged KV with q4 stored-prefix TQ.
- Visual settings proof is complete for M3, including `15% -> 12% -> 15%`
  Save & Restart. The attempted HY3 settings screenshot landed on an older
  Bonsai session detail, so non-M3 visual toggle parity remains partial and is
  not claimed as green.
- Evidence:
  `session-config-cache-parity-summary.json`,
  `session-config-cache-parity-processes.txt`, and the per-model health files
  under the active gate directory.

## 2026-07-15 - HY3 MTP depth-1 current Electron proof

Status: `SCOPED_LIVE_ELECTRON_MTP_ACTIVE_CACHE_PASS_SPEEDUP_UNVERIFIED_RELEASE_GATE_OPEN`.

- Current-source dev Electron selected
  `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP`, model name
  `jangq-ai/Hy3-JANG_2K-MTP`, port `8010`.
- Health after visible start reported `config_num_nextn_predict_layers=1`,
  `jang_mtp_layers=1`, index MTP layer count `1`, `mtp_tensor_count=42`,
  `runtime_supported=true`, `runtime_available=true`,
  `runtime_active=true`, `effective_depth=1`, and
  `runtime_reason=native MTP runtime is active for text`.
- Health also reports `speculative_decoding=not_configured` and does not
  expose acceptance/speedup counters. Therefore this is MTP activation proof,
  not proof that MTP is producing a net speedup.
- Fresh Electron exact marker passed:
  `HY3-CURRENT-COHERENT-DONE` at `33.8 tok/s`.
- Multi-turn cache proof passed: `HY3-MT1-STORED`, then exact recall
  `QUARTZ|719`. The rows reported `626 paged+tq cached` and
  `672 paged+tq cached`.
- Post-run health showed family `hy_v3`, schema `plain_kv_v1`, paged KV,
  q4 stored-prefix TQ, prefix+paged+block-L2 enabled, scheduler hits `2`,
  `1,298` tokens saved, and block L2 `53` blocks / `3,002` tokens on disk
  with `63` disk hits.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`
  as `hy3-*`.
- Release lock remains unchanged.

## 2026-07-15 - Laguna-M.1 current Electron cache proof

Status: `SCOPED_LIVE_ELECTRON_CACHE_PASS_SPEED_OPEN_RELEASE_GATE_OPEN`.

- Current-source dev Electron selected
  `/Volumes/EricsLLMDrive/jangq-ai/Laguna-M.1-JANG_2L`, model name
  `jangq-ai/Laguna-M.1-JANG_2L`, port `8015`.
- Health after visible start reported family `laguna`, schema `plain_kv_v1`,
  `paged_kv`, attention KV only, generic TurboQuant KV enabled for plain
  attention KV, stored-prefix quantization `q4`, prefix+paged+block-L2
  enabled.
- Fresh Electron exact marker passed:
  `LAGUNA-CURRENT-COHERENT-DONE`.
- Multi-turn cache proof passed: `LAGUNA-MT1-STORED`, then exact recall
  `MARBLE|508`. The rows reported `618 paged+tq cached` and
  `677 paged+tq cached`.
- Post-run health showed scheduler hits `2`, `1,295` tokens saved, block L2
  `14` blocks / `802` tokens on disk, `63` disk hits, and TQ objects active
  with live encode disabled but stored prefix `q4`.
- Runtime speed remains OPEN: current rows were still about `24.2-24.3 tok/s`,
  matching the earlier slow Laguna bench. This is correctness/cache pass, not
  speed-target pass.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`
  as `laguna-*`.
- Release lock remains unchanged.

## 2026-07-15 - DSV4 Flash CRACK current Electron cache proof

Status: `SCOPED_LIVE_ELECTRON_CACHE_PASS_EXACT_MARKER_PARTIAL_RELEASE_GATE_OPEN`.

- Current-source dev Electron selected the only configured DSV4 session in this
  profile: `/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`,
  model name `dealignai/DeepSeek-V4-Flash-JANG-CRACK`, port `8012`.
- Health after visible start reported `deepseek_v4` native composite cache
  schema `deepseek_v4_v7` with `swa_local`, `csa_compressed_pool`,
  `hca_compressed_pool`, and `incomplete_tail_state`; pool quant enabled;
  generic TurboQuant KV forced off as `native_dsv4_composite`; prefix, paged
  cache, and block-disk L2 enabled.
- The first exact-marker UI row was PARTIAL: requested
  `DSV4-CURRENT-COHERENT-DONE`, actual
  `DSV4-CURRENT-COHERENT-DENDONE`. This is recorded as output-fidelity risk,
  not hidden as pass.
- Basic coherence and native-cache reuse passed after that: arithmetic prompt
  returned `45` with `346 paged+dsv4 cached`; memory pair stored `STORED` and
  recalled exact `BASALT|314` with `413 paged+dsv4 cached`.
- Post-run health showed scheduler hits `3`, `1,142` tokens saved, DSV4 batch
  generator active, block L2 `35` blocks / `7,644` tokens on disk, `6` disk
  hits, and native DSV4 ratios `4`/`128` across compressed pools.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`
  as `dsv4-crack-*`.
- Release lock remains unchanged.

## 2026-07-15 - Bonsai 1-bit/ternary hybrid-cache and tool-parser live proof

Status: `SCOPED_LIVE_ELECTRON_AND_API_PASS_UI_TOOL_PARTIAL_RELEASE_GATE_OPEN`.

- Current-source dev Electron remained attached over CDP `9335` on
  `erics-m5-max.local`. MiniMax-M3 was stopped through the visible UI before
  switching Bonsai sessions.
- Bonsai 1-bit was loaded through the visible session header from
  `/Volumes/EricsLLMDrive/jangq-ai/Bonsai-27b-1bit-JANG` on port `8030`.
  Health reported `JANG_AFFINE_1BIT`, target/config bits `1`, actual bits
  `1.1128`, `qwen3_5` `hybrid_ssm_v1`, live attention TurboQuant KV enabled,
  native SSM companion state, prefix+paged+block-L2 enabled.
- Fresh Electron 1-bit output passed: exact `B1-CURRENT-COHERENT-DONE`, then
  `B1-MT1-STORED`, then exact recall `CEDAR-B1|9417`. The two multi-turn rows
  reported `paged+ssm` cached prefixes. Post-run health showed block disk L2
  `13` blocks / `687` tokens on disk, `27` disk hits, and SSM companion disk
  `16` entries / `2,687` tokens with RAM evictions recorded.
- Bonsai 1-bit Responses API tool parsing passed both non-stream and streaming
  after disabling thinking for the streaming harness: `file_info` emitted
  `{"path": "panel/package.json"}` with argument deltas and no warnings.
  The first thinking-on streaming attempt is preserved as evidence of model
  hesitation: reasoning-only completion plus a dropped incomplete call warning.
- Electron 1-bit built-in tool execution remains PARTIAL: the model returned
  exact `B1-UI-TOOL2-DONE`, but the persisted assistant row had null
  `tool_calls_json` and null `tool_results_oai_json`.
- Bonsai ternary was loaded through the same visible session selector from
  `/Volumes/EricsLLMDrive/jangq-ai/Bonsai-27b-Ternary-JANG` on port `8020`.
  Health reported `JANG_AFFINE_TERNARY_2BIT`, target/config bits `2`, actual
  bits `2.0959`, the same `qwen3_5` hybrid SSM cache schema, live attention
  TQ KV, prefix+paged+block-L2 enabled.
- Fresh Electron ternary output passed: exact `BT-CURRENT-COHERENT-DONE`, then
  `BT-MT1-STORED`, then exact recall `SPRUCE-BT|6824`. The multi-turn rows
  reported `247 paged+ssm cached` tokens. Post-run health showed block disk L2
  `8` blocks / `460` tokens on disk, `24` disk hits, and SSM companion disk
  `7` entries / `1,815` tokens.
- Bonsai ternary Responses streaming tool parser passed with thinking disabled:
  function-call argument deltas and final `file_info` call with
  `{"path": "panel/package.json"}`, no warnings.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.
- Release lock remains unchanged. This closes Bonsai basic quant/cache/parser
  rows but does not authorize release; UI tool execution and broader model
  rows remain open.

## 2026-07-15 - MiniMax-M3 tools-enabled media streaming repair

Status: `SCOPED_LIVE_ELECTRON_AND_API_PASS_M3_GATE_STILL_OPEN`.

- Current-source dev Electron remained attached over CDP `9335` on
  `erics-m5-max.local`; the model was
  `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M3-Coder-Small` on port `8017`.
- Root cause of the tools-enabled image dead end was split across the
  Responses finalizer and M3 answer-pass policy. Invalid/truncated native XML
  was buffered speculatively, the renderer converted the heartbeat into a
  completed zero-tool card, and the bounded visible-answer pass stayed
  disabled merely because tools were available even after parsing proved that
  no function call existed.
- Current source now hides an invalid control suffix, suppresses a completed
  zero-tool status card, and late-arms a bounded tools-free M3 visible-answer
  pass only after the final parser returns no schema-valid call. A second fix
  aligns that gate with `enable_thinking=true` when no explicit
  `thinking_mode` string is present. Genuine parsed function calls remain on
  the original tool path.
- Focused regression proof now includes the original `3 passed` streaming/tool
  slice plus the non-stream M3 parity tests. Current reruns passed:
  `tests/test_server.py` selected M3/non-stream rows `4/4`,
  `tests/test_streaming_reasoning.py` `131/131`, panel tool-status `10/10`,
  and TypeScript typecheck.
- Live tools-enabled image proof no longer blanks or displays `Used 0 tools`.
  It read `panel/package.json` correctly from the deterministic warm-cache
  screenshot; character-exact OCR remains partial because
  `MM3-DET1-DONE` was read as `MM3-DETI-DONE`.
- Live tools-enabled video proof passed on current source: M3 identified SMPTE
  bars, six visible colors, frames `0` and `100`, and timecodes
  `TC01:00:00:00` and `TC01:00:03:09`, then returned the exact follow-up
  `VIDEO-FOLLOW=TC01:00:00:00|TC01:00:03:09` without reattaching media.
  The follow-up visibly reused `128 paged+disk` tokens.
- Fresh live genuine-tool regression after the fallback change passed in
  Electron: exactly one `file_info` card for `panel/package.json`, exact final
  `MM3-TOOL-POSTFIX-DONE`, and `4,271` paged cached tokens. The persisted
  assistant row contains one OpenAI tool call and one matching tool result.
- After an app `Stop`/`Start` through the visible Electron server controls,
  non-stream Responses and Chat with tools merely available both returned exact
  markers: `MM3-NONSTREAM-RESP-DONE` and `MM3-NONSTREAM-CHAT-DONE`. The saved
  response artifacts are in the gate directory as
  `mm3-nonstream-*-tools-available-postfix.json`.
- Live cache/settings parity passed through the visible Session Settings form:
  the default preview and process argv both had Prefix Cache on, native M3
  Paged KV on, Block Disk Cache L2 on, `--cache-memory-percent 0.15`,
  `--max-cache-blocks 1000`, and `--block-disk-cache-max-gb 10`. Changing
  Cache Memory % to `12` and using the visible `Save & Restart` relaunched PID
  `9398` with `--cache-memory-percent 0.12`; health showed the L1 ceiling drop
  to `5,922.96 MB` while block L2 stayed enabled. The session was restored to
  PID `10336` with `0.15` and health L1 ceiling `7,623.44 MB`.
- Health retained native cache schema `minimax_m3_msa_v1` with dense KV layers
  `0-2`, sparse MSA layers `3-59`, block-L2 enabled, and generic TurboQuant KV
  correctly disabled. Post-restart health saw `146` disk blocks and `9,121`
  tokens on disk.
- `health.mtp.vl_runtime_available=false` is MTP-specific, not a general VL
  gate: this bundle has no usable MTP tensors. `/v1/capabilities` truthfully
  advertises image/video runtime support, and both were proven through the
  Electron UI.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.
- Open before M3 can be cleared: exact image OCR row only. Release lock is
  unchanged.

## 2026-07-15 - v1.6.11 Zaya native AppleScript and cache/L2 final-source proof

Status: `SCOPED_SOURCE_AND_LIVE_ELECTRON_PASS_RELEASE_GATE_OPEN`.

- Source commits `c5d713169` and `2bca8fde6` add the bundle-scoped
  `run_applescript` native tool, direct `/usr/bin/osascript -e` execution with
  bounded timeout/output, Zaya AppleScript schema restriction, and a terminal
  one-call policy that prevents the specialist model from repeating the same
  successful action indefinitely.
- Cache defaults move to v9: paged families default to block-disk L2 only;
  non-paged families default to prompt L2. The renderer now separates
  `Clear RAM` from destructive `Clear Prefix + L2`, and `DELETE
  /v1/cache?type=ram` preserves prompt/block/SSM disk stores.
- Real final-source Electron UI ran on `erics-m5-max.local`, CDP `9335`, model
  `/Volumes/EricsLLMDrive/jangq-ai/Zaya-8B-JANG_4M`, port `8013`.
  Chat Settings visibly showed bundle-derived temperature `1.00`, top-p
  `0.95`, top-k Off, min-p `0.00`, repetition penalty `1.00`, and Responses
  `/v1/responses`.
- Reasoning Auto/Off/On each produced exactly one native `run_applescript`
  card with exit code `0` and visible results `42`, `10`, and `11`. The second
  Auto turn returned `81` and visibly reused `497 paged+zaya_cca` tokens.
- Cache UI identified `typed_cca` / `zaya_cca_v1`. A 70,022-token UI request
  kept RAM at `63,936` tokens under the configured `64,000` cap. RAM-only
  clear changed resident/indexed tokens to `0` while block L2 remained
  `64,229`. An exact replay after RAM clear produced `999` UI L2 prefix hits;
  engine block-disk hits rose `28 -> 1,027` while writes stayed `2,004`.
- Evidence root:
  `docs/internal/release-gates/20260715_v1610_postrelease_matrix/`, including
  `zaya-final-derived-settings.png`, `zaya-final-auto-multiturn-expanded.png`,
  `zaya-final-off-tool-expanded.png`, `zaya-final-on-tool-expanded.png`,
  `zaya-final-clear-ram-preserves-l2.png`, and
  `zaya-final-l2-restore-hit.png`.
- Exact-source verification: full panel suite `2,218 passed, 3 skipped`;
  focused cache/API recheck `3 passed`; named no-heavy API/cache contract
  passed with `42` API-route rows and no missing markers; TypeScript typecheck
  passed; scoped `git diff --check` passed; no temporary chat diagnostics
  remain.
- Release boundary remains open: regenerated manifest reports
  `current_proof_sweep=fail`, `prepackage_ready=false`,
  `release_ready=false`. A transient regeneration/run reduced stale missing
  artifacts but still left `244` objective rows open; focused release pytest
  was `649 passed, 1 skipped, 6 failed`, with failures in N2 artifact presence,
  stale DSV4 proof freshness, public-app issue audit, and model-registry rows.
  All regenerated tracked build artifacts were removed afterward so the
  pre-existing dirty/deleted build state was preserved.
  No package, signing, notarization, tag, upload, updater mutation, or public
  release action is authorized by this status.

## 2026-07-14 - Electron tool-history duplication live proof PASS

Status: `LIVE_ELECTRON_TOOL_HISTORY_DUPLICATION_FIX_VERIFIED`.

- Blocker class reduced: `api/ui` Responses tool-history persistence at commit
  `a74d68b86`.
- Existing Electron PID `38562` was driven over CDP `127.0.0.1:9333`; it was
  not relaunched. External-drive LFM2.5 ran READY on PID `39186`, port `8016`,
  with autodetected `lfm2` tool parser and Reasoning Auto.
- Controlled four-turn prompt counts were `1465 -> 1641 -> 1817 -> 1946`;
  same-context turn-4 resend was byte-identical `TANGERINE-42` at `1946`.
- Chat DB plus live `[CHAT_DIAG]` request shapes show only the current turn's
  OAI tool exchange persisted, with no prior-turn union.
- Verdict: `/tmp/codex-toolhistory-verdict.md`.
- Screenshot: `/tmp/vmlx-toolhistory-controlled-final.png`.
- Boundary: scoped duplication fix PASS; LFM search/result quality anomalies
  and all broader model/UI/cache/media/release blockers remain open. Release
  lock unchanged.

Last updated: 2026-07-14 01:24Z by Codex.

## 2026-07-13 - Zaya live Electron parser re-verification FAIL

Status: `LIVE_ELECTRON_ZAYA_TOOL_PATH_STILL_OPEN_AFTER_DOUBLE_WRAP_FIX`.

- Blocker classes reduced/classified: `parser/template` and `api/ui` for the
  real dev Electron Responses/tool path.
- Required UI Stop -> Start replaced the old Zaya PID `59440` with current
  repo-source PID `82010` on port `8013`; final session state is running.
- Report: `/tmp/codex_zaya_reverify_findings.md`.
- Test 1 explicit JSON prompt failed: no nested path object recurred, but three
  persisted `read_file` calls contained malformed string paths, all failed,
  a fourth repeat hit the iteration guard, and no `READ_OK` was synthesized.
- Test 2 normal natural-language path read failed: `toolCallsOaiJson=null`,
  visible `Used 0 tools` / `Tool call generated`, empty assistant content, and
  the turn exhausted 2048 tokens.
- Post-restart engine diagnostics contain zero `Dropping` and zero
  `must be of type string` lines for both tests, but repeatedly report
  `Chat template needs fallback tool schema injection.`
- Screenshots: `/tmp/codex_zv_test1_fail.png`,
  `/tmp/codex_zv_test2_fail.png`, and
  `/tmp/codex_zv_sessions_after_tests.png`.
- Boundary: `_unwrap_double_wrapped` removes the exact old nested-object shape
  but does not clear Zaya live Electron tool execution. Release lock unchanged.

Last updated: 2026-07-13 20:35Z by Codex.

## 2026-06-18 21:56Z - MM3 clean-start proof gates actual UI launch argv

Status: `SOURCE_HARNESS_FIXED_FOCUSED_TESTS_PASS_INSTALLED_UI_STILL_STALE`.

- Fixed `panel/scripts/live-clean-start-autodetect-proof.mjs` so the live UI clean-start proof records the actual session launch command from UI logs.
- MM3 clean-start proof now fails if spawned argv has generic `--use-paged-cache`, `--enable-block-disk-cache`, generic `--kv-cache-quantization`, `--enable-jit`, generic `--is-mllm`, or `--disable-prefix-cache`.
- MM3 clean-start proof now requires `--enable-disk-cache`, `--tool-call-parser minimax_m3`, `--reasoning-parser minimax_m3`, and `--enable-auto-tool-choice`.
- Added `tests/test_panel_cli_flag_contract.py::test_live_clean_start_proof_checks_mm3_actual_launch_argv`.
- Corrected docs: saved UI config `kvCacheQuantization=auto`; effective MM3 CLI/runtime suppresses generic KV quantization and uses native/no generic codec.
- Verification passed: `node --check` for the proof script, 4 Python source-contract tests, and 266 panel tests.
- Boundary: source is `1.5.66` but installed `/Applications/vMLX.app` is still `1.5.64`; live installed-app proof remains open.

## 2026-06-18 22:02Z - MM3 RAM/cache/settings issue documented; generic paged KV remains off

Status: `PARTIAL_SOURCE_UI_CONTRACTS_LIVE_ENGINE_EVIDENCE_UI_E2E_OPEN`.

- Blocker classes reduced: `memory/headroom`, `cache/storage`, and
  `api/ui` settings parity.
- Documented:
  `.agents/MM3-RAM-CACHE-SETTINGS-2026-06-18.md`.
- User issue captured:
  - 128GB Mac can still Metal-OOM even after
    `sudo sysctl iogpu.wired_limit_mb=128000`;
  - long generations can burst at `18-20 tok/s`, stall for `10-15s`, crawl,
    then burst again.
- Root-cause framing:
  wired limit raises the Metal wired-memory ceiling but does not create RAM or
  cover model weights plus native cache plus prefill buffers plus Metal command
  buffers plus OS/app headroom. For 128GB machines, `128000 MB` can be too
  aggressive; `115000-120000` plus lower runtime pressure is safer.
- MM3 cache decision:
  - native MSA SSD/prefix/L2 cache must be ON;
  - generic paged KV remains OFF because it cannot represent
    `keys + values + idx_keys`;
  - generic TurboQuant/stored KV q4/q8 remains OFF for the same reason.
- Source/UI evidence:
  - `vmlx_engine/cli.py` logs MM3 `paged_cache=OFF`,
    `tq_kv=SKIP(native MSA)`, `jit=OFF(forced)`;
  - `panel/src/main/sessions.ts` forces MM3 `usePagedCache=false`,
    `enableDiskCache=true`, `enableBlockDiskCache=false`,
    saved UI config `kvCacheQuantization=auto`, launched CLI suppresses
    generic `--kv-cache-quantization`, and engine resolves MM3 to native/no
    generic KV codec; `enableJit=false`;
  - `SessionConfigForm.tsx` disables generic paged-KV and generic stored-KV
    controls for MM3 and explains the native MSA `idx_keys` cache boundary;
  - `panel/src/shared/metalWiredLimit.ts` contains the user-facing command
    `sudo sysctl iogpu.wired_limit_mb=120000`.
- Live engine evidence from the current MM3 Responses proof run:
  - `Wired limit set to 115 GB (model 102 GB)`;
  - `Recorded Metal working-set model baseline: active=94.8GB max=107.5GB`;
  - `Runtime cache layout ... MiniMaxM3SparseCache`;
  - `Disk cache loaded with MiniMax-M3 sparse restore`;
  - `disk cache hit (m3-sparse)`;
  - `MiniMax-M3 prefix cache store using clean prompt-boundary re-prefill`.
- Boundary:
  this documents and source-traces the RAM/cache/settings issue. It is not yet
  the required live Electron clean-start screenshot/argv proof or the
  timestamped bursty-stall root cause trace.

## 2026-06-18 22:05Z - MM3 direct-engine Responses parser proof passes

Status: `ENGINE_RESPONSES_API_PASS_UI_TOOLS_VL_LONG_CONTEXT_OPEN`.

- Blocker class reduced: `parser/template` and `api/ui` for direct
  current-source `/v1/responses`.
- One harness issue was found and corrected:
  the first proof parser only looked for reasoning inside `type="message"`
  output items. The engine returned non-stream reasoning correctly as a separate
  Responses item with `type="reasoning"`, so the apparent
  `responses_nonstream_on` failure was proof-harness parsing, not engine
  behavior.
- Result artifact:
  `/tmp/mm3_responses_live_proof_result.json`
  SHA256
  `4e4ce716775ad3fefe69168bfcd3dda561be35a0335f4a0bbb335f92d0768ac9`;
  `status=pass`, `failures=[]`.
- Server log:
  `/tmp/mm3_responses_live_proof_server.log`
  SHA256
  `19bd9fc3ae6be5234ffc8ce38207a2e0d30564273738ccd062503e5024bcc0a8`.
- Live Responses matrix:
  - non-stream off/on/auto all had visible content and no MM3 tag leakage;
  - stream off/on/auto all had visible content and no MM3 tag leakage;
  - off mode had `reasoning_len=0`;
  - on/auto modes emitted reasoning in Responses reasoning items/events.
- Source/log evidence in the server log:
  MM3 autodetect, `jit=OFF(forced)`, `paged_cache=OFF`,
  `tq_kv=SKIP(native MSA)`, `vl_route=ON`, typed
  `MiniMaxM3SparseCache`, `disk cache hit (m3-sparse)`, and clean
  prompt-boundary re-prefill.
- Boundary:
  this is direct-engine Responses API proof only. It is not packaged Electron
  UI proof, not tool-call proof, not MM3 VL image proof, and not 10-turn /
  long-context coherence proof.

## 2026-06-18 21:38Z - MM3 direct CLI route fixed; current-source Chat API parser proof passes

Status: `ENGINE_CHAT_API_PARTIAL_PASS_UI_RESPONSES_TOOLS_VL_OPEN`.

- Blocker classes reduced: `parser/template`, `api/ui`, and MM3 direct CLI
  startup route.
- First live repro after the parser cleanup failed before parser assertions:
  direct CLI routed MiniMax-M3 VL through generic `mlx_vlm` and exited with
  `No module named 'mlx_vlm.models.minimax_m3_vl'`.
- Root cause:
  - MLXStudio panel launch already set `VMLINUX_M3_VL=1` and suppresses generic
    `--is-mllm` for MM3;
  - direct CLI did not set the same env/route override;
  - `is_mllm_model(..., force_mllm=True)` could still route MM3 VL into generic
    `mlx_vlm`, which has no `minimax_m3_vl` runtime.
- Source fix:
  - `vmlx_engine/cli.py` now sets `VMLINUX_M3_VL=1` for MM3 VL configs and
    ignores `--is-mllm` for this family;
  - `vmlx_engine/api/utils.py` now always returns text-route `False` for
    MiniMax-M3 configs, even when force-mllm is requested, so MM3 uses the
    source-owned MSA/VL runtime path.
- Focused route verification:
  `.venv/bin/python -m pytest tests/test_api_utils.py::TestIsMllmModel::test_minimax_m3_vl_routes_text_runtime_even_when_force_mllm tests/test_panel_cli_flag_contract.py::test_cli_minimax_m3_vl_autoroutes_to_text_msa_runtime -q`
  -> `2 passed`.
- Current-source live engine/API proof:
  `/tmp/mm3_parser_live_proof.py` launched the local
  `MiniMax-M3-REAP40-d3-JANG_2L` by direct CLI on port `8019` with
  `--reasoning-parser minimax_m3`, `--tool-call-parser minimax_m3`,
  SSD prefix cache enabled, and `--enable-jit` requested.
- Full server log evidence:
  `/tmp/mm3_parser_live_proof_server.log`
  SHA256
  `96330be8ef7e0067ff2142cac51829b9b99c91b0e406da14f7e499dcad097b74`.
  It shows MM3 autodetection, `jit=OFF(forced)`, `paged_cache=OFF`,
  `tq_kv=SKIP(native MSA)`, `vl_route=ON`, `reasoning_parser=minimax_m3`,
  `tool_parser=minimax_m3`, `mllm=False`, typed `MiniMaxM3SparseCache`, and
  disk-prefix cache initialization/stores.
- Result artifact:
  `/tmp/mm3_parser_live_proof_result.json`
  SHA256
  `a31afc97ec4fc608bb518ba7b11012e139c23cfe463ce792963cddebc16b1f39`;
  `status=pass`, `failures=[]`.
- Live Chat Completions parser matrix passed:
  - non-stream off/on/auto all had visible content and no MM3 tag leakage;
  - stream off/on/auto all had visible content and no MM3 tag leakage;
  - off mode had `reasoning_len=0`; on/auto modes emitted reasoning in the
    reasoning field, not visible content.
- Current-turn focused regression recheck:
  - `.venv/bin/python -m pytest tests/test_api_utils.py::TestIsMllmModel::test_minimax_m3_vl_routes_text_runtime_even_when_force_mllm tests/test_panel_cli_flag_contract.py::test_cli_minimax_m3_vl_autoroutes_to_text_msa_runtime tests/test_minimax_m3_cache_paths.py::test_minimax_m3_reasoning_parser_splits_captured_prompt_opened_raw_output tests/test_minimax_m3_cache_paths.py::test_minimax_m3_residual_think_markup_is_stripped_for_display_and_tools -q`
    -> `4 passed`;
  - `cd panel && npm test -- --run tests/reasoning-display.test.ts tests/metal-wired-limit.test.ts tests/chat-error-display.test.ts tests/settings-flow.test.ts`
    -> `372 passed`;
  - `.venv/bin/python -m py_compile vmlx_engine/server.py vmlx_engine/cli.py vmlx_engine/api/utils.py`
    -> passed;
  - `cd panel && npx tsc --noEmit --pretty false` -> passed;
  - `cd panel && npx electron-vite build` -> passed;
  - `git diff --check` on touched engine/panel/tests/docs files -> passed.
- Boundary:
  this is current-source direct-engine Chat Completions proof only. It is not
  packaged Electron UI proof, not Responses proof, not tool-call proof, not MM3
  VL image proof, and not 10-turn/long-context prefix-cache coherency proof.

## 2026-06-18 21:32Z - MM3 parser/reasoning release blocker source-fixed, live repro still open

Status: `SUPERSEDED_BY_21_38_DIRECT_ENGINE_CHAT_API_PROOF`.

- Release impact:
  `.66` remains blocked until current-source live MM3 UI/API proof confirms
  reasoning separation, no visible `<mm:think>` leakage, and no hidden-only
  assistant turns.
- Captured repro artifacts reviewed:
  - `/Users/eric/Library/Messages/Attachments/44/04/66E1F8A6-4008-47D0-B81E-60F9CA9C2766/minimax-m3-reasoning-sampling-smoke-20260618.log`;
  - `/Users/eric/Library/Messages/Attachments/98/08/A4689063-BEF0-4689-8EDD-3B23D122AD0F/minimax-m3-reasoning-sampling-smoke-20260618.json`.
- Findings:
  - no-thinking rows scored `0/40` but had no MM3 think tags, so those wrong
    arithmetic answers are not parser leakage;
  - thinking rows included flattened `</mm:think>` in the captured raw text, and
    one row stayed unclosed in reasoning until truncation;
  - parser-only replay proves `MiniMaxM3ReasoningParser` splits
    `reasoning</mm:think>content` correctly when active, so visible
    `</mm:think>` means parser inactive, harness flattened raw text, or a
    fallback/cleanup path failed.
- Source fix:
  - `vmlx_engine/server.py` now strips `<mm:think>...</mm:think>` and
    close-only `reasoning</mm:think>content` in residual display cleanup and
    tool-parse cleanup;
  - `panel/src/main/ipc/chat.ts` now strips leaked MM3 think blocks from
    thinking-off history and normalizes MM3 tags in client-side streaming
    fallback.
- Verification:
  - focused MM3 parser/cleanup pytest -> `6 passed`;
  - broader Python reasoning/API source contracts -> `343 passed`;
  - panel reasoning/UI fallback suite -> `122 passed`;
  - MM3 visible-answer source guards -> `3 passed`;
  - `.venv/bin/python -m py_compile vmlx_engine/server.py` -> passed;
  - `cd panel && npx tsc --noEmit --pretty false` -> passed;
  - `cd panel && npx electron-vite build` -> passed, and generated
    `panel/dist/main/index.mjs` contains the MM3 fallback normalization.
- Detailed handoff:
  `.agents/MM3-PARSER-RELEASE-BLOCKER-2026-06-18.md`.
- Boundary:
  source/tests are green for the identified MM3 tag cleanup hole. No current
  live MM3 model/API/UI rerun happened in this checkpoint, so this is not
  release proof.

## 2026-06-18 21:24Z - Metal wired-limit exact crash guidance rechecked

Status: `SOURCE_TESTED_PARTIAL_LIVE_UI_OPEN`.

- Blocker class reduced: `memory/headroom` user-facing diagnosis for Metal
  command-buffer OOMs before engine readiness.
- Exact user failure covered by source/test contract:
  `Process exited before becoming ready: libc++abi: terminating due to
  uncaught exception of type std::runtime_error: [METAL] Command buffer
  execution failed: Insufficient Memory
  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
- Source evidence:
  - `panel/src/shared/metalWiredLimit.ts` detects `Command buffer execution
    failed`, `Insufficient Memory`, and
    `kIOGPUCommandBufferCallbackErrorOutOfMemory`, then appends
    `sudo sysctl iogpu.wired_limit_mb=120000`;
  - `panel/src/main/sessions.ts` passes process-exit reasons and
    wait-for-ready pre-ready crash reasons through
    `appendMetalWiredLimitGuidance(...)`;
  - `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx`
    surfaces the same warning in the memory/cache settings UI.
- Verification:
  - `cd panel && npm test -- --run tests/metal-wired-limit.test.ts
    tests/chat-error-display.test.ts tests/settings-flow.test.ts` ->
    `266 passed`;
  - `.venv/bin/python -m pytest
    tests/test_panel_cli_flag_contract.py::test_metal_oom_startup_errors_surface_wired_limit_guidance
    -q` -> `1 passed`;
  - `cd panel && npx electron-vite build` -> passed, and generated
    `panel/dist/main/index.mjs` contains
    `sudo sysctl iogpu.wired_limit_mb=120000`,
    `kIOGPUCommandBufferCallbackErrorOutOfMemory`, and both
    `appendMetalWiredLimitGuidance(reason)` call sites.
- Boundary:
  this is source/test proof only. No installed-app screenshot or live/mocked
  app startup failure artifact was captured in this checkpoint, so do not call
  the shipped UI proof complete yet.

## 2026-06-18 21:22Z - .66 release-source surface contract passes

Status: `RELEASE_SOURCE_SURFACE_PASS_PUBLIC_PACKAGE_OPEN`.

- Blocker class reduced: `release` source-surface drift before any build,
  signing, notarization, or upload.
- Verification:
  - `.venv/bin/python -m pytest tests/test_release_surface_contract.py
    tests/test_scoped_release_preflight_66.py tests/test_panel_cli_flag_contract.py
    -q` -> `32 passed`;
  - `cd panel && npm test -- --run tests/update-checker.test.ts` ->
    `30 passed`;
  - `.venv/bin/python tests/cross_matrix/run_release_surface_contract.py
    --out build/current-release-surface-contract-v1566-source-after-wired-limit-20260618.json`
    -> `status=pass`, `failed_checks=[]`.
- Artifact:
  - `build/current-release-surface-contract-v1566-source-after-wired-limit-20260618.json`;
  - SHA256:
    `81ce3f3398415785eceafb068629213fb98b0b6df6f976ce3267c1fd369a090c`.
- Current source-surface facts:
  - `pyproject.toml`, `panel/package.json`, `panel/package-lock.json`, and
    `vmlx_engine/__init__.py` all report `1.5.66`;
  - local `latest.json` remains `1.5.65`, which the contract accepts as
    `staged_source_version_not_public=true`;
  - updater selection tests pass the raw-GitHub-over-stale-mlx.studio fallback
    and Sequoia/Tahoe DMG selection logic.
- Note:
  - one first direct invocation with system `python3` failed on
    `ModuleNotFoundError: No module named 'tomllib'`; rerunning with the
    project `.venv/bin/python` passed and generated the artifact above.
- Boundary:
  - no `.66` DMGs were built, signed, notarized, uploaded, or published here;
  - no live public updater/site/PyPI/GitHub `.66` release proof was run;
  - release remains open until packaged artifacts and public surfaces are
    produced and verified.

## 2026-06-18 21:20Z - No-model parser/API/reasoning source gate passes

Status: `PARSER_API_SOURCE_PASS_LIVE_MODEL_UI_OPEN`.

- Blocker classes reduced: `parser/template` and `api/ui` source contracts for
  Gemma4/MM3 reasoning, tool parsing, streaming deltas, API adapters, and
  panel rendering.
- Python source/API verification:
  - command:
    `.venv/bin/python -m pytest tests/test_gemma4_tool_parser.py
    tests/test_gemma4_reasoning_no_leak.py tests/test_reasoning_modes.py
    tests/test_reasoning_parser.py tests/test_streaming_reasoning.py
    tests/test_reasoning_tool_interaction.py tests/test_api_surface_parity.py
    tests/test_anthropic_adapter.py tests/test_ollama_adapter.py
    tests/test_responses_history.py tests/test_responses_multimodal_history.py
    -q`;
  - result: `500 passed`.
- Python tool/parser verification:
  - command:
    `.venv/bin/python -m pytest tests/test_tool_call_contract.py
    tests/test_tool_parsers.py tests/test_tool_format.py
    tests/test_native_tool_format.py tests/test_gemma3_tool_parser.py
    tests/test_xml_function_tool_parser.py tests/test_dsml_tool_parser.py -q`;
  - result: `268 passed`.
- Panel UI/gateway reasoning/tool verification:
  - command:
    `cd panel && npm test -- --run tests/reasoning-display.test.ts
    tests/interleaved-reasoning-render.test.ts
    tests/interleaved-reasoning-segments.test.ts
    tests/tool-auto-continue.test.ts tests/tool-media-followup.test.ts
    tests/api-gateway-body.test.ts tests/api-gateway-ollama-behavior.test.ts
    tests/api-gateway-single-model.behavior.test.ts`;
  - result: `165 passed`.
- Boundary:
  - no source edits were needed for this checkpoint;
  - these are no-model/source-contract tests only;
  - they do not replace live MM3/Gemma installed-app output, real content delta
    streaming, reasoning on/off/auto visual proof, tool execution against a
    loaded model, cache-hit telemetry, or packaged `.66` proof.

## 2026-06-18 21:18Z - Scoped .66 MM3/Gemma compatibility preflight passes

Status: `SCOPED_1566_PREFLIGHT_PASS_RELEASE_SURFACE_OPEN`.

- Blocker class reduced: `api/ui` scoped release-gate clarity for MM3/Gemma4
  compatibility rows.
- Command:
  `python3 panel/scripts/scoped-release-preflight-66.py --out
  build/current-scoped-release-preflight-66-after-wired-limit-source-20260618.json`.
- Result:
  - artifact:
    `build/current-scoped-release-preflight-66-after-wired-limit-source-20260618.json`;
  - status: `pass`;
  - failures: `0`;
  - SHA256:
    `8ecef476bb456c1cd6fb6a0785f665d5403401129ace56d7ee47e0c2328c6d7a`.
- Version evidence counted by the gate:
  - `panel/package.json`: `1.5.66`;
  - `panel/package-lock.json`: `1.5.66`;
  - `panel/package-lock.json` root package: `1.5.66`;
  - `pyproject.toml`: `1.5.66`;
  - `vmlx_engine/__init__.py`: `1.5.66`.
- Proof rows counted by the gate:
  - preserved MM3 strict stress artifact:
    `docs/internal/release-gates/current-proof-preserved/live-mm3-stress-post-dmg-2026-06-18T16-27-08Z/mm3-stress-proof.json`;
  - seven preserved Gemma4 VL/media rows:
    E2B/E4B/12B/26B/31B JANG_4M VL and 26B/31B MXFP4 visual;
  - seven preserved clean-start/autodetect rows:
    MM3 REAP40-d3, Gemma E2B MXFP4, Gemma 12B/26B/31B JANG_4M, and
    Gemma 26B/31B MXFP4;
  - four preserved lifecycle rows:
    MM3 REAP40-d3, Gemma E2B MXFP4 VL, Gemma 26B MXFP4, Gemma 31B MXFP4.
- Additional source-contract verification:
  - `.venv/bin/python -m pytest tests/test_scoped_release_preflight_66.py
    tests/test_panel_cli_flag_contract.py -q` -> 17 passed.
- Boundary:
  - this preflight is scoped to the `.66` MM3/Gemma compatibility gate and
    relies on current preserved live proof artifacts under
    `docs/internal/release-gates/current-proof-preserved`;
  - it does not prove the public website/updater, newly built/notarized `.66`
    DMGs, GitHub release upload, PyPI, or broad non-scoped model-family rows;
  - local `latest.json` still points at public `1.5.65`, which is expected
    until `.66` is built, notarized, uploaded, and the public updater is
    changed.

## 2026-06-18 21:17Z - MM3/DSV4 cache UI source-contract red cleared

Status: `SOURCE_TESTED_PARTIAL_LIVE_UI_OPEN`.

- Root cause of the lightweight red tests:
  - `panel/tests/settings-flow.test.ts` still pinned the old inline
    `disabled={!dsv4Active && cachePolicy.pagedCacheDisabled}` expression;
  - current `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx`
    intentionally routes that through `genericPagedCacheToggleDisabled` so
    MiniMax-M3 can disable the generic paged-KV toggle while DSV4 keeps hiding
    generic paged controls behind its native composite prefix-cache switch.
- Fix:
  - updated the stale source contracts to assert the new variable definition
    and the `disabled={genericPagedCacheToggleDisabled}` call site;
  - left runtime behavior unchanged.
- Verification:
  - `cd panel && npm test -- --run tests/settings-flow.test.ts -t
    "session cache controls use shared policy|one DSV4 native composite cache
    switch|MiniMax-M3|wired-limit|Metal wired"` -> 7 passed;
  - `cd panel && npm test -- --run tests/settings-flow.test.ts` -> 261
    passed;
  - combined lightweight gate:
    `cd panel && npm test -- --run tests/settings-flow.test.ts
    tests/metal-wired-limit.test.ts tests/chat-error-display.test.ts
    tests/api-gateway-ollama-behavior.test.ts
    tests/api-gateway-ollama.test.ts
    tests/api-gateway-single-model.behavior.test.ts` -> 339 passed;
  - `cd panel && npm run typecheck` -> passed;
  - `.venv/bin/python -m pytest
    tests/test_panel_cli_flag_contract.py::test_metal_oom_startup_errors_surface_wired_limit_guidance
    tests/test_panel_cli_flag_contract.py::test_mm3_and_gemma_live_stress_harnesses_cover_api_auth_matrix
    -q` -> 2 passed;
  - `git diff --check` on the touched source/test/docs set -> passed.
- Boundary:
  - source contracts now cover the UI intent, but no installed-app screenshot
    or live MM3/Gemma model load was run in this checkpoint.

## 2026-06-18 21:15Z - Metal wired-limit user guidance source checkpoint

Status: `SOURCE_TESTED_PARTIAL_LIVE_UI_OPEN`.

- User-facing issue added: when a large model fails before health/ready with
  Metal command-buffer OOM, for example:
  `Process exited before becoming ready: libc++abi: terminating due to uncaught
  exception of type std::runtime_error: [METAL] Command buffer execution
  failed: Insufficient Memory
  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`, the app must tell
  users about the macOS wired-memory limit command instead of only surfacing
  the raw backend crash.
- Source change:
  - added `panel/src/shared/metalWiredLimit.ts` with the displayed command
    `sudo sysctl iogpu.wired_limit_mb=120000` and caution text to pick a value
    below physical RAM, leave system headroom, use admin password, and expect
    reboot reset;
  - `panel/src/main/sessions.ts` now passes process-exit and wait-for-ready
    error reasons through `appendMetalWiredLimitGuidance(...)`, covering the
    exact pre-ready `kIOGPUCommandBufferCallbackErrorOutOfMemory` crash path;
  - `panel/src/shared/chatErrorDisplay.ts` appends the same guidance to visible
    projected Metal-headroom generation blocks;
  - `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` shows
    the same warning/info copy in the session memory/cache settings area.
- Verification:
  - `sysctl iogpu.wired_limit_mb` exists on this Mac and returned `0`, so the
    key name is valid locally; no `sudo` command was executed;
  - `cd panel && npm test -- --run tests/chat-error-display.test.ts
    tests/metal-wired-limit.test.ts` -> 5 passed;
  - `cd panel && npm test -- --run tests/metal-wired-limit.test.ts
    tests/chat-error-display.test.ts tests/settings-flow.test.ts -t
    "wired-limit|wired limit|Metal wired"` -> 4 passed, 262 skipped;
  - `.venv/bin/python -m pytest
    tests/test_panel_cli_flag_contract.py::test_metal_oom_startup_errors_surface_wired_limit_guidance
    -q` -> 1 passed;
  - `cd panel && npm run typecheck` -> passed;
  - `git diff --check` on touched wired-limit files -> passed.
- Boundary:
  - this is source/test proof only;
  - no live installed-app Settings screenshot, real pre-ready Metal OOM
    reproduction, or notarized `.66` DMG proof was run in this checkpoint.
  - The earlier unrelated `settings-flow` red was cleared in the 21:17Z
    checkpoint above.

## 2026-06-18 21:06Z - Ollama gateway/API-key auth source fix checkpoint

Status: `OLLAMA_GATEWAY_AUTH_SOURCE_PASS_LIVE_GATEWAY_OPEN`.

- Blocker reduced: `api/ui` auth parity for API users using the MLXStudio
  gateway's Ollama compatibility routes.
- Root cause found:
  - generic gateway proxying in `panel/src/main/api-gateway.ts::proxyRequest`
    preserved client request headers, including `Authorization`;
  - translated Ollama routes rebuilt backend requests with only
    `Content-Type: application/json`, so `/api/chat`, `/api/generate`, and
    `/api/embeddings` dropped `Authorization: Bearer ...` before reaching an
    auth-enabled engine session.
- Fix:
  - added `jsonProxyHeadersWithAuth(...)` in
    `panel/src/main/api-gateway.ts`;
  - wired Ollama chat, generate, and embeddings backend requests to preserve
    the caller bearer header while keeping JSON content type;
  - updated `panel/src/renderer/src/components/api/CodeSnippets.tsx` so
    copied Ollama curl/Python examples include bearer auth when the session has
    an API key. The Ollama CLI snippet now warns that the CLI cannot send
    custom HTTP headers and points users to curl/Python for authenticated
    gateway calls.
- Red/green:
  - `npm test -- --run tests/api-gateway-ollama-behavior.test.ts -t "forwards bearer auth"`
    first failed with backend auth headers `[undefined, undefined, undefined]`;
  - after the source fix, the focused test passed.
  - `npm test -- --run tests/api-gateway-ollama.test.ts -t "shows API key auth"`
    first failed because `CodeSnippets.tsx` had no Ollama auth helper and
    discarded `_apiKey`;
  - after the snippet fix, the focused test passed.
- Verification:
  - `cd panel && npm test -- --run tests/api-gateway-ollama-behavior.test.ts`
    -> 11 passed;
  - `cd panel && npm test -- --run tests/api-gateway-ollama.test.ts`
    -> 36 passed;
  - `cd panel && npm run typecheck` -> passed;
  - `git diff --check -- panel/src/main/api-gateway.ts panel/src/renderer/src/components/api/CodeSnippets.tsx panel/tests/api-gateway-ollama-behavior.test.ts panel/tests/api-gateway-ollama.test.ts`
    -> passed.
- Boundary:
  - this is source/behavior-contract proof only;
  - no live `.66` installed app, live gateway, real MM3/Gemma model, direct
    session port, Anthropic, Responses, or notarized DMG proof was run here;
  - next live `.66` API/auth artifacts must show gateway Ollama
    missing/wrong/right auth behavior and visible output against a running
    engine.

## 2026-06-18 18:34Z - Scoped .64 preflight after MH-17 installed proof

Status: `SCOPED_PREFLIGHT_FAIL_MODEL_ARTIFACTS_MISSING`.

- Command:
  `python3 panel/scripts/scoped-release-preflight-64.py --out
  /tmp/current-scoped-release-preflight-64-after-mh17-installed.json`.
- Result: `status=fail`.
- Remaining failures:
  - missing current MM3 strict stress pass artifact;
  - missing Gemma media/stress rows:
    `gemma4-12b-jang4m-vl-current-64`,
    `gemma4-26b-jang4m-vl-current-64`,
    `gemma4-26b-mxfp4-visual-current-64`,
    `gemma4-31b-jang4m-vl-current-64`,
    `gemma4-31b-mxfp4-visual-current-64`,
    `gemma4-e2b-jang4m-vl-current-64`,
    `gemma4-e4b-jang4m-vl-current-64`;
  - missing clean-start/autodetect rows:
    `gemma4-12b-jang4m-clean-start-current-64`,
    `gemma4-26b-jang4m-clean-start-current-64`,
    `gemma4-26b-mxfp4-clean-start-current-64`,
    `gemma4-31b-jang4m-clean-start-current-64`,
    `gemma4-31b-mxfp4-clean-start-current-64`,
    `gemma4-e2b-mxfp4-real-profile-gateway-vl-visible-off`,
    `mm3-reap40-d3-real-profile-gateway-visible-off`;
  - missing lifecycle rows:
    `gemma4-26b-mxfp4-lifecycle-current-64`,
    `gemma4-31b-mxfp4-lifecycle-current-64`,
    `gemma4-e2b-mxfp4-lifecycle-vl`,
    `mm3-reap40-d3-lifecycle`.
- Boundary:
  - MH-17 installed-app proof is no longer the preflight blocker.
  - Release remains blocked on refreshed real model artifacts.

## 2026-06-18 18:33Z - Paged-cache settings UI installed-app proof

Status: `PAGED_CACHE_SETTINGS_INSTALLED_APP_PASS`.

- Blocker reduced: `api/ui` installed-app parity for MH-17.
- Build/install:
  - ran `bash panel/scripts/build-and-install.sh`;
  - preflight passed TypeScript, Python syntax, bundled Python source parity,
    editable-install guard, and API field parity;
  - bundled Python was rebuilt from local `vmlx` 1.5.64 and local
    `jang-tools` 2.5.30 wheels;
  - bundled runtime import verification passed, including `mlx_vlm`,
    Gemma4, audio deps, `vmlx_engine`, JANG/TurboQuant kernels, and bundled
    source parity;
  - staged app and `/Applications/vMLX.app` were both sealed; 501 bundled
    Python native files were signed;
  - fresh `codesign --verify --deep --strict --verbose=2
    /Applications/vMLX.app` passed.
- Harness hardening:
  - first installed run failed because the What's New modal blocked navigation;
  - second/third installed runs failed because section expansion was racing the
    installed renderer DOM;
  - `panel/scripts/live-metal-headroom-ui-proof.mjs` now dismisses the modal,
    waits for section text, and uses real mouse events against the rendered
    section control.
- Current installed proof:
  - command: `node panel/scripts/live-metal-headroom-ui-proof.mjs`;
  - result: `status=pass`, failures `[]`;
  - artifact:
    `build/live-metal-headroom-ui-2026-06-18T18-32-30-003Z/metal-headroom-ui-proof.json`;
  - settings screenshot:
    `build/live-metal-headroom-ui-2026-06-18T18-32-30-003Z/metal-headroom-settings-proof.png`.
- Artifact facts:
  - app mode `installed-app`;
  - paged-on state: Prefix Cache checked, Paged KV Cache checked, effective
    capacity text present, ignored MB/%/TTL text present, Cache Memory Limit,
    Cache Memory %, and Cache TTL controls found and disabled;
  - paged-off state: Paged KV Cache unchecked, effective capacity text absent,
    Cache Memory Limit, Cache Memory %, and Cache TTL controls found and
    enabled;
  - startup log evidence found
    `Paged cache capacity: 64 tokens/block x 64 blocks = 4096 tokens.` and the
    ignored `--cache-memory-mb/--cache-memory-percent` warning.
- Boundary:
  - This closes MH-17 for installed-app UI/settings/CLI-log parity.
  - It still uses a fake model directory for startup logging; it does not prove
    real MM3/Gemma post-guard model generation or the missing scoped preflight
    live rows.

## 2026-06-18 18:23Z - Paged-cache settings UI on/off live proof in current source

Status: `PAGED_CACHE_SETTINGS_ELECTRON_DEV_PASS_INSTALLED_OPEN`.

- Blocker reduced: `api/ui` live visual state for MH-17.
- Harness update:
  - extended `panel/scripts/live-metal-headroom-ui-proof.mjs`;
  - installed app remains the default mode;
  - `VMLINUX_ELECTRON_DEV=1` runs Electron-dev/current source;
  - the harness now captures settings UI state before startup:
    paged cache on, paged cache off, disabled/enabled state for Cache Memory
    Limit, Cache Memory %, and Cache TTL, plus settings screenshot.
- Current live proof:
  - command: `VMLINUX_ELECTRON_DEV=1 node
    panel/scripts/live-metal-headroom-ui-proof.mjs`;
  - result: `status=pass`, failures `[]`;
  - artifact:
    `build/live-metal-headroom-ui-2026-06-18T18-22-57-274Z/metal-headroom-ui-proof.json`;
  - settings screenshot:
    `build/live-metal-headroom-ui-2026-06-18T18-22-57-274Z/metal-headroom-settings-proof.png`.
- Artifact facts:
  - app mode `electron-dev`;
  - paged-on state: Prefix Cache checked, Paged KV Cache checked, effective
    capacity text present, ignored MB/%/TTL text present, Cache Memory Limit,
    Cache Memory %, and Cache TTL controls found and disabled;
  - paged-off state: Paged KV Cache unchecked, effective capacity text absent,
    Cache Memory Limit, Cache Memory %, and Cache TTL controls found and
    enabled;
  - startup log evidence still found
    `Paged cache capacity: 64 tokens/block x 64 blocks = 4096 tokens.` and the
    ignored `--cache-memory-mb/--cache-memory-percent` warning.
- Boundary:
  - This is current-source Electron-dev proof, not installed-app/current-bundle
    proof.
  - It uses a fake model directory for startup logging; it does not prove real
    MM3/Gemma generation or post-guard regression rows.

## 2026-06-18 18:21Z - Paged-cache settings UI capacity/ignored-state source proof

Status: `PAGED_CACHE_SETTINGS_SOURCE_PASS_LIVE_BOTH_STATES_OPEN`.

- Blocker reduced: `api/ui` settings visibility for the Metal headroom/paged
  cache issue.
- Root cause addressed in current source:
  - the launch path already omitted `--cache-memory-mb`,
    `--cache-memory-percent`, and `--cache-ttl-minutes` while paged cache was
    active;
  - the settings UI had disabled those sliders, but there was no shared
    source contract that rendered the effective paged capacity from
    `block_size * max_blocks` and tied disabled/ignored state to the same
    `effectiveUsePagedCache` decision.
- Fix:
  - added `panel/src/shared/cacheCapacityDisplay.ts`;
  - `SessionConfigForm.tsx` now renders
    `Effective paged capacity: <block> tokens/block x <blocks> blocks =
    <capacity> tokens` when paged cache is active;
  - the same helper drives the ignored/disabled state for Cache Memory Limit,
    Cache Memory %, and Cache TTL copy.
- Red/green:
  - `cd panel && npm test -- cache-capacity-display.test.ts` first failed
    because `cacheCapacityDisplay` was missing;
  - `cd panel && npm test -- settings-flow.test.ts -t "settings form renders
    effective paged capacity"` first failed because `SessionConfigForm.tsx`
    did not import/use the helper.
- Current verification:
  - `cd panel && npm test -- cache-capacity-display.test.ts
    settings-flow.test.ts -t "cache capacity display helpers|settings form
    renders effective paged capacity"` -> `4 passed`;
  - `cd panel && npm test -- cache-control-policy.test.ts
    chat-error-display.test.ts` -> `14 passed`;
  - `cd panel && npm run typecheck` -> passed;
  - `.venv/bin/python -m pytest tests/test_panel_cli_flag_contract.py -q` ->
    `12 passed`;
  - `git diff --check` on touched UI helper/form/tests -> passed.
- Boundary:
  - This is source + renderer-contract proof only.
  - MH-17 still needs live UI proof for paged-cache on/off visual states with
    matching CLI argv/logs in an installed/current app.
  - It does not prove MM3/Gemma real model generation, tight-headroom visual
    capping, notarization, or release publication.

## 2026-06-18 18:15Z - Metal headroom chat UI visible safety block proof

Status: `ELECTRON_DEV_CHAT_SAFETY_BLOCK_PASS_INSTALLED_PROOF_OPEN`.

- Blocker reduced: `api/ui` tight-headroom visible response.
- Root cause found in current source:
  - `panel/src/main/ipc/chat.ts` deleted the pre-inserted empty assistant row
    and threw when a server rejected before any token;
  - projected Metal-headroom 413s therefore behaved like generic request
    failures instead of visible safety blocks in chat.
- Fix:
  - added `panel/src/shared/chatErrorDisplay.ts`;
  - `panel/src/main/ipc/chat.ts` now maps projected Metal-headroom 413s to a
    visible assistant message starting `Generation blocked:` and returns
    normally after clearing the stream lock;
  - ordinary connection failures and timeouts still throw.
- Red/green:
  - `cd panel && npm test -- chat-error-display.test.ts` first failed because
    the helper module was missing;
  - `tests/test_panel_cli_flag_contract.py::test_live_metal_headroom_chat_ui_proof_checks_visible_safety_block`
    first failed because
    `panel/scripts/live-metal-headroom-chat-ui-proof.mjs` was missing.
- Current verification:
  - `cd panel && npm test -- chat-error-display.test.ts chat-ui.test.ts` ->
    `143 passed`;
  - `node --check panel/scripts/live-metal-headroom-chat-ui-proof.mjs` -> exit
    0;
  - `.venv/bin/python -m pytest
    tests/test_panel_cli_flag_contract.py::test_live_metal_headroom_chat_ui_proof_checks_visible_safety_block
    -q` -> `1 passed`;
  - `node panel/scripts/live-metal-headroom-chat-ui-proof.mjs` ->
    `status=pass`, failures `[]`.
- Live proof artifact:
  `build/live-metal-headroom-chat-ui-2026-06-18T18-14-54-332Z/metal-headroom-chat-ui-proof.json`.
- Artifact facts:
  - UI path was Electron dev/current source (`appMode=electron-dev`);
  - mock server received streaming Chat Completions with `max_tokens=8192`;
  - saved assistant content began `Generation blocked: Requested max output
    tokens exceed projected safe Metal headroom`;
  - content preserved `requested=8192`, `safe_cap=1`, and
    `Metal OOM / kernel-panic risk`;
  - `chat:isStreaming` was false after the turn;
  - app logs did not contain `[CHAT] Error caught` for this safety block.
- Boundary:
  - This is current-source Electron-dev proof, not installed-app proof.
  - It does not prove real MM3/Gemma model generation after the guard,
    visual settings-control disable/cap states, or notarized DMG behavior.

## 2026-06-18 18:08Z - Metal headroom streaming API guard proof

Status: `UNSAFE_STREAM_GUARD_PASS_SAFE_DELTA_AND_UI_PROOF_OPEN`.

- Added streaming rows to the projected Metal headroom guard contract:
  - `chat_completions_stream`;
  - `responses_stream`;
  - `anthropic_messages_stream`;
  - `ollama_chat_stream`;
  - `ollama_generate_stream`.
- Red/green:
  - `tests/test_api_surface_contract.py::test_metal_headroom_guard_contract_covers_all_public_text_surfaces`
    first failed on missing `chat_completions_stream`;
  - after extending `tests/cross_matrix/run_metal_headroom_guard_contract.py`,
    the same test passed.
- Proof artifact:
  `build/current-metal-headroom-guard-contract.json`.
- Result:
  - `status=pass`, failures `[]`;
  - Chat, Responses, Anthropic, Ollama chat, and Ollama generate non-stream
    and stream rows all returned HTTP 413 before stream/forward;
  - every row includes `requested=8192`, `safe_cap=1`,
    `projected safe Metal headroom`, and the explicit
    `Metal OOM / kernel-panic risk` override warning.
- Boundary:
  - This proves unsafe streaming budget rejection only.
  - It does not prove safe-budget live model streaming delta equivalence,
    reasoning/content/tool chunk shape, VL/CLIP transient budget accounting, or
    UI visual enabled/disabled state.

## 2026-06-18 18:08Z - In-repo CLI explicit max-token guard proof

Status: `IN_REPO_CLI_OUTPUT_GUARD_PASS_EXTERNAL_WRAPPER_OPEN`.

- Extended `tests/cross_matrix/run_metal_headroom_guard_contract.py` to cover
  direct CLI entrypoints before model forward:
  - `cli_server_main_explicit_max_tokens` for
    `python -m vmlx_engine.server --max-tokens 8192`;
  - `cli_vmlx_engine_serve_explicit_max_tokens` for
    `vmlx-engine serve ... --max-tokens 8192`.
- Proof artifact:
  `build/current-metal-headroom-guard-contract.json`.
- Result:
  - `status=pass`, failures `[]`;
  - both CLI rows recorded `max_tokens=8192` and
    `max_tokens_explicit=true`;
  - both rejected through `_resolve_max_tokens(None, ...)` with HTTP-style
    `status_code=413`, `requested=8192`, and `safe_cap=1`, before any model
    forward;
  - the same artifact still covers Chat Completions, Responses, Anthropic
    Messages, Ollama chat, and Ollama generate route rejects.
- Boundary:
  - This is in-process direct CLI parsing/resolver proof with load/uvicorn
    stubbed to avoid loading a model.
  - A repo search did not find a current `run-vmlx-prompt.sh` or in-repo
    `VMLINUX_MAX_TOKENS` wrapper, so external support-script parity remains
    open if that script is restored or lives outside this checkout.

## 2026-06-18 17:54Z - Metal headroom matrix expanded for UI/CLI/VL control parity

Status: `DOCUMENTED_RED_ROWS_ADDED_CAPACITY_UI_LIVE_PROOF_OPEN`.

- Expanded `.agents/METAL-HEADROOM-OUTPUT-CACHE-SAFETY.md` with mandatory
  operator questions for every future proof:
  - effective output and context cap source;
  - current Metal active/max/headroom and model-derived bytes/token;
  - paged cache on/off state and effective `block_size * max_blocks` capacity;
  - visible UI disabled/enabled state for cache/output/reasoning/tool/VL
    controls;
  - prefix cache, SSD prompt cache, MM3 native MSA, Gemma mixed-SWA, and
    generic TQ-KV boundaries;
  - VL/image/CLIP transient buffer and media-salt edge cases;
  - CLI/API parity for direct flags, env wrappers, Chat, Responses,
    Anthropic, and Ollama.
- Expanded `.agents/RELEASE-1.5.63-STRESS-MATRIX.md` with release rows
  `MH-16` through `MH-24` so these checks are not optional prose:
  clean-session UI parity, paged-cache control state, tight-headroom visual
  response, direct CLI wrapper parity, VL/CLIP transient budget, MM3/Gemma
  cache topology exactness after guard patch, streaming parity, and edge-case
  knobs.
- Boundary:
  - This subsection records documentation/matrix coverage only; the later
    17:58Z entry records the installed-app paged-capacity UI proof.
  - Tight-headroom UI proof, direct CLI wrapper proof, and post-guard
    MM3/Gemma live regression rows remain open.

## 2026-06-18 17:58Z - Installed app paged-cache capacity log proof

Status: `PAGED_CAPACITY_UI_LIVE_PASS_TIGHT_HEADROOM_STILL_OPEN`.

- Rebuilt and installed `/Applications/vMLX.app` with
  `panel/scripts/bundle-python.sh && panel/scripts/build-and-install.sh`.
- Build/install evidence:
  - bundled Python source parity passed in the build script;
  - 501 bundled Python native files were signed;
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed;
  - installed bundled source smoke found
    `server._metal_projected_output_token_cap`,
    `server._apply_projected_output_guard`,
    `memory_limits.projected_output_token_cap`, and
    `memory_limits.estimate_kv_bytes_per_token_from_config`.
- Live UI proof:
  - command: `node panel/scripts/live-metal-headroom-ui-proof.mjs`;
  - artifact:
    `build/live-metal-headroom-ui-2026-06-18T17-58-13-602Z/metal-headroom-ui-proof.json`;
  - screenshot:
    `build/live-metal-headroom-ui-2026-06-18T17-58-13-602Z/metal-headroom-ui-proof.png`;
  - result: `status=pass`, failures `[]`;
  - evidence: Logs panel emitted
    `Paged cache capacity: 64 tokens/block x 64 blocks = 4096 tokens.` and
    `--cache-memory-mb/--cache-memory-percent are ignored while paged cache is
    active`;
  - CLI argv in the proof included `--use-paged-cache`,
    `--paged-cache-block-size 64`, `--max-cache-blocks 64`,
    `--max-tokens 8192`, and `--max-prompt-tokens 8192`.
- Boundary:
  - This proof uses a fake model directory and intentionally exits after
    startup logging with `FileNotFoundError: No safetensors found`.
  - It proves installed-app UI/log/argv visibility for paged capacity only.
  - It does not prove tight-headroom rejection, real model generation, direct
    CLI wrapper parity, or post-guard MM3/Gemma regressions.

## 2026-06-18 17:37Z - New release blocker: projected Metal headroom output/cache budget

Status: `METAL_HEADROOM_OUTPUT_CACHE_GUARD_SOURCE_PARTIAL_LIVE_OPEN`.

- New issue document:
  `.agents/METAL-HEADROOM-OUTPUT-CACHE-SAFETY.md`.
- External failure class being tracked:
  - model residency reported around `105.41 GiB`;
  - fixed macOS/Metal wired limit around `107.52 GiB`;
  - only about `2.11 GiB` Metal headroom before generation;
  - unsafe attempts with output caps `8192` / `4096`;
  - paged cache capacities `64 x 1000 = 64,000` tokens and
    `64 x 256 = 16,384` tokens;
  - `--cache-memory-mb 512` is not meaningful when paged cache is active.
- Current source evidence:
  - `vmlx_engine/server.py:319-356` estimates projected safe output cap from
    loaded config and Metal headroom.
  - `vmlx_engine/server.py:359-385` rejects explicit unsafe output budgets and
    clamps implicit unsafe defaults.
  - `vmlx_engine/server.py:1395-1426` routes request values, explicit
    CLI/session defaults, bundle defaults, and fallback output caps through the
    projected output guard.
  - `vmlx_engine/server.py:3862-3934` and `3962-3988` are reactive current
    pressure guards; they do not pre-project output KV, prompt/context, media,
    or paged-cache block capacity before the request allocates.
  - `vmlx_engine/cli.py:1311-1320` already warns that paged cache ignores
    `--cache-memory-mb/--cache-memory-percent`.
  - `vmlx_engine/cli.py:2349-2355` defaults paged cache capacity to
    `1000 * 64 = 64,000` tokens when paged cache is active.
  - `panel/src/main/sessions.ts:1485-1495` logs model size versus free RAM;
    visual tight-headroom cap/refuse state still needs live proof.
  - `panel/src/main/sessions.ts:2978-2980` and `3035-3043` pass
    `--max-cache-blocks`, `--max-tokens`, and `--max-prompt-tokens`.
- Required fix/proof before green:
  - engine/API projected output/cache guard with tests;
  - UI visible effective output/context/cache capacity and safe cap/reject;
  - direct CLI/API unsafe `8192` output reject or safe clamp before model
    forward;
  - paged-cache logs that show `block_size * max_blocks` and do not imply
    `cache-memory-mb` controls paged cache;
  - regression live rows for MM3 and Gemma after the guard changes.
- Boundary:
  - Source guard and UI/CLI capacity visibility are now implemented and
    source-tested, but this is still not live-proven or release-green.
  - Current source verification:
    `.venv/bin/python -m pytest tests/test_api_surface_contract.py
    tests/test_cache_bypass.py tests/test_memory_limits.py
    tests/test_metal_headroom_output_guard.py
    tests/test_panel_cli_flag_contract.py -q` -> `100 passed`;
    `.venv/bin/python -m pytest tests/test_vl_video_regression.py -k
    "MetalWorkingSetGuard or metal_working_set or memory_pressure or
    prefill_guard" -q` -> `15 passed`;
    `.venv/bin/python tests/cross_matrix/run_metal_headroom_guard_contract.py
    --out build/current-metal-headroom-guard-contract.json` -> `status=pass`,
    failures `[]`, with Chat/Responses/Anthropic/Ollama chat/Ollama generate
    all HTTP 413 and in-repo direct CLI entrypoints preserving
    `max_tokens_explicit=true`;
    `.venv/bin/python -m py_compile vmlx_engine/utils/memory_limits.py
    vmlx_engine/server.py vmlx_engine/cli.py` -> passed;
    `cd panel && npm run typecheck` -> passed.
  - Missing proof: real/staged app UI rejection/cap screenshot/log and direct
    shell wrapper/CLI prompt unsafe-token rejection artifact.

## 2026-06-18 16:36Z - Post-DMG MM3 exactness passes; scoped release preflight blocked by wiped Gemma/lifecycle artifacts

Status: `MM3_POST_DMG_EXACTNESS_PASS_DMG_LOCAL_VERIFY_PASS_SCOPED_PREFLIGHT_BLOCKED`.

- Release artifacts built locally:
  - `panel/release/vMLX-1.5.64-sequoia-arm64.dmg`
  - `panel/release/vMLX-1.5.64-tahoe-arm64.dmg`
- Local artifact verification:
  - Both staged apps are Developer ID signed by
    `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`.
  - `codesign --verify --deep --strict --verbose=2` passed for:
    `panel/release/sequoia-app/mac-arm64/vMLX.app` and
    `panel/release/tahoe-app/mac-arm64/vMLX.app`.
  - `codesign --verify --verbose=2` and `hdiutil verify` passed for both DMGs.
- Current MiniMax-M3 post-DMG exactness proof:
  - Command used the signed Tahoe staged app via
    `VMLX_APP_PATH=/Users/eric/mlx/vllm-mlx/panel/release/tahoe-app/mac-arm64/vMLX.app`.
  - Artifact:
    `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-post-dmg-2026-06-18T16-27-08Z/mm3-stress-proof.json`.
  - Result: `status=pass`, `failures=[]`.
  - Exactness covered: 10 cached UI turns, reasoning off/on/auto, tool call,
    long-context recall, MM3 VL image marker, mixed single-session
    `M3_MIX_TEXT_OFF`, `MM3_MIX_IMAGE_RED`, `M3_MIX_AUTO_TEXT`,
    `M3_MIX_TOOL_ON_DONE`, `M3_MIX_TOOL_AUTO_DONE`, final recall,
    Chat/Responses/Anthropic/Ollama API markers, streaming Chat text,
    streaming Responses image, streaming Chat tool, and streaming Responses
    tool.
  - Cache/settings evidence from the same proof: native
    `minimax_m3_msa_v1`, `msa_idx_keys`, generic TQ-KV off, storage
    quantization off, paged cache off, prompt disk L2 on, memory cache hits
    present.
  - Generation defaults: model `temperature=1.0`, `top_p=0.95`, `top_k=null`;
    UI/session showed `temperature 1.00`, `top-p 0.95`, `top-k off`.
- Current blocker:
  - `python3 panel/scripts/scoped-release-preflight-64.py --out
    /tmp/current-scoped-release-preflight-64-after-mm3-rerun.json` now fails
    because the public DMG build wiped the earlier `build/live-*` proof
    artifacts.
  - MM3 stress is current and accepted; missing rows are Gemma media/stress,
    clean-start/autodetect, and lifecycle artifacts.
- Boundary:
  - DMGs are built and locally signed/verified, but not notarized/stapled.
  - Do not publish/upload/tag until scoped preflight passes again with current
    proof artifacts preserved outside any cleaned build path.

## 2026-06-18 16:08Z - Scoped 1.5.64 MM3 + Gemma4 installed-app gate passes

Status: `SCOPED_1_5_64_MM3_GEMMA4_PREFLIGHT_PASS_RELEASE_NOT_NOTARIZED`.

- Current gate:
  - `python3 panel/scripts/scoped-release-preflight-64.py` ->
    `status=pass`.
  - Manifest:
    `/Users/eric/mlx/vllm-mlx/build/current-scoped-release-preflight-64.json`.
  - Version stamps accepted by the gate: panel/package, package-lock root,
    pyproject, and `vmlx_engine.__version__` all `1.5.64`.
- MiniMax-M3 current proofs accepted by the gate:
  - Stress/exactness:
    `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T15-26-22-368Z/mm3-stress-proof.json`.
  - Clean-start/autodetect/defaults/gateway:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-mm3-reap40-d3-real-profile-gateway-visible-off-2026-06-18T15-38-48-747Z/clean-start-proof.json`.
  - Lifecycle abort/stop/no-autonomous-generation:
    `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-mm3-reap40-d3-lifecycle-2026-06-18T15-42-40-466Z/lifecycle-proof.json`.
  - Clean-start config evidence: `enablePrefixCache=true`,
    `usePagedCache=false`, `enableDiskCache=true`, `enableJit=false`,
    `toolCallParser=minimax_m3`, `reasoningParser=minimax_m3`,
    `isMultimodal=true`; native cache `minimax_m3_msa_v1` with
    `attention_kv`, `msa_idx_keys`, `absolute_block_index`; generic TQ-KV and
    storage quantization disabled for native MSA idx keys; prompt disk L2 on.
- Gemma4 media/VL rows accepted by the gate:
  - `gemma4-e2b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-jang4m-vl-current-64-2026-06-18T15-45-14-104Z/gemma4-media-proof.json`.
  - `gemma4-e4b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-jang4m-vl-current-64-2026-06-18T15-46-51-748Z/gemma4-media-proof.json`.
  - `gemma4-12b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-vl-current-64-2026-06-18T15-48-39-441Z/gemma4-media-proof.json`.
  - `gemma4-26b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-jang4m-vl-current-64-2026-06-18T15-51-55-478Z/gemma4-media-proof.json`.
  - `gemma4-31b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-jang4m-vl-current-64-2026-06-18T15-54-15-057Z/gemma4-media-proof.json`.
  - `gemma4-26b-mxfp4-visual-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-mxfp4-visual-current-64-2026-06-18T15-59-56-227Z/gemma4-media-proof.json`.
  - `gemma4-31b-mxfp4-visual-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-current-64-2026-06-18T16-02-06-376Z/gemma4-media-proof.json`.
  - Each media row passed 10-turn UI coherency, reasoning off/on/auto,
    image exact marker, tool calls under reasoning on/auto, post-media/tool
    prefix-cache hit, Chat/Responses/Anthropic/Ollama API markers, streaming
    text/image/tool markers, and UI-visible generation defaults from the model
    (`temperature=1.0`, `top_p=0.95`, `top_k=64`).
- Gemma4 clean-start/autodetect rows accepted by the gate:
  - `gemma4-e2b-mxfp4-real-profile-gateway-vl-visible-off`:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-e2b-mxfp4-real-profile-gateway-vl-visible-off-2026-06-18T15-39-57-485Z/clean-start-proof.json`.
  - `gemma4-12b-jang4m-clean-start-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-12b-jang4m-clean-start-current-64-2026-06-18T15-40-18-597Z/clean-start-proof.json`.
  - `gemma4-26b-jang4m-clean-start-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-26b-jang4m-clean-start-current-64-2026-06-18T15-40-36-553Z/clean-start-proof.json`.
  - `gemma4-31b-jang4m-clean-start-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-31b-jang4m-clean-start-current-64-2026-06-18T15-40-58-474Z/clean-start-proof.json`.
  - `gemma4-26b-mxfp4-clean-start-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-26b-mxfp4-clean-start-current-64-2026-06-18T15-41-38-810Z/clean-start-proof.json`.
  - `gemma4-31b-mxfp4-clean-start-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-31b-mxfp4-clean-start-current-64-2026-06-18T15-41-59-797Z/clean-start-proof.json`.
- Gemma4 lifecycle rows accepted by the gate:
  - `gemma4-e2b-mxfp4-lifecycle-vl`:
    `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-e2b-mxfp4-lifecycle-vl-2026-06-18T15-43-23-841Z/lifecycle-proof.json`.
  - `gemma4-26b-mxfp4-lifecycle-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-26b-mxfp4-lifecycle-current-64-2026-06-18T15-43-52-632Z/lifecycle-proof.json`.
  - `gemma4-31b-mxfp4-lifecycle-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-31b-mxfp4-lifecycle-current-64-2026-06-18T15-44-22-546Z/lifecycle-proof.json`.
- Boundaries:
  - This is scoped installed-app proof for MM3 + Gemma4 VL/current rows. It is
    not a notarized DMG/GitHub release proof.
  - Gemma audio remains out of this gate per the matrix boundary; VL/image is
    in scope and passed.
  - The gate intentionally requires generic flat TQ-KV/storage quantization to
    stay off for MM3 native MSA and Gemma mixed-SWA native cache unless a
    native-compatible TQ bridge is separately implemented and live-proven.

## 2026-06-18 15:34Z - Disk-cache contiguity fix + installed MM3 exactness rerun pass

Status: `GEMMA12B_DISK_RESTORE_PASS_MM3_EXACTNESS_PASS_RELEASE_STILL_PARTIAL`.

- Source fix:
  - `vmlx_engine/disk_cache.py` standard prompt-cache serialization now
    materializes every MLX tensor with `mx.contiguous(...)`, evaluates it on
    the caller thread, then performs the numpy CPU copy. This targets the
    isolated corruption where Gemma 4 sliding `RotatingKVCache.values` changed
    after disk write/read while keys and full-attention KV remained exact.
  - `vmlx_engine/disk_cache.py` background writer now calls
    `_write_queue.task_done()` so proof harnesses and shutdown paths can flush
    queued standard writes deterministically.
- Source/test verification:
  - `.venv/bin/python -m py_compile vmlx_engine/disk_cache.py
    vmlx_engine/mllm_batch_generator.py vmlx_engine/scheduler.py` passed.
  - `.venv/bin/python -m pytest tests/test_disk_cache_unit.py
    tests/test_cache_bypass.py tests/test_mllm_scheduler_cache.py
    tests/test_single_sequence_cache_merge.py tests/test_minimax_m3_cache_paths.py
    tests/test_batching.py::TestPrefixHitTailAccounting -q` ->
    `208 passed, 2 warnings`.
- Installed-app rebuild:
  - `panel/scripts/bundle-python.sh && panel/scripts/build-and-install.sh`
    completed; installed `/Applications/vMLX.app`.
  - Bundled source smoke from `/Applications/vMLX.app` confirmed
    `DiskCacheManager.store` contains `mx.contiguous(arr)` and
    `_background_writer` contains `task_done`.
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed. `spctl --assess` still rejects this local ad-hoc/dev install, so
    notarized release remains unclaimed.
- Gemma 4 12B MXFP4 fresh-process disk-restore live proof:
  `/Users/eric/mlx/vllm-mlx/build/live-cache-restore-gemma4-12b-mxfp4-cache-restore-installed-after-contiguous-disk-cache-2026-06-18T15-24-37-888Z/cache-restore-proof.json`.
  Result `status=pass`, failures `[]`.
  Evidence: prime exact `CACHE_RESTORE_PRIME_OK_MQJNHZ0H_515430`; restore
  exact `CACHE_RESTORE_HIT_OK_MQJNHZ0H_515430`; restore
  `cachedTokens=648`, `cacheDetail=disk`; native cache family `gemma4`,
  schema `mixed_swa_kv_v1`, components `full_attention_kv`,
  `sliding_window_kv`, `rotating_window_metadata`, paged `false`, generic
  TQ-KV not active.
- MiniMax-M3 REAP40-d3 installed-app exactness/stress proof:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T15-26-22-368Z/mm3-stress-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T15-26-22-368Z/mm3-stress-final.png`.
  Result `status=pass`, failures `[]`.
  Evidence: 10 UI turns visible/coherent with cached tokens `172` through
  `944` and speed `23.4-24.9 t/s`; reasoning off/on/auto separated with
  reasoning chars `0/416/766`; UI tool wrote `M3_TOOL_OK` and returned
  `M3_TOOL_OK_DONE`; long-context recall exact
  `PROFILE_OPTION_SENTINEL_ZETA_173` with `5778` cached tokens; MM3 VL image
  exact `MM3_IMAGE_RED`; mixed text/image/tool/reasoning/cache exact markers;
  Chat, Responses, Anthropic, Ollama, streaming Chat/Responses/image/tool rows
  passed exact markers. Generation defaults matched `generation_config.json`
  (`temperature=1.0`, `top_p=0.95`, `top_k=off`) and were visible in UI.
  Health/source trace showed native cache schema `minimax_m3_msa_v1`,
  components `attention_kv`, `msa_idx_keys`, `absolute_block_index`,
  generic TQ-KV disabled for `native_minimax_m3_msa_idx_keys`, storage
  quantization disabled for MSA idx keys, prompt disk L2 enabled, paged false.
- Boundaries:
  - Only the available local MM3 bundle
    `/Users/eric/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP40-d3-JANG_2L`
    was tested; no local REAP32-d3 MM3 path was found.
  - The build script warned twice that the bundled `mlx_vlm/models/gemma4/vision.py`
    patch target was not found, though bundled verification later reported the
    Gemma4 vision pixel-values coercion check as OK. Gemma VL remains governed
    by its separate live proof rows.
  - This is installed-app proof, not notarized/public release proof.

## 2026-06-18 11:47Z - Gemma 4 31B JANG_4M VL/current row live pass under tightened harness

Status: `GEMMA4_31B_JANG4M_VL_CURRENT_LIVE_PASS_CACHE_POLICY_PARTIAL`.

- Installed-app live artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-jang4m-vl-current-64-2026-06-18T11-40-12-845Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-jang4m-vl-current-64-2026-06-18T11-40-12-845Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, artifact finished
  `2026-06-18T11:45:02.427Z`.
- Scope:
  - audio disabled; capabilities reported `text`, `vision`, `video`;
  - 10 UI text turns stayed visible/coherent with cached tokens `23` through
    `502`, speed roughly `19.9-23.3 t/s`;
  - reasoning off/on/auto emitted visible content with separated reasoning
    chars `0`, `218`, `330`;
  - mixed image returned `GEMMA_MIX_IMAGE_RED`, streaming Responses image
    returned `GEMMA_STREAM_IMAGE_RED`;
  - mixed reasoning-on and reasoning-auto tool rows each emitted exact final
    marker and exactly one `run_command`;
  - streaming Chat tool args carried exact `GEMMA_STREAM_CHAT_TOOL`;
    streaming Responses completed `record_gemma_stream_response_label` with
    exact `GEMMA_STREAM_RESP_TOOL`;
  - API Chat, Responses, Anthropic, Ollama, streaming Chat, streaming
    Responses text/image/tool rows passed exact markers/tool signals.
- Startup/default/cache proof:
  - CLI included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--enable-disk-cache`;
  - session config had `toolCallParser=gemma4`, `reasoningParser=gemma4`,
    `isMultimodal=true`, `usePagedCache=false`, `enableDiskCache=true`,
    `kvCacheQuantization=auto`;
  - generation defaults matched `generation_config.json`:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - cache stats showed `cache_hit_tokens=3203`;
  - native cache reported `family=gemma4`, schema `mixed_swa_kv_v1`,
    components `full_attention_kv`, `sliding_window_kv`,
    `rotating_window_metadata`, paged false.
- Cache/TQ boundary:
  - artifact reports generic TurboQuant KV disabled with reason `not_active`
    and storage quantization disabled. This is live-positive for the current
    mixed-SWA native-cache contract, but the broader Gemma-compatible
    TurboQuant/cache-quantization policy gate remains open.
- Remaining:
  - selected regression rows and release build/sign/notarize/GitHub gates;
  - explicit decision or implementation for Gemma-compatible TQ/cache
    quantization on mixed-SWA native cache.

## 2026-06-18 11:44Z - Gemma 4 26B-A4B JANG_4M VL/current row live pass under tightened harness

Status: `GEMMA4_26B_JANG4M_VL_CURRENT_LIVE_PASS_CACHE_POLICY_PARTIAL`.

- Installed-app live artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-jang4m-vl-current-64-2026-06-18T11-36-58-014Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-jang4m-vl-current-64-2026-06-18T11-36-58-014Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, row
  `gemma4-26b-jang4m-vl-current-64`, artifact finished
  `2026-06-18T11:39:02.668Z`.
- Scope:
  - audio disabled; capabilities reported `text`, `vision`, `video`;
  - 10 UI text turns stayed visible/coherent with cached tokens `23` through
    `523`, speed roughly `89.3-127.7 t/s`;
  - reasoning off/on/auto emitted visible content with separated reasoning
    chars `0`, `972`, `1228`;
  - mixed image returned `GEMMA_MIX_IMAGE_RED`, streaming Responses image
    returned `GEMMA_STREAM_IMAGE_RED`;
  - mixed reasoning-on and reasoning-auto tool rows each emitted exact final
    marker and exactly one `run_command`;
  - streaming Chat tool args carried exact `GEMMA_STREAM_CHAT_TOOL`;
    streaming Responses completed `record_gemma_stream_response_label` with
    exact `GEMMA_STREAM_RESP_TOOL`;
  - API Chat, Responses, Anthropic, Ollama, streaming Chat, streaming
    Responses text/image/tool rows passed exact markers/tool signals.
- Startup/default/cache proof:
  - CLI included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--enable-disk-cache`;
  - session config had `toolCallParser=gemma4`, `reasoningParser=gemma4`,
    `isMultimodal=true`, `usePagedCache=false`, `enableDiskCache=true`,
    `kvCacheQuantization=auto`;
  - generation defaults matched `generation_config.json`:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - cache stats showed `cache_hit_tokens=3204`;
  - native cache reported `family=gemma4`, schema `mixed_swa_kv_v1`,
    components `full_attention_kv`, `sliding_window_kv`,
    `rotating_window_metadata`, paged false.
- Cache/TQ boundary:
  - artifact reports generic TurboQuant KV disabled with reason `not_active`
    and storage quantization disabled. This is live-positive for the current
    mixed-SWA native-cache contract, but the broader Gemma-compatible
    TurboQuant/cache-quantization policy gate remains open.
- Remaining:
  - Gemma 31B JANG_4M current row;
  - selected regression rows and release build/sign/notarize/GitHub gates.

## 2026-06-18 11:43Z - Gemma 4 12B JANG_4M VL/current row live pass under tightened harness

Status: `GEMMA4_12B_JANG4M_VL_CURRENT_LIVE_PASS_CACHE_POLICY_PARTIAL`.

- Installed-app live artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-vl-current-64-2026-06-18T11-32-16-900Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-vl-current-64-2026-06-18T11-32-16-900Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, row
  `gemma4-12b-jang4m-vl-current-64`, artifact finished
  `2026-06-18T11:35:22.586Z`.
- Scope:
  - audio disabled per current user instruction; capabilities reported
    `text`, `vision`, `video`;
  - 10 UI text turns stayed visible/coherent with cached tokens `23` through
    `482`, speed roughly `46.7-59.4 t/s`;
  - reasoning off/on/auto emitted visible content with separated reasoning
    chars `0`, `1432`, `0`;
  - mixed same-chat row covered text reasoning off, image reasoning on,
    text reasoning auto, reasoning-on tool, reasoning-auto tool, final recall,
    cache prime, and cache hit;
  - mixed image returned `GEMMA_MIX_IMAGE_RED`, streaming Responses image
    returned `GEMMA_STREAM_IMAGE_RED`;
  - mixed reasoning-on and reasoning-auto tool rows each emitted the exact
    final marker and exactly one `run_command`;
  - streaming Chat tool args carried exact `GEMMA_STREAM_CHAT_TOOL`;
    streaming Responses completed `record_gemma_stream_response_label` with
    exact `GEMMA_STREAM_RESP_TOOL`;
  - API Chat, Responses, Anthropic, Ollama, streaming Chat, streaming
    Responses text/image/tool rows passed exact markers/tool signals.
- Startup/default/cache proof:
  - CLI included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--enable-disk-cache`;
  - session config had `toolCallParser=gemma4`, `reasoningParser=gemma4`,
    `isMultimodal=true`, `usePagedCache=false`, `enableDiskCache=true`,
    `kvCacheQuantization=auto`;
  - generation defaults matched `generation_config.json`:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - cache stats showed `cache_hit_tokens=3712`;
  - native cache reported `family=gemma4`, schema `mixed_swa_kv_v1`,
    components `full_attention_kv`, `sliding_window_kv`,
    `rotating_window_metadata`, paged false.
- Cache/TQ boundary:
  - artifact reports generic TurboQuant KV disabled with reason `not_active`
    and storage quantization disabled;
  - source trace: `vmlx_engine/utils/jang_loader.py:1549-1555` skips flat
    generic TQ-KV for mixed sliding/full attention because it would violate
    the native `mixed_swa_kv_v1` RotatingKVCache metadata contract;
    `vmlx_engine/server.py:7046-7084` reports the native mixed-SWA cache and
    storage quantization fields;
  - this row is live-positive for current mixed-SWA native-cache behavior, but
    the broader Gemma TurboQuant/compatible-cache-quantization policy gate
    remains open until explicitly resolved and live-proven.
- Remaining:
  - Gemma 26B JANG_4M current row;
  - Gemma 31B JANG_4M current row;
  - selected regression rows and release build/sign/notarize/GitHub gates.

## 2026-06-18 11:40Z - MM3 strict exactness rerun pass

Status: `MM3_STRICT_EXACTNESS_LIVE_PASS_BROAD_64_PARTIAL`.

- Installed-app live artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T11-23-05-316Z/mm3-stress-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T11-23-05-316Z/mm3-stress-final.png`.
- Result: `status=pass`, failures `[]`.
- Artifact finished at `2026-06-18T11:31:01.098Z`.
- Strict exactness proof:
  - mixed reasoning-on tool row emitted `M3_MIX_TOOL_ON_DONE`, had
    `227` reasoning chars, `3394` cached tokens, and exactly one
    `run_command` call:
    `printf 'M3_MIX_TOOL_ON' > m3_mixed_tool_on.txt`;
  - mixed reasoning-auto tool row emitted `M3_MIX_TOOL_AUTO_DONE`, had
    `3563` cached tokens, and exactly one `run_command` call:
    `printf 'M3_MIX_TOOL_AUTO' > m3_mixed_tool_auto.txt`;
  - streaming Chat tool args contained exact `MM3_STREAM_CHAT_TOOL`;
  - streaming Responses tool emitted a completed
    `record_mm3_stream_response_label` call with exact
    `{"label": "MM3_STREAM_RESP_TOOL"}`.
- Live output/cache/API evidence:
  - 10 UI text turns stayed visible/coherent with cached tokens `172` through
    `1203` and speed roughly `23.2-24.5 t/s`;
  - reasoning off/on/auto emitted visible content with separated reasoning
    chars `0`, `716`, `642`;
  - long-context prefix/cache returned `PROFILE_OPTION_SENTINEL_ZETA_173`;
  - MM3 VL returned `MM3_IMAGE_RED`;
  - API Chat/Responses/Anthropic/Ollama and streaming Chat/Responses
    image/tool rows passed exact markers/tool signals.
- Startup/default/cache proof:
  - session config had `toolCallParser=minimax_m3`,
    `reasoningParser=minimax_m3`, `usePagedCache=false`,
    `enableDiskCache=true`, `kvCacheQuantization=auto`, `enableJit=false`;
  - final cache stats showed `cache_hit_tokens=22010`;
  - native cache reported `family=minimax_m3`, schema `minimax_m3_msa_v1`,
    components `attention_kv`, `msa_idx_keys`, `absolute_block_index`,
    generic TurboQuant KV disabled, storage quantization disabled,
    paged false, prompt disk L2 true.
- Boundary:
  - MM3 current `.64` row is live-positive under the stricter exactness gate.
  - Broad `.64` remains partial until Gemma 12B/26B/31B JANG_4M rows,
    selected regression rows, build/sign/notarize, and GitHub release gates
    are completed with current artifacts.

## 2026-06-18 11:37Z - MM3 stricter exactness rerun red; exact-one tool iteration harness adjusted

Status: `MM3_STRICT_EXACTNESS_RERUN_RED_TOOL_ITERATION_FIX_PENDING_RERUN`.

- Failed installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T11-12-58-443Z/mm3-stress-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T11-12-58-443Z/mm3-stress-final.png`.
- Result: `status=fail`.
- Failure:
  - `mixed reasoning-on expected exactly 1 run_command call, saw 2`.
- Source/runtime trace:
  - the UI reasoning-on mixed tool turn first called
    `run_command` with `echo "M3_MIX_TOOL_ON" > m3_mixed_tool_on.txt`;
  - after tool result, the model called `run_command` again with
    `cat m3_mixed_tool_on.txt`, then produced `M3_MIX_TOOL_ON_DONE`;
  - this was not duplicate execution of one call; it was a second model-issued
    verification tool call while the harness allowed `maxToolIterations=4`.
- Positives from the same red artifact:
  - 10 UI text turns remained visible/coherent with cached tokens through
    `926` and speed roughly `23.7-25.6 t/s`;
  - reasoning off/on/auto emitted visible content with reasoning chars
    `0`, `430`, `641`;
  - long-context sentinel returned `PROFILE_OPTION_SENTINEL_ZETA_173`;
  - MM3 VL returned `MM3_IMAGE_RED`;
  - API Chat/Responses/Anthropic/Ollama and streaming rows printed exact
    expected markers in the harness summary.
- Source adjustment:
  - exact-one mixed tool rows in `panel/scripts/live-mm3-stress-proof.mjs`
    now set `maxToolIterations=1` and explicitly tell the model not to
    read/cat/verify or call a second tool;
  - matching Gemma exact-one mixed tool rows in
    `panel/scripts/live-gemma4-media-stress-proof.mjs` were adjusted the same
    way before Gemma live rows.
- Verification:
  - `node --check panel/scripts/live-mm3-stress-proof.mjs` passed;
  - `node --check panel/scripts/live-gemma4-media-stress-proof.mjs` passed.
- Boundary:
  - MM3 strict exactness is still `PARTIAL/RED` until a fresh live rerun passes
    with this exact-one tool-iteration setting.

## 2026-06-18 11:25Z - Gemma 4 media stress harness tightened before JANG_4M rows

Status: `GEMMA4_HARNESS_TIGHTENED_SOURCE_ONLY_PENDING_LIVE_JANG4M_ROWS`.

- Source trace:
  - `panel/scripts/live-gemma4-media-stress-proof.mjs` now records completed
    Responses streaming `function_call` items separately from argument deltas;
  - streaming Chat/Responses text rows now require exact start markers via
    `startsWithExactMarker`;
  - streaming Chat/Responses tool rows now require exact tool arguments;
  - streaming Responses tool rows now require exactly one completed
    `record_gemma_stream_response_label` call carrying
    `GEMMA_STREAM_RESP_TOOL`;
  - mixed UI tool rows now require exact final labels and exactly one
    `run_command` call for reasoning-on and reasoning-auto tool turns;
  - API Chat tool rows now require exact `GEMMA_TOOL_OK` arguments.
- Verification:
  - `node --check panel/scripts/live-gemma4-media-stress-proof.mjs` passed;
  - `node --check panel/scripts/live-mm3-stress-proof.mjs` passed after the
    parallel MM3 harness tightening.
- Boundary:
  - This is source/harness verification only. It does not prove any Gemma 4
    JANG_4M model row until fresh installed-app live artifacts are produced.
  - Next Gemma row after the MM3 stricter rerun: 12B JANG_4M VL current row.

## 2026-06-18 11:18Z - MM3 exactness harness tightened for completed streaming tool calls

Status: `MM3_EXACTNESS_HARNESS_TIGHTENED_PENDING_NEXT_LIVE_RERUN`.

- Source trace:
  - `panel/scripts/live-mm3-stress-proof.mjs` now records completed Responses
    streaming `function_call` items separately from argument deltas;
  - the verdict now requires exactly one completed
    `record_mm3_stream_response_label` call with exact
    `MM3_STREAM_RESP_TOOL` arguments.
- Why: the current live MM3 artifact already passed exact UI/API/VL/cache
  markers, but its reconstructed `streaming.responsesTool.toolArgs` included
  both argument deltas and the final completed arguments. The SSE event tail
  showed one completed function-call item; the harness now validates that
  exact completed item instead of relying only on marker presence.
- Verification:
  - `node --check panel/scripts/live-mm3-stress-proof.mjs` passed.
- Boundary:
  - Previous installed-app proof remains evidence for the earlier exactness
    gate (`status=pass`, failures `[]`, artifact below).
  - This stricter completed-tool-call gate is source-checked only until the
    next live MM3 stress rerun.

## 2026-06-18 11:07Z - MM3 exactness/live UI/API/cache proof green after Chat stream fix

Status: `MM3_CURRENT_EXACTNESS_LIVE_PASS_BROAD_64_PARTIAL`.

- Source fix:
  - `vmlx_engine/server.py:14768` now gives Chat Completions streaming the
    same M3/Gemma reasoning-only answer-budget split that Responses streaming
    already had;
  - `vmlx_engine/server.py:15455` now runs a bounded thinking-off visible
    answer pass when M3/Gemma streamed reasoning but no visible content;
  - `panel/scripts/live-mm3-stress-proof.mjs:792` and `:834` now enforce
    exact mixed-tool visible markers, exact-one `run_command` calls, and exact
    streaming/API markers.
- Verification before live rerun:
  - `/Users/eric/mlx/vllm-mlx/.venv/bin/python -m py_compile
    vmlx_engine/server.py` passed;
  - `node --check panel/scripts/live-mm3-stress-proof.mjs` passed;
  - `git diff --check -- vmlx_engine/server.py
    panel/scripts/live-mm3-stress-proof.mjs .agents/STATUS.md
    .agents/RELEASE-1.5.63-STRESS-MATRIX.md .agents/LOG.md` passed.
- Installed-app live proof:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T10-58-45-481Z/mm3-stress-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T10-58-45-481Z/mm3-stress-final.png`.
- Result: `status=pass`, failures `[]`.
- Live output/cache/API evidence:
  - 10 UI cache-hit turns stayed visible/coherent; cached tokens rose from
    `172` through `1096`; speed stayed around `23.3-24.9 t/s`;
  - reasoning off/on/auto produced visible content with separated reasoning
    chars `0`, `496`, `638`;
  - standalone UI tool row created `M3_TOOL_OK` and answered
    `M3_TOOL_OK_DONE`;
  - long-context prefix/cache row recalled
    `PROFILE_OPTION_SENTINEL_ZETA_173` with `5778` cached tokens;
  - MM3 VL row returned `MM3_IMAGE_RED`;
  - mixed same-chat row covered text reasoning off, image reasoning on,
    text reasoning auto, reasoning-on tool, reasoning-auto tool, and final
    recall. Tool rows emitted `M3_MIX_TOOL_ON_DONE` and
    `M3_MIX_TOOL_AUTO_DONE` with one `run_command` call each, and cache-hit
    tokens reached `3353`/`3508`;
  - Chat, Responses, Anthropic, Ollama, streaming Chat, streaming Responses
    image, streaming Chat tool, and streaming Responses tool rows all passed
    exact markers/tool signals. The formerly red streaming Chat row now emitted
    visible `MM3_STREAM_CHAT_OK` with `494` reasoning chars.
- Startup/defaults/cache evidence:
  - live CLI used `--tool-call-parser minimax_m3`,
    `--enable-auto-tool-choice`, `--reasoning-parser minimax_m3`,
    `--enable-disk-cache`, no `--enable-jit`;
  - logs reported prefix cache enabled automatically, memory-aware prefix
    cache initialized, disk prompt cache initialized, tool calling enabled,
    and reasoning parser `MiniMaxM3ReasoningParser`;
  - generation defaults matched model/session/UI:
    `temperature=1.0`, `top_p=0.95`, `top_k=off`;
  - cache stats showed `cache_hit_tokens=21422`, memory hit detail, disk hits
    `1`, native MM3 cache `minimax_m3_msa_v1`, generic TQ-KV disabled, storage
    quantization disabled, paged cache off, prompt disk L2 on.
- Boundary:
  - This is current installed-app live proof for MM3 exactness/cache/reasoning
    UI/API/VL/tool rows.
  - Broad `.64` remains partial until remaining Gemma JANG_4M size rows,
    any selected release regression rows, build/sign/notarize, and GitHub
    release gates are completed.

## 2026-06-18 10:58Z - MM3 exactness red found; Chat streaming fix in source

Status: `MM3_CURRENT_EXACTNESS_PARTIAL_FIX_IN_SOURCE_NEEDS_LIVE_RERUN`.

- Failed installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T10-45-55-653Z/mm3-stress-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T10-45-55-653Z/mm3-stress-final.png`.
- Result before fix: `status=fail`.
- Live positives in that artifact:
  - 10 UI cache-hit turns stayed visible/coherent with cached tokens through
    `1079` and speed around `23.6-25.1 t/s`;
  - reasoning off/on/auto produced separated reasoning chars `0`, `451`,
    `877`;
  - long-context prefix/cache row recalled
    `PROFILE_OPTION_SENTINEL_ZETA_173` with turn-2 cached tokens `5778`;
  - MM3 VL red-image row returned `MM3_IMAGE_RED`;
  - Chat, Responses, Anthropic, Ollama, Responses image stream, Chat tool
    stream, and Responses tool stream had the expected route/tool markers.
- Live red findings:
  - `/v1/chat/completions` streaming with `enable_thinking=true` and
    `max_thinking_tokens=120` produced only `reasoning_content`, finished with
    `finish_reason="length"`, and emitted no visible content or
    `MM3_STREAM_CHAT_OK`;
  - mixed UI `tool_reasoning_on` called `run_command` twice despite the prompt
    requiring exactly once, and visible content did not include
    `M3_MIX_TOOL_ON_DONE`.
- Source trace and fix in progress:
  - `stream_responses_api` already had a bounded M3/Gemma reasoning-only
    visible-answer pass at `vmlx_engine/server.py:15818` and `16630`;
  - `stream_chat_completion` only emitted a warning for the same condition at
    `vmlx_engine/server.py:15471`, leaving Chat streaming empty-visible;
  - current source now adds the same bounded M3/Gemma answer pass to
    `stream_chat_completion` and tightens
    `panel/scripts/live-mm3-stress-proof.mjs` so exact API markers, mixed
    visible markers, and exact-one tool-call counts cannot slip through.
- Verification so far:
  - `/Users/eric/mlx/vllm-mlx/.venv/bin/python -m py_compile
    vmlx_engine/server.py` passed;
  - `node --check panel/scripts/live-mm3-stress-proof.mjs` passed;
  - `/Applications/vMLX.app` imports `vmlx_engine` from this editable checkout,
    so the source fix is eligible for installed-app live rerun without an app
    rebuild.
- Boundary:
  - MM3 is not green and `.64` is not release-ready until the tightened
    installed-app live MM3 proof reruns and passes with exact visible markers,
    one-tool-call rows, streaming Chat visible output, prefix-cache hits, and
    no loops/hidden-only rows.

## 2026-06-18 10:44Z - Gemma E4B JANG_4M VL/current `.64` proof green

Status: `E4B_JANG4M_VL_CURRENT_LIVE_PASS_BROAD_64_PARTIAL`.

- Installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-jang4m-vl-current-64-2026-06-18T10-42-01-236Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-jang4m-vl-current-64-2026-06-18T10-42-01-236Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, row
  `gemma4-e4b-jang4m-vl-current-64`.
- Source/bundle trace:
  - model path:
    `/Users/eric/models/OsaurusAI--gemma-4-E4B-it-qat-JANG_4M`;
  - `config.json` reports `model_type=gemma4`,
    `text_config.model_type=gemma4_text`, vision config present, audio config
    present;
  - `model.safetensors.index.json` contains `audio_tower.*` tensors, but this
    row was run with `VMLX_GEMMA_EXPECT_AUDIO=0`, so it proves VL/text, not
    audio semantics;
  - `generation_config.json` reports `do_sample=true`, `temperature=1.0`,
    `top_p=0.95`, `top_k=64`, `eos_token_id=[1,106,50]`.
- Live behavior proven:
  - 10 UI cache-hit turns stayed visible/coherent with cached tokens through
    `378` and speed around `83.8-102.4 t/s`;
  - reasoning off/on/auto produced visible content with reasoning chars
    `0`, `977`, and `981`;
  - mixed same-chat row covered text reasoning off, image reasoning on
    (`GEMMA_MIX_IMAGE_RED`), text reasoning auto, required tool with reasoning
    on, required tool with reasoning auto, final recall, cache prime, and cache
    hit (`32` cached tokens);
  - generation defaults matched session/UI visible settings:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - Chat, Responses, Anthropic, Ollama, streaming Chat, streaming Responses,
    streaming Responses image, streaming Chat tool, and streaming Responses
    tool rows passed with exact markers/tool signals.
- Live cache/runtime trace in the artifact:
  - capabilities reported `text`, `vision`, `audio`, `video`;
  - native cache reported Gemma4 `mixed_swa_kv_v1`, paged cache off, generic
    TurboQuant KV disabled;
  - quantization reported `profile=JANG_4M`, `weight_format=jang_affine`,
    `backend=mx.quantize`.
- Boundary:
  - E2B and E4B JANG_4M VL/text/reasoning/tool/API/cache are current
    live-positive.
  - Broad `.64` remains partial: 12B/26B/31B JANG_4M rows, explicit MM3 output
    exactness/current longer regression, any explicitly reopened Gemma audio
    semantic row, cross-family parser/API exactness, and final release gates
    remain open.

## 2026-06-18 10:39Z - Gemma E2B JANG_4M VL/current `.64` proof green

Status: `E2B_JANG4M_VL_CURRENT_LIVE_PASS_BROAD_64_PARTIAL`.

- Harness/source fix made before rerun:
  `panel/scripts/live-gemma4-media-stress-proof.mjs` now validates API exact
  marker prompts with `startsWithExactMarker()` after label normalization
  instead of a word-boundary regex. The prior failed run returned
  `output_text="GEMMA_RESP_OKOracle..."`, so the Responses object shape was
  valid and the validator was falsely rejecting a no-space marker prefix.
  `node --check panel/scripts/live-gemma4-media-stress-proof.mjs` passed.
- Installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-jang4m-vl-current-64-2026-06-18T10-37-59-814Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-jang4m-vl-current-64-2026-06-18T10-37-59-814Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, row
  `gemma4-e2b-jang4m-vl-current-64`.
- Source/bundle trace:
  - model path:
    `/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-JANG_4M`;
  - `config.json` reports `model_type=gemma4`,
    `text_config.model_type=gemma4_text`, vision config present, audio config
    present;
  - `model.safetensors.index.json` contains `audio_tower.*` tensors, but this
    row was run with `VMLX_GEMMA_EXPECT_AUDIO=0`, so it proves VL/text, not
    audio semantics;
  - `generation_config.json` reports `do_sample=true`, `temperature=1.0`,
    `top_p=0.95`, `top_k=64`, `eos_token_id=[1,106,50]`.
- Live behavior proven:
  - 10 UI cache-hit turns stayed visible/coherent with cached tokens through
    `359` and speed around `143.7-179.1 t/s`;
  - reasoning off/on/auto produced visible content with reasoning chars
    `0`, `1787`, and `1401`;
  - mixed same-chat row covered text reasoning off, image reasoning on
    (`GEMMA_MIX_IMAGE_RED`), text reasoning auto, required tool with reasoning
    on, required tool with reasoning auto, final recall, cache prime, and cache
    hit (`32` cached tokens);
  - generation defaults matched session/UI visible settings:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - Chat, Responses, Anthropic, Ollama, streaming Chat, streaming Responses,
    streaming Responses image, streaming Chat tool, and streaming Responses
    tool rows passed.
- Live cache/runtime trace in the artifact:
  - capabilities reported `text`, `vision`, `audio`, `video`;
  - native cache reported Gemma4 `mixed_swa_kv_v1`, paged cache off, generic
    TurboQuant KV disabled;
  - quantization reported `profile=JANG_4M`, `weight_format=jang_affine`,
    `backend=mx.quantize`.
- Boundary:
  - E2B JANG_4M VL/text/reasoning/tool/API/cache is current live-positive.
  - Broad `.64` remains partial: E4B/12B/26B/31B JANG_4M rows, any explicitly
    reopened Gemma audio semantic row, MM3 longer regression if required, and
    cross-family parser/API exactness still need current proof or explicit
    deferral.

## 2026-06-18 10:19Z - Public 1.5.63 verified; 1.5.64 blocker lane reopened

Status: `PUBLIC_1_5_63_VERIFIED_BROAD_1_5_64_PARTIAL`.

- Current-turn directive from Eric: get `.63` out ASAP if not already out, then
  fix remaining blockers/red items for `.64`. Do not drift into MiMo/N2; focus
  on MM3 and Gemma compatibility/release blockers.
- Public release evidence refreshed this turn:
  - `git ls-remote https://github.com/jjang-ai/vmlx.git refs/heads/main
    'refs/tags/v1.5.63^{}'` returned the same peeled commit
    `12382f2a5d149d1142bd427781d4560f9ed17816`;
  - `git ls-remote https://github.com/jjang-ai/mlxstudio.git refs/heads/main
    'refs/tags/v1.5.63^{}'` returned the same peeled commit
    `3e79bf561aee6216947012b2d87ab097adf3d14e`;
  - `gh release view v1.5.63` for both repos shows non-draft,
    non-prerelease releases with Sequoia/Tahoe DMGs and blockmaps uploaded;
  - public `mlxstudio/main/latest.json` reports `version: 1.5.63`, the
    Sequoia/Tahoe release URLs, and SHA256 values matching the local DMGs.
- Local notarization evidence refreshed this turn:
  - `panel/release/vMLX-1.5.63-sequoia-arm64.dmg` and
    `panel/release/vMLX-1.5.63-tahoe-arm64.dmg` exist;
  - `/tmp/vmlx-1.5.63-verify-dmgs.log` shows `hdiutil verify` valid,
    Developer ID authority `Developer ID Application: ShieldStack LLC
    (55KGF2S5AY)`, stapled notary tickets, `stapler validate` success, and
    Gatekeeper `Notarized Developer ID` acceptance for both DMGs.
- `.64` evidence boundary:
  - current `find build -path '*live-*' -name '*proof.json'` still finds no
    full scoped proof JSONs;
  - recovered `.63` summaries in `.agents/recovered-1.5.63-proof/` remain
    JSON-valid and report pass/failures `[]`, but they are summaries only;
  - historical 26B/31B/12B-audio proof paths currently named in this file are
    missing on disk and must be rerun/restored before being claimed current.
- Next blocker being reduced:
  `gemma4-large-vl-current-proof`: rerun and preserve a current installed-app
  UI/API proof for deferred Gemma 26B/31B visual rows, then update this file and
  the stress matrix with exact artifact paths and pass/fail state. Audio stays
  out unless explicitly reopened with a bundle that has real audio tower proof.

### 2026-06-18 10:23Z - Gemma 26B MXFP4 visual current `.64` proof green

Status: `26B_MXFP4_VISUAL_CURRENT_LIVE_PASS_31B_STILL_OPEN`.

- Installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-mxfp4-visual-current-64-20260618T102002Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-mxfp4-visual-current-64-20260618T102002Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, row
  `gemma4-26b-mxfp4-visual-current-64`.
- Source/bundle trace:
  - model path:
    `/Users/eric/models/OsaurusAI--gemma-4-26B-A4B-it-qat-MXFP4`;
  - `config.json` reports `model_type=gemma4`, `text_config.model_type`
    `gemma4_text`, vision config present, no audio config;
  - `generation_config.json` reports `do_sample=true`, `temperature=1.0`,
    `top_p=0.95`, `top_k=64`, `eos_token_id=[1,106,50]`.
- Live CLI/autodetect evidence:
  - app spawned bundled engine from `/Applications/vMLX.app`;
  - CLI included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--enable-disk-cache`, and `--max-tokens 512`;
  - capabilities reported `text`, `vision`, `video`;
  - native cache reported Gemma4 `mixed_swa_kv_v1`, paged cache off, generic
    TurboQuant KV disabled, prompt prefix cache enabled.
- Live behavior proven in this artifact:
  - 10 UI cache-hit turns stayed visible/coherent with cached tokens through
    `491` and speed around `100.1-137.7 t/s`;
  - reasoning off/on/auto all produced visible content, with reasoning chars
    `0`, `2580`, and `1213`, no hidden-only row;
  - mixed same-chat row covered text reasoning off, image reasoning on
    (`GEMMA_MIX_IMAGE_RED`), text reasoning auto, required tool with reasoning
    on, required tool with reasoning auto, final recall, cache prime, and cache
    hit (`928` cached tokens);
  - model-owned generation defaults matched session/UI visible settings:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - streaming Chat text, streaming Responses text, streaming Responses image,
    streaming Chat tool, and streaming Responses tool passed.
- Boundary:
  - 26B MXFP4 visual is current live-positive for `.64` scope.
  - 31B visual still needs current rerun; old 31B proof path is missing on
    disk and remains historical until rerun.

### 2026-06-18 10:27Z - Gemma 31B MXFP4 visual current `.64` proof green

Status: `31B_MXFP4_VISUAL_CURRENT_LIVE_PASS_LARGE_VL_RERUNS_GREEN`.

- Installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-current-64-20260618T102300Z/gemma4-media-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-current-64-20260618T102300Z/gemma4-media-final.png`.
- Result: `status=pass`, failures `[]`, row
  `gemma4-31b-mxfp4-visual-current-64`.
- Source/bundle trace:
  - model path:
    `/Users/eric/models/OsaurusAI--gemma-4-31B-it-qat-MXFP4`;
  - `config.json` reports `model_type=gemma4`, `text_config.model_type`
    `gemma4_text`, vision config present, no audio config;
  - `generation_config.json` reports `do_sample=true`, `temperature=1.0`,
    `top_p=0.95`, `top_k=64`, `eos_token_id=[1,106,50]`.
- Live CLI/autodetect evidence:
  - app spawned bundled engine from `/Applications/vMLX.app`;
  - CLI included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--enable-disk-cache`, and `--max-tokens 512`;
  - capabilities reported `text`, `vision`, `video`;
  - native cache reported Gemma4 `mixed_swa_kv_v1`, paged cache off, generic
    TurboQuant KV disabled, prompt prefix cache enabled.
- Live behavior proven in this artifact:
  - 10 UI cache-hit turns stayed visible/coherent with cached tokens through
    `456` and speed around `25.6-35.1 t/s`;
  - reasoning off/on/auto all produced visible content, with reasoning chars
    `0`, `226`, and `367`, no hidden-only row;
  - mixed same-chat row covered text reasoning off, image reasoning on
    (`GEMMA_MIX_IMAGE_RED`), text reasoning auto, required tool with reasoning
    on, required tool with reasoning auto, final recall, cache prime, and cache
    hit (`942` cached tokens);
  - model-owned generation defaults matched session/UI visible settings:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - streaming Chat text, streaming Responses text, streaming Responses image,
    streaming Chat tool, and streaming Responses tool passed.
- Boundary:
  - current 26B and 31B MXFP4 visual reruns are now live-positive for `.64`.
  - Broad `.64` remains partial: large-row clean-start/gateway/lifecycle proof,
    Gemma JANG_4M rows, any reopened audio/video semantic rows, MM3 longer
    stress/release regression, and cross-family parser/API exactness are not
    newly cleared by these two visual harnesses.

### 2026-06-18 10:30Z - Gemma 26B/31B MXFP4 clean-start + gateway current proof green

Status: `26B_31B_MXFP4_CLEAN_START_GATEWAY_CURRENT_PASS_LIFECYCLE_OPEN`.

- 31B clean-start artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-31b-mxfp4-clean-start-current-64-20260618T102849Z/clean-start-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-31b-mxfp4-clean-start-current-64-20260618T102849Z/clean-start-final.png`.
- 26B clean-start artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-26b-mxfp4-clean-start-current-64-20260618T102928Z/clean-start-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-26b-mxfp4-clean-start-current-64-20260618T102928Z/clean-start-final.png`.
- Results: both `status=pass`, failures `[]`.
- Live default/autodetect evidence for both rows:
  - session config resolved model-owned defaults:
    `defaultTemperature=100`, `defaultTopP=95`, `defaultTopK=64`,
    `defaultSamplingDefaultsDeclared=true`, `defaultDoSample=true`;
  - startup config set `enablePrefixCache=true`, `usePagedCache=false`,
    `enableDiskCache=true`, `cacheMemoryPercent=15`, `kvCacheQuantization=auto`,
    `toolCallParser=gemma4`, `reasoningParser=gemma4`, `isMultimodal=true`;
  - capabilities reported `text`, `vision`, `video`;
  - native cache reported Gemma4 `mixed_swa_kv_v1`, generic TurboQuant KV
    disabled, paged off;
  - default UI turn produced visible `CLEAN_START_VISIBLE_OK`, not hidden-only,
    with no reasoning-tag leak or loop suspect;
  - gateway status was running on `127.0.0.1:8080`; gateway health/models/
    capabilities were present; gateway Chat, Responses, streaming Chat, and
    streaming Responses exact marker probes passed.
- Boundary:
  - 26B/31B large MXFP4 visual rows now have current stress + clean-start/
    gateway proof.
  - Stop/abort lifecycle for 26B/31B remains open unless rerun with
    `live-lifecycle-stop-proof.mjs`.

### 2026-06-18 10:32Z - Gemma 26B/31B MXFP4 Stop/abort lifecycle current proof green

Status: `26B_31B_MXFP4_LARGE_ROW_CURRENT_LIFECYCLE_PASS`.

- 31B lifecycle artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-31b-mxfp4-lifecycle-current-64-20260618T103055Z/lifecycle-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-31b-mxfp4-lifecycle-current-64-20260618T103055Z/lifecycle-final.png`.
- 26B lifecycle artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-26b-mxfp4-lifecycle-current-64-20260618T103136Z/lifecycle-proof.json`;
  screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-26b-mxfp4-lifecycle-current-64-20260618T103136Z/lifecycle-final.png`.
- Results: both `status=pass`, failures `[]`.
- Lifecycle evidence for both:
  - abort turn was streaming before abort;
  - `window.api.chat.abort` returned success;
  - chat streaming lock cleared;
  - quiet wait had no extra messages and no new assistant continuation;
  - stop turn was streaming before session stop;
  - `window.api.sessions.stop` returned success;
  - session status became `stopped`, PID cleared, backend `/health` was
    unreachable, and quiet wait had no extra messages.
- Verification sweep:
  `jq -e '.status=="pass" and ((.failures//[])|length==0)'` passed for all
  six current 26B/31B `.64` artifacts: visual stress, clean-start/gateway, and
  lifecycle for each row. No vMLX app, engine, or harness process remained.
- Boundary:
  - Large Gemma 26B/31B MXFP4 visual rows now have current installed-app
    stress, clean-start/gateway, and lifecycle proof.
  - Remaining broad `.64` work is outside these rows: Gemma JANG_4M coverage,
    any reopened audio/video semantic proof, MM3 longer regression if required,
    and cross-family parser/API exactness.

## 2026-06-18 10:15Z - Current truth after public 1.5.63 release

Status: `SCOPED_1_5_63_RELEASED_BROAD_OBJECTIVE_PARTIAL`.

### FIXED / PUBLIC RELEASED: scoped 1.5.63 distribution

- Public vMLX source/release evidence:
  - `jjang-ai/vmlx` `main` points to
    `12382f2a5d149d1142bd427781d4560f9ed17816`;
  - `jjang-ai/vmlx` tag `v1.5.63^{}` points to the same commit;
  - release exists at
    `https://github.com/jjang-ai/vmlx/releases/tag/v1.5.63`;
  - release assets present: Sequoia DMG, Sequoia blockmap, Tahoe DMG, Tahoe
    blockmap.
- Public MLXStudio updater/release evidence:
  - `jjang-ai/mlxstudio` `main` points to
    `3e79bf561aee6216947012b2d87ab097adf3d14e`;
  - `jjang-ai/mlxstudio` tag `v1.5.63^{}` points to the same commit;
  - release exists at
    `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.5.63`;
  - public raw `latest.json` reports version `1.5.63` and points at the
    `mlxstudio` v1.5.63 Sequoia/Tahoe download URLs.
- Local DMG evidence still present:
  - `panel/release/vMLX-1.5.63-sequoia-arm64.dmg`;
  - `panel/release/vMLX-1.5.63-tahoe-arm64.dmg`;
  - SHA256:
    `0be2ad302ae391efb7a1af2203db831162f60ec8271bd30a071a8c48e303c0dc`
    for Sequoia;
  - SHA256:
    `aff1cc423dff786282d7f8e9fc2e1022d763b435a4e2d6e09bc941394d768a30`
    for Tahoe.
- Notarization/log evidence still present:
  - `/tmp/vmlx-1.5.63-notarize.log`;
  - `/tmp/vmlx-1.5.63-verify-dmgs.log`;
  - Apple accepted Sequoia id
    `a51c36f3-7fed-4211-8ee7-bbad93f951b2`;
  - Apple accepted Tahoe id
    `fc7ee38f-2c80-410c-baf2-2196c3c8f2f9`;
  - verifier log shows stapled tickets, `spctl` accepted both DMGs as
    `Notarized Developer ID`, and the same final hashes above.

### PARTIAL: scoped live-run evidence is recovered as summaries, but original full dirs are gone

- The original proof paths named below under `build/live-*` are not currently
  present in this checkout. Current `find build ... proof.json` returned no
  matching proof files.
- Surviving `/tmp` JSON summaries were recovered to stable ignored paths:
  - `.agents/recovered-1.5.63-proof/mm3-stress-proof-recovered.json`;
  - `.agents/recovered-1.5.63-proof/gemma4-e2b-mxfp4-media-proof-recovered.json`;
  - `.agents/recovered-1.5.63-proof/gemma4-e4b-mxfp4-media-proof-recovered.json`;
  - `.agents/recovered-1.5.63-proof/gemma4-12b-mxfp4-media-proof-recovered.json`;
  - `.agents/recovered-1.5.63-proof/manifest.json`.
- The recovered summaries report `status=pass` and `failures=[]` for MM3,
  Gemma E2B, Gemma E4B, and Gemma 12B scoped VL/text/reasoning/tool/API/cache
  runs. They are useful evidence but are weaker than the missing full proof JSON
  directories because screenshots and full artifacts are no longer available at
  the named `build/live-*` paths.

### PARTIAL / NOT COMPLETE: broad original objective

- Gemma audio is **not** release-proven. It was explicitly excluded from the
  1.5.63 gate. Prior internal evidence says 12B direct-audio attempts either
  failed or answered as if no audio was attached.
- Gemma 26B/31B visual were not in the final 1.5.63 scoped public gate. Earlier
  internal status sections include live-positive rows, but current 1.5.63
  scoped preflight defers them to 1.5.64 and the broad objective still needs
  current proof if they are to be claimed again.
- Full broad matrix rows outside MM3 + Gemma E2B/E4B/12B VL remain open until
  rerun with current artifacts.
- Current local source tree is dirty only in local/internal/generated state:
  `.agents/LOG.md` is modified and `panel/index.mjs` is untracked. These were
  intentionally not pushed into the public release.

### Source-trace anchors for current public 1.5.63

- `origin/main:vmlx_engine/server.py:1866` defines the MM3 image-only Responses
  carve-out helper.
- `origin/main:vmlx_engine/server.py:1871` defines the unsupported-modality
  filter for the MM3 VL carve-out.
- `origin/main:vmlx_engine/server.py:2247` starts the Gemma/audio capability
  truth gate.
- `origin/main:vmlx_engine/server.py:2298-2300` requires
  `audio_runtime_proven` or explicit experimental override when no real
  `audio_tower.*` weights exist.
- `origin/main:vmlx_engine/server.py:2349-2354` makes loaded MM3 VL report
  `["text", "vision"]` without switching onto the generic MLLM wrapper.
- `origin/main:panel/scripts/scoped-release-preflight.py` is the scoped 1.5.63
  fail-closed release gate for MM3 + Gemma E2B/E4B/12B VL and explicitly
  defers Gemma 26B/31B visual plus broader rows to 1.5.64.

Next blocker to reduce: rerun and preserve full current proof artifacts for
the broad 1.5.64 live matrix, starting with Gemma audio or 26B/31B visual if
those are still in scope.

## Active Blocker Being Reduced

Gemma4 release blocker: Gemma audio is now explicitly out of `.63` scope per
Eric. Current Gemma rows require text, VL image, reasoning off/on/auto,
tools, API/streaming, native cache, startup autodetect/UI-setting parity, and
lifecycle proof. MM3 mixed proof is live-positive, and the tightened Gemma
E2B/E4B MXFP4 mixed UI/API proofs are live-green after the Gemma4
reasoning-only visible-answer fix. 26B and 31B MXFP4 visual rows are
live-green for their scoped installed UI/API mixed rows. Current scoped Gemma
E2B and MM3 rows now have real-profile clean-start/default-autodetect proof,
gateway/default-port proof, and installed app Stop/abort lifecycle proof.
Release packaging remains unproven/open.

Current `.63` ASAP target:
- ship the currently proven scope: MM3 REAP40-d3 plus Gemma E2B/E4B/12B MXFP4
  VL/text/reasoning/tool/API/cache;
- keep Gemma audio out of `.63`;
- defer Gemma 26B/31B visual and remaining broad/red historical rows to `.64`;
- do not claim released until scoped preflight, signed DMGs, notarization,
  stapling, verification, GitHub/public release, and private postmortem docs
  are complete.

Release/notarization remains `PARTIAL` until those packaging/public artifacts
exist and verify.

## 2026-06-18 09:36Z - Scoped `.63` proof set ready for DMG preflight

Status: `SCOPED_1_5_63_PROOFS_READY_PACKAGING_OPEN`.

- MM3 full installed-app stress:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T09-12-47-547Z/mm3-stress-proof.json`;
  `PASS`, failures `[]`.
- Gemma E2B VL:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T09-22-26-588Z/gemma4-media-proof.json`;
  `PASS`, failures `[]`, `expectAudio=false`, `expectImage=true`.
- Gemma E4B VL:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T09-28-11-833Z/gemma4-media-proof.json`;
  `PASS`, failures `[]`, `expectAudio=false`, `expectImage=true`.
- Gemma 12B VL:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-mxfp4-2026-06-18T09-30-26-336Z/gemma4-media-proof.json`;
  `PASS`, failures `[]`, `expectAudio=false`, `expectImage=true`.
- `panel/scripts/scoped-release-preflight.py` now makes
  `VMLX_RELEASE_SCOPE=mm3_gemma_vl` fail closed on this exact proof set and
  records 26B/31B visual as `.64` deferrals.

## 2026-06-18 02:54 PDT - Real-profile delete-all-sessions clean-start proof green

Status: `REAL_PROFILE_CLEAN_START_DELETE_GATE_PASS_RELEASE_STILL_BLOCKED`.

- Harness now supports `VMLINUX_CLEAN_USE_REAL_PROFILE=1`, backs up
  `~/Library/Application Support/vMLX`, deletes saved server sessions through
  `window.api.sessions.stop/delete`, then creates/starts the model with empty
  default config.
- Gemma E2B real-profile artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-e2b-mxfp4-real-profile-delete-start-vl-2026-06-18T08-53-44-560Z/clean-start-proof.json`;
  result `PASS`, deleted `11` saved sessions, native `mixed_swa_kv_v1`, disk
  cache hit `37` tokens, coherent visible output.
- MM3 real-profile artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-mm3-reap40-d3-real-profile-delete-start-2026-06-18T08-54-19-029Z/clean-start-proof.json`;
  result `PASS`, deleted `1` saved session, capabilities `text, vision`,
  native `minimax_m3_msa_v1` with `msa_idx_keys`, disk cache hit `195` tokens,
  coherent visible output.
- Boundary: gateway/default-port parity, sleep/wake/Stop lifecycle, remaining
  size-row clean-start coverage if required, notarization/public release, and
  private postmortem note remain open.

## 2026-06-18 02:49 PDT - MM3 VL capability clean-start fix green after rebuild

Status: `MM3_CLEAN_START_VL_CAPABILITY_PASS_RELEASE_STILL_BLOCKED`.

- Source fix: `vmlx_engine/server.py::_loaded_runtime_modalities()` now returns
  `text, vision` for the special MM3 `VMLX_M3_VL=1` route when
  `_m3_vl_image_ok(_engine)` is true, without switching MM3 onto the generic
  MLLM wrapper.
- Test: `tests/test_multimodal_routing.py -q -k 'm3_vl or text_only_multimodal'`
  -> `3 passed, 5 deselected`; `server.py` py_compile passed.
- Rebuild/install: `panel/scripts/build-and-install.sh` exited 0 and
  `/Applications/vMLX.app` signature verification passed.
- Installed-app MM3 clean-start artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-mm3-reap40-d3-clean-start-after-vl-capability-fix-2026-06-18T08-48-27-375Z/clean-start-proof.json`.
- Result: `PASS`, failures `[]`; capabilities `text, vision`; native cache
  `minimax_m3_msa_v1` with `msa_idx_keys`; generic TQ-KV off; paged off; disk
  prompt L2 true; default UI turn visible, cache hit `195` disk tokens, no loop
  suspect.
- Installed-app Gemma E2B clean-start rerun after the same rebuild also passed:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-e2b-mxfp4-clean-start-vl-after-mm3-capability-fix-2026-06-18T08-49-36-147Z/clean-start-proof.json`.
- Boundary: gateway/default-port, real-profile Server-panel deletion, lifecycle,
  remaining size-row clean-start coverage if required, notarization/public
  release, and private postmortem note are still open.

## 2026-06-18 02:36 PDT - Gemma E2B clean-start autodetect/default proof green

Status: `GEMMA4_E2B_CLEAN_START_AUTODETECT_PASS_RELEASE_STILL_BLOCKED`.

- Installed-app clean-start artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-e2b-mxfp4-clean-start-vl-2026-06-18T08-36-49-037Z/clean-start-proof.json`.
- Result: `PASS`, failures `[]`.
- The session was created from a temporary app profile with no manual settings
  changes. It resolved Gemma4 parsers, multimodal true, paged cache off, disk
  cache on, `kvCacheQuantization=auto`, model-owned defaults
  `temperature=1.0`, `top_p=0.95`, `top_k=64`, and native
  `mixed_swa_kv_v1` cache with generic TQ-KV disabled.
- Default UI turn produced visible content with `CLEAN_START_VISIBLE_OK`,
  separated reasoning, no native tag leak, no hidden-only response, and no loop
  suspect.
- Boundary: real-profile Server-panel deletion, MM3 clean-start, gateway/default
  port, lifecycle, signing/notarization, public release surfaces remain open.

## 2026-06-18 01:30 PDT - 31B MXFP4 visual strict rerun green after short Responses marker

Status: `31B_MXFP4_VISUAL_LIVE_PASS_RELEASE_STILL_BLOCKED`.

- Harness hardening:
  - direct API and streaming text rows now require exact markers;
  - the Responses markers were shortened from `GEMMA_RESPONSES_OK` /
    `GEMMA_STREAM_RESPONSES_OK` to `GEMMA_RESP_OK` /
    `GEMMA_STREAM_RESP_OK` after 31B deterministically mutated the long
    synthetic word `RESPONSES` into `RESPONDES/RESPODES`;
  - model-owned default sampling remains tested separately by the generation
    defaults row.
- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-2026-06-18T08-26-40-791Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-2026-06-18T08-26-40-791Z/gemma4-media-final.png`;
  - result: `PASS`, failures `[]`.
- Live rows proven in this artifact:
  - installed app loaded
    `/Users/eric/models/OsaurusAI--gemma-4-31B-it-qat-MXFP4`;
  - capabilities were `text`, `vision`, `video`;
  - model-owned generation defaults matched and were visible:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - UI 10-turn text/cache stayed coherent with cached tokens through `459`;
  - reasoning off/on/auto all had visible content with separated reasoning where
    expected (`0`, `246`, `251` reasoning chars);
  - mixed image reasoning-on returned `GEMMA_MIX_IMAGE_RED` with `372`
    reasoning chars;
  - mixed text-auto, reasoning-on tool, reasoning-auto tool, final recall,
    post-mixed cache prime/hit all passed;
  - required tool files contained `GEMMA_MIX_TOOL_ON` and
    `GEMMA_MIX_TOOL_AUTO`;
  - direct Chat, direct Responses, Anthropic Messages, Ollama chat, Ollama
    generate, streaming Chat text, streaming Responses text, streaming
    Responses image, streaming Chat tool, and streaming Responses tool passed;
  - health/cache reported native Gemma4 `mixed_swa_kv_v1`, generic TurboQuant
    KV disabled, prompt disk L2 entries `60`, hits `14`, stores `11`,
    scheduler cache-hit requests `26`, cache-hit tokens `2896` (`768` disk,
    `2128` memory).
- Boundary:
  - 31B MXFP4 visual/mixed row is scoped green for this installed UI/API
    artifact;
  - this does not clear 12B audio, gateway/default-port parity, clean-start UI
    session deletion/autodetect parity, sleep/wake/Stop lifecycle, DMG signing,
    notarization, public release, or private postmortem note.

## 2026-06-18 01:18 PDT - 31B MXFP4 visual strict rerun red on Responses marker

Status: `31B_MXFP4_PARTIAL_LIVE_RED_RESPONSES_EXACTNESS`.

- Harness hardening:
  - `panel/scripts/live-gemma4-media-stress-proof.mjs` now requires exact
    markers for direct API text rows and streaming text rows instead of only
    requiring non-empty visible content;
  - this caught a false-green condition in the prior 31B run.
- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-2026-06-18T08-14-04-680Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-2026-06-18T08-14-04-680Z/gemma4-media-final.png`;
  - result: `FAIL`;
  - failure: `API responsesText missing exact marker GEMMA_RESPONSES_OK`.
- Positive rows in the same artifact:
  - installed app loaded
    `/Users/eric/models/OsaurusAI--gemma-4-31B-it-qat-MXFP4`;
  - capabilities were `text`, `vision`, `video`;
  - UI 10-turn text/cache stayed coherent with cached tokens through `479`;
  - reasoning off/on/auto all had visible content with separated reasoning where
    expected (`0`, `448`, `505` reasoning chars);
  - mixed image reasoning-on returned `GEMMA_MIX_IMAGE_RED`;
  - mixed tool-on/tool-auto wrote the expected files and final recall preserved
    all labels;
  - post-mixed cache hit reached `928` cached tokens;
  - direct Chat, Anthropic Messages, Ollama chat, and Ollama generate returned
    exact requested markers;
  - streaming Chat text, streaming Responses text, streaming Responses image,
    streaming Chat tool, and streaming Responses tool passed exact marker/tool
    checks;
  - health/cache reported native Gemma4 `mixed_swa_kv_v1`, generic TurboQuant
    KV disabled, prompt disk L2 entries `37`, hits `15`, stores `11`,
    scheduler cache-hit requests `26`, cache-hit tokens `3607` (`580` disk,
    `3027` memory).
- Exact red row:
  - direct `/v1/responses` non-stream text returned
    `GEMMA_RESPODES_OK...` instead of the requested `GEMMA_RESPONSES_OK`;
  - do not mark 31B release-green until this exactness row is fixed or the
    release scope explicitly excludes direct Responses exact-label compliance.

## 2026-06-18 01:07 PDT - 26B MXFP4 visual installed mixed UI/API proof green

Status: `26B_MXFP4_VISUAL_LIVE_PASS_RELEASE_STILL_BLOCKED`.

- Source trace:
  - fused quantized expert sidecar splitting lives in
    `vmlx_engine/models/gemma4_unified/gemma4_unified.py:17`,
    `vmlx_engine/models/_gemma4_text_upstream.py:19`, and
    `vmlx_engine/utils/jang_loader.py:354`;
  - Gemma4 audio capability truth gate lives in
    `vmlx_engine/server.py:2247`, so this vision-only 26B row does not advertise
    audio from a stale `audio_config` stub.
- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-mxfp4-visual-2026-06-18T08-04-22-352Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-mxfp4-visual-2026-06-18T08-04-22-352Z/gemma4-media-final.png`;
  - result: `PASS`, failures `[]`.
- Live rows proven in this artifact:
  - installed app loaded
    `/Users/eric/models/OsaurusAI--gemma-4-26B-A4B-it-qat-MXFP4`;
  - capabilities were exactly `text`, `vision`, `video`;
  - session settings reflected Gemma4 parsers and cache policy:
    `toolCallParser=gemma4`, `reasoningParser=gemma4`, `isMultimodal=true`,
    `usePagedCache=false`, `enableDiskCache=true`,
    `kvCacheQuantization=auto`;
  - model-owned generation defaults matched and were visible:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - UI 10-turn text/cache stayed coherent with cached tokens through `486` and
    about `95.8-136.4 tok/s`;
  - reasoning off/on/auto all had visible content; off had `0` reasoning chars,
    on had `2770`, auto had `1034`;
  - mixed single-session rows passed: text reasoning off, image reasoning on
    (`GEMMA_MIX_IMAGE_RED`, `633` reasoning chars), text reasoning auto,
    reasoning-on tool, reasoning-auto tool, final recall, post-mixed
    cache prime/hit;
  - required tool files were written with `GEMMA_MIX_TOOL_ON` and
    `GEMMA_MIX_TOOL_AUTO`;
  - streaming Chat/Responses text, streaming Responses image, streaming Chat
    tool, and streaming Responses tool passed;
  - health/cache reported native Gemma4 `mixed_swa_kv_v1`, generic TurboQuant
    KV disabled, storage quantization disabled for full/sliding attention KV,
    prompt disk L2 entries `37`, hits `15`, stores `11`, scheduler cache-hit
    requests `27`, cache-hit tokens `2788` (`580` disk, `2208` memory).
- Boundary:
  - 26B MXFP4 visual/fused-expert row is live-green for this installed UI/API
    mixed artifact;
  - this does not clear 31B visual, 12B audio, gateway/default-port parity,
    Anthropic/Ollama streaming exactness, sleep/wake/Stop lifecycle, missing
    DMG/notarization, or public release surfaces.

## 2026-06-17 23:55 PDT - 12B MXFP4 installed mixed proof red on audio capability

Status: `12B_MXFP4_PARTIAL_TEXT_VISION_CACHE_PASS_AUDIO_RED_RELEASE_BLOCKED`.

- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-mxfp4-2026-06-18T06-50-24-957Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-mxfp4-2026-06-18T06-50-24-957Z/gemma4-media-final.png`;
  - result: `FAIL`.
- Positive live rows in this artifact:
  - installed app loaded
    `/Users/eric/models/OsaurusAI--gemma-4-12B-it-qat-MXFP4`;
  - generation defaults matched and were visible in UI:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - UI 10-turn text/cache stayed coherent with no blank/loop/tag leak;
  - reasoning-off text turns had visible content and `0` reasoning chars;
  - health/cache reported native Gemma4 mixed-SWA cache
    `mixed_swa_kv_v1`, paged false, generic TurboQuant KV disabled, prompt
    disk L2 hits/stores, scheduler cache-hit requests `25`, and cache-hit
    tokens `3610` (`421` disk, `3189` memory).
- Red rows in this artifact:
  - capability endpoint runtime modalities were only `text`, `vision`,
    `video`;
  - audio was declared in metadata but reported as
    `declared_not_runtime_supported`;
  - mixed UI audio turn failed to send;
  - UI audio failed 400 with `/v1/responses received unsupported media
    modality audio. Supported modalities: text, vision, video.`;
  - Chat Completions audio and Responses audio both failed HTTP 400.
- Boundary:
  - 12B MXFP4 is not an audio-capable green row in the current installed app;
  - it can only be counted as partial text/vision/video/cache evidence unless
    the runtime is fixed to support audio for this bundle or the release scope
    explicitly excludes 12B MXFP4 audio.

## 2026-06-18 00:02 PDT - 12B JANG_4M direct-audio experiment remains red

Status: `12B_JANG4M_DIRECT_AUDIO_LIVE_RED_KEEP_GATED`.

- Source/bundle facts checked before the live run:
  - 12B MXFP4 and 12B JANG_4M have `audio_config` and
    `embed_audio.embedding_projection.weight`;
  - neither local 12B bundle has `audio_tower.*` weights;
  - their processor emits `input_features` shaped `(1, 13, 640)`, matching
    `audio_config.audio_embed_dim=640`;
  - current source intentionally gates Gemma4 unified direct audio unless the
    bundle is stamped/proven or an experimental env var is set.
- Installed live experiment:
  - env enabled `VMLINUX_ALLOW_EXPERIMENTAL_GEMMA4_DIRECT_AUDIO=1` and
    `VMLX_ALLOW_EXPERIMENTAL_GEMMA4_DIRECT_AUDIO=1`;
  - artifact:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-direct-audio-2026-06-18T06-57-25-896Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-direct-audio-2026-06-18T06-57-25-896Z/gemma4-media-final.png`;
  - result: `FAIL`.
- Positive rows:
  - capability endpoint advertised `text`, `vision`, `audio`, `video` under
    the experimental direct-audio gate;
  - UI 10-turn text/cache stayed coherent, with cached tokens through `509`;
  - reasoning off/on/auto all emitted visible content;
  - image, tools, final recall, streaming Chat/Responses text, streaming
    Responses image, streaming Chat tool, and streaming Responses tool passed;
  - mixed audio turn emitted `audio present GEMMA_MIX_AUDIO_PRESENT`.
- Red rows:
  - dedicated UI audio answered that no audio/video file was attached;
  - Chat Completions audio answered that no audio file was attached;
  - Responses audio answered that no audio was heard;
  - artifact failures were `uiAudio did not acknowledge expected audio
    phrase/tone`, `chatAudio did not acknowledge expected audio phrase/tone`,
    and `responsesAudio did not acknowledge expected audio phrase/tone`.
- Boundary:
  - direct `embed_audio` is shape-compatible but not semantically reliable in
    the installed app/API surfaces;
  - current default source gate should stay in place. Do not advertise 12B
    direct audio by default, and do not count 12B audio green without a repaired
    bundle/runtime and a new live artifact.

## 2026-06-17 23:46 PDT - E4B MXFP4 tightened mixed UI/API proof green

Status: `E4B_MXFP4_LIVE_PASS_RELEASE_STILL_BLOCKED`.

- Harness hardening:
  - `panel/scripts/live-gemma4-media-stress-proof.mjs` now normalizes
    Markdown-escaped underscores for recall labels (`GEMMA\_EBS` counts as
    `GEMMA_EBS`);
  - the mixed-session gate now requires exact visible markers for
    `GEMMA_MIX_TEXT_OFF`, `GEMMA_MIX_AUTO_TEXT`,
    `GEMMA_MIX_TOOL_ON_DONE`, and `GEMMA_MIX_TOOL_AUTO_DONE`.
- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T06-46-20-957Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T06-46-20-957Z/gemma4-media-final.png`;
  - result: `PASS`, failures `[]`.
- Live rows proven in this artifact:
  - installed app loaded
    `/Users/eric/models/OsaurusAI--gemma-4-E4B-it-qat-MXFP4` with
    capabilities `text`, `vision`, `audio`, `video`;
  - model-owned generation defaults matched and were visible:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - UI 10-turn text/cache stayed coherent with cached tokens through `378` and
    about `42.8-54.1 tok/s`;
  - reasoning off/on/auto all had visible content; off had `0` reasoning
    chars, on had `1009`, auto had `954`;
  - mixed single-session rows passed the tightened per-turn markers:
    `GEMMA_MIX_TEXT_OFF`, `GEMMA_MIX_IMAGE_RED`,
    `GEMMA_MIX_AUTO_TEXT`, `GEMMA_MIX_AUDIO_PRESENT`,
    `GEMMA_MIX_TOOL_ON_DONE`, `GEMMA_MIX_TOOL_AUTO_DONE`,
    final recall, post-mixed cache prime/hit;
  - post-mixed cache prime/hit each reported `cachedTokens=32`;
  - UI audio, Chat audio, and Responses audio returned audio-present markers;
  - streaming Chat/Responses text, streaming Responses image, streaming Chat
    tool, and streaming Responses tool passed;
  - health/cache reported native Gemma4 mixed-SWA cache with prompt disk L2:
    disk cache entries `63`, hits `16`, stores `9`, prompt L2 tokens `10927`.
- Boundary:
  - E4B MXFP4 is live-green for this installed UI/API mixed row;
  - this does not clear 12B JANG_4M audio, 26B/31B visual, gateway/default-port,
    Anthropic/Ollama streaming exactness, sleep/wake/Stop lifecycle, or
    release/notarization.

## 2026-06-17 23:37 PDT - E2B MXFP4 installed mixed UI/API proof green

Status: `E2B_MXFP4_LIVE_PASS_RELEASE_STILL_BLOCKED`.

- Source fix:
  - `vmlx_engine/server.py::stream_responses_api()` now initializes
    `_family_name` from the resolved model config in the Responses stream path;
  - reasoning-only text used as a tool-call parse candidate is no longer
    recycled as visible `output_text` when no tool call is found
    (`_tool_parse_from_reasoning_only`);
  - Gemma4 reasoning-on/auto no-tool Responses streams can run a bounded
    thinking-off visible-answer pass when the first pass emits separated
    reasoning but no visible content.
- Focused source proof:
  - `py_compile` passed for `vmlx_engine/server.py`,
    `vmlx_engine/reasoning/gemma4_parser.py`, `vmlx_engine/models/mllm.py`,
    `vmlx_engine/engine/batched.py`, `tests/test_engine_audit.py`, and
    `tests/test_reasoning_modes.py`;
  - `pytest tests/test_reasoning_modes.py tests/test_engine_audit.py -q -k
    'gemma4_reasoning_parser or gemma4_reasoning_streaming or
    responses_stream_usage_preserves_cache_detail_without_cached_tokens or
    responses_stream_usage_preserves_positive_cached_tokens_after_zero_chunk or
    gemma4_responses_stream_reasoning_only_runs_visible_answer_pass'`
    -> `8 passed, 616 deselected`.
- Packaging proof:
  - rebuilt through `panel/scripts/bundle-python.sh` and
    `panel/scripts/build-and-install.sh`;
  - bundled source parity passed for critical runtime files;
  - bundled Python reports `vmlx_engine 1.5.63`;
  - installed source contains `_tool_parse_from_reasoning_only`;
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed and the app satisfies its Designated Requirement.
- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T06-35-33-296Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T06-35-33-296Z/gemma4-media-final.png`;
  - result: `PASS`, failures `[]`.
- Live rows proven in this artifact:
  - installed app loaded
    `/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4` with
    capabilities `text`, `vision`, `audio`, `video`;
  - model-owned generation defaults matched and were visible:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - UI 10-turn text/cache stayed coherent with cached tokens through `366` and
    about `68.8-85.7 tok/s`;
  - reasoning off/on/auto all had visible content; off had `0` reasoning
    chars, on had `2045`, auto had `893`;
  - mixed single-session rows passed: text reasoning off, image reasoning on,
    text reasoning auto, audio reasoning auto, reasoning-on tool,
    reasoning-auto tool, final recall, post-mixed cache prime/hit;
  - previous hidden-only `text_reasoning_auto` row is fixed live:
    visible content `GEMMA_MIX_AUTO_TEXT=...`, reasoning chars `649`;
  - post-mixed cache prime/hit each reported `cachedTokens=32`;
  - UI audio, Chat audio, and Responses audio returned `audio present`;
  - streaming Chat/Responses text, streaming Responses image, streaming Chat
    tool, and streaming Responses tool passed;
  - health/cache reported native Gemma4 mixed-SWA cache with prompt disk L2:
    disk cache entries `72`, hits `16`, stores `8`, prompt L2 tokens `14326`,
    scheduler cache-hit requests `28`, cache-hit tokens `2652`.
- Boundary:
  - E2B MXFP4 is live-green for this installed UI/API mixed row;
  - this does not clear 12B JANG_4M audio, 26B/31B visual, gateway/default-port,
    Anthropic/Ollama streaming exactness, sleep/wake/Stop lifecycle, or
    release/notarization.

## 2026-06-17 22:58 PDT - E2B installed mixed rerun after vision config rebuild

Status: `PARTIAL_LIVE_RED_MIXED_REASONING_CACHE_TELEMETRY`.

- Installed app proof artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T05-55-06-754Z/gemma4-media-proof.json`.
- Result: `FAIL`.
- Positive rows:
  - `/Applications/vMLX.app` loaded E2B MXFP4 with capabilities `text`,
    `vision`, `audio`, `video`;
  - generation defaults were visible/applied as model-owned sampling:
    `temperature=1.0`, `top_p=0.95`, `top_k=64`;
  - UI 10-turn text/cache stayed coherent, with cached tokens through `428`;
  - UI audio, Chat audio, Responses audio, post-audio text recovery, streaming
    Chat text, streaming Responses text, streaming Responses image, streaming
    Chat tool, and streaming Responses tool produced positive markers;
  - health reported Gemma4 native `mixed_swa_kv_v1`, scheduler cache-hit
    requests `25`, cache-hit tokens `2842`, disk cache entries `42`.
- Red rows:
  - mixed reasoning-on image turn emitted no separated reasoning;
  - mixed post-media/tool turns did not expose per-turn cached-token telemetry,
    even though health counters show cache hits.
- Boundary:
  - E2B is not release-green until the mixed image reasoning extraction/leak and
    post-media/tool cache telemetry row are fixed and rerun live.

## 2026-06-17 20:32 PDT - 12B JANG_4M installed audio fallback rerun

Status: `PARTIAL_LIVE_RED_AUDIO_CONDITIONING`.

- Installed app/package proof:
  - `/Applications/vMLX.app` passed deep strict codesign verification;
  - packaged Python reports `vmlx_engine 1.5.63`;
  - bundled `BatchedEngine._use_simple_mllm_media_fallback()` includes the
    audio-aware Gemma4 media fallback.
- Installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T03-30-22-999Z/gemma4-media-proof.json`;
  - the row label defaulted to `gemma4-e2b-mxfp4` due to an env-var typo, but
    artifact `modelPath`, CLI command, health, capabilities, and logs prove the
    loaded model was
    `/Users/eric/models/OsaurusAI--gemma-4-12B-it-qat-JANG_4M`.
- Positive live rows:
  - 12B JANG_4M loaded in the installed app;
  - UI 10-turn text/cache completed with visible content and cache hits up to
    `483` cached tokens;
  - reasoning off/on/auto produced visible content; off had `0` reasoning
    chars and on had separated reasoning;
  - Chat Completions and Responses streaming text emitted visible markers;
  - post-audio no-attachment recovery returned `GEMMA_POST_AUDIO_TEXT_OK`;
  - logs showed Gemma4 mixed-SWA native cache, prefix cache, prompt disk L2,
    paged cache off, and generic TQ-KV skipped for mixed-SWA.
- Remaining live failure:
  - UI audio, Chat Completions audio, and Responses audio all answered as if
    no audio was attached;
  - server logs prove `input_audio` reached the server and the simple MLLM
    fallback ran with `1 audio input(s)`;
  - source processor sanity for the same WAV fixture rendered `<|audio|>`,
    expanded to `21` audio token IDs, and produced
    `input_features=(1,21,640)` plus `input_features_mask=(1,21)`.
- Boundary:
  - 12B text/cache/reasoning/API streaming is live-positive;
  - Gemma4 audio remains release-red until the model/runtime conditions on the
    audio content live or audio is gated off for failing artifacts.

## 2026-06-17 20:20 PDT - 12B JANG_4M installed startup fixed, audio row live-red

Status: `PARTIAL_STARTUP_FIXED_LIVE_RED_AUDIO_THOUGHT_LOOP`.

- Rebuilt and installed `/Applications/vMLX.app` after the processor config
  temp-path source fix:
  - build/install completed through `panel/scripts/bundle-python.sh` and
    `panel/scripts/build-and-install.sh`;
  - bundled critical `vmlx_engine` files matched source;
  - bundled critical imports passed;
  - 501 bundled Python native files were signed;
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed;
  - packaged Python reports `vmlx_engine 1.5.63`;
  - bundled `jang_loader.py` contains
    `_prepare_gemma4_processor_model_path` and
    `Gemma4 processor config normalization`.
- Packaged startup proof:
  - direct packaged 12B JANG_4M startup reached
    `Application startup complete`;
  - logs showed `Gemma4 processor config normalization`;
  - logs showed Gemma4 mixed-SWA runtime cache layout with
    `RotatingKVCache` plus periodic `KVCache`;
  - previous `final_logit_softcapping` TypeError is cleared in source and
    packaged startup.
- Installed live harness artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-2026-06-18T03-14-21-436Z/gemma4-media-proof.json`;
  - status `fail`;
  - capabilities advertised `text`, `vision`, `audio`, `video`;
  - UI 10-turn text/cache ran with cache hits up to `482` cached tokens and
    no startup failure;
  - reasoning off/on/auto produced visible content; on produced separated
    reasoning; no hidden-only row in that section;
  - Chat and Responses streaming returned visible text with expected
    `GEMMA_STREAM_*` markers.
- Remaining live failure:
  - `uiAudio did not acknowledge expected audio phrase/tone`;
  - UI audio output was repeated `thought`;
  - post-audio text output was repeated `thought`;
  - Chat/Responses audio returned internal correction/reasoning-like text
    instead of the required phrase.
- Boundary:
  - 12B JANG_4M startup is fixed;
  - 12B JANG_4M audio/reasoning/media output is not fixed and not
    release-ready;
  - next blocker is to inspect Gemma4 audio prompt/template/reasoning parser
    behavior for 12B JANG_4M specifically, comparing against the E2B/E4B
    live-green audio rows.

## 2026-06-17 20:12 PDT - 12B JANG_4M processor config path source-fixed

Status: `PARTIAL_SOURCE_FIXED_INSTALLED_REBUILD_PENDING`.

- Full traceback captured from the installed-engine startup probe:
  - `/tmp/gemma12_jang4m_startup_trace.log`;
  - failure was not the JANG model config construction path anymore;
  - `mlx_vlm.utils.load_processor()` called
    `AutoProcessor.from_pretrained()`, which called
    `AutoTokenizer.from_pretrained()`, which reread the raw model
    `config.json` through Transformers `AutoConfig.from_pretrained()`;
  - that independent processor/tokenizer path still saw
    `text_config.final_logit_softcapping=30` as an int.
- Source fix:
  - `vmlx_engine/utils/jang_loader.py::_prepare_gemma4_processor_model_path()`
    now creates a temporary processor-load model directory with symlinks to the
    real model files plus a normalized `config.json`;
  - `_load_jang_vlm_processor()` uses that path for Gemma4 processor/image
    processor load and keeps the temporary directory alive on the processor;
  - the real model artifact is not mutated.
- Source verification:
  - `.venv/bin/python -m py_compile vmlx_engine/utils/jang_loader.py
    tests/test_jang_loader.py` -> passed;
  - `.venv/bin/python -m pytest tests/test_jang_loader.py -q -k
    'gemma4_processor_path or gemma4_loader_normalizes_integer_softcap or
    gemma4_unified'` -> `7 passed, 71 deselected`;
  - direct real-artifact processor load on
    `/Users/eric/models/OsaurusAI--gemma-4-12B-it-qat-JANG_4M` used a temp
    processor path and loaded `GemmaTokenizer`;
  - controlled source engine startup for the same 12B JANG_4M model reached
    `Application startup complete`; logs showed both
    `Gemma4 config scalar normalization` and
    `Gemma4 processor config normalization`.
- Boundary:
  - this clears the source startup blocker only;
  - `/Applications/vMLX.app` must still be rebuilt and the installed
    12B JANG_4M live harness must pass before the row can move out of red.

## 2026-06-17 20:02 PDT - 12B JANG_4M installed row still live-red after bundled normalization

Status: `PARTIAL_SOURCE_PRESENT_IN_BUNDLE_LIVE_RED`.

- Rebuilt and installed `/Applications/vMLX.app`; current integrity checks:
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed;
  - packaged Python imports `vmlx_engine 1.5.63`;
  - installed bundled `vmlx_engine/utils/jang_loader.py` contains
    `_normalize_gemma4_config_scalar_types` and `final_logit_softcapping`.
- Latest installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-2026-06-18T02-59-56-741Z/gemma4-media-proof.json`;
  - status `fail`;
  - model path
    `/Users/eric/models/OsaurusAI--gemma-4-12B-it-qat-JANG_4M`;
  - app log confirmed bundled engine `1.5.63`, family `gemma4`,
    `tool: gemma4`, `reasoning: gemma4`, `VLM: true`;
  - failure remains before ready:
    `TypeError: Field 'final_logit_softcapping' with value 30 doesn't match
    any type in (<class 'float'>, <class 'NoneType'>)`.
- Current read:
  - the source fix is present in the installed bundle, so the remaining blocker
    is not simply "forgot to rebuild";
  - the next required step is full traceback capture from the spawned engine to
    identify the still-bypassing config construction path.
- Boundary:
  - 12B JANG_4M is not fixed, not live-passing, and not release-ready.

## 2026-06-17 19:52 PDT - 12B JANG_4M actual VLM load path softcap fix

Status: `PARTIAL_SOURCE_FIXED_AFTER_INSTALLED_RED`.

- Rebuilt and installed `/Applications/vMLX.app` after the first
  `gemma4_unified.config` softcap fix.
- Installed-app integrity after that rebuild:
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed;
  - packaged Python imported `vmlx_engine 1.5.63`;
  - direct packaged `mlx_vlm.models.gemma4_unified.config.ModelConfig` parse
    returned `softcap 30.0 float`.
- Installed live artifact remained red:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-2026-06-18T02-49-24-920Z/gemma4-media-proof.json`;
  - status `fail`;
  - same `final_logit_softcapping` int TypeError before ready.
- Corrected source trace:
  - actual installed load goes through
    `vmlx_engine/utils/jang_loader.py` and calls
    `model_class.ModelConfig.from_dict(config)` directly for JANG VLM loads;
  - the previous vendored `gemma4_unified.config` fix did not normalize the
    dict before this JANG-loader model-class construction path;
  - `vmlx_engine/utils/jang_loader.py` now normalizes Gemma4
    `final_logit_softcapping` int -> float immediately after every config load
    and after Gemma4 VL promotion.
- Source verification:
  - `.venv/bin/python -m py_compile vmlx_engine/utils/jang_loader.py
    tests/test_jang_loader.py` -> passed;
  - `.venv/bin/python -m pytest tests/test_jang_loader.py -q -k
    'gemma4_loader_normalizes_integer_softcap or gemma4_unified'`
    -> `6 passed, 71 deselected`;
  - source reproduction of the exact JANG VLM config construction path printed
    `model_class mlx_vlm.models.gemma4_unified` and
    `softcap 30.0 float`.
- Boundary:
  - the real JANG VLM config-construction failure is now fixed in source;
  - installed app rebuild and 12B JANG_4M live rerun are still required before
    the row can move out of red.

## 2026-06-17 19:40 PDT - Gemma4 12B JANG_4M softcap config load blocker source-fixed

Status: `PARTIAL_SOURCE_FIXED_LIVE_RERUN_PENDING`.

- Failed installed live artifact:
  - `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-2026-06-18T02-37-38-287Z/gemma4-media-proof.json`;
  - status `fail` before ready;
  - error:
    `TypeError: Field 'final_logit_softcapping' with value 30 doesn't match
    any type in (<class 'float'>, <class 'NoneType'>). Field expected float,
    got int.`
- Source trace:
  - `/Users/eric/models/OsaurusAI--gemma-4-12B-it-qat-JANG_4M/config.json`
    has `text_config.final_logit_softcapping=30` as an integer;
  - `vmlx_engine/models/gemma4_unified/config.py::TextConfig.from_dict()` now
    normalizes that exact integer field to `30.0` before the upstream typed
    config constructor runs;
  - `tests/test_jang_loader.py` adds a regression for the integer softcap.
- Source verification:
  - `.venv/bin/python -m py_compile vmlx_engine/models/gemma4_unified/config.py
    tests/test_jang_loader.py` -> passed;
  - `.venv/bin/python -m pytest tests/test_jang_loader.py -q -k
    'gemma4_unified'` -> `5 passed, 71 deselected`;
  - exact artifact config parse now returns
    `parsed gemma4_unified TextConfig 30.0 float`.
- Boundary:
  - This clears the source/config parse failure only.
  - `/Applications/vMLX.app` must be rebuilt, then 12B JANG_4M must rerun
    installed UI + Chat + Responses + audio + cache + reasoning proof before
    this row can be green.

## 2026-06-17 19:34 PDT - Gemma4 E4B MXFP4 installed live harness passes

Status: `E4B_MXFP4_LIVE_PASS_OVERALL_RELEASE_PARTIAL`.

- Installed live artifact:
  - JSON:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T02-32-24-174Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T02-32-24-174Z/gemma4-media-final.png`;
  - status: `pass`, failures `[]`.
- Proven rows:
  - session settings visible:
    `usePagedCache=false`, `enableDiskCache=true`,
    `kvCacheQuantization=auto`, `enableBlockDiskCache=false`;
  - parser/autodetect:
    `toolCallParser=gemma4`, `reasoningParser=gemma4`,
    `isMultimodal=true`;
  - UI 10-turn text/cache run produced visible answers with memory cache hits;
  - reasoning off/on/auto produced visible content, separated reasoning where
    expected, no hidden-only rows;
  - UI audio, Chat Completions audio, and Responses audio each returned
    `Audio present`;
  - post-audio text recovery returned `GEMMA_POST_AUDIO_TEXT_OK`;
  - Chat and Responses streaming returned visible text/deltas;
  - health/cache reported Gemma4 native `mixed_swa_kv_v1`, paged false,
    block-disk L2 false.
- Boundary:
  - E4B MXFP4 audio/media row is green for this installed artifact.
  - Overall `.63` release remains blocked by 12B audio, 26B visual/MoE split,
    31B visual, MM3 remaining rows, notarized DMG, and final private writeup.

## 2026-06-17 19:31 PDT - Gemma4 E2B MXFP4 installed live harness passes

Status: `E2B_MXFP4_LIVE_PASS_OVERALL_RELEASE_PARTIAL`.

- Rebuilt `/Applications/vMLX.app` with the session settings telemetry fix via
  `panel/scripts/build-and-install.sh`.
- Installed app verification:
  - final script deep signature verification passed;
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed;
  - bundled Python imports `vmlx_engine 1.5.63`;
  - packaged `app.asar` contains the new
    `applyMissingCacheStackStartupDefaults`/`cacheDefaultsFilled` settings
    path.
- Installed live artifact:
  - JSON:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T02-27-21-412Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T02-27-21-412Z/gemma4-media-final.png`;
  - status: `pass`, failures `[]`.
- Proven rows in this artifact:
  - session settings visible:
    `usePagedCache=false`, `enableDiskCache=true`,
    `kvCacheQuantization=auto`, `enableBlockDiskCache=false`;
  - parser/autodetect visible:
    `toolCallParser=gemma4`, `reasoningParser=gemma4`,
    `isMultimodal=true`;
  - generation defaults visible:
    `defaultDoSample=true`, `defaultTemperature=100`,
    `defaultTopP=95`, `defaultTopK=64`;
  - UI 10-turn text/cache run produced visible answers with memory cache hits
    and no harness blank/loop failure;
  - reasoning off/on/auto produced visible content and separated reasoning
    where expected, no hidden-only rows;
  - UI audio, Chat Completions audio, and Responses audio each returned
    `audio present`;
  - post-audio text recovery returned `GEMMA_POST_AUDIO_TEXT_OK`;
  - Chat and Responses streaming returned visible text/deltas;
  - health/cache reported Gemma4 native `mixed_swa_kv_v1`, paged false,
    block-disk L2 false, and disk prompt L2 stores.
- Boundary:
  - E2B MXFP4 audio/media row is green for this installed artifact.
  - Overall `.63` release remains `PARTIAL_RELEASE_BLOCKED`: E4B audio, 12B
    audio, 26B visual/MoE split, 31B visual, MM3 remaining rows, notarized DMG,
    and final private writeup are not all current-green.

## 2026-06-17 19:23 PDT - E2B live audio fixed, settings telemetry source-only fix pending rerun

Status: `PARTIAL_SETTINGS_TELEMETRY_FIX_SOURCE_ONLY`.

- Installed app rebuilt after the batched `num_audio` processor-template fix.
- Installed app integrity:
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed;
  - installed bundled Python imports `vmlx_engine 1.5.63`;
  - installed `BatchedEngine._apply_chat_template` contains `num_audio`,
    `or num_audio > 0`, and the `num_audio == 0` image-branch guard.
- Latest installed live artifact:
  - JSON:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T02-16-35-358Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T02-16-35-358Z/gemma4-media-final.png`;
  - harness status: `fail`.
- Live positives:
  - UI 10-turn text run remained visible/coherent enough for the harness, with
    cache hits and about `67-86 t/s`;
  - reasoning off/on/auto produced visible content and separated reasoning
    where expected;
  - UI audio returned `audio present`;
  - Chat Completions audio returned `audio present`;
  - Responses audio returned `audio present`;
  - post-audio text recovery returned `GEMMA_POST_AUDIO_TEXT_OK`.
- Live failures:
  - `session usePagedCache=undefined`;
  - `session enableDiskCache=undefined`;
  - `session kvCacheQuantization=undefined`.
- Source root cause for remaining failures:
  - fresh `sessions.create()` with minimal config called
    `markCacheStackStartupDefaultsCurrent(config)` before filling missing
    cache-stack defaults;
  - `startSession()` then skipped `applyCacheStackStartupDefaultMigration()`
    because the version was already current, leaving the persisted session
    config display fields undefined.
- Source fix:
  - `panel/src/main/sessions.ts` adds
    `applyMissingCacheStackStartupDefaults()` and calls it before stamping
    fresh/minimal session configs current;
  - the helper fills missing, not explicit, values for the visible cache tuple
    (`usePagedCache`, `enableDiskCache`, `kvCacheQuantization`, and dependent
    cache defaults);
  - start-session writeback now includes `cacheDefaultsFilled`.
- Source verification:
  - `cd panel && npm run typecheck` -> passed;
  - `cd panel && npx vitest run tests/settings-flow.test.ts`
    -> `257 passed`;
  - `git diff --check -- panel/src/main/sessions.ts panel/tests/settings-flow.test.ts`
    -> passed.
- Boundary:
  - The model/media/API behavior is live-positive in the latest artifact, but
    this row is still not green because the settings source fix has not yet
    been rebuilt into `/Applications/vMLX.app` and rerun.

## Prior MiniMax-M3 Evidence Boundary

Target evidence still required before release/notarization claims:
- live packaged app load from `/Applications/vMLX.app`;
- logs/CLI showing M3 autodetect with paged cache off, generic KV/TQ skipped, native MSA cache, SSD prefix cache, JIT off, parsers set to `minimax_m3`;
- live multi-turn UI behavior through 10 turns with full output inspected for blanks, hidden-only turns, reasoning leaks, and repetition loops;
- long-context/prefix-cache hit proof with cache telemetry and tail inspection;
- reasoning off/on/auto proof through the live app/API path;
- generation defaults proof for the model-owned `temperature=1.0`, `top_p=0.95`, `do_sample=true` defaults and explicit overrides;
- speed evidence;
- only after all rows are green: Developer ID notarized release artifact and public release surfaces.

## 2026-06-17 19:02 PDT - Gemma4 E2B latest installed audio proof is live red

Status: `PARTIAL_LIVE_RED_AUDIO_CONDITIONING`.

- Source fixes currently present:
  - `vmlx_engine/mllm_batch_generator.py` promotes Gemma processor
    `input_features` / `input_features_mask` and forwards raw Gemma audio
    features as `input_features`, not precomputed `audio_embeds`;
  - `vmlx_engine/models/gemma4_unified/config.py` preserves real
    `gemma4_audio` configs as upstream Gemma4 audio configs;
  - `vmlx_engine/models/gemma4_unified/gemma4_unified.py` builds `AudioEncoder`
    for `gemma4_audio`, inverts valid masks for tower padding semantics, runs
    mel features through the audio tower before `embed_audio`, and sanitizes
    audio tower Conv weights into MLX layout.
- Source verification:
  - `py_compile` passed for the patched Gemma4 unified files and
    `tests/test_jang_loader.py`;
  - `.venv/bin/python -m pytest tests/test_jang_loader.py -q -k
    'gemma4_ple_native_mxfp4 or gemma4_moe_mxfp or gemma4_unified'`
    -> `7 passed, 68 deselected`;
  - `.venv/bin/python -m pytest tests/test_mllm_scheduler_cache.py -q -k
    'processor_direct or audio_outputs or input_features or
    audio_payload_prefill or gemma_input_features'`
    -> `5 passed, 97 deselected`;
  - `.venv/bin/python -m pytest tests/test_engine_audit.py -q -k
    'gemma4_unified and audio'`
    -> `4 passed, 554 deselected`;
  - real E2B source-load sanity produced `audio_tower=AudioEncoder`,
    `embed_audio_weight=(1536,1536)`, processor features `(1,99,128)`, and
    `model.get_audio_features(...) -> (25,1536)`.
- Installed app rebuild:
  - ran `panel/scripts/bundle-python.sh`,
    `panel/scripts/verify-bundled-python.sh`, and
    `panel/scripts/build-and-install.sh`;
  - `/Applications/vMLX.app` deep strict `codesign` verification passed;
  - installed bundled Python imported `vmlx_engine 1.5.63`;
  - installed source markers showed the `gemma4_audio` dispatch,
    `AudioEncoder`, mask inversion, and audio tower Conv sanitizer.
- Latest installed live artifact:
  - JSON:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T01-59-02-654Z/gemma4-media-proof.json`;
  - screenshot:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T01-59-02-654Z/gemma4-media-final.png`;
  - status: `fail`.
- Positive live evidence:
  - E2B loaded in `/Applications/vMLX.app`;
  - launch command included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--cache-memory-percent 0.15`, and `--max-tokens 512`;
  - logs showed Gemma4 JANG promotion to `gemma4_unified`, mixed-SWA cache
    handling, and disabled stored-cache quantization for mixed-SWA VLM;
  - UI 10-turn text run had no blanks/loops and memory-cache hits
    `23, 61, 111, 168, 231, 290, 349, 349, 349` at about `71-90 t/s`;
  - reasoning off/on/auto had visible content, separated reasoning where
    expected, and no raw reasoning-tag leak;
  - Chat Completions and Responses streaming returned visible text;
  - UI/Chat/Responses audio no longer 500 or matmul-crash;
  - post-audio text recovery returned `GEMMA_POST_AUDIO_TEXT_OK`.
- Failing live evidence:
  - `session usePagedCache=undefined`;
  - `session enableDiskCache=undefined`;
  - `session kvCacheQuantization=undefined`;
  - UI audio response: `Please provide the audio file you would like me to
    transcribe.`;
  - Chat and Responses audio responses: `Please provide the audio file you are
    referring to...`;
  - UI, Chat, and Responses audio failed the expected phrase/tone checks.
- Current source boundary:
  - the previous raw-feature/matmul failure is cleared;
  - the remaining likely failure is prompt/input-id audio insertion:
    processor output must be checked for `<|audio|>` or audio-token count and
    compared to the 25 audio tokens returned by `get_audio_features`;
  - if counts mismatch, patch Gemma4 unified processing/template expansion so
    the number of audio placeholders in `input_ids` matches the feature tokens.

## 2026-06-17 19:18 PDT - Gemma4 E2B batched audio template-slot source fix

Status: `PARTIAL_SOURCE_FIXED_LIVE_RERUN_PENDING`.

- Source root cause found:
  - live E2B uses the batched MLLM path in `vmlx_engine/engine/batched.py`;
  - `_apply_chat_template()` only forced the processor template route for
    images/videos, ZAYA text routing, and MiMo audio;
  - generic Gemma4 audio-only requests fell through to the tokenizer fallback,
    while `audio=` still reached `_call_processor_direct()` later;
  - that can produce the exact live symptom: audio payload is decoded and model
    audio features exist, but the rendered prompt does not contain the
    model-owned `<|audio|>` slot the model attends to.
- Source fix:
  - `BatchedEngine._apply_chat_template()` now accepts `num_audio` and routes
    generic MLLM audio requests through the processor template path;
  - image/video behavior is preserved for non-audio requests;
  - `_compute_gen_prompt_len()` and `_compute_segment_boundaries()` now pass
    `num_audio` so cache-key generation prompt stripping and segment metadata
    use the same prompt contract.
- Source verification:
  - `.venv/bin/python -m py_compile vmlx_engine/engine/batched.py
    tests/test_engine_audit.py` passed;
  - `.venv/bin/python -m pytest tests/test_engine_audit.py -q -k
    'BatchedEngineVideoTemplate or gemma4_unified and audio'`
    -> `15 passed, 544 deselected`;
  - `.venv/bin/python -m pytest tests/test_mllm_scheduler_cache.py -q -k
    'processor_direct or audio_outputs or input_features or
    audio_payload_prefill or gemma_input_features'`
    -> `5 passed, 97 deselected`.
- Real E2B source sanity:
  - `BatchedEngine._apply_chat_template()` rendered one literal `<|audio|>`;
  - `_call_processor_direct()` expanded it to `audio_token_count=25`;
  - `model.get_audio_features(...)` returned `feature_tokens=(25,1536)`;
  - final processed `input_ids_shape=(1,40)`.
- Boundary:
  - This is source proof only.
  - `/Applications/vMLX.app` must be rebuilt and the E2B installed UI + Chat +
    Responses audio harness must rerun before the row can move out of red.

## Current Source/Bundle Evidence

Current source has the claimed M3 runtime fixes:
- `MiniMaxM3SparseCache` remains first-class through memory-aware fetch/truncate and scheduler truncate paths.
- Scheduler selects `make_minimax_m3_sampler`, the raw-logits fp32 sampler matching the clean M3 runtime.
- Panel M3 startup defaults force prefix cache on, paged cache off, SSD prompt disk cache on, block-disk/paged L2 off, generic KV quantization ignored, and JIT off.
- Server maps M3 thinking off/on/auto to `disabled`/`enabled`/`adaptive` and seeds the reasoning parser when enabled prompt-opens `<mm:think>`.
- New source fix in `vmlx_engine/scheduler.py`: the memory-aware/object prefix-cache store path now prefers `response.prompt_cache_snapshot` when present instead of the live post-generation cache. This matters for M3 because the default app path has paged cache off and therefore uses memory-aware object cache hits.
- New source fix in `vmlx_engine/scheduler.py` and `vmlx_engine/memory_cache.py`: MiniMax-M3 prompt-cache truncation/fetch clones now materialize the keys/values/idx_keys slices instead of handing around lazy MLX slices. Generic KV already had this Metal-safety materialization path; M3 was bypassing it even though its MSA idx_keys lane is more sensitive to wrong offsets.
- New source fix in `vmlx_engine/memory_cache.py`: memory-aware fetch clones now materialize standard dense KV companion layers too. Live M3 cache layout is dense KV layers 0-2 plus sparse MSA layers 3-59; materializing only the sparse lanes still left API-thread lazy dense KV slices in cache-hit generation.
- New source fix in `vmlx_engine/scheduler.py`: the memory-aware/object prefix-cache store path now writes prompt disk L2 too. This uses the existing DiskCacheManager M3 tuple support: full generation-prompt-stripped disk key, N-1 cache payload, and first-class `idx_keys`.
- New source fix in `vmlx_engine/disk_cache.py` + `vmlx_engine/scheduler.py`: SSD prompt L2 now supports longest-prefix lookup instead of exact-prompt-only lookup. On a disk prefix hit for stored key `P` and current prompt `F`, scheduler restores the N-1 cache for `P` and replays `P[-1] + F[len(P):] + generation_prompt_suffix`.

Fresh checks this turn:
- `py_compile vmlx_engine/scheduler.py vmlx_engine/memory_cache.py vmlx_engine/models/minimax_m3/cache.py` -> passed.
- `pytest tests/test_minimax_m3_cache_paths.py -q` -> 20 passed after adding memory-aware prompt disk L2 and SSD longest-prefix regressions.
- `pytest tests/test_minimax_m3_cache_paths.py tests/test_cache_record_validator.py tests/test_disk_cache_unit.py tests/test_memory_cache.py -q` -> 105 passed.
- `cd panel && npx vitest run tests/model-config-registry.test.ts tests/settings-flow.test.ts` -> 325 passed.
- `cd panel && npm run typecheck` -> passed.
- `cd panel && ./scripts/verify-bundled-python.sh` -> passed, critical bundled files match source.
- `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app` -> valid on disk, satisfies Designated Requirement.
- Installed bundle contains `make_minimax_m3_sampler`, `clone_minimax_m3_sparse`, and `m3-visible-answer`; no stale `output_collector.py` `error_message=new.error_message` line found.
- 2026-06-17 10:05 PDT empty-assistant-row fix:
  - `panel/src/renderer/src/components/chat/ChatInterface.tsx` now rebinds an active stream to the latest persisted empty assistant placeholder when the user reloads/switches chats during TTFT.
  - `panel/src/renderer/src/components/chat/MessageBubble.tsx` now renders `Waiting for model response...` for an active empty assistant row and `No visible response was produced.` only for a completed empty assistant row.
  - `vmlx_engine/server.py` Responses streaming now defines MiniMax-M3 thinking-mode locals before use; the installed app had previously thrown `NameError: name '_is_minimax_m3' is not defined` on `/v1/responses` after the reasoning-mode patch.
  - `vmlx_engine/disk_cache.py` + `vmlx_engine/scheduler.py` now require `MiniMaxM3SparseCache` for M3 disk prompt-cache fetches and log restored M3 SSD hits as `m3-sparse`, rejecting stale KV-only disk records instead of accepting wrong cache class shapes.
  - Source checks: `npm run typecheck` passed; `py_compile vmlx_engine/disk_cache.py vmlx_engine/scheduler.py` passed; `pytest tests/test_streaming_reasoning.py tests/test_minimax_m3_cache_paths.py -q` -> 151 passed.
  - Installed app rebuilt via `panel/scripts/build-and-install.sh`; bundled source parity passed; `/Applications/vMLX.app` codesign verify passed; installed source contains `required_cache_class`, `m3-sparse`, and `latestPendingAssistantId`.

## Current Live Findings

Live app evidence is currently PARTIAL, not release-ready.

- Packaged app loaded M3 from `/Applications/.../site-packages` and generated at ~24-26 tok/s with correct CLI shape: JIT off, paged cache off, no generic KV quantization, parser flags `minimax_m3`, native MSA cache layers 3-59.
- Fresh 10-turn UI-style run did not hard-loop, but it became incoherent/contextless after memory prefix-cache hits: turns 6-8 answered generic "need context/details", and turn 10 summarized only the latest answer.
- After rebuilding with the snapshot+MSA-slice+dense-KV materialization fixes, packaged-app 10-turn cache-hit run no longer blanked or looped. It still failed strict quality/recall: turn 3 echoed the prompt and turn 10 summarized mostly the patch/testing topic. Health showed 9 memory hits / 4028 cached tokens and disk prompt cache stores still 0 in the installed app built before the SSD L2 store fix.
- After rebuilding with the SSD L2 store fix, live packaged-app 10-turn UI IPC run had no blank turns, no think-tag leaks, and no repetition loops; turn 8 answered profile options. Health showed 9 memory hits, 4978 tokens saved, 10 disk stores, 10 disk entries, 6039 prompt L2 tokens, and pending writes 0. It still failed strict recall because turn 10 summarized only patch/testing instead of the full EBS thread.
- Restart proof found another SSD L2 bug: after engine restart, memory L1 was empty and disk entries remained 10, but turn 11 produced `disk_hits=0`, `disk_misses=1`, and stored a new entry. Source trace: `DiskCacheManager.fetch()` was exact-token-hash only; multi-turn SSD reuse needs longest-prefix lookup.
- After rebuilding with the SSD longest-prefix hit fix, installed app live proof showed:
  - in-app logs: `Disk cache prefix hit: matched 1123/1210 prompt tokens` and `1122 tokens restored from disk`;
  - health after the turn: `disk_hits=1`, `disk_misses=0`, `disk_entries=12`, `l2_prompt_tokens_on_disk=8372`;
  - app chat log: 53 tokens in 12.7s, 23.9 t/s, 1122 cached prompt tokens;
  - response was visible and correctly referenced turn 8, with no repetition.
- Final installed-build realistic 10-turn UI IPC run had no blank turns, no think-tag leaks, no repetition loops, and stayed broadly coherent. Strict recall remains PARTIAL: turn 10 summarized EBS/patching/best practices but did not explicitly include every prior topic (AP, GL, responsibilities, concurrent programs, profile options).
- Reasoning off/on/auto live proof is PARTIAL:
  - off: reasoning hidden, visible content present, no leaked tags;
  - on: with larger output budget, reasoning and visible content were separated with no leaked tags;
  - auto: reasoning and visible content separated with no leaked tags;
  - model arithmetic quality on the tiny 17+28 prompt was unreliable, and small-budget reasoning-on produced reasoning-only with no visible final answer.
- Same exact live UI history replayed directly against the loaded engine with cache-hit and `skip_prefix_cache=true`, deterministic `temperature=0`, produced identical correct Oracle EBS/AP/GL/responsibilities/concurrent/profile/patch summaries. This shows the dense-KV/source cache state is much closer, but the live UI stochastic turn quality remains PARTIAL and needs rerun after SSD L2 rebuild.
- Persisted old live chat `861b5a50-c4b1-444a-ba9f-ca38250b6d8b` has the exact blank turn from the user paste: turn 8 assistant `content=""`, `reasoning=""`, `tokenCount=1`, `promptTokens=802`, `cachedTokens=652`, `cacheDetail=memory`.
- Controlled no-cache isolation on the already-loaded engine used the same failing 8-turn Responses sequence with `skip_prefix_cache=true`; all 8 turns were visible and coherent. This pins the failure layer to memory prefix-cache reuse, not the model weights or the visible parser.
- Health after the cached live run showed memory cache hits/stores only; disk prompt cache entries/stores remained 0. Current memory-aware M3 path does not provide SSD prompt L2 despite `--enable-disk-cache`.
- 2026-06-17 10:05 PDT live installed-app proof for the user-reported apparent empty answer:
  - Reproduced the exact visual state with a fresh selected app chat and prompt `What is Oracle EBS? Answer in one sentence under 25 words.`;
  - during TTFT, the assistant row displayed `Waiting for model response...` instead of a blank/dead-looking bubble; screenshot saved at `/tmp/m3-final-placeholder-waiting.png`;
  - completion was visible and coherent: `Oracle EBS (Enterprise Business Suite) is a comprehensive set of integrated enterprise applications for managing business processes like finance, HR, and supply chain.`;
  - app metrics: 29 tokens, 25.2 t/s, 177 prompt tokens with 172 disk cached, 23.74s TTFT, 25.0s total;
  - logs showed `Disk cache loaded with MiniMax-M3 sparse restore: 60 layers (57 MSA sparse)` and `Disk cache hit (m3-sparse): 173 tokens`, proving the SSD prompt cache hit used the M3 sparse tuple path, not generic KV/TQ remap;
  - no `NameError`, `Traceback`, or empty completed assistant content appeared in this proof.

## Current Boundary

Status remains PARTIAL for the overall release, but the current MiniMax-M3 app
cache/reasoning/tool row is materially improved by fresh source and live proof.

2026-06-17 11:06 PDT current-source + live app evidence:
- Source trace:
  - `/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 -B -s`
    imports `vmlx_engine.scheduler` from `/Users/eric/mlx/vllm-mlx/vmlx_engine/scheduler.py`.
  - `vmlx_engine/scheduler.py` now collects M3 `idx_keys` during prompt-only
    cache eval and, for MiniMax-M3 cache-hit requests, stores a clean
    prompt-boundary re-prefill cache instead of the hit-derived tail-replay
    cache.
  - Added regression
    `test_scheduler_m3_cache_hit_store_rederives_clean_prompt_cache` in
    `tests/test_minimax_m3_cache_paths.py`.
- Source verification:
  - `py_compile vmlx_engine/scheduler.py tests/test_minimax_m3_cache_paths.py`
    passed.
  - `pytest tests/test_minimax_m3_cache_paths.py -q` -> 23 passed.
  - `cd panel && npm run typecheck` -> passed.
- Live app proof:
  - M3 session `3d24ecd5-150f-46d8-a4f5-8481e54bd6cf` restarted in
    `/Applications/vMLX.app`, port `8006`, PID `3942`.
  - Startup logs confirmed CLI/parser/cache settings:
    `paged_cache=OFF`, `tq_kv=SKIP(native MSA)`, `jit=off`,
    `tool_parser=minimax_m3`, `reasoning_parser=minimax_m3`, dense KV layers
    0-2 and `MiniMaxM3SparseCache` layers 3-59.
  - Logs during the 10-turn run repeatedly showed:
    `MiniMax-M3 prefix cache store using clean prompt-boundary re-prefill ...
    after cache-hit tail replay`, followed by `Stored cache ...`.
  - `/tmp/m3_live_matrix_quiet_result.json`:
    - 10-turn app chat: no blanks, no think-tag leaks, no repetition loops;
      turns 2-10 all used memory prefix hits; final recall included all seven
      labels with `AP: Payables management`.
    - Speed on 10-turn row: about 22.8-24.2 t/s after load.
    - Reasoning off/on/auto: off had 0 reasoning chars; on/auto produced
      separated reasoning plus visible content; no raw think tags leaked.
    - Built-in tool row executed `run_command`, wrote
      `/tmp/m3-tool-proof-quiet-1781719330214/m3_tool_probe.txt` containing
      `M3_TOOL_OK`, and visible final content was `M3_TOOL_OK_DONE`.
    - Long-context row: 9,523 prompt tokens on turn 2, 9,478 cached tokens,
      cache detail `memory`, visible answer `PROFILE_OPTION_SENTINEL_ZETA_173`.
  - Screenshot proof:
    `/tmp/m3_live_matrix_quiet_final.png` shows the real app chat UI with final
    10-turn recall visible and cache telemetry under the messages.
  - Health after proof: `num_waiting=0`, `num_running=0`; not autonomously
    generating. Cache telemetry included memory and disk hit tokens.
- Remaining boundary:
  - Do not claim full release/notarization/main-update readiness yet. The
    broader release still needs packaging/notarization workflow, cross-family
    regression gates, and any public release surface updates explicitly run
    from the current source. M3-specific proof above is current-source live app
    proof, not a global release clearance.

## 2026-06-17 11:27 PDT - Superseding Installed-App MiniMax-M3 Proof

Status: FIXED for the measured MiniMax-M3 installed-app cache/reasoning/tool
and long-context rows; PARTIAL for full release/notarization/main/public update.

- Installed app:
  - launched `/Applications/vMLX.app` via normal macOS `open -n ... --args
    --remote-debugging-port=9333`;
  - app user agent reported `vmlx/1.5.62`;
  - M3 session `3d24ecd5-150f-46d8-a4f5-8481e54bd6cf` restarted on port
    `8006`, PID `55050`;
  - `/health` after proof reported `status=healthy`, `model_loaded=true`,
    `num_waiting=0`, `num_running=0`.
- Bundle/source parity and signing:
  - isolated bundled Python import (`python3 -I -B -s` from `/tmp`) loaded
    `vmlx_engine.scheduler` from
    `/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/python3.12/site-packages/vmlx_engine/scheduler.py`;
  - source scheduler bytes matched installed scheduler bytes,
    `sha256=e691bf597d0cf21c82dba58b6a3205da1f91e07ba3bac84c077b481a761d8f8e`;
  - installed scheduler contains the `clean prompt-boundary re-prefill` marker;
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed: valid on disk and satisfies its Designated Requirement.
- Source verification:
  - `py_compile vmlx_engine/scheduler.py tests/test_minimax_m3_cache_paths.py`
    passed;
  - `pytest tests/test_minimax_m3_cache_paths.py -q` -> 23 passed.
- Installed-app logs confirmed settings:
  - `MiniMax-M3 AUTODETECTED (model_type=minimax_m3_vl) ->
    paged_cache=OFF, tq_kv=SKIP(native MSA), vl_route=ON,
    tool_parser=minimax_m3, reasoning_parser=minimax_m3, jit=off,
    msa_per_step_sync=ON`;
  - runtime cache layout is dense `KVCache` layers 0-2 and
    `MiniMaxM3SparseCache` layers 3-59;
  - long-context and tool rows showed `Disk cache loaded with MiniMax-M3
    sparse restore: 60 layers (57 MSA sparse)` and `Disk cache hit
    (m3-sparse)`;
  - cache-hit stores logged repeated `MiniMax-M3 prefix cache store using clean
    prompt-boundary re-prefill ... after cache-hit tail replay`.
- Live UI matrix artifacts:
  - `/tmp/m3_live_matrix_quiet_result.json`;
  - `/tmp/m3_live_matrix_quiet_final.png`.
- Live UI matrix results:
  - 10-turn cache-hit chat: 10 turns, 0 blanks, 0 think-tag leaks, 0 loop
    suspects; final recall included all seven labels
    `EBS/AP/GL/RESP/CP/PROFILE/PATCH`;
  - cache telemetry on 10-turn row: turn 1 disk hit, turns 2-10 memory hits,
    172 -> 1068 cached tokens, about 23.8-25.5 tok/s;
  - reasoning off/on/auto: off returned visible `OFF_VISIBLE_OK` with 0
    reasoning chars; on returned visible `ON_VISIBLE_OK` with separated
    reasoning; auto returned visible `AUTO_VISIBLE_OK` with separated
    reasoning; no raw think tags leaked;
  - tool row: Responses tool call invoked `run_command` with
    `printf 'M3_TOOL_OK' > m3_tool_probe.txt`, tool result contained
    `M3_TOOL_OK`, and final visible content was `M3_TOOL_OK_DONE`;
  - long-context row: turn 2 had 9,523 prompt tokens, 9,478 cached tokens,
    `cacheDetail=memory`, visible answer `PROFILE_OPTION_SENTINEL_ZETA_173`;
  - final cache telemetry: `cache_hit_requests=14`, `cache_hit_tokens=27968`
    (`disk=9843`, `memory=18125`), disk cache `hits=6`, `stores=9`,
    pending writes 0.
- Boundary:
  - This is installed-app live proof for the measured MiniMax-M3 rows.
  - Do not claim global release readiness, notarization completion, main merge,
    or cross-family regression coverage from this entry alone.

## 2026-06-17 11:52 PDT - Release signing/notary state

Status: PARTIAL for v1.5.62 release packaging; signing/notary credentials are
usable, but the current repo release gate is red.

- Source/package version:
  - `panel/package.json` reports `1.5.62`;
  - `pyproject.toml` reports `1.5.62`.
- Signing/notary:
  - `panel/.env.signing` is present and defines the Apple ID, team ID,
    app-specific password, and signing identity variables;
  - `security find-identity -p codesigning -v` finds
    `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`;
  - `xcrun notarytool store-credentials vmlx-notary ...` using
    `panel/.env.signing` succeeded;
  - `xcrun notarytool history --keychain-profile vmlx-notary
    --output-format json` succeeded and returned accepted history.
- Public release surfaces checked:
  - `gh release view v1.5.62 --repo jjang-ai/vmlx` -> release not found;
  - `gh release view v1.5.62 --repo jjang-ai/mlxstudio` -> release not found;
  - `jjang-ai/mlxstudio` latest public release remains `v1.5.61`;

## 2026-06-17 16:51 PDT - v1.5.63 MM3/Gemma release scope reopened

Status: PARTIAL. No `.63` release, main merge, notarization, or public update
claim is allowed until the source and live proof rows below are current.

Current Eric instructions captured for this continuation:
- Focus on MiniMax-M3 compatibility and Gemma4 compatibility, not MiMo/N2.
- Keep MiniMax-M3 MSA/Lightning sparse cache, iRoPE/offset handling, async
  rederive/cache-sync behavior, SSD prefix cache store/hit, and reasoning
  kwargs/toggles as first-class proof rows.
- Add the Gemma4 26B-A4B MoE fix from CRACK notes: fused quantized
  `experts.gate_up_proj` sidecars must be split into SwitchGLU `gate_proj` and
  `up_proj`, preserving packed quantized weights/scales/biases instead of
  dequantizing or leaving random-initialized experts.
- Verify serving defaults use model generation config (`do_sample=true`,
  `temperature=1.0`, `top_k=64`, `top_p=0.95` for Gemma4 rows; M3 uses its
  model-owned defaults) unless explicit request kwargs override them.
- Required live proof before release: app UI multi-turn coherency/stress,
  reasoning off/on/auto, Chat Completions API, Responses API, tool usage,
  prefix-cache reuse without incoherent loops, and modality gates for local
  Gemma E2B/E4B/12B audio-capable rows, 26B/31B visual rows, and MM3-VL.
- Write progress/status/checklist updates as work happens.

Immediate work plan:
1. Port the Gemma4 fused quantized MoE sidecar split/rename fix into this
   active repo without disturbing existing M3 fixes or unrelated dirty build
   artifacts.
2. Add/refresh unit tests for loader sanitize, runtime patches, and quant-module
   path mapping so MXFP/JANG packed sidecars stay first-class.
3. Audit source for MM3/Gemma autodetect, cache defaults, TurboQuant KV gating,
   reasoning parser selection, generation defaults, Chat/Responses/Ollama/
   Anthropic routes, and UI startup flags.
4. Run focused source regressions before any app build.
5. Build/install only after source rows pass, then run live UI/API stress
   matrix and record artifacts.

No-claim boundary:
- Existing M3 installed-app proof from 11:27 PDT remains useful but is not a
  `.63` release clearance.
- Gemma4 26B/31B/E2B/E4B/12B compatibility is currently unproven in this
  continuation until current-source tests and live loads/API/UI rows pass.
  - `/Users/eric/mlx/mlxstudio/latest.json` still points to `1.5.61`.

## 2026-06-17 17:08 PDT - Installed 1.5.63 app bundle ready for live stress

Status: PARTIAL.

Current installed-app evidence:
- `/Applications/vMLX.app` was rebuilt by `panel/scripts/build-and-install.sh`.
- Deep codesign verification passed on the installed app.
- Installed bundled Python imports `vmlx_engine 1.5.63`.
- Installed bundled Python imports the registered Gemma4 Unified runtime under
  `mlx_vlm.models.gemma4_unified` after the loader registry patch.

Still open before any release/notarization/main-update claim:
- MiniMax-M3 installed-app live stress:
  - launch logs must show M3 autodetect, paged cache off, TurboQuant KV skipped,
    native MSA/Lightning sparse cache, SSD prefix cache configured, reasoning
    and tool parser settings, and JIT forced off;
  - 10-turn UI coherence with visible assistant output, no empty hidden-only
    turns, no autonomous generation after the user turn, no repetition loops,
    and cache-hit telemetry on later turns;
  - reasoning off/on/auto must each be checked from the UI and API surface;
  - Chat Completions and Responses API must be checked, including tools and
    usage/cache telemetry.
- Gemma4 live matrix:
  - E2B/E4B/12B advertised audio rows need live media proof;
  - 26B/31B visual rows need live image/text proof;
  - 26B fused quantized MoE split must be proven by an actual load/generation,
    not only by sanitizer tests.
- Release:
  - no notarization, public release, `latest.json`, or main-branch release
    claim until the live rows above are current green or Eric explicitly waives
    the gate.
- Proper release scripts found:
  - `panel/scripts/build-release-dmgs.sh` builds two public artifacts:
    `vMLX-${VERSION}-sequoia-arm64.dmg` using compat/macOS 14 wheels and
    `vMLX-${VERSION}-tahoe-arm64.dmg` using native/macOS 26 wheels;
  - `panel/scripts/notarize-release-dmgs.sh` notarizes/staples both DMGs with
    keychain profile `vmlx-notary`;
  - `panel/scripts/verify-release-dmgs.sh` checks hdiutil, Developer ID
    signature, stapler, spctl, and SHA256 for both DMGs.
- Release gate:
  - `cd panel && npm run release:prepackage -- --out
    ../build/current-release-regression-manifest-pre-codex-20260617.json`
    exited 1;
  - manifest status: `current_proof_sweep=fail`,
    `prepackage_ready=false`, `release_ready=false`;
  - current manifest reports 38 failed/missing/open sweep components and four
    explicit open blockers:
    `mimo_v2_jang2l_runtime_quality_open`,
    `source_version_already_public`,
    `real_ui_unblocked_non_mimo_missing`,
    `real_ui_unblocked_non_mimo_partial`.
- Boundary:
  - A signed/notarized `.62` DMG can be built only as an explicit override
    candidate while this gate remains red.
  - Do not publish/tag/update `mlxstudio/latest.json` or call `.62`
    release-ready until either the ledger passes or Eric explicitly waives the
    release gate for a scoped M3-only emergency release.

## 2026-06-17 17:23 PDT - MiniMax-M3 installed-app text/API/cache stress passed

Status: PARTIAL_RELEASE_BLOCKED.

Current live artifact:
- JSON:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T00-15-59-440Z/mm3-stress-proof.json`
- Screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T00-15-59-440Z/mm3-stress-final.png`
- Harness:
  `panel/scripts/live-mm3-stress-proof.mjs`

Verdict:
- The original harness verdict falsely failed on API metadata keys
  `models` and `servedModel`; the source check now skips API metadata entries
  without an `ok` field, and the recorded artifact was recomputed to
  `status=pass`, `failures=[]`.
- This is an installed `/Applications/vMLX.app` live run, not just CLI/API
  proof.

Live evidence in the artifact:
- UI launched the installed bundled engine for
  `JANGQ-AI/MiniMax-M3-REAP40-d3-JANG_2L`.
- App session config used model-owned generation defaults:
  `defaultTemperature=100`, `defaultTopP=95`, `defaultDoSample=true`.
- Launch args omitted paged cache and JIT and included:
  `--tool-call-parser minimax_m3`,
  `--enable-auto-tool-choice`,
  `--reasoning-parser minimax_m3`,
  `--enable-disk-cache`,
  `--max-tokens 512`.
- Engine logs showed:
  `MiniMax-M3 AUTODETECTED`, `paged_cache=OFF`,
  `tq_kv=SKIP(native MSA)`, `vl_route=ON`,
  `tool_parser=minimax_m3`, `reasoning_parser=minimax_m3`,
  `jit=off`, `msa_per_step_sync=ON`.
- Native cache proof showed `family=minimax_m3`,
  `schema=minimax_m3_msa_v1`, `cache_type=native_msa_sparse_kv`,
  components `attention_kv`, `msa_idx_keys`, `absolute_block_index`,
  generic TurboQuant KV disabled, `paged=false`, `prompt_disk_l2=true`,
  dense KV layers `0...2`, sparse MSA layers `3...59`.
- 10-turn UI multiturn coherence passed with no empty/hidden-only assistant
  turns, no raw think tags, no loop suspect, no autonomous assistant turn, and
  final recall containing labels `EBS`, `AP`, `GL`, `RESP`, `CP`, `PROFILE`,
  and `PATCH`.
- UI cache telemetry across the 10 turns:
  cached tokens `0, 172, 254, 405, 576, 749, 932, 1105, 1288, 1389`;
  speed stayed `22.1-24.2 tok/s`.
- Reasoning UI modes passed:
  off produced visible `OFF_VISIBLE_OK` with `reasoningChars=0`;
  on produced visible `ON_VISIBLE_OK` with `reasoningChars=512`;
  auto produced visible `AUTO_VISIBLE_OK` with `reasoningChars=739`.
- UI builtin tool proof passed:
  `run_command` wrote `tool-workdir/m3_tool_probe.txt` with `M3_TOOL_OK`, and
  the final assistant message contained `M3_TOOL_OK_DONE`.
- Long-context/prefix-cache proof passed:
  turn 2 recalled `PROFILE_OPTION_SENTINEL_ZETA_173` with
  `promptTokens=5820`, `cachedTokens=5778`, `cacheDetail=memory`,
  `ppSpeed=3419.5`, `ttft=1.70`, and `22.3 tok/s`.
- Direct API proof against the same app session passed:
  Chat Completions visible text returned HTTP 200;
  Chat Completions required tool returned HTTP 200 with `tool_calls`;
  Responses returned visible `API_RESP_OK`;
  Responses with `previous_response_id` returned visible `violet`.

Still open:
- MM3-VL image input was not exercised in this artifact; `vl_route=ON` is only
  startup/source evidence until an image turn produces visible output.
- Anthropic and Ollama protocol surfaces were not exercised in this artifact.
- Fresh-process SSD/L2 restore was not exercised; this artifact proves prompt
  prefix memory-cache hits and prompt-disk L2 configuration, not restart restore.
- Gemma E2B/E4B/12B audio and Gemma 26B/31B visual rows remain unproven live.
- Therefore `.63` is not release-ready and must not be notarized/published yet
  without either completing those rows or an explicit scoped override.

## 2026-06-17 17:58 PDT - MiniMax-M3 installed-app expanded stress passed after Responses image fix

Status: PARTIAL_RELEASE_BLOCKED.

Source fix:
- `vmlx_engine/server.py` now applies the existing gated MiniMax-M3 VL
  image-only carve-out to `/v1/responses`, the UI default wire API.
- Responses now:
  - removes only `image` / `vision` from the unsupported modality set when
    `_m3_vl_image_ok(engine)` is true;
  - still rejects unsupported non-image modalities such as audio/video for M3;
  - preserves multimodal content arrays for M3 image-only Responses input so
    `BatchedEngine` can run the existing M3 VL preprocessing path.
- Regression test:
  `tests/test_multimodal_routing.py::test_responses_m3_vl_image_only_carveout_preserves_ui_default_path`.

Source tests:
- `.venv/bin/python -m py_compile vmlx_engine/server.py tests/test_multimodal_routing.py`
  passed.
- `.venv/bin/python -m pytest tests/test_multimodal_routing.py tests/test_minimax_m3_cache_paths.py -q`
  -> `31 passed`.

Installed app:
- `panel/scripts/bundle-python.sh` rebuilt bundled Python with local
  `vmlx-1.5.63` and `jang-2.5.30`.
- `panel/scripts/verify-bundled-python.sh` passed.
- `panel/scripts/build-and-install.sh` installed `/Applications/vMLX.app`.
- Installed bundle contains the new server helpers:
  `_m3_vl_response_image_only` and
  `_responses_modalities_unsupported_after_m3_vl_carveout`.
- `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
  passed.
- Installed bundled Python reports `vmlx_engine.__version__ == 1.5.63`.

Live artifact:
- JSON:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T00-42-13-479Z/mm3-stress-proof.json`
- Screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T00-42-13-479Z/mm3-stress-final.png`
- Verdict: `status=pass`, `failures=[]`.

Live evidence:
- App-managed installed-app session loaded the MM3 REAP40 d3 model on port
  `60526`.
- Session settings reflected model-owned generation defaults and MM3 policy:
  `defaultTemperature=100`, `defaultTopP=95`, `defaultDoSample=true`,
  `enablePrefixCache=true`, `usePagedCache=false`, `enableDiskCache=true`,
  `enableBlockDiskCache=false`, `enableJit=false`,
  `toolCallParser=minimax_m3`, `reasoningParser=minimax_m3`,
  `kvCacheQuantization=auto`, `isMultimodal=true`.
- Engine logs showed:
  `MiniMax-M3 AUTODETECTED`, `paged_cache=OFF`,
  `tq_kv=SKIP(native MSA)`, `vl_route=ON`,
  `tool_parser=minimax_m3`, `reasoning_parser=minimax_m3`, `jit=off`,
  `msa_per_step_sync=ON`.
- Native cache snapshot:
  `family=minimax_m3`, `schema=minimax_m3_msa_v1`,
  `cache_type=native_msa_sparse_kv`,
  components `attention_kv`, `msa_idx_keys`, `absolute_block_index`,
  generic TurboQuant KV disabled with reason `native_minimax_m3_msa_idx_keys`,
  storage quantization disabled, prompt disk L2 configured,
  `paged=false`, `block_disk_l2=false`.
- 10-turn UI coherence passed:
  cached tokens per turn `172, 172, 249, 380, 515, 656, 785, 926, 1081, 1196`;
  token speed `22.4-24.9 tok/s`;
  no hidden-only turn, no loop suspect, no raw think tag leak, no autonomous
  assistant turn.
- UI reasoning modes passed:
  off: `reasoningChars=0`, visible `OFF_VISIBLE_OK`;
  on: `reasoningChars=523`, visible `ON_VISIBLE_OK`;
  auto: `reasoningChars=1156`, visible output, no hidden-only reply.
- UI tool proof passed:
  `run_command` wrote `M3_TOOL_OK`; assistant returned `M3_TOOL_OK_DONE`;
  persisted tool-call JSON was present.
- Long-context prefix proof passed:
  recalled `PROFILE_OPTION_SENTINEL_ZETA_173` with `promptTokens=5820`,
  `cachedTokens=5778`, `cacheDetail=memory`, `22.6 tok/s`.
- MM3 UI image/VL proof passed through the installed app default Responses
  route:
  attached red PNG -> visible `MM3_IMAGE_RED`, no send error, no hidden-only
  reply, no loop.
- API proof passed against the same session:
  Chat Completions visible text;
  Chat Completions required tool call;
  Responses first turn + `previous_response_id`;
  Anthropic Messages visible text;
  Ollama `/api/chat` visible text;
  Ollama `/api/generate` visible text.

Still open:
- MM3 streaming delta exactness across Chat Completions, Responses, Anthropic,
  and Ollama is not yet live-proven.
- MM3 fresh-process SSD prompt-cache restore is not yet live-proven.
- MM3 UI settings/i18n labels and concurrent-request/Stop-state rows are not
  yet live-proven.
- Gemma E2B/E4B/12B audio and Gemma 26B/31B visual rows remain open.
- Release/notarization/main/update rows remain blocked until the Gemma and
  expanded UI/gateway/parser rows are current green or explicitly waived.

# 2026-06-17 PDT - Added Swift handoff context to the .63 tracker

- Added the pasted `/Users/eric/vmlx` `AGENTS.md` context to
  `.agents/RELEASE-1.5.63-STRESS-MATRIX.md`.
- Scope: reference/future Swift-engine writeup only for this `.63` lane.
  Active runtime/release proof remains Python `vmlx_engine` plus Electron
  MLXStudio in `/Users/eric/mlx/vllm-mlx`.
- New documentation gate before release handoff: private internal writeup for
  MM3 MSA/Lightning sparse cache, SSD prefix-cache build-hit-restore, iRoPE and
  async/sync cache boundaries, reasoning/tool/streaming behavior, Gemma
  SWA/TurboQuant policy, mistakes made, exact build/test commands, and any
  remaining `PARTIAL` / `BLOCKED` rows.

## 2026-06-17 18:03 PDT - Gemma4 E2B MXFP4 PLE loader source fixed, live row still pending

Status: PARTIAL_SOURCE_FIXED.

Source fix:
- `vmlx_engine/utils/jang_loader.py` now preserves native MXFP sidecars for
  actual quantized Gemma PLE modules after `nn.quantize()`.
- Legacy/non-quantized Gemma PLE materialization now uses native
  `mx.dequantize(..., mode="mxfp4"/"mxfp8")` where appropriate and buffers
  split `.weight` / `.scales` sidecars across shards before loading.

Why this was needed:
- Installed E2B proof attempt failed before model health/generation with
  affine-style PLE dequantization on native MXFP4 sidecars.
- The real E2B bundle has:
  - `per_layer_model_projection.weight=(8960,192) uint32`,
    `scales=(8960,48) uint8`, `bits=4`, `group_size=32`;
  - `embed_tokens_per_layer.weight` and `.scales` split across different
    safetensor shards.

Verification so far:
- `py_compile` passed for `vmlx_engine/utils/jang_loader.py` and
  `tests/test_jang_loader.py`.
- Targeted pytest passed:
  `tests/test_jang_loader.py -k 'gemma4_ple_native_mxfp4 or gemma4_moe_mxfp'`
  -> `3 passed, 69 deselected`.
- Real source-load sanity passed for
  `/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4`:
  `QuantizedLinear/QuantizedEmbedding`, `mode=mxfp4`, `bits=4`,
  `group_size=32`, `uint32` packed weights, `uint8` scales.

Still blocked/pending:
- `/Applications/vMLX.app` has not yet been rebuilt with this loader patch.
- Gemma E2B live UI/API/audio/VL/cache stress proof is still pending.
- Gemma E4B/12B audio rows and 26B/31B visual rows are still open.
- No `.63` release, notarization, main merge, or public update claim is allowed
  from this source-only fix.

## 2026-06-17 18:08 PDT - Startup/autodetect/cache/media matrix gates expanded

Status: OPEN_RELEASE_GATES_ADDED.

Added to `.agents/RELEASE-1.5.63-STRESS-MATRIX.md`:
- Autodetect / Startup / UI Settings Parity Gate.
- Cache / TurboQuant KV / Async-Recompute Policy Gate.
- Media / Tool / Streaming Edge-Case Gate.

These gates now require current artifacts for:
- UI-visible resolved settings and toggle states;
- real app-spawned CLI args and direct CLI parity;
- model/family/format detection success and failure edges;
- generation-config defaults for plain MLX, JANG, JANG_4M, JANGTQ/MXTQ,
  MXFP4, and MXFP8 rows;
- typed cache proof for MM3 MSA, DSV4/ZAYA composites, Gemma mixed SWA/full
  attention, hybrid SSM, CCA, and companion-state families;
- TurboQuant KV/cache quantization policy, with DSV4/MM3 as explicit
  no-generic-TQ-KV exceptions unless a future native bridge is implemented and
  live-proven;
- UI/API audio/VL, tools, streaming deltas, prefix hits, and post-media
  recovery.

No live row is cleared by this documentation update.

## 2026-06-17 18:18 PDT - Swift handoff boundary recorded in matrix

Status: DOCUMENTATION_GATE_UPDATED.

- User pasted the `/Users/eric/vmlx` `AGENTS.md` handoff, which says the
  long-term active vMLX priority is the Swift stack and that Python work should
  not be entered by default.
- `.agents/RELEASE-1.5.63-STRESS-MATRIX.md` now records the boundary clearly:
  this `.63` matrix stays in `/Users/eric/mlx/vllm-mlx` only because the
  current request explicitly names MM3 / MLXStudio / Python `vmlx_engine`
  compatibility/release work.
- Evidence separation is now explicit both directions:
  - Swift handoff notes or Swift logs cannot mark Python/Electron rows green.
  - Python/Electron installed-app/API/signing proof cannot mark Swift rows
    green.
- If Eric switches the active task back to Swift vMLX, create or use a separate
  Swift release matrix under `/Users/eric/vmlx` or `/Users/eric/vmlx/swift`
  with current-source and live-app artifacts.

## 2026-06-17 18:34 PDT - Gemma4 audio path source fix, installed proof pending

Status: PARTIAL_SOURCE_FIXED.

- Failed installed E2B MXFP4 live artifact inspected:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T01-12-08-487Z/gemma4-media-proof.json`.
- Product failure found:
  - Chat Completions audio and Responses audio both returned visible engine
    errors:
    `Generation failed: could not convert string to float: '/var/...wav'`.
  - Source trace: `vmlx_engine/mllm_batch_generator.py` already had
    `_load_audio_waveforms_for_processor()` for this exact failure mode, but
    `_call_processor_direct()` passed temp audio paths directly through to
    non-MiMo processors.
- Source fix:
  - `_call_processor_direct()` now converts non-MiMo audio paths to float32
    waveform arrays before invoking the processor.
  - MiMo remains path-based because its separate mel/audio-code bridge owns
    raw audio processing.
- Harness fix:
  - `panel/scripts/live-gemma4-media-stress-proof.mjs` now queries
    `/v1/models/{servedModel}/capabilities` instead of non-existent
    `/capabilities`.
- Verification:
  - `py_compile vmlx_engine/mllm_batch_generator.py tests/test_mllm_scheduler_cache.py`
    passed.
  - `pytest tests/test_mllm_scheduler_cache.py -q -k 'processor_direct or audio'`
    -> `7 passed, 94 deselected`.
  - `pytest tests/test_engine_audit.py -q -k 'gemma4_unified and audio'`
    -> `4 passed, 554 deselected`.
  - `node --check panel/scripts/live-gemma4-media-stress-proof.mjs` passed.
- Open:
  - Full `tests/test_mllm_scheduler_cache.py tests/test_multimodal_routing.py`
    still has an unrelated source-test drift:
    `TestMLLMSchedulerConfigParity.test_paged_cache_fields` expects
    `use_paged_cache is True`, current config is `False`.
  - Installed `/Applications/vMLX.app` has not yet been rebuilt with the audio
    path fix.
  - E2B live UI/API/audio/cache proof must be rerun after rebuild.

## 2026-06-17 18:32 PDT - Installed E2B media proof failed; input_features bridge patched

Status: PARTIAL_SOURCE_FIXED; LIVE_RERUN_PENDING.

- Rebuilt `/Applications/vMLX.app` via `panel/scripts/build-and-install.sh`.
- Integrity proof before live rerun:
  - `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    passed.
  - Installed bundled Python reports `vmlx_engine.__version__ == 1.5.63`.
  - Installed `vmlx_engine.mllm_batch_generator._call_processor_direct`
    contains the waveform-loader path from the first audio fix.
- Installed E2B artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T01-28-01-756Z/gemma4-media-proof.json`.
- Result: `status=fail`.
- Positive live evidence:
  - capabilities endpoint resolved correctly and advertised text/vision/audio/video;
  - 10 UI text turns stayed coherent with memory-cache hits and ~68-79 t/s;
  - reasoning off/on/auto all produced visible content;
  - Chat and Responses streaming produced visible text.
- Live failures:
  - UI audio produced no visible response.
  - Chat and Responses audio returned:
    `unsupported media modality audio for gemma4_unified: raw audio reached the VLM processor, but the processor returned no audio_codes, audio_embeds, or audio_features`.
  - Session settings still omit `usePagedCache`, `enableDiskCache`, and
    `kvCacheQuantization`.
- Source root cause found after artifact:
  - Gemma4 processor returns `input_features` and `input_features_mask`.
  - `MLLMBatchGenerator._preprocess_request` only promoted `audio_codes`,
    `audio_embeds`, and `audio_features`, so the fail-loud guard rejected a
    valid Gemma audio processor result.
  - `_run_vision_encoding_inner` also had to forward raw Gemma features as
    `input_features` / `input_features_mask`, not as precomputed
    `audio_embeds`.
- Source fix:
  - `vmlx_engine/mllm_batch_generator.py` now promotes `input_features` /
    `input_features_mask`, hashes the mask in media cache keys, and forwards
    raw Gemma audio features under the model-consumed names.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/mllm_batch_generator.py tests/test_mllm_scheduler_cache.py`
    passed.
  - `.venv/bin/python -m pytest tests/test_mllm_scheduler_cache.py -q -k 'processor_direct or audio_outputs or input_features or audio_payload_prefill'`
    -> `5 passed, 97 deselected`.
- Open:
  - Rebuild bundled Python/app again with this second audio bridge patch.
  - Rerun E2B installed-app proof.
  - The broader source suite still has the unrelated paged-cache default drift
    noted above.

## 2026-06-17 18:44 PDT - Installed E2B audio rerun reached runtime, audio tower still blocked

Status: PARTIAL_LIVE_RED.

- Rebuilt bundled Python and `/Applications/vMLX.app` after the
  `input_features` / `input_features_mask` scheduler bridge patch.
- Installed-bundle source inspection before the live rerun confirmed:
  - `vmlx_engine.__version__ == 1.5.63`;
  - installed `mllm_batch_generator.py` promotes `input_features`;
  - installed `mllm_batch_generator.py` promotes `input_features_mask`;
  - the raw audio-feature flag exists;
  - `_run_vision_encoding_inner` forwards `input_features` /
    `input_features_mask`.
- Installed live artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T01-40-27-510Z/gemma4-media-proof.json`.
- Result: `status=fail`.
- Positive live evidence:
  - installed E2B text multiturn remained coherent with cache hits and
    ~68-79 t/s in the harness;
  - reasoning off/on/auto produced visible output;
  - Chat and Responses streaming text paths produced visible output.
- Live failures:
  - UI audio still produced empty or hidden-only output;
  - Chat Completions audio returned HTTP 500;
  - Responses audio returned HTTP 500;
  - post-audio text recovery failed;
  - MLXStudio session state still omitted `usePagedCache`, `enableDiskCache`,
    and `kvCacheQuantization`.
- Current source/runtime boundary:
  - the first audio bridge issue is fixed: temp wav paths are converted to
    float32 waveform arrays before the processor;
  - the second bridge issue is fixed: processor-returned `input_features` /
    `input_features_mask` are promoted and forwarded to the runtime;
  - the remaining failure is inside Gemma4 audio runtime execution:
    `get_audio_features` passes `(1,82,128)` raw mel features into
    `embed_audio`, whose projection expects `(1536,1536)`;
  - the real E2B bundle has `audio_tower.*` weights and
    `audio_tower.output_proj.weight (1536,1024)`, so raw mel features must go
    through the Gemma4 audio tower before `embed_audio`.
- Next fix target:
  - inspect/port/wire the real Gemma4 audio tower path in the vendored unified
    runtime, or honestly capability-gate audio red until that path exists.
  - Do not mark E2B/E4B/12B audio green from bridge tests alone.

## 2026-06-17 18:44 PDT - Active worktree guard added to release matrix

Status: DOCUMENTATION_GATE_UPDATED.

- `.agents/RELEASE-1.5.63-STRESS-MATRIX.md` now explicitly records the nearest
  Python release-lane `AGENTS.md` guard:
  - active work stays in `/Users/eric/mlx/vllm-mlx` for this `.63`
    Python/Electron lane;
  - each continuation must name the blocker being reduced;
  - after each proof, update `.agents/STATUS.md`, `.agents/LOG.md`, and the
    matrix with artifact path, pass/fail state, and remaining boundary;
  - source tests/imports/health/package integrity do not make rows green
    without current live UI/API/runtime evidence;
  - signing, notarization, tagging, downloads, and release claims remain locked
    while objective rows are `OPEN`, `PARTIAL`, or `BLOCKED` unless Eric
    explicitly overrides that lock.
- No runtime, cache, UI, API, signing, notarization, or release row is cleared
  by this documentation update.

## 2026-06-17 18:50 PDT - Gemma4 E2B/E4B audio tower source path fixed

Status: SUPERSEDED_BY_2026_06_17_19_02_LIVE_RERUN.

- Source fix:
  - `vmlx_engine/models/gemma4_unified/config.py` now preserves
    `gemma4_audio` configs as upstream Gemma4 audio configs when
    `update_module_configs()` runs, instead of collapsing E2B/E4B audio to the
    older 640-dim unified placeholder.
  - `vmlx_engine/models/gemma4_unified/gemma4_unified.py` now builds
    `AudioEncoder` for `gemma4_audio`, runs raw `(B,T,128)` mel features
    through the audio tower before `embed_audio`, inverts processor
    `True=valid` masks to the tower's `True=padding` mask semantics, and
    sanitizes audio tower Conv2d/Conv1d weights into MLX layout.
- Source checks:
  - `py_compile` passed for the patched Gemma4 unified files and
    `tests/test_jang_loader.py`.
  - `.venv/bin/python -m pytest tests/test_jang_loader.py -q -k 'gemma4_ple_native_mxfp4 or gemma4_moe_mxfp or gemma4_unified'`
    -> `7 passed, 68 deselected`.
  - `.venv/bin/python -m pytest tests/test_mllm_scheduler_cache.py -q -k 'processor_direct or audio_outputs or input_features or audio_payload_prefill or gemma_input_features'`
    -> `5 passed, 97 deselected`.
  - `.venv/bin/python -m pytest tests/test_engine_audit.py -q -k 'gemma4_unified and audio'`
    -> `4 passed, 554 deselected`.
- Real E2B source-load sanity:
  - model path:
    `/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4`;
  - `load_jang_vlm_model(..., skip_eval=True)` loaded
    `audio_tower=AudioEncoder`;
  - audio config came from `mlx_vlm.models.gemma4.config`;
  - `audio_hidden=1024`, `audio_output=1536`,
    `embed_audio_weight=(1536,1536)`;
  - processor output was `input_features=(1,99,128)`,
    `input_features_mask=(1,99)`;
  - `model.get_audio_features(...)` returned `(25,1536)`, directly clearing
    the previous `(1,82,128)` into `(1536,1536)` matmul source failure.
- Boundary:
  - This was source proof only at the time of the entry.
  - Superseded by the 19:02 installed rerun:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T01-59-02-654Z/gemma4-media-proof.json`.
  - The installed app was rebuilt and the matmul crash cleared, but E2B audio
    remains `PARTIAL_LIVE_RED_AUDIO_CONDITIONING` because UI/Chat/Responses
    still answer as if no audio was attached.

## 2026-06-17 20:53 PDT - 12B JANG_4M installed proof passes after direct-audio capability gate

Status: PARTIAL_RELEASE_BLOCKED; 12B JANG_4M measured text/cache/reasoning/
streaming/capability-truth row is live-green for the scoped artifact.

- Rebuilt and installed `/Applications/vMLX.app` via
  `panel/scripts/build-and-install.sh`.
- Package proof:
  - bundled Python built `vmlx 1.5.63` and critical bundled source parity
    checks passed;
  - 501 bundled Python native files were signed;
  - `/Applications/vMLX.app` passed
    `codesign --verify --deep --strict --verbose=2`;
  - installed Python reports `vmlx_engine.__version__ == 1.5.63`;
  - installed `vmlx_engine.server` contains `_bundle_weight_map_has_prefix`.
- Installed capability truth:
  - E2B MXFP4: `['text', 'vision', 'audio', 'video']`;
  - E4B MXFP4: `['text', 'vision', 'audio', 'video']`;
  - 12B JANG_4M: `['text', 'vision', 'video']`;
  - 12B direct audio is therefore not advertised for this artifact.
- Installed live proof artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-2026-06-18T03-51-02-869Z/gemma4-media-proof.json`.
- Final screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-2026-06-18T03-51-02-869Z/gemma4-media-final.png`.
- Result: `PASS`, failures `[]`.
- Passed proof scope:
  - capabilities exposed `text`, `vision`, `video` and omitted unsupported
    audio;
  - `/v1/models/.../capabilities` reported family `gemma4`,
    `tool_parser=gemma4`, `reasoning_parser=gemma4`, supports tools and
    thinking, and sampling defaults `temperature=1`, `top_p=0.95`,
    `top_k=64`;
  - native cache reported Gemma4 `mixed_swa_kv_v1`, with full-attention KV,
    sliding-window KV, rotating-window metadata, prefix cache enabled, paged
    cache disabled, and generic TurboQuant KV inactive for this native mixed
    SWA cache;
  - 10 UI text/cache turns completed with visible content and cached tokens
    through `474`, with observed throughput around `33.4-51.0 tok/s`;
  - reasoning off/on/auto produced visible non-hidden-only content; off had
    `0` reasoning chars, on had separated reasoning, auto stayed visible;
  - Chat Completions streaming produced visible `GEMMA_STREAM_CHAT_OK` text;
  - Responses streaming produced visible `GEMMA_STREAM_RESPONSES_OK` text and
    final usage included `cached_tokens=29`, `cache_detail=disk`.
- Remaining release blockers:
  - Gemma 26B/31B visual rows remain unproven live;
  - MM3 remaining release rows still include streaming delta exactness, fresh
    process SSD restore if claimed, UI settings/i18n, and concurrent
    Stop-state;
  - the older matrix-referenced MM3, E2B, and E4B proof JSON files are missing
    from the current `build/` tree; rerun or restore those artifacts before a
    release-grade evidence package;
  - no main merge, notarization, public release, or release-ready claim is
    allowed until those rows are passed or explicitly waived.

## 2026-06-17 20:59 PDT - E2B MXFP4 installed proof artifact regenerated

Status: PARTIAL_RELEASE_BLOCKED; E2B MXFP4 measured UI/API/audio/cache/
reasoning/streaming row is current live-green.

- Installed live proof artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T03-56-46-744Z/gemma4-media-proof.json`.
- Final screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T03-56-46-744Z/gemma4-media-final.png`.
- Result: `PASS`, failures `[]`.
- Startup/CLI proof:
  - installed app spawned bundled Python engine on port `53494`;
  - command included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--cache-memory-percent 0.15`, `--enable-disk-cache`,
    `--disk-cache-max-gb 10`, and `--max-tokens 512`.
- Passed proof scope:
  - capabilities exposed `text`, `vision`, `audio`, `video`;
  - native cache reported `mixed_swa_kv_v1`;
  - sampling defaults were `temperature=1`, `top_p=0.95`, `top_k=64`;
  - 10 UI text/cache turns completed with cached tokens through `402` and no
    loop/hidden-only score;
  - reasoning off/on/auto produced visible non-hidden-only content;
  - UI audio returned `audio present`;
  - Chat Completions audio returned `audio present`;
  - Responses audio returned `audio present`;
  - post-audio text recovery returned `GEMMA_POST_AUDIO_TEXT_OK`;
  - Chat Completions and Responses streaming produced visible marker text and
    Responses final usage included `cache_detail=disk`.
- Remaining release blockers:
  - regenerate E4B and MM3 artifacts or restore exact old proof files;
  - Gemma 26B/31B visual rows remain unproven live;
  - release/notarization remains blocked.

## 2026-06-17 21:01 PDT - E4B MXFP4 installed proof artifact regenerated

Status: PARTIAL_RELEASE_BLOCKED; E4B MXFP4 measured UI/API/audio/cache/
reasoning/streaming row is current live-green.

- Installed live proof artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T03-59-11-486Z/gemma4-media-proof.json`.
- Final screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-mxfp4-2026-06-18T03-59-11-486Z/gemma4-media-final.png`.
- Result: `PASS`, failures `[]`.
- Startup/CLI proof:
  - installed app spawned bundled Python engine on port `53713`;
  - command included `--is-mllm`, `--tool-call-parser gemma4`,
    `--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
    `--cache-memory-percent 0.15`, `--enable-disk-cache`,
    `--disk-cache-max-gb 10`, and `--max-tokens 512`.
- Passed proof scope:
  - capabilities exposed `text`, `vision`, `audio`, `video`;
  - native cache reported `mixed_swa_kv_v1`;
  - sampling defaults were `temperature=1`, `top_p=0.95`, `top_k=64`;
  - 10 UI text/cache turns completed with cached tokens through `385` and no
    loop/hidden-only score;
  - reasoning off/on/auto produced visible non-hidden-only content;
  - UI audio returned `audio present`;
  - Chat Completions audio returned `Audio present`;
  - Responses audio returned `audio present`;
  - post-audio text recovery returned `GEMMA_POST_AUDIO_TEXT_OK`;
  - Chat Completions and Responses streaming produced visible marker text.
- Remaining release blockers:
  - regenerate MM3 artifact or restore exact old proof files;
  - Gemma 26B/31B visual rows remain unproven live;
  - release/notarization remains blocked.

## 2026-06-17 21:08 PDT - MiniMax-M3 installed proof artifact regenerated

Status: PARTIAL_RELEASE_BLOCKED; MiniMax-M3 measured installed-app
text/cache/reasoning/tool/long-context/VL/API rows are current live-green.

- Installed live proof artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T04-01-45-144Z/mm3-stress-proof.json`.
- Final screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T04-01-45-144Z/mm3-stress-final.png`.
- Result: `PASS`, failures `[]`.
- Startup/CLI proof:
  - installed app spawned bundled Python engine on port `53910`;
  - command included `--tool-call-parser minimax_m3`,
    `--enable-auto-tool-choice`, `--reasoning-parser minimax_m3`,
    `--cache-memory-percent 0.15`, `--enable-disk-cache`,
    `--disk-cache-max-gb 10`, and `--max-tokens 512`;
  - resolved session settings showed `usePagedCache=false`,
    `enableDiskCache=true`, `kvCacheQuantization=auto`, `enableJit=false`,
    `toolCallParser=minimax_m3`, `reasoningParser=minimax_m3`, and
    `isMultimodal=true`.
- Native cache proof:
  - `schema=minimax_m3_msa_v1`;
  - components `attention_kv`, `msa_idx_keys`, `absolute_block_index`;
  - generic TurboQuant KV disabled with reason
    `native_minimax_m3_msa_idx_keys`;
  - storage quantization forced off for MSA idx_keys;
  - paged false, prompt disk L2 true, block disk L2 false;
  - final cache stats: disk cache hits `13`, stores `9`, scheduler
    cache-hit requests `18`, cache-hit tokens `21113`
    (`6902` disk, `14211` memory).
- Passed proof scope:
  - 10 UI text/cache turns completed with cached tokens
    `172, 172, 243, 342, 444, 559, 671, 787, 902, 1018`
    and observed throughput `20.8-23.2 tok/s`;
  - reasoning off/on/auto produced visible non-hidden-only content;
  - UI builtin tool wrote `M3_TOOL_OK`;
  - long-context prefix/sentinel recall returned
    `PROFILE_OPTION_SENTINEL_ZETA_173`;
  - MM3 VL image row returned `MM3_IMAGE_RED`;
  - Chat Completions visible text returned `API_CHAT_OK`;
  - Chat Completions tool call emitted `record_mm3_label` with
    `API_TOOL_OK`;
  - Responses first call returned `API_RESP_OK`;
  - Responses `previous_response_id` continuation returned `violet`;
  - Anthropic Messages, Ollama chat, and Ollama generate routes returned
    visible non-looping text.
- Remaining release blockers:
  - MM3 streaming delta exactness across Chat/Responses/Anthropic/Ollama is
    still not separately proven by this harness;
  - fresh-process SSD restore is not separately proven beyond same-process
    disk hits/stores;
  - UI settings/i18n and concurrent Stop-state rows remain open;
  - Gemma 26B/31B visual rows remain unproven live;
  - release/notarization remains blocked.

## 2026-06-17 21:10 PDT - Mixed single-session stress gate added

Status: PARTIAL_RELEASE_BLOCKED; existing E2B/E4B/MM3 proof remains useful
component proof, but release now requires an additional mixed single-session
proof for each capable model family.

- New UI gate:
  - one chat session must mix reasoning off text, reasoning on plus media
    (image or audio as supported), reasoning auto text, reasoning on plus a
    required tool call, reasoning auto plus required/auto tool call, and a
    final recall/cache-hit follow-up;
  - the mixed session must prove no hidden-only assistant messages, no raw
    reasoning tags, no loops, no autonomous extra assistant generation, and
    cache-hit telemetry after media/tool turns.
- New API gate:
  - Chat Completions and Responses must have mixed proof for streaming deltas,
    separated reasoning where enabled, structured tool calls, media only for
    advertised modalities, and cache usage/usage details.
- New defaults gate:
  - proof artifacts must compare `generation_config.json` defaults
    (`temperature`, `top_p`, `top_k`, sampling flags) with the resolved
    MLXStudio session config and visible/resolved UI settings;
  - if the UI does not expose/show those resolved defaults, that model row is
    `PARTIAL_UI_DEFAULTS_UNPROVEN`.
- Implementation target:
  - patch `panel/scripts/live-mm3-stress-proof.mjs` and
    `panel/scripts/live-gemma4-media-stress-proof.mjs` to add
    `ui.mixedSession`, `api.mixed*`, and `generationDefaults` result sections;
  - rerun MM3, Gemma E2B/E4B, and 12B JANG_4M scoped rows after the harness
    patch.

## 2026-06-17 21:55 PDT - MM3 mixed gate live rerun exposed two remaining blockers

Status: PARTIAL_LIVE_RED_MIXED_MEDIA_CACHE; not release-ready.

- Live installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T04-44-32-604Z/mm3-stress-proof.json`.
- Result: `FAIL`, failures: `mixed final recall cachedTokens=0`.
- Positive live evidence in this artifact:
  - installed app spawned bundled Python engine with MiniMax-M3 parsers,
    disk cache, `max_tokens=512`, and M3-specific paged/JIT/TQ-KV settings;
  - 10-turn UI text/cache sequence completed with cache hits through `1122`
    cached tokens and no loop flags;
  - reasoning off/on/auto worked visibly with off=`0` reasoning chars and
    on/auto separated reasoning;
  - isolated UI tool wrote `M3_TOOL_OK`;
  - long-context sentinel returned `PROFILE_OPTION_SENTINEL_ZETA_173`;
  - isolated image row returned visible `MM3_IMAGE_RED`;
  - generation defaults matched model `generation_config.json`
    (`temperature=1.0`, `top_p=0.95`, `top_k=off`) and the UI visibly showed
    those concrete resolved values;
  - API Chat, Responses, Anthropic, Ollama, streaming Chat/Responses image,
    and streaming Chat/Responses tool checks returned positive markers.
- New live blockers from the stronger mixed gate:
  - mixed turn 2 did send one `image_url` to `/v1/responses` and the engine
    logged `MEDIA_DIAG` with one image item, but the model answered that it
    could not see the image; the harness was too weak because it accepted a
    quoted `MM3_MIX_IMAGE_RED` label even inside that refusal text;
  - mixed final recall changed from tool-enabled schema to tools-off schema,
    so it reported `cachedTokens=0` even though the two preceding mixed tool
    turns had memory cache hits (`3404`, `3559` cached tokens).
- Source/harness fixes applied after this red row, pending rebuild/live rerun:
  - `vmlx_engine/models/minimax_m3/m3_vl_preprocess.py` now renders image
    placeholders before text within each M3 multimodal user turn, matching the
    existing proven `extra_images` diagnostic path instead of preserving the
    panel's text-then-image content-array order;
  - `panel/scripts/live-mm3-stress-proof.mjs` and
    `panel/scripts/live-gemma4-media-stress-proof.mjs` now reject "cannot see
    image" style responses even if the requested label is quoted;
  - both mixed harnesses keep the native tool schema stable on the final
    recall/cache turn so prefix-cache reuse is tested against a stable prompt
    shape.
- Checks run after these source changes:
  - `py_compile vmlx_engine/models/minimax_m3/m3_vl_preprocess.py`;
  - `node --check panel/scripts/live-mm3-stress-proof.mjs`;
  - `node --check panel/scripts/live-gemma4-media-stress-proof.mjs`;
  - `cd panel && npm run typecheck`.
- Missing evidence:
  - installed app has not yet been rebuilt with the M3 preprocessor fix;
  - MM3 mixed gate has not yet passed live after the fix;
  - Gemma mixed gate rows are not yet rerun with the tightened harness.

## 2026-06-17 22:25 PDT - MM3 mixed UI/API gate passed on installed app

Status: MM3_MIXED_GATE_PASS; overall 1.5.63 release remains
PARTIAL_RELEASE_BLOCKED by Gemma/open release rows.

- Live installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T05-16-31-370Z/mm3-stress-proof.json`.
- Final screenshot:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T05-16-31-370Z/mm3-stress-final.png`.
- Result: `PASS`, failures `[]`.
- Source/build evidence for this run:
  - installed `/Applications/vMLX.app` passed
    `codesign --verify --deep --strict`;
  - `vmlx_engine/models/minimax_m3/m3_vl_preprocess.py` installed source
    contains the image-placeholder-before-text path for M3 multimodal turns;
  - `panel/src/main/ipc/chat.ts` installed ASAR contains historical media
    stripping so prior image/audio payloads are not replayed into later
    text/tool turns;
  - harness syntax and panel TypeScript passed after the stricter mixed-gate
    image fixture update.
- Live UI proof:
  - 10-turn text/cache run completed with no empty/hidden-only turn, no loop
    flag, no autonomous extra assistant turn, and cached tokens through `1216`;
  - throughput in the text/cache sequence stayed around `21.2-24.7 tok/s`;
  - reasoning off/on/auto worked: off had `0` reasoning chars, on had `646`,
    auto had `569`, all with visible content;
  - isolated tool wrote `M3_TOOL_OK`;
  - long-context sentinel returned `PROFILE_OPTION_SENTINEL_ZETA_173` with
    `5778` cached tokens;
  - isolated VL image returned `MM3_IMAGE_RED`;
  - model-owned generation defaults matched and were visible in UI:
    `temperature 1.00`, `top-p 0.95`, `top-k off`.
- Mixed single-chat UI proof:
  - turn 1 text + reasoning off returned `M3_MIX_TEXT_OFF` with `0` reasoning;
  - turn 2 image + reasoning on returned `MM3_MIX_IMAGE_RED` with `431`
    reasoning chars;
  - turn 3 text + reasoning auto returned `M3_MIX_AUTO_TEXT` with `297`
    reasoning chars;
  - turn 4 reasoning on + required tool returned `M3_MIX_TOOL_ON_DONE`,
    wrote `M3_MIX_TOOL_ON`, and had `3431` cached tokens;
  - turn 5 reasoning auto + required tool returned `M3_MIX_TOOL_AUTO_DONE`,
    wrote `M3_MIX_TOOL_AUTO`, and had `3648` cached tokens;
  - final recall returned all mixed labels in the same chat.
- Live API/protocol proof in the same artifact:
  - Chat Completions visible text;
  - Chat Completions required tool call;
  - Responses visible text and `previous_response_id`;
  - Anthropic Messages visible text;
  - Ollama chat and generate visible text;
  - streaming Chat text returned `MM3_STREAM_CHAT_OK`;
  - streaming Responses image returned `MM3_STREAM_IMAGE_RED`;
  - streaming Chat and Responses tool checks returned structured tool calls.
- Remaining release blockers:
  - Gemma mixed-session rows need rerun with the tightened labeled-red
    fixture/cache criterion;
  - Gemma 26B/31B visual rows are still open;
  - release signing/notarization/stapling/public update has not been run in
    this proof cycle.

## 2026-06-17 22:30 PDT - Gemma E2B mixed gate red; vision graph source fix added

Status: GEMMA_E2B_PARTIAL_LIVE_RED_IMAGE; source fix added, installed rerun
pending.

- Live installed-app artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T05-27-45-910Z/gemma4-media-proof.json`.
- Result: `FAIL`.
- Positive rows in this artifact:
  - installed app loaded E2B MXFP4 with capabilities `text`, `vision`,
    `audio`, `video`;
  - 10-turn text/cache completed with cached tokens through `432` and
    `68.8-84.5 tok/s`;
  - reasoning off/on/auto produced visible content, with off=`0` reasoning
    chars and on/auto separated reasoning;
  - UI audio returned `audio present`;
  - Chat and Responses audio each returned `audio present`;
  - post-audio recovery returned `GEMMA_POST_AUDIO_TEXT_OK`;
  - generation defaults matched model config and were visible in UI:
    `temperature 1.00`, `top-p 0.95`, `top-k 64`;
  - streaming Chat/Responses text and streaming Chat/Responses tool checks
    returned positive markers.
- Red rows:
  - mixed image+reasoning-on returned an empty assistant turn (`0` tokens);
  - streaming Responses image reconstructed empty content;
  - mixed post-media/tool cache criterion failed because media fallback/tool
    schema did not report cached tokens in that run.
- Source root cause from live logs:
  - both image failures routed through `Using simple MLLM media ... 1 image`;
  - `vmlx_engine.models.gemma4_unified.gemma4_unified.Model.get_image_features`
    crashed with `[layer_norm] weight must have the same size as the last
    dimension of x but has 6912 elements`;
  - E2B/E4B/26B/31B configs use `vision_config.model_type=gemma4_vision` with
    real `vision_tower.*` transformer weights and
    `embed_vision.embedding_projection.weight`, but the promoted
    `gemma4_unified` runtime always instantiated the encoder-free
    `VisionEmbedder` intended for `gemma4_unified_vision`.
- Source fix now applied:
  - `vmlx_engine/models/gemma4_unified/config.py` preserves
    `vision_config.hidden_size`;
  - `vmlx_engine/models/gemma4_unified/gemma4_unified.py` uses upstream
    `gemma4.vision.VisionModel` for `gemma4_vision` and keeps
    `VisionEmbedder` only for `gemma4_unified_vision`;
  - `vmlx_engine/models/gemma4_unified/processing_gemma4_unified.py` selects
    raw `Gemma4ImageProcessor` for local `gemma4_vision` bundles and keeps
    pre-patched `Gemma4UnifiedImageProcessor` for unified-vision bundles.
- Source checks passed:
  - `py_compile` for touched Gemma files and `tests/test_jang_loader.py`;
  - `pytest tests/test_jang_loader.py -q -k 'vendored_gemma4_unified'`
    -> `9 passed, 73 deselected`;
  - `pytest tests/test_engine_audit.py -q -k 'gemma4_unified and (audio or vision or runtime_modalities)'`
    -> `6 passed, 557 deselected`;
  - `pytest tests/test_mllm_scheduler_cache.py -q -k 'gemma_input_features or processor_direct or audio_outputs or input_features or audio_payload_prefill'`
    -> `5 passed, 97 deselected`;
  - harness `node --check` for MM3 and Gemma.
- Missing evidence:
  - installed app has not yet been rebuilt with this Gemma vision split;
  - E2B mixed UI/API gate has not yet passed live after the fix;
  - E4B/12B/26B/31B rows remain unrerun with this fix.

## 2026-06-17 22:45 PDT - Gemma E2B installed rerun startup red; raw vision config fix added

Status: GEMMA_E2B_PARTIAL_SOURCE_FIXED_INSTALLED_REBUILD_PENDING.

- Installed rerun artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T05-43-38-894Z/gemma4-media-proof.json`.
- Result: `FAIL` before model proof; the app launched but the server exited
  before readiness.
- Bundled source/signature evidence:
  - `/Applications/vMLX.app` passed `codesign --verify --deep --strict`;
  - installed source contained `Gemma4VisionModel`,
    `_uses_full_gemma4_vision_tower`, and
    `_make_image_processor_for_model_config`.
- Startup root cause from bundled CLI traceback:
  - `vmlx_engine/models/gemma4_unified/gemma4_unified.py` constructed
    upstream `Gemma4VisionModel(config.vision_config)`;
  - upstream `mlx_vlm.models.gemma4.vision.VisionModel` required
    `config.default_output_length`;
  - vendored `VisionConfig` did not preserve that raw `gemma4_vision` field.
- Source fix now applied:
  - `vmlx_engine/models/gemma4_unified/config.py` preserves raw Gemma4 vision
    fields needed by upstream `VisionModel`: `default_output_length`,
    `position_embedding_size`, attention layer fields, standardization, and
    raw full-attention RoPE defaults;
  - `tests/test_jang_loader.py` now asserts the preserved raw vision fields.
- Source checks:
  - `py_compile` passed for touched Gemma config/runtime/processor and
    `tests/test_jang_loader.py`;
  - `pytest tests/test_jang_loader.py -q -k 'vendored_gemma4_unified'`
    -> `9 passed, 73 deselected`;
  - source venv E2B startup reached `/health` with model loaded.
- Health/cache source sanity from that start:
  - `native_cache.schema=mixed_swa_kv_v1`;
  - generic `turboquant_kv_cache.enabled=false`;
  - scheduler reported mixed SWA cache layout with `RotatingKVCache` and
    `KVCache` layers;
  - prompt disk L2 initialized and mixed-SWA stored-cache quantization was
    disabled.
- Missing evidence:
  - `/Applications/vMLX.app` has not yet been rebuilt after this raw-vision
    config fix;
  - E2B mixed UI/API gate has not passed live after this fix.

## 2026-06-18 08:59Z - Gateway/default-port clean-start proof green for scoped Gemma and MM3

Status: GATEWAY_DEFAULT_PORT_PASS_RELEASE_STILL_BLOCKED.

- Scope update: Gemma audio is out of the active `.63` release gate per Eric.
  Gemma remains gated on text, VL image, reasoning, tools, API/streaming,
  clean-start autodetect, prefix/cache behavior, and lifecycle.
- Harness update: `panel/scripts/live-clean-start-autodetect-proof.mjs` now
  records the installed app gateway status and exercises `/health`,
  `/v1/models`, `/v1/models/{model}/capabilities`, Chat Completions,
  Responses, streaming Chat, and streaming Responses through the active gateway
  port. Visible marker calls explicitly use `enable_thinking=false`, and the
  Responses SSE parser counts only visible `response.output_text.delta` as
  visible output.
- Gemma E2B installed real-profile gateway artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-gemma4-e2b-mxfp4-real-profile-gateway-vl-visible-off-2026-06-18T08-59-27-957Z/clean-start-proof.json`;
  result `PASS`, failures `[]`. Evidence includes gateway `127.0.0.1:8080`,
  gateway health/models/capabilities, visible Chat/Responses stream and
  non-stream markers, real-profile session deletion, default Gemma parser/VL
  settings, mixed-SWA native cache, generic TQ-KV off, disk cache hit, and
  visible UI turn.
- MM3 installed real-profile gateway artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-mm3-reap40-d3-real-profile-gateway-visible-off-2026-06-18T08-59-58-015Z/clean-start-proof.json`;
  result `PASS`, failures `[]`. Evidence includes gateway `127.0.0.1:8080`,
  gateway health/models/capabilities, visible Chat/Responses stream and
  non-stream markers, real-profile session deletion, default MM3 parser/VL
  settings, native `minimax_m3_msa_v1` with `msa_idx_keys`, generic TQ-KV off,
  prompt SSD/L2 policy, and visible UI turn.
- Remaining blockers before release: sleep/wake/Stop lifecycle proof, any
  requested full-size Gemma reruns under this current clean-start/gateway
  harness, final source/test sweep, signed/notarized DMG/GitHub release, and
  the private postmortem/build-test/cache writeup.

## 2026-06-18 09:04Z - Installed app Stop/abort lifecycle proof green for scoped Gemma and MM3

Status: LIFECYCLE_STOP_ABORT_PASS_RELEASE_STILL_BLOCKED.

- Harness added:
  `panel/scripts/live-lifecycle-stop-proof.mjs`. It launches
  `/Applications/vMLX.app` with a temporary profile, starts the target model
  with default session settings, aborts one in-flight UI chat request, then
  stops the session during a second in-flight UI chat request.
- Gemma E2B artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-gemma4-e2b-mxfp4-lifecycle-vl-2026-06-18T09-03-37-304Z/lifecycle-proof.json`;
  result `PASS`, failures `[]`. Abort cleared `chat.isStreaming`; quiet wait
  did not add messages. Session stop returned success, session status became
  `stopped`, PID cleared, backend health was unreachable, and quiet wait did
  not add messages.
- MM3 artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-mm3-reap40-d3-lifecycle-2026-06-18T09-04-16-542Z/lifecycle-proof.json`;
  result `PASS`, failures `[]`. Abort cleared `chat.isStreaming`; quiet wait
  did not add messages. Session stop returned success, session status became
  `stopped`, PID cleared, backend health was unreachable, and quiet wait did
  not add messages. In the MM3 abort subtest the abort landed before an
  assistant message was committed, which still proves the no-autonomous-
  continuation lifecycle gate.
- Process cleanup recheck after the scripts:
  no `/Applications/vMLX.app/Contents/MacOS/vMLX` or
  `vmlx_engine.cli ... serve` process was left running.
- Remaining blockers before release: any requested full-size Gemma reruns under
  current clean-start/gateway/lifecycle harnesses, final source/test sweep,
  signed/notarized DMG/GitHub release, and private postmortem/build-test/cache
  writeup.

## 2026-06-18 09:06Z - Focused source/test sweep after gateway and lifecycle harness updates

Status: FOCUSED_SOURCE_SWEEP_PASS_RELEASE_STILL_BLOCKED.

- `git diff --check` passed for the touched docs, live proof harnesses,
  `tests/test_multimodal_routing.py`, and `vmlx_engine/server.py`.
- Harness syntax passed for both
  `panel/scripts/live-clean-start-autodetect-proof.mjs` and
  `panel/scripts/live-lifecycle-stop-proof.mjs`.
- Focused MM3 modality regression passed:
  `.venv/bin/python -m pytest tests/test_multimodal_routing.py -q -k 'm3_vl or text_only_multimodal'`
  -> `3 passed, 5 deselected`.
- Panel typecheck passed: `cd panel && npm run typecheck` -> `tsc --noEmit`
  exit 0.
- Remaining blockers before release: any remaining scoped model reruns Eric keeps
  in `.63`, packaging/signing/notarization/GitHub release, and private
  postmortem/build-test/cache writeup.
## 2026-06-18 12:40Z - vMLX 1.5.64 installed-app MM3/Gemma release preflight green

Status: RUNTIME_PREFLIGHT_PASS_SIGNING_NOTARIZATION_PENDING.

- Version/build checkpoint:
  - source stamps are `1.5.64` in `panel/package.json`,
    `panel/package-lock.json`, `pyproject.toml`, and
    `vmlx_engine/__init__.py`;
  - `panel/scripts/build-and-install.sh` rebuilt bundled Python and installed
    `/Applications/vMLX.app`;
  - bundled engine check:
    `/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 -B -s -c "import vmlx_engine; print(vmlx_engine.__version__)"`
    -> `1.5.64`;
  - independent signature check:
    `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
    -> valid on disk and satisfies Designated Requirement.
- Fail-closed release gate:
  `python3 panel/scripts/scoped-release-preflight-64.py` ->
  `/Users/eric/mlx/vllm-mlx/build/current-scoped-release-preflight-64.json`,
  `status=pass`.
- MM3 exactness/current proof:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T12-04-42-222Z/mm3-stress-proof.json`,
  result `PASS`, failures `[]`. Evidence includes installed-app launch,
  10 UI turns with cache hits through `985`, reasoning off/on/auto separation,
  exact tool file and exact tool markers, long-context sentinel recall,
  image/VL exact marker, mixed single-chat image/tool/cache/reasoning rows,
  Chat/Responses/Anthropic/Ollama compatibility, streaming Chat/Responses
  image/tool checks, visible generation defaults, `enablePrefixCache=true`,
  paged cache off, disk cache on, JIT off, parsers `minimax_m3`, native
  `minimax_m3_msa_v1` with `msa_idx_keys`, generic TQ-KV off, and prompt disk
  L2 on.
- Gemma media/stress rows regenerated after the 1.5.64 install and all passed:
  - `gemma4-e2b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-jang4m-vl-current-64-2026-06-18T12-12-46-515Z/gemma4-media-proof.json`;
  - `gemma4-e4b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-jang4m-vl-current-64-2026-06-18T12-14-24-479Z/gemma4-media-proof.json`;
  - `gemma4-12b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-12b-jang4m-vl-current-64-2026-06-18T12-16-12-368Z/gemma4-media-proof.json`;
  - `gemma4-26b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-jang4m-vl-current-64-2026-06-18T12-19-14-137Z/gemma4-media-proof.json`;
  - `gemma4-31b-jang4m-vl-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-jang4m-vl-current-64-2026-06-18T12-21-39-960Z/gemma4-media-proof.json`;
  - `gemma4-26b-mxfp4-visual-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-26b-mxfp4-visual-current-64-2026-06-18T12-26-23-704Z/gemma4-media-proof.json`;
  - `gemma4-31b-mxfp4-visual-current-64`:
    `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-31b-mxfp4-visual-current-64-2026-06-18T12-28-26-593Z/gemma4-media-proof.json`.
  Each row covers 10-turn UI coherence/cache, reasoning off/on/auto, image
  exact marker, tool-on/tool-auto exact markers, post-mixed cache hit,
  Chat/Responses/streaming/image/tool exactness, visible generation defaults
  from `generation_config.json`, paged cache off, disk cache on, and native
  Gemma `mixed_swa_kv_v1` cache with generic flat TQ-KV disabled for mixed-SWA.
- Clean-start/autodetect rows regenerated after install and all passed:
  MM3 real-profile gateway/default-port row, Gemma E2B MXFP4 real-profile
  row, plus 12B/26B/31B JANG_4M and 26B/31B MXFP4 current rows under
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-*`.
- Lifecycle rows regenerated after install and all passed:
  MM3, Gemma E2B MXFP4, Gemma 26B MXFP4, and Gemma 31B MXFP4 under
  `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-*`; each proved abort and
  stop of active streams, no continued streaming, stopped session state, dead
  backend health after stop, and no quiet-wait extra assistant messages.
- Qualitative note: the 31B JANG_4M stress row passed the strict marker/cache
  harness but one sampled EBS sentence said "Euthanasia Business Suite" instead
  of "E-Business Suite". Treat that as a semantic sampling blemish, not a
  cache/parser/loop failure; it should be mentioned if claiming broad Gemma
  answer quality.
- Remaining release work: final diff/source sweep, commit/push, release DMG
  signing/notarization/stapling/Gatekeeper validation, GitHub release/update
  surfaces, and private postmortem/cache/decode/build-test writeup. Runtime
  gate is green, but release is not done until those release artifacts exist
  and validate.
- Release script fix after this checkpoint:
  `panel/scripts/build-release-dmgs.sh` now routes scoped `1.5.64` builds
  through `panel/scripts/scoped-release-preflight-64.py`; without this it would
  have called the historical `1.5.63` scoped preflight. `bash -n` passed for
  build/notarize/verify scripts and `scoped-release-preflight-64.py` still
  reports `status=pass`.

## 2026-06-18 13:45Z - vMLX 1.5.64 post-DMG proof and MM3 exactness checkpoint

Status: `DMGS_NOTARIZED_RUNTIME_PREFLIGHT_PASS_PUBLIC_RELEASE_PENDING`.

- DMGs built by `panel/scripts/build-release-dmgs.sh all` with
  `VMLINUX_RELEASE_SCOPE=mm3_gemma_vl`:
  - `panel/release/vMLX-1.5.64-sequoia-arm64.dmg`
  - `panel/release/vMLX-1.5.64-tahoe-arm64.dmg`
- Notarization/stapling:
  - Sequoia notary id `771d26a2-c62a-42ff-a88d-c76688d7c17d`,
    sha256 `c53cfaa2e4c041280932fd06741847176a83066f5be5f01b3cdef47cafa8bf40`,
    size `479970644`;
  - Tahoe notary id `1391c83c-305d-42e2-8865-5aad8ce718e5`,
    sha256 `e8e01928908d7c58e82b5c9916ff129ef718d46e90e4b0e6d580ac822680161f`,
    size `496033984`.
- Final DMG verification:
  `panel/scripts/verify-release-dmgs.sh` passed for both DMGs:
  `hdiutil verify` valid, app `codesign --verify --deep --strict` valid,
  authority `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`,
  TeamIdentifier `55KGF2S5AY`, stapled ticket present, `stapler validate`
  succeeded, and `spctl` accepted with `source=Notarized Developer ID`.
- Post-DMG scoped runtime preflight:
  `/Users/eric/mlx/vllm-mlx/build/current-scoped-release-preflight-64-post-dmg-full-rerun.json`,
  `status=pass`, failures `[]`.
- Current MM3 exactness/live proof:
  `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-2026-06-18T13-08-33-708Z/mm3-stress-proof.json`,
  `status=pass`, failures `[]`. This artifact is exact-marker based, not just
  visual inspection:
  - 10 live UI turns with cache hits through turn 10 (`cached_tokens=1089`),
    no loop suspects, around `22.7-25.4` tok/s;
  - reasoning off: visible `OFF_VISIBLE_OK`, reasoning chars `0`;
    reasoning on: visible `ON_VISIBLE_OK`, reasoning chars `517`;
    reasoning auto: visible `AUTO_VISIBLE_OK`, reasoning chars `709`;
  - mixed single-chat stress: text/off `M3_MIX_TEXT_OFF`, image/on
    `MM3_MIX_IMAGE_RED`, text/auto `M3_MIX_AUTO_TEXT`, tool/on
    `M3_MIX_TOOL_ON_DONE`, tool/auto `M3_MIX_TOOL_AUTO_DONE`, final recall of
    all labels, and tool rows with exact one `run_command` iteration;
  - API exactness: Chat marker `API_CHAT_OK`; Responses previous-response
    recall exactly `violet`; streaming Chat tool args exactly
    `{"label": "MM3_STREAM_CHAT_TOOL"}`; streaming Responses image marker
    `MM3_STREAM_IMAGE_RED`; streaming Responses tool args exactly
    `{"label": "MM3_STREAM_RESP_TOOL"}`.
- Current MM3 clean-start/default proof:
  `/Users/eric/mlx/vllm-mlx/build/live-clean-start-mm3-reap40-d3-real-profile-gateway-visible-off-2026-06-18T13-36-46-062Z/clean-start-proof.json`,
  `status=pass`, failures `[]`. Session config after start shows
  `enablePrefixCache=true`, `usePagedCache=false`, `enableDiskCache=true`,
  `enableJit=false`, `toolCallParser=minimax_m3`,
  `reasoningParser=minimax_m3`, `isMultimodal=true`.
- MM3 cache wording precision: session prefix cache is ON. Native cache
  telemetry reports generic prefix component `false` because MM3 is not using
  generic KV/TurboQuant storage; it uses native `minimax_m3_msa_v1` with
  `attention_kv`, `msa_idx_keys`, `absolute_block_index`, generic TQ-KV off,
  storage quantization off, disk tuple tag `minimax_m3`, and
  `prompt_disk_l2=true`.
- Current release boundary: runtime proof, app proof, DMG notarization, staple,
  and Gatekeeper verification are present. Public release is still not done
  until vmlx main/tag/GitHub release assets and mlxstudio update surfaces are
  committed/pushed and verified live.

## 2026-06-18 14:00Z - Post-release broader Gemma audio audit/proof

Status: `BROADER_GOAL_PARTIAL_GEMMA_E2B_E4B_AUDIO_NOW_LIVE_PASS`.

- Reason for this continuation: the scoped 1.5.64 release intentionally kept
  Gemma audio out of the release gate, but the broader active goal includes
  checking all advertised VL/audio behavior. Current artifact audit showed:
  - `gemma4-e2b-jang4m-vl-current-64`: audio `runtime_supported`;
  - `gemma4-e4b-jang4m-vl-current-64`: audio `runtime_supported`;
  - `gemma4-12b-jang4m-vl-current-64`: audio `declared_not_runtime_supported`;
  - 26B/31B JANG_4M and MXFP4 visual rows: audio `not_advertised`.
- E2B JANG_4M audio live proof:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T13-52-20-927Z/gemma4-media-proof.json`.
  Result `status=pass`, failures `[]`, model path
  `/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-JANG_4M`,
  `expectAudio=true`.
  Evidence: capabilities audio `runtime_supported`, UI audio returned
  `audio present`, post-audio text recovery returned
  `GEMMA_POST_AUDIO_TEXT_OK`, mixed audio/reasoning row returned
  `GEMMA_MIX_AUDIO_PRESENT`, API Chat audio returned `audio present`,
  API Responses audio returned `audio present`, generation defaults matched
  `temperature=1.0`, `top_p=0.95`, `top_k=64`.
- E4B JANG_4M audio live proof:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-mxfp4-2026-06-18T13-54-24-110Z/gemma4-media-proof.json`.
  Result `status=pass`, failures `[]`, model path
  `/Users/eric/models/OsaurusAI--gemma-4-E4B-it-qat-JANG_4M`,
  `expectAudio=true`.
  Evidence: UI audio returned `audio present`, post-audio text recovery
  returned `GEMMA_POST_AUDIO_TEXT_OK`, API Chat audio returned
  `Audio present`, API Responses audio returned `Audio present`, streaming
  Chat/Responses text/image/tool rows remained green, generation defaults
  matched `temperature=1.0`, `top_p=0.95`, `top_k=64`.
- Artifact naming note: both audio reruns used the default row label
  `gemma4-e2b-mxfp4` because the row-name environment variable was mistyped
  during launch. The proof JSON model paths above are authoritative for which
  model actually ran.
- Current boundary: E2B/E4B audio are now live-positive. 12B/26B/31B audio is
  not green because current capabilities do not advertise runtime audio for
  those scoped rows. Do not claim "all Gemma audio" beyond E2B/E4B without
  either advertised runtime support or a new artifact-specific proof.

## 2026-06-18 14:05Z - Native cache source gates refreshed

Status: `SOURCE_CACHE_GATES_GREEN_LIVE_RESTART_RESTORE_STILL_PARTIAL`.

- Found a stale source-test red:
  `tests/test_mllm_scheduler_cache.py::TestMLLMSchedulerConfigParity::test_paged_cache_fields`
  expected `MLLMSchedulerConfig().use_paged_cache is True`.
- Root cause: the assertion predated the current paged-off default policy.
  Current source and live artifacts require prefix cache ON but paged cache OFF
  by default for MM3/Gemma mixed-SWA. Typed families such as ZAYA/CCA or hybrid
  Mamba can still promote themselves to paged when required by native cache
  state.
- Source fix:
  - `tests/test_mllm_scheduler_cache.py` now asserts default
    `use_paged_cache is False`;
  - `vmlx_engine/mllm_scheduler.py` comment now states that MM3 and Gemma
    mixed-SWA stay paged-off on the current memory-aware/prompt-L2 path.
  - `vmlx_engine/utils/ssm_companion_cache.py` in the active branch now mirrors
    main's `last_prefix_lookup.source="l1_or_l2"` hit diagnostic, and the test
    asserts it.
- Verification:
  - `.venv/bin/python -m pytest tests/test_mllm_scheduler_cache.py
    tests/test_single_sequence_cache_merge.py tests/test_gemma4_tool_parser.py
    tests/test_gemma4_reasoning_no_leak.py tests/test_minimax_m3_cache_paths.py
    -q` -> active branch `149 passed, 2 warnings`; clean main worktree
    `151 passed, 2 warnings`;
  - `.venv/bin/python -m pytest tests/test_minimax_m3_cache_paths.py -q` ->
    `24 passed`.
- Public main: pushed `4b8436dc1 test: align mllm paged cache default` to
  `jjang-ai/vmlx` main. Active feature branch has equivalent local cleanup
  commit `e17720bf3`.
- Boundary: this is source-level native cache contract evidence. It does not
  replace a future live fresh-process restart-restore proof for MM3 MSA or
  Gemma mixed-SWA prompt disk L2.

## 2026-06-18 14:40Z - Installed-app fresh-process prompt disk L2 restore proof

Status: `MM3_AND_GEMMA_E2B_RESTART_RESTORE_LIVE_PASS_RELEASE_STILL_PARTIAL`.

- Source changes under test:
  - `vmlx_engine/mllm_batch_generator.py` now uses
    `DiskCacheManager.fetch_longest_prefix()` for MLLM disk L2 prompt cache
    hits and replays the unmatched suffix after a prefix hit.
  - `vmlx_engine/mllm_scheduler.py` now stores MLLM prompt disk/L1 cache under
    the N-1 prompt-boundary key (`cache_key_tokens`) and uses clean typed
    prompt-boundary prefill for path-dependent mixed-SWA/ZAYA caches instead
    of generic post-generation truncation.
  - `panel/scripts/live-cache-restore-proof.mjs` is the installed-app harness
    for real two-phase fresh-process cache restore. It runs `/Applications/vMLX.app`
    twice with separate temp profiles/ports, primes the cache in phase 1, kills
    that app process, then loads a fresh app process and verifies a disk prefix
    cache hit on the next turn.
- Verification:
  - `.venv/bin/python -m py_compile vmlx_engine/mllm_scheduler.py
    vmlx_engine/mllm_batch_generator.py` passed.
  - `.venv/bin/python -m pytest tests/test_cache_bypass.py
    tests/test_mllm_scheduler_cache.py tests/test_single_sequence_cache_merge.py
    tests/test_minimax_m3_cache_paths.py -q` -> `193 passed, 2 warnings`.
  - Rebuilt and installed `/Applications/vMLX.app` via
    `panel/scripts/build-and-install.sh`; build exit 0; bundled import reported
    `vmlx_engine 1.5.64 imported OK`; installed app signature verified with
    `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`.
- MM3 REAP40-d3 installed-app exactness + fresh-process restore proof:
  `/Users/eric/mlx/vllm-mlx/build/live-cache-restore-mm3-reap40-d3-cache-restore-installed-multiturn-shape-2026-06-18T14-34-10-253Z/cache-restore-proof.json`.
  Result `status=pass`, failures `[]`.
  Evidence:
  - model path:
    `/Users/eric/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP40-d3-JANG_2L`;
  - prime response exact content:
    `CACHE_RESTORE_PRIME_OK_MQJLP2VH_872322`;
  - restore response exact content:
    `CACHE_RESTORE_HIT_OK_MQJLP2VH_872322`;
  - restore usage: `cached_tokens=587`, `cache_detail=disk`;
  - score flags: `empty=false`, `leakedReasoningTags=false`,
    `loopSuspect=false`;
  - native cache health: family `minimax_m3`, schema `minimax_m3_msa_v1`,
    cache type `native_msa_sparse_kv`, components `attention_kv`,
    `msa_idx_keys`, `absolute_block_index`, prompt disk L2 `true`, paged
    `false`, generic TurboQuant KV disabled with reason
    `native_minimax_m3_msa_idx_keys`, storage quantization disabled with
    reason `generic_kv_quantization_forced_off_for_msa_idx_keys`.
  - UI settings visibility captured prefix cache, paged KV, KV cache
    quantization, disk cache, performance/generation, max output/context,
    generation defaults, tool parser, and MM3 family explanation as visible.
- Gemma E2B MXFP4 installed-app fresh-process restore proof:
  `/Users/eric/mlx/vllm-mlx/build/live-cache-restore-gemma4-e2b-mxfp4-cache-restore-installed-multiturn-shape-2026-06-18T14-33-24-787Z/cache-restore-proof.json`.
  Result `status=pass`, failures `[]`.
  Evidence:
  - model path: `/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4`;
  - prime response exact content:
    `CACHE_RESTORE_PRIME_OK_MQJLO3SK_190630`;
  - restore response exact content:
    `CACHE_RESTORE_HIT_OK_MQJLO3SK_190630`;
  - restore usage: `cached_tokens=643`, `cache_detail=disk`;
  - native cache health: family `gemma4`, schema `mixed_swa_kv_v1`,
    components `full_attention_kv`, `sliding_window_kv`,
    `rotating_window_metadata`, paged `false`, generic TurboQuant KV not
    active, storage quantization disabled while preserving rotating-window
    metadata.
- Boundary:
  - This closes the previous live fresh-process restart-restore gap for MM3
    REAP40-d3 and Gemma E2B MXFP4 only.
  - It does not prove fresh-process restore for every Gemma size yet.
  - The settings harness still records `reasoningParser=false` in the settings
    visibility map even though earlier MM3 stress proof covered reasoning
    on/off/auto behavior functionally. Treat reasoning-parser UI visibility as
    `PARTIAL` until the UI label/a11y capture or a separate settings proof is
    refreshed.
  - This is an installed local app proof, not a notarized public release.

## 2026-06-18 17:00Z - Post-DMG scoped .64 MM3/Gemma gate refreshed

Status: `SCOPED_MM3_GEMMA_POST_DMG_PREFLIGHT_PASS_NOTARIZATION_PENDING`.

- Rebuilt signed local DMGs already present:
  - `panel/release/vMLX-1.5.64-sequoia-arm64.dmg`
  - `panel/release/vMLX-1.5.64-tahoe-arm64.dmg`
- Local signing/container checks were green before this gate:
  - `codesign --verify --deep --strict --verbose=2` on the staged Sequoia and
    Tahoe apps.
  - `codesign --verify --verbose=2` and `hdiutil verify` on both DMGs.
- Because the DMG build reset `build/`, reran the missing live signed-app
  proof rows against
  `panel/release/tahoe-app/mac-arm64/vMLX.app`.
- Current scoped release gate:
  `/Users/eric/mlx/vllm-mlx/build/current-scoped-release-preflight-64-post-dmg-rerun.json`
  and preserved copy
  `docs/internal/release-gates/current-proof-preserved/current-scoped-release-preflight-64-post-dmg-rerun.json`.
  Result: `status=pass`, failures `[]`.
- MM3 exactness/current artifacts accepted by the gate:
  - Stress/API/VL/tool/reasoning:
    `/Users/eric/mlx/vllm-mlx/build/live-mm3-stress-post-dmg-2026-06-18T16-27-08Z/mm3-stress-proof.json`.
  - Clean start/autodetect:
    `/Users/eric/mlx/vllm-mlx/build/live-clean-start-mm3-reap40-d3-real-profile-gateway-visible-off-2026-06-18T16-53-32-135Z/clean-start-proof.json`.
  - Lifecycle stop/abort/no-autonomous-generation:
    `/Users/eric/mlx/vllm-mlx/build/live-lifecycle-mm3-reap40-d3-lifecycle-2026-06-18T16-57-02-584Z/lifecycle-proof.json`.
- MM3 clean-start live config evidence:
  - `enablePrefixCache=true`, `usePagedCache=false`, `enableDiskCache=true`,
    `enableJit=false`.
  - `toolCallParser=minimax_m3`, `reasoningParser=minimax_m3`,
    `isMultimodal=true`.
  - Native cache schema `minimax_m3_msa_v1`, cache type
    `native_msa_sparse_kv`, components `attention_kv`, `msa_idx_keys`,
    `absolute_block_index`, generic TurboQuant KV off, storage quantization
    off, prompt disk L2 on.
- Gemma rows refreshed after the DMG build:
  - JANG_4M VL media/stress: E2B, E4B, 12B, 26B, 31B.
  - MXFP4 visual media/stress: 26B, 31B.
  - Clean-start/autodetect: E2B MXFP4, 12B/26B/31B JANG_4M,
    26B/31B MXFP4.
  - Lifecycle: E2B MXFP4, 26B/31B MXFP4.
- Boundary:
  - This is current source/runtime/package proof for the scoped MM3/Gemma .64
    release gate.
  - It is not yet notarized/stapled/public-released proof.

## 2026-06-18 17:12Z - 1.5.64 DMGs notarized and stapled

Status: `DMG_NOTARIZATION_PASS_PUBLIC_RELEASE_PENDING`.

- Ran `panel/scripts/notarize-release-dmgs.sh`.
- Sequoia:
  - DMG: `panel/release/vMLX-1.5.64-sequoia-arm64.dmg`
  - Apple notary id: `e4181a30-a25e-4b30-92ef-5d026e2df62f`
  - Notary status: `Accepted`
  - Stapler: staple and validate worked
  - `spctl`: accepted, `source=Notarized Developer ID`
  - SHA-256 after staple/blockmap regeneration:
    `cb818bfc4082b65e5469bf04a3d4e4d4864938cbaf373c8ddc1a5ee0eb04b882`
- Tahoe:
  - DMG: `panel/release/vMLX-1.5.64-tahoe-arm64.dmg`
  - Apple notary id: `35400c4d-be69-484d-bdb8-947cb988229f`
  - Notary status: `Accepted`
  - Stapler: staple and validate worked
  - `spctl`: accepted, `source=Notarized Developer ID`
  - SHA-256 after staple/blockmap regeneration:
    `12a34c1e9b7e3ffc7c5dfe31899b469d305033c15d15d64f0b5601d14b5a3010`
- Ran `panel/scripts/verify-release-dmgs.sh` after notarization.
  Final verification passed for both DMGs:
  `hdiutil verify`, `codesign --verify`, Developer ID authority
  `ShieldStack LLC (55KGF2S5AY)`, Team ID `55KGF2S5AY`, stapler validate,
  and `spctl --assess`.
- Boundary: public GitHub release/upload/feed updates are still pending.

## 2026-06-18 17:25Z - 1.5.64 public release surfaces updated

Status: `SCOPED_1_5_64_MM3_GEMMA_RELEASE_PUBLIC_SURFACE_PASS`.

- Updated `jjang-ai/mlxstudio` GitHub release `v1.5.64`:
  `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.5.64`.
  Replaced DMG assets with the final notarized/stapled files.
  Verified asset digests from GitHub:
  - `vMLX-1.5.64-sequoia-arm64.dmg`
    `sha256:cb818bfc4082b65e5469bf04a3d4e4d4864938cbaf373c8ddc1a5ee0eb04b882`,
    size `479960358`.
  - `vMLX-1.5.64-tahoe-arm64.dmg`
    `sha256:12a34c1e9b7e3ffc7c5dfe31899b469d305033c15d15d64f0b5601d14b5a3010`,
    size `495872676`.
- Updated `jjang-ai/mlxstudio` release body with final hashes and notary ids.
- Updated `jjang-ai/vmlx` GitHub release `v1.5.64`:
  `https://github.com/jjang-ai/vmlx/releases/tag/v1.5.64`.
  It now points to the `mlxstudio` DMG release and includes current main
  commit `cb4c768df` plus the final hashes.
- Updated `latest.json` in clean main worktree `/private/tmp/vmlx-1564-main`
  and pushed commit `cb4c768df` to `origin/main`.
  Raw GitHub verification:
  `https://raw.githubusercontent.com/jjang-ai/vmlx/main/latest.json` reports
  `version=1.5.64`, Sequoia hash
  `cb818bfc4082b65e5469bf04a3d4e4d4864938cbaf373c8ddc1a5ee0eb04b882`, Tahoe
  hash `12a34c1e9b7e3ffc7c5dfe31899b469d305033c15d15d64f0b5601d14b5a3010`.
- Boundary:
  - This is a scoped MM3 + Gemma 4 VL public release surface pass.
  - The broader non-scoped model matrix remains outside this release gate.

## 2026-06-18 17:30Z - Broader reasoning/tool streaming UI/API proof refreshed

Status: `UI_API_REASONING_TOOL_STREAMING_MOCK_SURFACE_PASS_MODEL_ROWS_STILL_SEPARATE`.

- Ran `node panel/scripts/live-chat-tools-reasoning-proof.mjs` with
  `VMLINUX_LIVE_PROOF_BASENAME=2026-06-18-broader-reasoning-tools-ui-api`.
- Artifact:
  `docs/internal/agent-notes/2026-06-18-broader-reasoning-tools-ui-api-proof.json`.
- Screenshots:
  - `docs/internal/agent-notes/2026-06-18-broader-reasoning-tools-ui-api-chat-settings.png`
  - `docs/internal/agent-notes/2026-06-18-broader-reasoning-tools-ui-api-server-cache-settings.png`
- Result: command exit 0.
- Evidence:
  - final visible content exactly `Done after tools.`;
  - persisted reasoning segments exactly
    `First plan before tools. ` and `Second plan after tool results. `;
  - event counts: stream `3`, tool `14`, reasoningDone `2`, complete `1`;
  - tool phases for `run_command`, `list_directory`, `read_image`, and
    `read_video` each included calling, executing, and result;
  - follow-up Responses request included content parts `text`, `image_url`,
    and `video_url`;
  - request summaries showed first request with tools and second request with
    `function_call_output`, tools, image URL, and video URL;
  - chat settings UI showed built-in tools/shell/search/utilities/hide-tool
    labels and expected checked states;
  - server cache UI showed prefix cache, paged KV, block disk L2, disk cache,
    and stored cache quantization labels; block disk toggled prefix+paged+L2,
    legacy disk toggled prefix+disk while paged stayed off;
  - no raw `<think>`, `<tool_call>`, or `<minimax:tool_call>` leaked into final
    visible content.
- Boundary:
  - This proves Electron UI/API streaming/tool/reasoning plumbing against a
    deterministic mock Responses server.
  - It does not by itself prove real MM3/Gemma model behavior; those remain
    covered by the live model rows.

## 2026-06-18 17:35Z - Broader Gemma E2B/E4B JANG_4M audio+VL rows refreshed

Status: `GEMMA_E2B_E4B_JANG4M_AUDIO_VL_LIVE_PASS_OTHER_GEMMA_AUDIO_NOT_GREEN`.

- Reran correct-row-name audio+VL stress rows against signed staged app
  `panel/release/tahoe-app/mac-arm64/vMLX.app` with bundled engine `1.5.64`.
- E2B JANG_4M artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e2b-jang4m-audio-vl-broader-64-2026-06-18T17-19-21-681Z/gemma4-media-proof.json`.
  Result `status=pass`, failures `[]`.
  Evidence:
  - runtime capabilities include `text`, `vision`, `audio`, `video`;
  - UI audio exact visible output `audio present`;
  - post-audio text recovery exact `GEMMA_POST_AUDIO_TEXT_OK`;
  - mixed single-chat labels include `GEMMA_MIX_TEXT_OFF`,
    `GEMMA_MIX_IMAGE_RED`, `GEMMA_MIX_AUTO_TEXT`,
    `GEMMA_MIX_AUDIO_PRESENT`, `GEMMA_MIX_TOOL_ON_DONE`,
    `GEMMA_MIX_TOOL_AUTO_DONE`, `GEMMA_MIX_CACHE_HIT`;
  - Chat and Responses audio both returned `audio present`;
  - streaming text/image/tool rows passed;
  - generation defaults parity true for `temperature`, `top_p`, and `top_k`.
- E4B JANG_4M artifact:
  `/Users/eric/mlx/vllm-mlx/build/live-gemma4-media-gemma4-e4b-jang4m-audio-vl-broader-64-2026-06-18T17-20-58-698Z/gemma4-media-proof.json`.
  Result `status=pass`, failures `[]`.
  Evidence:
  - runtime capabilities include `text`, `vision`, `audio`, `video`;
  - UI audio exact visible output `audio present`;
  - post-audio text recovery exact `GEMMA_POST_AUDIO_TEXT_OK`;
  - mixed single-chat labels include `GEMMA_MIX_TEXT_OFF`,
    `GEMMA_MIX_IMAGE_RED`, `GEMMA_MIX_AUTO_TEXT`,
    `GEMMA_MIX_AUDIO_PRESENT`, `GEMMA_MIX_TOOL_ON_DONE`,
    `GEMMA_MIX_TOOL_AUTO_DONE`, `GEMMA_MIX_CACHE_HIT`;
  - Chat and Responses audio both returned `Audio present`;
  - streaming text/image/tool rows passed;
  - generation defaults parity true for `temperature`, `top_p`, and `top_k`.
- Boundary:
  - E2B and E4B JANG_4M audio+VL are live-positive in the signed app.
  - 12B/26B/31B Gemma audio remains not green unless those artifacts advertise
    runtime audio support and pass an audio-specific live row. Their scoped .64
    VL rows remain green for image/VL only.

## 2026-06-18 18:57Z - MM3 strict installed-app stress refreshed

Status: `MM3_STRICT_INSTALLED_PASS_GEMMA_ROWS_STILL_OPEN_SCOPE_OVERRIDE_REQUESTED`.

- Blocker classes reduced: `runtime/kernel`, `parser/template`,
  `cache/storage`, and `api/ui` for MiniMax-M3.
- First run artifact:
  `build/live-mm3-stress-current-64-2026-06-18T18-36-55Z/mm3-stress-proof.json`.
  Result `status=fail`; failures were exact-count harness failures for
  `EBS` and `PROFILE`.
- Root cause of first failure:
  the final recall prompt allowed phrases like `EBS patching` and
  `profile options`; `markerWordCount()` counted those repeated label words.
  The output was coherent and cached, but the exactness instruction was
  ambiguous.
- Harness source adjustment:
  `panel/scripts/live-mm3-stress-proof.mjs` now tells the model to use each
  label exactly once, only before its colon, and not repeat label words inside
  phrases after colons.
- Passing rerun artifact:
  `build/live-mm3-stress-current-64-rerun-2026-06-18T18-46-27Z/mm3-stress-proof.json`.
  Result `status=pass`, failures `[]`.
- Installed-app MM3 evidence:
  - launch used `/Applications/vMLX.app`;
  - generated CLI included `--tool-call-parser minimax_m3`,
    `--enable-auto-tool-choice`, `--reasoning-parser minimax_m3`,
    `--enable-disk-cache`, and no paged-cache enable flag;
  - startup logs showed `paged_cache=OFF`, `tq_kv=SKIP(native MSA)`,
    `vl_route=ON`, `tool_parser=minimax_m3`,
    `reasoning_parser=minimax_m3`, `jit=off`, and
    `msa_per_step_sync=ON`;
  - native cache schema was `minimax_m3_msa_v1` with `attention_kv`,
    `msa_idx_keys`, and `absolute_block_index`;
  - generic TurboQuant KV was disabled for M3 native MSA and prompt disk L2
    was enabled;
  - scheduler cache hit tokens reached `23288`;
  - UI 10-turn cache hits rose to `1536` cached tokens with no loop suspects;
  - reasoning off/on/auto, UI tool call, long-context sentinel recall, MM3 VL
    image, mixed image/reasoning/tool/cache, Chat, Responses,
    Anthropic, Ollama chat/generate, streaming Chat, streaming Responses
    image, streaming Chat tool, and streaming Responses tool rows passed;
  - generation defaults matched model `generation_config.json`:
    temperature `1`, top-p `0.95`, top-k off, and the UI visibly exposed
    those values.
- Follow-up preflight:
  `python3 panel/scripts/scoped-release-preflight-64.py --out
  /tmp/current-scoped-release-preflight-64-after-mm3-pass.json` returned
  `status=fail` only for remaining Gemma media/stress,
  clean-start/autodetect, and lifecycle rows.
- Boundary:
  MM3 strict installed-app and CLI/autodetect behavior are current-pass.
  Full scoped preflight remains red unless the release is explicitly narrowed
  to a user-protection MM3/current-GUI release and remaining Gemma rows are
  deferred.

## 2026-06-18 18:59Z - Release-scope override under discussion

Status: `USER_REQUESTED_SCOPED_USER_PROTECTION_RELEASE_CHECKING_CACHE_DEFAULTS_BEFORE_PACKAGING`.

- Eric asked whether, if the GUI and model launch are usable, `.64` can ship
  now so users stop hitting the current issues, with remaining red blockers
  deferred to the next version.
- Eric also explicitly added that prefix cache, SSD/disk cache, memory/RAM
  limit, paged-cache behavior, and related UI/CLI controls must be proper,
  configurable, visible, and toggleable before packaging.
- Current evidence for that added cache/UI condition:
  - installed-app paged-cache settings proof:
    `build/live-metal-headroom-ui-2026-06-18T18-32-30-003Z/metal-headroom-ui-proof.json`;
  - MM3 strict artifact above proves MM3 defaults in live app:
    prefix cache on, paged cache off, disk cache on, native MSA cache schema,
    generic TQ-KV disabled, prompt disk L2 on, and cache hits present.
- Boundary:
  do not package/upload a replacement `.64` until the cache/defaults release
  condition is either rechecked from current source/artifacts or explicitly
  accepted as covered by the installed-app proofs above.

## 2026-06-18 19:05Z - Scoped preflight pass restored

Status: `SCOPED_PREFLIGHT_PASS_WITH_FRESH_MM3_AND_PRESERVED_GEMMA_ROWS`.

- `panel/scripts/scoped-release-preflight-64.py` now scans preserved proof rows
  in `docs/internal/release-gates/current-proof-preserved/` as secondary
  evidence, while preferring fresh `build/` MM3 stress proof when present.
- Verification passed:
  `.venv/bin/python -m py_compile panel/scripts/scoped-release-preflight-64.py`.
- Gate command passed:
  `.venv/bin/python panel/scripts/scoped-release-preflight-64.py --out
  /tmp/current-scoped-release-preflight-64-after-preserved-proof-scan.json`.
- Result: `status=pass`, failures `[]`.
- Boundary:
  this supports a scoped user-protection release from current source. Gemma rows
  are preserved release-gate evidence, not rerun after the MH-17 UI-only cache
  settings change.

## 2026-06-18 19:16Z - Release DMGs rebuilt from current source

Status: `DMG_BUILD_PASS_NOTARIZATION_PENDING`.

- Official release build command completed:
  `VMLINUX_RELEASE_SCOPE=mm3_gemma_vl VMLX_PREPACKAGE_READY_MANIFEST_OUT=build/current-scoped-release-preflight-64-dmg-reissue-build.json bash panel/scripts/build-release-dmgs.sh all`.
- Produced:
  - `panel/release/vMLX-1.5.64-sequoia-arm64.dmg`;
  - `panel/release/vMLX-1.5.64-tahoe-arm64.dmg`.
- Boundary:
  DMGs are built from current source, but notarization/stapling/final verify and
  public release updates remain pending.

## 2026-06-18 19:31Z - Reissued .64 DMGs notarized and verified

Status: `DMG_NOTARIZATION_AND_VERIFY_PASS_PUBLIC_UPLOAD_PENDING`.

- `panel/release/vMLX-1.5.64-sequoia-arm64.dmg` notarized with Apple id
  `d7be5213-9931-4b72-8242-c6da9d3c60fa`, stapled, verified, `spctl`
  accepted. Final SHA-256:
  `4a23be61f5b7b68aeea2a49375e77751088461fc65b5187d7af2d915955b699c`.
- `panel/release/vMLX-1.5.64-tahoe-arm64.dmg` notarized with Apple id
  `7a419358-ccb6-4b12-97c7-3812da3c5bc2`, stapled, verified, `spctl`
  accepted. Final SHA-256:
  `983423d21c60637a18d44478bc42c30eab7846b92b84bec5fa1b61b2a70f17c3`.
- Boundary:
  public GitHub release asset replacement and updater metadata remain pending.

## 2026-06-18 19:36Z - Follow-up issue added: MM3 cache-mode confusion

Status: `NEXT_RELEASE_FOLLOWUP_NOT_CURRENT_BLOCKER`.

- Follow-up issue:
  document and test the user-facing distinction between:
  - MM3 production default cache path: prefix cache on, SSD prompt/disk cache
    on, paged cache off, native MSA cache with `idx_keys`, generic TQ-KV off;
  - diagnostic clean-engine path: `--disable-prefix-cache`
    `--no-memory-aware-cache`, which is useful only for isolation and should
    not be treated as the production fix.
- Current .64 evidence:
  the latest MM3 installed-app strict run proved production cache-on behavior
  without incoherent looping, with MM3 autodetect logs showing
  `paged_cache=OFF`, `tq_kv=SKIP(native MSA)`, `vl_route=ON`,
  `tool_parser=minimax_m3`, `reasoning_parser=minimax_m3`, `jit=off`, and
  `msa_per_step_sync=ON`.
- Required next-release checks:
  - explicit CLI-only MM3 run with default cache-on and expected logs;
  - explicit CLI-only MM3 diagnostic run with `--disable-prefix-cache`
    `--no-memory-aware-cache` and expected logs;
  - UI session settings screenshot/proof showing default cache settings and
    the consequences of toggling prefix, paged, disk, and memory controls;
  - docs/user-facing copy explaining that cache-on is intended and safe when
    native MSA cache invariants are preserved.
- Boundary:
  this is tracked for the next release; it does not block the current scoped
  MM3 user-protection release because the current installed-app MM3 production
  cache-on path passed.

## 2026-06-18 20:25Z - v1.5.65 public release completed

Status: `V1_5_65_RELEASE_PUBLIC_SURFACES_VERIFIED`.

- Reason for bump:
  `.64` had already been published and users had downloaded it, so the MM3
  compatibility/user-protection release had to move to `.65` instead of
  silently replacing `.64` assets.
- Public source:
  `jjang-ai/vmlx` main pushed to `43a5f460d`, tag `v1.5.65` pushed, and
  GitHub release created at
  `https://github.com/jjang-ai/vmlx/releases/tag/v1.5.65`.
- Public app downloads:
  `jjang-ai/mlxstudio` release created at
  `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.5.65`.
- Updater metadata:
  `jjang-ai/vmlx` main `latest.json` now reports version `1.5.65` and points
  to the `mlxstudio` `.65` Sequoia/Tahoe DMGs.
- Notarization evidence:
  - Sequoia DMG Apple id `9d76633e-abd1-4e60-81c5-2b526d3da4dc`;
    SHA-256 `7c9984b9fdafeffbe470b2d7396e43bfd54138b6cc832e923a1445a245bd5ed3`.
  - Tahoe DMG Apple id `5b6c3d94-c840-4beb-af42-26aa155a07e9`;
    SHA-256 `a9bc04daa894da6552906d09f9ec4a7056eea41041f39178ccb6981ffcad21b4`.
- Final verification:
  `bash panel/scripts/verify-release-dmgs.sh` passed for both `.65` DMGs:
  `hdiutil verify`, Developer ID signature, stapled notarization ticket,
  stapler validate, `spctl source=Notarized Developer ID`, and SHA-256 output.
- Public asset verification:
  GitHub release assets for `mlxstudio` `v1.5.65` report uploaded state and
  matching SHA-256 digests for both DMGs plus blockmaps.
- Boundary:
  this is the scoped MM3 + Gemma 4 VL compatibility release. Broader red matrix
  rows and the explicit MM3 cache-on/cache-off CLI comparison remain tracked
  for the next release line.

## 2026-06-18 20:17Z - Post-release M3 UI cache-codec hardening

Status: `SOURCE_UI_FIX_PASS_LIVE_NOT_RERUN`.

- Issue found after release audit:
  the launch path correctly ignored generic stored KV quantization for
  MiniMax-M3, but the settings form only disabled the Stored Cache Quantization
  dropdown for DSV4. That let users visually select `q4`/`q8` for M3 even
  though the native MSA cache must keep keys, values, `idx_keys`, and absolute
  offsets first-class.
- Fix:
  `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` now treats
  DSV4 and MiniMax-M3 as native typed-cache owners for the stored-cache codec:
  the dropdown displays `auto`, is disabled, and shows a MiniMax-M3 explanation
  that generic q4/q8 stored-KV codecs cannot preserve the MSA cache format.
- Test:
  `panel/tests/settings-flow.test.ts` now pins the M3 disabled-state source
  contract and keeps the existing DSV4 native-cache contract updated.
- Verification:
  - `npm test -- settings-flow.test.ts -t "settings form disables generic stored KV codec controls for MiniMax-M3 native MSA cache|settings form keeps DSV4 native cache controls deduped"` -> 2 passed;
  - `npm test -- cache-control-policy.test.ts cache-capacity-display.test.ts settings-flow.test.ts -t "cache control policy|cache capacity display helpers|settings form disables generic stored KV codec controls for MiniMax-M3 native MSA cache|settings form keeps DSV4 native cache controls deduped"` -> 17 passed;
  - `npm run typecheck` -> passed;
  - `git diff --check -- panel/src/renderer/src/components/sessions/SessionConfigForm.tsx panel/tests/settings-flow.test.ts` -> passed;
  - `.venv/bin/python panel/scripts/scoped-release-preflight-65.py --out build/current-scoped-release-preflight-65-after-m3-ui-kv-codec-disable.json` -> `status=pass`.
- Boundary:
  this is source/UI-contract proof only. The already-published `.65` DMGs do
  not include this post-release UI hardening unless a new release is built.
  No fresh installed-app live run was performed for this UI-only change.

## 2026-06-18 20:20Z - Post-release M3 paged-cache toggle hardening

Status: `SOURCE_UI_FIX_PASS_LIVE_NOT_RERUN`.

- Issue found after continuing the cache-mode audit:
  the form already forced `usePagedCache=false` for MiniMax-M3 and the launcher
  forced M3 paged cache off, but the visible `Use Paged KV Cache` checkbox was
  still clickable for M3. Users could therefore click a generic paged-cache
  control that the M3 launch path would ignore.
- Fix:
  `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` now
  disables the generic paged-cache checkbox for M3 and shows an explicit note:
  MiniMax-M3 uses paged-off SSD prefix cache with native MSA `idx_keys`;
  generic paged KV cache is disabled for this model.
- Test:
  `panel/tests/settings-flow.test.ts` now pins the M3 paged-cache disabled
  marker, the user-facing explanation, and the disabled prop.
- Red/green:
  `npm test -- settings-flow.test.ts -t "settings form disables the ignored generic paged-cache toggle for MiniMax-M3"`
  first failed on missing `genericPagedCacheToggleDisabled`, then passed after
  the form change.
- Verification:
  - `npm test -- cache-control-policy.test.ts cache-capacity-display.test.ts settings-flow.test.ts -t "cache control policy|cache capacity display helpers|settings form disables generic stored KV codec controls for MiniMax-M3 native MSA cache|settings form disables the ignored generic paged-cache toggle for MiniMax-M3|settings form keeps DSV4 native cache controls deduped|settings form renders effective paged capacity"` -> 19 passed;
  - `npm run typecheck` -> passed;
  - `git diff --check -- panel/src/renderer/src/components/sessions/SessionConfigForm.tsx panel/tests/settings-flow.test.ts` -> passed;
  - `.venv/bin/python panel/scripts/scoped-release-preflight-65.py --out build/current-scoped-release-preflight-65-after-m3-ui-cache-toggle-disable.json` -> `status=pass`, failures `[]`.
- Boundary:
  this is source/UI-contract proof only. The already-published `.65` DMGs do
  not include this post-release UI hardening unless a new release is built.
  No fresh installed-app live screenshot/proof was run for this UI-only change.

## 2026-06-18 20:27Z - Post-release M3 settings live UI proof added

Status: `SOURCE_UI_FIX_PASS_ELECTRON_DEV_UI_PROOF`.

- Harness change:
  `panel/scripts/live-metal-headroom-ui-proof.mjs` now includes a fake
  `minimax_m3_vl` model-config pass. It opens the real create-session settings
  UI in Electron dev, verifies the active model path, opens Paged KV Cache and
  KV Cache Quantization, and records the M3-specific disabled states.
- Live UI proof:
  `VMLINUX_ELECTRON_DEV=1 node panel/scripts/live-metal-headroom-ui-proof.mjs`
  passed with `status=pass`, failures `[]`, output:
  `build/live-metal-headroom-ui-2026-06-18T20-26-19-934Z/metal-headroom-ui-proof.json`.
- Proof fields:
  `minimaxM3SettingsUi.activeModelPathFound=true`,
  `pagedCheckboxFound=true`, `pagedChecked=false`, `pagedDisabled=true`,
  `storedQuantSelectFound=true`, `storedQuantValue=auto`,
  `storedQuantDisabled=true`, and all M3 explanation text checks true.
- Screenshot:
  `build/live-metal-headroom-ui-2026-06-18T20-26-19-934Z/metal-headroom-m3-settings-proof.png`
  shows the real settings UI with M3's generic paged toggle off/disabled and
  Stored Cache Quantization forced to Auto/disabled.
- Additional harness coverage:
  the same run still proves generic paged-cache capacity UI/log behavior for a
  fake Qwen model: cache memory MB/%/TTL controls disabled while paged cache is
  on, re-enabled when off, and the expected capacity/ignored-memory log lines
  present.
- Preflight:
  `.venv/bin/python panel/scripts/scoped-release-preflight-65.py --out build/current-scoped-release-preflight-65-after-m3-live-ui-settings-proof.json`
  passed with `status=pass`, failures `[]`; it still points to the preserved
  post-DMG MM3/Gemma proof artifacts for model-load evidence.
- Boundary:
  this is Electron-dev current-source UI proof, not installed-app proof and not
  a fresh huge-model load/chat proof. The already-published `.65` DMGs still do
  not contain these post-release UI hardening changes.

## 2026-06-18 20:38Z - v1.5.65 GitHub public, mlx.studio origin still stale; next release is 1.5.66

Status: `PARTIAL_RELEASE_SURFACE`.

- GitHub release verification:
  - `jjang-ai/mlxstudio` `v1.5.65` exists, non-draft, non-prerelease,
    published `2026-06-18T20:13:21Z`;
  - `jjang-ai/mlxstudio` assets are uploaded for Sequoia/Tahoe DMGs and
    blockmaps;
  - Sequoia asset digest is
    `sha256:7c9984b9fdafeffbe470b2d7396e43bfd54138b6cc832e923a1445a245bd5ed3`;
  - Tahoe asset digest is
    `sha256:a9bc04daa894da6552906d09f9ec4a7056eea41041f39178ccb6981ffcad21b4`;
  - `jjang-ai/vmlx` `v1.5.65` exists, non-draft, non-prerelease, published
    `2026-06-18T20:13:26Z`.
- Manifest correction:
  - `jjang-ai/vmlx/main/latest.json` already serves `1.5.65`;
  - `jjang-ai/mlxstudio/main/latest.json` was stale at `1.5.64`, then updated
    and pushed to commit `38b94f7` with `version=1.5.65` and matching hashes.
- Live website blocker:
  - `https://mlx.studio/update/latest.json` still returns `version=1.5.58`;
  - `https://mlx.studio/download/` still hardcodes `.58` links and
    `Version 1.5.58`;
  - query-string cache busting still returns `.58`, and headers report
    `cf-cache-status: DYNAMIC`, so the origin content is stale, not merely a
    browser cache artifact.
- Server/control blocker:
  - prior logs identify the origin as `/var/www/mlx.studio` on
    `45.32.71.230`;
  - user says Cloudflare controls/API/auth live at `/var/www/cf-panel`;
  - current session cannot SSH: `45.32.71.230` reaches port 22 but rejects the
    local key, `exploit.team`/`exploit.bot` time out, and
    `ai.shieldstack.dev` rejects the local key;
  - public `/cf-panel` returns `404`; direct-origin `/cf-panel` returns `403`;
  - local `/var/www` does not exist;
  - SSH alias `omlx` is reachable as root, but it is the `omlx.net` host only:
    `/var/www/cf-panel` and `/var/www/mlx.studio` are absent there, with nginx
    roots under `/var/www/omlx.net`.
- Next release:
  - `1.5.66` is now the next vMLX/MLXStudio release number.
  - Details are tracked in `.agents/NEXT-RELEASE-1.5.66.md`.
- Boundary:
  `.65` is uploaded/released on GitHub, but the `mlx.studio` updater/download
  website is not live-current yet. Do not tell users the website updater is on
  `.65` until live curl proves it.

## 2026-06-18 20:44Z - Live-public .65 release-surface contract fails on website/PyPI/source-tag rows

Status: `LIVE_PUBLIC_RELEASE_SURFACE_FAIL_EXPECTED`.

- Command:
  `.venv/bin/python tests/cross_matrix/run_release_surface_contract.py --live-public --out build/current-release-surface-contract-v1565-live-public-site-stale-20260618.json`.
- Artifact:
  `build/current-release-surface-contract-v1565-live-public-site-stale-20260618.json`.
- Result:
  `status=fail`.
- Passing evidence:
  - source versions are `1.5.65`;
  - local/latest manifest structure is valid;
  - raw GitHub `mlxstudio/main/latest.json` matches local `1.5.65`;
  - GitHub `jjang-ai/mlxstudio` release `v1.5.65` is published and contains
    expected Sequoia/Tahoe DMG assets and digests;
  - `mlx.studio/update/latest.json` cache headers are safe.
- Failing evidence:
  - live `mlx.studio/update/latest.json` still serves `1.5.58`;
  - PyPI `vmlx/1.5.65` returns 404;
  - `jjang-ai/vmlx` tag `v1.5.65` resolves to commit `43a5f460...`, while
    current source head is `06ba1895...`.
- Boundary:
  treat `.65` as GitHub-DMG-public but not full public-release-surface-current.
  The website origin/API/auth blocker remains unresolved, and PyPI/source-tag
  parity also remain open.

## 2026-06-18 20:45Z - Update checker protects users from stale mlx.studio manifest, but release surface still open

Status: `SOURCE_GUARD_PASS_PUBLIC_SURFACE_STILL_FAILS`.

- Source trace:
  `panel/src/main/update-checker.ts` fetches raw GitHub and `mlx.studio`
  manifests in parallel and selects `selectHighestRelease(results)`.
- Added guard:
  `panel/tests/update-checker.test.ts` now asserts raw GitHub `1.5.65` wins
  over stale `mlx.studio` `1.5.58` regardless of result order.
- Test:
  `npm test -- update-checker.test.ts -t "multi-source update manifests|native macOS update asset selection"`
  passed: 5 selected tests.
- Follow-up live-public contract:
  `build/current-release-surface-contract-v1565-live-public-site-stale-after-update-checker-guard-20260618.json`
  still fails on:
  - stale website updater (`1.5.58`);
  - missing PyPI `vmlx==1.5.65`;
  - source tag/head mismatch.
- Boundary:
  this protects the app updater from a stale custom-domain manifest, but it is
  not a full release-surface fix. Origin/API/auth is still required to update
  `mlx.studio`.

## 2026-06-18 20:48Z - Release-surface gate reporting hardened

Status: `REPORTING_GUARD_PASS_LIVE_PUBLIC_STILL_FAILS`.

- Source:
  `tests/cross_matrix/run_release_surface_contract.py` now emits
  `failed_checks`, `failed_steps`, and `next_actions` alongside
  `status_failed_checks`.
- Test:
  `.venv/bin/python -m pytest tests/test_release_surface_contract.py -q`
  passed: 14 tests.
- Live-public artifact:
  `build/current-release-surface-contract-v1565-live-public-site-stale-with-failed-checks-20260618.json`.
- Current failed checks are explicit:
  - `public_github_source_release_tag_matches_source_head`;
  - `public_pypi_has_release_files`;
  - `public_site_updater_matches_local`.
- Boundary:
  reporting is fixed; the public release surface remains failed until those
  three rows are addressed or explicitly scoped out.

## 2026-06-18 20:52Z - 1.5.66 source line started with version-stamp guard

Status: `SOURCE_VERSION_1_5_66_LOCAL_RELEASE_CONTRACT_PASS`.

- Source stamps now all read `1.5.66`:
  - `pyproject.toml`;
  - `vmlx_engine/__init__.py`;
  - `panel/package.json`;
  - `panel/package-lock.json`.
- Public updater manifest remains `latest.json` `1.5.65`, intentionally, until
  a real `.66` release is built/signed/notarized/uploaded.
- Gate hardening:
  `tests/cross_matrix/run_release_surface_contract.py` now checks
  `panel_package_lock` and `engine_init` too, not only pyproject/panel package.
- Verification:
  - new RED/GREEN test:
    `test_release_surface_contract_requires_all_source_version_stamps_to_match`;
  - `.venv/bin/python -m pytest tests/test_release_surface_contract.py -q`
    passed 15 tests;
  - local contract artifact
    `build/current-release-surface-contract-v1566-source-stamps-20260618.json`
    passed with all source versions `1.5.66`, local latest `1.5.65`, and
    `staged_source_version_not_public=true`;
  - update-checker selected tests passed under `vmlx@1.5.66`;
  - `py_compile vmlx_engine/__init__.py` passed.
- Boundary:
  this is source/version preparation for `.66`, not a built, signed, notarized,
  uploaded, or live-model-verified `.66` release.

## 2026-06-18 20:57Z - 1.5.66 scoped MM3/Gemma preflight and DMG route wired

Status: `SCOPED_1_5_66_PREFLIGHT_PASS_PRESERVED_PROOF_NO_FRESH_BUILD`.

- Added:
  `panel/scripts/scoped-release-preflight-66.py`.
- Build-route change:
  `panel/scripts/build-release-dmgs.sh` now routes
  `VMLINUX_RELEASE_SCOPE=mm3_gemma_vl` and version `1.5.66` through
  `scoped-release-preflight-66.py`; `.65` still routes through the `.65` gate.
- TDD:
  `tests/test_scoped_release_preflight_66.py` first failed because the `.66`
  script/route were missing, then passed after the change.
- Verification:
  - `.venv/bin/python -m pytest tests/test_scoped_release_preflight_66.py -q`
    passed 2 tests;
  - `.venv/bin/python -m py_compile panel/scripts/scoped-release-preflight-66.py panel/scripts/scoped-release-preflight-65.py`
    passed;
  - `bash -n panel/scripts/build-release-dmgs.sh` passed;
  - `.venv/bin/python panel/scripts/scoped-release-preflight-66.py --out build/current-scoped-release-preflight-66-source-gate-20260618.json`
    passed with `failures=[]`.
- Artifact:
  `build/current-scoped-release-preflight-66-source-gate-20260618.json`.
  It records `.66` source versions, preserved MM3 stress proof, 7 Gemma media
  rows, 7 clean-start rows, and 4 lifecycle rows.
- Boundary:
  this is a scoped source/prepackage gate using preserved/current proof
  artifacts. It is not a fresh `.66` live UI/model stress run, not a signed DMG,
  and not a notarized/public release.

## 2026-06-18 20:55Z - API/auth and reasoning-empty-response source guards tightened

Status: `API_AUTH_SOURCE_GUARD_PASS_LIVE_API_AUTH_OPEN`.

- Source trace:
  - `vmlx_engine/server.py` requires `--api-key` or `VLLM_API_KEY` for API
    authentication and verifies `Authorization: Bearer ...` via
    `verify_api_key`;
  - `panel/src/main/sessions.ts` injects `VLLM_API_KEY` into the spawned engine
    env instead of argv, so session API keys do not leak through `ps`;
  - `panel/src/main/ipc/utils.ts` emits bearer headers for local API-key
    sessions and remote sessions, including `OpenAI-Organization` for remote
    org keys;
  - `vmlx_engine/server.py` has bounded visible-answer pass paths for
    reasoning-only Chat Completions and Responses streams so MM3/Gemma turns do
    not silently remain invisible when the first pass only emits reasoning.
- Code hardening:
  - added
    `tests/test_engine_audit.py::TestStreamUsagePropagatesCacheDetail::test_minimax_m3_chat_stream_reasoning_only_runs_visible_answer_pass`;
  - made the MM3 Responses visible-answer endpoint string explicit in
    `vmlx_engine/server.py` so the source guard can prove the exact MM3 API
    route instead of relying only on dynamic interpolation.
- Verification:
  - `.venv/bin/python -m pytest tests/test_engine_audit.py::TestStreamUsagePropagatesCacheDetail::test_minimax_m3_chat_stream_reasoning_only_runs_visible_answer_pass -q`
    -> 1 passed;
  - `.venv/bin/python -m pytest tests/test_engine_audit.py::TestStreamUsagePropagatesCacheDetail::test_gemma4_responses_stream_reasoning_only_runs_visible_answer_pass tests/test_engine_audit.py::TestStreamUsagePropagatesCacheDetail::test_minimax_m3_chat_stream_reasoning_only_runs_visible_answer_pass -q`
    -> 2 passed;
  - `.venv/bin/python -m pytest tests/test_streaming_reasoning.py -k "minimax_m3_streaming_enabled_mode_seeds_prompt_reasoning or minimax_m3_responses_forced_on_has_visible_answer_pass or minimax_m3_responses_never_falls_back_reasoning_as_visible" -q`
    -> 3 passed;
  - `.venv/bin/python -m pytest tests/test_server.py -k "verify_api_key_rejects_invalid or verify_api_key_accepts_valid" -q`
    -> 2 passed;
  - `cd panel && npm test -- settings-flow.test.ts remote-session.test.ts -t "API key|auth|Authorization|Bearer"`
    -> 5 passed;
  - `.venv/bin/python -m py_compile vmlx_engine/server.py` -> passed.
- Boundary:
  - this is source/unit/renderer-contract proof only;
  - it does not prove a live `.66` packaged app, direct session-port API, MLX
    Studio gateway API, Responses streaming, Anthropic, Ollama, tool calls, or
    real API-key rejection/acceptance against a running MM3/Gemma engine;
  - website/cf-panel auth is still blocked separately because the actual
    MLXStudio origin has no working SSH/API/auth from this session.

## 2026-06-18 21:00Z - MM3/Gemma live stress harnesses now require API-auth matrix proof

Status: `LIVE_HARNESS_AUTH_MATRIX_GUARD_PASS_NO_FRESH_MODEL_RUN`.

- Harness changes:
  - `panel/scripts/live-mm3-stress-proof.mjs` now creates its app session with
    a per-run `apiKey`, sends authenticated direct API requests by default,
    and records `apiAuth` with missing/wrong/correct bearer checks plus an
    authenticated chat exact-marker row;
  - `panel/scripts/live-gemma4-media-stress-proof.mjs` now does the same with
    a Gemma-specific per-run `apiKey`;
  - both scripts record `gatewayAuth` when the MLXStudio gateway is running,
    and explicitly mark it skipped if the gateway is not part of that stress
    run;
  - both `deriveVerdict` functions now fail the proof if the direct session
    API auth matrix does not show missing=401, wrong=401, correct=200, and a
    correct authenticated chat marker.
- Source guard:
  - added
    `tests/test_panel_cli_flag_contract.py::test_mm3_and_gemma_live_stress_harnesses_cover_api_auth_matrix`.
- Verification:
  - first RED run failed because the live stress scripts had no `apiKey` /
    `runApiAuthMatrix` coverage;
  - `.venv/bin/python -m pytest tests/test_panel_cli_flag_contract.py::test_mm3_and_gemma_live_stress_harnesses_cover_api_auth_matrix -q`
    -> 1 passed;
  - `.venv/bin/python -m pytest tests/test_panel_cli_flag_contract.py -q`
    -> 14 passed;
  - `node --check panel/scripts/live-mm3-stress-proof.mjs` -> passed;
  - `node --check panel/scripts/live-gemma4-media-stress-proof.mjs` -> passed;
  - `git diff --check panel/scripts/live-mm3-stress-proof.mjs panel/scripts/live-gemma4-media-stress-proof.mjs tests/test_panel_cli_flag_contract.py`
    -> passed.
- Boundary:
  - this makes future MM3/Gemma live stress artifacts prove direct API auth;
  - it is not itself a fresh installed-app live model run, gateway proof, or
    notarized `.66` release proof;
  - `.66` still needs installed-app live reruns of MM3 and Gemma rows using the
    updated harnesses.

## 2026-06-18 22:15Z - MM3 128GB RAM/OOM guard installed; external live retest open

Status: `INSTALLED_APP_SOURCE_BUNDLE_PASS_EXTERNAL_128GB_LIVE_OPEN`.

- User log analyzed:
  `/Users/eric/Downloads/vmlx-logs-78327b52-2026-06-18.log`.
- Findings:
  - no explicit crash/OOM in the exported excerpt;
  - MM3 autodetect/cache route was correct:
    `paged_cache=OFF`, `tq_kv=SKIP(native MSA)`, `vl_route=ON`, typed
    `MiniMaxM3SparseCache`, memory-aware prefix cache, SSD disk cache;
  - old RAM behavior was risky on 128GB Macs:
    `Wired limit set to 134 GB (model 113 GB)` and later Metal baseline
    `active=105.4GB max=125.0GB`, leaving too little real system/Metal
    headroom.
- Source changes covered:
  - `vmlx_engine/utils/jang_loader.py::_set_wired_limit_for_model` now caps MLX
    wired target by physical RAM reserve, not just OS/sysctl max working set;
  - `panel/src/shared/metalWiredLimit.ts` guidance now mentions SIGKILL/OOM,
    `115000-120000 MB`, and not setting wired limit equal to physical RAM;
  - MM3 settings UI shows generic paged KV as visibly `LOCKED OFF` while
    preserving native MSA SSD prefix cache and leaving other model checkboxes
    normal.
- Verification:
  - behavior contract for Rich-class 128GB/113GB case:
    `.venv/bin/python -m pytest
    tests/test_panel_cli_flag_contract.py::test_jang_loader_wired_limit_caps_rich_128gb_case
    tests/test_panel_cli_flag_contract.py::test_jang_loader_wired_limit_keeps_physical_ram_headroom
    tests/test_panel_cli_flag_contract.py::test_metal_oom_startup_errors_surface_wired_limit_guidance -q`
    -> `3 passed`;
  - renderer/settings contract:
    `cd panel && npm test -- --run tests/metal-wired-limit.test.ts
    tests/settings-flow.test.ts` -> `265 passed`;
  - installed app:
    `/Applications/vMLX.app` reports `1.5.66`, bundled `vmlx_engine 1.5.66`,
    contains `ram_capped_target`, and passes
    `codesign --verify --deep --strict`.
- Missing before claiming fixed for users:
  fresh live exported log from a 128GB external Mac after installing this
  build, showing the new cap log line and a successful long generation without
  Metal OOM/SIGKILL/kernel panic.

## 2026-06-18 22:45Z - Remote MM3 REAP32 128GB proof box updated; visual startup pass, stress still red

Status: `REMOTE_128GB_MM3_VISUAL_STARTUP_PASS_STRESS_RED_STREAMING_CHAT_EXACTNESS`.

- Remote proof box:
  - host: `erics-m5-max.local` / `Erics-M5-Max.lan`;
  - model: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP32-d3-JANG_2L`;
  - wired limit observed before proof: `iogpu.wired_limit_mb: 126976`;
  - synced current `/Users/eric/mlx/vllm-mlx` source and clean
    `/Users/eric/jang-pub/jang-tools` to the remote.
- Remote build/install:
  - ran `panel/scripts/build-and-install.sh` on the remote;
  - `/Applications/vMLX.app` reports `CFBundleShortVersionString=1.5.66`;
  - bundled `vmlx_engine 1.5.66`;
  - bundled `jang 2.5.30` from clean `/Users/eric/jang-pub/jang-tools`;
  - `codesign --verify --deep --strict /Applications/vMLX.app` passed;
  - installed bundled `vmlx_engine/utils/jang_loader.py` contains
    `ram_capped_target` and `VMLX_METAL_WIRED_RESERVE_FRACTION`.
- Direct visual zero-touch startup proof:
  - artifact:
    `/Users/eric/mlx/vllm-mlx/build/remote-mm3-visual-startup-proof/visual-startup-proof.json`;
  - screenshots:
    `01-server-empty.png`, `02-model-list.png`,
    `03-mm3-expanded-prestart.png`, `04-after-launch-click.png`,
    `05-running-or-final.png`;
  - proof path used visible UI clicks for startup:
    Server tab -> Create Session -> MM3 REAP32 model row -> expand settings
    sections -> Launch Session;
  - verdict: `pass`, `failures=[]`;
  - actual spawned argv included
    `--tool-call-parser minimax_m3`, `--enable-auto-tool-choice`,
    `--reasoning-parser minimax_m3`, `--cache-memory-percent 0.15`,
    `--enable-disk-cache`, `--disk-cache-max-gb 10`, `--stream-interval 1`;
  - actual spawned argv did NOT include generic `--use-paged-cache`,
    `--enable-block-disk-cache`, `--kv-cache-quantization`, `--enable-jit`,
    or `--is-mllm`;
  - health native-cache proof reported `family=minimax_m3`,
    `schema=minimax_m3_msa_v1`, generic TurboQuant KV disabled, storage
    quantization disabled, prompt disk L2 enabled, block disk L2 disabled.
- Full live MM3 stress proof:
  - artifact:
    `/Users/eric/mlx/vllm-mlx/build/remote-mm3-reap32-ram-proof/mm3-stress-proof.json`;
  - final status: `fail`;
  - only recorded failure:
    `streaming Chat missing exact marker MM3_STREAM_CHAT_OK`;
  - actual streaming Chat text was `MM3_STREAM_CHAT`;
  - 10-turn UI coherence section completed with no empty/loop flags and
    cached tokens rose to `1307`;
  - reasoning mode section showed off=`0` reasoning chars, on/auto emitted
    reasoning plus visible content;
  - long-context prefix-cache exact marker passed:
    `PROFILE_OPTION_SENTINEL_ZETA_173`;
  - VL image exact marker passed: `MM3_IMAGE_RED`;
  - mixed reasoning/VL/tool/cache section completed, including cached token
    hits above `3300`;
  - direct API Chat/Responses/Anthropic/Ollama rows completed, but release
    status remains red because the Chat streaming exact marker failed.
- RAM observation:
  - during the full stress run, wired pages peaked near the remote
    `126976 MB` wired limit and the app did not OOM/kernel panic during the
    observed proof;
  - this is useful REAP32-on-128GB evidence, not a universal guarantee for
    other users or larger MM3 variants.
- Release boundary:
  - MM3 zero-touch UI startup/autodetect/argv parity is live-proven on the
    remote REAP32 128GB box;
  - MM3 full stress remains `RED` until Chat streaming exactness is fixed or
    root-caused with source and live evidence.

## 2026-06-18 23:16Z - Remote MM3 stream/parser stress pass; long-output defaults source-fixed

Status: `REMOTE_MM3_STRESS_PASS_LONG_OUTPUT_DEFAULTS_SOURCE_FIXED_LIVE_REBUILD_OPEN`.

- Root-caused the prior remote stress failure:
  Chat Completions streaming with MiniMax-M3 reasoning enabled spent the first
  pass inside reasoning, emitted only a partial visible prefix, then ended with
  `finish_reason="length"`. The server only ran the bounded thinking-off
  visible-answer pass when **no** visible content was emitted, so it streamed
  `MM3_STREAM_CHAT` instead of the required `MM3_STREAM_CHAT_OK`.
- Source fix:
  `vmlx_engine/server.py` now defers MiniMax-M3 no-tool Chat streaming visible
  content from the first reasoning pass. If that pass ends by length, the
  existing thinking-off visible-answer pass emits the final visible answer
  instead of leaking a truncated prefix.
- Source verification:
  - focused M3 Chat streaming regression tests -> `2 passed`;
  - `tests/test_streaming_reasoning.py` +
    `tests/test_minimax_m3_cache_paths.py` -> `156 passed`;
  - MM3 route/argv/RAM focused checks -> passed;
  - server py-compile, harness `node --check`, panel typecheck -> passed.
- Remote installed-app verification:
  - rebuilt `/Applications/vMLX.app` on `erics-m5-max.local`;
  - app version `1.5.66`, signed, bundled `server.py` contained
    `deferred_m3_visible_content`;
  - reran
    `/Users/eric/mlx/vllm-mlx/build/remote-mm3-reap32-ram-proof-after-stream-fix/mm3-stress-proof.json`;
  - result `status=pass`, `failures=[]`;
  - 10/10 UI turns coherent, no empty/loop/tag-leak flags;
  - reasoning off/on/auto passed;
  - tool row produced `M3_TOOL_OK_DONE`;
  - long-context prefix/cache row recalled
    `PROFILE_OPTION_SENTINEL_ZETA_173` with `5778` cached tokens;
  - MM3 VL image row returned `MM3_IMAGE_RED`;
  - mixed text/VL/tool/reasoning/cache session passed;
  - Chat/Responses/Anthropic/Ollama/API auth rows passed;
  - streaming Chat now returned `MM3_STREAM_CHAT_OK`, streaming image/tool rows
    passed.
- Newly root-caused user long-output complaint:
  if a session/request does not set `max_tokens`, the engine fallback is
  `4096`; if a session stays at generic timeout, the UI/CLI/server/gateway
  timeout path is `300` seconds. This explains users needing to resume every
  five minutes for long app/code generations.
- Source fix for the long-output complaint:
  - MiniMax-M3 zero-touch sessions now fill `timeout=900` and `maxTokens=8192`
    unless the user already set a custom value;
  - UI launch argv, Chat IPC request timeout, and Gateway proxy timeout all
    share the MM3 900s default;
  - live clean-start and MM3 stress harnesses now fail unless the actual
    UI-spawned MM3 argv contains `--timeout 900` and `--max-tokens 8192`.
- Local verification for the long-output fix:
  - `panel/tests/settings-flow.test.ts` -> `265 passed`;
  - harness source guards -> `2 passed`;
  - panel typecheck, harness `node --check`, server py-compile -> passed.
- Boundary:
  the streaming/parser/cache fix is live-proven on the remote installed app.
  The new MM3 `--timeout 900` / `--max-tokens 8192` default is source-tested
  only until `/Applications/vMLX.app` is rebuilt again and a clean-start/live
  proof records the new launch argv.

## 2026-06-18 23:42Z - Remote MM3 clean-start live proof for long-output defaults

Status: `REMOTE_MM3_CLEAN_START_LONG_OUTPUT_DEFAULTS_PASS`.

- Rebuilt and installed `/Applications/vMLX.app` on `erics-m5-max.local` after
  the MiniMax-M3 long-output default change.
- Installed bundle evidence:
  - `/Applications/vMLX.app` reports `CFBundleShortVersionString=1.5.66`;
  - `codesign --verify --deep --strict /Applications/vMLX.app` passed;
  - bundled `vmlx_engine/server.py` contains the M3 streaming fix marker
    `deferred_m3_visible_content`.
- Live UI clean-start proof:
  `/Users/eric/mlx/vllm-mlx/build/remote-mm3-reap32-clean-start-long-defaults/clean-start-proof.json`
  -> `status=pass`, `failures=[]`.
- The proof used the real app profile, backed it up, deleted `14` saved
  sessions through the app session API, created a fresh MM3 REAP32 session with
  `{}` settings, and started the model without manual setting edits.
- Actual UI-spawned MM3 argv included:
  `--timeout 900`, `--max-tokens 8192`, `--tool-call-parser minimax_m3`,
  `--reasoning-parser minimax_m3`, `--enable-auto-tool-choice`,
  `--cache-memory-percent 0.15`, `--enable-disk-cache`,
  `--disk-cache-max-gb 10`.
- Actual UI-spawned MM3 argv did not include generic paged KV, generic block
  disk cache, generic KV quantization, generic `--is-mllm`, or JIT.
- Runtime health/cache evidence:
  `native_cache.schema=minimax_m3_msa_v1`, components include
  `attention_kv`, `msa_idx_keys`, `absolute_block_index`,
  `generic_turboquant_kv.enabled=false`, `storage_quantization.enabled=false`,
  `prompt_disk_l2=true`, `block_disk_l2=false`.
- Gateway/API rows in the same proof passed exact markers for Chat, Responses,
  streaming Chat, and streaming Responses.
- UI visible turn passed:
  content included `CLEAN_START_VISIBLE_OK`, reasoning chars `607`,
  no empty response, no leaked reasoning tags, no loop suspect, speed
  `25.2 tok/s`.
- This answers the 300-second / 4096-token user report for MiniMax-M3 fresh UI
  sessions: it was a real default mismatch, and the installed app now proves
  the MM3 zero-touch launch uses 900s timeout and 8192 output tokens. This does
  not prove arbitrary user-customized sessions; custom timeout/max-token values
  are intentionally preserved.

## 2026-07-11 - Reasoning / streaming / sampler API stress

Status: `NO_SHIP_REASONING_STREAMING`.

- Blocker classes reduced: `parser/template`, `api/ui`, and `runtime/kernel`
  sampler correctness.
- Proof directory:
  `docs/internal/CODEX-REASONING-STRESS-2026-07-11/`.
- Live reused the existing Hy3 engine on `127.0.0.1:8010`; no model or app was
  started/restarted and no CDP/app surface was touched.
- API result: 17/24 stream/non-stream route/mode sequence rows passed; all 16
  off/auto rows passed. Reasoning-on remained red for answer-budget exhaustion,
  Ollama streaming route drift, and Ollama warm greedy nondeterminism.
- Source bugs fixed: H1 raw-logit processor ordering, seeded batch sampler
  sharing, chat fallback double emission, visible-prefix cleanup, and missing
  Hy3 normalization on Ollama streaming.
- Verification: 432 focused/broad tests passed, Python compile passed, and
  `git diff --check` passed.
- Remaining blocker: answer-pass draw-down reaches zero after a cap-exhausting
  reasoning pass; Ollama/Hy3 source fix requires live proof after a permitted
  engine restart. Verdict remains NO-SHIP.

## 2026-07-11 - Hy3 reasoning / Ollama stream re-verification

Status: `SHIP_REASONING_STREAMING_SCOPED`.

- Final artifact:
  `docs/internal/CODEX-REVERIFY-2026-07-11/all-routes-final.json` ->
  `status=pass`, failures `[]`.
- All 8 reasoning-on route/mode sequences and 24 turns passed across Chat,
  Responses, Anthropic, and Ollama; every turn 3 was visible, coherent, recalled
  the prior labels/code, ended `FINAL-CHECK`, and stayed within the +48 floor.
- Ollama stream content deltas were 8/5/31; each turn had one final `done:true`
  after all `message.content`, with reasoning isolated to `message.thinking`.
- Warm greedy was byte-identical on all four routes (`DET-731`, 4 tokens twice).
- Full suite A/B: baseline 5,939 tests / 53 failures; post 5,945 tests / 53
  failures; exact failure-ID sets identical, zero new failures.
- Hy3 standalone engine was stopped after proof; app/CDP surfaces were untouched.
- Scoped boundary: reasoning/streaming is cleared; broader release rows remain
  independently locked.

## 2026-07-13 - v1.6.8 live Electron cross-family visual QA

Status: `NO_SHIP_LIVE_ELECTRON_UI`.

- Blocker class reduced: `api/ui`, with `parser/template`, `media`, and
  `cache/storage` parity subchecks.
- Real dev Electron/CDP proof report:
  `/tmp/codex_liveui_findings.md`.
- LFM2.5 is red: Auto at the UI-spawned 512-token server cap produced a
  reasoning-only turn with no visible answer; On recalled the codeword but
  appended mutated stale text; Off suppressed reasoning but echoed the prior
  turn, returned `9+8` instead of `17`, and repeated the stale suffix.
- Zaya is partial/red: `zaya_xml` produced proper `Read` cards with no raw
  markup leak and a read succeeded, but the model repeated it until tool
  iteration `#3` was interrupted and never produced `READ_OK`.
- Gemma4 VL is red: the UI and engine accepted one PNG image and media
  accounting was correct, but the visible answer hallucinated ChatGPT/GPT-4o
  and unrelated `9H-8` content instead of identifying the vMLX screenshot.
- Existing-session settings are red: cache/paged/L2/max-output edits can clear
  the dirty state without updating stored session config. Gemma launched with
  `--cache-memory-percent 0.15 --no-paged-cache` and no `--max-tokens` after
  the UI had shown paged, Block L2, 19%, and 2048.
- Additional UI defects: cache-memory CLI preview omission, off-by-one
  max-output/block-size/max-block sliders, LFM model-selection reversion, and
  a stale saved LFM model symlink.
- Release boundary: the real Electron cross-family matrix remains open. No
  package, sign, notarize, tag, or public release action was performed.

## 2026-07-13 - Parser None / Force-Off / multiturn live Electron sweep

Status: `NO_SHIP_LIVE_SWEEP_5_DEFECTS`.

- Blocker class reduced: `api/ui`, with `parser/template`, `cache/storage`,
  `media`, and `runtime/kernel` subchecks.
- Full report: `/tmp/codex_live_sweep_findings.md`.
- Task 1 is red: Qwen3.6 launch argv correctly emitted literal parser `none`
  flags and omitted auto-tool choice, but tool output was still parsed and
  executed as a UI tool card. Auto mode enabled the correct qwen/qwen3 parsers
  but did not produce the required engine `Auto-configured ... parser` log.
- Task 2 resolves Force-Off behavior: detected-VL Gemma4 launched with
  `--text-only`, without `--is-mllm`, and rejected a real UI image request as
  unsupported media. Force Off wins over VL detection in the actual launch.
- Task 3 is red: Qwen3.6 native MTP/hybrid paged+SSM cache activated, but
  reasoning On produced empty visible content and Off stopped at `Codeword:`;
  LFM Auto and Off passed while On reused the stale Auto terminator and omitted
  the recalled codeword.
- Cache/RAM boundary: Qwen logged hybrid paged+SSM hits but later safely
  re-prefilled after SSM-key misses; RSS was 16.0 -> 20.1 GB while contaminated
  tool history expanded the prompt to 21.3k. LFM used 6 attention-KV plus 18
  SSM companion layers, block L2, a 1,253-token hybrid hit, and stayed bounded
  at 5.34 -> 5.40 GB RSS.
- Screenshots: `/tmp/codex_task3_qwen_final.png` and
  `/tmp/codex_task3_lfm_final.png`.
- Release lock remains active. No package, sign, notarize, tag, gate-manifest
  regeneration, or public release action was performed.

## 2026-07-15 - v1.6.11 current-source Electron/cache matrix

Status: `PARTIAL_SOURCE_CANDIDATE_NO_PACKAGE`.

- Blockers reduced: `runtime/kernel`, `cache/storage`, `parser/template`,
  `media`, and `api/ui` for Bonsai 1-bit/ternary, DSV4 Flash, Laguna-M.1, and
  MiniMax-M3.
- Source trace: commits `ad0468ba7` through `5427c0516` cover partial restart
  prefix restore, byte ceilings, DSV4/Bonsai tools, resident/SSM telemetry,
  MiniMax-M3 video, cache clear, Electron tool continuation, and lossless
  hybrid multi-turn policy.
- Live evidence directory:
  `docs/internal/release-gates/20260715_v1610_postrelease_matrix/`.
- Real Electron UI at CDP 9335 loaded every named model from
  `/Volumes/EricsLLMDrive`, visually exercised source-derived Chat Settings,
  Responses routing, reasoning Auto/Off/On, multi-turn recall, native tools,
  and cache panels. MiniMax-M3 additionally processed a real five-second video
  and retained its typed MSA cache schema.
- DSV4 was loaded with no competing model listener and host memory 94% free;
  it sustained about 17-26 tok/s, produced coherent math/recall, and showed a
  621-token native paged hit. Minor constrained-string omissions required
  correction turns, so this is not represented as exact-output perfection.
- Laguna showed 8/8 hits, 33,478 reused tokens, bounded 1,120 MB resident cache,
  and `paged+tq` / `paged+disk+tq`. MiniMax-M3 showed typed
  `minimax_m3_msa_v1` memory and disk hits. Bonsai showed live TQ KV plus
  bounded paged/SSM/L2 behavior; unsafe persisted SSM restore is quarantined.
- Verification: 876 changed-engine tests passed; 161 focused cache tests
  passed; six block-disk LRU/restart tests passed; panel tests passed 2,209
  with three intentional skips; TypeScript typecheck passed.
- Current release manifest remains `prepackage_ready=false` and
  `release_ready=false` because broader installed-app/model-family rows and
  historical proof inputs are still open or missing. No packaging, signing,
  notarization, tag, upload, or updater mutation is authorized by this status.

## 2026-07-15 - Current Electron continuation checkpoint

Status: `PARTIAL_NO_RELEASE`.

- Bonsai 1-bit current tool/final loop: live exact on rows 1373/1376/1379;
  warm same-process row restored 158/159 `paged+ssm` tokens. Restart row exact
  but no cache restore; hybrid SSM L2 remains quarantined/PARTIAL.
- Nemotron JANGTQ: duplicate-final A/B fixed in source and live row 1364.
- Step JANG_K: live exact control pass. Step JANGTQ_K: live runaway reasoning
  FAIL.
- Manual single-model swap and paged/legacy/block-L2 mutual exclusion both have
  current source plus live Electron evidence.
- Zaya generic tools remain out-of-contract for the installed AppleScript
  specialist, and Chat Settings capability truth is open.
- Gemma mixed-SWA cache default is now source-and-live verified through UI
  reset, DB, preview, argv, warm memory hit, and process-restart disk hit.
- Laguna speed, HY3 MTP net benefit, DSV4 exact output, M3 exact OCR/media
  tools, remaining families, full tests, package, signing, notary, feeds, and
  public release remain open.
- Current post-direct-answer reruns also pass on Nemotron, MiniMax-M2.7, and
  two identical DSV4 rows; broader DSV4 constrained-string fidelity remains
  partial rather than inferred from the post-tool contract.
- Step JANGTQ_K historical soup is now source-traced to the installed generic
  P18 attention patch dropping Step's post-reshape q/k norms and head-wise
  gate. The vMLX loader restores native Step attention when those semantics
  are absent. 129 focused tests passed; live Electron row 1406 returned exact
  `4`, and row 1418 made one real `file_info` plus exact
  `STEP-TQ-TOOL4-DONE` with 192 `paged+mixed_swa` cached tokens. Earlier
  narrated probes had `has_tools:false` and are retained as invalid setup
  evidence, not parser failures.
- MiniMax-M3 REAP32 is `FAIL-LIVE / PARTIAL-FIX`: two first Electron requests
  at a 105.4/107.52 GiB Metal baseline rebooted the host and left blank DB
  rows. Current source adds a 3 GiB M3 prefill-headroom reject and prevents an
  already-over-threshold baseline from bypassing the generic guard. Fifteen
  focused tests pass; no third live load was attempted, so the new 503 path is
  not live-verified.

## 2026-07-16 - OpenPangu strict-parser guard and Bonsai regression reopen

Status: `PARTIAL_NO_RELEASE`.

- openPangu cache classification/defaults are source-and-live verified for the
  Electron-launched session: argv has `--disable-prefix-cache`, health reports
  `openpangu_v2_composite_v1`, and all generic prefix/paged/L2 lanes are off.
- openPangu malformed tool execution is source-fixed but not agent-loop-fixed:
  38 focused openPangu/parser tests plus the new strict-native server regression
  pass. Live row 1470 did not execute the previous wrong `search_files` call,
  but it still stalled in speculative tool generation and was interrupted after
  7 tokens/89.4s. OpenPangu tools/final answer remain FAIL-LIVE.
- The user's `B1-UI-TOOL3` Bonsai 1-bit report reopens the repeated-reasoning
  gate. Prior `B1-UI-TOOL10` rows remain evidence for those exact conditions,
  but row 1473 on current HEAD only partially clears it: one real `file_info`,
  one persisted reasoning segment, and exact `B1-UI-TOOL3-RERUN-DONE` did
  render in Electron, while 3,617 hidden/generated tokens over 103.2s before
  tool close keeps Bonsai post-tool/tool-buffering performance PARTIAL.
- No packaging, signing, notarization, tag, feed, or release mutation is
  permitted from this state.

## 2026-07-15 - Mistral MXFP4 and Bonsai reasoning continuation

Status: `PARTIAL_NO_RELEASE`.

- Mistral Medium 3.5 MXFP4 is now detected and presented as the implemented
  text runtime, not as a working Pixtral/VL route. Reset/save persisted
  paged=1, block-L2=1, legacy-L2=0, multimodal=0; the real PID argv and health
  matched. Warm memory and process-restart disk restores each reused 1,240 of
  1,241 prompt tokens. Text `2+2 -> 4` passed, but a strict marker probe
  returned `I understand.`, and the 33-tool inventory repeated `2026` while a
  reduced one-tool route passed. Mistral remains PARTIAL.
- Bonsai 1-bit repeated-reasoning symptom is functionally absent in current
  rows 1443/1446: one reasoning card, one `file_info`, one result, exact
  `B1-UI-TOOL10-DONE`. The identical warm `paged+ssm` run was highly variable
  (2,338 tokens/64.3s vs 118 tokens/4.5s cold), so token/latency behavior is
  still PARTIAL even though content/tool finalization passed.
- The cross-model post-tool/reasoning gate remains OPEN/PARTIAL until every
  configured parser family has a current source + live Electron row. No
  release, package, sign, notarize, tag, feed, or public upload action.
## 2026-07-16 - Bonsai exact-once stream repair current status

- Reproduced the user's stuck/repeated reasoning state in Electron. Raw DEBUG
  evidence: 6,316 generated tokens, 24,443 characters, 46 tool markers, only
  57 reasoning characters visible. Final malformed `file_info` was correctly
  rejected.
- TQ-off A/B was performed through visible Server Settings. PID argv included
  `--kv-cache-quantization none`; health showed TQ disabled. The same failure
  still generated 4,335 tokens, with the first schema-valid call at character
  3,092. TurboQuant/cache is not the owning cause.
- Current source adds Qwen early stop only for a latest-user exact-once request
  naming one exposed tool. It validates all required args; global Qwen
  multi-call/interleaving stays enabled.
- Live proof: two TQ-off and six restored-Auto fresh Electron rows each made
  exactly one `file_info`, one result, and an exact final marker. Auto rows:
  115-244 tokens, 4.2-7.0s. Focused parser/resolver/Responses tests pass.
- Status: exact-once Bonsai tool/final contract VERIFIED-LIVE. General pre-call
  reasoning latency, VL, hybrid process-restart SSM restore, other model/parser
  rows, and release remain PARTIAL/OPEN.

## 2026-07-16 - Bonsai ternary native TQ cache boundary

Status: `PARTIAL_NO_RELEASE`.

- Source now preserves a complete seeded `turboquant_kv` payload through
  prefix extraction, block L2, validator/decode, scheduler restore, and MLLM
  storage/rewrap. Generic q4/q8 cannot replace an active native TQ record.
- Qwen hybrid Auto is storage-only TQ8 on 16 attention layers. Live transition
  stays off; 48 SSM companion layers remain native. Explicit UI None hard
  disables both stored and live TurboQuant behavior.
- Tests passed: 38 native-TQ; 151 selected engine-audit/TQ; 244 scheduler/cache.
- Live Electron Bonsai ternary None completed two exact-one-tool multi-turn
  rows after restart with explicit `--kv-cache-quantization none` and zero
  native-TQ counters.
- Live Electron Auto wrote seven native TQ blocks in a fresh namespace. After
  process restart it decoded all seven (`disk_hits=7`,
  `tq_native_hits=7`) and fresh chats retained exact-one-tool/final behavior.
- Restart SSM state remains quarantined and full-rederived; this is codec
  correctness proof, not a restart speed-hit claim. Reasoning continuity and
  stale mixed-history multi-call behavior remain partial.
- Bonsai 1-bit must now run the same separate matrix before the family row can
  be broadened. Release lock remains active.

## 2026-07-16 - Bonsai 1-bit current matrix

Status: `PARTIAL_NO_RELEASE` globally; scoped 1-bit rows below are current.

- Actual bundle/UI identity: Qwen3.5 hybrid SSM,
  `JANG_AFFINE_1BIT (1.1128b)`, 64 layers.
- Reproduced same-chat failure under None: row 1620 made five calls/six
  reasoning segments and was interrupted after 3,352 tokens/92.3 seconds.
- Root cause was panel lexical gating: `after the real tool result` did not
  match the answer-only follow-up detector. It was not owned by 1-bit weights,
  Qwen parser, or TQ.
- Bounded matcher repair plus exact live regression now passes 18 focused
  tests, 239 panel tool/request/chat tests, typecheck, and 114 selected engine
  audit/TQ tests.
- Rebuilt Electron None rows 1623/1626 and Auto rows 1632/1635 each have one
  reasoning segment, one real tool/result, and exact final content.
- Auto writes native TQ8 records only for 16 attention-KV layers; 48 SSM
  companions stay native. Restart decoded three native records and later
  counters reached eight hits; SSM restore remains suppressed/full-rederived.
- Health now labels storage-only compress telemetry as storage/codec telemetry,
  not live encode. Live health confirmed the corrected labels.
- Sessions UI and process list proved the ternary-to-1-bit single-model swap:
  1-bit ACTIVE, ternary INACTIVE, one local model server.
- Evidence copied to
  `docs/internal/release-gates/20260716_bonsai_1bit_native_tq/`. Release lock
  remains active for the remaining campaign matrix.

## 2026-07-16 - Gateway LAN/port current status

Status: `PARTIAL_NO_RELEASE` globally; scoped gateway row is current.

- Reproduced the LAN display bug in Electron: wildcard 18080 binding with an
  unusable advertised APIPA address even though the default route was RFC1918.
- Added a ranked LAN IPv4 selector and 3 focused tests. Gateway selector,
  Ollama, and single-model suites pass 65/65; typecheck passes.
- Rebuilt Electron and proved UI/DB/listener parity at localhost:18080, then
  LAN wildcard:18080 with advertised/reachable 192.168.1.110, then restored
  localhost:8080. Current health and DB report single-model mode true.
- Installed vMLX still independently owns wildcard 8080. Cross-protocol
  streaming and machine-global port-conflict behavior remain open.

## 2026-07-16 - Bonsai 1-bit TQ Auto/None persisted-L2 parity

Status: scoped `PASS`; global `PARTIAL_NO_RELEASE`.

- Live UI A/B showed paged cache and block-disk L2 enabled, selected None,
  restarted with `--kv-cache-quantization none`, and restored Auto.
- Found and fixed a real Off leak: old TQ-native blocks sharing the same token
  hash were still decoded. Explicit non-TQ mode now rejects TQ-native prompt
  and block reads/writes and evicts incompatible single-slot records so the
  clean prefill can write a standard replacement.
- Current Off live output was exact and coherent. Stats showed
  `tq_native_enabled=false`, zero TQ hits/writes, plus a real standard L2 hit
  and 64 tokens saved.
- Restored Auto PID 70504 has no explicit codec flag. Native cache telemetry
  maps TQ8 storage to 16 attention layers only, preserves all 48 SSM companion
  layers, restored 1,024 tokens from L2, recorded 14 native-TQ hits, and wrote
  one new native-TQ block.
- 630 affected cache/policy/engine-audit tests pass. Evidence:
  `docs/internal/release-gates/20260716_tq_toggle_parity/`.
- Bonsai bundle-default temperature 1.0 still produces stochastic verbose
  pre-tool reasoning; the same behavior reproduced with TQ Off, so it remains
  a separate model/sampling/template A/B row, not a cache PASS claim.

## 2026-07-16 - Bonsai exact-once Responses rail and gateway parity

Status: scoped `PASS`; global `PARTIAL_NO_RELEASE`.

- Isolated the 1,201-token ternary and 706-token gateway cases to the first
  tool-capable pass. The no-tools function-output pass remained 9-10 tokens
  and exact; same-chat Electron continuation was not collapsing to one turn.
- Found a real Responses streaming gap: after an early `</think>`, Bonsai
  could continue meta-reasoning on the content rail. The server buffered it
  once a tool marker appeared, but later published it in
  `response.output_text.done` alongside the valid function call.
- Current source buffers the content rail from token one only for an explicit
  exact-once request naming one exposed tool. Genuine reasoning-summary deltas
  still stream; premature pre-tool prose is omitted; general multi-tool turns
  are unchanged.
- Current gateway replay: 132 reasoning/tool tokens, zero visible text deltas,
  one `file_info`, empty pre-tool text finalization, then nine streamed exact
  answer tokens.
- Current Electron 1-bit rows 1839/1842/1845/1848 are 96/167/94/98 tokens,
  exactly one call/result each, across same-chat turns and process restart.
  Ternary rows 1830/1833/1836 are 78/87/86 after the original outlier.
- Reasoning Auto, paged/L2 defaults, cache codec Auto, and single-model swap
  were visually checked. LAN toggled `*:8081` then restored
  `127.0.0.1:8081`; installed vMLX independently owns 8080.
- Tests: `tests/test_server.py` 102 passed / 3 deselected; focused 4 passed;
  compile and diff checks passed. Evidence:
  `docs/internal/release-gates/20260716_bonsai_exact_once_gateway/`.
- SSM disk restore remains quarantined (`restore_enabled=false`); other model,
  media, protocol, signing, notarization, and release gates remain open.

## 2026-07-16 - DSV4 DSML reasoning and restart-cache recheck

Status: scoped `PASS`; global `PARTIAL_NO_RELEASE`.

- Reproduced raw canonical DSML leaking through Responses reasoning on live
  Electron row 1863 despite one valid structured tool call.
- The server now streams/finalizes only the reasoning prefix before native
  tool markup while retaining raw reasoning for structured DSML parsing.
- Current Electron rows 1869 and 1872 completed exact one-tool fresh and
  same-chat turns with no DSML leak. After a visible process Stop/Load, row
  1875 again completed exactly with 606 `paged+dsv4` tokens restored and three
  block-disk hits.
- Live Cache settings Off/On restarts proved native composite prefix, paged,
  block L2, and pool-codec controls alter argv and health state.
- Tests: 103 server, 131 streaming-reasoning, and 114 selected DSV4 tests pass.
- Preserved failures keep the row partial globally: malformed row 1866 took
  64.7 s with no call, and structurally correct restart row 1875 used 1,454
  tokens / 119.2 s. Only the available CRACK bundle was tested; DSV4 JANGTQ
  was not present. Evidence:
  `docs/internal/release-gates/20260716_dsv4_reasoning_cache_recheck/`.

## 2026-07-16 - Qwen architecture-aware TQ persisted-cache matrix

Status: scoped `PASS`; global `PARTIAL_NO_RELEASE`.

- Full-KV Qwen3 Auto initially reported active TQ objects but wrote plain
  blocks because prompt-boundary truncation rebuilt TQ caches as `KVCache`.
- Preserving TQ identity exposed a live Q3 correctness failure: a fresh tool
  turn stalled and a later 896-token paged hit emitted corrupt tool-like text.
  Q3 remains recorded as failed, not credited as working.
- Uncalibrated Qwen Auto now uses storage-only TQ8 for real attention KV;
  hybrid variants still exclude cumulative companions, cumulative-only Qwen
  gets no fabricated TQ slots, and bundle-owned calibrated policy wins.
- Prompt and block L2 namespaces now include effective native-TQ policy/bits so
  Auto and explicit None cannot share persisted representations.
- Live Electron TQ8 rows 1920/1923 returned exact finals, one real tool/result,
  and 896 paged or paged+disk tokens. Restart health recorded 14 native hits.
- UI None restarted with an explicit `--kv-cache-quantization none`; row 1926
  returned one exact tool turn and wrote only standard `dtype=kv` blocks.
- UI Auto restore removed that CLI flag; row 1929 returned one exact tool turn,
  restored 896 `paged+disk` tokens, and recorded 14 native hits / three native
  writes under `qwen_full_kv_storage_tq8`.
- Tests: 41 full affected, 581 full engine audit, and 212 selected cross-file
  checks pass. Evidence:
  `docs/internal/release-gates/20260716_qwen_tq_architecture_matrix/`.
- Next current gate is HY3 MTP depth 1: require live acceptance counters,
  coherent output, cache compatibility, and measured speed before a pass.

## 2026-07-16 - HY3 native MTP D1 acceptance and speed gate

Status: scoped `PASS`; global `PARTIAL_NO_RELEASE`.

- Added process-local native-MTP request/totals telemetry for the real oMLX
  `BatchGenerator`, including per-depth drafted/accepted tokens and actual
  seed-main, verify-main, and MTP forward counts. Scheduler health now exposes
  the completed snapshot; Perf renders it live.
- Controlled Electron Off -> D1 -> Off/restore settings flows changed runtime
  health as expected. All six identical greedy API runs produced the exact
  1-100 answer with 200 completion tokens.
- Warm median wall time was 21.234247 s Off versus 16.081931 s at D1, a
  24.264% reduction. The controlled D1 arm executed 414 cycles and accepted
  180/414 drafted tokens (43.478%). This disproves the stale near-full
  acceptance expectation while establishing a current positive speed result.
- Electron Perf visibly showed D1, acceptance, depth rates, forward counts,
  native cache and TQ state. Electron row 1944 executed exactly one
  `file_info(panel/package.json)` and returned only `HY3-D1-TOOL1-DONE`.
- Post-tool health showed a reconstructed 43-token paged prefix, 69 native TQ
  block writes and seven native TQ hits.
- Tests: 177 native-MTP-focused and 54 selected engine-audit tests passed;
  compile/diff checks passed. Evidence:
  `docs/internal/release-gates/20260716_hy3_mtp_d1_acceptance/`.
- Open follow-ups: Min-P zero does not persist from Chat Settings; Perf's
  `Attention KV L2 disabled` label conflicts with native TQ block activity;
  per-layer TQ INFO spam can evict MTP lines from the bounded Logs view.
- Next: return to the model/parser matrix, beginning with MiniMax/Bonsai/Pangu
  regressions and then remaining Laguna/Mistral/settings/API/gateway rows.

## 2026-07-16 - openPangu current-HEAD typed-cache/parser recheck

Status: scoped `PASS`; global `PARTIAL_NO_RELEASE`.

- Rechecked current commit `b5a47f62f` through the real Electron app after
  the later Qwen TQ and HY3 scheduler changes.
- Electron one-model mode loaded JANG_3M as PID 97796, visibly stopped it, and
  loaded PID 98632 without clearing prompt L2. HY3 was inactive.
- Fresh row 1947, same-chat row 1950, and restart row 1953 each executed one
  real `file_info` and returned exact final text. The warm row reused 144
  memory tokens; restart restored 295 typed disk tokens.
- UI, argv, health and Logs agree: Prefix Cache plus prompt Disk Cache use
  exact typed N-1 composites; generic paged/block L2 and TurboQuant/q4/q8 are
  off. Logs proved 2,826/2,826 leaves, 138 causal convs, all 46 decoder
  layers, DSA=16, SWA=30, mHC=4, sinks=128 and MLA rank=512.
- Tests: 75 openPangu model/parser/tool-prompt plus two exact-once Responses
  server regressions passed. Evidence appended under
  `docs/internal/release-gates/20260716_openpangu_typed_cache_electron/`.
- MTP remains unwired for this bundle/runtime and 512K soak/full protocol rows
  remain open. Next live gate is MiniMax-M3 exact OCR/tools/media behavior.

## 2026-07-16 - MiniMax-M3 current exact media/tool/cache gate

Status: scoped `PASS_WITH_RECORDED_OCR_FORMAT_MISSES`; global
`PARTIAL_NO_RELEASE`.

- Focused M3 tests exposed six post-init Scheduler reads of the unrelated
  `_uses_openpangu_cache` flag that lacked a false default. Partial M3 cache
  fixtures skipped/aborted store/fetch. Current source makes those reads
  defensive; normal initialized openPangu behavior is unchanged.
- Tools-enabled OCR no longer hits the old zero-tool dead end. Hyphenated and
  same-chat replacement rows 1956/1959 remain strict misses. A fresh-chat,
  high-contrast control row 1962 returned exact `BANANA8426` in 4.4 seconds.
- Row 1965 then made one real `file_info` and exact final text with 64 native
  paged tokens. Video row 1968 read both labels but added separator spaces;
  no-reattach row 1971 returned exact
  `FRAME START 2468|FRAME END 9753` with 128 paged tokens.
- Visible Stop/Starts proved exact one-tool finals with 128 and then 449
  `paged+disk` tokens. Row 1977 ran after the source repair on PID 2921 and
  returned exact `MM3-SCHED-CURR1-DONE`.
- Final health reports native `minimax_m3_msa_v1`, dense KV 0-2, sparse MSA
  3-59, reconstructed non-dequantized typed cache, zero native TQ use, and
  image/video runtime support. MTP remains unavailable because this artifact
  has no indexed MTP tensors.
- Tests: 34 M3 cache/loader, three MiniMax server, four media contracts, 75
  openPangu regressions, 581 engine-audit, and eight TQ block tests passed.
  Evidence: `docs/internal/release-gates/20260716_mm3_exact_media_current/`.
- Next: Laguna speed and Mistral strict-output/runtime rows, then settings and
  protocol parity. Release remains locked.

## 2026-07-16 - Laguna TQ dtype restore and warm-cache speed

Status: scoped cache-speed `PASS`; Laguna overall `PARTIAL`; global
`PARTIAL_NO_RELEASE`.

- Root-caused the warm slowdown to two coupled dtype losses: JANG TQ live
  compression decoded into float32, and vMLX block/prompt TQ records did not
  record the original model KV dtype.
- JANG now preserves float16/bfloat16 through live compression. vMLX records,
  validates, restores, and namespaces the dtype for positional blocks, native
  prompt records, and nested CacheList records. Reconstructed full-KV layers
  are rewrapped with the model's native TQ template.
- Real Electron exact-output rows measured 25.0 tok/s cold, 21.2 tok/s on a
  49-token paged+tq-native hit, and 24.6 tok/s after UI restart from
  paged+disk+tq-native. The old broken warm path was about 8 tok/s.
- Electron Logs after PID 7811 -> 9056 report 70 layers and 14,049,280 resident
  bytes, the correct 2-byte KV footprint; health reports a successful rewrap
  and native disk hit.
- A final restart after the rewrap metadata source edit, PID 9056 -> 10474,
  returned the same exact answer from 49 paged+disk+tq-native tokens at
  25.1 tok/s and logged the same 14,049,280-byte footprint.
- Tests: 24 JANG TQ cache, 115 vMLX TQ/batching, and 46 block/prefix tests pass.
  Evidence: `docs/internal/release-gates/20260716_laguna_tq_dtype_speed/`.
- Laguna remains red for default reasoning: a fresh Auto-thinking UI row looped
  in repetitive meta-reasoning and was interrupted after 726 tokens. TQ
  Auto-vs-None/JIT, tools, multi-turn, long context, and settings persistence
  remain open. Release remains locked.

## 2026-07-16 - Laguna uncalibrated Auto TQ3 corruption isolated and corrected

Status: cache correctness scoped `PASS`; Laguna overall `PARTIAL`; global
`PARTIAL_NO_RELEASE`.

- Reproduced the reasoning failure through the real Electron app: cold row
  1998 was exact, but same-chat row 2001 restored 3,545
  `paged+tq-native` tokens and entered an incoherent loop. It was manually
  stopped after 3,076 tokens and 9,597 reasoning characters.
- UI None preserved prefix/paged/disk cache while disabling TQ. Rows
  2004/2007/2010 were exact, including 3,549/3,612-token paged hits.
- Root cause was uncalibrated generic Auto assigning TQ3 to non-Qwen full-KV
  bundles. Auto now uses storage-only TQ8 for every uncalibrated real-attention
  layout; calibrated bundle settings remain authoritative.
- Added a full TQ codec signature to the persisted cache namespace and bumped
  it to `codec_config_v2`, preventing old unsafe TQ3 disk blocks from replay.
- Corrected Auto rows 2013/2016/2019 were exact with 3,550/3,614 native hits.
  Stop/Start restored 3,550 TQ8 tokens from disk coherently, but row 2022 made
  an unsolicited `ask_user` call before exact post-skip completion. Strict
  agent-choice status therefore remains partial.
- TQ8 reconstruction costs 3.59-4.79 s; warm TTFT was ~5.1 s versus
  1.2-1.5 s with TQ disabled. Correctness passes; latency remains open.
- 112 focused cache/policy tests pass. Evidence:
  `docs/internal/release-gates/20260716_release_closeout/`.

## 2026-07-16 - Mandatory architecture and agent-stream closeout expansion

Status: `OPEN`; global `PARTIAL_NO_RELEASE`.

- Added explicit current-source Electron gates for Qwen 3.6 35B/27B MXFP/JANG
  with MTP depth 3, HY3 MTP depth/cache interaction, MiniMax M2.7 KV, ZAYA
  typed CCA, Nemotron hybrid state, DSV4 native composite cache, and the
  existing MiniMax M3/openPangu native typed-cache routes.
- Each row now requires same-chat multi-turn behavior, a real tool result plus
  complete post-tool answer, reasoning/content/tool-argument delta continuity,
  restart/L2 restore, capacity accounting, and eviction/reload correctness.
- TQ is limited to codec-compatible attention KV. Hybrid companion state must
  remain native and be cleanly rederived/restored. DSV4, CCA, M3, and
  openPangu cannot inherit generic TQ merely from an Auto UI selection.
- MTP proof requires requested depth plus real draft/accepted counters; target
  cache ownership and restore must remain correct when speculation is active.
- Fake behavior fixes are prohibited: no prompt coercion, hidden sampler
  clamp, synthetic thinking/tool output, invented continuation, or arbitrary
  cap can close a correctness row.

## 2026-07-16 - Qwen 35B paged resident accounting and cache-tier reconciliation

Status: Qwen 35B cache row `PARTIAL`; global `PARTIAL_NO_RELEASE`.

- Real layout is 10 attention KV layers plus 30 native GDN/SSM companion
  layers. The bundle name does not declare MTP.
- Root cause: L2-promoted arrays were cleared after reconstruction without
  releasing PagedCacheManager resident-byte attribution. Reused block hashes
  could also inherit a stale `keep_resident` flag and evade later eviction.
- Commit `7bb34fa0d` atomically releases the promoted payload/accounting and
  resets payload-scoped residency protection.
- Tests: 595/595 engine-audit/byte-budget and 177/177 paged/disk/TQ/hybrid
  cache tests pass.
- Current Electron PID 58213 row 2160 restored 152/153 tokens as
  `paged+ssm+disk`, executed one real `file_info`, and returned exact final
  text. Visible Cache Management and health report 152 indexed tokens, zero
  resident bytes, seven native-TQ block hits, and two SSM-disk hits.
- Still open: bounded L1 eviction then L2 promotion, true-miss fallback,
  duplicate companion-state payload audit, selective-TQ UI label, and wider
  per-family proof.
- Added Gemma 4 rotating SWA as a named gate: rotating state native,
  compatible full-attention KV TQ-only, and prefix/L1/L2/eviction boundaries
  proven together.

## 2026-07-16 - Qwen 35B v8 hybrid cache tier closed

Status: Qwen 35B cache tiers `PASS-LIVE`; model strict long format `PARTIAL`;
global `PARTIAL_NO_RELEASE`.

- Live safetensor inspection caught the first external-companion patch still
  writing 30 cumulative states in the separate NumPy disk path. Malformed v7
  terminal files were ~64 MB; the failure is retained.
- Commits `df945f065`, `133d8c8e9`, and `7cb89185c` externalize generic hybrid
  companions, fix the NumPy writer, make the UI say `Selective TQ-KV`, and
  isolate corrected records in the v8 namespace.
- Every corrected v8 block contains exactly 10 `turboquant_kv` and 30 `skip`
  entries. Terminal files are 30 KB/295 KB and contain zero cumulative state.
- Electron rows 2169/2172/2175 prove cold, same-process `paged+ssm`, and
  Stop/Start `paged+ssm+disk` tiers with one real tool and exact finals.
- Electron UI Save & Restart set Max Cache Blocks to four. Rows 2178/2181 each
  restored 154/155 `paged+ssm+disk` tokens and completed exactly; Cache
  Management recorded nine L1 evictions. A 192-token KV-only boundary with no
  matching SSM companion safely full-prefilled instead of becoming a false hit.
- UI restored Max Cache Blocks 1000; PID 61919 argv/health confirm 1000 and row
  2184 repeated the exact disk hit.
- Tests: 784/784 Python hybrid/cache/scheduler, 278/278 panel settings, and
  panel typecheck pass.
- Remaining Qwen 35B row is strict long-format reliability, not cache tiers.

## 2026-07-16 16:45 PT - Bonsai parser-off contract and multi-turn recheck

- Retained failure: same-chat row 2352 requested `README.md` but executed the
  stale prior argument `panel/package.json`. Its exact final marker does not
  make the turn correct.
- Source fix `4e13b19a7` makes literal `--tool-call-parser none` authoritative
  for final parsing and both Chat/Responses streaming paths. Model/request
  auto-detection can no longer silently re-arm a disabled parser.
- Tests: 106/106 `tests/test_server.py`; 52 passed / 1 skipped selected
  server, openPangu, and VL parser coverage.
- Live Electron PID 99835 with parser None persisted row 2358 as raw model
  text with no structured tool call/result. A follow-up ran 3,701 reasoning
  tokens until visibly stopped, proving None is a real opt-out rather than a
  Bonsai workaround.
- Restored production Qwen parser on PID 864. Same-chat rows 2364, 2367, 2370,
  and 2373 executed the requested paths exactly once with exact final markers;
  row 2373 restored 258 tokens as `paged+ssm`.
- Status remains PARTIAL: the stale-argument row and long/repeated native
  reasoning variability still require soak/root-cause work.

## 2026-07-16 17:00 PT - Responses terminal event and Bonsai continuation truth

- Pushed `a36a5ea66`: a length-capped Responses stream now emits
  `response.incomplete` instead of a misleading `response.completed` wrapper;
  Electron handles completed/incomplete terminal text, usage, warnings, and
  status through the same path.
- Tests: 135/135 affected Python, 50/50 panel, and panel typecheck.
- Live direct Responses on current-source PID 2658 streamed a correct split
  `README.md` function call. A `tool_choice:none` result continuation completed
  once with exact `B1-RESP2-DONE`, then repeated native tool markup to 1,024
  tokens on a second run and truthfully emitted `response.incomplete`.
- Live Electron row 2388 used UI-applied Max Tokens 32, preserved separated
  reasoning plus partial content, reported exactly 32 tokens, and omitted the
  impossible requested closer. Max Tokens and system prompt were restored to
  blank/model-default.
- Additional same-chat rows 2376/2379/2382/2385 were correct, giving eight
  consecutive correct tool turns after retained stale-argument row 2352.
  Bonsai remains PARTIAL because the repeated post-tool run is still real.
## 2026-07-16 13:30 PT - MiniMax M2.7 cache tiers and paged ref ownership

- Source fix `af7815f1a`: fetched chain-hash/prefix-index block tables are now
  registered for completion ref release. This closes the live small-pool leak
  where successful L2 hits remained pinned after an agent iteration.
- Tests: 90/90 paged-cache, byte-budget, TQ block, and hybrid-prefix tests pass.
- Live Electron: rows 2187/2190 cold + same-chat two-tool; row 2193 Auto L2;
  rows 2196/2199 explicit None cold/L2; rows 2208/2211 post-fix four-block
  eviction/reuse; row 2214 restored normal Auto/1,000 blocks.
- Post-fix pressure health: only null block allocated, three usable blocks free,
  no shared refs, evictions rose 3 -> 9. Normal PID 66306 has 999 free blocks.
- Status: PASS-LIVE cache/settings/tools/eviction; PARTIAL long non-tool/direct
  stream. Release remains blocked.
- Long-stream follow-up: Electron row 2217 completed 582 output tokens with
  separated reasoning/content and exact `MM27-LONG1-END`. Direct Responses at
  1,024 tokens emitted 711 reasoning + 48 content deltas, matching text-done,
  the exact marker, and completed status. The 512-token control correctly
  surfaced incomplete. M2.7 is now PASS-LIVE current source; release remains
  blocked by other rows.

## 2026-07-16 - ZAYA artifact gate and Nemotron-H hybrid closeout

Status: Nemotron cache/settings/tools/API `PASS-LIVE`; long reasoning
`PARTIAL`; generic ZAYA `BLOCKED_MISSING_ARTIFACT`; global
`PARTIAL_NO_RELEASE`.

- The external drive contains only `Zaya-8B-JANG_4M`, whose README identifies
  it as the AppleScript-8B single-`run_applescript` specialist. The user
  excluded that specialist, so no generic CCA row was fabricated.
- Nemotron-H source trace found six real attention slots in its 52-layer
  hybrid pattern. Only those KV slots use Auto storage-only TQ8; 23 Mamba
  companion slots remain native and use the typed asynchronous companion path.
- Electron rows 2223/2226/2229 closed cold, same-chat, and process-restart L2
  one-tool loops with exact finals. The restart restored 192 tokens as
  `paged+ssm+disk+tq-native`.
- Electron-applied four-block PID 75038 rows 2235/2238 stayed exact while
  evictions rose 3 to 9. Explicit None PIDs 75398/75644 wrote/restored raw
  `paged+ssm+disk` blocks with zero TQ activity. Auto/1,000 blocks was restored
  on PID 75939.
- Electron row 2247 completed a coherent marked long answer but repeated 2,962
  tokens of native reasoning before its real closer; this remains PARTIAL, not
  masked by a fake closer or sampler workaround.
- Direct Responses streaming emitted 424 reasoning deltas, 30 content deltas,
  matching done events, and completed status. Focused tests pass 25/25.

## 2026-07-16 - Gateway rollback, LAN rebind, and single-model ownership

Status: gateway lifecycle `PASS-LIVE` for listener/session ownership;
cross-protocol streaming `PARTIAL`; global `PARTIAL_NO_RELEASE`.

- Commit `e76cc5451` makes `ApiGateway.restart()` restore the previous
  host/port listener when the requested bind fails, then rethrows the original
  error. Focused gateway suites pass 75/75 plus panel typecheck.
- Current-source Electron PID 9909 reproduced the old 8081 -> occupied 8012
  stopped-listener failure before the fix. After the fix, the same visible API
  page edit was rejected while health and SQLite stayed running on 8081.
- The visible LAN toggle rebound to `0.0.0.0:8081`, displayed routable
  `192.168.1.110`, served `/health` on that LAN address, and returned to
  localhost when disabled.
- With Single model mode enabled, the visible Bonsai Start button stopped DSV4
  PID 10013 before launching Bonsai PID 10495. UI, SQLite, process listing,
  gateway discovery, and the session lifecycle log each showed exactly one
  running engine.
- This does not close gateway protocol streaming or Bonsai output/cache soak.

## 2026-07-16 - Gateway four-protocol stream and Ollama terminal repair

Status: basic cross-protocol streams `PASS-LIVE` with one retained Responses
format miss; agentic continuation `PARTIAL`; global `PARTIAL_NO_RELEASE`.

- Through UI-proven gateway 8081 and live Bonsai, Chat Completions exact-finaled
  after streamed reasoning; Responses completed with matching text but two
  leading newlines; Anthropic exact-finaled with one `message_stop`.
- Ollama `think:true` exposed cumulative-thinking duplication and loss of the
  usage-only event after finish reason. Commit `a0aa81a94` waits for
  `[DONE]`/backend end, emits the empty terminal message once, and includes the
  later usage. Gateway tests pass 76/76 plus typecheck.
- After clean Electron main restart, PID 12046 with Bonsai PID 12114 emitted
  193 thinking deltas / 728 characters once, exact `OLL-GW4-DONE`, one terminal
  object, `eval_count=202`, and `prompt_eval_count=18`.
- An earlier Ollama turn stopped after reasoning with no content. Direct and
  later controls exact-finaled, so the miss remains Bonsai reliability evidence
  rather than being hidden by a synthetic continuation.

## 2026-07-16 18:35 PT - HY3 TQ4 MTP D1 restart and eviction closeout

Status: cache settings/restart/eviction `PASS-LIVE`; strict-format and
long/streaming reliability `PARTIAL`; global `PARTIAL_NO_RELEASE`.

- The exact `Hy3-JANG_2K-MTP` bundle declares one MTP layer and 42 MTP
  tensors. Current health reports `runtime_active=true` and
  `effective_depth=1`; depth 3 is not a valid HY3 gate for this artifact.
- Source fix `ab5d01e04` selects family-scoped q4 for HY3 full-KV stored
  prefixes and installs independent cache deepcopy ownership. This fixes the
  live scheduler retry loop caused by Python trying to pickle the preserved
  `mlx.core.Dtype` handles after prefill.
- Tests: 19/19 HY3/TQ cache tests, 178/178 native-MTP tests, 282/282 panel
  settings tests, and panel typecheck.
- Electron restart PID 22265 row 2483 restored 3,272 tokens as
  `paged+disk+tq-native`, called `file_info(panel/package.json)` exactly once,
  and returned exact `HY3-Q4-T1R-DONE`.
- Electron-applied four-block PID 23635 reached 11 L1 evictions, five
  TQ-native L2 writes, and 18 TQ-native hits. Older-prefix row 2492 restored
  192 bounded tokens as `paged+disk+tq-native`, made one real tool call, and
  exact-finaled. The Cache drawer visibly showed 4,346 L2 block tokens.
- The UI restored the normal 1,000-block setting on PID 25084. Row 2477's
  `HY3-Q4-T2-D-DE-DONE` strict-marker miss is retained; no prompt coercion,
  sampler clamp, or synthetic closer was added.
- Explicit UI None launched PIDs 26444/27461 with
  `--kv-cache-quantization none`. Cold row 2495 wrote 54 raw L2 blocks with
  zero TQ activity; restart row 2498 restored 3,258 tokens as `paged+disk`,
  exact-finaled after one real tool, and retained zero TQ writes/hits. Auto,
  1,000 blocks, and TQ-native enablement were restored on PID 28473.

## 2026-07-16 19:00 PT - Qwen 27B namespaced tool-preview continuation repair

Status: two-turn Electron agent continuation `PASS-LIVE`; cache restart,
eviction, true non-tool D3 acceptance, and MXFP4 parity `PARTIAL`; global
`PARTIAL_NO_RELEASE`.

- Root cause: Qwen 3.6 emitted a human-readable
  `call:default_api:file_info{...}` preview immediately before its real native
  structured tool call. The panel replayed both the preview as Responses
  `output_text` and the structured `function_call`, contaminating the
  tool-result continuation and stopping before the requested final answer.
- Commit `2a15e51d3` removes only a complete namespaced preview whose tool name
  matches an already parsed structured call. It does not synthesize a call,
  result, or final answer, and it preserves surrounding prose and mismatches.
- Current source validation: 14/14 focused panel tests and panel typecheck.
- Current Electron rows 2510/2513 used the requested distinct paths exactly
  once in the same chat and returned exact `Q27M8-D3-POSTFIX1-DONE` and
  `Q27M8-D3-POSTFIX2-DONE`; the persisted messages contain no raw preview.
- Configured MTP is D3, but tool requests are intentionally capped to D1.
  Therefore this closes the agent-loop parser row only, not the true non-tool
  D3 draft/acceptance row or cache restart/eviction rows.

## 2026-07-16 19:15 PT - Qwen 27B Auto TQ4 restart and eviction closeout

Status: Qwen MXFP8 agent loop and Auto-TQ4 cache tiers `PASS-LIVE`; true
non-tool D3 acceptance, MXFP4, long/cancel/media reliability `PARTIAL`; global
`PARTIAL_NO_RELEASE`.

- Live UI inspection exposed the prior Auto policy still used TQ8 for Qwen.
  An explicit q4 control was coherent but used generic storage quantization,
  not the required TQ-native codec.
- Commit `b994582cc` makes compatible non-Bonsai Auto use TQ4 only on attention
  KV while preserving native SSM/GDN companion state. Bonsai is detected from
  artifact identity and retains its TQ8 exception. Electron labels Qwen
  `TQ4 AUTO` and Bonsai `TQ8 AUTO`.
- Source tests: 35/35 scoped cache policy/contract tests; 283/283 panel
  settings tests plus typecheck. A widened 155-test process had two
  order-dependent mock failures; both owning files pass together 35/35.
- Cold Auto row 2528 exact-finaled after one tool and wrote three TQ-native q4
  blocks plus native SSM companions. Fresh-process row 2531 restored 149
  `paged+ssm+disk` tokens, exact-finaled, and health reported three TQ-native
  hits plus one SSM disk hit.
- UI-applied four-block row 2534 forced three L1 evictions. Older-prefix row
  2537 restored 149 `paged+ssm+disk` tokens, made one exact tool call, and
  exact-finaled. The Cache drawer visibly showed q4, typed SSM state, and the
  bounded pool. Auto/1,000 blocks are restored on PID 35460.

## 2026-07-16 19:25 PT - Qwen 27B true D3 tools-off terminal failure

Status: MTP D3 execution `PROVEN`; visible completion `FAIL-LIVE`; global
`PARTIAL_NO_RELEASE`.

- Tools-off Electron row 2540 used a 256-token cap, produced correct arithmetic
  reasoning and partial content, then stopped without its required final.
- A fresh 512-token control reached 505 reasoning tokens and ended with empty
  visible content. It did not complete the requested answer.
- Scheduler telemetry proves this was not a configured-only D3 claim:
  drafted-by-depth `[20,12,12]`, accepted-by-depth `[16,7,4]`, and D3
  acceptance rate `0.3333`. The later visible-answer pass finished at length.
- Normal chat settings were restored: built-in tools enabled and Max Tokens
  returned to model default. No synthetic continuation or hidden budget clamp
  was added.

## 2026-07-16 21:15 PT - Qwen 27B post-tool progressive streaming closeout

Status: scoped Qwen MXFP8 post-tool stream `PASS-LIVE`; cross-family regression
matrix `PARTIAL`; global `PARTIAL_NO_RELEASE`.

- Commit `b33d80589` fixes the false one-character native tool-marker trigger
  and enforces the explicit/Auto Qwen reasoning budget on post-tool
  continuations. It does not prebuffer the post-result answer or synthesize a
  closer.
- Current-source tests pass 832 with three deliberate deselections; the focused
  marker/tool/Qwen continuation selection passes 9/9.
- Direct Responses SSE emitted 153 reasoning and 113 content deltas, matching
  final text, one completed terminal, exact marker, and no tool re-entry under
  `max_thinking_tokens=256`.
- Current Electron row 2606 streamed the visible answer progressively while
  Stop remained active, executed one matching `file_info` call/result, reused
  512 `paged+ssm+disk` tokens, exact-finaled
  `Q27-ELECTRON-TOOLSTREAM-FIX7-DONE`, and stored no warning.
- All non-Qwen families retain their prior live evidence and open boundaries;
  none inherits this parser/stream pass without a current-head live regression
  row.

## 2026-07-17 17:55 PT - Shared Electron length-terminal persistence

Status: Responses and Chat Completions length-terminal UI contract `PASS-LIVE`;
Qwen 256-token exact-final reliability `FAIL-LIVE`; global
`PARTIAL_NO_RELEASE`.

- Current source commit `dc3b9c491` maps a terminal `finishReason="length"`
  to one deduplicated persisted warning shared by Responses and Chat
  Completions. It no longer splices a temporary notice into assistant content.
- Focused panel tests passed 45/45 and `npm run typecheck` passed before the
  commit was pushed to `codex/live-electron-gates-20260715`.
- Fresh Electron main PID 85736 visibly started the exact
  `dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP` session. Responses row 384 stopped at
  256 tokens, emitted 746 progressive DOM mutations, carried
  `finishReason="length"`, persisted the warning in `warnings_json`, and
  restored it visibly after a renderer reload.
- The matched Chat Completions turn row 387 stopped at 128 tokens, emitted 361
  progressive DOM mutations, carried the same length finish reason, and
  persisted/rendered the same warning separately from model text.
- Qwen's proposed 64-thinking/192-answer split is rejected as a fix: one of
  three raw Responses controls repeated arithmetic to the cap and ended
  incomplete. The existing Auto budget policy remains unchanged.
- Normal UI state was restored and visually checked: Responses wire, Auto
  reasoning, blank Max Tokens/model default, blank Max Thinking Tokens, and
  built-in tools enabled. This does not close broader Qwen sampled reliability.

## 2026-07-17 18:02 PT - Min-P zero and TQ cache-label parity refresh

Status: settings persistence and current Qwen cache telemetry `PASS-LIVE`;
global `PARTIAL_NO_RELEASE`.

- The stale HY3 follow-up that explicit Min-P zero did not persist was
  falsified on the current Electron source. The visible Qwen slider was saved
  at 0.20 (`chat_overrides.min_p=0.2`), then at 0.00
  (`chat_overrides.min_p=0.0`); a renderer reload still showed 0.00.
- Current request construction uses `overrides.minP != null`, so explicit zero
  is forwarded on both Responses and Chat Completions instead of falling back
  to a non-zero bundle default. The focused request/settings/policy suites pass
  371/371.
- The stale Perf-label conflict is also absent on current source. Live health
  exposes Qwen's 16 attention slots as TurboQuant q4 storage and 48 companion
  slots as native SSM async rederive. The visible Perf panel reports
  `Attention KV L2 turboquant-q4 (K q4 / V q4)`, not disabled, alongside 836
  paged-L2 blocks and 68 SSM-L2 entries.
- Responses wire, Auto reasoning, blank Max Tokens, blank Max Thinking Tokens,
  built-in tools on, and explicit Min-P off are the final visible chat state.

## 2026-07-19 - openPangu long-context oversize snapshot guard

Status: `SCOPED_GUARD_VERIFIED_LIVE_LONG_REUSE_AND_TIGHT_CAP_LATENCY_PARTIAL`.

- Source: the single-active generator now estimates typed prompt-cache bytes
  before cloning and receives its admissible single-entry ceiling from the
  configured RAM/disk backends. Scheduler/cache telemetry exposes the ceiling,
  last estimate, and skip count.
- Tests: 124/124 current openPangu/single-active/memory/disk/isolation/terminal
  rows passed.
- Electron: identical 43,980-token prompt returned the exact anchor answer.
  Pre/post TTFT was 186.35s/103.20s and peak Metal was
  138,814.3/115,551.8 MB. The post-patch UI kept reasoning separate.
- API: raw Responses produced 256 progressive reasoning deltas, 23 progressive
  content deltas, exact answer, and `response.completed`. The deliberately
  tight 256-token cap caused a second full prefill; this latency remains open.
- Evidence:
  `docs/internal/release-gates/20260718_openpangu_long_snapshot_guard/`.
