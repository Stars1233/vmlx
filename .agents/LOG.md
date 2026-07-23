# Agent proof log

## 2026-07-23 - MiniMax M3 paged-on/off SSD partial and restart proof

- Real UI Stop/Start removed PID `11370`, closed port 8008, and loaded PID
  `15531` with zero scheduler/L1 tokens plus 807 retained SSD blocks.
- Paged-On restart request restored 6,912/6,948 tokens as `paged+disk`, hit 108
  SSD blocks, exact-finaled, and reached first output in 2.05 seconds.
- The real UI kept Block Disk L2 checked and enabled while Paged RAM was
  switched Off. PID `16285` launched as `block_disk_only` with zero RAM tokens
  and restored 6,912/6,947 tokens from SSD.
- A second Paged-Off real UI Stop/Start loaded PID `16816` from zero RAM state
  and restored 6,912/6,950 tokens from the same SSD lineage.
- A suffix-only negative reused only 128/8,030 tokens, confirming safe
  longest-prefix reuse rather than arbitrary later-substring matching.
- Restored Paged-On + Block Disk L2 through the UI on PID `17435`.
- Evidence:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/m3-tool-math-live/`.
- Disk-cap eviction, media, signed app, and release-wide gates remain open.

## 2026-07-23 - Laguna ordinary post-tool continuation keeps requested reasoning mode

- Root-caused the missing post-tool reasoning mode to the panel request builder:
  an ordinary exact-final follow-up and a true empty-answer recovery shared the
  same helper, which deleted completed tools and forced thinking Off.
- Split the policy boundary in the owning helper. Ordinary continuation only
  removes completed tools; the bounded true recovery retains its instruct-mode
  behavior.
- Current verification: 22 focused panel tests, typecheck, real Electron
  Start-button load, exact-one `file_info` execution, real result continuation,
  and True-to-True engine thinking resolution after the fix.
- A separate deterministic explicit-On row persisted a real reasoning rail.
  The model's extra visible calculation is retained as an exactness caveat.
- Evidence:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/laguna-post-tool-thinking-live.json`.
- Overall v1.6.17 remains partial.

## 2026-07-23 - Gemma mixed-SWA SSD hierarchy live closure

- Through the real Server Settings UI, Paged RAM Off left Block Disk L2
  enabled. PID `95934` exact/partial requests restored 11,058/11,008 tokens
  from SSD with zero RAM residency.
- UI Stop/Start PID `96185` restored exact and changed-tail prefixes from the
  same persisted SSD lineage. The 10 GB cap evicted 106 blocks and returned to
  7.900 GB.
- After restoring Paged RAM On, PID `96442` promoted 173 SSD blocks on the
  first request and used resident RAM on the second; every answer was exact.
- Evidence:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/gemma-mixed-swa-ssd-restart-live.json`.
- No source change was made from the separate legacy MLLM prompt-disk
  key/offset lead because this production block-disk path did not reproduce a
  failure. Broader architecture and release gates remain partial.

## 2026-07-23 - Gemma 4 mixed-SWA prefix reconstruction and live proof

- Pushed `9f5b1bde2`: structural cache-layout detection now recognizes
  vendored `mlx_vlm` KV/Rotating classes, mixed-SWA avoids the SSM companion
  path, empty reconstructed hits fall back safely, and block telemetry reports
  `block_ids`.
- Electron PID `95416` passed three closely inspected turns with separate
  reasoning rails, rendered math, one real `file_info` result, exact finals,
  and cross-turn recall. Raw Chat and Responses separated reasoning/content
  deltas and leaked no control markers.
- Exact warm restored 42 tokens as `paged+mixed_swa` and reported one block.
  A deterministic 495-token cold/warm decode was byte-identical and exact.
- Tests: 8 selected passed; compile and diff checks passed. Evidence:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/`.
- Overall `.17` remains partial; SSD-only/restart/refault/q4 and broader
  family/protocol/media/settings gates remain open.

## 2026-07-19 - minimum-width five-locale expansion and full-suite gate

- Expanded the live-found locale repair through Code/chat-empty screens,
  remote picker/form, markdown code-copy feedback, waiting/empty-response
  states, session side panels, image-session labels, and fallback errors.
- Full current suites pass: Python 6,153 passed / 96 skipped / 92 deselected;
  panel 2,326 passed / 3 skipped; typecheck and current Electron production
  build pass. The clean-JANG bundled-Python production gate passed immediately
  before the renderer-only expansion.
- Full Electron at 600x760 passed all five Code locales at 600/600 with no raw
  catalog keys or clipped main elements. Japanese remote picker/form was also
  opened and visually inspected. Transient wait/empty/image-panel rendering,
  secondary/destructive modals, native sheets, full accessibility, and the
  signed app remain explicitly partial.
- Evidence: `docs/internal/release-gates/20260719_minwidth_locale_drawers/`.

## 2026-07-16 - Gemma 4 mixed-SWA cache/settings closeout PASS, long output PARTIAL

- Pushed `3385cb019`: Auto now wraps only Gemma 4 full-attention slots with
  TQ4, preserves native rotating-SWA slots, and fails closed on layout mismatch.
  Focused verification passed 153/153.
- Electron Auto/restart rows 2425/2428/2431 exact-finaled with one real tool;
  row 2431 restored 704 `paged+mixed_swa+disk` tokens and recorded 44
  native-TQ disk hits.
- Visible 16-block pressure recorded 38 L1 evictions. Row 2464 then restored
  the older ALPHA prefix from TQ-native L2 and exact-finaled.
- Explicit None PID 15388 and row 2467 proved ordinary disk writes with zero
  TQ-native writes/hits. Auto/1,000 blocks was restored on PID 15797; row 2470
  recorded 704 `paged+mixed_swa+disk`, three TQ-native writes, eleven hits,
  one real tool, and exact final.
- Pushed `ba68f8fba`: the live drawer now says
  `TQ4 full-attention KV + native rotating SWA / MIXED AUTO`; 281 settings
  tests and typecheck passed. Long constrained output remains open.

## 2026-07-16 - Gemma 4 tool-stream early stop PASS, cache tier proof still open

- Root-caused the Gemma/Bonsai-style visible stall on Gemma 4 with current
  Electron process in `--no-paged-cache --kv-cache-quantization none`: raw
  Responses output had a valid native tool call by token 20, then hallucinated
  `<|tool_response>` and an answer instead of stopping.
- Fixed `Gemma4ToolParser` to opt into completed-call stream stop and truncate
  at the last regex-parseable `<|tool_call>...<tool_call|>` match.
- Tests: `tests/test_gemma4_tool_parser.py` 13/13.
- API proof: same multi-turn shape dropped to 28 output tokens / 20 heartbeats
  and emitted one `file_info({"path":"README.md"})`.
- Live Electron proof: rows 2265/2268, same chat
  `0c1261aa-4e09-41e7-986c-ade3d4074357`, each made one real `file_info`
  call/result and exact final. Row 2268 used a 218-token memory cache hit and
  finished in 3.4s. Screenshot:
  `docs/internal/release-gates/20260716_release_closeout/gemma-current/gemma-uifix2-live-pass.png`.
- Restored the session through UI to Auto/paged/block-L2 and restarted. PID
  81973 argv has `--use-paged-cache`, block size 64, max blocks 1000, and
  block-disk L2. Rows 2271/2274/2277 then passed exact tool/final rows with
  `paged+mixed_swa+disk`, resident `paged+mixed_swa`, and post-restart
  `paged+mixed_swa+disk` cache details.
- UI-constrained four-block eviction row passed: PID 82455 launched with
  `--max-cache-blocks 4`, rows 2280/2283 stayed exact with 192-token
  `paged+mixed_swa+disk` restores, and health recorded `l1_evictions=9`.
  Normal 1,000 blocks were restored on PID 82981.

## 2026-07-16 - openPangu current-HEAD Electron/cache recheck

- Rechecked commit `8cfc9f269` through Electron CDP `9335`; 59 focused
  model/parser/exact-once tests passed.
- One-model mode stopped Bonsai. The sole openPangu argv had
  `--no-paged-cache --enable-disk-cache` and no generic TQ flag.
- UI and health matched the typed composite policy: prefix and prompt L2 on;
  paged blocks, block L2, stored/live TQ, and q4/q8 off.
- Rows 1851/1854 each made exactly one tool call/result and exact finals; row
  1854 reused 144 typed memory tokens.
- Visible Stop/Load changed PID 75458 to 76278. Row 1857 restored 295 prompt
  tokens from disk and again made one call/result with the exact final.
- Logs proved all 46 layers, 16 DSA, 30 SWA, four mHC streams, 128 sinks, MLA
  rank 512, 2,826/2,826 weights, and 138 causal convolutions.
- MTP remains runtime-unwired; broad release remains locked.

## 2026-07-16 - Bonsai Responses reasoning finalization PASS, long-call/cache health PARTIAL

- Reproduced the panel/server split: the Responses server had authoritative
  final text or a complete terminal reasoning summary, but the Electron client
  kept speculative tool buffering and persisted only the pre-marker reasoning
  fragment.
- Added guarded text-buffer recovery and terminal reasoning reconciliation.
  Raw Qwen/MiniMax/Harmony tool-control syntax is rejected; earlier
  interleaved reasoning segments are preserved.
- Current Electron row 1797 visibly rendered 158 reasoning characters,
  executed exactly one `file_info`, persisted one result, and finalized exact
  `B1-REASON-DONE-LIVE1-DONE` in 87 tokens/9.2s on the reused 2,535-token chat.
- Fresh row 1791 and reused-chat row 1794 also made one call/result and exact
  finals. Raw direct Responses evidence made one call in 111 tokens.
- Kept row 1788 red for latency variance: 2,128 tokens and 62.5s before the
  same valid one-tool/final result. This is not hidden as a parser PASS.
- Tests: panel 24/24, TypeScript typecheck, engine audit 580/580.
- Health after the proof: paged hit rate 0.9375, block L2 82 native-TQ writes
  and 393 native-TQ hits; top-level TQ health still false and SSM disk restore
  still suppressed, so cache telemetry parity/restart reuse remain open.
- Evidence:
  `docs/internal/release-gates/20260716_bonsai_reasoning_stream/`.

## 2026-07-15 - HY3 MTP depth-1 current Electron PASS, speedup unverified

- Drove visible Electron/CDP `9335`; selected
  `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP`, port `8010`.
- Health proved preserved native MTP candidate active for text:
  config/jang/index all show one layer, `42` MTP tensors,
  `runtime_active=true`, `effective_depth=1`.
- Health still says `speculative_decoding=not_configured` and exposes no
  acceptance/speedup counters, so this is activation proof only.
- Exact marker passed: `HY3-CURRENT-COHERENT-DONE` at `33.8 tok/s`.
- Multi-turn recall passed: `QUARTZ|719`, with `626` then
  `672 paged+tq cached`.
- Post-run health: `hy_v3` `plain_kv_v1`, q4 stored-prefix TQ,
  prefix+paged/block-L2 on, scheduler hits `2`, tokens saved `1,298`,
  block-L2 `53` blocks / `3,002` tokens, `63` disk hits.

## 2026-07-15 - Laguna-M.1 current Electron cache PASS, speed OPEN

- Drove visible Electron/CDP `9335`; selected
  `/Volumes/EricsLLMDrive/jangq-ai/Laguna-M.1-JANG_2L`, port `8015`.
- Health proved `laguna` / `plain_kv_v1` paged KV, generic TQ KV for plain
  attention KV, stored-prefix q4, prefix+paged/block-L2 on.
- Exact marker passed: `LAGUNA-CURRENT-COHERENT-DONE`.
- Multi-turn recall passed: `MARBLE|508`, with `618` then
  `677 paged+tq cached`.
- Post-run health: scheduler hits `2`, tokens saved `1,295`, block-L2 `14`
  blocks / `802` tokens, `63` disk hits, TQ objects active.
- Speed remains open: current UI rows are still around `24 tok/s`, not the
  target-speed lane.

## 2026-07-15 - DSV4 Flash CRACK current Electron cache PASS, exact marker PARTIAL

- Drove visible Electron/CDP `9335`; selected only configured DSV4 session
  `/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`, port
  `8012`.
- Health proved native `deepseek_v4_v7` composite cache with SWA local, CSA
  compressed pool, HCA compressed pool, incomplete tail state, pool quant on,
  generic TQ KV forced off, paged cache and block-L2 on.
- Exact marker row was partial: actual
  `DSV4-CURRENT-COHERENT-DENDONE` instead of
  `DSV4-CURRENT-COHERENT-DONE`; first-token TTFT was `65.64s`.
- Basic coherence/cache rows passed: arithmetic `45` with `346 paged+dsv4`;
  memory recall `BASALT|314` with `413 paged+dsv4`.
- Post-run health: scheduler hits `3`, tokens saved `1,142`, block-L2 `35`
  blocks / `7,644` tokens, `6` disk hits, `DSV4BatchGenerator` active.
- Evidence: `dsv4-crack-health-*.json`,
  `dsv4-crack-current-cache-recall.png`, and
  `dsv4-crack-current-db-rows.json` under the active gate directory.

## 2026-07-15 - Bonsai 1-bit/ternary current Electron proof PASS, UI tool PARTIAL

- Drove the same dev Electron on `erics-m5-max.local` over CDP `9335`; M3 was
  stopped visibly before loading Bonsai variants.
- Bonsai 1-bit (`jangq-ai/Bonsai-27b-1bit-JANG`, port `8030`) loaded with
  health quant `JANG_AFFINE_1BIT`, actual bits `1.1128`, hybrid
  `qwen3_5`/`hybrid_ssm_v1`, live attention TQ KV, SSM companion state, paged
  cache, and block-disk L2.
- 1-bit Electron rows: exact `B1-CURRENT-COHERENT-DONE`, then exact recall
  `CEDAR-B1|9417`; multi-turn metrics reported `paged+ssm`. Health recorded
  block L2 writes/hits and SSM companion disk entries/evictions.
- 1-bit Responses tool parser passed in non-stream and streaming
  thinking-disabled mode with final `file_info({"path":"panel/package.json"})`.
  The thinking-on streaming attempt completed reasoning-only and dropped an
  incomplete call, so it is kept as a model/prompt sensitivity artifact.
- 1-bit Electron built-in tool execution is partial: exact marker returned but
  DB row had no tool call/result.
- Bonsai ternary (`jangq-ai/Bonsai-27b-Ternary-JANG`, port `8020`) loaded with
  health quant `JANG_AFFINE_TERNARY_2BIT`, actual bits `2.0959`, the same
  hybrid SSM/TQ/paged/block-L2 cache policy.
- Ternary Electron rows: exact `BT-CURRENT-COHERENT-DONE`, then exact recall
  `SPRUCE-BT|6824`; multi-turn metrics reported `247 paged+ssm cached`.
  Streaming Responses tool parser emitted argument deltas and final
  `file_info({"path":"panel/package.json"})` with no warnings.
- Evidence is under
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`
  as `bonsai-1bit-*` and `bonsai-ternary-*`. No release action.

## 2026-07-15 - MiniMax-M3 tools-enabled media stream repair PARTIAL

- Drove the existing dev Electron on `erics-m5-max.local` over CDP `9335`;
  model path `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M3-Coder-Small`, engine
  port `8017`. The app was not replaced by API-only proof.
- Reproduced the tools-enabled VL failure: real preprocessing logged `551`
  image tokens, but invalid/incomplete native tool XML ended as a blank answer
  with a speculative tool heartbeat.
- Patched the Responses finalizer, M3 late visible-answer policy, non-stream
  M3 answer-pass parity, and renderer zero-tool status handling. Current
  verification passed: selected server M3/non-stream rows `4/4`,
  `tests/test_streaming_reasoning.py` `131/131`, panel tool-status `10/10`,
  and TypeScript typecheck.
- Current Electron image row now returns grounded visible text and no false
  zero-tool card. Tool path OCR passed; exact marker OCR remains partial
  (`MM3-DETI-DONE` versus `MM3-DET1-DONE`).
- Current Electron video row passed color bars, frames/timecodes, and the exact
  no-reattach follow-up marker. Follow-up telemetry showed `128 paged+disk`
  reused tokens; health retained typed `minimax_m3_msa_v1` memory/disk state.
- Fresh Electron genuine-tool regression passed after the fallback change:
  exactly one `file_info` card/result for `panel/package.json`, exact final
  `MM3-TOOL-POSTFIX-DONE`, and `4,271` paged cached tokens.
- After visible Electron `Stop`/`Start`, live non-stream Responses/Chat with
  tools available but unused returned exact markers
  `MM3-NONSTREAM-RESP-DONE` and `MM3-NONSTREAM-CHAT-DONE`.
- Session Settings cache parity passed visually and through process/health:
  default `0.15`/paged/block-L2 -> changed `0.12` with `Save & Restart` and
  lower L1 ceiling -> restored `0.15` with block L2 still enabled.
- `mtp.vl_runtime_available=false` was traced to native MTP+VL availability,
  not general media support. `/v1/capabilities` and live Electron image/video
  evidence agree.
- Evidence:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/mm3-vl6-*`
  and `mm3-video-*-current-code-*`.
- Scoped status remains partial pending exact-OCR row. No release action.

## 2026-07-15 - Zaya native AppleScript plus RAM/L2 Electron proof PASS

- Real dev Electron on `erics-m5-max.local`, CDP `9335`; Zaya loaded only on
  port `8013` from `/Volumes/EricsLLMDrive/jangq-ai/Zaya-8B-JANG_4M`.
- Auto/Off/On live native cards returned AppleScript exit code `0` and results
  `42`, `10`, and `11`; Auto second turn returned `81` with a
  `497 paged+zaya_cca cached` annotation. No repeated-call loop occurred.
- UI settings showed model-derived `temperature=1.00`, `top_p=0.95`, top-k
  Off, min-p `0.00`, repetition penalty `1.00`, and Responses routing.
- Cache panel showed `typed_cca / zaya_cca_v1`, bounded `63,936 / 64,000`
  resident tokens, and the new `Clear RAM` / `Clear Prefix + L2` controls.
- Live RAM-only clear preserved `64,229` block-L2 tokens. Exact replay restored
  `999` prefix blocks; block-disk hits increased `28 -> 1,027` with writes
  unchanged at `2,004`.
- Evidence:
  `docs/internal/release-gates/20260715_v1610_postrelease_matrix/zaya-final-*.png`.
- Final checks: panel `2,218 passed, 3 skipped`; focused Python `3 passed`;
  named no-heavy API/cache gate passed (`42` API-route rows); TypeScript and
  diff checks passed.
- Release manifest remains red (`prepackage_ready=false`,
  `release_ready=false`). Transient full-gate regeneration still left `244`
  open checklist rows and release-focused pytest had `649 passed, 1 skipped,
  6 failed`, so no release-adjacent action was performed.

## 2026-07-14 - Electron tool-history duplication live proof PASS

- Artifact: `/tmp/codex-toolhistory-verdict.md`.
- Existing dev Electron app at CDP `127.0.0.1:9333`; no relaunch.
- Model: external-drive `dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`, live
  `lfm2` tool parser, Reasoning Auto, built-in coding tools enabled.
- Controlled prompt counts: `1465 -> 1641 -> 1817 -> 1946`; turn-4
  edit-and-resend remained `1946` and byte-identical `TANGERINE-42`.
- Persisted OAI history contains only each current turn's tool exchanges;
  engine request shapes contain no replayed union.
- Screenshot: `/tmp/vmlx-toolhistory-controlled-final.png`.
- Scoped target PASS; default 50k tool results can still create large but
  payload-linear context, and broader release lock remains unchanged.

## 2026-07-13 - Zaya live Electron parser re-verification FAIL

- Artifact: `/tmp/codex_zaya_reverify_findings.md`.
- UI-restarted Zaya from old PID `59440` to repo-source PID `82010`; server
  remains running on `127.0.0.1:8013`.
- Explicit JSON test: old nested `path` object absent, but persisted malformed
  string paths failed and repeated to iteration guard; no final `READ_OK`.
- Natural single-path regression: no persisted tool call, no read card, empty
  visible answer, and 2048-token exhaustion.
- Engine log after restart: no `Dropping` and no `must be of type string`
  diagnostics; repeated fallback tool-schema-injection warnings remain.
- Screenshots: `/tmp/codex_zv_test1_fail.png`,
  `/tmp/codex_zv_test2_fail.png`,
  `/tmp/codex_zv_sessions_after_tests.png`.
- Scoped verdict: Zaya parser/tool/UI row remains OPEN; release lock unchanged.

## 2026-07-11 - Reasoning / streaming / sampler stress

- Artifact: `docs/internal/CODEX-REASONING-STRESS-2026-07-11/api-stress.json`
  (`status=fail`, 17/24 sequence rows passed, 24 assertion failures).
- Failure transcript bundle:
  `docs/internal/CODEX-REASONING-STRESS-2026-07-11/api-stress-failures.json`.
- Proofread/verdict:
  `docs/internal/CODEX-REASONING-STRESS-2026-07-11/PROOFREAD.md` and
  `docs/internal/CODEX-REASONING-STRESS-2026-07-11/FINAL.md`.
- Unit proof: 432 passed.
- Remaining boundary: no live post-fix engine restart was allowed; answer-pass
  reservation and Ollama/Hy3 post-fix E2E remain open.

## 2026-07-11 - Reasoning / Ollama re-verification PASS

- Final live proof:
  `docs/internal/CODEX-REVERIFY-2026-07-11/all-routes-final.json`.
- Final report:
  `docs/internal/CODEX-REVERIFY-2026-07-11/FINAL.md`.
- Result: 8/8 reasoning-on route/mode sequences passed; Ollama stream emitted
  incremental content before a sole terminal line; all warm greedy pairs were
  byte-identical.
- Full-suite A/B JUnit sets had 0 new failures (53 identical pre/post failures;
  six added regression tests passed).
- Verdict: SHIP for the scoped reasoning/streaming set; broader release lock
  unchanged.

## 2026-07-13 - Live Electron UI QA FAIL

- Report: `/tmp/codex_liveui_findings.md`.
- Key screenshots:
  - LFM empty Auto answer:
    `/tmp/codex_ui_lfm_auto_reasoning_only.png`
  - LFM incorrect Off answer:
    `/tmp/codex_ui_lfm_off_wrong_answer.png`
  - Zaya successful card plus repeated-call interruption:
    `/tmp/codex_ui_zaya_true_off_repeat_loop_expanded.png`
  - Gemma accepted image but ungrounded answer:
    `/tmp/codex_ui_gemma_vl_wrong_answer.png`
  - Gemma prelaunch settings/preview mismatch:
    `/tmp/codex_ui_gemma_settings_configured.png` and
    `/tmp/codex_ui_gemma_cli_preview.png`
- Runtime parity evidence was read from the UI-spawned process argv and the
  app-owned session logs. LFM fresh-session argv matched 17% paged+Block-L2 and
  max 512; Zaya matched 15% paged and max 1024; Gemma launched old 15% /
  no-paged / model-owned output values after the settings UI visually accepted
  different values.
- Result: LFM FAIL, Zaya FAIL, Gemma VL FAIL, settings parity FAIL. Release lock
  unchanged.

## 2026-07-15 - Bonsai/DSV4/Laguna/MM3 live Electron and cache proof

- Evidence root:
  `docs/internal/release-gates/20260715_v1610_postrelease_matrix/`.
- Source commits under proof: `ad0468ba7`, `693ad5e1e`, `79fed9f8d`,
  `9e4202034`, `80aecb703`, `75d9b8ed2`, `2dba3c2c4`, `471c86029`,
  `b6563c340`, and `5427c0516`.
- Electron screenshots include each model's current-source load logs, Chat
  Settings, Auto/Off/On turns, multi-turn transcript, native tool result, and
  cache panel. MM3 evidence additionally includes video attachment,
  Responses preprocess logs, description, and follow-up recall.
- Cache/source suites: 876/876 changed-engine, 161/161 focused cache, 6/6
  block-disk LRU/restart, 2,209 panel tests with three skips, and TypeScript
  typecheck.
- Scoped result: source candidate evidence is current for the named models and
  paths. Repository-wide package/notarization result remains blocked by the
  freshly regenerated manifest (`prepackage_ready=false`,
  `release_ready=false`); no release-adjacent command was run.

## 2026-07-13 - Parser / Force-Off / multiturn live Electron sweep FAIL

- Report: `/tmp/codex_live_sweep_findings.md`.
- Real dev Electron app was driven over sole-client CDP `127.0.0.1:9333`; all
  recorded model evidence used external paths under `/Volumes/EricsLLMDrive`.
- Verbatim session argv proves Qwen parser-None flags, Qwen Auto parser flags,
  Gemma4 `--text-only`, Qwen native-MTP D3 configuration, and LFM hybrid
  paged+block-L2 configuration.
- Counted defects: parser None still allows tool parsing/execution; Auto lacks
  the required engine auto-configuration log; Qwen On has empty final content;
  Qwen Off truncates before recall/exact output; LFM On reuses the prior stale
  terminator and omits recall.
- Scoped verdict: `NO_SHIP + 5 defects`; release lock unchanged.

## 2026-07-15 - Cross-model post-tool warning/finalization continuation

- Added the current-source cross-model matrix at
  `docs/POST-TOOL-CROSS-MODEL-MATRIX-2026-07-15.md` so API-only or shared-code
  evidence cannot silently promote an Electron model row.
- Source: added narrow superseded-recovery-warning removal after visible
  recovery content exists; retained all unrelated warnings.
- Focused proof: 48/48 panel tests and TypeScript typecheck passed.
- HY3 Electron: one `file_info`, one result, exact
  `HY3-POSTTOOL1-DONE`, no warning, `19.0 t/s`.
- Bonsai ternary Electron: one `file_info`, one result, exact
  `BT-POSTTOOL1-DONE`, one reasoning segment, no warning, `31.3 t/s`.
- Laguna Electron: one `file_info`, one result, exact
  `LAG-POSTTOOL1-DONE`, no warning, `16.0 t/s`, `3,612 paged+tq` cached.
- LFM source/live: the native-template shortcut accepted placeholder arguments
  as concrete; explicit named tools now get request-bound scalar examples.
  Eight focused tests passed. Pre-fix Electron made malformed/repeated calls;
  post-fix row 1322 made one exact `file_info`, one result, exact
  `LFM-POSTTOOL4-DONE`, no warning.
- Broad File I/O/Search/Shell rerun row 1325 also passed with one exact
  `file_info`, one result, exact `LFM-POSTTOOL5-DONE`, and no warning.
- Qwen3.6 27B broad Electron row 1328 passed with one exact `file_info`, one
  result, exact `Q36-POSTTOOL1-DONE`, no warning, and `22.6 t/s`; MTP D3 and
  hybrid cache were active in health, without a net-speedup claim.
- Gemma4 12B broad Electron row 1331 passed one exact tool/final row. A new
  cache-default parity issue was recorded because DB/argv/health show paged,
  prompt-L2, and block-L2 off despite the session's prefix-cache setting.
- MiniMax-M2.7 pre-fix row truncated the requested slash path. The native XML
  example extractor now preserves path separators; 26 focused fallback/parser
  checks passed, and post-fix row 1337 made one exact call/result plus exact
  `MM27-POSTTOOL2-DONE` with no warning.
- DSV4 Electron: pre-fix exact final retained a stale warning; post-fix one
  tool/result row had `warnings_json=null` and no rendered stale warning.
  Strict marker remained red because the model emitted
  `DSV4-PPOSTOLL2-DONE`.
- Evidence copied to
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.
- Verdict: warning lifecycle fixed and live-proven; cross-model coverage and
  DSV4 exact fidelity remain partial. Release lock unchanged.

## 2026-07-15 - Step/Nemotron/Bonsai live continuation

- Archived Step JANGTQ_K runaway failure and Step JANG_K exact control pass.
- Reproduced Nemotron duplicate final twice, removed duplicated agent prompt
  directive, broadened exact-output suppression, and live-proved one exact
  marker after rebuild.
- Reproduced Bonsai post-tool re-entry with TQ auto and explicit TQ none; the
  A/B ruled out TQ as owner. Added a narrowly scoped planned direct-answer
  follow-up for explicit single-tool exact-final contracts. Rebuilt live row
  completed, warm identical row restored 158/159 tokens, restart row was exact
  without L2 restore.
- Live session settings changed KV quantization `auto -> none -> auto`; argv
  reflected `--kv-cache-quantization none` during the diagnostic run.
- Cache-tuple normalization and manual single-model swaps were verified in the
  real Electron profile. Release lock unchanged.

## 2026-07-15 - Gemma4 mixed-SWA default/L2 continuation

- Traced the real bundle to 40 sliding plus 8 full-attention layers. Kept
  generic paged cache off and used the compatible legacy prompt-L2 lane.
- Electron Reset Defaults visibly changed the stale all-off session to
  prefix/paged/legacy/block `1/0/1/0`; preview, saved DB, PID argv, and health
  matched.
- Cold/warm/restart rows 1385/1388/1391 each made one exact `file_info` call
  and exact final. Warm restored 156 tokens from memory; a new engine process
  restored 156 from disk and recorded two disk hits.
- Archived screenshots and a combined DB/argv/health proof artifact. Gemma
  cache-default parity is live-verified; campaign/release lock unchanged.

## 2026-07-15 - Cross-model direct-answer rerun

- Rebuilt Electron rows 1394/1397/1400/1403 covered Nemotron, MiniMax-M2.7,
  and DSV4 cold/warm after the Bonsai direct-answer repair.
- Every row made one exact `file_info` call, one result, one final, and no
  warning. DSV4 was byte-exact twice and warm-reused 619 composite-cache
  tokens; its older general constrained-string failures remain separately
  open.

## 2026-07-15 - Step JANGTQ_K attention recovery

- Current installed `jang_tools` P18 Step patch omitted post-reshape q/k norms
  and the head-wise attention gate. Added a Step-only version guard that keeps
  semantically complete P18 implementations and otherwise restores native
  `Step3p5Attention.__call__` after hydration.
- 129 focused Step/JANG loader/runtime tests passed.
- Electron Logs visibly showed the native-attention restoration. Row 1406
  returned exact `4`; row 1418 executed one real
  `file_info({"path":"panel/package.json"})`, returned exact
  `STEP-TQ-TOOL4-DONE`, and reused 192 `paged+mixed_swa` tokens.
- Tool probes 1409/1412 were sent with `has_tools:false`; the UI tool toggle
  and working directory were corrected before the valid row.
- Step VL/media and restart-L2 remain open; release lock unchanged.

## 2026-07-15 - MiniMax-M3 REAP32 host-reboot gate

- Sessions UI loaded REAP32 as the only engine. Health reported 105.4 GiB
  active against a 107.52 GiB Metal ceiling.
- Two independent first Electron tool requests left blank assistant rows and
  coincided with full host reboots; the second occurred before UI Stop could
  complete.
- Generic occupancy guard allowed the 98.0% baseline, while output projection
  clamped to 2,304 tokens but did not cover fixed prefill workspace.
- Added a narrow 3 GiB MiniMax-M3 prefill-headroom reject and prohibited
  baseline forgiveness when the baseline itself is over threshold. 15 focused
  tests pass.
- No third live load was attempted. Status remains `FAIL-LIVE / PARTIAL-FIX`;
  release lock unchanged.

## 2026-07-15 - Mistral MXFP4 + Bonsai repeat-reasoning probe

- Mistral detector/source tests: outer `mistral3` plus inner `ministral3` is
  text-only, force-text-only, paged-default; Mistral 4 remains VLM. Panel
  registry/settings tests 358/358 and typecheck passed.
- Live Electron reset/save DB tuple: paged/block/legacy/multimodal = 1/1/0/0.
  Actual PID argv and health matched. Warm and restart rows restored 1,240
  tokens (`paged`, then `paged+disk`); strict marker content failed as
  `I understand.`. Broad 33 tools failed; reduced one-tool route passed.
- Bonsai current rows 1443/1446 each executed one `file_info` and finished with
  exact `B1-UI-TOOL10-DONE` and one reasoning card. Warm row restored 160/161
  `paged+ssm` but took 2,338 tokens/64.3s, so functional finalization passes
  while reasoning latency variance remains open.
- Evidence copied into
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.
  Release lock remains active.

## 2026-07-16 - OpenPangu strict parser and reasoning/finalization reopen

- Captured live openPangu failure row 1467: gibberish visible output promoted
  into a wrong `search_files` tool call with malformed args and an execution
  error, despite the user asking for `file_info`.
- Added `OpenPanguToolParser.STRICT_NATIVE_TOOL_FORMAT` and server handling so
  strict native parsers do not fall back to generic repair when they find no
  valid native call. Added a regression using the live malformed debris shape.
- Focused verification passed: `tests/test_openpangu_v2.py` plus
  `tests/test_openpangu_tool_parser.py` = 38/38, and the targeted
  `test_engine_audit.py` strict parser regression passed in the earlier
  27-test selected run.
- Restarted openPangu from Electron. PID 15972 argv and health proved the UI
  path used the openPangu parser and disabled generic cache reuse. Live retry
  row 1470 produced no tool call/tool result, so the wrong generic execution
  was not reproduced, but it still stalled in "Generating tool call..." and was
  interrupted. Mark openPangu tools as FAIL-LIVE, not fixed.
- Added the user's `B1-UI-TOOL3` Bonsai 1-bit repeated-reasoning/no-final
  report back to the active issue list. It is not normal behavior; previous
  `B1-UI-TOOL10` pass rows do not close the newly reported current symptom.
- Reproduced the `B1-UI-TOOL3` shape on current HEAD in a fresh Bonsai 1-bit
  Electron chat with built-in Search tools enabled and working directory set.
  Row 1473 functionally passed (one `file_info`, exact final), but took 3,617
  generated tokens and 103.2s with only 28 visible reasoning chars. Source/log
  trace shows the first stream ended with `content: 0`, `reasoning: 28`,
  `tool calls: 1`, `buffered: true`, so the active bug is hidden pre-tool
  Qwen tool-call buffering/perf rather than the post-tool final row on this
  exact run.

## 2026-07-16 - openPangu JANG_3M typed cache and architecture recovery

- Root-caused the previously corrupted openPangu output below the parser: all
  138 checkpoint causal-convolution keys were missing their nested `.conv`
  module path, and the DSA indexer's checkpoint-normalized key path requires
  RMSNorm. Added strict weight-landing diagnostics; the real bundle now reports
  2826/2826 parameter leaves, 46 layers, and 138 convolutions.
- Implemented exact `OpenPanguV2LayerCache` round-trip and non-aliasing clone,
  immutable N-1 prompt snapshots, memory-aware prefix reuse, and full-precision
  prompt disk L2. Generic paged/block formats and reverse trimming remain
  fail-closed.
- Enforced the family policy in CLI and UI: TurboQuant KV and generic q4/q8 are
  off. Live argv contained no cache quant flag; logs showed
  `VMLX_DISABLE_TQ_KV=1`; health exposed both quantization surfaces disabled.
- A >2K live Electron prompt activated the real DSA sparse indexer in every
  configured DSA layer and the forward diagnostic reported all 46 layers,
  DSA=16, SWA=30, mHC=4, sinks=128, MLA rank=512, window=512.
- Cold, warm-memory, and repeated process-restart disk rows all made one exact
  `file_info` call and exact final marker. Final row 1527 restored 2,075/2,076
  tokens from disk; post-run stats counted 4,178 disk-restored tokens.
- Fixed the strict parser's false warning after a valid auto-mode tool/final
  turn and made the settings UI say `TURBOQUANT OFF` instead of ambiguous
  generic `AUTO` for openPangu.
- Scoped gate passes. MTP execution, 512K context, long soak, full protocol
  matrix, other model-family issues, and release/notarization remain open.
## 2026-07-16 - Bonsai Qwen exact-once bounded after live TQ A/B

- Reproduced 2,422/4,335/6,316-token hidden tool-generation runs through the
  active Electron Responses route. The 6,316-token row exposed duplicate
  speculative tool states and no final; raw output had 46 tool markers.
- Switched the real KV setting to None, restarted, confirmed explicit CLI flag
  and TQ-disabled health, then reproduced 4,335 tokens. First valid call was at
  character 3,092, so TQ was falsified as root cause.
- Implemented request-scoped Qwen complete-call stop for an explicit single
  named tool plus “exactly once”; schema-invalid calls do not stop and ordinary
  Qwen multi-call turns do not opt in.
- Restored Auto/TQ and ran six fresh Electron rows: 115-244 tokens, 4.2-7.0s,
  one `file_info`, one result, exact final on every row. Two TQ-off proof rows
  also passed. Release remains locked for unrelated open gates.

## 2026-07-16 - Bonsai ternary native TQ paged/L2 repair

- Traced native TQ loss/corruption across partial disk payloads, missing block
  tags, unseeded decode, scheduler demotion, MLLM truncation/extraction, and a
  disk-numpy mirror overwrite. Added native record validation/rewrap and
  suppressed generic q4/q8 when native TQ owns storage.
- Set uncalibrated Qwen-hybrid Auto to correctness-first TQ8 storage on actual
  attention slots with no live mid-decode transition; explicit bundle configs
  remain authoritative and UI None is a hard bypass.
- Passed 38 native-TQ, 151 selected audit/TQ, and 244 scheduler/cache tests.
- In Electron, Bonsai ternary None restarted with the explicit disable flag
  and two multi-turn tool rows passed. Auto wrote seven native TQ blocks; a
  process restart decoded seven native records and fresh exact-one-tool rows
  passed again.
- Kept the boundary honest: persistent SSM restore is still quarantined, so no
  restart cache-speed claim; reasoning segmentation and an old mixed-history
  repeated-tool row remain partial. Next row is the separate Bonsai 1-bit
  Auto/None/native-TQ/restart matrix.

## 2026-07-16 - Bonsai 1-bit exact-once continuation repair

- Loaded the distinct 1-bit bundle and confirmed manifest/UI identity at
  `JANG_AFFINE_1BIT (1.1128b)`.
- Under UI None, first turn passed but the same-chat phrase `after the real tool
  result` missed the panel's literal detector. Row 1620 executed five tools,
  reopened reasoning six times, and was interrupted at 3,352 tokens/92.3s.
- Widened only the bounded `after ... tool result` clause while keeping
  `exactly once` and `reply exactly` mandatory. Rebuilt Electron main.
- Current None rows 1623/1626 and Auto rows 1632/1635 now each make one real
  tool call/result and exact final with one reasoning segment.
- Auto wrote native TQ8 attention-KV records and a process restart decoded
  three of three, later eight total hits. SSM restart restore stayed
  quarantined and was not credited as reused cache.
- Corrected health naming so storage-only `compress()` calls appear as
  storage/codec telemetry, never live-encode telemetry while live encoding is
  disabled. Live row 1641 confirmed it.
- Sessions UI showed one ACTIVE 1-bit model and ternary INACTIVE; process list
  showed one model server. Tests: 18 focused, 239 panel slice, typecheck, 114
  engine audit/TQ. Release remains locked.

## 2026-07-16 - Gateway LAN address and port parity

- Live Electron A/B moved the dev gateway from localhost:8080 to
  localhost:18080, then wildcard:18080, then back to localhost:8080. SQLite,
  the Electron listener, dashboard URLs, and health matched each transition.
- Pre-fix LAN UI advertised link-local 169.254.62.28. Main source had selected
  the first external IPv4. Added a pure selector that rejects APIPA and ranks
  RFC1918 over CGNAT/public; 65 selected gateway tests and typecheck pass.
- Rebuilt main process advertised 192.168.1.110:18080; curl to that exact URL
  returned healthy with port 18080 and single-model mode true.
- Restored user-facing state to LAN off, port 8080, single-model on. Installed
  app's independent wildcard 8080 listener remains an explicit open caveat.

## 2026-07-16 - TQ toggle persisted-cache leak and current Electron A/B

- Reproduced an explicit UI None session reading three pre-existing TQ-native
  block records despite live TQ objects being disabled.
- Added CLI-environment/constructor policy to both prompt and paged block disk
  stores, with read/write rejection, incompatible-record eviction, allocator
  `has_block()` compatibility inspection, and truthful stats.
- Unit regressions cover an Auto-written TQ record reopened by an Off manager,
  both explicit constructors, and `VMLX_DISABLE_TQ_KV=1` derivation.
- Live current-source Off: exact Electron output, standard block L2 hit, 64
  tokens saved, zero TQ-native hits/writes. Live restored Auto: exact output,
  attention-only TQ8, native SSM companion state, 16 disk hits / 14 native-TQ
  hits / one native-TQ write, 1,024 tokens saved.
- Full affected suite: 630 passed. Current state is restored to Auto. No
  release action; remaining matrix and Bonsai reasoning variability are open.

## 2026-07-16 - Exact-once pre-tool prose isolation and gateway live proof

- Replayed Bonsai Responses as two real harness turns. The function-output
  continuation was exact and short; the first pass sometimes spent hundreds
  of tokens in reasoning before one valid call.
- Captured the protocol defect: genuine reasoning-summary deltas stopped after
  an early `</think>`, later meta-reasoning was buffered with the tool call,
  and `response.output_text.done` exposed it as assistant prose.
- Added an exact-once-only buffer from the start of the Responses stream,
  preserved reasoning-summary deltas during buffering, and blanked valid-call
  preamble text. Added regressions for post-call repetition and premature
  content-rail meta-reasoning.
- Restarted the real 1-bit server from Electron. Gateway first pass emitted
  zero text deltas, one call, and empty pre-tool done text; the result pass
  streamed `GW-B1-FIX1-DONE`. Electron row 1848 returned
  `B1-FIX-UI1-DONE`, one call/result, 98 tokens.
- Re-proved 1-bit and ternary same-chat/restart turns, UI Auto reasoning and
  cache defaults, single-model unload, and LAN/localhost gateway rebind.
- `tests/test_server.py`: 102 passed, 3 deselected. Release remains locked.

## 2026-07-16 - DSV4 reasoning-control boundary and restart proof

- Captured pre-fix row 1863 with canonical DSML exposed in
  `reasoning_content` while the structured `file_info` call still executed.
- Added tool-safe reasoning-prefix tracking to Responses streaming and final
  reasoning output; raw reasoning remains available to the final DSML parser.
- Added a regression requiring genuine reasoning deltas plus exactly one
  structured call and no DSML in any visible/completed reasoning surface.
- Live current-source fresh and same-chat rows 1869/1872 returned exact finals,
  one call/result each, and no DSML leak.
- Visible Stop/Load changed the model process and row 1875 restored 606 DSV4
  paged tokens from persisted blocks, then returned one exact call/result.
- UI cache-disable restart removed paged/L2/pool paths; explicit re-enable
  restored them. Final health used `DSV4BatchGenerator` and reported three
  disk hits.
- Updated stale openPangu-shared UI assertions and the `__new__`-only DSV4
  generator fixture. Test totals: 103 + 131 + 114 passed.
- DSV4 remains partial beyond this scoped repair because row 1866 malformed,
  row 1875 took 119.2 s / 1,454 tokens, and the tested volume contained only
  the CRACK bundle rather than DSV4 JANGTQ.

## 2026-07-16 - Full-KV Qwen native-TQ truncation, Q3 red row, TQ8 recovery

- Root-caused native write loss to `Scheduler._truncate_cache_to_prompt_length`
  replacing `TurboQuantKVCache` with plain `KVCache` at the prompt boundary.
- Preserved cache type, bits, seed, sink/transition policy, and the complete
  decoded state. Added detailed TQ state to prompt/block persistent namespaces
  in both schedulers.
- Live Q3 native storage was compact, but cache reuse was not correct: one
  fresh tool call stalled for 113.4 s and an 896-token hit produced corrupted
  tool-like output. Both red rows remain in the ledger.
- Changed only uncalibrated Qwen Auto to storage-only TQ8 on real attention KV.
  No change is applied to cumulative-only Qwen or bundle-calibrated policy.
- Visible Electron None -> Auto restarts proved CLI, health, disk dtype, tool
  execution, and cache namespace parity. Final Auto row 1929 restored 896
  tokens from disk and executed one `file_info` with exact completion.
- Test totals: 41 + 581 + 212 passing selections; compile and diff check pass.
  Ruff has a legacy 496-finding baseline and was not counted green.
- Release remains locked; continue with HY3 depth-1 MTP acceptance/speed, then
  the outstanding MiniMax/Bonsai/Pangu/Laguna/Mistral and settings/API rows.

## 2026-07-16 - HY3 D1 live acceptance telemetry and controlled A/B

- Published locked process-local oMLX native-MTP snapshots through scheduler
  health and counted per-depth acceptance plus seed/verify/MTP forwards.
- Electron Off and deterministic D1 settings were visibly exercised through
  restart; health reported disabled versus active D1 respectively.
- Identical greedy three-run arms were exact. Off wall seconds:
  23.496679/17.986732/21.234247. D1 wall seconds:
  22.627244/16.006324/16.081931. Median improvement: 24.264%.
- D1 aggregate: 180 accepted / 414 drafted (43.478%). Electron Perf rendered
  live counters; row 1944 executed one `file_info` and exact final text.
- Tests: 177 focused MTP and 54 selected engine-audit tests passed; compile
  and diff checks passed. Evidence saved under
  `docs/internal/release-gates/20260716_hy3_mtp_d1_acceptance/`.
- Release remains locked. Min-P zero persistence, cache-label parity and TQ log
  rate limiting remain open alongside the wider model/parser/settings gates.

## 2026-07-16 - openPangu b5 current-HEAD Electron recheck

- Loaded the real JANG_3M session through Electron single-model mode, then
  visibly Stop/Started it without clearing L2.
- Rows 1947/1950/1953 each made one real tool call and exact final text; memory
  restored 144 tokens and process-restart prompt L2 restored 295 tokens.
- Current health/argv/settings/logs show native full-precision typed composite
  caching only: no generic paged/block cache and no TurboQuant/q4/q8.
- Runtime log proved every decoder layer and advertised topology were used:
  DSA 16, SWA 30, mHC 4, 128 sinks, MLA rank 512, 138 causal convs.
- 75 focused openPangu and two exact-once server tests passed. Evidence updated
  in `docs/internal/release-gates/20260716_openpangu_typed_cache_electron/`.
- Release remains locked; MiniMax-M3 OCR/tools/media is next.

## 2026-07-16 - MiniMax-M3 exact OCR/video/tool and scheduler repair

- Reproduced two strict OCR-format misses without the old zero-tool dead end,
  then proved fresh-chat exact `BANANA8426` with tools enabled.
- Proved one real same-chat tool/final, two-frame video recognition, exact
  no-reattach follow-up, and process-restart typed cache reuse.
- Fixed post-init Scheduler reads of `_uses_openpangu_cache` to default false;
  four previously red M3 cache tests then passed.
- Current-source PID 2921 row 1977 executed one tool, exact final text, and
  restored 449 `paged+disk` native MSA tokens with no dequantization.
- Passed 34 + 3 + 4 + 75 + 581 + 8 selected tests, compile, and diff check.
  Evidence saved under
  `docs/internal/release-gates/20260716_mm3_exact_media_current/`.
- Release remains locked; Laguna/Mistral/settings/protocol rows remain open.

## 2026-07-16 - Laguna TQ dtype loss and warm decode repair

- Proved TurboQuant decode promoted the real model KV path to float32 both in
  live JANG compression and after vMLX block/prompt L2 restore.
- Added source-owned dtype preservation in JANG plus dtype metadata,
  validation, restoration, nested CacheList coverage, and a TQ-only cache
  namespace bump in vMLX.
- Exact Electron output measured 25.0 tok/s cold, 21.2 tok/s warm paged, and
  24.6 tok/s after Save & Restart from native TQ disk blocks.
- Restart Logs show 70 rewrapped layers at 14,049,280 bytes rather than the
  former float32-sized footprint. Tests pass 24 + 115 + 46.
- Final current-source PID 10474 repeated the exact disk row at 25.1 tok/s and
  the same 14,049,280-byte 70-layer rewrap footprint.
- Default Laguna reasoning remains red: a repetitive 726-token UI run required
  manual interruption. Release remains locked; continue with its parser/JIT
  matrix before Mistral and global settings/protocol rows.

## 2026-07-16 - Laguna uncalibrated TQ3 reasoning-loop A/B and TQ8 repair

- Reproduced cold-coherent -> warm-incoherent behavior in Electron at a 3,545
  token `paged+tq-native` boundary; preserved the interrupted row and screenshot.
- UI None proved ordinary paged reuse coherent on two same-chat turns while
  leaving prefix/paged/disk enabled.
- Changed uncalibrated Auto storage to TQ8 for real attention KV and added a
  codec-config fingerprint plus namespace bump so prior TQ3 blocks miss.
- Corrected Auto passed cold and two in-memory warm Electron rows. A real
  Stop/Start restored 3,550 TQ8 tokens from L2; the model chose `ask_user`, then
  completed exactly after visible Skip, so strict no-unsolicited-tool remains
  partial.
- 112 focused tests pass. Measured TQ8 reconstruction is 3.59-4.79 s, which
  remains a performance gate. Release remains locked.
## 2026-07-16 - Bonsai SSM L2 exact restart and disk-boundary repair

- Current-source cache-on repeats: rows 2037/2040/2043/2046 each persisted one
  tool call/result and exact final; health recorded RAM, disk, and native-TQ
  hits. The earlier loop remains preserved as an outlier.
- Removed automatic Qwen SSM L2 suppression and restarted through the visible
  Electron Server settings.
- Deterministic restart rows 2052 and 2064 restored 160/168 tokens as
  `paged+ssm+disk`, each with exact output and ~0.10s cache reconstruction.
- Added a fresh-process direct L2 probe for the scheduler-selected SSM boundary
  and a regression with an empty L1 length index.
- Longer continuation safely fell back when its 64-token KV boundary lacked a
  companion, then wrote a complete 64-token repair sidecar. Broad partial
  prefix acceleration stays PARTIAL.
- Tests: 581 engine-audit and 166 hybrid-focused passed. Release not started.
- Ternary default rows 2073/2076 and restart row 2079 were exact; restart
  restored 153 `paged+ssm+disk` tokens in ~0.101s with three native-TQ hits and
  two SSM L2 hits.
- Ternary None used CLI `--kv-cache-quantization none`, retained native
  prefix/paged/block L2, and recorded zero TQ hits/writes; Auto was restored.

## 2026-07-16 - Expanded mandatory runtime/cache/stream proof matrix

- Recorded Qwen 3.6 35B/27B, HY3 MTP, MiniMax M2.7, ZAYA/CCA, Nemotron
  hybrid, DSV4, M3, and openPangu as named current-source Electron gates.
- Added minimum two-turn and real tool-result continuation requirements,
  reasoning/content/tool-argument streaming assembly checks, MTP depth and
  acceptance telemetry, RAM/L2 hit accounting, and bounded-capacity eviction
  followed by correct reload or safe full-prefill.
- Pinned the architecture rule: generic TQ only encodes compatible attention
  KV; hybrid non-KV state stays native; DSV4/CCA/M3/openPangu stay on their
  typed codecs. No load-only or one-turn row is enough for release.
- Pinned the no-fake-fix rule in the release matrix. All new rows remain OPEN
  until source trace and live artifacts exist.

## 2026-07-16 - Paged L2 promotion accounting repair and 24h reconciliation

- Fixed phantom resident-byte accounting after disk-backed block
  reconstruction and cleared payload-scoped `keep_resident` on block reuse.
- Updated the stale disk-stats contract for the intentional
  `tq_native_enabled` field discovered by the widened cache suite.
- Passed 595/595 engine-audit/byte-budget tests and 177/177
  paged/disk/TQ/hybrid cache tests.
- Pushed `7bb34fa0d` to `origin/codex/live-electron-gates-20260715`.
- Electron restart PID 58213 row 2160 restored 152/153
  `paged+ssm+disk` tokens with one real tool and exact final text; Cache
  Management shows zero L1 resident bytes and 152 indexed tokens.
- Reconciled the current matrix to 45 commits ahead of `origin/main`, corrected
  Qwen 35B to its real 10-attention/30-companion graph, retained forced
  eviction/duplicate companion payload/selective-label as open, and added a
  current-source Gemma 4 rotating-SWA TQ/paged/L2/eviction row.

## 2026-07-16 - Qwen 35B external companion and forced-eviction closeout

- Source audit proved terminal generic block L2 duplicated 30 SSM/GDN states
  that the typed companion path required and overwrote anyway.
- First source implementation passed 783 tests, but live v7 file inspection
  found the NumPy writer still emitted `cumulative=30`; added direct coverage
  and repaired that owning branch instead of accepting the health counters.
- Pushed `df945f065`, `133d8c8e9`, and `7cb89185c`.
- Corrected v8 files contain 10 native-TQ attention entries plus 30 skips and
  no cumulative payloads. Python coverage is now 784/784; panel settings are
  278/278 plus typecheck.
- Live Electron rows 2169/2172/2175 close cold/RAM/restart tiers. UI-applied
  four-block rows 2178/2181 forced nine evictions and reloaded from L2 exactly;
  a missing SSM companion at a 192-token KV boundary safely full-prefilled.
- UI restored 1,000 blocks and PID 61919 row 2184 repeated exact disk restore.
- Cache tier is closed; strict long-format reliability remains partial.
- 2026-07-16 13:30 PT: fixed generic fetched-paged-hit ref ownership
  (`af7815f1a`), passed 90 focused tests, and live-proved MiniMax M2.7 Auto,
  explicit None, two-turn tools, L1/L2, forced eviction/reuse, and restored
  1,000-block settings. Long non-tool/direct stream remains open.
- 2026-07-16 13:40 PT: closed M2.7 long-stream row with Electron row 2217 and
  a completed 1,024-token Responses SSE proof; retained the correctly reported
  512-token incomplete control. M2.7 is PASS-LIVE current source.
- 2026-07-16 16:12 PT: closed the scoped DSV4 native-composite L2-to-L1
  retention and UI memory-budget parity row. Source now keeps restored
  `deepseek_v4` native block payloads resident but evictable after successful
  L2 reconstruction, and MLXStudio DSV4 settings/preview/session launch all
  emit Cache Memory %. Focused tests passed: 76/76 DSV4/paged byte-budget,
  280/280 settings, and panel typecheck. Live Electron PID 95494 launched from
  project `.venv` with `--cache-memory-percent 0.15`; health showed 15% L1.
  Row 2343 restored from DSV4 block L2 and exact-finaled; row 2346 reused 611
  resident `paged+dsv4` tokens with scheduler/block disk hits unchanged
  `2 -> 2`. Evidence under
  `docs/internal/release-gates/20260716_release_closeout/dsv4-current-head/`.
  DSV4 remains release-PARTIAL for forced eviction, long constrained output,
  reasoning/content stream soak, quiet speed, and exact JANGTQ artifact row.

- 2026-07-16: searched the external model drive for another ZAYA artifact.
  Only the user-excluded AppleScript specialist exists, so the generic CCA gate
  is recorded as blocked by missing artifact rather than mis-testing its
  intentionally restricted tool surface.
- 2026-07-16: closed Nemotron-H selective-cache/settings/tools/API rows through
  Electron. Exact cold/same-chat/restart tool rows covered resident TQ, typed
  SSM companion state, and disk restore. A UI-applied four-block pool forced
  nine evictions; explicit None proved raw L2 with zero TQ; Auto/1,000 was
  restored. Direct Responses completed with both reasoning and content deltas.
  The coherent long UI row repeated native reasoning before closing, so that
  reliability boundary remains PARTIAL and release remains locked.

- 2026-07-16 16:45 PT: pushed `4e13b19a7` to honor explicit Tool Parser None
  through final parsing and Chat/Responses streaming. Focused server coverage
  passed 106/106 plus 52 passed / 1 skipped affected selection. Electron PID
  99835 persisted raw unparsed output with no tool result under None. After
  restoring Qwen on PID 864, rows 2364/2367/2370/2373 completed four same-chat
  exact one-tool turns; the fourth reused 258 `paged+ssm` tokens. Retained row
  2352 as a real stale-argument failure and retained the reasoning variability,
  so Bonsai remains PARTIAL and release remains blocked.

- 2026-07-16 17:00 PT: pushed `a36a5ea66` for truthful Responses terminal
  events and matching Electron handling. A length-capped stream now emits
  `response.incomplete`; 135 Python tests, 50 panel tests, and typecheck pass.
  Electron row 2388 visibly stopped at UI-applied 32 tokens with separated
  reasoning/partial content, then Max Tokens was restored. Direct Bonsai
  Responses produced one exact result continuation and one 1,024-token repeated
  tool-markup incomplete continuation. Retained that red variability; eight
  consecutive same-chat UI tool turns after row 2352 were otherwise correct.
- 2026-07-16 17:32 PT: pushed `7d664e071` for typed DSV4/ZAYA disk-tier
  telemetry and panel cache-tier union. A UI-applied four-block DSV4 pool
  forced L1 evictions plus L2 writes/hits; after current-source Electron
  restart, PID 5485 row 2409 restored 598 tokens as `paged+dsv4+disk`, made
  one real tool call, and exact-finaled. The UI restored 1,000 blocks on PID
  5953. Raw Responses D4-RAW1 completed with 187 reasoning and 32 content
  deltas whose assembled text matched the done event. Electron row 2412 is
  retained red for hallucinated acronym meanings, repetition, and wrong exact
  closer. DSV4 cache/stream tiers pass; long quality/perf and release remain
  PARTIAL.
- 2026-07-16 17:27 PT: pushed `d49f500a3` to preserve explicit per-chat
  Min-P zero. The old UI converted zero to inherit and both wire builders
  omitted it, so a bundle-owned non-zero default could not be disabled.
  Current source sends zero on Responses and Chat Completions; 213 affected
  panel tests and typecheck pass. A clean Electron restart with the intended
  user-data profile and project `.venv` showed `0.00`, persisted SQLite
  `min_p=0.0`, and emitted live `[CHAT_DIAG]` request body `"min_p":0` on
  DSV4 PID 8935. Broader settings/gateway and release gates remain open.
- 2026-07-16 18:15 PT: pushed `e76cc5451` for transactional gateway restart.
  A real-listener test plus 74 related gateway tests and typecheck pass. In the
  clean current-source Electron app, the occupied-port edit preserved the
  running 8081 listener, LAN enabled a working `192.168.1.110:8081` health
  route and restored localhost, and the visible single-model Bonsai Start
  action stopped DSV4 PID 10013 before starting Bonsai PID 10495. UI, DB,
  process list, gateway health, and lifecycle log all showed one engine.
  Protocol streaming and global release gates remain PARTIAL.
- 2026-07-16 18:41 PT: pushed `a0aa81a94` for Ollama stream finalization. The
  translator no longer repeats accumulated thinking and now waits through the
  usage-only event to `[DONE]` before emitting the empty terminal object.
  Tests pass 76/76 plus typecheck. Fresh Electron PID 12046 / Bonsai PID 12114
  streamed 193 reasoning deltas once, exact `OLL-GW4-DONE`, and terminal usage
  202/18. Basic Chat/Responses/Anthropic/Ollama gateway streams were exercised;
  agentic continuation and one retained Responses whitespace miss remain open.
- 2026-07-16 18:35 PT: pushed `ab5d01e04` and `5e6a1f8a1` for HY3 full-KV
  TQ4 stored-prefix policy, batch-safe cache deepcopy, and truthful Electron
  settings labeling. Focused tests passed 19 Python HY3/TQ, 178 native-MTP,
  282 panel settings, plus typecheck. Electron PID 22265 restored 3,272
  `paged+disk+tq-native` tokens after process replacement; four-block PID
  23635 forced 11 L1 evictions, and row 2492 restored an older bounded prefix
  from TQ-native L2 with an exact one-tool final. Normal 1,000 blocks were
  restored on PID 25084. Retained row 2477's strict-marker miss and left the
  global release blocked.
- 2026-07-16 18:48 PT: closed HY3 explicit-None cache parity through the live
  Electron settings drawer. PIDs 26444/27461 launched with literal
  `--kv-cache-quantization none`; row 2495 wrote raw L2 with no TQ activity,
  and post-restart row 2498 restored 3,258 `paged+disk` tokens with one exact
  tool/final while TQ writes and hits remained zero. Auto/TQ4 and 1,000 blocks
  were restored on PID 28473. HY3 remains PARTIAL only for retained strict
  format and long/streaming reliability rows.
- 2026-07-16 19:00 PT: pushed `2a15e51d3` to stop replaying Qwen 3.6's
  redundant `call:default_api:<tool>{...}` preview as assistant `output_text`
  beside an already parsed structured tool call. Focused panel tests passed
  14/14 plus typecheck. Current Electron rows 2510/2513 executed distinct
  `file_info` arguments once each and exact-finaled without persisted raw
  preview. This closes the two-turn parser/continuation row only; configured
  D3 is capped to D1 for tool requests, and non-tool D3 acceptance,
  restart/L2, forced eviction, and MXFP4 remain open.
- 2026-07-16 19:15 PT: pushed `b994582cc` for TQ4 Auto on compatible
  non-Bonsai attention KV while preserving Bonsai TQ8 and native hybrid
  companions. Scoped tests pass 35/35 Python and 283/283 panel plus typecheck.
  Electron rows 2528/2531 prove cold q4 write and fresh-process
  `paged+ssm+disk` restore with TQ-native plus SSM hits; four-block rows
  2534/2537 forced three evictions and restored the older prefix exactly.
  Cache UI showed q4 and Auto/1,000 blocks were restored. True non-tool D3,
  MXFP4, and long/cancel/media rows remain open.
- 2026-07-16 19:25 PT: retained tools-off rows 2540/2542 as D3 terminal
  failures. The 256 control stopped after partial content; the 512 control
  ended after 505 reasoning tokens with no visible final. Scheduler telemetry
  proves real D3 drafting/acceptance (`[20,12,12]` drafted, `[16,7,4]`
  accepted, D3 rate 0.3333). Restored tools on and Max Tokens to model default.
- 2026-07-16 21:15 PT: pushed `b33d80589` for Qwen post-tool progressive
  reasoning/content streaming. A one-character backtick no longer falsely
  enters native tool buffering, ambiguous short marker suffixes are released
  safely, and explicit/Auto reasoning partitioning now applies after the tool
  result. Current source passes 832 tests (three deselected). Direct Responses
  emitted 153 reasoning and 113 content deltas with one completed terminal.
  Electron row 2606 visibly grew the answer while Stop was active, executed one
  matching `file_info` call/result, restored 512 `paged+ssm+disk` tokens, and
  exact-finaled `Q27-ELECTRON-TOOLSTREAM-FIX7-DONE`. Other families remain
  regression-gated; no release claim was made.
- 2026-07-18: published the full v1.6.11 checkpoint. The package was built
  from `95b2caa956c592a9caa706f2a790dcd5664721b7`; final tag, `origin/main`,
  closeout branch, and evidence head resolve to
  `df244c4a858df3894fa3911b270d6d1b175966d6`. Both Sequoia and Tahoe DMGs
  passed signing, notarization, staple, Gatekeeper, and installed Electron
  smoke. Public GitHub source/app releases, PyPI, raw/site feeds, and Homebrew
  are live. Final current-head release-surface verification reports pass with
  no failed checks. Evidence is under
  `docs/internal/release-gates/20260718_v1_6_11_release/`. This closes the
  release checkpoint only; retained model/media/protocol/UI rows remain active
  post-release work.

## 2026-07-19 - openPangu 44k typed-snapshot admission

- Reproduced the 43,980-token Electron row before patch: exact answer only
  after 186.35s TTFT; 22,090.8 MB exact native snapshot was copied and then
  rejected; peak Metal 138,814.3 MB.
- Added a global typed-snapshot pre-copy size gate to the single-active
  generator and truthful scheduler/cache telemetry. No native state format or
  model output policy changed.
- 124 focused tests passed. Real Electron replay returned the same exact answer
  at 103.20s TTFT and 115,551.8 MB peak. Raw Responses replay showed progressive
  reasoning and content plus a completed terminal event.
- Retained `PARTIAL`: the 256-token API cap was spent entirely in reasoning, so
  the existing visible-answer recovery performed a second full prefill because
  the 20.9 GB native boundary could not fit the 10 GB backends.

## 2026-07-19 - MiniMax M2.7 paged q4 partial/refault and telemetry truth

- The live Electron cache drawer applied a fresh dedicated Block Disk L2
  directory, 64-token blocks, and a four-block ceiling. The first prompt wrote
  a 178-token q4 native-TQ chain with exact 64+64+50-token records. Same-chat
  pressure evicted L1; a fresh chat refaulted the older partial boundary and
  exact-finaled. A visible process replacement then restored the same boundary
  from four persisted blocks with zero pre-request L1 tokens.
- The run exposed an instrumentation defect: later frugal indexed requests
  performed worker-side L2 payload reads while usage said only
  `paged+tq-native`. Commit `97a84fed5` records successful worker-reconstructed
  disk blocks and promotes the actual source to request/cache-execution detail.
  It does not infer a disk hit merely from L2 configuration. Focused cache and
  scheduler validation passes 114 with two intentional deselections.
- Patched PID 65685 proved the regression through Electron, raw Responses,
  and raw Chat. Each restored 178 `paged+disk+tq-native` tokens. Responses
  emitted 316 reasoning and 10 content deltas before completion; Chat emitted
  508 reasoning and 10 content deltas then finish, one terminal usage chunk,
  and `[DONE]`. The same Electron chat restored 192 disk tokens, executed one
  real `file_info(panel/package.json)`, and exact-finaled with its 5.2 KB
  result and no warning.
- Commit `135a2ef6b` preserves the current health, raw SSE/traces, settings,
  screenshots, block index, persisted UI rows, issue ledger, and matrix update
  under `docs/internal/release-gates/20260719_m27_paged_l2_partial_refault/`.

## 2026-07-19 - full-suite recovery and bundle evidence preservation

- The first panel full run found a stale MCP source-shape contract. The current
  contract now asserts both effective Responses/Chat tool-schema guards;
  focused MCP tests pass 9/9 and full panel passes 2,312 with three skips.
- The first Python run was invalid because noninteractive SSH omitted Node from
  PATH. The corrected run found one stale cache fake and one real bundled
  source drift. The fake now matches the production `get_stats()` interface;
  bundled Python was rebuilt from current vMLX plus clean JANG `9081c924`.
- While rebuilding, the bundle script was found deleting the entire tracked
  `build/` proof tree. `92935ada5` scopes cleanup to setuptools scratch. A real
  rebuild retained a sentinel and bundled verification passed all critical
  source hashes/imports.
- After canonical proof-artifact regeneration, the final isolated Python suite
  passed 6,125 with 96 skips and 92 deselections. Typecheck and the clean-JANG
  production build also passed. The canonical regression runner itself stays
  `open` for the named retained live/release rows.
- Evidence:
  `docs/internal/release-gates/20260719_full_suite_checkpoint/`.

## 2026-07-19 - M2.7 Chat/Responses effective no-tool parity

- Live red: MiniMax M2.7 Chat post-tool continuation with retained schemas,
  `tool_choice=none`, and thinking off emitted no content, then stop/warning/
  length. Responses with the same real 5.2 KB result completed.
- Root cause: parser seed and answer policy used public `request.tools` after
  the renderer had removed tools. Commit `ffb9ed7db` centralizes effective
  generation-tool availability across Chat/Responses stream/non-stream.
- Regression: 244 passed, three deselected. Live after a real Electron engine
  restart: Chat emitted 18 progressive content deltas, exact final, one stop,
  one usage, and `[DONE]`; retained-schema Responses emitted 19 content deltas
  and completed once.
- Electron Auto/tool/recall rows passed with separate reasoning/content, one
  real `file_info(panel/package.json)` result, no second tool, and no warning.
- Evidence: `docs/internal/release-gates/20260719_m27_protocol_parity/`.
  Protocol matrix remains `PARTIAL` pending Anthropic/Ollama and failure-
  recovery rows.

## 2026-07-19 - M2.7 Anthropic protocol repair

- Reproduced three distinct integration failures on the Electron-started
  MiniMax-M2.7 JANGTQ/MXTQ process: an empty Anthropic tool name from split
  Chat deltas, an orphaned MiniMax outer opener leaking native invoke XML, and
  retained schemas rendered despite Anthropic `tool_choice=none`.
- Fixed and pushed `c707bb61a`, `d7f74b982`, and `4a53f16e1`. No argument,
  tool name, answer, sampler, or token budget is synthesized.
- Current live Anthropic required-tool output contains one named `file_info`
  with exact path JSON. Its result continuation emits 17 progressive content
  deltas and exact final with one stop. Real Electron independently completes
  one built-in tool loop with exact visible final, separate reasoning, no
  warning, and `paged+disk+tq-native` cache detail.
- Focused regression is 119/119. Preserved raw before/after, controls, health,
  tests, SQLite, UI text, and screenshot under
  `docs/internal/release-gates/20260719_anthropic_tool_parity/`.
- Protocol parity remains `PARTIAL`; proceed to Ollama and failure-recovery
  rows rather than claiming campaign completion.

## 2026-07-19 - Ollama stream/tool and shared think-boundary repair

- Raw Ollama Chat reproduced `\n\n` before visible content. Matched direct
  Chat reproduced the same bytes, locating the defect in shared streaming
  reasoning extraction rather than the Ollama adapter or M2.7 artifact.
- `c1db6b745` strips only the structural whitespace-only boundary, preserves
  all later deltas, and passes 300 focused tests across Qwen3, DeepSeek-R1,
  MiniMax-M2, streaming, API, and audit coverage.
- Templated Generate then exposed an independent terminal bug: finish was
  emitted before usage and the later count row was discarded. `01d95b448`
  defers/merges terminal rows. Focused Ollama regression passes 36/36.
- After real Electron Stop/Start, live Chat and Ollama content were exact;
  `/api/chat` tool/result completed with one object-argument call and no
  second tool; `/api/generate` completed with one usage-bearing terminal.
- CDP captured progressive Electron DOM growth before the final answer, and
  the persisted row kept reasoning/content separate with no warning.
- Preserved before/after raw streams, tests, health, DOM samples, SQLite,
  screenshot, and UI text at
  `docs/internal/release-gates/20260719_ollama_stream_tool_parity/`.
- Protocol remains `PARTIAL` pending cancellation/disconnect/failure recovery
  and retained cross-family rows.

## 2026-07-19 - Responses cancellation and disconnect recovery repair

- Live red on the Electron-started MiniMax-M2.7 JANGTQ/MXTQ engine: explicit
  Responses cancellation returned HTTP 200 but finalized three partial bytes as a
  completed output item and `response.completed`.
- `ae498c70b` fixes the shared endpoint state machine: aborted/disconnected output
  is incomplete, cancellation reason is explicit, no answer retry/history write
  occurs, and exceptions use `response.failed`.
- Focused tests pass 111/111 selected. After real Electron Stop/Start, live cancel
  produced only `response.incomplete`; client disconnect reached idle and the
  immediate recovery streamed 12 exact content deltas before one completed
  terminal. Both partial ids returned 404 from the Responses history endpoint.
- Preserved pre/post SSE, summaries, disconnect/recovery, source trace, tests,
  Electron logs, and screenshot under
  `docs/internal/release-gates/20260719_response_cancel_disconnect/`.
- Keep the parent protocol row `PARTIAL`: safe live mid-stream exception injection,
  Chat cancel/disconnect, signed-app repeat, raw Generate multi-tool, and other
  model/parser families are still open.

## 2026-07-19 - Ollama and Electron simultaneous multi-tool proof

- The first live Ollama iteration returned one `tool_calls` terminal containing
  exactly `file_info({path: panel/package.json})` and
  `run_command({command: pwd})`; both wire argument values were objects.
- The harness executed only those real operations and supplied separate named tool
  messages. The next iteration produced 43 thinking rows, 30 visible content rows,
  exact final, one stop terminal, and no second call.
- The real Electron built-in loop independently showed two reasoning rails, both
  tool status cards, exact final, and no warning. SQLite row 372 records both calls
  and results plus `paged+disk+tq-native` reuse.
- Added and pushed adapter regression coverage at `1b35d7a9b`; 31 selected tests
  passed. Evidence: `docs/internal/release-gates/20260719_ollama_multitool/`.
- Parent matrix remains `PARTIAL` for the explicitly retained cross-family,
  signed-app, media, cancellation, and long-soak rows.

## 2026-07-19 - Chat disconnect and real Electron user stop

- Raw Chat closed after five content deltas; scheduler returned idle and the next
  stream exact-finaled with one stop, usage, and `[DONE]`.
- Electron prefill stop left only the user row. Electron mid-content stop captured
  visible counting through 76, persisted the real partial plus
  `[Generation interrupted]`, and returned idle. The immediate same-chat follow-up
  stored exact content with no warning.
- Restored the temporary UI controls to Auto/tools On/blank Max Tokens. Existing
  focused suites passed 7 Python and 368 panel tests.
- Evidence:
  `docs/internal/release-gates/20260719_chat_disconnect_stop_recovery/`.
- Continue with safe live engine-failure injection and the retained signed-app,
  gateway, cross-family, and soak rows; do not promote global protocol parity yet.

## 2026-07-19 - Bonsai partial-prefix and shared Responses finalizer repair

- Re-read bundle config and recorded the quant boundary: Bonsai 27B 1-bit is
  `JANG_AFFINE_1BIT`, not JANGTQ/MXTQ. Its hybrid cache graph is 16 attention
  KV lanes plus 48 native companion lanes.
- Live 6,336-token sibling-prefix rows hit resident `paged+ssm` twice and
  process-restart `paged+ssm+disk`; health showed q8 native-TQ attention block
  hits plus an SSM disk restore.
- Raw Responses exposed a shared leak where a rejected incomplete tool suffix
  on the reasoning rail became visible output and blocked answer synthesis.
  `359ce6b2b` keeps it private and re-arms the direct answer pass; 147 selected
  neighboring tests pass.
- After Electron Stop/Start to PID 1054, raw Responses completed the real
  two-round tool loop with separate progressive rails. Electron row 385 made
  one real `file_info` call, exact-finaled with no warning, and CDP recorded
  progressive final paint.
- Preserved current-source evidence under
  `docs/internal/release-gates/20260719_bonsai_partial_prefix_responses/`.
- Do not promote the overall release matrix: cross-parser and retained
  long/stochastic/media/eviction/signed-app rows remain open.

## 2026-07-19 - M3 current-source parser/stream recheck

- Started M3 from its real Electron session card; PID 2277 became the only
  local engine. Re-read the bundle and confirmed affine JANG plus native
  MSA/index cache rather than JANGTQ/MXFP or generic TQ KV.
- Auto no-tool and same-chat tool rows both streamed distinct reasoning and
  progressively painted visible content. The tool ran exactly once and used
  its real result; no warnings or leaked parser markers.
- Raw Responses and Chat each completed a six-case no-tool/tool/follow matrix
  with separate reasoning, argument, and content deltas and clean terminals.
- The 806-item focused selection exposed stale bundled engine source. With
  only that packaging verifier deselected, 759 passed and 46 skipped.
- Preserved current proof in
  `docs/internal/release-gates/20260719_m3_current_postfinalizer/`.
- Keep release blocked until `bundle-python.sh` refreshes current source at the
  chosen cutoff and the complete verifier passes without deselection.
## 2026-07-19 - Gemma4 current-source text/parser repeat

- Re-read the 26B bundle: affine `JANG_4M`, 25 sliding + 5 full-attention layers,
  Gemma4 reasoning/tool parsers, vision-capable, no MTP.
- Real Electron Start loaded PID 4530. No-tool row 394 was coherent and progressive
  but consumed 3,322 output tokens; tool row 397 made one real file call, exact-reported
  5.2 KB, restored 7,168 `paged+mixed_swa+disk`, and had no warning.
- The 512-token raw controls reproduced truthful output-limit terminals. Increasing
  only the explicit request cap to 4,096 yielded completed Responses and Chat streams
  with hundreds of separate reasoning deltas followed by progressive content deltas.
- Health confirms native live rotating cache objects plus q4 storage-boundary encoding
  for both KV lanes with rotating metadata preserved. Focused tests pass 361/361.
- Preserved screenshots, DB rows, health, both raw traces, and test output under
  `docs/internal/release-gates/20260719_gemma4_current_parser_stream/`.

## 2026-07-19 - DSV4 current Auto/parser/restart/L2 proof

- Pushed code commit `4e723f311` for the DSV4 Auto reasoning UI state.
- Rejected one bad-PATH Electron relaunch as invalid evidence, then launched the
  current app with the project venv; main logged the exact engine path.
- Used the real Sessions Start button to load PID 8882 and visually confirmed Auto
  selected after a full main-process restart.
- Electron row 406: two separate reasoning rails, one exact real file-info call,
  visible `The file size is 5.2 KB.`, no warning.
- Health after the turn: two disk hits, 3,173 native DSV4 L2 block tokens, zero
  generic TQ. Controlled raw Responses/Chat streams and clean terminals pass.
- Retained marker mutation, path sensitivity, and weak-prompt hallucination as
  PARTIAL; matched same-artifact reference-runtime A/B is still required.
- Tests pass 329 Python + 100 panel + typecheck. Evidence preserved under
  `docs/internal/release-gates/20260719_dsv4_current_parser_auto_stream/`.

## 2026-07-19 - Laguna current stream, TQ A/B, and eviction checkpoint

- Loaded Laguna-M.1 through the visible Electron Sessions Start path.
- Preserved natural reasoning/content progress, exact one-tool row, DOM paint
  progression, and post-restart history recall.
- Exercised raw Responses and Chat stream/non-stream no-tool, tool, and follow-up.
- Used the real UI to switch Auto/1000 to None/max-four and back. Explicit None
  reached argv and health; max-four forced eviction and partial disk refault.
- Isolated greedy cold-vs-q4 divergence to q4 stored-prefix restore: q4 warm runs
  are stable, while raw None and bypass-cold match full-precision cold exactly.
- Updated the stale source-contract test only; commit `6f7b29bc3` is pushed.
- Focused validation: 411 Python pass/1 skip, 771 panel pass, typecheck pass.
- Evidence saved under
  `docs/internal/release-gates/20260719_laguna_current_stream_tq_determinism_eviction/`.
- Performance/long soak, q4 cold-byte equivalence policy, bundled-Python refresh,
  and the remaining campaign rows stay open.

## 2026-07-19 - Mistral JANGTQ failure preserved; worklist reconciled

- Real Electron testing of the 88-layer Mistral Medium 3.5 JANGTQ2 bundle found
  two distinct failures: legacy prefill stall and newline-only MPP NAX decode.
  A dtype-aware FP32 NAX rerun reproduced the newline-only failure at a real UI
  64-token cap; the generation was stopped visibly.
- Reverted the unsafe broad Auto exception, passed 12 focused policy tests,
  committed/pushed `fad7356d4`, and preserved screenshots, health, source trace,
  and explicit missing gates in `20260719_mistral35_jangtq_prefill/`.
- Separately committed/pushed Jang `000e41c` after 23 live Metal-kernel tests;
  recorded it as a kernel correction only.
- Reconciled the master matrix/ledger/status against July 19 evidence in
  `20260719_current_reconciliation/`. Overall status remains
  `PARTIAL_NO_1_6_12_RELEASE`.
## 2026-07-19 - Qwen3.6 35B JANGTQ current-source stream/cache checkpoint

- Proved current Electron one-tool/final behavior and current raw Responses/Chat
  required-tool continuations.
- Proved real Stop/Start exact L2 restore and changed-suffix partial block reuse:
  2,560 cached tokens, 40 native-TQ disk hits, one complete SSM disk hit, exact
  changed answer.
- Found that startup said stored quantization `none` while health/live blocks said
  q4 native TQ. Traced `none` to the second generic wrapper, not native TQ; fixed
  the wording in `87e11c5ee`, reran 103 Python and 127 panel focused tests, and
  re-proved current-source Electron/API/cache behavior.
- Preserved strict-format sampling misses, unavailable live VL, and the generic
  Sessions-card `JANG` label as explicit remaining issues. Evidence is under
  `20260719_qwen35_jangtq_current/`.

## 2026-07-19 - Qwen JANGTQ card label fixed and live-rechecked

- Commit `54222003d` makes quant labels bundle-grounded and prevents provider
  directory names from classifying base MXFP children as JANG.
- Remote label/registry/card tests passed 94/94 plus typecheck.
- Fully relaunched Electron main, used the real Sessions Start button, and
  captured `JANGTQ2 (2b)` on both the Qwen card and active header. Affine JANG
  and base MXFP controls remained distinct.
- Fresh row 440 completed one real file tool with 3,904 cached hybrid tokens and
  non-empty content, but retained a strict marker typo as PARTIAL evidence.

## 2026-07-19 - HY3 current D1 agent/stream/restart gate

- Used Electron single-model Start to replace Qwen with HY3; health and argv
  confirmed the real text-only affine JANG_2K bundle, Hunyuan/qwen3 parsers,
  q4 stored prefixes, and native MTP depth 1.
- Electron rows 443/446/449 exact-finaled a no-tool calculation, one real
  file-info turn, and no-tool history recall with distinct reasoning and
  progressive paint.
- Raw curl-N Responses/Chat no-tool, tool, and post-result continuations passed.
- Replaced HY3 PID 27632 with 29852; row 452 restored 4,655 disk/TQ-native
  tokens, exact-finaled, and health proved current D1 draft/accept execution.
- Preserved 318/318 focused tests and evidence under
  `20260719_current_hy3_mtp/`; long soak and MTP-Off A/B remain partial.

## 2026-07-19 - Step 3.7 JANGTQ zero-patch image/video and restart-L2 gate

- Used the real Electron Start/Stop/Start path for
  `Step-3.7-Flash-JANGTQ_K`; the artifact remained correctly classified as
  JANGTQ/MXTQ codebook quantization rather than affine JANG or MLX MXFP.
- Reproduced the 169-placeholder/zero-embedding image failure and traced it to
  false truthiness of MLX `array([0])` patch metadata. Normalized MLX/NumPy
  metadata and removed the truthiness branch; no prompt/parser/sampler fix.
- Electron rows 455/458/461/464 prove image A cold, identical-A resident hit,
  same-shape B isolation, and return-A. Row 467 reads a real MP4 as exact
  `VIDEO-B-8264`. DOM mutation traces prove content paints progressively.
- Literal curl-N Chat/Responses traces keep reasoning and content separate and
  finish cleanly. The Chat trace classifies cold self-correction as native
  content, retained as a strict-format miss.
- After visible process replacement with zero L1 state, row 470 restored 4,290
  tokens as `paged+mixed_swa+disk`; health reports 68 disk and q4 native-TQ
  hits, exact content, and 1.71s TTFT.
- Preserved two telemetry defects: same-process disk counters can increase while
  per-request detail omits disk, and the restarted Electron header lost its PID
  while still showing Stop. Cold latency, larger video, and stochastic soak
  remain partial.
- Current focused tests pass 422/422. Evidence:
  `docs/internal/release-gates/20260719_current_step37_jangtq/`.

## 2026-07-19 - shared MLLM lazy-L2 cache-detail repair

- Traced Step row 464's missing `+disk` label to a shared MLLM omission:
  `fetch_cache` found an indexed frugal block chain before the worker lazily
  refaulted its payloads from L2 during reconstruction.
- Ported the text scheduler's worker-source promotion to the MLLM batch
  generator through `_paged_reconstruct_disk_source`; no Step-specific branch.
- Source reload plus real Electron rows 473/476 now report
  `paged+mixed_swa+disk`; the immediate same-process repeat records 68 worker
  disk blocks, exact content, and 1.99s TTFT.
- Expanded selected verification passes 513 with two intentional deselections.
- Preserved the independent post-restart PID-header failure as open.

## 2026-07-19 - Electron PID lifecycle repair

- Traced the missing Step PID to the main process dropping `proc.pid` from
  local `session:ready` payloads. The renderer summary type and two awaited
  start resolutions also omitted PID, while the Stop handler retained stale
  state.
- Added PID transport for spawned and monitored local sessions, preserved it in
  all shared-context start paths, and cleared it on Stop. Remote endpoints
  remain PID-less.
- Panel session PID/single-model/port selection passes 174/174 and typecheck.
- Fully relaunched Electron with the project venv in PATH. Visible
  Start/Stop/Start proved PID 38968 -> absent -> 39507 with SQLite and `ps`
  parity and a single local engine.

## 2026-07-19 - immediate Stop and first-turn prompt-disk eviction repair

- Reproduced a visible-answer/disk-miss sequence at the 10 GB prompt-L2
  ceiling. The model output was coherent; the new first-turn cache record was
  missing on restart.
- Traced two shared causes: shutdown cancellation could overtake deferred
  terminal cleanup, and one-message user chats were classified as assistant
  cache entries and could self-evict ahead of older user entries.
- Added the text/MLLM terminal-cleanup stop barrier and one-message role
  boundaries. Added concurrency and role regression tests; 119 focused tests
  pass.
- Relaunched the real Electron engine from patched source. Detected the final
  row in a 200 ms poll and clicked Stop immediately. The post-stop index kept
  the 1,322-token record as `user`; after restart the same chat restored 1,321
  disk tokens and exact-finaled with progressive content.
- Repeated Responses and Chat via detached on-box curl after UI restarts.
  Both restored 1,321 disk tokens; Chat emitted finish, one usage-only chunk,
  then one `[DONE]`.
- Committed and pushed source as `7a146eefb`. Evidence is under
  `20260719_prompt_disk_stop_role_durability/`. Release remains partial.

## 2026-07-19 - Responses usage event parity

- Compared the current official Responses streaming contract and generated
  OpenAI Python request types: terminal usage belongs to
  `response.completed`, while Responses `StreamOptions` exposes
  `include_obfuscation`, not Chat's `include_usage`.
- Gated the vMLX incremental `response.usage` event behind
  `X-vMLX-Stream-Usage: incremental`. Removed the Chat-style body field from
  Electron Responses requests and limited the private header to local engines.
- Added standard/private-extension server contracts, panel request-shape
  contracts, and standard stream-option decoding coverage.
- Remote current-source validation passed 83 Python selections and 111 panel
  selections; panel typecheck passed.
- Raw curl-N standard/extension A/B produced exact progressive outputs,
  contiguous sequence numbers, one completed terminal, and terminal usage. The
  standard stream had zero incremental usage events; the explicit private path
  had 337.
- Relaunched the complete Electron main process and confirmed the project venv
  engine. The visible Start button loaded openPangu PID 49982. A fresh no-tool
  turn streamed reasoning separately, painted visible answer prefixes before
  completion, exact-finaled, and persisted no warning or tool call.
- Committed and pushed source as `cc4251318`. Preserved the raw SSE, analysis,
  screenshots, DOM trace, DB rows, argv, health, tests, and source diff under
  `20260719_responses_usage_extension_parity/`.
## 2026-07-19 - minimum-width localization follow-through

- Live Electron at 600x760 disproved the prior assumption that the remaining
  work was layout-only: Korean About/API Keys, Create Session, Server Settings,
  history/inference controls, message actions, and TTS retained English copy.
- Localized the observed surfaces through the existing catalog and added the
  missing English/Chinese/Korean/Japanese/Spanish keys.
- HMR briefly displayed raw keys from an orphaned dev renderer. Killed only the
  two scoped dev Electron process trees, relaunched cleanly on CDP 9335, and
  confirmed the fresh renderer resolved translations. The startup log again
  found `/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine` version 1.6.11.
- UI-selected About/API Keys passed all five locales at 600x760 with no document
  overflow or sampled clipped control. Korean Chat, Create Session, remote
  endpoint, and the open Server Settings drawer passed the same probe.
- Focused panel contracts: 341 passed; typecheck passed. Full suites remain a
  separate current-source gate until their active reruns finish.
- Evidence: `docs/internal/release-gates/20260719_minwidth_locale_drawers/`.

## 2026-07-19 - Paged RAM + SSD L2 hierarchy repair

- Live LFM health showed the owning red: Paged On / Block L2 On indexed 528
  tokens but held zero RAM payloads because disk presence implicitly enabled
  frugal mode.
- `8a93aa910` makes the tiers independent, retains successful L2 promotions as
  evictable L1 entries, preserves typed native exceptions, and exposes the
  effective RAM mirror policy.
- Electron rows 599/602/605/608/611/614 cover cold, RAM, eviction, SSD,
  restart-partial, and disk-only-partial paths with exact visible markers and
  no warnings. Final UI state is Paged On / Block L2 On.
- Raw Chat, Responses, Anthropic, and Ollama stream/non-stream parity completed;
  strict formatting remains PARTIAL. 190 cache and 99 protocol tests pass.
- Evidence: `docs/internal/release-gates/20260719_paged_ram_ssd_hierarchy/`.

## 2026-07-19 - MiniMax M2.7 JANGTQ q4 hierarchy and protocol proof

- The real Electron Start action selected the exact M2.7 JANGTQ/MXTQ bundle,
  stopped the prior LFM process under single-model mode, and eagerly
  materialized about 38.3 GB before any prompt. The main log resolved the
  project venv engine.
- Live rows cover cold q4 native writes, RAM-only reuse, bounded eviction, SSD
  refault with promotion, zero-L1 process restart, and a 320-token partial SSD
  hit. A real Paged-Off / L2-On UI restart restored the partial prefix as
  `block-disk+tq-native` while resident bytes remained zero; Paged On was
  restored afterward.
- Raw Chat, Responses, Anthropic, and Ollama stream/non-stream outputs were
  non-empty, progressive, identical, and terminated correctly. A second run
  kept 369 reasoning deltas and eight visible deltas separate on every
  protocol.
- Raw Responses and Electron each executed exactly one real
  `file_info(panel/package.json)` continuation. Two distinct UI reasoning
  prompts produced different reasoning bytes; the three-turn UI chat retained
  history without duplicate tools or warnings.
- Preserved bundle/config truth, argv, settings, DB summaries, health, scripts,
  raw events, and screenshots under
  `docs/internal/release-gates/20260719_minimax_m27_tq_hierarchy_protocol/`.
- Scoped M2.7 verdict is live-pass; overall matrix remains partial.

## 2026-07-19 - MiniMax-M3 terminal-first cache cleanup

- Reproduced the M3 large-video terminal stall in raw Responses: progressive
  content stopped for 2.3453s while a synchronous clean prompt-boundary
  re-prefill ran before text-done/completed.
- Moved M3 hit-derived clean rederive scheduling out of
  `_process_batch_responses`; post-dispatch `_cleanup_finished` now owns both
  paged and object-cache materialization under the existing admission barrier.
- Updated cache-path and terminal-order contracts. Focused 51/51 and expanded
  101/101 selections pass.
- Electron Save & Restart changed PID 40588 -> 42270. Raw repeat emitted 40
  progressive deltas and terminaled 0.0415s after the last delta while
  restoring 1,701 `paged+disk` tokens.
- Fresh visible Electron video turn persisted non-empty output, no reasoning,
  no tool call, and no warning; UI Logs showed one L2 block reconstruction,
  deferred 1,701-token clean rederive, and 60 typed M3 layer states.
- Retained OCR/order misses as `PARTIAL`; no REAP32 retry due reboot risk.
- Evidence:
  `docs/internal/release-gates/20260719_m3_terminal_dispatch_large_video/`.

## 2026-07-19 - Generalized terminal-first path-dependent cache store

- Audited `_process_batch_responses` after the M3 fix and found the same
  pre-terminal clean-prefill lifecycle violation in DSV4, ZAYA, and mixed-SWA
  branches.
- Replaced all six architecture branches with shared typed deferred
  descriptors. The terminal finalizer now has zero direct clean-prefill calls;
  `_cleanup_finished` materializes paged/object stores after dispatch.
- Full DSV4 paged + ZAYA runtime plus M3/terminal/media/history selection:
  229 passed, 6 skipped.
- Relaunched through Electron Save & Restart (PID 42270 -> 43998). Raw M3
  terminal remained 0.0414s after the last content delta with 1,701
  `paged+disk` tokens. Fresh Electron row 662 independently persisted visible
  content without warning/tool/reasoning.
- DSV4, mixed-SWA, and ZAYA family live reloads remain explicit partials.
- Evidence:
  `docs/internal/release-gates/20260719_path_dependent_terminal_cleanup/`.
## 2026-07-19 - DSV4 terminal-first current-source live proof

- Live-proved generalized source `0c9436bce` on the Electron-started affine
  DSV4 CRACK artifact. Raw Responses emitted 11 progressive deltas/run and
  terminaled within 0.15 s for cold, RAM-hit, and changed-suffix requests.
- Exact warm saved 765 tokens. A non-terminal changed-suffix partial was
  rejected because it lacked complete CSA/HCA composite state, preserving
  correctness.
- Electron restarts 45021 -> 46544 -> 48507 eagerly loaded before first
  request. Exact SSD replay produced five raw disk promotions; Electron row 671
  restored 765 `paged+dsv4+disk` tokens. Generic TQ stayed off.
- Visible short output was exact and progressive. Strict long marker fidelity
  remains partial due intermittent `TERTERMINAL` duplication.
- Archived current raw JSON, restart health, UI/Logs/settings screenshots, and
  source/DB observations under
  `docs/internal/release-gates/20260719_dsv4_terminal_dispatch_native_l2/`.

## 2026-07-19 - ZAYA typed CCA current-source live proof

- Traced ZAYA's native `zaya_cca_v1` contract: generic TurboQuant is disabled,
  Paged Off is promoted to typed paged storage, and prefix chains without
  terminal convolution/previous-hidden state are rejected.
- Raw Responses cold/warm/changed-suffix runs streamed 26-39 visible deltas and
  terminaled within 17 ms. Exact warm saved 919 typed tokens; the unsafe
  changed suffix clean-prefilled.
- The Electron Sessions Start path eagerly loaded PID 50901. A fresh chat
  streamed exact `ZAYA-UI-FIRST-DONE`. `Save & Restart` replaced the process
  with PID 52039, which began with zero L1 and 2,101 L2 tokens before any
  request; Regenerate restored 529/537 as `paged+zaya_cca+disk` and exact-
  finaled.
- Focused contracts passed 49 with one broad-selection skip. Preserved raw
  JSON, health, driver, DOM trace, and visually inspected screenshots under
  `docs/internal/release-gates/20260719_zaya_typed_cca_terminal_l2/`.

## 2026-07-19 - safe injected stream failure closure

- Traced the global mid-stream exception contract end to end. Python already
  emitted partial text then failed terminal/usage; Electron consumed only the
  first error and cancelled early.
- Added shared nested-error extraction and deferred error propagation in the
  panel, plus Chat/Responses server contracts and a production-stream-function
  live harness.
- Ran literal curl-N failure/recovery pairs and real Electron visible textarea
  turns on both protocols. Visually inspected progressive partial, interrupted
  terminal, and immediate recovery screenshots. DB metrics and request-history
  assertions passed.
- Full validation passed (Python 6,185; panel 2,333; typecheck; Electron build).
  ESLint is N/A because no config exists; the exact tool output is retained.
- Committed/pushed source as `5f05ad72a`; public v1.6.12 is unchanged.
- Committed the proof/ledger bundle as `1cc329c05` and fast-forwarded `main`,
  `codex/live-electron-gates-20260715`, and the post-release branch to it.
  Revalidated the immutable v1.6.12 tag target, both public GitHub releases,
  both DMG hashes/staples/Gatekeeper verdicts, both installed-app signatures,
  all updater feeds, PyPI, and the Homebrew cask after the push.

## 2026-07-20 - Anthropic/Ollama injected failure recovery

- Reproduced the global Ollama false-success bug from real converter output:
  upstream `error` became no row (or an empty nonterminal raw row), then later
  `[DONE]` became `done:true`.
- Pinned the official native `{"error":"..."}` streaming contract, added a
  production-route regression, fixed all three streaming routes, and pushed
  source commit `d811270ad`.
- Ran live localhost production handlers with delayed partial output and an
  injected engine exception. Anthropic plus Ollama chat/templated/raw each
  ended natively without a false success; the immediately following request
  completed normally.
- Evidence and remaining boundaries are recorded under
  `20260720_anthropic_ollama_midstream_failure/`. Overall matrix remains
  partial; this was an API adapter closure, not a model/Electron/release row.

## 2026-07-20 - built, notarized, and live-proved v1.6.13 artifacts

- Rebuilt both production DMGs from pushed source `5fae65d38` with pinned clean
  JANG after fixing packaged child import isolation.
- Apple accepted Sequoia submission
  `bc4293f5-02f8-4f28-9cd3-d7bf51031f51` and Tahoe submission
  `4dbf39a0-d2ec-43a8-a126-ca24f3cdc3d0`; stapling, validation, codesign, and
  Gatekeeper checks passed.
- Copied exact artifacts to `erics-m5-max.local`; all DMG/blockmap SHA-256
  values matched before install. Did not touch `/Applications/vMLX.app`.
- Final Sequoia installed app loaded Gemma via real UI Start and passed a clean
  reasoning/tool/recall sequence. Exactly one `file_info(panel/package.json)`
  returned 5.2 KB. Raw Responses/Chat streamed separate rails and clean
  terminals. UI Stop/Start then restored 3,359 tokens from 53 q4-native disk
  blocks with exact visible completion.
- Final Tahoe installed app loaded independently, completed a visible coherent
  reasoning/content turn, streamed raw Chat with usage/DONE, and stopped via UI.
- Preserved screenshots, health, logs, SQLite rows, and raw SSE under the
  v1.6.13 release-checkpoint evidence directory. Publication is the remaining
  checkpoint step; broader matrix rows remain partial/open.

## 2026-07-20 - published and independently re-read v1.6.13

- Published the public source and four-asset DMG releases. Corrected the
  updater-repo lightweight tag to its 1.6.13 manifest commit `07c402d` while
  preserving the annotated source tag at `2f509f79d`.
- Published updater manifests, Homebrew cask `0b0f54c`, PyPI `vmlx==1.6.13`,
  and the `mlx.studio` origin feed; independently re-read exact versions,
  sizes, and hashes from every public surface.
- Ran publication and verification from `erics-m5-max.local`. Preserved the
  dirty older checkout and created a clean exact-tag release worktree.
- Re-read the installed Electron Gemma rows. The corrected Sequoia tool row is
  a real one-call `file_info` result (`5.2 KB`); the earlier raw-markup
  `221 bytes` row is excluded. Tahoe content is coherent but not marker-only.
- Checkpoint is public and usable within its named proof scope. The broader
  matrix remains partial/open; pause after final proof push and remote fetch.

## 2026-07-22 - v1.6.16 shared rails and settings restart work

- Added split-marker holdback across DeepSeek, ThinkXML, and M3 reasoning
  parsers; repaired Anthropic late block ordering/terminal handling; normalized
  Ollama streaming reasoning policy for M3/openPangu/Mistral4 and Off history.
- Live Laguna on the real Electron dev app proved separate UI reasoning,
  exact real tools, raw four-protocol progressive rails, and honest direct-rail
  Auto controls. Privacy-safe summaries/screenshots are committed; raw private
  reasoning was not committed.
- Fixed settings restart ownership: removed both post-stop delays, emitted
  `session:updated`, refreshed current PID/config in Chat Settings, and renamed
  running plain Save to `Save for Next Restart`. Focused panel/typecheck passed.
- Relaunched the patched main process with the venv PATH and current user-data
  directory. Two real Save & Restart cycles applied DEBUG then INFO, refreshed
  PID without staleness, and completed a post-restart exact turn with a
  disk/TQ-native promotion. Source/evidence/docs are pushed through
  `f9a4b6838` and synchronized while preserving unrelated dirty state.

## 2026-07-22 - MiniMax-M3 named-family Ollama and Electron reasoning proof

- Stopped Laguna through the UI and started the exact MiniMax-M3 Coder Small
  card through the real Sessions Start control. PID 60303 reached loaded health
  and roughly 52 GB RSS before its first request.
- Inspected the real bundle/config and visible settings: affine JANG_2L (not
  JANGTQ/MXTQ), temperature 1.0, top-p 0.95, top-k Off, MiniMax-M3 tool and
  reasoning parsers, native MSA cache, no generic TQ, no JIT.
- Captured Electron stream IPC: 245 reasoning events to 620 chars, ten content
  events to 18 chars, then one completion. Captured an exact one-tool UI turn
  and screenshots.
- Raw Responses emitted 83 reasoning plus eight content deltas and one
  completion. Direct Ollama On/Off and gateway Auto all exact-finaled with one
  terminal and clean separated fields.
- Recorded the truthful remaining boundaries. A follow-up public check
  corrected the release wording: `jjang-ai/vmlx` is the source release and has
  no assets, while `jjang-ai/mlxstudio` currently carries the two DMGs and two
  blockmaps. No package, tag, or public release action was performed.

## 2026-07-22 - reconciled exhaustive 1.6.16 worklist

- Re-read the current campaign, active worklist, global reasoning audit, LFM
  handoff, settings/default gate, release guard, release checkpoint, user
  attachment, historical model matrix, and historical cache/UI matrix.
- Kept the current campaign board authoritative where old matrices conflict;
  notably, the old June paged-Off default directive does not supersede the
  current architecture-aware policy or the user's later requirement.
- Added `OPEN-ISSUES-RANKED.md` with ranked acceptance criteria, family rows,
  no-repeat scoped closures, release provenance, and the final 1.6.16
  signing/notarization/publication gate.
- Checked GitHub release truth live: source v1.6.15 remains public; the
  MLXStudio distribution release exposes both DMGs/blockmaps with the retained
  SHA-256 values; both raw updater manifests still name 1.6.15.
- Checked repo truth: vMLX campaign head is synchronized locally/GitHub/remote
  proof checkout through `aa97a531b`; the behavior evidence checkpoint remains
  `6de9ce8ef`, and the branch is ten commits ahead of main.
  JANG GitHub main is not safely represented by the current dirty developer
  tree, so clean provenance remains an explicit packaging gate.

## 2026-07-22 - JANG 2.5.33 and current-source Laguna distribution proof

- Identified the public packaging split: signed v1.6.15 apps already contained
  the fixed per-module Laguna runtime, but PyPI `vmlx==1.6.15` allowed
  `jang>=2.5.29`, so Python/CLI installs could resolve a stale uniform-bit
  runtime and reproduce the `576/48 bits=8` failure.
- Published JANG 2.5.33 from `b788273e`; GitHub release and PyPI wheel/sdist
  digests match. Full clean JANG suites passed on both source boxes.
- Pushed vMLX `b6d38eac7` and `e4c6762ce`: require JANG 2.5.33, reject stale
  mixed-affine runtimes, and log imported runtime provenance. The focused vMLX
  set passed 370/370 on both boxes; engine-path tests passed 7/7.
- Relaunched the real remote Electron app cleanly. After its initial probe, it
  left the installer screen and the main log found the venv engine. Real Start
  loaded Laguna PID 39057 with `PYTHONPATH` pinned to the synchronized release
  checkout; session logs recorded JANG marker 1 before model load.
- Current-source Electron showed a separate reasoning rail and a subsequent
  exact-one real `file_info` continuation. Raw Responses Auto/Off and Chat
  separated reasoning/content with truthful terminals and no native marker
  leak. The global campaign remains partial; Paged-Off restart, long eviction,
  broader protocols/gateway, full suites, and release packaging remain open.

## 2026-07-22 - Qwen full-catalog parser and protocol release cut

- Fixed setup/Install Engine divergence by sharing the source-venv resolver
  between installation checks and session launch.
- Captured two real Qwen3.6 JANGTQ malformed closed tool wrappers and added
  request-schema-gated parser support. Missing arguments are not synthesized
  and unadvertised tools remain rejected.
- Proved 16/16 explicit Thinking-Off coding-harness flows through direct and
  gateway Chat, Responses, Anthropic, and Ollama in both modes.
- Live Electron executed `file_info` and `run_command` with warm
  `paged+ssm+disk+tq-native` reuse. The model's final copy dropped one path
  segment, retained as a strict-synthesis partial.
- Thinking-On required-tool A/B remains a real artifact/runtime limitation.
  Evidence is under `20260722_qwen35_release_checkpoint/` at `74dadd30c`.

## 2026-07-22 - Bonsai/Laguna partial SSD cache proof

- Treated old legacy disk cache as out-of-scope for the .16 emergency gate
  unless it is the explicit active tier. The release-critical tier is Block
  Disk Cache / SSD L2.
- Real Electron current-source Bonsai was exercised through UI multi-turn,
  exact tool use, Chat/Responses API reasoning separation, Paged-On partial SSD
  restore after restart, and Paged-Off disk-only partial SSD restore with zero
  resident paged RAM.
- Real Electron current-source Laguna JANG_4M was exercised with Paged-Off
  Block Disk L2 and proved a post-restart never-stored changed suffix restoring
  6,400 SSD tokens with mixed-SWA native state preserved.
- Real Electron current-source Gemma 4 E2B JANG_4M was created with app-derived
  parser/defaults and exercised both cache modes. Paged-On clean-restart
  suffix D restored 4,672 SSD tokens as `paged+mixed_swa+disk`; Paged-Off
  disk-only restart suffix C restored 4,672 SSD tokens as
  `block-disk+mixed_swa` with `ram_tokens_cached=0`.
- While the same Gemma session was loaded, captured three visible UI turns:
  separate reasoning rail plus exact answer, multi-turn recall with
  `68 paged+mixed_swa cached`, and one real `Info panel/package.json` tool card
  with exact final. Gateway Chat and Responses streamed separate reasoning and
  content; gateway Chat emitted a required `file_info` call and completed the
  tool-result continuation exactly.
- Swapped to Laguna JANG_4M and traced parser provenance before prompting:
  bundle vendor parser is `poolside_v1`, current argv is `glm47`/`deepseek_r1`,
  and source registers `poolside_v1` as aliases to those parser classes. Fresh
  UI proof passed reasoning/answer/tool with one real file-info card and
  `4861 block-disk+tq-native cached`. Gateway Chat Auto terminal and Chat tool
  rows passed; Gateway Responses remains partial because reasoning+hard prompt
  ended incomplete while short prompt completed without reasoning.
- Added the focused cache-control policy regression that Block Disk SSD/L2
  stays available whether Paged RAM is on or off, and that legacy disk is only
  available after both Paged RAM and Block Disk L2 are off. After syncing the
  patch to the live remote checkout, the focused policy test passed `18/18`.
- Updated the campaign README and ranked open issues. Bounded eviction/refault,
  corrupt/missing companion fallback, and remaining cache archetypes are still
  open.

## 2026-07-22 - Laguna Anthropic/Ollama parser and tool addendum

- While the Electron-started Laguna JANG_4M PID remained loaded, captured raw
  gateway Anthropic `/v1/messages` and Ollama `/api/chat` event streams.
- Anthropic hard prompt produced separate protocol-native reasoning deltas and
  text deltas with `message_stop`, no native marker leakage, but visible prose
  over-generation. Anthropic required-tool and tool-result continuation passed
  exact `file_info(panel/package.json)` and exact final
  `The package file is 5.2 KB.`.
- Ollama hard prompt produced separate `thinking` and content deltas with
  terminal `stop`, no native marker leakage, but visible prose over-generation.
  Ollama required-tool and tool-result continuation passed exact
  `file_info(panel/package.json)` and exact final
  `The package file is 5.2 KB.`.
- Retained artifact:
  `docs/internal/release-gates/20260722_laguna_r16_parser_ui_api/laguna-anthropic-ollama-gateway-proof.json`.
  Strict-format hard-prompt rows remain partial; parser/tool transport rows
  are scoped live passes.

## 2026-07-22 - Laguna four-block Paged-RAM eviction/refault

- Restarted the same Electron-started Laguna JANG_4M session with Paged RAM on,
  `maxCacheBlocks=4`, and Block Disk L2 on. Health confirmed
  `backend_mode=paged`, `capacity_tokens=192`, and native mixed-SWA q4 storage
  policy.
- Ran a bounded direct-engine store/changed-suffix/pressure/refault sequence.
  All rows exact-finaled, reached `[DONE]`, and leaked no native markers.
- L1 pressure took effect: `l1_evictions` rose `0 -> 6`; final refault disk
  hits rose `9 -> 12` while exact-finaling `LAG-FOUR-C`. The direct-engine
  health path did not populate `last_cache_execution`, so the proof relies on
  counter deltas plus output correctness.
- Restored the session to Paged RAM off, `maxCacheBlocks=1000`, Block Disk L2
  on; post-restore health confirmed those effective values.
- Retained artifact:
  `docs/internal/release-gates/20260722_laguna_r16_parser_ui_api/laguna-four-block-eviction-refault.json`.
  Low-limit `Block Cache Max (GB)` disk eviction/refault remains open.

## 2026-07-22 - Laguna Block Cache Max GB eviction/refault

- Ran the SSD budget proof in isolated Block Disk Cache directories. The
  first `0.03 GB` cap is retained as a negative control: the cache wrote and
  evicted `198` blocks immediately, so no refault could survive and two visible
  rows were not exact.
- Reran with `blockDiskCacheMaxGb=0.25` and neutral exact-marker prompts. All
  rows exact-finaled, reached `[DONE]`, and leaked no native markers. Final
  counters: `disk_writes=62`, `disk_hits=53`, `disk_evictions=59`,
  `blocks_on_disk=3`.
- The surviving-prefix refault row increased disk hits by `2` and exact-finaled
  `OK_CHARLIE_B`; the older-prefix after-pressure row exact-finaled
  `OK_ALPHA_C`.
- Restored the session to Paged RAM off, `maxCacheBlocks=1000`,
  `blockDiskCacheMaxGb=10`, default disk path, and Block Disk L2 on.
- Retained artifacts:
  `docs/internal/release-gates/20260722_laguna_r16_parser_ui_api/laguna-block-disk-gb-cap-eviction.json`
  and
  `docs/internal/release-gates/20260722_laguna_r16_parser_ui_api/laguna-block-disk-gb-cap-eviction-025gb.json`.
  Cache matrix remains partial for corrupt/missing companion fallback and
  untested architecture archetypes.

## 2026-07-22 - 1.6.16 scoped parser/cache preflight

- Added fail-closed `panel/scripts/scoped-release-preflight-16.py` and wired
  `panel/scripts/build-release-dmgs.sh` to accept
  `VMLINUX_RELEASE_SCOPE=r16_parser_cache`.
- The scoped gate validates only the emergency 1.6.16 parser/cache release
  checkpoint: source/package version stamps, updater hold at 1.6.15, cache
  terminology artifacts, Bonsai/Laguna/Gemma partial SSD cache with Paged RAM
  on/off, Laguna four-block RAM refault, Laguna 0.25 GB Block Disk cap refault,
  and current Gemma/Laguna/Bonsai/Qwen API reasoning/tool artifacts.
- Ran:
  `python3 panel/scripts/scoped-release-preflight-16.py --out build/current-scoped-release-preflight-16-parser-cache.json`
  and it reported `scope=r16_parser_cache`, `version=1.6.16`, `status=pass`.
- Also exercised the same preflight invocation shape used by the build script
  with `VERSION` read from `panel/package.json`; it reported `status=pass`.
- This does not close the broad old release regression manifest or full
  cross-family matrix. It only provides an auditable package-clearance path for
  a user-approved 1.6.16 emergency parser/cache checkpoint.

## 2026-07-22 - JANG 2.5.34 release provenance

- The default `/Users/eric/jang/jang-tools` checkout was dirty and on older
  `2.5.30`; production build correctly refused to bundle it.
- Created clean worktree
  `/Users/eric/jang/jang-tools-r16-2534` from current JANG `origin/main`
  (`2.5.33`), branch `codex/r16-vmlx-1.6.16-jang-tools`.
- Cherry-picked `8ae60a7` as `e3c4c24` to preserve DSV4 adaptive pool-cache
  residency, bumped package metadata to `2.5.34` in `6e28ff2`, and pushed the
  branch to both `jjang-ai/jangq` and `jangq-ai/jangq`.
- Ran focused JANG verification in the clean worktree:
  `53 passed` for DSV4 pool residency, pack, format, and writer tests, plus
  `py_compile` for the touched modules.
- Built `jang-2.5.34` sdist/wheel and `twine check` passed. PyPI upload is
  blocked by missing API-token credentials, not by package validation.

## 2026-07-22 - 1.6.16 source build and math renderer precheck

- Reran scoped vMLX preflight, scoped Python regression, cache-control policy,
  TypeScript, and diff-check; all passed.
- Ran `npm --prefix panel run build` with
  `VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/jang-tools-r16-2534/jang-tools`.
  The previous dirty-JANG guard no longer fired; bundled Python installed local
  `vmlx-1.6.16` and local clean `jang-2.5.34`.
- Bundled verification reported critical `vmlx_engine` and `jang_tools` files
  match source content, no editable installs, relocatable scripts, and critical
  import coverage including Gemma4, Qwen3-VL, JANGTQ, MiniMax, Step3.7, and
  audio/vision dependencies.
- Production renderer build completed and emitted KaTeX assets, which is the
  expected shipped path for the LaTeX display regression.
- Renderer focused tests for math markdown, reasoning display, and interleaved
  reasoning rendering passed `124/124`. Live API/Electron math proof remains
  pending; this row is source/build precheck only.

## 2026-07-22 - current Laguna UI/API tool and Responses rerun

- Classified the earlier current Laguna UI `file_info` miss as invalid proof:
  the proof chat had no `chat_overrides` row, so Electron sent no built-in
  tool schemas to the model.
- Reran the same current Electron app/session with Chat Settings overrides
  persisted through `chat:setOverrides`: `builtinToolsEnabled=true`,
  `workingDirectory=/Users/eric/mlx/vllm-mlx-r16-reasoning-p0-live`,
  explicit Thinking On, `maxTokens=512`, and `maxThinkingTokens=256`.
- Electron UI Responses wire exact-finaled
  `R16-LAGUNA-UI-TOOL-RESPONSES-DONE` after one real
  `file_info(panel/package.json)` result (`Size: 5.2 KB`).
- Electron UI Chat Completions wire exact-finaled
  `R16-LAGUNA-UI-TOOL-COMPLETIONS-DONE` after one real
  `file_info(panel/package.json)` result.
- Raw Chat Completions coding-harness rerun emitted first-stream
  `finish_reason=tool_calls`, parsed `file_info({"path":"panel/package.json"})`,
  accepted the real filesystem result, and second-stream exact-finaled
  `R16-LAGUNA-API-CHAT-TOOL-CONT-DONE`.
- Raw Responses rerun emitted separate reasoning-summary deltas, visible
  output-text deltas, `response.output_text.done`, and `response.completed`.
  It overexplained instead of obeying the exact visible-answer instruction, so
  strict-format model compliance remains `PARTIAL`; the transport terminal row
  is current-source proven.
- Final raw Chat tool health retained Paged-Off SSD/TQ reuse:
  `disk_hits=48`, `tq_native_hits=48`.
- Retained artifacts:
  `docs/internal/release-gates/20260722_v1_6_16_campaign/current-reruns/`.

## 2026-07-22 - start v1.6.17 consolidation and align Ollama private history

- Started from clean baseline `e0b49ec29` in isolated worktree
  `/Users/eric/mlx/vllm-mlx-r17-consolidation`.
- Reconciled the user-supplied `.17` requirements with the retained `.16`
  open matrix and wrote
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/README.md`.
- Audited direct Python and Electron-gateway Ollama request translation.
- Found direct route normalized assistant `message.thinking` before text/media
  conversion while gateway returned text messages unchanged and retained the
  alias on media messages.
- Updated:
  - `vmlx_engine/api/ollama_adapter.py`
  - `panel/src/main/api-gateway.ts`
  - `tests/test_ollama_adapter.py`
  - `panel/tests/api-gateway-ollama-behavior.test.ts`
- Canonical rule: non-empty existing `reasoning_content` wins; otherwise a
  non-empty assistant `thinking` becomes `reasoning_content`; the alias is
  removed even when empty.
- Verification:
  - `PYTHONPATH=$PWD /Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest -q
    tests/test_ollama_adapter.py tests/test_ollama_reasoning_parity.py
    -p no:cacheprovider` -> `36 passed`.
  - Full focused reasoning/parser/adapter/agentic protocol set after the
    change -> `472 passed`.
  - `npm test -- --run tests/api-gateway-ollama.test.ts
    tests/api-gateway-ollama-behavior.test.ts` -> `59 passed`.
  - `npm run typecheck` -> pass.
  - Selected paged/block-disk/TQ/hybrid-SSM cache regression set ->
    `111 passed`.
- Installed this worktree's locked panel dependencies with `npm ci`; did not
  reuse another checkout's `node_modules`.
- Live Electron/direct/gateway model proof remains open. No release readiness
  claim was made.

## 2026-07-22 - remote Ornith current-source live proof

- Removed only orphaned `.16`/`.17` dev Electron trees on the authorized M5
  Max; left the local stale installed `1.6.9` app untouched.
- Created a checkout-local `.17` project venv from the functioning `.16`
  environment, replaced its editable install with the `.17` checkout, and
  verified exact Python/module provenance.
- Clean dependency resolution itself failed on Python 3.13 because the current
  dependency graph selected unsupported `llvmlite 0.36.0`; release bundling
  must be reverified independently.
- Started the real Electron UI on CDP 9335 with a fresh isolated profile.
- Loaded Ornith/Qwen3.5 via the UI Start button and retained bundle/argv/health
  truth for parsers, q4 attention-only TQ storage, hybrid typed state, paged
  RAM, and block-disk L2.
- Completed three UI turns including visual math, separate reasoning, one
  truthful tool failure due to unset workspace, and a successful interleaved
  reasoning/tool/result continuation after configuring the workspace.
- Completed raw streamed Chat, Responses, Anthropic, and Ollama probes through
  the Electron gateway. Chat included a three-generation required-tool loop.
- Restart test found persisted TQ attention blocks but could not use the
  changed-tail partial prefix because no matching SSM companion boundary was
  restored. Logged this as an open hybrid partial-SSD defect, not a cache pass.


## 2026-07-22 - close Ornith hybrid partial SSD restart discovery

- Root cause: after restart, exact SSM boundary lookup could reach L2, but a
  shorter persisted companion boundary was invisible because the process-local
  length index started empty.
- Added validated sidecar boundary discovery and retained normal typed fetch as
  the sole acceptance path.
- Pushed source as `be6cc8497` after `54` focused and `167` broader cache tests.
- Live remote Electron/gateway proof accepted `9,216` of `9,279` SSD tokens
  with q4-TQ attention blocks and typed SSM state in both Paged-On and SSD-only
  modes. SSD-only kept RAM resident bytes/tokens at zero.
- Restored the Electron session to Paged RAM on + Block Disk on.

## 2026-07-22 - consolidate bundle generation-default selection

- Confirmed two main-process readers independently selected
  `generation_config.json`/JANG sampling defaults and disagreed on invalid
  `max_new_tokens`.
- Added `resolveBundleGenerationDefaults()` as the single pure owner and wired
  both Chat Settings and startup/session hydration through it.
- Kept thinking-budget/template capability detection in the existing
  main-process reader rather than broadening the shared module.
- Added cases for JANG field precedence, negative output limits, disabled
  top-k, and DSV4 repetition selection.
- Remote checks: `35 passed`, `318 passed`, `tsc --noEmit` passed, and
  `git diff --check` passed.
- Live Gemma 4 settings/request parity remains next; this entry is not runtime
  proof.

## 2026-07-23 - close scoped Gemma settings/health/default parity

- Reproduced the live status mismatch on current PID `97058`:
  `/v1/capabilities` exposed bundle sampling defaults while `/health` omitted
  them, and neither exposed one shared effective omitted-request view.
- Added `server.py::_model_effective_defaults_status()` and routed both
  endpoints through it. Bundle defaults and runtime-effective defaults remain
  separate; integer sampler fields are normalized without injecting argv
  overrides.
- Focused new/adjacent checks passed `4/4`; the broader runnable
  settings/health/capabilities selection passed `57`. Three existing async
  tests were not runnable because `pytest-asyncio` is absent from the project
  venv.
- Used the real Electron Server Settings Save & Restart control. PID `97849`
  loaded from this checkout's venv with the expected Gemma parsers, Paged RAM,
  Block Disk L2, and explicit TQ-none diagnostic configuration.
- Verified initial bundle defaults, exact SQLite/request/engine override
  propagation, New Chat, Reset, explicit Off across restart, and the distinction
  between blank bundle max-output and the engine's reported 16,384-token
  reasoning fallback.
- Fresh UI Auto output retained a separate reasoning rail, exact final, and
  disk-assisted paged cache hit. Raw direct Chat and dev-gateway Responses both
  emitted separate progressive reasoning/content, truthful terminals, exact
  finals, and zero marker leaks.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/gemma-settings-health-gateway-live.json`.

## 2026-07-23 - DSV4 bundle pool codec and math/API proof

- Traced the live DSV4 pool-codec mismatch to hard-coded panel and engine
  defaults that ignored `jang_config.cache.pool_quant_default=false`.
- Added one model-detection field and one engine bundle resolver; preserved
  explicit saved UI/env overrides and unstamped legacy fallback.
- Real Electron reset/save/start loaded PID `3423`. Health reports
  `pool_quant.enabled=false`, env `0`, native composite prefix/paged/block-L2
  active, generic TQ KV disabled.
- Tightened inline KaTeX parsing to one line and changed parse errors to escaped
  fallback. Live completed UI surface rerendered with zero `.katex-error`.
- Direct raw Chat SSE preserved LaTeX bytes. Direct reasoning Chat SSE kept
  reasoning/content deltas separate and terminalized once.
- Focused verification: panel `441`, math `13`, Python DSV4 `80`, typecheck
  pass, bundled-Python and Electron production build pass.

## 2026-07-23 - DSV4 exact native-cache and DSML live proof

- Ran one combined cache/protocol gate against real Electron-started PID 3423.
- Traced the prior zero cache state to the 256-token DSV4 snapshot threshold.
  A 2,187-token cold prompt stored 2,186 native composite tokens; the exact
  repeat restored all tokens as `paged+dsv4` and wrote nine L2 blocks.
- The changed-tail prompt exposed an explicit architecture boundary: matching
  non-terminal blocks have `deepseek_v4_pending` local state but no terminal
  CSA/HCA composite. The guard safely forced full prefill; safe partial typed
  checkpointing/rederive remains an implementation task.
- Direct streamed Chat and gateway streamed Responses each emitted one real
  `file_info` function call, consumed the actual 5,336-byte result, separated
  reasoning from visible content, and terminalized exactly once without DSML
  residue.
- Real Electron Chat showed a separate reasoning rail, one `Info` tool card,
  and a non-empty final. Exact final-marker compliance was partial.

## 2026-07-23 - fail closed on non-equivalent DSV4 composite cache restores

- Fixed block-L2 thread ownership at `e9eadc6cc`: block deserialization now
  runs on the scheduler model worker. A real Electron restart restored a
  simple 1,291-token DSV4 SSD prefix and exact-finaled without the prior MLX
  stream-owner exception.
- Three-turn Electron inspection then exposed stale visible replay from a
  269/337 partial DSV4 cache hit. The identical raw Responses history with
  prefix cache bypass exact-finaled.
- Replaying after a real Electron restart found a newly stored exact 336/337
  SSD checkpoint, but that path looped for 2,647 private-reasoning tokens with
  no visible answer. Exact N-1 shape is therefore not sufficient DSV4
  composite equivalence proof.
- Added the correctness gate in `e9149f566`: all DSV4 paged/L2 hits are
  released and hit credit is rolled back before full prefill.
- Current Electron PID `11278` logged the 336/337 rejection and all-337-token
  prefill. The old math replay disappeared. The full-prefill run independently
  looped for 2,616 private-reasoning tokens and was interrupted, so quality
  remains a separate release blocker.
- Verification: 70 selected DSV4/cache tests pass; Python compile and
  `git diff --check` pass.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/dsv4-l2-owner-and-equivalence-live.json`.

## 2026-07-23 - normalize effective reasoning parser and retain math proof

- Replaced duplicated parser enablement logic with
  `resolveEffectiveReasoningParser()` and
  `reasoningParserIsEnabled()`.
- Added a detection-gated Laguna v2 parser migration and hard opt-out semantics
  for both empty and literal `none`.
- Wired session launch/preview, Chat Settings, session shell/toolbar, chat IPC,
  Harmony selection, and Ollama capabilities to the shared resolution.
- Added behavior coverage for Ollama Auto/None/no-thinking capability and form
  display coverage for persisted literal None.
- Restarted the isolated dev Electron and used the real Start/Stop/Save &
  Restart controls. Auto PID `20507` launched `deepseek_r1`; explicit-None PID
  `20077` launched `none`; Auto was restored.
- Verified a fresh New Chat showed bundle/health sampling defaults.
- Ran a literal LaTeX Electron turn: two KaTeX nodes, zero errors, visibly
  formatted multiplication and fractions. Raw gateway Chat SSE reconstructed
  the exact literal commands with progressive content and one terminal.
- Diagnosed and fixed the proof driver retaining its CDP socket and leaving
  stale SSH clients. Removed only stale local SSH clients; Electron and model
  processes on the proof host were not killed by that cleanup.
- Remote result: `494` focused tests, typecheck, `node --check`, and diff check
  pass.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/laguna-parser-settings-math-live.json`.

## 2026-07-23 - prove Laguna partial SSD reuse with Paged RAM on and off

- Kept the exact Laguna S2.1 JANG_4M bundle and real Electron-started session.
- Paged On: wrote a 2,983-token prefix, cleared RAM only, then restored 2,944
  changed-tail tokens from `paged+disk+tq-native`; the next changed tail used
  resident `paged+tq-native`.
- Used the real Server Settings UI to turn off only In-Memory Paged Cache
  (RAM). Save & Restart produced PID `22169` with `--no-paged-cache` while SSD
  L2 and q4 native storage remained enabled.
- The first Paged-Off gateway request reused blocks written by the prior
  Paged-On process. Two gateway and one direct changed-tail requests all
  restored 2,944 tokens and exact-finaled.
- Ran three distinct Electron chats. The exact cross-chat replay restored
  6,528/6,580 tokens from SSD and visibly reported the cache detail and timing.
- Confirmed `scheduler_cache.total_tokens_cached` is indexed tokens; the
  authoritative disk-only aggregate stayed at zero resident tokens and bytes.
- Restored Paged-On plus SSD L2 through the UI; current PID `22853`.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/laguna-paged-on-off-ssd-partial-live.json`.

## 2026-07-23 - prove Block Cache Max SSD eviction

- Used the real Server Settings UI to set Laguna Block Cache Max to 1 GB and
  restart.
- PID `23420` started with `--block-disk-cache-max-gb 1`.
- The next write evicted 156 blocks and reduced 2.252 GB to 0.790 GB.
- Replayed an older prefix: zero cached-token credit, 52 SSD misses, safe full
  prefill, exact output, and bounded refill. Cumulative evictions reached 206.
- Restored the 10 GB default through the real UI; PID `23830` is healthy.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/laguna-ssd-capacity-eviction-live.json`.
## 2026-07-23 - R17 Gemma combined UI/API/media proof

- Started `JANGQ-AI/gemma-4-12B-it-qat-JANG_4M` with the real Electron Start
  button at PID `24290`.
- Inspected three UI turns closely: separate reasoning, exact
  `file_info(panel/package.json)` call/result, no second tool, history recall,
  progressive output, and one valid KaTeX fraction.
- Ran Chat, Responses, Anthropic, and Ollama raw streaming on the same process.
  All kept reasoning/content separate and used truthful terminal events;
  required tool continuations completed with the real `5.2 KB` result.
- Verified gateway raw LaTeX byte fidelity and Electron KaTeX display.
- Sent `/private/tmp/gemma-audio-marker.wav` through gateway and direct
  Responses with thinking Off; both exact-transcribed in 20 deltas.
- Attached the same WAV through Electron. SQLite retained a canonical
  `input_audio` part and Auto reasoning stayed in its rail, but transcript
  quality failed in both an existing chat and a fresh chat.
- Retained the failure rather than forcing reasoning Off or rewriting output.

## 2026-07-23 - R17 MiniMax M2.7 and global replacement-template cache keys

- Loaded MiniMax M2.7 through the real Electron Start button and combined the
  required three UI turns with reasoning, tool, history, KaTeX, timing, and
  cache inspection.
- Ran raw direct and gateway Chat, Responses, Anthropic, and Ollama reasoning
  streams plus direct/gateway Responses tool loops.
- Reproduced changed-tail partial SSD misses with Paged RAM Off, then traced
  them to `_generation_prompt_cache_extra_key`: replacement-style templates
  hashed user-tail text into every block key.
- Changed the fallback to hash only text after the with/without-generation
  render divergence and added a MiniMax-shaped regression test.
- Live after repair:
  - Paged Off: 1,600/1,631 tokens from `block-disk+tq-native`, RAM 0.
  - Paged On after UI restart: 1,600/1,634 from
    `paged+disk+tq-native`, then 1,600 from `paged+tq-native`.
- Remote selected tests: 132 passed.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/minimax-m27-ui-api-cache-live.json`.

## 2026-07-23 - R17 Gemma typed mixed-SWA SSD partial reuse

- Avoided rerunning the already-current Gemma reasoning/tool/KaTeX/API row.
- Used the real Electron settings and Save & Restart controls to prove:
  - Paged Off + SSD On restored 2,624/2,709 changed-tail tokens from
    `block-disk+mixed_swa` with zero resident RAM.
  - Paged On + SSD On restored 2,624/2,710 from SSD after restart, promoted
    state to RAM, then reused 2,624/2,711 from RAM for the next tail.
- Retained exact argv, health, cache execution, output, and UI checkbox
  evidence.
- Remote selected tests: 139 passed.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/gemma-mixed-swa-paged-on-off-ssd-live.json`.

## 2026-07-23 - R17 Bonsai hybrid partial SSD closure

- Created and launched a fresh Bonsai 1-bit Electron session with current
  model-derived defaults.
- Confirmed 16 attention-KV lanes use TQ8 SSD storage while 48 SSM/GDN lanes
  keep native companion state.
- Proved changed-tail partial reuse in three hierarchy states:
  - resident Paged-On: `paged+ssm+tq-native`;
  - process-restarted Paged-Off: `block-disk+ssm+tq-native`, zero RAM;
  - process-restarted Paged-On: `paged+ssm+disk+tq-native`, then RAM reuse.
- Every accepted hit carried a matching companion checkpoint; there was no
  KV-only credit or silent downgrade.
- Restored the Paged-On + SSD-On default before leaving the gate.
- Remote selected tests: 209 passed.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/bonsai-hybrid-paged-on-off-ssd-live.json`.

## 2026-07-23 - R17 Qwen 35 JANGTQ hybrid partial SSD closure

- Created a fresh Electron session for the exact Qwen 35 JANGTQ artifact and
  retained the pre-launch model-derived cache UI.
- Confirmed live JANGTQ/MXTQ codebook weight truth separately from cache
  storage: 2-bit routed experts, q4 attention-KV SSD storage, native SSM/GLA
  companion state.
- Proved the same 4,160-token shared boundary in three hierarchy states:
  - resident Paged-On: `paged+ssm+tq-native`;
  - process-restarted Paged-Off: `block-disk+ssm+tq-native`, zero L1 payload;
  - process-restarted Paged-On: `paged+ssm+disk+tq-native`, then resident RAM.
- Every changed-tail probe exact-finaled. The accepted disk hit included 65
  TQ-native attention blocks plus a matching SSM companion; no unsafe KV-only
  hit was credited.
- Restored the model-derived Paged-On + SSD-On state before leaving the row.
- Remote selected tests: 209 passed.
- Retained raw requests, health snapshots, and UI screenshots under:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/qwen35-jangtq-*`.

## 2026-07-23 - remove per-request gateway launch preflight

- Matched direct and gateway Responses on the current Qwen 35 JANGTQ PID.
  Gateway TTFT was `2.30-2.34s` versus direct `0.13-0.16s`.
- Traced the delay through
  `prepareSessionForRouting -> preflightSessionStart -> findEnginePath ->
  getDevelopmentProjectVenv`, whose fresh Python import measured
  `2.15/2.17/2.16s`.
- Changed the gateway to run launch preflight only when the resolved target is
  not already running. Kept one-model cleanup for running targets and kept
  preflight-before-unload/rollback for actual load transitions.
- Added a behavior test pinning both halves: no repeated preflight for the
  running target, but the competing running session is still stopped.
- Restarted only the isolated R17 Electron app, clicked real Start, and loaded
  Qwen 35 JANGTQ PID `92163`.
- Post-fix direct/gateway TTFB and total times overlapped. Raw gateway
  Responses retained 256 progressive deltas and one truthful incomplete
  terminal.
- The real UI follow-up displayed the exact same output and warning at
  `98.1 t/s`, with current typed cache telemetry.
- Remote selected tests: 133 passed, three skipped. Typecheck, production
  Electron build, and diff check passed.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/qwen35-gateway-running-session-latency-live.json`.

## 2026-07-23 - live Ollama history normalization and tool loop

- Kept the already Electron-started Qwen 35 JANGTQ PID `92163` loaded.
- Ran the existing allowlisted agentic harness through direct port `8006` and
  Electron gateway `8088` with identical Ollama request bodies.
- Both endpoints streamed separate private thinking, called real
  `file_info(panel/package.json)`, consumed its `5.2 KB` result, called real
  `run_command(pwd)`, consumed its repository-path result, and produced
  progressive visible final text with a truthful terminal.
- Direct and gateway tool calls and visible final bytes matched. Their first
  two reasoning payload hashes matched; the third reasoning payloads were
  fresh and distinct while visible output remained identical.
- Kept the row partial because the model omitted literal `STREAM` in the final
  requested marker. Kept the lower-budget diagnostic because direct stage two
  truthfully exhausted 512 tokens before emitting a tool call.
- Added no product fallback and made no new runtime code change.
- Retained:
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/qwen35-ollama-reasoning-history-tool-live.json`
  and
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/qwen35-ollama-reasoning-history-tool-live-1024.json`.

## 2026-07-23 - OpenPangu generation-key cache recovery

- Traced ordinary Chat/Responses cache misses to a global scheduler exclusion:
  `_cache_extra_keys` was treated as a reason to bypass all prefix/prompt-disk
  backends.
- Commit `568c1e105d3b` made the generation discriminator part of memory,
  prefix-trie, and disk keys. Current-head focused verification passed 152
  tests.
- Used the real isolated Electron Start/Stop controls for the exact affine
  `openPangu-2.0-Flash-JANG_3M` bundle.
- Proved exact L1 reuse, exact SSD reuse after restart, and changed-tail
  partial SSD reuse after another restart:
  - 1,946/1,947 exact prompt tokens from disk;
  - 3,543/3,565 shared-prefix tokens from disk plus 22-token new suffix.
- The same session already retained three visual turns and direct/gateway
  Chat, Responses, Anthropic, and Ollama agentic flows. Visual math rendered
  while raw API LaTeX/currency bytes remained unchanged.
- Kept architecture truth explicit: this is native typed prompt L2 with MLA,
  DSA indexer, rotating SWA, and convolution state. It is not generic paged,
  block-disk, or TurboQuant proof.
- Retained all current requests, SSE, health, argv, screenshots, source test
  log, and hashes under
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/openpangu-live/`.

## 2026-07-23 - M3 exact-once tool and malformed TeX checkpoint

- Reused the one current isolated Electron instance on CDP `9335` and loaded
  the exact MiniMax M3 bundle with the real Start button as PID `11370`.
- Reproduced the tool-loop defect: one requested `file_info` call executed
  three times because exactly-once retirement was tied to `reply exactly`.
- Added an independent named exactly-once contract and same-batch duplicate
  guard in `b482bec60`. The current live turn executed exactly once and
  continued to the truthful `5.2 KB` final.
- Inspected raw SQLite bytes for the following history turn. `$43` was
  prompt-required currency; the model emitted invalid `\(\(` nesting.
- Added shared completed/reasoning adjacent-delimiter normalization in
  `0592404d8`. Current DOM and screenshots show valid KaTeX, no fallback, and
  no raw TeX marker in the answer or expanded reasoning rail.
- Preserved four screenshots plus structured evidence under
  `docs/internal/release-gates/20260722_v1_6_17_consolidation/m3-tool-math-live/`.
- Remote tests passed: math 15, tool-auto-continue 23, combined tool/metrics
  26, and typecheck.
- Left direct/gateway API parity and the remaining M3 cache/media/release gates
  explicitly open.
- Ran current direct and Electron-gateway Chat/Responses/Anthropic/Ollama.
  All eight agentic routes separated reasoning, emitted one exact tool, consumed
  its real result, streamed a progressive final, and terminated truthfully.
- Stream/non-stream thinking-Off parity passed all four protocols at both
  endpoints with identical exact output and zero reasoning deltas.
- Raw math transport retained literal `$43`, delimiters, and the same
  model-owned `\×` on every route. Added `f34deae28` so the UI renders that
  invalid TeX spelling without fallback while preserving API bytes. Live DOM
  showed KaTeX in answer and reasoning with zero raw marker.
