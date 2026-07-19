# Agent proof log

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
