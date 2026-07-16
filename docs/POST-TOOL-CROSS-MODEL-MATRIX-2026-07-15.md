# Post-tool finalization cross-model matrix — 2026-07-15

Scope: current-source Electron behavior after a real built-in tool result. This
matrix tracks the user-visible failure class first observed on Bonsai 1-bit:
repeated reasoning-only continuations, missing final content, speculative or
stale warnings, false token-rate metrics, and duplicate tool execution.

Source contract under test:

- `panel/src/main/ipc/chat.ts` permits at most one answer-only recovery after a
  completed tool result, removes tool schemas only for that recovery, resets
  cross-request timing, and preserves the rolling streamed token rate. An
  explicit `exactly once` + `after the tool result` + `reply exactly` contract
  now puts the planned post-tool turn on the direct-answer rail before a model
  can enter a second native tool prefix.
- `panel/src/shared/responsesWarnings.ts` removes only superseded
  empty-visible-answer diagnostics after visible recovery content exists. It
  preserves parser, schema, cache, tool-drop, and previous-response warnings.
- `vmlx_engine/api/tool_calling.py` forces explicitly named LFM2 tools past a
  placeholder-bearing native-template shortcut and binds scalar request values
  such as `file_info.path` into the native Python-call example.
- Focused current-source verification after the latest direct-answer and
  prompt-dedup changes: 97/97 tests passed across `request-builder`,
  `tool-auto-continue`, and `tool-status-responsiveness`; the 276-test
  session/settings slice and TypeScript typecheck also passed.

The shared source path is not sufficient to mark a model green. Each row needs
a current Electron tool call, persisted call/result record, final rendered
content, reasoning/status inspection, and timing/warning inspection.

| Model/family | Current Electron post-tool result | Tool count/result | Final content | Reasoning/status/TPS | Verdict / remaining work |
|---|---|---|---|---|---|
| Bonsai 27B 1-bit (`qwen3_5`, hybrid SSM) | Current reproduction exposed the same stuck UI: 6,316 generated tokens/46 tool markers behind 57 visible reasoning characters. A TQ-off run still generated 4,335 tokens and showed its first valid call at raw character 3,092. Current source adds request-scoped exact-once early stop without changing ordinary Qwen multi-call turns. | Two TQ-off plus six restored-Auto rows each executed exactly one `file_info`; one matching result | Exact unique marker on all eight current rows | Restored-Auto rows used 115-244 tokens and 4.2-7.0s. TQ-off A/B proves TQ was not the cause. One TQ-off row needed 1,195 tokens before its first call; process-restart SSM restore remains quarantined. | `VERIFIED-LIVE` for the explicit exactly-once tool/final contract; `PARTIAL` for general pre-call reasoning latency, multi-call patterns, and process-restart hybrid L2 reuse. |
| Bonsai 27B ternary (`qwen3_5`, hybrid SSM) | Current-source fresh Electron row | Exactly one `file_info`; one matching result | Exact `BT-POSTTOOL1-DONE` | One reasoning segment; normal tool lifecycle; no warning; measured `31.3 t/s` | `VERIFIED-LIVE` for this row. Broader model/release gates remain separate. |
| HY3 JANG 2K MTP (`hy_v3`, qwen3 reasoning parser) | Current-source fresh Electron row | Exactly one `file_info`; one matching result | Exact `HY3-POSTTOOL1-DONE` | One reasoning segment; normal generating/calling/executing/result/processing/done lifecycle; `19.0 t/s`; no warning | `VERIFIED-LIVE` for this row. MTP net speedup remains unverified. |
| DSV4 Flash CRACK (`deepseek_v4`, native composite cache) | Earlier warning-fix row misspelled its final. Current direct-answer cold/warm rows 1400/1403 both passed. | Exactly one `file_info`; one matching result on both current rows | Exact `DSV4-DIRECT-RAIL1-DONE` twice | One short reasoning segment, no warning, no repeated tool; identical warm row restored 619 tokens as `paged+dsv4` | `VERIFIED-LIVE` for the current explicit single-tool post-tool contract. Broader non-tool constrained-string fidelity remains separately open because older rows mutated markers. |
| MiniMax-M3 Coder Small (`minimax_m3`) | Current genuine-tool Electron regression after M3 stream repair | Exactly one `file_info`; one matching result | Exact `MM3-TOOL-POSTFIX-DONE` | No completed zero-tool card; native M3 parsing retained | `VERIFIED-LIVE` for genuine-tool finalization. Exact image OCR remains open. |
| MiniMax-M3 REAP32 d3 Coder (`minimax_m3`) | The model loaded through Sessions at 105.4 GiB active against a 107.52 GiB Metal ceiling. Two separate first Electron tool requests left blank assistant rows and coincided with full host reboots. | No completed tool call; both rows remained empty | Empty | Generic 99% pressure guard allowed the 98.0% baseline and output projection only clamped to 2,304 tokens; fixed prefill workspace still exhausted the machine. Current source adds a 3 GiB M3 prefill-headroom reject and blocks baseline forgiveness when the baseline itself is over threshold; 15 focused tests pass. | `FAIL-LIVE / PARTIAL-FIX`: distinct bundle/runtime safety failure. Do not retry unchanged. The new 503 path is not live-verified because a third 105 GiB load was deliberately avoided. |
| Zaya (`openpangu_v2`) | Current bundle is the AppleScript specialist despite its generic path. | One native `run_applescript` on the in-contract row. Generic `file_info` probes were filtered back to `run_applescript` and ended without a final. | Visible post-tool completion only on the in-contract AppleScript row. | Repeated successful action prevented; UI still misleadingly exposes unrelated File/Search categories. | `VERIFIED-LIVE` only for the specialized AppleScript contract. Generic tool parity is `OUT-OF-CONTRACT`; model-derived UI capability truth remains open. |
| Laguna-M.1 (`laguna`) | Current-source fresh Electron row | Exactly one `file_info`; one matching result | Exact `LAG-POSTTOOL1-DONE` | Two phase-appropriate reasoning passages in one reasoning record; normal tool lifecycle; no warning; `16.0 t/s`; `3,612 paged+tq` cached tokens | `VERIFIED-LIVE` for this row. Decode speed remains a separate open gate. |
| LFM2.5 (`lfm2_moe`, hybrid SSM) | Broad-tools and Search-only pre-fix rows emitted malformed `path=': '` calls and repeated tools. Current-source post-fix Search-only and broad File/Search/Shell Electron rows passed. | Exactly one `file_info` with exact `panel/package.json`; one matching result | Exact `LFM-POSTTOOL5-DONE` on the broad row | One persisted reasoning record; normal tool lifecycle; no warning; `189.9 t/s`; `paged+ssm` cache detail | `VERIFIED-LIVE` for the broad row after request-bound LFM native example repair. |
| Qwen3.6 27B MXFP4 CRACK MTP (`qwen3_5`, hybrid SSM/VL) | Current-source broad File/Search/Shell Electron row | Exactly one `file_info`; one matching result | Exact `Q36-POSTTOOL1-DONE` | Two short phase-appropriate reasoning fragments; normal tool lifecycle; no warning; `22.6 t/s` | `VERIFIED-LIVE` for this row. Health also showed native MTP D3 and hybrid cache active, but this row does not prove MTP net speedup. |
| Gemma4 12B JANG 4M (`gemma4`, mixed SWA/full KV) | Current-source cold/warm/restart Electron rows 1385/1388/1391 | Exactly one `file_info`; one matching result on every row | Exact `GEM4-L2-TOOL1-DONE` on every row | Cold row coherent; warm row restored 156/157 tokens from `memory`; post-restart row restored the same 156 tokens from `disk`. UI Reset Defaults visibly selected legacy Disk Cache, preview/argv emitted `--no-paged-cache --enable-disk-cache`, DB stored `1/0/1/0`, and health recorded 2 disk hits. | `VERIFIED-LIVE`: mixed-SWA correctly uses non-paged prompt L2, not incompatible generic paged blocks. Tool/final/cache persistence all passed. |
| MiniMax-M2.7 Small JANGTQ (`minimax`, native reasoning) | Pre-fix broad row truncated `panel/package.json` to `panel`. Current-source post-fix broad row passed after slash-preserving native example repair. | Exactly one `file_info` with exact path; one matching result | Exact `MM27-POSTTOOL2-DONE` | Two phase-appropriate reasoning passages; normal tool lifecycle; no warning; `31.0 t/s`; `3,597 paged+tq` cached tokens | `VERIFIED-LIVE` for this row. Broader M2.7 reasoning-mode parity remains separate. |
| Step-3.7 Flash JANG_K (`step3p7`, hybrid) | Current Electron control row 1355 | Exactly one `file_info`; one matching result | Exact `STEP-JANGK-POSTTOOL2-DONE` | Two coherent reasoning phases; `27.6 t/s`; `448 paged+mixed_swa` cached; no warning | `VERIFIED-LIVE` for the JANG_K control. |
| Step-3.7 Flash JANGTQ_K (`step3p7`, hybrid) | Historical row 1349 ran away for 1,854 reasoning tokens. Current source restores native Step attention when the installed generic P18 patch lacks post-reshape q/k norms and the head-wise gate. Electron row 1418 passed after tools and workspace were visibly enabled. | Exactly one `file_info` with exact `panel/package.json`; one real result | Exact `STEP-TQ-TOOL4-DONE` | One concise reasoning segment; normal tool lifecycle; no warning; `41.5 t/s`; 192 `paged+mixed_swa` cached. Electron Logs visibly record the native-attention correctness guard. | `VERIFIED-LIVE` for current coherence and this post-tool row. Rows 1409/1412 were invalid tool setups (`has_tools:false`), not parser failures. Broader Step VL/media and restart-L2 gates remain separate. |
| Nemotron Omni Nano JANGTQ (`nemotron-h`, hybrid SSM) | Two pre-fix rows duplicated the requested final marker. The broad agent prompt repeated its final-response directive and exact-output suppression required a colon. Rebuilt Electron row 1364 passed. | Exactly one `file_info`; one matching result | Exact single `NEMO-POSTTOOL3-DONE` | Two coherent reasoning phases; `paged+ssm+disk+tq`; no warning | `VERIFIED-LIVE` after prompt dedup/exact-output detection repair. |
| MiMo and other configured families | No current Electron row for this exact failure class | Untested | Untested | Untested | `UNTESTED`; do not infer parity from shared panel code or older API-only runs. |

Current evidence root:
`docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.
Relevant screenshots include `hy3-posttool1-pass.png`,
`bt-posttool1-pass.png`,
`laguna-posttool1-pass.png`,
`lfm-posttool1-fail.png`, `lfm-posttool4-pass.png`, and
`lfm-posttool5-broad-pass.png`,
`q36-posttool1-pass.png`,
`gemma4-posttool1-pass.png`,
`mm27-posttool1-wrong-path-fail.png`, `mm27-posttool2-pass.png`,
`dsv4-posttool1-stale-warning.png`, and
`dsv4-posttool2-warning-cleared-strict-partial.png`. New evidence includes
`step-jangk-posttool2-pass.png`, `step-posttool1-aborted-fail.png`,
`nemotron-posttool3-pass.png`, `bonsai-1bit-posttool9-warm-pass.png`,
`bonsai-1bit-posttool9-restart-exact-no-l2.png`,
`gemma-cli-preview-defaults.png`, `gemma-warm-memory-pass.png`, and
`gemma-l2-restart-disk-pass.png`. The combined Gemma DB/argv/health artifact is
`gemma4-cache-ui-db-argv-health-proof.json`. Current post-direct-answer rows for
Nemotron, MiniMax-M2.7, and DSV4 are in
`direct-answer-cross-model-current-rows.json`; DSV4 visual/health evidence is
`dsv4-direct-rail1-warm-pass.png` and `dsv4-direct-rail1-warm-health.json`.
Current Step JANGTQ_K evidence is in `step-jangtq-current-rows.json`,
`step-jangtq-health.json`, `step-jangtq-coherence1-pass.png`,
`step-jangtq-tool4-pass.png`, and `step-jangtq-attention-guard-log.png`.
REAP32 crash evidence is in `m3-reap32-host-reboot-fail.txt`,
`m3-reap32-second-host-reboot-fail.txt`, and
`m3-reap32-overlimit-health-before-guard.json`.

Release boundary: `PARTIAL_NO_RELEASE`. This matrix does not clear Laguna
speed, HY3 measured MTP benefit, DSV4 exact-output fidelity, M3 exact image
OCR, M3 REAP32 live-safe rejection, Step VL/media and restart-L2 behavior,
Bonsai hybrid L2 restart reuse, remaining
model-family post-tool rows, package integrity, signing,
notarization, updater feeds, or public release.
