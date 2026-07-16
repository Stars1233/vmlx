# Post-tool finalization cross-model matrix — 2026-07-15

Scope: current-source Electron behavior after a real built-in tool result. This
matrix tracks the user-visible failure class first observed on Bonsai 1-bit:
repeated reasoning-only continuations, missing final content, speculative or
stale warnings, false token-rate metrics, and duplicate tool execution.

Source contract under test:

- `panel/src/main/ipc/chat.ts` permits at most one answer-only recovery after a
  completed tool result, removes tool schemas only for that recovery, resets
  cross-request timing, and preserves the rolling streamed token rate.
- `panel/src/shared/responsesWarnings.ts` removes only superseded
  empty-visible-answer diagnostics after visible recovery content exists. It
  preserves parser, schema, cache, tool-drop, and previous-response warnings.
- `vmlx_engine/api/tool_calling.py` forces explicitly named LFM2 tools past a
  placeholder-bearing native-template shortcut and binds scalar request values
  such as `file_info.path` into the native Python-call example.
- Focused current-source verification: 48/48 tests passed across
  `responses-warnings`, `tool-auto-continue`, and
  `tool-status-responsiveness`; TypeScript typecheck passed.

The shared source path is not sufficient to mark a model green. Each row needs
a current Electron tool call, persisted call/result record, final rendered
content, reasoning/status inspection, and timing/warning inspection.

| Model/family | Current Electron post-tool result | Tool count/result | Final content | Reasoning/status/TPS | Verdict / remaining work |
|---|---|---|---|---|---|
| Bonsai 27B 1-bit (`qwen3_5`, hybrid SSM) | Current-source row after the bounded-recovery fix | Exactly one `file_info`; one matching result | Exact `B1-UI-TOOL5-DONE` | Two phase-appropriate reasoning segments; measured `41.9 t/s`; no empty final | `VERIFIED-LIVE` for this row. Broader model/release gates remain separate. |
| Bonsai 27B ternary (`qwen3_5`, hybrid SSM) | Current-source fresh Electron row | Exactly one `file_info`; one matching result | Exact `BT-POSTTOOL1-DONE` | One reasoning segment; normal tool lifecycle; no warning; measured `31.3 t/s` | `VERIFIED-LIVE` for this row. Broader model/release gates remain separate. |
| HY3 JANG 2K MTP (`hy_v3`, qwen3 reasoning parser) | Current-source fresh Electron row | Exactly one `file_info`; one matching result | Exact `HY3-POSTTOOL1-DONE` | One reasoning segment; normal generating/calling/executing/result/processing/done lifecycle; `19.0 t/s`; no warning | `VERIFIED-LIVE` for this row. MTP net speedup remains unverified. |
| DSV4 Flash CRACK (`deepseek_v4`, native composite cache) | Pre-fix exact-final row retained a stale empty-answer warning. Post-fix row removed the warning. | Each row executed exactly one `file_info` with one matching result | Pre-fix exact `DSV4-POSTTOOL1-DONE`; post-fix model misspelled the marker as `DSV4-PPOSTOLL2-DONE` | Two reasoning phases; post-fix `warnings_json=null`; `18.3 t/s`; no repeated tool | `PARTIAL`: warning lifecycle is `VERIFIED-LIVE`; strict output fidelity remains red. |
| MiniMax-M3 Coder Small (`minimax_m3`) | Current genuine-tool Electron regression after M3 stream repair | Exactly one `file_info`; one matching result | Exact `MM3-TOOL-POSTFIX-DONE` | No completed zero-tool card; native M3 parsing retained | `VERIFIED-LIVE` for genuine-tool finalization. Exact image OCR remains open. |
| Zaya (`openpangu_v2`) | Current specialized terminal AppleScript row | One native `run_applescript`; terminal one-call policy | Visible post-tool completion | Repeated successful action prevented | `VERIFIED-LIVE` for the specialized AppleScript route, not generic `file_info` parity. |
| Laguna-M.1 (`laguna`) | Current-source fresh Electron row | Exactly one `file_info`; one matching result | Exact `LAG-POSTTOOL1-DONE` | Two phase-appropriate reasoning passages in one reasoning record; normal tool lifecycle; no warning; `16.0 t/s`; `3,612 paged+tq` cached tokens | `VERIFIED-LIVE` for this row. Decode speed remains a separate open gate. |
| LFM2.5 (`lfm2_moe`, hybrid SSM) | Broad-tools and Search-only pre-fix rows emitted malformed `path=': '` calls and repeated tools. Current-source post-fix Search-only and broad File/Search/Shell Electron rows passed. | Exactly one `file_info` with exact `panel/package.json`; one matching result | Exact `LFM-POSTTOOL5-DONE` on the broad row | One persisted reasoning record; normal tool lifecycle; no warning; `189.9 t/s`; `paged+ssm` cache detail | `VERIFIED-LIVE` for the broad row after request-bound LFM native example repair. |
| Qwen3.6 27B MXFP4 CRACK MTP (`qwen3_5`, hybrid SSM/VL) | Current-source broad File/Search/Shell Electron row | Exactly one `file_info`; one matching result | Exact `Q36-POSTTOOL1-DONE` | Two short phase-appropriate reasoning fragments; normal tool lifecycle; no warning; `22.6 t/s` | `VERIFIED-LIVE` for this row. Health also showed native MTP D3 and hybrid cache active, but this row does not prove MTP net speedup. |
| Gemma4 12B JANG 4M (`gemma4`, mixed SWA/full KV) | Current-source broad File/Search/Shell Electron row | Exactly one `file_info`; one matching result | Exact `G4-POSTTOOL1-DONE` | No reasoning fragments; normal tool lifecycle; no warning; `38.2 t/s`; 3,204-token memory-prefix hit | `VERIFIED-LIVE` for this row. Separate cache-default parity is red: UI config says prefix on but paged/prompt-L2/block-L2 off and health reports all three effective tiers off. |
| MiniMax-M2.7 Small JANGTQ (`minimax`, native reasoning) | Pre-fix broad row truncated `panel/package.json` to `panel`. Current-source post-fix broad row passed after slash-preserving native example repair. | Exactly one `file_info` with exact path; one matching result | Exact `MM27-POSTTOOL2-DONE` | Two phase-appropriate reasoning passages; normal tool lifecycle; no warning; `31.0 t/s`; `3,597 paged+tq` cached tokens | `VERIFIED-LIVE` for this row. Broader M2.7 reasoning-mode parity remains separate. |
| Step-3.7, Nemotron/Nemo, MiMo and other configured families | No current Electron row for this exact failure class | Untested | Untested | Untested | `UNTESTED`; do not infer parity from shared panel code or older API-only runs. |

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
`dsv4-posttool2-warning-cleared-strict-partial.png`.

Release boundary: `PARTIAL_NO_RELEASE`. This matrix does not clear Laguna
speed, HY3 measured MTP benefit, DSV4 exact-output fidelity, M3 exact image
OCR, remaining model-family post-tool rows, package integrity, signing,
notarization, updater feeds, or public release.
