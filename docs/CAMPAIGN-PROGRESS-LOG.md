
## 2026-06-22 — STREAMING-FOR-CODING-HARNESSES audit (content/reasoning/tool deltas, all 4 API surfaces)
Audited /v1/chat/completions, /v1/responses, /api/chat (ollama), /v1/messages (anthropic) streaming for
Cline/Roo/Cursor/Aider/Continue/OpenCode/Zed/Claude-Code compat. Branch reconcile/1.5.68.
HEADLINE: tool-call ARGUMENTS never streamed incrementally — buffered to end-of-stream, parsed whole,
dumped in one chunk (Responses API fakes 16-char re-chunking). All 23 parsers have extract_tool_calls_streaming
= DEAD CODE (abstract_tool_parser.py:157, zero runtime callers). Reasoning IS genuinely streamed per-token
for every family that has a parser (good).
P0 functional breaks:
 1. anthropic_adapter.py:675 finalize() stop_reason never checks finish_reason=='tool_calls' -> end_turn ->
    Anthropic harnesses never execute the tool. (non-stream path server.py:9134 already correct — mirror it)
 2. anthropic thinking blocks emit no signature_delta -> Claude Code 400 on replay (stream+non-stream)
 3. chat stream: tool-buffering branch yields nothing unless include_usage (server.py:15092/15226) -> silent stall
 4. chat stream: flush_chunk NameError when remainder empty (server.py:15565, yield outside if remainder)
 5. ollama /api/chat: include_usage never set -> eval_count/prompt_eval_count/tok-s = 0 (stream only; nonstream ok)
P1: anthropic usage zero (force include_usage); responses no response.in_progress + reasoning attached to MESSAGE
    item not a reasoning item + output_index wrong (server.py ~15951/16501/17204); chat typed-error paths skip
    [DONE] (server.py:15321-15359); delta.reasoning not mirrored (models.py:1028 exclude=True); ollama durations
    missing + done_reason='length' leak (ollama_adapter.py:470).
P2: BaseThinkingReasoningParser partial-tag split leak (think_parser.py:143; gptoss/gemma4 already guard);
    tool-buffering driven by fixed _TOOL_CALL_MARKERS list (server.py:2877) misses unlisted/bare-JSON formats.
HIGH-LEVERAGE: force include_usage internally fixes #5+anthropic-usage+#3; wiring extract_tool_calls_streaming
fixes incremental args on all 4 surfaces. PLAN: (A) anthropic adapter P0s, (B) force include_usage, (C) chat
robustness #3/#4/[DONE], (D) incremental tool args (TDD, wire dead streaming parsers), (E) responses reasoning
item, (F) parser partial-tag holdback. Then LIVE-UI proof via Codex gpt-5.5 computer-use across gemma E2B/E4B/
12B/26B/31B + drag-drop media passthrough.

### 2026-06-22 LANDED (UNIT-VERIFIED, uncommitted, reconcile/1.5.68) — streaming robustness batch 1
anthropic_adapter.py: (1) finalize stop_reason now authoritative on finish_reason=='tool_calls' -> tool_use
  (P0, harnesses execute the tool); (2) _close_thinking() emits signature_delta (base64 placeholder dm1seA==)
  before every thinking content_block_stop (P0, Claude Code replay no longer 400); (3) mid-stream {"error"}
  chunk -> Anthropic event:error + finalize short-circuit (P2, no silent EOF). 4/4 new tests pass
  tests/test_anthropic_stream_fixes.py.
server.py: (4) ollama_chat + create_anthropic_message force chat_req.stream_options.include_usage=True
  (P0 ollama eval_count/tok-s no longer 0; P1 anthropic usage real); (5) chat tool-buffering heartbeat now
  UNCONDITIONAL (usage attached only if requested) -> no silent stall during tool gen (P0); (6) 3 typed-error
  handlers now emit [DONE]+return (P1, no hang / no post-error tool-extraction); (7) flush_chunk NameError
  fixed (yield moved inside if remainder) (P0 silent death).
REGRESSION: 515 passed in narrow suite; broad run 10 fails ALL PRE-EXISTING (proven via git-stash isolation:
  same fails at clean HEAD) — mcp_policy x4, mllm _extract_multimodal_messages 3-vs-4 audio unpack x4+1
  (REAL media-passthrough bug, relevant to drag-drop ask), vl_video source-scan x1, VG coord-strip x1
  (_strip_visual_grounding_markup_for_display leaves "(40,100)"). My edits add ZERO regressions.
NOT LIVE-PROVEN YET (per Eric rule unit!=proof) — queue live-UI via Codex gpt-5.5 computer-use.
NEXT: (D) incremental tool-call arg streaming (wire dead extract_tool_calls_streaming) = headline ask;
  (E) responses reasoning-item + in_progress; (F) parser partial-tag holdback; then LIVE PROOF gemma
  E2B/E4B/12B/26B/31B + drag-drop media; then fix pre-existing media-unpack bug.
## 2026-07-15 - MiniMax-M3 tools-enabled media/streaming continuation

LIVE CURRENT-SOURCE RESULT: tools-enabled M3 image generation no longer ends
blank or creates a completed `Used 0 tools` card after an invalid native XML
suffix. The final Responses parser now discards unsafe control markup and
late-arms a bounded tools-free visible-answer pass only when no schema-valid
call exists; effective boolean thinking is treated consistently with explicit
thinking modes. Focused regression: 3/3 passed.

Electron image transport and embeddings are functional (`551` image tokens);
the deterministic screenshot row read `panel/package.json` but confused the
marker digit `1` with `I`, so character-exact OCR is PARTIAL. Current-source
Electron video and no-reattach recall PASS: SMPTE colors, frames 0/100,
`TC01:00:00:00`, `TC01:00:03:09`, and exact
`VIDEO-FOLLOW=TC01:00:00:00|TC01:00:03:09`. The follow-up reused 128
`paged+disk` tokens and health retained typed `minimax_m3_msa_v1` state.

Fresh genuine-tool post-fix regression PASS: the Electron row executed exactly
one `file_info` call for `panel/package.json`, persisted one matching OAI tool
call/result pair, and finished exactly `MM3-TOOL-POSTFIX-DONE` with `4,271`
paged cached tokens. Non-stream parity PASS after visible app `Stop`/`Start`:
Responses and Chat with tools available but unused returned exact
`MM3-NONSTREAM-RESP-DONE` and `MM3-NONSTREAM-CHAT-DONE`.

M3 cache/settings parity PASS: the visible Session Settings defaults matched
the spawned CLI (`0.15` memory percent, paged KV on, block L2 on, 1000 blocks,
10GB block L2). Editing Cache Memory % to `12` and clicking the visible
`Save & Restart` relaunched the server with `--cache-memory-percent 0.12` and
health L1 max `5,922.96 MB`; restoring `15` relaunched with `0.15` and health
L1 max `7,623.44 MB`, with block L2 still enabled.

The health field `mtp.vl_runtime_available=false` is deliberately scoped to
native MTP+VL and is false for this no-MTP bundle; general image/video support
is advertised by `/v1/capabilities` and proven in Electron. M3 stays OPEN for
exact OCR. Release gate remains locked.
## 2026-07-15 - HY3 MTP current-source continuation

LIVE CURRENT-SOURCE RESULT: HY3 loads with native MTP depth 1 active, produces
coherent exact output, and reuses paged+TQ cached prefixes. Net MTP speedup is
not proven because health does not expose acceptance counters and reports
`speculative_decoding=not_configured`.

- UI selected `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP` and launched
  port `8010`.
- Health: `config_num_nextn_predict_layers=1`, `jang_mtp_layers=1`,
  index MTP layer count `1`, `mtp_tensor_count=42`, `runtime_active=true`,
  `effective_depth=1`, `runtime_scope=text`.
- Health also reports `speculative_decoding=not_configured`; no acceptance or
  speedup counters were exposed.
- Electron: exact `HY3-CURRENT-COHERENT-DONE`; exact recall `QUARTZ|719`;
  multi-turn metrics `626 paged+tq cached` then `672 paged+tq cached`.
- Health after runs: `hy_v3` `plain_kv_v1`, q4 stored-prefix TQ,
  prefix+paged+block-L2, scheduler hits `2`, tokens saved `1,298`, block L2
  `53` blocks / `3,002` tokens / `63` disk hits.

Release remains locked: MTP activation and output/cache pass, MTP speedup
effectiveness is unverified.

## 2026-07-15 - Laguna-M.1 current-source continuation

LIVE CURRENT-SOURCE RESULT: Laguna-M.1 loads and passes exact marker,
multi-turn recall, paged+TQ cache, and block-L2 proof; speed remains open.

- UI selected `/Volumes/EricsLLMDrive/jangq-ai/Laguna-M.1-JANG_2L` and
  launched port `8015`.
- Health: `laguna` `plain_kv_v1`, paged KV, generic TQ KV enabled for plain
  attention KV, stored-prefix q4, prefix+paged+block-L2 enabled.
- Electron: exact `LAGUNA-CURRENT-COHERENT-DONE`; exact recall `MARBLE|508`;
  multi-turn metrics `618 paged+tq cached` then `677 paged+tq cached`.
- Health after runs: scheduler hits `2`, tokens saved `1,295`, block L2 `14`
  blocks / `802` tokens / `63` disk hits.
- Speed remains open: current UI rows still run around `24 tok/s`, consistent
  with earlier dedicated Laguna bench under target.

Release remains locked: Laguna correctness/cache passes, speed does not.

## 2026-07-15 - DSV4 Flash CRACK current-source continuation

LIVE CURRENT-SOURCE RESULT: the configured DSV4 Flash CRACK session loads and
uses the native DSV4 composite cache correctly, but exact-marker fidelity is
not clean.

- UI selected `/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`
  (`dealignai/DeepSeek-V4-Flash-JANG-CRACK`) and launched port `8012`.
- Health: `deepseek_v4_v7` native composite cache with SWA local, CSA/HCA
  compressed pools, incomplete tail state, pool quant enabled, generic TQ KV
  forced off, prefix+paged+block-L2 enabled.
- Exact marker partial: requested `DSV4-CURRENT-COHERENT-DONE`, returned
  `DSV4-CURRENT-COHERENT-DENDONE`.
- Coherence/cache pass: arithmetic returned `45` with `346 paged+dsv4 cached`;
  memory recall returned `BASALT|314` with `413 paged+dsv4 cached`.
- Health after runs: scheduler hits `3`, tokens saved `1,142`, block L2
  `35` blocks / `7,644` tokens / `6` disk hits.

Release remains locked: DSV4 is cache/coherence-pass but exact-fidelity
partial, and other model gates remain open.

## 2026-07-15 - Bonsai 1-bit/ternary current-source continuation

LIVE CURRENT-SOURCE RESULT: Bonsai 1-bit and ternary both load in the visible
Electron session flow, produce coherent exact markers, preserve multi-turn
state across `paged+ssm` cached prefixes, and expose the expected hybrid SSM
cache/TurboQuant KV policies in `/health`.

1-bit evidence:

- UI selected `jangq-ai/Bonsai-27b-1bit-JANG` and launched port `8030`.
- Health: `JANG_AFFINE_1BIT`, actual bits `1.1128`, `qwen3_5`
  `hybrid_ssm_v1`, live attention TQ KV, native SSM companion state,
  prefix+paged+block-L2 enabled.
- Electron: exact `B1-CURRENT-COHERENT-DONE`; exact recall `CEDAR-B1|9417`;
  multi-turn metrics `paged+ssm`.
- Health after runs: block L2 `13` blocks / `687` tokens / `27` hits; SSM
  companion disk `16` entries / `2,687` tokens.
- Responses API parser: non-stream and streaming thinking-disabled probes emit
  `file_info({"path":"panel/package.json"})`. The old UI row was invalid because
  its chat had builtin/file/search tools disabled. A fresh default-thinking row
  persisted exactly one `file_info`, one real result, and exact
  `B1-UI-TOOL5-DONE`.
- A valid pre-fix UI row exposed four reasoning-only post-tool retries, empty
  content, five fragment cards, and bogus `565.9 t/s`. The panel now performs
  one answer-only recovery, reports an error if content is still empty, resets
  timing across HTTP streams, and keeps the measured rolling rate. Live proof:
  two phase-appropriate reasoning segments, exact final content, `41.9 t/s`.

Ternary evidence:

- UI selected `jangq-ai/Bonsai-27b-Ternary-JANG` and launched port `8020`.
- Health: `JANG_AFFINE_TERNARY_2BIT`, actual bits `2.0959`, same hybrid
  SSM/TQ/paged/block-L2 policy.
- Electron: exact `BT-CURRENT-COHERENT-DONE`; exact recall `SPRUCE-BT|6824`;
  multi-turn metrics `247 paged+ssm cached`.
- Health after runs: block L2 `8` blocks / `460` tokens / `24` hits; SSM
  companion disk `7` entries / `1,815` tokens.
- Responses streaming parser emitted function-call argument deltas and final
  `file_info({"path":"panel/package.json"})` with no warnings.

Release remains locked: these rows close Bonsai basic quant/cache/parser proof,
not DSV4/Laguna/HY3/full-release proof.

## 2026-07-15 - Gateway/settings/localization live Electron continuation

- Gateway port collision truthfulness PASS: installed vMLX owned wildcard
  `8080`; dev gateway bound `*:8081`, and the UI plus DB immediately displayed
  `8081` rather than the requested stale port.
- LAN PASS: displayed `192.168.1.110:8081` answered `/health` from both Macs.
  The app was restored to localhost `127.0.0.1:8080` after the proof.
- Single-model PASS: Bonsai -> Hy3 -> Bonsai routing unloaded the prior PID
  before loading the target. Only one engine stayed resident, and Responses
  streaming remained incremental through the swap.
- HY3 settings enforcement PASS: visible Max Blocks save/restart changed UI,
  DB, argv, and health `1000 -> 900`, then restored all to `1000`. Expanded UI
  also matched prefix/paged/block-L2, 15%, block size 64, 10GB, and KV auto.
- Localization PASS: zh, ko, ja, es, and en controls were visibly exercised;
  each shipped catalog has the same 978 keys. English was restored.
- Focused panel verification: 777 passed, 3 skipped across gateway, settings,
  i18n, tool status/continuation, chat, and session-port suites; TypeScript
  typecheck clean. Focused Python cache/server/reasoning verification: 262
  passed, 3 deliberately deselected.

Campaign remains `PARTIAL_NO_RELEASE`. Open red/partial rows: Laguna decode
speed around 24 tok/s, HY3 MTP net speedup without acceptance counters, DSV4
exact-marker mutation, and M3 character-exact OCR. No tag, signing,
notarization, updater-feed mutation, or public release was performed.

## 2026-07-15 - Cross-model post-tool finalization continuation

The Bonsai repeated-reasoning/empty-final/TPS defect is now tracked as a
shared Electron gate rather than a Bonsai-only issue. The current matrix is
`docs/POST-TOOL-CROSS-MODEL-MATRIX-2026-07-15.md` and deliberately leaves
families without a current visible Electron row as `PARTIAL` or `UNTESTED`.

- Current source adds narrow warning reconciliation after a successful
  answer-only recovery. It removes only superseded current-response
  empty-visible-answer diagnostics and preserves parser, schema, cache,
  tool-drop, and previous-response warnings.
- Focused source verification passed: 48/48 tests across responses warnings,
  tool continuation, and tool status; TypeScript typecheck is clean.
- HY3 live Electron row: exactly one `file_info`, one matching result, exact
  `HY3-POSTTOOL1-DONE`, one reasoning segment, normal tool lifecycle, no
  warning, `19.0 t/s`, and `3,626 paged+tq` cached tokens.
- Bonsai ternary live Electron row: exactly one `file_info`, one matching
  result, exact `BT-POSTTOOL1-DONE`, one reasoning segment, no warning, and a
  measured `31.3 t/s`. This replaces the prior API-only post-tool evidence.
- Laguna live Electron row: exactly one `file_info`, one matching result,
  exact `LAG-POSTTOOL1-DONE`, no warning, `16.0 t/s`, and `3,612 paged+tq`
  cached tokens. Laguna decode-speed performance remains a separate open row.
- LFM2.5 initially failed both broad-tools and Search-only Electron rows: its
  placeholder-bearing native template produced malformed `path=': '` and up
  to three calls. Direct single-schema Responses correctly parsed the exact
  path, isolating prompt construction from the parser. Current source forces
  explicitly named LFM tools through a request-bound native example and binds
  scalar parameters instead of `VALUE_HERE`. Eight focused LFM prompt/parser
  tests passed. After visible Stop/Start, Electron row 1322 persisted exactly
  one `file_info({"path":"panel/package.json"})`, one matching result, exact
  `LFM-POSTTOOL4-DONE`, no warning, and `paged+ssm+disk` cache detail.
  A second post-fix row with broad File I/O, Search, and Shell all visibly
  enabled also passed: row 1325 made exactly one correct call/result and exact
  `LFM-POSTTOOL5-DONE`, with no warning.
- Qwen3.6 27B MXFP4 CRACK MTP broad File/Search/Shell Electron row 1328
  persisted exactly one `file_info({"path":"panel/package.json"})`, one
  result, exact `Q36-POSTTOOL1-DONE`, two short reasoning fragments, no
  warning, and `22.6 t/s`. Post-run health showed native MTP D3 active,
  hybrid SSM cache, block-L2/SSM companion stores, and live attention TQ
  telemetry. This is post-tool/cache telemetry, not measured MTP speedup proof.
- Gemma4 12B broad File/Search/Shell Electron row 1331 persisted exactly one
  `file_info({"path":"panel/package.json"})`, one result, exact
  `G4-POSTTOOL1-DONE`, no reasoning, no warning, `38.2 t/s`, and a 3,204-token
  memory-prefix hit. A separate cache-default red row was opened: the running
  session config has prefix enabled but paged, prompt L2, and block L2 off;
  argv contains `--no-paged-cache`; health reports effective native
  prefix/paged/block-L2 all false and zero L2 tokens.
- MiniMax-M2.7 broad Electron row 1334 exposed a semantic tool failure hidden
  behind an exact final marker: the native fallback truncated
  `panel/package.json` to `panel`. Source tracing found the generic scalar
  regex excluded `/`. A path-specific slash-preserving extractor and focused
  MiniMax regression were added; two fallback tests, five LFM regressions, and
  19 MiniMax parser tests passed. After visible Stop/Start, row 1337 persisted
  exactly one `file_info({"path":"panel/package.json"})`, one result, exact
  `MM27-POSTTOOL2-DONE`, two reasoning passages, no warning, `31.0 t/s`, and
  `3,597 paged+tq` cached tokens.
- DSV4 pre-fix live row: one tool/result and exact final content, but the stale
  `visible answer is empty` warning remained. Post-fix live row: one
  tool/result, visible final content, `warnings_json=null`, and `18.3 t/s`.
  The model returned `DSV4-PPOSTOLL2-DONE` instead of the requested exact
  marker, so warning cleanup is live-verified while DSV4 strict fidelity stays
  `PARTIAL`.
- Screenshots and DB-derived evidence were preserved under
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.

Campaign remains `PARTIAL_NO_RELEASE`. The remaining untested model/parser
families still need current Electron post-tool rows; all
previously listed model, settings, cache, media, package, signing, notary,
feed, and release gates remain in force.

## 2026-07-15 - Step, Nemotron, Bonsai and cache/settings continuation

- Fresh Electron session creation for Nemotron and Step JANG_K visually showed
  prefix on, paged on, block-L2 on, and legacy disk off; DB, argv, and health
  matched. The stale Step JANGTQ_K impossible `1/1/1` cache tuple was normalized
  to `1/0/1` on a rebuilt Electron constructor run.
- Manual single-model mode is now enforced by `startSession`, not only gateway
  routing. Live Zaya -> Step and Nemotron -> Bonsai swaps stopped the prior
  process first and kept exactly one engine during load.
- Step JANGTQ_K is red: 1,854 runaway reasoning tokens, no valid final. Step
  JANG_K control passed one exact tool/final row with `paged+mixed_swa` reuse.
- Nemotron initially duplicated its exact final twice. Source tracing found a
  duplicated final-response rule and a colon-only exact-output detector.
  After prompt repair, rebuilt row 1364 made one tool call and one exact final
  with `paged+ssm+disk+tq` detail.
- The current Bonsai exact-tool path no longer repeats reasoning or hangs in a
  second tool prefix. Rebuilt rows 1373/1376/1379 made one tool call and one
  exact final; identical warm row 1379 restored 158/159 tokens as
  `paged+ssm`. A process restart remained exact but restored zero tokens, so
  hybrid SSM L2 restart reuse remains PARTIAL under the current quarantine.
- Focused verification passed: 97 prompt/tool tests, 276 session/settings
  tests, 33 Zaya prompt tests, and TypeScript typecheck. Full-suite and release
  gates remain pending.
- Evidence root:
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.
- Campaign remains `PARTIAL_NO_RELEASE`; no release-adjacent command ran.

## 2026-07-15 - Gemma4 mixed-SWA cache-default and restart-L2 proof

- Source trace against the real 12B bundle found 40 `sliding_attention` and
  eight `full_attention` layers. The registry and CLI intentionally reject the
  generic paged serializer for this mixed-SWA layout; the compatible default is
  memory-aware prefix cache plus legacy prompt-disk L2.
- The existing volume-backed session had both disk tiers off. In the real
  Electron Session Settings screen, `Reset all parameters to defaults`
  visibly turned legacy Disk Cache on. CLI Preview showed
  `--no-paged-cache --enable-disk-cache`; Save persisted prefix/paged/legacy/
  block as `1/0/1/0`.
- Starting the session through Electron enforced single-model mode, unloaded
  Bonsai, and launched Gemma PID 44635 with the previewed flags. Health exposed
  the prompt disk tier and `mixed_swa_kv_v1` with 4-bit storage-boundary KV.
- Electron row 1385 cold-passed exactly one `file_info` and exact
  `GEM4-L2-TOOL1-DONE`. Identical fresh-chat row 1388 restored 156/157 prompt
  tokens from `memory` and stayed exact.
- `Save & Restart` replaced the engine with PID 44896 without clearing L2.
  The new process discovered two prompt-L2 entries (2,994 tokens). Identical
  row 1391 restored 156/157 tokens with `cacheDetail:"disk"`, executed exactly
  one tool, and returned the exact final. Health recorded two disk hits.
- Evidence is under the current release-gate root, especially
  `gemma4-cache-ui-db-argv-health-proof.json`,
  `gemma-cli-preview-defaults.png`, `gemma-warm-memory-pass.png`, and
  `gemma-l2-restart-disk-pass.png`.

Gemma4 cache-default parity is `VERIFIED-LIVE`. Campaign remains
`PARTIAL_NO_RELEASE` because Bonsai hybrid restart L2, Step JANGTQ_K,
remaining model/parser/media rows, full tests, package integrity, signing,
notary, feeds, and public release are still open.

## 2026-07-15 - Post-direct-answer cross-model live rerun

The Bonsai repair was rerun through three additional parser/runtime families
on the rebuilt Electron app, rather than inferred from shared TypeScript:

- Nemotron row 1394: exactly one `file_info`, one result, exact
  `NEMO-DIRECT-RAIL1-DONE`, one reasoning segment, no warning, and
  `paged+ssm+disk+tq` reuse.
- MiniMax-M2.7 row 1397: exactly one slash-preserving
  `file_info({"path":"panel/package.json"})`, one result, exact
  `MM27-DIRECT-RAIL1-DONE`, one reasoning segment, no warning, and
  `paged+disk+tq` reuse.
- DSV4 rows 1400 and 1403: both produced exactly one tool/result and byte-exact
  `DSV4-DIRECT-RAIL1-DONE`, with one short reasoning segment and no warning.
  The identical warm row restored 619 tokens as `paged+dsv4`.

This closes the Bonsai-style repeated-reasoning/missing-final symptom for these
current explicit single-tool contracts. It does not erase older evidence that
DSV4 can mutate constrained strings in other prompts, so its broader fidelity
row remains partial. Evidence: `direct-answer-cross-model-current-rows.json`,
`dsv4-direct-rail1-warm-pass.png`, and
`dsv4-direct-rail1-warm-health.json` under the active release-gate root.

## 2026-07-15 - Step-3.7 JANGTQ_K attention and Electron tool recovery

Historical Step JANGTQ_K soup is now traced to a current runtime regression,
not reassigned to the pre-proven bundle. The installed/general JANGTQ P18 QKV
patch replaced native `Step3p5Attention.__call__` but normalized q/k before the
head reshape and omitted Step's head-wise `g_proj` sigmoid gate. The corrected
implementation existed in historical JANG commit `44a3c55` but is not in the
current JANG main lineage.

- Current vMLX source adds a version-tolerant Step-only guard. It inspects the
  installed P18 implementation and keeps it only when both post-reshape q/k
  norm and head-wise gate semantics are present; otherwise it restores the
  captured native Step attention after hydration. Routed TurboQuant expert
  paths remain enabled.
- Focused verification passed 129 tests across `test_jang_loader.py` and the
  Step VLM/runtime/crash-audit suites.
- The real Electron log visibly reports that the unsafe P18 patch was detected
  and native Step attention was restored. Row 1406 then returned exact `4`
  with coherent reasoning instead of soup.
- Two subsequent narrated/no-tool probes were invalid setups: current request
  diagnostics showed `has_tools:false`. After the Chat Settings tool toggle
  was visibly enabled and the working directory was set to the repo, row 1418
  executed exactly one `file_info({"path":"panel/package.json"})`, received the
  real 5.2 KB file result, emitted one concise reasoning segment, and returned
  exact `STEP-TQ-TOOL4-DONE`. Metrics were 41.5 t/s with 192
  `paged+mixed_swa` cached tokens.
- Evidence: `step-jangtq-current-rows.json`, `step-jangtq-health.json`,
  `step-jangtq-coherence1-pass.png`, `step-jangtq-tool4-pass.png`, and
  `step-jangtq-attention-guard-log.png` under the active release-gate root.

Step JANGTQ_K coherence and this explicit post-tool contract are
`VERIFIED-LIVE`; Step VL/media and restart-L2 behavior remain unverified.
Campaign status stays `PARTIAL_NO_RELEASE`. MiMo and other configured parser
families still need their own current Electron rows, and no package, signing,
notary, feed, tag, or public-release action is cleared by this result.

## 2026-07-15 - MiniMax-M3 REAP32 first-prefill host reboot

The configured `jangq-ai/MiniMax-M3-REAP32-d3-Coder` row is a distinct live
safety failure, not a pass inferred from MiniMax-M3 Coder Small.

- Sessions UI single-model handoff stopped Step and loaded only REAP32. Health
  reported `107939.9 active_mb`, correctly converted to 105.4 GiB, while the
  M5 Max MLX ceiling is 107.52 GiB. MTP metadata declared one layer but the
  bundle index had zero MTP tensors; `vl_runtime_available=false` was also
  truthful despite present vision weights.
- The first Electron `file_info` turn left a blank assistant row and the host
  rebooted. The stale `running` session was reconciled to inactive when the
  same dev profile restarted.
- A controlled second load reproduced the same 105.4 GiB baseline. The
  generic request guard allowed 98.0% occupancy under its 99% threshold and
  projected output only clamped to 2,304 tokens. The second first request again
  left a blank row and rebooted the host before UI Stop completed.
- Current source now permits loaded-model baseline forgiveness only when the
  baseline itself is below the configured threshold. It also rejects any
  MiniMax-M3 request before the handler when less than 3 GiB of Metal headroom
  remains. The error explicitly identifies first-prefill risk and the opt-out;
  it does not synthesize output or silently change sampling.
- Ten focused Metal-guard tests and five memory/audit checks pass. A third
  REAP32 live load was deliberately not attempted, so the new 503 path remains
  source-verified but live-unverified.
- Evidence: `m3-reap32-host-reboot-fail.txt`,
  `m3-reap32-second-host-reboot-fail.txt`, and
  `m3-reap32-overlimit-health-before-guard.json` under the active gate root.

Verdict: `FAIL-LIVE / PARTIAL-FIX`. Release lock remains active. The host must
not be exposed to another unchanged REAP32 first prefill merely to obtain a
green screenshot.

## 2026-07-16 - Bonsai exact-once Qwen stream bound

The user-reported repeated Bonsai reasoning/tool state was reproduced through
the running Electron dev app, then separated from cache quantization with a
visible settings and process-argv A/B.

- Pre-fix Responses/Electron traces generated 2,422, 4,335, and 6,316 tokens.
  The 6,316-token row contained 24,443 raw characters and 46 `<tool_call>`
  markers while the UI exposed only 57 reasoning characters and duplicate
  speculative tool status. The final malformed call was correctly rejected.
- Selecting `None (disable stored quant + live TQ-KV)` in Server Settings
  produced a new PID with literal `--kv-cache-quantization none`; health showed
  zero TQ objects/encoded layers. The same failure still took 4,335 tokens.
  Its first fully canonical, schema-valid `file_info` call occurred at raw
  character 3,092, followed by roughly 13.5K characters of repetition. This
  falsifies TurboQuant and stale hybrid-cache reuse as the owning cause.
- Qwen's parser-wide multi-call behavior remains unchanged. The server now
  enables its complete-call detector only when the latest user request names
  exactly one exposed tool and explicitly requires it exactly once. The first
  closed call must contain every required non-empty argument before the engine
  request can be stopped; the existing eight-chunk grace window remains.
- The 423-test parser/server/reasoning/tool-format suite and all 578 engine
  audit checks pass. The focused Responses regression covers schema
  validation, truncation, engine abort, one emitted function item, no
  post-call leak, and preservation of ordinary Qwen multi-call turns. The
  release manifest is 319/320: one freshness row is blocked by an unrelated
  pre-existing deletion of
  `build/current-regression-suite-after-pr-intake-matrix-refresh-20260609.json`.
- Live TQ-off Electron rows completed in 24.9s/1,195 tokens and 7.0s/279
  tokens. After restoring the visible KV setting to Auto, the process argv no
  longer carried the explicit `none` flag, health again showed TQ on the 16
  attention layers with 48 native SSM companions, and six fresh Electron rows
  completed in 4.2-7.0s/115-244 tokens. Every row persisted exactly one
  `file_info({"path":"panel/package.json"})`, one result, and its exact final
  marker.

Verdict: `VERIFIED-LIVE` for Bonsai's explicit exact-once tool/final contract;
`PARTIAL` for general reasoning latency and other Qwen multi-call patterns.
This does not close Bonsai VL (`vl_runtime_available=false`), hybrid SSM
process-restart restore quarantine, cross-model parser rows, or the release
gate. Campaign remains `PARTIAL_NO_RELEASE`.
