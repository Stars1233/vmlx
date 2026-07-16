
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
- DSV4 pre-fix live row: one tool/result and exact final content, but the stale
  `visible answer is empty` warning remained. Post-fix live row: one
  tool/result, visible final content, `warnings_json=null`, and `18.3 t/s`.
  The model returned `DSV4-PPOSTOLL2-DONE` instead of the requested exact
  marker, so warning cleanup is live-verified while DSV4 strict fidelity stays
  `PARTIAL`.
- Screenshots and DB-derived evidence were preserved under
  `docs/internal/release-gates/20260715_140235_hy3_dsv4_mm3_exhaustive_electron/`.

Campaign remains `PARTIAL_NO_RELEASE`. Laguna and the other untested
model/parser families still need current Electron post-tool rows; all
previously listed model, settings, cache, media, package, signing, notary,
feed, and release gates remain in force.
