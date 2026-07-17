# vMLX 1.6.11 release closeout matrix — 2026-07-16

Status: `PARTIAL_NO_RELEASE`.

This is the current additive closeout view over `docs/internal/ISSUE-LEDGER.md`,
`.agents/STATUS.md`, the July 15–16 live proof directories, the shared wiki
production gate, and the current branch. Older contradictory rows remain in
their original ledgers for provenance; the newest source-plus-live row wins and
superseded conclusions are called out here.

## Release truth

- Working branch: `reconcile/1.5.68`; current scoped code head `f993e36b8` and
  prior matrix/evidence head `13206d490`;
  typed-settings,
  non-MTP architecture-hint, paged resident-accounting, typed hybrid-companion
  ownership, and v8 cache-namespace repairs plus their focused tests are pushed
  to the closeout branch described below.
- Push target: `origin/codex/live-electron-gates-20260715`.
- At scoped code head `d9cef0b0c`, the branch is 111 commits ahead of
  `origin/main` and zero behind. Matrix-only commits may follow the scoped code
  head.
- Source versions are `1.6.11` in `pyproject.toml`,
  `vmlx_engine/__init__.py`, and `panel/package.json`.
- Public GitHub app release, PyPI, and `mlxstudio/latest.json` are all 1.6.10.
- The Laguna parser-default migration is committed and pushed as `7b45676ce`.
  Current Electron main launched PID 32806 with `--tool-call-parser glm47`,
  and the session is stamped migration version 1.
- No package, version bump, tag, signing, notarization, feed update, PyPI
  upload, or GitHub release is allowed until the red rows below close.

## Current blockers

| Area | Status | Current evidence | Required closeout |
|---|---|---|---|
| Laguna parser migration | PASS-LIVE / COMMITTED | Electron UI/DB/argv migrated to `glm47`; rows 1992/1995 each executed one `file_info` and exact final text; 94 parser/migration tests and panel typecheck passed | Keep as a release regression row |
| Laguna reasoning | PARTIAL-LIVE | Cold row 1998 was exact. Auto's old uncalibrated TQ3 warm row 2001 restored 3,545 tokens, looped incoherently, and was stopped after 3,076 generated tokens. None rows 2004/2007/2010 and corrected Auto TQ8 rows 2013/2016/2019 were exact. Restart row 2022 stayed coherent but made an unsolicited `ask_user` call before exact post-skip completion | Repeat restart/disk row without unsolicited tool; long reasoning soak and strict byte-format closeout; no forced sampler or synthetic think tags |
| Laguna cache/perf | PASS-LIVE correctness / PARTIAL latency | UI None left prefix/paged/L2 on and produced exact 3,549/3,612-token paged hits. Auto now uses uncalibrated TQ8 with codec-config namespace invalidation; exact 3,550/3,614-token native hits and a coherent 3,550-token disk restore were observed | TQ8 reconstruction costs 3.6-4.8s and warm TTFT ~5.1s versus 1.2-1.5s None; optimize/accept with measured release budget, plus long-context/eviction proof |
| Bonsai hybrid restart | PASS-LIVE exact boundary / PARTIAL partial prefix | Two independent 1-bit PID replacements restored 160/168 tokens and ternary restored 153 tokens as `paged+ssm+disk`, with native-TQ plus SSM disk hits, ~0.10s reconstruction, one tool, and exact finals. A longer 64-token KV prefix without a companion safely full-prefilled and wrote the missing checkpoint | Repeat partial-prefix repair as a measured hit, then long-context/eviction |
| Bonsai current-HEAD regression | PASS-LIVE agent correctness + restart + eviction / PARTIAL soak | Current Electron PID 83540 launched 1-bit with `--tool-call-parser qwen`, `--reasoning-parser qwen3`, paged cache, Block Disk L2, and Auto. Rows 2286/2289 prove same-chat two-turn exact one-tool/final behavior. Fresh exact old repro row 2292 produced one real `file_info(panel/package.json)` and exact `B1-UI-TOOL3-DONE`; screenshot `bonsai-b1-ui-tool3-current-pass.png` captured the live UI. Health after the rows shows `tq_native_enabled=true`, 17 native-TQ L2 writes, four block-disk hits, SSM companion disk stores, and only one local serve process for Bonsai. UI restart produced PID 84219; rows 2294/2298 again finalized with one real tool and exact visible markers. The changed-prefix row correctly recorded `hybrid_kv_without_ssm_hits=1` / `reason=no_ssm_companion_state`; source trace `mllm_batch_generator.py:5814-5863` releases that KV-only candidate and full-prefills rather than using an unsafe hybrid prefix. A second UI restart to PID 84984 replayed the exact row-2298 prompt; row 2301 restored 154 tokens as `paged+ssm+disk`, `disk_hit=true`, `reconstructed=true`, `dequantized=true`, `tq_native_hits=3`, and no `hybrid_kv_without_ssm`, while again producing one real tool and exact final. UI-applied four-block PID 85595 (`--max-cache-blocks 4`) rows 2304/2306 each executed one real tool and exact finals; health showed `max_blocks=4`, `free_blocks=0`, `l1_evictions=9`, `l2_block_tokens_on_disk=21303`, `l2_ssm_tokens_on_disk=18482`, `tq_native_enabled=true`, and SSM companion evictions. UI restored 1,000 blocks on PID 85909. Earlier 1-bit loop row 2028 remains retained. Current repro row 2349 now proves the agent loop does finish: one real `file_info(panel/package.json)`, exact `B1-UI-TOOL3-DONE`, `cachedTokens=158`, `cacheDetail=paged+ssm+disk`, TQ8 native hits, and SSM companion disk hits. It also retains the reliability issue: 4,222 chars / 1,101 tokens of native reasoning before the tool call, visible as a large Reasoning panel. | Long-context/output soak without forcing sampler defaults; keep long pre-tool reasoning under UI/API observation and determine whether a non-fake request policy can reduce it without disabling reasoning globally |
| Bonsai native Qwen tool stream | PASS-LIVE scoped / PARTIAL broader catalog | Commit `f993e36b8` makes Qwen and Step native-template classification mutually exclusive, preserves Qwen's shipped `<tools>` scaffold for ordinary auto-tool turns, and recognizes the live exact-once wording `after its result`. Real Electron row 54 executed one `file_info(panel/package.json)`, persisted one matching call/result, streamed 61 reasoning plus 13 content events, and returned exact `B1-ELECTRON-TOOL-TEMPLATE2-DONE` in 7.1s. An identical fresh chat restored 157/158 prompt tokens as `paged+ssm`, reduced TTFT from 0.46s to 0.20s, and again completed one tool with 59/13 progressive events. A visible Stop/Load replaced PID 34438 with PID 34884; the identical request then restored 157/158 as `paged+ssm+disk`, streamed 55/13 events, executed one tool, and exact-finaled. Raw Responses emitted 117 reasoning deltas, one function item, then 14 content deltas and exact completed text after the function output. | Cover named `run_command`, multi-tool chains, and other Qwen artifacts before broad release classification |
| Bonsai media-keyed hybrid cache | PASS-LIVE image + video-A cache / PARTIAL cross-video exactness+catalog | The real 1-bit `Qwen3_5ForConditionalGeneration` artifact advertises vision config plus image/video tokens. Image A cold returned exact `Q27-EXACTONCE-ELECTRON2-DONE` in 14 progressive paints; identical A restored 4,963/4,964 tokens as `paged+ssm` and reduced TTFT from 21.33s to 0.69s. Same-shape marker image B was a zero-cache miss and returned exact `B1-MEDIA-B-DONE`; return-A restored A at 0.66s. PID 34884→36409 then restored A as `paged+ssm+disk` at 1.64s with 78 native-TQ disk hits and one SSM disk hit. Video A cold returned exact `FRAME START 2468` / `FRAME END 9753` in 15 progressive paints; identical A restored 2,933/2,934 `paged+ssm` tokens and reduced TTFT from 8.19s to 0.66s. Alternate videos were zero-cache misses and never leaked A, but Bonsai abbreviated their visible digits (`ALT START 1` / `ALT END 86`, then `START` / `END`), retained as OCR quality misses. Return-A was exact at 0.64s. Visible PID 36409→37342 restored A as `paged+ssm+disk` at 1.72s with 46 native-TQ disk hits and one SSM hit. Real `curl -N` Responses requests reused the Electron image and video prefixes and emitted 14/15 timed `response.output_text.delta` events plus one completed terminal each. Current focused media/scheduler/SSM tests pass 219 with six intentional skips. | Improve/characterize alternate-video OCR exactness without fake postprocessing; cover other advertised Qwen3.5/Bonsai artifacts. Auto-thinking image row remains retained after 1,024 reasoning paints and a truncated marker. |
| Mistral Medium 3.5 | DEFERRED BY USER | Prior text load/cache observations are retained, but the user explicitly excluded further Mistral MXFP4 testing from this closeout run | Do not spend this campaign on Mistral MXFP4; it is not used to claim a current release pass |
| DSV4 CRACK | PASS-LIVE eager/cache/settings/eviction/stream/tool-scope tiers / PARTIAL strict long quality+perf | Commit `1e15c94bd` makes DSV4 start materialize stored parameters before a first prompt; live health before any request reported `last_request_time=null` and about 99.7 GB active model memory. Native composite cache and DSML separation pass, including 3,244/3,245-token RAM and `paged+dsv4+disk` restores. Commit `012c1fe90` aligns broad-catalog fallback validation with native DSV4 tool scoping, preserves explicit path binding, and maps explicit no-tool turns to standard `tool_choice:none`. Fresh Electron row 153 executed one real `file_info(panel/package.json)`; rows 156/159 continued without tools; row 162 painted reasoning and visible content progressively across 1,595 DOM mutations. Raw Responses with all 33 tools emitted ten progressive content deltas and completed once. Strict format remains red: row 159 used two sentences and row 162 ended `D4 STREAM COMP complete.` instead of the requested marker. | Diagnose/retest constrained exact-output and long factual reliability without sampler coercion; quiet speed; exact JANGTQ bundle only if locally available |
| MiniMax-M3 | PASS-LIVE scoped exact OCR/tools/video / PARTIAL ambiguous glyphs+media cache+REAP | Visible single-model switch unloaded Bonsai PID 37342 and started only M3 PID 37842 on 8017. The original digit/dash marker retained the known quality miss (`B1`→`B81`, hyphens→en dashes) while streaming ten clean content paints. A visually inspected all-letter fixture then returned byte-exact `MAGNOLIA CACHE DONE` twice in seven progressive paints with Thinking Off and no tool/reasoning events; real `curl -N` Responses independently emitted seven output-text deltas, matching done text, and one completed terminal. Both identical Electron requests reported zero cached tokens. This matches source ownership: active M3 VL forces every image/video request through full prompt with `cache_to_use=None` because all image-token positions and pixel grids must be present in one vision-splice forward (`scheduler.py:6112-6129`). Text MSA prefix/paged/L2 proof does not imply media reuse. | Design and prove a family-owned cached-vision boundary before enabling M3 media prefix reuse; improve/characterize digit/dash OCR without output rewriting; live REAP32 503 guard only if it can be exercised without host-reboot risk |
| openPangu | PASS-LIVE scoped / PARTIAL long-context+protocol | Source policy `_apply_openpangu_cache_policy` forces paged/block/TQ off and preserves typed MLA KV, DSA indexer, rotating-SWA metadata, causal-conv state, 128 sinks, and mHC runtime. Electron-loaded 3M PID 86212/86842/87268 launched with `--no-paged-cache --enable-disk-cache`, no KV quantization, and no block L2; Bonsai was unloaded by the single-model swap. Rows 2310/2313 prove same-chat exact one-tool finals; row 2313 hit 152 memory tokens. PID 87268 exact first-turn replay row 2322 restored 152 tokens from prompt Disk L2 (`cacheDetail=disk`, TTFT 0.18s), executed one real tool, and returned exact final. Health reports `native_path_dependent_composite`, schema `openpangu_v2_composite_v2`, `generic_turboquant_kv.enabled=false`, paged false, prompt disk L2 true | 512K/long-context soak, full protocol matrix, and broader openPangu bit-variant coverage; MTP remains detection-only/unwired for this family |
| Cross-model post-tool | PARTIAL | Many named families pass exact one-tool/final rows | MiMo and every remaining configured parser family need current Electron rows |
| Settings parity | PARTIAL | Cache defaults, Auto/None, typed-setting restart, selective-TQ Cache/Perf labeling, explicit Tool Parser None, explicit per-chat Min-P zero, gateway LAN/port persistence, and single-model swap now have scoped source-plus-live proof. Commit `d49f500a3` preserves slider zero in SQLite and both wire builders; clean current-source Electron PID 8935 displayed Min P `0.00`, DB stored `min_p=0.0`, and live `[CHAT_DIAG]` serialized `"min_p":0`. Commit `4e13b19a7` keeps parser None literal. Current-source Electron PID 9909 also exercised gateway conflict rollback, LAN rebind/restore, and the session-manager single-model swap below. | Complete remaining UI/DB/preview/argv/health rows across model-derived defaults and cache controls; retain parser None, Min-P zero, gateway rollback, and single-model swap as regression rows |
| API/protocol parity | PARTIAL | Responses now emits the standards-matching `response.incomplete` terminal event for length-capped streams, and the Electron client consumes completed/incomplete final text, usage, warnings, and status symmetrically (`a36a5ea66`). Current Bonsai gateway controls streamed across all four requested surfaces: Chat Completions returned exact `OAI-GW1-DONE`; Responses emitted 151 reasoning, seven content deltas, matching done text, and `response.completed` (with two leading visible newlines retained as a strict-format miss); Anthropic emitted thinking plus exact `ANT-GW1-DONE` and one `message_stop`; Ollama's current-source `think:true` row emitted 193 reasoning deltas once, exact `OLL-GW4-DONE`, and a single empty-message terminal with usage. A controlled Responses `tool_choice:none` result continuation completed once but repeated native tool markup to the cap on another run, so tool-result synthesis remains variable. | Non-stream equivalents, stable auto tool/result continuation through each protocol, disconnect/stop/follow-up, and strict Responses formatting |
| Gateway lifecycle | PASS-LIVE lifecycle+basic streams / PARTIAL agent protocols | Commit `e76cc5451` makes restart transactional: a rejected port change restores the prior listener and rethrows the original error. Through the current Electron API page, changing running gateway `127.0.0.1:8081` to DSV4's occupied `8012` first reproduced the old stopped-listener bug, then the fixed build rejected the conflict while health and SQLite remained running on 8081. LAN UI enable rebound to `0.0.0.0:8081`, displayed routable `192.168.1.110`, served `/health` over that LAN address, and rebound to localhost when disabled. With Single model mode enabled, the visible Bonsai Start control stopped DSV4 PID 10013 and launched Bonsai PID 10495; UI, SQLite, process listing, gateway discovery, and `[SESSIONS]` lifecycle log all showed exactly one running engine. Commit `a0aa81a94` waits for usage and `[DONE]` before emitting Ollama's empty-message terminal, preventing cumulative thinking duplication and premature loss of `eval_count`; current Electron PID 12046 / Bonsai PID 12114 proved the live stream above. | Agentic tool/result continuation per protocol, disconnect/error recovery, and repeated unload/reload swap soak |
| Full tests/build | OPEN | Current hybrid ownership/cache changes: 784/784 Python hybrid/cache/scheduler tests, 278/278 panel settings tests, and panel typecheck pass. The fetched-block ref-ownership repair adds 90/90 focused paged/TQ/hybrid tests. Parser/Responses terminal coverage passed 135/135 Python, 50/50 panel, and panel typecheck. Typed DSV4 disk-tier telemetry passed 76/76 DSV4/paged-byte-budget tests, three focused scheduler assertions, 43/43 relevant panel tests, and panel typecheck. Explicit Min-P zero passed 213/213 affected panel tests plus typecheck. Gateway transactional restart and Ollama terminal behavior passed 76/76 focused panel tests plus typecheck. Shared terminal-dispatch ordering passed 4/4 behavioral contracts and 50/50 affected tests; the wider cache/batching slice passed 260/261 with one retained unrelated source-string assertion. Cross-model tool inheritance passed 299/299 affected panel tests plus typecheck. Terminal/exact-once head `9618e2e46` passes 131/131 selected tests. Qwen media-keyed hybrid cache head `9982d9ae2` passes 218 tests with six intentional skips across the full ZAYA/Qwen media, MLLM scheduler-cache, and SSM-companion files. Current DSV4 eager/tool-scope commits add 18/18 eager tests, 40/40 combined tool-fallback/hardening tests, 162/162 affected panel tests, and panel typecheck. | Focused suites after each fix, full Python/panel suite, bundled-Python gate, clean release build |
| Eager session materialization | PASS-LIVE scoped DSV4 / OPEN other deferred routes | Source `vmlx_engine/utils/tokenizer.py:1123-1134` now calls the DSV4 JANGTQ loader with `skip_params_eval=False`. A visible Electron Stop/Start completed before any prompt; health reported `last_request_time=null`, `model_loaded=true`, and about 99.7 GB active model memory. The focused DSV4 eager suite passed 18/18. This proves DSV4 only, not every lazy architecture route. | Keep DSV4 as a regression row; inventory and prove other routes that still defer materialization |
| Responsive Electron chrome | OPEN | The user reports top navigation/icons/text can compress or overlap at narrow window sizes. No current-source visual width matrix has been captured. | Resize the real Electron window through bounded widths, fix layout ownership, and retain before/after screenshots without hiding controls |
| Packaging/public release | BLOCKED | Public truth remains 1.6.10 | Build Sequoia/Tahoe, sign, notarize, staple, Gatekeeper verify, install-smoke, publish GitHub/PyPI/feed |

### DSV4 eager materialization and broad-tool continuation — current source

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

### Bonsai post-dispatch q8 hybrid regression — current source

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
| Plain full attention KV | Paged/prompt cache; uncalibrated Auto uses storage-only TQ8; lower bits require bundle-owned calibration; codec fields are part of the persisted namespace | Qwen full-KV and Laguna scoped pass; broader family regression matrix open |
| Qwen/Bonsai hybrid GDN/SSM | Eligible slots come from the real layer graph, not a family-name constant. Qwen 35B has 10 attention KV plus 30 companion layers; tested Bonsai bundles have 16 attention KV plus 48 companion layers. Only attention KV is TQ encoded; companion state remains native with clean boundary capture/rederive plus fingerprinted SSM L2 | Qwen 35B and two 1-bit plus one ternary Bonsai restart restores pass with native TQ8 + SSM disk; current Bonsai 1-bit PID 83540/84219/84984/85595 writes native-TQ L2 blocks and SSM companion disk records while preserving exact multi-turn tool behavior. Changed-prefix native-TQ candidates without an SSM checkpoint safely full-prefill; exact-prefix replay restores cleanly as `paged+ssm+disk`; forced four-block capacity evicts L1 while keeping L2 block+SSM stores intact. Broad long-context coverage remains partial |
| Other hybrid SSM/GLA | Architecture allow-list plus native companion state and async clean-prefill rederive | Nemotron-H current-source Auto/None, L1/L2, and forced-eviction rows pass with exactly six attention slots TQ-eligible and native Mamba companion state; per-family proof remains required and no name-only inference is allowed |
| Gemma 4 mixed rotating SWA | Rotating SWA state remains native; only compatible full-attention KV may be TQ encoded. Prefix lookup, resident paged blocks, L2 disk promotion, companion-state restore/rederive, and bounded eviction must agree on one valid boundary | Parser/tool-loop fix is current-source PASS: raw Responses trace proved the model generated a valid `<|tool_call>` by token 20 then hallucinated client-owned `<|tool_response>`/answer text; source now opts Gemma into completed-call stream stop and truncates at the regex-parseable native call boundary. Focused parser tests pass 13/13. Direct multi-turn Responses proof dropped from 97 output tokens / 82 heartbeats to 28 output tokens / 20 heartbeats and emitted one `file_info({"path":"README.md"})`. Live Electron same-chat rows 2265/2268 each executed one real `file_info` and exact finals; row 2268 reused 218 memory tokens and completed in 3.4s. Restored Auto/paged/L2 rows 2271/2274/2277 then proved `paged+mixed_swa+disk`, resident `paged+mixed_swa`, and post-restart `paged+mixed_swa+disk` exact tool continuations. UI-constrained four-block rows 2280/2283 stayed exact while L1 evictions reached 9 and both rows restored 192 tokens as `paged+mixed_swa+disk`; normal 1,000 blocks were restored on PID 82981. None A/B recheck and long-output cache proof remain PARTIAL. |
| DSV4 Flash | Native `deepseek_v4_v7` SWA + CSA/HCA composite and pool codec; never generic TQ KV | CRACK eager/cache/settings/eviction/stream/tool-scope tiers pass scoped: start-time materialization is live-proven before any prompt; source keeps restored DSV4 L2 payloads resident but evictable; `Scheduler._typed_paged_cache_detail` preserves `+disk`; the panel unions cache tiers across tool iterations. A 3,244-token prefix restored from RAM and then `paged+dsv4+disk`. Broad 33-tool raw Responses and real Electron one-tool plus no-tool continuations complete without the prior literal `response` loop, and long UI reasoning/content painted progressively. Strict constrained formatting and broader long quality remain red. |
| MiniMax-M3 | Native `minimax_m3_msa_v1`, dense KV 0–2 plus sparse MSA/index state 3–59; generic TQ off | Cache/restart scoped pass |
| openPangu 2.0 Flash | Native typed MLA + DSA/SWA + mHC + 128-sink composite; generic paged/block/TQ off | Current 3M Electron rows pass scoped tools, same-chat memory hit, process-restart prompt Disk L2 hit, and single-model swap; long-context/protocol soak remains partial |
| ZAYA/CCA | Typed CCA state; generic TQ off until typed parity exists | Historical live proof; current release regression row still required |
| VLM/video/audio | Architecture cache plus canonical media salt and real media payload | Qwen 3.6 27B image transport and Qwen video-frame fallback/cache are scoped PASS-LIVE: real pixels/MP4, media-keyed RAM hit, cross-media isolation, bypass, block+SSM disk restore, progressive Chat streaming, and visible Electron persistence. Current source bounds fallback video frames before image-token expansion, keys local temporary media by content bytes, and preserves native video tensors/grids. Other advertised-family media matrices remain open. Evidence: `qwen27-media-cache-current/`. |

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
| Qwen 3.6 27B `...-MTP` | The same hybrid cache invariant applies, and MTP is eligible because the actual model/bundle name says `MTP`. Compatible non-Bonsai attention KV uses TQ4; native SSM/GDN companion state remains full precision and independently restored/rederived. Media prompts add a canonical pixel/video-derived side-key; the N-1 cache must be produced while the media tensors remain live. | MTP depth 1 and 3 launch/health, real draft/accepted counters, cold + two-turn + tool continuation, RAM hit, restart/L2 hit, forced eviction/reload, and media A/B isolation with coherent streaming output. | PASS-LIVE agent loop+Auto-TQ4 cache+terminal-stream+image/video-media tiers / PARTIAL long-reasoning and broader-variant reliability: prior current-source rows prove exact distinct tools, q4 plus native-SSM RAM/restart/forced-eviction behavior, truthful UI settings, D1 tool policy, and genuine D3 draft/accept counters. Commit `9982d9ae2` moves clean Qwen media N-1 capture before tensor release and persists attention blocks plus native SSM/GDN under one media key. The image proof is in `qwen27-media-cache-current/`. Current source additionally bounds Qwen video fallback frames before image expansion, uses local media-content hashes rather than random temp-frame paths, and retains native video tensors/grids. Live MP4 A produced exact `FRAME START 2468`/`FRAME END 9753`; alternate B returned only `ALT START 1357`/`ALT END 8642`; return A restored 2,927 `paged+ssm` tokens at 0.9620s. Bypass emitted no cache usage and the following normal A still hit. Electron Stop/Start left L1 empty but L2 populated; A restored as `paged+ssm+disk` at 1.1044s with 46 native-TQ block-disk hits and one SSM-disk hit. A real Electron video attachment visibly completed with both markers. Focused tests: 6 terminal-guard + 1 native-video cache + 6 fallback-bound + 4 media-key. The older text tool/terminal/MTP proofs remain in `qwen27-mtp-stream-cache-current/`. Long native reasoning latency, larger-context cancellation, and broader variant reliability remain open. |
| HY3 MTP | The exact `Hy3-JANG_2K-MTP` bundle declares one MTP layer and 42 MTP tensors, so the runtime must use depth 1 rather than inventing a depth-3 gate. HY3's plain attention KV may use the family-scoped TQ4 stored-prefix codec; live decode stays native, and MTP batch split/verify must own independent cache copies. | Depth-1 draft/accepted counters, same-chat multi-turn tool loop, process-restart L2 restore, forced eviction/reload, explicit None A/B, and coherent long/streaming output. | PASS-LIVE cache+settings+restart+eviction / PARTIAL reliability+long: commit `ab5d01e04` selects HY3 full-KV TQ4 and installs an independent `TurboQuantKVCache.__deepcopy__`, fixing the live `cannot pickle 'mlx.core.Dtype' object` scheduler retry loop; 19 focused TQ tests and 178 native-MTP tests passed. Commit `5e6a1f8a1` reports `Native HY3 KV + TQ4 stored prefixes / TQ4 AUTO` in settings; 282 settings tests and typecheck passed, and the label is visible in Electron. PID 22265 row 2483 survived UI process restart, restored 3,272 tokens as `paged+disk+tq-native`, executed one `file_info`, and exact-finaled. UI-applied four-block PID 23635 produced rows 2488/2489, 11 L1 evictions, five TQ-native L2 writes, and 18 TQ-native hits; older-prefix row 2492 restored the bounded 192 tokens as `paged+disk+tq-native`, executed exactly one tool, and returned exact `HY3-Q4-T1R-DONE`. Explicit UI None launched PIDs 26444/27461 with `--kv-cache-quantization none`; cold row 2495 wrote 54 raw blocks with zero TQ activity, and restart row 2498 restored 3,258 tokens as `paged+disk`, made one real tool call, and exact-finaled while TQ writes/hits remained zero. UI restored Auto, 1,000 blocks, and TQ-native enablement on PID 28473. Same-chat rows 2474/2480 exact-finaled, but row 2477 emitted `HY3-Q4-T2-D-DE-DONE`; strict-format reliability therefore remains PARTIAL. |
| MiniMax M2.7 | Ordinary KV attention may use calibrated or correctness-safe TQ storage; parser/reasoning rails must survive multi-turn tool continuation. | Auto and None UI/argv/health A/B, two-turn tool loop, RAM/L2 restore, eviction, long visible answer, and streaming rail continuity. | PASS-LIVE current source: rows 2187/2190 prove cold plus same-chat two-tool continuation and a 173-token resident `paged+tq-native` hit. PID 63682 row 2193 restored 173/177 as `paged+disk+tq-native`. None mode PID 64194 launched with explicit `--kv-cache-quantization none`, wrote raw `dtype=kv` blocks, and PID 64579 row 2199 restored 161/165 as `paged+disk` with zero TQ activity. Commit `af7815f1a` repairs fetched-block ref ownership; under the UI-applied four-block ceiling, PID 65838 rows 2208/2211 completed exact tool loops, returned all three usable blocks to the free queue, and raised L1 evictions from 3 to 9. Normal 1,000-block Auto was restored on PID 66306 and row 2214 repeated the exact 173-token disk hit. Electron row 2217 produced a coherent 582-token reasoning/content answer with the exact terminal marker. A direct Responses stream with a 1,024-token budget emitted 711 reasoning deltas, 48 content deltas, matching text-done, and `response.completed(status=completed)` with its exact marker. The controlled 512-token cap correctly reported `status=incomplete` instead of pretending completion. |
| ZAYA / CCA | Typed CCA state owns its cache. Generic TQ is forbidden unless a typed CCA codec has source and live parity. | Typed cold/warm/restart/eviction rows plus multi-turn tool and reasoning/content stream. | BLOCKED current generic row: the external drive contains only the `AppleScript-8B-JANG_4M` single-tool specialist, which the user excluded from this campaign. This is a missing-artifact gate, not a runtime failure. |
| Nemotron hybrid | Eligible attention KV may be TQ encoded; non-KV hybrid state remains native and is async clean-prefill rederived/restored. Family selection must come from config/layers, not a name match. | Auto/None A/B, cold + two-turn + tool continuation, L2 restart, eviction, long output, no reasoning leak. | PASS-LIVE cache/settings/tools/API / PARTIAL repeated long reasoning: rows 2223/2226 were exact cold and same-chat one-tool turns, with 162 tokens restored as `paged+ssm+tq-native`. PID 74652 row 2229 restored 192 tokens as `paged+ssm+disk+tq-native`. UI-applied four-block PID 75038 rows 2235/2238 stayed exact while evictions rose 3 to 9 and three usable blocks returned free. Explicit None PIDs 75398/75644 rows 2241/2244 wrote and restored raw `paged+ssm+disk` blocks with zero TQ activity. Auto/1,000 blocks is restored on PID 75939. Electron row 2247 completed a coherent marked answer but repeated 2,962 tokens of native reasoning before the real `</think>`; retained as reliability PARTIAL. Direct Responses emitted 424 reasoning deltas, 30 content deltas, matching done events, and `response.completed`. Focused source tests pass 25/25. |
| Gemma 4 rotating SWA | TQ applies only to compatible full-attention KV. Rotating SWA cache remains native, and a prefix hit is valid only when both lanes share a restorable boundary; otherwise safely rederive/full-prefill. | Auto/None UI/argv/health A/B, cold + two-turn + tool continuation, resident paged hit, L2 restart promotion, forced eviction/reload, true-miss fallback, and coherent long output. | PASS-LIVE cache/settings/tools/eviction/restart / PARTIAL long-output: commit `3385cb019` makes UI/CLI Auto select q4 only for full-attention slots, keeps rotating-SWA slots native, and fails closed on layout mismatch; 153 focused tests passed. Auto rows 2425/2428/2431 and final row 2470 each made one real tool call and exact final; row 2431 survived process replacement with 704 `paged+mixed_swa+disk` tokens and 44 native-TQ disk hits. A visible 16-block pressure run produced 38 L1 evictions; post-eviction row 2464 restored 704 tokens from TQ-native L2 and exact-finaled. Explicit None PID 15388 launched with `--kv-cache-quantization none`; row 2467 exact-finaled while ordinary disk writes rose to three and TQ writes/hits stayed zero. UI Auto/1,000 blocks is restored on PID 15797; row 2470 restored 704 `paged+mixed_swa+disk` tokens with three TQ-native writes and eleven hits. Commit `ba68f8fba` fixes the settings drawer to show `TQ4 full-attention KV + native rotating SWA / MIXED AUTO` instead of an SSM/GLA label; 281 settings tests and typecheck passed, and the label is visible in Electron. Coherent constrained long-output remains open. Evidence: `docs/internal/release-gates/20260716_gemma4_mixed_swa_tq4/`. |
| DSV4 Flash | Native DSA/SWA/CSA/HCA composite and pool codec only; never generic TQ KV. | Composite cache health, eager load, cold/warm/restart/eviction, multi-turn agent loop, reasoning/content stream continuity and coherent constrained output. | PASS-LIVE eager/cache/settings/eviction/stream/tool-scope tiers / PARTIAL strict long quality+perf: source trace `vmlx_engine/utils/tokenizer.py:1123-1134` owns eager DSV4 materialization; before-prompt live health reported no request plus about 99.7 GB active memory. Native composite cache source and live proof retain the earlier eviction tiers plus a current 3,244-token RAM and `paged+dsv4+disk` restore. Commit `012c1fe90` scopes fallback validation to DSV4's native explicit/recent-tool selection, retains concrete explicit path binding, and applies standard `tool_choice:none` to directive-shaped Electron turns. Raw Responses with all 33 tools streamed and completed; Electron row 153 executed one real tool, rows 156/159 continued without a call, and row 162 progressively painted reasoning/content. Focused tests: 18 eager, 40 combined Python tool/hardening, 162 panel, and typecheck. Row 159 violated one-sentence form, row 162 missed the exact marker, and prior long row 2412 failed quality; those gates remain red. Evidence: `dsv4-eager-current/`, `dsv4-cache-current/`, `dsv4-tool-scope-current/`. |
| MiniMax M3 / openPangu | Native typed architecture cache only; generic TQ remains off. | openPangu 3M current rows now pass scoped tools/restart prompt L2; MiniMax-M3 exact OCR repeat and broader protocol/media boundaries remain partial. | PARTIAL |

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
6. Build, sign, notarize, staple, verify, install-smoke, and publish 1.6.11 only
   after every release-blocking row above is green.
