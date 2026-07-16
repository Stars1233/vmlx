# vMLX 1.6.11 release closeout matrix — 2026-07-16

Status: `PARTIAL_NO_RELEASE`.

This is the current additive closeout view over `docs/internal/ISSUE-LEDGER.md`,
`.agents/STATUS.md`, the July 15–16 live proof directories, the shared wiki
production gate, and the current branch. Older contradictory rows remain in
their original ledgers for provenance; the newest source-plus-live row wins and
superseded conclusions are called out here.

## Release truth

- Working branch: `reconcile/1.5.68` at `4e13b19a7`; typed-settings,
  non-MTP architecture-hint, paged resident-accounting, typed hybrid-companion
  ownership, and v8 cache-namespace repairs plus their focused tests are pushed
  to the closeout branch described below.
- Push target: `origin/codex/live-electron-gates-20260715`.
- After fetching `origin`, the committed branch is 68 commits ahead of `origin/main` and
  zero behind.
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
| Mistral Medium 3.5 | DEFERRED BY USER | Prior text load/cache observations are retained, but the user explicitly excluded further Mistral MXFP4 testing from this closeout run | Do not spend this campaign on Mistral MXFP4; it is not used to claim a current release pass |
| DSV4 CRACK | PASS-LIVE cache/settings tier / PARTIAL quality+perf | Native composite cache and DSML separation pass; Electron PID 95494 launched from project `.venv` with `--cache-memory-percent 0.15`, health reported 15% L1 ceiling, row 2343 restored DSV4 state from block L2 and returned exact `DSV4-FIX1-DONE`, and row 2346 reused 611 resident `paged+dsv4` tokens with no new scheduler/block-disk hits | Constrained-string repeat matrix, reasoning/tail quality, quiet speed, forced four-block eviction/reload, and exact JANGTQ bundle only if locally available |
| MiniMax-M3 | PARTIAL-LIVE | Typed MSA cache, tools, OCR control, and video pass; two OCR formatting misses retained | Exact deterministic OCR repeat; live 503 guard for REAP32 only if it can be exercised without host-reboot risk |
| openPangu | PASS-LIVE scoped / PARTIAL long-context+protocol | Source policy `_apply_openpangu_cache_policy` forces paged/block/TQ off and preserves typed MLA KV, DSA indexer, rotating-SWA metadata, causal-conv state, 128 sinks, and mHC runtime. Electron-loaded 3M PID 86212/86842/87268 launched with `--no-paged-cache --enable-disk-cache`, no KV quantization, and no block L2; Bonsai was unloaded by the single-model swap. Rows 2310/2313 prove same-chat exact one-tool finals; row 2313 hit 152 memory tokens. PID 87268 exact first-turn replay row 2322 restored 152 tokens from prompt Disk L2 (`cacheDetail=disk`, TTFT 0.18s), executed one real tool, and returned exact final. Health reports `native_path_dependent_composite`, schema `openpangu_v2_composite_v2`, `generic_turboquant_kv.enabled=false`, paged false, prompt disk L2 true | 512K/long-context soak, full protocol matrix, and broader openPangu bit-variant coverage; MTP remains detection-only/unwired for this family |
| Cross-model post-tool | PARTIAL | Many named families pass exact one-tool/final rows | MiMo and every remaining configured parser family need current Electron rows |
| Settings parity | PARTIAL | Cache defaults, Auto/None, gateway LAN, single-model swap, typed-setting restart, selective-TQ Cache/Perf labeling, and explicit Tool Parser None now have scoped source-plus-live proof. Commit `4e13b19a7` prevents request/model auto-detection and streaming marker buffering from silently re-enabling a disabled parser; Electron PID 99835 launched with literal `--tool-call-parser none`, and row 2358 persisted raw model text with no structured tool call/result. | Fix Min-P zero persistence; rerun UI/DB/preview/argv/health matrix including port conflict and LAN/gateway state; retain parser None as a regression row |
| API/protocol parity | OPEN | Selected Responses/gateway rows pass | Streaming + non-stream Chat, Responses, Anthropic, Ollama; tools/result continuation; disconnect/stop/follow-up |
| Gateway lifecycle | PARTIAL | Routable LAN address and localhost/LAN rebinding pass | Port conflict UX, cross-protocol streaming, single-model unload/reload state |
| Full tests/build | OPEN | Current hybrid ownership/cache changes: 784/784 Python hybrid/cache/scheduler tests, 278/278 panel settings tests, and panel typecheck pass. The fetched-block ref-ownership repair adds 90/90 focused paged/TQ/hybrid tests. Parser-None repair passed 106/106 `test_server.py`, plus 52 passed / 1 skipped across selected server/openPangu/VL parser coverage. | Focused suites after each fix, full Python/panel suite, bundled-Python gate, clean release build |
| Packaging/public release | BLOCKED | Public truth remains 1.6.10 | Build Sequoia/Tahoe, sign, notarize, staple, Gatekeeper verify, install-smoke, publish GitHub/PyPI/feed |

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
- Bonsai remains `PARTIAL` because row 2352 and the earlier 4,222-character
  reasoning turn prove variability. No sampler clamp, prompt coercion, hidden
  reasoning disable, or argument rewriting was added.

## Architecture-specific cache truth

| Architecture | Production cache contract | Current status |
|---|---|---|
| Plain full attention KV | Paged/prompt cache; uncalibrated Auto uses storage-only TQ8; lower bits require bundle-owned calibration; codec fields are part of the persisted namespace | Qwen full-KV and Laguna scoped pass; broader family regression matrix open |
| Qwen/Bonsai hybrid GDN/SSM | Eligible slots come from the real layer graph, not a family-name constant. Qwen 35B has 10 attention KV plus 30 companion layers; tested Bonsai bundles have 16 attention KV plus 48 companion layers. Only attention KV is TQ encoded; companion state remains native with clean boundary capture/rederive plus fingerprinted SSM L2 | Qwen 35B and two 1-bit plus one ternary Bonsai restart restores pass with native TQ8 + SSM disk; current Bonsai 1-bit PID 83540/84219/84984/85595 writes native-TQ L2 blocks and SSM companion disk records while preserving exact multi-turn tool behavior. Changed-prefix native-TQ candidates without an SSM checkpoint safely full-prefill; exact-prefix replay restores cleanly as `paged+ssm+disk`; forced four-block capacity evicts L1 while keeping L2 block+SSM stores intact. Broad long-context coverage remains partial |
| Other hybrid SSM/GLA | Architecture allow-list plus native companion state and async clean-prefill rederive | Nemotron-H current-source Auto/None, L1/L2, and forced-eviction rows pass with exactly six attention slots TQ-eligible and native Mamba companion state; per-family proof remains required and no name-only inference is allowed |
| Gemma 4 mixed rotating SWA | Rotating SWA state remains native; only compatible full-attention KV may be TQ encoded. Prefix lookup, resident paged blocks, L2 disk promotion, companion-state restore/rederive, and bounded eviction must agree on one valid boundary | Parser/tool-loop fix is current-source PASS: raw Responses trace proved the model generated a valid `<|tool_call>` by token 20 then hallucinated client-owned `<|tool_response>`/answer text; source now opts Gemma into completed-call stream stop and truncates at the regex-parseable native call boundary. Focused parser tests pass 13/13. Direct multi-turn Responses proof dropped from 97 output tokens / 82 heartbeats to 28 output tokens / 20 heartbeats and emitted one `file_info({"path":"README.md"})`. Live Electron same-chat rows 2265/2268 each executed one real `file_info` and exact finals; row 2268 reused 218 memory tokens and completed in 3.4s. Restored Auto/paged/L2 rows 2271/2274/2277 then proved `paged+mixed_swa+disk`, resident `paged+mixed_swa`, and post-restart `paged+mixed_swa+disk` exact tool continuations. UI-constrained four-block rows 2280/2283 stayed exact while L1 evictions reached 9 and both rows restored 192 tokens as `paged+mixed_swa+disk`; normal 1,000 blocks were restored on PID 82981. None A/B recheck and long-output cache proof remain PARTIAL. |
| DSV4 Flash | Native `deepseek_v4_v7` SWA + CSA/HCA composite and pool codec; never generic TQ KV | CRACK cache/settings tier pass: source keeps restored DSV4 L2 payloads resident but evictable, panel now emits visible cache memory budget, PID 95494 argv includes `--cache-memory-percent 0.15`, row 2343 used L2, and row 2346 proved RAM-first reuse with unchanged disk-hit counters. Quality/performance and bounded eviction remain partial |
| MiniMax-M3 | Native `minimax_m3_msa_v1`, dense KV 0–2 plus sparse MSA/index state 3–59; generic TQ off | Cache/restart scoped pass |
| openPangu 2.0 Flash | Native typed MLA + DSA/SWA + mHC + 128-sink composite; generic paged/block/TQ off | Current 3M Electron rows pass scoped tools, same-chat memory hit, process-restart prompt Disk L2 hit, and single-model swap; long-context/protocol soak remains partial |
| ZAYA/CCA | Typed CCA state; generic TQ off until typed parity exists | Historical live proof; current release regression row still required |
| VLM/video/audio | Architecture cache plus canonical media salt and real media payload | M3/Qwen selected rows pass; advertised-family matrix open |

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
| Qwen 3.6 35B MXFP/JANG (name has no `MTP`) | Hybrid layout is derived from the real 10-attention/30-companion layer graph. TQ encode/decode applies only to eligible attention KV; GDN/SSM companions remain native and are cleanly rederived/restored. This artifact is not assigned an MTP gate. | Cold + two-turn + tool continuation, RAM hit, restart/L2 hit, and forced eviction/reload with coherent output. | PASS-LIVE cache tiers / PARTIAL long strict format: rows 2169/2172/2175 prove cold, same-process `paged+ssm`, and restart `paged+ssm+disk`; v8 disk files contain exactly 10 `turboquant_kv` + 30 `skip` entries and no cumulative duplicates. UI-applied four-block rows 2178/2181 restored 154 tokens from L2 with exact tool/final output, forced nine L1 evictions, and safely full-prefilled when 192 KV tokens lacked a matching companion. PID 61919 restored the normal 1,000-block setting and row 2184 repeated the exact disk hit. The ambiguous long-tool row and clarified minor period-format miss remain retained. |
| Qwen 3.6 27B `...-MTP` | The same hybrid cache invariant applies, and MTP is eligible because the actual model/bundle name says `MTP`. | MTP depth 1 and 3 launch/health, real draft/accepted counters, cold + two-turn + tool continuation, RAM hit, restart/L2 hit, and forced eviction/reload with coherent output. | PARTIAL: typed D3 Save & Restart now has UI/DB/argv/health parity; tools-on D1-capped multi-turn passes; tools-off D1 and D3 both looped in reasoning; cache restart/eviction rows remain open |
| HY3 MTP | Native MTP depth is the requested value and yields measured accepted draft tokens. Prompt/L2 records include the owning target-model cache state; speculative output is never treated as a substitute cache. | Depth 1 and depth 3 A/B, acceptance and latency counters, multi-turn tool loop, restart restore, and eviction/reload. | PARTIAL: depth-1 speed row exists; depth-3/cache interaction is open |
| MiniMax M2.7 | Ordinary KV attention may use calibrated or correctness-safe TQ storage; parser/reasoning rails must survive multi-turn tool continuation. | Auto and None UI/argv/health A/B, two-turn tool loop, RAM/L2 restore, eviction, long visible answer, and streaming rail continuity. | PASS-LIVE current source: rows 2187/2190 prove cold plus same-chat two-tool continuation and a 173-token resident `paged+tq-native` hit. PID 63682 row 2193 restored 173/177 as `paged+disk+tq-native`. None mode PID 64194 launched with explicit `--kv-cache-quantization none`, wrote raw `dtype=kv` blocks, and PID 64579 row 2199 restored 161/165 as `paged+disk` with zero TQ activity. Commit `af7815f1a` repairs fetched-block ref ownership; under the UI-applied four-block ceiling, PID 65838 rows 2208/2211 completed exact tool loops, returned all three usable blocks to the free queue, and raised L1 evictions from 3 to 9. Normal 1,000-block Auto was restored on PID 66306 and row 2214 repeated the exact 173-token disk hit. Electron row 2217 produced a coherent 582-token reasoning/content answer with the exact terminal marker. A direct Responses stream with a 1,024-token budget emitted 711 reasoning deltas, 48 content deltas, matching text-done, and `response.completed(status=completed)` with its exact marker. The controlled 512-token cap correctly reported `status=incomplete` instead of pretending completion. |
| ZAYA / CCA | Typed CCA state owns its cache. Generic TQ is forbidden unless a typed CCA codec has source and live parity. | Typed cold/warm/restart/eviction rows plus multi-turn tool and reasoning/content stream. | BLOCKED current generic row: the external drive contains only the `AppleScript-8B-JANG_4M` single-tool specialist, which the user excluded from this campaign. This is a missing-artifact gate, not a runtime failure. |
| Nemotron hybrid | Eligible attention KV may be TQ encoded; non-KV hybrid state remains native and is async clean-prefill rederived/restored. Family selection must come from config/layers, not a name match. | Auto/None A/B, cold + two-turn + tool continuation, L2 restart, eviction, long output, no reasoning leak. | PASS-LIVE cache/settings/tools/API / PARTIAL repeated long reasoning: rows 2223/2226 were exact cold and same-chat one-tool turns, with 162 tokens restored as `paged+ssm+tq-native`. PID 74652 row 2229 restored 192 tokens as `paged+ssm+disk+tq-native`. UI-applied four-block PID 75038 rows 2235/2238 stayed exact while evictions rose 3 to 9 and three usable blocks returned free. Explicit None PIDs 75398/75644 rows 2241/2244 wrote and restored raw `paged+ssm+disk` blocks with zero TQ activity. Auto/1,000 blocks is restored on PID 75939. Electron row 2247 completed a coherent marked answer but repeated 2,962 tokens of native reasoning before the real `</think>`; retained as reliability PARTIAL. Direct Responses emitted 424 reasoning deltas, 30 content deltas, matching done events, and `response.completed`. Focused source tests pass 25/25. |
| Gemma 4 rotating SWA | TQ applies only to compatible full-attention KV. Rotating SWA cache remains native, and a prefix hit is valid only when both lanes share a restorable boundary; otherwise safely rederive/full-prefill. | Auto/None UI/argv/health A/B, cold + two-turn + tool continuation, resident paged hit, L2 restart promotion, forced eviction/reload, true-miss fallback, and coherent long output. | PASS-LIVE parser/tool continuation, Auto L2 tiers, and forced eviction / PARTIAL None+long-output: rows 2265/2268 prove same-chat Electron two-turn tool continuation after parser repair. Rows 2271/2274/2277 prove exact tool continuations with `paged+mixed_swa+disk` (3,264 tokens), resident `paged+mixed_swa` (543 tokens), and post-restart `paged+mixed_swa+disk` (709 tokens). UI-applied four-block PID 82455 rows 2280/2283 stayed exact and restored 192 tokens from L2 while health showed `l1_evictions=9` and reconstruction 0.015s. Normal 1,000 blocks were restored on PID 82981 with the expected argv. None recheck and coherent long-output row remain open before full cache release credit. |
| DSV4 Flash | Native DSA/SWA/CSA/HCA composite and pool codec only; never generic TQ KV. | Composite cache health, cold/warm/restart/eviction, multi-turn agent loop, reasoning/content stream continuity and coherent constrained output. | PASS-LIVE cache/settings tier / PARTIAL quality+eviction: source trace `prefix_cache.py::_block_payload_has_dsv4` plus `BlockAwarePrefixCache.reconstruct_cache` keeps restored DSV4 native payloads in L1 and clears temporary protection via `paged_cache.py::make_resident_payload_evictable`; panel source `sessions.ts`, `SessionSettings.tsx`, and `SessionConfigForm.tsx` now pass DSV4 Cache Memory % to launch/preview/UI. Tests: 76/76 DSV4/paged byte-budget and 280/280 settings plus typecheck. Live Electron evidence: `dsv4-tiered-l1-argv.txt`, `dsv4-tiered-l1-start-health.json`, rows 2343/2346, and `dsv4-tier-row2-live.png`. Forced eviction, long constrained output, and reasoning/content stream soak remain open. |
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
  assumption. Bonsai/Qwen hybrid currently remains TQ8 until a Q4 hybrid
  restart/eviction proof shows exact tool/reasoning/content behavior with a
  matched SSM companion. For other compatible non-composite KV families, Q4 is
  the target Auto storage width only after source classification excludes typed
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
- HY3 MTP depth 1 is active and measured: controlled warm median improved
  21.234247s to 16.081931s, with 180/414 draft tokens accepted.
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
   reasoning retained as PARTIAL. Run Gemma 4 rotating-SWA tier/eviction next.
2. Close DSV4 native-composite quality/stream/eviction, HY3 D3/cache, and the
   remaining M3/Pangu long/media boundaries. Do not test Mistral MXFP4 in this
   campaign per the user's explicit instruction.
3. Re-prove Bonsai forced eviction/repair boundaries, retaining its recorded
   sampling miss as reliability evidence; keep Qwen 35B's long-format miss in
   the reliability ledger without reopening its now-closed cache tier row.
4. Close the remaining Laguna unsolicited-tool/long-context/latency rows and
   run the complete settings and protocol matrix through Electron/gateway.
5. Run focused and full tests, audit the dirty tree, commit/push only scoped
   files, and merge/integrate the closeout branch deliberately.
6. Build, sign, notarize, staple, verify, install-smoke, and publish 1.6.11 only
   after every release-blocking row above is green.
