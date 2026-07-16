# vMLX 1.6.11 release closeout matrix — 2026-07-16

Status: `PARTIAL_NO_RELEASE`.

This is the current additive closeout view over `docs/internal/ISSUE-LEDGER.md`,
`.agents/STATUS.md`, the July 15–16 live proof directories, the shared wiki
production gate, and the current branch. Older contradictory rows remain in
their original ledgers for provenance; the newest source-plus-live row wins and
superseded conclusions are called out here.

## Release truth

- Working branch: `reconcile/1.5.68` at `7cb89185c`; typed-settings,
  non-MTP architecture-hint, paged resident-accounting, typed hybrid-companion
  ownership, and v8 cache-namespace repairs plus their focused tests are pushed
  to the closeout branch described below.
- Push target: `origin/codex/live-electron-gates-20260715`.
- After fetching `origin`, the committed branch is 49 commits ahead of `origin/main` and
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
| Bonsai current-HEAD regression | PASS-LIVE scoped | Two default-temperature 1-bit cache-on chats produced four exact one-tool finals; ternary passed two turns plus restart. Ternary None preserved prefix/paged/L2 with zero TQ activity, and Auto was restored. Earlier 1-bit loop row 2028 remains retained | Quantify 1-bit sampling outlier rate and run long-context/eviction without forcing sampler defaults |
| Mistral Medium 3.5 | DEFERRED BY USER | Prior text load/cache observations are retained, but the user explicitly excluded further Mistral MXFP4 testing from this closeout run | Do not spend this campaign on Mistral MXFP4; it is not used to claim a current release pass |
| DSV4 CRACK | PARTIAL-LIVE | Native composite cache and DSML separation pass; malformed row took 64.7s, restart tool row took 119.2s/1454 tokens | Constrained-string repeat matrix, reasoning/tail quality, quiet speed; exact JANGTQ bundle only if locally available |
| MiniMax-M3 | PARTIAL-LIVE | Typed MSA cache, tools, OCR control, and video pass; two OCR formatting misses retained | Exact deterministic OCR repeat; live 503 guard for REAP32 only if it can be exercised without host-reboot risk |
| openPangu | PARTIAL-LIVE | Exact typed cache/tools/restart and full 46-layer architecture pass; generic TQ correctly off | 512K/long-context soak, full protocol matrix; MTP remains unavailable for current artifact |
| Cross-model post-tool | PARTIAL | Many named families pass exact one-tool/final rows | MiMo and every remaining configured parser family need current Electron rows |
| Settings parity | PARTIAL | Cache defaults, Auto/None, gateway LAN, single-model swap, typed-setting restart, and selective-TQ Cache/Perf labeling have scoped proof | Fix Min-P zero persistence; rerun UI/DB/preview/argv/health matrix including port conflict and LAN/gateway state |
| API/protocol parity | OPEN | Selected Responses/gateway rows pass | Streaming + non-stream Chat, Responses, Anthropic, Ollama; tools/result continuation; disconnect/stop/follow-up |
| Gateway lifecycle | PARTIAL | Routable LAN address and localhost/LAN rebinding pass | Port conflict UX, cross-protocol streaming, single-model unload/reload state |
| Full tests/build | OPEN | Current hybrid ownership/cache changes: 784/784 Python hybrid/cache/scheduler tests, 278/278 panel settings tests, and panel typecheck pass. The fetched-block ref-ownership repair adds 90/90 focused paged/TQ/hybrid tests. | Focused suites after each fix, full Python/panel suite, bundled-Python gate, clean release build |
| Packaging/public release | BLOCKED | Public truth remains 1.6.10 | Build Sequoia/Tahoe, sign, notarize, staple, Gatekeeper verify, install-smoke, publish GitHub/PyPI/feed |

## Architecture-specific cache truth

| Architecture | Production cache contract | Current status |
|---|---|---|
| Plain full attention KV | Paged/prompt cache; uncalibrated Auto uses storage-only TQ8; lower bits require bundle-owned calibration; codec fields are part of the persisted namespace | Qwen full-KV and Laguna scoped pass; broader family regression matrix open |
| Qwen/Bonsai hybrid GDN/SSM | Eligible slots come from the real layer graph, not a family-name constant. Qwen 35B has 10 attention KV plus 30 companion layers; tested Bonsai bundles have 16 attention KV plus 48 companion layers. Only attention KV is TQ encoded; companion state remains native with clean boundary capture/rederive plus fingerprinted SSM L2 | Qwen 35B and two 1-bit plus one ternary Bonsai restart restores pass with native TQ8 + SSM disk; missing partial checkpoints safely full-prefill and repair; forced eviction and broad long-context coverage remain partial |
| Other hybrid SSM/GLA | Architecture allow-list plus native companion state and async clean-prefill rederive | Per-family proof required; no name-only Qwen inference |
| Gemma 4 mixed rotating SWA | Rotating SWA state remains native; only compatible full-attention KV may be TQ encoded. Prefix lookup, resident paged blocks, L2 disk promotion, companion-state restore/rederive, and bounded eviction must agree on one valid boundary | Prior UI/DB/argv/warm/restart evidence is scoped only; current-source forced eviction and tier-fallback proof remain open |
| DSV4 Flash | Native `deepseek_v4_v7` SWA + CSA/HCA composite and pool codec; never generic TQ KV | CRACK scoped cache pass; quality/performance partial |
| MiniMax-M3 | Native `minimax_m3_msa_v1`, dense KV 0–2 plus sparse MSA/index state 3–59; generic TQ off | Cache/restart scoped pass |
| openPangu 2.0 Flash | Native typed MLA + DSA/SWA + mHC + sink composite; generic paged/block/TQ off | Typed prefix/prompt-L2 pass |
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
| MiniMax M2.7 | Ordinary KV attention may use calibrated or correctness-safe TQ storage; parser/reasoning rails must survive multi-turn tool continuation. | Auto and None UI/argv/health A/B, two-turn tool loop, RAM/L2 restore, eviction, long visible answer, and streaming rail continuity. | PASS-LIVE cache/settings/tools/eviction / PARTIAL long stream: rows 2187/2190 prove cold plus same-chat two-tool continuation and a 173-token resident `paged+tq-native` hit. PID 63682 row 2193 restored 173/177 as `paged+disk+tq-native`. None mode PID 64194 launched with explicit `--kv-cache-quantization none`, wrote raw `dtype=kv` blocks, and PID 64579 row 2199 restored 161/165 as `paged+disk` with zero TQ activity. Commit `af7815f1a` repairs fetched-block ref ownership; under the UI-applied four-block ceiling, PID 65838 rows 2208/2211 completed exact tool loops, returned all three usable blocks to the free queue, and raised L1 evictions from 3 to 9. Normal 1,000-block Auto was restored on PID 66306 and row 2214 repeated the exact 173-token disk hit. Long non-tool visible answer and direct streaming rail soak remain open. |
| ZAYA / CCA | Typed CCA state owns its cache. Generic TQ is forbidden unless a typed CCA codec has source and live parity. | Typed cold/warm/restart/eviction rows plus multi-turn tool and reasoning/content stream. | OPEN current release row |
| Nemotron hybrid | Eligible attention KV may be TQ encoded; non-KV hybrid state remains native and is async clean-prefill rederived/restored. Family selection must come from config/layers, not a name match. | Auto/None A/B, cold + two-turn + tool continuation, L2 restart, eviction, long output, no reasoning leak. | OPEN |
| Gemma 4 rotating SWA | TQ applies only to compatible full-attention KV. Rotating SWA cache remains native, and a prefix hit is valid only when both lanes share a restorable boundary; otherwise safely rederive/full-prefill. | Auto/None UI/argv/health A/B, cold + two-turn + tool continuation, resident paged hit, L2 restart promotion, forced eviction/reload, true-miss fallback, and coherent long output. | OPEN current-source tier/eviction row; older warm/restart settings evidence is retained only as scoped history |
| DSV4 Flash | Native DSA/SWA/CSA/HCA composite and pool codec only; never generic TQ KV. | Composite cache health, cold/warm/restart/eviction, multi-turn agent loop, reasoning/content stream continuity and coherent constrained output. | PARTIAL |
| MiniMax M3 / openPangu | Native typed architecture cache only; generic TQ remains off. | Existing scoped rows plus long-context, restart/eviction, and protocol/streaming completion. | PARTIAL |

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

1. Complete MiniMax M2.7's remaining long non-tool visible-answer and direct
   streaming rail soak; its Auto/None, multi-turn tool, RAM/L2, and eviction
   rows are closed. Then run ZAYA/CCA, Nemotron hybrid, and Gemma 4
   rotating-SWA tier/eviction rows.
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
