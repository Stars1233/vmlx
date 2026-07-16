# vMLX 1.6.11 release closeout matrix — 2026-07-16

Status: `PARTIAL_NO_RELEASE`.

This is the current additive closeout view over `docs/internal/ISSUE-LEDGER.md`,
`.agents/STATUS.md`, the July 15–16 live proof directories, the shared wiki
production gate, and the current branch. Older contradictory rows remain in
their original ledgers for provenance; the newest source-plus-live row wins and
superseded conclusions are called out here.

## Release truth

- Working branch: `reconcile/1.5.68` at `f67684e09156`; scoped Bonsai SSM
  L2 source and ternary live-evidence commits are pushed to the closeout
  branch described below.
- Push target: `origin/codex/live-electron-gates-20260715`.
- After fetching `origin`, the committed branch is 38 commits ahead of `origin/main` and
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
| Mistral Medium 3.5 | PARTIAL-LIVE | Text load/cache works; broad tool prompt repeated `2026`, strict marker returned `I understand.` | Root-cause model/runtime/template/parser behavior; long output and reduced/broad tool parity |
| DSV4 CRACK | PARTIAL-LIVE | Native composite cache and DSML separation pass; malformed row took 64.7s, restart tool row took 119.2s/1454 tokens | Constrained-string repeat matrix, reasoning/tail quality, quiet speed; exact JANGTQ bundle only if locally available |
| MiniMax-M3 | PARTIAL-LIVE | Typed MSA cache, tools, OCR control, and video pass; two OCR formatting misses retained | Exact deterministic OCR repeat; live 503 guard for REAP32 only if it can be exercised without host-reboot risk |
| openPangu | PARTIAL-LIVE | Exact typed cache/tools/restart and full 46-layer architecture pass; generic TQ correctly off | 512K/long-context soak, full protocol matrix; MTP remains unavailable for current artifact |
| Cross-model post-tool | PARTIAL | Many named families pass exact one-tool/final rows | MiMo and every remaining configured parser family need current Electron rows |
| Settings parity | PARTIAL | Cache defaults, Auto/None, gateway LAN, and single-model swap have scoped proof | Fix Min-P zero persistence, native-TQ Perf label, Laguna parser migration; rerun UI/DB/preview/argv/health matrix |
| API/protocol parity | OPEN | Selected Responses/gateway rows pass | Streaming + non-stream Chat, Responses, Anthropic, Ollama; tools/result continuation; disconnect/stop/follow-up |
| Gateway lifecycle | PARTIAL | Routable LAN address and localhost/LAN rebinding pass | Port conflict UX, cross-protocol streaming, single-model unload/reload state |
| Full tests/build | OPEN | Current Bonsai/Qwen cache selection: 196/196 pass | Focused suites after each fix, full Python/panel suite, bundled-Python gate, clean release build |
| Packaging/public release | BLOCKED | Public truth remains 1.6.10 | Build Sequoia/Tahoe, sign, notarize, staple, Gatekeeper verify, install-smoke, publish GitHub/PyPI/feed |

## Architecture-specific cache truth

| Architecture | Production cache contract | Current status |
|---|---|---|
| Plain full attention KV | Paged/prompt cache; uncalibrated Auto uses storage-only TQ8; lower bits require bundle-owned calibration; codec fields are part of the persisted namespace | Qwen full-KV and Laguna scoped pass; broader family regression matrix open |
| Qwen3.5/Bonsai hybrid GDN/SSM | TQ only on the 16 attention KV slots; 48 companion states remain native; clean boundary capture/rederive plus fingerprinted SSM L2 | Two 1-bit and one ternary exact restart restores pass with native TQ8 + SSM disk; missing partial checkpoints safely full-prefill and repair; long-context partial |
| Other hybrid SSM/GLA | Architecture allow-list plus native companion state and async clean-prefill rederive | Per-family proof required; no name-only Qwen inference |
| Gemma mixed SWA | Native rotating cache for SWA, compatible full-attention lane only; legacy prompt L2 default | UI/DB/argv/warm/restart scoped pass |
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
| Qwen 3.6 35B MXFP/JANG (name has no `MTP`) | Hybrid layout is derived from the real layer graph. TQ encode/decode applies only to eligible attention KV; GDN/SSM companions remain native and are cleanly rederived/restored. This artifact is not assigned an MTP gate. | Cold + two-turn + tool continuation, RAM hit, restart/L2 hit, and forced eviction/reload with coherent output. | OPEN |
| Qwen 3.6 27B `...-MTP` | The same hybrid cache invariant applies, and MTP is eligible because the actual model/bundle name says `MTP`. | MTP depth 1 and 3 launch/health, real draft/accepted counters, cold + two-turn + tool continuation, RAM hit, restart/L2 hit, and forced eviction/reload with coherent output. | PARTIAL: typed D3 Save & Restart now has UI/DB/argv/health parity; tools-on D1-capped multi-turn passes; tools-off D1 and D3 both looped in reasoning; cache restart/eviction rows remain open |
| HY3 MTP | Native MTP depth is the requested value and yields measured accepted draft tokens. Prompt/L2 records include the owning target-model cache state; speculative output is never treated as a substitute cache. | Depth 1 and depth 3 A/B, acceptance and latency counters, multi-turn tool loop, restart restore, and eviction/reload. | PARTIAL: depth-1 speed row exists; depth-3/cache interaction is open |
| MiniMax M2.7 | Ordinary KV attention may use calibrated or correctness-safe TQ storage; parser/reasoning rails must survive multi-turn tool continuation. | Auto and None UI/argv/health A/B, two-turn tool loop, RAM/L2 restore, eviction, long visible answer, and streaming rail continuity. | OPEN |
| ZAYA / CCA | Typed CCA state owns its cache. Generic TQ is forbidden unless a typed CCA codec has source and live parity. | Typed cold/warm/restart/eviction rows plus multi-turn tool and reasoning/content stream. | OPEN current release row |
| Nemotron hybrid | Eligible attention KV may be TQ encoded; non-KV hybrid state remains native and is async clean-prefill rederived/restored. Family selection must come from config/layers, not a name match. | Auto/None A/B, cold + two-turn + tool continuation, L2 restart, eviction, long output, no reasoning leak. | OPEN |
| DSV4 Flash | Native DSA/SWA/CSA/HCA composite and pool codec only; never generic TQ KV. | Composite cache health, cold/warm/restart/eviction, multi-turn agent loop, reasoning/content stream continuity and coherent constrained output. | PARTIAL |
| MiniMax M3 / openPangu | Native typed architecture cache only; generic TQ remains off. | Existing scoped rows plus long-context, restart/eviction, and protocol/streaming completion. | PARTIAL |

Current Qwen 27 settings-parity evidence: the Electron number field published
`3` before blur, SQLite persisted `nativeMtpDepth=3` with override enabled, PID
52719 launched with `--native-mtp-depth 3`, `/health::mtp.effective_depth` was
`3`, and `qwen36-27-mtp-d3-settings-parity.png` visibly records the current
model, PID, Server Settings drawer, and depth. This proves the settings
round-trip only; it does not clear the reasoning-loop or cache-behavior rows.

## Non-negotiable correctness invariants

- No prompt coercion, hidden sampler clamps, forced thinking tags, synthetic
  tool output, invented continuation, or arbitrary output cap may be used to
  make a gate appear green. Fix the layer that owns the defect.
- Assign MTP gates only to actual model/bundle names containing `MTP`. Do not
  infer MTP eligibility from a Qwen, HY, Nemotron, or other family name alone.
- Cache keys and persisted records must cover model/runtime fingerprint,
  architecture codec, quantization parameters, original KV dtype, media salt,
  MTP mode/depth where relevant, and every state needed for exact restore.
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

1. Close the remaining Laguna unsolicited-tool/long-context/latency rows while
   preserving the TQ3 failure and Auto/None/TQ8 A/B artifacts.
2. Re-prove the repaired Bonsai partial-prefix boundary as an actual hit and
   run long-context/eviction; retain the default-sampling loop outlier as a
   reliability row.
3. Close Mistral and DSV4 quality/performance rows, then M3/Pangu long/media
   boundaries and remaining post-tool families.
4. Run the complete settings and protocol matrix through the real Electron
   app and gateway.
5. Run focused and full tests, audit the dirty tree, commit/push only scoped
   files, and merge/integrate the 37-commit branch deliberately.
6. Build, sign, notarize, staple, verify, install-smoke, and publish 1.6.11 only
   after every release-blocking row above is green.
