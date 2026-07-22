# vMLX 1.6.16 release campaign control board

Date: 2026-07-22 (America/Los_Angeles)

Status: `ACTIVE / NOT RELEASE-READY`.

This is the canonical control board for the next Python/Electron vMLX release.
It starts from public v1.6.15 follow-up source commit
`7b940b070dc8ab7afe014561c5094853e16a29c4` on branch
`codex/v1.6.16-release-campaign-20260722`. The immutable v1.6.15 tag and its
signed evidence remain unchanged.

Latest behavior-bearing tested cutoff:
`e4c6762ce982c1878a82feec142fda929ebc3311`. The first immutable evidence
checkpoint containing the retained MiniMax-M3 artifacts is
`6de9ce8eff206e8a77f65f2ab191c2b3aa971390`; later documentation-only commits
do not silently promote the runtime cutoff.
The clean source checkout is
`/Users/eric/mlx/vllm-mlx-release-1.6.15`; the live Electron proof checkout on
`erics-m5-max.local` is `/Users/eric/mlx/vllm-mlx-release-1.6.13`. Both were at
the same commit for the retained Laguna, settings, and MiniMax-M3 proofs. The dev launcher executable was
`/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`, with `PYTHONPATH` selecting
the synchronized release proof checkout rather than the launcher's source
tree.

The campaign must not be called complete from focused tests, source inspection,
or one successful model turn. A row closes only when its current source trace,
focused regression tests, real Electron computer-use proof, raw protocol proof,
and durable evidence artifacts are all named. Results from an older source may
remain useful controls but are `STALE-REUSE-CANDIDATE` until the owning source
is shown unchanged or the proof is rerun.

## Inputs reconciled into this board

- `AGENTS.md`, `.agents/STATUS.md`, and `.agents/LOG.md`
- `docs/internal/ISSUE-LEDGER.md`
- `../20260716_release_closeout/CURRENT-MATRIX.md`
- `../20260722_active_worklist/README.md`
- `../20260722_global_reasoning_source_audit/README.md`
- `../20260722_lfm_reasoning_tool_stream/README.md`
- `../20260722_settings_defaults_parser/README.md`
- `../20260721_dsv4_direct_quality_boundary/README.md`
- `../20260721_affine_jang_qwen_vl_panel/README.md`
- `../20260721_developer_conversion_lifecycle_current/README.md`
- `../20260722_single_model_unmanaged_engine_sweep/README.md`
- `../20260722_release_checkpoint_1_6_15/README.md`
- `../20260722_public_1615_laguna_provenance/README.md`
- `../20260722_jang_2533_laguna_distribution/README.md`
- `OPEN-ISSUES-RANKED.md`, the living exhaustive 1.6.16 checklist
- User campaign attachment
  `/Users/eric/.codex/attachments/668d3b34-ca92-49c4-aeb2-5e0ded8f6fc0/pasted-text-1.txt`

## Public release truth at campaign start

GitHub currently exposes public `v1.6.15` as the latest release, published
2026-07-22 at 09:05:45 UTC. The annotated tag resolves to
`2dc90921ea8604f4ec4c62e196621007fbb1cbbf`; the source checkout's version
surfaces (`pyproject.toml`, `panel/package.json`, and `latest.json`) remain
1.6.15. The `jjang-ai/vmlx` **source** release has zero attached assets by
design. The separate `jjang-ai/mlxstudio` **distribution** release currently
has four public assets: Sequoia and Tahoe DMGs plus both blockmaps. Its DMG
digests match the retained checkpoint evidence:
`c1bfa6e6b62e2e322461fd549203599f912dc4688e2c31e86d83d7b68c69a4cf`
and `ae5a41c60fd79a39238e03fd74c1df2f5d92a2e57df8a60ff58ee34e248eb4be`.
The updater manifests in both repositories currently name 1.6.15 and those
distribution URLs. This is not an asset gap; it is a two-repository release
layout that must be verified separately again for 1.6.16. The next checkpoint
is a new version, never a retag or overwrite of 1.6.15.

## Non-negotiable proof contract

For every live model generation retained as evidence:

1. Launch the model with the real Electron Sessions **Start** control on the
   M5 Max proof host using CDP `127.0.0.1:9335`; retain the dev-log engine-path
   line, eager-loaded pre-request health, real argv, and a screenshot.
2. Ground architecture, quantization, modalities, parser, template, reasoning,
   MTP, and generation defaults in the exact bundle's `config.json`,
   `generation_config.json`, `tokenizer_config.json`, `chat_template.jinja`,
   and `jang_config.json`. Never infer JANG affine, JANGTQ/MXTQ, or base MLX
   MXFP from a repository name alone.
3. Inspect the visual stream while it runs. Preserve progressive reasoning and
   visible-content mutations, the final frame, SQLite reasoning/content/tool
   fields, warnings, and metrics. Non-empty content is mandatory unless the
   truthful terminal is a tool call or an explicitly incomplete/failed result.
4. Capture raw timed protocol events for the same behavior. Reasoning must use
   `reasoning_content` or the protocol-native reasoning rail; visible content
   must use content/output-text deltas. No raw `<think>`, parser delimiters,
   tool markup, stale reasoning replay, terminal batch-only answer, false
   success terminal, or missing terminal is acceptable.
5. A tool gate must execute the real declared tool, pass the real result back,
   allow more reasoning, then either execute the next authorized tool or emit a
   progressive final exactly once. Never inject a fake tool result or repair
   model output after generation.
6. Cache claims require health counters plus output correctness. A cached row
   is not a pass if it loops, leaks another prompt/media answer, loses native
   companion state, or produces a false terminal.
7. Store sanitized source/test/live artifacts in the owning release-gate
   directory before changing the row to `VERIFIED-LIVE`.

## Ranked blockers

### P0 — shared release-critical behavior

| ID | Status | Work and acceptance |
|---|---|---|
| `R16-REASONING-RAILS` | `PARTIAL / RELEASE-CRITICAL` | Current Laguna Electron plus Chat/Responses/Anthropic/Ollama proof, current M3 Electron/Responses/Ollama proof, and the three scoped source/test repairs are retained below. Auto remains model-owned variable reasoning where the bundle supports it; it must resolve to an enabled reasoning-capable request, not force the model to expose reasoning on every easy prompt. Explicit On and Off must be honored. Still close non-stream behavior, remaining parser families, a live late-reasoning family, and live openPangu/Mistral4 Ollama normalization. Separate reasoning/content, progressive paint/wire events, one truthful terminal, and no inline native tags are mandatory. |
| `R16-AGENTIC-HARNESS` | `OPEN / RELEASE-CRITICAL` | Use a coding-harness-shaped client for no-tool, auto, required, explicit function choice, real result continuation, two-tool interleaving, final synthesis, cancellation, disconnect, injected backend failure, and immediate recovery. Exercise direct and Electron gateway paths. A reasoning-only final, dropped required argument, repeated tool, hallucinated tool result, or false terminal fails the row. |
| `R16-STREAM-METRICS` | `OPEN` | Compare Electron `metrics_json` against raw timed SSE for the same prompt. Report TTFT, prompt processing speed, decode tokens/s after first output token, reasoning tokens, visible tokens, tool/fallback pauses, and wall time separately. Reject terminal-batch answer painting and misleading blended two-pass TPS. |
| `R16-CACHE-HIERARCHY` | `OPEN / RELEASE-CRITICAL` | Prove cold store, resident RAM hit, partial-block reuse, L1 eviction, L2 SSD refault, process restart restore, and safe full-prefill fallback for standard KV, hybrid SSM/GDN, mixed SWA, CCA, M3 sparse, DSV4 composite, and openPangu native prompt disk. With Paged Off and L2 On, partial prefix reuse must come from SSD with zero resident paged bytes. With Paged On, lookup order must use matching RAM blocks first and SSD when absent. Cross-chat and cross-session reuse must not leak unrelated suffixes or media. |
| `R16-SETTINGS-PARITY` | `PARTIAL / RELEASE-CRITICAL` | Compare bundle defaults to visible Chat Settings, SQLite, IPC/request payload, preview/argv, and engine-resolved kwargs/health. Cover temperature, top-p, top-k including Off/-1/large values, min-p zero, repetition penalty, max output, max context, reasoning Auto/On/Off, tool/reasoning parsers, MTP, modalities, cache toggles, block size/count, RAM percentage, L2 size/path, LAN/port, and Single Model. First use must inherit the bundle; saved per-chat/per-session values must survive restart; reset/Auto must remove the override. Commit `951eab25d` plus the retained live gate close the plain-Save wording, restart delay, update broadcast, and live PID subdefects. The broader cross-model/default/reset/failure matrix remains open. |
| `R16-CACHE-LABEL` | `VERIFIED-LIVE_SCOPED` | Source `4558dac06` introduced **In-Memory Paged Cache (RAM)**; the current checkpoint consistently names the persistent tier **Block Disk Cache (SSD / L2)** across Settings, Cache, Perf, capacity text, locales, and CLI help. Live verification found the earlier glyph-only tooltip suppression did not cover wrapper padding; the wrapper now owns the click boundary and both toggles rechecked `true -> true`. Backend flags/defaults/eligibility are unchanged; explicit Off plus disk-only L2 remains a separate cache-behavior regression row. |
| `R16-LAGUNA-MIXED-BIT-PROVENANCE` | `VERIFIED-LIVE_SCOPED` | The reported `(…,576)` weight / `(…,48)` scales / `bits=8` failure is a 6-bit 3072-input affine module being treated as 8-bit, not an attention `g_proj` slice. The exact signed v1.6.15 bundled engine completed cache-disabled S-2.1 JANG_2L and JANG_4M streams; the real signed Electron app completed an S-2.1 UI turn. The stale signed `/Applications/vMLX.app` version 1.6.9 reproduced the error byte-for-byte on the same JANG_4M artifact. Require exact executable and imported-module provenance for every release proof. |
| `R16-SINGLE-MODEL-GATEWAY` | `PARTIAL` | Preserve the scoped unmanaged-engine sweep, rollback, disconnect, backend-loss, and non-stream atomicity fixes. Still run a bounded multi-client soak with repeated real UI model swaps, only one resident process, eager Start-before-first-message load, concurrent route/swap attempts, active-request LAN/port rollback, occupied port, stale target, late loader failure, unload/reload, and recovery across all four protocols. |

### P1 — architecture and family gates

| Family / artifact class | Current truthful boundary | Required 1.6.16 work |
|---|---|---|
| Laguna S-2.1 JANG_2L and JANG_4M | Strong scoped live cache/reasoning/protocol evidence exists; long SWA quality and reasoning reliability remain partial. JANG_4M live speed was about 49–52 tok/s after the dtype repair. | Rerun only shared changed rails. Add >512-token SWA-boundary coherence, disk-only process restart, long agentic tool continuation, saved settings restart, sampler/TPS parity, and bounded eviction. Keep q4 only on full-attention KV; native rotating SWA state is not flattened. |
| Qwen3.6 35B/27B JANGTQ/MXTQ, MTP-named variants | 35B single-tool four-protocol and sampler rows are scoped passes. | Preserve JANGTQ/MXTQ Hadamard-codebook identity, not affine JANG or base MXFP. Run multi-tool/cancellation/non-stream, MTP depth from named/configured artifacts only, cache/media deltas, and one 27B representative. |
| HY3 JANG with named MTP | Native MTP depth-1 scoped API/Electron/cache proof exists. | Prove actual configured depth for each MTP-named artifact, accepted/compressed proposal-token accounting, reuse safety, long/stochastic quality, and current shared rails without hidden MTP enablement for non-MTP artifacts. |
| Bonsai 27B 1-bit and other real Bonsai variants | q8 TQ hybrid storage is the explicit Bonsai exception; exact tool/restart/media scoped rows exist. Long pre-tool reasoning and partial-prefix breadth remain partial. | Re-run shared rails and partial-prefix repair/eviction. Confirm q8 only for Bonsai attention KV, native SSM/GDN rederive/companion state, L2 cross-chat/restart, and no incoherent loops or reasoning-only finalization. Ground each variant in its real Qwen-family config. |
| Ornith / other Qwen-family variants | Not globally promoted by the Qwen3.6/Bonsai rows. | Bundle-ground parser, modalities, MTP, quant format, sampler defaults, hybrid-state topology, Auto reasoning, one multi-turn tool loop, cache hierarchy, and media if advertised. |
| MiniMax M2.7 | Existing scoped proof is M2.7 JANGTQ/MXTQ full-KV text-only, not affine JANG. | Add the requested affine-JANG M2.7 artifact if locally present. M2.7 remains text-only. Prove full-KV q4 TQ, cache hierarchy, Auto/On/Off, all four protocols, tools, settings, and no false VL claim. |
| MiniMax M3 | Current-source real Electron Start/eager load, Auto reasoning/content IPC separation, exact one-tool continuation, raw Responses, and direct/gateway Ollama On/Off/Auto are scoped passes. Native MSA health is explicit; generic TQ remains Off. Live health reports `vl_runtime_available=false`, so prior media transport does not promote this current runtime to a VL pass. | Preserve native dense KV plus sparse/MSA lightning-indexer state; do not apply generic TQ to indexer state. Still regress image/video availability and salt A/B/A after the VL runtime is available, partial/restart/eviction cache, Ollama non-stream/post-tool continuation, long video terminal delay, and REAP variants. |
| DSV4 Flash affine/JANG and any exact JANGTQ artifact | Short reasoning/content stream and native composite cache scoped pass. Medium/long Auto loops in direct generation remain an honest failure boundary. | Preserve MLA plus exact bundle/source-owned local/global compressed branches (SWA/CSA/HCA) and native pool codec; generic TQ stays Off. Regress cold/warm/partial/restart/eviction, long/short coherence, DSML tools, four protocols, and eager materialization. Do not blame or rewrite the official artifact without an independent matched A/B. |
| Gemma 4 JANG_4M/MXFP8, dense/MoE and rotating-SWA variants | Signed 1.6.15 Gemma mixed-SWA q4 cache row passed scoped. Anthropic late reasoning and broader media remain open. | Regress mixed rotating/full KV cache, q4 only where eligible, late-reasoning Anthropic event indices, sampler defaults, image/video/audio only where the exact artifact advertises them, media salt, tools, and long context. |
| Nemotron Nano/Omni | Auto-detection and selected Omni media/audio rows are scoped, not family-wide. | Ground Parakeet/RADIO/audio/VL tensors in sidecars and index. Prove audio and advertised image/video in Electron and raw APIs, parser Auto persistence, hybrid cache/TQ companion state, tool loop, and no modality overclaim. |
| Step Flash / Step 3.7 | Image/video mixed-SWA cache scoped proof exists; cold latency, stochastic quality, PID telemetry, audio breadth are partial. | Regress shared rails, larger video, audio only if bundle-advertised, media salt/restart, post-media tool turns, native reasoning, cache metrics, and restarted PID UI. |
| openPangu 2.0 Flash | Typed composite prompt-disk path exists; generic TQ is intentionally Off. | Preserve exact MLA/DSA/SWA/attention-sink/mHC architecture from bundle/source, not user-memory shorthand. Prove long context, protocol/tool breadth, Paged-Off native disk behavior, sampler defaults, and safe no-generic-TQ health. |
| LFM2.5 | API Chat/Responses reasoning and single-tool continuation passed scoped; Electron, Anthropic/Ollama, restart disk proof remain partial. | Add real Electron Auto reasoning and progressive content, Anthropic/Ollama continuation, hybrid SSM q4 attention cache plus native state rederive, Paged-Off SSD partial/restart, and honest output-budget completion. |
| ZAYA/CCA and Laguna/Nemotron-like CCA/hybrid routes | Selected typed CCA proof exists. | Preserve typed CCA state and eligible KV-only q4 storage. Prove partial/restart/eviction, Auto reasoning, settings, tools, and media where advertised. |
| Mistral 3.5/Pixtral JANGTQ/MXTQ | Current exact artifact loads but blank/whitespace output is retained blocked. | Root-cause model port/quantized coverage/template/parser/media integration. Do not test Mistral base MXFP4 as a substitute. Require coherent Electron and raw API output before any release claim. |

### P2 — product breadth and release engineering

| ID | Status | Work and acceptance |
|---|---|---|
| `R16-MEDIA-BREADTH` | `PARTIAL` | For every advertised modality, use real Electron attachment plus raw Chat/Responses/Anthropic/Ollama or documented protocol-native extension. Prove same-media reuse, same-shape different-media miss, A/B/A salt isolation, restart L2 restore, post-media text/tool recovery, progressive reasoning/content, and no M2.7 VL claim. |
| `R16-JANG-CONVERSION` | `PARTIAL` | Preserve the current affine JANG Developer Tools lifecycle pass. Complete affine profiles/custom mixes and generic JANGTQ/MXTQ conversion, overwrite/low-disk/unwritable/calibration/MoE/resume/error cases, metadata identity, independent artifact reload, coherent multi-turn/tool/media output, and cache compatibility. Keep `/Users/eric/jang` synchronized and version only real package changes. |
| `R16-MAX-BUDGETS` | `OPEN` | Keep max output tokens distinct from model context length across UI, DB, API adapters, preview/argv, and engine. A truthful `incomplete/max_output_tokens` is not a stream failure; an unexplained low hidden cap or context clamp is. |
| `R16-RESPONSIVE-UX` | `PARTIAL` | Preserve current min-width toolbar/drawer fixes. Exercise remaining wait/empty/image states, secondary modals, translated labels, keyboard/screen-reader semantics, stale missing-path repoint/remove UX, and the new cache labels at minimum width. Remove dead/zombie source when an owning path is replaced, with focused tests. |
| `R16-FULL-SUITES` | `STALE-MUST-RERUN-AFTER-CUTOFF` | Run full Python pytest, full panel Vitest, TypeScript typecheck, production Electron build, bundled-Python verification, clean-JANG parity, and release regression manifest after the final source cutoff. Focused suites do not close this row. |
| `R16-RELEASE` | `BLOCKED` | Only after selected P0s and declared P1 cutoff are `VERIFIED-LIVE`: bump every version surface to 1.6.16, build from a clean pinned JANG tree, create separate Sequoia/Tahoe DMGs, Developer ID sign, notarize, staple, deep verify, Gatekeeper verify, install-smoke both, run signed-app Electron/API/cache proofs, tag exact built source, publish GitHub source/MLXStudio/PyPI/Homebrew/updater manifests, re-download public artifacts, verify hashes/version truth, and update this board plus the ledger. |

## Non-repetitive execution order

1. Fix shared reasoning/protocol/stream contracts first. Use one small,
   reliable representative per parser family, then rerun only the affected
   family deltas. Do not repeatedly run the same Qwen/Bonsai single-tool prompt.
2. Fix settings/default persistence and the cache terminology in the same
   Electron drawer pass. Validate a representative set chosen for unusual
   defaults: MiniMax M2.7 top-k 40, openPangu top-p 0.8/top-k 151552, an
   Off/-1 top-k artifact such as HY3/ZAYA, and a non-neutral Nemotron/Gemma row.
3. Run cache archetypes as a matrix: standard KV, hybrid SSM/GDN, mixed SWA,
   typed CCA, M3 sparse, DSV4 composite, and openPangu native. Reuse an existing
   live family proof when the owning source is unchanged; run new cold/warm/
   partial/eviction/restart proofs when cache source changes.
4. Fold media, agentic tools, settings, and cache checks into the same model
   sessions where possible. Each retained prompt must still have an unambiguous
   acceptance marker and truthful negative-control classification.
5. Run conversion, gateway soak, responsive UX, full suites, and packaging only
   after shared runtime defects stop changing the cutoff.

## Progress ledger

| Date | Commit/source | Change | Focused proof | Electron proof | Raw API proof | Status |
|---|---|---|---|---|---|---|
| 2026-07-22 | `7b940b070` | Opened 1.6.16 campaign from post-1.6.15 main; added explicit cache terminology gate | Documentation/source reconciliation only | Not run for this row | Not run for this row | `ACTIVE / NOT READY` |
| 2026-07-22 | `4558dac06` | Renamed the visible RAM tier and fixed tooltip clicks toggling checkbox settings | 301 focused panel tests + typecheck passed on proof host | 1400x900 and 600x760 section/toggle/help inspected; checked state preserved | Existing DB/argv and pre-request loaded health agree on Paged RAM On plus Block L2 | `VERIFIED-LIVE_SCOPED` |
| 2026-07-22 | `ff293d1e7` | Added accumulated-prefix holdback for split reasoning markers across DeepSeek, ThinkXML, and MiniMax M3 parsers | Included in 210/210 combined reasoning/protocol focused tests | Laguna current-source UI showed a separate reasoning rail with no visible marker leakage; natural character-split timing was not forced live | Laguna current-source protocol rows had no visible control-marker leakage | `VERIFIED-SOURCE_TEST_SCOPED / LIVE REPRESENTATIVE` |
| 2026-07-22 | `95e954045` | Balanced Anthropic late-reasoning block transitions, indices, and terminal handling | Included in 210/210 combined reasoning/protocol focused tests | No live late-reasoning family was exercised | Laguna normal reasoning-before-text produced blocks 0/1, 222/99 thinking/text deltas, one terminal | `VERIFIED-SOURCE_TEST_SCOPED / LIVE NORMAL-ORDER ONLY` |
| 2026-07-22 | `230c822f2` | Normalized streaming Ollama reasoning policy for MiniMax M3, openPangu, Mistral4, and Off-history stripping | Included in 210/210 combined reasoning/protocol focused tests | Laguna UI unaffected and coherent | Laguna generic Ollama route produced 222 thinking events, 99 content events, one terminal; named-family normalization is not yet live-proven | `VERIFIED-SOURCE_TEST_SCOPED / LIVE REPRESENTATIVE` |
| 2026-07-22 | `230c822f2` | Current Laguna S-2.1 JANG_2L reasoning, tool-loop, and protocol representative | Bundle/config grounded; combined focused suite 210/210 | Real Electron chat retained one separate reasoning turn and two exact one-tool continuations; third turn restored 664 `paged+tq-native` RAM tokens | Chat, Responses, Anthropic, and Ollama streams were progressive and terminal-complete; exact counts are in the sanitized summary | `VERIFIED-LIVE_SCOPED / GLOBAL ROW PARTIAL` |
| 2026-07-22 | `951eab25d` | Synchronized settings updates, Save & Restart sequencing, and Chat Settings live PID | 503/503 focused panel tests + typecheck + locale parse + diff check | Plain Save kept the current process and exposed the next-restart contract; two real Save & Restarts changed PID/config/argv without the old wait; displayed PID matched SQLite/ps; post-restart exact turn restored 80 disk/TQ-native tokens | Current health matched Paged RAM/L2/q4 policy and recorded two disk promotions/TQ-native hits | `VERIFIED-LIVE_SCOPED / GLOBAL SETTINGS ROW PARTIAL` |
| 2026-07-22 | `f9a4b6838` | Current MiniMax-M3 Ollama normalization, reasoning rail, exact tool loop, and native sparse-cache boundary | Bundle hashes and live health ground affine JANG_2L, 60-layer MSA topology, parser defaults, sampler defaults, and intentional no-generic-TQ policy | Real Sessions Start eagerly materialized PID 60303 before a prompt; UI IPC recorded 245 reasoning updates then 10 content updates and one completion; a separate turn executed exactly one real `file_info` and exact-finaled | Responses emitted 83 reasoning plus eight content deltas and one completion; direct Ollama On/Off and gateway Auto streamed separate rails with one terminal and no native-tag leak | `VERIFIED-LIVE_SCOPED / GLOBAL ROWS PARTIAL` |
| 2026-07-22 | `6de9ce8ef` | Preserved the MiniMax-M3 live artifacts and reconciled source-vs-distribution v1.6.15 release truth | JSON evidence parses; diff check clean | Reuses the exact retained current-source screenshots/IPC capture without claiming a new model run | Reuses the retained raw Responses/Ollama captures without changing the behavior cutoff | `DOCUMENTATION CHECKPOINT` |
| 2026-07-22 | `WORKTREE after aa97a531b` | Corrected Anthropic combined tool-result/follow-up ordering and exact-one Chat finalization; added a reusable four-protocol two-tool harness | 44/44 focused adapter/server/harness tests; evidence JSON parse and diff check passed | Reuses the current Qwen Electron Start/load and two-tool UI proof; no new UI generation is claimed for this raw-wire row | Natural direct Anthropic stream completed two real tools plus progressive exact final with separate reasoning; gateway completed the first tool but truthfully failed its second-tool continuation | `VERIFIED-LIVE_SCOPED / GLOBAL AGENTIC ROW PARTIAL` |
| 2026-07-22 | current cache-name checkpoint after `4ee7befad` | Unified RAM/SSD cache terminology across settings/status/CLI and moved tooltip suppression to the outer wrapper | 293 focused panel tests + typecheck and 94 CLI/cache tests passed on both source boxes | Server Settings, Cache, and Perf inspected through CDP 9335; RAM and SSD wrapper clicks stayed `true -> true`; status surfaces showed `RAM paged + SSD L2` and `Block Disk L2 (SSD)` | Live CLI help distinguishes Apple unified memory and supported SSD-only mode; no new cache-reuse claim | `VERIFIED-LIVE_SCOPED` |
| 2026-07-22 | signed v1.6.15 versus stale signed v1.6.9 provenance control | Classified the Laguna `576/48 bits=8` crash as stale uniform-bit runtime behavior, not S-2.1 `g_proj` slicing | Signed 1.6.15 runtime lines 261-306 derive 6-bit module width; stale 1.6.9 lines 185-199 apply top-level 8-bit uniformly | Real signed 1.6.15 Tahoe Electron Start loaded S-2.1 and exact-finaled `REL1615-LAGUNA-UI-DONE` without warnings | Signed 1.6.15 bundled engine exact-finaled cache-disabled JANG_2L and JANG_4M; signed 1.6.9 negative control reproduced the exact dequant error | `VERIFIED-LIVE_SCOPED / GLOBAL RELEASE ROW PARTIAL` |
| 2026-07-22 | `b6d38eac7`, `e4c6762ce`, JANG `b788273e` / 2.5.33 | Repaired the Python/CLI mixed-affine distribution contract: vMLX now requires JANG 2.5.33, rejects stale runtimes, and logs runtime provenance | JANG full suite 574 passed/37 skipped on both boxes; vMLX focused set 370 passed on both boxes; panel engine-path set 7 passed | Real Start loaded S-2.1 under the synchronized release checkout and public 2.5.33 wheel; separate reasoning/content and exact one-tool continuation persisted without warnings | Current Responses Auto/Off and Chat streams separated reasoning/content, terminalized truthfully, had no inline marker leak, and warm-hit q4 native cache | `VERIFIED-LIVE_SCOPED / GLOBAL RELEASE ROW PARTIAL` |

## Cache terminology and tooltip proof

Source commit `4558dac06` introduced the plain-language RAM label. The current
checkpoint completes the terminology/status surfaces and corrects the shared
tooltip click boundary:

- section and control: `In-Memory Paged Cache (RAM)`;
- help: Apple unified memory is the fast RAM tier; Block Disk Cache (SSD / L2)
  is persistent and may remain enabled when the RAM tier is Off;
- the earlier inner-glyph `Tooltip.handleClick` fix did not cover clicks on
  wrapper padding. The outer wrapper now owns `preventDefault()` and
  `stopPropagation()`, so every help-target click nested inside a checkbox
  label avoids the label's toggle action;
- all five locale section labels were updated; backend config field
  `usePagedCache` and CLI flag `--use-paged-cache` were deliberately retained.

Focused proof on `erics-m5-max.local`:

- `tests/settings-flow.test.ts` plus `tests/i18n-consistency.test.ts`: 301/301
  passed;
- `npm run typecheck`: passed;
- normal-width visual artifact: `cache-label-normal.png`;
- 600x760 visual artifact: `cache-label-minwidth.png`; DOM width was 600/600,
  label rectangle `x=290..565`, tooltip `x=253.76..541.76`, and the checkbox
  remained checked before and after opening help;
- persisted session config retained `usePagedCache:true`;
- live PID 81068 argv retained `--use-paged-cache`, block size 64, 1000 max
  blocks, and Block Disk L2;
- after a real `/admin/wake`, `cache-label-health.json` records
  `model_loaded=true`, `last_request_time=null`, paged RAM enabled, disk-only
  false, 63,936-token configured RAM capacity, and Block L2 present.

Current-checkpoint follow-up proof is retained separately under
`docs/internal/release-gates/20260722_cache_names_ram_ssd/`: both outer
wrapper clicks kept the checked value `true -> true`; Cache rendered
`Block Disk Cache (SSD / L2)`; Perf rendered `RAM paged + SSD L2` and
`Block Disk L2 (SSD)`; and the focused test sets passed on both source boxes.

No model generation was run for this UI/settings proof, and no cache reuse or
output-coherence claim is added here.

## Current Laguna reasoning, protocol, tool-loop, and cache boundary

Cutoff `230c822f2` was synchronized to the live proof checkout before the real
Electron Start action loaded
`/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L` as PID 53268 on port
8018. Its actual bundle identifies `model_type=laguna`, JANG affine-mixed
weights (not JANGTQ/MXTQ), 48 layers with one full-attention layer followed by
three sliding-window layers, window 512, `deepseek_r1` reasoning, `glm47`
tools, text-only modality, and temperature/top-p/top-k/min-p defaults
`1/1/20/0`. The visible Chat Settings controls matched those sampling defaults.

The retained three-turn Electron chat has:

- row 183: exact `R16-LAG-UI-DONE`, a separate 404-character reasoning rail,
  no tool call, no warning;
- row 186: exactly one `file_info(panel/package.json)`, a real result reporting
  5.2 KB, exact `R16-LAG-UI-TOOL-DONE SIZE=5.2 KB`, no warning;
- row 189: exactly one visibly rendered `file_info(pyproject.toml)`, exact
  `R16-LAG-UI-TOOL-VISIBLE-DONE`, no warning, and 664 restored
  `paged+tq-native` tokens.

Raw current-process protocol summaries:

- Chat Completions: 228 reasoning deltas, 98 content deltas, one stop, one
  usage event, no visible marker leak. The model added explanatory prose before
  the requested marker, so transport passed while strict exact format did not.
- Responses: 21 progressive content deltas and one completed terminal. The
  model selected its direct visible rail for the easy prompt.
- Anthropic: a strong prompt produced thinking block 0 then text block 1, 222
  thinking deltas, 99 text deltas, and one `message_stop`.
- Ollama: the same strong prompt produced 222 `message.thinking` events, 99
  content events, and one terminal.

Easy Chat/Anthropic/Ollama A/B prompts all selected direct visible output, so
the absence of a private rail on those turns is a model-variable Auto control,
not evidence of adapter loss. The live current process has 5,132 L2 block
tokens on disk but zero disk hits; this row proves only the observed resident
RAM/native-TQ warm hit, not SSD refault or restart restore.

Durable evidence:

- `laguna-reasoning-protocol-ui-summary.json`: sanitized counts, hashes, exact
  visible finals, DB-derived tool metadata, metrics, and cache boundary;
- `laguna-ui-reasoning.png`: collapsed private rail plus exact visible final;
- `laguna-ui-tool-visible.png`: both real tool cards, exact finals, and metrics.

The raw SSE/NDJSON captures and full private reasoning text are deliberately
not committed.

## Settings restart fix and live proof at this cutoff

The live DEBUG-to-INFO negative control initially looked like a lost settings
update. Source trace and the next real Stop/Start show a narrower result:
plain Save deliberately persists for the next restart and does not restart the
running process; the subsequent PID 53268 correctly omitted the DEBUG argv
flag. Commit `951eab25d` then repaired the three actual UI/session defects:

1. Save & Restart now proceeds directly from the already-awaited Stop promise
   to Start instead of subscribing after the stop event and waiting 15 seconds.
   The parallel full settings screen no longer adds its fixed 2.5-second delay.
2. Ordinary config updates read back the durable session and emit
   `session:updated`, allowing every mounted consumer to refresh.
3. Chat Settings now treats the current `SessionsContext` PID as authoritative,
   including explicit `undefined` when stopped, rather than retaining stale
   detailed-session state.

Both running-session settings surfaces now label the non-disruptive action
`Save for Next Restart`; Save & Restart remains available after that save.
Focused proof passed 503/503 panel tests, TypeScript typecheck, locale parsing,
and `git diff --check` on the synchronized proof checkout.

Live Electron proof changed INFO to DEBUG using plain Save: PID 56197 and its
argv stayed unchanged, while Save & Restart produced PID 57093 with
`--log-level DEBUG`. The same path then saved INFO without changing PID 57093;
Save & Restart produced PID 58440, whose process start was within three seconds
of the click and whose loaded health was observed within 18 seconds. The new
argv omitted the DEBUG flag. Chat header, drawer, SQLite, and `ps` all agreed
on PID 58440 and INFO, with the visible success message
`Restarted with new settings.`

A fresh post-restart Electron turn exact-finaled `R16-SET-POST-DONE` with no
warning or tool call and restored 80 `paged+disk+tq-native` tokens. Health
recorded two disk promotions and two TQ-native hits under the Laguna mixed-SWA
q4 policy. This closes the settings/restart/PID subdefects and one Paged-On
restart refault; it does not close cross-model settings defaults, failure
recovery, Paged-Off disk-only partial reuse, or eviction.

Durable evidence: `settings-restart-pid-proof.json`,
`settings-debug-restart.png`, `settings-info-select.png`, and
`settings-post-restart-turn.png`.

## Current MiniMax-M3 native-cache, reasoning, Ollama, and tool boundary

At source cutoff `f9a4b6838`, the real Electron Sessions **Start** control
loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiniMax-M3-Coder-Small` as PID
60303 on port 8003 before any prompt. Pre-request RSS was approximately 52 GB,
the card visibly reported Running, and the dev log named
`/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`. This is eager
materialization evidence rather than a load-on-first-message observation.

The exact bundle is `minimax_m3_vl` with affine mixed JANG/JANG_2L weights,
not JANGTQ/MXTQ. Live health reports a 60-layer native
`minimax_m3_msa_v1` cache: dense KV layers 0-2 and sparse MSA layers 3-59 with
`attention_kv`, `msa_idx_keys`, and `absolute_block_index`. Generic
TurboQuant KV is intentionally Off because it cannot preserve indexer state;
Paged RAM and Block Disk L2 are enabled. JIT is Off for the Lightning-Indexer
path. The UI matched bundle defaults temperature 1.00, top-p 0.95, top-k Off,
Auto reasoning, MiniMax-M3 tool/reasoning parsers, and the 1,048,576-token
context. Live health also reports `vl_runtime_available=false`; this run makes
no current image/video claim despite the bundle's vision tensors.

Electron Responses proof:

- an injected IPC observer saw 245 progressive reasoning updates grow from 3
  to 620 characters, then ten content updates grow from 1 to 18 characters,
  then one completion with the same 620/18 split;
- the visible final was exact `R16-M3-UI-IPC-DONE`, with no inline native
  marker;
- a separate fresh chat emitted 233 reasoning characters, exactly one real
  `file_info({"path":"panel/package.json"})`, the real 5.2 KB result, and
  exact `R16-M3-UI-TOOL-DONE SIZE=5.2 KB` with no warning;
- that tool turn reported a 256-token `paged` hit.

Raw protocol proof:

- Responses: 83 reasoning-summary deltas, eight output-text deltas, exact
  `M3-RESP-STREAM-DONE`, and one `response.completed` status `completed`;
- direct Ollama `think:true`: 512 thinking deltas followed by nine content
  deltas and one `done:true/stop`, exact `M3-OLLAMA-ON-DONE`;
- direct Ollama `think:false`: zero thinking deltas, nine content deltas, one
  terminal, exact `M3-OLLAMA-OFF-DONE`;
- Electron gateway Ollama Auto: 160 thinking deltas, nine content deltas, one
  terminal, exact `M3-GATEWAY-AUTO-DONE`;
- none of the visible streams contained `<think>` or `<mm:think>` markers.

The transient SQLite placeholder changes rowid when final `INSERT OR REPLACE`
persists the completed message. That artifact did not indicate inline
rendering: current UI IPC proves reasoning and content were separate while
streaming. Still open for M3 at this cutoff are native sparse-cache
partial/eviction/process-restart proof, VL runtime availability and media salt,
Ollama non-stream/post-tool continuation, and REAP variants.

Durable evidence: `m3-reasoning-ollama-ui-summary.json`,
`m3-loaded-session.png`, `m3-ui-reason.png`, and `m3-ui-tool.png`.

## Cache tier naming and tooltip boundary at the current cutoff

The Electron settings/status surfaces now use `In-Memory Paged Cache (RAM)`
for the fast Apple unified-memory tier and `Block Disk Cache (SSD / L2)` for
the persistent tier. Command-line flag names remain stable. The live Server
Settings, Cache, and Perf views were inspected through CDP 9335; the latter
reported `RAM paged + SSD L2` on a real Laguna mixed-SWA/q4 process.

Live verification also found a real interaction defect: clicking padding in a
tooltip wrapper could toggle its enclosing checkbox because suppression was
owned only by the inner `?` glyph. The event boundary now lives on the outer
wrapper, and both RAM and SSD/L2 tooltip clicks rechecked `true -> true` while
showing the tooltip. Focused panel tests/typecheck and CLI/cache tests passed
on both synchronized source checkouts. This is a terminology/interaction
closure only; it does not substitute for the Paged-Off partial/restart or
eviction gates below. Evidence is in
`docs/internal/release-gates/20260722_cache_names_ram_ssd/`.

## Release stop conditions

Do not package, tag, publish, or describe v1.6.16 as ready while any selected
P0 row is `OPEN`, `FAIL`, or `PARTIAL`, or while the release-cutoff full suites
and signed-app install smoke are missing. A user-approved checkpoint may retain
explicit P1/P2 limitations only when the release notes and this board name them
without converting them into passes.
