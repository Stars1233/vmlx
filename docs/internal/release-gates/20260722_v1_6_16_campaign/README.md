# vMLX 1.6.16 release campaign control board

Date: 2026-07-22 (America/Los_Angeles)

Status: `ACTIVE / NOT RELEASE-READY`.

This is the canonical control board for the next Python/Electron vMLX release.
It starts from public v1.6.15 follow-up source commit
`7b940b070dc8ab7afe014561c5094853e16a29c4` on branch
`codex/v1.6.16-release-campaign-20260722`. The immutable v1.6.15 tag and its
signed evidence remain unchanged.

Current tested cutoff: `230c822f2e36967d8f6050b47f820e40b7a21f46`.
The clean source checkout is
`/Users/eric/mlx/vllm-mlx-release-1.6.15`; the live Electron proof checkout on
`erics-m5-max.local` is `/Users/eric/mlx/vllm-mlx-release-1.6.13`. Both were at
the same commit for the retained Laguna proof. The dev launcher executable was
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
- User campaign attachment
  `/Users/eric/.codex/attachments/668d3b34-ca92-49c4-aeb2-5e0ded8f6fc0/pasted-text-1.txt`

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
| `R16-REASONING-RAILS` | `PARTIAL / RELEASE-CRITICAL` | Current Laguna Electron plus Chat/Responses/Anthropic/Ollama live proof and the three scoped source/test repairs are retained below. Auto remains model-owned variable reasoning where the bundle supports it; it must resolve to an enabled reasoning-capable request, not force the model to expose reasoning on every easy prompt. Explicit On and Off must be honored. Still close non-stream behavior, remaining parser families, a live late-reasoning family, and live M3/openPangu/Mistral4 Ollama normalization. Separate reasoning/content, progressive paint/wire events, one truthful terminal, and no inline native tags are mandatory. |
| `R16-AGENTIC-HARNESS` | `OPEN / RELEASE-CRITICAL` | Use a coding-harness-shaped client for no-tool, auto, required, explicit function choice, real result continuation, two-tool interleaving, final synthesis, cancellation, disconnect, injected backend failure, and immediate recovery. Exercise direct and Electron gateway paths. A reasoning-only final, dropped required argument, repeated tool, hallucinated tool result, or false terminal fails the row. |
| `R16-STREAM-METRICS` | `OPEN` | Compare Electron `metrics_json` against raw timed SSE for the same prompt. Report TTFT, prompt processing speed, decode tokens/s after first output token, reasoning tokens, visible tokens, tool/fallback pauses, and wall time separately. Reject terminal-batch answer painting and misleading blended two-pass TPS. |
| `R16-CACHE-HIERARCHY` | `OPEN / RELEASE-CRITICAL` | Prove cold store, resident RAM hit, partial-block reuse, L1 eviction, L2 SSD refault, process restart restore, and safe full-prefill fallback for standard KV, hybrid SSM/GDN, mixed SWA, CCA, M3 sparse, DSV4 composite, and openPangu native prompt disk. With Paged Off and L2 On, partial prefix reuse must come from SSD with zero resident paged bytes. With Paged On, lookup order must use matching RAM blocks first and SSD when absent. Cross-chat and cross-session reuse must not leak unrelated suffixes or media. |
| `R16-SETTINGS-PARITY` | `OPEN / RELEASE-CRITICAL` | Compare bundle defaults to visible Chat Settings, SQLite, IPC/request payload, preview/argv, and engine-resolved kwargs/health. Cover temperature, top-p, top-k including Off/-1/large values, min-p zero, repetition penalty, max output, max context, reasoning Auto/On/Off, tool/reasoning parsers, MTP, modalities, cache toggles, block size/count, RAM percentage, L2 size/path, LAN/port, and Single Model. First use must inherit the bundle; saved per-chat/per-session values must survive restart; reset/Auto must remove the override. Current source trace found that plain Save deliberately persists for the next restart, while Save & Restart has a late-listener delay, ordinary config saves do not broadcast `session:updated`, and Chat Settings can show a stale PID after restart. These subdefects remain open until patched and live-reproved. |
| `R16-CACHE-LABEL` | `VERIFIED-LIVE_SCOPED` | Source `4558dac06` renames Electron `Paged Cache` / `Use Paged KV Cache` to **In-Memory Paged Cache (RAM)** and identifies **Block Disk Cache (L2)** as SSD. It also prevents help-tooltip clicks from toggling checkbox settings. Focused tests, typecheck, normal/minimum-width visual proof, unchanged persisted state/argv, and effective loaded health are named below. Backend flags/defaults/eligibility are unchanged; explicit Off plus disk-only L2 remains a separate cache-behavior regression row. |
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
| MiniMax M3 | Image/video/tool/cache transport scoped pass; exact OCR and REAP remain partial. | Preserve native dense KV plus sparse/MSA lightning-indexer state; do not apply generic TQ to indexer state. Regress image/video salt A/B/A, partial/restart cache, Ollama normalization, long video terminal delay, and agentic tool finalization. |
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

## Cache terminology and tooltip proof

Source commit `4558dac06` changes only user-facing terminology/help and the
shared tooltip click contract:

- section and control: `In-Memory Paged Cache (RAM)`;
- help: Apple unified memory is the fast RAM tier; Block Disk Cache (L2) is the
  persistent SSD tier and may remain enabled when the RAM tier is Off;
- `Tooltip.handleClick` calls `preventDefault()` before `stopPropagation()` so
  a help click nested inside a checkbox label does not activate the checkbox;
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

## Settings restart findings at this cutoff

The live DEBUG-to-INFO negative control initially looked like a lost settings
update. Source trace and the next real Stop/Start show a narrower result:
plain Save deliberately persists for the next restart and does not restart the
running process; the subsequent PID 53268 correctly omitted the DEBUG argv
flag. Three actual UI/session defects remain:

1. Save & Restart subscribes for `session:stopped` only after awaited Stop has
   already emitted it, causing the 15-second fallback wait.
2. Ordinary config updates do not emit `session:updated`, so other mounted UI
   consumers can keep stale config.
3. Chat Settings merges live status/port but not live PID, so it displayed old
   PID 52809 after the engine had restarted as 53268.

Plain Save also needs clearer "Save for Next Restart" wording while running.
No settings row is promoted until the shared source fix, focused tests, and a
new Electron Save & Restart prove prompt PID replacement, argv/config parity,
drawer persistence, Chat Settings PID parity, and prompt restart failure
handling.

## Release stop conditions

Do not package, tag, publish, or describe v1.6.16 as ready while any selected
P0 row is `OPEN`, `FAIL`, or `PARTIAL`, or while the release-cutoff full suites
and signed-app install smoke are missing. A user-approved checkpoint may retain
explicit P1/P2 limitations only when the release notes and this board name them
without converting them into passes.
