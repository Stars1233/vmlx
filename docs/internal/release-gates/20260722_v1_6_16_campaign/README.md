# vMLX 1.6.16 release campaign control board

Date: 2026-07-22 (America/Los_Angeles)

Status: `ACTIVE / NOT RELEASE-READY`.

This is the canonical control board for the next Python/Electron vMLX release.
It starts from public v1.6.15 follow-up source commit
`7b940b070dc8ab7afe014561c5094853e16a29c4` on branch
`codex/v1.6.16-release-campaign-20260722`. The immutable v1.6.15 tag and its
signed evidence remain unchanged.

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
| `R16-REASONING-RAILS` | `OPEN / RELEASE-CRITICAL` | Close Chat Completions, Responses, Anthropic, and Ollama stream and non-stream reasoning behavior across representative parser families. Auto remains model-owned variable reasoning where the bundle supports it; it must resolve to an enabled reasoning-capable request, not force the model to expose reasoning on every easy prompt. Explicit On and Off must be honored. Separate reasoning/content, progressive paint/wire events, one truthful terminal, and no inline native tags are mandatory. Include the known Ollama M3/Mistral4 normalization, Ollama Off-history stripping, Anthropic late-reasoning block-index, and partial-delimiter holdback source findings. |
| `R16-AGENTIC-HARNESS` | `OPEN / RELEASE-CRITICAL` | Use a coding-harness-shaped client for no-tool, auto, required, explicit function choice, real result continuation, two-tool interleaving, final synthesis, cancellation, disconnect, injected backend failure, and immediate recovery. Exercise direct and Electron gateway paths. A reasoning-only final, dropped required argument, repeated tool, hallucinated tool result, or false terminal fails the row. |
| `R16-STREAM-METRICS` | `OPEN` | Compare Electron `metrics_json` against raw timed SSE for the same prompt. Report TTFT, prompt processing speed, decode tokens/s after first output token, reasoning tokens, visible tokens, tool/fallback pauses, and wall time separately. Reject terminal-batch answer painting and misleading blended two-pass TPS. |
| `R16-CACHE-HIERARCHY` | `OPEN / RELEASE-CRITICAL` | Prove cold store, resident RAM hit, partial-block reuse, L1 eviction, L2 SSD refault, process restart restore, and safe full-prefill fallback for standard KV, hybrid SSM/GDN, mixed SWA, CCA, M3 sparse, DSV4 composite, and openPangu native prompt disk. With Paged Off and L2 On, partial prefix reuse must come from SSD with zero resident paged bytes. With Paged On, lookup order must use matching RAM blocks first and SSD when absent. Cross-chat and cross-session reuse must not leak unrelated suffixes or media. |
| `R16-SETTINGS-PARITY` | `OPEN / RELEASE-CRITICAL` | Compare bundle defaults to visible Chat Settings, SQLite, IPC/request payload, preview/argv, and engine-resolved kwargs/health. Cover temperature, top-p, top-k including Off/-1/large values, min-p zero, repetition penalty, max output, max context, reasoning Auto/On/Off, tool/reasoning parsers, MTP, modalities, cache toggles, block size/count, RAM percentage, L2 size/path, LAN/port, and Single Model. First use must inherit the bundle; saved per-chat/per-session values must survive restart; reset/Auto must remove the override. |
| `R16-CACHE-LABEL` | `OPEN-UX` | Rename Electron `Paged Cache` / `Use Paged KV Cache` to **In-Memory Paged Cache (RAM)**. Help text must say it is the fast unified-memory prefix/block tier and distinguish it from **Block Disk Cache (L2)** on SSD. Do not call it GPU RAM on Apple silicon, rename backend flags, change defaults, enable unsupported architectures, or break explicit Off plus disk-only L2. Verify wording and controls at normal and minimum window widths plus UI/DB/preview/argv/health parity. |
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

## Release stop conditions

Do not package, tag, publish, or describe v1.6.16 as ready while any selected
P0 row is `OPEN`, `FAIL`, or `PARTIAL`, or while the release-cutoff full suites
and signed-app install smoke are missing. A user-approved checkpoint may retain
explicit P1/P2 limitations only when the release notes and this board name them
without converting them into passes.
