# vMLX Issue Ledger (ADDITIVE — never delete rows, only append + update Status)

Rules of this ledger (Eric directive 2026-07-10):
- ADDITIVE: every issue, potential issue, smaller nit, and regression gets a
  row. Rows are never removed; when resolved, flip Status and add the
  resolving commit + the LIVE proof that closed it.
- LIVE-PROOF-ONLY: an issue is CONFIRMED or CLOSED only by live runtime
  evidence — load the model, run multiturn, look at actual output + cache
  stats + stream deltas + logs. Code-reading yields a LEAD (evidence
  "code-only"), never a confirmation or a closure.
- Both reviewers (Claude + codex gpt-5.6-sol) append here. Cross-reference the
  per-area notes in CODEX-AUDIT-2026-07-10/ and REAUDIT-round2/.

Severity: C / H / M / L(nit) / P(otential, unproven).
Evidence: LIVE / code-only.
Status: OPEN / FIXED / VERIFIED-LIVE / WONTFIX / NOT-A-DEFECT / BUNDLE / BASELINE.

## 2026-07-18 public checkpoint override

- Public release checkpoint: `v1.6.11` is released. Packaged engine/source
  commit: `95b2caa956c592a9caa706f2a790dcd5664721b7`. Final annotated tag,
  `origin/main`, closeout branch, and evidence head:
  `df244c4a858df3894fa3911b270d6d1b175966d6`.
- Current live release-surface verification is `pass` with no failed checks;
  signed/notarized Sequoia and Tahoe DMGs, PyPI, GitHub releases, updater feeds,
  and Homebrew are documented in
  `docs/internal/release-gates/20260718_v1_6_11_release/`.
- The release closes the packaging/publication gate only. It does not
  implicitly close any `OPEN`, `FAIL-LIVE`, or `PARTIAL` runtime row in this
  additive ledger. Those rows remain active post-release work.

## 2026-07-19 current-source full-suite and bundle-safety checkpoint

- `fb9689968` repairs two stale full-suite contracts: the MCP policy test now
  pins both real Responses/Chat built-in-tool injection guards, and the
  worker-dequant fake implements the production memory-cache capacity API.
- `92935ada5` fixes a real packaging defect: `bundle-python.sh` no longer
  deletes the whole tracked `build/` proof tree while cleaning setuptools
  scratch. A real clean-JANG rebuild preserved a build sentinel.
- Current source results: Python **6,125 passed / 96 skipped / 92 deselected**;
  panel **2,312 passed / 3 skipped**; typecheck, bundled hash/import verifier,
  and clean-JANG production build all pass.
- Canonical current-regression status remains `open`; absent MiMo artifacts,
  staged packaged-integrity/signing drift for this post-release head, and the
  retained live model/protocol/media/UI rows remain explicit blockers or
  partials. No model generation ran in this source/build checkpoint.
- Evidence:
  `docs/internal/release-gates/20260719_full_suite_checkpoint/`.

## Fixed + verified this cycle (v1.6.6 lane)
| ID | Sev | Issue | Status | Fix | Live proof |
|----|-----|-------|--------|-----|-----------|
| C1 | C | non-stream answer pass took a fresh full max_tokens (total > cap) | VERIFIED-LIVE | dc8e1600d + round2 shared-cap | small-max_tokens reasoning req: completion <= cap |
| C2 | C | paged block hash collisions (a:bc == ab:c; 1 == "1") -> cross-condition KV | VERIFIED-LIVE | 8245a6107 + round2 key recursion | 7 collision tests + dtype + int/str-key |
| H1 | H | MLLM decode passed raw logits to a logprob sampler | FIXED | 5520b1759 | token0 + decode share normalized contract |
| H2 | H | L2 cache key ignored weight shards (in-place swap reused stale KV) | VERIFIED-LIVE | a76d217ab | key changed on weight-index mutation |
| H3 | H | disk L2 bf16->fp16 cast lossy / overflow-to-inf (F21 root cause) | VERIFIED-LIVE | 8245a6107 | Hy3 cold/warm/warm/restart byte-identical Paris. |
| H4 | H | invalid MTP depth -> D3; blocked tuning leaked depth; false gate advert | VERIFIED-LIVE | c963e37a0 | temp0.9 Hy3 uses rejection-sample path |
| H5 | H | fail-open detection (JANG/config/hybrid/MLA/TQ-unknown) | FIXED | 5520b1759 | fail-closed contract tests |
| MTP | H | Hy3 MTP never engaged (2 stale blocks); depth policy | VERIFIED-LIVE | 2b770a3fd + 47f9758a2 | d1 +10-14%, autodetect effective_depth=1 |
| EOS | H | multi-eos stop set inert for :opensource tokenizers | VERIFIED-LIVE | 47365762f | eos_token_ids installed live |
| F20 | H | memory_cache returned stored ref on clone-gate fail | VERIFIED-LIVE | 5f1e241d2 | hybrid clone isolation |

## OPEN — engine
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| UI-minwidth-i18n | M | Minimum-width Electron controls and fixed-width settings drawers could be clipped; observed Chat Settings and icon accessibility strings remained English in Korean | VERIFIED-LIVE-SCOPED (current source) | Source contracts + current-source Electron at 600x760 + before/after screenshots + CDP geometry + dismissed native confirm + theme-cycle and keyboard probes | The retained signed v1.6.13 screenshot clips toolbar controls beyond 600 px. Before the current repair, the Korean Server drawer occupied `x=216..600` inside an `x=260..600` overflow-hidden main pane, hiding 44 px and label prefixes. Shared responsive wrappers and `w-full max-w-*` drawer roots now keep Server at `x=260..600` and Chat at `x=280..600`; toolbar document width is 600/600 with no clipped controls. The observed Chat/settings strings are catalog-backed, the real bulk-delete action opened a localized Korean native confirm and was dismissed without deleting three chats, and theme/voice icon controls now expose localized title/aria names. Live theme clicks traversed dark/light/system and restored dark; 25 unique base-Chat Tab stops were named and fully within 600x760 with zero failures. Complete panel suite: 2,340 passed / 3 skipped; typecheck and direct Electron source build pass. Still open: forced transient wait/empty/image states, remaining secondary custom modals, drawer/modal-specific keyboard and full screen-reader semantics, and next signed-app repeat. Evidence: `docs/internal/release-gates/20260719_minwidth_locale_drawers/`, `docs/internal/release-gates/20260720_minwidth_drawer_followthrough/`, and `docs/internal/release-gates/20260720_minwidth_accessibility_followthrough/`. |
| #78 | M | TurboQuant live encode inert engine-wide (compress_after=0; no config param) | VERIFIED-LIVE | LIVE (Hy3/Qwen3.6/Gemma) | `compress_after` is threaded through loader/config/clone paths and real codec counters advance. All three default OFF: Hy3 coherence gate failed; Qwen/Gemma resident-memory gates failed. Capabilities/logs are truthful and make no 5x claim. |
| A-cap | M | /v1/capabilities returns 404 (real path /v1/models/{id}/capabilities) | VERIFIED-LIVE | LIVE | `/v1/capabilities` returned 200 for loaded Gemma and exposed truthful TQ/native-cache data. |
| A-omni | M | attention-only live-TQ allow-list covers Qwen3.6 not Nemotron-Omni | VERIFIED-LIVE | LIVE | Nemotron: 6 attention TQ + 23 native SSM slots; warm 702-token `paged+ssm+tq`, exact output, SSM companion entry present. |
| A-lag | M | Laguna cannot reach documented SWA opt-in loader path | VERIFIED-LIVE | LIVE | Dedicated loader reached opt-in: 10 full-attention TQ + 30 RotatingKVCache; warm 601-token `memory+tq`, exact output. |
| seed | L | text chat/completions ignore `seed` (image endpoints only) | VERIFIED-LIVE | LIVE | Request-local keyed sampling wired through chat/completions/Responses/Anthropic/Ollama and MTP. Fresh-cache same-seed completion byte-identical, different seed diverged; SSE produced visible `STREAM-SEED-OK`. |
| #45 F1 | M | q4 stored-prefix cold != warm first-token divergence | OPEN (awaits Eric) | LIVE (3 models) | inherent to q4 store; needs per-family gate |
| M3-stream-zero | H | MiniMax-M3 tools-enabled media turn could buffer invalid XML, erase the visible answer, and render a completed zero-tool card | VERIFIED-LIVE (current source) | LIVE Electron + focused tests | Finalizer hides invalid control suffix; renderer hides completed speculative zero-tool heartbeat; M3 late answer pass runs only after parser proves no valid call. Current image/video turns produce visible grounded output with no zero-tool card, and the genuine-tool post-fix row still executes exactly one `file_info`. |
| M3-nonstream-parity | H | MiniMax-M3 non-stream Chat/Responses may skip its bounded visible-answer fallback whenever tools are merely available | VERIFIED-LIVE (current source) | 2 functional tests + live API after Electron restart | Non-stream Chat/Responses now late-arm the bounded M3 answer pass only when no schema-valid call exists. Live tools-available/no-call markers returned exactly `MM3-NONSTREAM-RESP-DONE` and `MM3-NONSTREAM-CHAT-DONE`. |
| M3-chat-namespace-stream | H | MiniMax-M3 raw Chat could stream a terminally truncated `]<]minimax[>` namespace separator as visible content immediately before an otherwise valid tool call | VERIFIED-LIVE (current source) | Source trace + 46 focused tests + LIVE Electron + raw Chat/Responses SSE | The M3 parser now strips the complete namespace separator and its observed one-character terminal truncation; the shared Chat marker boundary buffers the namespace before any byte escapes. PID 67052 -> 67856 via visible Save & Restart. Raw Chat first pass emitted 24 reasoning deltas, zero content, one valid `file_info(path)` call, `tool_calls`, and DONE; its continuation emitted 20 progressive content deltas. Responses emitted one valid call and a 75-content-delta continuation. Fresh Electron row 138 executed the tool once and exact-finaled `MM3-NS-UI-FIX-DONE SIZE=5.2 KB` with no warning. Evidence: `docs/internal/release-gates/20260718_mm3_chat_namespace_stream/`. |
| M3-cache-settings | M | M3 paged/block-L2 UI defaults and setting changes could diverge from spawned CLI/health | VERIFIED-LIVE (current source) | LIVE Electron settings + process argv + health | Default Session Settings matched CLI/health; `15% -> 12% -> 15%` Save & Restart cycle changed process argv and L1 ceiling while preserving native paged/block-L2. |
| M3-image-exact-ocr | L | M3 grounded image response is character-inexact on deterministic marker (`1` read as `I`) | OPEN | LIVE Electron | Image transport and tool path passed; do not call exact visual extraction green until the marker is byte-exact. |
| M3-REAP32-prefill-safety | H | MiniMax-M3 REAP32 loads at 105.4 GiB against a 107.52 GiB Metal ceiling, then can reboot the host on its first text/tool prefill | FAIL-LIVE / PARTIAL-FIX | LIVE Electron + `/health` + host boot time + DB + 15 focused source tests | Two controlled Electron first requests left empty assistant rows and coincided with full host reboots. The generic 99% guard and output-token projection were insufficient at 98.0% occupancy. Current source refuses any M3 request with under 3 GiB Metal headroom and no longer lets an already-over-threshold model use baseline forgiveness. A third live load was deliberately not attempted; current 503 behavior is source-tested but not live-verified. |
| Bonsai-1bit-current | H | Bonsai 1-bit current-source runtime needed live quant/cache/coherence/parser proof | VERIFIED-LIVE (current source) | LIVE Electron + `/health` + Responses API | UI-selected `jangq-ai/Bonsai-27b-1bit-JANG` loaded as `JANG_AFFINE_1BIT` actual bits `1.1128`; exact marker and exact recall passed with `paged+ssm`; health showed block-L2 and SSM companion disk stores; Responses and Electron both executed exactly one `file_info`. |
| Bonsai-ternary-current | H | Bonsai ternary current-source runtime needed live quant/cache/coherence/parser proof | VERIFIED-LIVE (current source) | LIVE Electron + `/health` + Responses API | UI-selected `jangq-ai/Bonsai-27b-Ternary-JANG` loaded as `JANG_AFFINE_TERNARY_2BIT` actual bits `2.0959`; exact marker and exact recall passed with `247 paged+ssm cached`; health showed block-L2 and SSM companion disk stores. Current Electron post-tool row persisted exactly one `file_info`, one result, exact `BT-POSTTOOL1-DONE`, one reasoning segment, no warning, and `31.3 t/s`. |
| Bonsai-ui-tool-exec | M | Bonsai Electron built-in tool execution could complete the tool but end in repeated reasoning-only retries with no visible answer | VERIFIED-LIVE exact-once contract / PARTIAL broader performance (current source) | Source trace + 423 parser/server tests + 578 engine-audit checks + LIVE Electron/DB/argv/health | Raw current-source traces proved the model emitted a schema-valid `file_info` call early, then repeated 10-46 native call blocks until 4,335/6,316 output tokens. Qwen remains globally multi-call capable; the server now enables early stop only when the latest user request names one exposed tool and explicitly requires it exactly once. The parser validates required arguments before stopping. With current source, TQ-off rows completed in 24.9s/1,195 tokens and 7.0s/279 tokens; after restoring UI Auto/TQ, six consecutive fresh Electron rows completed in 4.2-7.0s/115-244 tokens, each with one real `file_info`, one result, and an exact final marker. |
| Bonsai-reasoning-token-variance | M | Default-thinking Bonsai can spend thousands of hidden/reasoning/tool-prefix tokens before an otherwise valid single tool call | PARTIAL (post-call repetition bounded; pre-call sampling variance remains) | LIVE Electron + engine logs + TQ on/off A/B + focused Responses regression | Pre-fix current rows generated 2,422, 4,335, and 6,316 tokens; the raw 4,335-token TQ-off trace contained its first valid call at character 3,092 and still generated another ~13.5K characters. This falsified TQ/cache as the root cause. The exact-once stream gate prevents post-call repetition without changing ordinary Qwen multi-call turns. Current TQ-auto live rows are 115-244 tokens/4.2-7.0s across six repeats, but one TQ-off row still needed 1,195 tokens before its first valid call, so general reasoning latency is not performance-green. |
| Mistral35-MXFP4-current | H | Mistral Medium 3.5 MXFP4 needed truthful text/VL capability, cache-setting parity, cache reuse, and tool/instruction behavior proof | PARTIAL-LIVE | Source trace + 358 focused panel tests + LIVE Electron/DB/argv/health | Detector now reports the implemented Ministral3 text runtime (`isMultimodal=false`, `forceTextOnly=true`, paged default), while Mistral Small 4 remains VLM. Reset/save persists paged=1, block-L2=1, legacy-L2=0, multimodal=0; PID argv and health match. Exact `4` passed. Identical fresh chats restored 1,240/1,241 tokens from RAM (`paged`) and after restart (`paged+disk`, 20 disk hits, q4 KV). Behavior remains red: broad 33-tool prompt repeated `2026`; a reduced one-tool/Search route passed exactly; the cache probe answered `I understand.` instead of the required marker. No VL runtime is claimed. |
| Step-JANGTQ-attention | H | The installed generic JANGTQ P18 fusion replaced Step-3.7 attention without preserving its post-reshape q/k norms or head-wise `g_proj` gate, producing deterministic reasoning soup | VERIFIED-LIVE (current source) | Historical source A/B + 129 focused tests + LIVE Electron/log/DB/health | The vMLX loader now inspects the installed P18 semantics and restores native `Step3p5Attention.__call__` only when the fusion lacks both Step invariants. The visible log records that guard. Electron row 1406 returned exact `4`; row 1418 made one real `file_info`, one result, one concise reasoning segment, exact `STEP-TQ-TOOL4-DONE`, and 192 `paged+mixed_swa` cached tokens. |
| post-tool-cross-model | H | The shared Electron post-tool loop can repeat reasoning, lose final content, misreport TPS, retain stale warnings, or repeat tools; every parser/model family needs a separate live row | OPEN/PARTIAL | Current source + LIVE Electron matrix | Bonsai 1-bit, Bonsai ternary, HY3, Laguna, LFM, Qwen3.6, Gemma4, MM2.7, M3, Nemotron, both Step quant paths, and DSV4 now have exact one-tool/final rows. Current Nemotron 1394, MM2.7 1397, DSV4 1400/1403, and Step JANGTQ_K 1418 pass without repeated reasoning or missing final content. MiMo and other configured families remain untested, so this campaign-wide row stays partial. See `docs/POST-TOOL-CROSS-MODEL-MATRIX-2026-07-15.md`. |
| post-tool-warning-lifecycle | M | A successful answer-only recovery could retain an intermediate `visible answer is empty` warning in the final assistant row/UI | VERIFIED-LIVE (current source) | 48 focused tests + LIVE DSV4 Electron/DB | `dropSupersededRecoveryWarnings` narrowly removes only superseded current-response empty-answer diagnostics after visible content exists. DSV4 pre-fix row 1301 showed exact final plus stale warning; post-fix row 1304 has one tool/result, visible final content, and `warnings_json=null`. The model misspelled the strict marker, so DSV4 fidelity remains separately open. |
| LFM2-explicit-tool-placeholder | H | LFM2 native-template shortcut treated placeholder argument examples as concrete, causing Electron `file_info` to emit malformed `path=': '` and repeat calls | VERIFIED-LIVE (current source) | Source trace + 8 focused tests + LIVE Electron/DB | Explicitly named LFM tools now force request-bound fallback examples; scalar values such as `path panel/package.json` replace `VALUE_HERE`. Pre-fix broad/Search-only rows made 2/3 calls with malformed arguments. Post-fix broad File/Search/Shell row 1325 made exactly one `file_info({"path":"panel/package.json"})`, one result, exact `LFM-POSTTOOL5-DONE`, and no warning. |
| LFM2-MXFP4-native-reasoning-tool | H | Base-MLX MXFP4 LFM2.5 must use its native reasoning rail; its current required-tool turn can still emit malformed arguments and replay stale visible content | PARTIAL-LIVE (current source) | Bundle source trace + 14 Python tests + 88 panel tests/typecheck + LIVE Electron/API | `LFM2.5-8B-A1B-MXFP4-CRACK` is MXFP4 (not affine JANG/JANGTQ), has six attention plus 18 SSM/conv layers, and explicitly forbids synthetic `<think>` prefill. Auto reasoning streamed and exact-finaled. The current Electron required `file_info` row parsed `{"path":": "}`, failed execution, leaked faux JSON, and replayed the prior marker. Responses now truthfully ends unmet required-tool requests with `response.failed`/`tool_calls_required`, not `response.completed`. Gateway Ollama detail parity is closed separately below. Evidence: `docs/internal/release-gates/20260720_lfm_native_reasoning_protocol/`. |
| LFM2-diskonly-tqoff-chain | H | In Paged-Off/Block-L2 mode, explicit TQ None could leave later q4-native rows in a content-hash chain, then a bounded hybrid prompt could queue SSM rederive beyond the actual stored KV boundary and never wake the idle scheduler | VERIFIED-LIVE scoped at `d23a4a37f`; LFM family PARTIAL | Source trace + 326 focused tests + raw Chat/Responses + real Electron + process restart | Every explicit-None ordinary page write now evicts an incompatible native-TQ row; rejected/shortened hits reconcile refs and token credit; text/MLLM loops service queued SSM rederive; the companion retargets to the stored block boundary. Live `LFM2.5-8B-A1B-MXFP4-CRACK` cold/warm/restart proof exact-finaled and restored 576/716 tokens as `block-disk+ssm` from nine ordinary L2 pages plus one SSM disk hit, with zero L1 resident bytes and zero TQ-native writes/hits. Required tools, exact-setting Anthropic/Ollama/cancel/failure breadth, larger eviction/fault injection, signed app, and other families remain open. Evidence: `docs/internal/release-gates/20260720_lfm_diskonly_tq_off_truth/`. |
| gateway-ollama-fastapi-detail | M | Ollama gateway converted a FastAPI `{"detail":"..."}` 400 into generic `Backend request failed` | VERIFIED-LIVE (current source) | 53 gateway tests + typecheck + current-source Electron relaunch + UI Start + 8-row direct/gateway probe | `sendOllamaBackendError` now reads `detail` as well as `error`/`message` and preserves backend status. After current-source relaunch, the real UI Start loaded LFM PID 26730; direct/gateway Chat, Responses, Anthropic, and Ollama all returned 400 with the complete native thinking-off incompatibility message. Evidence: `docs/internal/release-gates/20260720_lfm_native_reasoning_protocol/lfm-off-protocol-rejection-after-gateway.json`. |
| Q36-post-tool-current | H | Qwen3.6 current broad-tools Electron post-tool finalization needed proof independent of shared qwen parser source | VERIFIED-LIVE (current source) | LIVE Electron + DB + `/health` | Broad File/Search/Shell row 1328 made exactly one `file_info({"path":"panel/package.json"})`, one result, exact `Q36-POSTTOOL1-DONE`, two short reasoning fragments, no warning, and `22.6 t/s`. Health showed MTP D3 and hybrid SSM/TQ/cache telemetry active; MTP net speedup is not claimed by this row. |
| G4-post-tool-current | H | Gemma4 current broad-tools Electron post-tool finalization needed proof with its own parser/reasoning behavior | VERIFIED-LIVE (current source) | LIVE Electron + DB | Broad File/Search/Shell row 1331 made exactly one `file_info({"path":"panel/package.json"})`, one result, exact `G4-POSTTOOL1-DONE`, no reasoning fragments, no warning, and `38.2 t/s`. |
| G4-cache-default-parity-current | H | Gemma4 mixed-SWA needs its architecture-correct non-paged prompt L2 default reflected consistently in UI, DB, argv, health, warm reuse, and restart restore | VERIFIED-LIVE (current source) | Source trace + LIVE Electron + DB + process argv + `/health` + restart | Real bundle has 40 sliding and 8 full-attention layers; detector/CLI exclude generic paged blocks. Electron Reset Defaults visibly enabled legacy Disk Cache; preview and PID emitted `--no-paged-cache --enable-disk-cache`; DB stored prefix/paged/legacy/block `1/0/1/0`. Exact cold/warm/restart tool rows passed; warm restored 156/157 tokens from memory and process-restart restored the same 156 from disk. Health recorded 2 prompt-L2 disk hits. |
| MM27-slash-path-current | H | MiniMax native fallback truncated slash-bearing generic tool paths, so Electron inspected `panel` instead of `panel/package.json` while still returning the requested final marker | VERIFIED-LIVE (current source) | Source trace + 26 focused tests + LIVE Electron/DB | `_render_xml_examples` now has a path-specific slash-preserving extractor. Pre-fix row 1334 made one wrong `file_info({"path":"panel"})`; post-fix row 1337 made one exact `file_info({"path":"panel/package.json"})`, one result, exact `MM27-POSTTOOL2-DONE`, no warning, and `3,597 paged+tq` cached tokens. |
| DSV4-crack-cache-current | H | DSV4 Flash CRACK current-source native composite cache needed live verification | VERIFIED-LIVE (current source) | LIVE Electron + `/health` | UI-selected configured DSV4 CRACK session loaded `deepseek_v4_v7` native composite cache with SWA/CSA/HCA pools, pool quant, generic TQ KV forced off, paged and block-L2 on. Arithmetic and recall passed with `paged+dsv4`; health showed DSV4BatchGenerator and block-L2 hits. |
| DSV4-crack-exact-marker | M | DSV4 Flash CRACK exact marker fidelity has failed older general prompts but must stay exact on the bounded post-tool direct-answer rail | OPEN/PARTIAL overall; post-tool subrow VERIFIED-LIVE | LIVE Electron + DB | Older requests mutated two markers. Current direct-answer cold/warm rows 1400/1403 each made one tool/result and returned byte-exact `DSV4-DIRECT-RAIL1-DONE`; warm reused 619 `paged+dsv4` tokens. The shared post-tool symptom is closed for this contract, while broader constrained-string reliability still needs a separate repeat matrix. |
| Laguna-current-cache | H | Laguna-M.1 current-source prompt/cache path needed live verification | VERIFIED-LIVE (current source) | LIVE Electron + `/health` | UI-selected `jangq-ai/Laguna-M.1-JANG_2L` loaded `laguna` `plain_kv_v1`, paged KV, q4 stored-prefix TQ, prefix+paged+block-L2 on. Exact marker and recall passed with `paged+tq`; block-L2 hits recorded. Current Electron post-tool row persisted exactly one `file_info`, one result, exact `LAG-POSTTOOL1-DONE`, no warning, and `3,612 paged+tq` cached tokens. |
| Laguna-speed | M | Laguna-M.1 remains far below expected decode speed | OPEN | LIVE Electron + prior dedicated bench | Current UI rows still around `24 tok/s`; correctness/cache is not a speed pass. |
| HY3-mtp-depth1-active | H | HY3 MTP depth-1 must be actually present and active, not metadata-only | VERIFIED-LIVE (current source) | LIVE Electron + `/health` | UI-selected `jangq-ai/Hy3-JANG_2K-MTP`; config/jang/index all show one MTP layer, `42` tensors, `runtime_active=true`, `effective_depth=1`, text scope. Exact output and recall passed. |
| HY3-mtp-speedup | M | HY3 MTP depth-1 may be active without measurable speedup | OPEN/PARTIAL | LIVE Electron + `/health` | Health says `speculative_decoding=not_configured` and exposes no acceptance/speedup counters. Current rows run `26-34 tok/s`; activation is proven, net speedup is not. |
| cache-default-parity-current | H | Every prefix-cache family must default to its compatible L2 lane and match UI/session/CLI/health | PARTIAL-LIVE | Session DB + process argv + health + M3/Gemma visual UI | M3 and the paged families use block L2. Gemma mixed-SWA now has visual settings, DB, preview, PID, health, warm-memory, and restart-disk proof for legacy prompt L2. Bonsai hybrid restart reuse remains quarantined; other architecture rows and manual toggle permutations are not all current, so the campaign-wide row remains partial. |

## OPEN — bundle / quant (NOT engine defects)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| Q-tool | - | Qwen CRACK required-tool -> HTTP 400 | NOT-A-DEFECT | LIVE | correct API semantics; damaged bundle emitted no tool |
| Q-think | - | Qwen CRACK reasoning on/auto hidden-only | BUNDLE | LIVE | quant-damage reasoning-runaway; C1 caps budget |
| Q-mtp | - | Qwen CRACK declares 1 MTP layer, 0 tensors | BUNDLE | LIVE | metadata_inconsistent; engine correctly no-activates |
| Hy3-longform | P | temp-0.9 long-form word corruption (CJK-in-word) seen in 1 UI run | OPEN-UNREPRO | LIVE partial | API regate: 0 CJK across 4 arms; re-verify live UI |

## OPEN — test / release hygiene
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| BASE-42 | - | full-suite environmental/stale-artifact failures (historical label) | BASELINE | LIVE A/B: 54 -> 54 | Final 5,786 pass / 54 fail / 94 skip / 92 deselect; zero NEW failures. Eleven campaign tests added as passes. |
| BOX-pyver | L | box bundled-python stale 1.6.2 vs source 1.6.6 | OPEN (box-local) | LIVE | max2 release bundle IS 1.6.6; rebuild box bundle |

## Regression watch (a future change could break these — verify LIVE each build)
- paged block-hash canonical encoding (C2): any extra_keys schema change must re-run collision tests.
- disk bf16 native store (H3): a future "safetensors can't do bf16" assumption would reintroduce F21.
- MTP depth order env > sidecar > family > D3: adding a family must not shadow Qwen D3.
- eos dialect fallback: a new variant-suffixed family must resolve its stop set (unresolved-eos warnings = 0 live).
- TQ family policy: any new family must land in the correct cache lane (plain-KV TQ / hybrid split / SWA rotating / MLA-exclude) — prove via load-log layout + determinism battery.

## Appended 2026-07-10 (deep cross-family stress, LIVE)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| HY3-deep | - | Hy3-JANG_2K-MTP: off/high-split/longform(526w,0cjk,worst6=2)/tool-stream/multiturn-recall | VERIFIED-LIVE PASS | LIVE stream | 2 cache hits, 0 tracebacks |
| GEM-deep | - | gemma-4-12B: off/longform(545w,0cjk)/tool-stream/multiturn-recall | VERIFIED-LIVE PASS | LIVE stream | 0 tracebacks |
| GEM-effort | L | gemma4 reasoning_effort=high alone does NOT split thinking (rc=0); enable_thinking=true DOES (rc=973/1058); answer always correct(90) | OPEN (consistency) | LIVE 3-variant probe | pre-existing; UI uses enable_thinking (works); NOT v1.6.6 regression; reasoning_effort→enable_thinking mapping needs per-family matrix (cf Mistral none/high), not a release-lane tweak |

## Appended 2026-07-10 (Eric directives — reasoning parity, MM2.7, MTP UI)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| REASON-parity | H | reasoning_effort / reasoning="auto"|on|off must map to EACH family's own reasoning trigger across ALL surfaces (OpenAI chat, Anthropic /v1/messages, Ollama) + UI. gemma4 has its OWN reasoning kwarg; reasoning_effort alone doesn't fire it (live-proven). If a model supports reasoning it must engage via the standard param. | OPEN-PARTIAL | LIVE | Gemma 12/12 and Hy3 12/12 PASS after concrete effort->thinking mapping. MiniMax full artifact is 8/12; OpenAI greedy native-reasoning loops and the supplied Anthropic-off cell sends the same body as auto. UI live proof blocked by browser runtime. |
| MM27-reason | M | MiniMax-M2.7 is a NATIVE always-reasoning model; vMLX has a CUSTOM reasoning-OFF path. Verify reasoning on(default)/off/auto all honored across UI+API and off actually suppresses thinking. | OPEN-PARTIAL | LIVE | Full artifact 8/12; custom off is correct on OpenAI/Ollama. OpenAI greedy on/auto/effort reaches token limit without final. No hidden sampling or output repair added. |
| MTP-ui | M | After model load the UI must SHOW MTP settings (like Qwen MTP models) + engine native type. Verify Hy3-JANG_2K-MTP surfaces MTP depth/native-type in UI post-load. | VERIFIED-LIVE 2026-07-11 | LIVE dev-app screenshot | Performance→ENGINE panel renders post-load: MTP active D1 (text), MTP Depth D1, Scope text-only, Policy deterministic-defaults, Gate greedy=identity-verify/stochastic=rejection-sampling, MTP Tensors 42/0, Native Cache paged_kv, TQ-KV enabled, Attention KV L2 q4/g64. See HY3-UI-LIVE-PROOF-2026-07-11/. |

## Appended 2026-07-10 (Eric: per-arch cache policy — NO ASSUMPTIONS, live-verify each)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| CACHE-policy | H | Intended (Eric): new UI session starts WITH prefix cache; KV-component families get TQ ENCODE on their KV part — gemma rotating-SWA KV, qwen hybrid-SSM attention-KV (+ async-rederive SSM), hy3 plain-KV layers. Must LIVE-verify per family what ACTUALLY happens (cache layout, TQ objects, whether encode FIRES, async rederive) vs intended. DO NOT ASSUME what applies where. | VERIFIED-LIVE (encode defaults gated off) | LIVE per-family | Hy3 plain 80 TQ; Qwen hybrid 16 TQ + 48 native SSM and paged+ssm; Gemma 8 full TQ + 40 rotating. Prefix hits and exact outputs proven. Encode remains OFF because no family passed every coherence+memory gate. |
| GRADE-rule | - | Every test cell must be graded PASS/FAIL by LIVE proof; each FAIL gets a fix. | POLICY | Eric directive | applies to reasoning parity + cache policy + MTP UI matrices |

| PAGED-toggle | M | UI paged cache default OFF (correct); toggling it ON must actually work end-to-end (UI→gateway→engine spawn --use-paged-cache→paged blocks + TQ on KV). CLI arm M2 passed; UI-toggle path needs live proof. | VERIFIED-LIVE 2026-07-11 | LIVE dev-app session | Running Hy3 session saved config usePagedCache:true → spawned argv has --use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000; live cache path = paged_kv with TQ on attention_kv (Native Cache paged_kv, 1295 paged+tq cached tokens reused). config→argv round-trip faithful. |

## Graded reasoning-parity matrix result (LIVE 2026-07-10)
Hy3-JANG_2K-MTP: 12/12 PASS (all surfaces x off/auto/effort/enable).
gemma-4-12B: 11/12 PASS. ONLY fail = OpenAI reasoning_effort (think=0);
gemma Ollama effort (think=950) + Anthropic effort (think=1082) PASS.
=> Precise defect: OpenAI /v1/chat/completions reasoning_effort does not imply
enable_thinking for a supports_thinking family lacking an effort normalizer
(hy3/dsv4 have one; gemma4 does not). Fix owned by codex impl campaign;
verify live after. Answer correct + leak-free in ALL cells.

## Appended 2026-07-10 (Eric Q: hy3 22 tok/s in UI history)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| MTP-len-curve | P | Hy3-JANG_2K-MTP UI turns read 22.3 tok/s on LONG gens (1183/1098 tok) but 31.8/33.6 on short (41-86 tok); MTP was ON (depth-1). 22 = normal sequence-length decode decay, NOT a regression/MTP-off. But: MTP +10% was only measured at 600 tok — measure MTP benefit vs context length (short/med/long) to get the full curve. | OPEN (measure) | LIVE (chat metrics_json) | short-turn 31-34 confirms MTP engaged; prefix cache working (108 tok/s warm on 4590-tok cached prompt, pp 8255) |

## #78 ENGINE-AWARE TQ contract (Eric 2026-07-10 — VERIFY codex against this)
TQ encode is engine-aware: encode may ONLY act on the TurboQuantKVCache path,
which is ALREADY excluded (make_cache patch, jang_loader.py:1558-1646) for
families that must NOT take TQ:
- MM3 / MiniMax-M3 (native MSA, idx_keys lane) — NO TQ
- DSV4-Flash / deepseek_v4 (MLA+DSA latent) — NO TQ (is_mla_model)
- MLA: DeepSeek V2/V3, GLM-5.1, Mistral 4 — NO TQ
- MiMo-V2 (asymmetric full/SWA) — NO TQ
- gemma/Laguna mixed-SWA — TQ only on full-attn slots under VMLX_SWA_TQ
=> When #78 wires compress_after into TurboQuantKVCache, encode is structurally
impossible on the excluded families (they never create that object). Per-family
LIVE coherence gate applies ONLY to Hy3 plain-KV, Qwen3.6 attention-KV,
gemma full-attn (opt-in). VERIFICATION GATE: codex #78 must not add encode
anywhere that bypasses the exclusion; confirm live these families still show
native/MSA/MLA cache (no TQ objects) + coherent after the change.

### #78 TQ per-COMPONENT awareness (Eric 2026-07-10 refinement)
TQ is component-aware, not just model-aware. hybrid_tq_cache.build_hybrid_turboquant_make_cache
wraps ONLY layer_type=="attention" KV slots in TurboQuantKVCache
(policy tag _vmlx_hybrid_tq_policy="attention_kv_only"); every SSM/GDN/conv
companion slot keeps its NATIVE class (append(slot)) with async re-derivation —
NO TQ, NO encode. So within Qwen3.6 hybrid: attention-KV -> TQ (+future encode);
GatedDeltaNet/SSM state -> native full-precision + async rederive.
VERIFY (live): after #78, Qwen3.6 load-log shows attention layers TQ + SSM
layers native companion; encode counters advance ONLY on attention-KV slots,
never on SSM/companion. Same for gemma full-attn vs sliding slots.

## Appended 2026-07-10 (Codex implementation campaign grades)

| ID | Sev | Issue / test | Status | Evidence | Notes |
|----|-----|--------------|--------|----------|-------|
| TQ-HY3 | H | Hy3 live encode coherence gate | FAIL-SAFE | LIVE | Encode counters advanced on 80 KV slots; cache and seeded output worked, but reasoning was 10/12 armed vs 12/12 disabled. Default OFF. |
| TQ-Q36 | H | Qwen3.6 attention-only encode memory gate | FAIL-SAFE | LIVE | 16 attention TQ + 48 native SSM, exact output and `paged+ssm`; resident delta +1,637,280 bytes. Default OFF. |
| TQ-GEM | H | Gemma4 full-attention encode memory gate | FAIL-SAFE | LIVE | 8 full TQ + 40 rotating, exact output and 1,596-token hit; resident delta +2,298,176 bytes. Default OFF. |
| REASON-GEM | H | Gemma effort/on/off/auto x OpenAI/Ollama/Anthropic | VERIFIED-LIVE PASS | LIVE 12/12 | Former OpenAI effort failure fixed; correct 90 km/h and leak-free. |
| REASON-HY3 | H | Hy3 effort/on/off/auto x OpenAI/Ollama/Anthropic, encode disabled | VERIFIED-LIVE PASS | LIVE 12/12 | Confirms reasoning mapping and #78 safe gate. |
| REASON-MM27 | H | MiniMax-M2.7 full artifact matrix | OPEN-FAIL 8/12 | LIVE | OpenAI greedy native reasoning has no final by 900 tokens; supplied Anthropic-off cell is identical to auto. Custom off itself passes. |
| CACHE-NEMO | H | Nemotron attention-only TQ + native SSM warm restore | VERIFIED-LIVE PASS | LIVE | 702 cached tokens, `paged+ssm+tq`, exact output, SSM companion entry. |
| CACHE-LAG | H | Laguna mixed-SWA dedicated-loader opt-in | VERIFIED-LIVE PASS | LIVE | 601 cached tokens, `memory+tq`, exact output. |
| CACHE-MM27-Q4 | H | Memory-prefix q4 cross-request stream ownership | VERIFIED-LIVE PASS | LIVE | Before: empty 200 / process abort. After CPU-packed storage: 56-token memory hit and native reasoning generation runs. |
| UI-TSC | M | Panel settings/MTP/native-type compile | FIXED | graded test | `tsc --noEmit` PASS; 187 focused Vitest tests PASS. |
| UI-LIVE-20260710 | H | Real UI paged toggle + MTP native-type rendering | OPEN-BLOCKED | tool boundary | Mandatory in-app browser connector rejected missing sandbox metadata. No visual claim made. |
| DEAD-20260710 | M | Six unreferenced helpers | FIXED | grep + AST + suite | Removed exact-zero-reference wrappers; dynamic compat/import/parser hooks retained and documented. |

## Appended 2026-07-10 (Eric: settings-enforcement worry — ALL settings ALL models)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| SETTINGS-enforce | H | Worry: UI settings (esp. paged cache default-off; toggle on/off; session restart; model switch) may not be ENFORCED into the spawned engine. Mechanism EXISTS: RESTART_REQUIRED_KEYS covers all cache/MTP/kv/batch/parser keys; (re)start rebuilds argv from config via cacheLaunchPolicy w/ per-family overrides. But that's a CODE LEAD. | OPEN (needs LIVE) | code-only | LIVE-prove per model x per setting: toggle → spawned `pgrep -af vmlx-serve` argv changes, no stale engine, runtime honors it. See docs/internal/UI-CLI-PARITY-TQ-AWARENESS.md B2. RELEASE-GATING. |

## Reasoning/streaming set — CLOSED, SHIP (LIVE 2026-07-11)
Eric's UI-behavior worry was RIGHT: reasoning/streaming changes shifted behavior.
Codex parallel proofread+stress caught reasoning-ON failing 7/8 rows (empty
content; Ollama-stream answer misrouted to message.thinking). Fixes:
- codex 0cde4f19e: sampler norm order, seeded-sampler sharing, answer-pass
  double-emit, visible-prefix deletion, Ollama/Hy3 normalization (5 bugs).
- Claude f2b7e8c12: bounded answer-pass floor (48) — runaway reasoning no
  longer starves visible answer to empty; overage bounded (not audit-C1 2x).
- codex 2a5c2d445: Ollama-stream answer -> message.content (misroute fixed),
  Hy3 answer-marker holdback=0, family-label helper.
RE-VERIFY (codex live, all 4 routes x stream/non-stream x 3-turn reasoning-on):
24/24 PASS, every turn-3 non-empty+coherent, no leak/misroute, overage 36<=48,
warm greedy byte-identical DET-731, full-suite zero new failures. SHIP.
All pushed origin/main, both nodes synced. Author Jinho Jang throughout.

## 2026-07-12 — ANSWER-PASS-DSV4-DEGEN: salvage context malformed for deepseek_v4/step3p7
FOUND (Codex live-UI QA, DSV4-Flash-JANG-CRACK, Reasoning mode, maxTokens=64):
never-empty salvage emitted degenerate looping "ans=BLUE-FALROON+37...+DERIV+DERIV"
(turn 2 copied the garbage via history). Mechanics of the rail were CORRECT
(112 = 64 + 48 floor; non-empty streamed; UI rendered reasoning box + perf line).
ROOT CAUSE (proven by template-render A/B): answer pass appended
{role:assistant, content:"", reasoning_content:trunc}; DSV4 template DROPS
reasoning_content -> renders EMPTY assistant turn + EOS + SECOND generation
prompt (<Assistant><eos><Assistant></think>) = OOD prompt -> degeneration.
Direct API replay WITHOUT the appended turn on the same engine: clean
"BLUE-FALCON, 37, Paris is capital of France", finish=stop (LIVE-PROVEN).
step3p7 template: renders back-to-back assistant turns (also malformed);
NOTE step3p7 direct rail is real — ensure_thinking_off_sentinel closes its
always-open <think> in simple/batched/mllm prompt paths.
FIX: _ANSWER_PASS_FRESH_CONTEXT_FAMILIES={deepseek_v4,step3p7} -> answer pass
re-runs ORIGINAL messages at all 4 sites (chat/responses x stream/non-stream);
legacy 9-family appended-reasoning behavior untouched.
Commits: max2 e4b2f13f3 / box 6113241b7. Tests 5/5 + answer-pass suite 23/23.
STATUS: OPEN until live re-proof on reloaded DSV4 (stream salvage clean at
maxTokens=64) — box currently owned by Codex MM3 row. Author Jinho Jang.

## 2026-07-12 — ANSWER-PASS-DSV4-DEGEN: LIVE RE-PROOF PASS (stream path)
Reloaded DSV4 on fixed engine (box ba715c6b9 lineage). Same failing scenario
(stream, thinking-on, max_tokens=64): reasoning 259 chars (budget exhausted)
-> salvage "The 37th BLUE-FALCODE is 37-BLUE-FALCON. Paris is France's
capital. Final: 37-BLUE-FALCON, Paris." finish=stop — coherent prose, zero
looping (was "+DERIV+DERIV" garbage). Turn 2 multiturn: "codeword=BLUE-FALCON;
sum=42; capital=Paris" — correct recall+arithmetic, 6s warm vs 69s cold
(prefix reuse working engine-side). Codex UI pixel re-verify in flight.

## 2026-07-12 — ANSWER-PASS-M2-DEAD-FAMILY: MM2.7 reasoning-only EMPTY root-caused + FIXED
FOUND (Codex live-UI matrix): MM2.7-Small Thinking-On at tight budget streamed
reasoning then EMPTY visible + "reasoning only" warning (graded release FAIL).
REPRO (API stream, max_tokens=64): reasoning 278 chars, content "", finish=length.
ROOT CAUSE: MiniMax-M2.x bundles resolve family_name "minimax";
_REASONING_ANSWER_PASS_FAMILIES only contained "minimax_m2" — which is the
family's REASONING PARSER name, never a family (registry grep: minimax_m2
appears ONLY as reasoning_parser). Dead entry -> rail never armed for M2.x.
Also minimax template renders the appended reasoning turn as back-to-back
]~b]ai turns (same malformation class as deepseek_v4/step3p7; render-proven).
FIX: arm "minimax" (+keep "minimax_m2") in families + label map; add both to
_ANSWER_PASS_FRESH_CONTEXT_FAMILIES. Commits: max2 69d6cebe3 / box ba715c6b9.
Tests 6/6 file, 31/31 selection. LIVE RE-PROOF PASS: same request now emits
"BLUE-FALCON 37 Paris is the capital of France" finish=stop.
Direct-rail sanity (fresh thinking-off 48tok): "BLUE-FALCON, 37, Paris is the
capital of France." finish=stop. Author Jinho Jang.
LESSON: family-set entries must be validated against what the registry
actually emits as family_name — parser names are NOT family names.

## 2026-07-12 — JANGTQ SUPPORT MATRIX (Eric: "forgotten jangtq support")
Static: jang_tools fast path present in box venv; ALL 7 local JANGTQ bundles
have dict mxtq_bits (no scalar slow-path trap); registry detects all 7
correctly (families/parsers/cache: nemotron_h ssm_attention, qwen3_5_moe
hybrid, laguna kv, minimax kv, mistral3 kv, step3p7 full_sliding_kv).
LIVE ENGINE GATES (vmlx-serve, greedy, ports 8033-8036):
- Nemotron-Omni-Nano-JANGTQ-CRACK: ALL PASS — coherent, recall sum=42,
  nothink clean, warm==warm byte-equal, cache_detail paged+ssm+tq (24/31).
  "JANGTQ v2 loaded 5.4s native TQ". NOTE cosmetic: "0.0-bit avg" in load line.
- Nemotron-Omni-Nano-JANGTQ4-CRACK: ALL PASS (same profile).
- Laguna-XS.2-JANGTQ: coherent+recall+nothink PASS; G4 flagged = COLD-vs-warm
  whitespace diff (probe design compares cold run vs warm run; MoE inherent).
  cache_detail memory 63/64. Needs warm-vs-warm re-gate for a clean verdict.
- Qwen3.6-35B-A3B-JANGTQ-CRACK: ALL PASS — cache_detail paged+ssm (hybrid GDN,
  TQ policy per autodetect matrix), 5 prefix HITs. NOTE minor deterministic
  quant artifact: fib list rendered "0, 1, 1, , 2, 3, 5, 8, 13" (byte-stable).
- MiniMax-M2.7-Small-JANGTQ: running in app :8003 — coherent (Rayleigh),
  UI On/Off proven by Codex pixels (post minimax-arm fix).
PENDING: Mistral-Medium-3.5-128B-JANGTQ (38G) + Step-3.7-Flash-JANGTQ_K (69G)
big-2 round (needs DSV4 stopped); Laguna warm-vs-warm re-gate.
HARNESS NOTE: first round false LOAD=FAILs were bind Errno48 on a reused port
(TIME_WAIT) — per-model ports fixed it. pkill "vmlx-serve serve" does NOT
match app engines (python -m vmlx_engine.cli serve) — app sessions unaffected.
STACKED CODEBOOK (Eric flag): loader has codebook_vq path (_is_codebook_vq_model
+ guarded experimental import; models/codebook.py present BOTH nodes, tracked
since v1.3.93). ZERO codebook_vq bundles on the external drive → cannot live-
gate; needs a bundle to validate load+cache behavior. Author Jinho Jang.

## 2026-07-12 — ENGINE-FATAL-METAL-CALLBACK: root cause of "DSV4 UI row" false FAILs
Crash report python3.13-2026-07-12-161249.ips: SIGABRT via uncaught C++
exception from mlx::core::gpu::check_error(MTL::CommandBuffer*) inside a Metal
addCompletedHandler callback thread -> std::terminate -> whole engine process
dies. Trigger = GPU oversubscription: I ran the JANGTQ probe fleet (CLI serve,
12-19GB loads) CONCURRENTLY with the app DSV4 session (67GB) — both Codex's
DSV4 UI row (15:59 session stop) and my 274s-TTFT "[Generation interrupted]"
row (16:12 crash, victim was the Laguna probe server; DSV4 session -> error)
were contention casualties, NOT the answer-pass fix (API-stream proven clean).
SIDE FINDING: M2.7 session record went "stopped" while engine pid 41092 kept
serving :8003 — session-state desync / orphaned engine under the same chaos
window (cleaned up by kill; watch for recurrence under normal ops).
RULES ADDED: (1) NEVER run probe engines concurrently with app sessions on the
box — Metal errors under contention are FATAL (no catch possible on callback
threads); serial only. (2) "Engine randomly died" под load => check
DiagnosticReports python*.ips for check_error/MTLCommandBuffer first.
UPSTREAM LEAD (non-blocking): MLX check_error throwing on completion-handler
threads means any transient Metal error kills the server; a graceful
degrade would need an upstream MLX change or process-level supervision.
Author Jinho Jang.

## 2026-07-12 — JANGTQ matrix addendum
- Laguna-XS.2-JANGTQ: warm-vs-warm (W2==W3) byte-equal PASS — earlier G4 flag
  confirmed cold-vs-warm MoE router-flip (inherent). ALL GATES PASS. 6/7 clean.
- Mistral-Medium-3.5-128B-JANGTQ: LOAD=BLOCKED-BY-DESIGN — load_mistral3.py:98
  NotImplementedError gate (JANGTQ2 2-bit dense TQ stalls full prefill,
  degenerate text; points to mxfp4 bundle; VMLX_ALLOW_UNSTABLE_MISTRAL35_JANGTQ=1
  debug hatch). Correct fail-closed behavior, NOT a regression.
- Step-3.7-Flash-JANGTQ_K (69G) running last.

## 2026-07-12 — JANGTQ MATRIX FINAL (7/7 verdicts, #101 CLOSED)
| Bundle | Verdict |
| Nemotron-Omni-Nano-JANGTQ-CRACK | ALL PASS (paged+ssm+tq) |
| Nemotron-Omni-Nano-JANGTQ4-CRACK | ALL PASS |
| Laguna-XS.2-JANGTQ | ALL PASS (warm==warm byte-equal re-gate) |
| Qwen3.6-35B-A3B-JANGTQ-CRACK | ALL PASS (known ", ," deterministic quant artifact) |
| MiniMax-M2.7-Small-JANGTQ | PASS (in-app; UI On/Off Codex-pixel-proven) |
| Mistral-Medium-3.5-128B-JANGTQ | BLOCKED-BY-DESIGN (load_mistral3.py fail-closed gate; correct) |
| Step-3.7-Flash-JANGTQ_K | LOADS after jang_tools audio_config fix; QUANT-DEFECTIVE (greedy soup all modes, deterministic; engine machinery sound: cache 19/20, warm==warm) — confirms 07-04 memory |
NEW FIX (jang repo, both nodes): load_jangtq_vlm._mlx_vlm_skeleton forced
audio_config={} into every bundle -> update_module_configs resolved
model_class.AudioConfig -> AttributeError killed EVERY Step-3.7 JANGTQ VLM
load. Fix = setdefault None (box jang 8260f88 / max2 jang a9a4fc1 + both venv
copies). Regression: Qwen3.6-JANGTQ rows byte-identical pre/post.
OPEN QUESTION for Eric: give Step-3.7-Flash-JANGTQ_K the same fail-closed
production gate as Mistral-3.5-JANGTQ (clean error + escape hatch) instead of
load-and-soup? Codebook-VQ: loader path present both nodes, ZERO bundles on
drive to gate — needs a bundle. Author Jinho Jang.

## 2026-07-12 — RECLASSIFY (Eric correction): Step-3.7-JANGTQ_K soup = ENGINE-SUSPECT, not "quant defect"
Eric: models on the external drive were PROVEN coherent before landing there.
Codex jang-repo memory confirms: "Step 3.7 JANGTQ runtime fix — runtime bug,
COHERENT BUNDLE, upload prep" — the bundle validated coherent at build time
and a RUNTIME fix existed. Precedent: openPangu 42->4 (blamed quant, was
ENGINE positional bug), MXFP4 expert-scramble (RETRACTED), DSV4 "too big"
(app bug). The 07-04 "greedy soup = quant" memory is UNPROVEN.
Also re-examine: Mistral-3.5-JANGTQ fail-closed gate rationale ("2-bit dense
stalls prefill, degenerate") may itself paper over the same engine-side
JANGTQ hydration bug.
NEXT: decisive A/B — jang_tools reference runtime vs vmlx engine on the SAME
Step-3.7-Flash-JANGTQ_K bundle (queued after Codex UI lane; serial GPU rule).
If reference is coherent -> vmlx step3p7 JANGTQ hydration path is guilty.
RULE (standing, Eric): drive bundles are pre-proven — output damage defaults
to ENGINE-SUSPECT until an A/B against the build-time reference says otherwise.

## 2026-07-12 — Codex backlog lane verdicts (post minimax-arm fix)
PASS: MiniMax-M2.7 On/Off/Auto at normal budget + UI==API parity per mode
(reasoning presence On[t,t]/Off[f,f]/Auto[t,t]; all stream to [DONE], no
leaks) — #82 MiniMax core CLOSED by re-grade.
NEW FAILS (all ENGINE-SUSPECT per pre-proven-bundle rule, tasks #103/#104):
- DSV4 UI normal budget (Auto, 2048): BOTH turns loop malformed "[verb]" text
  to cap; 24 prompt tokens (no tools in prompt). Earlier On+64 UI turn was
  coherent — deltas: Auto mode + 2048 budget. ALSO engine cache_hit_requests=0
  in that run (dsv4PrefixCache:true) — hit accounting or engagement gap.
- Zaya tools: <parameter=path> pseudo-markup + AppleScript register leaked as
  visible content to 2048 cap, toolStatuses=[] (2890 prompt tokens — tools WERE
  in prompt). Parser-dialect gap suspicion (zaya_xml vs qwen-style fragments).
- Zaya think On: reasoning_len 0,0 at 2048 cap (known-open #91 remnant, now
  confirmed at normal budget in UI).
Lane A wave-1 committed both nodes (22a5823d2/199f37591): #233 deadlock,
stop_token_ids crash + decode-loop stop fallback, lockless clear/reset_stats.
NEXT QUEUE: #103 DSV4 loop repro (box free now) -> #104 Zaya -> Step-JANGTQ_K
A/B vs jang_tools reference -> remaining triage (TQ "!" hole, convert path).

## 2026-07-12 (later) — #103 RESOLVED environmental + paged-default campaign start
#103 DSV4 "[verb] loop": ENGINE PROVEN HEALTHY. Full API matrix on :8005
({cold,warm} x {stream,non-stream} x {Auto, reasoning.effort=high, EXACT panel
fields enable_thinking+thinking_mode+chat_template_kwargs, +/-reasoning_effort}
x short-exact-prompt x 2048) -> EVERY run clean "42" + reasoning captured
(53-78 chars), no loop, no cap. UI turn-1 smoking gun = 55.4s TTFT / 0.4 pp-s
on 24 tokens = GPU-contention/thrash starvation (env), not engine; turn-2
inherited turn-1 garbage as context. Thread B (cache_hit=0) BENIGN: short
prompts (<256 store threshold) never stored -> turn-1 uncacheable, turn-2 miss
legit. REMAINING: one clean live-UI run under no contention to close.

CAMPAIGN (Eric 2026-07-12): paged cache DEFAULT ON all autodetect + UI parity;
Zaya tool+reasoning parser selectable; codebook stacked-vs-none awareness;
reasoning->answer streaming continuity for agentic loops (no random stop);
DSV4 ~20 t/s + large usable context (small cache footprint); dual-perspective
(Codex proofreads every fix, both doubt everything). Codex diag (bzwixvdj6)
delivered file:line change-map for all 3 engine/panel lanes.

LANE 1 (paged default) ENGINE DONE (uncommitted, pending Codex proofread
bm7adgc05): cli.py adds --no-paged-cache opt-out (serve+bench) + generic
default-ON block (~1000-1017) for autodetected families EXCEPT
{minimax_m3,minimax_m3_vl,openpangu_v2,gemma4}; DSV4/Zaya self-manage; hybrid
SSM auto-switches. Requires continuous_batching. PANEL parity still TODO
(sessions.ts migration bump v-next, SessionConfigForm default ON, launch must
pass --no-paged-cache when OFF so stale saved-false != silent; plumb
/v1/capabilities paged state; legacy-disk OFF + block-disk ON when paged).

LANE 3 (streaming continuity) CONFIRMED at source: server.py:16350-16352 chat
stream emits finish_reason=stop on reasoning-only final chunk BEFORE answer-pass
runs (16821) -> agentic harness stops early. Fix = suppress first-pass stop when
answer-pass will run (not content_was_emitted AND (m3_)reasoning_only_answer_
enabled AND accumulated_reasoning) AND guarantee answer-pass path always emits a
terminal finish (Risk B: empty-answer paths 17087-17182 must emit length/error,
not bare stop). Non-stream + Responses variants analogous (Codex mapped).

LANE 2 (Zaya) per Codex: zaya_xml tool + qwen3 reasoning ARE registered/
selectable (engine zaya_tool_parser.py:19-20, registry:109-113). Real leak
suspect = sessions.ts:1460-1466 preserves stale toolCallParser='' (None) for
ZAYA -> launch omits parser -> raw <parameter=path> leaks. MUST live-confirm
the session's actual saved parser value, not assume.

## 2026-07-12 — PAGED-DEFAULT-ON (#86/#87): text families default paged ON + full UI↔engine parity
DIRECTIVE (Eric, reverses prior paged-default-OFF): paged cache default ON for
ALL autodetected text models at startup; Electron UI default toggles MUST match
what the engine launches with (no drift).
LANDED max2 282fe0b03 (7 files, +192/-19):
- Engine cli.py serve: use_paged_cache defaults ON for autodetected text family
  when continuous_batching + prefix cache active; skips MLLM/VL (is_mllm_model
  force-aware), DSV4 (own composite paged), and paged-incompatible set
  {minimax_m3, minimax_m3_vl, openpangu_v2, gemma4, gemma4_text}; honors explicit
  --no-paged-cache / --use-paged-cache. Added --no-paged-cache (disable_paged_cache).
- Panel parity: launch emits --use-paged-cache OR --no-paged-cache matching the
  effective policy (NO silent OFF); CACHE_STACK_STARTUP_DEFAULTS_VERSION 7->8;
  staleV7GenericPagedOff migration (full v7 default tuple) flips ONLY untouched v7
  text sessions ON, preserves user-modified + detected-OFF families; migration runs
  in updateSession() before markCurrent. Registry defaults paged ON text / OFF MM;
  dense-KV VL forced OFF. detectedUsePagedCache threaded into SessionConfigForm +3
  parents; native forced-paged gated on detectedUsePagedCache===true (Gemma
  mixed-SWA now unchecked/enabled == launch OFF; hybrid/step-3.7 still forced ON);
  renderer preview emits --no-paged-cache OFF branch. --no-paged-cache blocked from
  additionalArgs (main + renderer).
FAMILY PARITY (UI checkbox == launch flag == engine): generic text ON; dense VL
OFF; hybrid VL/Zaya/mamba/nemotron/lfm2 ON; Gemma mixed-SWA OFF; M3 OFF; openPangu
OFF; DSV4 opt-in/out; step-3.7 ON.
VERIFY: tsc --noEmit clean; 4-pass Codex GPT-5.6 dual-review (13 issues found+fixed
across passes; pass-4 GO-WITH-CHANGES → renderer OFF-branch added → GO).
EXCLUDES pending #98 (MLLM paged byte-ceiling). Author Jinho Jang.
STATUS: committed max2, push + box sync in flight; NEXT live Electron per-family
matrix (UI toggle == /tmp/vmlx-dev.log launch flag).

## Post-v1.6.8 live-Electron campaign (2026-07-13, Codex gpt-5.6-sol drove real dev app CDP :9333)
Codex live-Electron pass returned NO_SHIP_LIVE_ELECTRON_UI with 10 defects API/curl testing missed. Dual-agent fix loop (Codex = live UI eyes, Claude = root-cause + fix). All results LIVE on the running dev app at commit 3e71d82e / dae7336f4.

| ID | Sev | Issue | Status | Fix | Live proof |
|----|-----|-------|--------|-----|-----------|
| LE1 | H | existing-session cache edits not persisted — save clears dirty state but migration reset stored paged/L2/%/max-out to family defaults; engine launched --no-paged-cache --cache-memory-percent 0.15 no --max-tokens while UI showed paged/L2/19%/2048 | VERIFIED-LIVE | f0d0661bf (sessions.ts: migrate stored baseline first, merge user edits on top) | Codex A3: saved config + launched argv both show --cache-memory-percent 0.19 --use-paged-cache --enable-block-disk-cache --max-tokens 2048 |
| LE2 | M | CLI preview omitted --cache-memory-mb/percent under paged (H1 regression: dropped guard in sessions.ts real emit but not preview copy) | VERIFIED-LIVE | f0d0661bf (SessionSettings.tsx) | Codex A2: preview shows --cache-memory-percent 0.17 + --cache-memory-mb 1024 under paged |
| LE3 | L | paired number/range sliders off by one (Block 64/65, MaxBlocks 1000/1001, MaxOut 512/513, MaxCtx 1024/1025) — min=1 off-steps the range grid (1+k*step) | VERIFIED-LIVE | 3e71d82e9 (SliderField: anchor range track to step grid) | Codex A1: number==range for all 4 controls |
| LE4 | M | Zaya double-wrapped tool args {"path":{"path":..,"offset":..}} -> tool rejects path-as-object -> model loops to iteration guard | FIXED (nested shape eliminated live); residual is model trait | dae7336f4 (zaya_tool_parser _unwrap_double_wrapped + 2 unit tests) | Codex Zaya restart PID 82010: nested {"path":{...}} NO LONGER occurs, 0 Dropping / 0 must-be-of-type-string lines |
| LE5 | H | Zaya tool calls still fail — model emits AppleScript programs as path value + colon garbage, or no tool call (natural prompt) burns 2048 budget empty | NOT-A-DEFECT (model trait) | n/a — Zaya-8B-JANG_4M trained/quantized AppleScript persona; parser extracts faithfully | Codex Zaya: 3 calls all AppleScript/garbage path after double-wrap fix; parser not at fault |
| LE6 | M | LFM2.5 multiturn "contamination" (T3 repeats stale CODEWORD STORED suffix) | NOT-A-DEFECT (model/transcript conditioning) | n/a | Codex cache-ON vs cache-OFF (--disable-prefix-cache, SSM companion disabled, 0 cache telemetry) IDENTICAL failure; arithmetic passes at 2048 |
| LE7 | M | Gemma-4-12B-JANG_4M VL grounding unreliable (screenshot hallucinated, red->salmon) | NOT-A-DEFECT (4-bit VL quant quality; plumbing OK) | n/a | Codex red-square control: model saw uniform red field (vision wired); media.items=1 data_url=1 --is-mllm |
| LE8 | L | Gemma "Reasoning Off not honored" (engine enable_thinking=True despite drawer Off) | NOT-A-DEFECT | n/a — stored chat_overrides.enable_thinking was NULL (Auto) for tested chat; another Gemma chat has 0 persisted fine | DB check: chat 0e455c49 enable_thinking NULL; d1a2c0b4 enable_thinking 0. Request builder correct. |
| LE9 | L | chat model-selection reverts to stopped/other model after nav (composer disabled) — raw path-string match missed the chat model under an aliased path | VERIFIED-LIVE | e498912b5 (App.tsx handleChatSelect matches by shared modelIdentity/sessionMatchesModelPath) | Codex live: alias-backed LFM chat bound to running /Volumes LFM session (composer enabled), survived nav-away/reselect AND two-running-model (Zaya+LFM) test — did NOT fall back to Zaya |
| LE10 | L | starting a session whose path is a dead symlink threw Model-not-found though the same model exists at a valid path | VERIFIED-LIVE | e498912b5 (sessions.ts re-resolve by identity, single-match only) + ccb31dfea (guard persist vs UNIQUE) | Live dev-app: log "re-resolved by identity to /Volumes/...", engine launched /Volumes path on :8001, 0 Model-not-found, 0 UNIQUE/persist errors |
| LE11 | H | replayed prior tool calls/results were unioned into each new assistant row, causing super-linear prompt growth | VERIFIED-LIVE | a74d68b86 (`currentTurnToolStart` + current-turn-only harvest) | Existing Electron app over CDP :9333, external LFM2.5 Responses tools: controlled prompt `1465 -> 1641 -> 1817 -> 1946`, byte-identical resend at `1946`; chat DB and `[CHAT_DIAG]` show no prior-turn union. Verdict `/tmp/codex-toolhistory-verdict.md`. |

## 2026-07-15 — LIVE-ELECTRON-GATEWAY-CACHE-TOOLS: scoped current-source continuation

- `GATEWAY-BOUND-PORT`: VERIFIED-LIVE. With installed vMLX owning wildcard
  `8080`, the dev gateway fell back to `*:8081`; the API UI and DB now display
  the actual bound port, and LAN `/health` was reachable from the controlling
  Mac. Restored to `127.0.0.1:8080` afterward.
- `SINGLE-MODEL-SWAP`: VERIFIED-LIVE. Gateway Bonsai -> Hy3 -> Bonsai requests
  killed the previous engine PID before loading the target; only one local
  model remained resident. Responses deltas stayed incremental through swap.
- `CACHE-SETTINGS-ENFORCE`: VERIFIED-LIVE on Hy3. Visible Max Blocks edit
  changed UI/DB/argv/health `1000 -> 900`, then restored all four to `1000`.
  Prefix, paged, block-L2, 15%, 64-block, 1000-block, 10GB, and KV-auto values
  were visually expanded and matched the live process.
- `I18N-ALL-SHIPPED`: VERIFIED-LIVE for zh/ko/ja/es/en API controls; all five
  catalogs have the same 978 flattened keys. English was restored.
- `BONSAI-UI-TOOLS`: VERIFIED-LIVE after correcting the old invalid setup
  (that chat had builtin/file/search tools disabled). Fresh default-thinking
  row persisted exactly one `file_info`, its real result, and exact
  `B1-UI-TOOL5-DONE`.
- `POST-TOOL-REASONING-RETRY`: FIXED+VERIFIED-LIVE. A valid pre-fix row made
  one tool call, then four reasoning-only continuations and no content (five
  fragment cards, 4,604 tokens). The panel now performs at most one explicit
  answer-only recovery, removes tool schemas only for that retry, uses the
  local direct rail, and surfaces an error if content remains empty. Live row
  produced two phase-appropriate reasoning segments and exact final content.
- `POST-TOOL-TPS`: FIXED+VERIFIED-LIVE. Final metrics retain the rolling
  streamed rate when Responses buffering makes cumulative delta-arrival time
  invalid. Live changed from impossible `261-565 t/s` to `41.9 t/s`, matching
  the measured stream.

Campaign status remains `PARTIAL_NO_RELEASE`: Laguna decode speed, measured
Hy3 MTP benefit, DSV4 exact-marker fidelity, and M3 image-character exactness
remain open. No public release/notarization/feed mutation performed.

## 2026-07-15 — CURRENT-ELECTRON-CROSS-MODEL-CONTINUATION

- `BONSAI-REPEATED-REASONING`: VERIFIED-LIVE for the current single-tool loop.
  The original row 1289 was a genuine failure: five reasoning fragments,
  4,604 generated tokens, one completed tool, and empty visible content. Two
  later probes exposed a second form: the post-tool model re-entered native
  tool generation indefinitely. Source now recognizes only the explicit
  `exactly once` + `after the tool result` + `reply exactly` contract and sends
  its planned follow-up without tools on the direct-answer rail. Rebuilt rows
  1373/1376/1379 each made one tool call and one exact final; row 1379 restored
  158/159 prompt tokens from `paged+ssm`.
- `BONSAI-HYBRID-L2-RESTART`: PARTIAL. Process-restart row 1382 remained
  coherent/exact but restored zero cached prompt tokens. Qwen hybrid SSM L2
  restore is intentionally quarantined after cross-restart numeric divergence;
  block-L2 remains populated but the runtime full-prefills on the missing SSM
  companion. Do not promote this to an L2 restart pass.
- `NEMOTRON-DUPLICATE-FINAL`: VERIFIED-LIVE. The broad agent prompt contained
  the final-response directive twice, and the exact-output suppression matched
  only `reply exactly:`. Nemotron therefore emitted the marker twice in rows
  1358/1361. Current source removes the duplicate directive and recognizes
  ordinary `reply exactly MARKER`; rebuilt row 1364 produced one tool, one
  marker, `paged+ssm+disk+tq`, and no warning.
- `STEP-JANGTQ-K-COHERENCE`: VERIFIED-LIVE on current source. Historical row
  1349 remains valid pre-fix evidence: 1,854 runaway reasoning tokens and no
  valid tool/final. Source comparison found that the installed generic P18
  attention patch omitted Step's post-reshape q/k norms and head-wise
  `g_proj` gate. The loader now preserves P18 only when both semantics are
  present and otherwise restores native Step attention after JANGTQ hydration.
  The real Electron log visibly records the guard. Row 1406 returned exact
  `4`; with tools visibly enabled and the working directory set, row 1418 made
  exactly one real `file_info({"path":"panel/package.json"})`, one result,
  one concise reasoning segment, and exact `STEP-TQ-TOOL4-DONE`, with 192
  `paged+mixed_swa` cached tokens. Rows 1409/1412 are retained as invalid setup
  evidence because their request diagnostics explicitly show `has_tools:false`.
- `CACHE-TUPLE-NORMALIZATION`: VERIFIED-LIVE for mutual exclusion. Current
  source normalizes persisted `paged=true + legacy-disk=true + block-L2=true`
  sessions to `paged=true + legacy=false + block-L2=true` independent of
  migration version. A rebuilt Electron constructor changed the stale Step
  JANGTQ_K row in the real profile; 276 focused session/settings tests and
  typecheck passed. Gemma's older all-off cache default remains a separate red
  row.
- `MANUAL-SINGLE-MODEL`: VERIFIED-LIVE. Starting Step JANG_K from Sessions
  while Zaya was active logged the explicit stop-before-start transition and
  the engine-process count stayed exactly one through loading. Later
  Nemotron-to-Bonsai swap reproduced the same behavior.
- `ZAYA-SPECIALIST-UI-TRUTH`: OPEN. The configured Zaya path is an AppleScript
  specialist whose README defines only `run_applescript`; the main process
  correctly filters generic tools, but Chat Settings still visually exposes
  unrelated File/Search/Shell categories. Generic `file_info` probes are
  out-of-contract and ended without final content.
- Release boundary remains `PARTIAL_NO_RELEASE`. No package, sign, notarize,
  tag, feed, or public-release mutation is authorized.

## 2026-07-16 - OpenPangu and cross-model reasoning/tool finalization correction

- `OPENPANGU-CACHE-PARITY`: VERIFIED-LIVE for settings/argv/health only.
  Electron launched PID 15972 with `--tool-call-parser openpangu`,
  `--enable-auto-tool-choice`, `--reasoning-parser deepseek_r1`, and
  `--disable-prefix-cache`; health reported `openpangu_v2_composite_v1` with
  prefix, paged, prompt-L2, and block-L2 all inactive and zero cache hits.
- `OPENPANGU-STRICT-NATIVE-PARSER`: SOURCE-PASS / LIVE-PARTIAL. The parser now
  declares strict native ownership so malformed `<|tool_call_start|>` debris is
  not reinterpreted by generic repair as another executable tool family. Focused
  parser/openPangu tests pass 38/38, and the added server regression reproduces
  the live malformed `search_files` promotion and returns no call.
- `OPENPANGU-AGENTIC-TOOLS`: FAIL-LIVE. Before the strict parser guard, Electron
  row 1467 promoted gibberish into a wrong `search_files` call with malformed
  args and executed it. After the guard, row 1470 no longer had
  `tool_calls_oai_json` or tool results, but it still stalled in
  "Generating tool call..." after 7 tokens/89.4s and had to be interrupted.
  Do not claim openPangu tool/final completion works.
- `BONSAI-B1-UI-TOOL3-REGRESSION`: VERIFIED-LIVE for the explicit
  exactly-once contract; broader reasoning performance remains PARTIAL. The
  current reproduction generated 6,316 tokens/24,443 raw characters and 46
  tool markers while Electron showed only 57 reasoning characters and two
  speculative tool states. A TQ-off A/B repeated the defect at 4,335 tokens;
  its first schema-valid call appeared at character 3,092, proving the cache
  codec was not the cause. Current source gives Qwen a request-scoped early
  stop only when the user names one exposed tool and explicitly says exactly
  once; ordinary multi-call/interleaved Qwen turns remain open. Two TQ-off and
  six restored-Auto Electron rows each executed exactly one `file_info`, one
  result, and an exact final marker. Auto rows measured 115-244 tokens and
  4.2-7.0s. One TQ-off row needed 1,195 tokens before its first valid call, so
  general pre-call sampling latency is still not performance-green.
- `CROSS-MODEL-POST-TOOL-FINALIZATION`: OPEN. Add Bonsai 1-bit, Bonsai
  ternary, openPangu, MiniMax-M3, DSV4, HY3, Laguna, Step, Nemotron, Mistral,
  Gemma, LFM, Qwen, and Zaya to the current retest matrix for: one real tool
  call, no duplicate reasoning cards, visible final `content` delta/done,
  parser-family correctness, no generic fallback mispromotion, and cache detail
  truth. Release boundary remains `PARTIAL_NO_RELEASE`.

## 2026-07-16 - OpenPangu follow-up: prompt/status fixed, live output still red

- `OPENPANGU-PROMPT-FALLBACK`: SOURCE-PASS / LIVE-FAIL. Source now treats
  `openpangu` / `openpangu_v2` as its own native fallback family, keeps the
  template `tools` kwarg narrowed to explicitly requested tools, and injects a
  request-bound JSON-list example in the real
  `<|tool_call_start|> ... <|tool_call_end|>` format. Focused tests passed
  41/41 selected (`tests/test_openpangu_v2.py`,
  `tests/test_tool_prompt_fallback.py`, `tests/test_openpangu_tool_parser.py`).
  Live Electron row `[PG2-UI-TOOL1]` still failed: Pangu emitted malformed
  reasoning text, no `tool_calls_oai_json`, no tool result, and had to be
  interrupted after 125.2s.
- `OPENPANGU-MTP-HEALTH`: FIXED+VERIFIED-LIVE for status reporting only.
  `native_mtp.inspect_native_mtp_bundle` now accepts openPangu's documented
  included MTP layers stored as extra `model.layers.46-48` rather than `mtp.*`
  keys and reports `weights_present_runtime_unwired` with
  `runtime_mtp_mode=included_but_dropped_for_runtime`. After UI restart on PID
  replacement, `/health` reported `issues: []`, `runtime_supported:false`, and
  `runtime_available:false`.
- `OPENPANGU-CACHE-ARCHITECTURE`: VERIFIED-LIVE as no generic reuse. Running
  server argv includes `--disable-prefix-cache`; health reports
  `openpangu_v2_composite_v1`, components `mla_latent_kv`,
  `dsa_indexer_state`, `swa_rotating_window`, and
  `path_dependent_conv_state`, with generic prefix/paged/prompt-L2/block-L2 all
  false/unsupported. Do not enable generic paged/L2 for Pangu until a typed
  composite prompt-boundary codec exists.
- `OPENPANGU-RUNTIME-COHERENCE`: FAIL-LIVE. API diagnostics against the same
  live server show the issue is below UI finalization: a simple
  `Answer exactly: PANGU-SIMPLE-OK` prompt loops/corrupts the marker under both
  thinking off and thinking on, and a tool prompt emits malformed native JSON
  (`{"name: "file_info"...}`) that the strict parser correctly rejects. Treat
  Pangu tool/parser rows as blocked by runtime/quant/template coherence until
  the base generation row is coherent.

## 2026-07-16 - OpenPangu parser repair source/API partial, Electron still red

- `OPENPANGU-NATIVE-PARSER-TRUNCATION`: SOURCE-PASS / API-PARTIAL. The
  dedicated parser now accepts only bounded openPangu-native truncations: a
  whole-turn native JSON-list payload, optionally detagged by tokenizer decode,
  whose tool-call object is complete but whose closing list bracket or
  `<|tool_call_end|>` sentinel is missing. It still rejects missing object
  braces, embedded prose, and trailing commentary. Focused openPangu tests pass
  47/47 selected across `tests/test_openpangu_tool_parser.py`,
  `tests/test_tool_prompt_fallback.py`, and `tests/test_openpangu_v2.py`.
- `OPENPANGU-RESPONSES-FIRST-CALL`: API-PASS for first-call extraction only.
  After UI restart of `jangq-ai/openPangu-2.0-Flash-JANG_2L`, direct
  `/v1/chat/completions` with `enable_thinking:false` returned
  `finish_reason:"tool_calls"` and one `file_info` call with
  `{"path":"panel/package.json"}`. Direct non-stream `/v1/responses` with
  `enable_thinking:false` returned one `output[].type="function_call"` for the
  same tool/path. This does not prove post-tool final content.
- `OPENPANGU-POST-TOOL-FINAL`: LIVE-FAIL. Direct two-step API continuation
  after a valid `file_info` result did not emit the exact requested marker; it
  generated corrupted/repeated text such as `PG2-FINAL2-D-ONE-2000...` or
  summaries and ended by length. Do not claim openPangu full agent loop works.
- `OPENPANGU-ELECTRON-FRESH-TOOLS`: LIVE-FAIL. Fresh Electron chat
  `d3e7ec71-3eba-4b36-ad88-24b8d586e138` with DB-verified
  `chat_overrides.enable_thinking=0` and built-in tools enabled still failed:
  `[PG2-UI-FRESH1]` hung in tool buffering and was interrupted after 77.7s with
  zero tokens; after streaming parser repair and restart, `[PG2-UI-FRESH2]`
  finished in 0.9s with the visible warning
  `The 'openpangu' native tool parser did not produce a schema-valid function
  call`, no `tool_calls_oai_json`, and no tool result. Screenshot:
  `/tmp/pangu-ui-fresh2-zero-tool-red.png`.
- `CHAT-SETTINGS-THINKING-OFF-PERSISTENCE`: VERIFIED-LIVE for the fresh Pangu
  chat only. The first automated Off+Save attempt clicked the wrong `Save`
  button and left `enable_thinking=NULL`; the correct Chat Settings Save button
  at the inference panel persisted `enable_thinking=0` in
  `chat_overrides`. This confirms the setting can persist, but the row above
  proves it is not sufficient to make Pangu Electron tool use pass.

## 2026-07-16 - openPangu JANG_3M correction after runtime and typed-cache repair

- `OPENPANGU-RUNTIME-COHERENCE`: VERIFIED-LIVE for the current JANG_3M scoped
  row, superseding the prior JANG_2L failure above. Source remaps all three
  causal-convolution checkpoint names to the nested runtime modules, transposes
  their tensors, uses the checkpoint-exact DSA RMSNorm, and fails closed on
  incomplete landing. Live startup reported 2826/2826 leaves, 46 layers, and
  138 causal convolutions.
- `OPENPANGU-ARCHITECTURE-EXECUTION`: VERIFIED-LIVE at 2,104 tokens. Every one
  of the 16 DSA layers logged real sparse activation above top-k=2048. The same
  forward reported all 46 production layers, 30 SWA layers, four mHC streams,
  128 attention sinks, MLA KV rank 512, and the 512-token SWA window.
- `OPENPANGU-NO-TURBOQUANT`: VERIFIED-LIVE across UI, DB, argv, startup logs,
  health, and cache stats. The UI says `TURBOQUANT OFF`; saved quantization is
  `none`; no q4/q8/TQ flag appears in argv; logs set `VMLX_DISABLE_TQ_KV=1`;
  health reports both generic KV quantization and TurboQuant disabled.
- `OPENPANGU-EXACT-TYPED-CACHE`: VERIFIED-LIVE for cold, warm-memory, and
  process-restart prompt-disk paths. The typed N-1 record owns MLA KV, DSA
  indexer state, SWA rotation metadata, and all three convolution states.
  Generic paged/block codecs remain off. Final row 1527 restored 2,075/2,076
  prompt tokens from disk, then completed one exact tool call and final marker.
- `OPENPANGU-AGENTIC-TOOLS`: VERIFIED-LIVE for the exact one-tool Electron row.
  Row 1527 persisted one `file_info({"path":"panel/package.json"})`, one result,
  and only `PG3M-CACHE-20260716-A-DONE` as visible final content; the earlier
  false schema warning no longer appeared.
- `OPENPANGU-MTP-RUNTIME`: OPEN. Three MTP layers are detected in config but
  dropped by the current runtime. No active MTP or speedup claim is made.
- `OPENPANGU-LONG-CONTEXT`: PARTIAL. Sparse DSA was live-proven at 2,104 tokens;
  the advertised 524,288-token limit and long-context retrieval quality were
  not exercised.
- `CROSS-MODEL-POST-TOOL-FINALIZATION`: remains OPEN for all families not
  covered by current live rows and for long-soak/reasoning-card performance.
  The openPangu scoped recovery does not close Bonsai variance or other family
  gates. Release remains `PARTIAL_NO_RELEASE`.

## 2026-07-16 - Bonsai ternary native TQ storage boundary

- `BONSAI-TERNARY-TQ-AUTO`: VERIFIED-LIVE for storage only. Auto now resolves
  to native TQ8 for the 16 attention-KV layers; mid-decode compression is off
  and all 48 SSM companions remain native. Health and startup policy agree.
- `BONSAI-TERNARY-TQ-NONE`: VERIFIED-LIVE. The visible None selection persisted,
  restarted with `--kv-cache-quantization none`, set the hard-disable env, and
  produced no native TQ writes/hits. Two fresh multi-turn exact-one-tool rows
  completed coherently.
- `BONSAI-TERNARY-PAGED-L2-NATIVE-TQ`: VERIFIED-LIVE for attention-KV record
  integrity. Seven `turboquant_kv` blocks were written and all seven decoded
  after process restart (`disk_hits=7`, `tq_native_hits=7`). The source path
  preserves seed, complete offset, cache type, and native TQ rewrap through
  prefix/paged/block-disk/scheduler/MLLM boundaries.
- `BONSAI-HYBRID-SSM-RESTART`: PARTIAL by design. Persistent SSM restore stays
  quarantined because prior live numeric comparison diverged. Restart uses the
  decoded attention-KV record only as diagnostic evidence and full-prefills or
  asynchronously rederives required SSM state. No restart acceleration claim.
- `BONSAI-TERNARY-REASONING-CONTINUITY`: PARTIAL. Fresh exact-one-tool rows
  finalize correctly, but one clean multi-turn row reopened a short reasoning
  segment after the tool result. An older mixed-history chat repeated the same
  tool four times; it is retained as a failure row pending stale-history and
  ordinary multi-call soak coverage.
- `BONSAI-1BIT-NATIVE-TQ-MATRIX`: OPEN. Do not infer 1-bit behavior from the
  separately loaded ternary bundle. Auto/None, native TQ disk boundaries,
  reasoning, exact tool finalization, and process restart need current Electron
  evidence on `Bonsai-27b-1bit-JANG`.
- `RELEASE`: `PARTIAL_NO_RELEASE`; no package, signature, notarization, tag,
  feed, or upload is cleared by this scoped result.

## 2026-07-16 - Bonsai 1-bit exact-once continuation and native TQ

- `BONSAI-1BIT-BUNDLE-IDENTITY`: VERIFIED-LIVE. Source manifest says
  `JANG_AFFINE_1BIT`, storage bits 1, lossless runtime expansion bits 2, actual
  bits 1.1128; the running Electron header showed the same profile/actual bits.
- `BONSAI-1BIT-MULTITURN-EXACT-ONCE-PRE-FIX`: FAIL-LIVE. Row 1620 executed five
  tools and six reasoning segments, generated 3,352 tokens/92.3 seconds, and
  was interrupted. Source matched only `after [the] tool result`; the live
  phrase `after the real tool result` therefore retained tool schemas across
  follow-ups.
- `BONSAI-1BIT-MULTITURN-EXACT-ONCE-CURRENT`: VERIFIED-LIVE for the bounded
  contract. The matcher now allows same-clause modifiers but still requires
  `exactly once` and `reply exactly`. None rows 1623/1626 and Auto rows
  1632/1635 each persisted one reasoning segment, one named real tool, one
  result, and exact final content. General multi-call work remains open.
- `BONSAI-1BIT-TQ-NONE`: VERIFIED-LIVE. Visible select, DB, PID argv, and health
  agreed on the hard disable; no native TQ counters were present.
- `BONSAI-1BIT-TQ-AUTO`: VERIFIED-LIVE for storage only. Health resolved native
  TQ8 on 16 attention layers, 48 native SSM companions, live encode disabled,
  and persisted native block records.
- `BONSAI-1BIT-TQ-RESTART`: VERIFIED-LIVE for record integrity. Row 1629 wrote
  three native blocks; restart row 1632 decoded all three and finished exactly.
  Current counters later reached `tq_native_hits=8`. SSM persistent restore is
  still suppressed and receives no speed/cache-token credit.
- `TQ-HEALTH-TELEMETRY-LABEL`: VERIFIED-LIVE. Storage-only compress calls now
  surface as `storage_encode_telemetry` plus neutral
  `codec_compress_telemetry`; `live_encode_telemetry` is absent while
  `live_encode_enabled=false`.
- `GATEWAY-SINGLE-MODEL-SWAP`: VERIFIED-LIVE for the ternary-to-1-bit switch.
  Sessions displayed one ACTIVE 1-bit server and ternary INACTIVE, and the
  process list contained one local model server.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`; ordinary multi-call soak, VL,
  cross-family parser rows, all remaining models/protocols, packaging, signing,
  notarization, feeds, and public release remain open.

## 2026-07-16 - Gateway port/LAN address live correction

- `GATEWAY-PORT-ASSIGNMENT`: VERIFIED-LIVE. UI change to 18080 persisted in
  SQLite and moved the dev Electron listener to the selected port. Restoring
  8080 moved the listener and UI URL back to localhost:8080; current health,
  DB, and listener agree.
- `GATEWAY-LAN-ADDRESS`: VERIFIED-LIVE after fix. Pre-fix wildcard binding was
  correct but the dashboard advertised APIPA `169.254.62.28`. Source now
  rejects unusable/link-local IPv4 and ranks RFC1918 before CGNAT/public.
  Post-fix Electron advertised `192.168.1.110:18080`, actual listener was
  `*:18080`, and the exact advertised health URL responded successfully.
- `GATEWAY-LAN-SELECTOR-TESTS`: PASS. Three selector tests plus the gateway
  Ollama and single-model suites passed 65/65; panel typecheck passed.
- `GATEWAY-SINGLE-MODEL-STATE`: VERIFIED-LIVE for the current named swap. DB,
  health, and UI report single-model mode enabled; the dashboard showed one
  running Bonsai 1-bit server, matching the current Sessions/process proof
  that starting 1-bit left ternary inactive and one model process.
- `GATEWAY-PORT-CONFLICT-CAVEAT`: OPEN. The installed app separately owns
  wildcard port 8080 and was intentionally left running. The dev app's exact
  host/port behavior is proven, but machine-global conflict policy and the full
  OpenAI/Anthropic/Ollama streaming matrix remain open.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 - Qwen3.6 MTP UI reasoning-replay / missing visible answer (P0)

- `Q36MTP-UI-REASONING-REPLAY`: OPEN (P0). Live pre-patch Electron evidence
  (20:35-20:38, before the 20:53 patched relaunch): in a chat with a prior
  run_command tool turn, two different `[Q36MTP-UI-NOTOOL2/3]` exact-string
  prompts each rendered a Reasoning block BYTE-IDENTICAL to the first NOTOOL
  turn (same 1300 chars / 456 tokens, still reasoning about the first turn
  chunks) and an EMPTY visible assistant answer — no output text, no tool
  call. Timing stats were live per turn (TTFT 2.37s vs 4.65s, prompt 1245 vs
  1722), so a real generation ran while stale reasoning was displayed and no
  content was surfaced. Distinct signature from the answer-pass reasoning-only
  rows: this is stale reasoning REPLAY plus missing visible content.
- Caveats: evidence app predates commit `9d8a730ec`
  (toolHistoryReplay/answer-pass fixes) and ran with Engine Manager
  "Not installed" (venv missing from SSH-launched PATH). Patched app with
  correct env relaunched 20:53; live Codex UI retest of this exact repro is
  in progress. Treat as a global bug class (any reasoning model emitting
  reasoning then no visible answer) until the patched retest proves otherwise.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 - Relaunched Electron does not adopt live orphaned model server

- `Q36MTP-UI-ORPHAN-SERVER-NOT-ADOPTED`: OPEN. After the 20:53 Electron dev
  relaunch, the Qwen3.6-27B-MXFP8-CRACK-MTP session UI shows
  "Model is not running." + Load Model button while that session's
  vmlx-serve (spawned 20:45 by the pre-relaunch app) is still alive on port
  8032 with `/health` `model_loaded=true`, `status=standby_soft`. Either the
  panel fails to re-adopt a healthy orphaned server on startup (process
  handle lost across relaunch) or it renders standby_soft as not-running.
  Consequence: every chat turn in the session is blocked or silently spawns a
  duplicate 27B server. Live evidence: CDP screenshot 21:16
  (release-gates/20260717_qwen36_mtp_stream_history/ui-proof/02-turnA.png,
  banner visible) + concurrent health JSON. Needs a structural adopt-or-truth
  fix, not a label tweak.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 - CORRECTION + new bug: single-model swap left adopted server alive

- `Q36MTP-UI-ORPHAN-SERVER-NOT-ADOPTED`: CORRECTED — adoption WORKED
  (`[STARTUP] Adopted 1 vmlx-engine process(es)` in the 20:53 relaunch log).
  The "Model is not running" banner had a different cause: single-model mode
  stopped session e5a12a4d when session 08921892 (Hy3, ~73.7GB) was started
  (`[SESSIONS] single-model mode: stopping session e5a12a4d... before
  starting 08921892...`). Renderer banner was truthful for a stopped session.
- `SINGLE-MODEL-SWAP-ORPHANS-ADOPTED-SERVER`: OPEN (P1). That single-model
  stop did NOT kill the ADOPTED Qwen3.6 server: /health on 8032 still
  answered `model_loaded=true, standby_soft` ~15 min after the stop, and the
  Hy3 start logged `RAM: 63.9 GB free / 137 GB` with the explicit
  may-exceed-memory warning — i.e. the swap left ~27GB resident and loaded a
  73.7GB model on top (double-residency contention class, same failure mode
  as the 07-14 DSV4 case). Suspected cause: stop path for adopted sessions
  (process handle null, killPid by pid/group) not reaching the adopted pid on
  the single-model swap rail. Needs source trace + live re-proof of the swap.
- Codex gpt-5.6-sol xhigh CDP run (21:14-21:17) correctly reported
  FAIL/UNVERIFIED for turns A-E: composer disabled by the stopped session;
  it changed nothing (`model_load_attempted:false`). Evidence:
  ui-proof.json + 01-initial/02-turnA.png (to be copied into the gate dir).
- Post-restart live turn on 8032 (21:3x, panel log): content 50 chars +
  reasoning 1596 chars, 634 tokens, TTFT 0.40s — visible content emitted on
  the patched panel for a plain turn. NOT yet proof for the NOTOOL replay
  signature; UI A-E rerun + Auto-mode API repro in flight.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 - Re-correction after source trace: swap-orphan UNCONFIRMED; add log-retention defect

- `SINGLE-MODEL-SWAP-ORPHANS-ADOPTED-SERVER`: DOWNGRADED to UNCONFIRMED.
  Source trace shows stopSession() handles adopted pids (SIGTERM, 1.5s wait,
  SIGKILL, then port fallback), and pid 10970's engine log ends in a
  GRACEFUL uvicorn shutdown — consistent with the panel killing it correctly
  at either the swap or the later manual restart. My health-probe ordering vs
  the swap is ambiguous (concurrent live manual use), so the orphan claim is
  not proven. Needs a CONTROLLED live repro: adopt server, enable
  single-model, start a second model, verify the first process dies.
- The 21:13 "silent server death" is most plausibly a MANUAL Stop:
  stopSession() emits no console line and deletes the session log buffer,
  which exactly matches the observed no-log-line + empty getLogs + stopped
  DB row. Not counted as an engine crash without a controlled repro.
- `STOP-DESTROYS-SESSION-LOG-BUFFER`: OPEN (P2, diagnosability). sessions.ts
  stopSession() does `this.logBuffers.delete(sessionId)`, so after ANY stop
  (manual, swap, monitor) the server's stderr history is gone and no crash
  postmortem is possible from the app. Fix direction: retain last N lines in
  a stopped-session buffer or append to a per-session on-disk log.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 22:0x - Q36 MTP: replay class FIXED-LIVE; NEW scoped P0 = tool-call never emitted

- `Q36MTP-UI-REASONING-REPLAY`: CLOSED-BY-LIVE-PROOF for the patched build.
  Codex gpt-5.6-sol UI run2 (real Electron session, prior tool turn, both
  NOTOOL prompts): visible answers EXACT (`QP10-...-ZA60`, `HG10-...-QR60`),
  reasoning 3001 vs 3121 chars (NOT byte-identical), Busan recall PASS, no
  blank/leak/loop on turns A/B/D/E. API rail same-server 4/4 PASS with
  enable_thinking omitted (Auto parity). Evidence:
  release-gates/20260717_qwen36_mtp_stream_history/{q36-auto-api-repro.json,
  ui-proof/run2-*.png, ui-proof/run2-ui-proof.json}.
- `Q36MTP-TOOL-CALL-NOT-EMITTED`: OPEN (P0, scoped). UI run2 turn C
  ("Use the run_command tool to run pwd") = reasoning rail streams 501 chars
  ending literally with "Let's start with the tool call." then the stream
  ENDS: no tool_call item, no visible content, 164 completion tokens, live
  stats (18.0 t/s, TTFT 2.52s). Model was cut at the reasoning->tool-call
  boundary or its tool tokens were swallowed. Tools-enabled Responses API
  repro (auto + required + no-tools control) running to isolate engine vs
  panel. This matches Eric's global "reasoning then no tool/no answer"
  complaint — treat as cross-family suspect until proven scoped.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 22:2x - Q36MTP tool rail: fix landed d5e9e5177, tool loop VERIFIED-LIVE

- `Q36MTP-TOOL-CALL-NOT-EMITTED`: ROOT-CAUSED + FIXED + PARTIALLY CLOSED.
  Two stacked panel defects in the chat rail (chat.ts): (1) zero-tool
  speculative-buffer restore re-entered emitDelta marker detection, so
  restored text containing the hallucinated `<run_command>` dialect was
  re-suppressed permanently (live-proven pre-fix: content 0, buffered true;
  post-fix-v1 live: content 96, buffered false); (2) the final persistence
  sanitizer stripped the same dialect to an EMPTY answer (DB content len 0,
  renderer "No visible response was produced"). Fix d5e9e5177: restore
  bypasses detection + fences dialect text; sanitizer never-empty guard
  preserves verbatim text when stripping would blank a no-tool turn.
  Tests: responses-stream-recovery 9/9, panel suite 2296 green, typecheck.
- VERIFIED-LIVE on the patched app (22:17): full native tool loop — server
  emitted function_call (tool calls: 1), panel executed builtin run_command,
  continuation completed, visible answer "The current working directory is:
  /Users/eric/mlx/vllm-mlx", stats live. Evidence: ui-proof/run6-toolloop-pass.png.
  API rail control: q36-tool-api-repro.json (tools-auto + tools-required both
  emit function_call; no-tools control emits visible text).
- REMAINING PARTIAL: the hallucinated-dialect zero-tool path end-to-end
  (dialect text visibly rendered + persisted) is unit-pinned but not yet
  observed live post-v2 (model emitted a native call this run). Next natural
  occurrence or forced repro will close it. Gateway/API rails untouched by
  the fix (no import of the shared module outside ipc/chat.ts — verified).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 22:4x - Q36MTP row: dialect path CLOSED-BY-LIVE-PROOF; NOTOOL regression green on v2/v3

- `Q36MTP-TOOL-CALL-NOT-EMITTED`: CLOSED. Live on the v3 app: forced dialect
  turn logged buffering -> "Restoring 34 authoritative" -> "Sanitizer emptied
  a 44-char answer with no executed tool - preserving verbatim in a fence";
  DB persisted the fenced <run_command>/pwd text (44 chars, visible). Guard
  v3 keys on receivedToolCalls/allGeneratedContent because the speculative
  "generating" status had masked the v2 guard (live-caught: empty 12-char
  fence). NOTOOL-A/B regression on patched app: exact strings, reasoning
  differs, prefix cache hits (5241/5399 and 5241/5502 cached). Tool loop
  pass (run6) stands. Evidence: ui-proof/run6-toolloop-pass.png,
  ui-proof/run7-dialect-guard-pass.png, panel log lines in gate README.
- `RELEASE`: remains `PARTIAL_NO_RELEASE` (matrix rows pending).

## 2026-07-17 23:0x - Bonsai row: swap-kill verified, q8 TQ policy verified, two truthfulness defects

- `SINGLE-MODEL-SWAP-ORPHANS-ADOPTED-SERVER`: CLOSED-NOT-REPRODUCIBLE.
  Controlled repro: with adopted Qwen server pid 19298 live on 8032, starting
  Bonsai session 5fd14571 logged the single-model stop and pid 19298 was DEAD
  within seconds. The 21:0x suspicion was concurrent-manual-use observation
  ordering, not a defect.
- Bonsai-27b-1bit-JANG live row (port 8030, paged+disk, TQ Auto):
  bundle ground truth model_type=qwen3_5 + vision_config (autodetect
  family/VLM CORRECT); q8 policy ACTIVE per /v1/cache/stats
  turboquant_kv_cache: enabled=true, storage q8 (key/value bits 8),
  auto_policy bonsai_hybrid_*, tq_native_writes=15/tq_native_hits=135,
  native_cache hybrid_ssm_typed. Cold+warm chat turns: exact visible
  BONSAI-T1-OK / BONSAI-LONG-OK, reasoning separated; scheduler hits=6,
  tokens_saved=2703 (870-token warm prefix reused).
- `HEALTH-KV-QUANT-FLAG-FALSE-WHILE-TQ-ACTIVE`: OPEN (P1, truthfulness;
  carries over 20260716 PARTIAL, now precisely reproduced). /health
  kv_cache_quantization={enabled:false} while storage TQ q8 is active with
  live native writes/hits. Suspect: summary reads live_encode_enabled
  (false, compress_after=0 per TQ-KV-NEVER-compresses finding) instead of
  storage_encode_enabled. Fix in vmlx_engine server health assembly +
  regression test; then live re-proof.
- `USAGE-CACHED-TOKENS-MISSING-NONSTREAM-CHAT`: OPEN (P1, truthfulness).
  Non-stream /v1/chat/completions usage on Bonsai omitted
  cached_tokens/cache_detail on a warm 870-token identical prompt while
  scheduler_cache counted the hit (hits 6, tokens_saved 2703). Responses/
  stream rails report these (Q36 rows show cache_detail). Verify grammar per
  CACHE_DETAIL_GRAMMAR.md across rails; fix usage assembly; retest.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 23:2x - Health kv-quant truthfulness FIXED-LIVE; JIT parity row opened

- `HEALTH-KV-QUANT-FLAG-FALSE-WHILE-TQ-ACTIVE`: CLOSED-BY-LIVE-PROOF.
  Root cause: _kv_cache_quantization_status read only legacy
  scheduler._kv_cache_bits (explicit live quantization) and was blind to the
  TurboQuant storage codec. Fix: helper consumes the computed
  _turboquant_kv_cache_status at both /health and /v1/cache/stats call
  sites; storage codec reports enabled=true mode=turboquant-storage with
  bits/key_bits/value_bits/auto_policy; legacy bits report mode=live.
  Tests: tests/test_kv_quant_status_truthfulness.py 5/5 +
  turboquant contract 8/8. Live: Bonsai 8030 restart -> /health
  kv_cache_quantization {enabled:true, mode:turboquant-storage, bits:8,
  turboquant-q8, bonsai_hybrid_attention_kv_storage_tq8}. Panel consumers
  checked: PerformancePanel renders bits (mirrored), CachePanel reads the
  TQ section directly.
- `JIT-COMPILE-UI-CLI-PARITY`: OPEN (Eric directive). For models that do not
  support JIT/mx.compile (VLM/mllm streaming path logs "Ignoring stale JIT
  flag"), the UI toggle must display OFF and the CLI preview/args parity must
  omit/disable it. Verify per-model in the settings-parity pass; add to the
  UI wiring matrix.
- `USAGE-CACHED-TOKENS-MISSING-NONSTREAM-CHAT`: still OPEN — next: trace
  cached_tokens attribution on the batched/MLLM non-stream rail (get_usage
  forwards it; the GenerationOutput arrives with 0 despite scheduler hits).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 23:4x - JIT parity: launch rail verified live; visual toggle/preview check in flight

- `JIT-COMPILE-UI-CLI-PARITY`: PARTIAL progress. Source trace shows all three
  surfaces share one guard set (VLM/mlx-vlm, TurboQuant, hybrid SSM, DSV4,
  M3, Zaya CCA, FlashMoE, distributed): sessions.ts effectiveEnableJit gates
  launch args + logs "Ignoring stale JIT flag because <reason>";
  SessionSettings.tsx:412+663 uses the same expression for the CLI preview;
  SessionConfigForm.tsx:1270 masks the toggle checked-state with the same
  actives. LIVE: running Bonsai (VLM+TQ) argv contains ZERO --enable-jit
  despite stored default enableJit:true. Remaining evidence: visual toggle
  state + preview text per session — box-side Codex gpt-5.6-sol computer-use
  run in flight (box codex auth RESTORED by Eric). Watch for drift risk:
  form uses multimodalActive/flashMoeActive naming vs isVLM/effective* in
  the other two copies — preview logic is a COPY, keep synced.
- `USAGE-CACHED-TOKENS-MISSING-NONSTREAM-CHAT`: bounded source audit running
  (attribution of response.cached_tokens=0 on warm non-stream MLLM hits).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-17 23:1x - JIT parity CLOSED live; hybrid SSM companion never stored (new P1)

- `JIT-COMPILE-UI-CLI-PARITY`: CLOSED-BY-LIVE-PROOF. Box-side Codex
  gpt-5.6-sol run (jit-findings.json + 7 jit-*.png in the q36 gate ui-proof
  dir; interaction guard: nothing changed/saved/restarted): Bonsai (VLM+TQ
  hybrid) and Qwen3.6 MTP both show JIT toggle UNCHECKED + disabled and CLI
  preview WITHOUT --enable-jit; my own screenshot read confirms the truthful
  amber banner ("JIT is disabled for hybrid SSM/Mamba cache models...");
  running Bonsai argv has zero --enable-jit despite stored default true.
  All three surfaces share one guard set (sessions.ts effectiveEnableJit,
  SessionSettings.tsx preview copy, SessionConfigForm checked-mask).
- `HYBRID-SSM-COMPANION-NEVER-STORED`: OPEN (P1 — supersedes/explains
  USAGE-CACHED-TOKENS-MISSING-NONSTREAM-CHAT). Source audit + live probe:
  Bonsai warm 870-token "hit" was counted by prefix_cache.fetch_cache
  (hits/tokens_saved increment BEFORE consumer acceptance,
  prefix_cache.py:452-519) but the hybrid gate REJECTED reuse for missing
  SSM companion (mllm_batch_generator.py:5866-5916 -> release + full
  re-prefill; req._cached_tokens stays 0 from :5602). Live ground truth:
  /v1/cache/stats ssm_companion.entries=0 after MULTIPLE completed requests
  on this hybrid model — companion state is never persisted to L1, so warm
  hybrid reuse never happens and tokens_saved is fiction. Fix plan (audit):
  (1) root-cause SSM companion capture/store after generation
  (_mark_required_ssm_checkpoint / state cache store on _cleanup_finished);
  (2) gate tokens_saved credit on actual reuse or expose
  hybrid_kv_without_ssm counter; (3) hardening: mllm_scheduler.generate()
  should stamp request._cached_tokens/_cache_detail onto final_output
  (stream rail latches, non-stream does not — server.py:17116 vs :6641).
  Note: Q36MTP server DOES report paged+ssm cached — its companion works;
  Bonsai path is the broken one. CACHE_DETAIL_GRAMMAR.md exists on this
  checkout (audit worktree flag was stale).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 00:0x - SSM companion narrowed to silent hybrid-detection failure; engine affine-JIT default hygiene

- `HYBRID-SSM-COMPANION-NEVER-STORED`: narrowed. Bonsai server log contains
  ZERO SSM store/rederive lines -> the store gate
  (mllm_batch_generator.py:6596, _is_hybrid && _ssm_companion_enabled) never
  fires. _is_kv_like DOES handle TurboQuantKVCache (by class name, :1463),
  so prime suspect is hasattr(language_model, "make_cache")=False for the
  JANG-affine-loaded wrapper -> _hybrid_kv_positions None -> _is_hybrid
  False, SILENTLY (no warning). Box Codex gpt-5.6-sol xhigh dispatched with
  load-probe + fix + honest-accounting + non-stream-stamping + regression
  plan (/tmp/PROMPT-ssm.txt, report to /tmp/ssm-fix-report.md, uncommitted).
- `ENGINE-AFFINE-JIT-DEFAULT-HYGIENE`: OPEN (P3). Engine cli logs "JANG
  affine model detected - defaulting --enable-jit ON" for Bonsai (hybrid VLM
  TQ) even though the panel deliberately omits the flag; the runtime later
  refuses ("JIT: Skipping mx.compile - MLLM hybrid cache contains
  ArraysCache"), so JIT is NOT actually active (defense-in-depth held; the
  JIT-COMPILE-UI-CLI-PARITY closure stands). Hygiene fix: gate the affine
  JIT default on the same mx.compile-safety conditions so logs/args do not
  claim ON for models the runtime will refuse.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 00:3x - HYBRID-SSM-COMPANION + cached-usage rows CLOSED-BY-LIVE-PROOF

- `HYBRID-SSM-COMPANION-NEVER-STORED` + `USAGE-CACHED-TOKENS-MISSING-NONSTREAM-CHAT`:
  CLOSED-BY-LIVE-PROOF (commit above). Codex gpt-5.6-sol authored the fix
  (report /tmp/ssm-fix-report.md); I ran gates + live proof: 123 pytest pass
  (new tests/test_hybrid_ssm_companion_regressions.py + scheduler cache + TQ
  contract + kv-quant); Metal probe scripts/probe_mllm_hybrid_detection.py
  on the real Bonsai bundle shows 16 KV + 48 ArraysCache and resolver picks
  language_model correctly; post-restart live: cold 871-token request
  restored cached 832 `paged+ssm+disk` (restart-restore axis), warm cached
  832 `paged+ssm` IN NON-STREAM USAGE, ssm_companion.entries=2, tokens_saved
  backed by real reuse. Note: fresh-load legacy detection also passes, so
  the live zero-entries state was runtime-path dependent; the new WARNING on
  non-hybrid resolution ensures any recurrence is visible, and the resolver
  covers the non-forwarding wrapper shape (pinned by regression).
  Minor model quirk observed: cold answer typo "BONSEI-SSM-OK" (1-bit model
  stochastic; warm exact) — not an engine defect.
- `RELEASE`: remains `PARTIAL_NO_RELEASE` (matrix rows pending).

## 2026-07-18 00:5x - Bonsai UI multiturn: T1/T2 PASS; tool-continuation stale-replay + cached>prompt telemetry

- Bonsai UI multiturn (session 5fd14571, chat REBOUND from the Qwen chat via
  the header model-switcher — carries full prior history): T1 exact
  `BONSAI-UI-T1-OK` (17 chars + 339 reasoning, 50 t/s); T2 recall exact
  `42-Busan` with 5720/5758 paged+ssm cached, TTFT 0.68s — companion fix
  visibly working in UI. Screenshot bonsai-ui-multiturn.png (gate ui-proof).
- `BONSAI-TOOL-CONTINUATION-STALE-REPLAY`: OPEN (P1). T3 "run pwd": tool
  loop completed (2x run_command chips, both executed) but the FINAL visible
  answer is a VERBATIM copy of the earlier [Q36MTP-V2-NOTOOL-A] user prompt
  + its `QP10-...-ZA60` answer from the rebound history — never mentions the
  pwd result. 1255 tokens genuinely generated (not a render bug; persisted
  in DB). Suspects: tool-continuation history assembly for a
  model-switched chat confusing the model, or 1-bit model parroting; needs
  continuation-request capture to decide. Eric's emission class — do not
  minimize as model quirk without the capture.
- `USAGE-CACHED-GT-PROMPT-ON-CONTINUATION`: OPEN (P2, telemetry truth). T3
  stats line: `2077 prompt (5720 paged+ssm cached)` — cached exceeds prompt.
  Likely the continuation usage line mixes the first stream's cached count
  with the follow-up prompt count (panel aggregation or engine usage).
- Also noted: chats REBIND to another session via header switcher by design;
  cross-model history carryover is then fed to the new model — matrix rows
  should use fresh chats unless testing carryover deliberately.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 01:1x - Fresh-chat control clean; cached>prompt reproduces on fresh chat (upgrade)

- `BONSAI-TOOL-CONTINUATION-STALE-REPLAY`: SCOPED by control test. FRESH
  Bonsai chat + same tool prompt = tool emitted, executed once, final answer
  correct pwd result (239 tokens, 6.5s). The stale-replay manifests only on
  the REBOUND chat carrying cross-model history — history-conditioned model
  parroting on 1-bit; panel/engine tool machinery sound on fresh chats for
  both Bonsai and Qwen. Remaining action: capture the continuation
  requestMessages for a rebound chat before final classification; matrix
  rows use fresh chats.
- `USAGE-CACHED-GT-PROMPT-ON-CONTINUATION`: UPGRADED (P1, engine suspicion).
  Reproduces on a FRESH chat: continuation stats `pp: 481 tokens (3904
  cached)` — cached 8x the prompt, impossible as a prefix match. Suspect the
  new cached-token stamping/credit path (1657ed312: max() latch or block
  credit counting stored-sequence length instead of matched-prefix length)
  over-credits on tool continuations. Trace _record_cache_hit credit size vs
  matched tokens + the stamping max() on follow-up streams; fix + regression;
  re-proof that warm hits still report correctly after.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 01:3x - USAGE-CACHED-GT-PROMPT CLOSED-BY-LIVE-PROOF

- `USAGE-CACHED-GT-PROMPT-ON-CONTINUATION`: CLOSED (commit above). Root
  cause was PANEL aggregation, not the engine credit path: promptTokens was
  overwritten per stream (last stream wins) while cachedTokens latched the
  max across streams — pairing the continuation prompt (481) with the first
  stream cached count (3904). Fix: fold per-stream pairs into exchange
  totals at each follow-up boundary with per-stream clamping. Panel 2296
  tests + typecheck green; live fresh-chat tool turn now reports
  pp 4743 (4384 cached) — coherent. Engine fix 1657ed312 exonerated (its
  per-stream values were individually truthful).
- `RELEASE`: remains `PARTIAL_NO_RELEASE` (next: affine-JIT hygiene P3,
  rebound-chat continuation capture, DSV4-Flash row, remaining matrix).
- Bundled-python note for next release build: engine changes tonight
  (1657ed312, aff768f75) require scripts/bundle-python.sh before packaging.

## 2026-07-18 02:1x - ENGINE-AFFINE-JIT-DEFAULT-HYGIENE CLOSED-BY-LIVE-PROOF

- `ENGINE-AFFINE-JIT-DEFAULT-HYGIENE`: CLOSED (commit above). cli.py affine
  JIT default now gated on is_mllm/hybrid cache_type; Bonsai restart logs
  "JIT default stays OFF: multimodal/VLM streaming path is not mx.compile
  safe" and the old "defaulting --enable-jit ON" line is gone (0 vs 1 in
  session log). Regression tests/test_affine_jit_default_gate.py.
- `RELEASE`: remains `PARTIAL_NO_RELEASE` (next: DSV4-Flash row, then
  Hy3/Gemma/Laguna/Step/Nemotron/MiniMax matrix; bundle-python before any
  packaging).

## 2026-07-18 02:5x - DSV4-Flash row: cache axes VERIFIED-LIVE; effort-none degeneration reopened

- DSV4-Flash live row (session a6810958, port 8012, 97GB load OK, 99.7GB
  active): autodetect CORRECT (family deepseek-v4, dsml/deepseek_r1 parsers,
  native composite prefix cache active). Cache truth: turboquant enabled
  FALSE (never generic TQ on DSV4 — honored), native_cache
  deepseek_v4_v7/native_composite, kv_cache_quantization truthfully false
  (aff768f75 does not mislabel DSV4), block_size 256. API: effort=high exact
  DSV4-high-OK with 317-char reasoning (cold AND warm); long-prompt warm hit
  cached 978/979 `paged+dsv4` IN NON-STREAM USAGE (d841fc799/1657ed312
  working on the typed path). Long-prompt cold/warm answers coherent and
  consistent (paraphrase, not exact marker — instruction softness, noted).
- `DSV4-EFFORT-NONE-DEGENERATION`: REOPENED (P1, quality). reasoning_effort
  none correctly yields reasoning_len 0 (20260507 encoder contract holds)
  but the visible answer degenerates into a repetition loop live:
  "No-go-go-go-all-all-all-...". Same class as the 20260507 open
  quality caveat and #103 verb-loop. Needs direct-rail sampling/template
  investigation on current source (fresh-context fix e4b2f13f3 lineage).
- Remaining for DSV4 row: UI fresh-chat turn, long-Responses fragment
  retest, restart-restore axis.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 03:2x - DSV4 row: UI turn + long-Responses retest PASS

- DSV4 UI fresh-chat turn (Responses rail, wake-from-soft-sleep): exact
  visible `DSV4-UI-T1-OK`, reasoning 99 chars separated, TTFT 9.0s incl
  wake, persisted. Long-Responses fragment row (was matrix PARTIAL/FAIL,
  111-char fragment after forced </think>): RETESTED-PASS on current
  source — streamed /v1/responses long prompt + high effort produced 434
  visible chars in 108 content deltas ending exactly DSV4-LONGRESP-DONE,
  963 reasoning chars in 271 deltas, response.completed clean.
- DSV4 row remaining: effort-none degeneration P1 (open), restart-restore/
  L2 axis for the native composite store (warm detail showed paged+dsv4
  without +disk — verify dsv4 L2 write/restore in a dedicated pass).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 03:4x - Eric directive: per-model q4 TQ confirmation + deep paged-eviction/L2 checks (TO-DO rows)

- `MATRIX-Q4-TQ-POLICY-CONFIRMATION`: TO-DO (Eric directive). For Step 3.7,
  Gemma, Hy3, Laguna, Nemotron, MiniMax 2.7: confirm each gets the PROPER
  q4 TurboQuant storage policy on compatible attention KV (Bonsai is the q8
  exception, already verified; DSV4/M3/openPangu native typed caches stay
  TQ-free — verified for DSV4). Check /v1/cache/stats turboquant_kv_cache
  storage bits + auto_policy per model during each row.
- `MATRIX-PAGED-EVICTION-L2-DEEP-CHECK`: TO-DO (Eric directive). For each
  matrix model: deep-check paged cache EVICTION behavior (fill past the L1
  RAM ceiling/cache-memory-percent, verify free-block eviction with disk-L2
  write-through FIRST, evictions counter increments, no corruption after
  eviction) and L2 DISK FALLBACK (evicted-from-memory prefix must restore
  from the OLD stored L2 disk blocks: disk_hits increment, cache_detail
  gains +disk, output stays coherent). Also verify stale/older L2 stores
  from prior sessions are used when RAM cache is cold (proven once on
  Bonsai cold-after-restart paged+ssm+disk — repeat per architecture).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 04:0x - Hy3+MTP row: q4 TQ + MTP d1 + cold-L2/warm-paged VERIFIED-LIVE

- Hy3-JANG_2K-MTP (session 08921892, port 8010, 73.7GB): autodetect correct
  (hy3, hunyuan/qwen3, VLM false). MTP depth-1 ACTIVE (config
  num_nextn_predict_layers=1, jang_mtp_layers=1, index tensors present).
  Q4 TQ POLICY CONFIRMED (MATRIX-Q4-TQ-POLICY row, Hy3 ✓):
  kv_cache_quantization {enabled:true, mode:turboquant-storage, bits:4,
  turboquant-q4} — truthful summary working on a third architecture.
  API cold: exact "Busan HY3-cold-DONE", reasoning 1239 separated, detail
  `paged+disk+tq-native` (10 tokens restored from OLD disk L2 —
  MATRIX-PAGED-EVICTION-L2 partial evidence: cross-session L2 reuse on
  cold). Warm: exact "Busan HY3-warm-DONE", 640/695 cached,
  `paged+tq-native`, coherent + self-consistent (MoE MTP gate met; byte
  inequality expected and observed only in reasoning length).
- Hy3 row remaining: UI fresh-chat turn; full eviction-past-L1-ceiling
  deep check (fill beyond cache-memory-percent, verify write-through-first
  eviction + restore-from-evicted-L2).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 04:2x - Hy3 UI turn PASS; row green except eviction deep-check

- Hy3 UI fresh-chat turn: exact `HY3-UI-T1-OK`, reasoning 276 chars in rail,
  4096/4163 cached paged hit, 24.9 t/s, persisted. Hy3 row now VERIFIED-LIVE
  for: autodetect, q4 TQ policy, MTP d1, API cold(disk-L2)+warm(paged)
  coherence, reasoning separation, non-stream cached usage, UI turn.
  Remaining: eviction-past-ceiling deep check (MATRIX-PAGED-EVICTION-L2 row,
  shared pass across models).
- Next: Gemma row — investigate session e4a79e4c (gemma-4-12B, port 8000,
  status=error) first; prefer -it- 26B variant per directive; SWA per-layer
  TQ expectations: mixed_attention Rotating+KVCache, NO paged+ssm telemetry,
  q4 TQ confirmation.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 04:4x - Gemma row: q4 TQ + mixed-SWA cache VERIFIED-LIVE (API)

- Gemma 4-12B-it-qat-JANG_4M (external drive session 4a161eb9, port 8009):
  autodetect correct (gemma4 family/parsers, VLM true, JIT correctly
  ignored). Q4 TQ CONFIRMED (MATRIX-Q4-TQ row, Gemma ✓): turboquant-q4 bits
  4/4, auto_policy mixed_* (SWA per-layer). API cold exact
  "Busan GEMMA-cold-DONE" + 594-char reasoning; warm exact with 768/806
  cached, detail `paged+mixed_swa` — correct mixed-SWA telemetry, NO
  paged+ssm (contract held). Non-stream cached usage working (4th arch).
- Note: errored session e4a79e4c (12B, LOCAL .mlxstudio path) — path exists
  on disk; error cause not investigated (external variant preferred per
  directive); minor follow-up.
- Gemma remaining: UI turn; eviction/L2 deep-check (shared pass).
- Matrix q4-TQ progress: Hy3 ✓ Gemma ✓; remaining Step/Laguna/Nemotron/MM2.7.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 05:0x - Gemma UI turn PASS; row green except shared eviction pass

- Gemma UI fresh-chat turn: exact `GEMMA-UI-T1-OK`, reasoning 324 chars in
  rail, 45 t/s, persisted. Gemma row VERIFIED-LIVE: autodetect, q4 TQ
  (mixed-SWA policy), cold/warm with paged+mixed_swa, reasoning separation,
  non-stream cached usage, UI turn. Remaining: shared eviction/L2 deep pass.
- Matrix progress: Qwen3.6 ✓ Bonsai ✓ DSV4 (effort-none P1 open) ✓ Hy3 ✓
  Gemma ✓; next Laguna, Step 3.7, Nemotron, MM2.7, M3.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 05:2x - Laguna row PARTIAL: q4 TQ + cold pass; two open findings

- Laguna-M.1-JANG_2L (session 01bf750a, port 8015): autodetect correct
  (laguna, glm47/qwen3). Q4 TQ CONFIRMED (turboquant-q4 bits 4/4). Cold
  exact "Busan LAGUNA-cold-DONE" + reasoning separated. Warm cached 832,
  detail `paged+tq-native`, non-stream usage working (5th arch).
- `LAGUNA-DEFAULT-REASONING-CHECK`: OPEN. Default turn (no reasoning
  params) produced 877-char reasoning_content with exact visible answer.
  July-11 contract said template-omitted-default-OFF; M.1 may legitimately
  default thinking ON in its template — ground against the bundle
  chat_template before classifying. Visible content non-empty (the old
  empty-output failure is NOT present).
- `LAGUNA-WARM-TRAILING-SPILL`: OPEN (P2). Warm turn visible content ran
  past the answer marker: "Busan\n\nLAGUNA-warm-DONE\n\nThe problem
  involves a se..." — analysis-like prose after the marker ONLY on the
  warm/cached turn (cold clean). Suspect glm47 parser state on cache-restore
  or model rambling; needs streamed-delta capture on a warm turn to decide.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 05:4x - Laguna verdicts: default-reasoning = engine override vs template-OFF; spill = model ramble

- `LAGUNA-WARM-TRAILING-SPILL`: CLOSED-NOT-A-PARSER-BUG. Streamed warm turn:
  visible exactly "The favorite harbor is Busan." (29 chars), reasoning 1159
  chars fully separated, no spill. The earlier non-stream trailing prose was
  model ramble after the answer (reasoning was separately captured there
  too). glm47 parser clean on the stream rail.
- `LAGUNA-DEFAULT-REASONING-CHECK`: RECLASSIFIED as engine default-injection
  finding (P2). GROUND TRUTH: M.1 chat_template generation prompt emits
  `</think>` (thinking OFF) when enable_thinking is unset/false — the
  template explicitly DEFAULTS OFF. Yet default API turns produce ~900-1150
  chars reasoning => the engine/registry passes enable_thinking=True by
  default for family laguna, overriding the template-owned default
  (July-11 contract lineage: model-owned defaults must win). Action: check
  model_config_registry think_in_template/default-thinking for laguna and
  align to template default; regression + live re-proof.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 06:0x - CORRECTION: Laguna default-reasoning is INTENTIONAL (Eric directive)

- `LAGUNA-DEFAULT-REASONING-CHECK`: CLOSED-INTENTIONAL. model_configs.py
  laguna entry documents the design explicitly: the template GATES think on
  enable_thinking (default OFF in template), think_in_template=False, and
  per ERIC DIRECTIVE all reasoning-capable families default reasoning ON via
  architecture_hints.default_enable_thinking=True (Auto -> ON, template
  renders <think>). Observed default-turn reasoning is the intended
  behavior; visible content exact; template gating verified live both ways
  earlier (default ON produced reasoning; the template-side </think> path is
  the explicit-off rail). My 05:4x reclassification was wrong — reverted.
- Laguna row now GREEN except UI turn + shared eviction pass: autodetect,
  q4 TQ, cold/warm paged+tq-native, non-stream cached usage, glm47 stream
  separation (spill disproven), default-reasoning intentional.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 06:2x - Laguna UI turn PASS; row GREEN. Matrix 6/9.

- Laguna UI fresh-chat turn: exact `LAGUNA-UI-T1-OK`, reasoning 277 in rail,
  4096/4158 cached, persisted. Laguna row GREEN (except shared eviction
  pass). Matrix: Qwen3.6 ✓ Bonsai ✓ DSV4 ✓(effort-none P1 open) Hy3 ✓
  Gemma ✓ Laguna ✓; remaining Step 3.7, Nemotron, MM2.7, M3.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 06:3x - Eric directive: test BOTH JANGTQ and JANG-affine variants per family

- `MATRIX-JANGTQ-VS-AFFINE-VARIANTS`: TO-DO (Eric directive). JANGTQ
  (mxtq-packed TurboQuant weights) and plain JANG affine are DIFFERENT
  loader/runtime paths — where both exist locally, test both per family:
  Step 3.7 (JANGTQ_K sess 8022 AND JANG_K sess 8023 — do both), Qwen3.6
  (35B JANGTQ-CRACK vs 27B MXFP8/JANG_4M), Bonsai (1bit-JANG affine done;
  JANGTQ variant if present), Nemotron (JANGTQ/JANGTQ4 vs MXFP4), MiniMax
  2.7 (JANG_K), Hy3 (JANG_2K-MTP done — check for JANGTQ variant), Laguna
  (JANG_2L done — M.1/XS JANGTQ variants if present). Each variant: load,
  autodetect, TQ policy truth, cold/warm, coherence.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 06:5x - Step 3.7 JANG_K: q4 TQ + cold pass; WARM reasoning-only EMPTY (P1)

- Step-3.7-Flash-JANG_K (session 3ec7c78c, port 8023): autodetect correct
  (step-3.7-flash, step3p5/qwen3, VLM, JIT ignored). Q4 TQ CONFIRMED
  (turboquant-q4 4/4). Cold: exact "Busan STEP-cold-DONE", reasoning 734
  separated.
- `STEP37-WARM-REASONING-ONLY-EMPTY`: OPEN (P1 — Eric emission class).
  WARM identical non-stream turn: content EMPTY, reasoning 1557, cached 704
  detail paged+mixed_swa. Cold clean / warm broken => cache-restore
  interaction with the step template/answer-pass (no instruct rail —
  supportsInstructMode false limits recovery; prior lineage: step3p7
  non-stream reasoning-preamble leak + long-reasoner floor). Repro is
  deterministic-ish: long prompt cold then identical warm on 8023. Next:
  streamed warm capture + answer-pass trace for step family on warm turns.
- Matrix note: Step detail is paged+mixed_swa (SWA arch) — consistent.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 07:1x - STEP37 P1 SHARPENED: non-stream rail runaway reasoning; stream rail clean

- `STEP37-WARM-REASONING-ONLY-EMPTY` renamed scope ->
  `STEP37-NONSTREAM-RUNAWAY-REASONING` (P1). Evidence on 8023, same warm
  long prompt: STREAM rail = reasoning 439 chars, closes think, visible
  "\nBusan STEP-warm2-DONE", finish stop. NON-STREAM rail = reasoning 1557
  (cap 400) then 5706 (cap 1500), NEVER closes think, finish length, content
  EMPTY — twice. Deterministic stream-vs-non-stream divergence on the same
  server => the non-stream path renders/parametrizes differently for the
  step family (suspects: answer-pass/direct-rail template kwargs, stop
  tokens, enable_thinking/budget plumbing on non-stream; lineage: step3p7
  non-stream reasoning-preamble leak + no instruct rail). NEXT: diff the
  exact rendered prompt + sampling params between rails in server.py for
  step-3.7; fix; pytest; live re-proof non-stream warm visible answer.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 07:3x - Step UI turn PASS (stream rail); rail-diff Codex dispatched

- Step 3.7 UI fresh-chat turn: exact `STEP-UI-T1-OK`, reasoning 164 in rail
  — UI/stream rail clean, reinforcing STEP37-NONSTREAM-RUNAWAY-REASONING as
  non-stream-rail-specific. Box Codex gpt-5.6-sol dispatched on the source
  diff (report /tmp/step-rail-fix-report.md, uncommitted changes).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 07:5x - STOP-DESTROYS-SESSION-LOG-BUFFER fixed (active on next app relaunch)

- `STOP-DESTROYS-SESSION-LOG-BUFFER`: FIXED (commit above), tests
  session-log-retention 3/3 + panel 2299 green + typecheck. Stop retains
  the buffer with a marker; start resets; delete still drops. NOTE: main-
  process change — active after the NEXT Electron relaunch (deferred to the
  next natural relaunch point to avoid churn during the Step rail work).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 08:1x - CORRECTION: Step37 empty-answer = stochastic long-reasoner budget floor, NOT rail divergence

- `STEP37-NONSTREAM-RUNAWAY-REASONING`: RECLASSIFIED ->
  `STEP37-LONG-REASONER-BUDGET-FLOOR` (known lineage: 20260712 gate "verify
  budget before blaming parser"). Expanded sample (8 runs, cap 1500):
  stream 6/6 pass with reasoning 164-2597 chars; non-stream 1/4 pass (pass
  at 5106 chars reasoning!), fails ONLY when reasoning ~exceeds the cap
  (finish=length, empty content). Reasoning length is stochastic; the
  rails are symmetric — earlier 2-sample "deterministic divergence" claim
  RETRACTED. Engine behavior is explainable: explicit small max_tokens +
  no-instruct-rail family (supportsInstructMode false) = no recovery when
  budget dies mid-think; UI already applies reasoning-aware 4x fallback
  when the cap is unset. IMPROVEMENT row (P3): consider a server-side
  reasoning-aware floor/warning for explicit small caps on long-reasoner
  families. Codex rail-diff run died (compaction "Request blocked", 260k
  tokens, no diffs) — takeover produced this verdict instead.
- Step JANG_K row now GREEN (autodetect, q4 TQ, cold exact, warm paged+
  mixed_swa with adequate budget, stream 6/6, UI turn) with the P3
  improvement note. Next: JANGTQ_K variant per MATRIX-JANGTQ-VS-AFFINE.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 08:4x - Step JANGTQ_K variant GREEN (MATRIX-JANGTQ-VS-AFFINE: Step both variants done)

- Step-3.7-Flash-JANGTQ_K (session e04ccb1d, port 8022, mxtq loader path):
  autodetect identical to affine variant (step-3.7-flash, step3p5/qwen3,
  VLM, JIT ignored); q4 TQ confirmed. Cold exact "Busan STEPTQ-cold-DONE"
  (reasoning 1437, stop); warm exact with reasoning 7899 CLOSED (finish
  stop — budget-floor thesis further confirmed: adequate cap lets even
  ~2000-token reasoning close), cached 704 `paged+mixed_swa`. Step family:
  BOTH variants green. Matrix 7/9 + variant coverage advancing.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 09:0x - Nemotron JANGTQ row GREEN. Matrix 8/9.

- Nemotron-Omni-Nano-JANGTQ-CRACK (session ccb65b5a, port 8024): autodetect
  correct (nemotron-h, nemotron/deepseek_r1, VLM false, JIT correctly off
  for TQ). Q4 TQ CONFIRMED. Cold exact "Busan NEMO-cold-DONE" (739
  reasoning, stop); warm exact, 768 cached, `paged+ssm+tq-native` — SSM
  companion working on the Nemotron-H hybrid path. Media/Omni rows remain
  TODO per matrix (text-only row here). MXFP4 variant has NO session yet —
  create + test in the variant pass.
- Matrix q4-TQ confirmations: Hy3 ✓ Gemma ✓ Laguna ✓ Step(x2) ✓ Nemotron ✓;
  remaining MM2.7. Matrix rows: 8/9 core (MM2.7 + M3 remain).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 09:3x - MM2.7 row: q4 TQ complete (6/6 families); PAGED-DEFAULT MISSING on session (P2)

- MiniMax-M2.7-Small-JANGTQ (session f73768eb, port 8005): autodetect
  correct (minimax family, minimax/minimax_m2 parsers, NO VL — contract
  held). Q4 TQ CONFIRMED — MATRIX-Q4-TQ-POLICY-CONFIRMATION row COMPLETE:
  Hy3/Gemma/Laguna/Step(x2)/Nemotron/MM2.7 all turboquant-q4, Bonsai q8
  exception, DSV4/M3-class native excluded. Cold+warm exact answers,
  reasoning separated, coherent (#95 lineage holds).
- `MM27-PAGED-DEFAULT-MISSING`: OPEN (P2, cache-defaults wiring). Session
  argv contains NO --use-paged-cache/--enable-prefix-cache/
  --enable-block-disk-cache; scheduler warm miss (hits 0, misses 2) on an
  identical ~750-token prompt. Per paged-default-ON directive (excludes
  only M3/openpangu_v2/gemma4), MM2.7 should get paged+disk by default.
  Suspect stale saved session config predating the cache-stack migration
  (check cacheStackStartupDefaultsVersion migration path in sessions.ts for
  this session) — CACHE-DEFAULTS-UI-WIRING-MATRIX row to verify + fix.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 09:5x - MM27 paged-default root-caused: migration best-effort detection failure sticks forever

- `MM27-PAGED-DEFAULT-MISSING`: ROOT-CAUSED (sessions.ts:741+). The v8
  paged-default-ON migration resolves per-family paged capability
  best-effort at migration time; on detection failure it leaves the generic
  branch at paged-OFF (code comment: "detection best-effort; leave
  undefined -> paged-off"), then v9/v10 stamp
  cacheStackStartupDefaultsVersion=10 so it NEVER retries. Session f73768eb
  config: usePagedCache=false, enablePrefixCache=true, version=10.
  FIX DIRECTION: at session start, when usePagedCache=false with no
  explicit user opt-out marker and the family is paged-capable (not
  M3/openpangu_v2/gemma4), re-resolve capability (bundle is reachable at
  start) and flip to the directive default; needs a user-explicit-off
  marker to avoid overriding real choices. Also open question: prefix
  cache enabled yet warm identical prompt MISSED twice on the minimax LLM
  path (hits 0/misses 2) — verify prefix-hit keying for minimax without
  paged in the same fix pass.
- Still queued: M3 row (a563f316), M3 JANG_2L error readout, eviction/L2
  deep pass, DSV4 effort-none P1, Nemotron MXFP4 variant, UI turns
  Step/Nemotron/MM2.7/M3, settings parity.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 10:1x - CORRECTION: MM27 paged-off is branch policy (v7); question for Eric

- `MM27-PAGED-DEFAULT-MISSING`: RECLASSIFIED — NOT a wiring bug on this
  branch. sessions.ts migration generic branch documents: "Generic /
  Gemma-SWA / MoE / hybrid: paged cache OFF by default (Eric directive
  2026-06-30 v7, supersedes v6 ON)". MM2.7 (generic MoE) paged-off +
  version 10 is CONSISTENT with reconcile/1.5.68 policy. The 2026-07-12
  paged-default-ON directive (which reversed v7) lives in the v1.6.x line.
  ALSO withdrawing the "warm miss" sub-claim: I read scheduler_cache (the
  PAGED stats — rightly idle with paged off); the non-paged prefix-cache
  stats section was not checked. QUESTION FOR ERIC: should reconcile/1.5.68
  adopt the 07-12 paged-default-ON policy (excl. M3/openpangu_v2/gemma4)?
  If yes, the start-time re-resolution fix design from 23aabe891 applies;
  holding implementation until confirmed.
- Follow-up kept: verify MM2.7 non-paged prefix-cache hit behavior by
  reading the correct stats section on the next MM2.7 pass.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 10:3x - M3 sessions triage; fresh Coder-Small session created

- M3 session triage: e00c22bf (JANG_2L) error cause = model path missing
  (/Users/eric/.mlxstudio/... local, absent) — same class as gemma e4a79e4c
  and a563f316 Coder (also missing local). 78ad8607 is a remote:// session.
  REAL drive bundles: MiniMax-M3-Coder-Small + MiniMax-M3-REAP32-d3-Coder.
  Created fresh session 3c9ca4bf (Coder-Small, port 8017, external drive)
  via renderer API and started it — M3 row proceeds on it.
- Stale-local-path sessions (3x) flagged: consider a session-health sweep
  that marks sessions with missing model paths and offers re-point to the
  drive (LE10 lineage) — improvement row.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 10:5x - M3 row GREEN. CORE MATRIX 9/9 COMPLETE.

- MiniMax-M3-Coder-Small (fresh session 3c9ca4bf, port 8017, external
  drive): autodetect correct (minimax_m3 family/parsers), JIT absent from
  argv, kv_cache_quantization truthfully false (native typed MSA, no
  generic TQ). Cold exact "Busan M3-cold-DONE" (424 reasoning, stop); warm
  exact, 832 cached, detail `paged` (typed idx_keys path), reasoning 2076
  separated, finish stop.
- CORE MATRIX 9/9: Qwen3.6-MTP ✓ Bonsai ✓ DSV4 ✓ Hy3+MTP ✓ Gemma ✓
  Laguna ✓ Step(x2 variants) ✓ Nemotron-JANGTQ ✓ MM2.7 ✓ M3 ✓ — all with
  live cold/warm cache-detail evidence, reasoning separation, exact-answer
  turns; q4-TQ row complete across six families + q8 Bonsai + native
  exclusions verified.
- REMAINING before release: eviction/L2 deep pass, DSV4-EFFORT-NONE P1,
  Nemotron MXFP4 + remaining JANGTQ-vs-affine variants, UI turns for
  Nemotron/MM2.7/M3, settings-parity sweep, MM2.7 non-paged prefix
  verification, media/VL rows (Omni/Gemma/Qwen images), paged-policy
  question for Eric, bundle-python + packaging/signing/notarization gates.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 11:2x - Eviction probe round 1: capacity healthy, eviction NOT yet exercised (PARTIAL)

- `MATRIX-PAGED-EVICTION-L2-DEEP-CHECK`: PARTIAL round 1 (Bonsai 8030,
  /tmp/evict-probe.json). 16 distinct ~2200-token prompts: hits 0->17,
  oldest prefix re-queried with 2207/2208 cached `paged+ssm` and exact
  coherent answer — multi-prefix L1 retention + restore healthy. BUT
  evictions stayed 0: total ~35k tokens < 64k block capacity (1000x64),
  so the eviction axis was NOT exercised; also my probe read disk_writes
  from the wrong health field (null) — fix the field path. ESCALATION:
  round 2 with ~35 prompts (>64k tokens) to force eviction, then re-query
  an evicted prefix expecting disk_hits increment + `+disk` detail.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 11:4x - MATRIX-PAGED-EVICTION-L2-DEEP-CHECK CLOSED-BY-LIVE-PROOF

- Round 2 (Bonsai 8030, 35 distinct ~2200-token prompts, evidence
  bonsai-evict-probe.json): evictions 0 -> 107 -> 142, first eviction at
  fill #30 (~66k tokens — exactly past the 64k block capacity). Re-query of
  the FIRST (evicted) prefix: cached 2207/2208, detail `paged+ssm+disk`,
  disk_hits 0 -> 35, exact coherent answer EVICT-BASE-OK. Eric-requested
  end-to-end behavior PROVEN: fill past L1 ceiling -> eviction (write-
  through to L2 first) -> evicted prefix restores from OLD disk-L2 blocks
  with SSM companion intact -> coherent continuation. Round-1 addendum:
  16-prefix concurrent retention also healthy. Residual nit: probe read
  disk_writes from wrong health path (null) — counter exists in
  block_disk_cache section (verified earlier: disk_writes 15).
- `RELEASE`: remains `PARTIAL_NO_RELEASE` (remaining: DSV4 effort-none P1,
  UI turns Nemotron/MM2.7/M3, MXFP4 + JANGTQ variants, MM2.7 prefix
  verification, settings parity, media rows, packaging gates).

## 2026-07-18 12:0x - DSV4 effort-none root-caused: untuned generation_config stub (artifact-level)

- `DSV4-EFFORT-NONE-DEGENERATION`: ROOT-CAUSED, reclassified artifact-config.
  Bundle generation_config.json is a transformers auto-stub
  (_from_model_config:true) declaring temperature 1.0 + top_p 1.0, no
  repetition penalty; jang_config carries no sampling. Engine policy honors
  bundle-declared values (does not synthesize) => effort-none direct rail
  samples at temp 1.0/top_p 1.0 raw and degenerates ("No-go-go-go...");
  reasoning-on masks it (long CoT stabilizes the final answer). The
  20260507 exact-"35" pass was a near-single-token answer (low exposure).
  Artifact-level evidence per the trust rule: the stub marker itself.
  RECOMMENDATION (Eric): regenerate/patch the DSV4-Flash-JANG-CRACK
  generation_config with tuned sampling (e.g. temp~0.6/top_p 0.95/rep-pen)
  — jang-tools domain; engine change NOT indicated. Workaround: per-chat
  sampling overrides. Engine P1 CLOSED-NOT-ENGINE.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 12:3x - MM2.7 prefix VERIFIED; new P2: chat-session modelpath identity mismatch

- MM2.7 non-paged prefix: VERIFIED-LIVE — warm identical turn hit
  (scheduler_cache hits 1, tokens_saved 49, memory-aware entry resident
  23.7MB) + legacy disk L2 storing (3 entries, 1563 tokens,
  tq_native_enabled). Earlier miss concern fully resolved; MM2.7 row green
  except UI turn (blocked by the new P2 below).
- `CHAT-SESSION-MODELPATH-IDENTITY-MISMATCH`: OPEN (P2, UI truth). Live
  three-way split: session record model_path=/Users/eric/models/JANGQ-AI/
  MiniMax-M2.7-Small-JANGTQ (local), running argv=/Volumes/EricsLLMDrive/
  jangq-ai/... (LE10 launch re-resolution), chat binding=drive path (header
  dropdown). Renderer matches chat.model_path to sessions[].model_path by
  RAW STRING -> no match -> "Model is not running" banner + disabled
  composer while the session server is healthy and DB status=running.
  Renderer reload does NOT fix (rebuilds from same mismatched paths).
  FIX: use the LE10 modelIdentity/sessionMatchesModelPath helper in the
  renderer active-session lookup (App.tsx:82 / SessionView chat binding)
  AND/OR persist the re-resolved path back to the session record at launch.
  Repro: any session whose stored path differs from launch-resolved path.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 12:5x - CHAT-SESSION-MODELPATH-IDENTITY-MISMATCH CLOSED-BY-LIVE-PROOF

- Refined root cause: TWO session rows for one model (path-prefix twins:
  ~/models symlink vs /Volumes real dir — create dedupes by raw path); the
  chat stayed pinned to the STOPPED twin while the usable one served.
  Fix (commit above): App.tsx active-session resolution falls back from a
  non-usable pinned session to a usable same-identity session
  (sessionMatchesModelPath). Live: MM2.7 chat composer enabled, banner
  gone, UI turn exact `MM27-UI-T1-OK` (29 tokens, wake from standby).
  Panel 2301 tests + typecheck green. Relaunch also ACTIVATED the
  stop-log-retention fix (0fbddfce8) — postmortem buffers now live.
  Follow-up (P3): dedupe session creation by model identity to stop
  spawning path-prefix twins in the first place.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 13:1x - Nemotron UI turn PASS; MXFP4 variant session created + starting

- Nemotron JANGTQ UI fresh-chat turn: exact `NEMO-UI-T1-OK`, reasoning 85
  in rail, 80.5 t/s. Nemotron JANGTQ row fully green.
- Nemotron-Omni-Nano-MXFP4-CRACK variant session created (d9c7b9c1, port
  8034, external drive) and starting — MATRIX-JANGTQ-VS-AFFINE Nemotron
  pair in progress. Remaining UI turn: M3 only.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 13:4x - Nemotron variant pair complete; M3 UI turn PASS; ALL matrix UI turns green

- Nemotron-Omni-Nano-MXFP4-CRACK (d9c7b9c1, port 8034): q4 TQ confirmed,
  cold exact "Busan NEMOMX-cold-DONE", warm exact with 768 cached
  `paged+ssm+tq-native` — MATRIX-JANGTQ-VS-AFFINE Nemotron pair COMPLETE
  (JANGTQ + MXFP4 both green).
- M3 UI fresh-chat turn: exact `M3-UI-T1-OK`, reasoning 109 in rail,
  streamed clean (F13 lineage clear), 128 cached. ALL matrix UI turns now
  green: Qwen3.6/Bonsai/DSV4/Hy3/Gemma/Laguna/Step/Nemotron/MM2.7/M3.
- Remaining before release: Qwen3.6-35B JANGTQ variant (optional), settings
  parity sweep, media/VL rows, session-dedupe P3, then HELD gates
  (bundle-python, packaging/signing/notarization — Eric permission) and
  Eric decisions (paged policy for this branch; DSV4 sampling regen).
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 14:0x - Settings-parity sweep: M3 PASS (UI settings == argv)

- M3 session settings vs running argv (8017), all matched: tool/reasoning
  parser dropdowns=minimax_m3/minimax_m3 == argv; Continuous Batching
  true==flag; Prefix Cache true==engine default-on; Paged KV true==
  --use-paged-cache (M3 typed paged path); Block Disk L2 true==
  --enable-block-disk-cache; Legacy Disk false==no flag; JIT false==no
  flag; Auto Tool Choice true==flag. Settings-parity row: M3 PASS (JIT
  parity closed earlier across Bonsai/Qwen; parser/cache parity now proven
  on M3). Preview element not located in drawer DOM this pass — preview
  parity remains covered by the earlier jit-findings (both sessions) only;
  note for a future sweep pass.
- Remaining: media/VL row, Qwen3.6-35B JANGTQ variant, session-dedupe P3,
  HELD gates + Eric decisions.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 14:2x - Gemma VL/media row: API image turn functional PASS

- Gemma 12B VL API image turn (8x8 solid-red data-URI PNG): image ingested
  (291 prompt tokens incl image), visible answer "Pink GEMMA-IMG-DONE."
  with marker, reasoning separated, no crash. Color named Pink for pure
  red — consistent with the known LE7 Gemma-VL-4bit quant color-fidelity
  trait (model, not engine). Media rail FUNCTIONAL on the API path.
  UI attach-flow turn deferred as follow-up (file-dialog automation);
  media-salted cache axis not exercised this pass (single image) — both
  noted for the media deep pass.
- `RELEASE`: remains `PARTIAL_NO_RELEASE`.

## 2026-07-18 14:5x - Qwen3.6-35B JANGTQ variant GREEN; variant matrix complete

- Qwen3.6-35B-A3B-JANGTQ-CRACK (session 100cb088, port 8029 — stored path
  valid, loaded fine): q4 TQ confirmed; cold exact "Busan-Q35-cold-DONE"
  (model chose no-think, visible fine); warm exact with reasoning 1643
  separated, 896 cached `paged+ssm`. MATRIX-JANGTQ-VS-AFFINE now covers:
  Step (JANG_K + JANGTQ_K), Nemotron (JANGTQ + MXFP4), Qwen3.6 (27B MXFP8
  + 35B JANGTQ), Bonsai (1bit affine), Hy3 (JANG_2K-MTP), Laguna (JANG_2L),
  MM2.7 (JANGTQ), Gemma (JANG_4M qat) — variant coverage substantially
  complete for locally-available bundles.
- LAST code item: session-dedupe-by-identity P3. All else HELD for Eric:
  bundle-python + packaging/signing/notarization; paged-policy decision
  (v7 OFF vs 07-12 ON on this branch); DSV4 sampling regen (jang-tools).
- `RELEASE`: remains `PARTIAL_NO_RELEASE` pending held gates.

## 2026-07-18 15:2x - Session dedupe CLOSED. CAMPAIGN TESTABLE SCOPE COMPLETE.

- Session-dedupe-by-identity: CLOSED-BY-LIVE-PROOF (commit above). Create
  with a path-variant of an existing identity reused the row (34 -> 34
  sessions). Panel 2302 tests + typecheck green.
- CAMPAIGN STATE (2026-07-17/18 overnight): 14 defects closed by live
  proof; core matrix 9/9 with UI turns; JANGTQ-vs-affine variant coverage
  across 8 families; q4-TQ policy row complete; eviction/L2 lifecycle
  proven (142 evictions, disk restore); settings parity (M3 + JIT rows);
  VL media rail functional; hosts+GH synced throughout.
- REMAINING (all HELD for Eric): (1) packaging chain — bundle-python.sh,
  build-release-dmgs, notarize, verify, publish (explicit permission
  required); (2) paged-default policy for reconcile/1.5.68 (v7 OFF
  currently, 07-12 ON directive exists in v1.6.x line); (3) DSV4
  generation_config sampling regen (jang-tools); (4) optional follow-ups:
  UI-attach media flow, media-salt axis, Omni audio rows, box codex
  compaction issue.
- `RELEASE`: `PARTIAL_NO_RELEASE` — blocked ONLY on the held gates above.

## 2026-07-18 15:0x - COMPLETION CLAIM RETRACTED; campaign resumed (Eric directive + Codex source audit)

- The 07-18 03:15 "campaign testable scope complete" claim is RETRACTED. HEAD
  937fd7639 closed ledger rows only; CURRENT-MATRIX.md retains release-critical
  PARTIAL/OPEN/BLOCKED rows (full suites, protocol parity non-stream +
  tool-continuation stability, gateway agent soak, media/Omni rows, locales/
  modals at min width, eager non-DSV4 routes, MiMo + remaining parser families,
  settings-parity remainder, stale-session UX, family PARTIAL sub-axes).
  Status remains PARTIAL_NO_RELEASE. Campaign resumed per Eric 07-18 directive
  (10 testable workstreams; packaging still held pending explicit go; publish
  requires separate explicit PUBLISH).
- FULL suites launched (not focused): pytest tests/ -k "not Async"
  (/tmp/full-pytest-20260718.log) + panel vitest run
  (/tmp/full-vitest-20260718.log).

## 2026-07-18 15:0x - DSV4-EFFORT-NONE 12:0x diagnosis CORRECTED: jang_config DOES declare tuned sampling

- The 07-18 12:0x claim "jang_config carries no sampling" is FALSE. The
  official bundle /Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK
  declares jang_config.json::chat.sampling_defaults = {temperature 0.6,
  top_p 0.95, repetition_penalty_chat 1.05, repetition_penalty_thinking 1.0,
  max_new_tokens 4096} (file mtime Jul 12 00:51 — predates the wrong entry;
  the earlier grep only scanned TOP-LEVEL keys and missed the nested block).
- Offline ground truth on current source (.venv, server module, _model_path set
  to the bundle): _resolve_temperature(None)=0.6, _resolve_top_p(None)=0.95,
  rep_pen chat 1.05 — the engine honors bundle-declared values; the
  generation_config 1.0/1.0 stub is correctly shadowed by the JANG chat stamp.
  Panel chat rail sends temperature/top_p ONLY when explicitly overridden.
- CONSEQUENCE: the effort-none degeneration cause is NOT established. The
  "regen the bundle" recommendation is WITHDRAWN. Next per Eric: controlled
  same-artifact A/B (jang-tools/reference vs vmlx, matched prompts+sampling)
  before any blame; no artifact mutation without permission.
- Item remains OPEN as DSV4-EFFORT-NONE-DEGENERATION (cause unknown).

## 2026-07-18 15:4x - FULL-SUITE gate: first complete run triaged (9 fails -> 3 classes)

- FULL pytest (5931 collected effective): 9 failed / 5922 passed / 96 skipped.
  FULL panel vitest: 2302/2305 GREEN (0 fail). Panel typecheck: PASS. Logs
  preserved: release_closeout/full-suite-20260718/.
- CLASS 1 (stale test anchors, behavior intact — FIXED): (a) engine_audit
  error-recovery window missed delete_block_table because 1657ed312 inserted
  hit-credit finalization before it (call intact at mllm_scheduler.py:3819;
  window widened + detach_request forbidden in-window); (b) engine_audit
  recompress anchor broke on the deliberate cache-owner arg change
  (_cache_model or language_model — anchor updated); (c) mistral decode-gate
  contract expected reasoning "mistral" but registry ground truth for BOTH
  Mistral-Medium-3.5 bundles is reasoning=None ([THINK]-in-content by design;
  502f656db reconcile was correct) — expectation fixed + None-aware flag check.
- CLASS 2 (env-only): public_app_issue_audit x2 + verify-bundled-python needed
  node/PATH; with PATH the audit still FAILS on issue 119: gemma26
  memory-stress + real-UI artifacts (agent-notes 20260531 proofs) do not exist
  ANYWHERE in git or on disk — pre-existing red on this branch; closure =
  fresh Gemma-26B live rows (folded into Gemma family work).
  verify-bundled-python exit 1 = REAL bundle drift (engine changed);
  panel/scripts/bundle-python.sh dev-sync launched (NOT packaging).
- CLASS 3 (missing regenerable artifact): release_regression_manifest freshness
  fails because untracked build/current-regression-suite-...-20260609.json is
  gone from every working tree; regenerating fresh via
  run_current_regression_suite.py (in flight; sub-suite
  noheavy_api_cache_contract rc=1 to triage).
- ALSO: test_tq_disk_cache + test_tq_paged_block_cache FAIL under full suite
  but PASS in isolation — order-dependent state pollution, OPEN to root-cause.

## 2026-07-18 21:1x - FULL Python suite green; no-heavy meta-audit remains OPEN

- The order-dependent TurboQuant failures were test-process pollution, not a
  live cache defect. `test_explicit_kv_quantization_disables_loader_turboquant`
  invokes the real CLI, which intentionally sets `VMLX_DISABLE_TQ_KV=1` for a
  server lifetime, but the test stops at `uvicorn.run` and remained in the same
  pytest process. An autouse fixture now restores every CLI-mutated TQ/SSM
  policy variable after each test. The entire polluting file followed by both
  formerly failing disk-cache tests passed 10/10.
- Two remaining failures were stale source-string contracts: the post-tool
  Qwen Auto partition now uses `_auto_thinking_partition_allowed` in both Chat
  and Responses, and the MiniMax-M3 visible-answer path uses the shared
  progressive streamer. Tests now pin helper behavior and endpoint delta
  wiring instead of removed inline/log literals.
- Bundled Python was rebuilt from current vMLX source plus clean JANG source
  `9081c924` and passed version, critical source/JANG hashes, relocatable
  shebangs, and all critical imports.
- Current-source full Python suite with Node in PATH: **5942 passed, 96 skipped,
  261 deselected** in 246.95s. Evidence:
  `docs/internal/release-gates/20260718_full_suite_test_isolation/`.
- The generated no-heavy regression orchestrator is still **OPEN**, not green.
  Its child pytest/vitest commands generally pass, but several proof runners
  retain stale exact test-name/count/source-marker expectations; MiMo local
  bundles are absent from their hard-coded paths; and packaged signing remains
  outside this post-release source/test checkpoint. These meta-audit rows are
  being reconciled separately and are not converted into runtime proof.

## 2026-07-18 19:0x - Step JANGTQ Auto-tool empty final recovered; cross-family Auto-stream gate remains open

- `STEP37-AUTO-INVALID-TOOL-EMPTY`: scoped current-source fix is
  `VERIFIED-LIVE` on the JANGTQ/MXTQ artifact, not inherited by affine JANG or
  base MLX MXFP routes. An ordinary no-tool Electron request with the full
  built-in catalog made Step emit malformed native XML; the schema validator
  dropped it and the old stream finalized reasoning-only/empty. The new
  Step-only intent-gated retry removes only tool schemas, preserves native
  qwen3 reasoning, and requires a real close marker before progressively
  exposing content. Explicit/required/named tool requests remain fail-closed.
- Live Electron PID 56622, port 8022: row 42 recovered to exact
  `3861 STEP-POSTREL-FIX-T2-DONE`; row 45 retained same-chat `CEDAR-720` and
  visibly streamed the four-line final character-by-character; row 48 emitted
  exactly one real `file_info(panel/package.json)`, two separate progressive
  reasoning rails, exact 5.2 KB post-tool content, and reused 512
  `paged+mixed_swa` tokens.
- Raw temperature-0 parity: Responses no-tools 144/9 reasoning/content
  deltas; Responses two-tools Auto 216/9; Chat two-tools Auto 255/10 plus stop
  and `[DONE]`. All three exact-finaled. One earlier default-sampling response
  inserted `WASHINGTON` into a strict marker despite structurally correct
  152/21 streaming; it remains a semantic reliability FAIL sample.
- Source tests: 352 passed / 3 deselected across server/tool format/prompt/
  parser suites; py_compile and diff-check pass. Evidence:
  `docs/internal/release-gates/20260718_step_auto_reasoning_tool_recovery/`.
- `AUTO-REASONING-CROSS-FAMILY-STREAM`: OPEN/PARTIAL. Registered reasoning
  parsers are deepseek_r1, gemma4, minimax_m2, minimax_m3, mistral,
  openai_gptoss, qwen3, and think_xml. Every locally configured family must
  still have a current-head representative prove Auto progressive reasoning,
  progressive content, clean terminal, no-tool Auto, required tool, and
  post-tool continuation in Electron plus raw Chat/Responses. Current-head
  representatives now include qwen3 (Step/Qwen), minimax_m2, deepseek_r1
  (DSV4), and minimax_m3; this campaign row remains open for the other
  configured parser families. No blanket family pass is inferred from one
  representative.
- Global release/campaign status remains `PARTIAL`; no new release action is
  authorized by this scoped fix.

## 2026-07-18 19:4x - Qwen 3.6 35B JANGTQ ordinary Auto output recovered; explicit-tool variability retained

- `Q35-JANGTQ-AUTO-PARTITION`: `VERIFIED-LIVE` for the ordinary no-tool Auto
  path on the JANGTQ/MXTQ artifact. An attached tool catalog no longer disables
  Qwen's bounded reasoning/output partition when the request has no explicit
  tool intent. Required, named, or explicit tool turns remain unpartitioned
  and fail closed. The policy covers stream/non-stream Chat and Responses.
- Real Electron `Save & Restart` replaced PID 61979 with 63899. Row 126 kept
  3,773 reasoning characters separate and exact-finaled three non-empty lines.
  Same-chat row 129 executed exactly one real
  `file_info({"path":"panel/package.json"})`, exact-finaled 5.2 KB, and
  restored 325 `paged+ssm+disk` tokens. Raw Responses emitted 237 reasoning
  deltas, 14 progressive content deltas, and one completed terminal. Raw Chat
  emitted 1,024/353 reasoning/content deltas for ordinary Auto, a valid
  explicit `file_info`, and then 152/17 deltas for the exact real-result
  continuation with no repeated call. Chat finish reasons and `[DONE]` were
  correct; the ordinary model answer added an explanation before the marker,
  retained as a strict-format miss.
- Bundle/runtime truth: `weight_format=mxtq`, `profile=JANGTQ2`, routed expert
  bits 2, `turboquant_codebook`; this is not affine JANG or base MLX MXFP.
  Cache truth is a separate q4 attention-KV TurboQuant storage axis with native
  SSM companion persistence.
- `Q35-JANGTQ-EXPLICIT-TOOL-RELIABILITY`: `PARTIAL-STOCHASTIC`. Electron row
  63 and one distinct raw prompt produced a native `file_info` candidate
  missing required `path`; schema validation rejected it without execution.
  Repeated same-chat row 69 reused prior output without calling again, and row
  72 spent 52,343 reasoning characters / 214.5 seconds before eventually
  calling and finalizing. Positive controls are 11/11 fresh Electron tool
  turns, 3/3 fresh Auto-to-tool pairs, the final-source pair, and 12/12 exact
  raw prompt repeats. This supports prompt/history/sampling-sensitive native
  emission rather than a deterministic parser/transport defect, but repeated
  tool soak remains open for coding-harness reliability.
- Focused affected tests: 101 passed / 105 deselected, plus py_compile. No
  guessed arguments, synthetic calls, forced thinking-off retry, sampler clamp,
  or prompt coercion was added. Evidence:
  `docs/internal/release-gates/20260718_qwen35_jangtq_auto_partition/`.

## 2026-07-18 20:4x - Chat tool delta id accumulation repaired; MiniMax-M2.7 current-source q4/L2 agent stream re-proven

- `CHAT-TOOL-DELTA-ID-ACCUMULATION`: `VERIFIED-LIVE` on current source. The
  early Chat START delta already introduced tool-call index 0's id, but the
  terminal function-data delta repeated the entire string. Standards-style SDK
  accumulation therefore produced `call_abccall_abc`. The final delta now
  omits id/type (and the repeated role) for index 0 while subsequent calls still
  introduce their own identifiers. The raw M2.7 stream reconstructed exactly
  one `call_369ebbcf`, correct `file_info(path)`, `tool_calls`, and one DONE.
- Raw Chat result continuation emitted 46 reasoning plus 14 progressive content
  deltas and exact-finaled. Responses emitted one valid function call, then 55
  reasoning plus 15 progressive content deltas and one completed terminal.
- Real Electron Save & Restart moved the model to PID 72865. Fresh row 159
  executed one real `file_info(panel/package.json)` call, kept reasoning
  separate, returned non-empty exact `MM27-UI-CURRENT-DONE SIZE=5.2 KB`, and
  stored no warning. The screenshot visibly identifies the JANGTQ artifact and
  current PID.
- Bundle truth remains `weight_format=mxtq`, `profile=JANGTQ2`; this is neither
  affine JANG nor base MLX MXFP. KV-prefix storage is a separate q4 TurboQuant
  axis. After process restart, four identical raw requests restored 838 tokens
  as `paged+disk+tq-native`; health recorded 22 native-TQ L2 hits, three writes,
  and no dequantization on the last reconstruction.
- Broad affected regression: 430 passed / 3 deselected. Evidence:
  `docs/internal/release-gates/20260718_minimax_m27_tq4_agent_stream/`.
- Retained boundary: prompt/tool-schema rendering changed the second Electron
  prompt enough to miss the full conversation prefix. Later same-chat and
  identical restart requests prove resident and disk reuse at matching
  boundaries; maximal reuse across every history/template transition is not
  claimed. Campaign/release status remains `PARTIAL`.

## 2026-07-18 21:4x - Cross-matrix proof counts and clean-checkout bootstrap repaired; canonical sweep remains OPEN

- Commit `db07a6fc1` centralizes ANSI-safe pytest/Vitest result parsing for
  all 15 affected cross-matrix runners. Each runner now hashes the shared
  parser, preventing a parser change from leaving an apparently current proof.
- The no-heavy orchestrator no longer crashes before its own objective-digest
  producer on a clean checkout. Provisional checkpoints truthfully report a
  pending digest; final status still requires the generated artifact.
- Stale gate anchors were reconciled to current source. In particular, the
  Native-MTP gate no longer asserts the removed policy that JANG_2K blocks MTP;
  it now pins the measured-stamp policy and has no missing markers.
- Focused verification: **222 passed, 1 skipped**. Canonical sweep focused
  sub-suite: **656 passed, 1 skipped, 232 deselected**. Full panel verification
  in the same source interval: **2311 passed, 3 skipped**, typecheck pass, and
  production build pass using clean JANG source `9081c924`.
- The canonical sweep is honestly `OPEN`, not a campaign pass. Current failed
  steps are missing MiMo local bundles, signing preflight, release manifest,
  and the release gate's open objective digest. The required current DSV4 live
  default-cache tool-loop artifact is also absent. Missing historical live
  Qwen speed, Gemma/Ling quality, broad real-Electron, and cross-family smoke
  artifacts remain open rather than being inferred from source tests.
- No model generation ran in this checkpoint. Evidence:
  `docs/internal/release-gates/20260718_cross_matrix_count_parser/`.

## 2026-07-18 22:4x - DSV4 native pool codec preserved across RAM/L2; deterministic agent stream re-proven

- `DSV4-V8-TYPED-POOL-CACHE`: `VERIFIED-LIVE` for deterministic native
  composite prefix reuse and the one-tool Responses/Electron loop. The
  cache-hit tail had explicitly disabled realization before allocator clear,
  while prompt snapshots and paged reconstruction silently rebuilt
  `PoolQuantizedV4Cache` as base `DeepseekV4Cache`. Those paths now preserve
  the pool codec and realize the uncached tail before clearing MLX allocator
  state. Schema `deepseek_v4_v8` invalidates v7 blocks captured after the old
  type downgrade.
- This artifact is affine JANG (`JANG_AFFINE`), not JANGTQ/MXTQ or base MLX
  MXFP. Its cache is the DSV4-native 43-layer SWA plus CSA/HCA compressed-pool
  composite. Health reports pool quant enabled and generic TurboQuant KV
  disabled; the disk store recorded zero TQ-native writes/hits.
- Real Electron rows 189/192 produced byte-identical reasoning/content,
  exactly one real `file_info(panel/package.json)`, and a 340-token
  `paged+dsv4` resident hit. After the v8 bump, rows 195/198 repeated an exact
  one-tool final across visible Stop/Start; row 198 restored 338 tokens as
  `paged+dsv4+disk`, with two disk hits and no warning.
- Current raw Responses cold/warm/skip-control runs normalized to the same
  schema-valid function call. Each emitted 78 separate reasoning deltas, two
  progressive argument deltas, and one completed terminal. The real tool
  result continuation emitted 15 progressive content deltas and exact
  `DSV4-RAW-V8-DONE SIZE=5.2 KB` before completion.
- Focused current-source validation: **813 passed, 1 skipped**. Evidence:
  `docs/internal/release-gates/20260718_dsv4_v8_typed_cache_electron/`.
- Retained boundary: this closes deterministic typed cache and streaming
  correctness, not the separate temperature-0.6 strict-format/long-quality
  reliability row. Overall campaign/release status remains `PARTIAL`.

## 2026-07-18 23:0x - MiniMax-M3 Auto reasoning/tool stream current-head closure

- `M3-AUTO-REASONING-AGENT-STREAM`: `VERIFIED-LIVE` for current-source text
  Auto reasoning and the real one-tool continuation through Electron,
  Responses, and Chat Completions. The real bundle declares
  `minimax_m3_vl`, `JANG_2L` affine mixed quantization, native sparse MSA
  cache, tools, thinking, and vision. It is not JANGTQ/MXTQ or base MXFP.
- The real Electron Start button replaced DSV4 with M3 PID 19963 under the
  one-model policy. Before any prompt, `/health` reported `model_loaded=true`,
  `last_request_time=null`, schema `minimax_m3_msa_v1`, dense KV layers 0-2,
  sparse MSA layers 3-59, and generic TurboQuant forced off. The fresh-chat UI
  visibly showed Auto reasoning, Responses wire, built-in tools enabled, and
  bundle defaults temperature 1.0/top-p 0.95; the controlled run then applied
  temperature 0 and max 512 without changing Auto.
- Electron row 201 stored 2,065 reasoning characters separately from 541
  visible characters. Timed screenshots show Waiting at 1s, 42 reasoning
  characters at 3s, 413 at 6s, then progressive visible content. Same-chat
  row 204 executed exactly one real `file_info(panel/package.json)`, received
  5.2 KB, and grew visible content from 178 to 394 characters before final;
  it retained 1,851 separate reasoning characters and no warning.
- Raw Responses no-tool Auto emitted 262 reasoning and 46 content deltas with
  one completed terminal. Required-tool emitted 19 reasoning and two argument
  deltas plus a schema-valid `file_info`; the real-result continuation emitted
  15 reasoning and 119 content deltas before completion. Raw Chat emitted
  304/51 reasoning/content deltas for no-tool Auto, 19 reasoning plus two tool
  deltas and `finish_reason=tool_calls` for the call, then 37/125 deltas and
  `finish_reason=stop` for continuation; every Chat stream ended with `[DONE]`.
- The model refused synthetic `M3-*` exact-output markers as prompt-injection
  patterns while still producing coherent non-empty answers and executing the
  requested real tool. This is retained as model policy/strict-format
  behavior, not misclassified as a parser, cache, UI paint, or API terminal
  failure. Evidence:
  `docs/internal/release-gates/20260718_m3_auto_agent_stream/`.
- M3 larger-video, digit OCR, terminal-delay, and REAP32 headroom rows remain
  `PARTIAL`; campaign/release status remains `PARTIAL`.

## 2026-07-18 23:1x - openPangu 3M native Auto/tool protocols current-head closure

- `OPENPANGU-AUTO-REASONING-AGENT-STREAM`: `VERIFIED-LIVE` for current-source
  text Auto reasoning and one-tool continuation through Electron, Responses,
  and Chat Completions. The real bundle is affine importance-quantized
  `JANG_3M` with no Hadamard rotation; it is not JANGTQ/MXTQ or base MXFP.
- Electron Start replaced M3 with openPangu PID 21745 and left exactly one
  model running. Before a prompt, health reported `model_loaded=true`,
  `last_request_time=null`, and the native `openpangu_v2_composite_v2` MLA +
  DSA indexer + rotating SWA + path-dependent convolution state. Generic TQ,
  generic paged blocks, and block L2 were off; exact typed prompt-disk L2 was
  on, matching the architecture and UI/argv policy.
- Bundle defaults appeared in the UI as temperature 1.0, top-p 0.8 and Auto
  reasoning with Responses/tools enabled. The controlled run changed only
  temperature to 0 and max tokens to 512. Electron row 207 stored 897
  reasoning and 406 visible characters separately, coherent and warning-free.
  Same-chat row 210 executed exactly one real
  `file_info(panel/package.json)`, received 5.2 KB, and returned the requested
  one-sentence visible answer with 468 reasoning characters and no warning.
- Raw Responses no-tool/tool/follow emitted 389/38 reasoning-content, 35
  reasoning plus two argument, and 124/151 reasoning-content deltas; every
  case completed. Raw Chat emitted 250/38, 35 plus two tool deltas, and
  185/145; call/follow finish reasons were `tool_calls`/`stop`, and each ended
  with `[DONE]`. No generic TQ/paged path was introduced.
- The bundle contains stored MTP hints but its name does not declare MTP and
  the tensor index has no MTP tensors. Health truthfully reports the current
  openPangu runtime drops those extra-layer heads and leaves MTP inactive; no
  depth was invented for this non-MTP-named artifact.
- Evidence:
  `docs/internal/release-gates/20260718_openpangu_auto_agent_stream/`.
- Long-context stress and broader protocol/cancellation soak remain
  `PARTIAL`; campaign/release status remains `PARTIAL`.

## 2026-07-19 06:2x - openPangu long-context snapshot admission guard

- `OPENPANGU-LONG-SNAPSHOT-ADMISSION`: `VERIFIED-LIVE` for the owning
  pre-copy guard. A 43,980-token Electron request produced the exact anchor
  answer, but the pre-fix path copied a 22,090.8 MB native composite that both
  configured cache backends then rejected. It took 186.35s to first token and
  peaked at 138,814.3 MB Metal.
- `SingleBatchGenerator` now receives the largest valid RAM/disk single-entry
  ceiling, estimates typed prompt state before deep copy, skips an inadmissible
  boundary, and reports estimate/limit/skip telemetry. The post-patch Electron
  replay returned the byte-identical answer with separate reasoning/content at
  103.20s TTFT, 426.2 prompt tok/s, and 115,551.8 MB peak. No RAM/L2 entry was
  falsely reported. Focused current-source coverage is 124/124.
- `OPENPANGU-LONG-RESPONSES-STREAM`: `VERIFIED-LIVE` for wire emission,
  `PARTIAL` for latency. Raw Responses emitted 256 timed reasoning deltas,
  then—because a 256-token cap was exhausted inside reasoning—23 timed content
  deltas from the bounded direct-answer pass, followed by output-text done and
  response completed. Exact content was
  `cedar-7319|quartz-4821|harbor-9652`.
- Retained limitation: a native boundary larger than every backend is not
  reusable, and a tight reasoning cap can therefore cause two full prefills.
  The advertised 524,288-token limit remains unproven. Evidence:
  `docs/internal/release-gates/20260718_openpangu_long_snapshot_guard/`.

## 2026-07-19 00:4x - Seeded TQ restore startup cost closed for text + MLLM schedulers

- `HY3-TQ-DECODER-FIRST-RESTORE-LATENCY`: `VERIFIED-LIVE` on commit
  `7472c1ad1`. The live 80-layer HY3 cache owns 80 distinct q4 codec seeds,
  while the storage decoder LRU retained only 32. The first disk hit also paid
  the first real packed-codec invocation. The text scheduler now retains 256
  codec pairs and materializes the bundle-derived q4 codecs on its pinned
  model worker before readiness. The same 36-token `paged+disk+tq-native`
  boundary moved from 9.688565s to 0.951356s reconstruction and Electron TTFT
  from 9.88s to 1.14s; an 82-token/two-block replay reconstructed in 1.044455s
  and exact-finaled. Evidence:
  `docs/internal/release-gates/20260719_hy3_tq_decoder_warmup/`.
- `MLLM-TQ-DECODER-FIRST-RESTORE-LATENCY`: `VERIFIED-LIVE` for current
  Bonsai source on commit `727da2e44`. The initial hook did not
  cover `--is-mllm`; `_start_mllm` now warms the MLLM scheduler's real patched
  language-model cache owner on the same load/step worker. Electron Logs show
  exactly 16 q8 TurboQuant slots warmed and 48 ArraysCache companion slots
  retained. Fresh-chat restart replay restored 74 tokens as
  `paged+ssm+disk` at 0.21s TTFT with eight q8 TQ-native block hits, one SSM
  companion hit, and no unsafe KV-without-SSM reuse. Same-chat Electron
  `file_info` executed exactly once and exact-finaled; raw Chat/Responses
  emitted separate progressive reasoning/content and correct terminal events.
  The stochastic no-tool UI row rambled past the requested terminal marker and
  remains a strict-format reliability failure. Evidence:
  `docs/internal/release-gates/20260719_bonsai_mllm_tq_warmup/`.
- `NONPAGED-PROMPT-DISK-L2-RESTORE`: `OPEN` (P1 cache-axis gate, Eric
  directive). With paged cache explicitly Off and prompt disk cache/L2 On, an
  exact prefix must still be indexed, found, restored, and reused across a
  process restart. This gate also requires a non-zero partial-prefix match:
  prove block-aligned partial reuse, prove only the unmatched tail is
  recomputed, and prove the same partial boundary restores from L2 after a
  process restart. Repeat the partial-prefix/block-reuse axis with paged cache
  On where the representative architecture supports paged blocks, so exact
  hits cannot mask a broken partial lookup or eviction/refault path. Prove a
  plain-KV representative and an
  architecture-specific typed-cache representative through Electron UI -> DB
  -> argv -> health plus raw API streaming. The result must report prompt
  `disk` reuse without fabricated paged/block hits when paged is Off, and must
  report the actual reused-token/block boundary rather than only a generic
  cache-hit label. Explicit Off must persist.
- `NONPAGED-PROMPT-DISK-L2-RESTORE / PLAIN-FULL-KV`: `VERIFIED-LIVE` on
  current source for MiniMax-M2.7-Small-JANGTQ. This is a text-only
  `minimax_m2` bundle whose weights are JANGTQ/MXTQ (`JANGTQ2`, routed experts
  2-bit and attention/shared/embed/head 8-bit); it is not JANG affine or base
  MLX MXFP. The UI was set to Prefix On, Paged Off, Block L2 Off, Prompt Disk
  On and stored-codec Auto. The saved preview and process argv contained
  `--no-paged-cache --enable-disk-cache` with the dedicated prompt-cache
  directory, and health selected q4 TurboQuant storage for its 62-layer plain
  KV cache. Only this model remained loaded after the UI Start swap.
- Electron PID 44250 wrote two durable q4 TQ-native prompt records. After UI
  Stop/Start to PID 45185, a longer same-chat prompt restored 2,235/2,310
  prompt tokens from the 2,236-token disk boundary, recomputed only the tail,
  recalled the requested fact exactly, kept reasoning separate, and completed
  with 0.88s TTFT. Logs report `matched 2236/2305`; health reports one prompt
  disk hit and one TQ-native hit with zero paged/block-disk tokens.
- Raw Responses independently wrote a 1,393-token boundary. After UI restart
  to PID 45913, a longer request restored 1,392/1,458 input tokens, emitted
  124 reasoning-summary deltas and 26 progressive content deltas, returned the
  exact retained value, and emitted output-text done plus response completed.
  After another UI restart to PID 46340, Chat Completions restored the same
  boundary for a different longer turn (1,392/1,460), emitted 58 reasoning and
  14 content deltas, returned the exact value, finished `stop`, and emitted
  `[DONE]`. Evidence:
  `docs/internal/release-gates/20260719_nonpaged_prompt_disk_partial/`.
- This closes only the plain full-KV, paged-Off prompt-L2 subrow. The
  architecture-specific typed-cache representative and the paged-On
  block-aligned partial/eviction/refault rows remain `OPEN`; therefore the
  parent gate and campaign/release status remain `PARTIAL`.
- Current focused cache selection initially exposed one stale constructor-
  bypass fixture: `test_scheduler_uses_minimax_m3_logits_sampler_for_msa_cache`
  used `object.__new__(Scheduler)` without declaring the no-RAM/no-disk cache
  state now read by snapshot admission. The fixture now explicitly sets both
  cache backends to `None`; the same selection reran `107 passed`.
- Focused current-source validation for the MLLM extension: 143 passed. The
  campaign/release status remains `PARTIAL`; no release action is authorized by
  these scoped rows.

## 2026-07-19 01:1x - openPangu typed non-paged prompt-disk partial axis

- `NONPAGED-PROMPT-DISK-L2-RESTORE / TYPED-OPENPANGU / THINKING-OFF`:
  `VERIFIED-LIVE` on source head `6e5653f56`. The real Electron session for
  `openPangu-2.0-Flash-JANG_3M` used Prefix On, Paged Off, Block L2 Off,
  Prompt Disk On, and the architecture-owned typed composite policy. The
  actual argv contained `--no-paged-cache --enable-disk-cache`; health kept
  TurboQuant KV, paged blocks, and block-disk tokens at zero. This artifact is
  JANG affine (`JANG_3M`, 3.83-bit average, asymmetric `mx.quantize`, no
  Hadamard rotation), not JANGTQ/MXTQ and not base MLX MXFP.
- The Thinking-Off Electron base wrote a 780-token exact typed N-1 prompt
  snapshot. After real Electron Stop/Start process replacement, the longer
  same-chat turn restored 779/840 prompt tokens from disk, recomputed the
  unmatched tail, recalled the requested fact exactly, and completed. The log
  records `Disk cache prefix hit: matched 780/840 prompt tokens`; health
  records `cache_detail=disk`, zero blocks, and no reconstruction or
  dequantization.
- Raw Responses after another Electron restart independently restored 779/838
  input tokens from the same prompt-disk boundary, emitted 14 progressive
  output-text deltas, exact-finaled, and emitted both output-text done and
  response completed. Raw Chat after another Electron restart restored
  779/840 tokens, streamed progressive content, exact-finaled, emitted
  `finish_reason=stop`, and emitted `[DONE]`.
- The final Chat store crossed the configured 10 GB prompt-disk limit and
  logged one entry eviction after the successful restore/store. This is
  evidence that size enforcement ran, but it is not yet an eviction/refault
  closure because the evicted key was not requested again.
- `OPENPANGU-AUTO-REASONING-PROMPT-DISK-PARTIAL`: `OPEN` (P1 correctness and
  latency). The Auto base emitted a separate 786-character reasoning rail,
  exact visible output, and a durable 1,432-token typed prompt record. After
  process restart the longer same-chat follow-up was coherent and exact but
  restored zero tokens; health recorded one disk miss followed by an
  independent 1,495-token store. Current source has persisted-reasoning replay,
  but the observed Auto follow-up did not share an admissible token prefix
  with the stored prompt-only boundary. Compare the rendered/tokenized base
  prompt, replayed reasoning item, injected/open `<think>` boundary, and N-1
  snapshot before changing policy. Do not use prompt coercion, discard hidden
  reasoning, or add a model-specific fake cache hit.
- Auto acceptance: fresh Electron Auto base with separate reasoning/content,
  process restart, longer same-chat Auto follow-up with non-zero typed disk
  partial reuse and unmatched-tail-only prefill, then equivalent raw Responses
  and Chat streams with no stale reasoning replay. Evidence:
  `docs/internal/release-gates/20260719_openpangu_typed_nonpaged_partial/`.
- The parent non-paged gate remains `PARTIAL`: the typed Thinking-Off subrow
  joins the plain full-KV pass, while openPangu Auto reasoning and the generic
  paged-On block-aligned partial/eviction/refault row remain open.

## 2026-07-19 08:32 - shared prompt-disk N-1 payload-prefix repair

- `PROMPT-DISK-NMINUS1-PAYLOAD-PREFIX-INDEX`: `VERIFIED-LIVE` on source
  commit `a96a44559`. Token-level analysis proved that the 1,432-token
  openPangu Auto base and 1,495-token replay shared exactly 1,431 tokens: the
  full typed N-1 payload. Only the non-owned generation sentinel differed
  (`<think>` token 148905 versus replayed `</think>` token 148906). The old
  disk index hashed only the full N-token key and could not find the reusable
  payload. This was a shared prompt-disk lookup bug, not a model, JANG affine,
  reasoning-parser, or quant artifact failure.
- SQLite prompt records now index `payload_prefix_hash=hash(tokens[:-1])`.
  Longest-prefix lookup accepts a different Nth token only when the complete
  N-1 payload hash is exact, loads the existing record by its stored full-key
  hash, and re-feeds the current boundary plus unmatched tail. Exact lookup is
  unchanged, earlier payload divergence is rejected, and legacy null-hash
  rows remain exact-only until safely backfilled or rewritten. The focused
  current-source selection passed 84/84 tests across standard, typed, and
  TQ-native disk paths.
- `OPENPANGU-AUTO-REASONING-PROMPT-DISK-PARTIAL`: `VERIFIED-LIVE` for the
  cache/output axes. The real Electron Auto base emitted separate reasoning
  and exact visible content. After UI Stop/Start, the longer same-chat turn
  restored 1,431/1,495 tokens from prompt disk, produced fresh non-identical
  reasoning, exact-finaled, and logged the N-1 boundary re-feed. Raw Responses
  after an independent UI restart restored 1,431/1,495, emitted 262 separate
  reasoning deltas and 15 progressive content deltas, exact-finaled, and
  completed. Raw Chat after another UI restart restored 1,431/1,497, emitted
  300 reasoning and 16 content deltas, exact-finaled, stopped, and emitted
  `[DONE]`.
- `CHAT-STREAM-INCLUDE-USAGE-PARITY`: `OPEN / FAIL` (P1 protocol parity).
  The same raw Chat stream exposed a shared server behavior: with
  `stream_options.include_usage=true`, 317 intermediate chunks each carried a
  non-null usage object instead of a single terminal usage chunk before
  `[DONE]`. Cache restore, reasoning/content separation, and terminal output
  passed, but Chat protocol parity must not be closed until source is changed
  and current live Chat plus Electron metrics prove a single final usage event
  without regressing progressive rendering.
- Evidence:
  `docs/internal/release-gates/20260719_prompt_disk_payload_prefix_index/`.
  The parent cache/release gate remains `PARTIAL`: generic paged RAM partial
  reuse plus forced eviction/block-disk refault/restart restore are still
  required on a compatible model, and the new Chat usage row remains open.

## 2026-07-19 01:49 - Chat terminal usage and finish ordering

- `CHAT-STREAM-INCLUDE-USAGE-PARITY`: `VERIFIED-LIVE` on source commit
  `5358842b2` for the shared Chat Completions serializer/finalizer. The live
  pre-fix openPangu stream carried 317 non-null, growing usage objects on
  intermediate reasoning/content chunks. The server now emits `usage:null`
  on ordinary chunks and one choices-empty total-usage chunk before `[DONE]`.
  This was a global Chat wire defect, not a model, parser, quant, or cache
  failure.
- The same live rerun exposed and closed a second ordering defect in the
  generic terminal-finish guard. When no prior finish chunk existed, it used
  to append synthetic `finish_reason=stop` at `[DONE]`, after a previously
  emitted usage-only tail. The guard now detects that tail and inserts
  `finish_reason=stop` before it. Current raw order was 388 null-usage ordinary
  chunks, finish at index 387, exactly one choices-empty usage total at index
  388, then `[DONE]`; no ordinary chunk omitted usage. Reasoning remained
  separate/progressive and visible content exact-finaled.
- The Electron chat override was grounded in SQLite as
  `wire_api=completions`, and its settings UI showed `/v1/chat/completions`.
  A retained screenshot sequence shows a no-tool Auto turn growing from 61 to
  1,370 reasoning characters before exact visible
  `CHAT-USAGE-UI3-DONE`; final metrics reported 414 output, 364 prompt, 290
  memory-cached, 25.3 tok/s, and 0.87s TTFT. This proves the UI still derives
  live progress from real deltas after per-chunk usage removal.
- The same Chat wire then executed exactly one `file_info` call with valid
  `{"path":"panel/package.json"}`, consumed the 5.2 KB tool result, and
  exact-finaled `CHAT-USAGE-TOOL-DONE SIZE=5.2 KB` with separate reasoning,
  one matching OAI call/result, coherent metrics, and no warning.
- Current validation: 666/666 Python stream/parser tests, 84/84 panel tests,
  and TypeScript typecheck passed. Evidence:
  `docs/internal/release-gates/20260719_chat_terminal_usage_parity/`.
- This closes the scoped shared Chat usage/order blocker, not the full
  protocol or model matrix. Other families, Responses/Anthropic/Ollama live
  rows, paged block eviction/refault, media, gateway, full suites/build, and
  release gates remain `PARTIAL` or `OPEN` until their own current-source live
  evidence is recorded.

## 2026-07-19 09:0x - explicit no-tool schema/prefix stability

- `ELECTRON-NO-TOOL-SCHEMA-PREFIX-STABILITY`: `VERIFIED-LIVE` on source
  commit `258cf16f9`. With built-in tools persistently enabled, the panel
  recognized an explicit current-turn “do not call tools” directive but still
  sent the full tool catalog plus `tool_choice=none`. On the next Responses
  turn, MiniMax tool-template fallback inserted that catalog at the prompt
  front; prompt size changed 297 to 1,185, the cached prefix no longer
  matched, and three L1 evictions occurred. This was a shared Electron request-
  builder defect, not a MiniMax, JANGTQ, reasoning-parser, or cache-codec
  failure.
- Both Responses and Chat builders now omit tool definitions entirely for a
  guarded explicit no-tool directive. Normal tool-enabled turns remain
  unchanged. Focused panel validation passed 24/24 and TypeScript typecheck.
- After a full Electron main-process relaunch, the current-source base logged
  `has_tools=false`, exact-finaled with separate reasoning, and stored a q4
  native-TQ paged boundary. The same-chat follow-up also omitted tools,
  restored 192 tokens as `paged+disk+tq-native`, recalled the exact private
  fact, exact-finaled, and stored no warning.
- The cache result is deliberately classified as same-process L2, not RAM:
  with block disk enabled the default frugal policy keeps the L1 chain index
  but releases the duplicate payload after q4-native disk write-through.
  RAM-resident reuse with Block L2 Off, then forced eviction/L2 refault,
  partial-block reuse, and process-restart restore remain the active parent
  gate.
- Evidence:
  `docs/internal/release-gates/20260719_no_tool_schema_prefix_stability/`.

## 2026-07-19 09:07 - no-tool directive variant and paged RAM partial reuse

- `ELECTRON-WITHOUT-TOOLS-DIRECTIVE`: `VERIFIED-LIVE` on source commit
  `69246de78`. The explicit phrase `Without tools` was not covered by the
  guarded current-turn parser. The first RAM acceptance attempt therefore
  changed from `has_tools=false` on the base to `has_tools=true` on the
  follow-up, expanded the prompt 182 to 1,061 tokens, missed the prefix, and
  evicted three blocks. This was a shared parser/request-shape defect, not a
  MiniMax M2.7, JANGTQ, TurboQuant, or paged-cache failure.
- The shared parser now recognizes directive-shaped `without tools` and
  `without using any tools` while negative tests protect quoted/explanatory
  text. Focused panel validation passed 20/20 and TypeScript typecheck.
- `M27-PAGED-RAM-PARTIAL`: `VERIFIED-LIVE`. After a full Electron relaunch,
  the model was started from the real UI with Block Disk L2 Off. Both current-
  source Responses requests logged `has_tools=false`. The follow-up restored
  178 tokens from three 64-token RAM-resident q4 native-TQ blocks as
  `paged+tq-native`, including the partial terminal block, exact-recalled
  `SFACT-11=N-15263`, preserved separate fresh reasoning, and stored no
  warning. Health recorded 192 RAM tokens, 13,066,128 resident L1 bytes, zero
  disk hits, and zero L2 tokens.
- Forced eviction/L2 refault, L2 partial-block reconstruction, server-process
  restart restore, and raw Responses/Chat streaming remain the active parent
  gate. Evidence is in
  `docs/internal/release-gates/20260719_no_tool_schema_prefix_stability/`.

## 2026-07-19 09:25 - M2.7 paged q4 partial L2/refault and truthful detail

- `M27-PAGED-Q4-L2-PARTIAL-REFAULT`: `VERIFIED-LIVE` on source commit
  `97a84fed5`. The real Electron UI applied a new empty block-L2 directory,
  64-token blocks, and a four-block ceiling. The cold base wrote a 178-token
  q4 native-TQ chain of 64+64+50 tokens. Same-chat pressure evicted L1 state;
  a fresh Electron chat refaulted the old partial-terminal chain from L2 and
  exact-finaled. A real UI process replacement then began with zero L1 tokens
  and four persisted disk blocks and restored the same 178 tokens with three
  native-TQ disk hits and exact visible output.
- `PAGED-FRUGAL-WORKER-DISK-DETAIL`: `VERIFIED-LIVE` after repair. Live raw
  probes found that later frugal indexed hits read their q4 payloads from L2
  while usage under-reported `paged+tq-native`. Fetch-time sampling happened
  before worker reconstruction performed the reads. Successful reconstruction
  now records actual disk blocks and promotes that fact into request detail;
  it does not infer disk merely because L2 is enabled. Focused validation is
  114 passed with two intentional deselections.
- Patched live evidence: Electron and later same-process Responses/Chat all
  reported 178 `paged+disk+tq-native` tokens. Responses emitted 316 reasoning
  and 10 content deltas before completion. Chat emitted 508 reasoning and 10
  content deltas, followed by finish, one usage-only chunk, and `[DONE]`.
  A same-session required-tool turn restored 192 disk tokens, executed exactly
  one real `file_info(panel/package.json)`, consumed its 5.2 KB result, and
  exact-finaled without a warning.
- Evidence:
  `docs/internal/release-gates/20260719_m27_paged_l2_partial_refault/`.
- This closes only the M2.7 paged full-KV child row. Paged-Off prompt-disk
  partial restore, other cache architectures, media/gateway soak, full suites,
  build, and release remain separate gates.

## 2026-07-19 03:27 - effective no-tool state after a real tool result

- `EFFECTIVE-NO-TOOL-PARSER-SEED`: `VERIFIED-LIVE` on source commit
  `ffb9ed7db`. A raw MiniMax M2.7 Chat continuation retained its public tool
  schemas for history fidelity while setting `tool_choice=none` and
  `enable_thinking=false`. Before the fix it emitted no visible content,
  contradictory `stop` then `length` terminals, and a reasoning-only warning.
  The same loaded artifact and tool result completed through Responses, which
  isolated the failure to endpoint/parser integration rather than the
  JANGTQ/MXTQ artifact.
- The renderer had correctly stripped tools, but Chat and Responses parser-
  seed/answer-policy paths re-read the public `request.tools`. The shared
  `_tools_available_for_generation` helper now uses the effective prompt tool
  set and always returns false for `tool_choice=none`. Streaming and non-
  streaming Chat/Responses use the same contract. No tool argument, output,
  thinking tag, sampler, or output budget is synthesized or coerced.
- Focused current-source validation passed 244 tests with three intentional
  deselections. After a real Electron Stop/Start, the identical Chat request
  emitted 18 progressive content deltas, exact-finaled
  `M27-CHAT-TOOL-CONTINUE-DONE SIZE=5.2 KB`, emitted one stop, one terminal
  usage event, and one `[DONE]`; 173 tokens restored as
  `paged+disk+tq-native`. A retained-schema Responses continuation separately
  emitted 19 progressive content deltas and one completed terminal.
- The current Electron three-turn row also passed separate Auto reasoning and
  content, exactly one real `file_info(panel/package.json)` call/result/final,
  and a no-second-tool same-chat recall. SQLite preserves the reasoning,
  content, tool call, tool result, cache detail, and warnings independently.
- Evidence:
  `docs/internal/release-gates/20260719_m27_protocol_parity/`.
- Scoped status is `VERIFIED-LIVE` for current-source Electron, Chat, and
  Responses. Overall protocol parity remains `PARTIAL`: Anthropic, Ollama,
  cancellation/disconnect/mid-stream recovery, and signed-app repeat are open.

## 2026-07-19 03:4x - Anthropic tool-name, MiniMax marker, and no-tool parity

- `ANTHROPIC-SPLIT-TOOL-NAME`: `VERIFIED-LIVE` on commit `c707bb61a`.
  Chat introduced a tool id before its name; the Anthropic adapter opened an
  invalid empty-name block. It now buffers by tool index until id and name are
  both known. Current live output has one `file_info` block, exact path JSON,
  no error, and one terminal stop.
- `MINIMAX-ORPHAN-OUTER-OPENER`: `VERIFIED-LIVE` on commit `d7f74b982`.
  M2.7 emitted a complete invoke and outer close after the opener was consumed.
  The parser now recovers only that unambiguous native-control shape. A visible
  invoke without the MiniMax close remains content. Raw Anthropic and a real
  Electron built-in-tool turn each executed exactly one valid `file_info`.
- `ANTHROPIC-EFFECTIVE-NO-TOOL-PROMPT`: `VERIFIED-LIVE` on commit
  `4a53f16e1`. The route previously rendered retained public schemas even when
  `tool_choice=none`, causing meta-text before the exact post-tool answer. The
  patched route uses the effective prompt tool set consistently. Live follow-up
  emitted the exact marker over 17 content deltas, no reasoning/tool block,
  one `end_turn`, and one `message_stop`.
- Focused current-source validation passed 119/119 selected tests. Evidence:
  `docs/internal/release-gates/20260719_anthropic_tool_parity/`.
- Scoped Anthropic status advances to `VERIFIED-LIVE`. Overall protocol parity
  remains `PARTIAL`: Ollama, cancellation/disconnect/injected mid-stream
  recovery, signed-app repeat, and other parser families remain open.

## 2026-07-19 03:5x - shared reasoning separator and Ollama terminal parity

- `THINK-STREAM-STRUCTURAL-SEPARATOR`: `VERIFIED-LIVE` on commit
  `c1db6b745`. Direct Chat and Ollama both exposed two structural newlines
  before thinking-enabled visible content. The shared think-tag streaming
  parser now suppresses only the whitespace-only boundary before the first
  visible byte; later formatting is preserved. Qwen3, DeepSeek-R1, and
  MiniMax-M2 split/same-delta paths are pinned by 300 focused passing tests.
- `OLLAMA-GENERATE-USAGE-TERMINAL`: `VERIFIED-LIVE` on commit `01d95b448`.
  Templated `/api/generate` previously emitted the finish terminal and dropped
  the later usage terminal. The wrapper now defers and merges upstream finish
  and usage into one `done:true` row. Live after has exact progressive output,
  one terminal, `eval_count=134`, and `prompt_eval_count=74`.
- Current-source `/api/chat` stream/non-stream and the tool loop pass for M2.7:
  separate thinking/content, exact final, one terminal, nonzero usage, one
  `file_info` with object arguments, and no second call after its real result.
- Real Electron CDP sampling recorded progressive reasoning growth through
  13/23/43/62/81/108 tokens before exact visible output. SQLite kept 529
  reasoning characters separate, no tool call, and no warning.
- Evidence:
  `docs/internal/release-gates/20260719_ollama_stream_tool_parity/`.
- Overall protocol parity remains `PARTIAL`: cancellation/disconnect/injected
  mid-stream failure/recovery, signed-app repeat, raw generate, multi-tool, and
  other model/parser family live rows remain open.

## 2026-07-19 11:05 - Responses cancellation truth and disconnect recovery

- `RESPONSES-CANCEL-FALSE-SUCCESS`: `VERIFIED-LIVE` on commit `ae498c70b`.
  Before the repair, cancelling a long stream after three content deltas returned
  HTTP 200 but finalized partial text `1, ` as a completed output item inside a
  `response.completed` terminal. This was a shared Responses finalization defect,
  not MiniMax-M2.7, JANGTQ/MXTQ, TurboQuant, reasoning-parser, or cache behavior.
- Aborted or detected-disconnect streams now retain incomplete item state, emit
  `response.incomplete` with `reason=cancelled`, skip the visible-answer retry, and
  do not enter Responses history. A mid-stream exception now emits
  `response.failed` instead of a contradictory `response.completed` envelope.
- Regression validation passed 111 selected cancellation/Responses/reasoning/API
  tests with 741 deselected. The exception terminal is directly pinned, but safe
  live HTTP fault injection remains open.
- After a real Electron Stop/Start, PID 95088 cancelled response
  `resp_7b8a2f8c5881` after three progressive content deltas: HTTP 200 cancel, one
  incomplete output item, one `response.incomplete`, reason `cancelled`, zero
  active requests, and no persisted history. A separate client disconnect after
  five deltas reached idle in 1.12s; immediate recovery streamed 12 content deltas
  to exact marker `M27-AFTER-DISCONNECT-PATCH-DONE` and completed once.
- Evidence:
  `docs/internal/release-gates/20260719_response_cancel_disconnect/`.
- Scoped cancellation/disconnect recovery is `VERIFIED-LIVE`. Overall protocol
  parity remains `PARTIAL` for safe live mid-stream exception injection, Chat
  cancellation/disconnect behavior, signed-app repeat, raw Generate multi-tool,
  and remaining model/parser families.

## 2026-07-19 11:14 - Ollama and Electron simultaneous multi-tool loop

- `M27-OLLAMA-MULTITOOL`: `VERIFIED-LIVE` on test commit `1b35d7a9b`.
  The Electron-started M2.7 JANGTQ/MXTQ process emitted exactly two valid Ollama
  calls in one terminal: `file_info(panel/package.json)` and `run_command(pwd)`.
  Both arguments were objects, the terminal reason was `tool_calls`, and only the
  exact requested real operations were executed.
- After both named tool results, the next Ollama stream emitted fresh separate
  reasoning, 30 progressive content rows, the exact final marker, one stop
  terminal, and no repeated call. Focused adapter/protocol validation passed
  31 selected tests with 119 deselected.
- A fresh real Electron chat independently executed the same two built-ins once
  each. Row 372 preserves both call ids, arguments, matching results, 740 reasoning
  characters, exact visible content, no warning, and 192
  `paged+disk+tq-native` cached tokens. Health returned idle afterward.
- Evidence: `docs/internal/release-gates/20260719_ollama_multitool/`.
- This closes only the M2.7 Ollama/Electron two-tool child row. Other parser
  families, signed-app repeat, media tools, cancellation, and long-loop soak remain
  open.

## 2026-07-19 11:2x - Chat disconnect and Electron user-stop recovery

- `CHAT-CLIENT-DISCONNECT-RECOVERY`: `VERIFIED-LIVE` on current source
  `576e12733`. A raw Chat client closed after five progressive content deltas; the
  Electron-started engine returned idle in 1.078s. The immediate fresh Chat request
  streamed 12 exact content deltas, one stop, one usage tail, and `[DONE]`.
- `ELECTRON-USER-STOP-RECOVERY`: `VERIFIED-LIVE`. Stopping during prefill created
  no false assistant row. Stopping after the visible UI painted through integer 76
  retained actual partial content plus `[Generation interrupted]`, real partial
  metrics, and no warning/tool/reasoning fabrication. An immediate same-chat turn
  exact-finaled `M27-ELECTRON-AFTER-STOP2-DONE`; health was idle.
- Temporary controls were restored to Auto reasoning, built-in tools On, and blank
  Max Tokens. Existing focused coverage passed 7 Python selections and 368 panel
  tests. Evidence:
  `docs/internal/release-gates/20260719_chat_disconnect_stop_recovery/`.
- Remaining failure parent row: safe live mid-stream engine exception, signed-app
  repeat, gateway network loss, other parser/model families, and prolonged soak.

## 2026-07-19 - Bonsai partial-prefix and shared Responses post-tool finalization

- Artifact truth was re-read from the live bundle: this model is Qwen3.5
  `JANG_AFFINE_1BIT` (1.1128 actual bits), not JANGTQ/MXTQ. Its 64-layer graph
  has 16 full-attention KV lanes and 48 linear-attention companion lanes.
- Real Electron Auto launched q8 storage for attention KV only plus native
  companion state. A 6,336-token sibling prefix reused as `paged+ssm` from RAM
  twice and as `paged+ssm+disk` after visible process replacement. Health
  records 99 native-TQ disk hits and a successful SSM checkpoint restore.
- A release-blocking raw Responses continuation reproduced a shared finalizer
  defect: rejected reasoning-channel tool markup was copied into output text,
  the tools-free answer pass was skipped, and the terminal was incomplete.
  Commit `359ce6b2b` keeps that text private and re-arms the existing answer
  pass. All three neighboring test files pass 147 with three deselected.
- Current PID 1054 live proof: raw Responses emitted 454 reasoning deltas and
  one valid tool call, then 185 separate reasoning plus 18 progressive exact
  content deltas after the real result, ending completed. Fresh Electron row
  385 independently executed one real tool, exact-finaled, had no warning, and
  visibly painted `SIZE=` through `SIZE=5.2 KB` over multiple DOM mutations.
- Scoped verdict: `PASS-LIVE` for Bonsai Auto partial-prefix RAM/L2 and this
  shared Responses continuation defect. Overall matrix remains `PARTIAL` for
  cross-parser repeats, long/stochastic/media/eviction/signed-app rows. Evidence:
  `docs/internal/release-gates/20260719_bonsai_partial_prefix_responses/`.

## 2026-07-19 - M3 current-head Auto/tool recheck and bundled-engine drift

- Real Electron Sessions-card Start loaded MiniMax-M3 PID 2277 and stopped
  Bonsai. Bundle/config and health identify affine `JANG_2L` with native
  `minimax_m3_msa_v1`; generic TQ KV remains correctly Off.
- Electron Auto rows 388/391 produced non-empty separate reasoning/content.
  The same-chat second turn executed one real `file_info`, returned the 5.2 KB
  result, reused 8,980 `paged+disk` tokens, and painted the final progressively
  with no warning.
- Current raw Responses and Chat each passed Auto no-tool, required-tool, and
  real-result continuation. Reasoning, tool arguments, and content streamed on
  separate rails; every protocol emitted its proper terminal.
- Focused source/runtime validation is 759 passed, 46 skipped, one deliberate
  packaging-verifier deselection.
- New release blocker `BUNDLE-PYTHON-SOURCE-DRIFT`: current `server.py` hash
  `8e462960...` differs from bundled hash `193ad562...` after the Responses
  repair. Do not package until `bundle-python.sh` is run and the verifier is
  rerun without deselection.
- Evidence:
  `docs/internal/release-gates/20260719_m3_current_postfinalizer/`.
- Scoped M3 text verdict is `PASS-LIVE`; M3 larger-media/OCR/terminal-delay/
  REAP32/signed-app and cross-parser rows remain `PARTIAL`.
## 2026-07-19 - Gemma 4 current parser/stream/cache recheck

- `GEMMA4-CURRENT-AUTO-STREAM`: `PASS-LIVE` with adequate request headroom.
  Electron Start loaded the 26B affine `JANG_4M` bundle as PID 4530 with Gemma4
  reasoning/tool parsers, Responses wire, tools On, prefix+paged+L2 On. Row 394
  persisted non-empty coherent content separately from reasoning; row 397 executed
  exactly one real `file_info(panel/package.json)`, returned 5.2 KB, restored 7,168
  `paged+mixed_swa+disk` tokens, and stored no warning. CDP observed progressive
  visible-content painting.
- Raw Responses and Chat each passed no-tool, required-tool, and tool-result
  continuation at 4,096 output tokens with separate reasoning/content deltas and clean
  completed/stop terminals. The 512-token controls correctly ended
  `response.incomplete(max_output_tokens)` and Chat `length`; they are retained as
  negative cap controls, not counted as complete streams.
- `GEMMA4-DEFAULT-REASONING-EFFICIENCY`: `PARTIAL`. The short default UI prompt used
  3,322 output tokens and 15,629 reasoning characters before a coherent two-sentence
  answer. No replay/leak/freeze was observed, but stochastic verbosity remains open;
  do not mask it with a hidden cap or sampler clamp.
- Cache truth: live rotating slots remain native and generic live TQ is Off, while
  stored prefix/block payloads use q4 for full and sliding KV and preserve rotating
  metadata. Health after the current turns records 10 prefix hits, 7,950 tokens saved,
  56 scheduler disk hits, and 239 native-TQ L2 hits.
- Focused current-source validation: 361 passed. Evidence:
  `docs/internal/release-gates/20260719_gemma4_current_parser_stream/`.
- Release remains blocked on the already-recorded source/bundled Python hash drift.

## 2026-07-19 - DSV4 current-source Auto UI, parser stream, and restart/L2 recheck

- `DSV4-AUTO-UI-STATE`: `VERIFIED-LIVE`. Commit `4e723f311` declares the
  DSV4 `dsml`/`deepseek_r1` reasoning capabilities and adds a real Auto choice
  for an absent override. After a full Electron main-process relaunch (not only
  renderer HMR), the settings drawer visibly selected Auto and left
  Instruct/Reasoning/Max unselected.
- `DSV4-ELECTRON-RESTART-TOOL`: `VERIFIED-LIVE`. The correct-PATH dev log
  found `/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`; the real Sessions
  Start action loaded PID 8882 before a prompt. Fresh row 406 kept two reasoning
  rails separate, made exactly one `file_info(panel/package.json)` call, used
  the real 5.2 KB result, produced non-empty visible content, and stored no warning.
- `DSV4-NATIVE-L2-RESTART`: `VERIFIED-LIVE`. Post-turn health identifies the
  `dsv4` batch path, two disk hits, 3,173 L2 block tokens, and zero generic TQ
  writes/hits. This artifact is affine JANG, not JANGTQ/MXTQ/MXFP; its 43-layer
  native SWA+CSA/HCA pool cache remains outside generic TurboQuant.
- `DSV4-PROTOCOL-STREAM`: `VERIFIED-LIVE` for controlled current prompts.
  Identical Responses/Chat prompts emitted byte-identical separated reasoning
  and progressive content; both protocols completed one real tool continuation
  with clean terminals. This rules out an endpoint-specific re-emission defect
  for the controlled row.
- `DSV4-STRICT-MARKER-AND-STOCHASTIC-QUALITY`: remains `OPEN/PARTIAL`.
  Both APIs mutated one synthetic marker before endpoint emission, a weaker
  stochastic prompt duplicated a path, and another weak post-tool prompt
  hallucinated package contents. No artifact blame is assigned; a matched
  same-artifact reference-runtime A/B remains required.
- Focused validation: 329 Python tests, 100 panel tests, and panel typecheck pass.
  Evidence: `docs/internal/release-gates/20260719_dsv4_current_parser_auto_stream/`.
- Release stays `PARTIAL/BLOCKED`: bundled Python remains stale relative to
  current source, and the other recorded matrix rows remain open.

## 2026-07-19 - Laguna current stream, settings, q4 determinism, and eviction recheck

- `LAGUNA-ELECTRON-STREAM`: `VERIFIED-LIVE`. The real Sessions Start action
  loaded Laguna-M.1 on port 8015. Natural row 409 kept 3,244 reasoning
  characters separate from a non-empty two-sentence answer. Tool row 412 made
  exactly one `file_info(panel/package.json)` call, used the real 5.2 KB result,
  and painted the final answer progressively; the DOM trace records visible
  content-length growth from 30 through 208 rather than one terminal batch.
- `LAGUNA-MULTITURN-RESTART`: `VERIFIED-LIVE`. After two real UI Save & Restart
  cycles, row 415 recalled the prior 5.2 KB tool result without a new tool call,
  produced visible content, stored no warning, and restored 4,980
  `paged+disk+tq-native` tokens.
- `LAGUNA-PROTOCOL-PARITY`: `VERIFIED-LIVE` for the controlled matrix. Raw
  Responses and Chat Completions, stream and non-stream, preserve separate
  reasoning/content, one schema-valid tool call, real-result continuation, and
  clean completed/stop/DONE terminals.
- `LAGUNA-CACHE-SETTINGS-PARITY`: `VERIFIED-LIVE`. Auto/1000 displayed and
  launched q4 stored-prefix TQ with paged+block L2. Explicit UI None/max-four
  launched `--kv-cache-quantization none --max-cache-blocks 4`, showed TQ Off,
  and health reported zero TQ objects/writes/hits. Restoring Auto removed the
  explicit mode and health again reported `uncalibrated_full_kv_storage_tq4`.
- `LAGUNA-EVICTION-PARTIAL-PREFIX`: `VERIFIED-LIVE` with TQ Off. Four blocks
  forced ten L1 evictions and ten disk writes; the oldest 4,538-token prompt
  refaulted 192 tokens from three disk blocks as `paged+disk` and reproduced
  exact answer `166`.
- `LAGUNA-Q4-COLD-EQUIVALENCE`: `PARTIAL/OBSERVED-MISMATCH`. Four greedy q4
  disk restores are byte-identical to one another, but differ from the first
  full-precision cold answer. Three bypass-cold runs and cold plus four raw
  `paged+disk` restores with explicit None are byte-identical to the cold
  baseline. The observed boundary is lossy q4 stored-KV restoration; no sampler,
  prompt, or output rewrite was added to hide it.
- `LAGUNA-LATENCY-LONG-SOAK`: remains `OPEN/PARTIAL`. Controlled correctness
  passes, but the natural row is about 23.8 tok/s and restart recall TTFT is
  5.10 s; long agent reliability and the performance target are not closed.
- Validation: 411 Python passed / 1 skipped, 771 panel passed, and TypeScript
  typecheck passed. Test-only commit `6f7b29bc3` updates one stale source
  assertion to the effective-tool helper already used by both streaming paths.
- Evidence: `docs/internal/release-gates/20260719_laguna_current_stream_tq_determinism_eviction/`.
- Release remains `PARTIAL/BLOCKED`; bundled Python is stale and the other
  explicit matrix rows remain open.

## 2026-07-19 - Mistral3 JANGTQ prefill/output failure and current-worklist reconciliation

- `MISTRAL35-JANGTQ-OUTPUT`: `BLOCKED_CURRENT_ARTIFACT_RUNTIME` on pushed head
  `fad7356d4`. The official 88-layer `mistral3/ministral3` JANGTQ2 bundle passes
  strict hydration (`swapped=616 skipped=0`). The legacy path did not reach TTFT
  after more than two minutes. The temporary dense-Mistral MPP NAX Auto exception
  reduced TTFT to about 14 seconds but emitted only newline tokens. A second live
  Electron attempt after FP32 A/B NAX accumulation reproduced newline-only output
  at an explicit 64-token cap and was visibly cancelled.
- Projection-level official-weight agreement did not predict full-model output,
  so commit `fad7356d4` withdraws the broad exception and restores the safe generic
  MXTQ/JANGTQ policy. Twelve focused policy tests pass. No prompt coercion, sampler
  clamp, fabricated output, or official-artifact mutation was added.
- The independent Jang NAX wide-range correction is pushed as `000e41c` after 23
  live Metal-kernel tests. It is not claimed as a Mistral runtime fix.
- Evidence:
  `docs/internal/release-gates/20260719_mistral35_jangtq_prefill/`.
- Documentation/source reconciliation is preserved at
  `docs/internal/release-gates/20260719_current_reconciliation/README.md`.
  Current status is `PARTIAL_NO_1_6_12_RELEASE`: the post-release head needs
  selected current-head model/API/Electron rows, complete suites/build, a fresh
  bundled-Python rebuild, and the full Sequoia/Tahoe signing/notarization chain.
## 2026-07-19 - Qwen3.6 35B JANGTQ current-source stream, tool, and L2 partial-prefix gate

- `Q35-JANGTQ-CURRENT-TEXT-AGENT`: `PASS-LIVE_SCOPED` at pushed source
  `87e11c5ee`. The real Electron Sessions Start path loaded the MXTQ/JANGTQ2
  artifact; a fresh current-source chat executed exactly one real
  `file_info(panel/package.json)` and exact-finaled `5.2 KB` with reasoning
  separate, visible content non-empty, no warning, and 256
  `paged+ssm+disk` tokens. Raw current-source Responses and Chat each produced
  one valid tool call and streamed the real-result continuation over 256
  reasoning plus 18 content deltas to clean terminals with usage.
- `Q35-HYBRID-Q4-L2-PARTIAL-PREFIX`: `PASS-LIVE_SCOPED`. After real Electron
  Stop/Start, an exact tool replay restored seven q4 native-TQ blocks and two
  SSM disk checkpoints. A separate 2,587-token seed then changed only its final
  suffix after another restart: 2,560 tokens restored from 40 disk blocks, all
  40 native-TQ hits, with one complete 30-layer SSM disk hit and exact changed
  output in 0.476s. Current source repeated the row in 0.492s.
- `Q35-TQ-STARTUP-TRUTH`: fixed by `87e11c5ee`. The old log said stored
  quantization `none`, falsely implying TQ storage was disabled. Source trace
  and live counters show `none` applies only to the second generic
  QuantizedKVCache wrapper; architecture-selected q4 attention-TQ remains the
  prefix/paged/L2 codec. The corrected log says so without altering the proven
  cache path or double-quantizing it. Focused validation: 103 Python and 127
  panel tests.
- `Q35-SESSION-CARD-QUANT-LABEL`: `PASS-LIVE` at pushed commit `54222003d`.
  The shared label formatter now reads current top-level JANG sidecar profiles,
  the detector returns `JANGTQ2 (2b)`, and SessionCard limits its fallback to
  the actual bundle basename rather than provider directories. After a complete
  Electron-main relaunch, the real Sessions card and active header both showed
  `JANGTQ2 (2b)`; affine Bonsai/DSV4/Gemma labels remained JANG, while base
  Nemotron MXFP4 and a base MXFP4 child under `jangq-ai/` had no false badge.
  The latter was UI-only classification proof; the excluded Mistral MXFP4 model
  was not loaded or generation-tested. Focused panel validation is 94/94 plus
  typecheck. Fresh row 440 then executed one real file tool with non-empty final
  content and 3,904 `paged+ssm+disk` tokens, but misspelled the exact marker;
  strict sampled formatting/tool reliability therefore remains PARTIAL.
- Retained reds: strict sampled formatting/tool reliability remains PARTIAL;
  advertised vision remains OPEN because live health says
  `vl_runtime_available=false`.
- Evidence:
  `docs/internal/release-gates/20260719_qwen35_jangtq_current/`.

## 2026-07-19 - HY3 native-MTP D1 current-source Electron/API/cache gate

- `HY3-CURRENT-AUTO-D1-AGENT`: `VERIFIED-LIVE_SCOPED` at source cutoff
  `0e09ce789`. The real bundle is affine `JANG_2K` (not JANGTQ/MXFP), declares
  one next-token layer, and contains 42 MTP tensors. Electron single-model mode
  stopped Qwen PID 26427 and loaded HY3 PID 27632 with Hunyuan tools, qwen3
  reasoning, Auto q4 stored prefixes, and `--native-mtp-depth 1`.
- Rows 443/446/449 were exact three-turn finals with distinct 844/1,263/483
  character reasoning. Row 446 executed exactly one real `file_info`; row 449
  called no tool and recalled both prior values. MutationObserver traces prove
  the visible finals painted progressively after reasoning/tool execution.
- Raw curl-N Responses emitted 215+14 no-tool deltas, one valid required tool,
  and 107+17 post-result deltas. Chat emitted 142+13, one valid required tool,
  and 89+16 post-result deltas. Finals were exact; terminals/usage/DONE ordering
  was clean with no parser leakage or warning.
- `HY3-CURRENT-D1-Q4-L2`: `VERIFIED-LIVE_SCOPED`. The first process recorded
  1,194 drafted/497 accepted tokens. After Electron Stop/Start to PID 29852,
  row 452 restored 4,655/4,872 tokens as `paged+disk+tq-native`, exact-recalled
  without a tool, and independently recorded 87 drafted/35 accepted tokens.
  Health counted 73 disk hits, all 73 native-TQ hits, and zero resident L1 bytes.
  Accepted/main-history tokens therefore reach the q4 prefix boundary; rejected
  speculative drafts are not claimed as persisted history.
- No source fix was needed. Current focused HY3/MTP/Hunyuan/TQ/reasoning tests
  pass 318/318. Retained PARTIAL: long/stochastic soak and a new current-source
  MTP-Off versus D1 performance A/B. VL/audio/video are N/A for this text-only
  artifact. Evidence: `docs/internal/release-gates/20260719_current_hy3_mtp/`.

## 2026-07-19 08:21 - Step 3.7 JANGTQ zero-patch VL repair and current media/L2 gate

- `STEP37-ZERO-PATCH-MLX-METADATA`: `FIXED_SOURCE + VERIFIED-LIVE_SCOPED`.
  The real Electron image request failed with 169 placeholders and zero
  embeddings even though `pixel_values=(1,3,728,728)` arrived. The owning
  defect was `array([0])` patch metadata being reduced to `[]` by Python
  truthiness. The model boundary now normalizes MLX/NumPy arrays to a 1-D
  integer list and uses an explicit `None` check. No parser, sampler, prompt,
  or quant workaround was added.
- `STEP37-CURRENT-MEDIA-SALT`: `VERIFIED-LIVE_SCOPED`. Electron rows
  455/458/461/464 prove A cold, identical-A 4,290-token resident hit,
  same-shape B miss/no-A-leak, and return-A reuse. Row 467 reads the real
  four-second MP4 exactly as `VIDEO-B-8264`. DOM observers prove progressive
  visible paints after separate reasoning rails.
- `STEP37-CURRENT-L2-TQ4`: `VERIFIED-LIVE_SCOPED`. A visible Stop/Start left
  zero L1 tokens and 15,987 L2 tokens. Row 470 restored all 4,290 tokens as
  `paged+mixed_swa+disk` with `disk_hit=true`, 68 block-disk hits, 68 q4
  native-TQ hits, exact content, and 1.71s TTFT.
- `STEP37-CURRENT-API-STREAM`: `VERIFIED-LIVE_SCOPED`. Literal curl-N Chat
  emitted 46 reasoning plus 42 content deltas, stop/usage/DONE. Responses
  emitted 73 reasoning-summary plus six exact content deltas and one completed
  terminal with 223 cached input tokens.
- `STEP37-CONTENT-STRICTNESS`: `PARTIAL/FAIL`. Cold image generations placed
  self-correction prose after the reasoning terminator. Raw Chat proves it was
  genuine content, not parser leakage or Electron batching. Do not hide it.
- `STEP37-SAME-PROCESS-DISK-DETAIL`: `FIXED_SOURCE + VERIFIED-LIVE_SCOPED`.
  Row 464 remains the control where aggregate block-disk/native-TQ counters
  increased but per-request detail omitted disk. The MLLM path now merges its
  fetch-time sample with `BlockAwarePrefixCache._last_reconstruct_disk_blocks`,
  matching the existing text scheduler. After source reload, rows 473/476 both
  reported 4,290 `paged+mixed_swa+disk` tokens; the immediate same-process row
  recorded `disk_hit=true` and `disk_blocks=68`.
- `STEP37-RESTART-PID-UI`: `FIXED_SOURCE + VERIFIED-LIVE_SCOPED`. The local
  `session:ready` paths now include the real PID; `SessionsContext` carries it
  through event and promise paths and clears it on Stop. A full Electron-main
  relaunch logged the project `.venv/bin/vmlx-engine`. Visible Start showed PID
  38968 in both Server and Chat; visible Stop cleared the header and SQLite PID;
  the next visible Start showed PID 39507, matching SQLite and `ps`. Exactly one
  local engine was running. Panel PID/session/port tests pass 174/174 plus
  typecheck. Evidence is in the Step current-source gate.
- `STEP37-COLD-MEDIA-LATENCY`: `PARTIAL`. Cold image TTFT remained
  44.44-44.87s and video TTFT 55.21s. Larger video and retained stochastic
  loop soak remain open without sampler coercion.
- Expanded current verification passes 513 with two intentional deselections.
  Evidence:
  `docs/internal/release-gates/20260719_current_step37_jangtq/`.

## 2026-07-19 - Immediate-Stop prompt-disk durability and first-turn role eviction

- `PROMPT-DISK-TERMINAL-STOP-DURABILITY`: `FIXED_SOURCE + VERIFIED-LIVE` at
  `7a146eefb`. Text `EngineCore` and multimodal `MLLMScheduler` now honor the
  existing terminal-cleanup barrier before cancelling their loop on shutdown.
  The final delta still dispatches before cache persistence; only shutdown
  waits for an in-flight terminal cleanup.
- `PROMPT-DISK-FIRST-TURN-ROLE`: `FIXED_SOURCE + VERIFIED-LIVE`. A single user
  or system message now produces a real segment boundary instead of falling
  back to `assistant` priority. The pre-fix 1,539-token one-turn base logged a
  store and then evicted itself at the 10 GB ceiling; its first follow-up
  missed. After the repair, an immediate visible Electron Stop retained the
  new 1,322-token entry as `user` and evicted the older 1,582-token LRU entry.
- `OPENPANGU-NONPAGED-SSD-PARTIAL-CURRENT`: `VERIFIED-LIVE`. After visible
  process replacement, Electron restored 1,321/1,395 tokens from disk with
  zero resident L1 bytes, separate fresh reasoning, progressive exact content,
  and no warning. Independent detached Responses and Chat clients after UI
  restarts both restored 1,321 disk tokens and exact-finaled. Chat preserved
  strict finish -> one usage-only chunk -> `[DONE]` ordering.
- `RESPONSES-STREAM-USAGE-EVENT-PARITY`: `OPEN / INVESTIGATE`. The detached
  Responses request opted into `stream_options.include_usage` and received 483
  vMLX `response.usage` extension events plus terminal usage. This gate proves
  delivery and accounting, but does not assume that custom event is part of
  the current public Responses protocol.
- Focused validation passes 119/119. Evidence:
  `docs/internal/release-gates/20260719_prompt_disk_stop_role_durability/`.
  Generic paged/block-L2/TQ rows remain N/A for openPangu and open on their
  compatible-model matrix rows. Overall release remains `PARTIAL`.

## 2026-07-19 - Responses stream usage event parity

- `RESPONSES-STANDARD-USAGE-EVENTS`: `FIXED_SOURCE + VERIFIED_LIVE_SCOPED` at
  `cc4251318`. Chat-style `stream_options.include_usage` can no longer cause an
  ordinary Responses stream to emit the private `response.usage` event. A raw
  current-source stream produced 383 reasoning and nine content deltas, zero
  incremental usage events, one completed terminal, final usage, and exact
  visible content.
- `RESPONSES-LOCAL-INCREMENTAL-USAGE`: `VERIFIED_LIVE_SCOPED`. The explicitly
  header-gated local extension produced 337 incremental usage events plus one
  completed terminal with final usage, without changing reasoning/content or
  event-sequence correctness.
- `ELECTRON-RESPONSES-USAGE-REQUEST-SHAPE`: `FIXED_SOURCE +
  VERIFIED_LIVE_SCOPED`. The panel omits the nonstandard body field and sends
  the private header only to local engines. A fully relaunched Electron main,
  real visible Start, and fresh chat proved character-wise reasoning and
  progressive visible content, exact non-empty final output, server-backed
  metrics, no tool call, and no warning.
- Expanded current-source validation is 83 Python plus 111 panel tests and
  clean typecheck. Evidence:
  `docs/internal/release-gates/20260719_responses_usage_extension_parity/`.
- Retain `PARTIAL`: live remote-provider smoke, signed packaged-app repeat,
  failure/disconnect soak, and the rest of the explicit release matrix.

## 2026-07-19 - current-head Responses usage/gateway recovery reconciliation

- Status: `VERIFIED-LIVE_SCOPED` at source cutoff `76e8d6c1e`; overall API
  parity remains `PARTIAL`.
- This is a bounded current-head spot-check and matrix reconciliation, not a
  third replacement for the stronger retained gates. Source keeps public
  Responses usage on `response.completed`, exposes incremental
  `response.usage` only through `X-vMLX-Stream-Usage: incremental`, and sends
  that private header only from the local Electron client.
- Direct Laguna without the private header emitted seven progressive content
  deltas, exact `USAGE-STANDARD-CURRENT-OK`, one `response.completed`, terminal
  usage `27/10/37`, and zero `response.usage` events. Explicit negotiation
  emitted ten private usage events, the same seven content deltas/exact final,
  and terminal usage with 23 `paged+tq-native` cached tokens.
- The ordinary gateway request emitted eight progressive content deltas, exact
  `USAGE-GATEWAY-CURRENT-OK`, one completed terminal, terminal usage
  `28/11/39`, and zero private usage events. This proves the current gateway
  did not inject the local telemetry header into a public client request.
- The earlier explicit-cancel gate remains authoritative for cancellation
  semantics: incomplete item/terminal state, `reason=cancelled`, no successful
  history, zero active requests, and immediate exact recovery after a client
  disconnect. The current Laguna client-abort spot-check again reached idle and
  a gateway follow-up completed; real Electron row 725 then visibly returned
  exact `UI-CANCEL-RECOVERY-OK`, with null reasoning/warnings and
  `paged+disk+tq-native` telemetry.
- Remaining: non-stream equivalents, stable tool-result continuation through
  every protocol, Chat/Anthropic/Ollama cancellation, safe live mid-stream
  exception injection, remote-provider smoke, and signed-app repetition.
- Evidence:
  `docs/internal/release-gates/20260719_responses_usage_extension_parity/` and
  `docs/internal/release-gates/20260719_response_cancel_disconnect/`.

## 2026-07-19 - Block-disk-only partial-prefix gate

- `GENERIC-BLOCK-L2-PAGED-OFF`: `PASS-LIVE scoped`. Current source now treats
  Block Disk Cache (L2) as an independent durable block backend when Paged RAM
  is explicitly Off. It does not substitute the legacy memory prefix cache if
  SSD initialization fails, and it does not drop the last KV payload until the
  queued block write is durably visible.
- `BLOCK-CAPACITY-TRUTH`: `FIXED_SOURCE + VERIFIED-LIVE`. The engine reserves
  block 0. UI, DB/argv, launch log, and health now report three usable 64-token
  blocks / 192 tokens from four configured blocks; idle utilization is `0.0`
  rather than counting the reserved null block as 25% user cache.
- `BLOCK-DISK-COUNTER-TRUTH`: `FIXED_SOURCE + VERIFIED-LIVE`. Public scheduler
  disk hit/miss fields now use actual BlockDiskStore reads, while promotion-only
  counters remain separately named. Final current-process health showed 21
  actual disk hits, 21 TQ-native hits, and zero resident KV bytes.
- The real Electron UI loaded MiniMax-M2.7 JANGTQ via Start with
  `--no-paged-cache`, block L2 On, and q4 TQ prefix storage. After process
  replacement, an identical fresh chat restored 192
  `block-disk+tq-native` tokens and exact-finaled. A raw long Chat API repeat
  independently restored 192/846 tokens and streamed seven content deltas.
- Timed Chat and Responses SSE separately proved reasoning deltas, progressive
  visible-content deltas, and correct terminal events. One tools-enabled
  synthetic Electron replay unnecessarily chose write/read tools; that
  model-choice observation is retained instead of being called a strict no-tool
  pass. The loop still returned the exact final marker.
- Validation: 13 focused Python passes, 299 panel passes, clean typecheck, and a
  passing aggregate cache contract with 454 cache-family and 115 panel-policy
  selections. Evidence:
  `docs/internal/release-gates/20260719_block_disk_only_partial/`.
- This closes only the compatible generic full-KV disk-only exact/partial/restart
  row for the exercised M2.7 artifact. Hybrid companion rederive,
  native/typed/mixed-SWA families, signed-app repetition, and overall release
  readiness remain separately `PARTIAL`/`OPEN`.

## 2026-07-19 - Qwen3.6 JANGTQ hybrid Block L2 with Paged RAM Off

- `Q35-HYBRID-BLOCK-L2-PAGED-OFF`: `FIXED_SOURCE + PASS-LIVE scoped`.
  UI policy, persistence, preview, and argv now permit an explicitly supported
  hybrid architecture to keep Block Disk L2 On while Paged RAM is Off. The
  engine refuses a silent RAM fallback if the authoritative SSD store fails.
- The real Electron Start action loaded
  `dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`, whose live health identifies MXTQ /
  JANGTQ rather than affine JANG. DB, preview, and process argv agree on
  `--no-paged-cache`, Block L2 On, Auto q4 attention-KV storage, qwen3
  reasoning, and qwen tool parsing.
- Cold row 569 stored the 5,165-token prefix. After visible Stop/Start, row 575
  restored 5,165/5,166 tokens as `block-disk+ssm+tq-native` with 0.45s TTFT.
  Changed-suffix row 578 restored 5,120/5,175 tokens with 0.67s TTFT. Health
  recorded actual SSD and q4 native-TQ hits, typed SSM companion disk hits,
  and zero paged-KV resident bytes.
- A same-process warm row exposed a false `paged+ssm+disk` label. Current source
  now reports `block-disk+ssm+tq-native` only when the disk-only manager and
  native-TQ reconstruction counters prove those tiers. The pre-fix screenshot
  is retained as evidence of the live red.
- Raw Chat and Responses streams kept reasoning and visible content separate,
  emitted progressive content deltas, and completed normally. Electron row 581
  executed exactly one real `file_info` call and exact-finaled with the 5.2 KB
  result. Raw strict formatting remains `PARTIAL` because the model added math
  prose before the requested lines.
- Validation: 911 expanded Python tests, 304 panel tests, and typecheck pass.
  Evidence:
  `docs/internal/release-gates/20260719_qwen35_hybrid_diskonly/`.
- Retain `OPEN`: Paged-On RAM-to-SSD hierarchy, explicit TQ Off, this row's
  Ollama/Anthropic parity, fault injection, signed-app repeat, other families,
  and overall release readiness.

## 2026-07-19 - Generic Paged RAM + block-disk L2 hierarchy

- `PAGED-L1-L2-INDEPENDENCE`: `FIXED_SOURCE + PASS-LIVE_SCOPED` at pushed
  commit `8a93aa910`. Block-disk L2 no longer silently enables frugal mode and
  discards ordinary Paged-On RAM payloads. Health exposes the effective
  `ram_mirror_policy` and `paged_frugal` state.
- Real Electron LFM rows prove cold write-through, a 306-token RAM-only hit
  with zero SSD reads, bounded block eviction, a 306-token SSD refault, process
  restart with zero L1 state, and a 256-token partial SSD prefix. The panel then
  switched Paged Off / L2 On and restored 256 tokens as `block-disk+ssm` while
  remaining at zero resident bytes. Paged On was restored afterward.
- Raw Chat, Responses, Anthropic, and Ollama stream/non-stream requests all
  emitted non-empty identical content progressively and reached their native
  terminal events. LFM strict formatting is retained as PARTIAL because it
  added explanation after the requested lines.
- Validation: 190/190 selected cache-family tests and 99/99 protocol/adapter
  tests. Evidence:
  `docs/internal/release-gates/20260719_paged_ram_ssd_hierarchy/`.
- Retain `OPEN`: LFM TQ eligibility/native encoding (this run reported zero
  TQ-native blocks), tool/reasoning protocol breadth, signed app, fault
  injection, full suites/build, and the rest of the family matrix.

## 2026-07-19 - LFM2.5 selective q4 TurboQuant KV checkpoint

- `LFM-SELECTIVE-TQ4`: `FIXED_SOURCE + VERIFIED-LIVE_SCOPED` at pushed commit
  `748929fe3`. The tested `LFM2.5-8B-A1B-MXFP4-CRACK` artifact is base MLX
  MXFP4, not affine JANG and not JANGTQ/MXTQ. Its native 24-slot cache has six
  full-attention `KVCache` positions and 18 `ArraysCache` convolution/SSM
  companions.
- Auto mode now derives LFM's real 64-wide attention heads, installs q4
  `TurboQuantKVCache` only at positions 2/6/10/14/18/21, and preserves every
  companion slot in native full precision. Health reports
  `uncalibrated_selective_attention_kv_storage_tq4`, `turboquant_native`, and
  `tq_native_enabled=true`; explicit user Off still bypasses this Auto branch.
- Real Electron settings showed Paged On, Block L2 On, and Auto. A cold UI turn
  wrote nine q4-native SSD blocks. A fresh-chat replay reused 576/1,204 prompt
  tokens as `paged+ssm+tq-native`, exact-finaled visibly, and wrote the matching
  typed SSM companion checkpoint.
- A real Electron Stop/Start replaced PID 69378 with 69763. Before the first
  request, L1 held zero tokens while SSD retained nine q4-native blocks plus one
  SSM companion entry. The first post-restart request restored all nine blocks
  (`disk_hits=9`, `tq_native_hits=9`) and the SSM checkpoint (`hits=1`), saving
  576 tokens.
- The deliberately 64-token raw Responses probe emitted 63 progressive content
  deltas and `response.output_text.done`, then correctly terminated as
  `response.incomplete`; its strict-format answer is therefore `PARTIAL`, not a
  cache failure and not a completed protocol-quality row.
- Validation: 63/63 focused hybrid-TQ/model-inspector tests. Evidence:
  `docs/internal/release-gates/20260719_lfm_native_tq4/`.
- Retain `OPEN`: the same native-TQ codec with Paged RAM explicitly Off,
  explicit Auto-to-Off live parity, full four-protocol/tool/cancel breadth,
  larger-context eviction, fault injection, signed-app repeat, and full suites.

## 2026-07-19 - MiniMax M2.7 JANGTQ full hierarchy/protocol checkpoint

- `M27-JANGTQ-Q4-HIERARCHY`: `VERIFIED-LIVE_SCOPED` on source cutoff
  `b31fdca95`. The tested artifact is full-KV `minimax_m2` with MXTQ/JANGTQ2
  weights; it is not affine JANG and not base MXFP. Auto cache storage selected
  q4 native TurboQuant for all 62 attention-KV layers. Bonsai remains the only
  all-q8 cache exception.
- `M27-PAGED-RAM-THEN-L2`: Electron rows 617/620/623/626 prove cold write,
  a 352-token RAM hit with zero disk reads, bounded eviction, and a 352-token
  SSD refault/promotion. Health recorded native q4 writes and hits.
- `M27-RESTART-PARTIAL`: PID replacement began with zero L1 payloads and 12
  persisted SSD blocks. Changed-suffix row 629 reused 320/360 tokens as
  `paged+disk+tq-native` and exact-finaled.
- `M27-PAGED-OFF-SSD-PARTIAL`: after the real UI set Paged Off / Block L2 On,
  health reported `block_disk_only`, `ram_mirror_policy=disk_only`, and zero
  resident bytes. Changed-suffix row 632 reused 320/361 tokens directly as
  `block-disk+tq-native`, exact-finaled, and kept resident bytes at zero.
  Paged On / L2 On was restored afterward.
- `M27-EAGER-SINGLE-MODEL`: the exact Sessions Start action stopped the prior
  LFM PID and left one local engine. Before a request, health reported
  `model_loaded=true`, `last_request_time=null`, and about 38.3 GB active
  memory. The Electron log resolved the project `.venv/bin/vmlx-engine`.
- `M27-FOUR-PROTOCOL-STREAM`: stream/non-stream Chat, Responses, Anthropic,
  and Ollama all returned HTTP 200, identical non-empty progressive visible
  bytes, and native terminal events. Native reasoning separately produced 369
  reasoning deltas and eight content deltas on each protocol with no think-tag
  leakage.
- `M27-TOOL-MULTITURN`: raw Responses emitted exactly one valid
  `file_info(panel/package.json)` and exact-finaled its result continuation.
  Electron row 647 independently executed the tool once and exact-finaled;
  same-chat rows 650/653 recalled the prior result without a second call.
  Electron reasoning rows 641/644 produced distinct reasoning bytes and
  non-empty exact visible answers.
- Evidence:
  `docs/internal/release-gates/20260719_minimax_m27_tq_hierarchy_protocol/`.
- This closes only the named M2.7 child rows. MiniMax M3 VL/video, typed
  DSV4/openPangu, hybrid SSM/GDN, mixed-SWA, gateway/network soak, remaining
  eager routes, signed-app repetition, and the overall release remain open.

## 2026-07-19 - MiniMax-M3 terminal dispatch and 14-second video

- Status: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED`; global release remains
  `PARTIAL_NO_1_6_12_RELEASE`.
- `M3-TERMINAL-STALL`: live raw Responses reproduced a 2.3453-second gap
  between the last progressive content delta and `response.output_text.done`.
  Source trace found the clean M3 sparse-MSA prompt-boundary re-prefill inside
  `_process_batch_responses`, before EngineCore could dispatch the terminal
  `RequestOutput`.
- `M3-DEFERRED-CLEAN-REDERIVE`: both paged and object-cache M3 hit paths now
  schedule a typed descriptor. `_cleanup_finished` performs the required clean
  N-1 re-prefill after terminal dispatch while the existing admission barrier
  keeps the next turn out until persistence finishes.
- After a real Electron Save & Restart replaced PID 40588 with 42270, the same
  raw video request emitted 40 progressive content deltas, one text-done, one
  completed, and reduced last-content-to-done to 0.0415 seconds. It restored
  1,701 tokens as `paged+disk`.
- A fresh visible Electron chat attached the 14-second/28-frame MP4, used
  Responses / Thinking Off / temperature 0 / tools off, and persisted a
  non-empty answer with 1,701 `paged+disk` tokens, no reasoning, no tool call,
  and no warning. UI Logs showed an L2 reconstruction followed by deferred
  clean rederive and 60-layer typed store.
- M3 remained native `minimax_m3_msa_v1`: dense KV 0-2, sparse MSA/idx_keys
  3-59, generic TQ disabled, zero native-TQ block writes/hits.
- OCR/order quality remains `PARTIAL` (`BANANA8426` -> `BANANA84426`,
  `9753` -> `97553`/`9755`, plus one raw extra line). REAP32 remains excluded
  from live retry because prior attempts rebooted the host.
- Focused validation: 51/51; expanded terminal/M3/media/Responses selection:
  101/101.
- Evidence:
  `docs/internal/release-gates/20260719_m3_terminal_dispatch_large_video/`.

## 2026-07-19 - Path-dependent terminal cleanup generalized

- Status: `FIXED_SOURCE`; `VERIFIED-LIVE_SCOPED` on current-source M3.
  DSV4/ZAYA/mixed-SWA live family rows remain `PARTIAL` until reloaded.
- Source audit found six path-dependent cache branches in
  `_process_batch_responses`: DSV4 paged, ZAYA paged, mixed-SWA paged/object,
  and M3 paged/object. Four direct clean-prefill calls could hold terminal SSE
  and Electron finalization behind a full prompt rederive.
- All six now schedule the shared `_deferred_prompt_cache`; the terminal
  finalizer has zero `_prefill_for_prompt_only_cache` calls. Cleanup performs
  the unchanged typed N-1 rederive after collector dispatch and retains the
  next-turn admission barrier.
- Expanded source/runtime contracts: 229 passed, 6 skipped. This includes the
  full DSV4 paged and ZAYA runtime files; real small ZAYA tests still prove
  clean CCA storage and reuse through synchronous-default Scheduler.step.
- Current-source M3 Save & Restart replaced PID 42270 with 43998. Raw video
  streaming again restored 1,701 `paged+disk` tokens, emitted 40 content
  deltas, and terminaled 0.0414s after the last delta. Electron row 662
  independently returned non-empty visible content with no reasoning/tool/
  warning and the same 1,701-token restore.
- DSV4, Gemma/Step/MiMo mixed-SWA, and available ZAYA artifacts still require
  their own live Electron/API terminal measurements before those family rows
  can be closed.
- Evidence:
  `docs/internal/release-gates/20260719_path_dependent_terminal_cleanup/`.

## 2026-07-19 - ZAYA typed CCA terminal and L2 proof

- `ZAYA-TYPED-CCA-CURRENT`: `VERIFIED-LIVE_SCOPED` on source cutoff
  `1aa5f8e49`. The exercised `Zaya-8B-JANG_4M` bundle is affine JANG, not
  JANGTQ/MXTQ. Health exposed the native `zaya_cca_v1` record with standard
  KV, convolution state, previous hidden state, and no-state MoE slots.
- `ZAYA-GENERIC-TQ-BOUNDARY`: `PASS-SAFE-EXCEPTION`. Generic q4/q8 and
  TurboQuant remain disabled because a KV-only record cannot restore
  path-dependent CCA companion state. Explicit Paged Off is auto-promoted to
  typed paged storage for this family rather than admitting an incomplete SSD
  cache.
- Raw Responses cold/exact-warm/changed-suffix runs emitted 26-39 progressive
  content deltas and completed within 17 ms of the last visible delta. Exact
  warm saved 919 tokens; the changed suffix clean-prefilled instead of using
  a non-terminal typed chain without complete CCA state.
- Real Electron Start eagerly materialized PID 50901 before any request. A
  fresh short UI answer visibly grew to exact `ZAYA-UI-FIRST-DONE`. Real
  `Save & Restart` replaced it with PID 52039, starting with 2,101 SSD tokens
  and zero L1 tokens; Regenerate restored 529/537 as
  `paged+zaya_cca+disk`, promoted nine blocks, and exact-finaled.
- Focused validation: 49 passed, 1 skipped. Raw long strict formatting remains
  `PARTIAL`; media, tools/protocol breadth, cancellation/failure, eviction,
  alternate ZAYA variants, signed app, and the overall release remain open.
- Evidence:
  `docs/internal/release-gates/20260719_zaya_typed_cca_terminal_l2/`.

# 2026-07-19 current active continuation after `0c9436bce`

Overall status: `PARTIAL_NO_RELEASE`. A scoped DSV4 current-source proof now
closes the shared terminal-first lifecycle for that native composite family,
but the following remain explicit release gates:

1. Compatible cache families: prove SSD-only partial-prefix/block reuse with
   Paged Off, then RAM-first -> SSD-refault -> full-prefill fallback with Paged
   On. Include eviction, process restart, accurate counters, q4 native-TQ for
   eligible models, q8 only for Bonsai, and native typed exceptions.
2. Protocol parity: current-source stream/non-stream Chat Completions,
   Responses, Anthropic, and Ollama, including separate reasoning/content,
   automatic/required/no-tool continuations, cancellation, disconnect,
   injected failure, recovery, and follow-up.
3. Electron parity: per-chat overrides and model-derived defaults must match
   DB, preview, argv, health, and actual request behavior. Explicit Off must be
   honored; Auto must resolve from the real bundle/architecture.
4. Gateway/session lifecycle: one-model mode must unload the old process before
   eagerly materializing the selected model; cover repeated swaps, port
   conflicts, LAN rollback, listener recovery, and stale missing-model paths.
5. Required live family rows: mixed-SWA Gemma/Step/MiMo and ZAYA for the shared
   terminal change, then remaining MiniMax M2.7/M3, openPangu, DSV4, Laguna,
   Nemotron, Qwen/JANGTQ, Gemma, LFM, Step, and advertised media/audio routes.
6. Full Python/panel suites, typecheck/build, version/feed truth, signed and
   notarized Sequoia/Tahoe install smoke, then an explicitly authorized public
   release.

Scoped DSV4 evidence:
`docs/internal/release-gates/20260719_dsv4_terminal_dispatch_native_l2/`.
The native `deepseek_v4_v8` path restored exact terminal composite prefixes
from RAM and SSD with progressive Responses/Electron output and sub-0.15-second
raw terminal gaps. Arbitrary changed-suffix partial reuse remains deliberately
rejected because non-terminal blocks lack complete CSA/HCA state. Strict output
fidelity is still partial due intermittent `TERTERMINAL` duplication.

## 2026-07-19 - Gemma mixed-SWA block SSD without paged RAM

- Status: `VERIFIED-LIVE_SCOPED`; overall release remains
  `PARTIAL_NO_1_6_12_RELEASE`.
- The panel now exposes the engine's authoritative mixed-SWA Block Disk L2 route
  when Prefix Cache and Block L2 are on but Paged KV Cache is explicitly off.
  The actual Electron `Save & Restart` argv used `--no-paged-cache`,
  `--kv-cache-quantization none`, and `--enable-block-disk-cache`.
- Cold, exact-warm, changed-tail partial-prefix, and process-restart Electron
  rows returned exact non-empty visible answers with separate progressive
  reasoning. Exact hits restored 1,646 tokens and the changed-tail row restored
  1,536/1,649 tokens as `block-disk+mixed_swa`.
- Before the restart hit, the new process had zero L1 indexed/resident tokens
  and 5,419 L2 tokens on SSD. The first request produced 26 disk promotions,
  zero new writes, zero resident L1 bytes, and zero TQ writes/hits.
- Raw Responses cold/warm/partial runs each emitted 12 progressive content
  deltas, one text-done, one completed terminal, and a 21.5-22.7 ms terminal
  gap. Focused validation passed 7 + 7 Python, 300 panel, typecheck, and diff
  check.
- The real UI was restored to Paged On + Auto afterward; PID 59856 health
  reports paged RAM and stored-prefix `turboquant-q4` for compatible mixed-SWA
  KV storage.
- Step/MiMo and Chat/Anthropic/Ollama repetition remain open. Evidence:
  `docs/internal/release-gates/20260719_gemma_mixed_swa_disk_only_ui/`.

## 2026-07-19 - Step 3.7 mixed-SWA SSD-only bounded proof

- Status: `VERIFIED-LIVE_SCOPED_SHORT_PREFIX`; long tight-memory cold store is
  `PARTIAL`; overall release remains `PARTIAL_NO_1_6_12_RELEASE`.
- The real Step JANGTQ_K bundle has 45 full/sliding attention layers, a
  512-token window, typed `step3p7_full_sliding_kv` records, and no MTP tensors.
  The panel now allows Paged Off + Block L2 for this exact subtype and visibly
  discloses the tight-headroom long-store limitation.
- Gateway single-model Start stopped Gemma PID 59856 and eagerly loaded Step PID
  60732 before a prompt with exactly one engine remaining.
- The real settings drawer launched PID 62212 with `--no-paged-cache`,
  `--kv-cache-quantization none`, and Block L2. A 136-token cold prompt wrote
  three SSD blocks; exact warm restored 135/136; changed-tail partial restored
  one 64-token block; PID 63165 restored 135/136 from disk after restart. L1
  resident bytes and TQ writes/hits stayed zero.
- Raw Responses with the supported low-reasoning rail emitted 102/85 reasoning
  deltas, 9 content deltas per turn, one text-done, one completed terminal, and
  26.5-26.8 ms terminal gaps.
- A 1,343-token cold prompt streamed an exact answer but intentionally wrote no
  cache: the tight-Metal guard rejected a second clean prefill to avoid OOM.
  No unsafe force override was used; long cold store remains open.
- Paged On + Auto was restored through the real UI; PID 64768 health reports
  paged RAM and stored-prefix TQ4. Focused checks: 14 Python, 300 panel,
  typecheck, and diff check. Evidence:
  `docs/internal/release-gates/20260719_step37_mixed_swa_disk_only_ui/`.

## 2026-07-19 - openPangu current-head exact memory/SSD restart recheck

- Status: `VERIFIED-LIVE_SCOPED` for exact prompt-memory and first-turn
  prompt-SSD restore on source `117c3d206`; overall release remains
  `PARTIAL_NO_RELEASE`.
- The real Electron Sessions Start action stopped Step and eagerly loaded only
  openPangu PID 65893 before a request (`last_request_time=null`). The actual
  argv used `--no-paged-cache --enable-disk-cache`; health kept generic TQ and
  block-disk L2 off for native schema `openpangu_v2_composite_v2`.
- Cold Electron stored an 817-token typed boundary. A fresh exact chat restored
  817 tokens from memory. After a real UI Stop/Start replaced the process with
  PID 66691, pre-request health had zero memory/L1 entries and 6,502 prompt-L2
  tokens; the first identical UI turn restored 817 tokens as `disk`, exact-
  finaled, and streamed distinct reasoning plus visible content progressively.
- Raw Responses emitted separate progressive reasoning/content and clean
  terminals for cold/exact/forward-prefix requests. The forward request hit
  592 memory tokens but produced both A and B markers after exhausting 512
  reasoning tokens, so strict B-only output remains `PARTIAL`.
- Generic paged blocks, arbitrary block partial reuse, and generic TurboQuant
  are architecture-incompatible/N/A for openPangu's MLA/DSA/SWA/causal-conv
  composite and must be proven on compatible families instead.
- Evidence:
  `docs/internal/release-gates/20260719_openpangu_current_disk_restore/`.

## 2026-07-19 - Laguna eager Start checkpoint

- Status: `VERIFIED-LIVE_SCOPED` for eager Start/materialization on current head
  `7d48071e2`; Laguna performance/reliability and overall release remain
  `PARTIAL`.
- The real Electron Sessions Start control stopped the prior LFM process and
  launched only `Laguna-M.1-JANG_2L` PID 70292. Before any request, health
  reported `model_loaded=true`, `last_request_time=null`, and 82,631.3 MB
  active memory. The process argv selected `glm47`, qwen3 reasoning, Paged On,
  Block L2 On, and Auto q4 TurboQuant storage.
- The first fresh Electron turn persisted 1,524 separate reasoning characters
  and a coherent two-sentence visible answer ending `LAGUNA-EAGER-DONE`; there
  was no tool call or warning. The attempted DOM paint observer exited without
  saving samples, so this row does not replace the existing current Laguna raw
  SSE/Electron progressive-paint evidence and makes no new streaming-timing
  claim.
- Evidence: `docs/internal/release-gates/20260719_laguna_eager_current/`.
- Retain `OPEN`: natural decode is still about 24 tok/s, restart cache TTFT is
  slower than warm RAM, long-agent reliability/strict formatting remain
  partial, and other deferred loader routes still need eager proof.

## 2026-07-19 - eager Start evidence reconciliation

- Status: `VERIFIED-LIVE_SCOPED` for the already-recorded DSV4, Laguna, Step,
  openPangu, Gemma mixed-SWA, HY3 native-MTP D1, and MiniMax M2.7 JANGTQ
  routes; remaining loader classes, repeated sleep/wake swaps, and signed-app
  repetition remain `PARTIAL`/`OPEN`.
- This is a documentation reconciliation, not a new runtime run. Each named
  gate used the real Electron Start or Stop/Start control and captured health
  before the first request with the model loaded and no prior request. The
  corresponding evidence directories are now linked from the master matrix's
  eager-materialization row so these routes are not repeatedly retested.
- No claim is generalized to an untested loader class. In particular, Mistral
  MXFP4 remains excluded by user directive, and Mistral JANGTQ2 remains on its
  separately documented blocked runtime row.

## 2026-07-19 - stale local model-path recovery UI

- Status: `VERIFIED-LIVE_SCOPED` for current filesystem classification, visible
  no-Start recovery actions, native-chooser repoint, and explicit removal;
  signed-app repetition remains open.
- A disposable missing-path session appeared in the real Electron dashboard
  under `MISSING MODEL (1)` with an unavailable-directory warning, usable-twin
  hint, and only `Repoint model path` / `Remove session` actions. The rendered
  Remove action deleted the fixture from UI and SQLite without stopping or
  altering active Laguna PID 70292.
- Source ownership is explicit classification without silent mutation, bundle
  validation, identity-change confirmation, duplicate/running-session guards,
  and transactional chat rebinding. Focused panel validation passes 8/8 plus
  typecheck.
- The complementary real Repoint action opened the native directory chooser,
  updated the fixture to `/private/tmp/vmlx-repoint-live/Laguna-M.1-JANG_2L`,
  and moved the card from `MISSING MODEL` to normal `INACTIVE` Start/Delete.
  The disposable session and directory were removed afterward; active Laguna
  PID 70292 remained unchanged.
- Evidence:
  `docs/internal/release-gates/20260719_stale_path_recovery_live/`.

## 2026-07-19 - Laguna soft-sleep/wake UI soak

- Status: `VERIFIED-LIVE_SCOPED` for three consecutive soft-sleep/Wake cycles;
  deep sleep and cross-model swap soak are now covered by their later gates,
  while signed-app repetition remains open.
- The real Electron moon and Wake controls drove all six transitions. Every
  wake reached DB `running` plus health `healthy/model_loaded=true`; every
  sleep returned DB to `standby/soft` plus health
  `standby_soft/model_loaded=true`.
- PID 70292 was unchanged throughout, and the final process list contained
  exactly one engine. Logs preserve three engine/app Wake pairs and three
  soft-sleep pairs. No inference ran, so this row makes no streaming or cache
  reuse claim.
- Evidence:
  `docs/internal/release-gates/20260719_laguna_soft_sleep_soak/`.

## 2026-07-19 - Laguna deep-sleep unload and Wake

- Status: `VERIFIED-LIVE_SCOPED` for real UI policy configuration, automatic
  deep sleep, visible deep-standby state, in-process Wake reload, and default
  restoration; cross-model swaps are now covered by the later one-model soak,
  while signed-app repetition remains open.
- The session settings UI set Light/Deep to `0`/`1`; the idle Laguna session
  automatically entered `standby/deep`. Health reported
  `standby_deep/model_loaded=false`, while the same PID 70292 and its listening
  process remained alive. Electron visibly rendered `Deep Sleep` plus Wake.
- Wake reloaded the model without replacing PID 70292. The settings UI then
  restored `10`/`30`, and final state returned to
  `standby/soft/model_loaded=true` with exactly one engine process.
- No generation ran, so this row makes no streaming/cache-reuse claim.
- Evidence:
  `docs/internal/release-gates/20260719_laguna_deep_sleep_ui/`.

## 2026-07-19 - repeated one-model Electron Start swap soak

- Status: `VERIFIED-LIVE_SCOPED` for two MiniMax M2.7/Laguna round trips using
  the real Electron Start controls; signed-app repetition remains open.
- With gateway single-model mode live, the PID sequence was
  `70292 -> 78868 -> 79430 -> 80033 -> 80479`. Every Start stopped the prior
  session and endpoint before the replacement became active; SQLite and `ps`
  showed exactly one local engine after every transition.
- Both model loads completed before any request. Health reported
  `model_loaded=true` and `last_request_time=null`; M2.7 and Laguna argv/parser/
  cache/quant identities matched their distinct JANGTQ2 and affine JANG routes.
- The Electron main log preserved the venv engine PATH line and all four
  stop-before-start transitions. Final state was restored through the real moon
  control to Laguna `standby/soft`, PID 80479, with M2.7 stopped.
- No generation ran, so this row makes no streaming, tool, cache-hit, or L2
  restore claim.
- Evidence:
  `docs/internal/release-gates/20260719_one_model_swap_soak/`.

## 2026-07-19 - gateway downstream-disconnect cleanup and recovery

- Status: `VERIFIED-LIVE_SCOPED` for downstream client disconnects through the
  live Electron gateway on streaming and non-stream Chat Completions,
  Anthropic Messages, Ollama chat, and Ollama generate; signed-app repetition,
  safe injected engine failures, stable agentic tool/result continuation, and
  broader model/parser coverage remain `PARTIAL`/`OPEN`.
- Source trace found two owning defects. The gateway watched the consumed
  request body and installed its response-close listener only after upstream
  headers, so a non-stream client could disappear while headerless inference
  remained active. The Anthropic non-stream adapter also internally consumed
  `stream_chat_completion`, leaving no Starlette streaming response to own the
  disconnect receive channel.
- `panel/src/main/api-gateway.ts` now binds downstream response closure before
  proxy headers and destroys the upstream request when the downstream socket
  closes early. `vmlx_engine/server.py` now uses a bounded active receive drain
  only for non-stream adapter consumption, aborting the exact response while
  leaving normal stream ownership unchanged.
- Before the fix, non-stream Chat remained at `num_running=1` for every sample
  across 10 seconds after the client timeout. After the gateway repair, Chat,
  Ollama chat, and Ollama generate returned idle in 0.037/0.031/0.029 seconds.
  Anthropic initially retained the orphan, independently proving the adapter
  defect; after its repair it returned idle in 0.034 seconds.
- Current-source recovery requests completed exactly on all four surfaces with
  one truthful terminal and usage. Streaming aborts emitted progressive partial
  bytes but no false terminal, returned idle in 0.030-0.034 seconds, and were
  followed by exact completed recovery streams.
- The real Electron app was reloaded against the patched main process and used
  its Save & Restart control to replace Laguna PID 80479 with PID 88506. A
  visible recovery turn progressively grew
  `UI-GATEWAY-DISCONNECT-FIX-` -> `...-O` -> `...-OK` before terminal metrics;
  SQLite preserved exact visible content, null reasoning/warnings, and the DOM
  observer recorded 46 mutations.
- Focused validation passes: 5 selected Python disconnect/stream tests, 63
  Anthropic adapter tests, 77 panel gateway tests, panel typecheck, and
  `git diff --check`.
- Evidence:
  `docs/internal/release-gates/20260719_gateway_disconnect_recovery/`.

## 2026-07-19 - vMLX 1.6.12 signed public release checkpoint

- Status: `PUBLIC-RELEASED VERIFIED-LIVE` for this exact checkpoint. The
  checkpoint does not close the deferred family/stress rows below.
- Runtime/source checkpoint:
  `6de1096eca0ea2d5516ad64d6e79da98f3ae20a2`.
- Complete validation on that source: 6,186 Python tests passed / 185 skipped;
  2,332 panel tests passed / 3 skipped; TypeScript typecheck passed; bundled
  engine 1.6.12 and clean JANG 2.5.31 compatibility passed.
- Fresh production artifacts were built for Sequoia and Tahoe, signed by
  `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`, freshly notarized,
  stapled, and accepted by Gatekeeper. Notary IDs are
  `8b4a213b-a856-4659-8aa9-146ba211c163` (Sequoia) and
  `4fb3b188-5c57-4eb2-a909-85a917ee31b4` (Tahoe), both Accepted.
- Exact installed-app proof used each real Start and Stop control. Sequoia
  completed three Electron turns with separate reasoning, exactly one real
  `file_info` tool, post-tool continuation, cross-turn recall, raw Responses
  and Chat progressive streams, and a live 100-token
  `paged+mixed_swa+tq-native` cache hit. Tahoe independently completed a
  reasoning/content UI turn and a raw Responses stream with 147 reasoning
  deltas, 17 content deltas, one completed terminal, and usage.
- Artifact SHA-256: Sequoia
  `704d87edf168a73d4ca2d94e8cb6190ca593ada71bca181bf369c84ea13ae421`;
  Tahoe
  `81b9205a722282cc1eec75713c18dec3efc34ed76e3bcaf6587147e0ce372c49`.
- Canonical record:
  `docs/internal/release-gates/20260719_release_checkpoint_1_6_12/README.md`.
- Public source and DMG releases are
  `https://github.com/jjang-ai/vmlx/releases/tag/v1.6.12` and
  `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.12`. PyPI,
  Homebrew, both GitHub updater manifests, and the `mlx.studio` edge feed all
  publicly report 1.6.12 with their recorded hashes.
- The standard GitHub PyPI workflow still needs its trusted-publisher mapping
  repaired (`invalid-publisher`); this release used the existing authenticated
  publisher credential on the trusted live-model Mac after building/checking
  the exact tagged artifacts.
- Explicitly retained after publication: broad signed-app family repetition,
  remaining parser-family tool rows, safe injected mid-stream failure, long
  stochastic/latency/media soaks, remaining accessibility/modal breadth,
  openPangu 512K work, DSV4 controlled reference A/B, and the broader gateway/
  swap soak. These stay `PARTIAL`/`OPEN`; release packaging does not promote
  them.

## 2026-07-19 - post-release mid-stream engine-failure recovery

- Status: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED` on pushed source
  `5f05ad72a`; public v1.6.12 remains sealed and does not contain this
  post-release fix.
- Root cause was shared Electron SSE ownership, not a model family. Chat and
  Responses already emitted progressive partial text and authoritative failed
  terminals/usage, but `panel/src/main/ipc/chat.ts` threw on the first error
  event and cancelled the reader before terminal usage arrived.
- The panel now defers ordinary server errors until the stream terminal,
  recognizes `response.failed`, extracts nested failed-response errors, and
  preserves immediate handling for expected backend disconnects.
- Literal curl-N production-stream-function probes passed Chat and Responses
  failure/recovery ordering. The real Electron dev app visibly painted
  `RESP-PARTIAL-` / `CHAT-PARTIAL-` before failure, persisted exact partial
  content with 2 output and 5 prompt tokens, showed the failure, and completed
  immediate same-chat recoveries exactly. Recovery history retained the safe
  partial prefix but stripped the UI-only interruption marker.
- Complete validation: 6,185 Python passed / 95 skipped / 92 deselected using
  the released clean JANG 2.5.31 source; 2,333 panel passed / 3 skipped;
  typecheck and Electron main/preload/renderer production build passed.
- This closes safe injected engine failure for dev Electron plus raw
  Chat/Responses. Gateway network-loss injection, Anthropic/Ollama injected
  failures, signed-app repetition, and unrelated family/stress rows remain
  open. Evidence:
  `docs/internal/release-gates/20260719_midstream_failure_recovery/`.
- Proof/ledger commit `1cc329c05` was pushed by fast-forward to `main`,
  `codex/live-electron-gates-20260715`, and the post-release branch. A final
  public-surface audit revalidated the unchanged v1.6.12 tag target, both DMG
  hashes/staples/Gatekeeper results, both installed-app signatures, all updater
  feeds, PyPI, and the Homebrew cask; see the release-checkpoint README.

## 2026-07-20 - Anthropic/Ollama injected mid-stream failure closure

- Status: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED` at pushed source `d811270ad`;
  public v1.6.12 is unchanged and does not contain this post-release fix.
- The Anthropic adapter already emitted a native error and suppressed normal
  finalization. Ollama's chat, templated-generate, and raw-generate converters
  instead dropped the structured error; their route wrappers then converted
  the later `[DONE]` into a false successful `done:true` row.
- All Ollama converters now emit the official native terminal
  `{"error":"..."}` row. Route state clears deferred done/tool state and
  suppresses later success/usage synthesis after an error.
- Literal curl-N failure/recovery pairs through the production handlers proved
  two progressive visible chunks before every failure, no false success
  terminal, and immediate normal completion on Anthropic plus all three Ollama
  streaming rails.
- Validation: 30 adapter/route tests, two existing Chat/Responses mid-stream
  selections, one Ollama route-cap selection, py_compile, and diff check pass.
- Evidence:
  `docs/internal/release-gates/20260720_anthropic_ollama_midstream_failure/`.
- Still open: gateway network-loss injection, signed-app repetition, broader
  parser/model agentic loops, non-stream pre-header failures, and unrelated
  family/cache/media/stress rows.

## 2026-07-20 - v1.6.13 packaged checkpoint proof

- Status before publication: `SIGNED_NOTARIZED_VERIFIED_LIVE_PUBLICATION_PENDING`
  at pushed source `5fae65d38`.
- The initial installed Sequoia candidate exposed a real packaged-runtime bug:
  launch cwd `/Users/eric` placed `/Users/eric/mlx` on Python's path beside the
  bundled `mlx` package. `5fae65d38` now launches bundled Python with
  `PYTHONSAFEPATH=1` and a stable bundled-binary cwd; focused and full panel
  tests pass after the fix.
- Both corrected DMGs were rebuilt, Developer-ID signed, freshly notarized,
  stapled, independently verified, transferred byte-for-byte to
  `erics-m5-max.local`, installed separately, and accepted by Gatekeeper.
- Final Sequoia Electron proof includes clean reasoning/content, one real
  `file_info` call/result, multi-turn recall, progressive Responses/Chat SSE,
  q4-native L2 writes, and a process-restart 3,359-token
  `paged+mixed_swa+disk+tq-native` restore with 53 disk/q4-native hits.
- Final Tahoe proof includes real Start/load/generation/Stop plus raw Chat SSE
  with separate reasoning/content, stop, usage, and DONE.
- Honest retained observations: the 512-token Responses control correctly
  truncated; Gemma strict marker-only behavior is partial because completed
  rows can add coherent explanatory prose. No output rewrite or hidden sampler
  coercion was added.
- Canonical evidence:
  `docs/internal/release-gates/20260720_release_checkpoint_1_6_13/`.
- Broader model/parser/cache/media/gateway/stress rows remain exactly as OPEN or
  PARTIAL in the master matrix; packaging does not promote them.

## 2026-07-20 - v1.6.13 public checkpoint released

- Status: `PUBLIC_CHECKPOINT_RELEASED_BROADER_MATRIX_PARTIAL`.
- Source release `jjang-ai/vmlx@v1.6.13` peels to
  `2f509f79d7829119308a36a02f13fd590dd2010e`. The public
  `jjang-ai/mlxstudio@v1.6.13` release contains both signed/notarized DMGs and
  blockmaps with GitHub-reported digests matching the built artifacts.
- Updater repo `main` and its lightweight `v1.6.13` tag resolve to
  `07c402d426f125e1ded175b34d52a16e3769dd8a`. Both GitHub manifests and the
  `mlx.studio` origin serve version 1.6.13 with the exact Sequoia/Tahoe hashes.
- PyPI serves `vmlx==1.6.13` with wheel SHA-256 `363e5e3e...a2100` and sdist
  SHA-256 `bbd2141b...1f24`. Homebrew main `0b0f54c` serves cask 1.6.13 with
  the exact Sequoia hash and passes `brew style`.
- Publication, public re-read, and source synchronization were driven from
  `erics-m5-max.local`. Its older dirty checkout was not overwritten; a clean
  release worktree was created at the exact tag.
- Gemma output verdict remains scoped: corrected Sequoia turns are coherent,
  separately parsed, and include a real one-call tool loop; the earlier raw
  tool-markup/`221 bytes` row is excluded. Tahoe is coherent but adds prose
  before an exact marker, so strict-format reliability remains PARTIAL.
- Operational follow-up: repair the GitHub/PyPI trusted-publisher OIDC mapping.
  Broader family/parser/cache/media/gateway/stress rows remain PARTIAL/OPEN and
  resume only after the requested release pause.

## 2026-07-20 - LFM2.5 MXFP4 native reasoning / required-tool protocol

- Status: `PARTIAL_LIVE` on the post-v1.6.13 remote release checkout.
- The real bundle is base MLX MXFP4, not affine JANG or JANGTQ/MXTQ. Its native
  template has no `enable_thinking` branch and the bundle README explicitly
  forbids synthetic `<think>` prefill. Current source removes the old LFM-only
  sentinel, exposes Auto/On, and rejects explicit Off across protocol surfaces.
- A fresh Electron Auto/no-tool turn streamed a separate reasoning rail and
  progressive visible content, then exact-finaled
  `LFM-UI-AUTO-CURRENT-DONE`.
- The required `file_info(panel/package.json)` Electron turn remains red for
  this artifact: parsed arguments were `{"path":": "}`, execution failed, faux
  tool JSON leaked into content, and the prior marker replayed. The older
  JANG_2L proof remains scoped to that separate artifact.
- The shared Responses bug is fixed: an unmet required-tool contract now emits
  `tool_calls_required` and terminates only as `response.failed`; it is not
  stored as successful history. Current raw API proof reproduces the model miss
  and the truthful failed terminal.
- Follow-up gateway source now recognizes FastAPI's `detail` error field.
  After a full current-source Electron relaunch and real UI Start of LFM PID
  26730, all eight direct/gateway Chat, Responses, Anthropic, and Ollama rows
  preserved status 400 and the complete model-family incompatibility message.
  Validation: 53 gateway tests and TypeScript typecheck. This gateway subrow is
  `VERIFIED-LIVE_SCOPED`.
- Validation: 14 selected Python tests, 88 panel tests, TypeScript typecheck,
  and diff check. Evidence:
  `docs/internal/release-gates/20260720_lfm_native_reasoning_protocol/`.
## 2026-07-20 - Gemma 4 media-order, 1120-budget, mixed-SWA L2, and fallback recheck

- Current host/repo: `erics-m5-max.local`,
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`, branch
  `codex/postrelease-ui-drawers-20260720`.
- Bundle identity is grounded in the real files: affine `JANG_4M`, not
  JANGTQ/MXTQ; `gemma4_unified`; 40 rotating-SWA plus eight full-attention
  layers; vision+audio advertised; video not advertised.
- `7687f237b` fixes the Electron composer contract for Gemma only: visual media
  precedes text and audio follows it. The pre-fix `[text,image]` rows looped or
  mangled output even with cache disabled. The corrected request log shows
  `[image_url,text]`.
- `a0abd7ab3` adds explicit validated Gemma image budgets
  70/140/280/560/1120 across Chat, Responses, batching, UI persistence, and
  media-cache identity. The real UI selected 1120 and sent it; prompt size rose
  from 328 to 1,144 tokens.
- Structural image transport is current-source PASS-LIVE, but exact OCR is
  **PARTIAL**: Electron A5 returned `jiang-ai/...`; raw A6/A7/A8 also changed
  one or more characters/case from
  `jangq-ai/gemma-4-12B-it-qat-JANG_4M`. Do not call that output proper and do
  not hide it with postprocessing.
- Raw Responses A6 emitted progressive reasoning from 1.44s, progressive
  content from 7.40s, and a completed terminal at 8.28s. Identical warm reuse
  restored 1,137/1,138 tokens as `paged+mixed_swa+tq-native` and began at
  0.32s.
- Visible Electron Stop/Start cleared L1. A7 restored 1,137 tokens as
  `paged+mixed_swa+disk+tq-native`; health recorded 18 disk promotions plus 18
  native-TQ hits, with the 40 rotating layers kept native.
- The first pre-fix restart row exposed a separate P0 fallback bug: after the
  512-token reasoning rail, the direct answer pass emitted literal `thought`
  and ended incomplete. Cache restoration was numerically coherent; the server
  prompt/stream layer was wrong.
- Root cause and fix (`1b89e1118`): buffer the degraded Gemma `thought\n`
  channel from its first partial chunk, and rerun the fallback from original
  context because Gemma's real template ignores a bare assistant
  `reasoning_content` turn. Forced A8 at a 64-token thinking cap emitted 57
  reasoning events followed by 30 progressive content events, no `thought`
  leak, and `response.completed`.
- Validation: dedicated family test 9/9; expanded server/Gemma/reasoning/media
  selection 57 passed with two intentional skips. Evidence:
  `docs/internal/release-gates/20260720_gemma4_media_stream_cache/`.
- Remaining: controlled same-artifact OCR reference A/B, different-media
  1120-budget salt isolation and return-A, audio, post-media text/tool turns,
  bounded eviction, and signed-app repetition.
- Follow-up media-salt proof is now `PASS-LIVE_SCOPED`: same-size screenshot A
  had two active cards and B had one; identical-prompt A/B/A returned
  `2`/`1`/`2`. B had no cached-token claim. Return-A restored 1,097/1,098 tokens
  as `paged+mixed_swa+tq-native`; after a real Electron Stop/Start it restored
  the same 1,097 tokens as `paged+mixed_swa+disk+tq-native`, with 18 disk
  promotions/native-TQ hits and no writes.
- Follow-up post-media history/tool proof is `PASS-LIVE_SCOPED`: the same
  Electron chat recalled the prior image marker on a text-only turn. After the
  real Chat Settings UI enabled built-in tools, Gemma emitted exactly one
  `file_info(panel/package.json)` call, consumed the 5.2 KB result, and
  exact-finaled with no warning.
- Remaining Gemma rows are now the controlled OCR reference A/B, advertised
  audio, non-advertised-video rejection, bounded eviction, and signed-app
  repetition. Exact OCR remains `PARTIAL`; these cache/history passes do not
  promote it.

## 2026-07-20 - Nemotron Omni audio media-salt and Electron history replay

- Current host/repo: `erics-m5-max.local`,
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`, branch
  `codex/postrelease-ui-drawers-20260720`.
- Bundle files classify the artifact as JANGTQ/MXTQ Hadamard-codebook, not
  affine JANG and not base MLX MXFP. `config_omni.json` advertises Parakeet
  16-kHz audio, C-RADIO vision, and video tokens.
- `NEMO-OMNI-MEDIA-SALT`: `VERIFIED-LIVE_SCOPED`. Root cause was text-only
  conversation hashing in `vmlx_engine/omni_multimodal.py`. An identical-text
  orange->blue replay returned stale `MARKER=ORANGE-4729`. Current source adds
  media identity to every user-turn signature, resets on a mismatched prefix,
  and rehydrates the latest historical media for a text-only follow-up. The
  post-fix blue replay exact-finaled `MARKER=BLUE-6813` with progressive
  content, stop, usage, and DONE.
- `NEMO-OMNI-PANEL-HISTORY`: `VERIFIED-LIVE_SCOPED`. The shared Electron path
  previously stripped all historical media, turning the next request into
  `chatIsMultimodal:false` and bypassing Omni. Current source preserves prior
  media only when family detection says `nemotron-h` and the selected bundle
  has `config_omni.json`. A clean no-attachment second turn logged
  `preserveHistoricalMediaForOmni:true`, retained `input_audio`, reached the
  Omni dispatcher, and continued a matching persistent conversation prefix.
- Clean Electron chat outputs were `READY`, `blue6813`, and exact
  `BLUE-6813`; separate reasoning lengths were 445/367/354 and hashes were all
  different. The turn-2 missing hyphen is retained as a strict-format miss,
  not rewritten or hidden.
- Raw Chat audio emitted 146 reasoning plus 24 content deltas, terminal stop,
  usage, and DONE. Raw Responses emitted 160 reasoning-summary plus 26 text
  deltas, matching done events, one completed item, and one completed response
  with usage.
- Real Electron Stop/relaunch plus UI Start proved eager load before the first
  request (`last_request_time=null`, 9,348.1 MB active) and retained the exact
  venv `[Engine Manager] Found in PATH` log.
- Validation: 25 focused Python tests; full panel 77 files / 2,346 passed / 3
  skipped; typecheck. Full Python ran 6,202 pass / 96 skip / 92 deselect and
  intentionally failed only because the bundle still had the pre-fix source.
  `bundle-python.sh` was then run against the clean detached JANG checkout;
  the complete bundled-runtime verifier and the formerly failing test pass.
- `NEMO-OMNI-MEDIA-L2`: `OPEN` at this historical source checkpoint. The Omni
  dispatcher's persistent KV+SSM conversation was process-local. Do not use
  ordinary scheduler paged/TQ/L2 counters to claim process-restart
  media-session restore. The later post-v1.6.14 closure is recorded below.
- Evidence:
  `docs/internal/release-gates/20260720_nemotron_omni_audio/`.

## 2026-07-20 - v1.6.14 public signed checkpoint

- Status: `PUBLIC_CHECKPOINT_RELEASED_BROADER_MATRIX_PARTIAL`.
- Exact source: annotated tag object
  `420b3d91c54e5626164ea49faf7ee6783641df53` peels to
  `e1776a485e8a85f3957b79030e12f4c312eda04b`. Source release:
  `https://github.com/jjang-ai/vmlx/releases/tag/v1.6.14`.
- Complete current-source validation: 6,203 Python passed / 96 skipped / 92
  deselected, 77 panel files / 2,346 passed / 3 skipped, TypeScript typecheck,
  Electron production compile, and bundled engine 1.6.14 + clean JANG 2.5.31
  verification.
- Both final DMGs were built from the exact source, Developer ID signed by
  ShieldStack LLC (`55KGF2S5AY`), Apple notarized, stapled, Gatekeeper
  accepted, and installed separately on `erics-m5-max.local`.
- Final SHA-256:
  - Sequoia: `345fd1ec02bf039b4a113bc617c5fa4eca7c057577a100212e3587dd1bc8022c`;
  - Tahoe: `d77b49ede22d47f7cc2ebb3f3ecfe1b4425f92c05c20eff7be9d2ab6c97a739d`.
- Sequoia signed-app proof: real Electron Launch Session loaded the affine
  Gemma 4 JANG_4M bundle; two distinct prompts produced separate reasoning
  and non-empty content, multi-turn recall, one exact real
  `file_info(panel/package.json)` call/result/final loop, and raw Responses
  streaming with 337 reasoning plus 50 content deltas and a completed terminal.
- Tahoe signed-app proof: fresh launch plus literal UI Stop/Start loaded the
  installed Tahoe engine, restored 73 tokens across process/profile/variant as
  `paged+mixed_swa+disk+tq-native` with 0.29 s TTFT, and raw Chat emitted 185
  reasoning plus 18 content deltas, stop, and DONE.
- The tools-Off raw-markup row and tools-On/unset-working-directory error row
  are retained but excluded from the successful tool verdict. Only the later
  one-call/5.2-KB/exact-final row counts as PASS.
- Public DMG release:
  `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.14`. GitHub reports
  all four DMG/blockmap sizes and hashes equal to the final artifacts.
- PyPI serves the exact 1.6.14 wheel/sdist; Homebrew commit `47a691a2`
  serves cask 1.6.14; both GitHub manifests and `mlx.studio` are byte-identical
  at SHA-256 `e19da155...d79c6a7c` and carry both exact platform hashes.
- Honest retained partials: Gemma UI2 progressively completed but overthought
  for 72.8 s / 2,782 tokens; strict marker-only behavior and exact OCR remain
  PARTIAL; Nemotron Omni process-restart/L2 media state remains OPEN. No hidden
  sampler clamp, output rewrite, or fake cache/model behavior was added.
- Canonical source/live/public evidence:
  `docs/internal/release-gates/20260720_release_checkpoint_1_6_14/`.
- This checkpoint closes publication/signing for these exact artifacts only.
  Every broader family/protocol/cache/media/gateway/stress row retains its
  existing PASS/PARTIAL/OPEN status in the master matrix.

## 2026-07-20 - Nemotron Omni q4-KV/native-SSM process-restart session L2

- Current host/repo: `erics-m5-max.local`,
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`, branch
  `codex/postrelease-ui-drawers-20260720`.
- Artifact truth remains JANGTQ/MXTQ Hadamard-codebook (`profile=JANGTQ2`),
  not affine JANG and not base MLX MXFP. Runtime persistence quantizes only
  attention KV to q4 and retains Mamba/SSM `ArraysCache` natively.
- `NEMO-OMNI-MEDIA-L2`: `VERIFIED-LIVE_SCOPED` on current post-v1.6.14
  source. `vmlx_engine/omni_multimodal.py` now fingerprints the exact bundle,
  binds the snapshot to the exact conversation/media prefix, atomically saves
  the architecture-owned mixed cache, and rejects schema/fingerprint/prefix/
  topology mismatches. `vmlx_engine/server.py` forwards the effective loaded
  Block Disk Cache toggle and exposes architecture-specific health.
- Exact-current-source Electron proof: seed PID 82562 exact-finaled `SEEDED`;
  real UI Stop/Start produced fresh PID 82724; the post-restart turn painted
  progressive separate reasoning and exact `FIR-9928`. The DB row is nonempty
  with no warning/tool; health records `hits=1`, 0.000317 s restore, q4/native
  codecs, and no error.
- Exact-current-source raw Chat proof after another UI restart: 129 reasoning
  deltas, 13 content deltas, exact `blue6813 FIR-9928`, one stop, one usage,
  and one DONE. Health again recorded a real architecture-session hit.
- Explicit-Off negative control: the real UI unchecked L2 and applied Save &
  Restart. Health reported `enabled=false` while the file remained present;
  exact `OFF-PATH-ACTIVE` completed with hits/stores still zero. The same UI
  restored L2 On and restarted; final screenshots/health preserve On.
- Validation: expanded Omni/multimodal 45 passed / 563 deselected; complete
  `tests/test_server.py` 119 passed / 3 deselected; diff check passed.
- Honest remaining boundary: latest exact-prefix snapshot only. Multi-snapshot
  LRU, architecture-file partial-prefix reuse, bounded eviction, image/video
  restart controls, Responses/Anthropic restart repetition, and signed-app
  repetition remain OPEN. Public v1.6.14 predates this fix.
- Evidence:
  `docs/internal/release-gates/20260720_nemotron_omni_session_l2/`.

## 2026-07-20 - Gemma 4 Unified direct-audio mask/cache closure

- Host/repo: `erics-m5-max.local`,
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`, branch
  `codex/postrelease-ui-drawers-20260720`.
- Artifact truth: `gemma4_unified`, affine `JANG_4M`, encoder-free
  `embed_audio` projection, 40 rotating-SWA plus eight full-attention layers,
  audio+vision, no advertised video. It is not JANGTQ/MXTQ or base MXFP.
- `GEMMA4-DIRECT-AUDIO-MASK`: `VERIFIED-LIVE_SCOPED`. The continuous-batching
  wrapper incorrectly forwarded the processor's 2-D padding mask into Gemma's
  causal language attention. Same-artifact logits differed from direct
  mlx-vlm by max `18.86328125`; omitting that mask was bit-identical (`0.0`).
  Current source drops only ordinary 2-D-or-lower processor masks and preserves
  explicit higher-rank attention masks.
- `GEMMA4-AUDIO-UI`: `VERIFIED-LIVE_SCOPED` with Thinking Off. The real
  Electron UI visibly attached a WAV and emitted a non-empty exact-content
  transcription in 26 tokens, 0.48 s TTFT, and 1.0 s total, with no reasoning
  rail, loop, warning, tool markup, or truncation.
- `GEMMA4-AUDIO-STREAM`: `VERIFIED-LIVE_SCOPED`. Raw Responses under explicit
  TQ None and restored Auto q4 each emitted 21 progressive content deltas,
  output-text done, completed terminal, and the exact lowercase transcript.
- `GEMMA4-AUDIO-CACHE`: `VERIFIED-LIVE_SCOPED`. Resident reuse restored
  218/219 tokens as `paged+mixed_swa+tq-native`. After real Electron Save &
  Restart, the first request restored 218/219 as
  `paged+mixed_swa+disk+tq-native`; health recorded four disk hits and four
  native-TQ hits with no writes.
- `GEMMA4-AUTO-AUDIO-QUALITY`: `PARTIAL`. The fix removed the pre-fix repeated
  `0.02e+19` collapse, but Auto-thinking UI turns overthought for 2,896 and
  2,271 tokens; the second said no audio was attached. Do not hide this with
  forced Thinking Off, output rewriting, prompt coercion, or sampler clamps.
- Validation: capability selection 7/7; Gemma/audio scheduler selection 9/9;
  diff check passed. Evidence:
  `docs/internal/release-gates/20260720_gemma4_audio_mask_cache/`.

## 2026-07-20/21 - model-derived generation-settings UI parity

- `CHAT-SAMPLING-DEFAULTS-PARITY`: `VERIFIED-LIVE-SCOPED / BROADER MATRIX
  PARTIAL`. The affine Gemma JANG route now has current-source end-to-end proof
  from exact bundle metadata (`temperature=1.0`, `top_p=0.95`, `top_k=64`) to
  persisted session detection, visible Chat Settings, outgoing Responses body,
  and resolved engine kwargs.
- Root cause for the explicit-Off failure was shared request serialization:
  Chat/Responses and both Ollama translations dropped `top_k=0`, so the engine
  correctly re-inherited bundle `64`. Current source preserves zero. A real
  Electron Save produced a request with `"top_k":0` and engine kwargs without
  top-k; exact visible output completed with no warning. Raw Responses, Chat,
  and Ollama each streamed nine content deltas and a valid terminal.
- Real Reset now clears sampler fields to SQL `NULL` rather than storing a
  stale copy. The drawer visibly returned to Top K 64 and the next request
  omitted the override; engine kwargs resolved `top_k:64` from the bundle.
- Removed the engine-only Ling/Bailing top-k 20 fallback. Undeclared sampling
  now remains neutral unless supplied by request, startup/session CLI, or
  bundle metadata. No Ling/Bailing artifact exists in the active roots, so this
  sub-row is `PASS-SOURCE`, not live.
- Focused validation: panel 519/519 plus typecheck; engine sampling audit 51/51;
  generation-default matrix pass (28 panel + 61 engine + 6 live-path metadata
  audit + 22 startup contracts). Evidence:
  `docs/internal/release-gates/20260720_sampling_defaults_ui_runtime/`.
- Remaining: repeat this exact visual/payload/runtime chain on JANGTQ/MXTQ,
  base MLX/MXFP, DSV4/M3 native typed routes, and a bundle whose repetition
  penalty is non-neutral. Do not promote the broad matrix from this Gemma row.
- `ENGINE-VERSION-TRUTH`: `OPEN`. The restarted v1.6.14 Electron source log
  reported the PATH `vmlx-engine` version as 1.6.12. Trace package/source/bundle
  version surfaces before the next checkpoint; do not infer runtime source from
  the displayed version alone.

## 2026-07-20 - current-source DSV4/M3 typed cache and M3 tool continuation

- `DSV4-M3-CURRENT-TYPED-CACHE`: `VERIFIED-LIVE_SCOPED`. Both artifacts were
  selected and launched through the real Electron UI with single-model mode
  enabled; each eagerly materialized before its first request and replaced the
  prior engine. DSV4 retained its native SWA/CSA/HCA composite path with pool
  codec and no generic TQ. M3 retained dense KV layers 0-2 plus native MSA
  index-key layers 3-59 and no generic TQ.
- DSV4 exact warm reuse restored 1,722/1,723 tokens as `paged+dsv4`; after a
  real UI restart it restored the same prefix as `paged+dsv4+disk`. Its
  nonterminal partial request safely recomputed because terminal composite
  CSA/HCA state was absent. The explicit pool-codec Off UI control changed
  health and raw progressive output still exact-finaled.
- `M3-PERSISTED-TQ-TRUTH`: fixed. The loader rejected TQ, but the disk-store
  health/admission gate remained enabled. The M3 CLI branch now sets
  `VMLX_DISABLE_TQ_KV=1`; after real Electron Save & Restart, health reports
  native MSA, `tq_native_enabled=false`, and a real 23-block SSD restore.
- M3 exact warm reused 1,495/1,500 tokens. Same-process partial reuse restored
  1,472/1,512; after process replacement and empty L1, a new suffix restored
  1,472/1,514 from SSD as `paged+disk`, then stored only the new tail.
- Current Electron M3 turns have non-empty output and prompt-distinct reasoning.
  The second same-chat turn generated and executed exactly one
  `file_info(panel/package.json)` call and exact-finaled
  `M3-HEAD-TOOL-DONE SIZE=5.2 KB` with no warning or zero-tool card.
- Validation: 132 focused Python tests. This cache/tool gate did not itself
  rerun media support; the later current-head media capability correction and
  representative live image/video proof are recorded below. Evidence:
  `docs/internal/release-gates/20260720_dsv4_m3_current_typed_cache/`.

## 2026-07-20 - current-source gateway protocols and one-model ownership

- `GATEWAY-AGENTIC-PROTOCOLS`: `VERIFIED-LIVE_SCOPED`. The real Electron API
  drawer showed the current dev gateway at `127.0.0.1:8088`, LAN Off, Single
  Model On, and one running MiniMax-M3 backend. Raw stream and non-stream Chat,
  Responses, Anthropic, and Ollama calls all returned HTTP 200, identical
  non-empty output, progressive stream deltas, and truthful native terminals.
- With Auto reasoning, every protocol emitted 512 reasoning deltas separately
  from 12 progressive content deltas and exact-finaled. This proves transport
  separation, not reasoning economy: the complete 512-token budget was used.
- `GATEWAY-TOOL-CONTINUATION`: `VERIFIED-LIVE_SCOPED`. Chat, Responses,
  Anthropic, and Ollama each emitted one schema-valid
  `file_info(panel/package.json)` call, consumed the real 5.2 KB result, and
  progressively exact-finaled without a second call. OpenAI stream/non-stream
  no-tool and continuation controls also completed.
- `GATEWAY-ONE-MODEL-AUTOSWAP`: `VERIFIED-LIVE_SCOPED`. Requests through 8088
  performed M3 -> DSV4 -> M3 replacement. Every state had
  `single_model_mode=true`, only the requested backend running, exactly one
  engine process, progressive exact output, and updated gateway health.
- Focused validation: 92 Python adapter passes; 87 panel gateway/session passes
  with three skips. Evidence:
  `docs/internal/release-gates/20260720_gateway_agentic_ownership_current/`.
- Retained `PARTIAL/OPEN`: current LAN enable/rollback, port-conflict and
  network-loss injection, concurrent/long gateway soak, signed-app repetition,
  other parser/model families, and media-bearing gateway turns. A stale
  installed app remains separately bound to 8081; this gate deliberately
  isolated the current dev app on 8088 and does not claim stale-listener cleanup.
- `GATEWAY-Q27-MTP-REASONING`: `VERIFIED-LIVE_SCOPED` at `616b0f3c8`.
  The real API drawer showed localhost 8088, LAN Off, Single Model On, and only
  Qwen3.6-27B-MXFP4-MTP running. Sequential Chat, Responses, Anthropic, and
  Ollama streams each returned HTTP 200, 460-469 separate reasoning deltas,
  12 progressive content deltas, exact `Q27-MTP-GATEWAY-REASON-DONE VALUE=95`,
  and the protocol-native terminal. Gateway health and `ps` still showed one
  Qwen engine afterward. This adds Qwen parser/MTP transport breadth; it does
  not close LAN/port/failure/cancel/media-bearing gateway rows. Evidence:
  `docs/internal/release-gates/20260720_gateway_agentic_ownership_current/`.

## 2026-07-20 - MiniMax-M3 current media capability telemetry correction

- `M3-MEDIA-CAPABILITY-TRUTH`: `VERIFIED-LIVE_SCOPED`. The earlier current-head
  note incorrectly used `/health.mtp.vl_runtime_available=false` as a media
  verdict. That field correctly describes inactive MTP for this non-MTP
  artifact. The owning `/v1/capabilities` surface reports current runtime
  modalities text/vision/video, 907 vision tensors, and no unwired media lane.
- Real Electron proof used a fresh M3 chat, actual file-input attachment,
  Thinking Off, tools Off, temperature 0, and Responses wire. The image was
  visibly rendered and exact-finaled `MAGNOLIA CACHE DONE`; SQLite retained
  non-empty content, null reasoning, no warning, and no tool state.
- Raw Responses sent the real six-frame MP4 and exact-finaled `BANANA8426`
  across four progressive content deltas, zero reasoning events under Off, one
  text-done, and one completed terminal. Last-content-to-completed was
  0.039767 s.
- This refreshes representative current-head image/video transport only.
  Broader OCR quality, stochastic Auto media, signed-app repetition, and
  REAP32 headroom remain `PARTIAL/OPEN`. Evidence:
  `docs/internal/release-gates/20260720_m3_media_capability_truth_current/`.

## 2026-07-20 - Gemma native-video versus sampled-frame capability truth

- `G4-VIDEO-CAPABILITY-TRUTH`: `PASS-SOURCE+LIVE_SCOPED`. The tested affine
  12B bundle explicitly declares native video false while carrying a video
  token/processor that the current engine safely routes through sampled image
  frames. Source now preserves both facts: native declaration false, runtime
  bridge supported. `/v1/capabilities` no longer adds bridge-only video to
  `declared_modalities`, but retains video in the runtime-supported surface.
- Real Electron Start-button loading materialized PID 20930 on port 8141 before
  a request. The visible drawer matched bundle defaults Auto/1.00/0.95/top-k
  64/repetition 1.00. A fresh blind MP4 turn emitted progressive visible
  prefixes and completed cleanly with null reasoning/no tools/no warning.
- `G4-VIDEO-BLIND-OCR`: `FAIL-QUALITY/PARTIAL-ROOT-CAUSE`. The fresh UI final
  was `FRANCMASSONIC`; raw direct-PNG and MP4-frame routes also missed the
  visible `BANANA8426` marker while retaining progressive content and truthful
  terminals. A history-contaminated same-chat exact answer is excluded.
  Because direct image and video fallback fail on the same pixels, this is not
  promoted as a video-only bridge bug and is not blamed on the trusted artifact
  without a controlled reference-runtime A/B.
- Validation: 39 focused capability tests and 739 expanded
  engine/multimodal/Gemma/scheduler tests passed. Evidence:
  `docs/internal/release-gates/20260720_gemma4_video_capability_bridge_current/`.

## 2026-07-20 - Nemotron Omni omitted-cap Auto finalization and image-session proof

- `NEMO-OMNI-OMITTED-MAX`: `VERIFIED-LIVE_SCOPED` after a current-source
  repair. A fresh real Electron Auto image turn retained 849 reasoning
  characters but stopped at exactly 256 tokens with empty content and length
  warnings, although the UI Max Tokens field was blank, the bundle default was
  16,384, and server sampling logs resolved 16,384. Responses rebuilt its
  internal Chat request from unresolved `request.max_output_tokens`; the Omni
  bridge then applied its old 256 fallback.
- Current Chat, Responses, and Anthropic Omni entries pass the same resolved
  max-token/temperature/top-p values used by their ordinary generation rails.
  The patched Electron Auto turn produced separate 284-character reasoning
  plus exact `BANANA8426` / `NEMO-IMG-B2-AUTO-DONE` content in 107 tokens, no
  warning, and no tool state. Raw omitted-max Responses emitted 257 reasoning
  plus 24 content deltas and one completed event; Chat emitted 351 reasoning
  plus 23 content deltas, stop, usage, and `[DONE]`.
- `NEMO-OMNI-IMAGE-SESSION`: `VERIFIED-LIVE_SCOPED` in-process. Real image A
  exact-finaled `vMLX` over progressive character paints and wrote a 50.97 MB
  q4-attention-KV/native-SSM snapshot. A no-attachment follow-up exact-recalled
  it at 0.37 s TTFT; logs recorded `continuing conversation (prefix matches)`
  and zero new images. Fresh image B logged a reset and returned the distinct
  unseen code without leaking A.
- Bundle/start truth: real Electron Start eagerly loaded the MXTQ/JANGTQ2
  artifact before a request and left exactly one engine. Health reports q4 TQ
  only for attention layers 3/7/11/15/19/24, native SSM companion layers, and
  async rederive; this is not affine JANG, base MXFP, or generic all-layer KV.
- `NEMO-OMNI-VIDEO-SESSION-L2`: `VERIFIED-LIVE_SCOPED` for transport,
  streaming, and disk restoration. The fresh Electron MP4 turn had 33 visible
  UI states, separate reasoning, exact unseen `BANANA8426`, and one final
  message. Real Stop/Start changed PID 23620 to 24498 and health showed the
  replacement loaded with `last_request_time=null`. The no-attachment
  follow-up restored 51,847,398 bytes of q4 attention KV plus native SSM in
  0.000328 s (`hits=1`, `misses=0`). Raw omitted-max Responses emitted 447
  reasoning plus 25 content deltas and one completed event.
- Exact-format quality remains `PARTIAL`: the post-restart Electron follow-up
  appended an unrequested Python block after the correct two lines. No output
  filter was added. The architecture store is one `latest.safetensors`;
  latest-only replacement behavior, bounded/multi-snapshot eviction, and
  different-history partial-prefix matching remain open.
- Focused validation: 30 passed / 580 deselected plus `py_compile` and diff
  check. Evidence:
  `docs/internal/release-gates/20260720_nemotron_omni_media_cache_current/`.

## 2026-07-20 - Gemma 4 MoE token-only audio capability and video L2

- `GEMMA4-MOE-AUDIO-CAPABILITY`: `VERIFIED-LIVE_SCOPED` after a current-source
  repair. The real 26B-A4B JANG_4M artifact is a 30-layer, 128-expert top-8
  MoE with vision but `audio_config=null` and no audio tower. Pre-fix
  capabilities nevertheless advertised audio because a reserved top-level
  `audio_token_id` reached a generic heuristic. Gemma 4 now fails closed on a
  token-only audio stamp. Patched `/v1/capabilities` reports text/vision/video
  and `audio=not_advertised`; raw Responses WAV returns HTTP 400.
- `GEMMA4-MOE-VIDEO-L2`: `VERIFIED-LIVE_SCOPED`. Real Electron `Launch
  Session` stopped the prior Nemotron engine and eagerly loaded one Gemma
  process before a request. Fresh MP4 exact-finaled over 16 progressive UI
  states with separate reasoning. After real process replacement and empty
  L1, a fresh identical media/prompt row restored 327/328 tokens as
  `paged+mixed_swa+disk+tq-native`, with six disk promotions, six native-TQ
  hits, and 0.33 s TTFT. Raw omitted-max Responses emitted 100 reasoning plus
  18 content deltas and one completed event.
- A no-attachment follow-up is not counted as cache proof: it was coherent and
  exact but health recorded zero hits because ordinary Gemma history omitted
  the old media payload. Fifteen focused tests passed. Evidence:
  `docs/internal/release-gates/20260720_gemma4_moe_media_capability_current/`.
- `GEMMA4-MOE-POSTMEDIA-TOOL`: `VERIFIED-LIVE_SCOPED` in the current dev
  Electron app. Built-in tools were enabled through the real Chat settings
  drawer after the MP4 turn. The next no-attachment request rendered 16
  progressive states, separate 253-character reasoning, one exact
  `file_info(panel/package.json)` call, the real 5.2 KB result, and exact
  `G4MOE-POSTMEDIA-TOOL1-DONE SIZE=5.2 KB` visible content. SQLite row 205 has
  one call/result, no warning, 0.16-second TTFT, and 1.8-second total time.
  The owning request-shape log records zero attachments and `has_tools=true`,
  so stale video was not resent into the tool turn. Raw API post-media
  continuation and other Gemma variants remain open; this is not a family-wide
  closure.

## 2026-07-20 - Qwen3.6 27B MXFP4-MTP video, native MTP, and restart L2

- `Q27-MXFP4-MTP-CLASSIFICATION`: `VERIFIED-LIVE_SCOPED`. The exact tested
  artifact is base MLX MXFP4, not affine JANG or JANGTQ/MXTQ. Bundle/index
  truth records 333 vision and 23 MTP tensors. The real UI launch selected
  native MTP depth 3 with text+VL scope and eagerly materialized one engine
  before a request.
- `Q27-MTP-VIDEO-UI-STREAM`: `VERIFIED-LIVE_SCOPED`. Real file-input MP4
  attachment exact-finaled `BANANA8426` plus the requested marker. The 86-state
  DOM trace proves progressive reasoning followed by progressive content;
  SQLite has non-empty content, separate reasoning, and no tool/warning. MTP
  telemetry recorded 127/192 accepted drafts including depth-2/depth-3 accepts.
- `Q27-MTP-VIDEO-RESTART-L2`: `VERIFIED-LIVE_SCOPED`. A visible Stop/Start
  emptied L1 while retaining SSD state. The identical fresh-chat media/prompt
  restored 2,225/2,226 tokens from all 35 q4-native attention blocks plus the
  native SSM companion, exact-finaled over 81 UI states, and reduced TTFT from
  7.68 s to 0.60 s. TurboQuant is restricted to the 16 attention-KV layers;
  48 SSM layers retain native companion state and async rederive.
- `Q27-MTP-RAW-RESPONSES-VIDEO`: `VERIFIED-LIVE_SCOPED`. Omitted-max raw
  Responses emitted 155 reasoning deltas, 16 progressive content deltas,
  exact content, one text-done, and one completed event. It recorded 130/194
  accepted MTP drafts including deeper accepts.
- `Q27-MTP-POSTVIDEO-TOOL`: `VERIFIED-LIVE_SCOPED`. In the same real chat,
  enabling built-in tools through the visible drawer produced 50 UI states,
  separate reasoning, exactly one `file_info(panel/package.json)` call, one
  real 5.2 KB result, exact visible content, and no warning. Tool-bearing MTP
  correctly capped to depth 1 with no deeper drafts or dropped call.
- Remaining: media-salt/partial-prefix variants, raw API post-media tool use,
  Chat/Anthropic/Ollama media, explicit MTP policy variants, bounded eviction
  and fault injection, longer inputs, 35B MoE MTP, Bonsai/Ornith breadth, and
  signed-app repetition. Evidence:
  `docs/internal/release-gates/20260720_qwen36_27b_mxfp4_mtp_video_current/`.

## 2026-07-20 - DSV4 long-prefill memory, UI chunk parity, and output split

- `DSV4-LONG-PREFILL-MEMORY`: `VERIFIED-LIVE_SCOPED`. The tested affine JANG
  CRACK artifact retained its native 43-layer SWA/CSA/HCA composite cache and
  separate pool codec; generic TurboQuant remained off. Source now rejects an
  unsafe deep-copy prompt snapshot before allocation using the active backend
  budget and Metal headroom, and DSV4 JANG materializes hidden state at each
  layer boundary for multi-token prefill so all 43 lazy CSA/HCA graphs do not
  coexist. The exact 23,477-token Electron prompt survived at about 104.9 GB
  peak active memory instead of the prior Metal OOM.
- `DSV4-PREFILL-STEP-UI-ARGV`: `VERIFIED-LIVE_SCOPED`. DSV4 still suppresses
  inapplicable batch-size flags, but the real preview and Start path now pass
  the user's `--prefill-step-size`. Current live argv records 512.
- `DSV4-LONG-AUTO-QUALITY`: `FAIL/PARTIAL`. Memory survival is not output
  correctness. The 23,477-token Auto turn generated 23,560 reasoning
  characters, exhausted 4,144 output tokens, and returned wrong visible
  content. A 7,875-token Auto turn visibly streamed 1,228 UI changes but also
  looped until manually stopped at 2,288 tokens. No output filter, hidden
  sampler clamp, forced Thinking Off, or synthetic answer was added.
- `DSV4-MEDIUM-INSTRUCT-STREAM`: `VERIFIED-LIVE_SCOPED`. The same 7,875-token
  prompt under Instruct produced 18 progressive Electron states and exact
  non-empty three-line content. Raw Responses Thinking emitted 512 distinct
  reasoning deltas followed by 29 content deltas and one completed terminal;
  raw Instruct emitted 31 progressive content deltas.
- `DSV4-DETERMINISTIC-COMPOSITE-L2`: `VERIFIED-LIVE_SCOPED`. After real UI
  process replacement, the deterministic row restored 7,874 tokens as
  `paged+dsv4+disk`; the immediate resident repeat restored the same prefix as
  `paged+dsv4`. Health records 31 disk promotions, 15,748 saved tokens, and the
  exact native typed components. The direct JANG reference A/B remains
  `BLOCKED` because its legacy loader rejected the official sidecar shape
  before model load; no artifact-level conclusion is made.
- `CHAT-SAMPLING-DEFAULTS-DSV4`: `VERIFIED-LIVE_SCOPED`. Bundle JANG chat
  defaults 0.6/0.95/top-k Off/min-p Off/repetition 1.0/max 4096 matched the
  visible DSV4 drawer and detection API. Eight-session metadata parity was
  captured, but each non-DSV4 session still needs its own visual/payload/runtime
  chain before promoting broad family parity.
- Focused validation: 67 DSV4 cache tests, 12 snapshot-admission selections,
  286 panel settings-flow tests, panel typecheck, Python compile checks, and
  seven JANG DSV4 overlap/pool tests. Evidence:
  `docs/internal/release-gates/20260720_dsv4_long_context_snapshot_budget_current/`.
- Overall campaign and release status remain `PARTIAL`: DSV4 Auto quality,
  broader protocols/media/fault injection, other model-derived sampler visual
  rows, and signed-app repetition are not closed by this scoped gate.

## 2026-07-20 - effective Chat sampling defaults under native-MTP startup policy

- `CHAT-SAMPLING-NATIVE-MTP-EFFECTIVE-PARITY`: `VERIFIED-LIVE_SCOPED` after a
  current-source repair. The Qwen3.6 27B MXFP4-MTP bundle declares stochastic
  defaults 1.0/.95/top-k 20, but the session's default native-MTP
  `deterministic` mode launches the server with `deterministic-defaults`, which
  intentionally changes omitted requests to 0/1/top-k Off/min-p Off. The
  pre-fix drawer showed bundle values while live resolved kwargs used greedy
  values. This was a panel/effective-policy mismatch, not an artifact defect.
- Current source applies session startup policy after bundle/model detection.
  Both Chat Settings entry points pass the serialized session config, and the
  renderer type contract now includes native-MTP detection. `auto`/`off` MTP
  modes retain bundle defaults; explicit chat values keep request precedence.
- Patched Qwen Electron proof: the visible inherited drawer is now
  0/1/Off/Off/1, same-chat output exact-finaled with no warning and
  `paged+ssm+tq-native`, and runtime logs resolved 0/1. A real UI override to
  1/.95/20 exact-finaled over seven progressive states and runtime logs kept
  1/.95/20. Real Reset returned the drawer to 0/1/Off/Off/1.
- Raw Responses A/B independently exact-finaled: omitted values emitted eight
  content deltas and one completed terminal at resolved 0/1; explicit values
  emitted ten deltas and one completed terminal at resolved 1/.95/20. Native
  MTP stayed active at depth 3; the explicit stochastic row used its rejection
  acceptance path and accepted tokens at depths 1, 2, and 3.
- Ordinary non-MTP-policy spot checks also pass live: Laguna JANGTQ visibly
  matched and resolved .7/.9/max 2048; MiniMax-M3 visibly matched and resolved
  1/.95 and restored 128 tokens from disk. DSV4 -> Laguna -> M3 -> Qwen UI
  Starts left exactly one Qwen engine process.
- A clean Electron/process relaunch independently retained the repair: startup
  found the intended venv engine, the real Start button eagerly loaded Qwen
  with `last_request_time=null`, and the fresh renderer still showed
  0/1/Off/Off/1. The following Electron turn exposed a separate 1211-character
  reasoning rail, 113 observed progressive UI states, and the exact non-empty
  visible final without a warning.
- Validation: 561 focused panel tests across five files and TypeScript
  typecheck. Evidence:
  `docs/internal/release-gates/20260720_sampling_effective_mtp_current/`.
- Retain `PARTIAL`: a non-neutral repetition-penalty artifact, sampling A/B on
  Chat/Anthropic/Ollama, signed-app repetition, and the remaining catalog are
  not closed by this representative gate.

## 2026-07-20 - Qwen gateway sampler translation across four protocols

- `Q27-GATEWAY-SAMPLING-PRECEDENCE`: `VERIFIED-LIVE_SCOPED` on current source.
  The already Electron-loaded Qwen3.6 27B MXFP4-MTP backend received 16 live
  gateway calls: Chat, Responses, Anthropic, and Ollama; each with omitted and
  explicit sampling; each in stream and non-stream mode.
- All 16 returned HTTP 200 and exact non-empty visible markers. The eight
  streams emitted 10-19 time-separated content deltas, kept reasoning empty
  under explicit thinking-off, and ended with their native terminal
  (Chat stop+DONE, Responses completed, Anthropic message_stop, Ollama done).
- Engine logs independently resolved every omitted route to temperature 0 /
  top-p 1 under deterministic native-MTP startup policy. Explicit routes
  resolved temperature 1 / top-p .95 / top-k 20; Chat/Responses/Ollama also
  retained repetition penalty 1.05. Explicit temperature 1 correctly skipped
  deterministic MTP, while omitted temperature 0 exercised it.
- Gateway health remained `single_model_mode=true`, only Qwen was running,
  exactly one engine process existed, and the scheduler returned to zero active
  requests. Focused validation: 92 Python adapter tests, 82 panel gateway tests,
  and TypeScript typecheck passed.
- Evidence:
  `docs/internal/release-gates/20260720_q27_gateway_sampling_protocol_ab/`.
- Retain `PARTIAL`: this does not repeat cancellation/failure injection, media
  payloads, tool-result continuation under stochastic sampling, concurrent
  clients, LAN/port rollback, signed app, or other parser/model families.

## 2026-07-21 - Qwen gateway media preservation and answer-stream completion

- `Q27-GATEWAY-MEDIA-TRANSLATION`: `VERIFIED-LIVE_SCOPED`. The exact tested
  artifact is `dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP`: base MLX MXFP4 with
  native MTP, not affine JANG or JANGTQ/MXTQ. Anthropic video/audio blocks,
  direct Ollama `videos` and `audio`/`audios`, and Electron-gateway Ollama
  media extensions now reach typed model content instead of being dropped.
- `Q27-AUTO-PARTIAL-ANSWER-STREAM`: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED`.
  Current-source Chat, Responses, Anthropic, and Ollama media requests each
  returned exact stream and non-stream answers. Streams emitted 9-14
  progressive content deltas, 86-154 separate reasoning deltas, and one native
  terminal. A reserved answer pass now reconciles against the already-visible
  byte prefix and emits only the missing suffix; divergence fails closed.
- `Q27-NONSTREAM-TERMINAL`: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED`. Chat now
  adopts the completed answer pass terminal rather than retaining the first
  pass's `length`; translated Ollama therefore also reports `stop`. Responses
  reports `completed` and Anthropic reports `end_turn`.
- `DEV-ELECTRON-SOURCE-PIN`: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED`. A clean
  Electron launch with no parent `PYTHONPATH` loaded through the real Start
  path and spawned the engine with
  `PYTHONPATH=/Users/eric/mlx/vllm-mlx-release-1.6.13`. This prevents a sibling
  editable checkout from silently supplying runtime source during dev proof.
- Real Electron MP4 proof produced 90 UI states, separate reasoning, and exact
  `BANANA8426` / `Q27-MEDIA-UI-CURRENT-DONE`. The clean source-pin follow-up
  persisted exact visible content with separate reasoning and no warning.
- Validation: 131 focused Python tests, 58 panel tests, TypeScript typecheck,
  and `git diff --check`. The new reconciliation helper has four production
  call sites; no superseded stream branch was retained solely for tests.
- Retain `PARTIAL`: audio has source/contract coverage but no audio-capable
  live model in this gate; Auto reasoning verbosity, media tools,
  cancellation/fault injection, alternate-media salt, signed-app repetition,
  and other model families remain open. Evidence:
  `docs/internal/release-gates/20260721_gateway_media_stream_repair_current/`.

## 2026-07-21 - Electron gateway downstream-disconnect recovery

- `GATEWAY-CLIENT-DISCONNECT-FOUR-PROTOCOL`: `VERIFIED-LIVE_SCOPED`. The
  current Electron-owned gateway on port 8088 received real Chat, Responses,
  Anthropic, and Ollama streams. Each downstream client closed after three
  visible deltas and before a terminal. Backend activity returned to zero in
  24.88-28.08 ms after close.
- Each immediate recovery request returned HTTP 200, exact visible content,
  8-10 progressive deltas, its native terminal, and an idle backend. Gateway
  health retained Single Model On, one Qwen engine, and all other sessions
  stopped.
- The following real Electron chat turn produced 88 observed UI states,
  separate 891-character reasoning, exact
  `GATEWAY-UI-DISCONNECT-RECOVERY-DONE`, no warning/tool payload, and 130
  `paged+ssm+disk+tq-native` cached tokens. The screenshot was visually
  inspected and backend health remained idle.
- Source/dead-code trace confirms both shared close helpers have four
  production route call sites; no source edit or redundant model/test matrix
  was needed.
- Retain `PARTIAL`: explicit request-ID cancel, upstream/backend connection
  loss through the gateway, concurrent disconnect/swap soak, active-request
  LAN/port failure, and signed-app repetition remain open. Evidence:
  `docs/internal/release-gates/20260721_gateway_disconnect_recovery_current/`.
