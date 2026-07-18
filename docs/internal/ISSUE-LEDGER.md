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
| #78 | M | TurboQuant live encode inert engine-wide (compress_after=0; no config param) | VERIFIED-LIVE | LIVE (Hy3/Qwen3.6/Gemma) | `compress_after` is threaded through loader/config/clone paths and real codec counters advance. All three default OFF: Hy3 coherence gate failed; Qwen/Gemma resident-memory gates failed. Capabilities/logs are truthful and make no 5x claim. |
| A-cap | M | /v1/capabilities returns 404 (real path /v1/models/{id}/capabilities) | VERIFIED-LIVE | LIVE | `/v1/capabilities` returned 200 for loaded Gemma and exposed truthful TQ/native-cache data. |
| A-omni | M | attention-only live-TQ allow-list covers Qwen3.6 not Nemotron-Omni | VERIFIED-LIVE | LIVE | Nemotron: 6 attention TQ + 23 native SSM slots; warm 702-token `paged+ssm+tq`, exact output, SSM companion entry present. |
| A-lag | M | Laguna cannot reach documented SWA opt-in loader path | VERIFIED-LIVE | LIVE | Dedicated loader reached opt-in: 10 full-attention TQ + 30 RotatingKVCache; warm 601-token `memory+tq`, exact output. |
| seed | L | text chat/completions ignore `seed` (image endpoints only) | VERIFIED-LIVE | LIVE | Request-local keyed sampling wired through chat/completions/Responses/Anthropic/Ollama and MTP. Fresh-cache same-seed completion byte-identical, different seed diverged; SSE produced visible `STREAM-SEED-OK`. |
| #45 F1 | M | q4 stored-prefix cold != warm first-token divergence | OPEN (awaits Eric) | LIVE (3 models) | inherent to q4 store; needs per-family gate |
| M3-stream-zero | H | MiniMax-M3 tools-enabled media turn could buffer invalid XML, erase the visible answer, and render a completed zero-tool card | VERIFIED-LIVE (current source) | LIVE Electron + focused tests | Finalizer hides invalid control suffix; renderer hides completed speculative zero-tool heartbeat; M3 late answer pass runs only after parser proves no valid call. Current image/video turns produce visible grounded output with no zero-tool card, and the genuine-tool post-fix row still executes exactly one `file_info`. |
| M3-nonstream-parity | H | MiniMax-M3 non-stream Chat/Responses may skip its bounded visible-answer fallback whenever tools are merely available | VERIFIED-LIVE (current source) | 2 functional tests + live API after Electron restart | Non-stream Chat/Responses now late-arm the bounded M3 answer pass only when no schema-valid call exists. Live tools-available/no-call markers returned exactly `MM3-NONSTREAM-RESP-DONE` and `MM3-NONSTREAM-CHAT-DONE`. |
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
