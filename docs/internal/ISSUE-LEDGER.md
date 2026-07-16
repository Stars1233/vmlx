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
| Bonsai-ui-tool-exec | M | Bonsai Electron built-in tool execution could complete the tool but end in repeated reasoning-only retries with no visible answer | VERIFIED-LIVE functional / PARTIAL performance (current source) | Source trace + LIVE Electron + DB + raw Responses SSE | `292f99b28` bounds empty post-tool recovery to one answer-only pass; `0cc2ee8f1` recognizes the explicit one-tool/exact-final contract and removes tools only for its planned follow-up. Current Electron rows 1443 and 1446 each made exactly one `file_info`, one real result, one reasoning card, and exact `B1-UI-TOOL10-DONE`; no repeated reasoning cards or missing final remain. The identical warm row was slow/stochastic (2,338 generated tokens, 64.3s versus 118 tokens, 4.5s cold), so latency/token variance is not closed. |
| Bonsai-reasoning-token-variance | M | Default-thinking Bonsai can spend thousands of hidden/reasoning/tool-prefix tokens before an otherwise valid single tool call | OPEN/PARTIAL | LIVE Electron + engine logs + direct raw SSE | Cold Electron row 1443 finished in 118 tokens/4.5s. Identical `paged+ssm` row 1446 finished correctly but consumed 2,338 tokens/64.3s. A direct Responses control completed a valid call in 213 output tokens and exposed a coherent reasoning summary, so the shared stream/parser completes; stochastic reasoning latency remains variable and must not be called performance-green. |
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
- `BONSAI-B1-UI-TOOL3-REGRESSION`: REOPENED/UNVERIFIED-CURRENT. Earlier rows
  1443/1446 and screenshots `b1-ui-tool10-*` remain valid evidence for that
  exact prompt/settings state, but the user's later live `B1-UI-TOOL3` report
  shows repeated reasoning cards, one `Info panel/package.json` result, and no
  final content. Current HEAD rerun row 1473 did finish one real
  `file_info` call and exact `B1-UI-TOOL3-RERUN-DONE` with one persisted
  reasoning segment, so missing-final was not reproduced in that fresh chat.
  However it generated 3,617 tokens over 103.2s before the tool closed, with
  only 28 visible reasoning characters and the UI showing tool-call buffering.
  Treat the user-visible repeated/stuck reasoning symptom as abnormal and keep
  Bonsai `FUNCTIONAL PASS / PERF-HIDDEN-GEN PARTIAL`.
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
