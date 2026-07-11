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
| MTP-ui | M | After model load the UI must SHOW MTP settings (like Qwen MTP models) + engine native type. Verify Hy3-JANG_2K-MTP surfaces MTP depth/native-type in UI post-load. | OPEN | LIVE UI needed | compare to Qwen MTP model UI; capabilities.mtp + native cache type must render |

## Appended 2026-07-10 (Eric: per-arch cache policy — NO ASSUMPTIONS, live-verify each)
| ID | Sev | Issue | Status | Evidence | Notes |
|----|-----|-------|--------|----------|-------|
| CACHE-policy | H | Intended (Eric): new UI session starts WITH prefix cache; KV-component families get TQ ENCODE on their KV part — gemma rotating-SWA KV, qwen hybrid-SSM attention-KV (+ async-rederive SSM), hy3 plain-KV layers. Must LIVE-verify per family what ACTUALLY happens (cache layout, TQ objects, whether encode FIRES, async rederive) vs intended. DO NOT ASSUME what applies where. | VERIFIED-LIVE (encode defaults gated off) | LIVE per-family | Hy3 plain 80 TQ; Qwen hybrid 16 TQ + 48 native SSM and paged+ssm; Gemma 8 full TQ + 40 rotating. Prefix hits and exact outputs proven. Encode remains OFF because no family passed every coherence+memory gate. |
| GRADE-rule | - | Every test cell must be graded PASS/FAIL by LIVE proof; each FAIL gets a fix. | POLICY | Eric directive | applies to reasoning parity + cache policy + MTP UI matrices |

| PAGED-toggle | M | UI paged cache default OFF (correct); toggling it ON must actually work end-to-end (UI→gateway→engine spawn --use-paged-cache→paged blocks + TQ on KV). CLI arm M2 passed; UI-toggle path needs live proof. | OPEN | LIVE UI needed | verify spawned engine argv + paged path active + determinism |

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
