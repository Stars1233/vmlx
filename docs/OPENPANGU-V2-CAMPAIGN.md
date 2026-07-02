# openPangu-2.0-Flash (openpangu_v2) — vmlx Python integration campaign

**Started 2026-07-02.** Living status board — update as work lands. Scope set by Eric
(verbatim intent): PR reviews #223/#218 grounded-commented (DONE, posted), then
"analyze and read all about openpangu and start the integration for all autodetect
with proper prefix cache paged cache cache pooling quant and also mtp detection for
depth 3 pass with proper cache sync between dsa swa and other embedding attention
mla whatnot and make sure no other functions and settings that shpuldnt be" +
"test and fix live" + "and notarize etc".

Additional standing directives (2026-07-02 session):
- Keep status + all needed matrix live UI multiturn chatting tests documented as we go.
- Check parity of the settings in UI vs params passed to CLI (argv) + logs.
- BOTH /v1/responses API AND proper UI live chat multiturn checking is crucial — every time.
- Qwen 3.5/3.6 + Gemma 4: video AND VL (and audio where present) processing must properly work (separate priority, task #43).
- Testing = multiturn CROSS-MATRIX with variated combos (e.g. turn1 VL no reasoning →
  turn2 tool call WITH reasoning → …) covering every capability the model has;
  UI must emit and show responses/answers correctly.
- Consider interleaved reasoning + content delta streaming issues; batching.
- Custom kernel may be needed — or modify the existing DSV4 one. (Current call:
  REUSE `jang_tools.dsv4.mlx_model.hc_split_sinkhorn` — fused Metal kernel, validated
  vs torch reference — for the mHC Sinkhorn. If openpangu profiling shows the DSA
  indexer or conv path needs its own kernel, extend the DSV4 kernel family.)

## Ground truth / references

- Model: `~/models/openpangu/openPangu-2.0-Flash` (bf16 HF snapshot, config.json is canonical).
- JANG bundle: `~/models/JANGQ-AI/openPangu-2.0-Flash-JANG_2L` (39.9GB, avg 3.17b,
  per-tensor `tensor_quantization_manifest` in jang_config.json — 684 entries,
  incl. the ambiguous `*_mhc_module.phi` 2-bit/gs128).
- jang_config capabilities stamp: `family=openpangu_v2, cache_type=hybrid,
  reasoning_parser=deepseek_r1, tool_parser=qwen, think_in_template=true`.
  MTP stamp: `depth 3 (layers 46-48), spec_decoding_ready=true, layer_prefix=model.layers`.
- PROVEN reference implementation: vmlx-swift branch `feat/openpangu-v2`
  (OpenPanguV2*.swift) — live-coherent on JANG_2L incl. multiturn recall.
  Port log with all 10 bugs: `~/jang/docs/openpangu-v2-port.md`.
- Runtime plan: `~/jang/docs/research/2026-07-01-next-gen-architectures/06-…runtime-plan.md`.
- Ascend reference: gitcode ascend-tribe/openPangu-2.0-Infer (omni_npu `_mhc_*_naive`).

## Architecture invariants baked into the Python port (from the Swift bug list)

1. Conv ORDER: `q_a_proj → qa_conv → q_a_layernorm` and `kv_a split → compresskv_conv
   → kv_a_layernorm` — conv on RAW latent BEFORE layernorm. `o_conv` after attention,
   before o_proj.
2. Conv residual: `y = conv(x) + x` (reference residual_connection=1).
3. Conv weight PyTorch `[C,1,3]` → MLX `[C,3,1]` (sanitize transpose).
4. Sink mask POLARITY: boolean masks are true=attend; the 128 sink columns MUST be
   prepended as True. (THE Swift coherence bug — sinks masked out during prefill made
   the model context-blind.)
5. Sinks: `kv_b_proj(kv_a_layernorm(param_sink_compressed_kv))` per-head expand;
   `param_sink_k_pe` raw, NO rope; position-free.
6. mHC: phi[24,4H] = DSV4 `fn`; branch_alpha[3] = scale; branch_beta[24] = base
   directly; expand uses comb TRANSPOSED; hc_eps=1e-6 (not rms_norm_eps); merge
   gate = sigmoid, NO +eps, no sum-to-1.
7. block_post_layernorm weight is `[4*hidden]=10240` on the FLATTENED 4-stream
   residual (9 layers).
8. MoE: sigmoid scores; SELECT on scores+e_score_correction_bias; WEIGHT with
   unbiased; renorm × 2.5. Router `mlp.gate.weight` fp16 (must stay Linear).
9. rope: theta 6.4M, rope_interleave=false → traditional=False (split-half); scale
   192^-0.5, no mscale.
10. fp32 SDPA on L==1 decode (MLA absorb-drift, Ornith-397B lesson).
11. `e_score_correction_bias` lives at `mlp.` level in the bundle → held directly on
    the MoE module in Python (key path matches, no rename — improvement over Swift).
12. phi bit-ambiguity: solved via jang_config `tensor_quantization_manifest`-driven
    quantization overrides in the loader (improvement over Swift's dequant-freeze).
13. router_sliding_window=3 rolling bias: NOT implemented — training-time smoothing;
    inference uses static e_score_correction_bias (Swift proven coherent without it).
14. MTP layers 46-48 dropped at sanitize (mtp_mode=off); MTP DETECTION stays
    metadata-driven from jang_config.mtp (depth 3) via native_mtp.py — same
    "detected but runtime-unwired" bucket as DSV4. Do NOT add openpangu_v2 to
    `_RUNTIME_SUPPORTED_FAMILIES`.

## Integration surface (files)

| # | File | Status | What |
|---|------|--------|------|
| 1 | vmlx_engine/models/openpangu_v2/cache.py | DONE | OpenPanguV2LayerCache (KV/Rotating + indexer KV + 3 conv states; is_trimmable=False; state/meta_state round-trip) |
| 2 | vmlx_engine/models/openpangu_v2/openpangu_v2.py | DONE (untested) | Full model: MLA+convs+sinks, DSA indexer (top-2048, no-op ≤2048), mHC (jang_tools hc_split_sinkhorn kernel), sandwich norm, MoE, sanitize |
| 3 | vmlx_engine/models/openpangu_v2/register.py | TODO | sys.modules install under mlx_lm.models.openpangu_v2 (minimax_m3 pattern) |
| 4 | vmlx_engine/models/openpangu_v2/__init__.py | TODO | exports |
| 5 | vmlx_engine/utils/jang_loader.py | TODO | register hook pre-load + manifest-driven `_post_load_quantization_overrides` for openpangu_v2 |
| 6 | vmlx_engine/model_configs.py | TODO | static ModelConfig (family openpangu_v2, cache_type/subtype decision below, eos 148902, reasoning deepseek_r1, tool qwen) |
| 7 | vmlx_engine/cli.py | TODO | register call + family policy: auto-ignore --enable-jit (M3 precedent), text-only, prefix/paged safety |
| 8 | panel/src/main/sessions.ts | TODO | family defaults: timeout, cache toggles; argv parity |
| 9 | tests/ | TODO | unit: config parse, sanitize (conv transpose, MTP drop), gate math, sink mask polarity, cache state round-trip, mtp detection depth-3 |
| 10 | block_disk_store.py typed lane | DEFERRED | openpangu_v2 SSD lane (kv+indexer+conv states) — Phase 2, after live coherence |

## Cache / settings policy (the "no functions that shouldn't be" list)

- TurboQuant-KV: AUTO-SKIPPED (kv_lora_rank=512 → is_mla_model → TQ-KV off). Verify in logs.
- Prefix/paged cache: OpenPanguV2LayerCache → detect_cache_type UNKNOWN → stores skip
  safely (no silent false hits; conv state is path-dependent). Warm reuse = Phase 2
  typed lane. MUST VERIFY the store actually skips (not crashes) live.
- Paged cache default: OFF (v7 policy) — openpangu_v2 should be registered with
  cache_type that does NOT force paged (decision: `kv` + subtype
  `openpangu_v2_composite`, DSV4 pattern), overriding the converter's coarse
  `hybrid` stamp which would misroute into SSM hybrid handling.
- --enable-jit: auto-ignore for openpangu_v2 (dynamic DSA top-k selection = same
  class of mx.compile hazard as M3 MSA).
- Native MTP runtime: OFF (detection-only). EAGLE3: N/A.
- Norm shift (+1): should NOT fire (DeepSeek-style norms ~1.0); `_jang_needs_norm_shift`
  auto-detect must return False — verify at load.

## Verification gates (all must pass on JANG_2L, remote erics-m5-max.local)

Engine-level (preliminary sanity only, NOT final proof):
- G1 structural: loads, all 46 layers, no missing/unused tensor complaints.
- G2 first-token: "The capital of France is" → argmax ≈ Paris (forward correctness).
- G3 greedy 40 tokens coherent; thinking=false fast path.
- G4 DSA vs SWA: layer 3 (DSA) vs layer 4 (SWA window 512) instrumented on same prompt.
- G5 multiturn recall (fact from turn 1 recalled turn 3+) — conv-state carry proof.
- G6 >2048-token prompt: indexer activates (log top-k path), needle-in-haystack.
- G7 reasoning parser: <think> split works via deepseek_r1 parser; budget backstop.

Full matrix (final proof — Responses API + UI live chat, per mandatory rules):
- M1 /v1/chat/completions streaming + non-streaming: multiturn, reasoning on/off/auto.
- M2 /v1/responses: same matrix; interleaved reasoning/content delta streaming —
  watch for reasoning-in-content leaks, empty deltas, malformed finish.
- M3 UI live chat (dev-build, CDP :9333): multiturn variated cross-matrix —
  turn1 plain no reasoning → turn2 reasoning → turn3 tool call w/ reasoning →
  turn4 recall check. UI must render responses/answers correctly (no blank
  bubbles, no tag leaks). openpangu is text-only → no VL/video/audio rows.
- M4 tool calls: qwen tool parser fires; streaming tool_call id consistency (#219 fix).
- M5 batching: 2+ concurrent requests, no cross-request contamination (conv states
  are per-request cache objects — verify batch path doesn't share).
- M6 settings↔argv parity: every UI toggle lands in spawned argv + startup logs;
  toggling changes argv (v7 protocol); openpangu policy flags visible in logs.
- M7 RAM: stable across 15+ turn soak, wired ceiling respected.

## Status log

- 2026-07-02: PR #223 + #218 grounded reviews POSTED on GitHub. Integration map
  built (3 explorer agents). cache.py + openpangu_v2.py written (proven-math port
  from Swift + deepseek_v32 indexer + jang_tools Sinkhorn kernel). Register/loader/
  configs/cli/panel wiring next; then smoke on remote, then matrix.
- 2026-07-02 (cont): ALL wiring landed + pushed (06d3b06af + 2716f9c9c):
  register/__init__, model_configs entry, cli registration + policy/transparency
  log, jang_loader manifest overrides + family-gate registration, panel defaults,
  tests/test_openpangu_v2.py (10/10 green; prefill-vs-incremental equivalence
  diff 0.005). Synthetic smoke: sanitize/prefill/14-step decode past SWA window
  + DSA topk/state-roundtrip ALL PASS. Adjacent regressions: 227 pass, 1
  pre-existing failure (step37 runtime_scope KeyError — fails at HEAD without
  these changes, known debt). JANG_2L bundle rsynced to erics-m5-max.local
  (~40GB, 5m10s). NOTE: remote /Users/eric/mlx/vllm-mlx working tree is DIRTY
  (uncommitted iter-32/34/35 session fixes in server.py/adapters/cli.py —
  tasks #29/#30); openpangu testing uses a DETACHED WORKTREE at
  /Users/eric/mlx/vllm-mlx-openpangu instead — do not clobber the dirty tree.
  G1/G2 (real-bundle load + first-token probe) launched on remote.
- 2026-07-02 LIVE GATES: G1 PASS (JANG_2L loads in 5s via v2 mmap, 39.0GB peak,
  after tokenizer trust_remote_code fix 4552d8ad5 — first live bug, found+fixed).
  **G2 PASS: "The capital of France is" → top-1 " Paris"** (prefill 0.3s) — the
  full forward (MLA+convs+sinks+mHC+MoE+manifest quant overrides) is
  numerically correct on the real 92B bundle. G3/G5 (greedy coherence +
  multiturn recall) running.
- 2026-07-02 G3/G5: **G5 PASS (multiturn recall "Blue"), Tokyo/Mars PASS,
  33.9 t/s decode** (vs Swift 0.5-1.5). Math collapses (2+2→"2",
  17*23+thinking→zeros-loop) = the documented **no-AWQ 2-bit-expert quant
  signature** (jang_config calibration_method=weights, switch_mlp up=2bit) —
  NOT a runtime bug. Converter-side fix: rebuild with --awq; higher-bit A/B
  is the definitive confirm. Template `thinking` kwarg verified (True→open
  rail, False→pre-closed).
- 2026-07-02 server launch findings (fixed, d1a588487): stamp cache_type=hybrid
  overrode registry kv contract; generic affine-JIT default re-enabled JIT.
  Startup log now shows cache_type=kv/openpangu_v2_composite, jit off. ✓
- 2026-07-02 API matrix round 1: ALL surfaces returned content=None — root
  cause: openPangu template keys on `thinking` not `enable_thinking`; the
  parser-seed probe rendered with enable_thinking (jinja swallows unknown
  kwargs) → always assumed open rail; + 2-bit reasoner never closes </think>.
  FIX (6ff6ac4a5): _normalize_openpangu_thinking (M3 pattern, 3 call sites),
  native-kwarg parser-seed render, openpangu_v2 added to the reasoning-only
  bounded thinking-off answer backstop (chat+responses, stream+non-stream).
  Matrix round 2 running.
- 2026-07-02 API matrix round 2 (post 6ff6ac4a5): **M1a PASS** (content "Tokyo."
  + reasoning preserved, backstop fired), **M1b PASS** (thinking-off yields
  visible content; 2-bit model narrates reasoning-style prose in content and
  runs to cap — quant behavior, noted), **M1c PASS** (streaming: 250 reasoning
  deltas + content deltas, NO think-tag leak, finish=stop), **M2 PASS**
  (/v1/responses "sky" + reasoning block), **M1d PASS** (API multiturn recall
  "Biscuit"). **M4 OPEN**: tool_calls never parse — ROOT CAUSE FOUND: openPangu
  emits a JSON list inside <|tool_call_start|>/<|tool_call_end|> (ids
  148903/148904); the stamped qwen parser cannot match. Next: dedicated
  `openpangu` tool parser + registry entry switch + stamp neutralization
  (task #44). Remaining after that: M3 UI live-chat matrix via dev-build
  (CDP), M5 batching, M6 argv parity, M7 RAM soak, then the release chain
  (rebuild+notarize DMGs with paged-off v7 — still pending from earlier scope).
- 2026-07-02 task #44 LANDED (db123a433 + 73db720d2 + 6dbb27384): dedicated
  `openpangu` tool parser (JSON-list-between-special-tokens; raw-output scan
  so mid-reasoning calls survive think-strip; streaming buffer/emit with
  stable per-call ids per #219; SUPPORTS_NATIVE_TOOL_FORMAT — template
  handles role=tool + tool_calls natively). model_configs openpangu_v2
  tool_parser qwen→openpangu; model_config_registry neutralizes the stale
  sidecar stamp (tp=None next to ct/cst, block hoisted above the generic tp
  application which had run first); panel applyJangCapabilities mirrors the
  neutralization (panel passes --tool-call-parser explicitly, so the engine
  fix alone was not enough for UI sessions). CLI choices +openpangu/
  openpangu_v2. Tests: tests/test_openpangu_tool_parser.py (16) +
  test_openpangu_v2 registry pin updated + VALID_TOOL_PARSERS — 29 pass;
  parser-registry contract + CLI-choice coverage + panel registry vitest
  (69) green. M4 needs live re-verify on the remote box next (unit-proven
  only here; no model runs on max2).

- 2026-07-02 M4 LIVE RETRY (post tool-parser db123a433..5f1b6aef0): startup log
  shows "Auto-configured tool parser from registry: openpangu" (autodetect +
  stamp-neutralization proven live). **STREAMING: tool call emitted after 165
  reasoning deltas and parsed — 1 START chunk, 1 stable id (#219 contract),
  TTFT 0.44s** → tool-calls-mid-reasoning WORKS live. OPEN FOLLOW-UPS:
  (1) finish_reason was "length" not "tool_calls" on the streaming run —
  verify the finish-reason mapping when calls were parsed + whether
  generation should halt at <|tool_call_end|>; (2) non-stream sample emitted
  no call within 700 toks (temp 1.0, 2-bit reasoning rambles) — inspect raw
  output to confirm model-behavior vs extraction before closing M4;
  (3) M5 batching, M6 argv parity (UI manual parser select), M7 RAM soak,
  TTFT formal rows, UI live-chat matrix, then notarize chain.

- 2026-07-02 M4 follow-up (1) FIXED — streaming finish_reason="tool_calls":
  root cause: no server convention existed for terminating the turn when a
  complete tool call parses mid-stream (other families rely on the model
  emitting EOS after the call; degraded 2-bit openPangu keeps narrating to
  max_tokens → final chunk finish_reason="length", #46 finish path never
  fires). FIX = family-agnostic opt-in: ToolParser.STREAM_STOPS_AFTER_
  COMPLETE_CALL + stream_tool_calls_complete()/stream_tool_call_stop_
  truncate() (abstract_tool_parser); openpangu parser opts in (closed
  <|tool_call_start|>[...]<|tool_call_end|> pair IS the end of the turn by
  format contract; a new/partial START after the last END resets the check
  so multi-block turns are never cut). Server: stream_chat_completion +
  stream_responses_api abort generation (engine.abort_request + break, same
  convention as the disconnect path) after an 8-chunk grace window once the
  active parser reports the turn complete, truncate post-call rambling
  (channel-aware: content vs mid-reasoning), then the EXISTING post-stream
  extraction emits the #46 tool_calls data + finish chunks
  (finish_reason="tool_calls") + [DONE]. All other parsers keep default
  False — zero behavior change for families with interleaved/sequential
  post-call output. Non-stream path confirmed already correct
  (finish_reason = "tool_calls" if tool_calls, family-agnostic) — untouched.
  Tests: tests/test_openpangu_tool_parser.py +10 (parser contract, resolver,
  e2e fake-engine stream regression: complete call + 40 ramble chunks →
  abort fired, stream not drained, finish_reasons==["tool_calls"], START id
  reused, no ramble leak, [DONE]) — 25/25; adjacent: tool parser suites +
  streaming/reasoning/server/native-format/registry 600+ pass (pre-existing
  only: test_step37_flash_jang_config + 12 test_engine_audit failures proven
  identical at HEAD via stash-diff). Needs live re-verify on remote next.

- 2026-07-02 M4 CLOSED LIVE: streaming tool call → finish_reason=["tool_calls"] only, get_weather parsed (START+data chunks, stable id), early-stop fired (wall 26.5s, no length overrun) on JANG_2L @ :8003 at abb8cc29f. Remaining rows: M5 batching, M6 UI argv parity + manual parser select, M7 RAM soak, UI live-chat matrix, notarize chain.
