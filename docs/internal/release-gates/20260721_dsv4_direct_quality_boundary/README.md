# DSV4 direct-model quality boundary and current-head stream proof

Date: 2026-07-21
Host: `erics-m5-max.local`
Repository: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
Branch: `codex/postrelease-ui-drawers-20260720`
Source: `6081b1da4b04830fd6b9a237530f2da6421ad2fb`

## Scope and artifact identity

This is a targeted follow-up to
`20260720_dsv4_long_context_snapshot_budget_current/`. It does not repeat the
already-passing resident/restart cache matrix. The exact artifact is
`/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`: affine JANG
using MLX affine quantized matmul, not JANGTQ/MXTQ and not base MLX MXFP.

Current `/health` identifies the architecture-owned `deepseek_v4_v8` cache:
43 layers comprising two local SWA layers, 21 ratio-4 CSA compressed-pool
layers, and 20 ratio-128 HCA compressed-pool layers. The native pool codec is
enabled and generic TurboQuant KV is forced off. This gate never substitutes a
generic KV/TQ cache for that composite state.

## Direct-model controlled A/B

`direct-model-ab.py` loads the official artifact with the current
`load_jang_model` implementation, renders the exact SQLite row-225 Auto prompt,
and calls `mlx_lm.generate` directly. The server, API adapters, parser,
scheduler, paged RAM cache, and block-disk L2 are absent. Generic TQ KV is also
disabled. Before generation, the bundle encoder and tokenizer
`apply_chat_template(enable_thinking=True)` produced byte-identical 28,780-byte
prompts with SHA-256
`6c8923b79a64fcd72324cd9e442248c536ceb06d044f097ad07caefc5cf275de`
and 7,879 tokens; `prompt-render-equality.json` preserves that comparison. The
current stdout results are transcribed in
`direct-model-ab-summary.json`; the full raw completions were not persisted, so
that limitation is explicit.

All three 7,879-token direct runs failed to close `</think>` or produce the
requested terminal marker within 512 output tokens:

- native pool codec on, bundle stochastic defaults (`0.6/.95`): looped planning
  and hallucinated record values;
- native pool codec on, greedy (`0/1`): recovered both real secrets but repeated
  its plan and never reached visible content;
- native pool codec off, greedy (`0/1`): reproduced the same 1,526-character
  looping completion as the pool-on greedy arm.

This proves the retained long/medium Auto-quality failure is not created by
Electron painting, Responses finalization, prompt/paged/L2 reuse, generic TQ,
stochastic sampling, or the native DSV4 pool codec. It does **not** prove an
artifact defect: the same affine artifact has not yet run in an independent
architecture implementation. The legacy standalone JANG loader still rejects
the sidecar because it expects the removed top-level `format` key.

## Current-head live Electron boundary

The real Electron Sessions Start control launched DSV4 PID 76078 and stopped
the prior model under Single Model mode. Before the first request,
`dsv4-current-health-before.json` recorded `model_loaded=true`,
`last_request_time=null`, and 99,724.7 MB active model memory. The Chat Settings
drawer visibly matched the bundle-owned Auto defaults: temperature 0.60, top-p
0.95, top-k Off, min-p 0, repetition 1.00, and model-default maximum 4096. Only
a bounded 1024-token per-chat ceiling was set for this short probe; tools were
Off.

The Electron prompt exact-finaled
`DSV4-CURRENT-BOUNDARY-DONE VALUE=45`. The DOM trace recorded 40 distinct
states: reasoning grew progressively from 3 to 235 displayed characters, then
visible content grew through `DSV`, `DSV4-C`,
`DSV4-CURRENT-BOUNDARY`, and the exact final. SQLite row 284 contains separate
reasoning and content, 81 output tokens, no tool call, and no warning. The final
screenshot retains the live drawer and output together.

## Raw Responses stream

The already Electron-loaded server received a separate short raw
`/v1/responses` request with omitted thinking control, so the bundle's Auto
default owned the rail. `dsv4-current-api-sse.json` records:

- 44 timed `response.reasoning_summary_text.delta` events, first at 1,189 ms;
- 12 timed `response.output_text.delta` events, first at 3,330 ms;
- exact content `DSV4-CURRENT-API-DONE VALUE=42`;
- one `response.completed` terminal at 3,878 ms.

`dsv4-current-health-after-streams.json` then identifies the owning
`DSV4BatchGenerator`, the affine JANG codec/profile, native composite cache,
native pool codec On, paged and block-L2 On, and generic TQ Off. These short
prompts are below one 256-token cache block, so this gate makes no cache-hit
claim; the prior long gate remains the cache evidence owner.

## Source and dead-code trace

- `vmlx_engine/loaders/load_jangtq_dsv4.py` owns the mandatory long-context and
  pool-codec activation for the current DSV4 loader.
- `vmlx_engine/utils/dsv4_batch_generator.py`, `vmlx_engine/scheduler.py`, and
  `vmlx_engine/prefix_cache.py` own the DSV4 typed snapshot/store/restore path.
- `vmlx_engine/server.py::_resolve_dsv4_thinking_policy` maps omitted Auto to
  the bundle default; its `tools_present` and `tool_choice` parameters are
  currently forwarded by five production call sites but are not read in the
  resolver. They are recorded as dead parameter surface. They were not removed
  in this evidence-only checkpoint because that source change would require a
  new engine restart and behavioral rerun; removal should be paired with the
  next DSV4 code change rather than triggering a duplicate model campaign.
- Historical CHANGELOG descriptions are release history, not active runtime
  branches, and were not rewritten.

## Verdict

- Short Auto reasoning/content/terminal streaming: **PASS-LIVE scoped** in
  Electron and raw Responses on current source.
- Long/medium Auto quality: **FAIL/PARTIAL**, reproduced below the API/cache
  layers in direct model generation.
- Artifact-versus-shared-architecture root cause: **PARTIAL/BLOCKED** until an
  independent DSV4 implementation can consume this affine bundle or a matched
  unquantized/reference artifact is available.

No model-specific output rewrite, prompt coercion, hidden sampler clamp, or
generic-cache substitution was added.

## Evidence inventory

- `direct-model-ab.py`
- `direct-model-ab-summary.json`
- `prompt-render-equality.json`
- `dsv4-current-boundary-ui-trace.json`
- `dsv4-current-boundary-ui.png`
- `dsv4-current-api-sse.json`
- `dsv4-current-health-before.json`
- `dsv4-current-health-after-streams.json`
- `electron-assistant-row-284.json`
