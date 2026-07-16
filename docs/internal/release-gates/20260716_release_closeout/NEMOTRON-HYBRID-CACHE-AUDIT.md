# Nemotron-H hybrid cache and stream audit — 2026-07-16

Status: `PASS-LIVE` for cache/settings/tools/API; `PARTIAL` for long-reasoning
reliability. Global status remains `PARTIAL_NO_RELEASE`.

## Artifact and architecture trace

- Bundle: `/Volumes/EricsLLMDrive/dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK`.
- Real config declares `model_type=nemotron_h`, 52 layers, and
  `hybrid_override_pattern=MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`.
  The six `*` slots are attention KV; 23 Mamba slots and 23 expert/no-state
  slots are not generic KV caches.
- `vmlx_engine/utils/hybrid_tq_cache.py` derives eligibility from the actual
  hybrid layout and wraps only `layer_type == "attention"` slots in
  `TurboQuantKVCache`.
- `vmlx_engine/utils/ssm_companion_cache.py` owns Nemotron-H native companion
  checkpoints and asynchronous clean-prefill restore/rederive.
- `vmlx_engine/cli.py` maps Auto to correctness-first storage-only TQ8 for the
  compatible attention KV lane while leaving SSM state native. The model
  registry supplies `nemotron` tool parsing and `deepseek_r1` reasoning.
- The cold write made exactly 42 TQ storage encodes: six eligible attention
  slots across seven persisted blocks. This matches the real layer graph and
  rules out name-wide wrapping.

## Live Electron evidence

| Gate | Current evidence | Verdict |
|---|---|---|
| Settings and launch parity | Expanded Electron Settings visibly showed prefix cache, paged cache, block size 64, max blocks 1,000, block-disk L2 10 GB, KV Auto, auto tool choice, `nemotron`, and `deepseek_r1`. PID 75939 argv and health match those controls. | PASS |
| Single-model swap | Starting Nemotron in Sessions changed the UI to Active (1), made ZAYA inactive, and left exactly one serve process. | PASS |
| Cold and same-chat multi-turn | Rows 2223 and 2226 each called one real `file_info`, persisted its result, and returned exact `NEMO-LIVE1-DONE` / `NEMO-LIVE2-DONE`. Turn two restored 162 tokens as `paged+ssm+tq-native`. | PASS |
| Process restart and L2 | PID 74652 row 2229 restored 192 tokens as `paged+ssm+disk+tq-native`, called one tool, and returned the exact final. | PASS |
| Forced bounded eviction | Electron set max blocks to four and restarted PID 75038. Rows 2235/2238 remained exact, evictions rose from 3 to 9, and all three usable blocks returned to the free queue. | PASS |
| Explicit None | Electron selected None. PID 75398 argv carried `--kv-cache-quantization none`; row 2241 wrote raw blocks with zero TQ activity. PID 75644 row 2244 restored 156 tokens as `paged+ssm+disk`, still with zero TQ activity. | PASS |
| Auto restoration | Electron restored Auto and max blocks 1,000. PID 75939 argv and health report selective storage-only TQ8 and 999 free blocks at clean start. | PASS |
| Long visible answer | Row 2247 visibly completed twelve coherent numbered lines and exact `NEMO-LONG1-DONE`. It generated 2,962 reasoning tokens and repeated answer drafting before emitting its real `</think>`. Content and reasoning stayed separated and the answer did not truncate. | PARTIAL reliability |
| Responses streaming | `nemotron-responses-stream.json` contains 424 reasoning deltas, 30 output-text deltas, matching reasoning/text done events, and `response.completed(status=completed)` with exact marker text. | PASS |

The bundle itself defaults thinking on and opens the native `<think>` rail;
the configured parser matches that contract. Because the long row eventually
emitted the real closer and a complete visible answer, the repeated internal
drafting is retained as a model/runtime reliability observation, not hidden by
a forced closer, sampler clamp, prompt coercion, or synthetic continuation.

## Focused tests and artifacts

The current source passed 25 focused Nemotron/selective-attention-TQ tests from
`test_model_inspector.py`, `test_model_config_registry.py`,
`test_tool_parsers.py`, `test_streaming_reasoning.py`,
`test_impl_campaign_20260710.py`, and `test_engine_audit.py`.

Screenshots and the raw Responses stream are in `nemotron-current/`. These
artifacts close this family’s cache, settings, tool-continuation, and API
streaming rows. They do not close the global release or the remaining repeated
long-reasoning reliability row.
