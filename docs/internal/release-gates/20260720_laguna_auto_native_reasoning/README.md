# Laguna native Auto reasoning and answer-pass gate

Date: 2026-07-20

Repository: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Bundle: `/Users/eric/.mlxstudio/models/JANGQ-AI/Laguna-XS.2-JANGTQ`

## Verdict

The scoped Laguna reasoning regression is `PASS` on the source under test. The broader Laguna family and release matrix remain `PARTIAL` because this gate does not cover Laguna tools, long-context soak, process-restart L2 restoration, eviction, or every protocol.

The bundle is JANGTQ/MXTQ, profile `JANGTQ2`: routed experts are 2-bit and attention/shared/embed/lm-head tensors are 8-bit. It is not affine JANG and not base MLX MXFP.

## Root causes

1. Auto reasoning policy contradicted the real bundle template. The template uses `enable_thinking | default(false)`, while the Python and Electron registries forced Laguna Auto to `true`. A live Auto turn then generated 1,978 reasoning tokens and repeated `tenant` until manual interruption.
2. The bounded visible-answer pass replayed the truncated reasoning assistant turn. Laguna's template did not faithfully replay that turn, so the direct pass continued planning in visible content. Laguna now uses the original-message fresh-context answer pass.

This is an integration/policy fix. The official quantized artifact was not blamed or changed.

## Source trace

- `vmlx_engine/model_configs.py`: Laguna keeps reasoning support and the `qwen3` parser, but its authoritative `default_enable_thinking` is now `False`.
- `panel/src/main/model-config-registry.ts`: the Electron-side family registry uses the same native-off Auto default.
- `vmlx_engine/server.py`: Laguna is included in `_ANSWER_PASS_FRESH_CONTEXT_FAMILIES`; explicit On/Off continues to override Auto.
- `tests/test_model_config_registry.py`, `tests/test_reasoning_modes.py`, `tests/test_answer_pass_families_dsv4_step37.py`, and `panel/tests/model-config-registry.test.ts` pin those contracts.

## Live evidence

### Failure reproduction

- Real Electron app, CDP `127.0.0.1:9335`, real Start button, Laguna PID under the app.
- Auto reasoning produced 5,885 progressive DOM updates but no visible answer. It was manually interrupted at 1,978 tokens after 65.9 seconds.
- Persisted reasoning entered a long `tenant tenant ...` loop.
- Evidence: `laguna-auto-tenant-loop.png`, `laguna-auto-dom-trace-summary.json`, and `failure-summary.json`.

This proves the UI was progressively painting deltas; it was not a renderer batching/freeze failure.

### Same-artifact reference A/B

- Direct `jang_tools.laguna.runtime`, exact prompt, greedy: completed at 302 tokens with EOS and the requested marker.
- Sampled `temperature=0.7`, `top_p=0.9`, seeds 0/1/2: all completed at 255/281/420 tokens with the requested marker and no `tenant` repetition.
- Evidence: `laguna-reference-direct.log` and `laguna-reference-sampled.log` (runtime diagnostics followed by JSON records).

### TurboQuant isolation

- Controlled same-loaded-model 64-token greedy comparison between native cache and the mixed 10x `TurboQuantKVCache` plus 30x rotating-cache layout produced identical token arrays.
- `max_abs_logit_diff=0.0`; `first_token_difference=null`.
- Evidence: `laguna-tq-numeric.log` (runtime diagnostics followed by the JSON result).

This proves the tested cold/live `compress_after=0` cache objects did not cause the deterministic failure. It does not by itself close process-restart restore or eviction.

### Patched Auto, real Electron

- Model started from the real Electron UI from current source.
- Chat override remained Auto (`enable_thinking=null`), tools disabled.
- Server resolved `enable_thinking=False`.
- Visible answer was non-empty and ended with `LAG-AUTO-NATIVE-UI1-DONE`.
- 36 output tokens, 32.2 tok/s, 91 prompt tokens, 0.22 s TTFT, 1.4 s total.
- DOM observer: 155 snapshots and 124 distinct rendered lengths.
- Evidence: `laguna-auto-native-ui-pass.png`, `laguna-auto-native-ui-row.json`, and `ui-stream-summary.json`.

### Patched explicit On, raw Responses API

- `max_thinking_tokens=256`, seed 0.
- Post-fix stream: 167 reasoning deltas and 132 content deltas; 772 reasoning characters and 582 content characters.
- Terminal `response.completed` present; no failed event; no `tenant` repetition.
- Visible content begins directly with the numbered answer rather than the pre-fix mid-planning prefix.
- Evidence: `api-stream-summary.json`.

### Patched explicit On, real Electron

- Settings changed in the UI to explicit On and max thinking 256.
- Persisted override: `enable_thinking=1`, `max_thinking_tokens=256`, built-in tools disabled.
- Server request resolved `enable_thinking=True` and `thinking_budget=256`.
- UI showed a separate 488-character reasoning rail and a non-empty four-line answer ending in `LAG-EXPLICIT-UI1-DONE`.
- 148 output tokens, 31.6 tok/s, 177 prompt tokens, 90 cached tokens, cache detail `paged+disk+tq-native`, 0.25 s TTFT, 5.0 s total.
- DOM observer: 676 snapshots and 526 distinct rendered lengths.
- Evidence: `laguna-explicit-ui-pass.png`, `explicit-ui-row.json`, and `ui-stream-summary.json`.

## Tests

- Python focused set after both fixes: 219 passed (`test_answer_pass_families_dsv4_step37.py`, `test_model_config_registry.py`, `test_reasoning_modes.py`).
- Panel registry suite: 85 passed.
- Panel TypeScript typecheck: passed.

## Remaining Laguna rows

- `PARTIAL`: tool-call loop and post-tool continuation on current source.
- `PARTIAL`: three-turn history/recall with explicit reasoning.
- `PARTIAL`: process-restart L2 restoration, partial-prefix reuse, and eviction on current source.
- `PARTIAL`: long-context reliability and latency soak.
- `PARTIAL`: Chat Completions, Anthropic, and Ollama parity for this exact fix.
