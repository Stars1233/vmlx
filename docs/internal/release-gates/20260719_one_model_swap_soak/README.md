# One-model Electron Start swap soak — 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` at source head
`677c408507fdba5d24994ba2a8baa706d7c1bbce`.

This gate used the real dev Electron Sessions dashboard over CDP 9335. It did
not invoke the session IPC directly and did not send an inference request.

## Source and artifact truth

- `panel/src/main/sessions.ts:1542-1592` serializes manual Start transitions
  when `gateway_single_model_mode=true`, stops every other running/loading/
  standby local session first, and fails closed if the old engine cannot stop.
- `panel/src/main/engine-manager.ts:177-203` resolves the installed engine from
  PATH. The live Electron main log contains:
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- MiniMax M2.7 bundle:
  `config.json` declares `model_type=minimax_m2`, 62 layers, 196608 context;
  `jang_config.json` declares `weight_format=mxtq`, profile `JANGTQ2`, routed
  expert 2-bit and attention/dense/embed/head 8-bit.
- Laguna bundle: `config.json` declares `model_type=laguna`, 70 layers, 262144
  context. The API parser pair is intentionally `reasoning=qwen3` and
  `tool=glm47`: `vmlx_engine/model_configs.py:215-252` documents the `<think>`
  reasoning rail and Laguna's `<arg_key>/<arg_value>` tool XML. The bundle's
  `poolside_v1` name is not treated as an installed server parser identifier.

## Live sequence

Initial state was Laguna soft standby, PID 70292, with MiniMax M2.7 stopped.
Single-model mode was live in gateway health.

| Step | Real UI action | New active PID | Prior engine | Pre-request health |
|---|---|---:|---|---|
| 1 | MiniMax M2.7 `Start` | 78868 | Laguna stopped, 8015 down | healthy, loaded, `last_request_time=null` |
| 2 | Laguna `Start` | 79430 | M2.7 stopped, 8014 down | healthy, loaded, `last_request_time=null` |
| 3 | MiniMax M2.7 `Start` | 80033 | Laguna stopped | healthy, loaded |
| 4 | Laguna `Start` | 80479 | M2.7 stopped | healthy, loaded, `last_request_time=null` |

After every step, SQLite and `ps` showed exactly one local
`vmlx_engine.cli serve` process. Electron main-log lines 474, 481, 532, and
568 preserve each stop-before-start transition. MiniMax M2.7 health identified
JANGTQ2/TurboQuant weights separately from affine JANG, stored-prefix
`turboquant-q4`, paged RAM enabled, and block-disk L2 enabled. Laguna health
identified affine `JANG_2L`, the `glm47`/`qwen3` argv, stored-prefix
`turboquant-q4`, paged RAM enabled, and block-disk L2 enabled.

The final real moon control returned Laguna PID 80479 to
`standby/soft`; health reported `standby_soft` and `model_loaded=true`.

## Visual evidence

- `one-model-swap-before.png` — initial Laguna soft standby.
- `one-model-swap-m27-1.png` — M2.7 is the only running session, PID 78868.
- `one-model-swap-laguna-1.png` — first reverse swap.
- `one-model-swap-laguna-2.png` — second reverse swap, Laguna PID 80479.
- `one-model-swap-final-soft.png` — final restored Light Sleep state.

## Boundary

This is a lifecycle/eager-materialization proof only. No generation ran, so it
makes no output-streaming, reasoning, tool-loop, prefix-hit, eviction, or L2
restore claim. Those remain owned by their model/protocol/cache gates. The
signed packaged-app repetition also remains open.
