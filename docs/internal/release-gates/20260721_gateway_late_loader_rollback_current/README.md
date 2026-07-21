# Gateway late-loader rollback and error-truth gate — 2026-07-21

## Verdict

`VERIFIED-LIVE_SCOPED` for a target that passes the local-path preflight but
fails during actual engine loading while gateway Single Model mode is enabled.
The current Electron main process restores the displaced healthy backend,
returns `model_load_failed` with the loader's real diagnostic, and leaves the
restored route usable through both raw gateway streaming and the real Chat UI.

This gate used a tiny, explicitly synthetic invalid safetensors fixture under
`/private/tmp`. It did not alter, distrust, or infer a defect in any official
JANG, JANGTQ/MXTQ, or base-MLX artifact. It does not promote any architecture,
model-quality, parser, media, or cache-matrix row.

## Retained pre-fix behavior

The synthetic target had a valid model directory and readable `config.json`,
so it passed the non-mutating preflight added by the earlier failed-target
transaction repair. Its deliberately invalid `model.safetensors` failed only
when the engine loader parsed the file.

On the old boolean JIT-load contract, the real gateway request failed in
9.949086 seconds but was falsely returned as:

```text
code=model_load_timeout
Model 'codex-late-load-fail-20260721' failed to load within 120s
```

The displaced Nemotron backend was restored, proving the transaction branch
executed, but the public diagnosis still collapsed an immediate loader error
into a deadline timeout. `02-old-code-restored-false-timeout.png` retains the
visible restored Electron state from that run.

## Root cause and source repair

`panel/src/main/api-gateway.ts::_doJitLoad()` returned only `boolean`.
`false` represented three materially different outcomes:

1. `startSession` or wake threw immediately;
2. the session entered the engine `error` state before readiness;
3. the real 120-second JIT deadline expired.

`prepareSessionForRouting()` therefore had no truthful outcome to expose and
always labeled `false` as `model_load_timeout`.

Current source replaces that lossy branch with one typed `JitLoadResult`:

- `ready`
- `load_failed` plus loader detail
- `timeout`

Only the deadline path now produces `model_load_timeout`. Start/wake failures
and an early engine error produce `model_load_failed`; the same displaced
session rollback runs for either failure class. The obsolete boolean result
and its inference branch were removed rather than retained as compatibility or
zombie code. `source-diff.patch` contains the scoped source/test change summary;
the commit itself remains the authoritative exact diff.

## Current-source Electron transaction proof

The Electron development app was fully relaunched from this checkout using
`/Users/eric/.vmlx-v1613-responsive-dev` and CDP 9335. The real UI `Start`
control loaded `dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK` before any request;
the main-process log printed:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

The gateway then stopped Nemotron for the valid-path synthetic target. The
target failed inside `load_safetensors`, and the current request returned HTTP
503 in 7.563482 seconds with the truthful surface:

```text
code=model_load_failed
Model 'codex-late-load-fail-20260721' failed to load: Process exited before becoming ready: RuntimeError: [load_safetensors] Invalid json header length file /private/tmp/codex-late-load-fail-20260721/model.safetensors
```

Rollback restored Nemotron as the sole running backend at PID 79320; the
synthetic session remained `error` with no PID. Gateway health reported
`single_model_mode=true` and `active_requests=0`; model health reported
`model_loaded=true` and `last_request_time=null`. The real Sessions screenshot
`03-current-restored.png` visibly shows the restored running session.

## Raw stream and visible UI recovery

A raw Chat Completions request through the Electron-owned gateway and restored
Nemotron process emitted 16 separate content deltas that progressively formed
`GATEWAY-LATEFAIL-RECOVERY-DONE`, then a `finish_reason=stop` chunk and
`[DONE]`. Usage was 37 prompt, 17 completion, 54 total tokens. See
`gateway-recovery.sse`.

The first UI recovery attempt is retained as a negative control. Automation
clicked a generic Off control rather than the Off control inside the exact
`Enable Thinking` section. SQLite proved `enable_thinking` was still inherited;
the 64-token turn stored 254 reasoning characters, empty visible content, and
reasoning-only plus truncation warnings. This is not represented as a product
fix or model failure. See `04-ui-auto-cap-negative.png` and
`ui-negative-control.json`.

The correction selected Off in the exact `Enable Thinking` section and Save;
SQLite then showed `enable_thinking=0`, `builtin_tools_enabled=0`, and
`max_tokens=64`. The same-chat follow-up produced 53 observed DOM states whose
visible suffix grew through `GATEWAY-LA`, `GATEWAY-LATEFAIL`,
`...RECOVERY2-D`, and finally exact
`GATEWAY-LATEFAIL-UI-RECOVERY2-DONE`. The terminal row had non-empty content,
no reasoning, warnings, or tools, and reported 32
`paged+ssm+tq-native` cached tokens. This is route-restoration and UI-streaming
proof only; the cache detail is observed telemetry, not a new cache gate.
`06-ui-clean-final.png` visibly retains the complete final answer and metrics
after the settings drawer was closed, without running another generation.

## Validation and retained boundaries

- `focused-validation.txt`: 29/29 focused gateway behavior tests passed and
  panel typecheck completed successfully.
- The added tests distinguish an immediate loader failure from an explicitly
  stubbed deadline exhaustion and pin rollback for both.
- `git diff --check` was clean before staging.
- Longer concurrent load/swap/failure soak, a signed packaged-app repeat, and
  an installed-app OS-port collision remain separate open rows.
- The synthetic fixture and disposable session are removed after evidence
  capture; `07-cleanup-active-only.png` visibly shows one active Nemotron and
  no synthetic target after renderer reload. Official model directories are
  unchanged.
