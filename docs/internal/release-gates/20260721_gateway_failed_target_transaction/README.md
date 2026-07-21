# Gateway failed-target transaction gate — 2026-07-21

## Verdict

`VERIFIED-LIVE_SCOPED` for the reproducible stale/missing local-model target.
The gateway now rejects that target before Single Model mode unloads the healthy
backend, returns a truthful preflight error, and leaves the existing Electron-
started backend loaded. A failure that occurs only after a target passes
preflight remains `PASS-SOURCE+CONTRACT / PARTIAL-LIVE`: rollback is implemented
and covered by the focused behavior test, but this gate did not fabricate a
late loader failure in a valid official bundle.

## Retained pre-fix failure

- A disposable saved session named `codex-missing-swap-target` pointed at the
  confirmed-absent path
  `/Users/eric/.mlxstudio/models/Codex-Missing-Swap-Target-20260721`.
- The real Electron Sessions page showed one active DSV4 backend and the
  missing target. Routing a gateway Chat Completions request to the missing
  model first stopped the active DSV4 session and only then discovered the
  missing path. SQLite recorded DSV4 as stopped with no PID and no engine
  process remained. `prefix-failure-stranded-ui.png` is the visible stranded
  state.
- The old API error was also false: it reported `model_load_timeout` and
  "failed to load within 120s" for an immediate local path rejection. The
  0.054267-second timing in `prefix-summary.json` is from the retained repeat
  after DSV4 had already been stranded; it is not represented as the timing of
  the destructive first transition.

## Source repair

- `panel/src/main/sessions.ts`
  - `preflightSessionStart` performs the non-mutating engine/path/format check
    before Single Model replacement.
  - `validateLocalSessionTarget` is the one shared validation owner. The old
    inline duplicate was removed; `_startSessionInner` repeats the same helper
    immediately before spawn to close filesystem races.
  - Direct UI/session starts also preflight before stopping other local models.
- `panel/src/main/api-gateway.ts`
  - `prepareSessionForRouting` preflights before `enforceSingleModelMode`.
  - It records the displaced running/standby backend and restores it if JIT
    loading fails after a successful preflight.
  - All gateway surfaces receive structured `model_preflight_failed` versus
    `model_load_timeout` diagnostics instead of converting both cases to a
    timeout.
- `panel/tests/api-gateway-single-model.behavior.test.ts`
  - Pins no stop/start calls on preflight rejection.
  - Pins restoration of the displaced backend after a later target-start
    failure.

The active-source search found one owner of the missing-model diagnostic and
only production call sites for the new helpers. This change removes duplicated
validation rather than adding a compatibility/zombie branch. See
`source-diff.patch`.

## Current-source live proof

The Electron app was fully relaunched from this checkout with CDP 9335 and the
dev log printed:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

The real Electron `Start` button eagerly loaded
`dealignai/DeepSeek-V4-Flash-JANG-CRACK` on port 8002 as PID 80040. The final
missing-target request then produced:

```text
HTTP 503 in 0.030499s
code=model_preflight_failed
Model 'codex-missing-swap-target' cannot be loaded: Model not found at: ...
```

Before and after SQLite records are byte-equivalent for the relevant state:
DSV4 remains `running`, PID 80040; the disposable target remains `stopped` with
no PID. `postfix-before-health.json` and `postfix-after-health.json` are exactly
equal and report the same healthy, loaded DSV4 backend. The Electron screenshot
`postfix-preserved-active-ui.png` visibly shows `Missing model (1)` beside
`Active (1)` DSV4 after the failure. The main-process log contains the preflight
rejection and no unload/JIT-start line for the bad target.

A short gateway stream through the surviving DSV4 process then emitted 256
incremental reasoning chunks, ten incremental content chunks, a stop chunk,
and `[DONE]`; the trace shows first reasoning at 04:25:52.158, first content at
04:26:03.577, and terminal at 04:26:04.061. This proves the preserved route was
still usable and streaming. It does **not** promote DSV4 exact-output quality:
the visible content was `GATEWAY-R-RECOVERY-DONE`, not the requested
`GATEWAY-RECOVERY-DONE`, consistent with the separately retained DSV4 Auto
quality boundary.

## Validation and evidence

- `focused-tests.txt`: 36 passed (28 gateway lifecycle plus eight session path
  recovery tests).
- `typecheck.txt`: panel TypeScript check completed successfully.
- `diff-check.txt`: empty, meaning `git diff --check` found no whitespace error.
- `postfix-before-state.json`, `postfix-after-state.json`: unchanged PID/status.
- `postfix-before-health.json`, `postfix-after-health.json`: unchanged healthy
  backend.
- `postfix-current-processes.txt`: exactly one DSV4 engine command at PID 80040.
- `postfix-error-body.json`: truthful current-source error surface.
- `surviving-route-recovery.sse` and
  `surviving-route-recovery-trace.txt`: raw progressive gateway stream.
- `prefix-failure-stranded-ui.png` and
  `postfix-preserved-active-ui.png`: retained pre-fix and current-source real
  Electron UI states.
- `fixture-cleaned-active-ui.png`: after evidence capture, the disposable
  session was removed from SQLite, its path remained absent, and a renderer
  reload visibly retained only `Active (1)` DSV4 plus the original sessions.

## Remaining boundary

- `PARTIAL-LIVE`: inject a real post-preflight loader failure without mutating
  or distrusting an official bundle, then confirm the previous backend is
  restored in the Electron UI. The restoration branch is source-tested now.
- Longer concurrent load-failure/model-swap soak and signed packaged-app
  repetition remain separate matrix rows.
