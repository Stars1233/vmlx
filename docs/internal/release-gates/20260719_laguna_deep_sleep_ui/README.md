# Laguna deep-sleep unload and wake gate

Date: 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` for real Electron power-policy configuration,
automatic deep sleep, visible deep-standby state, in-process Wake reload, and
settings restoration on current head `1924b3ea4`. Cross-model swap soak and
signed-app repetition remain open.

## Source ownership

- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx:1240-1278`
  exposes Auto-Sleep plus separate light/deep minute controls.
- `panel/src/main/sessions.ts:2995-3023` calls `/admin/deep-sleep`, then persists
  `standby/deep` only after the engine succeeds.
- `panel/src/main/sessions.ts:3025-3064` wakes a standby session and returns it
  to the existing health-monitored load path without replacing the process.
- `panel/src/renderer/src/components/sessions/SessionView.tsx:380-399` and
  `SessionCard.tsx:310-323` render Deep Sleep and the visible Wake action.

## Live policy and unload proof

The real session settings UI temporarily set Light Sleep to `0` (disabled) and
Deep Sleep to `1` minute; `01-deep-policy-settings-ui.png` shows the actual
numeric controls. The already-idle Laguna session then automatically entered
deep sleep through the configured policy. The engine/app log records:

`Entered deep sleep — model unloaded, process alive`

At the captured deep boundary:

- Electron visibly rendered `Deep Sleep` and Wake while retaining PID 70292
  (`05-deep-dashboard-ui.png`);
- SQLite stored `standby/deep` with PID 70292 (`02-deep-session.json`);
- health reported `standby_deep` and `model_loaded=false`
  (`03-deep-health.json`);
- process inspection showed the same single serve process still alive
  (`04-deep-process.txt`).

The visible Wake action reloaded the model in the same PID. A second explicit
Electron deep-sleep snapshot was used only to preserve the raw deep-state
artifacts; its immediate Wake completed in about 39 seconds according to the
engine log. Both deep sleeps and both reloads are retained in
`06-deep-wake-log.json`.

## Restoration

The real settings UI restored Light/Deep timeouts to `10`/`30`. After the final
Wake, the rendered moon action returned Laguna to its original soft-standby
boundary. Final SQLite and health report `standby/soft`, PID 70292,
`standby_soft`, and `model_loaded=true`; exactly one engine process remains.
See `07-final-restored-session.json`, `08-final-restored-health.json`, and
`09-final-process.txt`.

No inference request ran in this gate, so it makes no output-streaming or cache
reuse claim. Deep sleep intentionally unloads the model and clears its resident
runtime state.

## Remaining boundary

Repeat unload/reload across another loader class, exercise repeated one-model
cross-model swaps, and repeat in the signed app. Those are not inherited from
this Laguna same-process lifecycle proof.
