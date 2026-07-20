# Laguna real-UI soft-sleep/wake soak

Date: 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` for three consecutive real Electron soft-sleep
and Wake cycles on current head `6af4831f3`. Deep sleep, cross-model swap soak,
and signed-app repetition remain open.

## Source ownership

- `panel/src/renderer/src/components/sessions/SessionCard.tsx:310-323` owns
  visible Wake, while lines 356-364 own the moon sleep action.
- `panel/src/main/ipc/sessions.ts:315-337` routes those actions to the session
  manager.
- `panel/src/main/sessions.ts:2965-2993` accepts soft sleep only for a running
  local session and persists `standby/soft` after the engine succeeds.
- `panel/src/main/sessions.ts:3025-3064` accepts Wake only from standby and
  returns the session to the health-monitored loading path.

## Live Electron proof

The already-loaded `jangq-ai/Laguna-M.1-JANG_2L` process began in soft standby
as PID 70292. Three times in succession, the rendered Wake button was clicked,
DB became `running` with no standby depth, and engine health became
`healthy/model_loaded=true`. Each corresponding rendered moon button was then
clicked, DB returned to `standby/soft`, and health returned to
`standby_soft/model_loaded=true`.

All six transitions retained PID 70292. Each wake reached healthy within the
bounded live poll (3.8-6.1 seconds observed), and each soft sleep reached
standby in under two seconds. At the final boundary, process inspection showed
exactly one `vmlx_engine.cli serve` process. The engine/app logs contain three
`Woke from soft sleep` pairs and three `Entered soft sleep` pairs; see
`02-sleep-wake-log.json`.

The final real dashboard shows the single Laguna card in soft standby with a
visible Wake control. See `01-cycle3-standby-ui.png`. Final SQLite, health, and
process truth are preserved in `03-final-session.json`,
`04-final-health.json`, and `05-engine-process.txt`.

No inference request ran in this gate, so it makes no output-streaming or cache
reuse claim. Soft sleep intentionally clears caches while leaving the model
loaded.

## Remaining boundary

Exercise deep sleep/model unload and repeated cross-model one-model swaps, then
repeat the lifecycle in the signed app. Those are not inherited from this
same-process soft-sleep soak.
