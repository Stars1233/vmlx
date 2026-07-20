# Stale local model-path recovery UI gate

Date: 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` for missing-path classification, safe action
surface, native-chooser repoint, and explicit removal in the real Electron
dashboard on current head `fd83128c8`. Signed-app repetition remains open.

## Source ownership

- `panel/src/main/session-model-path.ts:33-68` classifies local paths against
  current filesystem truth without silently rewriting or pruning SQLite.
- `panel/src/renderer/src/components/sessions/SessionCard.tsx:272-299` replaces
  Start/Delete with Repoint/Remove whenever `modelPathMissing` is true.
- `panel/src/main/ipc/sessions.ts:178-244` validates the selected bundle and
  requires confirmation when model identity changes.
- `panel/src/main/sessions.ts:2459-2511` rejects running/duplicate-path unsafe
  repoints and performs the explicit database/history rebind.

## Live Electron proof

A disposable stopped local-session fixture pointed at the deliberately absent
directory
`/Volumes/EricsLLMDrive/__vmlx_missing_model_path_live__/Laguna-M.1-JANG_2L`.
The real dashboard classified it under `MISSING MODEL (1)`, displayed the
unavailable-directory warning and usable-twin hint, and exposed exactly
`Repoint model path` and `Remove session` in the card viewport. No Start control
was present. See `01-missing-path-ui.png`.

The rendered `Remove session` button was clicked with its confirmation accepted.
The card disappeared from the live dashboard and the fixture row disappeared
from SQLite. The unrelated active Laguna process remained PID 70292, and its
post-removal health still reported `standby_soft`, `model_loaded=true`, and model
`jangq-ai/Laguna-M.1-JANG_2L`. See `02-after-remove-ui.png`.

The fixture was temporary and is no longer present. No model was started or
stopped by this gate.

The fixture was then recreated and the real `Repoint model path` button opened
the native macOS directory chooser. Selecting the disposable valid bundle
`/tmp/vmlx-repoint-live/Laguna-M.1-JANG_2L` updated SQLite to the canonical
`/private/tmp/...` path and moved the card from `MISSING MODEL` to ordinary
`INACTIVE`, where Start/Delete replaced Repoint/Remove. See
`03-after-native-repoint-ui.png`. The disposable session was deleted through
its own rendered Delete button and the temporary bundle directory was removed.
Active Laguna PID 70292 remained unchanged throughout.

## Focused validation

- `tests/session-model-path-recovery.test.ts`: 8/8 passed.
- Panel TypeScript typecheck passed.
- Logs: `focused-panel.log` and `typecheck.log`.

## Remaining boundary

Repeat this flow in the next signed app. This scoped dev-Electron gate does not
claim that a repointed bundle with only `config.json` is startable; the fixture
was intentionally never started because this row owns path recovery UX, not
model loading.
