# Single-model unmanaged-engine sweep proof — 2026-07-22

Verdict: `VERIFIED-LIVE` for the scoped lifecycle issue only.

This gate proves that when `gateway_single_model_mode=true`, starting a model from the real Electron UI stops a healthy local `vmlx-engine` process even if that process is not represented by an active running DB session row.

## Source trace

- `panel/src/main/sessions.ts`
  - `startSession()` now calls `stopDetectedLocalEnginesForSingleModel(sessionId)` before `_startSessionInner(sessionId)`.
  - `stopDetectedLocalEnginesForSingleModel()` enumerates live local engine processes with `detect()`, preserves only the healthy target process for the target session, and stops other detected engines.
  - `terminateDetectedLocalEngine()` kills the live PID, escalates to `SIGKILL` if needed, and marks the owning local row stopped by path, port, or PID.
  - `detectAndAdoptAll()` now routes through `pruneDetectedProcessesForSingleModel()` so adoption cannot re-import multiple healthy engines while single-model mode is enabled.
- `panel/tests/session-single-model-start.test.ts`
  - Guards manual Start ordering.
  - Guards stale detected engine termination.
  - Guards adoption pruning.
  - Guards already-running target adoption.

## Focused checks

Remote checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Commands:

```sh
cd /Users/eric/mlx/vllm-mlx-release-1.6.13/panel
export PATH=/Users/eric/.local/node/bin:$PATH
npm test -- --run tests/session-single-model-start.test.ts
npm run typecheck
```

Result:

- `tests/session-single-model-start.test.ts`: 5/5 passed.
- `tsc --noEmit`: passed.

## Live Electron/process evidence

User data: `/Users/eric/.vmlx-v1613-responsive-dev`

The dev Electron app was relaunched with the engine virtualenv in `PATH`. `dev-log-filtered.txt` includes:

- `[STARTUP] Using vMLX userData override: /Users/eric/.vmlx-v1613-responsive-dev`
- `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`
- `[Engine Manager] Version: 1.6.14`
- `[SESSIONS] single-model mode: stopping detected engine pid=62262 port=8051 model=/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK before starting 6e71c6e8-1b61-4963-81a1-d1110986c205`

After clicking Laguna Start in the real Electron UI, `state.txt` shows:

- `gateway_single_model_mode|true`
- only one live local model process:
  `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L --port 8008`
- no live `LFM2.5` process.

Artifacts:

- `state.txt`
- `dev-log-filtered.txt`
- `ui-text-after-sweep.txt`
- `ui-after-sweep.png`

## Scope limits

This gate does not close release packaging, notarization, full protocol parity, or the full model matrix. It only closes the single-model unmanaged-process sweep/adoption bug for this source state with real Electron UI proof.
