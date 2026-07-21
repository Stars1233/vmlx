# Gateway active-request host/port reconfiguration — current source

Date: 2026-07-21

Status: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED`

Scope: MLXStudio Electron dev main from
`/Users/eric/mlx/vllm-mlx-release-1.6.13`, gateway `127.0.0.1:8088`, and the
already-selected `dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP` session. This exact
artifact is base MLX `MXFP4` plus native MTP depth 3, not affine JANG and not
JANGTQ/MXTQ.

## Failure reproduced before the source repair

A real gateway Chat stream produced three visible deltas, then the Electron API
dashboard's visible LAN toggle was clicked. `ApiGateway.restart()` called
`server.close()` before rebinding. Node stopped accepting new connections but
waited for the existing stream to finish. The UI optimistically changed to
`Localhost only (127.0.0.1)`, while eight consecutive health probes to
`127.0.0.1:8088` returned no HTTP response. The original stream finally ended
normally after 21,266.62 ms, after which the gateway logged `Stopped` and
rebound to `127.0.0.1:8088`.

This was a gateway listener lifecycle failure, not a model decode failure.

Source trace:

- `panel/src/main/api-gateway.ts::restart` stopped the live listener before it
  knew whether active proxied responses existed.
- `panel/src/main/index.ts::gateway:restart` wrote `gateway_port` and
  `gateway_host` before the replacement listener had bound, duplicating the
  success-only persistence already owned by `ApiGateway.start()`.

## Repair

- `ApiGateway` tracks non-liveness HTTP responses from admission through
  `finish`/`close`.
- `/` and `/health` remain available and do not themselves block an idle
  listener change.
- `restart()` rejects host/port mutation before `server.close()` whenever a
  model/API response remains active. The current listener and persisted
  settings remain untouched, and the renderer's existing error path rolls the
  optimistic LAN toggle back.
- Gateway health reports `active_requests`; the Electron status payload exposes
  `activeRequests` from the same owner.
- The superseded IPC-side pre-bind DB writes were deleted. Host/port persistence
  now occurs only after `_tryListen()` succeeds.

## Current-source live proof

The Electron dev main was fully relaunched after the edit. Startup printed:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

The real Server-page `Start` button eagerly loaded the Qwen session before any
request; live health showed `model_loaded=true` and `last_request_time=null`.

During a 127-delta Chat stream, the proof driver clicked the real LAN toggle
after three content deltas:

- all eight health probes stayed HTTP 200 on port 8088;
- health reported at least one active request throughout the rejection window;
- the visible API dashboard showed `Gateway cannot change host or port while 1
  model request is active`;
- the visible LAN state rolled back to `Localhost only (127.0.0.1)`;
- the original stream completed with `length` plus `[DONE]`, without transport
  error;
- the immediate recovery emitted nine progressive deltas, exact-finaled
  `GATEWAY-ACTIVE-RECONFIGURE-RECOVERY-DONE`, and ended `stop` plus `[DONE]`;
- final gateway health reported `active_requests=0`, port 8088, and Single Model
  On.

Additional real UI controls:

- entering occupied model port `8005` displayed the conflict and retained
  gateway `8088`, model `8005`, DB `gateway_port=8088`, and
  `gateway_host=127.0.0.1`;
- idle LAN On successfully rebound to `0.0.0.0:8088` and displayed
  `192.168.1.110:8088`;
- idle LAN Off successfully restored `127.0.0.1:8088`; this is the retained
  final state.

Artifacts:

- `q27-gateway-active-reconfigure-postfix.json`
- `gateway-active-reconfigure-postfix-proof.png`
- `gateway-port-conflict-postfix.png`
- `gateway-lan-idle-success.png`
- `gateway-lan-restored-localhost.png`

## Validation and boundary

- Gateway-focused panel tests: 91 passed / 3 skipped across five files.
- TypeScript typecheck: pass.
- `git diff --check`: pass.
- Dead-code trace: one success-only gateway settings writer remains in
  `ApiGateway._tryListen()`; the two premature IPC writers were removed.

This closes active-request LAN/port rejection, idle LAN rebind, and occupied
session-port rollback for the current dev Electron gateway. It does not close
concurrent model-swap/loss races, repeated long soak, other running-model
families, signed-app repetition, or the full test suites.
