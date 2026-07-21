# Current-source Electron gateway non-stream atomicity

Status: `VERIFIED-LIVE_SCOPED`

Date: 2026-07-21

Source commit: `e70902c42` (`fix(gateway): keep non-stream responses atomic`)

## Release blocker reduced

`api/ui`: a generic non-stream proxy response could commit the backend HTTP
status and leak a partial JSON prefix before the backend connection reset. The
client then observed a false HTTP 200 with malformed/truncated JSON instead of
one truthful failure response. This affected generic OpenAI Chat Completions,
Responses, and Anthropic Messages routes through the Electron-owned gateway.

## Owning source change

- `panel/src/main/api-gateway.ts::proxyRequest`
  - determines stream/non-stream before committing response headers;
  - installs downstream-cancel and upstream-lifecycle guards for both modes;
  - buffers a non-stream response until the backend `end` event, then commits
    status, headers, and the complete JSON body once;
  - leaves the streaming byte-forwarding/native failure-terminal path intact.
- The superseded unconditional header/body-forward path was replaced at the
  owner. No endpoint-specific Chat/Responses/Anthropic workaround or output
  repair was added.
- `panel/tests/api-gateway-ollama-behavior.test.ts` adds the shared regression
  across all three generic JSON routes and repairs the test harness's stale
  `preflightSessionStart` mock.

## Automated validation

From `panel/` at the tested source:

```text
npm test -- --run tests/api-gateway-ollama-behavior.test.ts
1 file passed; 18 tests passed

npm run typecheck
tsc --noEmit (exit 0)

git diff --check
exit 0
```

The first full-suite attempt before the stale mock repair failed before proxy
execution because the current gateway calls `preflightSessionStart` and the
old test mock did not define it. The current full file, not only the new case,
is the recorded 18/18 pass.

## Real Electron proof

The dev Electron app was fully relaunched from the current checkout with:

```text
userData: /Users/eric/.vmlx-v1613-responsive-dev
CDP: 127.0.0.1:9335
gateway: 127.0.0.1:8088
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
[Engine Manager] Version: 1.6.14
```

`electron-api-drawer.png` is the visually inspected real API drawer. It shows
the app-owned endpoint at `localhost:8088`, LAN disabled/local-only, Single
Model enabled, and OpenAI/Anthropic/Ollama connection surfaces. No model was
loaded for this transport-only fault injection, avoiding a duplicate family
smoke.

A controlled healthy backend was temporarily mapped to one stopped local
session. For the first request on each route it emitted HTTP 200 headers plus
the literal incomplete prefix `{"partial":`, then reset the socket. Its
second request on the same route returned complete recovery JSON. Requests
were sent to the Electron-owned gateway, not directly to the backend.

`live-gateway-proof.json` records:

| Route | First request | Immediate second request |
|---|---|---|
| `/v1/chat/completions` | 502 parseable JSON, `backend_connection_closed` | 200 parseable recovery JSON |
| `/v1/responses` | 502 parseable JSON, `backend_connection_closed` | 200 parseable recovery JSON |
| `/v1/messages` | 502 parseable JSON, `backend_connection_closed` | 200 parseable recovery JSON |

The main-process log owns all three warnings:

```text
[gateway] Proxy response 127.0.0.1:18089/v1/chat/completions ended prematurely: backend response aborted
[gateway] Proxy response 127.0.0.1:18089/v1/responses ended prematurely: backend response aborted
[gateway] Proxy response 127.0.0.1:18089/v1/messages ended prematurely: backend response aborted
```

`backend-observed-requests.log` independently records exactly two requests per
route. `session-restored.json` proves the temporary mapping was restored to its
original stopped state on port 8007 with no PID, and the controlled backend was
stopped.

## Boundary

This closes the retained non-stream partial-body-loss row for the current dev
Electron gateway. It does not promote signed-app parity, model quality,
reasoning/tool parsers, streaming model generation, cache/media behavior, or
the broader gateway concurrency/stress matrix. Those rows keep their existing
statuses and should not be rerun merely because this shared transport owner
changed.
