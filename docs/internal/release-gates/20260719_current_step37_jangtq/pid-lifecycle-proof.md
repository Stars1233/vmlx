# Electron session PID lifecycle proof

Date: 2026-07-19

Scope: current-source Python/Electron dev build, real renderer on CDP 9335,
session `e04ccb1d-75b8-4eef-b716-c803e11af256`, Step port 8022.

## Launch gate

The Electron main process was fully relaunched with the preserved development
profile and the repository virtual environment in `PATH`. Its live log emitted:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
[Engine Manager] Version: 1.6.11
```

## Visible lifecycle

1. The real Sessions-card **Start** control loaded the model. Server and Chat
   both displayed `PID 38968`. SQLite stored `pid=38968,status=running,port=8022`,
   and `ps` showed that exact Python `vmlx_engine.cli serve` process.
2. The real Chat-header **Stop** control terminated PID 38968. The Chat header
   displayed Start with no PID, SQLite stored `pid=null,status=stopped`, and
   `ps` no longer returned the old PID.
3. The real Chat-header **Start** control loaded a new process. The Chat header
   displayed `PID 39507`; SQLite and `ps` both reported PID 39507 on port 8022.
   The process inventory contained exactly one local `vmlx_engine.cli serve`.

Screenshots:

- `pid-fix-running-server.png`
- `pid-fix-running-chat-header.png`
- `pid-fix-stopped-chat-header.png`
- `pid-fix-restarted-chat-header.png`

## Source and focused tests

- `panel/src/main/sessions.ts`: local ready events include the spawned or
  adopted PID; remote endpoints remain PID-less.
- `panel/src/renderer/src/contexts/SessionsContext.tsx`: the shared session type
  includes PID, awaited starts preserve it, and Stop clears it.
- `panel/tests/session-pid-lifecycle.test.ts`: transport, monitored-ready, and
  clear-on-stop contracts.
- Focused result: 174 passed across PID lifecycle, single-model start, and
  session-port UI tests; `tsc --noEmit` passed.

Verdict: `VERIFIED-LIVE_SCOPED` for the Electron PID lifecycle only. This does
not promote the full model, protocol, cache, packaging, or release matrix.
