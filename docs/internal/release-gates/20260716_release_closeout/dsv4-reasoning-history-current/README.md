# DSV4 reasoning-history replay — current proof

Status: `PASS-LIVE` for ordered Responses reasoning/tool history replay and
progressive warm streaming; `PARTIAL` for strict byte-format reliability and
long-context closeout.

## Source trace

- Scoped code commit: `35b444ce356e` (`fix(responses): replay reasoning across
  tool turns`).
- `vmlx_engine/server.py` accepts Responses `reasoning` input items, preserves
  assistant reasoning history, and keeps reasoning-only turns.
- `panel/src/shared/toolHistoryReplay.ts` reconstructs persisted assistant
  history in native per-iteration reasoning/call/result/final order.
- `panel/src/main/ipc/chat.ts` uses the shared reconstruction for both
  Responses and Chat Completions and persists raw reasoning boundary slots.

## Live Electron evidence

- Electron main PID 71823 and DSV4 PID 71928 were restarted from the current
  dev source before the proof; `live-processes.txt` records the processes.
- `request-shape-and-server-log.txt` records Responses `reasoning` items around
  the function call/result and server reconstruction as seven ordered
  messages.
- SQLite row 351 coherently recalled the prior real tool result:
  `panel/package.json` and `5.2 KB`.
- Warm row 354 restored 274 tokens as `paged+dsv4` at 1.28s TTFT and completed
  coherently with no tool or warning.
- `dsv4-history-fix4-warm-mid-2s.png` captures progressive reasoning already
  visible two seconds into the warm request; the matching complete screenshot
  captures the terminal answer.

## API and test evidence

- `raw-seeded-results.txt` records five seeded raw Responses replays: all five
  completed coherently with progressive reasoning/content; one of five met the
  requested byte-exact marker.
- `focused-python-tests.txt`: 55/55 focused Python tests passed.
- `focused-panel-tests.txt`: 121/121 focused panel tests passed.
- `panel-typecheck.txt`: panel TypeScript typecheck passed.

The 1/5 strict marker result remains a release blocker. No sampler coercion,
synthetic output, or output cleanup is claimed or used by this proof.
