# Laguna parser-default migration — current live gate

Verdict: scoped `PASS-LIVE`. Laguna reasoning remains a separate release
blocker.

## Root cause

The Python registry and the real Laguna/Poolside bundle require the GLM-style
`<arg_key>/<arg_value>` tool parser, but the Electron family registry persisted
the older `qwen` default. Updating the registry alone did not repair existing
sessions. The first Save & Restart only restarted the model child while the
Electron main process still contained the old code, so PID 13108 retained
`--tool-call-parser qwen` and the DB migration version remained null.

## Source trace

- `panel/src/main/model-config-registry.ts` maps Laguna to `glm47`.
- `panel/src/shared/sessionConfigMigrations.ts` versions parser defaults and
  migrates only the known stale Laguna `qwen` value once.
- `panel/src/main/sessions.ts` applies that migration during detected-family
  startup defaults and stamps new adopted sessions at the current version.
- Explicit non-stale parser choices remain unchanged.

## Live Electron evidence

- The dev Electron main process was rebuilt/restarted with the same
  `/Users/eric/.vmlx-v1611-cachefix-dev` profile and CDP port 9335.
- Starting Laguna from the visible session card launched PID 32806 with
  `--tool-call-parser glm47 --reasoning-parser qwen3`.
- SQLite persisted `toolCallParser=glm47` and
  `modelParserDefaultsVersion=1`.
- Expanded Server Settings DOM values were `glm47`, `qwen3`, and detected
  family `laguna`.
- Same-chat row 1992: one `file_info(panel/package.json)`, one result, exact
  `LAG-GLM47-CURRENT1-DONE`, no warning, and 3,072
  `paged+disk+tq-native` cached tokens.
- Fresh-chat row 1995: one `file_info(panel/package.json)`, one result, exact
  `LAG-GLM47-FRESH1-DONE`, no warning, and 3,136 `paged+tq-native` cached
  tokens.

## Tests

- Registry plus migration tests: 94 passed after the explicit-choice guard.
- TypeScript typecheck passed.

## Remaining Laguna blockers

- Default Auto reasoning previously looped in repetitive meta-reasoning and
  was manually interrupted after 726 output tokens.
- Auto versus None cache/JIT behavior, long-context tails, broader tools,
  multi-turn quality, and full protocol parity remain open.
