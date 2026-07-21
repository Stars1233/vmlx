# Development engine version/source truth gate — 2026-07-21

## Verdict

`VERIFIED-LIVE_SCOPED` for development Electron engine discovery. The version
returned by the real renderer-to-main IPC now describes the source tree the
selected console-script shim will actually import at session launch. Packaged
build isolation is unchanged.

This closes the explicit `ENGINE-VERSION-TRUTH` row only. It does not claim a
new signed-app/package release, and no generation, parser, cache, or model
quality row was rerun or promoted.

## Retained pre-fix mismatch

The app and checkout source surfaces were all 1.6.14:

- `panel/package.json`
- `pyproject.toml`
- `vmlx_engine/__init__.py`

The discovered shim was
`/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`. Without the development
checkout on `PYTHONPATH`, its interpreter reported the sibling editable
installation as 1.6.12. With the same `PYTHONPATH` used by actual development
session launches, the same shim reported 1.6.14. The real pre-fix Electron log
therefore showed the selected PATH engine as version 1.6.12 even though model
processes imported current source.

## Root cause and source repair

`panel/src/main/sessions.ts` already pins `engineResult.sourceRoot` on
`PYTHONPATH` when a development checkout reuses a sibling/system engine shim.
`panel/src/main/engine-manager.ts::getVersionFromBinary()` instead cleared
`PYTHONPATH` unconditionally. Its displayed version described the shim's stale
editable install, not the code the app would run.

Current source adds one `getDevelopmentSourceRoot()` owner. It returns a valid
repo source root only in development and returns `null` for packaged builds.
Every system/path shim probe receives that root; the child interpreter sets
`PYTHONPATH` to it before importing `vmlx_engine.__version__`. The existing
`getBundledSourcePath()` reuses the same owner instead of keeping a duplicate
dev-root calculation. The stale mismatch and duplicated path logic were
removed rather than hidden with a version string override.

The packaged path remains unchanged: a signed build still reads bundled
dist-info first, uses bundled Python as authoritative, and never shadows it
with a development source root.

## Current-source live Electron proof

The exact old dev Electron main and its orphaned CDP process were stopped; the
installed `/Applications/vMLX.app` was not touched. A clean main/preload/
renderer rebuild launched this checkout with the existing v1.6.14 user-data
directory and CDP 9335.

Startup and a direct call through the real renderer IPC produced:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
[Engine Manager] Version: 1.6.14
```

```json
{"installed":true,"path":"/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine","version":"1.6.14","method":"unknown"}
```

The real Sessions `Start` control then eagerly loaded the existing Nemotron
session as PID 82494. `electron-start-current.png` visibly shows one active
session and that PID. Before any request, `/health` returned
`model_loaded=true`, the exact model name, and `last_request_time=null`.
The live process environment contained:

```text
PYTHONPATH=/Users/eric/mlx/vllm-mlx-release-1.6.13
```

That pairs the displayed 1.6.14 version with the same source root used by the
actual Electron-started engine process.

## Validation and boundaries

- `engine-path-policy.test.ts` and `engine-spawn-path-isolation.test.ts`:
  7/7 passed.
- Panel typecheck passed.
- `git diff --check` was clean.
- No model prompt was sent; output-emission, reasoning, tool, and cache rows
  are intentionally N/A for this version/source gate.
- The next signed/package checkpoint must independently verify bundled
  dist-info, packaged IPC, staged DMG contents, and installed-app behavior.
