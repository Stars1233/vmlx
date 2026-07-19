# 2026-07-19 Full-suite checkpoint

Status: PASS for the source/full-suite/build checkpoint. Broader live runtime
and release-matrix rows remain OPEN/PARTIAL.

Baseline source: `e76d54d1f` on `reconcile/1.5.68`.

## FS-1 — stale panel MCP source-shape contract

- Symptom: the first full panel run failed `mcp-policy.test.ts` because it
  required the obsolete literal `if (overrides?.builtinToolsEnabled)`.
- Root cause: current `chat.ts` independently guards Responses and Chat
  Completions tool-schema injection with
  `overrides?.builtinToolsEnabled && !userForbidsToolCalls`; built-in tool
  execution and remote MCP execution remained separate.
- Change: assert that both API builders contain the effective injection guard,
  rather than pinning an older standalone `if` spelling.
- Acceptance evidence:
  - `tests/mcp-policy.test.ts`: 9/9 passed.
  - full panel: 73/73 files, 2,312 passed, 3 skipped.
  - `npm run typecheck`: passed.
- Durable raw log: `/tmp/vmlx-full-panel-mcpfix-20260719.log` on the test host.

## FS-2 — cache worker-dequant test double missing production capacity API

- Symptom: `test_worker_schedule_dequantizes_flagged_memory_cache_hit` failed
  before reaching its worker-dequant assertion because `_FakeMemoryCache`
  lacked `get_stats()`.
- Root cause: production scheduler snapshot-capacity accounting calls the
  memory-cache `get_stats()` interface; the narrow test double had not tracked
  that interface addition.
- Change: give the fake the production capacity interface. No runtime cache
  behavior was changed.
- Acceptance evidence: `tests/test_cache_hit_worker_dequant.py`: 2/2 passed.
- Full-suite status: the corrected pre-bundle run reached 6,123 passed and 96
  skipped, with only FS-4 remaining. Final post-bootstrap rerun is pending.

## FS-3 — invalid first Python-suite environment

- Symptom: the first full Python invocation reported 13 failures. Twelve were
  app/release audit subprocesses failing with `node: command not found`; the
  remaining failure was FS-2.
- Root cause: the noninteractive SSH command omitted
  `/Users/eric/.local/node/bin` from `PATH`, unlike the real Electron/release
  environment.
- Resolution: do not count that run as a product result. Rerun the complete
  suite with `.venv/bin`, the local Node bin, and system bins in `PATH`.
- Invalid-run log: `/tmp/vmlx-full-python-20260719.log`.
- Corrected-run log: `/tmp/vmlx-full-python-pathfix-20260719.log`.

## FS-4 — bundled Python source drift

- Symptom: the corrected full Python suite finished with one failure because
  bundled `vmlx_engine/server.py` did not match current source.
- Root cause: post-release engine fixes had not yet been copied into the
  packaged Python runtime.
- Resolution: rebuild with the clean JANG source checkout at the same revision
  as the dirty working checkout:
  `/Users/eric/.cache/vmlx-release/jang-clean-9081c924/jang-tools` at
  `9081c92476a63b912f4d2ce96146674971b5c83e`.
- Acceptance evidence:
  - `verify-bundled-python.sh` reports matching critical vmlx_engine and
    jang_tools hashes and all critical imports present.
  - clean-JANG `npm run build` completed all main/preload/renderer production
    bundles.
- Raw logs:
  - `/tmp/vmlx-bundle-python-clean-jang-20260719.log`
  - `/tmp/vmlx-verify-bundled-python-20260719.log`
  - `/tmp/vmlx-panel-build-cleanjang-20260719.log`

## FS-5 — bundle operation erased the repository proof tree

- Symptom: `bundle-python.sh` used `rm -rf "$VMLX_LOCAL/build"`. The repository
  stores tracked cross-matrix evidence under that same directory, so bundling
  deleted proof prerequisites and made later public-app/manifest tests depend
  on timing or stale local state.
- Root cause: a broad stale-wheel cleanup conflated setuptools scratch with
  the repository's durable release-gate artifact root.
- Change: clean only packaging-owned `build/lib`, `build/bdist.*`,
  `build/temp.*`, `build/scripts-*`, and root egg-info paths.
- Acceptance evidence:
  - bash syntax check passed.
  - focused release contracts: 3/3 passed.
  - a real clean-JANG bundle rebuild retained
    `build/vmlx-bundle-preserve-probe-20260719.txt` and printed
    `PRESERVE_PROBE_PASS`.
- Raw log: `/tmp/vmlx-bundle-python-preserve-proof-20260719.log`.

## Canonical artifact bootstrap

Because the old bundler had already removed proof prerequisites, the canonical
current-regression orchestrator was rerun from current source. It returned
`status=open`, not a release pass. Its retained failures/open rows include the
absent local MiMo bundle, staged packaged-integrity/signing drift for the
post-release head, and broad DSV4/Qwen/Gemma/cross-family live requirements.
The public-app audit itself regenerated as `status=open` rather than `fail`.
Raw log: `/tmp/vmlx-current-regression-bootstrap-20260719.log`.

## Current source/build results

- Panel full suite: 73/73 files, 2,312 passed, 3 skipped.
- Panel typecheck: passed.
- Bundled Python source/hash/import verification: passed.
- Clean-JANG production build: passed.
- Python full suite: 6,125 passed, 96 skipped, 92 deselected, two third-party
  `librosa` deprecation warnings; zero failures in 254.04 seconds.
- Final raw log: `/tmp/vmlx-full-python-final-bootstrap-20260719.log`.

## Live-proof classification

This checkpoint is a source/test gate. It did not load a model or run a chat
generation, so Electron model-load, output-emission, tool-loop, and cache-hit
runtime rows are N/A here. Existing model-specific live evidence is not
reclassified by these test-only changes.
