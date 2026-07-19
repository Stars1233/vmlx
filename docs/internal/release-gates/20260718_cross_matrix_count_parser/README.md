# Cross-matrix count parser and clean-checkout sweep — 2026-07-18

Status: `SOURCE_META_AUDIT_PASS_CAMPAIGN_OPEN_NO_NEW_MODEL_LIVE_PROOF`.

## Scope and source trace

- Base source head: `6d813c39d20ca6d10f5b0967bc3ac2e615f71a24`.
- Scoped fix commit: `db07a6fc1b319c65bc8d47e72f18c3e46a8bbb5b`.
- `tests/cross_matrix/output_counts.py` now owns ANSI-safe pytest, Vitest,
  and nested-runner count parsing. The explicit Vitest `Tests` row is parsed
  before generic `N passed` text, so a colored `Test Files` count cannot be
  mistaken for the test count.
- Fifteen proof runners use the shared parser and include it in their
  `SOURCE_HASH_FILES`, so a parser change makes old proof artifacts stale.
- `run_current_regression_suite.py` can checkpoint a clean checkout before
  the generated objective digest exists. A provisional digest is reported as
  pending; the final summary still requires the real generated artifact.
- Stale exact marker anchors were reconciled to current behavior for cache
  controls, generation defaults, request-scoped thinking budgets, VL Force-Off,
  launch-memory admission, and Native MTP. The Native-MTP gate now pins the
  current rule: a JANG_2K profile alone does not disable an indexed MTP
  artifact; only a bundle-declared measured block may do so.

## Verification

- Focused runner/contract matrix: **222 passed, 1 skipped**.
  See `focused-contract-tests.log`.
- The canonical no-heavy orchestrator reached the current artifact path and
  its 658-test focused sub-suite returned **656 passed, 1 skipped, 232
  deselected**. The earlier missing-current-artifact freshness failure is gone.
  See `canonical-noheavy-sweep.log` and `current-regression-suite.json`.
- Full panel Vitest from the same source interval: **2311 passed, 3 skipped**;
  TypeScript typecheck passed; the production Electron build passed with
  `VMLX_JANG_TOOLS_SOURCE` set to the clean JANG `9081c924` checkout. See
  `full-panel-vitest.log`, `panel-typecheck.log`, and
  `panel-production-build.log`.
- Python compilation and `git diff --check` passed for every modified runner.

## Honest open boundary

The canonical orchestrator remains `status=open`. Current failed steps are:

- `mimo_v2_local_bundle_metadata_contract`: both configured local bundle paths
  are absent.
- `packaged_integrity_contracts`: current host signing preflight is blocked.
- `release_regression_manifest`: required current live proof rows remain open.
- `release_gate_skip_app`: the objective digest is open and release readiness
  is therefore false.

The tool-call source tests pass, but the tool runner remains open because the
required current DSV4 default-cache live artifact is absent. Historical live
speed, Gemma/Ling quality, broad real-Electron, and cross-family smoke evidence
that is missing from the current working tree is not promoted to a pass.

No model was loaded or generated during this source/meta-audit checkpoint.
Accordingly, it provides no new Electron, streaming API, cache-hit, media, or
tool-loop runtime clearance. Those rows require separate current-head live
proof.
