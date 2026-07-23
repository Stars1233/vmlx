# v1.6.17 bundled-Python and exact-head suite gate

Date: 2026-07-23 (America/Los_Angeles)

Status: `PREPACKAGE BUNDLE + FULL SUITES PASS / SIGNED DMGS OPEN`.

## Frozen source boundary

- vMLX branch:
  `codex/v1.6.17-consolidation-20260723`.
- Version/provenance head:
  `ee366371a6daf19e7d9d4d8fe7c06385f391f6f9`.
- vMLX source version:
  `1.6.17`.
- Clean JANG source:
  `/Users/eric/jang/.worktrees/jang-tools-r16-2534/jang-tools`.
- JANG source commit:
  `2e85095f55b1b64cbb6e8264c82a074a3cc28250`.
- JANG source version:
  `2.5.34`.
- The JANG commit is pushed on
  `origin/codex/r16-vmlx-1.6.16-jang-tools`. At capture time,
  `jjang-ai/jangq` main and public PyPI `jang` remained at `2.5.33`.
  Main/PyPI reconciliation is therefore a publication dependency, not an
  inherited pass.

## Packaging-source correction

The packaging default pointed at `/Users/eric/jang/jang-tools`, a JANG 2.5.31
checkout with numerous untracked package files. The old guard checked tracked
diffs only, so an untracked module under `jang_tools/` could enter a public app
without a commit or provenance record.

The 1.6.17 bundler now fails closed on:

- tracked or untracked changes under `pyproject.toml` or `jang_tools/`;
- a non-Git JANG source unless a smoke-only override is explicit;
- source/install JANG version mismatch;
- JANG below the vMLX floor `2.5.33`.

It writes `vmlx-bundle-provenance.json` with the vMLX commit/version, exact
JANG commit/version, and MLX wheel platform. The verifier independently
requires that manifest, compares its JANG commit/version with the selected
source, enforces the distribution version floor, and retains the existing
critical-file SHA checks.

The contaminated default tree was exercised deliberately. It exited `1`
before deleting the existing bundle (`before=present`, `after=present`).
Evidence: `r17-dirty-jang-guard.log`.

## Actual Sequoia-compatible bundle proof

The clean JANG source rebuilt the real relocatable bundled Python with:

- Python `3.12.12`;
- vMLX `1.6.17`;
- JANG `2.5.34` at exact commit `2e85095f…`;
- MLX wheel platform `macosx_14_0_arm64`;
- no editable installs;
- no build-machine or `/Applications/vMLX.app` console-script shebangs;
- isolated user-site behavior;
- a final size of approximately `1.4G`.

`verify-bundled-python.sh` passed:

- vMLX version and critical-source SHA parity;
- exact JANG provenance and critical-source SHA parity;
- Gemma 3/4 and Qwen VL imports;
- image, vision, and audio dependencies;
- Hy3, MiMo, Step, Kimi, DSV4, and TurboQuant/JANGTQ modules;
- Gemma/Step registration;
- Kimi, DSV3.2/GLM, and Mistral MLA runtime patches.

Evidence:

- `r17-bundle-sequoia.log`
- `r17-verify-bundle-sequoia.log`
- `vmlx-bundle-provenance.json`

## Exact-head suites

The 1.6.17 scoped preflight passed with no failures:
`current-scoped-release-preflight-17.json`.

The first broad Python invocation inherited the minimal non-login SSH PATH and
could not find Homebrew `node`. Twelve tests failed at process launch while
`6,398` passed. The log is retained as
`r17-exact-head-full-python-with-bundle.log`; it is not counted as a product
failure or a pass.

The complete command was rerun with
`PATH=/opt/homebrew/bin:/usr/local/bin:$PATH` and no manual test deselection.
It passed:

```text
6410 passed, 97 skipped, 92 deselected, 2 warnings in 265.67s
```

This includes the bundled-Python integrity row that the previous source-only
run manually omitted.

At the same head, panel verification passed:

- `86/86` test files;
- `2,491` tests passed and `3` skipped;
- TypeScript `tsc --noEmit`;
- production Electron build with KaTeX assets.

Evidence:

- `r17-exact-head-full-python-with-bundle-path-fixed.log`
- `r17-exact-head-panel-full.log`
- `r17-exact-head-panel-typecheck.log`
- `r17-exact-head-panel-build.log`

## Remaining release boundary

This gate permits the signed artifact build. It does not prove or claim:

- Developer-ID-signed Sequoia and Tahoe DMGs;
- Apple acceptance, stapling, Gatekeeper, or final SHA-256;
- installed signed-app Electron/API smoke;
- JANG main/PyPI publication;
- vMLX source tag, GitHub release, PyPI, updater, website, or Homebrew
  publication.

Those remain downstream gates. The broader campaign matrix also remains
`PARTIAL`.
