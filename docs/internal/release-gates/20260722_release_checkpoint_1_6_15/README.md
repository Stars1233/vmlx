# vMLX 1.6.15 release-candidate checkpoint — 2026-07-22

Status: `BLOCKED_BEFORE_SIGNED_RELEASE`.

This directory records the current 1.6.15 checkpoint attempt on the Python /
Electron vMLX release branch. It is not a public release record: no signed DMG,
notarized artifact, tag, updater feed, PyPI upload, Homebrew update, or GitHub
release was produced from this gate because Developer ID signing is blocked by
keychain access in the current SSH session.

## Source state

- Repository: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Branch: `codex/postrelease-ui-drawers-20260720`
- Current pushed source HEAD: `8bd3716f2`
- Version surfaces: `1.6.15` in `panel/package.json`, `panel/package-lock.json`,
  `pyproject.toml`, and `vmlx_engine/__init__.py`.
- JANG source used for bundling: `/Users/eric/jang` clean at
  `801209c13c189ebb8fb4d1596748a336f568da38`.

## Reasoning / Auto policy for this checkpoint

`Auto` remains model/template-driven. It is allowed to emit no reasoning rail for
easy prompts. The release invariant is narrower and stricter:

- explicit UI/API reasoning-on must be passed to the tokenizer/template layer;
- if the model emits reasoning, it must stream separately from visible content;
- raw `<think>`, private reasoning tags, or tool/parser markers must not leak
  into visible content;
- required tool calls must produce real protocol tool/function-call events and
  complete with a real tool-result continuation.

Current Laguna S-2.1 raw Chat/Responses proof for this invariant is preserved in
`../20260722_laguna_reasoning_tool_stream_current/current_pid4279_after_template_mirror/`.

## Passing gates on this source

| Gate | Status | Evidence |
| --- | --- | --- |
| Full Python suite after 1.6.15 bump | PASS | `full-python-pytest-1.6.15.log`: `6290 passed, 96 skipped, 92 deselected, 2 warnings in 254.16s` |
| Full panel Vitest after packaging fix | PASS | `panel-vitest-full-after-packaging-fix.log`: `80 passed`, `2405 passed`, `3 skipped` |
| TypeScript typecheck after packaging fix | PASS | `panel-typecheck-after-packaging-fix.log`: `tsc --noEmit` completed |
| Production panel build before packaging | PASS | `panel-build.log`: bundled-python verifier passed and Electron Vite production build completed |
| Bundled Python verification | PASS | `panel-build.log` and DMG-build logs show bundled engine version `1.6.15`, critical vmlx/jang source hash parity, and critical imports OK |
| Packaging `.exe` regression test | PASS | `8bd3716f2` adds `panel/tests/release-packaging.test.ts`; focused run passed 2/2 |

## Scoped prepackage gate

The default `npm run release:prepackage` broad manifest remains `BLOCKED`:

- `release-prepackage.log` records `current_proof_sweep=fail`,
  `prepackage_ready=false`, and `release_ready=false`.
- The manifest is still tied to broad historical matrix artifacts and open
  family rows. It is not satisfied by the targeted 1.6.15 reasoning/packaging
  fixes.

The release build was therefore attempted with the script-supported scoped path:

```sh
VMLINUX_RELEASE_SCOPE=codex_ui_only ./scripts/build-release-dmgs.sh all
```

This is not a declaration that the broad matrix is complete. It means the
checkpoint would require post-build live Codex UI validation of the signed app
as the substantive scoped gate before public publication.

## Packaging fix landed during this attempt

First DMG build attempt failed when electron-builder tried to Developer-ID sign
bundled Windows launchers from `pip/_vendor/distlib`:

- failing log: `build-release-dmgs.log`
- failing file example:
  `bundled-python/python/lib/python3.12/site-packages/pip/_vendor/distlib/t32.exe`

Fix:

- source commit: `8bd3716f2 fix(release): drop bundled Windows launchers before signing`
- file: `panel/scripts/electron-builder-after-pack.cjs`
- behavior: remove bundled `site-packages/pip/_vendor/distlib/*.exe` Windows
  launcher stubs before electron-builder traverses the app for signing.
- focused proof: `panel/tests/release-packaging.test.ts` passed.

Second DMG build attempt reached this fixed hook:

```text
[afterPack] removed 6 bundled Python Windows launcher stubs: t32.exe, t64-arm.exe, t64.exe, w32.exe, w64-arm.exe, w64.exe
[afterPack] normalized ad-hoc signatures for 503 bundled Python native files
```

## Current hard blocker

Developer ID signing is not usable from the current SSH session.

Evidence:

- `build-release-dmgs-after-exe-fix.log` fails while signing
  `scipy/sparse/linalg/_eigen/arpack/_arpacklib.cpython-312-darwin.so` with
  `errSecInternalComponent`.
- `codesign-keychain-blocker.log` records:
  - `security show-keychain-info ... build.keychain-db: User interaction is not allowed`
  - `security show-keychain-info ... login.keychain-db: User interaction is not allowed`
  - the Developer ID identity exists in `security find-identity`;
  - manual Developer-ID signing of a copied SciPy `.so` fails with
    `errSecInternalComponent`;
  - manual Developer-ID signing of a copied Electron binary also fails with
    `errSecInternalComponent`.

Therefore no Sequoia/Tahoe public DMG was produced, signed, notarized, stapled,
verified, installed-smoked, uploaded, tagged, or published in this attempt.

## To resume release after keychain access is fixed

Run from an interactive/unlocked signing environment on `erics-m5-max.local`:

```sh
cd /Users/eric/mlx/vllm-mlx-release-1.6.13/panel
export PATH=/Users/eric/.local/node/bin:$PATH
export PYTHON=/Users/eric/mlx/vllm-mlx/.venv/bin/python
export VMLINUX_RELEASE_SCOPE=codex_ui_only
./scripts/build-release-dmgs.sh all
./scripts/notarize-release-dmgs.sh
./scripts/verify-release-dmgs.sh
```

After those pass, still required before public release:

- install-smoke both final DMGs;
- run live signed-app Electron UI smoke on the scoped checkpoint model(s);
- update this release checkpoint with DMG sizes, sha256 hashes, notary
  submission IDs, stapler/spctl/codesign evidence, and smoke evidence;
- tag/push the exact released commit;
- update GitHub release artifacts, updater `latest.json`, PyPI/Homebrew only
  after publication authority is confirmed for this version.
