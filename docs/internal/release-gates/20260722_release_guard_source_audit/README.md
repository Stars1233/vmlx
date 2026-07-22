# Release packaging guard source audit

Date: 2026-07-21 (America/Los_Angeles)

Scope: read-only source/repository audit. No package, DMG, signing,
notarization, tagging, or publishing was run.

Verdict: `RELEASE BLOCKED` until the P0 guards below are addressed and the
current-source runtime matrix is complete.

## Current repository truth at audit time

- vMLX branch `codex/postrelease-ui-drawers-20260720` was pushed, but was 71
  commits ahead and 1 behind `origin/main`.
- Source version remained `1.6.14`; no local or remote `v1.6.15` tag existed.
- JANG GitHub `origin/main` was `801209c`, but `/Users/eric/jang` was a
  different dirty checkout. A clean worktree pinned to `801209c` is required.
- The existing bundled JANG provenance pointed to an older source worktree,
  and the bundled MLX wheel was Tahoe-only.
- Noninteractive remote PATH did not contain `node`; the actual executable is
  `/Users/eric/.local/node/bin/node`.
- `panel/node_modules` was a symlink into a sibling checkout, not an owned
  dependency tree for this release checkout.

## P0 guard gaps

1. `scripts/bundle-python.sh` ignores untracked JANG files, does not require an
   exact expected JANG commit, and installs the entire local JANG tree. It has
   no equivalent vMLX clean-tree/commit/upstream guard before packaging local
   source.
2. The Electron extraResource copy includes raw `vmlx_engine/**`; current
   packaged-app gates do not prove parity of that full raw tree.
3. Sequoia and Tahoe are rebuilt from mutable source paths without a shared
   immutable provenance manifest or before/after SHA and clean-tree checks.
4. Packaged import verification does not require `sys.executable`,
   `vmlx_engine.__file__`, and `jang_tools.__file__` to reside inside the app.
5. Missing bundled JANG source can be reported as SKIP instead of FAIL, and
   hash coverage omits some Laguna/format-writer/runtime files.
6. The `codex_ui_only` path can bypass the offline release manifest and must
   not be used for a public release.

## Required guard work

- Add a stdlib source preflight before bundling and before each flavor:
  tracked+untracked clean trees, exact pushed vMLX/JANG SHAs, version parity,
  owned non-symlink `node_modules`, and an immutable JSON provenance manifest.
- Carry the manifest into both apps; assert identical source SHAs and the
  correct flavor-specific MLX wheel tag.
- Fail packaged gates for imports outside the app, missing JANG source,
  absolute development `direct_url.json`, poisoned Python environment, or raw
  source-tree mismatch.
- Run the packaged Python/app gate separately for both Sequoia and Tahoe.
- Build from a clean JANG worktree pinned to `801209c` and a clean, pushed,
  current-runtime-proven vMLX release head.

No public publish should proceed from the existing bundled Python tree or the
sibling `node_modules` symlink.
