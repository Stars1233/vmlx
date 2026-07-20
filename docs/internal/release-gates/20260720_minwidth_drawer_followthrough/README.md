# Minimum-width drawer and localization follow-through

Date: 2026-07-20

Scope: `api/ui` only. This checkpoint fixes and proves current-source Electron
toolbar/drawer behavior at a 600 x 760 viewport. It does not clear the broader
model, cache, media, gateway, accessibility, installed-app, or release matrix.

## Source trace

- `panel/src/renderer/src/components/layout/ChatModeToolbar.tsx`
  - wraps the session toolbar at narrow widths;
  - makes the model selector shrinkable;
  - places Chat and Server drawers in explicit, right-aligned `w-full`
    containers bounded by `max-w-80` and `max-w-96`.
- `panel/src/renderer/src/components/chat/ChatSettings.tsx`
  - bounds the drawer to `w-full max-w-80`;
  - routes the observed agent/tool/profile/status/close/Brave controls through
    the existing translation catalogs.
- `panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx`
  - replaces the unconditional 384 px root width with
    `w-full max-w-96`, so it cannot extend outside a narrower main pane.
- `panel/tests/layout-shell.test.ts` and
  `panel/tests/i18n-consistency.test.ts` pin the responsive and catalog-backed
  source contracts.

The exact current diff is preserved in `source.diff`.

## Retained failure controls

1. The signed v1.6.13 Sequoia app at 600 px clipped lifecycle/settings controls
   beyond the viewport. See `signed-sequoia-toolbar-before.png`.
2. Before the Server drawer width repair, live current-source Electron measured:
   - visible main pane: `x=260..600` (340 px);
   - Server drawer: `x=216..600` (384 px);
   - the main element had `overflow:hidden`, so the drawer's first 44 px and
     the beginnings of Korean labels were hidden.
   See `korean-server-drawer-before.png`.

These are retained failures, not passing evidence.

## Current-source live Electron proof

Host: `Erics-M5-Max.lan`

Worktree base: `f8bc773a8e82f4e5fcea8db3ead2fee012d1470b`

Electron:

- source: `/Users/eric/mlx/vllm-mlx-release-1.6.13/panel`;
- CDP: `127.0.0.1:9335`;
- user data: `/Users/eric/.vmlx-v1613-responsive-dev`;
- engine lookup in the dev main log:
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.

At 600 x 760:

- the toolbar has document width `600/600` and no horizontally clipped
  controls (`current-source-toolbar-after.png`);
- the Korean Server drawer and wrapper both measure exactly `x=260..600`,
  matching the visible main pane; labels are fully visible
  (`korean-server-drawer-after.json` and `.png`);
- the Korean Chat drawer measures `x=280..600`; translated agent controls are
  present and the stale English target list is empty
  (`korean-chat-drawer-after.json` and `.png`);
- the real Clear All action opened a native Korean `confirm` sheet:
  `모든 모델의 채팅 3개를 모두 삭제하시겠습니까?` followed by the localized
  irreversible-deletion warning. The sheet was dismissed and all three chats
  remained (`korean-native-confirm.json`).

## Verification

- Complete panel suite: `76` files, `2,339 passed`, `3 skipped`.
  See `full-panel.log`.
- TypeScript typecheck: pass. See `typecheck.log`.
- Main/preload/renderer production source build via `electron-vite build`:
  pass. See `electron-vite-build.log`.
- `git diff --check`: pass before evidence capture.
- The umbrella `npm run build` correctly refused because the external
  `/Users/eric/jang/jang-tools` checkout contains tracked changes. The dirty
  source guard was not bypassed. See `bundle-safety-refusal.log` and
  `bundle-safety-exit-code.txt`.

## Verdict

`VERIFIED_LIVE_CURRENT_SOURCE_SCOPED` for the 600 px toolbar, Chat/Server drawer
width, observed Korean Chat Settings localization, and dismissed Korean native
bulk-delete confirmation.

Still partial/open:

- repeat the repaired surfaces in a future signed/notarized app;
- remaining secondary/transient modals and full keyboard/screen-reader traversal;
- image/wait/empty-response states that still have source-contract-only proof;
- the full model/cache/media/protocol/gateway/eager-load release matrix.
