# Minimum-width accessibility/localization follow-through

Date: 2026-07-20

Status: `VERIFIED-LIVE-CURRENT-SOURCE-SCOPED`; overall production matrix remains `PARTIAL`.

## Scope

This follow-through audits the remaining English-only icon controls found in the
real Korean Electron Chat surface at the supported 600x760 minimum window size.
It does not claim a new signed-app package or a model/runtime gate.

Base source before this change:
`9531eea958b9ee7c53bc5d8c48b78f3286006baa` on
`erics-m5-max.local:/Users/eric/mlx/vllm-mlx-release-1.6.13`.

## Source trace

- `panel/src/renderer/src/components/chat/VoiceChat.tsx`
  - uses the existing `chat.voice.*` catalog for recording/transcription titles
    and known error fallbacks;
  - exposes the same localized state label through `title` and `aria-label`;
  - includes `sessionId` and `t` in the recording callback dependency list.
- `panel/src/renderer/src/components/ui/theme-toggle.tsx`
  - removes the literal `Theme: ${theme}` title;
  - exposes localized theme state through both `title` and `aria-label`.
- `panel/src/renderer/src/i18n/locales/{en,zh,ko,ja,es}.json`
  - adds a complete `common.theme.{dark,light,system}` key tree.
- `panel/tests/i18n-consistency.test.ts`
  - pins all VoiceChat catalog calls and both accessible-name contracts.

## Live Electron proof

The real current-source Electron app ran on the other Mac at CDP 9335 with
profile `/Users/eric/.vmlx-v1613-responsive-dev`, Korean locale, and a 600x760
viewport.

- `live-visible-button-catalog.json` records the voice button as
  `title=aria-label=클릭하여 음성 입력 시작` and the theme button as
  `title=aria-label=어두운 테마`.
- `live-theme-cycle.json` records the real click sequence and effective DOM
  state: dark / `어두운 테마` -> light / `밝은 테마` -> system /
  `시스템 테마` -> restored dark / `어두운 테마`.
- `live-keyboard-focus.json` records 25 unique Tab stops on the current Chat
  surface. Every stop had a text/title/aria/placeholder name and remained
  fully inside the 600x760 viewport; `failures` is empty.
- `live-600px-ko-accessibility.png` is the visually inspected final surface;
  SHA-256 is `c23fb6ec07d147600cb705e94982c30fc2a37ab402918906a2d9d6fc8199816a`.
- `live-renderer-reload.json` records a clean explicit renderer reload with no
  console error or `pageerror` event and a non-empty Korean app surface.

Updating all five locale JSON files caused Vite to invalidate the mixed-export
i18n module and perform a development-only full-page reload. A screenshot taken
during that reload captured an empty root. An explicit reload immediately
mounted the complete app with no renderer errors. This is retained as an HMR
observation, not counted as packaged-product blank-window evidence.

## Verification

- Focused i18n suite: 12/12 passed.
- Complete panel suite: 76 files, 2,340 passed, 3 skipped.
- TypeScript `tsc --noEmit`: passed.
- Direct production `electron-vite build`: main, preload, and renderer passed.
- `git diff --check`: passed.

Logs are preserved in this directory as `panel-full-test.log`,
`panel-typecheck.log`, `panel-electron-vite-build.log`, and
`git-diff-check.log`.

## Honest boundary

This closes the observed theme/voice accessible-name defects and the base Chat
keyboard traversal at minimum width. Forced transient answer/image states,
remaining secondary custom modals, a full screen-reader semantics pass,
drawer/modal-specific keyboard traversal, and the next signed-app repeat remain
open. No model was loaded or generated in this scoped UI follow-through.
