# Minimum-width locale and drawer gate — 2026-07-19

Status: `VERIFIED-LIVE-SCOPED`; this is not a release-wide UI pass.

## Defect

At the 600x760 minimum proof window, the base chrome no longer overflowed, but
several visible surfaces ignored the selected locale: Create Session and its
remote endpoint, Server Settings title/footer, session/chat controls, chat
history grouping and bulk-delete copy, Casual/Expert mode, message actions,
TTS, version/copyright, and API-key settings. A second current-source sweep
found the same defect in the Code and chat-empty screens, remote-model picker,
stream-wait/empty-response states, markdown code-copy feedback, session side
panels, image-session labels, and several fallback errors. The Server Settings
and side-panel close buttons also lacked localized accessible names.

## Source repair

- Routed those strings through the existing renderer locale catalog in
  `App.tsx`, `CreateSession.tsx`, `ServerSettingsDrawer.tsx`, `SessionView.tsx`,
  `ChatModeToolbar.tsx`, `ChatHistory.tsx`, `InferenceMode.tsx`,
  `MessageBubble.tsx`, and `VoiceChat.tsx`.
- Removed the singleton English markdown renderer labels: every rendered code
  block now gets the active catalog's default-language and Copy labels, and
  click feedback uses the current locale instead of literal `Copy/Copied!`.
- Added matching keys to all five shipped catalogs: English, Chinese, Korean,
  Japanese, and Spanish.
- Extended `i18n-consistency.test.ts` so missing keys, placeholder drift, and
  these source surfaces fail the catalog contract.

## Live Electron evidence

The full Electron main process was relaunched from current source with
`VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1611-cachefix-dev` and CDP 9335. The
startup log reported:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
[Engine Manager] Version: 1.6.11
```

At 600x760, About/API Keys was selected through the real UI and then switched
through all five locale buttons. Every row reported
`documentElement.scrollWidth == clientWidth == 600` and the visible interactive
element clipping probe returned `[]`. The rendered copy included:

- Korean: `버전 1.6.11`, `API 키`, `Brave Search API 키`, `HuggingFace 토큰`.
- Japanese: `バージョン 1.6.11`, `API キー`.
- Chinese: `版本 1.6.11`, `API 密钥`.
- Spanish: `Versión 1.6.11`, `CLAVES DE API`.
- English: `Version 1.6.11`, `API KEYS`.

Korean minimum-width live rows additionally exercised Chat, Create Session,
the remote endpoint tab, and the open Server Settings drawer. All had a 600 px
document width, zero sampled clipped controls, and localized labels. The chat
view visibly rendered `시작`, `채팅`, `서버`, `말하기`; Create Session rendered
`세션 생성`, `로컬 모델`, `원격 엔드포인트`; the drawer rendered
`서버 설정`, `저장`, `재설정`, with a localized close accessible name.

The expanded sweep was then exercised in the same running Electron main:

- the Code screen rendered the catalog strings in all five locales at 600 px;
  every measured row had `scrollWidth == clientWidth == 600`, no raw catalog
  key, and no clipped main-surface element;
- Japanese Chat opened the real model picker and its inline remote-endpoint
  form. `リモートエンドポイントに接続`, `リモート接続`, the URL/model/key
  placeholders, and Connect state were visible without panel overlap;
- representative Code screenshots were manually inspected in English and
  Spanish, and the Japanese remote picker was manually inspected.

Screenshots:

- `ui-minwidth-ko-chat-fixed2.png`
- `ui-minwidth-ko-about-fixed.png`
- `ui-minwidth-ja-about-fixed.png`
- `ui-minwidth-zh-about-fixed.png`
- `ui-minwidth-es-about-fixed.png`
- `ui-minwidth-en-about-fixed.png`
- `ui-minwidth-ko-create-fixed2.png`
- `ui-minwidth-ko-remote-fixed2.png`
- `ui-minwidth-ko-server-drawer-fixed2.png`
- `ui-minwidth-{en,zh,ko,ja,es}-code-expanded2.png` (English uses
  `ui-minwidth-en-code-expanded.png`; Spanish uses the `expanded2` capture)
- `ui-minwidth-ja-remote-expanded.png`

## Verification boundary

Current-source verification after the repair:

- focused i18n/layout/chat slice: **341 passed**;
- complete panel suite: **2,326 passed, 3 skipped**;
- complete Python suite with the clean JANG source: **6,153 passed, 96 skipped,
  92 deselected**;
- TypeScript typecheck: passed;
- production renderer/main/preload build with clean JANG source
  `9081c92476a63b912f4d2ce96146674971b5c83e`: passed, including bundled
  Python critical hash/import verification.

The first production-build attempt intentionally stopped on the dirty primary
JANG checkout. The passing build used the previously documented clean JANG
checkout rather than bypassing that safety gate. Logs are retained as
`full-panel-final.log`, `clean-jang-build.log`, and
`electron-vite-build-final.log`.

The first complete Python run used the corrected Node `PATH` but no explicit
clean-JANG source. It reached 6,151 passes and exposed two failures: the bundle
verifier compared the clean bundle against the dirty primary JANG checkout,
and an old contract still demanded the literal English Clear All aria label.
The contract now requires the localized catalog key. Both affected tests pass
with the clean JANG input, and the complete clean-input rerun passed 6,153
tests. The failed first run remains preserved as `full-python-first-run.log`;
the passing current-source run is `full-python-clean-jang.log`.

This closes the observed hardcoded-copy and minimum-width failures on the
named and newly swept surfaces in the current dev Electron build. The stream
wait/empty state and image-session side panels have source and contract proof
but were not forced into their transient live states in this pass. It also does
not prove every secondary/destructive modal, native confirmation sheet, full
keyboard/screen-reader traversal, or the signed packaged application; those
remain explicit follow-up rows.
