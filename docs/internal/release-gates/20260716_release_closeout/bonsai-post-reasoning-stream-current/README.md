# Bonsai post-reasoning Electron stream proof — 2026-07-17

Status: `PASS-LIVE_SCOPED_BONSAI_ELECTRON_PAINT`; broader cross-model and
release gates remain `PARTIAL_NO_RELEASE`.

## Source trace

- Commit `a7b34bc4a` changes the shared Electron stream boundary, not Bonsai
  model output.
- `panel/src/main/ipc/chat.ts::streamSSE` now yields after each visible
  `chat:stream` delta while draining an already-buffered SSE read.
- `panel/src/renderer/src/components/chat/MessageBubble.tsx::useTypewriter`
  drains a just-finished streaming backlog rather than snapping it when the
  completion event is coalesced with pending content.
- Focused and wider affected panel verification passed 301/301 plus
  `tsc --noEmit`.

## Live verification

- Pre-fix current Electron row 360 persisted a coherent 992-token answer, but
  the observer saw its post-reasoning visible answer in one terminal paint.
- A raw `/v1/responses` probe against the same running Bonsai engine emitted
  406 reasoning deltas and 46 timed content deltas before one completed event.
  This isolated batching to the Electron main/renderer boundary.
- After a true Electron-main replacement, current row 363 exact-finaled while
  the DOM observer recorded 173 distinct visible-content mutations over
  1.998 seconds.
- Row 363 restored 216 tokens as `paged+ssm+disk`; health recorded four
  native-TQ q8 disk hits and one SSM companion disk hit.

## Boundary

This proves the repaired Electron paint path for the live Bonsai 1-bit
Responses route. It does not prove every model family, media route, protocol,
long-context case, or release package. Those remain open matrix rows.
