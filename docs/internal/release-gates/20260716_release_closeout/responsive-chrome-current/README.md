# Responsive Electron chrome evidence

Scope: current Electron dev build at commit `073d15d3c`, 2026-07-17.

## Live failure

- At 700 px before the repair, the session Stop control occupied x=697..739
  while the viewport and body were 700 px wide. The header's horizontal
  overflow affordance was not visible, so the control was clipped.
- At 600 px before the repair, the body overflowed to 651 px, the language and
  settings controls ended at x=613 and x=639, and the session toolbar extended
  through x=739.

## Source trace

- `panel/src/renderer/src/components/layout/TitleBar.tsx` gives every mode
  button a title and accessible label, compacts only its visible text below
  720 px, and keeps the title bar within its viewport.
- `panel/src/renderer/src/components/sessions/SessionView.tsx` wraps the
  session header and moves the complete action toolbar onto a full-width row
  below 800 px instead of clipping it.

## Live post-repair matrix

- 600x650: body `600/600`; all first 20 chrome controls inside the viewport.
  Mode buttons remained individually accessible as 28 px icon controls. The
  session toolbar wrapped; Stop ended at x=584.
- 700x650: body `700/700`; every sampled chrome control visible; maximum right
  edge x=688.
- 900x700: body `900/900`; every sampled chrome control visible; maximum right
  edge x=888 and the session toolbar returned to a single row.
- 1400x900: body `1400/1400`; all sampled controls visible after restoring the
  working window size.

The directory retains before/after screenshots at 600, 700, and 900 px plus
the restored 1400 px state. The layout contract suite passed 189/189 and panel
TypeScript typecheck passed. This proves the current Server/session chrome; it
does not assert that every modal or every translated locale has been visually
audited at every width.
