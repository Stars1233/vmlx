# Cache tier naming and tooltip-click proof

Date: 2026-07-22

Verdict: `VERIFIED-LIVE_SCOPED` for terminology and tooltip interaction only.
This checkpoint does not change cache defaults, launch flags, storage formats,
lookup order, eviction, or restore behavior.

## User-facing contract

- `In-Memory Paged Cache (RAM)` names the fast, non-persistent block tier in
  Apple unified memory. The UI deliberately does not call it GPU RAM because
  Apple Silicon CPU and GPU share the same unified-memory pool.
- `Block Disk Cache (SSD / L2)` names the persistent content-addressed block
  tier. With the RAM tier enabled it is L2; supported architectures may also
  use it as the authoritative SSD-only tier when the RAM tier is disabled.
- Backend and command-line compatibility names such as
  `--use-paged-cache` remain unchanged.

## Source trace

- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` owns the
  Server Settings labels, explanations, and tooltip behavior.
- `panel/src/renderer/src/components/sessions/CachePanel.tsx` owns the live
  cache-status heading.
- `panel/src/renderer/src/components/sessions/PerformancePanel.tsx` owns the
  compact cache-stack and SSD-L2 status labels.
- `panel/src/shared/cacheCapacityDisplay.ts` and `panel/src/main/sessions.ts`
  render effective-capacity and launch-log explanations.
- `vmlx_engine/cli.py` explains the same RAM/SSD distinction in `serve` and
  `bench` help without changing flags.
- All five shipped locale JSON files were updated for the primary setting
  label.

## Regression found during live UI verification

The first Electron click on the outer `?` tooltip wrapper also toggled the
containing checkbox. The tooltip's event suppression lived only on its inner
glyph, so clicking wrapper padding followed the enclosing `<label>` default
action. `SessionConfigForm.tsx` now owns `ref`, click suppression, and hover
handlers on the outer wrapper. The unsaved negative-control change was
immediately restored and was never persisted or restarted.

Live recheck through the running dev Electron app on CDP 9335:

- RAM tooltip wrapper: checked `true -> true`; full Apple unified-memory
  explanation appeared.
- SSD/L2 tooltip wrapper: checked `true -> true`; full persistent-tier and
  SSD-only explanation appeared.
- `r16-cache-names-tooltip-fixed.png` and
  `r16-cache-names-block-tooltip-fixed2.png` retain those visible states.

## Current live status-surface proof

- `r16-cache-panel-names.png`: real Qwen3.6 35B JANGTQ Electron session,
  Cache panel text contained `BLOCK DISK CACHE (SSD / L2)` and displayed the
  existing live block/SSM cache counters. The heading is below the screenshot
  viewport, so the accompanying CDP text read is the label evidence; the
  screenshot retains the exact session and panel context.
- `r16-perf-panel-cache-names.png`: real Laguna S-2.1 JANG_2L Electron
  session, Perf panel visibly shows `RAM paged + SSD L2` and the active q4
  attention-KV cache contract. The CDP text read also reported
  `Block Disk L2 (SSD) — 227 blocks / 13,802 tokens` lower in the same panel.
- `r16-cache-names-expanded.png`: Server Settings visibly shows the expanded
  `In-Memory Paged Cache (RAM)` section.
- `r16-cache-cli-help.txt`: live remote CLI help names Apple unified memory,
  the SSD-only tier, and Block Disk Cache without the obsolete claim that
  block-disk persistence always requires the RAM tier.

The active Qwen process was replaced by a concurrently started Laguna process
between the Cache and Perf captures, consistent with the configured
single-model ownership lane. Laguna was subsequently stopped by that separate
lane before a `/health` artifact could be retained. Therefore this directory
does not claim a new process-lifetime or cache-restore result; the scoped proof
is UI terminology and interaction only.

## Tests

Both synchronized source checkouts ran:

- panel: 293 focused tests passed;
- panel TypeScript typecheck passed;
- Python CLI/DSV4 cache contracts: 94 tests passed;
- `git diff --check` passed.

Outputs are retained in `local-focused-tests.txt` and
`remote-focused-tests.txt`.
