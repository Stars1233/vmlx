# Path-dependent cache terminal-first cleanup

Date: 2026-07-19

Source verdict: `FIXED_SOURCE`.  Current-source live verdict is
`VERIFIED-LIVE_SCOPED` for MiniMax-M3 only.  DSV4, ZAYA, and mixed-SWA family
runtime rows remain `PARTIAL` until each is reloaded and measured through raw
streaming plus the Electron UI.

## Audit result

The M3 terminal-delay repair exposed the same lifecycle violation in other
path-dependent cache branches.  Before this change,
`Scheduler._process_batch_responses()` had six terminal-path clean-prefill
branches:

| Family/cache | Paged branch | Object branch |
|---|---:|---:|
| DSV4 composite SWA/CSA/HCA | 1 | 0 |
| ZAYA CCA | 1 | 0 |
| Mixed full/SWA (Gemma/Step/MiMo class) | 1 | 1 |
| MiniMax-M3 MSA | 1 | 1 |

Each direct `_prefill_for_prompt_only_cache()` could hold the final SSE/UI
terminal event behind a full clean prefill.  The response finalizer now has
zero direct calls.  All six branches schedule `_deferred_prompt_cache` with an
explicit family, mode (`paged` or `object`), and clean N-1 key.

`Scheduler._materialize_deferred_prompt_cache()` runs from
`_cleanup_finished()` only after EngineCore dispatches terminal output.  The
existing `_terminal_cleanup_complete` barrier still prevents the next request
from entering while the clean cache is extracted/stored.

This does not change cache math or introduce synthetic output behavior.  It
only moves the already-required clean rederive to the correct lifecycle phase.

## Validation

Focused current-source selection:

```text
235 collected
229 passed
6 skipped
```

The selection includes terminal-order guards, stream terminal guards, M3 typed
cache/parser/media/history, the complete DSV4 paged-cache file, and the
complete ZAYA runtime file.  Two real small ZAYA scheduler tests still execute
the clean typed-CCA rederive and prove the prefix remains fetchable after
`Scheduler.step()` performs its synchronous-default cleanup.

Static terminal ownership guard:

```text
_process_batch_responses direct _prefill_for_prompt_only_cache calls: 0
deferred paged descriptors: 4
deferred object descriptors: 2
families: DSV4, ZAYA, Mixed-SWA, MiniMax-M3
```

See `focused-tests.txt` and `source-callsite-inventory.json`.

## Current-source live M3 regression

A real Electron Save & Restart replaced PID 42270 with PID 43998.  The exact
14-second video probe then restored 1,701 `paged+disk` tokens, emitted 40
progressive content deltas, one text-done, and one completed event.  The final
content-to-done gap was 0.0414 seconds and content-to-completed was 0.0419
seconds.

A new visible Electron chat independently attached the MP4 and persisted row
662 with non-empty content, no reasoning, no tool call, and no warning:

```text
31 tokens
1,706 prompt (1,701 paged+disk cached)
1.18s TTFT
2.5s total
```

The visible Logs panel showed 27 blocks/1,701 tokens restored, one block
reconstructed from L2, terminal-first deferred scheduling, and a later 60-layer
typed M3 store.

Artifacts:

- `global-terminal-mm3-ui-final.png`
- `global-terminal-mm3-ui-logs.png`
- `m3-current-source-terminal.json`

OCR quality remains partial exactly as documented in the preceding M3 gate.

## Remaining live rows

No blanket runtime claim is made for the other families.  Required follow-up:

1. DSV4 Flash: raw Chat/Responses and visible Electron cache-hit/cold rows;
   verify native composite cache and no terminal hold.
2. Mixed-SWA: Gemma 4 and Step 3.7 (plus MiMo where configured), raw and
   Electron; verify rotating/full layer store plus TQ-compatible full-KV lanes.
3. ZAYA: use an available external-drive artifact; if unavailable, retain
   source/test proof only and keep the runtime row partial.

openPangu is not part of this defect: its exact typed N-1 snapshot is captured
before decode and does not call the clean-prefill helper in response
finalization.
