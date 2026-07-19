# MiniMax-M3 post-content terminal dispatch and 14-second video gate

Date: 2026-07-19

Scope verdict: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED` for the M3 terminal-delay
contract and large-video transport/cache reuse.  OCR/order quality remains
`PARTIAL`; REAP32 remains excluded from live retry because its earlier first
requests rebooted the host.  This is not a global release verdict.

## Root cause and source repair

The Responses stream was progressive until its last content delta, then stayed
silent for about 2.35 seconds before `response.output_text.done` and
`response.completed`.  Electron displayed the same pause before finalization.

The engine already called `Scheduler.step(defer_finished_cleanup=True)` and
delivered terminal `RequestOutput` objects before `_cleanup_finished`.  M3 was
still defeating that contract inside `Scheduler._process_batch_responses`: a
cache-hit request synchronously called `_prefill_for_prompt_only_cache()` to
rebuild the clean sparse-MSA prompt boundary before the terminal output could
be returned.

The fix keeps the required clean rederive but stores a typed deferred
descriptor during response finalization.  `_cleanup_finished` materializes the
clean M3 cache only after EngineCore has put the terminal output in the
collector.  Next-turn admission remains blocked on
`_terminal_cleanup_complete`, so cache ownership/durability is unchanged.

Source:

- `vmlx_engine/scheduler.py`
  - both paged and object-cache M3 hit branches schedule the shared
    `_deferred_prompt_cache` descriptor;
  - `_materialize_deferred_prompt_cache()` performs the clean N-1 prefill,
    typed-state extraction, optional legacy prompt-disk mirror, and normal
    paged/block-L2 store during post-dispatch cleanup.
- `vmlx_engine/engine_core.py`
  - existing terminal-first collector dispatch and cleanup admission barrier.
- `tests/test_terminal_dispatch_before_cache_cleanup.py`
- `tests/test_minimax_m3_cache_paths.py`

## Live Electron proof

The real Electron session used CDP 9335 and the exact Sessions/Server UI.  Save
& Restart replaced PID 40588 with PID 42270 and relaunched this argv:

```text
... MiniMax-M3-Coder-Small ... --tool-call-parser minimax_m3
--enable-auto-tool-choice --reasoning-parser minimax_m3
--cache-memory-percent 0.15 --use-paged-cache
--paged-cache-block-size 64 --max-cache-blocks 1000
--enable-block-disk-cache --block-disk-cache-max-gb 10 --stream-interval 1
```

The fresh chat visibly used Thinking Off, temperature 0, max tokens 256,
Responses wire, and built-in tools off.  It attached the real 1400x900,
28-frame, 14-second MP4 and sent the exact same prompt used before the fix.

Electron row 659:

```text
The video shows the following distinct alphanumeric markers:
- FRAME START 2468
- BANANA84426
- FRAME END 97553
```

Metrics: 31 output tokens; 1,706 prompt tokens; 1,701 `paged+disk` cached;
1.04s TTFT; 2.4s total.  SQLite retained non-empty visible content, no
reasoning, no tool call, and no warning.  The UI Logs showed the real
`video_url` request, 1,496 video tokens, a 27-block/1,701-token hit, one L2
block reconstruction, then the deferred rederive after terminal dispatch.

Artifacts:

- `mm3-terminal-fix-settings.png`
- `mm3-terminal-fix-ui-final.png`
- `mm3-terminal-fix-ui-logs.png`
- `ui-db-row.json`
- `ui-log-excerpt.txt`

## Raw Responses terminal timing

The exact 14-second `input_video` probe was repeated before and after the
source reload.  Both runs had progressive content deltas, one text-done event,
one completed event, and no reasoning/tool event.  The measured terminal gap
changed as follows:

| Measurement | Before | After |
|---|---:|---:|
| Last content to `response.output_text.done` | 2.3453s | 0.0415s |
| Last content to `response.completed` | 2.3458s | 0.0419s |
| Progressive content deltas | 30 | 40 |
| Cached input | 1,664 paged | 1,701 paged+disk |

The after-run restored all 1,701 cache-key tokens after the process restart.
Health then reported two scheduler hits, zero misses, 3,402 tokens saved, 28
block-disk hits, 1,738 resident RAM tokens, and 21,132 tokens on SSD.

M3 cache truth remained architecture-native:

- family/schema: `minimax_m3` / `minimax_m3_msa_v1`
- dense KV layers: 0-2
- sparse MSA layers with `idx_keys`: 3-59
- generic TurboQuant KV: disabled with reason
  `native_minimax_m3_msa_idx_keys`
- block-disk native-TQ writes/hits: 0/0

The store capability field `tq_native_enabled=true` is not evidence that M3
used TQ; its per-request/native-cache telemetry and counters prove it did not.

See `responses-terminal-before-after.json`, `health-after.json`, and
`terminal-cache-order.log`.

## Quality boundary

Transport, progressive streaming, terminal ordering, cache restoration, and
visible Electron persistence pass this scoped gate.  Exact OCR does not:

- fixture truth includes `BANANA8426`; M3 emitted `BANANA84426`;
- fixture truth includes `FRAME END 9753`; M3 emitted `FRAME END 97553` or
  `9755`;
- one raw repeat also emitted an extra `FRESH IMAGE` line and repeated BANANA.

Those are retained as model/runtime quality `PARTIAL`, not hidden by the
streaming repair.  The larger-video transport row is now exercised; a broader
quality catalog and REAP32 headroom remain open.

## Tests

Focused cache/terminal contracts: 51/51.

Expanded terminal, M3 cache/parser, multimodal history, media cache, and
Responses history selection: 101/101.

See `focused-tests.txt`.

## Fixture provenance

`mm3-large-abab-14s.mp4` is a mechanical A/B/A/B concat of already committed
real gate videos, normalized to 1400x900 at 2 fps.  It is 141,203 bytes and its
SHA-256 is:

```text
b79f92e10cdf5580c648b3ff5f3ce580f48cd2c3fcafa6d3c2eb3f8f3c1858ab
```
