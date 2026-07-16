# DSV4 reasoning, cache-control, and restart proof (2026-07-16)

Overall status: **PARTIAL — no release**.

This packet proves the scoped DSML reasoning-boundary repair and current
native DSV4 cache-control/restart path in the live Electron dev build. It does
not prove the unavailable DSV4 JANGTQ artifact, stable latency, or the broader
model/release matrix.

## Current source trace

- `vmlx_engine/server.py:18073,18653-18689` tracks the already-emitted safe
  reasoning prefix. Exact-once requests stream genuine reasoning while
  suppressing native DSML once its marker appears.
- `vmlx_engine/server.py:18917-18932,19635-19642` uses the cleaned reasoning
  prefix for the final reasoning-summary and completed Responses output while
  retaining the raw accumulator for structured tool parsing.
- `tests/test_server.py:2850` exercises DSML on the reasoning rail and requires
  one structured `file_info` call with no DSML in deltas, done text, or the
  completed response.
- `tests/test_dsv4_paged_cache.py` now asserts the current shared
  native-typed-cache UI ownership rules after openPangu routing was added.
- `tests/test_dsv4_batch_generator_speed.py` explicitly initializes its
  `__new__`-only no-snapshot fixture; production construction already sets the
  snapshot controls in `DSV4BatchGenerator.__init__`.

## Live Electron evidence

Model actually loaded:
`/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`.
No DSV4 JANGTQ bundle was present on the inspected volume, so this run does not
claim JANGTQ proof.

Current server argv included `--tool-call-parser dsml`,
`--reasoning-parser deepseek_r1`, `--dsv4-enable-prefix-cache`,
`--use-paged-cache`, and `--enable-block-disk-cache`.

- Pre-fix DB row 1863 reproduced raw DSML in `reasoning_content` despite one
  valid `file_info(README.md)` and exact visible final content. See
  `dsv4-current-reasoning-dsml-leak.png`.
- Post-fix fresh row 1869 returned exactly `DSV4-FIX1-DONE`, with one real
  `file_info(panel/package.json)`, one result, no warning, and no DSML leak.
- Same-chat row 1872 returned exactly `DSV4-FIX2-DONE`, with one real
  `file_info(README.md)`, one result, no warning, and only the safe reasoning
  prefix. See `dsv4-dsml-reasoning-fix-multiturn.png`.
- After a visible Stop/Load process restart, row 1875 again returned exactly
  `DSV4-FIX1-DONE`, one call, and one result. Electron showed 606
  `paged+dsv4` cached prompt tokens, 1.16 s TTFT, and the health endpoint
  reported three block-disk hits with `DSV4BatchGenerator` active. See
  `dsv4-current-restart-disk-hit.png`.
- The expanded Cache settings were visually inspected. Turning DSV4 Native
  Composite Prefix Cache off and saving restarted with only
  `--disable-prefix-cache`; health then reported prefix, paged, block L2, and
  pool quant disabled. Re-enabling composite and the independent pool codec
  restored all four. See the settings screenshots.

## Current tests

- `tests/test_server.py`: 103 passed, 3 deselected.
- `tests/test_streaming_reasoning.py`: 131 passed.
- Selected DSV4 cache/tool/restart/finalizer/generator suite: 114 passed.
- `git diff --check`: clean for the four scoped files.

## Open gates

- Row 1866 remains a preserved failure sample: malformed native tool output,
  no call, and 64.7 s total time.
- Restart row 1875 was structurally correct but consumed 1,454 tokens and
  119.2 s. DSV4 termination/latency variability is therefore still PARTIAL.
- Current live proof covers the available CRACK bundle only, not the missing
  JANGTQ bundle.
- Cross-protocol, media, other-model, signing, notarization, and release gates
  remain open.
