# Qwen full-KV mixed TurboQuant live gate

Date: 2026-07-17

Commit: `b64a16c61090a20ce7e520f4fa48ed70fe882dab`

Scoped verdict: **PASS-LIVE for the tested Qwen3-0.6B full-KV artifact**. This is
not a global TurboQuant, model-matrix, or release pass. Other Qwen subtypes,
hybrid families, VL/media routes, and the full release gates remain PARTIAL.

## Root cause

The paged/L2 transport was structurally intact: block slicing, native TQ
metadata, q4 scalar/batched decode, and restore axes all matched. The missing
gate was semantic model-scale codec quality. Existing tests checked only
shape/finite values and batch-vs-scalar equality.

On the real `mlx-community/Qwen3-0.6B-8bit` runtime, uniform native TQ q4
produced exact cold answers but corrupted every identical warm restore tested,
starting at 1,576 cached tokens. Uniform native TQ q8 stayed exact. A first/
last-three q8 boundary with q4 bulk removed gibberish but still produced one
strict-format miss. The accepted policy keeps q4 on 16 of 28 layers and uses q8
on the first/last six boundary layers.

## Source trace

- `vmlx_engine/utils/turboquant_config.py:107-201` owns the uncalibrated Auto
  policy. Full-KV Qwen now resolves to
  `qwen_full_kv_storage_tq4_critical_tq8`, q4 bulk, q8 boundary layers, and no
  mid-request live transition.
- `vmlx_engine/server.py:7891-7958` derives actual cache-template bit sets and
  reports `turboquant-q4/q8-mixed` rather than flattening mixed storage to q4.
- `tests/test_hybrid_live_tq_kv.py:121-178` pins both policy fields and the real
  cache object distribution.
- `CachePanel.tsx`, `PerformancePanel.tsx`, and `SessionConfigForm.tsx` show the
  q4/q8 bit sets and the first/last-six policy in the dev UI.

## Preserved red evidence

Uniform q4 cold rows were exact, while their identical warm rows restored
`paged+tq-native` and emitted newlines, digits, or repeated q-like fragments.
The failing cached-token counts were 1,576; 3,112; 6,184; 10,792; and 15,642.
This is treated as a real cache correctness failure, not model behavior, because
the identical cold prompts and the q8/mixed A/B rows were coherent.

The narrower q4 plus first/last-three q8 trial was coherent at long context but
had one strict-format miss (`CACHE-BOUNDARY Q3TQMIXR1-DONE` instead of the
requested marker only). It was not accepted as the final boundary.

## Current live proof

- Final first/last-six policy: 6/6 exact cold-to-warm pairs, with cached prefixes
  from 1,316 through 12,040 tokens. Each warm row reported
  `paged+tq-native` and emitted multiple incremental Responses text deltas.
- Process restart without clearing L2: the 12,036-token row returned exact
  `Q3C6R5-DONE` from `paged+disk+tq-native`, then exact again from RAM-backed
  `paged+tq-native`. Health recorded 756 native-TQ disk hits.
- Health exposed real `key_bits_values=[4,8]`, `value_bits_values=[4,8]`,
  `stored_prefix_quantization=turboquant-q4/q8-mixed`, and
  `auto_policy=qwen_full_kv_storage_tq4_critical_tq8`.
- Real Electron dev UI visibly showed `TQ4 bulk attention KV + TQ8 boundary
  layers` and `MIXED TQ4/8 AUTO`.
- Fresh Electron multi-turn Responses chat:
  - row 390 called `file_info(panel/package.json)` exactly once and ended exactly
    `Q3-MIX-UI1-DONE`;
  - row 393 reused 148 `paged+tq-native` tokens, called
    `file_info(README.md)` exactly once, and ended exactly
    `Q3-MIX-UI2-DONE`;
  - row 396 continued without a tool and ended exactly `Q3-MIX-UI3-DONE`.
  The renderer recorded 203 mutations for turn 1 and 294 mutations / 292
  distinct snapshots for turn 3; reasoning and final content were visibly
  progressive rather than a single final paint.

The third turn's prompt metric did not show broad conversation-prefix reuse;
health's last execution still reconstructed the 148-token shared prefix. That
is sufficient to prove this tested multi-turn path did not bypass cache
entirely, but multi-turn reuse efficiency remains a separate PARTIAL matrix row.

## Tests

- `tests/test_hybrid_live_tq_kv.py`: 21 passed.
- `tests/test_hybrid_live_tq_kv.py` + `tests/test_tq_paged_block_cache.py`:
  36 passed.
- selected `tests/test_engine_audit.py` Qwen/TQ/cache set: 169 passed.
- `panel/tests/settings-flow.test.ts`: 283 passed.
- panel TypeScript typecheck passed.
- Python compile and scoped `git diff --check` passed.

Release remains locked pending the remaining model/protocol/media/cache-pressure,
full-suite, signing, notarization, and update-feed gates.
