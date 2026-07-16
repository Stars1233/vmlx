# MiniMax M2.7 current-source cache and agent-loop audit

Date: 2026-07-16  
Source after repair: `af7815f1a`  
Overall: `PASS-LIVE` cache/settings/tools/eviction; `PARTIAL` long direct stream.

## Source contract

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`.
- Bundle graph: `model_type=minimax_m2`, 62 attention-only layers, eight KV
  heads, head dimension 128, `use_mtp=false`.
- `vmlx_engine/model_configs.py:1181-1201` registers ordinary `cache_type=kv`,
  `tool_parser=minimax`, `reasoning_parser=minimax_m2`, and the two native stop
  markers.
- `vmlx_engine/cli.py:704-761` distinguishes omitted Auto from explicit None.
  The live Auto policy selects stored TQ8 for this compatible full-KV graph;
  explicit None sets `VMLX_DISABLE_TQ_KV` and passes the literal CLI flag.
- `vmlx_engine/prefix_cache.py:1484-1488`, `1608-1635`, and `1664-1690`
  establish fetched-table ref ownership for chain-hash and prefix-index hits.
  This is the owning fix for the live small-pool pin/leak.

## Focused tests

```text
tests/test_paged_cache.py                    45 passed
tests/test_paged_byte_budget.py              14 passed
tests/test_tq_paged_block_cache.py            9 passed
tests/test_hybrid_prefix_cache.py            22 passed
total                                        90 passed
```

The new regression test stores three usable blocks in a four-block manager,
releases them to cached-free state, fetches them, verifies that the fetched
table is registered, and verifies completion returns every ref to zero and all
three blocks to the free queue.

## Live Electron rows

| Row/PID | Route | Evidence | Result |
|---|---|---|---|
| 2187 / 62630 | Auto cold, first chat turn | 177 prompt tokens; one `file_info(panel/package.json)`; exact `MM27-LIVE1-DONE` | PASS |
| 2190 / 62630 | Auto same-chat second turn | 173 cached as `paged+tq-native`; one `file_info(vmlx_engine/model_configs.py)`; exact `MM27-LIVE2-DONE` | PASS |
| 2193 / 63682 | Auto after visible Stop/Load Model | 173/177 `paged+disk+tq-native`; 0.30s TTFT; exact tool/final | PASS |
| 2196 / 64194 | Explicit None cold | argv contains `--kv-cache-quantization none`; TQ disabled; exact tool/final | PASS |
| 2199 / 64579 | Explicit None restart | 161/165 `paged+disk`; no TQ tag/activity; exact tool/final | PASS |
| 2202, 2205 / 64928 | Four-block pressure before repair | coherent outputs, but three usable blocks remained pinned; logs reported `Out of cache blocks`; eviction stayed zero | FAIL retained; root cause |
| 2208 / 65838 | Four-block Auto after repair | 173/177 `paged+disk+tq-native`; exact tool/final; health returned three free blocks and recorded three evictions | PASS |
| 2211 / 65838 | Distinct pressured request | 64-token disk/TQ prefix; exact tool/final; refs free after completion; evictions rose to nine | PASS |
| 2214 / 66306 | Restored normal Auto/1,000 blocks | 173/177 `paged+disk+tq-native`; exact tool/final; 999 free blocks after completion | PASS |

## Persisted payload truth

Auto namespace: `/Users/eric/.cache/vmlx-engine/block-cache/4fcc1f43edca`.

- A 64-token record is 8,877,576 bytes and indexed as `dtype=turboquant_kv`.
- It contains 373 tensors: one metadata tensor plus six packed/norm tensors for
  each of 62 layers.
- Embedded metadata marks layer 0 through 61 as `turboquant_kv`, with 8-bit K/V,
  eight KV heads, 64 tokens, and head dimension 128.
- The corresponding raw BF16 KV payload is approximately 16.25 MB per block;
  the stored TQ file is approximately 8.88 MB.

None namespace: `/Users/eric/.cache/vmlx-engine/block-cache/b9f58de8797d`.

- The index records `dtype=kv`, never `turboquant_kv`.
- A 64-token raw record is 32,518,867 bytes because the correctness-preserving
  NumPy bridge holds BF16-origin values as float32 rather than clipping through
  float16.
- Auto and None do not read one another's persisted blocks.

## Visual evidence

- `mm27-auto-cache-settings-controls.png`
- `mm27-live2-multiturn-pass.png`
- `mm27-live1-disk-restart-pass.png`
- `mm27-none-setting-before-restart.png`
- `mm27-none-disk-restart-pass.png`
- `mm27-auto-4blocks-before-restart.png`
- `mm27-fixed-eviction-reuse-pass.png`
- `mm27-restored-normal-disk-pass.png`

## Remaining gate

Do not promote this row to a full model/release pass until a long tools-off
visible answer and a direct streaming/API rail soak prove complete reasoning
deltas, content deltas, stop/final events, and no truncation or parser stall.
