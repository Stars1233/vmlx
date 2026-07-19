# Block-disk-only partial prefix gate (2026-07-19)

## Verdict

`PASS-LIVE scoped` for the generic full-attention KV path exercised by
`jangq-ai/MiniMax-M2.7-Small-JANGTQ`:

- Paged RAM was explicitly Off while Prefix Cache and Block Disk Cache (L2)
  remained On.
- The real Electron Start action launched the current source with
  `--no-paged-cache`, `--enable-block-disk-cache`, 64-token blocks, and four
  configured blocks.
- Because block 0 is reserved, the effective capacity was correctly reported
  as three usable blocks / 192 tokens on the UI, launch log, and `/health`.
- A process restart restored an identical fresh-chat prefix as 192
  `block-disk+tq-native` tokens from SSD and produced the exact visible marker.
- A separate long raw Chat API repeat restored a 192/846-token partial prefix,
  streamed seven content deltas, and finished with `stop` plus usage.
- Chat Completions and Responses both streamed reasoning separately from
  progressive visible content and emitted their correct terminal events.
- The active disk-only pool retained zero KV payload bytes in RAM after each
  completed request.

This is not a campaign-wide cache or release pass. Native/typed cache families,
hybrid SSM companion rederive, mixed-SWA, signed-app repetition, and the retained
release matrix remain separately gated.

## Artifact identity

The live bundle is JANGTQ/MXTQ, not affine JANG and not base MLX/MXFP:

- `quantization.codec=turboquant_codebook`
- `weight_format=mxtq`
- `profile=JANGTQ2`
- routed experts are 2-bit while attention/dense/embed/head roles are 8-bit
- stored prefix KV uses TurboQuant q4 (`key_bits=4`, `value_bits=4`)
- MiniMax-M2.7 is text-only; no VL capability is claimed

See `diskonly-m27-health-final-process.json` and
`diskonly-m27-health-final-hit.json`.

## Issues found and disposition

| ID | Finding | Disposition | Evidence |
|---|---|---|---|
| DISKONLY-01 | Block SSD L2 was coupled to `use_paged_cache`, so explicit Paged Off could suppress durable block reuse. | Fixed and current-source live-proven. The block-aware index is built when paged RAM **or** block L2 is active. | Source changes in `cli.py`, `scheduler.py`, `server.py`; session DB, argv, health, restart hit. |
| DISKONLY-02 | A pure SSD backend needed a durability barrier; otherwise a background write could be followed by dropping the only RAM payload before the file/SQLite row was visible. | Fixed. `write_block_async` reports enqueue success and `wait_for_blocks` gates RAM-payload release. Write failure retains a telemetry-visible RAM fallback rather than silently losing the prefix. | `block_disk_store.py`, `prefix_cache.py`, 13 focused Python tests. |
| DISKONLY-03 | Exact and partial prefix discovery had to survive process restart without a resident trie/payload. | Fixed and live-proven. Persisted partial sizes plus chained block hashes restored 192 tokens after restart; a real tensor test restores 7/8 tokens. | Electron row 563, raw API cache-hit summary, `test_disk_only_store_and_restart_restore_exact_partial_prefix`. |
| DISKONLY-04 | UI policy forced Block L2 off when Paged was off and fresh-session normalization could overwrite an explicit toggle. | Fixed in shared policy, create/reset/save/preview paths. Explicit Off is source/test-proven; the live row proves the independent `Paged Off + Block L2 On` combination. | `cacheControlPolicy.ts`, session components, 299 panel tests, `electron-session-cache-config.json`. |
| DISKONLY-05 | UI and launch logs overstated capacity as `block_size * max_blocks`; the engine permanently reserves block 0. A four-block pool was shown as 256 tokens although only 192 were usable. | Fixed on every current surface. UI minimum is now two blocks; capacity is `block_size * (max_blocks - 1)`. | Final settings screenshot, corrected live log, `/health usable_blocks=3 capacity_tokens=192`, unit tests. |
| DISKONLY-06 | Idle utilization counted the reserved null block, reporting 25% usage in a four-block pool before any user cache existed. | Fixed. Utilization now excludes the null block and reports `0.0` on the freshly restarted idle engine. | `paged_cache.py`, `test_reserved_null_block_is_zero_utilization`, final health JSON. |
| DISKONLY-07 | `scheduler_cache.disk_hits` counted only promotions while `block_disk_cache.disk_hits` counted actual successful SSD reads, yielding contradictory health values in disk-only mode. | Fixed without erasing the old meaning. Public disk hit/miss fields use BlockDiskStore; promotion and lookup-only counters remain separately named. | `prefix_cache.py`, focused tensor round-trip test, final health (`disk_counter_source=block_disk_store`). |
| DISKONLY-08 | Disk-cache help text still said block L2 “works with paged cache,” implying it required paged RAM. | Fixed. Current text explicitly covers Paged On or explicitly Off and states mutual exclusion with legacy prompt disk format. | `diskonly-m27-final-settings-truth.png`. |
| DISKONLY-OBS-01 | One tools-enabled replay interpreted the synthetic word `DISKONLY` as a file task, performed `write_file` then `read_file`, and took 113.3s. | Classified as model/tool-choice behavior for this ambiguous synthetic prompt, not hidden as a cache pass. The loop still executed two schema-valid calls/results, restored 576 aggregate SSD tokens across three iterations, emitted the exact final marker, and stored no warning. The untracked `secret.txt` test artifact was removed. Raw no-tool Chat/Responses rows did not call tools and streamed correctly. | Electron row 566 and final screenshot. |
| HARNESS-OBS-01 | The first filler generator used BSD `jot` with the wrong argument order and flooded the shell. | Harness-only; no request reached the app. Replaced with bounded Node string generation. Not counted as a runtime failure or pass. | Command transcript only; excluded from product evidence. |

## Source trace

- Engine policy and backend construction:
  `vmlx_engine/cli.py`, `vmlx_engine/scheduler.py`
- SSD durability and visibility barrier:
  `vmlx_engine/block_disk_store.py`, `vmlx_engine/prefix_cache.py`
- disk-only block metadata, partial discovery, capacity/utilization truth:
  `vmlx_engine/paged_cache.py`
- health/capability truth:
  `vmlx_engine/server.py`
- UI defaults, explicit-toggle preservation, preview/argv, capacity/help copy:
  `panel/src/shared/cacheControlPolicy.ts`,
  `panel/src/shared/cacheCapacityDisplay.ts`, `panel/src/main/sessions.ts`, and
  the four session settings components in this commit
- Regression coverage:
  `tests/test_block_disk_default.py`, `tests/test_batching.py`,
  `tests/test_paged_cache.py`, panel cache/settings tests, and the cache
  architecture contract orchestrator

## Live Electron proof

### Settings, DB, preview, and argv

The real session ID `86b9d9f1-7987-46b4-8a45-a25c9cf0109b` persisted:

- `usePagedCache=false`
- `enablePrefixCache=true`
- `enableBlockDiskCache=true`
- `enableDiskCache=false`
- `pagedCacheBlockSize=64`
- `maxCacheBlocks=4`
- custom SSD directory
- `kvCacheQuantization=auto`
- cache-default migration version 11

The real Electron Start action launched PID 20946 with the project venv engine
and the expected `--no-paged-cache` / block-L2 flags. The log contains:

```text
Block disk-only index capacity: 64 tokens/block x 3 usable blocks (4 configured; 1 reserved) = 192 indexed tokens.
PagedCacheManager initialized: block_size=64, max_blocks=4, usable_blocks=3, max_tokens=192, backend=block-disk-only
```

See `electron-session-cache-config.json`, `live-engine-argv.txt`,
`live-log-excerpt.txt`, `diskonly-m27-final-settings-truth.png`, and
`diskonly-m27-final-corrected-logs.png`.

### Cold/write, partial reuse, and process restart

The initial empty directory wrote three 64-token q4 TQ-native blocks. The
capacity-limited store logged 192/1490 tokens and released every persistent RAM
payload after the durability barrier. After visible Stop/Start and later a full
Electron dev relaunch, identical fresh chats restored three blocks / 192 tokens
from SSD.

Representative persisted rows:

- row 557: exact `DISKONLY-BASE-DONE`, one real automatic tool, 192
  `block-disk+tq-native` cached tokens
- row 560: same-chat exact `SECRET=cobalt-7319`, no tool, non-empty reasoning and
  visible content
- row 563: post-process-restart exact `DISKONLY-BASE-DONE`, no tool, 192
  `block-disk+tq-native` cached tokens
- row 566: final current-process two-tool continuation, exact final marker, 576
  aggregate `block-disk+tq-native` cached tokens

Final health after the current-process row reports:

- three scheduler hits, zero misses, 576 tokens saved
- three usable blocks / 192-token effective index capacity
- utilization `0.0` after completion
- 14 durable files representing 753 cached tokens
- 21 actual SSD hits and 21 TQ-native hits
- zero resident KV bytes and zero resident byte budget

See `electron-assistant-rows.json`, `diskonly-m27-restart-disk-hit.png`,
`diskonly-m27-final-electron-multitool.png`, and final health JSON.

## Raw API streaming proof

### Chat Completions

`chat-sse-summary.json` records:

- 330 reasoning deltas
- 15 progressive content deltas
- exact content `CHAT-CONTENT-ONE\nCHAT-CONTENT-DONE`
- `finish_reason=stop`
- one `[DONE]` marker in the raw timed SSE artifact

The first reasoning byte arrived at 1784497406.221126. Content was emitted over
multiple timed events rather than appearing as one final batch.

### Responses

`responses-sse-summary.json` records:

- 512 reasoning-summary deltas
- 16 progressive output-text deltas
- exact content `RESPONSES-CONTENT-ONE\nRESPONSES-CONTENT-DONE`
- one `response.completed`, zero `response.failed`
- `response.output_text.done`, content-part done, and output-item done in the raw
  artifact

The first content delta was at 1784497462.738294 and the last at
1784497463.138246, demonstrating progressive final-answer emission after the
reasoning rail.

### API partial-prefix repeat

`api-cache-hit-sse-summary.json` records a long exact repeat with:

- 192 cached tokens out of 846 prompt tokens
- `cache_detail=block-disk+tq-native`
- 71 reasoning deltas and seven content deltas
- exact `API-CACHE-BASE-DONE`
- `finish_reason=stop`

## Focused and aggregate tests

- Python disk-only/cache truth: 13 passed, 117 deselected
- Panel cache/settings: 299 passed
- TypeScript: `tsc --noEmit` passed
- Aggregate cache architecture contract: `status=pass`, no failed or missing
  markers
  - API cache status orchestrator: pass
  - cache-family selection: 454 passed, 656 deselected
  - panel cache launch policy: 115 passed, 209 skipped by name filter

Artifacts: `focused-python-tests.txt`, `focused-panel-tests.txt`,
`typecheck.txt`, and `cache-architecture-contract.json`.

## Remaining gates

- Repeat disk-only Off/On and partial-prefix behavior on additional compatible
  full-KV families; this row proves MiniMax-M2.7 only.
- Hybrid SSM/GDN families still require their separate attention-KV q4 plus
  native companion async-rederive proof.
- Native M3 MSA, DSV4 composite, openPangu typed prompt L2, and mixed-SWA
  families must stay on their architecture-specific rows; this generic result
  must not be generalized to them.
- Signed/notarized app repetition and the retained cross-family release matrix
  remain open for a later release cutoff.
