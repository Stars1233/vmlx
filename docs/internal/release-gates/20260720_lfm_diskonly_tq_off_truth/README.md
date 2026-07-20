# LFM Paged-Off / TQ-Off SSD restore truth

Date: 2026-07-20

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Base HEAD before this scoped repair: `e8755c6b2e4a8abe5cfb38b0a284fb61129a0634`

Public checkpoint boundary: v1.6.14 is already public at tagged source
`e1776a485`. This post-release cache repair is newer than that tag and is not
part of the public v1.6.14 binaries.

## Verdict

- LFM block-disk reuse with Paged Cache **Off**, Block Disk L2 **On**, and
  stored KV quantization explicitly **None**: **PASS-LIVE scoped on current
  post-release source**.
- Partial-prefix reuse from SSD without a resident paged-KV payload:
  **PASS-LIVE scoped**. The live process restored nine 64-token blocks, saved
  576 of 716 prompt tokens, and retained zero resident L1 KV bytes.
- Hybrid SSM companion alignment and process-restart restore:
  **PASS-LIVE scoped**. The SSM snapshot was retargeted from the full 712-token
  prompt boundary to the actual 576-token stored KV boundary, persisted, and
  restored after process replacement.
- Explicit TQ Off codec truth: **PASS-LIVE scoped**. The restart proof records
  `tq_native_enabled=false`, `tq_native_writes=0`, and `tq_native_hits=0`.
- Reasoning/content output emission: **PASS-LIVE scoped** through raw Chat,
  raw Responses, and the real Electron dev app. Reasoning and visible content
  were separate; visible content was non-empty and exact.
- Overall LFM family and campaign: **PARTIAL**. Required-tool reliability,
  full protocol/cancellation breadth under this exact setting, larger-context
  eviction/fault injection, signed-app repetition, and other model families
  remain open.

## Bundle-grounded identity

The exact live bundle was:

`/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`

The preserved bundle summary proves:

- `model_type=lfm2_moe`;
- base MLX `MXFP4`, not affine JANG and not JANGTQ/MXTQ;
- 24 language layers: attention at indices `2, 6, 10, 14, 18, 21`, with 18
  SSM/convolution companion layers;
- no `jang_config.json`;
- model-owned generation configuration derived from the bundle.

See `bundle-config-summary.json` and `bundle-generation-config.json`.

## Root cause

Two shared cache/storage defects combined.

1. Block-disk rows are keyed by token-prefix content, not by storage codec. A
   prior Auto/q4 run could therefore leave q4-native pages under the same
   hashes used by a later explicit-None run. Lookup rejected the first
   incompatible q4 page, but the cold completion queued ordinary KV writes for
   the full chain. The background writer deduplicated later hashes against the
   old q4 rows, leaving a mixed q4/plain chain. Later reconstruction failed
   with missing tensor data, while optimistic lookup telemetry could still
   over-credit the request.
2. The 716-token prompt had 712 cacheable prompt tokens, but the configured
   ten-block pool reserves block zero and could store only nine 64-token pages,
   or 576 tokens. The deferred hybrid SSM rederive remained keyed to 712. The
   async engine loop also considered itself idle because scheduler
   `has_requests()` ignored the rederive queue, so the usable 576-token SSM
   checkpoint was never produced.

## Source repair

- `vmlx_engine/block_disk_store.py`
  - every ordinary KV write in explicit TQ-Off mode now inspects the matching
    hash and evicts an incompatible native-TQ row before the asynchronous
    writer can deduplicate it;
  - compatible ordinary rows remain deduplicated.
- `vmlx_engine/scheduler.py`
  - unusable or failed paged hits release refs and roll back optimistic hit
    credit;
  - accepted shortened prefixes keep only the tokens actually consumed;
  - completed cache hits finalize their accounting;
  - queued SSM rederive is retargeted to the block table's actual stored
    boundary;
  - the scheduler remains active while a hybrid SSM rederive task is queued.
- `vmlx_engine/mllm_scheduler.py`
  - the multimodal scheduler likewise remains active for its batch generator's
    queued hybrid SSM rederive.
- `tests/test_tq_paged_block_cache.py`
  - reproduces a three-page q4 chain followed by explicit-None ordinary writes
    without pre-reading every page, and requires all pages to restore as plain
    KV.
- `tests/test_hybrid_ssm_companion_regressions.py`
  - pins rejected-hit rollback/ref release, accepted-prefix credit,
    text/MLLM idle-rederive wakeup, and 706-to-576 boundary retargeting.

`source-and-test.diff` preserves the exact pre-commit source/test diff.

## Focused validation

The expanded cache/scheduler selection passed **326/326**, with no failures,
errors, or skips. The JUnit header records `tests="326" failures="0"
errors="0" skipped="0"` in `focused-cache-tests.xml`.

The selection covers block-disk serialization/defaults, N-1 prefix handling,
hybrid cache clone/state behavior, SSM companion regressions, text and MLLM
schedulers, paged cache and hash collisions, byte budgets, prefix caches,
native-TQ disk storage, and TQ prefix clone behavior.

This is a focused cache/scheduler gate, not a new full Python/panel suite run.

## Live cold and warm Chat proof

Prompt marker: `LFMSSD-9381`

Required visible answer: `LFM-SSD-NONE-RETARGET-DONE`

The first unique raw Chat request was a true cold miss:

- prompt tokens: 716;
- separate progressive reasoning and visible content;
- exact visible answer;
- clean stop/usage/`[DONE]` terminal;
- log records a 712-token queued SSM rederive, a partial 576-token KV store,
  retargeting to 576, and completion of the 576-token SSM checkpoint.

The exact warm raw Chat request then reported:

- `cached_tokens=576`;
- `cache_detail=block-disk+ssm`;
- nine SSD blocks;
- no TQ encode/decode activity;
- exact visible answer and clean terminal ordering.

The byte-preserved captures are `raw-chat-cold.sse(.gz)` and
`raw-chat-warm.sse(.gz)`.

## Real Electron proof

The real Electron app on CDP 9335 used the persisted session settings:

- Prefix Cache On;
- Paged Cache Off;
- ten configured blocks, nine usable blocks, 576-token capacity;
- Block Disk Cache (L2) On;
- Stored Cache Quantization `None`;
- `--no-paged-cache` in the actual engine argv.

The fresh UI chat progressively painted a separate reasoning rail and exact
visible content. SQLite row 130 contains:

- `content="LFM-SSD-NONE-RETARGET-DONE"`;
- separate reasoning;
- `cachedTokens=576`;
- `cacheDetail="block-disk+ssm"`;
- no tool call and no warning.

See `electron-warm-trace.json`, `electron-chat-db.json`, the first-paint/final
screenshots, `electron-settings-paged-off-l2-on.png`,
`electron-settings-tq-none.png`, and `engine-argv.txt`.

## Process-restart proof

The app performed a real Electron Stop/Start before the final request.
Pre-request health showed:

- `model_loaded=true`, `last_request_time=null`;
- zero L1 resident/indexed KV tokens and zero in-memory SSM entries;
- 61 ordinary L2 blocks containing 3,538 tokens;
- 19 SSM disk entries containing 3,562 tokens;
- native TQ disabled with zero q4-native activity.

The first request in the replacement process was a real Electron Regenerate
through Responses. It exact-finaled and reported 576
`block-disk+ssm` cached tokens with 0.15-second TTFT. Post-request health shows:

- scheduler `tokens_saved=576`;
- nine disk promotions;
- `l1_resident_bytes=0`;
- one SSM disk hit, zero misses;
- no TQ-native write or hit.

The UI log records `SSM disk HIT: N=576`, nine reconstructed L2 blocks, and a
576-token hybrid cache hit. See `restart-pre-request-health.json`,
`restart-post-ui-health.json`, `electron-restart-trace.json`,
`electron-restart-log-text.txt`, and the restart screenshots.

## Raw Responses proof

The independent raw `/v1/responses` stream after restart emitted separate
reasoning-summary and output-text deltas, exact visible content, one reasoning
terminal, one output-text terminal, one output-item terminal, and exactly one
`response.completed`. Final usage was 716 input / 238 output / 954 total with
576 cached tokens and `block-disk+ssm` detail. No error event occurred.

See `raw-responses-warm.sse(.gz)`.

## Stop state

Work stopped through the real Electron UI after the proof:

- session `89a68557-452b-4850-b189-ba9d57c4d6c5` is `status=stopped` with
  `pid=null`;
- port 8016 has no listener;
- `electron-handoff-stopped.png` records the visible stopped state;
- `final-stopped-session.json` records the final SQLite row.

The persisted session intentionally still has the experiment settings:
Paged Off, ten configured blocks, stored KV quantization None, and L2 On. The
next agent must not assume those are the desired steady-state settings.

## Remaining boundaries

- LFM required-tool generation is still a current live failure in
  `../20260720_lfm_native_reasoning_protocol/README.md`: the model emitted a
  malformed `file_info` path and no schema-valid call. The Responses endpoint
  now fails that request truthfully; this cache repair does not change it.
- Anthropic and Ollama were not rerun under this exact Paged-Off/TQ-Off source
  state.
- Tool-result continuation, cancellation/disconnect/failure cleanup, larger
  capacity/eviction, fault-injected disk writes, partial-prefix variations,
  model swaps, sleep/wake, and signed-app repetition remain open here.
- Cross-prompt stale-reasoning comparison was not part of this exact-cache row;
  identical reasoning on Regenerate of the identical prompt is expected and is
  not promoted as a distinct-prompt replay test.
- Strict superseding-suffix controls remained model-behavior PARTIAL and were
  not hidden by output rewriting or prompt coercion.
- No other family is cleared by this LFM-specific live evidence. The shared
  source regression tests reduce the common risk, but each relevant family
  still needs its own current-source live UI/API row.
- No package, signing, notarization, tag, or public release action was performed
  for this post-v1.6.14 change.

## Resume order

1. Read the successor handoff at
   `docs/internal/agent-notes/2026-07-20-lfm-diskonly-tq-off-stop-handoff.md`.
2. Re-read `AGENTS.md`, `.agents/STATUS.md`, `.agents/LOG.md`,
   `docs/internal/ISSUE-LEDGER.md`, and
   `../20260716_release_closeout/CURRENT-MATRIX.md` from the live checkout.
3. Confirm branch/HEAD/upstream and the stopped port before changing state.
4. Through the real UI, restore the LFM session to the intended steady-state
   cache policy before using it for unrelated work; verify UI, SQLite, launch
   preview, argv, and health rather than assuming the settings took effect.
5. Continue the highest-priority retained matrix row, not another duplicate
   LFM short smoke.
