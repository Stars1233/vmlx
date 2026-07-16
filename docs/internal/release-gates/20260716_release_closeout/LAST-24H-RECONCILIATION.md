# vMLX 1.6.11 last-24-hour reconciliation — 2026-07-16

Status: `PARTIAL_NO_RELEASE`.

This is a current-source reconciliation, not a replacement for the detailed
artifacts. At source commit `af7815f1a`, the branch is 51 commits ahead and zero
behind `origin/main`, and zero ahead/behind the pushed closeout branch. There
were 53 commits in the preceding 24 hours. The public release remains 1.6.10.

## Workstream reconciliation

| Workstream | Relevant commits | Current proof level | Still red |
|---|---|---|---|
| Typed model caches | `c5d713169`, `2bca8fde6`, `5cb6de1dc`, `cee23cec7`, `f85bafb64`, `f5a75d15a`, `9deb23483`, `b62955450`, `335c2f812`, `2f5b7786d` | ZAYA, MiniMax-M3, and openPangu have scoped source plus historical/current Electron rows recorded in the matrix | Current-release ZAYA eviction/stream row; M3/Pangu long/eviction/protocol rows |
| Hybrid SSM/GDN and TQ storage | `51bd720fb`, `d524b631b`, `63512c555`, `103cb9e92`, `4b96bbe6c`, `7bb34fa0d`, `df945f065`, `133d8c8e9`, `7cb89185c`, `af7815f1a` | Bonsai and Qwen 35B have native-TQ attention-block plus native SSM-disk restart evidence. Qwen 35B now has cold, RAM, restart, forced-eviction, post-eviction L2 reload, and safe KV-only-miss/full-prefill Electron rows; v8 block files contain no duplicate companion state. The generic fetched-paged-hit ownership leak found with MiniMax M2.7 is repaired and live pressure-tested. | Same tier matrix per remaining family; Qwen 27 MTP cache rows; Bonsai forced eviction; Qwen 35B strict long-format reliability only |
| MTP | `b5a47f62f`, `b0b21ed12` | HY3 depth-1 acceptance/speed is scoped; non-MTP Qwen 35B telemetry is now correctly inactive despite a nested architecture hint | HY3 depth-3/cache interaction; Qwen 27 MTP depth-1/depth-3 cache and loop rows |
| Agent streaming/parsers | `26fb7cd54`, `292f99b28`, `ea89ff55d`, `7aae50003`, `969aff76a`, `b111197f8`, `048d3c16a`, `88857fe52`, `9b14fc66c`, `0cc2ee8f1`, `ea40a0a3e`, `50dedf1db`, `a593f0630`, `54a08fce8`, `8cfc9f269`, `7b45676ce` | Multiple named Electron exact-tool rows exist; MiniMax speculative zero-tool finalization and Laguna parser migration have focused tests and scoped live evidence | Full Chat/Responses/Anthropic/Ollama streaming assembly, disconnect/stop, long output, interleaved reasoning, and every remaining parser family |
| Architecture/runtime | `289f45900`, `757c6e30e`, `3ddcf1349` | Step attention, DSV4 reasoning separation, and Laguna dtype/cache reconstruction have scoped source evidence | DSV4 quality/performance/eviction, Laguna unsolicited tool and latency, remaining model rows |
| Settings/lifecycle | `6d8c2ac30`, `626f8524d`, `df945f065` | Typed settings are committed before restart; scoped LAN/single-model evidence exists; Qwen four-block capacity was applied through Save & Restart and selective TQ is labeled truthfully in Cache/Perf | Min-P zero persistence, wider Auto/None parity, port conflict, LAN/gateway protocol parity |

## Current cache acceptance contract

For every eligible model, prefix reuse is three-tiered:

1. Use a valid resident L1/paged block.
2. If L1 is absent, promote a matching L2 disk record.
3. If neither tier is complete and valid, safely rederive or full-prefill.

For hybrid or mixed-cache models, a hit is valid only when every required
component can reach the same boundary. Attention KV may use TQ only when its
actual cache class is codec-compatible. SSM, GDN, rotating SWA, CCA, DSV4
composite, MiniMax-M3 MSA, and openPangu composite state stay on their owning
native formats unless a typed codec is explicitly implemented and proven.

Required live evidence per family is: cold miss, same-chat resident hit,
process-restart L2 promotion, forced eviction then correct reload, a true miss
that full-prefills safely, resident/indexed byte accounting, encode/decode and
companion restore/rederive counters, two user turns, one real tool result, and
a complete reasoning/content stream.

## Current Qwen 35B cache repair

Source trace:

- The real layer graph is 10 attention KV layers plus 30 GDN/SSM companion
  layers. It is not an MTP artifact.
- `7bb34fa0d` makes disk-reconstruction cleanup release paged resident-byte
  attribution and prevents `keep_resident` from leaking when a block is reused.
- `df945f065`, `133d8c8e9`, and `7cb89185c` externalize generic hybrid
  cumulative state to its typed companion store, repair the separate NumPy
  disk-writer branch found by live inspection, and invalidate malformed v7
  records through the v8 namespace.
- Current coverage passes 784/784 Python hybrid/cache/scheduler tests,
  278/278 panel settings tests, and panel typecheck.

Live Electron evidence:

- PID 58213, row 2160: 152/153 `paged+ssm+disk`, one schema-valid
  `file_info`, one real tool result, exact final text.
- Cache Management and health: 152 indexed tokens, 0 L1 resident bytes,
  seven native-TQ block hits, two native SSM companion hits.
- Corrected v8 block inspection: every file contains exactly 10
  `turboquant_kv` plus 30 `skip` entries; no terminal block contains cumulative
  companion state. Malformed v7 terminal files were about 64 MB; corrected v8
  terminal files are about 30 KB/295 KB.
- Rows 2169/2172/2175 prove cold, same-process, and process-restart tiers.
  UI-applied four-block rows 2178/2181 each restored 154 tokens from disk with
  one real tool and exact final text. The visible Cache drawer recorded nine
  L1 evictions and a safe full-prefill when 192 KV-only tokens had no matching
  SSM companion.
- The UI restored 1,000 blocks; PID 61919 launched with that argv and row 2184
  repeated the exact `paged+ssm+disk` hit.
- Screenshots: `qwen35-resident-accounting-pass.png`,
  `qwen35-cache-management-postfix.png`, `qwen35-v8-ram-hit.png`,
  `qwen35-v8-disk-hit.png`, `qwen35-v8-selective-tq-cache.png`,
  `qwen35-evict-settings-4blocks.png`,
  `qwen35-v8-forced-eviction-pass.png`, and
  `qwen35-v8-restored-normal.png`.

The Qwen 35B cache tier is `PASS-LIVE`. The artifact remains `PARTIAL` only for
its retained long strict-format/reliability miss; that is not hidden by the
cache result.

## Current MiniMax M2.7 cache and agent-loop repair

Source trace:

- The real bundle reports `model_type=minimax_m2`, 62 ordinary attention KV
  layers, no hybrid companion state, and no MTP. The registry selects the
  `minimax` tool parser and `minimax_m2` reasoning parser.
- Auto resolves to storage-only TQ8 for all 62 compatible KV layers; explicit
  None disables TQ while leaving prefix, paged, and block-disk L2 enabled.
- `af7815f1a` registers both chain-hash and prefix-index fetched block tables in
  `_request_tables`. Completion cleanup can now release the request refs instead
  of leaving disk-promoted blocks permanently pinned after an agent iteration.
- 90/90 focused paged-cache, byte-budget, TQ block, and hybrid-prefix tests pass.

Live Electron evidence:

- Rows 2187 and 2190 are one same-chat two-turn tool loop. Both execute one real
  `file_info` and finish with exact visible markers; row 2190 restores 173 prompt
  tokens from resident `paged+tq-native` state.
- PID 63682 row 2193 restores 173/177 tokens from `paged+disk+tq-native` after a
  visible Stop/Load Model cycle. A real full block contains 62
  `turboquant_kv` layer records, 8-bit K/V metadata, and is indexed as
  `dtype=turboquant_kv`.
- Explicit None produced PID 64194 with `--kv-cache-quantization none`, zero TQ
  telemetry, and raw `dtype=kv` files in a separate namespace. PID 64579 row
  2199 restored 161/165 tokens as plain `paged+disk` and completed exactly.
- Before the ownership fix, the UI-applied four-block ceiling left all three
  usable blocks pinned and logged `Out of cache blocks`. After the fix, PID
  65838 rows 2208 and 2211 both completed real tool loops; health returned to
  `allocated_blocks=1`, `free_blocks=3`, `shared_blocks=0`, and live evictions
  advanced from 3 to 9 while L2 hits/writes remained active.
- The UI restored Auto and 1,000 blocks. PID 66306 row 2214 repeated the exact
  173/177 disk hit; post-request health showed `free_blocks=999` and no shared
  refs. Screenshots and the full evidence ledger are in `MM27-CACHE-AUDIT.md`.

MiniMax M2.7 is `PASS-LIVE` current-source for cache tiers, Auto/None parity,
two-turn tools, restart, forced eviction, long visible output, and direct
Responses streaming. Electron row 2217 produced a coherent 582-token separated
reasoning/content answer with its exact terminal marker. The 1,024-token direct
stream emitted 711 reasoning deltas and 48 content deltas, matched the assembled
text in `response.output_text.done`, retained the exact marker, and ended with
`response.completed(status=completed)`. A controlled 512-token budget ended
`status=incomplete` after reasoning consumed the cap, which is the correct
harness-visible result rather than a false completed response.

## Explicit additions and exclusions

- Gemma 4 rotating SWA is now a named current-source row: native rotating SWA
  plus TQ-compatible full-attention KV, prefix/L1/L2 tiering, eviction, and
  safe fallback must be proven together.
- Only bundles whose actual model name says `MTP` receive MTP gates.
- Official JANGQ and dealignai quantized artifacts are trusted inputs; runtime
  incoherence is investigated in vMLX dispatch, layer utilization, cache,
  parser, streaming, template, sampler, or settings ownership.
- Mistral MXFP4 is excluded from this campaign by explicit user instruction.
- No package build, signing, notarization, tag, feed, PyPI, or GitHub release
  begins while any release row remains red.
