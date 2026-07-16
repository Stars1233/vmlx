# vMLX 1.6.11 last-24-hour reconciliation — 2026-07-16

Status: `PARTIAL_NO_RELEASE`.

This is a current-source reconciliation, not a replacement for the detailed
artifacts. At commit `7bb34fa0d`, the branch is 45 commits ahead and zero
behind `origin/main`, and zero ahead/behind the pushed closeout branch. There
were 47 commits in the preceding 24 hours. The public release remains 1.6.10.

## Workstream reconciliation

| Workstream | Relevant commits | Current proof level | Still red |
|---|---|---|---|
| Typed model caches | `c5d713169`, `2bca8fde6`, `5cb6de1dc`, `cee23cec7`, `f85bafb64`, `f5a75d15a`, `9deb23483`, `b62955450`, `335c2f812`, `2f5b7786d` | ZAYA, MiniMax-M3, and openPangu have scoped source plus historical/current Electron rows recorded in the matrix | Current-release ZAYA eviction/stream row; M3/Pangu long/eviction/protocol rows |
| Hybrid SSM/GDN and TQ storage | `51bd720fb`, `d524b631b`, `63512c555`, `103cb9e92`, `4b96bbe6c`, `7bb34fa0d` | Bonsai and Qwen 35B have native-TQ attention-block plus native SSM-disk restart evidence. Qwen row 2160 restored 152/153 tokens after Electron restart, ran one real tool, returned exact final text, and left 0 resident bytes | Forced eviction/L2 promotion/true-miss fallback per family; duplicate companion-payload audit; truthful selective-TQ UI label; Qwen 27 MTP cache rows |
| MTP | `b5a47f62f`, `b0b21ed12` | HY3 depth-1 acceptance/speed is scoped; non-MTP Qwen 35B telemetry is now correctly inactive despite a nested architecture hint | HY3 depth-3/cache interaction; Qwen 27 MTP depth-1/depth-3 cache and loop rows |
| Agent streaming/parsers | `26fb7cd54`, `292f99b28`, `ea89ff55d`, `7aae50003`, `969aff76a`, `b111197f8`, `048d3c16a`, `88857fe52`, `9b14fc66c`, `0cc2ee8f1`, `ea40a0a3e`, `50dedf1db`, `a593f0630`, `54a08fce8`, `8cfc9f269`, `7b45676ce` | Multiple named Electron exact-tool rows exist; MiniMax speculative zero-tool finalization and Laguna parser migration have focused tests and scoped live evidence | Full Chat/Responses/Anthropic/Ollama streaming assembly, disconnect/stop, long output, interleaved reasoning, and every remaining parser family |
| Architecture/runtime | `289f45900`, `757c6e30e`, `3ddcf1349` | Step attention, DSV4 reasoning separation, and Laguna dtype/cache reconstruction have scoped source evidence | DSV4 quality/performance/eviction, Laguna unsolicited tool and latency, remaining model rows |
| Settings/lifecycle | `6d8c2ac30`, `626f8524d` | Typed settings are committed before restart; scoped LAN and single-model evidence exists | Min-P zero persistence, Auto/None and selective-TQ labels, cache-limit enforcement, port conflict, LAN/gateway protocol parity |

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
- The repair passes 595/595 engine-audit/byte-budget tests and 177/177
  paged/disk/TQ/hybrid cache tests.

Live Electron evidence:

- PID 58213, row 2160: 152/153 `paged+ssm+disk`, one schema-valid
  `file_info`, one real tool result, exact final text.
- Cache Management and health: 152 indexed tokens, 0 L1 resident bytes,
  seven native-TQ block hits, two native SSM companion hits.
- Screenshots: `qwen35-resident-accounting-pass.png` and
  `qwen35-cache-management-postfix.png`.

This is still `PARTIAL`: forced bounded eviction, disk promotion after that
eviction, true-miss fallback, duplicate companion-state storage, and the
selective-TQ UI label remain open.

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
