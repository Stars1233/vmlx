# MiniMax M2.7 paged q4 L2 partial-block, eviction, restart, and stream proof

Status: `VERIFIED-LIVE` on source commit
`97a84fed5` (`fix(cache): report worker L2 reconstruction`).

This is a current-source Python/Electron proof on the real
`jangq-ai/MiniMax-M2.7-Small-JANGTQ` bundle. It is JANGTQ/MXTQ
(`weight_format=mxtq`, profile `JANGTQ2`), not affine JANG and not base MLX
MXFP. MiniMax M2.7 is text-only.

## Real Electron configuration

The settings were changed through the running Electron app on CDP 9335 and
applied with `Save & Restart`:

```text
Prefix Cache: On
Paged KV Cache: On
Block size: 64
Max cache blocks: 4
Block Disk Cache (L2): On
Legacy prompt disk cache: Off
Stored KV codec: Auto (effective q4 native TurboQuant storage)
L2 directory: /Users/eric/.cache/vmlx-engine/live-proof-m27-paged-l2-final-20260719
```

The directory did not exist before the run. PID 63575 launched with the
matching argv and health reported zero blocks, zero L1 tokens, zero disk
hits, and zero disk writes. This excludes contamination by older evidence.

## Partial terminal block and same-process refault

The cold Electron base produced exact visible `M27-L2F-BASE-DONE`, separate
reasoning, and no warning. Health and the SQLite block index showed three q4
native-TQ blocks for 178 tokens:

```text
64 tokens, 62 layers, turboquant_kv
64 tokens, 62 layers, turboquant_kv
50 tokens, 62 layers, turboquant_kv
```

The 50-token terminal record is the deliberate non-block-aligned acceptance
case. A same-chat follow-up restored all 178 tokens, exact-recalled
`LFACT-11=Q-84712`, and stored a longer 192-token bounded prefix under the
four-block ceiling. Health recorded an L1 eviction. A fresh Electron chat
then reissued the original base and restored the older 178-token boundary as
`paged+disk+tq-native`, exact-finaling again. This is a real post-pressure L2
refault, not a RAM-resident mirror: frugal mode reported zero L1 resident
bytes while block-disk native-TQ hits increased.

## Process-restart disk-only restore

Electron `Save & Restart` replaced PID 63575 with PID 64404. Before any new
request, health proved that process memory was empty while disk persisted:

```text
l1_indexed_tokens=0
ram_tokens_cached=0
l1_resident_bytes=0
blocks_on_disk=4
l2_block_tokens_on_disk=242
```

The identical fresh-chat base restored 178 tokens from the 64+64+50 disk
chain with three native-TQ hits and exact visible output. No prompt recompute
or warning was substituted for the restore.

## Defect found: worker L2 reads were under-reported

The cache behavior itself was correct, but the live protocol run exposed a
truthfulness defect. Once the first disk restore rebuilt the in-process chain
index, later frugal requests still read all q4 payloads from disk, and block-
disk hits increased from 3 to 21. Their per-request usage nevertheless said
only `paged+tq-native`.

Root cause: `Scheduler.add_request()` sampled disk-hit counters around
`fetch_cache()`, but an indexed frugal chain has no resident payload. Its
actual L2 reads occur later in `BlockAwarePrefixCache.reconstruct_cache()` on
the worker. The earlier sample could not observe those reads.

The repair records successful worker-side disk block reconstruction, promotes
that fact to `_paged_disk_hit`, and updates request/cache-execution detail
before streaming usage is emitted. It does not infer disk from configuration;
only blocks actually read during successful reconstruction are counted.

Focused current-source validation:

```text
targeted regression: 2 passed
batching + TQ paged + paged unit + byte-budget: 114 passed, 2 deselected
```

## Patched live UI and API proof

Electron restarted the patched source as PID 65685. Pre-request health again
showed zero L1 tokens and the same four persisted blocks. The first UI request
restored 178 tokens as `paged+disk+tq-native`, exact-finaled, and logged:

```text
worker reconstructed 3 paged block(s) from L2
```

Two later same-process raw requests exercised the exact regression case:

- Responses: 316 `response.reasoning_summary_text.delta` events, 10
  `response.output_text.delta` events, exact visible
  `M27-L2F-BASE-DONE`, then text-done and response-completed. Completed usage
  reported 178 `paged+disk+tq-native` cached tokens.
- Chat Completions: 508 non-empty reasoning deltas and 10 non-empty content
  deltas, exact visible marker, then `finish_reason=stop`, exactly one
  choices-empty usage chunk, and `[DONE]`. Ordinary chunks kept `usage=null`.
  Terminal usage reported 178 `paged+disk+tq-native` cached tokens.

Finally, the same restored Electron chat enabled tools, restored 192 tokens as
`paged+disk+tq-native`, executed exactly one real
`file_info({"path":"panel/package.json"})`, consumed the 5.2 KB result, and
exact-finaled `M27-L2F-TOOL-DONE SIZE=5.2 KB` with separate reasoning and no
warning.

## Evidence

- `m27-l2-final-settings.png`: visible fresh-directory UI policy.
- `m27-l2-health-fresh.json`: clean initial process.
- `m27-l2-block-index.json`: exact 64+64+50 partial chain and later block.
- `m27-l2-refault.png`: same-process post-eviction refault.
- `m27-l2-health-refault.json`: eviction and native-TQ L2 counters.
- `m27-l2-health-after-restart-before-request.json`: empty L1/persisted L2.
- `m27-l2-restart-restore.png` and `m27-l2-health-restart-restore.json`:
  disk-only process-restart restore.
- `m27-l2-fix-ui-restore.png`: patched current-source Electron restore.
- `m27-l2-fix-responses.sse` / `.trace`: raw timed Responses stream.
- `m27-l2-fix-chat.sse` / `.trace`: raw timed Chat stream.
- `m27-l2-fix-health-after-api.json`: post-protocol counters.
- `m27-l2-fix-tool.png`, `m27-l2-fix-health-after-tool.json`, and
  `m27-l2-ui-rows.json`: required-tool continuation and persisted rows.
- `m27-l2-fix-argv.txt`: exact final server command.

## Scope and remaining work

This closes the MiniMax M2.7 paged q4 child row for partial-terminal storage,
bounded-pool eviction, same-process L2 refault, process-restart disk restore,
truthful per-request detail, progressive Responses/Chat streams, and one
post-restore tool loop. It does not promote every model family or the release.
Paged-Off legacy/prompt-disk partial-prefix restore is a separate architecture
matrix row, as are hybrid SSM companions, mixed SWA, native DSV4/M3/openPangu
typed stores, media salt, gateway soak, full suites/build, and packaging.
