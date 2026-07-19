# openPangu typed non-paged prompt-disk partial-prefix proof

Date: 2026-07-19  
Source head: `6e5653f56feea89434b5cb8b34f1b1558b69806e`  
Model: `jangq-ai/openPangu-2.0-Flash-JANG_3M`  
Overall status: **PARTIAL**

This gate keeps the two observed reasoning modes separate:

- **Thinking Off: VERIFIED-LIVE** for Electron, Responses, and Chat
  Completions after real Electron Stop/Start process replacement.
- **Auto reasoning: OPEN** for partial prompt-disk reuse. The model produced
  coherent, exact answers, but the same-chat restart follow-up did not reuse
  the durable 1,432-token base boundary.

The Off pass must not be used to close the Auto row.

## Artifact and architecture identity

The live bundle is JANG affine, not JANGTQ/MXTQ and not base MLX MXFP:

- `bundle-jang-summary.json`: `format=jang`, `profile=JANG_3M`, asymmetric
  `mx.quantize`, `hadamard_rotation=false`, actual average 3.83 bits.
- `op-chat-partial-health.json`: runtime codec
  `affine_quantized_matmul`; the JANG sidecar is present and the JANGTQ
  runtime is false.
- It is text-only. The bundle advertises MLA/MoE and no vision, audio, or
  video.

The live process used the architecture-owned cache policy:

- 46 `OpenPanguV2LayerCache` layers.
- composite schema `openpangu_v2_composite_v2` with MLA latent KV, DSA
  indexer state, rotating SWA window state, and path-dependent causal-conv
  state.
- generic TurboQuant KV, paged blocks, and block-disk L2 remained disabled.
- prompt-disk snapshots remained full precision and exact typed N-1.

`source-trace.txt` preserves the current source paths for the openPangu policy,
typed store, longest-prefix prompt-disk lookup, and replay code. The owning
code is in `vmlx_engine/cli.py`, `vmlx_engine/scheduler.py`,
`vmlx_engine/disk_cache.py`, and
`panel/src/shared/toolHistoryReplay.ts`.

Focused current-source checks passed 38/38 Python tests across the openPangu
runtime, disk-cache unit path, and N-1 disk-prefix contract, plus 13/13 panel
tests for tool-history replay and interleaved reasoning segments. Those source
checks do not override the live Auto miss. Outputs are in
`focused-python-tests.txt` and `focused-panel-tests.txt`.

## UI and launch parity

The real Electron dev app on CDP 9335 configured and launched this session:

- Prefix cache: On
- Paged cache: Off
- Block-disk L2: Off
- Prompt disk cache: On
- TurboQuant KV: Off / native composite policy
- Prompt disk directory:
  `/Users/eric/.cache/vmlx-engine/live-proof-openpangu-typed-partial-20260719`

`op-typed-nonpaged-ui.png` records the UI. `openpangu-argv.txt` records the
actual process with `--no-paged-cache --enable-disk-cache` and the same
directory. Health records zero RAM/paged/block-L2 tokens and non-zero prompt
disk tokens. Only the openPangu server remained after the Start swap.

## Electron Thinking-Off partial restore: PASS

The base turn (`rows 283/285`) produced exactly `OP-OFF-BASE-DONE` with no
reasoning rail and stored a 780-token prompt snapshot. Electron Stop/Start
replaced the process. The longer same-chat turn (`rows 286/288`) then:

- returned exactly `OP-OFF-PARTIAL-DONE VALUE-82372`;
- reported 840 prompt tokens and 779 disk-cached tokens;
- logged `Disk cache prefix hit: matched 780/840 prompt tokens`;
- reported `cache_detail=disk`, zero blocks, no reconstruction, and no
  dequantization;
- recomputed only the unmatched tail and completed normally.

Evidence: `openpangu-electron-rows.json`, `op-off-base-pass.png`,
`op-off-partial-pass.png`, `op-off-partial-logs.png`,
`op-off-base-health.json`, `op-off-restart-health.json`, and
`op-off-partial-health.json`.

## Raw Responses after Electron restart: PASS

Electron Stop/Start replaced PID 48611 with PID 49320. A raw streamed
Responses request replayed the base user/assistant history and appended a
longer fact-retrieval turn with thinking disabled. It:

- restored 779/838 input tokens from prompt disk;
- emitted 14 non-empty progressive `response.output_text.delta` events;
- returned exactly `OP-API-PARTIAL-DONE VALUE-82372`;
- emitted `response.output_text.done` and `response.completed`;
- reported `cache_detail=disk`, zero blocks, and no TQ/dequantization.

Evidence: `op-api-start-health.json`, `op-api-partial.sse`, and
`op-api-partial-health.json`.

## Raw Chat Completions after another Electron restart: PASS

Electron Stop/Start replaced PID 49320 with PID 49554. A raw streamed Chat
Completions request on a different longer follow-up:

- restored 779/840 prompt tokens from the same 780-token disk boundary;
- emitted progressive content chunks rather than one terminal batch;
- returned exactly `OP-CHAT-PARTIAL-DONE VALUE-82372`;
- emitted `finish_reason=stop` and `[DONE]`.

The subsequent 840-token store crossed the configured 10 GB limit and logged
one disk-entry eviction. The hit occurred before that eviction and retained
the intended prefix. Health continued to report prompt-disk tokens only;
there were no fabricated paged/block-L2 hits.

Evidence: `op-chat-restart-health.json`, `op-chat-partial.sse`,
`op-chat-partial-health.json`, and `op-chat-partial-logs.png`.

## Auto reasoning partial restore: OPEN

The Auto base (`rows 277/279`) produced a separate 786-character reasoning
rail and exact visible `OP-TYPED-BASE-DONE`, then wrote a durable 1,432-token
typed prompt snapshot. After Electron process replacement, the longer
same-chat turn (`rows 280/282`) remained coherent and returned exact
`OP-TYPED-PARTIAL-DONE STATE-49059`, but it reported no cached tokens.

`op-partial-health.json` confirms zero cache hits/tokens and one prompt-disk
miss after restart, followed by a separate 1,495-token store. This is a real
cache-reuse failure, not a formatting failure and not a model-artifact claim.

Current source contains Responses history replay for persisted reasoning, but
the observed prompt still did not share an admissible stored prefix with the
base prompt. The precise token-boundary cause remains under investigation:
reasoning-item serialization, the template's injected/open `<think>` boundary,
and the N-1 prompt-only snapshot must be compared byte/token-for-token. Do not
fix this with prompt coercion, hidden output caps, or by discarding private
reasoning.

Acceptance test for closing the Auto row:

1. Start from a fresh dedicated directory in the real Electron app.
2. Run an Auto-reasoning base turn with separate, non-empty reasoning and
   visible content.
3. Electron Stop/Start the model process.
4. Run a longer same-chat Auto follow-up.
5. Prove a non-zero typed prompt-disk partial match, only unmatched-tail
   prefill, separate progressive reasoning/content deltas, a non-empty
   terminal answer, and no stale reasoning replay.
6. Repeat with raw Responses and Chat Completions.

Evidence: `op-typed-base-pass.png`, `op-typed-partial-result.png`,
`op-base-health.json`, `op-restart-health.json`, `op-partial-health.json`,
`openpangu-electron-rows.json`, and `openpangu-chat-overrides.json`.

## Non-claims / remaining work

- This does not prove paged or block-disk cache for openPangu; those paths are
  intentionally incompatible with its path-dependent composite state.
- The observed size-limit eviction is not yet a complete eviction/refault
  proof because the evicted key was not subsequently requested and restored.
- The generic paged-On block-aligned partial/eviction/refault row still needs
  a compatible representative.
- openPangu Auto reasoning cache reuse remains OPEN.
- Campaign and release status remain PARTIAL.
