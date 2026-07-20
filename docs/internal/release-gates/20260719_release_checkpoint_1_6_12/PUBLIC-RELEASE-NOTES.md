# vMLX 1.6.12

Cache-hierarchy, streaming-recovery, and gateway-lifecycle checkpoint.

## Highlights

- Makes block-disk L2 a usable prefix tier even when paged RAM is disabled,
  while preserving the RAM-first then disk-refault hierarchy when both tiers
  are enabled. Partial-prefix restore, eviction, immediate-stop durability,
  and restart refault are covered by the current scoped evidence rows.
- Hardens q4 TurboQuant storage for eligible KV components and hybrid
  attention paths without conflating it with model-weight formats. JANG
  affine, JANGTQ/MXTQ Hadamard-codebook, and base MLX MXFP remain distinct;
  DSV4, openPangu, MiniMax-M3, Gemma mixed-SWA, and other typed/native cache
  paths retain their architecture-specific policies.
- Fixes abandoned non-stream gateway ownership across Chat Completions,
  Anthropic Messages, and Ollama so a disconnected client cancels the upstream
  request before headers and an immediate follow-up can recover normally.
- Tightens progressive reasoning/content/tool streaming and terminal/usage
  ordering across current Qwen, HY3, Step, MiniMax, Anthropic, Ollama, and
  Responses routes, including no-tool prompts and rejected/incomplete native
  tool markup.
- Improves model-derived UI truth for quant labels, parser/reasoning/cache
  settings, single-model swaps, session PID/sleep lifecycle, eager Start
  materialization, port/LAN rollback, and missing-model repoint/removal.
- Extends current media ownership and cache-key coverage for Step and Nemotron
  Omni image/video/audio paths while preserving MiniMax-M2.7 as text-only.

## Verification

- Runtime source checkpoint: `6de1096eca0ea2d5516ad64d6e79da98f3ae20a2`;
  tag: `v1.6.12`.
- Full Python suite: `6,186 passed, 185 skipped`.
- Full panel suite: `2,332 passed, 3 skipped`; TypeScript typecheck passed.
- Both Sequoia and Tahoe DMGs passed bundled-runtime verification, Developer ID
  signing, fresh Apple notarization, stapling, Gatekeeper verification, and
  exact installed-app smoke.
- The Sequoia app completed a three-turn UI reasoning/tool/recall loop, raw
  Responses and Chat streaming, and a live
  `paged+mixed_swa+tq-native` cache hit. Tahoe independently completed a UI
  reasoning/content turn and a raw Responses stream with separate reasoning
  and content deltas, terminal completion, and usage.

## macOS builds

- Sequoia/Sonoma-compatible build (`macosx_14_0_arm64`; macOS 14.5+):
  `vMLX-1.6.12-sequoia-arm64.dmg`
- Tahoe-native build (`macosx_26_0_arm64`):
  `vMLX-1.6.12-tahoe-arm64.dmg`

## Checksums

- Sequoia SHA-256:
  `704d87edf168a73d4ca2d94e8cb6190ca593ada71bca181bf369c84ea13ae421`
- Tahoe SHA-256:
  `81b9205a722282cc1eec75713c18dec3efc34ed76e3bcaf6587147e0ce372c49`

## Known follow-up

This is a tested release checkpoint, not closure of every retained family or
stress row. Broader signed-app model repetition, longer stochastic and media
soaks, remaining parser-family tool rows, injected mid-stream failure, and the
documented Laguna/Bonsai/Step/DSV4/M3/openPangu/LFM/Nemotron follow-ups remain
on the post-release worklist.
