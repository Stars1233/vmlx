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

- Source commit: `PENDING`; tag: `v1.6.12`.
- Full Python suite: `PENDING`.
- Full panel suite: `2,332 passed, 3 skipped`; TypeScript typecheck passed.
- Both Sequoia and Tahoe DMGs: `PENDING` Developer ID signing, Apple
  notarization, stapling, Gatekeeper verification, and installed-app smoke.

## macOS builds

- Sequoia/Sonoma-compatible build (`macosx_14_0_arm64`; macOS 14.5+):
  `vMLX-1.6.12-sequoia-arm64.dmg`
- Tahoe-native build (`macosx_26_0_arm64`):
  `vMLX-1.6.12-tahoe-arm64.dmg`

## Checksums

- Sequoia SHA-256: `PENDING`
- Tahoe SHA-256: `PENDING`

## Known follow-up

This is a tested release checkpoint, not closure of every retained family or
stress row. Broader signed-app model repetition, longer stochastic and media
soaks, remaining parser-family tool rows, injected mid-stream failure, and the
documented Laguna/Bonsai/Step/DSV4/M3/openPangu/LFM/Nemotron follow-ups remain
on the post-release worklist.
