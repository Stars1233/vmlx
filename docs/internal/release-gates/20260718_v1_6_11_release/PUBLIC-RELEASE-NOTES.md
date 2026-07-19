# vMLX 1.6.11

Streaming, typed-cache, and packaged-runtime checkpoint.

## Highlights

- Streams post-reasoning and post-tool visible answers progressively across Responses, Chat Completions, and Electron instead of withholding a second-pass answer until completion.
- Hardens multi-turn tool continuation, terminal events, cancellation/recovery, and reasoning/content separation.
- Extends architecture-typed prefix, paged, and block-disk L2 reuse for hybrid and mixed-attention families while keeping native exceptions explicit.
- Improves model-derived cache/parser/reasoning settings parity, gateway port/LAN truth, model switching, missing-path handling, and narrow-window UI behavior.
- Adds current media/cache routing work for supported Qwen, Bonsai, Gemma, Step, and MiniMax-M3 families. MiniMax-M2.7 remains text-only.

TurboQuant is a KV-cache storage codec. It is distinct from JANG affine model weights, JANGTQ/MXTQ Hadamard-codebook model weights, and base MLX MXFP model weights.

## Verification

- Source commit: `95b2caa956c592a9caa706f2a790dcd5664721b7`; tag: `v1.6.11`.
- Full Python suite: 6,100 passed, 96 skipped, 92 deselected.
- Full panel suite: 2,311 passed, 3 skipped; TypeScript typecheck and the production Electron build passed.
- Both DMGs passed bundled-runtime verification, Developer ID signing, Apple notarization, stapling, Gatekeeper assessment, final DMG verification, and isolated installed-app smoke.
- The Sequoia installed app used its real Electron Start button and bundled engine to load Gemma 4 12B, then completed three UI turns including exactly one real `file_info` tool, post-tool continuation, multi-turn recall, paged plus mixed-SWA cache reuse, and raw Responses/Chat SSE.
- The Tahoe installed app independently loaded the same model through its bundled engine and completed UI and Responses streaming.

## macOS builds

- Sequoia/Sonoma-compatible build (`macosx_14_0_arm64`; macOS 14.5+): `vMLX-1.6.11-sequoia-arm64.dmg`
- Tahoe-native build (`macosx_26_0_arm64`): `vMLX-1.6.11-tahoe-arm64.dmg`

## Checksums

- Sequoia SHA-256: `c1a8dcd17563a772b83e64bfb443aabedb46111a1e0e500ab69c4fc49143cb2b`
- Tahoe SHA-256: `26b0b30a80e3576d2a2f967d9882966fde9637ed85f5fee2f7ec95a265e74308`

## Known follow-up

This is a tested release checkpoint, not closure of every retained model-family stress row. Larger/longer media, Omni audio, gateway soak, and remaining family-specific PARTIAL rows continue after this release.
