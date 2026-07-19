# vMLX 1.6.11 release checkpoint

Date: 2026-07-18

Verdict: **PASS for the v1.6.11 public checkpoint; broader retained model-family rows remain PARTIAL.**

## Source and taxonomy

- Packaged engine/source commit:
  `95b2caa956c592a9caa706f2a790dcd5664721b7`.
- Final annotated `v1.6.11` tag, `origin/main`, closeout branch, and release
  evidence head: `df244c4a858df3894fa3911b270d6d1b175966d6`.
- Clean JANG tools source: `9081c92476a63b912f4d2ce96146674971b5c83e`.
- JANG affine, JANGTQ/MXTQ, and base MLX MXFP are separate model-weight formats. JANGTQ/MXTQ is the Hadamard/codebook lane. TurboQuant is a separate KV-cache storage codec.

## Source gates

- Full Python suite: `6100 passed, 96 skipped, 92 deselected, 2 warnings`.
- Full panel suite: `73/73` files; `2311 passed, 3 skipped`.
- TypeScript typecheck: PASS.
- Electron production build: PASS.
- Bundled Python verification: PASS for both flavors; source and JANG hashes matched and 503 packaged native files were signed in each app.

## Signed and notarized artifacts

| Flavor | Apple submission | Final SHA-256 | Size | Result |
| --- | --- | --- | ---: | --- |
| Sequoia | `2d789d8b-9aa9-4d28-a221-55963a20ab99` | `c1a8dcd17563a772b83e64bfb443aabedb46111a1e0e500ab69c4fc49143cb2b` | 506,574,893 | Accepted, stapled, Gatekeeper accepted |
| Tahoe | `7f0ab5ab-dc56-4e5f-be37-71f99826e199` | `26b0b30a80e3576d2a2f967d9882966fde9637ed85f5fee2f7ec95a265e74308` | 522,656,073 | Accepted, stapled, Gatekeeper accepted |

The official `panel/scripts/verify-release-dmgs.sh` passed both final DMGs.

## Installed Sequoia package

- Isolated app: `/Applications/vMLX-1.6.11-Sequoia-Checkpoint.app`; CDP `9441`.
- CDP User-Agent: `vmlx/1.6.11`.
- Packaged log: `[Engine Manager] Found bundled Python with vmlx_engine 1.6.11 (from dist-info)`.
- The first direct-binary launch inherited Eric's home directory and the duplicate-MLX guard correctly rejected `/Users/eric/mlx`. The final proof relaunched from `/`, matching Finder's working-directory behavior; this failed harness attempt is retained in `sequoia-installed-app.log`.
- Real UI Start loaded `/Volumes/EricsLLMDrive/jangq-ai/gemma-4-12B-it-qat-JANG_4M`; `/health` reported healthy, loaded, `model_type=mllm`, `engine_type=batched`.
- UI T1 produced separate reasoning and non-empty content ending `REL1611-SEQ-T1-DONE`; metrics recorded `3584 paged+mixed_swa+disk cached`.
- UI T2 emitted exactly one `file_info(panel/package.json)`, displayed a live partial post-tool content delta, and completed `REL1611-SEQ-T2-DONE`; DB records one OAI call/result and no warning.
- UI T3 recalled `AURORA-611` and `5.2 KB` without another tool and completed `REL1611-SEQ-T3-DONE`.
- Responses SSE contains 300 reasoning-summary deltas, 50 output-text deltas, terminal text/item events, and `response.completed`.
- Chat Completions SSE contains separate `reasoning_content` and `content` deltas, a stop chunk, usage, and `[DONE]`.

## Installed Tahoe package

- Isolated app: `/Applications/vMLX-1.6.11-Tahoe-Checkpoint.app`; CDP `9442`.
- CDP User-Agent: `vmlx/1.6.11` and bundled engine `1.6.11`.
- Real UI Start independently loaded the same Gemma 4 12B bundle.
- UI T1 completed with non-empty content ending `REL1611-TAH-T1-DONE`; metrics recorded `3584 paged+mixed_swa+disk cached`.
- Responses SSE emitted progressive output-text deltas and completed `REL1611-TAH-API-DONE`.

## Evidence files

- `sequoia-installed-app.log`, `sequoia-installed-app-rootcwd.log`, and `tahoe-installed-app.log`
- `sequoia-cdp-version.json`, `tahoe-cdp-version.json`, and both health JSON files
- `sequoia-ui-*.png`, `tahoe-*.png`, final UI text, and per-turn DB JSON
- `sequoia-responses-stream.sse`, `sequoia-chat-stream.sse`, `tahoe-responses-stream.sse`, and curl timing traces
- A fresh rerun of the final DMG verifier, including stapled-ticket validation and Gatekeeper assessment, is preserved in this gate.

## Public surfaces

- Source release: `https://github.com/jjang-ai/vmlx/releases/tag/v1.6.11`.
- DMG release: `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.11`; target commit `38dafaf3d92ac0355316ea63770c5a17fbc512db`.
- GitHub reports the uploaded DMG digests as the same final SHA-256 values above. Blockmap SHA-256 values are `bc1b69159fef0acdaafe4129764d1a752c7c9a52d20da929c59b2a5af22234ab` (Sequoia) and `c32201b8403ba7254f43e4b3a84ea589aa0e5e7d17d0206832cee69222fbef81` (Tahoe).
- PyPI `vmlx==1.6.11`: wheel SHA-256 `4a5acd898a94b2714440f51a31f668fde1150b811a0e000136b71489b658345a`; sdist SHA-256 `9a2a65c807e01086b20ba4293212ec5dd4a14f5a4c889598b5acd854c47aa4e4`.
- `mlxstudio/main/latest.json` and `https://mlx.studio/update/latest.json` both serve `1.6.11` with matching Sequoia/Tahoe hashes.
- Homebrew cask commit `6db8a2cbbc321d8c29aeb16b2a47ed088d0eb5fa` points to the Sequoia/Sonoma-compatible `1.6.11` DMG and passed `brew style`.

## Release boundary

This checkpoint does not reclassify historical PARTIAL rows as complete. Larger and longer media, Omni audio, gateway soak, and remaining family-specific reliability/latency/eviction rows stay on the post-release worklist.
