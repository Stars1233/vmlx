# Homebrew cask checkpoint — v1.6.15

Status: PASS for the public `jjang-ai/homebrew-mlxstudio` cask surface.

## Published source

- Repository: `https://github.com/jjang-ai/homebrew-mlxstudio`
- Public `main`: `d4f0ab4293ce89096754925a14716c7c8e068ade`
- Version/checksum commit: `e21a5ab946f02ea48b8d53e9e147106d0109059e`
- Livecheck/modern macOS syntax commit: `d4f0ab4293ce89096754925a14716c7c8e068ade`
- Cask version: `1.6.15`
- Cask artifact: `vMLX-1.6.15-sequoia-arm64.dmg`
- Declared SHA-256: `c1bfa6e6b62e2e322461fd549203599f912dc4688e2c31e86d83d7b68c69a4cf`

## Current public verification

- Ruby syntax: PASS (`Syntax OK`).
- `brew style`: PASS (`1 file inspected, no offenses detected`).
- `brew livecheck`: PASS (`mlxstudio: 1.6.15 ==> 1.6.15`).
- `brew audit --cask --strict --online`: PASS (exit 0).
- `brew fetch --cask --force`: PASS (`Cask mlxstudio (1.6.15)`, exit 0).
- Downloaded artifact: 505,489,566 bytes.
- Downloaded artifact SHA-256: `c1bfa6e6b62e2e322461fd549203599f912dc4688e2c31e86d83d7b68c69a4cf`.
- GitHub release asset API reports the same byte size and SHA-256 digest.

The first strict online audit exposed a cask metadata issue: without an
explicit GitHub latest-release livecheck, Homebrew inferred the unrelated
date-like version `20260415-0612`. Commit `d4f0ab4` adds the authoritative
GitHub release livecheck and updates the deprecated macOS dependency syntax.
The strict audit passed after that correction.

No install command was run because installing the cask would replace or
conflict with the user's existing `/Applications/vMLX.app`. The release
checkpoint already contains independent signed-app install and Gatekeeper
proof; this lane validates the public Homebrew resolution, download, and
checksum boundary.

## Evidence files

- `homebrew-cask-public.rb`
- `homebrew-public-head.json`
- `homebrew-release-asset.json`
- `homebrew-checksums.txt`
- `homebrew-style.log`
- `homebrew-livecheck.log`
- `homebrew-audit.log`
- `homebrew-fetch.log`
