# vMLX 1.6.16 released checkpoint

Date: 2026-07-22 (America/Los_Angeles)

Status: `RELEASED_SCOPED / FOLLOW-UP OPEN`.

This file is a post-publication record. The exact behavior-bearing source that
was built, signed, notarized, and tagged is:

```text
jjang-ai/vmlx tag: v1.6.16
source commit: f3426fb412588856a47f53b8549700267605c00f
source release: https://github.com/jjang-ai/vmlx/releases/tag/v1.6.16
distribution release: https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.16
distribution manifest commit: 3f2f0a4
```

The source `main` branch may contain documentation-only commits after the tag.
Do not treat those commits as the exact built source unless a later release
record says so.

## Release scope

This is the public v1.6.16 emergency parser/cache/bundled-runtime checkpoint.
It is not a closure of every retained family/media/gateway stress row.

The scoped package gate was:

```text
VMLINUX_RELEASE_SCOPE=r16_parser_cache
```

It passed from the release checkout and wrote:

```text
build/current-release-regression-manifest-pre-dmg-release-build.json
scope=r16_parser_cache
version=1.6.16
status=pass
```

The broad retained matrix remains follow-up work, especially broader
family/media repetition, complete gateway soak, remaining protocol variants,
full retained stress rows, PyPI JANG publication, and every row still marked
`PARTIAL`, `OPEN`, or `BLOCKED` in this campaign directory.

## Bundled-file provenance

Build input source:

```text
vmlx source: /Users/eric/mlx/vllm-mlx-r16-reasoning-p0
vmlx commit: f3426fb412588856a47f53b8549700267605c00f
vmlx version: 1.6.16
```

Bundled JANG source:

```text
jang source: /Users/eric/jang/jang-tools-r16-2534/jang-tools
jang branch: codex/r16-vmlx-1.6.16-jang-tools
jang commit: 2e85095f55b1b64cbb6e8264c82a074a3cc28250
jang package version: 2.5.34
```

The release build installed both packages from local source into the bundled
Python runtime. Build log evidence:

```text
build/vmlx-1.6.16-rebuild-jang-version-build.log:473
==> Installing vmlx-engine + jang_tools (local source)...

build/vmlx-1.6.16-rebuild-jang-version-build.log:474
using local vmlx at /Users/eric/mlx/vllm-mlx-r16-reasoning-p0

build/vmlx-1.6.16-rebuild-jang-version-build.log:490
using local jang-tools at /Users/eric/jang/jang-tools-r16-2534/jang-tools

build/vmlx-1.6.16-rebuild-jang-version-build.log:543
ok bundled vmlx_engine version matches package.json (1.6.16)

build/vmlx-1.6.16-rebuild-jang-version-build.log:546
ok bundled critical vmlx_engine files match source content

build/vmlx-1.6.16-rebuild-jang-version-build.log:547
ok bundled critical jang_tools files match source content

build/vmlx-1.6.16-rebuild-jang-version-build.log:599
bundled-python: all critical imports ok
```

The same verifier rows passed again for the Tahoe app bundle in the same build
log.

Mounted-DMG import smoke:

```text
build/vmlx-1.6.16-rebuild-jang-version-bundled-import-smoke.log
sequoia: bundled vmlx_engine=1.6.16, bundled jang_tools=2.5.34
tahoe: bundled vmlx_engine=1.6.16, bundled jang_tools=2.5.34
```

## Built artifacts

Local artifact paths:

```text
panel/release/vMLX-1.6.16-sequoia-arm64.dmg
panel/release/vMLX-1.6.16-sequoia-arm64.dmg.blockmap
panel/release/vMLX-1.6.16-tahoe-arm64.dmg
panel/release/vMLX-1.6.16-tahoe-arm64.dmg.blockmap
```

Public URLs:

```text
https://github.com/jjang-ai/mlxstudio/releases/download/v1.6.16/vMLX-1.6.16-sequoia-arm64.dmg
https://github.com/jjang-ai/mlxstudio/releases/download/v1.6.16/vMLX-1.6.16-sequoia-arm64.dmg.blockmap
https://github.com/jjang-ai/mlxstudio/releases/download/v1.6.16/vMLX-1.6.16-tahoe-arm64.dmg
https://github.com/jjang-ai/mlxstudio/releases/download/v1.6.16/vMLX-1.6.16-tahoe-arm64.dmg.blockmap
```

SHA256:

```text
dc2580136d253c293ce8b8d1e9983a82179902a055605999408482f07760c2c2  vMLX-1.6.16-sequoia-arm64.dmg
e421b4e75fbd2bf9a499b9f9471bd501d3d475a31fa4ead4cc10cf3dad5f6f3f  vMLX-1.6.16-tahoe-arm64.dmg
835813bd49b803478a0b187cc97881a83e6553cf794af6778097a1bb1ecfcd09  vMLX-1.6.16-sequoia-arm64.dmg.blockmap
8405a2c87ce041f00073dae1f990533d100d6ca688097cbf849440119ff32e41  vMLX-1.6.16-tahoe-arm64.dmg.blockmap
```

SHA512 for updater manifest:

```text
Sequoia DMG:
fkAX2RO6WzdwhbUrjj+KSgm1JucujY1iNMWe4VDp7aFyB3s23F7uSS5JdkNsQTsEDFxfoE2/CVAiK/B5mrwi/w==

Tahoe DMG:
CuxPhx7L0AKly6IvEkTITZ+0C4ONqLF0BDCpVmUdClSyv86rDlOkV3fG3PPSIk2XaC02FCmxFVaKupS/pdSKZQ==
```

## Signing, notarization, stapling, and Gatekeeper

Signing identity:

```text
Developer ID Application: ShieldStack LLC (55KGF2S5AY)
```

Notarization results:

```text
Sequoia notarization id: 4065f621-193e-4fdf-a420-0b9f73016a1c
Tahoe notarization id: c73d1038-58b2-4d0f-b38d-c50269a88670
```

Verification evidence:

```text
build/vmlx-1.6.16-rebuild-jang-version-notarize.log
build/vmlx-1.6.16-rebuild-jang-version-verify.log
build/vmlx-1.6.16-rebuild-jang-version-bundled-import-smoke.log
```

Both DMGs passed:

```text
hdiutil verify: checksum VALID
codesign --verify --deep --strict
xcrun stapler staple
xcrun stapler validate
spctl accepted
source=Notarized Developer ID
```

## Updater manifest

The public updater feed is:

```text
https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json
```

Distribution repo state:

```text
jjang-ai/mlxstudio main: 3f2f0a4
jjang-ai/mlxstudio tag: v1.6.16
```

The manifest names version `1.6.16`, the Sequoia and Tahoe public DMG URLs, and
the SHA256/SHA512 hashes listed above.

## Current retained follow-up boundaries

Do not represent this checkpoint as full campaign closure. Still open or
partial after the public 1.6.16 checkpoint:

- Broader family repetition beyond the scoped Laguna/Qwen/Bonsai/Gemma cache
  and reasoning proof set.
- Full media/audio/video breadth across Nemotron, Step, Gemma, Qwen variants,
  MiniMax M3, DSV4, and other advertised modality artifacts.
- Exhaustive Chat/Responses/Anthropic/Ollama gateway soak across all retained
  parser families.
- Remaining DSV4 long-output quality, MiniMax M3 sparse/MSA restart/eviction,
  openPangu native prompt disk, and LFM/Nemotron/Step residual rows.
- Complete retained full-matrix closure and any broad test rows not explicitly
  named in the scoped preflight.
- PyPI publication for the JANG package. The clean JANG source and GitHub
  branches were pushed, but PyPI credentials were unavailable in this run.
