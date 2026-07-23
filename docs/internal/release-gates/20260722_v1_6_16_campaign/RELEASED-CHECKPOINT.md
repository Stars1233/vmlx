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
distribution manifest commit: e60316b
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
jang commit: 6e28ff20f6f2df60145a4c05fcbd77423b0745c2
jang package version: 2.5.34
```

The release build installed both packages from local source into the bundled
Python runtime. Build log evidence:

```text
build/vmlx-1.6.16-build-release-dmgs.log:473
==> Installing vmlx-engine + jang_tools (local source)...

build/vmlx-1.6.16-build-release-dmgs.log:474
using local vmlx at /Users/eric/mlx/vllm-mlx-r16-reasoning-p0

build/vmlx-1.6.16-build-release-dmgs.log:490
using local jang-tools at /Users/eric/jang/jang-tools-r16-2534/jang-tools

build/vmlx-1.6.16-build-release-dmgs.log:543
ok bundled vmlx_engine version matches package.json (1.6.16)

build/vmlx-1.6.16-build-release-dmgs.log:546
ok bundled critical vmlx_engine files match source content

build/vmlx-1.6.16-build-release-dmgs.log:547
ok bundled critical jang_tools files match source content

build/vmlx-1.6.16-build-release-dmgs.log:599
bundled-python: all critical imports ok
```

The same verifier rows passed again for the Tahoe app bundle in the same build
log.

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
986ce9a44f7cff90d34a5d79cbb72a6eb7dc63b87c1dc0ef37c0605a33564b8a  vMLX-1.6.16-sequoia-arm64.dmg
6a31f573e37cf8089abe26b70dfcd9403167348a30ab07d7bfe4843b714b1621  vMLX-1.6.16-tahoe-arm64.dmg
e16ee322c653092f30bf6f9aecaf2c88d8224e032e5c98077294bf8dc63447f4  vMLX-1.6.16-sequoia-arm64.dmg.blockmap
cc4e7e1d3363e771555071143faaca18d0e3da7c53005f7a057a2239b5e30120  vMLX-1.6.16-tahoe-arm64.dmg.blockmap
```

SHA512 for updater manifest:

```text
Sequoia DMG:
JdlAb7DVYyLMpPdDT0B202svKtIT4gttu5z24ea4la2Nwsu9a8XQuMJ1aifp7c/iqFK2L+o8YQwfDiXyUCmIGQ==

Tahoe DMG:
goTAcQeF4e6uVQlQHOXckFXXYKCSWlBjvEF4C/Z7PYRh6BiFKQjxmjEj5zHYZ2TQl7Sgx8wIobzAx+rYkWcxrw==
```

## Signing, notarization, stapling, and Gatekeeper

Signing identity:

```text
Developer ID Application: ShieldStack LLC (55KGF2S5AY)
```

Notarization results:

```text
Sequoia notarization id: 3b95132a-be73-41c4-8ac6-e428a8d5a748
Tahoe notarization id: de595b3d-7cea-4644-919c-389dd37894ed
```

Verification evidence:

```text
build/vmlx-1.6.16-notarize-release-dmgs.log
build/vmlx-1.6.16-verify-release-dmgs.log
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
jjang-ai/mlxstudio main: e60316b
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
