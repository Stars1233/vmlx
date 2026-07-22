# vMLX 1.6.15 public checkpoint — 2026-07-22

Status: `PUBLIC_CHECKPOINT_RELEASED_BROADER_MATRIX_PARTIAL`.

vMLX 1.6.15 is a public Python/Electron release checkpoint. The exact tagged
source was built into separate Sequoia and Tahoe applications, Developer ID
signed, Apple-notarized, stapled, independently verified, installed, and driven
through the real Electron Start/Stop and chat controls. The source release,
MLXStudio DMG release, updater feeds, package-manager surfaces, and durable
evidence were then published and re-read from their public endpoints.

This checkpoint does **not** close the broader all-family/media/gateway/stress
matrix. Older `PARTIAL`, `OPEN`, or artifact-specific `BLOCKED` rows remain
retained unless a newer named source-plus-live proof explicitly supersedes them.

## Exact source and version truth

- Source repository: `jjang-ai/vmlx`
- Release worktree: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Release branch: `codex/postrelease-ui-drawers-20260720`
- Exact built source commit: `344b6c88e46ce3eaf4aeb32108a48ed7144c7d2f`
- Annotated tag: `v1.6.15`
- Tag object: `2dc90921ea8604f4ec4c62e196621007fbb1cbbf`
- Tag peel: `344b6c88e46ce3eaf4aeb32108a48ed7144c7d2f`
- Manifest publication commit: `f724b122dcc733b2120bba8d1cb168cdf38e9c66`
- Source release: <https://github.com/jjang-ai/vmlx/releases/tag/v1.6.15>
- Version `1.6.15` is present in `panel/package.json`,
  `panel/package-lock.json`, `pyproject.toml`, and
  `vmlx_engine/__init__.py`.
- Bundled JANG source: `/Users/eric/jang` at
  `801209c13c189ebb8fb4d1596748a336f568da38`, package version `2.5.31`.
  The tracked JANG tree is clean and `HEAD...origin/main` is `0 0`; unrelated
  untracked research/backup files were intentionally left untouched.

## Source, suite, and production-build gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Full Python suite | PASS | `full-python-pytest-1.6.15.log`: `6290 passed, 96 skipped, 92 deselected, 2 warnings in 254.16s` |
| Full panel Vitest | PASS | `panel-vitest-full-after-packaging-fix.log`: 80 files passed, 2405 tests passed, 3 skipped |
| TypeScript typecheck | PASS | `panel-typecheck-after-packaging-fix.log`: `tsc --noEmit` completed |
| Production panel build | PASS | `panel-build.log` and `build-release-dmgs-merged-344b6c88e.log` |
| Bundled Python/source parity | PASS | Build log verifies vmlx `1.6.15`, JANG `2.5.31`, critical source hashes/imports, and 503 signed native files |
| Packaging regression | PASS | `panel/tests/release-packaging.test.ts` covers removal of six bundled Windows launcher stubs before macOS signing |

The broad `npm run release:prepackage` historical matrix remains negative in
`release-prepackage.log`. Publication used the script-supported scoped release
path and the signed-app live gate below. This is an honest checkpoint boundary,
not a claim that every retained family row is green.

## Signed and notarized artifacts

- Build log: `build-release-dmgs-merged-344b6c88e.log`
- Notary log: `notarize-release-dmgs-344b6c88e.log`
- Independent verifier: `verify-release-dmgs-344b6c88e.log`

Both apps pass deep/strict `codesign`, identify ShieldStack LLC team
`55KGF2S5AY`, pass `stapler validate`, and are Gatekeeper-accepted as
`Notarized Developer ID`.

| Artifact | Bytes | SHA-256 | Notary submission |
| --- | ---: | --- | --- |
| `vMLX-1.6.15-sequoia-arm64.dmg` | 505489566 | `c1bfa6e6b62e2e322461fd549203599f912dc4688e2c31e86d83d7b68c69a4cf` | `d967e0f8-8760-4471-944a-60989357e542` |
| `vMLX-1.6.15-tahoe-arm64.dmg` | 521245148 | `ae5a41c60fd79a39238e03fd74c1df2f5d92a2e57df8a60ff58ee34e248eb4be` | `bb654f20-2cc3-475f-9a32-c398a59dae75` |
| `vMLX-1.6.15-sequoia-arm64.dmg.blockmap` | 526356 | `0b675ec4351e5e78b6e84b4ff87c4844e75237d01bb46d4f620f5bc7f4c83555` | N/A |
| `vMLX-1.6.15-tahoe-arm64.dmg.blockmap` | 544722 | `56c482d2c0dd506cb76d0aa919c2069d5bc8bcd748ea799e17187f9003ca7f35` | N/A |

The final public DMG URLs were downloaded afresh and streamed through
`shasum -a 256`; both hashes matched the table above. GitHub's public asset API
also reports the same sizes and SHA-256 digests.

Public DMG release:
<https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.15>

## Installed signed-app Electron proof

The final DMGs were installed independently as:

- `/Applications/vMLX-1.6.15-Sequoia-Checkpoint.app`
- `/Applications/vMLX-1.6.15-Tahoe-Checkpoint.app`

Both report app version `1.6.15`, pass codesign/Gatekeeper checks after
installation, and loaded the bundled `vmlx_engine 1.6.15` using the real
Electron Sessions **Start** control. The packaged-app log line is
`[Engine Manager] Found bundled Python with vmlx_engine 1.6.15 (from dist-info)`;
packaged apps intentionally use bundled Python rather than a development
`.venv/bin/vmlx-engine` PATH.

Sequoia evidence:

- `sequoia-installed-app.log`
- `sequoia-ui-three-turn.png`
- `sequoia-ui-three-turn-rows.json`
- `sequoia-chat-sse.txt` and `sequoia-chat-sse.trace`
- `sequoia-responses-sse.txt` and `sequoia-responses-sse.trace`
- `sequoia-health-loaded.json`, `sequoia-health-after-streams.json`,
  `sequoia-health-after-warm.json`, and
  `sequoia-health-after-restart-disk.json`

The three real UI turns retained non-empty visible answers and distinct
reasoning rails. Turn 2 executed exactly one real
`file_info(panel/package.json)` and completed
`REL1615-SEQ-UI2-DONE SIZE=5.2 KB`; turn 3 recalled the prior facts without a
second tool call. Chat SSE emitted progressive separate `reasoning_content` and
content deltas, terminal `stop`, usage, and `[DONE]`. Responses emitted
protocol-native reasoning summary deltas followed by progressive output-text
deltas and one `response.completed` terminal.

Tahoe evidence:

- `tahoe-installed-app.log`
- `tahoe-ui.png` and `tahoe-ui-rows.json`
- `tahoe-chat-sse.txt` and `tahoe-chat-sse.trace`
- `tahoe-health-loaded.json` and `tahoe-health-after-ui-api.json`

The real UI emitted separate reasoning and exact visible content
`REL1615-TAH-UI1-DONE`. Raw Chat emitted progressive reasoning/content deltas,
terminal `stop`, usage, and `[DONE]`. Both model engines were stopped again
through the real UI after proof.

## Cache hierarchy proof in the signed app

The Sequoia installed app exercised Gemma 4's mixed rotating-SWA/full-attention
cache contract:

- cold: 127 q4 native-TQ L2 writes covering 8002 tokens; q4 applies only to
  full-attention KV boundaries while rotating state remains native;
- warm: 49 cached tokens reported as `paged+mixed_swa+tq-native` with successful
  native-TQ reconstruction/dequantization;
- real UI Stop/Start: the same prompt restored from disk as
  `paged+mixed_swa+disk+tq-native`, with one disk block and eight native-TQ
  slots/hits.

This signed-app row confirms the named Gemma mixed-cache path. It does not
promote other architecture-specific cache families without their own retained
live evidence.

## Public distribution and updater truth

- Source release: `jjang-ai/vmlx` `v1.6.15`, public, non-draft,
  non-prerelease, published `2026-07-22T09:05:45Z`.
- DMG release: `jjang-ai/mlxstudio` `v1.6.15`, public, non-draft,
  non-prerelease, published `2026-07-22T09:11:19Z`.
- MLXStudio manifest commit/tag: `95135bab9242110d99fc4dd1275809d6352b00ec`
  / `v1.6.15`.
- Raw `jjang-ai/vmlx/main/latest.json`, raw
  `jjang-ai/mlxstudio/main/latest.json`, and
  <https://mlx.studio/update/latest.json> were purged/re-read and are
  byte-identical with SHA-256
  `80a15ab8a52b360b8f3a07f6546acd33ccbe56bfe17250be44ade4b92c192b5f`.
- `public-vmlx-latest.json`, `public-mlxstudio-latest.json`, and
  `public-site-latest.json` retain those public reads.
- PyPI `vmlx==1.6.15` is public. The wheel is 1720560 bytes with SHA-256
  `cffa81c3b4093394bd70874a9b4623ef3651cbfea0d3442ceecc7bb06be21f0e`;
  the sdist is 2744159 bytes with SHA-256
  `1114d8bd5872a6d2e5b6b1d5fc6b547560b76190339ae160fc5aaa77fae07c4c`.
  Public downloads are byte-identical to a clean detached build at the exact
  tag. A clean no-deps venv installed from the public index and reported
  `1.6.15` from both distribution metadata and `vmlx_engine.__version__`.
  Official workflow run `29907439101` validated the tag, built, and passed
  `twine check`, then failed only at trusted-publisher authorization with
  `invalid-publisher`; the authorized authenticated fallback published the
  same exact-tag artifacts. See `PYPI.md` and its sanitized supporting files.
- Homebrew tap `jjang-ai/homebrew-mlxstudio` is public at
  `d4f0ab4293ce89096754925a14716c7c8e068ade`. The cask declares version
  `1.6.15` and the exact Sequoia digest above. `brew style`, livecheck, strict
  online audit, and forced cask fetch passed; the fetched 505489566-byte file
  independently hashes to the expected Sequoia SHA-256. A cask install was not
  run because it would replace the user's existing `/Applications/vMLX.app`;
  both final DMGs were already independently installed under isolated names
  for the signed-app proof above. See `HOMEBREW.md` and its sanitized logs.

## Retained scope after this checkpoint

The 1.6.15 release is a usable public checkpoint, not a global production-readiness
claim. Continue from `docs/internal/ISSUE-LEDGER.md` and
`../20260716_release_closeout/CURRENT-MATRIX.md`. In particular, do not erase
retained `PARTIAL`, `OPEN`, or artifact-specific `BLOCKED` rows for broader
family/media breadth, long-context and stochastic quality, gateway/network
soak, repeated model swaps, stale-path UX, or models not exercised by these two
signed-app smokes.

## Publication hygiene

- The exact built/tagged source commit remains immutable; manifest/evidence
  follow-ups are later commits.
- `.agents/LOG.md`, `.agents/STATUS.md`, `panel/node_modules`, secrets, and
  unrelated dirty/untracked JANG research files were not staged.
- Release notes contain no AI attribution and retain the requested
  `@Hornsan1` credit.
