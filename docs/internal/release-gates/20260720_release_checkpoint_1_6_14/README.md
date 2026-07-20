# vMLX 1.6.14 public release checkpoint — 2026-07-20

Status: **PUBLIC CHECKPOINT RELEASED — BROADER MATRIX PARTIAL**

This is the canonical record for the requested v1.6.14 public checkpoint.
It records the exact tagged source, complete suite results, built artifact
hashes, Apple notary submissions, signed-installed-app Electron proof, raw API
stream proof, and independent public-surface reads. It does not promote the
separately retained model-family, protocol, media, gateway, quality, or stress
rows.

All build, model, signed-app, publication, and public re-read work was performed
on `erics-m5-max.local`. The controller checkout at `/Users/eric/vmlx` was not
used as the Python/Electron runtime or packaging host.

## Exact source and release refs

- Source worktree:
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`.
- Release branch: `codex/postrelease-ui-drawers-20260720`.
- Tagged source commit:
  `e1776a485e8a85f3957b79030e12f4c312eda04b`.
- Annotated source tag object:
  `420b3d91c54e5626164ea49faf7ee6783641df53`, which peels to the exact
  tagged source commit above.
- Public source manifest commit on `origin/main`:
  `3a7ac0188a50d135f90bd0e090f6efc47671fdb7`.
- Public updater manifest commit:
  `dab40cff2a9f4857b516015a7312b99648d2559f`.
- Annotated updater tag object:
  `8acf927b121d78010154f4edec1dc05ea0d8c741`, which peels to the updater
  manifest commit above.

## Included source changes since v1.6.13

- `9531eea95` keeps narrow settings drawers visible instead of allowing
  controls to overlap or become inaccessible at minimum width.
- `5c3a3b9ed` localizes icon accessibility states.
- `24330de7e` honors LFM's native reasoning-mode contract and makes unmet
  required-tool behavior terminate truthfully instead of recording a false
  successful assistant turn.
- `1d04d49b3` preserves Ollama gateway backend error details.
- `7687f237b` uses Gemma 4's bundle-required media ordering.
- `a0abd7ab3` exposes and forwards validated Gemma 4 image-token budgets.
- `1b89e1118` keeps Gemma reasoning-only recovery on a fresh direct rail and
  prevents degraded `thought` control text from leaking into visible content.
- `26f3dfb59` honors Laguna's native Auto reasoning policy.
- `8b2e07dfe` salts Nemotron Omni conversation state with media identity,
  rehydrates the correct prior media for text-only continuation, and prevents
  cross-media stale replay.
- `e1776a485` aligns Python and Electron version surfaces at 1.6.14.

The detailed family evidence remains in the linked July 20 gate directories.
This checkpoint packages that current source; packaging itself is not evidence
that every retained family row is closed.

## Complete source/package gates

| Gate | Status | Current evidence |
| --- | --- | --- |
| Complete Python suite | PASS | 6,203 passed, 96 skipped, 92 deselected, 2 warnings in 234.75 s; JUnit: `~/.cache/vmlx-release/1.6.14/python-junit-release.xml` |
| Complete panel suite | PASS | 77 files; 2,346 passed, 3 skipped |
| TypeScript typecheck | PASS | `npm run typecheck` completed on exact v1.6.14 source |
| Electron production compile | PASS | Production panel/Electron compile completed on exact source |
| Bundled Python verification | PASS | Bundled engine reports 1.6.14; clean pinned JANG 2.5.31 and required runtime imports/source checks passed |
| Sequoia/Tahoe package build | PASS | Both DMGs were built after `bundle-python.sh` from the exact release source |
| Apple signing/notarization | PASS | Both apps/DMGs are Developer ID signed, notarized, stapled, and Gatekeeper accepted; exact submission IDs are below |
| Signed installed-app smoke | PASS scoped | Both final DMGs were installed separately, launched with isolated profiles, loaded the real Gemma bundle through Electron controls, generated visibly, exercised raw API streaming, and stopped through the UI |
| Public surfaces | PASS-PUBLIC | GitHub source/DMG releases, updater manifests, `mlx.studio`, PyPI, and Homebrew were downloaded again and matched the recorded versions and hashes |

## Final signed artifacts

| Artifact | Bytes | SHA-256 | SHA-512 (base64) |
| --- | ---: | --- | --- |
| `vMLX-1.6.14-sequoia-arm64.dmg` | 505,846,456 | `345fd1ec02bf039b4a113bc617c5fa4eca7c057577a100212e3587dd1bc8022c` | `qgQ/Ailo/o8C/zWkMwIkGwYaUjpoWotSf7e06bRLIZ0CCWWZvbF7FqJL7DHYclh16i2F5MpB+bYalADU8yCUtg==` |
| `vMLX-1.6.14-sequoia-arm64.dmg.blockmap` | 528,344 | `99fb36df897654f78cffc347c2aee3ef3551dc50ddf1f16acfa47ab1b3830d98` | — |
| `vMLX-1.6.14-tahoe-arm64.dmg` | 521,680,787 | `d77b49ede22d47f7cc2ebb3f3ecfe1b4425f92c05c20eff7be9d2ab6c97a739d` | `ri1ZgjK3vIZ5z7D+lc+Ax2RIAWU2+Os8O6Qu2HU8i1Dv+HMAsRsEWC9w/BQgW0agmeiVMyj0E3+g87kWjiGptA==` |
| `vMLX-1.6.14-tahoe-arm64.dmg.blockmap` | 542,774 | `eaa3aea294406fb1137f5c75fd34b78d9e84fc2feee030145a93adc4d1a80436` | — |

Notary submissions:

- Sequoia DMG: `5566cce1-48a1-405e-95ba-8b9f466f49a1`.
- Sequoia app ZIP: `b3d5b151-ac9f-42b7-aba1-02873a1a2ac6`.
- Tahoe DMG: `7bb7cf76-5dd2-4975-a0f6-2f0592049324`.
- Tahoe app ZIP: `9031ef00-3067-4220-9389-f686c45e08ab`.

Both final DMGs passed `hdiutil verify`. Their installed apps passed strict
`codesign` verification, `stapler validate`, and `spctl` as ShieldStack LLC,
team `55KGF2S5AY`.

## Installed signed-app proof

### Bundle-grounded model configuration

The exercised model was the real bundle
`/Users/eric/.mlxstudio/models/JANGQ-AI/gemma-4-12B-it-qat-JANG_4M`.
Its own `config.json` and JANG metadata identify:

- `model_type=gemma4_unified`;
- affine `weight_format=jang_affine`, profile `JANG_4M` — not JANGTQ/MXTQ
  and not base MLX MXFP;
- 40 rotating sliding-window layers plus eight full-attention layers;
- Gemma 4 reasoning/tool parsers;
- advertised vision and audio support.

Before launch, the visible settings showed Prefix Cache On, Paged Cache On,
Block Disk L2 On, and KV quantization Auto. Runtime health/logs then showed
1,000 configured paged blocks, 40 native rotating-cache layers, and q4 stored
TurboQuant only for the eight compatible full-attention KV slots. No claim is
made that rotating SWA state was converted into a generic full-KV cache.

### Sequoia

- Installed app:
  `/Applications/vMLX-1.6.14-Sequoia-Checkpoint.app`.
- Isolated profile: `/Users/eric/.vmlx-v1614-sequoia-checkpoint`.
- CDP/engine ports: 9463/8143.
- The real `Launch Session` control launched the bundled engine from the
  installed app. The log records `Found bundled Python with vmlx_engine
  1.6.14 (from dist-info)`; the UI showed no error toast.
- UI turn 1 separated reasoning from non-empty visible content and completed:
  `The result of multiplying 37 by 19 is 703... REL1614-SEQ-UI1-DONE`.
- UI turn 2 recalled 703 from the prior turn, computed 700, emitted eight
  numbered lines, and completed `REL1614-SEQ-UI2-DONE`. It restored 73 tokens
  as `paged+mixed_swa+tq-native` with 0.43 s TTFT.
- UI turn 2 spent 72.8 seconds and 2,782 tokens repeatedly checking a simple
  format constraint. DOM capture showed progressive repaint and the visible
  answer completed, so this is not a batch-at-end streaming failure; it remains
  a model verbosity/latency PARTIAL.
- With built-in tools visibly enabled and working directory set to
  `/Users/eric/mlx/vllm-mlx-release-1.6.13`, the counted tool turn emitted
  exactly one `file_info({"path":"panel/package.json"})`, received the real
  `5.2 KB` result, and exact-finaled `REL1614-SEQ-UI5-TOOL-DONE` with no warning.
- Literal Responses SSE emitted 337 reasoning-summary deltas and 50 output
  text deltas, then one reasoning done, one output done, one output item done,
  and one `response.completed` with usage. Final visible content was coherent
  and non-empty.
- The UI Stop control closed engine port 8143 after proof.

### Tahoe

- Installed app:
  `/Applications/vMLX-1.6.14-Tahoe-Checkpoint.app`.
- Isolated profile: `/Users/eric/.vmlx-v1614-tahoe-checkpoint`.
- CDP/engine ports: 9464/8144.
- A fresh `Launch Session` first loaded the model. The UI was then stopped and
  the literal `Start` button reloaded the exact installed Tahoe bundled engine;
  the packaged command is retained in the Logs screenshot/text artifact.
- Startup found the shared L2 store with 21 entries / 0.36 GB. The identical
  fresh-chat prompt restored 73 tokens across process, profile, and signed-app
  variant as `paged+mixed_swa+disk+tq-native`, with 0.29 s TTFT. The visible
  answer matched the coherent Sequoia result and marker.
- Literal Chat Completions SSE emitted 185 separate `reasoning_content`
  deltas and 18 separate `content` deltas, then `finish_reason=stop` and
  `[DONE]`. The final answer was coherent and non-empty.
- Logs recorded TQ write-through for eight compatible TQ slots while retaining
  the 40 rotating slots as typed rotating state.
- The UI Stop control closed engine port 8144 after proof.

## Tool/setup controls that are not counted as passes

- `REL1614-SEQ-UI3-TOOL` ran with built-in tools visibly Off. Gemma printed
  raw tool-like markup. This is retained as a negative control and is not a
  parser/tool success.
- `REL1614-SEQ-UI4-TOOL` ran after tools were enabled but before a working
  directory was set. The app truthfully persisted one structured call and the
  structured result `Error: Working directory not set...`. It is evidence of
  error propagation, not successful tool execution.
- Only `REL1614-SEQ-UI5-TOOL`, after visible UI configuration, is counted as
  the installed-app tool-loop PASS.

## Public publication and independent re-read

- Public source release:
  `https://github.com/jjang-ai/vmlx/releases/tag/v1.6.14`. It is non-draft and
  non-prerelease; public metadata reports publication at
  `2026-07-20T19:41:50Z`.
- Public DMG release:
  `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.14`. It is non-draft
  and non-prerelease; public metadata reports publication at
  `2026-07-20T19:43:23Z` and all four exact sizes/digests above.
- `jjang-ai/vmlx/main/latest.json`,
  `jjang-ai/mlxstudio/main/latest.json`, and
  `https://mlx.studio/update/latest.json` were freshly downloaded on the proof
  host. They are byte-identical, have SHA-256
  `e19da155e7853763cd953dc85a24ff092e19c83505535560cecda491d79c6a7c`,
  report version 1.6.14, and contain the exact Sequoia/Tahoe hashes.
- The custom origin update preserved the previous feed as
  `/var/www/mlx.studio/update/latest.json.20260720T194502Z.bak`.
- PyPI publicly serves `vmlx==1.6.14`:
  - wheel: 1,701,660 bytes, SHA-256
    `da84e4a68f4994c0c3ea2bfede490519f03a4c7f39a59974d1f00ace8c40afa4`;
  - sdist: 2,706,884 bytes, SHA-256
    `6483819395f74c0a46b2c513cde442da6aa900ff23e17930164c107ee7405157`.
- Homebrew commit `47a691a28487482aa486463af9ce4791748ee7fd`
  publishes `jjang-ai/homebrew-mlxstudio/Casks/mlxstudio.rb` at version 1.6.14
  with the exact Sequoia hash. `brew style` reported one file and no offenses.
- The fresh public verifier output and downloaded public metadata are committed
  under `evidence/public-proof/`.

## Evidence inventory

The committed `evidence/` directory contains:

- Sequoia and Tahoe UI screenshots before/after load, settings, Logs, answers,
  tool success, disk restore, and final Stop;
- sanitized installed-app logs;
- health snapshots;
- SQLite session and message exports including reasoning/content/tool fields;
- literal Responses and Chat Completions SSE captures;
- the complete Python JUnit file;
- the freshly downloaded source/DMG release metadata, updater manifests, PyPI
  metadata, Homebrew cask, and public-verification transcript.

## Honest scoped verdicts

- Installed-app model loading, Gemma bundle-grounded settings, multi-turn
  history, one real tool call/result/continuation, Responses/Chat streaming,
  mixed-SWA/q4 storage boundaries, paged reuse, and disk restore are
  `PASS-LIVE_SCOPED` for the exact signed checkpoint artifacts.
- Strict marker-only behavior and short-prompt reasoning economy remain
  `PARTIAL`. The 72.8-second UI2 reasoning rail completed progressively but is
  not acceptable evidence of low latency.
- Exact Gemma OCR remains `PARTIAL` from the earlier media gate; this signed
  checkpoint did not rerun image OCR.
- Nemotron Omni process-restart/L2 media-session restore remains `OPEN`.
  Ordinary scheduler cache counters are not substituted for Omni conversation
  persistence.
- The two setup/control tool rows remain retained failures and are excluded
  from the successful tool verdict.

## Retained PARTIAL / OPEN campaign work

The authoritative detailed list remains
`docs/internal/release-gates/20260716_release_closeout/CURRENT-MATRIX.md`.
This release does not close:

- every-family signed-app repetition, long/stochastic quality, latency, strict
  formatting, and longer agent/tool soak;
- the remaining Chat/Responses/Anthropic/Ollama cancellation, network-loss,
  disconnect, failure-recovery, and gateway soak matrix;
- the retained MiniMax M2.7/M3, openPangu, DSV4, Laguna, Bonsai, Step,
  Nemotron, Qwen, Gemma, LFM, and other architecture-specific breadth rows;
- remaining image/video/audio attachment, same-media, different-media salt,
  restart/L2, post-media text/tool, and Omni media persistence axes;
- openPangu 512K, DSV4 controlled reference-runtime sampling A/B, larger
  eviction/partial-prefix/fault-injection campaigns, and cache-disabled
  variants not already proven in their dedicated gates;
- remaining translated modal/drawer/accessibility/minimum-width breadth,
  repeated model-swap/LAN/port-conflict soak, and broader eager-load coverage.

## Checkpoint stop condition

The exact v1.6.14 source and updater tags are public, both DMGs are signed,
notarized, stapled, installed-smoked, and publicly downloadable, PyPI and
Homebrew are current, and the three updater feeds match byte-for-byte. Once
this record and the master ledgers are committed and pushed, work pauses as
requested. The retained PARTIAL/OPEN rows remain the continuation list for a
later campaign.
