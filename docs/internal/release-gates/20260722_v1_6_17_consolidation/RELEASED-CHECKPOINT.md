# vMLX 1.6.17 released checkpoint

Date: 2026-07-23 (America/Los_Angeles)

Status: `PUBLIC CHECKPOINT RELEASED / RETAINED MATRIX PARTIAL`.

This document freezes what was actually released, what was verified against
the signed artifacts, and what remains required. It does not promote the
broader campaign matrix to complete.

## Exact source and public surfaces

- vMLX source/tag:
  `d03ec297ef269bceb8a805726f6733a8c497f130` / `v1.6.17`.
- vMLX source release:
  <https://github.com/jjang-ai/vmlx/releases/tag/v1.6.17>.
- DMG distribution commit/tag:
  `580f56b` / `v1.6.17`.
- DMG release:
  <https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.17>.
- Homebrew cask commit:
  `8850e51`.
- PyPI:
  <https://pypi.org/project/vmlx/1.6.17/>.
- Bundled JANG tools:
  `2.5.34`, source commit
  `f7583e445e8bf75abe49b5cd0a19370a1d5ceb41`.

The GitHub trusted-publisher workflow run `30059283499` built and validated
the release artifacts but failed authentication because PyPI has no matching
trusted publisher for `repo:jjang-ai/vmlx:environment:pypi`. The exact tagged
source was then built locally, passed `twine check`, and uploaded using the
existing protected `.pypirc` credential. No credential value was printed,
copied into source, or committed.

The canonical GitHub updater feed and source-repository mirror are current at:

<https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json>

<https://raw.githubusercontent.com/jjang-ai/vmlx/main/latest.json>

The separately hosted mirror
`https://mlx.studio/update/latest.json` remained at `1.6.15` at release time.
SSH to `45.32.71.230` failed public-key authentication, so this mirror is
`BLOCKED` rather than falsely reported updated.

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| `vMLX-1.6.17-sequoia-arm64.dmg` | `d79bcaf6959870ac8db7b1f3554b170fab84395653fefcc8b11e78eb8929b312` |
| `vMLX-1.6.17-sequoia-arm64.dmg.blockmap` | `b312df0cfed0ad07b583079fbd68774f22e67e3d8879f0f5d1b3c82ab7831301` |
| `vMLX-1.6.17-tahoe-arm64.dmg` | `f8f27dfefb10d5a09325fd90985af8b8afa19f124b2821e8afe1c536512f8feb` |
| `vMLX-1.6.17-tahoe-arm64.dmg.blockmap` | `b8e3329b80d04272b130baaafe7d3ca2c7cc2cf9b174d297acd271a98f0517cd` |

Post-signing validation:

- every staged app passed the recursive audit of `518` Mach-O leaves;
- both DMGs passed `hdiutil verify`;
- both apps passed deep/strict Developer ID verification;
- both DMG containers were accepted by Apple, stapled, and validated;
- copied installed apps passed Gatekeeper with
  `source=Notarized Developer ID`.

Notarization IDs:

- Sequoia:
  `19b61c46-acdd-4ef4-a5d2-54a056f42af6`;
- Tahoe:
  `e999d5ab-f5c6-4406-9fee-0a3d50dc7f8c`.

Source logs on the M5 Max:

- `build/r17-build-release-dmgs-final-d03ec297e.log`;
- `build/r17-notarize-final-d03ec297e.log`;
- `build/r17-verify-final-d03ec297e.log`;
- `build/r17-final-artifact-hashes-d03ec297e.log`.

## Mounted and installed artifact provenance

Both mounted DMGs reported:

- app version `1.6.17`;
- bundled Python `3.12.12`;
- bundled `vmlx_engine=1.6.17`;
- bundled `jang_tools=2.5.34`;
- package imports resolving below the mounted app's
  `Contents/Resources/bundled-python/python` directory.

Both installed copies passed deep/strict code-sign checks, Gatekeeper, and
bundled imports. The live app log reported:

```text
[Engine Manager] Engine version check: installed=1.6.17, source=1.6.17, needsUpdate=false
[Engine Manager] Found bundled Python with vmlx_engine 1.6.17 (from dist-info)
```

The installed apps were launched with isolated user data and CDP ports. Older
1.6.14/1.6.15/dev app processes were not used as proof and were not
terminated.

The test host ran macOS Tahoe `26.3.2`. Both artifacts were exercised on that
host; there is no genuine Sequoia-OS runtime row in this checkpoint.

## Exact-head source gates

At `d03ec297e`:

- panel: `86` files, `2493 passed`, `3 skipped`;
- TypeScript `tsc --noEmit`: pass;
- production Electron build: pass;
- Python: `6411 passed`, `97 skipped`, `92 deselected`, with one failure in
  the one-shot environment because its fallback JANG source was `2.5.31`;
- the isolated bundle verifier against exact JANG `2.5.34` passed.

This is recorded as a composed source gate. It is not misreported as a single
all-green Python invocation.

## Signed-app UI and API smoke

### Sequoia artifact

The real create-session flow selected and loaded
`mlx-community/Qwen3-0.6B-8bit` with the packaged engine on port `8000`.
The signed UI showed a separate Reasoning rail and a non-empty visible answer.
Direct and gateway Chat SSE each produced:

- non-empty `delta.reasoning_content`;
- progressive `delta.content`;
- exact visible `SIGNED-R17-API-OK`;
- `stop` and `[DONE]`;
- no visible `<think>`, tool, or `[THINK]` marker.

The tiny Qwen model did **not** satisfy the tool-quality row: it hallucinated a
size instead of emitting `file_info`. This is retained as a model-level smoke
failure, not presented as a tool pass.

The signed app then loaded
`JANGQ-AI/MiniMax-M3-Coder-Small` on port `8001`. UI turns had separate
reasoning and visible content. The exact-tool prompt was refused by the model
instead of emitting a tool call. Exact source-level M3 and HY3 tool-loop
evidence remains under `m3-tool-math-live/` and
`release-candidate-smoke/`, but the signed-artifact tool repetition is
honestly `FAIL`.

### Restart/L2 refault proof

After two M3 turns had stored typed blocks, the model was stopped with the real
UI control and restarted from the same signed app/session. A history follow-up
returned the correct comparison and displayed:

```text
827 prompt (600 paged+disk cached)
```

Live health after the restarted turn reported:

- `cache_hit_requests=1`;
- `cache_hit_tokens=600`;
- detail `paged+disk`;
- `disk_blocks=10`;
- `reconstruction_ok=true`;
- `disk_hits=10`;
- native schema `minimax_m3_msa_v1`;
- components
  `attention_kv`, `msa_idx_keys`, `absolute_block_index`;
- generic TurboQuant correctly disabled for the native sparse indexer tuple.

This is current signed-app evidence for one typed M3 restart refault. It does
not establish the full cross-family eviction matrix.

### Tahoe artifact

The notarized Tahoe app opened the same persisted profile, visibly restored
both saved sessions, and started M3 through the real UI. A fresh UI chat and
fresh direct/gateway Chat requests each produced separate reasoning and the
non-empty visible answer:

```text
2 plus 2 equals 4.
```

One adversarial persisted-history UI turn emitted `27` reasoning characters
and then terminalized with no visible answer or tool call. This is logged as
the P0 `Q36MTP-UI-REASONING-REPLAY` class. Fresh UI, direct, and gateway
requests were healthy, but the persisted-history failure is not rationalized
or marked fixed.

Retained screenshots:

- `signed-release/r17-signed-sequoia-loaded.png`;
- `signed-release/r17-signed-sequoia-ui-turn1-final.png`;
- `signed-release/r17-signed-sequoia-ui-turn2.png`;
- `signed-release/r17-signed-sequoia-m3-ui-turn3-restart.png`;
- `signed-release/r17-signed-tahoe-restored-sessions.png`;
- `signed-release/r17-signed-tahoe-m3-fresh-ui.png`.

## Required cache proof procedure retained for follow-up

The following is a mandatory proof row, not a design note:
The canonical reusable protocol, required retained artifacts, architecture
variants, and invalid-proof examples are in
`CACHE-EVICTION-REFAULT-PROOF.md`.

1. Start from a clean cache and submit prompt/prefix `A`.
2. Submit enough distinct `B`, `C`, `D`, and later prompts to exceed the
   configured in-memory block budget.
3. Prove `A` is no longer resident in RAM while its valid blocks remain on
   L2 SSD.
4. Submit `A` plus a never-cached suffix.
5. Require the longest valid prefix match from SSD, refault only matching
   blocks, prefill only the unmatched suffix, and produce coherent output.
6. Require rising disk-hit/restored-token counters, an explicit execution
   record, and a measured prefill/TTFT saving rather than relying on a log
   phrase.
7. Restart the model/session and repeat the replay.
8. Repeat with in-memory paged cache enabled and disabled.
9. Repeat under a small configured L2 GB cap and prove least-recently-used
   disk blocks are removed while a surviving prefix still refaults.
10. Repeat under a small prefix-cache RAM percentage and prove old unused RAM
    blocks evict instead of exceeding the configured ceiling.
11. Verify an unsafe RAM percentage produces a visible OOM-risk warning based
    on physical device capacity.
12. Verify stored TurboQuant KV encode/decode for plain attention caches and
    architecture-native restore/rederive for SSM, GDN/CCA, mixed SWA, M3
    sparse indexer, DSV4 composite state, and OpenPangu native prompt state.
13. Corrupt or remove one companion artifact and require a safe cache miss or
    async rederive, never a partial wrong-state restore.

This procedure must be applied to every representative cache architecture.
Existing scoped Laguna, Bonsai, Qwen, Gemma, M3, HY3, DSV4, and OpenPangu
artifacts may satisfy individual subrows, but no universal pass may be inferred
from one family.

## Retained follow-up work

- Fix and repeat the persisted-history reasoning-only P0 row.
- Repeat installed-app tool loops with a model that actually emits the
  requested tools; do not promote the Qwen/M3 refusals.
- Complete the cache proof procedure above across all cache architectures.
- Complete Paged-Off SSD-only partial-prefix breadth, eviction/refault, L2
  capacity rotation, RAM-cap eviction, and the OOM-risk warning.
- Complete media A/B/A salt, restart/L2, and post-media tool continuation for
  advertised image/video/audio families.
- Repeat Chat, Responses, Anthropic, and Ollama agent loops across retained
  parser families and longer histories.
- Complete gateway multi-client/model-swap/LAN/port/disconnect soak.
- Close retained quality/performance rows for Laguna, Bonsai, Step, DSV4,
  M3, OpenPangu, LFM, Gemma, Nemotron, Qwen, and other listed families.
- Repair PyPI trusted-publisher configuration so future releases do not
  require the local credential fallback.
- Update the separately hosted `mlx.studio` mirror once server access is
  restored.

## Verdict

`v1.6.17` is a public, signed, notarized, installed-app-tested checkpoint.
It is **not** a declaration that the complete model/media/cache/gateway matrix
is done. The retained work above remains `PARTIAL` or `FAIL` exactly as stated.
