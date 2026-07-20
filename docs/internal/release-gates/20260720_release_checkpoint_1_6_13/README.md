# vMLX 1.6.13 release checkpoint — 2026-07-20

Status: **PUBLIC CHECKPOINT RELEASED — BROADER MATRIX PARTIAL**

This is the canonical record for the requested v1.6.13 public checkpoint. The
exact source, suite counts, artifact hashes, notary submissions, installed-app
evidence, and public surfaces below are current. This release does not promote
the separately retained model, parser, media, gateway, or stress rows.

## Included source changes

- `5f05ad72a` fixes shared Electron Chat/Responses consumption of a stream that
  emits progressive output and later fails. The UI now consumes the
  authoritative failed terminal and usage, persists the safe partial prefix,
  strips UI-only interruption text from recovery history, and completes an
  immediate same-chat retry.
- `d811270ad` fixes Ollama mid-stream failure conversion and route ownership.
  Chat, templated generate, and raw generate now emit native `error` objects;
  a later upstream `[DONE]` cannot create a false `done:true` success.
- `8111e4799` records the scoped Anthropic/Ollama production-handler proof.
- `e7cf91237` bumps all Python/Electron version surfaces to `1.6.13`.
- `57a88388c` permits explicitly isolated secondary installed-app proof
  profiles without disturbing the user's ordinary app instance.
- `5fae65d38` isolates bundled engine child imports with `PYTHONSAFEPATH=1`
  and a stable bundled-binary working directory. This fixes the installed app
  abort seen when a launch cwd contained another top-level `mlx` directory.
- Version surfaces are bumped together in `pyproject.toml`,
  `vmlx_engine/__init__.py`, `panel/package.json`, and
  `panel/package-lock.json`.

## Source and protocol evidence already current

- Chat/Responses evidence:
  `../20260719_midstream_failure_recovery/README.md`.
- Anthropic/Ollama evidence:
  `../20260720_anthropic_ollama_midstream_failure/README.md`.
- Literal `curl -N --no-buffer` failure/recovery pairs use the production
  stream generators, handlers, and adapters. Failures retain progressive
  content, end on the native failed terminal, never synthesize success, and
  leave immediate recovery usable.

## Release gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Complete Python suite | PASS | 6,185 passed, 99 skipped, 92 deselected, 2 warnings; JUnit: `~/.cache/vmlx-release/1.6.13/python-junit-release.xml` |
| Complete panel suite | PASS | 76/76 files; 2,336 passed, 3 skipped after the spawn-isolation regression was added |
| TypeScript typecheck | PASS | `npm run typecheck` on v1.6.13 source |
| Bundled Python verification | PASS | engine 1.6.13 plus pinned clean JANG 2.5.31; critical source hashes/imports matched |
| Sequoia/Tahoe production build | PASS | Exact pushed source `5fae65d38`; both bundles were rebuilt after the installed-app import-path fix |
| Apple notarization/stapling | PASS | Accepted submissions `bc4293f5-02f8-4f28-9cd3-d7bf51031f51` (Sequoia) and `4dbf39a0-d2ec-43a8-a126-ca24f3cdc3d0` (Tahoe); independent verification saw stapled tickets and Gatekeeper acceptance |
| Exact installed-app smoke | PASS scoped | On `erics-m5-max.local`, both final DMGs were installed separately, passed `codesign --deep --strict` and `spctl`, loaded Gemma through the real Electron Start control, generated in the UI, streamed raw API output, and stopped through the UI |
| Public release surfaces | PASS-PUBLIC | Source and four-asset DMG releases, both GitHub manifests, `mlx.studio` feed, PyPI, and Homebrew were independently re-read at 1.6.13 with exact hashes |

## Final artifacts

| Artifact | Bytes | SHA-256 | SHA-512 (base64) |
| --- | ---: | --- | --- |
| `vMLX-1.6.13-sequoia-arm64.dmg` | 509,252,693 | `21cf069cd1adf7d0a3903ee96290986248f2ed7a634df6944122f6e5e16490f2` | `7JerdVJljbgb/ImsBH0ELRFkoPonplhVLQUhcxePfIDhuJmcEDUUrc3xTdFOT/N5MaIoXcecpSeYV1z7wyZuVw==` |
| `vMLX-1.6.13-tahoe-arm64.dmg` | 525,095,986 | `a244eedb3f94400dee8a068966851ed9bfb7c39eef07846191d51fbc7d60b887` | `uCmn8mLomXM13owVnUBqboEasKQDgpvMG194hMt6DjcWVBrnmB1OIeDq2XdIvrdYom5TWnZgNLOTxUJj94L/yQ==` |
| Sequoia blockmap | 531,335 | `829677b79b9e16d7b9628072c0df004fbe88ea3706b720f4b2dceb70f64ce8df` | — |
| Tahoe blockmap | 544,852 | `f966303753ba3c41c5a8ba039ae6c26278e5a5e894f0ccece4b6c7d4f30be8e7` | — |

Remote hashes matched the build machine byte-for-byte before installation.

## Public surface verification

- Source release: `https://github.com/jjang-ai/vmlx/releases/tag/v1.6.13`.
  The annotated tag object `e536c9fefb90919f74f22be2e1fb9a62d9be7185`
  peels to source commit `2f509f79d7829119308a36a02f13fd590dd2010e`.
- DMG release: `https://github.com/jjang-ai/mlxstudio/releases/tag/v1.6.13`.
  It is public, non-draft, and non-prerelease. GitHub reports all four assets
  uploaded with the exact byte counts and SHA-256 digests in this record.
- The updater-repo `v1.6.13` lightweight tag and `main` both resolve to
  `07c402d426f125e1ded175b34d52a16e3769dd8a`, the 1.6.13 manifest commit.
  The tag was corrected before checkpoint close because the initial release
  creation had captured the preceding 1.6.12 manifest commit.
- `jjang-ai/vmlx/main/latest.json`,
  `jjang-ai/mlxstudio/main/latest.json`, and
  `https://mlx.studio/update/latest.json` all publicly serve 1.6.13 with the
  exact Sequoia and Tahoe hashes. The custom origin update preserved its prior
  manifest as
  `/var/www/mlx.studio/update/latest.json.20260720T100947Z.bak`.
- PyPI publicly serves `vmlx==1.6.13`. Wheel: 1,699,110 bytes, SHA-256
  `363e5e3ee5a2ff45a8e675f4d31005a2bb77a9fa2078c280ca80b3f76b7a2100`.
  Sdist: 2,702,455 bytes, SHA-256
  `bbd2141b1fa83c809492da5dea705a74437b1f3ade8d10cc007785b8f1541f24`.
  Both match the distributions built and checked from the exact tag on
  `erics-m5-max.local`.
- Homebrew commit `0b0f54ca59c8f371de1f070bed2720306cf722c0`
  publishes cask version 1.6.13 with the exact Sequoia hash. `brew style`
  inspected the cask with no offenses.
- The repository PyPI workflow built and checked the tagged distributions but
  its trusted-publisher step failed because the GitHub/PyPI OIDC mapping is
  still not configured. Publication used the existing authenticated
  `.pypirc` on the trusted live-model Mac without exposing the credential.
  Repairing trusted publishing remains an operational follow-up, not an
  unpublished-artifact blocker.

## Cross-box source and runtime truth

- Runtime/install proof was performed on `erics-m5-max.local`, not on the
  build Mac. The two installed 1.6.13 apps remain independently addressable at
  CDP ports 9461 and 9462 with their model engines stopped after proof.
- The other Mac's older dirty checkout at `/Users/eric/mlx/vllm-mlx` was left
  untouched. A clean release worktree was created at
  `/Users/eric/mlx/vllm-mlx-release-1.6.13` and verified at the exact tagged
  commit. Post-release documentation is synchronized there after its final
  proof commit rather than overwriting the dirty checkout.

## Installed-app live proof

The final proof host was `erics-m5-max.local` (Apple M5 Max, macOS 26.3.2).
Local-machine activity was limited to source/build/sign/notary operations.

### Sequoia artifact

- The real Electron Start action launched the bundled command from
  `/Applications/vMLX-1.6.13-Sequoia-Checkpoint.app/.../python3` on port 8141.
  Session logs contain no duplicate-MLX import abort.
- The loaded `gemma-4-12B-it-qat-JANG_4M` artifact was identified as affine
  `JANG_4M`, not JANGTQ/MXTQ or base MLX MXFP. Health exposed Gemma's native
  mixed-SWA cache with q4 TurboQuant only at stored full/sliding-attention KV
  boundaries and preserved rotating-window metadata.
- A clean three-turn UI chat retained separate reasoning and visible content,
  executed exactly one real `file_info(panel/package.json)` call/result
  (`5.2 KB`), and recalled both earlier markers without another tool call.
- The clean UI rows restored 3,264 and 3,359 tokens as
  `paged+mixed_swa+tq-native`. Before restart, health recorded 90 q4-native
  block writes and 5,384 L2 tokens.
- After visible Stop/Start, the identical fresh-chat prompt restored 3,359
  tokens as `paged+mixed_swa+disk+tq-native`: `disk_hit=true`, 53 disk blocks,
  53 q4-native hits, reconstruction/dequantization successful, and exact final
  marker.
- Raw Responses with a 2,048-token cap emitted 120 separate
  reasoning-summary deltas, 93 content deltas, and one `response.completed`
  with final usage. Raw Chat with usage enabled emitted 113 reasoning deltas,
  23 content deltas, `finish_reason=stop`, usage 61/171/232, and `[DONE]`.

### Tahoe artifact

- The independent Tahoe DMG launched its own bundled engine on port 8142 via
  the real UI Start control and loaded the same model before any request.
- The visible Electron turn completed with separate reasoning and non-empty
  coherent content. Raw Chat emitted 141 reasoning deltas, 24 content deltas,
  `finish_reason=stop`, usage 63/211/274, and `[DONE]`.
- The real UI Stop control shut the engine down after proof.

Evidence is committed under `evidence/`, including screenshots, sanitized
session logs, SQLite row exports, health snapshots, and literal raw SSE files.

## Honest failure/partial observations

- An initial Sequoia tool prompt was sent while the per-chat built-in-tools
  toggle was visibly Off and no working directory was configured. Gemma
  printed raw tool-like markup and hallucinated `221 bytes`; that row is not
  counted as a tool PASS. Once the UI toggle was enabled and
  `/Users/eric/mlx/vllm-mlx` was saved, two independent tool turns executed
  exactly once with real persisted calls/results reporting `5.2 KB`. This was
  test setup, not a parser closure claim.
- The first raw Responses probe used `max_output_tokens=512` and correctly
  ended `response.incomplete` with reason `max_output_tokens`; it is retained
  as a truthful truncation control. The 2,048-token rerun completed.
- Gemma sometimes adds explanatory visible prose despite an exact-output
  instruction (seen in the completed Responses and Tahoe UI rows). Streaming,
  terminal, cache, and coherence checks pass; strict marker-only reliability
  remains `PARTIAL` and no output rewrite or hidden sampler coercion was added.

## Retained PARTIAL / OPEN work after this checkpoint

Packaging does not promote any row below. The detailed authoritative list
remains `docs/internal/release-gates/20260716_release_closeout/CURRENT-MATRIX.md`.

- broad signed-app repetition of every model family, parser, and cache type;
- remaining parser-family and cross-model post-tool Electron rows;
- gateway network-loss injection and longer repeated swap/unload/LAN/port soak;
- long/stochastic output quality, latency, and strict-format reliability for
  Laguna, Bonsai, Step, DSV4, MiniMax-M3, openPangu, LFM, Nemotron, and other
  retained families;
- additional image/video/audio attachment, same-media reuse, media-salt,
  restart/L2, and post-media tool axes not already linked as scoped proof;
- openPangu 512K work, DSV4 controlled reference-vs-vMLX sampling A/B, and
  other architecture-specific performance/quality follow-ups;
- remaining translated modal/drawer/accessibility/minimum-width breadth and
  stale-path cleanup beyond the already proved repoint/remove flow.

## Checkpoint stop condition

The exact artifacts are signed, notarized, stapled, installed-smoked,
published, and independently re-read from their public surfaces. After this
record and the master ledgers are committed/pushed and the other Mac fetches
that post-release proof commit, work pauses as requested. The retained
PARTIAL/OPEN rows above remain the continuation list for a later campaign.
