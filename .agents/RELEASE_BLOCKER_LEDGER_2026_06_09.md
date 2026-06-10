# vMLX / MLXStudio release blocker ledger - 2026-06-09

Scope: active Python engine and Electron/panel app in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

## 2026-06-09 18:00 PDT checkpoint DMGs

Eric explicitly overrode the red prepackage gate for a checkpoint build. Current-source vMLX 1.5.56 Sequoia/Tahoe DMGs were built, Developer ID signed, notarized, stapled, blockmap-regenerated, and post-staple verified from `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

- Sequoia: `panel/release/vMLX-1.5.56-sequoia-arm64.dmg`
  - SHA-256: `014ef3a9d729bf6b63091e28c82cfe86a9921397aa3d27621cab5f0e0541652f`
- Tahoe: `panel/release/vMLX-1.5.56-tahoe-arm64.dmg`
  - SHA-256: `272f9c9551fa99332b66c0a686083d94d0b2bf7c5359d310d4983d322dd01686`

Commands used:

```sh
security unlock-keychain -p vmlx-release ~/Library/Keychains/vmlx-build.keychain-db
security set-keychain-settings ~/Library/Keychains/vmlx-build.keychain-db
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k vmlx-release ~/Library/Keychains/vmlx-build.keychain-db
VMLINUX_CHECKPOINT_RELEASE_OVERRIDE=1 \
  VMLX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-checkpoint-dmg-override-20260609.json \
  panel/scripts/build-release-dmgs.sh all
VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh
panel/scripts/verify-release-dmgs.sh
```

Final verify reported `Notarization Ticket=stapled`, `source=Notarized Developer ID`, `TeamIdentifier=55KGF2S5AY`, valid `hdiutil` checksums, and Developer ID signatures for both DMGs.

Boundary: the override manifest remains `status=fail`, `prepackage_ready=false`, and `release_ready=false`. This is a checkpoint artifact only unless Eric explicitly publishes it as such. Do not claim full production clearance for N2/MiMo/Gemma/media/tool/UI/cache/runtime rows from notarization alone. No tag, appcast/latest.json mutation, GitHub release publish, public download update, or PyPI publish was performed.

Hard boundary: no fake fixes, no forced sampling/parser/cache behavior to hide runtime/model bugs, and no signing/notarization/tag/download release until the runtime, cache, parser, UI, and installed-app rows are green or Eric explicitly overrides.

Reporter credit: include GitHub `@Hornsan1` in next release notes/changelog/public acknowledgement for reported runtime/model/UI/API issues.

## Active blockers and proof requirements

1. Responses streaming tool arguments

- Need raw SSE local direct vs gateway vs tunnel comparison on the same model and same request.
- Do not fix by disabling reasoning or reducing reasoning effort.
- Current synthetic/local engine paths can emit non-empty `response.function_call_arguments.delta` and `response.function_call_arguments.done`; next proof must use the reported model/request and deployed tunnel path.
- Specific issue to trace: streaming code near line `13592` checks `if tc_args:`; reported failure means `tc_args` is empty. Trace why `_parse_tool_calls_with_parser` returns empty args when reasoning is on, and trace how the streaming loop accumulates tool-call deltas.
- Gateway/tunnel/port/wake/sleep/session routing must be checked separately from engine parsing. A Cloudflare/gateway model miss or stale sleeping backend is not the same as a local parser bug.
- No executable `{}` argument payload should be emitted for required tool calls when required args are actually missing; fail closed instead.
- Current split: local no-heavy guards cover empty XML required-arg fail-closed behavior and output-index ordering. Gemma4 E2B direct/gateway captures preserve `record_fact` args and valid indices, but the public tunnel does not advertise `gemma4-e2b-sse`. Qwen35 public tunnel preserves args and reasoning events, but reuses `output_index=0` for both message and function_call. Treat Qwen35 as deployed/tunnel freshness unless current-source direct/gateway raw SSE reproduces the duplicate index.
- Current same-model Qwen35 direct proof: `build/current-responses-raw-sse-parity-qwen35-direct-source-vs-tunnel-20260609.json` captures current source against `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`; direct source uses `message=[0]`, `function_call=[1]`, preserves `{"value": "blue-cat"}`, and keeps reasoning events on. The artifact still fails because gateway is missing and the tunnel still duplicates `output_index=0`.
- Next proof: panel-gateway raw SSE with reasoning enabled and the exact Qwen35 tunnel request. If gateway uses output index `1`, rebuild/redeploy the tunnel backend and recapture; if it duplicates `0`, reopen the source streaming finalization path.

2. MiniMax random Chinese / visible planning

- Not cleared.
- Do not clamp sampling, inject sentinels, or hide visible planning as a "fix."
- Need isolate cache-on vs cache-off, TurboQuant KV vs none, L2 vs no L2, parser/template boundary, request/session reuse, and model-owned `generation_config.json`.
- Random Chinese or random-language output is likely cache continuation, prompt-contract contamination, parser/tool memory injection, L2 disk stale restore, or logit/decode-loop corruption until proven otherwise.
- Read prior docs/artifacts for similar cross-model cache/language/planning leaks before touching defaults.

3. MiMo V2.5

- JANGTQ_2 speed/cache is partially good, but release proof is not complete.
- JANG_2L, tool/JSON/loop, media, L2 restore, UI parity, and installed-app proof remain open.
- 2026-06-10 update: MiMo JANG_2L installed-app text/cache is now green in
  `build/current-real-ui-installed-app-mimo-v25-jang2l-text-cache-proof-20260610.json`.
  The local rebuilt `/Applications/vMLX.app` loaded the 105 GiB row, produced
  exact visible text turns, used native `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`, hit paged cache, and wrote block L2. This does not
  clear MiMo installed-app tools, media, JANGTQ_2 exactness, speed, public
  tunnel SSE, or Developer ID DMG readiness.
- 2026-06-10 update: MiMo JANG_2L installed-app tools are now explicitly red in
  `build/current-real-ui-installed-app-mimo-v25-jang2l-tools-proof-20260610.json`.
  The installed app reached the real built-in tool surface and executed one
  `run_command` on the second turn, but failed `long_tool_loop`: the first
  marker mutated to `REAL_UI_LAND_TOOL_ONE`, the first probe file was missing,
  and visible output degraded into repetitive tool-planning prose. Cache/L2 was
  positive, so do not chase prefix/L2/TurboQuant as the primary blocker.
- 2026-06-10 update: MiMo JANG_2L installed-app image/media is now explicitly
  red in `build/current-real-ui-installed-app-mimo-v25-jang2l-image-proof-20260610.json`.
  The installed app attached an image and server `MEDIA_DIAG` saw `image_url`,
  but the runtime returned the honest guard `400 - unsupported media modality
  image because the loaded runtime is text-only; supported modalities: text`.
  Forced MLLM is intentionally overridden because media weights are preserved
  but unwired. Cache/L2 was positive, so do not claim media support or chase
  cache as the blocker.
- 2026-06-10 update: MiMo JANGTQ_2 installed-app short text/cache is green in
  `build/current-real-ui-installed-app-mimo-v25-jangtq2-text-cache-proof-20260610.json`.
  The rebuilt app loaded the 79 GiB bundle, produced exact
  `MIMO_JANGTQ2_TEXT_ONE` / `MIMO_JANGTQ2_TEXT_TWO`, hit paged cache with
  `cache_hit_tokens=42`, and wrote block L2. This does not clear the broader
  JANGTQ_2 artifact exactness/tool/media/source-vs-quant blockers.
- 2026-06-10 update: MiMo JANGTQ_2 installed-app image/media is now explicitly
  red in `build/current-real-ui-installed-app-mimo-v25-jangtq2-image-proof-20260610.json`.
  The app attached an image and server `MEDIA_DIAG` saw `image_url`, but the
  runtime returned the honest text-only guard. Vision tensors/config metadata
  are present, but this runtime is not wired for media.
- Do not run source-vs-quant comparisons if RAM-blocked unless Eric explicitly allows.
- Exactness failures must not be papered over by parser repair, sampling clamps, cache disabling, or JSON repair.
- Confirm runtime dynamically reads artifact config for bit size, grouped experts, stacked vs legacy layout, JANG/JANGTQ/MXFP metadata, and model-owned generation defaults.

4. N2 / Qwen-family JANG and JANGTQ

- Need live tool, reasoning, parser, cache, Responses, streaming args, and UI proof.
- Especially cover `gdn_sink`, MTP, hybrid SSM/cache, paged cache, TurboQuant KV encode/decode, block-disk L2, and no parser/reasoning leaks.
- JANG_1L should fit with careful RAM handling; treat current blocker as careful live-proof scheduling and memory discipline, not permanent infeasibility.
- Do not launch N2/JANG_1L below the preflight headroom gate after Metal OOM evidence. If preflight says `do_not_launch`, clear RAM or schedule later instead of forcing it.
- 2026-06-10 update: refreshed no-load preflight
  `build/current-n2-pro-jang1l-local-memory-preflight-20260610-after-installed-app-proofs.json`
  still says `decision=do_not_launch`: payload `110.57 GiB`, required
  available `118.57 GiB`, observed available `112.77 GiB`, gap `5.8 GiB`.
  Eric explicitly overrode the launch-safe gate, so
  `build/current-n2-jang1l-live-chat-cache-override-20260610.json` launched
  anyway with smaller batches/cache knobs. It still failed before health with
  Metal OOM after `Wired limit set to 115 GB (model 119 GB)`. JANG_1L still
  needs a real lower-peak runtime strategy before release support can be
  claimed.
- 2026-06-10 launch-safe refresh:
  `build/current-n2-pro-jang1l-local-memory-preflight-launch-safe-20260610.json`
  and `build/current-n2-jang1l-chat-cache-launch-safe-20260610.json` confirm
  the safe gate still skips before launch: available `114.22-114.23 GiB`,
  required `118.57 GiB`, gap about `4.35 GiB`. Requested tool, Responses,
  Responses stream, and L2 restart probes were recorded but not launched.
- 2026-06-10 high-free launch attempt:
  `build/current-n2-pro-jang1l-local-memory-preflight-ultrafree-20260610.json`
  still says strict `decision=do_not_launch` with observed available
  `114.09 GiB`, required `118.57 GiB`, gap `4.48 GiB`. Per Eric's instruction
  to launch one-at-a-time anyway,
  `build/current-n2-jang1l-live-chat-cache-ultrafree-20260610.json` ran with
  lowered JANG_1L headroom (`3 GiB`) plus low-peak knobs and still failed at
  `phase=server_startup`: qwen3_5_moe/JANG_1L detection, qwen parser, qwen3
  reasoning parser, hybrid cache, attention-only TurboQuant KV, native SSM
  companion state, mmap JANG loader, `123` shards, bfloat16 for `512` experts,
  `Wired limit set to 115 GB (model 119 GB)`, then Metal OOM before health.
- N2 JANGTQ_2 proof does not clear N2 JANG_1L.
- Keep architecture names explicit in every proof: base Qwen/Qwen35 MXFP8-MTP direct-source proof does not clear Nex/N2 Pro 397B JANG_1L, and N2 JANG_1L does not clear regular Qwen MTP/JANGTQ rows. Record `format`, `weight_format`, `artifact_profile`, MTP depth, `gdn_sink`, hybrid SSM/native-cache schema, TurboQuant KV state, and media weight backing from loaded health/config rather than inferred family names.

5. DSV4

- Memory-unit harness fix is landed by the other agent.
- Still needs live default-cache tool-loop proof when memory gate allows.
- Must use native SWA/CSA/HCA cache behavior, not a generic fake KV substitute.
- Do not claim DSV4 release clearance from load, health, narrow cache proof, or memory-label cleanup alone.

6. Gemma 4 QAT/native MXFP4/MXFP8/JANG

- Source startup and several source smokes exist, but full MXFP4/MXFP8/JANG_4M/QAT media/cache/UI/installed-app/tunnel matrix remains open.
- Need E2B, E4B, 12B, 26B, and 31B downloaded/present from JANGQ HF repos for full multi-turn, tool-call, parser, cache-reuse, coherency, and media proof.
- Incoherent multilingual/token-soup output from older app versions is likely real runtime/load/decode corruption until proven otherwise. Known class: Gemma4 MoE/native-MXFP sidecar hydration or shared-KV/load compatibility can corrupt generation.
- `ModuleNotFoundError: No module named 'mlx_vlm.models.gemma4_unified'` is a release blocker until the packaged/bundled runtime has the current compatibility alias/shim and installed-app parity is green.
- VLM/image prefill guard recovery and post-error text recovery still need proof. A failed image turn must not poison later text turns in the same chat/session.
- Audio/video claims must be weight-backed and runtime-proven. Token metadata alone is not native audio proof.

7. Step 3.7

- Metadata/runtime route matrix is improved.
- Do not make a fake `has_vision=false` release claim unless the release row is explicitly scoped text-only.
- Tool dialect loops, raw XML-like leaks, tool-result continuation, multi-turn synthesis, and thinking-template mismatch remain model/runtime parser issues to prove per path.
- If Step3p7 VLM runtime is advertised, prove live media or fail closed with honest capability reporting.

8. Structured JSON/XML

- Repair/validation is application and benchmark hygiene, not a substitute for runtime coherence.
- Guided/schema decoding should only be claimed if real runtime support exists.
- JSON/XML repair should report raw parse success, repair-needed status, normalized object, schema validation result, and retry/fail-closed behavior.

9. UI/CLI/API parity

- Parser, reasoning, cache type, prefix cache, paged cache, L2 disk cache, TurboQuant KV, MTP, max output tokens, max context, generation defaults, and media settings must match CLI, API, panel settings, installed-app launch, and persisted session settings.
- Responses endpoints must cover content-delta streaming, function-call argument streaming, final object consistency, previous response/session reuse, cancel/restart, and gateway/tunnel parity.
- Do not clear panel/UI rows from CLI source proof alone.

10. Release/package/sign/notarize

- Developer ID signing and notary profile access are currently usable after the documented keychain unlock/partition-list sequence; `build/current-signed-checkpoint-dmg-readiness-20260609.json` records fresh signing `pass` and notarization `pass`.
- Current release packaging is still blocked on rebuilding current-source DMGs, notarizing/stapling/verifying those current artifacts, and clearing or explicitly scoping the remaining runtime/model/UI/cache rows.
- No signing, notarization, tag, public download update, or release announcement until runtime/model/UI/cache blockers are green or Eric explicitly overrides.
- If a release is forced with known open rows, the release notes must list exact open rows and not imply full clearance.
- 2026-06-10 update: `/Applications/vMLX.app` was rebuilt and locally installed
  with `panel/scripts/build-and-install.sh` after a pre-rebuild audit found only
  `vmlx_engine/utils/jang_loader.py` stale. The refreshed installed-app runtime
  parity audit is
  `build/current-installed-app-runtime-parity-audit-after-local-install-20260610.json`
  with `status=pass`, `missing_or_stale=[]`, bundled engine hash parity true,
  and packaged source hash parity true. This clears the local installed-app
  runtime/source parity blocker only; it is not a Developer ID notarized DMG
  release and does not clear model-specific installed-app chat proofs or the
  remaining runtime/model/UI/cache rows.
- 2026-06-10 N2 installed-app update:
  `build/current-real-ui-installed-app-n2-jangtq2-responses-tools-cache-20260610.json`
  is `status=pass` for the local rebuilt `/Applications/vMLX.app` plus
  `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`. It proves
  installed-app UI, `/v1/responses`, two built-in `run_command` calls,
  tool-result continuation, visible content deltas, server cache controls,
  no raw parser/reasoning leak, `hybrid_ssm_v1`, attention-only TurboQuant KV,
  native SSM companion state, block L2, and SSM disk hit. It does not clear
  public tunnel SSE, N2 audio, N2 JANG_1L, stricter prompt quality, or
  Developer ID DMG package/sign/notarize/release readiness.
- 2026-06-10 N2 installed-app media update:
  `build/current-real-ui-installed-app-n2-jangtq2-image-proof-20260610.json`
  and `build/current-real-ui-installed-app-n2-jangtq2-video-proof-20260610.json`
  are both `status=pass` for the same local rebuilt `/Applications/vMLX.app`
  and N2 JANGTQ2 row. They prove installed-app image/video attachment
  persistence, server `MEDIA_DIAG` detection, image `num_images_processed=1`,
  video base64 MP4 decode and `4` extracted frames, visible answers `Red` and
  `solid red screen`, no raw parser/reasoning leak, `hybrid_ssm_v1`,
  attention-only TurboQuant KV, native SSM companion state, block L2, and SSM
  companion disk stores. They do not clear N2 audio, N2 JANG_1L, public tunnel
  SSE, or Developer ID DMG package/sign/notarize/release readiness.
- 2026-06-10 N2 installed-app audio update:
  `build/current-real-ui-installed-app-n2-jangtq2-audio-proof-20260610.json`
  is `status=fail` for the same local rebuilt `/Applications/vMLX.app` and N2
  JANGTQ2 row. It proves installed-app audio attachment plumbing reached the
  server boundary: the app completed two text turns first, forced multimodal
  for one attached audio file, server `MEDIA_DIAG` saw `input_audio`, and the
  API returned the honest guard `400 - unsupported media modality audio;
  supported modalities: text, vision, video`. Runtime/cache remained green
  before the gate with `hybrid_ssm_v1`, attention-only TurboQuant KV,
  `cache_detail=paged+ssm`, block L2, and SSM companion disk stores. This does
  not clear N2 audio support, N2 JANG_1L, public tunnel SSE, or Developer ID DMG
  package/sign/notarize/release readiness.
- 2026-06-10 Gemma installed-app update:
  `build/current-real-ui-installed-app-gemma4-12b-mxfp4-responses-tools-cache-20260610.json`
  is `status=pass` for the local rebuilt `/Applications/vMLX.app` plus
  `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4`. It proves
  installed-app UI, `/v1/responses`, two built-in `run_command` calls,
  tool-result continuation, visible content deltas, server cache controls,
  no raw parser/reasoning leak, MXFP4 affine matmul with Metal NA active,
  native `mixed_swa_kv_v1`, paged mixed-SWA cache, and block L2. It does not
  clear Gemma installed-app media, Gemma audio, public tunnel SSE, or Developer
  ID DMG package/sign/notarize/release readiness.
- 2026-06-10 Gemma installed-app image update:
  `build/current-real-ui-installed-app-gemma4-12b-mxfp4-image-proof-20260610.json`
  is `status=pass` for the same local rebuilt `/Applications/vMLX.app` and
  Gemma 12B QAT MXFP4 row. It proves installed-app image attachment persistence,
  server `MEDIA_DIAG` detection, Gemma media fallback with `1 image(s)`,
  visible answer `Red`, no raw parser/reasoning leak, MXFP4 affine matmul with
  Metal NA active, native `mixed_swa_kv_v1`, paged mixed-SWA cache, and block
  L2 write. It does not clear installed-app video, Gemma audio, public tunnel
  SSE, or Developer ID DMG package/sign/notarize/release readiness.
- 2026-06-10 Gemma installed-app video update:
  `build/current-real-ui-installed-app-gemma4-12b-mxfp4-video-proof-20260610.json`
  is `status=pass` for the same local rebuilt `/Applications/vMLX.app` and
  Gemma 12B QAT MXFP4 row. It proves installed-app video attachment persistence,
  server `MEDIA_DIAG` detection, base64 MP4 decode, `25` frames at `25.0 fps`,
  `4` extracted frames, Gemma media fallback, visible answer describing a solid
  red background, no raw parser/reasoning leak, MXFP4 affine matmul with Metal
  NA active, native `mixed_swa_kv_v1`, paged mixed-SWA cache, and block L2
  write. It does not clear Gemma audio, public tunnel SSE, or Developer ID DMG
  package/sign/notarize/release readiness.
- 2026-06-10 Gemma installed-app audio update:
  `build/current-real-ui-installed-app-gemma4-12b-mxfp4-audio-proof-20260610.json`
  is `status=fail` for the same local rebuilt `/Applications/vMLX.app` and
  Gemma 12B QAT MXFP4 row. It proves installed-app audio attachment plumbing
  reached the server boundary: the app completed two text turns first, forced
  multimodal for one attached audio file, server `MEDIA_DIAG` saw
  `input_audio`, and the API returned the honest guard `400 - unsupported media
  modality audio; supported modalities: text, vision, video`. Runtime/cache
  remained green before the gate with MXFP4 affine matmul, Metal NA active,
  native `mixed_swa_kv_v1`, generic TurboQuant KV correctly disabled,
  `cache_detail=paged+mixed_swa`, and block L2 writes. This does not clear
  Gemma audio support, public tunnel SSE, or Developer ID DMG
  package/sign/notarize/release readiness.
- 2026-06-10 Gemma JANG4M installed-app update:
  `build/current-real-ui-installed-app-gemma4-12b-jang4m-responses-tools-cache-20260610.json`
  is `status=pass` for the local rebuilt `/Applications/vMLX.app` and
  `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`. It proves installed-app
  `/v1/responses`, built-in `run_command` tool calls, `previous_response_id`
  tool-result continuations, visible assistant turns, streaming deltas,
  JANG affine Metal NA, native mixed-SWA cache, and block L2. It does not clear
  JANG4M installed-app video/audio, larger Gemma QAT rows, public tunnel SSE, or
  package/sign/notarize/release readiness.
- 2026-06-10 Gemma JANG4M installed-app image update:
  `build/current-real-ui-installed-app-gemma4-12b-jang4m-image-proof-20260610.json`
  is `status=pass` for the local rebuilt `/Applications/vMLX.app` and
  `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`. It proves installed-app
  image attachment persistence, Gemma media fallback with `1 image(s)`, visible
  answer `Red`, no raw parser/reasoning leak, JANG affine Metal NA, native
  mixed-SWA cache, and block L2 writes. It does not clear JANG4M installed-app
  video/audio, public tunnel SSE, or package/sign/notarize/release readiness.
- 2026-06-10 Gemma JANG4M installed-app video update:
  `build/current-real-ui-installed-app-gemma4-12b-jang4m-video-proof-20260610.json`
  is `status=pass` for the local rebuilt `/Applications/vMLX.app` and
  `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` with explicit
  `max_prompt_tokens=12000`. It proves installed-app video attachment
  persistence, server `MEDIA_DIAG` video detection, base64 MP4 decode, 4-frame
  extraction from the 25 fps fixture, visible solid-red-screen answer, no raw
  parser/reasoning leak, JANG affine Metal NA, native mixed-SWA cache, and block
  L2 writes. It does not clear JANG4M installed-app audio, default-4k video,
  public tunnel SSE, or package/sign/notarize/release readiness.
- 2026-06-10 Gemma JANG4M installed-app audio update:
  `build/current-real-ui-installed-app-gemma4-12b-jang4m-audio-proof-20260610.json`
  is `status=fail` for the local rebuilt `/Applications/vMLX.app` and
  `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`. It proves audio
  attachment plumbing reached runtime: `persistedAudioAttachment=true`, server
  `MEDIA_DIAG` saw `input_audio`, and the server decoded the base64 WAV. The
  audio turn still ended with empty visible assistant content and no
  `audio_where_supported` surface, so JANG4M installed-app audio remains red.
  This does not invalidate installed-app text/tools/image/video green rows, and
  it does not clear public tunnel SSE or package/sign/notarize/release readiness.
- Proper release mechanics are documented in `/Users/eric/wiki/infra/apple-notarization.md`; do not invent an alternate path. The canonical keychain is `~/Library/Keychains/vmlx-build.keychain-db`, the Developer ID identity is `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`, and notarization uses the `vmlx-notary` keychain profile.
- If signing returns `errSecInternalComponent`, fix key access with the documented sequence and retry once after the partition-list grant settles:

```sh
security unlock-keychain -p vmlx-release ~/Library/Keychains/vmlx-build.keychain-db
security set-keychain-settings ~/Library/Keychains/vmlx-build.keychain-db
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k vmlx-release ~/Library/Keychains/vmlx-build.keychain-db
```

- Current Python/Electron DMG flow is the repo script path, not a manual ad-hoc app signing path:

```sh
VMLINUX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-pre-dmg-release-build-<scope>.json panel/scripts/build-release-dmgs.sh all
VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh
panel/scripts/verify-release-dmgs.sh
```

- `panel/scripts/build-release-dmgs.sh` first runs `tests/cross_matrix/run_release_regression_manifest.py --require-prepackage-ready`; if that ledger fails, the script must stop before DMG build. Do not bypass this stop except under an explicit checkpoint-release override that records the open rows in the release notes.
- `panel/scripts/notarize-release-dmgs.sh` submits the final signed Sequoia and Tahoe DMG containers, staples both, regenerates blockmaps, runs `spctl`, and prints final SHA-256 values. `panel/scripts/verify-release-dmgs.sh` is the post-staple verification pass.

## Other-agent integration requirements

- Before release, include all pushed source fixes from the other agent and verify they are present in the active worktree, `origin/main`, bundled Python, and installed app.
- Track issue/fix coverage by architecture: dense, MoE/routed, hybrid SSM, SWA/CSA/HCA, MLLM/VLM, audio/video, JANG, JANGTQ/MXTQ, MXFP4, MXFP8, QAT, native MTP.
- A fix for one family does not clear another family unless the same path is live-proven.
- Do not duplicate old work from `/Users/eric/vmlx`, ADLab, Max2 transport notes, or deprecated Swift lanes.

## Immediate focus order

1. Keep status/docs current so a second agent can pick up without guessing.
2. Fix and prove Responses streaming tool arguments with reasoning enabled on the real reported model/request and direct/gateway/tunnel surfaces.
3. Continue Gemma QAT/native MXFP4/MXFP8/JANG proof after downloads are available, including multi-turn, tools, parser leaks, cache reuse, media, and installed app.
4. Continue MiMo and N2 live runtime proof under memory gates, not source-vs-quant heavy comparisons.
5. Rebuild current-source DMGs only after the chosen checkpoint scope is explicit; then notarize/staple/verify with the documented `vmlx-build.keychain-db`/`vmlx-notary` path.
