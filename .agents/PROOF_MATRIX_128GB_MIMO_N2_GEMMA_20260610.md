# 128GB Checkpoint Proof Matrix - MiMo, N2, Gemma

Scope: current Python engine worktree
`/Users/eric/mlx/vllm-mlx-finite-launch-guard`. This file is for the
release/checkpoint lane. Do not use it to claim full release clearance; it
separates what was actually loaded and proven from what remains red.

## Proven Live On 128GB Host

### Dev-App Detector and Settings Launch Parity

Artifacts:

- `build/current-panel-settings-contract-proof-20260610-mimo-n2-gemma-launch-parity.json`
- `build/current-panel-exact-local-model-detect-mimo-n2-gemma-20260610.json`
- `build/current-installed-app-runtime-parity-audit-after-june10-devapp-proofs-20260610.json`
- `build/current-installed-app-runtime-parity-audit-after-local-install-20260610.json`
- `build/current-release-regression-manifest-after-local-installed-app-parity-20260610.json`

Proven:

- Panel settings/launch contract is current-source green:
  `status=pass`, `missing_source_markers=[]`, panel settings tests passed
  `315`, model registry tests passed `66` in the contract artifact,
  engine model registry passed `140`, and CLI flag contract passed `9`.
- Exact local Gemma 12B MXFP4 and JANG4M directories now autodetect as
  `family=gemma4`, `cacheType=rotating_kv`, `usePagedCache=true`,
  `toolParser=gemma4`, `reasoningParser=gemma4`, and multimodal.
- Exact local MiMo JANG_2L and JANGTQ_2 directories now autodetect as
  `family=mimo_v2`, `cacheSubtype=mimo_v2_asymmetric_swa`,
  `usePagedCache=true`, `toolParser=xml_function`, and no automatic reasoning
  claim.
- Exact local N2 JANGTQ2 directory autodetects as `qwen3.5-moe`,
  `cacheType=hybrid`, paged, Qwen tools, Qwen3 reasoning, TurboQuant, and
  multimodal.
- Exact local N2 JANG_1L directory autodetects as `qwen3.5-moe`,
  `cacheType=hybrid`, paged, Qwen tools, Qwen3 reasoning, but
  `forceTextOnly=true` until VL and memory-safe runtime proof exist.

Fixes made:

- Added panel `gemma4_unified` and `gemma4_unified_text` model-type aliases so
  Gemma 4 unified bundles do not fall to `unknown`.
- Aligned panel MiMo detection with the Python registry: XML tools remain on,
  `mimo_v2_asymmetric_swa` cache subtype is exposed, paged cache is forced for
  that subtype, and MiMo reasoning is not auto-advertised without visible-final
  thinking proof.
- Installed-app runtime parity was stale before the local rebuild: the June 10
  audit found only `vmlx_engine/utils/jang_loader.py` mismatched in both bundled
  Python and packaged source mirrors. `/Applications/vMLX.app` was rebuilt and
  reinstalled with `panel/scripts/build-and-install.sh`; rerun audit
  `build/current-installed-app-runtime-parity-audit-after-local-install-20260610.json`
  is `status=pass`, `missing_or_stale=[]`, bundled engine hash parity true,
  packaged source hash parity true, `serve_help_runs=true`,
  `xml_function_tool_parser_cli=true`, parser/reasoning settings wired,
  model-owned generation defaults wired, max-output/max-context settings wired,
  Responses stream cache-detail metrics wired, and single-model gateway cache
  endpoint routing wired.
- The rebuilt `/Applications/vMLX.app` also passed
  `codesign --verify --deep --strict --verbose=2`; this was a local app-dir
  install, not a Developer ID notarized DMG release.
- Regenerated no-heavy release manifest
  `build/current-release-regression-manifest-after-local-installed-app-parity-20260610.json`
  remains `status=fail`, `prepackage_ready=false`, and `release_ready=false`,
  but its current proof sweep has `installed_app_runtime_parity_audit=true` and
  `staged_app_runtime_parity_audit=true`.

Not proven:

- Electron UI clicked chat transcript for these exact rows.
- Model-specific installed-app chat transcript parity for these exact rows.
- N2 JANG_1L memory-safe live startup.
- MiMo JANGTQ_2 exactness.
- Gemma audio/video semantic E2E.
- Same-model direct/gateway/tunnel raw SSE deployed parity.
- Full release/prepackage readiness; the regenerated manifest still has open
  blockers outside installed-app runtime parity.

### Gemma 4 12B MXFP4 and JANG4M

Artifacts:

- `build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json`
- `build/current-gemma4-12b-mxfp4-jang4m-live-runtime-audit-20260610.json`
- `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-dev-app-proof-20260610.json`
- `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-image-proof-20260610.json`
- `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-video-proof-20260610.json`
- `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-audio-proof-20260610.json`
- `build/current-real-ui-installed-app-gemma4-12b-mxfp4-responses-tools-cache-20260610.json`
- `build/current-real-ui-installed-app-gemma4-12b-mxfp4-image-proof-20260610.json`
- `build/current-real-ui-installed-app-gemma4-12b-mxfp4-video-proof-20260610.json`
- `build/current-real-ui-installed-app-gemma4-12b-mxfp4-audio-proof-20260610.json`

Proven:

- MXFP4 and JANG4M both loaded through current vMLX server.
- Image/media path returned correct visible color answer.
- Conservative text runtime passed for both rows.
- Visible answer, multi-turn recall, reasoning-on visible answer, required
  tool call, and cache endpoint sanity passed.
- Real Electron dev-app Gemma 12B QAT MXFP4 Responses/tools/cache proof is
  green. The app loaded `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4`,
  executed the built-in `run_command` loop twice, used scoped Responses
  `function_call_output` follow-ups with `previous_response_id`, and created
  the expected probe files containing `REAL_UI_LIVE_TOOL_ONE` and
  `REAL_UI_LIVE_TOOL_TWO`.
- The MXFP4 app proof recorded content deltas for both assistant turns
  (`16` and `31` trace events), `responses_delta_streaming`,
  `responses_cache_detail_usage`, `long_tool_loop`, and
  `tool_l2_cache_integrated`.
- Runtime proof in the same app run: `weight_format=mxfp4`, `profile=MXFP4`,
  `weight_matmul_dispatch=mlx_affine_quantized_matmul`,
  `metal_na_active_on_host=true`, `passthrough_tensor_count=349`, active memory
  about `7773.6 MB`, and peak memory about `10518 MB`.
- Cache proof in the same app run: Gemma mixed-SWA cache,
  `cache_detail=paged+mixed_swa`, `cache_hit_tokens=3538`,
  final `cached_tokens=2688`, `l2_block_tokens_on_disk=3588`,
  block-disk `disk_hits=30`, and `disk_writes=58`.
- Real Electron dev-app Gemma 12B QAT MXFP4 image/VL proof is green. The app
  persisted an image attachment, server `MEDIA_DIAG` observed one `image_url`,
  the Gemma media fallback ran with `1 image(s)`, and the assistant answered
  `Red` for the red-image semantic probe.
- The MXFP4 image proof also showed MXFP4 affine matmul with Metal NA active,
  mixed-SWA cache, `cache_detail=paged+mixed_swa`, `cached_tokens=20`,
  `l2_block_tokens_on_disk=64`, and block-disk `disk_writes=2`.
- Real Electron dev-app Gemma 12B QAT MXFP4 video/VL proof is green for the
  same 1-second 64x64 solid-red MP4 fixture used by the N2 proof. The app
  persisted a `video_url` attachment, server `MEDIA_DIAG` observed one
  `video_url`, the server decoded the base64 MP4, reported
  `25 total frames @ 25.0 fps`, extracted `4 frames`, and the assistant
  answered `The video shows a solid red screen.`
- The MXFP4 video proof also showed MXFP4 affine matmul with Metal NA active,
  mixed-SWA cache, `cache_detail=paged+mixed_swa`, `cached_tokens=20`,
  `l2_block_tokens_on_disk=65`, and block-disk `disk_writes=2`.
- Real Electron dev-app Gemma 12B QAT MXFP4 audio proof is red by explicit
  runtime guard, not by crash. The app attempted an audio turn, server
  `MEDIA_DIAG` saw `input_audio`, and the API returned `400`:
  `/v1/chat/completions received unsupported media modality audio. Supported
  modalities: text, vision, video.`
- Local rebuilt installed app proof is now green for Gemma 12B QAT MXFP4
  Responses/tool/cache. `/Applications/vMLX.app` launched as
  `uiLaunchMode=installed-app`, adopted the real server for
  `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4`, used
  `/v1/responses`, executed two built-in `run_command` calls, and produced
  visible assistant turns containing `REAL_UI_LIVE_TOOL_ONE` and
  `REAL_UI_LIVE_TOOL_TWO`.
- The installed-app Gemma MXFP4 run verified the probe files exactly:
  `real_ui_tool_probe_1.txt=REAL_UI_LIVE_TOOL_ONE\n` and
  `real_ui_tool_probe_2.txt=REAL_UI_LIVE_TOOL_TWO\n`.
- The installed-app run recorded renderer content deltas (`count=16` and
  `count=24`), `eventCounts.tool=156`, `eventCounts.stream=40`,
  `eventCounts.complete=2`, server cache controls, settings persistence, and
  no raw parser/reasoning leak.
- Installed-app Gemma MXFP4 runtime/cache evidence: `weight_format=mxfp4`,
  `profile=MXFP4`, `weight_matmul_dispatch=mlx_affine_quantized_matmul`,
  `metal_na_active_on_host=true`, active memory about `7772.4 MB`, peak memory
  about `10512.9 MB`, native cache `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`,
  `cache_hit_tokens=3538`, `l2_block_tokens_on_disk=3584`,
  `l2_tokens_on_disk=3584`, block-disk `disk_hits=30`, and
  `disk_writes=58`.
- Local rebuilt installed app image/VL proof is also green for Gemma 12B QAT
  MXFP4. The app persisted the image attachment, the server `MEDIA_DIAG` saw
  one `image_url`, the Gemma media fallback ran with `1 image(s)`, and the
  assistant answered `Red` for the red-image semantic probe.
- The installed-app image run recorded `vl_image`, `installed_app_ui`,
  `server_cache_controls`, content deltas (`count=21`, `12`, and `1`), no raw
  parser/reasoning leak, `weight_format=mxfp4`, Metal NA active, native
  `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cached_tokens=20`,
  `l2_block_tokens_on_disk=70`, `l2_tokens_on_disk=70`, and `disk_writes=2`.
- Local rebuilt installed app video/VL proof is now green for the same Gemma
  12B QAT MXFP4 row. The app persisted a video attachment, server `MEDIA_DIAG`
  saw one `video_url`, the server decoded the base64 MP4, reported `25 total
  frames @ 25.0 fps`, extracted `4 frames`, routed those frames through the
  Gemma media fallback, and the assistant answered `The video shows a solid red
  background with no movement or changes.`
- The installed-app video run recorded `video_where_supported`,
  `installed_app_ui`, `server_cache_controls`, content deltas (`count=21`,
  `12`, and `1`), no raw parser/reasoning leak, `weight_format=mxfp4`, Metal NA
  active, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`,
  `cached_tokens=20`, `l2_block_tokens_on_disk=70`, `l2_tokens_on_disk=70`, and
  `disk_writes=2`.
- Local rebuilt installed app audio proof is red by explicit runtime guard, not
  by crash or cache failure. The installed app launched, completed two visible
  text turns, forced multimodal for one attached audio file, server
  `MEDIA_DIAG` saw one `input_audio`, and `/v1/chat/completions` returned
  `400`: `/v1/chat/completions received unsupported media modality audio.
  Supported modalities: text, vision, video.`
- The installed-app audio boundary run still recorded the Gemma runtime/cache
  surfaces before the failing audio turn: active memory about `7562.3 MB`, peak
  about `7872.4 MB`, `weight_format=mxfp4`, Metal NA active, native
  `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cached_tokens=20`,
  `l2_block_tokens_on_disk=70`, `l2_tokens_on_disk=70`, and block-disk
  `disk_writes=2`. Generic TurboQuant KV remained correctly disabled for this
  mixed-SWA family.

Not proven:

- Installed-app audio support for the MXFP4 row. Installed-app image and video
  are green; installed-app audio is classified red by honest unsupported
  modality guard.
- Audio weight-backed E2E for the MXFP4 dev-app row is red; current source
  honestly rejects audio with supported modalities `text, vision, video`.
- Full larger Gemma QAT matrix through UI/installed app.
- Tunnel/gateway parity for these exact Gemma rows.
- Gemma 12B QAT MXFP4 dev-app audio support. Image and video are green in the
  source dev app; audio is honestly gated as unsupported.
- The second MXFP4 visible answer begins with the plain word `thought`; this is
  not a raw `<think>` or parser markup leak and the leak gates passed, but do
  not hide this visible-final style caveat.

### Gemma 4 12B JANG4M Real Dev-App Proof

Artifact:

- `build/current-real-ui-live-model-gemma4-12b-jang4m-dev-app-proof-20260610.json`
- `build/current-real-ui-live-model-gemma4-12b-jang4m-video-proof-20260610.json`
- `build/current-real-ui-live-model-gemma4-12b-jang4m-audio-proof-20260610.json`

Raw ignored proof captures:

- `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-responses-tools-cache-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-image-cache-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-video-cache-max12k-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-audio-cache-20260610-proof.json`

Proven:

- Real Electron dev app launched with `npm run dev`, connected to a real vMLX
  server loading `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`.
- Responses UI rail passed with built-in `run_command` tools enabled.
- Two-turn tool loop executed real commands, wrote/read
  `REAL_UI_LIVE_TOOL_ONE`, wrote `REAL_UI_LIVE_TOOL_TWO`, and persisted tool
  execution/result phases in the chat UI state.
- Responses stream trace carried visible content deltas and server metrics.
  The two turns ended with `cachedTokens=2687` and `cachedTokens=2901`,
  `cacheDetail=paged+mixed_swa`, and live decode around `45 t/s`.
- Chat Completions UI rail passed with image attachment enabled.
- The attached red image reached the Gemma MLLM media path and returned visible
  `Red`; `media.imageSemanticVerified=true`.
- Server/cache controls were visible in the app session and matched the
  expected cache labels.
- Gemma native cache surfaced as mixed-SWA with prefix cache, paged cache, and
  block-disk L2 writes.
- No raw parser/tool markup leak and no hidden reasoning leak were observed in
  either dev-app run.
- Real Electron dev-app video pass is now green for a 1-second solid-red MP4
  when the session context cap is raised to `12000` prompt tokens. The app
  persisted a `video_url` attachment, the server decoded the base64 MP4,
  extracted frames, routed it through the MLLM media fallback, and the assistant
  answered `solid, dark red screen...`.
- The same video path with the default `4096` prompt cap failed honestly with
  `HTTP 413 prompt_too_long`, reporting about `8315` prompt tokens. Therefore
  Gemma 12B JANG4M video support is proven only with an adequately large context
  cap for this fixture, not with the 4k cap.
- The video pass also showed mixed-SWA cache/L2 surfaces:
  `cache_detail=paged+mixed_swa`, `cached_tokens=20`,
  `l2_block_tokens_on_disk=84`, and `disk_writes=2`.
- Real Electron dev-app audio attachment plumbing is now classified, but red:
  the app persisted an `input_audio` part, the server decoded the base64 WAV,
  and visible assistant output streamed, but the model did not transcribe the
  generated phrase `audio present`. The failed audio proof still shows cache
  and launch surfaces green: `cache_detail=paged+mixed_swa`,
  `cacheHitTokens=67`, `l2_tokens_on_disk=67`, `disk_writes=2`, and server
  cache controls verified.

Not proven:

- Installed packaged app parity.
- DMG package/sign/notarize/release readiness.
- Local panel session manager starting this exact model from launch args; these
  app proofs used a remote session connected to the server started by the proof
  harness.
- Gemma audio semantic E2E. The current dev-app proof reaches runtime but
  fails semantic verification, so do not claim audio support for this row.
- N2 or MiMo dev-app chat proof.
- Same-model public tunnel raw SSE parity.

### MiMo V2.5 JANGTQ_2

Artifacts:

- `build/current-mimo-v25-jangtq2-live-cb-cache-text-20260610.json`
- `build/current-mimo-v25-jangtq2-exactness-variant-probe-live-20260610/result.json`

Proven:

- JANGTQ_2 bundle loaded and served on the 128GB host.
- Model footprint is about `79.45 GiB` on disk; final health showed about
  `76.4GB` active memory in the earlier cache/text row.
- Native MiMo mixed full/sliding attention cache surfaced as
  `mixed_swa_kv_v1`, subtype `mimo_v2_asymmetric_swa`.
- Prefix cache, paged cache, q8 storage-boundary KV quantization, and
  block-disk L2 were active.
- Live cache proof observed paged cache hit and L2 block writes.
- Exact HTTP requests completed without hidden think tags.

Red:

- Exact output fidelity is not release-green. The same live server mutated:
  `blue-cat -> blue`, `B7-CAT-09 -> B7CAT-09` / `B7 CAT-09`, and tool args
  likewise lost characters.
- Short cache/text row also returned `ACKCB-742` instead of `ACK-CB-742`.

Next implementation target:

- Do not chase cache/parser/JSON repair for this red row. The current evidence
  says MiMo JANGTQ_2 cache plumbing works, but decode/artifact/logit exactness
  is wrong. Investigate artifact quant contract, codebook/decode path, or
  runtime token/logit contract.

### MiMo V2.5 JANG_2L

Artifact:

- `build/current-mimo-v25-jang2l-live-cb-cache-text-20260610.json`
- `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-proof-20260610.json`
- `build/current-mimo-v25-jang2l-chat-tool-boundary-20260610.json`
- `build/current-mimo-v25-jang2l-chat-tool-boundary-fulltools-20260610.json`
- `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-after-toolchoice-proof-20260610.json`
- `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-followup-proof-20260610.json`
- `build/current-real-ui-live-model-mimo-v25-jang2l-image-proof-20260610.json`
- `build/current-real-ui-live-model-mimo-v25-jang2l-responses-tools-proof-20260610.json`

Proven:

- 105 GiB local JANG_2L bundle loaded on 128GB host.
- Final health showed about `104997.8 MB` active and `105956.2 MB` peak.
- Uses `mlx_affine_quantized_matmul`; Metal NA active on host.
- Native MiMo mixed full/sliding cache surfaced as `mixed_swa_kv_v1`, subtype
  `mimo_v2_asymmetric_swa`.
- Prefix cache, paged cache, q8 storage-boundary KV quantization, and
  block-disk L2 were active.
- Paged cache hit observed: `cached_tokens=38`, `cache_detail=paged`.
- L2 block writes observed: `l2_tokens_on_disk=62`.
- Exact short output preserved: both repeats returned `ACK-CB-742`; first-token
  system probe returned `OK`.
- Direct source Chat tool boundary with only `run_command` passed for both
  `auto` and specific/required tool modes, with paged cache and L2 positive.
- Direct source Chat boundary with the full panel tool surface proved the
  failure shape: full auto mode is not green (`HTTP 413` nonstream and stream
  `prompt_too_long`, `4754` tokenized prompt tokens against max `4096`), while
  specific/required `run_command` still produced a tool call.
- Panel request assembly now sends a specific `tool_choice` only when the latest
  user message explicitly names exactly one available built-in tool. This is
  covered for both Chat Completions and Responses request bodies, and multiple
  named tools/no explicit tool names remain unpinned.
- Panel Chat Completions tool-result follow-ups now suppress the original
  explicit single-tool `tool_choice` after a tool call has executed. The
  follow-up proof no longer shows the earlier
  `tool_choice='required' was set but the model did not produce any tool calls`
  final-answer error.
- Post-fix real Electron dev-app MiMo attempts executed `run_command` tool calls
  and streamed visible content with cache/L2 telemetry. The best post-fix app
  run recorded `eventCounts.tool=90`, `eventCounts.stream=144`,
  `eventCounts.complete=2`, `cached_tokens=2344`, `cache_detail=paged`,
  `l2_block_tokens_on_disk=1074`, and `disk_writes=18`.
- The follow-up proof executed tool calls in the real dev app with
  `persistedToolCount=135`, `eventCounts.stream=239`,
  `eventCounts.complete=2`, `cacheHitTokens=8072`, verified server cache
  controls, and `l2_block_tokens_on_disk=4580`.
- Real Electron dev-app MiMo JANG_2L image/VL proof is now classified red by
  an explicit runtime text-only guard. The run requested `--is-mllm`, loaded the
  105 GiB artifact, completed text turns, and then the server `MEDIA_DIAG`
  observed one `image_url`, but `/v1/chat/completions` returned `400`:
  `received unsupported media modality image because the loaded runtime is
  text-only. Supported modalities: text.`
- The same image run proved MiMo JANG_2L runtime/cache surfaces in-app:
  `model_type=llm`, `JANG_2L_322_D3E16`,
  `mlx_affine_quantized_matmul`, Metal NA active, `mixed_swa_kv_v1`,
  `mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cached_tokens=39`,
  `l2_block_tokens_on_disk=110`, and `l2_tokens_on_disk=110`.
- The server log explains the boundary: MiMo V2 preserved media weights
  override forced MLLM mode because bundle metadata marks vision/audio as
  `unwired weights_preserved_text_runtime`; the runtime routes this artifact
  text-only.
- Real Electron dev-app MiMo JANG_2L Responses tool proof is now classified.
  The app used `/v1/responses`, the first turn emitted a built-in
  `run_command` tool call, the app sent a scoped tool-result follow-up with
  `previous_response_id=resp_7aed9d6ca76f`, and the first tool loop completed.
- The same Responses run showed real MiMo runtime/cache pressure rather than a
  shallow API check: first response took `424` tokens in `289.0s`, live decode
  was about `1.2 t/s`, peak memory was `109374.2 MB`, paged cache hit tokens
  reached `1071`, last cache execution used `cached_tokens=687`,
  `l2_block_tokens_on_disk=3784`, block-disk `disk_hits=18`, and
  `disk_writes=60`.

Red / not proven:

- Robust required/auto tool exactness in the dev-app row. The post-fix app run
  did call `run_command`, but MiMo mutated requested semantics:
  `REAL_UI_LIVE_TOOL_ONE` became `REAL_UI_LAND_TOOL_ONE`, and requested
  working-directory files became `/tmp/real_ui_land_tool_one.txt` /
  `/tmp/real_ui_land_tool_two.txt`; the expected probe files were not created.
- A stricter prompt using explicit `printf` / `cat` shell commands and a fixed
  working directory also failed: first assistant content was only
  `The user wants me to run run a specific command with the exact argument:`
  and the second assistant content was empty.
- The `long_tool_loop` release surface remains red for MiMo JANG_2L.
- The latest follow-up proof still fails `long_tool_loop`: MiMo mutated
  `REAL_UI_LIVE_TOOL_ONE` / `REAL_UI_LIVE_TOOL_TWO` into
  `REAL_UI_LAND_TOOL_ONE` / `REAL_UI_LAND_TOOL_TWO` and wrote `/tmp` files
  instead of the configured working-directory probe files.
- Responses stream/nonstream tool path for JANG_2L remains red for release. A
  real dev-app Responses run proved first-turn tool call plus
  `previous_response_id` follow-up, but the two-turn loop failed with
  `CDP timeout: Runtime.evaluate` while the second turn was still active; no
  final tool probe file semantics were verified.
- Fresh-process L2 restore for JANG_2L.
- VL/audio/video runtime. Image/VL is now explicitly red by a text-only runtime
  guard even when forced MLLM is requested; do not claim MiMo JANG_2L media
  support from preserved media weights.
- Long context usability beyond the short cache proof.

Next implementation target:

- Keep the panel `tool_choice` fix; it reduces the app/full-tool-surface
  failure without faking a tool call. Do not generalize it beyond explicit
  single-tool user requests.
- Next MiMo work is model/artifact/decode/tool-argument exactness. Cache/L2 is
  not the current blocker: post-fix app attempts have paged cache hits and
  block-disk L2 writes.
- Continue JANG_2L into Responses and fresh-process L2 restore. Media honesty is
  now classified for image/VL: the current artifact is text-only at runtime
  despite preserved media weights.

### Nex/N2 Pro JANGTQ2

Artifact:

- `build/current-n2-jangtq2-live-chat-cache-responses-l2-20260610.json`

Proven:

- 101 GiB local JANGTQ2 bundle loaded and served on 128GB host.
- Final health showed about `104202.2 MB` active and `105212.9 MB` peak.
- Native cache is `hybrid_ssm_v1` with components:
  `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Live attention TurboQuant KV is enabled only for attention KV layers;
  SSM companion state remains native.
- Tight-memory allocator drains occurred during prefill/decode.
- Chat cache proof passed with stable text `ACK`.
- Chat cache hit returned `cached_tokens=8`, `cache_detail=paged+ssm`.
- Required chat tool passed with args `{"query": "alpha"}`.
- Responses nonstream required tool passed.
- Responses tool-result follow-up with `previous_response_id` passed and
  returned `DONE`.
- Responses streaming required tool passed with args present across surfaces.
- Fresh-process L2 restart restore passed:
  `cache_detail=paged+ssm+disk`, block disk `disk_hits=1`, SSM companion disk
  `hits=1`.
- Final L2 totals: `l2_block_tokens_on_disk=271`,
  `l2_ssm_tokens_on_disk=271`, store sum `542`.

Not proven:

- Installed-app/UI path for this exact 20260610 proof.
- Dev-app image/video for this exact profile are proven in the app section
  below; audio remains red by explicit unsupported-modality guard.
- Same-model direct/gateway/tunnel public parity.
- JANG_1L profile.

Next implementation target:

- Treat JANGTQ2 as the N2 checkpoint candidate. It is the profile with real
  live 128GB cache/API/tool/L2 proof.

### Nex/N2 Pro JANGTQ2 Real Dev-App Proof

Artifact:

- `build/current-real-ui-live-model-n2-jangtq2-dev-app-proof-20260610.json`
- `build/current-n2-jangtq2-responses-stream-boundary-20260610.json`
- `build/current-real-ui-live-model-n2-jangtq2-dev-app-delta-proof-20260610.json`
- `build/current-real-ui-live-model-n2-jangtq2-dev-app-prevresp-proof-20260610.json`
- `build/current-real-ui-live-model-n2-jangtq2-image-proof-20260610.json`
- `build/current-real-ui-live-model-n2-jangtq2-video-proof-20260610.json`
- `build/current-real-ui-live-model-n2-jangtq2-audio-proof-20260610.json`
- `build/current-real-ui-installed-app-n2-jangtq2-responses-tools-cache-20260610.json`
- `build/current-real-ui-installed-app-n2-jangtq2-image-proof-20260610.json`
- `build/current-real-ui-installed-app-n2-jangtq2-video-proof-20260610.json`
- `build/current-real-ui-installed-app-n2-jangtq2-audio-proof-20260610.json`

Raw ignored proof captures:

- `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-cache-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-cache-longdelta-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-prevresp-default-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-prevresp-longdelta-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-audio-20260610-proof.json`

Proven:

- Real Electron dev app launched with `npm run dev`, connected to a real vMLX
  server loading `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`.
- The 101 GiB N2 JANGTQ2 profile loaded in the app proof harness; final health
  showed about `103807.6 MB` active and `108294.9 MB` peak in the longer
  attempt.
- Real Electron dev-app image/VL proof is green. The app persisted an image
  attachment, the server classified the model as `mllm`, `MEDIA_DIAG` observed
  one `image_url` content part, the runtime processed `num_images_processed=1`,
  and the assistant answered `Red` for the red-image semantic probe.
- The N2 image proof showed hybrid SSM/TurboQuant/L2 state in the same run:
  `cache_detail=paged+ssm`, `cached_tokens=18`,
  `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`,
  `l2_tokens_on_disk=118`, block-disk `disk_hits=3`, and SSM companion stores
  `2`.
- For the media prompt itself, the server intentionally skipped prefix/paged
  cache store because media embeddings are path-dependent; this is the honest
  media cache boundary, not a cache failure.
- Real Electron dev-app video/VL proof is also green for a 1-second 64x64
  solid-red MP4. The app persisted a `video_url` attachment, the server decoded
  the base64 MP4, reported `25 total frames @ 25.0 fps`, extracted `4 frames`,
  processed `num_images_processed=4`, and the assistant answered
  `The video shows a solid red screen with no visible movement or change.`
- The N2 video proof showed the same hybrid cache/L2 shape as the image proof:
  `cache_detail=paged+ssm`, `cached_tokens=18`,
  `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`,
  `l2_tokens_on_disk=118`, block-disk `disk_hits=3`, and SSM companion stores
  `2`.
- Real Electron dev-app audio proof is red by explicit runtime guard, not by
  crash. The app attempted an audio turn, server `MEDIA_DIAG` saw
  `input_audio`, and the API returned `400`:
  `/v1/chat/completions received unsupported media modality audio. Supported
  modalities: text, vision, video.`
- Responses UI rail reached `/v1/responses` and completed two turns.
- Built-in `run_command` tool loop executed and wrote/read the expected probe
  files: `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`.
- Native cache surfaced as `hybrid_ssm_v1` / `hybrid_ssm_typed` with
  `attention_kv`, `ssm_companion_state`, and `async_rederive`.
- TurboQuant KV was active for attention KV layers only; SSM companion state
  stayed native.
- App/cache endpoints showed `paged+ssm` cache hits, block-disk L2 writes, and
  SSM companion disk stores. The longer attempt ended with
  `l2_block_tokens_on_disk=4626`, `l2_ssm_tokens_on_disk=23454`, and
  `l2_tokens_on_disk=28080`.
- Server cache controls were visible and verified in the dev app.
- No raw parser/tool markup leak and no hidden reasoning leak were observed.
- Separate real Electron dev-app Responses delta pass, without built-in tools,
  cleared the app renderer/content-delta surface. It loaded the same N2 JANGTQ2
  row, used `/v1/responses`, and produced two assistant messages with
  multi-event visible deltas:
  - first trace: `count=21`, `N` -> `N2_APP_DELTA_ONE is ready...`
  - second trace: `count=24`, `N` -> `N2_APP_DELTA_TWO is ready...`
- That delta pass recorded `responses_delta_streaming`,
  `responses_cache_detail_usage`, `cache_hit_telemetry`,
  `native_cache_status`, `l2_disk_storage`, and `server_cache_controls` in the
  proof surfaces. Runtime/cache evidence included `cache_detail=paged+ssm`,
  `cached_tokens=45`, `l2_block_tokens_on_disk=120`,
  `l2_ssm_tokens_on_disk=274`, and `l2_tokens_on_disk=394`.
- Current panel source now sends Responses in-turn tool-result follow-ups as
  scoped `function_call_output` input with `previous_response_id`, instead of
  replaying the whole input and re-applying the original explicit tool choice.
  The app proof logs this twice:
  `Responses tool follow-up using previous_response_id=... with 1 function_call_output item(s)`.
- After that fix, the default real Electron dev-app N2 JANGTQ2 Responses
  tool/cache/delta row is green:
  `build/current-real-ui-live-model-n2-jangtq2-dev-app-prevresp-proof-20260610.json`
  has `status=pass`.
- The green combined app proof covers built-in `run_command`, tool-result
  continuation, visible markers, renderer deltas, cache/L2, and settings:
  assistant messages were `Done. REAL_UI_LIVE_TOOL_ONE` and
  `Done - this is the second UI turn. REAL_UI_LIVE_TOOL_TWO`; the tool probe
  files contained `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`.
- The same green proof recorded two multi-delta assistant traces (`count=8`
  and `count=15`), `responses_delta_streaming`,
  `responses_cache_detail_usage`, `long_tool_loop`, and
  `tool_l2_cache_integrated`.
- Runtime/cache evidence in the green combined app proof:
  `hybrid_ssm_v1`, attention-only TurboQuant KV, native SSM companion state,
  `cache_detail=paged+ssm`, `l2_block_tokens_on_disk=3579`,
  `l2_ssm_tokens_on_disk=17083`, `l2_tokens_on_disk=20662`,
  `block_disk_writes=59`, `block_disk_hits=110`, and `ssm_disk_hits=1`.
- Local rebuilt installed app proof is now green for the same default N2
  JANGTQ2 checkpoint path. `/Applications/vMLX.app` launched as
  `uiLaunchMode=installed-app`, adopted the real N2 server, used
  `/v1/responses`, executed two built-in `run_command` calls, and produced
  visible assistant turns `Done. REAL_UI_LIVE_TOOL_ONE` and
  `Done - this is the second UI turn. REAL_UI_LIVE_TOOL_TWO`.
- The installed-app run verified the probe files exactly:
  `real_ui_tool_probe_1.txt=REAL_UI_LIVE_TOOL_ONE` and
  `real_ui_tool_probe_2.txt=REAL_UI_LIVE_TOOL_TWO`.
- The installed-app run also recorded renderer content deltas (`count=8` and
  `count=15`), `eventCounts.tool=106`, `eventCounts.stream=23`,
  `eventCounts.complete=2`, server cache controls, settings persistence, and
  no raw parser/reasoning leak.
- Installed-app runtime/cache evidence stayed architecture-correct:
  `hybrid_ssm_v1`, components `attention_kv`, `ssm_companion_state`,
  `async_rederive`, attention-only TurboQuant KV storage boundary, native SSM
  companion state, `cache_detail=paged+ssm`, `cached_tokens=384`,
  `l2_block_tokens_on_disk=3582`, `l2_ssm_tokens_on_disk=17086`,
  `l2_tokens_on_disk=20668`, `block_disk_writes=59`, `block_disk_hits=110`,
  and `ssm_disk_hits=1`.
- The installed-app proof also records the honest N2 MTP boundary:
  `mtp_status=metadata_inconsistent`, `runtime_active=false`, because
  `jang_config.drop_mtp=true` while config declares one MTP layer.
- Local rebuilt installed app image/VL proof is green for N2 JANGTQ2. The app
  persisted the image attachment, server `MEDIA_DIAG` saw one `image_url`, the
  runtime processed `num_images_processed=1`, and the assistant answered `Red`.
- The installed-app image run recorded `vl_image`, `installed_app_ui`,
  `server_cache_controls`, no raw parser/reasoning leak, `hybrid_ssm_v1`,
  attention-only TurboQuant KV, `cache_detail=paged+ssm`, `cached_tokens=18`,
  `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`,
  `l2_tokens_on_disk=118`, block-disk `disk_hits=3`, `disk_writes=2`, and SSM
  companion disk stores `2`.
- Local rebuilt installed app video/VL proof is also green for N2 JANGTQ2. The
  app persisted the video attachment, server `MEDIA_DIAG` saw one `video_url`,
  the server decoded the base64 MP4, extracted `4` frames from a 25 fps
  one-second clip, processed `num_images_processed=4`, and the assistant
  answered `The video shows a solid red screen with no visible movement or
  change.`
- The installed-app video run recorded `video_where_supported`,
  `installed_app_ui`, `server_cache_controls`, no raw parser/reasoning leak,
  `hybrid_ssm_v1`, attention-only TurboQuant KV, `cache_detail=paged+ssm`,
  `cached_tokens=18`, `l2_block_tokens_on_disk=50`,
  `l2_ssm_tokens_on_disk=68`, `l2_tokens_on_disk=118`, block-disk
  `disk_hits=3`, `disk_writes=2`, and SSM companion disk stores `2`.
- Local rebuilt installed app audio proof is red by the same honest
  unsupported-modality guard as the source dev app, not by crash or cache
  failure. The installed app launched, completed two visible text turns, forced
  multimodal for one attached audio file, server `MEDIA_DIAG` saw one
  `input_audio`, and `/v1/chat/completions` returned `400`:
  `/v1/chat/completions received unsupported media modality audio. Supported
  modalities: text, vision, video.`
- The installed-app audio boundary run still recorded the N2 runtime/cache
  surfaces before the failing audio turn: active memory about `103805 MB`, peak
  about `104453.9 MB`, `hybrid_ssm_v1`, attention-only TurboQuant KV,
  `cache_detail=paged+ssm`, `cached_tokens=18`,
  `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`,
  `l2_tokens_on_disk=118`, block-disk `disk_hits=3`, `disk_writes=2`, and SSM
  companion disk stores `2`.

Red:

- A stricter custom long-delta prompt remains red:
  `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-prevresp-longdelta-20260610-proof.json`
  failed because the second turn did not create `real_ui_tool_probe_2.txt` and
  visible output degenerated into repeated `!` after a tool-choice-required
  error. Do not generalize the default checkpoint pass to that stricter prompt.
- Audio, public tunnel SSE parity, N2 JANG_1L, stricter custom long-delta prompt
  quality, and release readiness remain open. Image and video are green in both
  the source dev app and rebuilt installed app; audio is honestly gated as
  unsupported. Installed-app Responses/tool/cache and image/video media parity
  are green for the default checkpoint N2 JANGTQ2 path only; this is not a
  Developer ID notarized public DMG release claim.

Next implementation target:

- Raw direct server and panel gateway SSE are now proven green for N2 Responses
  tool plus tool-result continuation content deltas:
  `build/current-n2-jangtq2-responses-stream-boundary-20260610.json` is
  `status=pass`. Direct follow-up produced `16` output-text deltas and gateway
  follow-up produced `14` output-text deltas. Both completed on the same served
  model and returned `N2_DIRECT_DELTA_ONE` / `N2_DIRECT_DELTA_TWO`.
- N2 JANGTQ2 now has a single green default dev-app proof for built-in tool
  loop, Responses tool-result continuation, content-delta streaming,
  hybrid-SSM/TurboQuant KV cache, and L2. Remaining N2 rows are installed-app
  parity, audio, public tunnel parity, N2 JANG_1L memory strategy, and the
  stricter custom prompt quality red row.

## Red Live Attempts

### Nex/N2 Pro JANG_1L

Artifact:

- `build/current-n2-jang1l-live-chat-cache-responses-20260610.json`
- server log: `build/current-n2-jang1l-live-chat-cache-responses-20260610.server.log`

Observed:

- 110.59 GiB local bundle exists.
- Run was launched with `--jang1l-required-extra-headroom-gib 1`, so this was
  not a default preflight skip.
- Server startup began, model detection selected `qwen3_5_moe`, qwen parser,
  qwen3 reasoning parser, hybrid cache, and JANG text-only route.
- Quant shape inference patched 482 modules because config said uniform
  bits=2/group_size=128 while safetensors shapes required per-module overrides.
- Loader attempted 123 safetensors shards, bfloat16 for 512 experts, hidden
  4096.
- Loader set wired limit to the Metal cap: `Wired limit set to 115 GB (model
  119 GB)`.
- Process aborted before health with:
  `[METAL] Command buffer execution failed: Insufficient Memory`.

Conclusion:

- JANG_1L is not proven usable on this 128GB host. It is not merely untested and
  not merely skipped by a conservative gate; a live startup attempt crashed in
  Metal OOM before health.

Next implementation target:

- Implement an actual 128GB runtime strategy for JANG_1L before claiming it:
  lower peak loader/eval pressure, deferred/chunked eval that does not require
  full model command-buffer residency, smaller prefill/eval staging, or a
  JANG_1L-specific memory path. Do not claim N2 JANG_1L support from JANGTQ2.

## What To Tell The Other Agent

- Prioritize checkpoint release around proven rows: Gemma 12B MXFP4/JANG4M,
  MiMo JANG_2L short cache/text, and N2 JANGTQ2 full chat/cache/Responses/L2.
- Do not spend time proving generic cache. For MiMo use `mixed_swa_kv_v1`; for
  N2 use `hybrid_ssm_v1` with attention TQ KV plus native SSM companion.
- MiMo JANGTQ2 is loaded/cached but exactness-red; do artifact/logit/decode
  diagnosis, not parser repair.
- MiMo JANG_2L is the stronger MiMo checkpoint candidate for load/cache/text,
  but post-fix app tool exactness is still red. The panel now pins
  `tool_choice` only for explicit single-tool user requests, and the app can
  execute MiMo-generated `run_command` calls, but MiMo still mutates filenames
  and sentinel text. Other agent should work artifact/logit/decode/tool-arg
  exactness before claiming app tool support.
- N2 JANGTQ2 is the stronger N2 checkpoint candidate; it has live hybrid
  SSM/TQ/L2/tool/Responses proof.
- N2 JANG_1L needs a real memory-strategy fix. The current failure is a Metal
  OOM during loader/eval, not lack of attempt.
- Other agent should keep the new panel detector boundary: MiMo auto mode is
  XML tools + asymmetric-SWA paged cache, not auto reasoning; Gemma unified
  aliases must stay mapped to Gemma4 parsers and rotating mixed-SWA cache.
- Keep signed DMG release notes honest: say which profiles are checkpoint
  supported and which are experimental/red.
