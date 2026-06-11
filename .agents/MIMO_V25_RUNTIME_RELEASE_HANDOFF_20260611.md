# MiMo V2.5 Runtime Release Handoff - 2026-06-11

Current lane boundary:

- Do not clear MiMo release rows by parser repair, JSON repair, string
  post-processing, sampling clamps, or cache disabling.
- Do not claim source-vs-quant first divergence until the deliberate source and
  quant endpoints are up and the first-divergence probe runs.
- Do not use N2 JANG_1L work as MiMo evidence.

## Proven

JANGTQ_2 installed-app no-media Responses/tools/cache:

- Artifact:
  `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-responses-tools-cache-deterministic-printf-bundled-python-20260611-proof.json`
- Status: `pass`.
- Proves installed app UI, bundled Python, real loaded
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`, Responses API
  streaming, built-in auto tool loop, exact deterministic tool-result
  continuation, generation defaults, parser/language leak checks, settings
  persistence, server cache controls, native MiMo mixed-SWA cache status, cache
  endpoint stats, cache-hit telemetry, block-disk L2 storage, and tool/L2 cache
  integration.
- Runtime detection: `turboquant_codebook`, `JANGTQ_2`, 423 routed-expert
  TurboQuant targets, prestacked switch layout, `jangtq_runtime=true`, custom
  TurboQuant kernels.
- Cache proof: 3 cache-hit requests, 10463 hit tokens, 3732 L2 block tokens,
  no SSM companion because MiMo uses mixed-SWA KV not hybrid SSM.
- Memory proof: about 76.6GB active / 81.5GB peak.
- Speed evidence: live UI samples were 34.2 and 40.0 tok/s. This is near the
  current 40 tok/s floor but not enough to call every MiMo speed row green.

JANG_2L installed-app no-media Responses/tools/cache:

- Artifact:
  `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-responses-tools-cache-deterministic-printf-bundled-python-20260611-proof.json`
- Status: `pass`.
- Proves installed app UI, bundled Python, real loaded
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`, Responses
  streaming, built-in auto tool loop, deterministic tool-result continuation,
  native mixed-SWA cache, L2 storage, and cache-hit telemetry.
- Runtime detection says affine quantized matmul, Metal affine symbols
  available/active, and no TurboQuant runtime.
- Speed evidence remains bad: live UI samples were 1.6 and 1.7 tok/s with
  about 105GB active / 109GB peak. Do not promote JANG_2L as a fast release
  path without a deeper affine fusion/runtime fix.

MiMo media honesty:

- Current local JANG_2L/JANGTQ_2 bundles are `weights_preserved_text_runtime`.
- Panel detection has been fixed to respect the text-runtime MiMo capability
  stamp instead of treating `vision_config` alone as live multimodal support.
- Forced MLLM/media transport can reach parts of the runtime, but semantic
  media quality and release-capable image/video/audio are not proven.

Current-source JANGTQ_2 image/media classification, 2026-06-11:

- Default artifact:
  `docs/internal/agent-notes/current-real-ui-source-mimo-v25-jangtq2-image-icon-current-source-20260611-proof.json`
  is `status=fail`. It proves the default source/app path is still honestly
  text-only for this artifact: `modelForceTextOnly=true`,
  `chatIsMultimodal=false`, server `model_type=llm`, and image media stripped
  before the API body because config metadata says
  `weights_preserved_text_runtime`.
- Overlay artifact:
  `docs/internal/agent-notes/current-real-ui-source-mimo-v25-jangtq2-image-icon-media-overlay-20260611-proof.json`
  is `status=pass` only with
  `VMLINUX_MIMO_V2_ENABLE_TEXT_RUNTIME_MEDIA_OVERLAY=1`.
- Overlay proof details: current Electron dev UI, source Python server,
  `model_type=mllm`, image preserved in request body, server media diag saw one
  `image_url`, loader auto-enabled MiMo media from preserved sidecars,
  459 preserved media tensors assigned, one image processed,
  `vision_encoding_time=0.1617s`, `generation_tps=50.8`,
  `cache_hit_requests=1`, `cache_hit_tokens=34`, `ram_tokens_cached=88`,
  `l2_block_tokens_on_disk=88`, `disk_writes=2`, and `vl_image` surface green.
- Release meaning: the overlay proof shows current source can run the media
  bridge, but default release behavior remains text-only until the artifact is
  remade/promoted to an explicit multimodal runtime contract or the app/runtime
  intentionally sets the overlay env for this model family.

## Not Proven

- JANGTQ_2 exactness: current artifact evidence still shows literal mutation
  after valid parser structure, including `blue-cat -> blue-1/bluecat` and
  `B7-CAT-09 -> B7CAT-09/B7CCAT-09`.
- Source-vs-quant root cause: blocked until source endpoint
  `erics-m5-max2.local:8126` and local quant endpoint `127.0.0.1:8897` are
  actually listening.
- Media release quality: image/video/audio semantic E2E, post-media text
  recovery, media cache salt, fresh-process L2 restore, and installed-app
  parity are not green.
- JANG_2L speed: still release-red even though no-media tool/cache proof passes.
- Signed/notarized release readiness: still blocked by broader checklist rows.

## Current Diagnosis

The speed issue is split by artifact/runtime path:

- JANGTQ_2 uses the custom TurboQuant fused gate/up path and reaches about
  34-40 live tok/s in installed-app no-media proof.
- JANG_2L uses the affine MLX gather-qmm path. Even with the MiMo affine
  SwitchGLU fast path installed, installed-app proof remains around 1.6-1.7
  live tok/s. Health saying Metal affine is available is not sufficient release
  proof.
- The user-facing very low t/s reports are not explained by missing cache alone:
  JANG_2L is slow with cache and L2 present; JANGTQ_2 has much better decode
  under the same installed-app proof shape.

## Other-Agent Next Actions

1. Bring up the deliberate MiMo source endpoint and local quant endpoint, then
   run:
   `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py`
   without parser/JSON repair.
2. If source passes literals and JANGTQ_2 fails, inspect artifact quant/logit
   contract and rebuild/reupload the JANGTQ_2 artifact with a literal-safe
   profile.
3. If source also mutates literals, classify MiMo source/model behavior
   honestly and do not block on parser changes.
4. For speed, prioritize JANGTQ_2 for checkpoint users. Treat JANG_2L as
   functional but slow unless someone implements and proves a deeper affine
   fused gate/up/down path or another real runtime acceleration.
5. For media, keep default MiMo bundles text-only until live semantic
   image/video/audio E2E and media L2/restart proof are green. If another lane
   wants MiMo image in the checkpoint, it must either remake/promote metadata
   from `weights_preserved_text_runtime` to an explicit media runtime after
   E2E, or explicitly wire the overlay launch env and rerun source plus
   installed-app UI media proof; do not claim default media from source-overlay
   evidence alone.
