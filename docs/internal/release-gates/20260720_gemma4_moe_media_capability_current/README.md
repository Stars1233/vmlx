# Gemma 4 26B-A4B MoE media capability and video-cache gate

Date: 2026-07-20

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

## Exact artifact classification

The tested artifact is
`/Volumes/EricsLLMDrive/dealignai/Gemma-4-26B-A4B-it-JANG_4M-CRACK`.
Its real sidecars declare a 30-layer Gemma 4 text graph with MoE enabled, 128
experts, top-8 routing, 25 rotating/sliding-attention layers, and five
full-attention layers. Weights are affine JANG_4M at 4.26 measured bits; this
is not JANGTQ/MXTQ or base MLX MXFP. The artifact has a vision configuration,
`audio_config=null`, no audio tower, no MTP tensors, and a processor-backed
video route.

The top-level config retains `audio_token_id=258881`. That reserved token is
not evidence of a runnable audio tower.

## Root cause and source repair

Before the repair, `/v1/capabilities` contradicted itself: artifact-declared
modalities omitted audio, but runtime modalities and `status_by_modality`
advertised audio as supported. `_bundle_declares_native_audio()` fell through
to a generic token-ID heuristic after finding no audio configuration.

Current source fails closed for `model_type=gemma4`: without a real audio
configuration/capability path, an audio token alone does not advertise audio.
Audio-capable Gemma 4 bundles with actual audio configuration/weights retain
their existing path. No prompt rewrite, output filter, synthetic modality, or
bundle mutation was added.

## Live Electron and API evidence

- Real `Launch Session` stopped Nemotron PID 24498 and eagerly loaded only
  Gemma PID 25329 on port 8004. Health was loaded with
  `last_request_time=null`; the exact argv selected Gemma tool/reasoning
  parsers, paged cache, and block-disk L2.
- The visible creation drawer matched the bundle's generation defaults:
  temperature 1.00, top-p 0.95, and top-k 64. Prefix, paged RAM, and block L2
  were On. The codec label correctly described TQ4 full-attention KV plus
  native rotating SWA.
- After real Stop/Start replaced PID 25329 with 25806, current capabilities
  were exactly `text, vision, video`; audio was `not_advertised`. A raw
  Responses WAV request returned HTTP 400 with the supported list and did not
  enter generation.
- Fresh Electron MP4 input produced 16 progressive UI states, separate
  457-character reasoning, exact `BANANA8426`, the exact requested marker,
  and no warning/tool state. TTFT was 1.40 seconds.
- Real Stop/Start replaced PID 25806 with 26100 and cleared L1 while retaining
  six 327-token blocks on L2. A fresh reattachment with the byte-identical
  prompt restored 327/328 tokens as
  `paged+mixed_swa+disk+tq-native`, recorded six disk promotions and six
  native-TQ hits, reduced TTFT to 0.33 seconds, and exact-finaled again.
- The owning log shows base64 MP4 decode, six source frames, and the source
  Gemma frame-fallback converting them to one real sampled image frame. Raw
  omitted-max Responses emitted 100 separate reasoning deltas, 18 content
  deltas, one text-done, and one completed event with exact code/marker.
- A same-chat, no-attachment restart follow-up was coherent and exact at 0.29
  seconds, but it had zero cache hits because ordinary Gemma history omitted
  the old media payload. It is retained as history/output evidence and is
  explicitly excluded from the L2 claim.
- A later same-chat **post-video automatic tool turn** enabled the built-in
  coding tools through the real Chat settings drawer and called
  `file_info(panel/package.json)` exactly once. The 16 captured DOM states show
  progressive separate reasoning, the tool card, `Processing tool results`,
  and progressive visible content before the exact final
  `G4MOE-POSTMEDIA-TOOL1-DONE SIZE=5.2 KB`. SQLite row 205 preserves one
  schema-valid call, one real `Size: 5.2 KB` result, non-empty visible content,
  no warning, 0.16-second TTFT, and 1.8-second total time. The owning
  `CHAT_DIAG` request reports zero current attachments and `has_tools=true`;
  historical video bytes were not resent or mistaken for the new text/tool
  request. This closes the exercised Electron post-media text/tool transition,
  not raw API post-media continuation for every Gemma variant.

## Evidence map

- Bundle/source truth: `bundle-config.json`, `bundle-generation_config.json`,
  `bundle-jang_config.json`.
- Pre/post-fix surfaces: `capabilities-before-request.json`,
  `capabilities-after-audio-capability-fix.json`, and
  `raw-responses-audio-rejection.json`.
- UI load/settings: `ui-create-session-derived-settings.png`,
  `ui-loaded-before-request.png`, and `health-before-request.json`.
- Fresh video: `ui-video-a1-dom-trace.json`, `ui-video-a1-row.json`, and paired
  screenshots.
- Restart controls: `health-after-video-restart-before-followup.json`,
  `ui-video-a2-l2-row.json`, and the corresponding DOM trace/screenshots.
- Valid L2 replay: `ui-video-a1-disk-row.json`,
  `health-after-ui-video-a1-disk.json`, `ui-video-a1-disk-dom-trace.json`, and
  `ui-log-after-video-a1-disk.png`.
- Raw API: `raw-responses-video-omitmax-auto.sse` and
  `ui-log-after-raw-video.png`.
- Post-media tool transition: `ui-chat-settings-before-tool.png`,
  `ui-postmedia-tool-dom-trace.json`, paired first-paint/final screenshots,
  `ui-postmedia-tool-row.json`, and `ui-postmedia-tool-request-shape.txt`.

## Validation and remaining scope

- `py_compile vmlx_engine/server.py`: passed.
- Fifteen focused Gemma audio/video/runtime-capability tests: passed.
- `git diff --check`: passed.

This closes one real 26B-A4B MoE artifact. It does not promote all Gemma 4
variants, native video, audio on this vision-only artifact, signed-app repeat,
bounded eviction, alternate-video salt isolation, or longer stochastic media
quality. The already-proven 12B unified dense audio row remains separate.
