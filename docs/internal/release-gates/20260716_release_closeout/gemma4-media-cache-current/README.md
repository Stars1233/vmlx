# Gemma 4 media-keyed mixed-SWA cache proof

Status: `PASS-LIVE` for the scoped Gemma 4 image/video media-cache path at
commit `cc1562a2b`; broader Gemma catalog, audio, long-output, and alternate
video-content quality remain `PARTIAL`.

## Source trace

- `mllm_batch_generator.py:4891-4928` enables the existing clean
  media-conditioned N-1 producer by default for config-derived `gemma4` and
  `gemma4_unified`, while preserving explicit `off`, bypass, canonical media
  side-key, and real media-placeholder requirements.
- `mllm_scheduler.py:1028-1072` applies the same policy at admission and keeps
  ZAYA CCA excluded from this Gemma-owned path.
- `mllm_scheduler.py:3096-3118` stores the captured media-conditioned 48-layer
  boundary directly for mixed-SWA layouts. It does not run the old deferred
  text-only clean prefill after pixel/video tensors have been released.
- The stored boundary contains native rotating-SWA state plus the compatible
  full-attention TQ4 slots already selected by Gemma's mixed cache layout.
  Block-disk telemetry reported `tq_native_enabled=true`; this change does not
  TQ-encode rotating-SWA state.

The exact excerpts are in `source-trace.txt`.

## Automated proof

At commit `cc1562a2b`:

```text
.venv/bin/python -m pytest -q tests/test_zaya_runtime.py tests/test_mllm_scheduler_cache.py
167 passed, 6 skipped, 2 third-party warnings in 4.43s
```

The new contracts cover default-on plus explicit-off policy and prove a Gemma
mixed-SWA store consumes the captured media boundary without calling the
text-only path-dependent helper.

## Electron image proof

The live model was
`/Volumes/EricsLLMDrive/jangq-ai/gemma-4-12B-it-qat-JANG_4M` on port 8009.
The visible Sessions start loaded PID 52537 before any request, and the tested
restart used PID 54193 with the same UI-derived argv: `--is-mllm`, paged cache,
64-token blocks, 1,000 blocks, block-disk L2, and Auto mixed-cache selection.

- Cold image A, row 174: exact `BANANA8426`, 305 prompt tokens, no cache hit.
- Identical fresh A, row 177: exact `BANANA8426`, 304/305 tokens restored as
  `paged+mixed_swa`, TTFT 0.26s versus 0.60s cold.
- Different same-shape image B, row 180: zero cached tokens and exact
  `B1-MEDIA-B-DONE`. No A marker leaked.
- Return A, row 183: exact `BANANA8426`, 304/305 `paged+mixed_swa` tokens,
  TTFT 0.33s.
- Visible Stop/Start removed the process and left zero RAM/indexed tokens while
  69 L2 blocks remained. Row 186 then returned exact `BANANA8426` with 304/305
  `paged+mixed_swa+disk` tokens. Health recorded five block-disk hits and five
  native-TQ hits with zero selected-prefix scheduler disk misses.

The Electron bubbles are captured in `image-a-cold.png`, `image-a-ram.png`,
`image-b-cold.png`, `image-a-return.png`, and `image-a-l2.png`.

## Electron video proof

`gemma4-media-a-video.mp4` is a real six-frame MP4 generated from the inspected
2800x1800 marker fixture. The attachment route classified it as `video/mp4`,
the Responses wire body contained `video_url`, the server decoded a temporary
MP4, and Gemma's fallback sampled a real frame before vision prefill.

- Cold video A, row 189: exact `BANANA8426`, 304 prompt tokens, no cache hit.
- Identical fresh video A, row 192: exact `BANANA8426`, 303/304 tokens restored
  as `paged+mixed_swa`, TTFT 0.31s versus 0.66s cold.
- Visible process restart left zero L1 tokens and 74 L2 blocks. Row 195 then
  returned exact `BANANA8426` with 303/304 `paged+mixed_swa+disk` tokens.

The server logged a 303-token clean media boundary before tensor release, use
of the captured 48-layer mixed-SWA state, and a media-side-key paged store.
Screenshots are `video-a-ready.png`, `video-a-cold.png`, `video-a-ram.png`, and
`video-a-l2.png`.

## Raw Responses streaming proof

A real `curl -N` request sent the MP4 as a base64 `video_url` to
`/v1/responses` with thinking enabled. The stream delivered 89 independently
timed `response.reasoning_summary_text.delta` events from 0.679s through
2.742s, then six `response.output_text.delta` events from 2.786s through
2.896s (`BAN`, `ANA`, `8`, `4`, `2`, `6`), and one `response.completed` at
2.918s. Final usage was 304 input, 106 output, 410 total tokens; final output
was exactly `BANANA8426`.

## Remaining gates

- Audio is advertised by this artifact but is not proven in this row.
- A second video with different visual content is still needed for the same
  cross-video isolation strength already shown by image A/B.
- Broader Gemma 4 artifacts and coherent constrained long output remain open.
- Full-suite, packaged-app, signing, notarization, and release gates remain
  open.
