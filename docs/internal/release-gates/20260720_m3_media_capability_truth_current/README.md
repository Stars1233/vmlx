# MiniMax-M3 current media capability truth — 2026-07-20

Status: `VERIFIED-LIVE_SCOPED` for the current image UI and video Responses
rows; broader media catalog/quality and REAP remain `PARTIAL`.

Cutoff: `af0184caeddb85d83f0239ceb849b812d452f5ea` on
`codex/postrelease-ui-drawers-20260720`, exercised on
`erics-m5-max.local` through the real Electron dev app/CDP 9335 and direct
backend `127.0.0.1:8003`.

## Telemetry root cause

The prior current-head note treated `/health.mtp.vl_runtime_available=false`
as an M3 media-runtime failure. That field belongs to the MTP status object.
This artifact is deliberately not an MTP artifact: its name, JANG sidecar, and
tensor index do not declare MTP, so that status correctly says MTP is inactive.

The owning media surface is `/v1/capabilities`, backed by
`server.py::_m3_vl_media_ok()` and `_loaded_runtime_modalities()`. The current
loaded engine reports:

- runtime modalities: text, vision, video;
- 907 vision tensors present;
- no unwired vision/video modalities;
- native M3 MSA cache with dense KV layers 0-2 and sparse index-key layers
  3-59;
- generic TurboQuant disabled for the native sparse tuple.

See `m3-current-capabilities.json` and `m3-current-health-after-media.json`.

## Real Electron image proof

A fresh M3 chat used the real Chat Settings drawer with Thinking Off, tools
Off, temperature 0, Max Tokens 256, and Responses wire. Playwright attached
the committed 2800x1800 marker PNG through the real file input. The UI visibly
rendered the image and returned exact `MAGNOLIA CACHE DONE`.

The persisted assistant row has non-empty exact content, null reasoning,
8 output tokens, 744 prompt tokens, 3.04 s TTFT, no warning, and no tool state.
`m3-current-media-final.png` preserves the visible result.

## Raw Responses video proof

`m3-current-media-probe.py` sent the committed six-frame MP4 as an
`input_video` data URL with Thinking Off and temperature 0. The retained raw
events in `m3-current-media-probe.json` show:

- HTTP 200;
- exact `BANANA8426`;
- four progressive `response.output_text.delta` events;
- zero reasoning events under explicit Off;
- one `response.output_text.done` and one `response.completed`;
- 0.039767 s from the last content delta to completion;
- 1,682 input and five output tokens.

## Boundaries

- This corrects only the mistaken MTP-versus-media telemetry interpretation
  and refreshes representative image/video proof on the current head.
- Prior content-keyed RAM/L2 media isolation and larger-video proofs remain
  in `../20260716_release_closeout/mm3-media-cache-current/` and
  `../20260719_m3_terminal_dispatch_large_video/`; they were not redundantly
  rerun here.
- Exact OCR across a broader fixture catalog, stochastic Auto-thinking media,
  signed-app repetition, and REAP32 headroom remain `PARTIAL`/`OPEN`.
- No model output rewrite, OCR shim, hidden sampler clamp, or synthetic media
  behavior was added.
