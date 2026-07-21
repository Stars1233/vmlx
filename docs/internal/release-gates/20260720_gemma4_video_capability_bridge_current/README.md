# Gemma 4 native-video versus frame-fallback truth — 2026-07-20

Status: `PASS-SOURCE+LIVE_SCOPED` for capability truth, Electron/API media
transport, progressive output, and terminal completion. Blind marker reading is
`FAIL-QUALITY/PARTIAL-ROOT-CAUSE`; no native-video or grounding promotion is
made.

Branch: `codex/postrelease-ui-drawers-20260720`, based on
`60291760df21e59582979687fa55697077a01575` and committed with this gate. Live
host: `erics-m5-max.local`, real Electron dev app/CDP 9335, backend port 8141.

## Artifact and source truth

The tested official affine bundle was
`/Volumes/EricsLLMDrive/jangq-ai/gemma-4-12B-it-qat-JANG_4M`.

Its `config.json` and `jang_config.json` both explicitly declare:

- text, vision, and audio `true`;
- native video `false`;
- `video_token_id=258884` and a real `Gemma4UnifiedVideoProcessor` sidecar;
- hybrid rotating-SWA/full-attention language cache.

Those facts are compatible only when native artifact capability and the
runtime compatibility bridge are reported separately. Current source now:

- collects explicit modality booleans from both sidecars;
- makes an explicit `video=false` authoritative for
  `_bundle_declares_native_video()`;
- recognizes Gemma's already-implemented sampled-image-frame bridge in
  `_bundle_supports_video_frame_fallback()`;
- excludes bridge-only video from `media.declared_modalities` while keeping
  it in top-level/runtime modalities when the exact token, vision config, and
  video processor are present.

The data path is not synthetic output: `batched.py::_video_frame_fallback_messages`
decodes the actual MP4 and rewrites it to real sampled `image_url` parts before
Gemma vision prefill. No output text is inserted or corrected.

`/v1/capabilities` after the real Start-button load reported:

- runtime modalities `text, vision, audio, video`;
- declared modalities `text, vision/image, audio, multimodal` (no native
  video);
- video status `runtime_supported` through the source-traced bridge;
- mixed-SWA native cache, q4 storage-boundary KV, paged RAM, and block-disk L2;
- affine `JANG_4M`, not JANGTQ/MXTQ/MXFP.

## Real Electron load and settings

The model was selected from the real model dropdown and started with the real
`Start` button. The UI moved through `Loading...` to `Stop`; PID 20930 listened
on 8141 with the UI-derived argv containing `--is-mllm`, Gemma reasoning/tool
parsers, 64-token/1,000-block paged cache, and block-disk L2. No request was
needed to materialize the model.

The fresh Chat Settings drawer visibly matched `generation_config.json`:
Auto thinking, temperature 1.00, top-p 0.95, top-k 64, and repetition penalty
1.00. `g4-video-fallback-settings.png` preserves this visual parity. The
controlled media turns then explicitly set Thinking Off, temperature 0, and a
256-token cap; built-in tools remained Off.

## Streaming and quality evidence

The real six-frame MP4 has SHA-256
`842e3540854bfec12eda7701f9baa25fcd182ac3dff39088c758f095355111ed`.
Every inspected decoded frame visibly contains heading `FRESH IMAGE` and large
lower marker `BANANA8426`; see `g4-video-frame0.png` and
`g4-video-montage.png`.

### Fresh Electron blind turn

A truly fresh chat, with no earlier marker in its history, attached the MP4
through the real file input. It returned `FRANCMASSONIC`, not `BANANA8426`.
This is a real quality failure, not hidden as a pass.

The UI nevertheless satisfied the transport/emission contract:

- persisted row 175 has non-empty content, null reasoning, no warning, and no
  tool call;
- 1,122 prompt tokens, 11 output tokens, 1.45 s TTFT, 1.6 s total;
- the DOM mutation trace shows visible prefixes `F`, `FR`, `FRA`, ...,
  `FRANCMASSONIC`, followed by terminal metrics rather than one batched answer;
- `g4-fresh-video-blind-final.png` visibly shows the attached video and wrong
  final without a parser/control-token leak.

The earlier same-chat blind turn that returned `BANANA8426` is explicitly not
counted: the immediately preceding user prompt and assistant response already
contained the marker, so it was history-contaminated.

### Raw Responses A/B

Timed raw Responses with Thinking Off and an explicit 1,120 image-token budget
used the same blind prompt.

- Direct decoded PNG: `REMANAS66`, five progressive content deltas, zero
  reasoning events, one text-done, one completed terminal.
- MP4 frame fallback: `FRANCMASSONIC`, six progressive content deltas, zero
  reasoning events, one text-done, one completed terminal.
- The MP4 terminal followed the last content delta by 0.023442 s.

Both routes are wrong on the same pixels, so this gate does not identify a
video-only decode/cache defect. It also does not blame the trusted quantized
artifact: a controlled external/reference-runtime A/B is still required to
separate shared vMLX vision preprocessing from artifact/runtime visual quality.

## Automated validation

```text
39 selected capability/modality/video tests passed
739 expanded engine, multimodal, Gemma artifact/image/reasoning/tool, and
MLLM scheduler tests passed (two third-party librosa warnings)
```

## Boundaries and next work

- Native Gemma video remains correctly `false`; sampled-frame MP4 transport is
  current-source/live `PASS`.
- Blind OCR/grounding is `FAIL/PARTIAL`, not release-promoted.
- E2B/E4B audio-capable MoE-like Gemma variants and 26B/31B vision-only
  variants require separate bundle-grounded live rows; capabilities must not
  be generalized from this 12B unified artifact.
- Signed-app repetition and a same-artifact reference-runtime visual A/B remain
  open.
- No prompt answer injection, output rewrite, OCR shim, sampler clamp, or
  synthetic media behavior was added.
