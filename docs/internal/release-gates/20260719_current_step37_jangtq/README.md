# Step 3.7 JANGTQ current image/video, stream, and L2 gate

Date: 2026-07-19

Status: `SCOPED_CURRENT_SOURCE_PASS / FAMILY_PARTIAL`

Current source before this scoped fix was `ca1bb94d8`. The source/test/evidence
commit for this gate is recorded after commit below. This gate does not promote
the full release matrix.

## Artifact and launch truth

- Bundle:
  `/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANGTQ_K`.
- This is JANGTQ/MXTQ codebook quantization (`weight_format=jangtq`, profile
  `JANGTQ_K`, 2.841 measured bits), not JANG affine and not base MLX MXFP.
- The bundle contains Step vision tensors but no MTP tensors. The nested
  `num_nextn_predict_layers=3` field is an architecture hint only; no MTP flag
  was launched and health reports the artifact inactive for MTP.
- Real Electron Sessions Start launched port 8022 with `--is-mllm`,
  `--tool-call-parser step3p5`, `--reasoning-parser qwen3`, paged cache,
  64-token blocks, Block Disk L2, and Auto q4 stored-prefix quantization.
- Health identifies 12 full-attention q4 TurboQuant storage slots plus 33
  native rotating-SWA slots. Rotating metadata remains native.
- Electron Chat Settings visibly showed Auto/On reasoning, low/medium/high,
  Responses wire, and built-in tools enabled.

## Failure control and root cause

The first current-source Electron PNG request failed before decode:

```text
ValueError: Step3.7 image placeholder count does not match multimodal embeddings:
placeholders=169 embeddings=0
```

The attachment reached the server as real pixel data. A temporary diagnostic
showed `pixel_values=(1,3,728,728)` and `num_patches=array([0])`.
`mlx_vlm.prepare_inputs` legitimately normalized the zero-patch metadata to an
MLX array. `Model._process_image_input` used
`image_input.get("num_patches") or []`; false truthiness of `array([0])`
discarded the base-image feature while prompt placeholders remained.

## Source repair

- `vmlx_engine/models/step3p7_mlx_vlm.py:708-749` normalizes MLX and NumPy
  `num_patches` arrays to a one-dimensional Python integer list and rejects
  nested metadata.
- `vmlx_engine/models/step3p7_mlx_vlm.py:780-819` uses an explicit `None`
  check, preserving `[0]` and therefore the base image embedding.
- `tests/test_step37_mlx_vlm_runtime.py:771-787` covers MLX and NumPy
  `array([0])` metadata.
- The temporary diagnostic was removed before verification; there is no
  production-only logging branch.

## Focused verification

The current source passed 422 selected tests:

```text
tests/test_step37_mlx_vlm_runtime.py
tests/test_step3p7_mllm_detection_guard.py
tests/test_step37_vlm_runtime_audit.py
tests/test_step37_crash_falsification_contract.py
tests/test_mllm_scheduler_cache.py
tests/test_vl_media_cache_contract.py
tests/test_reasoning_modes.py
tests/test_streaming_reasoning.py
tests/test_step3p5_tool_parser.py
tests/test_reasoning_tool_interaction.py

422 passed, 2 warnings in 3.88s
```

## Live Electron image matrix

All turns used the real Electron renderer on CDP 9335 and the real attachment
input. SQLite rows are sanitized in `assistant-rows.json`.

| Row | Media | Result | Cache | TTFT | Verdict |
|---|---|---|---|---:|---|
| 455 | image A `VISION-A-7319` | correct marker plus visible self-correction prose | cold | 44.44s | image transport PASS; strict format FAIL |
| 458 | identical image A | exact `VISION-A-7319` | 4,290 `paged+mixed_swa`; pixel hit | 0.97s | resident reuse PASS |
| 461 | same-shape image B `VISION-B-2846` | correct marker plus visible self-correction; no A leakage | cold, zero prefix hit | 44.87s | media-salt isolation PASS; strict format FAIL |
| 464 | return image A | exact `VISION-A-7319` | 4,290 `paged+mixed_swa` | 2.38s | return-A reuse PASS; telemetry inconsistency retained |
| 467 | real 4-second MP4 B | exact `VIDEO-B-8264` | cold | 55.21s | video transport/OCR/paint PASS; cold latency PARTIAL |
| 470 | image A after visible Stop/Start | exact `VISION-A-7319` | 4,290 `paged+mixed_swa+disk` | 1.71s | process-restart L2 PASS |

DOM observers recorded progressive final-answer paints. For example, row 470
painted `V`, `VI`, `VIS`, through `VISION-A-7319`; it did not appear as one
terminal batch.

The A -> B miss -> A return sequence proves the original media salt selects
distinct prefixes. After process restart, health began with zero L1 entries and
15,987 L2 tokens. Row 470 then reported `disk_hit=true`, 68 block-disk hits,
68 native-TQ q4 hits, successful reconstruction/dequantization, and no new
write.

## Raw API streaming

Literal `curl -N` requests used actual PNG data URLs.

- Chat Completions: 46 separate reasoning deltas, then 42 progressive content
  deltas, `finish_reason=stop`, usage, and `[DONE]`. The content was
  `VISION-A...” ... VISION-A-7319`, proving the unwanted self-correction is
  genuine model `delta.content`, not a reasoning-parser or Electron batching
  defect.
- Responses repeat: 73 `response.reasoning_summary_text.delta` events, six
  progressive `response.output_text.delta` events forming exact
  `VISION-A-7319`, all done events, one `response.completed`, and usage with
  223/224 cached input tokens.
- Neither request emitted a tool call or parser marker.

## Retained issues

1. `STEP-CONTENT-STRICTNESS`: PARTIAL/FAIL. Some cold Step generations place
   self-correction prose after the model's reasoning terminator. Raw Chat proves
   it is native content. Do not hide or rewrite it in a parser.
2. `STEP-COLD-MEDIA-LATENCY`: PARTIAL. Image cold TTFT was 44.44-44.87s and
   video cold TTFT was 55.21s. Warm and disk restore are much faster, but the
   cold latency remains visible.
3. `STEP-SAME-PROCESS-DISK-DETAIL`: FAIL telemetry. Row 464 increased aggregate
   block-disk/TQ-native hit counters by 204 while the per-request
   `last_cache_execution.disk_hit` stayed false and UI detail omitted `disk`.
   The clean restart row correctly reported disk. The mixed-tier source field
   needs an owning-layer audit.
4. `STEP-RESTART-PID-UI`: FAIL telemetry. After Electron Stop/Start, the model
   was healthy and the UI showed Stop, but the active header no longer displayed
   a PID.
5. `STEP-STOCHASTIC-SOAK`: PARTIAL. The retained unseeded 1,024-token reasoning
   loop remains a failure control. No sampler coercion or hidden output clamp
   was added.
6. `STEP-LARGER-VIDEO`: PARTIAL. The real four-second MP4 passes; a larger and
   more varied video soak remains.
7. Health's legacy `quantization_format.type=jang` is not sufficient to classify
   this artifact. The authoritative `quantization.codec=turboquant_codebook`,
   `weight_format=jangtq`, and `profile=JANGTQ_K` fields must remain visible so
   JANG affine, JANGTQ/MXTQ, and MLX MXFP are not conflated.

## Evidence files

- `step-current-loaded.png`
- `step-current-chat-settings.png`
- `step-current-image-a-attached.png`
- `step-stopped-after-diag.png`
- `step-current-image-a-fixed.png`
- `step-current-image-a-repeat-ram.png`
- `step-current-image-b-isolation.png`
- `step-current-image-a-return.png`
- `step-current-video-b.png`
- `step-current-restarted-loaded.png`
- `step-current-image-a-restart-disk.png`
- `assistant-rows.json`
- `api-stream-summary.json`
- `restart-l2-health-summary.json`
- `focused-tests.txt`
