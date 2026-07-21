# Converted affine-JANG Qwen VL panel gate — 2026-07-21

Status: `PASS-LIVE_MEDIA+STREAM / PARTIAL_STRICT_FORMAT`.

This is a narrow follow-up to
`../20260721_electron_jang4m_conversion_agent/`. It does not repeat or promote
that gate's text, tool, cache, quantizer, or protocol rows.

## Artifact identity

- Model: `/Users/eric/models/Codex-Quant-Probe-OsaurusAgent-9b-JANG_4M`
- This is affine `mx.quantize` JANG_4M at 4.66 measured bits. It is **not**
  JANGTQ/MXTQ Hadamard-codebook quantization and is **not** base MLX MXFP.
- The converted tensor index contains 1,260 keys, including the same 333
  vision-tower keys counted in the source artifact. `jang_config.json`
  declares Qwen3.5, hybrid SSM, vision, and `has_vision=true`; it does not carry
  the older `runtime_verified/vision_verified` compatibility stamp.

## Defects and source repair

1. `panel/src/main/model-config-registry.ts` required the historical runtime
   stamp and therefore forced this real converted vision artifact text-only.
   The detector now accepts either the stamp or actual indexed vision tensors,
   while explicit `has_vision=false` and metadata-only/text extracts still fail
   closed.
2. `panel/src/main/sessions.ts::updateSessionConfig` discarded submitted
   `undefined` values before merge. For tri-state settings, the renderer uses
   `undefined` to mean Auto, so a stale explicit `isMultimodal=false` could not
   be cleared. Explicitly submitted undefined keys now delete the saved
   override and participate in restart-required detection; omitted keys remain
   untouched.
3. Active source comments referred to deleted
   `docs/AUDIT-QWEN-AFFINE-JANG-VLM.md` and described the resolved legacy
   M-RoPE fallback as current policy. Those zombie references were removed or
   rewritten around the current vMLX-owned `qwen3_5_family` runtime. The
   runtime/env branches themselves have production launch/registry/loader call
   sites and were not deleted.

## Real Electron proof

- The development app used CDP 9335 and the real Sessions UI. The Auto VLM
  selection removed the saved `isMultimodal` key from SQLite; this was checked
  after clicking **Save Settings**.
- A real **Start** click eagerly materialized PID 71790 before any request.
  The dev log contained
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- The process argv included `--is-mllm`, `--reasoning-parser qwen`,
  `--tool-call-parser qwen`, `--use-paged-cache`,
  `--enable-block-disk-cache`, and `--stream-interval 1`.
- Pre-request health reported `model_loaded=true`, `model_type=mllm`,
  `engine_type=batched`, and `last_request_time=null`. The MTP subsection's
  `vl_runtime_available=false` refers to the missing native-MTP artifact, not
  the ordinary vision route; the same health object reports 333 vision tensors.
- A new Electron chat attached the real PNG through the file input. Engine
  diagnostics recorded one `image_url`, `engine_is_mllm=true`, and
  `MLLMBatchGenerator: Using VLM's language_model for batched generation`.
- `ui-stream-trace.json` records 17 distinct painted states. Visible content
  grew at 16.657 s, 16.761 s, 16.865 s, and was complete at 16.967 s. The
  temporary probe's `completed=false` is a harness false negative: its terminal
  predicate incorrectly required a second copy of the marker even though the
  frame and SQLite row were terminal.
- SQLite row 275 has non-empty content, a separate 271-character reasoning
  field, no tool calls, no warnings, and 16.7 s total time. The image-derived
  inner text was correct:
  `QUANT-UI-T3-DONE PATH=panel/package.json SIZE=5.2 KB`.
- Strict formatting is **partial**: the requested outer prefix was `QAFF`, but
  the model emitted `QUAFF`. This byte-level model output miss is retained and
  is not hidden with parser rewriting, prompt coercion, or synthetic cleanup.

## Raw Responses image stream

The same current engine source and Electron-loaded converted artifact were
probed directly through `/v1/responses` with the screenshot and
`media_omitmax_probe.py`:

- HTTP 200;
- 51 `response.reasoning_summary_text.delta` events;
- 19 progressive `response.output_text.delta` events;
- one `response.completed` terminal;
- 4,964 input tokens and 73 output tokens;
- first reasoning delta at 11.944 s, first content delta at 12.561 s, terminal
  at 12.793 s;
- exact visible output:
  `QUANT-UI-T3-DONE PATH=panel/package.json SIZE=5.2 KB`.

This proves API media transport and separate progressive reasoning/content for
this artifact. It is not Chat/Anthropic/Ollama media parity.

## Cache evidence boundary

The Electron log shows a clean media-conditioned 8,792-token N-1 boundary,
24 SSM companion layers, 138 block-disk write-through blocks, and the paged
prefix store. This gate proves **store only**. It does not claim resident hit,
media-salt isolation, partial-prefix reuse, eviction, or restart/L2 restore for
this newly converted artifact; those remain open under the prior conversion
gate rather than being re-run here.

## Focused validation

- Python affine-Qwen registry/API/loader selection: 12 passed.
- Panel model detection and settings persistence: 374 passed.
- Panel TypeScript typecheck: passed.

## Evidence files

- `qaff-vl-ui1-result.png` — final real Electron frame.
- `qwen-affine-vlm-loaded.png` — active model after real Start.
- `qwen-affine-chat-ready.png` — real chat surface with attachment control.
- `ui-stream-trace.json` — progressive DOM snapshots; see the documented
  harness false-negative predicate above.
- `pre-request-health.json` — eager `mllm` health before generation.
- `live-proof-summary.json` — compact API/UI/process/DB observations.

## Remaining

- Converted-artifact video, audio where advertised, media-salt A/B/A,
  partial-prefix reuse, forced eviction, and restart/L2 restore.
- Chat/Anthropic/Ollama media parity for this converted artifact.
- Strict-format reliability beyond the one exact raw API pass and one UI
  outer-prefix miss.
- Signed-app repetition and other affine profiles/custom mixes.
