# Step 3.7 image/video runtime and mixed-SWA cache proof

Status: `PASS-LIVE` for the scoped Step 3.7 image and one-video cache path at
commit `c305b18b5`; commit `8b0e23dc1` separately makes the native
reasoning-only UI/API contract truthful. Strict raw formatting, cold-store
latency, alternate-video isolation, stochastic reasoning reliability, broader
artifacts, and long-output reliability remain `PARTIAL`.

## Root cause and source trace

The first real Electron image row reached Step's source-owned vision stack but
persisted zero tokens. The server traceback was:

```text
step3p7_mlx_vlm.py::_pixels_to_nhwc
mx.transpose(pixel_values, (0, 2, 3, 1))
TypeError: Invoked with types: ndarray, tuple
```

The shared MLLM request path asks processors for NumPy tensors. Step's direct
processor tests used MLX tensors, so the source model never normalized the live
NumPy boundary. `step3p7_mlx_vlm.py:708-739` now converts NumPy pixel,
patch-pixel, and image-embedding tensors to MLX at the model boundary before
any MLX transpose/reshape. No output rewriting, sampler coercion, or fake
fallback was added.

`mllm_batch_generator.py:4891-4930` and
`mllm_scheduler.py:1028-1074,3098-3120` then admit config-derived `step3p7`
media under a canonical side-key, while retaining an explicit cache-policy-off
escape hatch, and store the captured media-conditioned N-1 boundary directly.
The 45-layer boundary keeps 33 native
rotating-SWA layers plus 12 compatible full-attention TQ4 slots; rotating-SWA
state is not TQ encoded.

Exact source excerpts are in `source-trace.txt`.

## Automated proof

At commit `c305b18b5`:

```text
.venv/bin/python -m pytest -q \
  tests/test_step37_mlx_vlm_runtime.py \
  tests/test_step3p7_mllm_detection_guard.py \
  tests/test_zaya_runtime.py \
  tests/test_mllm_scheduler_cache.py
205 passed, 6 skipped, 2 third-party warnings in 5.31s
```

The contracts cover NumPy-to-MLX media normalization, default-on plus explicit
cache-policy-off media admission, and captured mixed-SWA storage without a
text-only path-dependent re-prefill. “Off” here refers to the cache feature,
not a model reasoning mode.

## Electron image proof

The live artifact was
`/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANGTQ_K`, PID 55586 before the
fix and PIDs 56338/57192 after source restarts. Its UI-derived argv included
`--is-mllm`, Auto stored-prefix TQ4, 64-token paged blocks, 1,000 blocks, and
block-disk L2. Health/source reported 12 full-attention TQ slots and 33 native
rotating-SWA slots. The artifact name does not say MTP and its tensor index has
no MTP tensors, so MTP correctly remained inactive despite an architectural
three-layer hint.

- Pre-fix row 198: zero tokens; live NumPy/MLX transpose traceback.
- Source-fixed policy-off rows 201/204: exact `BANANA8426`, no cached tokens,
  TTFT 19.38s and 19.60s. This proved vision correctness and the original cache
  miss separately.
- Cache cold row 207: exact `BANANA8426`; 2,203 prompt tokens. The clean N-1
  store pass raised TTFT to 38.83s and is retained as a measured cold cost.
- Identical A row 210: exact `BANANA8426`, 2,202/2,203 tokens restored as
  `paged+mixed_swa`, TTFT 0.94s.
- Different same-shape B row 213: zero cached tokens and exact
  `B1-MEDIA-B-DONE`; no A marker leaked.
- Return A row 216: exact `BANANA8426`, 2,202 resident tokens, TTFT 1.67s.
- Visible Stop/Start left zero L1 tokens and 97 disk blocks. Row 219 restored
  2,202/2,203 as `paged+mixed_swa+disk`, returned exact `BANANA8426`, and health
  reported 35 block-disk plus 35 native-TQ hits with zero scheduler disk misses.

## Electron video and raw Responses proof

The real six-frame MP4 was attached as `video/mp4`; the app sent `video_url`,
the server decoded the file, and the Step fallback sampled one real frame.

- Cold row 222: exact `BANANA8426`, 373 prompt tokens, TTFT 6.82s.
- Resident row 225: exact `BANANA8426`, 372/373 `paged+mixed_swa` tokens,
  TTFT 0.98s.
- After visible process restart, health showed zero L1 tokens and 103 disk
  blocks. Row 228 restored 372/373 as `paged+mixed_swa+disk`, returned exact
  `BANANA8426`, and health reported six block-disk/native-TQ hits.

A real `curl -N /v1/responses` request reused the Electron video prefix and
emitted 120 independently timed reasoning deltas, followed by six content
deltas, then one completed terminal. Usage reported 373 input tokens including
372 cached. The raw final was `\nBANANA8426`; the leading newline is retained as
a strict-format miss even though the Electron bubble persisted `BANANA8426`.

## Current native reasoning contract

The real template always opens `<think>` and has no `enable_thinking` branch.
Commit `8b0e23dc1` therefore hides Thinking Off in Electron, advertises only the
reasoning mode plus low/medium/high efforts, rejects API Off requests with HTTP
400, and removes Step from the generic thinking-off answer retry. Live Electron
row 300 again returned exact `BANANA8426` with 376/377
`paged+mixed_swa+disk` tokens after process restart. A deterministic Responses
cold/RAM/restart-disk sequence streamed both rails and returned the same exact
marker every time. One temperature-0.6 disk-hit probe looped inside reasoning
and ended incomplete, so broader Step output reliability remains `PARTIAL`.
See `../step37-reasoning-mode-current/`.

## Remaining gates

- Optimize or explicitly accept the measured double-prefill cold-store cost.
- Prove a distinct-content video B miss/return-A isolation row.
- Exercise larger/longer real video without exceeding the 70 GB model's Metal
  headroom.
- Reproduce and trace the retained temperature-0.6 reasoning loop without
  hidden sampler coercion; deterministic cache equivalence does not close that
  stochastic reliability row.
- The generic streamed-error surface is now closed at scoped source commit
  `57d5bcd0f`: both Chat Completions error chunks and Responses
  `response.failed` events visibly fail and leave no assistant success row.
  See `../stream-error-surface-current/`. A signed packaged-app repeat remains
  open.
- Full-suite, packaged-app, signing, notarization, and public release remain
  open.
