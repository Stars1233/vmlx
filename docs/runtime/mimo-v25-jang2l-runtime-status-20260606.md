# MiMo V2.5 JANG_2L Runtime Status - 2026-06-06

## Scope

Active repo: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

Do not use `/Users/eric/vmlx`, ADLab, TB/RDMA workflows, or Swift-only paths for this work. This status is for the vMLX Python engine and MLXStudio-facing runtime behavior.

## Current verdict

MiMo V2.5 JANG_2L is not release-green.

Working pieces are real and live-proven, but required tool calling is still red and speed is far below the expected 40+ tok/s target.

## Live proofs completed

### Text and cache path

Latest full smoke artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-keep-native-tools-20260606/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Passing checks:

- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: `prompt_tokens_details.cached_tokens=67`.
- `text_cache_repeat_2`: `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- L2 block disk cache wrote 4 blocks / 141 tokens in the smoke.
- Native cache metadata reports `mixed_swa_kv_v1`, full plus rotating/sliding-window layout preserved.

### TurboQuant/KV cache split

Diagnostic artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-diagnostic-no-tq-kv-20260606`

Finding:

- Generic TurboQuant KV corrupted MiMo mixed full/SWA rotating-cache semantics.
- Disabling generic TQ-KV made the repeated cached text path pass.
- Runtime fix now skips flat generic TurboQuant KV when native cache contains `RotatingKVCache` / `BatchRotatingKVCache`.
- This is not a fake guard: native rotating cache remains active, and storage-boundary q4 cache quantization remains separate from generic TQ-KV.

### Pressure guard split

Finding:

- The old 98% Metal working-set guard rejected valid follow-up requests after cache growth.
- Runtime now clears reclaimable MLX cache before rejecting and evaluates non-cache pressure.
- Default guard threshold moved to 99%.
- This let MiMo progress past text cache and multiturn in live smoke.

## Remaining red blockers

### MIMO-TOOL-001: required tool calling corrupts output

Failing request:

`Use the record_fact tool exactly once with value blue-cat. Do not answer in visible text.`

Expected tool:

`record_fact({"value":"blue-cat"})`

Observed with batched MLLM, paged cache, L2 block cache, native tools preserved:

- HTTP 400 because `tool_choice=required` produced no parsed tool calls.
- Generation emitted 96 tokens at about 1.6 tok/s.
- Raw preview: `<tool_call>` followed by punctuation/Chinese-comma garbage, not `<function=record_fact>`.

Observed with `--kv-cache-quantization none`:

- Same failure.
- Same raw preview shape.
- Rules out q4 KV quantization as the root cause.

Observed with SimpleEngine, `--no-continuous-batching`, `--disable-prefix-cache`, `--kv-cache-quantization none`:

- Same failure class.
- Raw preview is punctuation garbage.
- Rules out batched scheduler, paged prefix cache, L2 disk cache, and KV quantization as the sole cause.

Observed with thinking enabled in SimpleEngine:

- Still HTTP 400.
- Generated 97 tokens, but no visible parse text after reasoning extraction.
- Does not fix tool calling.

Current classification:

- Not a parser-only issue: parser supports `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`.
- Not a generic TQ-KV issue: failure reproduces with `--kv-cache-quantization none`.
- Not a paged/L2 issue: failure reproduces with prefix cache disabled and SimpleEngine.
- Likely one of: installed MiMo artifact corruption, MiMo template/decode incompatibility in current runtime, tokenizer/template mismatch, or model not trained/stable for required tool calls in this quant.

### MIMO-SPEED-001: speed far below target

Observed live smoke speeds:

- First short text request: about 0.1 tok/s due large first prefill/load path.
- Cached repeat: about 1.1 tok/s.
- Multiturn: about 1.2 tok/s.
- Required tool failure path: about 1.6 tok/s.

Target:

- 40+ tok/s expected by Eric.

Current classification:

- Red. Need separate profiling after correct artifact/runtime path is established.
- Do not claim release readiness or sign/notarize while this remains red.

### MIMO-VL-001: MiMo vision/audio/video runtime unwired

Observed capabilities report:

- Runtime modalities: `text`.
- Declared/preserved modalities: `vision`, `image`, `video`, `audio`.
- Status: preserved but unwired for image/video/audio.

Current classification:

- Red for VL/audio/video release.
- Text-only smoke cannot clear VL/audio/video.

## Engine changes made in this slice

Files changed:

- `vmlx_engine/mllm_batch_generator.py`
- `vmlx_engine/utils/tokenizer.py`
- `vmlx_engine/utils/jang_loader.py`
- `vmlx_engine/server.py`
- `vmlx_engine/prefix_cache.py`
- `vmlx_engine/api/tool_calling.py`
- `tests/test_engine_audit.py`

Implemented behavior:

- MiMo cached text fast path passes absolute position ids and seeds decode position state.
- Generic TurboQuant KV skips native rotating/full attention cache layouts.
- JANG TurboQuant KV patcher skips native rotating/full attention cache layouts.
- Metal working-set guard clears reclaimable cache and checks non-cache pressure before rejecting.
- Paged tensor-backed prefix cache no longer keeps full request cache references after L2 write-through.
- XML function fallback requires concrete `<function=name>` examples, not generic `example_function_name` only.
- XML function fallback injects into first user turn and preserves native `tools` during re-render.
- Required-tool failures now log a bounded raw preview for diagnosis.

Focused source tests:

`tests/test_engine_audit.py -k "native_rotating_cache or metal_ws_pressure_guard_reclaims_cache_before_reject or paged_cache_tensor_store_drops_full_request_cache_reference or mimo_v2_cached_text_fast_path_uses_absolute_positions or xml_function_tool_fallback_requires_concrete_function_examples"`

Latest result:

- 6 passed, 515 deselected.

## Next required checks

1. Obtain the intended/current MiMo V2.5 JANG_2L artifact without using ADLab work as implementation guidance.
2. Compare artifact metadata and hashes against the installed `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` copy.
3. Run the same required-tool SimpleEngine test on the replacement/current artifact.
4. If replacement works, classify installed local model as corrupt/bad and replace/delete the old MiMo copies.
5. If replacement fails identically, classify this as runtime/template/decode compatibility and continue in vMLX Python engine.
6. Only after required tools pass, run full smoke with paged cache, L2 block cache, native rotating cache, and no generic TQ-KV.
7. Only after text/tools pass, implement/prove MiMo image/video/audio runtime. Current MiMo media is preserved but unwired.
8. Only after all relevant families pass E2E should release/sign/notarize proceed.

## Release gate

Do not release, sign, notarize, or call MiMo/MLXStudio/vMLX green yet.

Current release status: red.

## Remote canonical artifact comparison

Checked `erics-m5-max2.local` on 2026-06-06.

Remote canonical path:

`/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`

Local canonical path:

`/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`

Comparison result:

- Local size: `106G`.
- Remote size: `106G`.
- Local shard count: `150`.
- Remote shard count: `150`.
- `config.json` SHA-256 matches: `c52d416c847d75c3c6cf5fb4c7eb984bbe868a552bd063659ffcc9fc3199ffe8`.
- `model.safetensors.index.json` SHA-256 matches: `5868a0220ff6610a962da52211ebb59b4a98b09a1c7785c20633a711dd9b006b`.
- `chat_template.jinja` SHA-256 matches: `3134ac101acd29d3ab41297707cc1a85699f5f0acb283fdeb0681e3750998403`.
- `model-00001-of-00150.safetensors` SHA-256 matches: `b5c8623063d66fea9fd6578dc354dc82fbf7a007ae6262397b9263a4ed2baccf`.
- `model-00150-of-00150.safetensors` SHA-256 matches: `4cdad7d9e973a8617851666ea1ab12bf67a4b8ca00ffd35ea5a0996c1f49308f`.

Conclusion:

- Re-downloading the canonical remote MiMo JANG_2L artifact is not expected to fix the current tool-call failure.
- The local installed artifact appears to match the remote canonical artifact for metadata and sampled shards.
- The remote contract also records expected text generation speed around `1.97-2.64 tok/s`, not `40+ tok/s`, for this 310B / 15B-active bundle on local M5 Max.
- Remote notes say vision/audio are preserved in the bundle but runtime support was deferred. This matches current vMLX capabilities: text runtime only, media preserved but unwired.

## Reproducible harness KV-none proof

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-kvnone-extra-args-20260606/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Command boundary:

- Runner used `VMLINUX_BENCH_EXTRA_SERVE_ARGS='--kv-cache-quantization none'`.
- Served command tail includes `--kv-cache-quantization none`.
- Health reports `kv_cache_quantization.enabled=false`.
- Health reports `turboquant_kv_cache.enabled=false`.
- Native cache remains `mimo_v2` / `mixed_swa_kv_v1` with prefix, paged cache, and block-disk L2 enabled.

Results:

- `text_cache_repeat_1`: HTTP 200, `ACK`.
- `text_cache_repeat_2`: HTTP 200, `ACK`, `cached_tokens=67`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, `blue cat`.
- `tool_required`: HTTP 400, no parsed tool calls.
- Raw preview: `<tool_call>` followed by punctuation/fullwidth-comma garbage instead of `<function=record_fact>`.

Conclusion:

- MiMo tool corruption is not caused by q4 KV cache storage or generic TurboQuant KV.
- The failure persists with cache storage quantization disabled while the intended prefix/paged/L2 cache stack remains active.
