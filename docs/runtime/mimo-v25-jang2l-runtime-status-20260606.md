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
- MiMo XML-function fallback instructions stay inside the rendered ChatML
  system turn when MiMo's template drops synthetic fallback messages; vMLX no
  longer prefixes those instructions before `<|im_start|>system`.

Focused source tests:

`tests/test_engine_audit.py -k "native_rotating_cache or metal_ws_pressure_guard_reclaims_cache_before_reject or paged_cache_tensor_store_drops_full_request_cache_reference or mimo_v2_cached_text_fast_path_uses_absolute_positions or xml_function_tool_fallback_requires_concrete_function_examples"`

Latest result:

- 6 passed, 515 deselected.

Additional focused parser/template proof after ChatML fallback fix:

`tests/test_tool_fallback_injection.py::test_mimo_xml_function_direct_fallback_stays_inside_chatml_system`

Latest result:

- 1 passed.

Full focused parser/fallback slice:

`tests/test_tool_fallback_injection.py::test_mimo_xml_function_fallback_matches_parser_dialect tests/test_tool_fallback_injection.py::test_mimo_xml_function_direct_fallback_stays_inside_chatml_system tests/test_xml_function_tool_parser.py`

Latest result:

- 12 passed.

## Next required checks

1. Obtain a corrected/current MiMo V2.5 JANG_2L artifact without using ADLab work as implementation guidance.
2. Prefer HTTP/local network transfer only for the model artifact itself; do not pull implementation guidance from ADLab/TB/RDMA notes.
3. Compare artifact metadata and hashes against the installed `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` copy.
4. Run the same direct XML continuation probe and required-tool SimpleEngine test on the replacement/current artifact.
5. If replacement works, classify the installed local model as corrupt/bad, enumerate exact old MiMo paths, then replace/delete the old MiMo copies.
6. If replacement fails identically, classify this as model/tool-template/decode compatibility and continue in vMLX Python engine only with legitimate fixes such as constrained/guided XML decoding.
7. Do not synthesize fake tool calls from `tool_choice=required` or user text as a release fix.
8. Only after required tools pass, run full smoke with paged cache, L2 block cache, native rotating cache, and no generic TQ-KV.
9. Only after text/tools pass, implement/prove MiMo image/video/audio runtime. Current MiMo media is preserved but unwired.
10. Only after all relevant families pass E2E should release/sign/notarize proceed.

## Release gate

Do not release, sign, notarize, or call MiMo/MLXStudio/vMLX green yet.

Current release status: red.

## 2026-06-07 ChatML tool-fallback framing fix and live result

Source change:

- `vmlx_engine/api/tool_calling.py` now keeps MiMo/XML-function fallback tool
  instructions inside the first rendered ChatML system turn if MiMo's template
  drops synthetic fallback messages during re-render.
- `tests/test_tool_fallback_injection.py` now covers this regression.

No-heavy proof:

- `py_compile` passed for `vmlx_engine/api/tool_calling.py` and
  `tests/test_tool_fallback_injection.py`.
- Focused parser/fallback tests passed: 12 passed.
- Render proof for the real local MiMo tokenizer showed
  `inside_system=True` and `prefix_outside_chatml=False`.

Live proof:

`build/current-all-local-model-smoke-mimo-v25-jang2l-after-chatml-tool-fallback-20260607`

Command boundary:

- Current source venv.
- Installed model:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Continuous batching, paged cache, block-disk L2, and native mixed full/SWA
  cache active.
- `--kv-cache-quantization none`.
- No media/video/reasoning; tools included.

Live positives:

- `text_cache_repeat_1`: HTTP 200, `ACK`.
- `text_cache_repeat_2`: HTTP 200, `ACK`, `cached_tokens=67`,
  `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, `blue cat`.
- Block-disk L2 wrote 4 blocks / 141 tokens.
- Native cache remained `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`.
- Generic TurboQuant KV remained disabled for this diagnostic.

Live failure:

- `tool_required`: HTTP 400, no parsed tool calls.
- Server raw preview:
  `<tool_call>\n\n：，，，。：，，。。。。。。，。。。。，。。，。。，，...`
- The model still failed to emit `<function=record_fact>` or
  `<parameter=value>blue-cat</parameter>`.
- Generation speed for the tool failure was about `1.7 tok/s`.

Classification:

- The ChatML fallback-framing bug is fixed.
- MiMo required-tool generation remains red after the framing fix.
- The remaining blocker is deeper than parser aliasing or prompt placement:
  current decode/model/runtime path still cannot reliably continue the
  XML-function grammar for required tool calls.
- Do not clear MiMo until a corrected artifact, source-vs-quant divergence fix,
  or a real constrained/guided XML decoder is implemented and live-proven
  without synthetic tool-call fabrication.

## 2026-06-07 source-vs-quant harness tool row and stale-cache cleanup

Source change:

- `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py` now
  includes the current release-blocking XML tool row:
  `xml_tool_required_record_fact`.
- The runner now extracts OpenAI `tool_calls`, records source/quant tool-call
  signatures, and classifies tool divergence separately from visible-text
  divergence.
- Proof pointers in the MiMo current-audit runner and release-regression
  manifest now point at:
  `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json`.

Focused validation:

- `py_compile` passed for the source-vs-quant runner, MiMo current audit,
  release manifest, and source-vs-quant tests.
- `tests/test_mimo_v2_source_vs_quant_probe.py`: 8 passed.

Current preflight:

`build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json`

- `status=missing_prerequisites`.
- Source path exists on Max2:
  `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`.
- Quant path exists locally:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Source endpoint `http://erics-m5-max2.local:8126` is not healthy.
- Quant endpoint `http://127.0.0.1:8897` is not currently running.
- This is preflight evidence only; it cannot classify model-upload vs runtime
  until both endpoints run and rows execute.

Current audit:

`build/current-mimo-v2-jang2l-current-audit-after-tool-row-source-quant-preflight-stale-clean-20260607.json`

- `status=open`.
- `stale_local_state_absent=true` after deleting the stale HF remote-code cache:
  `/Users/eric/.cache/huggingface/modules/transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L`.
- Remaining blockers:
  `mimo_long_prompt_coherence_blocked`,
  `mimo_tool_protocol_blocked`,
  `mimo_decode_speed_below_release_target`,
  `mimo_cb_system_prompt_working_set_pressure_blocked`,
  `mimo_source_vs_quant_first_divergence_missing_or_failed`,
  `mimo_vl_audio_video_unwired`.

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

## Direct continuation probe

Artifact:

`build/current-mimo-v25-tool-continuation-probe-20260606`

Command boundary:

- Served installed `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Used SimpleEngine: `--no-continuous-batching --disable-prefix-cache --kv-cache-quantization none`.
- Used `/v1/completions`, not Chat Completions tool parsing.
- Prompt was rendered with the real MiMo chat template plus concrete `mimo_xml_function` fallback example.
- No OpenAI tool parser, no `tool_choice=required`, no paged prefix cache, and no generic TurboQuant KV participated in this probe.

Results:

- Base rendered prompt output:
- `<think><think><think>。，，，。。。，，，，，。。，，，，，，。。，，，。。，。，。。。。，，，。，。，，。。。`
- Assistant-prefilled prompt ending in `<tool_call>\n<function=record_fact>\n<parameter=value>\n` output:
- `blue\n\n\n\n\n，，\n\n。，，，，，。，，，，。，。，，，，，，。，，。。。。。。。。。。。。。。。`
- Base probe speed: 48 completion tokens in 50.42s, about `1.0 tok/s`.
- Prefilled-function probe speed: 48 completion tokens in 29.52s, about `1.6 tok/s`.

Conclusion:

- The current installed MiMo JANG_2L path cannot reliably continue the native XML tool grammar even when the runtime has already supplied the `<function=record_fact>` and `<parameter=value>` prefix.
- This rules out parser-only failure, `tool_choice=required` response wrapping, continuous batching, prefix cache, paged L2, q4 KV storage, and generic TurboQuant KV as sole causes.
- Remaining likely causes are model/artifact/tool-template training incompatibility or missing optimized MiMo/JANG_2L decode support for this bundle.
- A runtime-side fake tool-call synthesis would be incorrect and must not be used as a release fix.
- A legitimate future runtime fix would need constrained/guided XML tool decoding or a proven better MiMo artifact that naturally emits valid XML tools.

## Remote `~/jang` runtime comparison

Remote checked:

`erics-m5-max2.local:/Users/eric/jang`

Findings:

- Remote implementation doc: `research/MIMO-V2.5-IMPLEMENTATION.md`.
- Remote quant contract: `docs/runtime/2026-05-27-mimo-v2-jang-2l-quant-contract.md`.
- Remote runtime source: `jang-tools/jang_tools/mimo_v2/mlx_model.py`.
- Local bundled runtime SHA-256: `7b6a0a524b486907f5e2fd2c5a122c1dc58726be1d8d6c2c1304a9aafda57f7d`.
- Remote runtime SHA-256: `a33ec86fc409e83feccf049c74d4c19be813ec6f5d130a6a20bf19d713d9372d`.

Remote docs confirm:

- The promoted coherent canonical MiMo JANG_2L path was measured at about `1.974-2.640 generation tok/s` on the local M5 Max.
- The docs explicitly say the current affine MLX path is coherent but not a `30 tok/s` path on this local M5 Max.
- A faster candidate reached `5.626 generation tok/s` on cached France but failed arithmetic with output `45.` for `2 + 2`, so it was rejected.
- Current 40+ tok/s expectations therefore require a different optimized artifact/runtime class, not just a serving flag on this 106GB affine JANG_2L bundle.

Runtime integration delta:

- Local bundled `jang_tools` lacked the remote MiMo compiled single-token router.
- vMLX now installs a narrow fallback compiled router during MiMo MLLM registration if the bundled `jang_tools` runtime is stale.
- The patch preserves the existing vMLX `inputs_embeds` compatibility wrapper and does not override newer `jang_tools` runtimes that already expose `run_compiled_mimo_decode_router`.

Live fallback-router probe:

Artifact:

`build/current-mimo-v25-compiled-router-live-probe-20260606`

Result:

- Server log confirms `Installed vMLX fallback compiled MiMo-V2 decode router`.
- Prompt prefilled through `<tool_call>\n<function=record_fact>\n<parameter=value>\n` still generated `blue` plus punctuation/newline garbage.
- Short first request speed was 16 completion tokens in 30.67s, about `0.5 tok/s`; this does not prove a speed win and includes first compiled-router request overhead.

Conclusion:

- The stale-runtime compiled-router integration miss is real and now patched in vMLX.
- It is not sufficient to clear `MIMO-TOOL-001`.
- It is not sufficient to clear `MIMO-SPEED-001`.
- Do not call MiMo fixed until either a better artifact is proven or a real constrained/guided XML decoder plus optimized routed-expert decode path is implemented and E2E-proven.
