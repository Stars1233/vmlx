# Upstream MLX runtime intake - 2026-06-09

Scope: fixes-only upstream intake for current vMLX Python engine work. No packaging,
signing, notarization, tags, downloads, or release actions.

## Implemented in vMLX source

1. `mlx-lm` PR #1370, BatchRotatingKVCache `meta_state` bool parsing.
   - Local venv reproduced the bug: `bool("False")` reloaded `rotated=True`.
   - vMLX fix: `vmlx_engine/runtime_patches/mlx_lm_compat.py` patches the
     setter to parse string booleans explicitly.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_batch_rotating_kv_cache_false_meta_state_roundtrips`.

2. `mlx-lm` PR #1354, LFM2 MoE sigmoid routing.
   - Local venv still used `softmax(gate(x))`, folded `expert_bias` into weights,
     and had no `routed_scaling_factor`.
   - vMLX fix: runtime patch replaces LFM2 MoE call with sigmoid routing,
     uses `expert_bias` only for top-k selection, applies
     `routed_scaling_factor`, and defaults it to `1.0` for old configs.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_lfm2_moe_runtime_patch_uses_sigmoid_route_and_bias_only_for_selection`.

3. `mlx-lm` PR #1361, Gemma4 channel-thinking auto-detection.
   - Local venv still classified `<|channel>/<channel|>` as thinking even when
     Gemma4 also has `<|think|>`, which can produce empty visible content.
   - vMLX fix: runtime patch makes `_infer_thinking()` return no default thinking
     rail for that Gemma4 vocab shape.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_gemma4_thinking_detection_patch_skips_channel_mode_when_think_token_present`.

4. Gemma4 audio placeholder turn terminator.
   - In-flight edit preserved: Gemma4 audio placeholders are inserted before
     `<turn|>` as well as `<end_of_turn>`, `<|im_end|>`, and `</s>`.
   - Proof: `tests/test_mllm_scheduler_cache.py::test_gemma4_audio_processor_prompt_uses_turn_pipe_terminator`.

5. `mlx-vlm` PR #1321, Gemma4 video processor tolerance for standard HF config
   keys.
   - Local venv reproduced the upstream gap: pinned
     `mlx_vlm.models.gemma4.processing_gemma4.Gemma4VideoProcessor.__init__`
     has no `**kwargs`, so real `processor_config.json` keys such as
     `do_convert_rgb`, `do_sample_frames`, `resample`, and `return_metadata`
     can reject processor construction before image/video inputs are wired.
   - vMLX fix: `vmlx_engine/runtime_patches/mlx_vlm_compat.py` filters unknown
     HF video processor keys for this pinned constructor while preserving all
     accepted Gemma4 video settings.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_gemma4_video_processor_accepts_hf_config_kwargs`.

6. `mlx-lm` issue #1326 / PR #1327, short-prompt think-token search clamp.
   - Local pinned `mlx_lm.tokenizer_utils.TokenizerWrapper._find()` still used
     `start = start or 0`, so a caller using `start=len(prompt)-11` can pass a
     negative reverse-search start for one-token or short thinking prompts.
   - vMLX fix: runtime patch clamps `_find()` start/end bounds and returns
     `-1` for empty sequences or windows shorter than the searched sequence.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_tokenizer_wrapper_find_clamps_negative_reverse_start`.

7. `mlx-lm` PR #1349, Gemma4 Unified text-runtime mapping and encoder-free
   vision-weight sanitizing.
   - Local pinned `mlx_lm.utils.MODEL_REMAPPING` had no `gemma4_unified` entry,
     and `mlx_lm.models.gemma4.Model.sanitize()` did not skip
     `vision_embedder.*` weights.
   - vMLX fix: runtime patch maps `gemma4_unified` to `gemma4` and strips
     `vision_embedder.*` before delegating to the existing Gemma4 sanitize path.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_mlx_lm_gemma4_unified_maps_to_text_runtime_and_strips_vision_embedder`.

8. `mlx-vlm` PR #1301, Gemma4 shared-KV layer loading.
   - Local pinned `mlx_vlm.models.gemma4.language.Gemma4TextModel.__init__`
     still instantiated unused K/V projections for layers covered by
     `num_kv_shared_layers`, and the language sanitize path did not remove stale
     K/V weights from shared layers.
   - vMLX fix: runtime patch marks shared-KV layers as `kv_shared_only`, removes
     unused `k_proj`/`v_proj`/`k_norm`/`v_norm` modules, and filters stale shared
     K/V weights in both outer and language sanitize paths.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_mlx_vlm_gemma4_shared_kv_layers_drop_unused_kv_modules_and_weights`.

9. `mlx-vlm` PR #1332, Qwen3-VL / Qwen3-VL-MoE deepstack visual embeds during
   chunked prefill.
   - Local pinned `mlx_vlm.models.qwen3_vl*.language.LanguageModel.__call__`
     sliced `visual_pos_masks` from `n_to_process` but still forwarded the full
     `deepstack_visual_embeds` list to every chunk. It also left the sliced mask
     longer than the current input window.
   - vMLX fix: runtime patch realigns both `visual_pos_masks` and
     `deepstack_visual_embeds` to the current chunk window for single-sequence
     chunked prefill, using the number of visual tokens before the chunk to slice
     the deepstack rows.
   - Proof: `tests/test_mlx_lm_runtime_patches.py::test_qwen3_vl_chunked_prefill_slices_deepstack_embeds_to_visual_window`.

10. `mlx-vlm` PR #1328, LFM2.5-VL projector layernorm loading.
    - Local pinned `Lfm2VlMultiModalProjector` replaced disabled projector
      layernorm with `nn.Identity`, so mlx-format checkpoints carrying
      `multi_modal_projector.layer_norm.*` weights can have unexpected/load-missed
      projector keys even though the flag should only control whether the
      layernorm is applied.
    - vMLX fix: runtime patch always materializes a `LayerNorm` module for
      loading, records `projector_use_layernorm`, and skips applying the layernorm
      when the config disables it.
    - Proof: `tests/test_mlx_lm_runtime_patches.py::test_lfm25_vl_projector_materializes_layernorm_without_applying_when_disabled`.

## Upstream items checked but not blindly ported

- `mlx-lm` PR #1167 (think-token `None` property guard): already present in the
  local pinned wheel; no vMLX patch needed.
- `mlx-lm` PR #1347 (BPE cleanup-space streaming decode mismatch): inspected,
  but needs a local streaming-vs-final decode mismatch repro before changing
  vMLX detokenization behavior.
- `mlx-lm` PR #1336 (`<tool_call` marker without closing `>`): vMLX streaming
  marker list already contains `<tool_call`; parser/final extraction remains
  guarded by required-schema filtering. Keep testing #192 direct/gateway/tunnel
  raw SSE, but do not synthesize missing args from preamble text.
- `mlx-lm` PR #1371 (OpenAI tool-call arguments stay JSON strings): vMLX has
  endpoint-specific normalization for chat-template replay and API outputs.
  Do not flatten this globally without proving each template path.
- `mlx-lm` PR #1373 (tool calls closed by EOS without end marker): relevant to
  Mistral-style tool streams; still needs local server-state mapping before port.
- `mlx-vlm` PR #1324 (APC disk hit promotion): vMLX already has L2-to-L1
  promotion in scheduler/prefix-cache paths; treat as covered unless a focused
  second-hit disk latency regression proves otherwise.
- `mlx-vlm` PR #1336 (Gemma4 materialized unused shared-KV weights): separate
  from the implemented #1301 shared-KV module/sanitize fix. Keep mapped as a
  future load-key regression candidate before changing materialized weight
  handling.
- `mlx-vlm` PR #1313/#1334 (Qwen3.5/3.6 quantized KV and MTP prefill): relevant
  to Qwen3.6/MTP/gdn_sink lanes, but requires local mapping against
  `utils/mlx_vlm_compat.py` and current Qwen VLM language source before port.
- 2026-06-09 follow-up on `mlx-vlm` PR #1313/#1334: local pinned Qwen attention
  still has the scalar `cache.offset` fallback from #1313, but vMLX also wraps
  Qwen3.5/3.6 VLM attention/language for native MTP, gdn_sink, and text-only
  1D RoPE in `vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py`. Do not paste the
  upstream patch over that path. Add a focused batched quantized-cache offset
  regression against the vMLX patch before changing the attention implementation.
- `mlx-vlm` PR #1316 (thinking enabled by default for chat templates): inspected
  and not ported as a global default. vMLX intentionally keeps Auto/unset
  thinking unresolved for Qwen3.6, MiniMax, and Gemma tool paths, with explicit
  request/server controls winning. Do not convert Auto to a hidden thinking-off
  rail to paper over template/runtime issues.
- `mlx-lm` PR #1377 (top-k interval wording): documentation/error-message-only
  upstream fix. No runtime behavior to port.
- `mlx-vlm` PR #1325 (Qwen3-VL visual masks during chunked prefill): covered by
  the local PR #1332 backport above for the single-sequence source path; batched
  continuous prefill remains a separate proof row if a local regression appears.
- LFM2.5 VL live media rows still need real model proof after the PR #1328
  load shim. The implemented no-load patch only proves projector key/materialized
  module compatibility.

## Other-agent reminders

- Keep #192 on the list as a current fail-closed/source boundary. Current
  source tests prove missing required `cmd` in XML is rejected; they do not prove
  public installed/tunnel behavior.
- Do not replace model-owned metadata with hidden sampling/parser/cache defaults.
- Do not call release-ready from these patches. They are source/no-heavy fixes
  and must be followed by live model, installed app, raw SSE, cache/L2, and UI
  proof per family.
