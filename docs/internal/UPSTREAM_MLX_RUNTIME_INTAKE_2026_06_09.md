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

## Upstream items checked but not blindly ported

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
- `mlx-vlm` PR #1336 (Gemma4 materialized unused shared-KV weights): likely
  relevant to mlx-format Gemma4 checkpoints, but local Gemma4 text/vendored
  model shape differs. Needs a load-weight key regression before patching.
- `mlx-vlm` PR #1313/#1334 (Qwen3.5/3.6 quantized KV and MTP prefill): relevant
  to Qwen3.6/MTP/gdn_sink lanes, but requires local mapping against
  `utils/mlx_vlm_compat.py` and current Qwen VLM language source before port.

## Other-agent reminders

- Keep #192 on the list as a current fail-closed/source boundary. Current
  source tests prove missing required `cmd` in XML is rejected; they do not prove
  public installed/tunnel behavior.
- Do not replace model-owned metadata with hidden sampling/parser/cache defaults.
- Do not call release-ready from these patches. They are source/no-heavy fixes
  and must be followed by live model, installed app, raw SSE, cache/L2, and UI
  proof per family.
