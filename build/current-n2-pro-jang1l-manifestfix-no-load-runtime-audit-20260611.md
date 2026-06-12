# N2 Manifestfix No-Load Runtime Audit - 2026-06-11

Artifact:

`/Volumes/EricsLLMDrive/jangq-ai/Nex-N2-Pro-JANG_1L-manifestfix-20260611`

Source:

`/Volumes/EricsLLMDrive/jangq-ai/sources/Nex-N2-Pro`

## Result

This was a no-weight-load audit of routing, prompt rendering, EOS/media metadata,
and mRoPE preservation.

- Source `is_mllm_model`: `true` via `config_json_vision_config`.
- JANG affine artifact `is_mllm_model`: `false` via
  `affine_qwen_hybrid_jang_text_only`.
- Registry family: `qwen3_5_moe`.
- Registry cache: `hybrid`.
- Registry parsers: tool `qwen`, reasoning `qwen3`.
- Registry `think_in_template`: `true`.

## Prompt / Thinking Template

Prompt:

`Reply with exactly: blue cat`

Source and artifact raw prompt ids match. Source and artifact rendered chat
template ids also match for `enable_thinking=false`.

Rendered artifact suffix with thinking disabled:

```text
<|im_start|>user
Reply with exactly: blue cat<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The first-token logits proof therefore scored the first visible token after a
closed empty think block, not a raw instruction-completion prompt.

## EOS / Tokens

- Artifact `config.eos_token_id`: `248046`.
- Artifact `text_config.eos_token_id`: `248046`.
- Tokenizer maps `248046` to `<|im_end|>`.
- Tokenizer maps `248044` to `<|endoftext|>` / pad in the source note.

## mRoPE / VL Metadata

Artifact preserves:

- `mrope_interleaved=true`
- `mrope_section=[11, 11, 10]`
- `partial_rotary_factor=0.25`
- `rope_theta=10000000`
- `max_position_embeddings=262144`
- `image_token_id=248056`
- `video_token_id=248057`
- `vision_start_token_id=248053`
- `vision_end_token_id=248054`

However, this artifact is routed text-only by vMLX today. Preservation of these
fields is not proof that the affine JANG artifact consumes the MLLM/VL mRoPE
runtime path.

## Media Capability Boundary

The artifact has `vision_config` and JANG capability metadata, but vMLX routes
plain affine Qwen hybrid JANG through text-only mode. This is intentional in
`model_config_registry.py` for `_is_affine_jang_qwen_hybrid_vlm(...)` unless a
native indexed MTP/VL route is available.

Do not claim N2 media healthy from this artifact. The honest state is
preserved/unwired metadata for current affine JANG text runtime.

## Proof Files

Full JSON:

`build/current-n2-pro-jang1l-manifestfix-no-load-runtime-audit-20260611.json`
