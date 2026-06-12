# N2 JANG_1L lm_head6 Raw Decode Probe - 2026-06-11

Artifact:

`/Users/eric/jangq-ai/Nex-N2-Pro-JANG_1L-full-runtimefit-lmhead6-20260611`

## Result

Status: fail, output-quality blocked.

Raising `lm_head` to 6-bit did not fix the repeated space-token collapse. The
first Chat Completions probe and the cache-hit retry both generated token id
`220` eight times. Tokenizer decode with special tokens kept/skipped is eight
spaces in both cases.

Observed diagnostic excerpt:

```json
{
  "completion_tokens": 8,
  "decode_keep_specials_head": "        ",
  "decode_skip_specials_head": "        ",
  "finish_reason": "length",
  "raw_text_head": "        ",
  "text_head": "",
  "token_ids_count": 8,
  "token_ids_head": [220, 220, 220, 220, 220, 220, 220, 220]
}
```

## Probe Configuration

- Source vMLX server on port `8136`.
- `VMLINUX_FORCE_TQ_AUTO=1`.
- `--reasoning-parser none`.
- `--default-enable-thinking false`.
- Deterministic sampling: `temperature=0`, `top_p=1`.
- Paged cache + block disk cache enabled.
- Prompt: `Reply with exactly: blue cat`.

## Runtime Evidence

- Static artifact facts from shared lane:
  - `jang_tools validate`: `VALID`.
  - 123 shards, 2725 tensors.
  - Indexed bytes: `114471981680` (`106.61 GiB`), disk usage `107G`.
  - No expert pruning.
  - `lm_head` 6-bit; embeddings 4-bit.

- Loader/runtime:
  - JANG_1L affine quantized matmul path active.
  - Metal NA active on host.
  - Runtime detected Qwen3.6/N2 hybrid cache: 15 attention KV layers and 45 SSM
    companion layers.
  - Live TurboQuant KV active for attention layers only.
  - Storage-boundary KV quantization q4 active for prefix/paged/L2.
  - Quant-shape inference patched 377 modules because top-level config still
    claims uniform `bits=2 group_size=128` while tensor shapes encode mixed
    precision.

- First Chat response:
  - HTTP 200.
  - `message.content=null`.
  - `completion_tokens=8`.
  - `finish_reason=length`.
  - Raw token ids: eight `220` tokens.
  - Speed: 8 tokens in 44.10s, `0.2 tok/s`.

- Cache-hit Chat retry:
  - HTTP 200.
  - `cached_tokens=10`, `cache_detail=paged+ssm+tq`.
  - Raw token ids: eight `220` tokens.
  - Speed: 8 tokens in 17.86s, `0.4 tok/s`.
  - Health after retry showed cache reconstruction OK:
    `reconstructed=true`, `dequantized=true`, `reconstruction_seconds=0.084994`.
  - L2 totals: 10 block tokens on disk, 10 SSM companion tokens on disk.

- Memory after retry:
  - Active: `108494.8 MB`.
  - Peak: `109629.5 MB`.
  - Cache: `204.1 MB`.

## Classification

This artifact is not healthy. The 6-bit `lm_head` one-variable test did not
change the first-token/output path: greedy decode still selects repeated token
`220`, which decodes to spaces before Chat/Responses assembly.

This remains not a Chat Completions assembly bug, not tokenizer
special-token filtering, and not reasoning-parser suppression. Cache reuse is
also not the current output-quality root cause: the cache-hit path used
`paged+ssm+tq`, reconstructed/dequantized successfully, and still generated the
same repeated token.

The next most direct JANG-side tests are source/high-bit first-token logits and
top-k comparison for this exact prompt, then a one-variable artifact that raises
the next logits-sensitive or early-path precision boundary. Since both
linear-attn input 4-bit restoration and `lm_head` 6-bit failed, those are not
sufficient explanations by themselves.

## Cleanup

The vMLX server on port `8136` was stopped after the probe. No intentional N2
vMLX probe server remains from this run.
