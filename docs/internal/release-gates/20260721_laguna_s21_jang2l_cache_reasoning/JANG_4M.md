# Laguna S-2.1 JANG_4M q4/L2 dtype and performance gate

Date: 2026-07-21

Host: `erics-m5-max.local` (Apple M5 Max)

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Artifact: `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_4M`

Electron profile: `/Users/eric/.vmlx-v1613-responsive-dev`

## Defect and owning fix

The JANG_4M cache hierarchy had a real restart-only performance defect. Cold
and same-process paged-q4 turns decoded near 50 tokens/s, while a process
restart that restored the same prefix from block-disk L2 decoded near
16 tokens/s.

`BlockAwarePrefixCache` uses float32 NumPy arrays as the CPU bridge for cache
states whose original dtype is bfloat16. `_numpy_block_slice` previously
ignored the saved original dtype for ordinary positional and rotating cache
states. The block writer therefore persisted Laguna's 36 rotating-SWA layers
as float32. The 12 full-attention layers were stored separately as native q4
TurboQuant records and remained bfloat16.

The fix in `vmlx_engine/prefix_cache.py::_numpy_block_slice` restores the
model-owned dtype before returning disk payloads for ordinary positional KV,
`RotatingKVCache`, and MiniMax-M3 native positional arrays. The quantized-KV
NumPy source path now also retains the dtype captured before its float32
bridge. It does not add a Laguna-only decode branch, prompt rewrite, output
repair, sampler clamp, or synthetic reasoning behavior.

`tests/test_hybrid_prefix_cache.py::test_numpy_disk_writer_restores_bfloat16_positional_dtypes`
pins both plain KV and rotating-KV restoration, including rotating metadata.
The focused cache suites pass 34/34.

## Current disk format

The stale model-specific cache directory was removed only after stopping the
real Electron session. A new cold Electron turn wrote two blocks (64 and 29
tokens). Direct safetensors-header inspection shows, for each block:

- 48 cache layers total.
- 12 `turboquant_kv` full-attention layers at indexes
  `0,4,8,...,44`.
- q4 key and q4 value storage with bfloat16 source dtype.
- 36 `rotating_kv` SWA layers.
- every rotating key/value tensor is `BF16`, with original dtype recorded as
  `mlx.core.bfloat16`.

The running `/health` contract agrees: `mixed_swa_kv_v1`, q4 storage applied
to `full_attention_kv_only`, native rotating state preserved, and restore
policy `decode_full_attention_tq_and_restore_rotating_state`.

## Real Electron results

All three turns used the exact prompt marker
`LAG-S21-4M-Q4-DTYPE-FIX-DONE 45` and persisted non-empty visible content,
separate reasoning, no warning, and no tool payload.

| Row | Cache path | Cached | Decode | TTFT | Visible result |
|---|---|---:|---:|---:|---|
| 371 | cold | 0/94 | 51.3 tok/s | 0.65 s | exact marker |
| 374 | paged RAM + native q4 | 93/94 | 51.7 tok/s | 0.56 s | exact marker |
| 377 | process-restart block L2 + native q4 | 93/94 | 48.9 tok/s | 0.47 s | exact marker |

The real Stop/Start controls replaced the engine and eagerly loaded PID 5317
before a request. Its argv restored JIT, paged RAM, 64-token blocks, 1000
blocks, 15% cache RAM, and 10 GB block L2. Before the restart turn health had
zero RAM cache tokens and the two 93-token L2 blocks. After the turn it
recorded two disk promotions, two native-TQ hits, successful reconstruction,
and 93 promoted RAM tokens.

The 1,399-token restart turn produced a 4,087-character reasoning rail. It was
long and repetitive, but it streamed through to the exact visible final at
48.9 tok/s without a truncation warning. This is retained as a model-quality
observation; it is not rewritten or misreported as a short-reasoning pass.

Evidence:

- `jang4m-ui-db-rows.json`
- `jang4m-health-after-l2.json`
- `jang4m-block-format.json`
- `laguna4m-q4-dtype-ram-pass.png`
- `laguna4m-q4-dtype-l2-pass.png`

## Paged-Off SSD-only result

The real Server drawer turned `Use Paged KV Cache` Off while leaving Block
Disk Cache L2 and q4 Auto enabled, then applied the change with Save & Restart.
PID 7217 eagerly loaded with `last_request_time=null`. Before generation,
health reported `backend_mode=block_disk_only`, `paged_ram_enabled=false`, two
existing L2 blocks/93 tokens, and zero resident cache bytes.

Electron row 380 restored 93/94 prompt tokens from those SSD blocks as
`block-disk+tq-native`, reconstructed two disk blocks/24 full-attention TQ
layer-block entries, retained `l1_resident_bytes=0`, and decoded at 50.7 tok/s
with 0.45-second TTFT. This is both partial-block reuse (64 + 29 tokens) and
full SSD-only payload reuse without a paged-RAM payload mirror.

The visible output added `18 + 27 = 45` before the requested marker. Cache,
stream completion, and non-empty visible output pass; strict exact formatting
fails for this stochastic turn. The failure is preserved in the screenshot
and row JSON and was not rewritten.

The same real drawer then restored Paged On and Save & Restart created PID
7739. Final health has `backend_mode=paged`, `paged_ram_enabled=true`, q4 Auto,
block L2 still enabled with both 93-token blocks, zero RAM tokens before any
new request, and `last_request_time=null`.

Additional evidence:

- `jang4m-disk-only-health.json`
- `jang4m-disk-only-ui-row.json`
- `laguna4m-diskonly-result.png`
- `laguna4m-final-paged-restored.png`

## Verdict and remaining Laguna work

`VERIFIED-LIVE_SCOPED` for the JANG_4M q4 mixed-SWA storage format, same-process
RAM reuse, process-restart L2 restore, dtype correctness, eager load, saved
cache settings, exact visible completion, and the restart decode regression.

Still open: long context past the 512-token SWA boundary,
Chat/Anthropic/Ollama tool-result continuations, broader settings/sampling
identity, and longer agentic quality soak. Cross-family Chat/Responses
reasoning-content semantics remain a separate release gate and are not
inferred from these Electron DB rows.
