# Bonsai MLLM q8 TQ startup warmup - 2026-07-19

Status: VERIFIED-LIVE for the scoped MLLM q8 attention-KV + native SSM
companion restore path. Strict-format reliability remains PARTIAL; this is not
a release-wide claim.

## Source trace

- The initial text-scheduler warmup did not execute for `--is-mllm` routes.
  Bonsai uses `BatchedEngine._start_mllm()` and `MLLMScheduler`, not the text
  `Scheduler` path.
- `vmlx_engine/engine/batched.py` now invokes the MLLM scheduler warmup on the
  same pinned executor used for model load and every MLLM step, before the
  scheduler advertises readiness.
- `vmlx_engine/mllm_scheduler.py` resolves the actual patched language-model
  `make_cache` owner and calls the shared storage-codec warmup only for its
  `TurboQuantKVCache` slots. It does not encode, replace, or synthesize the
  `ArraysCache` SSM companion slots.
- Explicit TQ/Prefix Off remains honored: the warmup exits when `_tq_active`
  or prefix cache is false.

## Current-source tests

- `pytest tests/test_tq_paged_block_cache.py tests/test_mllm_scheduler_cache.py tests/test_hybrid_live_tq_kv.py -q`
- Result: 143 passed, with two third-party librosa deprecation warnings.

## Live Electron / cache evidence

- Real Electron Start loaded
  `/Volumes/EricsLLMDrive/jangq-ai/Bonsai-27b-1bit-JANG` as the only local
  serve process (PID 42444, port 8030).
- Bundle/runtime truth: `JANG_AFFINE_1BIT`, `actual_bits=1.1128`, not JANGTQ
  or base MLX MXFP. Cache Auto remained the Bonsai-specific q8 policy.
- In-app Logs showed a 64-layer typed layout with exactly 16
  `TurboQuantKVCache` attention slots and 48 `ArraysCache` companion slots.
- In-app Logs then reported:
  `MLLM TurboQuant storage decoder startup warmup: configs=16 arrays=128 bytes=8470528 codec_probes=16 probe_tokens=64 probe_heads=4 seconds=1.085`.
- Fresh-chat exact replay restored 74 tokens as `paged+ssm+disk`, TTFT 0.21s.
  Health recorded eight q8 TQ-native block hits, one SSM companion disk hit,
  zero unsafe KV-without-SSM reuse, and q8 K/V storage.
- The no-tool visible answer was non-empty and reasoning was separate, but the
  stochastic model repeated `B1-CONTENT-PROGRESSIVE` after the requested
  terminal marker. This is retained as a strict-format/ramble failure.
- Same-chat Electron tool turn executed exactly one real
  `file_info(panel/package.json)`, received 5.2 KB, and exact-finaled
  `B1-CURRENT-TOOL-DONE SIZE=5.2 KB` with no warning.

## Raw API streaming evidence

- Temperature-zero Chat: 256 reasoning deltas, 10 progressive content deltas,
  exact sentence `31 times 9 is indeed 279.`, one `stop`, and `[DONE]`.
- Temperature-zero Responses: 256 reasoning-summary deltas, 10 output-text
  deltas, matching output-text done, and one `response.completed`.
- Required-tool Responses: 62 reasoning deltas, two progressive function
  argument deltas reconstructing exactly
  `{"path": "panel/package.json"}`, and one completed terminal.

## Artifacts

- `bonsai-start-health.json`
- `bonsai-q8-ssm-replay-health.json`
- `bonsai-q8-ssm-replay-row.json`
- `bonsai-q8-ssm-replay-format-partial.png`
- `bonsai-current-tool-row.json`
- `bonsai-current-tool-pass.png`
- `bonsai-current-api-streams.json`

Classification: q8 cache ownership, MLLM startup warmup, SSM companion restore,
Electron tool continuation, and raw delta/terminal transport pass for these
rows. Stochastic exact-output/ramble behavior remains PARTIAL and is not
hidden by parser, prompt, sampler, or output rewriting.
