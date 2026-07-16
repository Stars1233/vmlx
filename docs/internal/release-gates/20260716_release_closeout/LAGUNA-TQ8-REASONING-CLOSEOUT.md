# Laguna reasoning and uncalibrated Auto TQ closeout

Status: `PARTIAL`; release remains locked.

## Source trace

- `vmlx_engine/utils/jang_loader.py` generated 3-bit/4-bit TQ defaults for any
  legacy JANG bundle without a model-owned `turboquant` block.
- `vmlx_engine/utils/turboquant_config.py::apply_uncalibrated_auto_tq_policy`
  upgraded only recognized Qwen layouts to storage-only TQ8. Laguna therefore
  retained uncalibrated 3-bit storage despite having plain full-attention KV.
- `vmlx_engine/prefix_cache.py::compute_model_cache_key` keyed TQ persistence by
  a boolean plus stored q4/q8 selection, but not by TQ key bits, value bits,
  critical layers, seed, sinks, transition threshold, or Auto policy.
- The correction applies storage-only TQ8 to every uncalibrated configuration
  with real attention slots. Bundle-owned calibrated policy is unchanged.
  `turboquant_storage_signature` now fingerprints every codec field, and the
  persisted TQ namespace is bumped to `codec_config_v2`.

## Live Electron A/B

All rows used the running dev Electron app, Laguna-M.1-JANG_2L, reasoning on,
prefix cache on, paged cache on, disk cache on, and JIT on.

| Arm | Rows | Result |
|---|---:|---|
| Cold, old Auto/TQ3 | 1998 | Exact `FINAL=45`; no cache hit |
| Same-chat old Auto/TQ3 | 2001 | Restored 3,545 `paged+tq-native` tokens, emitted 9,597 chars of repeated/incoherent reasoning, no answer; manually interrupted after 3,076 generated tokens |
| UI None | 2004/2007/2010 | Cold exact; warm exact with 3,549 and 3,612 `paged` tokens; UI -> argv -> health confirmed TQ off while prefix/paged/disk stayed on |
| Corrected Auto/TQ8 | 2013/2016/2019 | Cold exact; warm exact with 3,550 and 3,614 `paged+tq-native` tokens; health reported `uncalibrated_full_kv_storage_tq8` and 8-bit key/value storage |
| Stop/Start disk restore | 2022 | Health recorded 3,550 `paged+disk+tq-native` tokens and 56 native disk hits. Output remained coherent, but the model made one unsolicited `ask_user` call; after visible Skip, the same Electron agent loop returned exact `TQ8-COLD1=45` |

The five-request streaming Responses soak with `skip_prefix_cache=true` reached
`response.completed` 5/5 with separated reasoning and visible answers. Each
visible answer was semantically exact but included leading/trailing newlines,
so strict byte formatting remains partial.

## Performance truth

- UI None paged hits: 1.18-1.46 s TTFT.
- Corrected Auto/TQ8 paged hits: 5.06-5.09 s TTFT.
- Health measured TQ reconstruction at 3.59-4.79 s for roughly 3,550 tokens.

Correctness is restored for the tested memory and disk cache paths, but the
TQ8 decode/reconstruction latency is still a release-performance row.

## Tests

- 112/112 passed across `test_hybrid_live_tq_kv.py`,
  `test_tq_paged_block_cache.py`, `test_tq_disk_cache.py`, and
  `test_cache_bypass.py`.
- Added direct contracts for Laguna uncalibrated TQ8 policy and cache namespace
  separation across codec bit settings.

## Remaining Laguna gates

- Repeat restart/disk exact-output rows without an unsolicited tool.
- Long-context and eviction/reload proof.
- Decide or optimize the measured TQ8 reconstruction latency.
- Complete strict Responses whitespace and full protocol coverage.
