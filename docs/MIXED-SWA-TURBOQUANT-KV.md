# Mixed-SWA per-layer TurboQuant KV (`VMLX_SWA_TQ`)

Status: **opt-in, default off.** Introduced in v1.6.4.

## What it does

Mixed sliding/full attention models (gemma-4, Laguna, Step-3.7) previously skipped
TurboQuant KV entirely. A flat TQ wrap replaces *every* cache slot, which destroys the
`RotatingKVCache` sliding-window metadata that the `mixed_swa_kv_v1` cache contract
depends on.

Only the **full-attention** slots want TQ. Their KV grows with the whole context, so
that is where the memory is. The sliding slots are capped at a 512-token window, and
`TurboQuantKVCache` is monotonic-growth (no `max_size` / `keep` / trim), so it cannot
honour a window at all.

With `VMLX_SWA_TQ=1` the loader assigns, per slot:

| slot kind          | cache class          |
|--------------------|----------------------|
| `full_attention`   | `TurboQuantKVCache`  |
| `sliding_attention`| `RotatingKVCache` (native, untouched) |

Observed layouts:

| model | slots | full → TQ | sliding → rotating |
|---|---|---|---|
| gemma-4-12B-it-qat | 48 | 8 | 40 |
| gemma-4-26B-A4B-it-qat (MoE) | 30 | 5 | 25 |
| gemma-4-31B-it-qat | 60 | 10 | 50 |
| Step-3.7-Flash-JANG_K | 45 | 12 | 33 |

### Guard

`is_mixed_swa_tq_supported()` requires exactly one declared `layer_type` per native cache
slot. Without a 1:1 mapping we cannot tell which slot is the unbounded one, so TQ stays
off. This is why `gemma-4-E2B/E4B` (35/42 declared layer types, 15/18 cache slots) are
never eligible.

## Which models actually reach this path

Reaching it requires `jang_loader._patch_turboquant_make_cache`, which not every loader
calls, and the earlier skip gates must not fire first.

| family | reaches SWA-TQ? | why |
|---|---|---|
| gemma-4 12B / 26B-A4B / 31B | **yes** | verified live |
| gemma-4 E2B / E4B | no | `is_mla_model` gate fires first (`gemma4_unified`), before the SWA gate |
| Laguna-XS.2 (mxfp4, JANGTQ) | no | loads through `jang_tools/laguna/runtime.py`, which never calls the TQ patch |
| Step-3.7-Flash | **yes** | verified live (12 full / 33 sliding) |

## Why it is not the default

`TurboQuantKVCache` **never compresses during decode.** `compress_after` defaults to `0`,
`TurboQuantConfig` has no such field, and neither `make_turboquant_cache` nor
`build_hybrid_turboquant_make_cache` passes one. A TQ cache therefore holds plain float KV
and behaves like a `KVCache` with extra machinery. The only `.compress()` call in the tree
is `_recompress_to_tq`, reached from paged-cache and disk-L2 reconstruction — and the paged
cache is off by default.

Consequence: today the flag costs nothing and buys nothing **in memory**. Outputs are
byte-identical to baseline. Making the encode real means wiring `compress_after`
end-to-end, which **will change decode numerics for every TQ family** (MiniMax-M2.7's 62
TQ layers, Qwen3.6's every-4th, and these gemma slots) and therefore needs a per-family
coherence gate before it can be turned on.

It does buy one thing today, as a side effect rather than by design: on the models whose
full-attention slots become TQ, the F1 cold≠warm divergence disappears (see Determinism).
That is not a reason to default it on — it is a pointer at the real F1 fix, which is to
stop the lossy q4 round-trip on stored `KVCache` prefixes (task #45).

## Determinism

Verified live, A = default vs B = `VMLX_SWA_TQ=1`, greedy (`temperature=0`).

| check | gemma-4-12B | gemma-4-26B-A4B (MoE) | Step-3.7-Flash |
|---|---|---|---|
| decode determinism (3× cold, distinct salts) | PASS both | PASS both | PASS both |
| cross-flag equality (A ≡ B) | PASS (byte-equal) | PASS (byte-equal) | PASS (byte-equal) |
| prefix-cache pollution guard | PASS both | PASS both | PASS both |
| multiturn recall over prefix hits | PASS both | PASS both | PASS both |
| image / image-stream / image-multiturn | PASS both, byte-equal | PASS both, byte-equal | n/a (text) |
| audio (no audio tower) | rejected `400`, correct | rejected `400`, correct | n/a |
| cold == warm | A: **F1**, B: **byte-faithful** | F1 on both | A: **F1**, B: **byte-faithful** |
| co-batch equivalence (`max-num-seqs=2`) | PASS | **flaky on both** (A 6/12, B 9/12 mismatch) | not re-run (covered by F18) |

Step-3.7's 74 GB bundle cannot host two servers on a 128 GB box, so A and B were run
sequentially on one port each and their raw outputs diffed offline
(`step_seq.py` → `step_cmp.py`) rather than side by side.

Step-3.7 independently reproduces the gemma-4-12B signature: byte-equal to baseline, and
`VMLX_SWA_TQ=1` *removes* F1. That is a third model confirming F1 is the q4 stored-prefix
round-trip on `KVCache` slots — in both models the flag's warm answer equals the baseline's
**cold** answer, i.e. the cold answer is the faithful one and baseline's warm answer is the
q4-degraded one.

Two pre-existing issues surfaced here. Neither is caused by this feature — both reproduce
on the default path with the flag off:

- **F1 (cold ≠ warm):** a prefix-cache hit does not byte-reproduce the fresh answer, though
  warm is itself stable. It is the q4 stored-prefix round-trip. Interestingly, on
  gemma-4-12B `VMLX_SWA_TQ=1` *removes* it: TQ slots are not `KVCache`, so
  `_quantize_cache_for_storage` leaves them alone and the clone rebuilds them exactly.
- **gemma-4-26B-A4B co-batch nondeterminism:** at `max-num-seqs=2` the greedy answer for a
  prompt differs solo vs co-batched, intermittently, on the default path. Consistent with
  MoE expert-routing flipping on near-ties under batched matmul numerics.

Note when measuring: `cache_salt` changes the cache **key**, so a salted request is always
cold and never reuses. Comparing a salted "cold" against an unsalted "warm" is vacuous.
Use a unique unsalted prompt and call it three times: call 1 is the miss, calls 2–3 are hits.

## Prefix-cache safety

Enabling this exposed a real corruption, fixed in the same release. `TurboQuantKVCache` is
not a `KVCache` subclass, so `MemoryAwarePrefixCache._clone_cache_for_fetch._safe()`
rejected it and the whole layer list fell through to the **stored-reference** return. TQ is
monotonic-growth, so decode appended directly into the cached entry: the stored 18-token
entry's offset walked 18 → 34 → 50 → 66 while its key length stayed 18, and a later
"What is the capital of France?" hit replayed the polluted prefix and answered "Berlin."

`_truncate_cache` now rebuilds TQ layers as fresh independent caches. Pure-TQ families
(M2.7, Qwen3.6) never surfaced this because the batched extract materializes TQ to float
before storage, so their stored entries are `QuantizedKVCache`. Only a *mixed* layer list
stores a live TQ object. Regression test: `tests/test_turboquant_prefix_cache_clone.py`.
