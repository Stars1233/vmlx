# Qwen 3.6 35B hybrid v8 cache audit — 2026-07-16

Status: cache tiers `PASS-LIVE`; model long-format reliability `PARTIAL`.

Model: `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`.
This official dealignai quantized artifact is treated as a trusted input.

## Source trace

- Real runtime graph: 40 cache layers, with attention KV at
  `3,7,11,15,19,23,27,31,35,39` and 30 GDN/SSM companion layers.
- Auto storage policy: native TurboQuant q8 for the 10 attention KV layers;
  live decode encoding remains off (`compress_after=0`), so there is no false
  live-resident-memory reduction claim.
- Companion policy: native full-precision typed SSM state plus clean-prefill
  rederive on a missing boundary.
- `7bb34fa0d`: release disk-promoted arrays and resident-byte attribution
  atomically; reset payload-scoped `keep_resident` on block reuse.
- `df945f065`: route generic hybrid terminal cumulative state to the typed
  companion store and label attention-only TQ as selective in the UI.
- `133d8c8e9`: fix the NumPy disk-writer branch that live safetensor inspection
  showed was still writing cumulative state.
- `7cb89185c`: bump the persisted namespace to v8 so malformed v7 files cannot
  replay.

## Persisted v8 payload inspection

Directory: `/Users/eric/.cache/vmlx-engine/block-cache/f16751f34ea4`.

| Tokens | File bytes | Tags |
|---:|---:|---|
| 64 | 712,584 | 10 `turboquant_kv`, 30 `skip` |
| 64 | 712,584 | 10 `turboquant_kv`, 30 `skip` |
| 64 | 712,584 | 10 `turboquant_kv`, 30 `skip` |
| 64 | 712,584 | 10 `turboquant_kv`, 30 `skip` |
| 2 | 30,345 | 10 `turboquant_kv`, 30 `skip` |
| 64 | 712,584 | 10 `turboquant_kv`, 30 `skip` |
| 64 | 712,584 | 10 `turboquant_kv`, 30 `skip` |
| 26 | 294,550 | 10 `turboquant_kv`, 30 `skip` |

No v8 file contains a `cumulative` entry. For comparison, the two malformed
v7 terminal files containing 30 cumulative entries were 64,427,560 and
64,691,660 bytes. The files remain outside the v8 namespace as retained
failure evidence.

## Live Electron tier rows

| Row | PID / setting | Result |
|---:|---|---|
| 2169 | PID 60796, cold v8 namespace | No cached-token metric; full prefill, one real `file_info`, exact `Q36J-V8-COLD-DONE` |
| 2172 | PID 60796, same process | 154/155 `paged+ssm`; one real tool, exact final |
| 2175 | PID 61067 after visible Stop/Start | 154/155 `paged+ssm+disk`; one real tool, exact final |
| 2178 | PID 61405, UI Save & Restart with Max Cache Blocks 4 | 154/155 `paged+ssm+disk`; one real tool, exact final |
| 2181 | PID 61405 after bounded eviction | 154/155 `paged+ssm+disk`; one real tool, exact final |
| 2184 | PID 61919 after UI restored Max Cache Blocks 1000 | 154/155 `paged+ssm+disk`; one real tool, exact final |

The four-block Cache drawer showed nine L1 evictions, one block allocated out
of four, 16 block-disk hits, selective q8 attention KV L2, and a safe fallback:
192 reusable KV-only tokens had no matching SSM companion, so the scheduler
full-prefilled instead of reporting a false hybrid hit. After the gate, both
argv and health again reported 1,000 blocks.

## Current tests

- 595/595 engine-audit and byte-budget tests passed for resident accounting.
- 177/177 paged/disk/TQ/hybrid tests passed for the first repair.
- 784/784 current hybrid prefix, companion, TQ block, MLLM scheduler, and
  engine-audit tests passed after external companion ownership.
- 278/278 panel settings tests and TypeScript typecheck passed.

## Retained partial rows

- The ambiguous tools-on long prompt attempted an invalid `write_file` call.
- The clarified direct-display long prompt streamed through its final marker
  but missed requested periods.

Those are strict-format/reliability evidence and remain open; they do not
invalidate the now-complete cache tier proof and are not attributed to the
official quant artifact without an owning runtime diagnosis.
