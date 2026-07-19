# Non-paged prompt-disk partial-prefix proof

Status: `PARTIAL` for the parent gate; `VERIFIED-LIVE` for the plain full-KV
representative on source head `727da2e44adb2c88e2a714fd45bfa45d9e0d7b1b`.

## Artifact and cache identity

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`
- `config.json`: `model_type=minimax_m2`, 62 layers, 48 attention heads and 8
  KV heads.
- `jang_config.json`: `weight_format=mxtq`, profile `JANGTQ2`, routed-expert
  bits 2 and attention/dense/embed/head bits 8. This is JANGTQ/MXTQ, not JANG
  affine and not base MLX MXFP.
- Health selected generic plain-attention q4 TurboQuant storage:
  `mode=turboquant-storage`, `bits=4`,
  `stored_prefix_quantization=turboquant-q4`, live encode Off.
- MiniMax M2.7 is text-only. No VL claim is made.

## Source trace

- `panel/src/shared/cacheControlPolicy.ts`: paged and legacy prompt-disk L2 are
  mutually exclusive, while explicit paged Off keeps prompt disk eligible.
- `panel/src/main/sessions.ts`: persists the settings and maps the effective
  policy to `--no-paged-cache`, `--enable-disk-cache` and the selected prompt
  cache directory.
- `vmlx_engine/cli.py`: forwards prompt disk independently of paged/block L2
  into `SchedulerConfig`.
- `vmlx_engine/disk_cache.py::fetch_longest_prefix`: finds only stored lengths
  that can prefix the current token sequence and delegates the winning hash to
  the validated/TQ-aware loader.
- `vmlx_engine/scheduler.py::_disk_prefix_hit_tail_and_cached_tokens`: observes
  the N-1 snapshot contract by refeeding the final matched token and only the
  unmatched tail.
- `vmlx_engine/scheduler.py` disk lookup calls `fetch_longest_prefix` before
  falling back to exact lookup, and marks actual native packed restores.

## Electron UI and process proof

The real session UI was used for every setting change and every process
Stop/Start. `m27-nonpaged-ui-config.png` visibly shows Paged Off, Block Disk L2
Off, Prompt Disk On and stored-codec Auto. `m27-session-db.json` preserves the
saved session row. `m27-argv.txt` proves the running process received:

```text
--no-paged-cache --enable-disk-cache
--disk-cache-dir /Users/eric/.cache/vmlx-engine/live-proof-m27-nonpaged-partial-20260719
```

The UI Start swap stopped Bonsai and left one engine process. Before any
request, health had `last_request_time=null`, paged false, block disk false,
prompt disk entries visible after restart, and q4 TQ storage active.

The first Electron request had 2,241 prompt tokens and returned exactly
`M27-NP-BASE-DONE` with reasoning held separately. Its asynchronous writer was
allowed to reach `pending_writes=0`; two q4 TQ-native prompt records totaling
4,476 indexed tokens existed before stopping PID 44250.

After UI restart to PID 45185, the longer same-chat turn returned exactly
`M27-NP-PARTIAL-DONE VALUE-84873`. The row and screenshot show:

```text
2310 prompt (2235 disk+tq+tq-native cached)
0.88s TTFT
```

The in-app log independently reports:

```text
TQ-native disk cache loaded: 62 layers ... 2236tok_tq.safetensors
Disk cache prefix hit: matched 2236/2305 prompt tokens
```

The one-token difference is the intentional N-1 payload contract: the final
matched prompt token is re-fed with the 69-token unmatched tail. Health records
one prompt-disk hit and one TQ-native hit, with zero block-disk tokens.

## Raw API streaming proof

`m27-api-base.sse` contains a fresh streamed Responses base turn. It emitted
226 reasoning-summary deltas and 16 content deltas, then output-text done and
response completed. Its 1,393-token q4 TQ-native boundary was fully flushed to
disk before UI restart.

After UI restart to PID 45913, `m27-api-partial.sse` used the same prior
user/assistant history plus a new user turn. It restored 1,392 of 1,458 input
tokens from disk, emitted 124 reasoning-summary deltas and 26 content deltas,
returned exactly `M27-API-PARTIAL-DONE CODE-74985`, then emitted output-text
done and response completed. The in-app log reports `matched 1393/1453`.

After another UI restart to PID 46340, `m27-chat-partial.sse` crossed the same
persisted boundary through Chat Completions. It restored 1,392 of 1,460 prompt
tokens, emitted 58 reasoning deltas and 14 progressive content deltas, returned
exactly `M27-CHAT-PARTIAL-DONE CODE-14275`, finished `stop`, and emitted
`[DONE]`. The log reports `matched 1393/1455`.

## Current test and gate truth

Focused selection:

```text
tests/test_disk_prefix_n_minus_1.py
tests/test_minimax_m3_cache_paths.py
tests/test_cache_bypass.py
```

The first run exposed one stale constructor-bypass fixture:
`test_scheduler_uses_minimax_m3_logits_sampler_for_msa_cache` used
`object.__new__(Scheduler)` without declaring the no-RAM/no-disk cache state
now read by snapshot admission. The fixture now explicitly sets
`memory_aware_cache=None` and `disk_cache=None`. The identical selection reran:

```text
107 passed in 8.71s
```

See `focused-tests.txt` for the current-source output.

Still required before the parent gate closes:

- architecture-specific typed-cache partial restart restore (openPangu v2),
- paged-On block-aligned partial reuse,
- forced RAM eviction followed by block-L2 refault,
- current-source rerun after any resulting fix,
- scoped commit/push and an honest matrix update.
