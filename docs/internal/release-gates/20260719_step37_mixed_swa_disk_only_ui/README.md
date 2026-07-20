# Step 3.7 full/sliding-KV block SSD without paged RAM

Date: 2026-07-19

Base source: `782abb33aefb0dae3409470c8c87ef4823b7cc1a` plus the scoped Step panel diff in `source.diff`.

Verdict: `VERIFIED-LIVE_SCOPED_SHORT_PREFIX`; long tight-memory cold-prompt store remains `PARTIAL`. This is not a release-ready or family-wide claim.

## Bundle and architecture truth

Artifact: `/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANGTQ_K`.

- format/profile: JANGTQ / `JANGTQ_K` (not affine JANG and not base MXFP);
- top-level `model_type=step3p7`, text model `step3p5`;
- 45 attention layers in a 1:3 full/sliding pattern;
- sliding window 512;
- no MTP tensors in this artifact;
- tool parser `step3p5`, reasoning parser `qwen3`;
- runtime cache schema `mixed_swa_kv_v1`, subtype `step3p7_full_sliding_kv`, with full-attention KV, sliding-window KV, and rotating metadata.

The panel now permits explicit Paged Off when Prefix Cache and Block Disk L2 are on for this exact subtype. The UI includes the tight-headroom limitation rather than implying every long cold prompt will be stored.

## Gateway one-model and eager-load proof

With Gateway single-model mode visibly enabled, clicking the real Step `Start` card:

- stopped Gemma PID 59856 and cleared its session PID/status;
- eagerly loaded Step PID 60732 in 33 seconds;
- produced `model_loaded=true` with `last_request_time=null` before any prompt;
- left exactly one local `vmlx_engine.cli serve` process.

A later full Electron-main relaunch found `/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine` and the real Step Start path eagerly loaded PID 61575.

## SSD-only launch proof

After a full Electron-main relaunch with the patched session manager, the real settings drawer showed Paged enabled as a user control. The drawer persisted Paged Off + Block L2 On + quantization None and `Save & Restart` launched PID 62212 with:

```text
--no-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000
--kv-cache-quantization none
--enable-block-disk-cache --block-disk-cache-max-gb 10
```

Before the first request health reported `backend_mode=block_disk_only`, `paged_ram_enabled=false`, `l1_resident_bytes=0`, no SSD entries in the qnone partition, `native_cache.schema=mixed_swa_kv_v1`, and TurboQuant disabled.

## Long-prompt limitation

The 1,343-token cold Electron prompt returned exact visible `STEP-SSDONLY-QNONE-A-DONE` with 38 progressive DOM mutations and separate reasoning, but wrote zero cache blocks. The live log is explicit:

```text
Skipping mixed-SWA VLM paged cache store ... tight-memory clean prompt prefill disabled to avoid Metal OOM (prompt_tokens=1343)
```

Step occupied enough of the Metal working set that the post-terminal clean prompt re-prefill guard rejected the second long prefill. No unsafe force environment override was used. This long-cold-store row remains partial.

## Bounded Electron SSD proof

A safe 136-token prompt produced the following real UI rows:

- cold A: exact visible `STEP-SSD-SHORT-A-DONE`, separate reasoning, 52 DOM mutations, 135 indexed tokens, three SSD block writes, zero resident L1 bytes, and zero TQ activity;
- exact warm A: `135/136 block-disk+mixed_swa cached`, exact visible output, and 49 DOM mutations;
- changed-tail B: records 001-006 stayed identical while 007-008 changed; it restored exactly one complete 64-token block out of 138, returned exact `STEP-SSD-SHORT-B-DONE`, and produced 82 DOM mutations;
- restart A: PID 62212 was later replaced by PID 63165. Before its first request the new process had zero L1 indexed/resident tokens and 208 SSD tokens. The first Electron turn restored `135/136 block-disk+mixed_swa cached`, produced three disk promotions, zero writes, and exact progressive content.

The visible answer was non-empty on every turn. Reasoning was separate and prompt-specific; no reasoning replay, parser marker leak, tool call, warning, loop, or truncation was observed.

## Raw Responses proof

The first attempt with `enable_thinking=false` returned HTTP 400 because this artifact advertises no native thinking-off/instruct route. The passing test used supported `reasoning_effort=low`, tools disabled, and a 512-token output budget.

- exact A: 102 reasoning-summary deltas, 9 content deltas, one text-done, one completed terminal, exact visible marker, 26.8 ms last-content-to-completed gap;
- partial B: 85 reasoning-summary deltas, 9 content deltas, one text-done, one completed terminal, exact visible marker, 26.5 ms terminal gap.

## Restored default policy

After proof, the real settings drawer restored Paged On + Auto and restarted as PID 64768. Effective argv contains `--use-paged-cache` and no explicit quant flag. Health before a request reports paged RAM plus stored-prefix `turboquant-q4`, 4-bit full/sliding KV storage, and preserved rotating metadata.

## Validation

- Panel cache/settings selection: 300 passed.
- Panel TypeScript typecheck: passed.
- Python mixed-SWA/disk-only scheduler selection: 7 passed.
- Python cache-detail/native-schema selection: 7 passed.
- `git diff --check`: passed.

These are focused tests. The full Python/panel/build release gate remains open.

## Remaining

- Design/prove a long-prompt store path that does not require an unsafe second clean prefill, or retain the current fail-safe skip.
- Repeat Chat Completions, Anthropic, Ollama, tool continuation, and media cache-salt rows on current source.
- MiMo is blocked because no runnable MiMo bundle is present on the active drive; only old converter/runtime source was found.
