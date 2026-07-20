# Gemma 4 mixed-SWA block SSD without paged RAM

Date: 2026-07-19

Base source: `4343e8913ef9132ebc313dc94c74281571ea282a` plus the scoped panel policy diff in `source.diff`.

Verdict: `VERIFIED-LIVE_SCOPED`. This closes the Gemma 4 mixed-SWA manual-toggle row for unquantized Block Disk L2 with Paged KV Cache off. It does not close Step/MiMo, media, the remaining protocol matrix, the full test/build gate, or release readiness.

## Source trace

- `vmlx_engine/mllm_scheduler.py` initializes `PagedCacheManager` when prefix cache is enabled and either paged RAM or block-disk L2 is enabled. With paged RAM off it selects `block_disk_only`, passes `disk_only=True`, retains a zero-byte payload mirror, and fails closed if the SSD store cannot initialize.
- `vmlx_engine/mllm_batch_generator.py` labels this typed route `block-disk+mixed_swa` and preserves full-attention KV, sliding-window KV, and rotating-window metadata.
- `panel/src/main/sessions.ts`, `SessionConfigForm.tsx`, `SessionSettings.tsx`, and `cacheControlPolicy.ts` now expose the already-supported mixed-SWA SSD-only engine route through UI, preview, persistence, and argv. DSV4, ZAYA, M3, openPangu, and the unproven Step subtype keep their existing architecture policy.
- `cache-control-policy.test.ts` and `settings-flow.test.ts` pin Paged Off + Block L2 On for Gemma mixed-SWA, no paged RAM ceiling, and forced Paged On when Block L2 is absent.

## Bundle and launch truth

Artifact: `/Volumes/EricsLLMDrive/jangq-ai/gemma-4-12B-it-qat-JANG_4M`.

This is affine `JANG_4M`, not JANGTQ/MXTQ/base MXFP. Its 48-layer text cache is 40 sliding-window plus 8 full-attention layers. The typed runtime schema was `mixed_swa_kv_v1` with `full_attention_kv`, `sliding_window_kv`, and `rotating_window_metadata`.

The real Electron settings drawer accepted Paged Off, Block Disk L2 On, and stored cache quantization None. `Save & Restart` produced PID 57549 and then PID 59518 with these effective flags:

```text
--no-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000
--kv-cache-quantization none
--enable-block-disk-cache --block-disk-cache-max-gb 10
```

Before the first post-restart request, `/health` reported `model_loaded=true`, `last_request_time=null`, `backend_mode=block_disk_only`, `paged_ram_enabled=false`, `disk_only=true`, `l1_indexed_tokens=0`, `l1_resident_bytes=0`, 5,419 L2 tokens on SSD, and zero TurboQuant writes/hits.

After the proof, the same real drawer restored Paged On + Auto and restarted as PID 59856. The persisted session config has `usePagedCache=true`, `enableBlockDiskCache=true`, and `kvCacheQuantization=auto`; argv contains `--use-paged-cache`, and health reports paged RAM plus stored-prefix `turboquant-q4` for compatible mixed-SWA KV storage.

## Live Electron rows

All turns used the running Electron dev build over CDP 9335.

- Cold A: exact visible `G4-SSDONLY-QNONE-A-DONE`, separate prompt-specific reasoning, 1,647 prompt tokens, no cached-token claim, 26 new SSD block writes, zero resident L1 bytes, and zero TQ activity.
- Exact warm A: exact visible answer with `1646/1647 block-disk+mixed_swa cached`; SSD hit counters advanced while resident L1 bytes remained zero.
- Partial B: a fresh chat preserved the prefix through record 046 and changed only records 047-048 plus the marker. It returned exact `G4-SSDONLY-QNONE-B-DONE` with `1536/1649 block-disk+mixed_swa cached`. The DOM observer captured 72 mutations with progressive separate reasoning and visible content. Disk hits advanced from 78 to 150 and only two changed-tail blocks were written.
- Process-restart A: after PID 57549 -> 59518, the first fresh Electron request restored `1646/1647 block-disk+mixed_swa cached`, returned exact visible content, produced 39 DOM mutations, and ended with 26 disk promotions, zero new writes, zero TQ writes/hits, and zero resident L1 bytes.

Reasoning bytes differed between A and B; no stale reasoning replay, tool call, warning, parser marker leak, or empty visible answer was observed.

## Raw Responses stream

`g4-ssdonly-responses-hierarchy.json` contains a cold/exact-warm/changed-suffix sequence against `/v1/responses` while the same Electron-started PID remained in SSD-only mode.

- all three requests returned HTTP 200 and exact non-empty visible output;
- each emitted 12 progressive `response.output_text.delta` events;
- each emitted exactly one `response.output_text.done` and one `response.completed`;
- terminal delay after the last content delta was 21.5-22.7 ms;
- final health remained `block_disk_only`, `paged_ram_enabled=false`, `l1_resident_bytes=0`, `tq_native_writes=0`, and `tq_native_hits=0`.

## Focused validation

- Python scheduler mixed-SWA/disk-only selection: 7 passed.
- Python cache-detail/native-schema selection: 7 passed.
- Panel policy/settings flow: 300 passed.
- Panel TypeScript typecheck: passed.
- `git diff --check`: passed.

These are focused tests, not the full Python/panel/build release gate.

## Evidence files

- `source.diff`
- `session-rows.json`
- `session-config-restored.json`
- `g4-diskonly-pre-request-health.json`
- `g4-restored-paged-auto-health.json`
- `g4-ssdonly-responses-hierarchy.json`
- `g4-ssdonly-qnone-b-partial-observe.json`
- `g4-ssdonly-qnone-a-restart-observe.json`
- `g4-settings-pagedoff-tqnone-before-restart.png`
- `g4-ssdonly-qnone-restart-pass.png`
- `g4-settings-restored-paged-auto.png`
- focused test logs

## Remaining gates

- Repeat the SSD-only exact/partial/restart matrix independently for Step and MiMo before broadening their UI policy.
- Exercise Chat Completions, Anthropic, and Ollama in addition to the Responses proof in this directory.
- Continue current-source rows for MM2.7, M3, openPangu, DSV4, Laguna, Nemotron/Nemo, Qwen/JANGTQ, LFM, and advertised image/video/audio routes.
- Finish gateway one-model unloading, eager-load inventory, settings parity, failure/cancellation soak, full suites/build, signing, notarization, and release gates.
