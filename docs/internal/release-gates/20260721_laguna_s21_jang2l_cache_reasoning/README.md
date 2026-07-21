# Laguna S-2.1 JANG_2L reasoning/tool/cache gate and JANG_4M supplement

Date: 2026-07-21

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Electron data: `/Users/eric/.vmlx-v1613-responsive-dev`

CDP: `127.0.0.1:9335`

The main gate covers the current `jangq-ai/Laguna-S-2.1-JANG_2L` artifact
loaded through the real Electron Start/Save-and-Restart controls. The
`JANG_4M.md` supplement adds current live proof for the JANG_4M q4/L2 dtype and
restart-performance defect. It does not promote long-context quality or the
full gateway protocol matrix.

## Artifact and source contract

- The bundle is text-only Laguna S-2.1 with 48 layers: 12 full-attention KV
  slots and 36 sliding-window slots. The current JANG reference requires
  `RotatingKVCache(keep=0)` for SWA and the shipped tokenizer behavior.
- `vmlx_engine/utils/turboquant_config.py::apply_mixed_swa_auto_tq_policy`
  assigns q4 only to full-attention slots and stamps
  `mixed_swa_full_attention_kv_storage_tq4`.
- `vmlx_engine/scheduler.py` creates a block-disk-only backend when L2 is on
  and paged RAM is off; it refuses to silently substitute RAM if SSD setup
  fails.
- `vmlx_engine/paged_cache.py::get_computed_blocks` checks L1 first, then L2,
  including known terminal partial sizes, and promotes reconstructable disk
  blocks.
- `vmlx_engine/prefix_cache.py::reconstruct_cache` retains promoted payloads
  under Paged On and releases transient payloads under disk-only mode.
- `vmlx_engine/model_config_registry.py` now treats a concrete JANG bundle's
  stamped `default_enable_thinking` as the Auto default for eligible families.
- `vmlx_engine/server.py` now probes the scheduler-owned loaded model when
  reporting effective TurboQuant bits/policy and reports mixed-SWA storage as
  full-attention-only q4 plus native rotating state.
- The Laguna loader intentionally does not force `fix_mistral_regex=True`.
  The JANG reference says Poolside serves the shipped tokenizer and a one-sided
  regex rewrite would make vMLX tokenization diverge. A live probe showed the
  current test prompt has identical IDs with and without the flag, so this is
  a parity decision rather than a claimed quality improvement.

## Live results

| Row | Status | Current evidence |
|---|---|---|
| Electron eager load and settings restart | PASS-LIVE scoped | UI Save-and-Restart replaced PID 96221 with 98484, then 98916 and 99170. The final argv restored `--use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000 --enable-block-disk-cache --block-disk-cache-max-gb 10`; final health has `model_loaded=true`, `last_request_time=null`. |
| Bundle-derived chat settings | PASS-LIVE scoped | The drawer displayed Auto, temperature 1.00, top-p 1.00, top-k 20, min-p 0.00, repetition penalty 1.00, and model-default max tokens. Logs resolved these values and `enable_thinking=True`. |
| Auto reasoning equivalence | PASS-API scoped | Greedy/cache-bypassed Auto and explicit On each emitted 256 reasoning and 42 content deltas with byte-identical reasoning/content SHA-256. Explicit Off emitted zero reasoning and 79 content deltas. The 256-token Auto/On controls truthfully ended `response.incomplete`; they are stream-separation controls, not completion-quality passes. See `laguna-s21-reasoning-auto-ab.json`. |
| Auto reasoning in Electron | PASS-LIVE scoped | `LAG-S21-UI-AUTO4` visibly painted a Thinking rail, persisted 1,724 reasoning characters separately, then a non-empty concise final ending `LAG-S21-UI-AUTO4-DONE` with no warning. See the mid/final screenshots and DB row 323. |
| Stochastic empty-think control | OBSERVED / not parser failure | Auto row 320 closed the think rail without reasoning and emitted its work as visible content. The deterministic Auto/On A/B plus row 323 show the Auto route itself is wired; this retained row documents native stochastic empty-think behavior and lower answer quality. |
| Electron one-tool continuation | PASS-LIVE scoped | Row 317 called built-in `file_info` exactly once with `panel/package.json`, persisted one OAI call and one result, and exact-finaled `LAG-S21-TOOL1-DONE SIZE=5.2 KB`; no warning or XML leak. |
| Raw Responses tool loop | PASS-LIVE scoped | Required-tool request emitted two progressive argument deltas, exactly one schema-valid call, and one completed terminal with no visible prose. `previous_response_id` continuation emitted 16 progressive content deltas, no repeat tool, exact final, and one completed terminal. See `laguna-s21-responses-tool-loop.json`. |
| Cold store format | PASS-LIVE scoped | The 334-token cold Electron row stored six blocks. Logs recorded 36 `rotating_kv` plus 12 `turboquant_kv` layers per block, q4 K/V policy, and native rotating metadata. |
| Paged RAM partial-prefix reuse | PASS-LIVE scoped | Row 329 reused 320/334 tokens, exactly five 64-token blocks, as `paged+tq-native`; TTFT moved from 0.52 s cold to 0.30 s and the exact marker remained correct. |
| Restart L2 restore | PASS-LIVE scoped | After PID replacement and zero RAM tokens, row 332 restored 320 tokens as `paged+disk+tq-native`. Health recorded five disk promotions, five TQ-native hits, 60 full-attention TQ layer-block entries, successful reconstruction, and exact output. |
| Bounded L1 eviction and L2 recovery | PASS-LIVE scoped | With eight configured blocks/seven usable, a distinct chained prefix caused five L1 evictions. Re-requesting the original prefix in row 338 restored all five blocks from SSD as `paged+disk+tq-native`; cumulative L1 evictions reached 11 and output stayed exact. |
| SSD-only partial-prefix reuse | PASS-LIVE scoped | Real UI set Paged Off and Block L2 On. PID 98916 launched with `--no-paged-cache`; health reported `backend_mode=block_disk_only`, `paged_ram_enabled=false`, `l1_resident_bytes=0`. Row 341 restored 320 tokens as `block-disk+tq-native`, with five disk/TQ hits, 60 TQ layer-block entries, and exact output. |
| Final restored policy | PASS-LIVE scoped | Real UI restored Paged On and 1000 blocks. PID 99170 has Paged RAM + block L2, 48 disk blocks retained, q4 K/V storage, full-attention-only TQ, 36 native rotating slots, and no request yet. See `final-restored-defaults-health.json`. |

## Retained negative controls

- Explicit-On row 314 separated reasoning correctly but violated the exact-only
  instruction by adding extra visible text and a malformed marker before the
  correct marker. This is a model-format miss, not hidden by the gate.
- Auto row 320 emitted no reasoning rail and produced a self-correction. The
  deterministic Auto/On A/B demonstrates that omission does not select a
  different prompt/parser route; it remains a stochastic quality observation.
- The bounded 256-token Auto/On API controls ended `response.incomplete` after
  the server's visible-answer pass. They prove progressive channel emission,
  not a completed terminal under that deliberately small cap.

## Focused validation

- `tests/test_laguna_loader.py`, the Laguna registry tests, and reasoning-mode
  tests: 9 selected tests passed.
- Earlier combined registry/status validation in this working set: 129 passed.
- Full Python and panel suites are not rerun by this scoped gate.

## Still open

1. Complete JANG_4M Paged-Off/SSD-only restore, one-tool/protocol, sampling,
   and long-window rows. The q4 mixed-SWA disk dtype, RAM hit, restart L2 hit,
   settings restart, eager load, and performance regression are closed in
   `JANG_4M.md`.
2. Run Chat Completions, Anthropic, and Ollama tool-result continuations for
   S-2.1; this gate covers Electron Responses plus raw Responses.
3. Measure long-context correctness across the 512-token SWA boundary and the
   documented Laguna performance budget. Do not infer those from a 334-token
   cache prompt.
4. Keep the model-level stochastic exact-format/empty-think variability visible;
   do not add output rewriting, synthetic think tags, or hidden sampler clamps.
5. Audit Chat Completions and Responses across parser families for separate
   reasoning deltas/`reasoning_content`, progressive visible content, valid tool
   continuation, and truthful terminal events. Inline `<think>` leakage is a
   failure; it is not inferred closed from the Electron DB split alone.
