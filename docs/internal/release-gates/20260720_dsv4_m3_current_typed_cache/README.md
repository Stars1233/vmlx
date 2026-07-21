# DSV4 Flash and MiniMax-M3 current-source typed-cache gate

Date: 2026-07-20 (America/Los_Angeles)

Host/repo: `erics-m5-max.local`,
`/Users/eric/mlx/vllm-mlx-release-1.6.13`, branch
`codex/postrelease-ui-drawers-20260720`.

This gate is deliberately architecture-specific. It does not flatten DSV4's
native composite state or MiniMax-M3's sparse-index state into generic KV, and
it does not treat a successful short generation as cache proof.

## Verdict

| Row | Status | Current live evidence |
|---|---|---|
| DSV4 real Electron load/eager ownership | PASS-LIVE scoped | The real `+ New Session` and `Launch Session` controls loaded `dealignai/DeepSeek-V4-Flash-JANG-CRACK`. Before a request, health had `last_request_time=null` and about 99.7 GB active memory. Single-model mode stopped the prior engine. UI, argv, and health agreed on paged 256-token blocks, Block L2, DSML/deepseek parsers, native DSV4 cache, and no generic TQ. |
| DSV4 typed cache hierarchy | PASS-LIVE exact-prefix / architecture-safe partial miss | Cold Responses emitted nine progressive text deltas and exact `DSV4-CACHE-CURRENT-DONE`. Same-process exact reuse restored 1,722/1,723 tokens as `paged+dsv4`. After real Electron Save & Restart, the first request restored the same 1,722 tokens as `paged+dsv4+disk`. A forward/partial request was intentionally recomputed because nonterminal fragments lacked the terminal composite CSA/HCA state; it completed exactly without falsely claiming a cache hit. |
| DSV4 pool codec explicit Off | PASS-LIVE scoped | The real Server Settings drawer disabled `DSV4 CSA/HCA Pool Codec` and applied Save & Restart. Before the request, health reported pool quantization disabled and `last_request_time=null`. Raw Responses emitted nine progressive content deltas, matching done text, and one completed terminal for exact `DSV4-POOL-OFF-DONE`. |
| DSV4 visible output/sampling | PASS-LIVE scoped | The real UI showed bundle-derived temperature 0.60, top-p 0.95, top-k Off, repetition penalty 1.00. Electron content was non-empty and exact `DSV4-HEAD1-DONE VALUE=45`, with 297 characters of separately persisted reasoning and no tool/warning. |
| M3 real Electron load/eager ownership | PASS-LIVE scoped | The real Electron model selector and `Launch Session` loaded `JANGQ-AI/MiniMax-M3-Coder-Small`, stopped DSV4, and left one engine. Before a request, health reported `last_request_time=null` and about 80.4 GB active memory. UI/argv/health agreed on paged 64-token blocks, Block L2, MiniMax parsers, no draft model, and no generic TQ. |
| M3 typed RAM/SSD partial-prefix reuse | PASS-LIVE scoped | Health exposes 60 layers: dense KV 0-2 and `MiniMaxM3SparseCache` 3-59 with `attention_kv`, `msa_idx_keys`, and `absolute_block_index`. Cold exact output stored asynchronously after terminal dispatch; settled health held 1,626 cached tokens. Warm exact restored 1,495/1,500 tokens. A same-process partial restored 1,472/1,512. After real Electron Save & Restart with empty L1, a never-stored suffix restored 1,472/1,514 from SSD as `paged+disk`, with 23 disk hits/promotions, then stored only the new tail. |
| M3 persisted-TQ truthfulness | PASS-SOURCE+LIVE | The loader already rejected generic TQ for MSA tuples, but `block_disk_cache.tq_native_enabled` was incorrectly true because the CLI did not set the process-level disk-store gate. `vmlx_engine/cli.py` now sets `VMLX_DISABLE_TQ_KV=1` for native M3 MSA. After Electron Save & Restart, health reports `generic_turboquant_kv.enabled=false`, `storage_quantization.enabled=false`, and `tq_native_enabled=false`; 23 disk hits restored native unquantized records. |
| M3 two-turn reasoning/tool/content loop | PASS-LIVE scoped | Electron turn 1 visibly preserved a 1,224-character reasoning rail and exact non-empty `M3-HEAD1-DONE`. Same-chat turn 2 preserved different 262-character reasoning, generated exactly one `file_info({"path":"panel/package.json"})`, executed one result, and exact-finaled `M3-HEAD-TOOL-DONE SIZE=5.2 KB` with no warning. It reused 256 `paged+disk` tokens. No zero-tool card, stale reasoning replay, leaked namespace/control marker, or missing terminal content appeared. |
| M3 MTP | N/A for this artifact | `config.text_config.num_nextn_predict_layers=1` is only an architecture hint. The bundle name, JANG sidecar, and tensor index declare no MTP artifact; health correctly reports `status=not_configured`. MTP is not force-enabled. |
| M3 VL | PARTIAL / not promoted by this gate | The artifact has vision config and weights, but current health says `vl_runtime_available=false`. Prior media gates remain provenance; a current-source UI/API image/video row is still required before promoting current VL. |

## Source trace and validation

- `vmlx_engine/cli.py`: the `_m3_forced_no_kvq` branch now applies the same
  no-TQ contract to the disk store that already governed the native loader.
- `tests/test_turboquant_cache_contract.py`:
  `test_minimax_m3_native_msa_disables_tq_for_live_and_persisted_cache`
  pins the CLI/env contract.
- Focused current-source validation: 132/132 passed across
  `test_turboquant_cache_contract.py`, `test_minimax_m3_cache_paths.py`,
  `test_dsv4_paged_cache.py`, and `test_dsv4_contract_hardening.py`.

## Evidence map

- Electron screenshots:
  `dsv4-create-cache-settings.png`, `dsv4-eager-loaded.png`,
  `dsv4-chat-inference-settings.png`, `dsv4-head1-ui-clean.png`,
  `dsv4-pool-off-before-restart.png`,
  `m3-create-cache-settings-full.png`, `m3-eager-loaded-ui.png`,
  `m3-chat-settings-current.png`, `m3-head1-current.png`, and
  `m3-head-tool-current.png`.
- Persisted Electron rows: `electron-db-rows.json`.
- Timed raw streams: `dsv4-cache-*-current.tsv`,
  `dsv4-pool-off-stream-current.sse`, and `m3-cache-*-current.tsv`.
- Health/cache telemetry: matching `*-health-current.json` files, including
  pre-request and settled snapshots.
- Source/test evidence: `source.diff` and `focused-tests.log`.

## Retained boundaries

- DSV4 partial reuse is not safe without terminal composite state. Full
  recomputation is the correct current behavior, not a reason to synthesize a
  hit.
- DSV4 longer constrained-output quality/performance and current-source
  agentic protocol breadth retain their existing matrix status.
- M3 current-source image/video capability is not claimed here. REAP32 remains
  excluded because of the documented host-reboot risk.
- This gate does not close gateway-wide Chat/Responses/Anthropic/Ollama,
  LAN/port rollback, broader media, signed-app repetition, or the version
  mismatch between the Electron source and PATH engine display.
