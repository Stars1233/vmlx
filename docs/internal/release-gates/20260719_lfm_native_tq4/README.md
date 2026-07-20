# LFM2.5 selective q4 TurboQuant checkpoint — 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` for Auto selective q4 storage, Paged RAM reuse,
typed SSM companion storage, and restart-from-SSD restore. Overall LFM and
release status remain `PARTIAL`.

## Artifact identity

- Model: `/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`
- `model_type=lfm2_moe`, `Lfm2MoeForCausalLM`, 24 native cache slots.
- Full-attention KV positions: 2, 6, 10, 14, 18, 21.
- Remaining 18 slots are native convolution/SSM `ArraysCache` companions.
- Weight path is base MLX MXFP4. It is not affine JANG and not JANGTQ/MXTQ.

## Source trace

Pushed commit: `748929fe3 fix(cache): enable selective LFM TurboQuant`.

- `vmlx_engine/utils/hybrid_tq_cache.py` adds `lfm2`/`lfm2_moe` to the
  architecture-gated selective attention-KV path.
- `vmlx_engine/utils/model_inspector.py` derives missing standard head widths
  from exact `hidden_size / num_attention_heads`; this bundle resolves to 64,
  matching the MLX LFM runtime.
- `vmlx_engine/cli.py` keeps Auto native TQ enabled for LFM and disables the
  second generic stored-quant wrapper. Explicit `none/q4/q8` remains outside
  this Auto branch and keeps its existing exact-user-choice behavior.
- `vmlx_engine/utils/tokenizer.py` retains native companion slots and replaces
  only real `KVCache` slots with `TurboQuantKVCache`.
- `tests/test_hybrid_live_tq_kv.py` and `tests/test_model_inspector.py` cover
  the allow-list, q4 policy, 64-wide dimensions, and untouched companions.
- Focused validation: 63 passed.

## Real Electron proof

The dev Electron app was controlled through CDP 9335. The session settings
visibly showed Paged On, Block Disk L2 On, and KV Cache Quantization Auto. Save
and Restart loaded PID 69378 from the project `.venv/bin/vmlx-engine`; a later
visible Stop/Start replaced it with PID 69763 before any post-restart request.

Initial health reported:

- `turboquant_kv_cache.enabled=true`
- `stored_prefix_quantization=turboquant-q4`
- `auto_policy=uncalibrated_selective_attention_kv_storage_tq4`
- `native_cache.schema=hybrid_ssm_v1`
- q4 `turboquant_native` on the six attention layers only
- native full-precision SSM companion policy on the other 18 slots
- `block_disk_cache.tq_native_enabled=true`

Cold UI row 716 wrote nine q4-native blocks (576 tokens). Fresh-chat UI row
719 visibly exact-finaled `LFM-NATIVE-TQ4-A-DONE`, reported
`576 paged+ssm+tq-native cached`, and stored the 576-token SSM companion state.
The DOM observer captured incremental updates during both turns.

After Electron Stop/Start, health before the first request had zero L1 tokens,
nine persisted q4-native blocks, and one persisted SSM companion entry. The
first request then recorded:

- scheduler `tokens_saved=576`
- block SSD `disk_hits=9`
- native TQ `tq_native_hits=9`
- SSM companion SSD `hits=1`
- L1 promotion back to 576 tokens

## Output boundary

The post-restart raw Responses probe intentionally used only 64 output tokens.
It emitted 63 progressive content deltas and `response.output_text.done`, then
`response.incomplete`. The response was coherent but did not reach the exact
marker. This is a `PARTIAL` output-cap row, not evidence of a cache corruption
and not a completed protocol-quality claim.

## Evidence files

- `lfm-before-native-tq.png`
- `lfm-config-native-tq.png`
- `lfm-config-paged-open.png`
- `lfm-ui-cold-pass.png`
- `lfm-ui-warm-pass.png`
- `lfm-ui-cold-observe.json`
- `lfm-ui-warm-observe.json`
- `lfm-native-health-before.json`
- `lfm-native-health-after-cold.json`
- `lfm-native-health-after-warm.json`
- `lfm-native-health-before-restart-hit.json`
- `lfm-native-health-after-restart-hit.json`
- `lfm-native-restart-response.json`
- `lfm-native-process-argv.txt`
- `lfm-native-db-rows.json`

## Still open

- Native q4 TQ with Paged RAM explicitly Off and Block L2 as the only block
  tier.
- Explicit Auto-to-Off live proof and restoration to Auto.
- Larger-context eviction/fault injection.
- Current-source Chat/Responses/Anthropic/Ollama tool, cancel, and recovery
  breadth.
- Full suites, installed/signed app, and release gates.
