# openPangu current-HEAD post-Qwen/HY3 recheck

Date: 2026-07-16

Commit under test: `b5a47f62f`

Verdict: `SCOPED_PASS / BROAD_RELEASE_GATE_OPEN`

## Source trace

- `vmlx_engine/models/openpangu_v2/openpangu_v2.py:682-724` refuses cache/layer mismatch, executes all decoder layers, merges four mHC streams, and logs the DSA/SWA/sink/MLA topology actually traversed.
- `vmlx_engine/models/openpangu_v2/cache.py:168-209` reconstructs the typed DSA/SWA cache from metadata and clones only a full, non-aliasing logical boundary; arbitrary reverse truncation is intentionally unsupported.
- `vmlx_engine/cli.py:232-253` forces generic paged/block cache and all TurboQuant/q4/q8 lanes off for openPangu's path-dependent composite state.
- `vmlx_engine/server.py:7873-7924` exposes `openpangu_v2_composite_v2`, its exact typed state components, prompt-memory/disk policy, and the reason generic TQ is disabled.

## Live Electron verification

- Electron one-model mode loaded `/Volumes/EricsLLMDrive/jangq-ai/openPangu-2.0-Flash-JANG_3M` as PID 97796, then visibly stopped it and loaded PID 98632 without clearing L2. HY3 became inactive.
- Launch argv used `--no-paged-cache --enable-disk-cache`, with no TurboQuant/q4/q8 flag.
- Health reported the native `openpangu_v2_composite_v2` cache with MLA latent KV, DSA indexer, rotating SWA and path-dependent convolution state; `turboquant_kv_cache.enabled=false`.
- Logs proved strict landing of 2,826/2,826 parameter leaves, 138 causal convolutions, and traversal of all 46 decoder layers: DSA=16, SWA=30, mHC=4, attention sinks=128, MLA KV rank=512, SWA window 512, max context 524,288.
- Electron row 1947: exactly one `file_info(panel/package.json)` and exact `PG-B5-1-DONE` in 5.1 s.
- Same-chat row 1950: exactly one `file_info(README.md)`, exact `PG-B5-2-DONE`, and 144 typed memory-cached tokens in 6.5 s.
- After the visible process restart, row 1953: exactly one `file_info(pyproject.toml)`, exact `PG-B5-3-RESTART-DONE`, and 295 typed disk-cached tokens in 7.7 s.
- Post-restart health showed one prompt-disk hit, 295 cache-hit tokens, 3,302 prompt-L2 tokens, zero block-L2 tokens, and `dequantized=false`.
- Server settings visibly showed Prefix Cache and prompt Disk Cache enabled, generic paged/block controls unavailable, `openPangu typed composite cache`, `TURBOQUANT OFF`, and stored quantization locked to None.

## Tests

- 75/75 openPangu model/parser/tool-prompt tests passed.
- 2/2 exact-once Responses server regressions passed.

## Boundary

- MTP heads/config are detected, but this JANG_3M bundle's current runtime has no landed MTP tensors and reports `weights_present_runtime_unwired`; no MTP speed claim is made.
- The 512K limit, sustained long-context soak, concurrency beyond the enforced single-active queue, and the full external protocol matrix were not rerun here.
- No package, version, signature, notarization, tag, feed, or public release surface changed.
