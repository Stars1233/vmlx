# Current source/release reconciliation — 2026-07-19

Source cutoff audited: `a8864c6b6` (`reconcile/1.5.68`)

Push target: `origin/codex/live-electron-gates-20260715`

Status: `PARTIAL_NO_1_6_12_RELEASE`

This is a document/source reconciliation over committed evidence. No model was launched specifically for this audit, so it does not create new live passes. A row is listed as live only when the named evidence directory contains current source trace plus real Electron and/or raw API artifacts. Older live rows remain useful evidence but require a current-head rerun when an owning server/parser/cache path changed afterward.

## Documentation rule for continuing work

Every investigated issue must be recorded before moving on:

1. artifact identity and architecture/quant/cache classification from the real bundle;
2. owning source path and exact change, or an explicit `NO_SOURCE_FIX` result;
3. focused tests with counts;
4. real Electron Start/Stop/settings/turn evidence and raw API SSE evidence where applicable;
5. output-emission checks: non-empty visible text, distinct reasoning across prompts, separate reasoning/content/tool deltas, progressive paint, terminal and usage ordering, and no leak/loop/truncation;
6. cold/RAM/L2/restart/partial-prefix/eviction cache axes as applicable;
7. an explicit `VERIFIED-LIVE`, `FAIL`, `PARTIAL`, or `BLOCKED` verdict and the missing acceptance evidence;
8. scoped commit and push identifiers.

## Current evidence-backed status

| Area | Current status | Evidence / missing proof |
|---|---|---|
| Public v1.6.11 | `RELEASED` | `20260718_v1_6_11_release/` preserves signed/notarized Sequoia and Tahoe release proof. This does not cover the current post-release head. |
| Current full suites/build | `STALE-PASS / RERUN-REQUIRED` | `20260719_full_suite_checkpoint/` passed 6,125 Python, 2,312 panel, typecheck, bundle verification, and production build at an earlier source. Shared server/parser/loader changes landed afterward. |
| Bundled Python | `STALE / BLOCKING` | M3 proof `20260719_m3_current_postfinalizer/` recorded source/bundle hash drift; `bundle-python.sh` is mandatory at the next cutoff. |
| MiniMax M2.7 text/protocol | `VERIFIED-LIVE scoped` | `20260719_m27_protocol_parity/`, `anthropic_tool_parity/`, `ollama_stream_tool_parity/`, `ollama_multitool/`, `response_cancel_disconnect/`, and `chat_disconnect_stop_recovery/`. Other parser families and safe live mid-stream fault injection remain open. |
| Paged-Off prompt-disk partial prefix | `VERIFIED-LIVE scoped` | M2.7 JANGTQ q4: `20260719_nonpaged_prompt_disk_partial/`. openPangu typed N-1 Auto/Off: `20260719_prompt_disk_payload_prefix_index/`. These prove disk reuse without paged cache; they do not generalize to every architecture. At cutoff `a8864c6b6`, no owning prompt-disk/prefix source changed after the live gates and the exact current disk/paged/TQ union passed 261 tests (2 deselected). |
| Paged q4 partial/eviction/L2 | `VERIFIED-LIVE scoped` | `20260719_m27_paged_l2_partial_refault/` proves 64+64+50 partial blocks, bounded L1 eviction, same-process L2 refault, process-restart disk-only restore, progressive Chat/Responses, and a post-restore tool loop. `vmlx_engine/prefix_cache.py` and its focused tests are unchanged since `97a84fed5`; the current 261-test union passes. |
| Chat terminal usage | `VERIFIED-LIVE scoped` | `20260719_chat_terminal_usage_parity/` proves ordinary `usage:null`, one final choices-empty usage chunk, then `[DONE]`; Electron paint remained progressive. Cross-family current-head repeats remain open. |
| Bonsai 27B 1-bit | `VERIFIED-LIVE scoped / PARTIAL family` | `20260719_bonsai_partial_prefix_responses/` proves q8 attention-KV plus native companion state, 6,336-token RAM/L2 partial prefix, progressive exact tool continuation. Long pre-tool reasoning, stochastic/media/soak and signed-app repeats remain open. |
| MiniMax M3 | `VERIFIED-LIVE text scoped / PARTIAL family` | `20260719_m3_current_postfinalizer/` proves current-source Electron and Chat/Responses text/tool streams with native MSA cache. Larger media/OCR/terminal delay/REAP and packaged repeat remain open. |
| Gemma 4 mixed SWA | `VERIFIED-LIVE text/cache scoped / PARTIAL family` | `20260719_gemma4_current_parser_stream/` proves progressive text/tool streams and `paged+mixed_swa+disk` restore. Default reasoning consumed 3,322 output tokens; media and broader restart/soak remain open. |
| DSV4 | `VERIFIED-LIVE controlled scoped / PARTIAL quality` | `20260719_dsv4_current_parser_auto_stream/` proves Auto UI, tool/parser streams and native typed L2 with generic TQ Off. Strict marker/long factual reliability and matched reference A/B remain open. |
| Laguna | `VERIFIED-LIVE controlled scoped / PARTIAL performance` | `20260719_laguna_current_stream_tq_determinism_eviction/` proves Electron/API streams, UI settings, q4 restart and bounded eviction. Cold full-precision and q4-restored greedy answers differed; latency/long soak remain open. |
| Mistral Medium 3.5 JANGTQ2 | `BLOCKED_CURRENT_ARTIFACT_RUNTIME` | `20260719_mistral35_jangtq_prefill/`: strict 616/616 hydration, legacy prefill stall, NAX newline-only decode, FP32 NAX repeat; failed auto exception reverted by `fad7356d4`. Matched known-good same-artifact runtime comparison is required. Mistral MXFP4 remains excluded by user directive. |
| Qwen 3.6 35B JANGTQ | `VERIFIED-LIVE scoped / PARTIAL family` | `20260719_qwen35_jangtq_current/` proves current Electron/API/tool/stream, q4 hybrid RAM/L2 partial-prefix, and quant-label truth. Strict sampled reliability and unavailable VL remain open. |
| HY3 MTP | `VERIFIED-LIVE scoped / PARTIAL family` | `20260719_current_hy3_mtp/` proves current Electron four-turn history/tool behavior, progressive paint, literal curl Responses/Chat tool continuations, real D1 draft/accept counters, q4 process-restart L2 restore, and one-model swap. Long/stochastic soak and a fresh MTP-Off performance A/B remain open. |
| Step 3.7 JANGTQ | `VERIFIED-LIVE scoped / PARTIAL family` | `20260719_current_step37_jangtq/` fixes zero-patch MLX metadata and proves Electron image A/A/B/A media-salt behavior, distinct real MP4, raw Chat/Responses streaming, and 4,290-token q4 mixed-SWA restart/L2 restore. Shared MLLM cache-detail accounting is now live-proven on an immediate same-process frugal L2 refault. Retained: cold latency, native strict-content misses, missing restarted PID header, larger-video/stochastic soak. |
| openPangu long, Nemotron, Zaya/CCA, other families | `PARTIAL / CURRENT-HEAD RERUN REQUIRED` | Retained earlier gates exist, but selected current-head Electron plus raw API reruns remain required before a new release claim. Media-capable rows require actual image/video/audio evidence and media-salt isolation. |
| Settings/gateway/single-model | `PARTIAL` | Scoped UI/DB/argv/health, LAN rollback, port conflict, and one-model swap evidence exists in the master matrix. Cross-family model-derived defaults, explicit Off, repeated swaps, network-loss recovery and soak remain open. |
| Current public bump | `BLOCKED` | Choose a 1.6.12 cutoff only after current full suites, regenerated bundled Python, selected must-pass current-head Electron/API matrix, clean builds, signature/notarization/staple/Gatekeeper/install-smoke, then explicit publication. |

## Next mandatory sequence

1. Update the master ledger/matrix with this reconciliation and retain older contradictory rows for provenance.
2. Continue current-head cross-family output-emission/protocol/cache regression rows, prioritizing openPangu, Nemotron hybrid, Zaya/CCA, and remaining media/audio representatives. Qwen JANGTQ, HY3 MTP, and Step now have scoped current-head gates but retain their listed family partials.
3. Close settings/gateway/single-model repeated-swap and narrow-window/locale rows selected for the release cutoff.
4. Rerun complete Python and panel suites, typecheck, and production build after the final runtime fix.
5. Rebuild bundled Python from the exact current vMLX and clean Jang revisions; verify hashes/imports.
6. Build/sign/notarize/staple/verify/install-smoke both Sequoia and Tahoe DMGs. Publish only after an explicit final release decision.

No row above is promoted by documentation alone.
