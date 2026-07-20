# Python/Electron vMLX stop handoff — 2026-07-20

## Stop instruction and scope

Eric explicitly asked to pause and stop all runtime work, then requested a
detailed written handoff. Do not restart models or continue implementation
merely because open rows exist. Resume only after a new user instruction.

This handoff is for the legacy Python/Electron vMLX/MLXStudio checkout on the
other Mac, not the local Swift repository.

- Host: `erics-m5-max.local`
- Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Branch: `codex/postrelease-ui-drawers-20260720`
- Upstream: `origin/codex/postrelease-ui-drawers-20260720`
- Base before the pending scoped cache repair:
  `e8755c6b2e4a8abe5cfb38b0a284fb61129a0634`
- Electron profile: `/Users/eric/.vmlx-v1613-responsive-dev`
- CDP: `127.0.0.1:9335`
- Python runtime venv: `/Users/eric/mlx/vllm-mlx/.venv`

Public release truth at this stop:

- v1.6.14 is already publicly released at source tag `e1776a485`.
- The canonical signed/notarized/public proof is
  `docs/internal/release-gates/20260720_release_checkpoint_1_6_14/README.md`.
- Current post-release source contains later work, including Nemotron Omni
  session-L2 commit `e8755c6b2`, plus the pending LFM cache repair documented
  below. These later changes are not part of public v1.6.14.
- The broader matrix is still `PARTIAL`; public checkpoint success must not be
  expanded into a global family/protocol/media/cache claim.

## What was being fixed at stop

Blocker class: `cache/storage` with `api/ui` live proof.

Exact model:
`/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`.

This is base MLX MXFP4, not affine JANG and not JANGTQ/MXTQ. It is a 24-layer
`lfm2_moe` hybrid with six attention layers and 18 SSM/convolution companion
layers.

The failure required all of these conditions:

- Prefix Cache On;
- Paged Cache Off;
- Block Disk L2 On;
- stored KV quantization explicitly None;
- a shared L2 directory previously containing Auto/q4-native rows;
- a small block pool that stored a shorter KV prefix than the full cacheable
  prompt.

Two defects were found:

1. Explicit TQ Off could reject the first old q4 row but leave later q4 rows in
   the same hash chain. The asynchronous ordinary-KV writer then deduplicated
   against those incompatible rows, creating a mixed chain.
2. Hybrid SSM rederive stayed queued at the full prompt boundary even when the
   block pool stored only 576 tokens, and the scheduler idle predicate did not
   wake for that queued rederive.

## Pending scoped source/test changes

These are intentional user/agent changes and must not be reverted:

- `vmlx_engine/block_disk_store.py`
- `vmlx_engine/scheduler.py`
- `vmlx_engine/mllm_scheduler.py`
- `tests/test_tq_paged_block_cache.py`
- `tests/test_hybrid_ssm_companion_regressions.py`

The source changes:

- replace codec-incompatible q4 rows on every explicit-None ordinary page
  write;
- roll back optimistic hit credit and release refs for rejected reconstruction;
- retain only accepted credit on shortened prefixes;
- retarget SSM rederive to the actual paged block-table boundary;
- keep text and MLLM async loops awake for queued hybrid SSM rederive.

Expanded focused validation passed 326/326. JUnit and the exact diff are in:

`docs/internal/release-gates/20260720_lfm_diskonly_tq_off_truth/`

## Current live proof

The detailed canonical proof is:

`docs/internal/release-gates/20260720_lfm_diskonly_tq_off_truth/README.md`

Named current evidence:

- cold raw Chat exact final after full prefill;
- warm raw Chat exact final with 576 `block-disk+ssm` cached tokens;
- real Electron fresh-chat visible progressive reasoning/content and exact
  persisted answer;
- real Electron Stop/Start with pre-request zero L1 and zero SSM RAM entries;
- first post-restart Electron Responses turn restoring nine ordinary SSD pages
  plus the 576-token SSM companion;
- independent raw Responses stream with separate rails and one completed
  terminal;
- health proving zero resident L1 payload and zero native-TQ activity.

This closes only the exact LFM Paged-Off/TQ-Off/SSD partial-prefix row. It does
not close the LFM family or the shared matrix.

## Final stopped state

- The real Electron Stop control was used.
- Session `89a68557-452b-4850-b189-ba9d57c4d6c5` is stopped with `pid=null`.
- Port 8016 has no listener.
- No model generation or API request should be assumed active.
- The session still persists the experimental settings: Prefix On, Paged Off,
  ten blocks, KV quantization None, Block L2 On, and the isolated LFM proof
  directory.

Do not silently reset or reuse those settings. On an authorized continuation,
restore the intended steady-state policy through the real UI and prove the
effective value across UI, SQLite, launch preview, argv, and health.

## Working-tree ownership and staging boundary

At the start of this repair, the branch and upstream both pointed to
`e8755c6b2`. The intended scoped commit consists only of the five source/test
files above plus the evidence README/artifacts, this handoff, the canonical
matrix update, and the issue-ledger update.

Preserve these unrelated/local-only paths:

- `.agents/STATUS.md` and `.agents/LOG.md` are local coordination state. Update
  them, but never commit them.
- `panel/node_modules` is untracked vendor noise. Never stage it.
- `AGENTS.md` is a local worktree guard. Never commit it.

Before further work, run `git status --short`, compare `HEAD` with `@{u}`, and
inspect the last scoped commit. Do not assume this handoff's base is still the
live head.

## Canonical read order for the next agent

Read these files from the live checkout, in this order:

1. `AGENTS.md`
   - non-negotiable Python/Electron lane, evidence rules, release lock, and
     local-only file rules.
2. `.agents/STATUS.md`
   - newest local operational state. This file is intentionally uncommitted.
3. `.agents/LOG.md`
   - chronological proof/actions. This file is intentionally uncommitted.
4. `docs/internal/release-gates/20260716_release_closeout/CURRENT-MATRIX.md`
   - authoritative additive public-checkpoint and retained PARTIAL/OPEN matrix.
5. `docs/internal/ISSUE-LEDGER.md`
   - defect-level status and current evidence pointers.
6. `docs/internal/release-gates/20260720_release_checkpoint_1_6_14/README.md`
   - exact public v1.6.14 tag, signed/notarized artifacts, installed UI proof,
     public surface hashes, and retained boundaries.
7. `docs/internal/release-gates/20260720_lfm_diskonly_tq_off_truth/README.md`
   - the last scoped cache/storage repair and live proof.
8. `docs/internal/release-gates/20260720_lfm_native_reasoning_protocol/README.md`
   - LFM native reasoning truth and still-failing required-tool row.
9. `docs/internal/CACHE_DETAIL_GRAMMAR.md`
   - cache-detail naming contract; do not invent telemetry labels.
10. `docs/internal/release-gates/20260720_nemotron_omni_session_l2/README.md`
    - latest post-v1.6.14 architecture-owned q4-attention/native-SSM session
      persistence proof at base commit `e8755c6b2`.

Do not look for the previously mentioned
`docs/internal/PYTHON_ENGINE_MODEL_GATE_MATRIX.md` or
`docs/internal/CACHE-DEFAULTS-UI-WIRING-MATRIX.md`; neither exists in this
checkout at this stop. The canonical matrix and ledger above are the current
replacements.

## High-signal current gate documents by domain

### Shared API, streaming, terminal, and recovery

- `docs/internal/release-gates/20260719_midstream_failure_recovery/README.md`
- `docs/internal/release-gates/20260720_anthropic_ollama_midstream_failure/README.md`
- `docs/internal/release-gates/20260719_response_cancel_disconnect/README.md`
- `docs/internal/release-gates/20260719_chat_disconnect_stop_recovery/README.md`
- `docs/internal/release-gates/20260719_chat_terminal_usage_parity/README.md`
- `docs/internal/release-gates/20260719_responses_usage_extension_parity/README.md`
- `docs/internal/release-gates/20260719_ollama_multitool/README.md`
- `docs/internal/release-gates/20260719_ollama_stream_tool_parity/README.md`
- `docs/internal/release-gates/20260719_anthropic_tool_parity/README.md`

### Cache hierarchy, disk-only, partial prefix, and lifecycle

- `docs/internal/release-gates/20260719_paged_ram_ssd_hierarchy/README.md`
- `docs/internal/release-gates/20260719_block_disk_only_partial/README.md`
- `docs/internal/release-gates/20260719_nonpaged_prompt_disk_partial/README.md`
- `docs/internal/release-gates/20260719_prompt_disk_payload_prefix_index/README.md`
- `docs/internal/release-gates/20260719_prompt_disk_stop_role_durability/README.md`
- `docs/internal/release-gates/20260719_path_dependent_terminal_cleanup/README.md`
- `docs/internal/release-gates/20260716_tq_toggle_parity/SUMMARY.md`

### Current family/runtime gates

- Qwen 3.6/JANGTQ/MTP:
  - `docs/internal/release-gates/20260719_qwen35_jangtq_current/README.md`
  - `docs/internal/release-gates/20260719_qwen35_hybrid_diskonly/README.md`
  - `docs/internal/release-gates/20260717_qwen36_mtp_stream_history/README.md`
  - `docs/internal/release-gates/20260717_qwen_fullkv_mixed_tq/README.md`
- MiniMax:
  - `docs/internal/release-gates/20260719_minimax_m27_tq_hierarchy_protocol/README.md`
  - `docs/internal/release-gates/20260719_m3_terminal_dispatch_large_video/README.md`
  - `docs/internal/release-gates/20260719_m3_current_postfinalizer/README.md`
- DSV4/ZAYA/openPangu:
  - `docs/internal/release-gates/20260719_dsv4_terminal_dispatch_native_l2/README.md`
  - `docs/internal/release-gates/20260719_zaya_typed_cca_terminal_l2/README.md`
  - `docs/internal/release-gates/20260719_openpangu_current_disk_restore/README.md`
  - `docs/internal/release-gates/20260719_openpangu_typed_nonpaged_partial/README.md`
- Gemma/Step/Laguna/Bonsai/HY3:
  - `docs/internal/release-gates/20260720_gemma4_media_stream_cache/README.md`
  - `docs/internal/release-gates/20260719_gemma_mixed_swa_disk_only_ui/README.md`
  - `docs/internal/release-gates/20260719_current_step37_jangtq/README.md`
  - `docs/internal/release-gates/20260719_laguna_current_stream_tq_determinism_eviction/README.md`
  - `docs/internal/release-gates/20260716_bonsai_current_head/SSM-L2-RESTORE-CLOSEOUT.md`
  - `docs/internal/release-gates/20260719_current_hy3_mtp/README.md`
- Nemotron Omni:
  - `docs/internal/release-gates/20260720_nemotron_omni_audio/README.md`
  - `docs/internal/release-gates/20260720_nemotron_omni_session_l2/README.md`

### App/UI/settings/release

- `docs/internal/release-gates/20260719_one_model_swap_soak/README.md`
- `docs/internal/release-gates/20260719_gateway_disconnect_recovery/README.md`
- `docs/internal/release-gates/20260719_stale_path_recovery_live/README.md`
- `docs/internal/release-gates/20260719_minwidth_locale_drawers/README.md`
- `docs/internal/release-gates/20260720_minwidth_drawer_followthrough/README.md`
- `docs/internal/release-gates/20260720_minwidth_accessibility_followthrough/README.md`
- `docs/internal/release-gates/20260719_full_suite_checkpoint/README.md`
- `docs/internal/release-gates/20260720_release_checkpoint_1_6_14/README.md`

Older gate directories remain useful provenance, but the newest source-plus-
live row in `CURRENT-MATRIX.md` wins when conclusions conflict.

## Retained work: highest priority

Do not rerun already-green short prompts just to accumulate screenshots. Pick a
retained row that lacks current proof or has a current failure.

1. **LFM required tools and exact protocol breadth**
   - Root-cause the malformed native required-tool arguments without forced
     calls, parser-side value repair, prompt coercion, or artifact blame.
   - Rerun Chat/Responses/Anthropic/Ollama stream and non-stream, required/auto/
     no-tool, real result continuation, cancellation, disconnect, and recovery.
   - Keep the new Paged-Off/TQ-Off cache row as a regression, not the whole
     family gate.
2. **Cross-family post-tool/reasoning emission**
   - Ensure every parser family emits distinct progressive reasoning and
     content, never finalizes reasoning-only, never terminal-batches a fallback
     answer, and does not repeat tools or stale reasoning across distinct
     prompts.
   - Retained gaps include MiMo and any configured parser/model not covered by
     the current exact-one-tool matrix.
3. **Cache storage breadth**
   - Paged Off: prove partial SSD prefix/block reuse, restart restore, and
     explicit TQ Off for additional compatible hybrid/full-KV families.
   - Paged On: prove RAM-first lookup, SSD fallback/refault, partial matching,
     bounded eviction, fault injection, and honest hit accounting.
   - Preserve architecture-specific policies: standard full KV may use q4 TQ;
     Bonsai is the q8 exception; hybrid SSM compresses compatible attention KV
     while preserving/rederiving companion state; Gemma mixed SWA, DSV4
     composite, ZAYA CCA, openPangu, M3 sparse cache, and Omni sessions retain
     their typed/native rules.
4. **Media breadth**
   - Real Electron attachments and API content parts for image, video, and
     audio where advertised.
   - Same-media reuse, different-media salt isolation, restart/L2 restore,
     post-media text recovery, and post-media tools.
   - MiniMax M2.7 is text-only; M3 is the MiniMax family with VL routes.
5. **Settings/startup parity and lifecycle**
   - Model-derived reasoning/tool/cache/MTP/media defaults across UI, SQLite,
     preview, argv, capabilities, and health.
   - Explicit user Off must remain honored; Auto remains the default policy
     where compatible.
   - Single-model mode must unload the old model before starting the next,
     avoid duplicate RAM residency, handle port conflicts/LAN rollback, and
     recover from disconnects.
   - Expand eager materialization beyond already-proven routes: model load must
     occur on session Start, before the first message.
6. **Family quality/reliability gaps retained by the matrix**
   - MiMo exactness/media/runtime boundaries;
   - Qwen 27B/35B MTP largest-context/cancellation/media/variant breadth;
   - DSV4 controlled reference-runtime sampling A/B and long exact code/file
     output;
   - Laguna latency/reliability/eviction;
   - Bonsai partial-prefix and long pre-tool reasoning;
   - Step stochastic completion/larger-video;
   - M3 larger-video/OCR/terminal-delay breadth;
   - openPangu long-context/protocol breadth;
   - Gemma per-quant and media/audio exactness;
   - Mistral 3.5/Pixtral JANGTQ blank/quality rows if still configured.
7. **UI/accessibility and soak**
   - Remaining custom/destructive modals, drawer/modal keyboard traversal,
     screen-reader semantics, transient states, translated labels, and signed-
     app minimum-width repetition.
   - Repeated swaps, unload/reload, soft/deep sleep, port conflicts, LAN,
     gateway, and network-loss soak.
8. **Release discipline**
   - v1.6.14 is the last proven public checkpoint. Post-release commits need
     fresh full suites, bundled-Python parity, installed-app proof, signing,
     notarization, publication, and public re-read before any later version can
     be called released.
   - Do not package or publish merely because one new cache/family row passes.

## First authorized resume actions

1. Re-read all canonical files above and `git status --short`.
2. Confirm the stopped LFM state and no port-8016 listener.
3. Confirm the scoped handoff commit is present locally and on the upstream
   branch; do not overwrite newer work if HEAD moved.
4. Through the real Electron UI, restore desired steady-state LFM settings
   (expected policy should be chosen from the then-current matrix, not from
   memory), then verify DB/preview/argv/health.
5. Choose one highest-priority red row and record `Changed / Proved / Still
   open / Next / Forbidden until green` after the proof.

## Explicit stop boundary

No model is running for this LFM session. No further tests, model loads,
packages, signatures, notarizations, tags, or publications are authorized by
this handoff itself.
