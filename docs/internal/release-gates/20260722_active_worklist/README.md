# 2026-07-22 Active Worklist and Proof Tracker

Status: `PARTIAL_NO_RELEASE`.

Host/repo for this checkpoint:

- Host: `erics-m5-max.local`
- Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
- Branch: `codex/postrelease-ui-drawers-20260720`
- Current pushed head at tracker creation: `9e1c98a22 docs(qwen): prove anthropic ollama tool continuation`

This tracker is intentionally conservative. It exists to keep the current
post-release work tied to the campaign matrix rather than closing one-off rows
from isolated prompts. A row below may be considered closed only when the
current source trace and live Electron/API evidence are both named.

## Source of truth read before continuing

- `AGENTS.md`
- `.agents/STATUS.md`
- `.agents/LOG.md`
- `docs/internal/ISSUE-LEDGER.md`
- `docs/internal/release-gates/20260716_release_closeout/CURRENT-MATRIX.md`
- `docs/internal/release-gates/20260721_reasoning_protocol_global/README.md`
- `docs/internal/release-gates/20260721_laguna_s21_jang2l_cache_reasoning/README.md`
- `docs/internal/release-gates/20260721_qwen36_jangtq_tool_parser_current/README.md`
- `docs/internal/release-gates/20260721_jangtq_sampling_chat_lifecycle_current/README.md`
- `docs/internal/release-gates/20260720_sampling_defaults_ui_runtime/README.md`
- `docs/internal/release-gates/20260720_dsv4_m3_current_typed_cache/README.md`

## Already closed scoped rows to avoid repeating unless they regress

These are not release-wide passes. They are named scoped closures already backed
by current-source evidence and should be reused instead of rerunning the same
model/prompt loops.

| Area | Current scoped evidence | Boundary still open |
|---|---|---|
| Qwen3.6 35B JANGTQ/MXTQ single-tool parser | `20260721_qwen36_jangtq_tool_parser_current/`; source `9e1c98a22`; live Electron exact one `file_info`; raw Chat, Responses, Anthropic, and Ollama single-tool continuation | Multi-tool interleaving, cancellation/failure, non-stream breadth, SSD-only/media rows, duplicate-send classification |
| JANGTQ sampler/session lifecycle on Qwen3.6 | `20260721_jangtq_sampling_chat_lifecycle_current/`; bundle-owned temp/top-p/top-k shown in Electron and raw Responses | Base MLX/MXFP, DSV4/M3 native-cache routes, and non-neutral repetition-penalty default |
| Gemma sampler Top-K zero/reset scoped chain | `20260720_sampling_defaults_ui_runtime/`; explicit Top-K Off and reset-to-bundle path through UI/SQLite/request/engine/API | JANGTQ/MXFP/native-cache family repeats and non-neutral repetition penalty |
| Laguna S2.1 JANG_2L Responses/Electron tool/cache | `20260721_laguna_s21_jang2l_cache_reasoning/`; Auto reasoning, Electron one-tool, raw Responses one-tool, q4 mixed-SWA cache, paged RAM partial, restart L2, eviction, SSD-only partial | Chat/Anthropic/Ollama result continuations, long-context/SWA-boundary quality, JANG_4M protocol rows, strict-format variability |
| DSV4/M3 typed cache current source | `20260720_dsv4_m3_current_typed_cache/`; DSV4 composite cache and M3 sparse-index cache not flattened into generic TQ | DSV4 long quality/protocol breadth; M3 current VL/video row remains PARTIAL |
| Gateway non-stream atomic JSON | `20260721_gateway_nonstream_atomic_current/`; parseable 502 on reset plus immediate recovery | Broader soak, simultaneous config mutations, signed-app repetition, model/parser breadth |
| Eager materialization representatives | `CURRENT-MATRIX.md` names DSV4, Laguna, Step, openPangu, Gemma, HY3, MiniMax M2.7, and swap soak | Inventory only loader classes not represented; signed-app repeat |

## Current release blockers that remain active

| Priority | Blocker | Why it remains open | Required live proof before closing |
|---|---|---|---|
| P0 | Global reasoning/content/tool rail correctness | The global reasoning gate is still `PARTIAL`; historical failures include inline `<think>`, reasoning-only finalization, post-reasoning batched content, incomplete tool markers, and false terminals | For representative parser families, raw Chat + Responses + Anthropic + Ollama must stream reasoning deltas on reasoning rails, visible content deltas on content rails, no inline/native marker leak, one truthful terminal, and tool-result continuation from a real result. Electron must visibly paint reasoning and post-reasoning content progressively. |
| P0 | Agentic tool loops across protocol surfaces | Single-tool scoped rows exist, but broad `reasoning -> tool -> result -> reasoning -> tool/final` behavior is not globally closed | Use actual coding-harness-shaped API calls: no-tool, required-tool, auto-tool, real result continuation, repeated/interleaved tool calls, cancellation/recovery, and no hallucinated tool results. |
| P0 | Laguna S2.1 complete runtime row | JANG_2L has strong scoped proof, but JANG_4M and full protocol/long-context/sampling rows remain open | Electron Start, settings/defaults, Auto reasoning, one-tool/final, Chat/Responses/Anthropic/Ollama continuations, long context beyond SWA window, q4 TQ full-attention KV only, native rotating SWA, paged and SSD-only partial reuse, restart L2, eviction, truthful speed/TTFT. |
| P0 | SSD/block-disk L2 partial prefix behavior | Several scoped models prove pieces, but the matrix still requires architecture-aware breadth and explicit Paged-Off behavior | For standard KV/TQ, hybrid SSM/GDN, mixed-SWA, typed CCA, M3 sparse, DSV4 composite, and openPangu native prompt disk: cold store, RAM hit, evicted SSD refault, Paged-Off SSD-only partial match, Paged-On L1 then L2 fallback, restart restore, and safe full-prefill when companion/native state is missing. |
| P0 | Model-derived Chat Settings defaults | User observed sliders not matching inherent bundle defaults; existing proof covers Gemma and Qwen scoped rows only | For selected non-duplicate artifacts, compare `generation_config.json`/`config.json`/`jang_config.json` to UI drawer, SQLite, preview/argv, request payload, and engine-resolved kwargs. Explicit Off/zero must remain honored and Auto must return to bundle inheritance. |
| P0 | Reasoning Auto default | User directive: Auto should default reasoning on for eligible reasoning models. Some rows show Auto wired, but broad family proof remains incomplete | Bundle-ground `default_enable_thinking` or model registry contract; prove UI Auto sends `enable_thinking=true` where expected and raw APIs honor explicit on/off without inline thinking or hidden output stalls. |
| P1 | Gateway and one-model lifecycle | Multiple scoped lifecycle fixes exist, but broader soak remains partial | Electron visual API drawer, LAN/port rollback, repeated model swaps, concurrent swap requests, one model resident, unload/reload, backend-loss/disconnect recovery, active-request reconfigure, and signed-app repetition. |
| P1 | Media-capable families | Many scoped media rows exist, but current matrix still retains family breadth and M3/Gemma video quality boundaries | UI attach and raw API image/video/audio for Qwen/Bonsai, Gemma, Step, Nemotron/Omni, MiniMax M3 only, and other advertised routes; media-salted cache; same-media reuse; different-media miss; post-media text/tool recovery; no MiniMax M2.7 VL claim. |
| P1 | Special native architectures | User explicitly flagged DSV4 Flash and MiniMax M3 cache/attention stacks | DSV4 must stay native composite DSV4 cache with MLA/SWA/CSA/HCA and no generic TQ substitution. M3 must stay native dense-KV plus sparse/MSA index cache with generic TQ off. Live proof must include cold/warm/partial/restart and coherent output, not just load. |
| P1 | MiniMax M2.7 / M3 distinction | M2.7 is text-only full-KV; M3 is the MiniMax VL-capable route | Keep M2.7 on full-KV q4 TQ text/tool/protocol rows. Run VL only on M3 artifacts with health showing runtime availability and actual image/video tensor path. |
| P1 | OpenPangu native composite route | Existing row proves no generic TQ and prompt Disk L2 scoped behavior; long/protocol remains partial | Preserve typed MLA/DSA/SWA/sinks/mHC route. Do not enable TurboQuant. Prove long-context/protocol behavior and prompt disk reuse within safe native boundary. |
| P1 | Metrics/TPS truthfulness | Historical UI rows showed content emitted in a terminal batch and misleading token/s | Raw timed SSE and DOM mutation evidence must match UI TPS claims. If terminal usage exists, final TPS must be based on terminal usage/timing; live rolling TPS must be labeled or fixed if chunk-count based. |
| P2 | JANG/JANGTQ conversion pipeline | Developer Tools lifecycle is closed scoped; generic JANGTQ/MXTQ conversion remains open | Real Electron conversion lifecycle plus generated artifact reload/coherent output, metadata correctness, overwrite/low-disk/calibration/MoE/resume rows, and no affine/JANGTQ confusion. |
| P2 | Version/release truth and checkpoint packaging | Public v1.6.14 exists while broader matrix is partial; next release needs clean selected cutoff | Only after critical runtime rows selected for checkpoint are current-source proven: run full Python/panel suites, bundle Python from clean JANG, build Sequoia/Tahoe DMGs, sign, notarize, staple, verify, install-smoke, update manifests/releases. |

## Immediate next non-duplicate work selection

The next target should not rerun the already-closed Qwen/Bonsai/M2.7 single-tool
rows. The best current reducer is:

1. `settings + parser/template`: audit and fix model-derived sampler/default
   parity for non-duplicate artifacts whose defaults stress the UI/API path:
   MiniMax M2.7 top-k 40, OpenPangu top-p 0.8/top-k 151552/no generic TQ, HY3 or
   ZAYA top-k -1/off, and a Nemotron non-Gemma temperature row.
2. Fold the same live run into `reasoning/content rails` by using Auto/on/off
   probes and one required-tool continuation on one selected model, rather than
   separate repetitive prompts.
3. Then return to Laguna S2.1 JANG_4M/JANG_2L protocol breadth and long-SWA
   correctness with the same proof style.

## Rules for subsequent closures

- Do not blame official `jangq-ai` or `dealignai` quantized artifacts unless a
  matched reference A/B proves the artifact itself fails outside vMLX.
- Do not add fake behavior: no synthetic thinking tags, hidden sampler clamps,
  forced outputs, prompt-only parser masking, output rewrites, fake tool
  results, or generic cache substitution for native architectures.
- Every source change must have focused tests, a live Electron proof where it
  affects UI/runtime behavior, raw API proof where it affects protocols, and a
  committed evidence artifact under `docs/internal/release-gates/`.
- Keep `.agents/STATUS.md`, `.agents/LOG.md`, this tracker, and
  `CURRENT-MATRIX.md` honest: `PARTIAL` remains `PARTIAL` until current live
  proof closes the exact row.
