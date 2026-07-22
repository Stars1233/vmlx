# vMLX 1.6.16 exhaustive ranked open-issues checklist

Last reconciled: 2026-07-22 (America/Los_Angeles)

Status: `ACTIVE / NOT RELEASE-READY`.

This is the living to-do list for the Python/Electron 1.6.16 campaign. The
canonical narrative and retained proof live in `README.md`; this file is the
compact requirement-by-requirement checklist. A checked item requires current
source trace, focused tests, live Electron computer-use evidence, raw protocol
evidence where applicable, and a committed sanitized artifact. Source reading,
an older transcript, a focused test, a load-only run, or one short answer is not
enough.

Reconciled evidence cutoff:

- behavior-bearing runtime source: `f9a4b6838b398312a951e78299e5457ce35d68b7`;
- first M3 evidence checkpoint: `6de9ce8eff206e8a77f65f2ab191c2b3aa971390`;
- current branch: `codex/v1.6.16-release-campaign-20260722`;
- clean source checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.15`;
- live proof checkout: `erics-m5-max.local:/Users/eric/mlx/vllm-mlx-release-1.6.13`;
- current branch is ten commits ahead of `origin/main` and not yet release-integrated;
- version surfaces remain 1.6.15. Do not bump until the selected runtime cutoff
  stops changing.

Historical matrices were also inspected but are not silently authoritative:

- `/Users/eric/mlx/vllm-mlx-168rel/docs/internal/PYTHON_ENGINE_MODEL_GATE_MATRIX.md`
  is a May-era local matrix with valuable family failure history but stale
  source paths, policies, and TODO state. Its unclosed rows are represented
  below; its old PASS labels do not override current live gates.
- `/Users/eric/mlx/vllm-mlx/docs/internal/CACHE-DEFAULTS-UI-WIRING-MATRIX.md`
  is a June-era matrix whose paged-Off generic default predates the current
  architecture-aware policy. It is retained as historical evidence, not as a
  conflicting release directive.

## Status language

- `VERIFIED-LIVE_SCOPED`: the named behavior and artifact only.
- `PARTIAL`: useful evidence exists, but the acceptance row is incomplete.
- `OPEN`: required proof or implementation is missing.
- `FAIL-LIVE`: a current run contradicted the acceptance criterion.
- `BLOCKED_CURRENT_ARTIFACT_RUNTIME`: the exact available artifact cannot yet
  produce a releasable result; do not substitute an easier artifact.
- `STALE-REUSE-CANDIDATE`: older evidence may be reused only after proving the
  owning source path is unchanged.

## Do not repeat these scoped rows unless their owning source changes

- [x] Plain-language cache UI: **In-Memory Paged Cache (RAM)** versus
  **Block Disk Cache (L2)** SSD, including tooltip-click safety and minimum-width
  visual proof (`4558dac06`). Backend option names remain unchanged.
- [x] Split-marker holdback source/test coverage for `<think>`, `[THINK]`, and
  `<mm:think>` plus a representative live no-leak row (`ff293d1e7`). Family
  breadth remains below.
- [x] Normal Save/Save & Restart sequencing, update broadcast, and live PID
  truth for one Laguna session (`951eab25d`). Cross-family/default/failure
  persistence remains open.
- [x] Current Laguna JANG_2L Electron reasoning, one-tool continuation, four
  streamed protocols, q4 full-attention storage, Paged-On RAM reuse, and a
  same-process Paged-Off SSD partial-prefix row. Disk-only process restart and
  long eviction remain open.
- [x] Current MiniMax-M3 real Start-before-prompt load, native MSA health,
  Electron reasoning/content IPC separation, one real file tool, raw Responses,
  and direct/gateway Ollama Auto/On/Off (`6de9ce8ef`). Current VL availability
  and sparse-cache restart/eviction remain open.
- [x] Qwen3.6 35B JANGTQ/MXTQ four-protocol single-tool continuation and
  model-derived sampler representative. Multi-tool/cancellation/non-stream and
  other Qwen variants remain open.
- [x] LFM Chat/Responses progressive reasoning and exact single-tool result
  continuation. Electron, Anthropic/Ollama, and disk restart remain open.
- [x] v1.6.15 remains immutable. Its source release is in `jjang-ai/vmlx`; its
  signed DMGs are in `jjang-ai/mlxstudio`. 1.6.16 must repeat both surfaces.

## P0: release-critical shared runtime contracts

### P0.1 Reasoning, visible content, and terminal correctness

- [ ] Auto is model-owned variable reasoning. For an eligible family, Auto
  must enable the model's native reasoning-capable route without forcing a
  visible reasoning rail on every easy prompt. Explicit On and Off must remain
  honored or return an explicit unsupported-mode error.
- [ ] Chat Completions streams private reasoning only in
  `delta.reasoning_content`, visible output only in `delta.content`, fragmented
  tool arguments only in `delta.tool_calls`, then one truthful finish and
  `[DONE]`.
- [ ] Responses streams reasoning-summary events, output-text events, function
  events, and exactly one completed/incomplete/failed terminal with monotonic
  item/content indices.
- [ ] Anthropic balances `content_block_start/delta/stop` for normal reasoning
  and the adversarial Gemma4 visible-text then late-reasoning order; indices
  must be monotonic and one `message_stop` must end success.
- [ ] Ollama Auto/On/Off normalizes family-specific reasoning policy in stream
  and non-stream modes; `think:false` strips historical private reasoning;
  one `done:true` carries the truthful `done_reason`.
- [ ] No raw reasoning or parser marker appears in visible content, including
  split `<think>`, `[THINK]`, `<mm:think>`, DSML/XML/JSON tool control text, or
  reasoning-only malformed tool residue.
- [ ] A reasoning-only first pass never finalizes as an empty successful
  assistant message. Any bounded answer pass must itself stream progressively
  and expose truthful per-pass timing/usage.
- [ ] Different prompts do not reuse byte-identical stale reasoning. No
  looping, repeated preamble, replacement characters, broken Markdown/math,
  EOS drift, random suffix, or terminal-batched visible answer is accepted.

Named live deltas still required after current scoped proofs:

- [ ] MiniMax-M3 Ollama non-stream plus post-tool continuation.
- [ ] Mistral4/openPangu Ollama stream/non-stream On/Off and family-specific
  effort mapping.
- [ ] Gemma4 Anthropic visible-text to late-reasoning to final/tool ordering.
- [ ] Qwen/Bonsai two-turn Ollama Off history stripping.
- [ ] DSV4 Responses Auto/On/Off plus one real DSML tool result.
- [ ] MiniMax M2.7 streaming On/Off/Auto plus a real tool result.
- [ ] LFM Electron plus Anthropic/Ollama continuation.

### P0.2 Coding-harness-shaped agentic tool loops

- [ ] Implement or retain a lightweight terminal harness with allowlisted real
  tools `file_info(path="panel/package.json")` and
  `run_command(command="pwd")`. Never inject fake results or execute arbitrary
  model-authored commands.
- [ ] On direct model ports and the Electron gateway, exercise streamed and
  non-streamed Chat, Responses, Anthropic, and Ollama.
- [ ] Cover no-tool, Auto, required, explicit function choice, two sequential
  tools, real-result continuation, fresh post-result reasoning, and one final
  visible synthesis. Ollama's lack of public `tool_choice` is tested by exposing
  only the permitted tool for each round.
- [ ] Reject missing required arguments, malformed fragments, unknown tools,
  repeated calls, duplicate sends, hallucinated results, stale result IDs, and
  false success terminals.
- [ ] Cancel/disconnect during the post-tool continuation, verify no false
  completed terminal or completed-history row, wait for zero running/waiting
  requests, then exact-final an immediate recovery request.
- [ ] Retain timestamped sanitized JSONL: request mode, delta kinds/times,
  hashes/lengths, tool IDs/arguments, real-result hash, terminal type/count,
  cancellation time, recovery latency, and normalized direct/gateway diff.

### P0.3 Cache hierarchy and TurboQuant/native-state correctness

For each archetype, prove coherent cold store, resident hit, partial-prefix
match, L1 eviction, SSD refault, process restart restore, missing/corrupt
companion fallback, cross-chat and cross-session reuse, and unrelated-suffix or
media isolation.

- [ ] Standard full KV: q4 TurboQuant storage by default, including MiniMax
  M2.7 text-only; explicit UI/API Off must disable TQ without disabling safe
  cache reuse.
- [ ] Hybrid SSM/GDN: q4 only for eligible attention KV; native SSM/GDN state
  is restored or cleanly re-derived. Bonsai alone uses the documented q8
  attention-storage exception. No companion mismatch may produce coherent-looking
  but wrong tokens.
- [ ] Mixed rotating/SWA: q4 only for eligible full-attention KV; native
  rotating-window state remains typed. Cover Laguna, Gemma4, and Step variants.
- [ ] Typed CCA: preserve all CCA path-dependent state; enable q4 only for a
  proven eligible KV component. A missing typed companion forces safe prefill.
- [ ] MiniMax-M3: preserve native `attention_kv`, `msa_idx_keys`, and absolute
  block index. Generic TQ remains Off. Prove partial, eviction, and disk restart.
- [ ] DSV4 Flash: preserve the exact MLA/local-global composite and native pool
  codec. Generic TQ remains Off. Prove partial, eviction, disk restart, and
  coherent short/long output.
- [ ] openPangu 2.0 Flash: preserve exact source/bundle MLA/DSA/SWA/attention
  sinks/mHC prompt state. Generic TQ remains Off. Prove native Paged-Off disk
  behavior and long context.
- [ ] Paged Off plus L2 On: partial prefix must be found on SSD with zero L1
  resident paged bytes. Restart the process and repeat the partial match.
- [ ] Paged On plus L2 On: use matching RAM blocks first, promote missing
  blocks from SSD, and full-prefill only absent or unsafe blocks.
- [ ] Block size/count, memory percentage, L2 size/path, partial terminal block,
  N-1 prompt snapshot, eviction order, and media salt appear truthfully in
  health/counters. Cache metrics alone never override output-quality failure.
- [ ] Native MTP: only artifacts named/configured for MTP enable it; requested
  depth and actual accepted depth agree; proposal/accepted counts are truthful;
  only accepted canonical tokens enter reusable history/cache.

### P0.4 Settings, defaults, budgets, restart, and metrics parity

- [ ] For every representative artifact, hash/read `config.json`,
  `generation_config.json`, `tokenizer_config.json`, `chat_template.jinja`, and
  `jang_config.json` before judging the UI.
- [ ] First-use Chat Settings visibly inherit temperature, top-p, top-k
  including Off/-1/large values, min-p zero, repetition penalty, max output,
  reasoning Auto, parser defaults, and tool availability from the bundle.
- [ ] Saved chat settings are tied to the intended chat/session and survive app
  and engine restart. Reset/Auto removes explicit overrides and returns to
  bundle inheritance; it must not write a hidden forced value.
- [ ] Server Settings persist and agree across UI, SQLite, IPC, preview, actual
  argv/env, health, and resolved generation kwargs: parsers, reasoning default,
  MTP, modalities, prefix/paged/L2/TQ/JIT/batching, block size/count, RAM %, L2
  size/path, port/LAN, and Single Model.
- [ ] Normal Save, Save & Restart, manual Stop/Start, failed restart, late loader
  failure, and stopped-session update paths retain one truthful process/PID and
  do not silently discard settings.
- [ ] `max_output_tokens` is never confused with model context length. UI, DB,
  all protocol adapters, preview, argv, and engine report the same effective
  values. A real explicit budget may truthfully yield incomplete; no unexplained
  32/48/96/160-token hidden cap is accepted.
- [ ] For the same prompt, compare Electron `metrics_json` to raw SSE timings.
  Separately report TTFT, prefill speed, decode speed after first output token,
  reasoning/visible/tool tokens, answer-pass/tool pauses, and wall time. Do not
  blend two-pass work into a misleading token/s value.

Minimum unusual-default representatives:

- [ ] MiniMax M2.7 top-k 40/full-KV q4.
- [x] openPangu top-p 0.8/top-k 151552 is visually scoped; payload/argv/resolved
  runtime and restart remain open.
- [ ] HY3/ZAYA top-k Off or -1.
- [ ] Nemotron/Gemma non-neutral repetition or temperature.
- [ ] base MLX/MXFP, affine JANG, JANGTQ/MXTQ, M3 native, and DSV4 native
  detector classes without conflating their formats.

### P0.5 Gateway, one-model ownership, eager loading, and recovery

- [ ] Single Model On repeatedly swaps through real Electron Start controls and
  leaves exactly one owned model process/resident allocation. Single Model Off
  permits the configured multi-model behavior.
- [ ] Start eagerly materializes model weights before the first chat request for
  every loader class; the card, PID/RSS, pre-request health, and log prove it.
- [ ] Direct and gateway routes target the selected session without stale port
  or model routing. Occupied port, stale path, failed target, late loader
  failure, backend loss, disconnect, and unload/reload roll back or recover.
- [ ] Active-request LAN/port changes fail atomically with visible rollback;
  idle changes rebind correctly. Soak concurrent clients, swaps, cancellation,
  backend loss, and immediate recovery across all four protocols.

### P0.6 Release-source and packaging provenance

- [ ] Merge/integrate the exact proven 1.6.16 branch into `main`; build only
  from the immutable pushed commit. Current branch is ahead of main and is not
  release integration.
- [ ] Create a clean GitHub-backed JANG release worktree. `/Users/eric/jang` is
  currently dirty; the earlier `/Users/eric/jang-release-prep-20260721` path is
  missing; `/Users/eric/mlx/jangq-release-clean-4129f28` tracks the local repo,
  not GitHub directly. Do not bundle until its exact GitHub SHA, clean tracked
  and untracked state, and required files are proven.
- [ ] Preflight tracked and untracked cleanliness, expected vMLX/JANG SHAs,
  version parity, owned non-symlink `node_modules`, Node PATH, and an immutable
  shared provenance manifest before bundling and before each flavor.
- [ ] Packaged imports (`sys.executable`, `vmlx_engine.__file__`,
  `jang_tools.__file__`) remain inside the app; missing JANG, absolute dev
  `direct_url.json`, poisoned env, raw-source mismatch, or offline-manifest
  bypass is a hard failure.

## P1: required model-family and architecture rows

Every row uses real Electron Start, at least three meaningful turns including
one tool/result continuation, bundle-ground settings, raw protocol deltas, and
the applicable cache/media axes. Official JANGQ/dealignai artifacts are trusted
inputs; integration is investigated first. Do not replace a named quant/model
with an easier unrelated model.

| Family/artifact | Required remaining proof | Status |
|---|---|---|
| Laguna S-2.1 JANG_2L/JANG_4M | >512-token SWA coherence, disk-only process restart, bounded eviction, long agent loop, sampler/TPS parity, saved settings restart, truthful ~40+ tok/s on the named JANG_4M where expected | `PARTIAL` |
| Qwen3.6 35B/27B named MTP artifacts | Distinguish JANGTQ/MXTQ Hadamard-codebook from affine JANG and base MLX/MXFP; multi-tool, cancellation, non-stream, MTP depth/accept accounting, q4 attention plus native hybrid state, image/video and salt | `PARTIAL` |
| HY3 JANG with MTP | Exact configured MTP depth, proposal/accept/compression accounting, cache safety, stochastic/long quality, shared protocol rails | `PARTIAL` |
| Bonsai 27B 1-bit and variants | q8 attention-only TQ exception, SSM/GDN companion rederive, partial prefix/eviction/restart, long pre-tool reasoning, no looping or reasoning-only final | `PARTIAL` |
| Ornith and other Qwen variants | Exact config-derived parser, modality, MTP, quant, sampler, hybrid topology, three turns, tool loop, cache, media | `OPEN` |
| MiniMax M2.7 affine JANG | Text-only, full-KV q4 TQ, no VL claim, Auto/On/Off, all protocols, tool, settings, partial/eviction/restart | `OPEN / ARTIFACT MUST BE LOCATED` |
| MiniMax M3 affine JANG_2L / variants | Current text/reasoning/tool/Ollama scoped pass; still VL runtime availability, image/video salt A/B/A, post-media tool, sparse partial/eviction/restart, non-stream/post-tool Ollama, REAP variants | `PARTIAL` |
| DSV4 Flash affine/JANG and exact JANGTQ variants | Native composite cache/pool only, DSML tools, all protocols, eager load, partial/eviction/restart, long/medium Auto quality. Independent matched reference A/B before blaming or mutating artifact | `FAIL-LIVE LONG AUTO / PARTIAL` |
| Gemma4 JANG_4M/MXFP8 dense/MoE/rotating-SWA | Mixed rotating/full q4 eligibility, Anthropic late reasoning, unusual sampler defaults, image/video/audio only when bundle advertises, salt/restart/post-media tool, long context | `PARTIAL` |
| Nemotron Nano/Omni | Bundle/index proof for Parakeet/RADIO/audio/VL, Auto/parser persistence, hybrid q4 attention/native state, image/video/audio UI+API, salt/restart/tool | `PARTIAL` |
| Step Flash / Step 3.7 | Shared rails, cold latency/stochastic quality, restarted PID telemetry, larger video, audio only if advertised, salt/restart/post-media tool | `PARTIAL` |
| openPangu 2.0 Flash | Exact typed architecture, no generic TQ, long context, sampler/runtime parity, all protocols/tools, Paged-Off native prompt disk partial/restart | `PARTIAL` |
| LFM2.5 | Current Chat/Responses scoped pass; still Electron, Anthropic/Ollama, three-turn history, q4 attention/native SSM, Paged-Off partial and restart, truthful large output budget | `PARTIAL` |
| ZAYA/CCA and related typed CCA | Complete typed state, eligible KV-only q4 if proven, safe missing companion, partial/eviction/restart, settings/tools/media | `PARTIAL` |
| Mistral 3.5/Pixtral JANGTQ/MXTQ | Root-cause current blank/whitespace output across model port, quantized coverage, template, parser, and media. Do not test base Mistral MXFP4 as a substitute | `BLOCKED_CURRENT_ARTIFACT_RUNTIME` |

## P2: media, conversion, and product breadth

- [ ] For every exact bundle-advertised modality, use the real Electron file
  picker/attachment surface plus raw protocol payloads. Never infer support
  from family naming or outer config alone.
- [ ] Cover Qwen/Bonsai/Ornith image/video, Gemma4 image/video/audio where
  tensors/config advertise it, Step media, Nemotron/Omni audio/image/video,
  and MiniMax-M3 image/video. MiniMax M2.7 remains text-only.
- [ ] Prove same-media reuse, same-shape different-media miss, A/B/A media-salt
  isolation, partial prefix, eviction, restart L2, post-media text, post-media
  tool continuation, progressive reasoning/content, and truthful terminal.
- [ ] Complete Electron JANG conversion for affine profiles/custom mixes and
  generic JANGTQ/MXTQ. Cover overwrite, low disk, unwritable target,
  calibration/AWQ/imatrix, large MoE, cancel/resume, metadata, independent
  reload, multi-turn/tool/media/cache behavior, and chat-quality output.
- [ ] Audit `/Users/eric/jang` missing/untracked production files. Move only
  reviewed production changes into a clean branch, run its full relevant suite,
  push the exact JANG source, and version it only if package behavior changes.
- [ ] Finish minimum-width and translated UI states, secondary modals,
  keyboard/screen-reader semantics, stale missing-model-path repoint/remove,
  wait/empty/image states, and dead/zombie owner cleanup.

## Final 1.6.16 release gate

- [ ] Freeze the selected source cutoff and update this checklist, `README.md`,
  `ISSUE-LEDGER.md`, `.agents/STATUS.md`, `.agents/LOG.md`, and the retained
  closeout matrix with exact PASS/PARTIAL/FAIL boundaries.
- [ ] Run the full Python suite, full panel Vitest, typecheck, production build,
  bundled-Python/source parity, release regression manifest, and clean-diff
  checks on the final cutoff. Focused suites do not count.
- [ ] Bump all version surfaces to 1.6.16 only after the cutoff is frozen.
- [ ] Build separate Sequoia and Tahoe apps/DMGs from the same provenance
  manifest; Developer ID sign, notarize, staple, deep codesign verify,
  Gatekeeper verify, and hash.
- [ ] Install both under isolated names and run signed-app Electron Start/Stop,
  reasoning/content/tool/history, gateway/API, cache warm/restart, and engine
  path/version proof.
- [ ] Tag exact built source; publish `jjang-ai/vmlx` source release,
  `jjang-ai/mlxstudio` DMGs/blockmaps, PyPI, Homebrew, and updater manifests.
  Re-download public artifacts, rehash, verify public versions/URLs, and retain
  the reads. No AI attribution in public text.

Until every selected release gate above has named current evidence, the only
truthful verdict is `PARTIAL / NOT RELEASE-READY`.
