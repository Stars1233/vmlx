# vMLX 1.6.17 consolidation campaign

Started: 2026-07-22 (America/Los_Angeles)

Status: `ACTIVE / PARTIAL / NOT RELEASE-READY`

## Source and proof boundary

- Baseline: public v1.6.16 follow-up head
  `e0b49ec2910649b554af14508eb42e6d4eec6f31`.
- Worktree:
  `/Users/eric/mlx/vllm-mlx-r17-consolidation`.
- Branch: `codex/v1.6.17-consolidation-20260723`.
- v1.6.16 remains the released scoped checkpoint. It is not evidence that the
  rows below are complete.
- This campaign does not inherit a live pass from an old PID, installed app,
  transcript, `/tmp` capture, or another checkout.
- No v1.6.17 version bump, tag, package, signing, notarization, updater change,
  or publication is authorized until the live rows below are current.

## Required per-family proof shape

A family row is not complete until the same exact bundle has all applicable
evidence below:

1. Read and retain hashes or relevant fields from `config.json`,
   `generation_config.json`, `tokenizer_config.json`,
   `chat_template.jinja`, and `jang_config.json`.
2. Start the model with the real Electron Start button. Record app/source,
   bundled-engine/import provenance, argv, PID, health, and absence of an error
   toast. Model materialization must occur before the first prompt.
3. Run at least three Electron turns and inspect the visible UI after every
   turn:
   - private reasoning appears only in the Reasoning rail;
   - visible content is non-empty when the turn should answer;
   - content streams progressively after reasoning rather than appearing only
     at terminal;
   - one real tool call executes and its real result continues to a final
     answer;
   - no raw reasoning/tool markers, stale replay, replacement characters,
     broken Markdown/KaTeX, random suffix, duplicated output, EOS drift, or
     false terminal appears;
   - TTFT, prompt/decode TPS, token counts, and cache counters are retained and
     compared with raw timing.
4. While that Electron-started PID is loaded, run at least three multi-turn API
   sequences through the gateway and direct model port. Cover Chat
   Completions, Responses, Anthropic Messages, and Ollama in stream and
   non-stream mode as applicable. Inspect raw bytes/events, not only the final
   JSON.
5. Cover no-tool, required/explicit tool, automatic tool, tool-result
   continuation, cancellation/disconnect, truthful failure, and immediate
   recovery. Never inject a hallucinated tool result.
6. Prove cold miss, warm resident hit, changed-tail partial match, L1
   eviction, SSD refault, process restart restore, new-chat reuse, new-session
   reuse, and safe fallback for missing/corrupt typed companions.
7. For media bundles, prove real image/video/audio input, media-salted A/B/A
   cache isolation, post-media text recovery, and post-media tool continuation.

## Priority 0: shared contracts

### P0.1 Canonical settings and defaults

- [ ] Replace duplicated panel model-default readers with one tested canonical
  resolution contract. UI display, persisted session metadata, command
  preview, launch argv, engine `/health`, and per-request resolved kwargs must
  agree.
- [ ] Preserve precedence:
  explicit request/chat override > explicit saved user setting > bundle JANG
  chat stamp > generation config > documented family fallback. Never turn a
  display value into a hidden server override.
- [ ] First-use Chat Settings must reflect bundle temperature, top-p, top-k
  including Off/-1/vocab-sized values, min-p including zero, repetition
  penalty including mode-specific values, max output, max context, reasoning
  Auto, parser choice, tool policy, MTP, and media capability.
- [ ] User-edited Chat Settings remain scoped to that chat/session. New Chat
  and new Session behavior must be explicitly tested so stale overrides do not
  silently replace model-owned defaults.
- [ ] Save, Save & Restart, app restart, session restart, model swap, sleep,
  and wake must preserve every user-owned server control and recompute every
  model-owned control from the current bundle.
- [ ] Max output and max context remain separate end-to-end. The UI must not
  use `max_new_tokens` as a hidden prompt-context or server-startup override.
- [ ] Remove dead readers, stale migration branches, duplicate parser/default
  maps, and unreachable compatibility code only after tests prove the owning
  path.

### P0.2 Reasoning, parser, and protocol normalization

- [ ] Auto preserves native variable-reasoning policy. Explicit On/Off is
  honored or rejected explicitly when unsupported. No prompt coercion,
  synthetic tags, hidden sampler clamps, or fabricated reasoning rail.
- [ ] Chat emits private reasoning only as `delta.reasoning_content`, visible
  text only as `delta.content`, tool fragments only under `delta.tool_calls`,
  followed by one truthful finish and `[DONE]`.
- [ ] Responses emits monotonic reasoning, output-text, function-call, and
  exactly one terminal event.
- [ ] Anthropic block ordering remains balanced for reasoning, visible text,
  tools, tool results, and late reasoning.
- [ ] Ollama normalizes request flags and prior assistant `message.thinking`
  identically through the direct Python route and Electron gateway.
- [ ] Split `<think>`, `[THINK]`, `<mm:think>`, DSML, Qwen XML, MiniMax XML,
  and JSON tool markers never leak into visible content.
- [ ] Reasoning-only first passes do not terminate as empty success. Any answer
  pass streams progressively and reports per-phase timing honestly.
- [ ] Model-specific parser aliases resolve to one canonical parser identity
  in detection, UI, argv, health, and live stream behavior.

### P0.3 Cache hierarchy and storage limits

- [ ] Paged Off + Block Disk L2 On finds SSD partial prefixes with
  `ram_tokens_cached=0`, including after restart and across new chats/sessions.
- [ ] Paged On + Block Disk L2 On uses matching RAM blocks first, promotes
  missing safe blocks from SSD, and prefills only absent/unsafe tail tokens.
- [ ] The Paged RAM block limit and Block Disk GB slider cause observable,
  bounded eviction/refault without deleting unrelated cache roots.
- [ ] Standard KV defaults to q4 TurboQuant storage unless explicitly disabled.
  Only eligible attention KV is quantized for hybrid/rotating families.
- [ ] Typed cache state is never flattened:
  hybrid SSM/GDN, Laguna/Gemma/Step rotating or mixed SWA, ZAYA CCA,
  MiniMax-M3 MSA/indexer state, DSV4 composite state, and openPangu native
  prompt state each need family-correct restore/rederive behavior.
- [ ] Media salt, model/config/weights hash, runtime/schema fingerprint,
  quant/TQ signature, and typed companion identity participate in safe cache
  identity.

### P0.4 Electron/gateway lifecycle

- [ ] One-model-only gateway mode eagerly starts the selected model, unloads
  the previous managed model, rejects failed unloads truthfully, and never
  routes to a stale/unmanaged PID.
- [ ] Port conflicts, LAN changes/rollback, client disconnect, mid-stream
  backend failure, session stop/start, sleep/wake, and repeated model swaps
  recover without false-success events or orphan engines.
- [ ] Electron Chat and Responses wires both receive private reasoning,
  progressive visible content, tool calls/results, usage, cache detail, and
  truthful terminal state.
- [ ] KaTeX/Markdown rendering is visual-only. Raw direct/gateway API payloads
  remain byte-faithful for backslashes, dollar currency, dollar math,
  parentheses, brackets, code fences, JSON, and whitespace-sensitive output.

## Priority 0: representative family matrix

Every row below requires the shared proof shape, not load-only proof.

- [ ] MiniMax M2.7 JANG affine: reasoning/tool streams, standard KV q4 TQ,
  restart SSD partial reuse, text-only media truth.
- [ ] MiniMax M3: adaptive Auto/On/Off, tool loops, native MSA/indexer typed
  cache, partial/eviction/restart, advertised media.
- [ ] Laguna S2.1 JANG_2L and JANG_4M: native Poolside template/parser truth,
  variable Auto plus explicit On/Off, mixed rotating/SWA cache, q4 eligible
  attention KV, partial SSD reuse, eviction/refault, target performance, no
  gibberish or delayed terminal-batched answer.
- [ ] Gemma 4 JANG_4M and MXFP8: mixed-SWA cache, Anthropic late-reasoning
  ordering, tool/JSON/code exactness, image/audio/video where advertised.
- [ ] Qwen 3.6 27B and 35B MXFP8/JANGTQ with native MTP: MTP autodetect and
  accepted-depth truth, hybrid GDN/SSM companion cache, Qwen reasoning/tool
  parsers, image/video rows where advertised.
- [ ] Bonsai and Ornith: native Qwen-family reasoning/tools, q8 Bonsai cache
  exception only where documented, changed-tail SSD partial reuse, history Off
  stripping, media if advertised.
- [ ] Hy3 JANG: MTP autodetect, native thinking contract, tools, multimodal
  video/audio path, typed cache.
- [ ] DSV4 Flash: DSML tools, Auto/On/Off, coherent long output, exact
  MLA/local-global composite cache and native pool codec.
- [ ] Step Flash: text stability, reasoning/tool dialect, video/audio only when
  the exact artifact advertises and implements them, rotating/native cache.
- [ ] Nemo/Nemotron Omni: reasoning/tools plus real image/audio/video bridge,
  Parakeet path, media-salted cache, restart recovery.
- [ ] LFM: Electron plus four protocol routes, three-turn tools/history,
  hybrid cache, Paged-Off SSD restore.
- [ ] openPangu/Mistral4/ZAYA: family-specific effort/tool/parser rules and
  native typed prompt/cache state.

## Priority 1: JANG conversion and distribution

- [ ] Audit the current `jjang-ai/jangq` main and the source used for vMLX
  bundles. Do not treat the old dirty developer checkout as release truth.
- [ ] Certify affine JANG separately from JANGTQ/MXTQ Hadamard-rotation
  codebook formats and base MLX MXFP. Do not collapse their loaders, metadata,
  kernels, or cache storage policy.
- [ ] Prove conversion, interruption/resume, overwrite/low-disk/unwritable
  failure, produced-bundle metadata, independent reload, coherent multi-turn
  text/tool/media output, and cache behavior.
- [ ] Keep GitHub source branches current. PyPI publication remains a separate
  credentialed release action and must be verified publicly before vMLX depends
  on the new version.

## Current checkpoint

### R17-001 Electron gateway Ollama history normalization

Status: `SOURCE+FOCUSED_TEST PASS / LIVE MODEL PROOF OPEN`

Finding:

- Direct Python Ollama normalized prior assistant `message.thinking` into
  `reasoning_content`.
- Electron gateway text-only history returned the original message unchanged;
  media history forwarded `thinking` as an unknown field.
- Therefore a follow-up through the gateway could render a different native
  prompt from the direct route.

Change:

- Normalize assistant `thinking` before the text/media split in both routes.
- Remove the alias even when empty.
- Preserve an already supplied non-empty canonical `reasoning_content`.

Focused evidence:

- Python:
  `tests/test_ollama_adapter.py tests/test_ollama_reasoning_parity.py`
  -> `36 passed`.
- Full focused reasoning/parser/adapter/agentic protocol set after the change
  -> `472 passed`.
- Panel:
  `api-gateway-ollama.test.ts api-gateway-ollama-behavior.test.ts`
  -> `59 passed`.
- Panel typecheck -> pass.

Missing:

- Real Electron-started model with a three-turn Ollama follow-up that consumes
  prior private reasoning.
- Raw direct versus gateway body/event comparison on the same model and
  prompts.
- Required tool/result continuation after the normalized history.

### Cache source audit checkpoint

Status: `SOURCE+FOCUSED_TEST PASS / LIVE FAMILY MATRIX OPEN`

- Standard/paged/block-disk mechanics: `101 passed`.
- Hybrid/SSM companion mechanics: `28 passed`.
- Current-branch combined selected cache regression run after the first patch:
  `111 passed`.
- Source supports disk-only changed-tail probes and Paged-On L1-then-L2
  promotion.
- Remaining source risk: the MLLM `BlockDiskStore` construction does not pass
  `expected_num_layers`, weakening early header-level layer-count rejection.
- No new live model proof was run for this audit, so no cache family row is
  newly closed here.

### R17-002 isolated Electron profile and current-source engine provenance

Status: `SOURCE+FOCUSED TEST PASS / DEV LIVE PASS / SIGNED BUNDLE OPEN`

Source head and host:

- Git head: `ef0d3a8cd`.
- Electron checkout:
  `/Users/eric/mlx/vllm-mlx-r17-consolidation`.
- Live host: `Erics-M5-Max.lan` (`erics-m5-max.local`), macOS `26.3.2`.
- The local `/Applications/vMLX.app` is the stale `1.6.9` install and was not
  used or modified.

Finding and change:

- A fresh `VMLX_USER_DATA_DIR` failed before the database could open because
  the override directory did not exist.
- `panel/src/main/user-data-dir.ts` now creates the override directory before
  calling `app.setPath`.
- `panel/tests/app-user-data-isolation.test.ts` pins the startup ordering.

Focused evidence:

- `app-user-data-isolation.test.ts` -> `5 passed`.
- Panel typecheck -> pass.
- A fresh remote profile at
  `/Users/eric/.vmlx-r17-consolidation-dev` opened without the prior SQLite
  startup failure.

Exact dev-engine provenance:

- The remote current-source project venv resolved to
  `/Users/eric/mlx/vllm-mlx-r17-consolidation/.venv/bin/python3`.
- Python reported `vmlx_engine.__version__ == 1.6.16`.
- Python imported
  `/Users/eric/mlx/vllm-mlx-r17-consolidation/vmlx_engine/__init__.py`.
- Electron logged:
  `Found development project venv:
  /Users/eric/mlx/vllm-mlx-r17-consolidation/.venv/bin/python3`.

Open packaging findings:

- The first clean `uv pip install -e .` on Python `3.13.11` failed while
  resolving `librosa -> numba 0.53.1 -> llvmlite 0.36.0`; that llvmlite
  release rejects Python 3.13. The live dev venv was instead cloned from the
  functioning v1.6.16 project venv and its editable `vmlx` install was replaced
  with this checkout, then provenance was re-read as above.
- This is valid current-source dev proof, not release-bundle proof.
- `bundle-python.sh`, bundled critical-import verification, signed app
  provenance, Sequoia/Tahoe packaging, installation, signing, notarization,
  stapling, and installed-app smoke remain open.

### R17-003 Ornith/Qwen3.5 Electron, gateway, and cache boundary

Status: `LIVE PARTIAL / PARSERS+STREAMING PASS / PARTIAL SSD REUSE FAIL`

Bundle and launch:

- Bundle:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/Ornith-1.0-9B-JANG_4M`.
- `config.json`: `model_type=qwen3_5`,
  `Qwen3_5ForConditionalGeneration`, hybrid 24 linear-attention plus 8
  full-attention layers, 262144 text context, vision config present.
- Weight quantization: affine JANG_4M, group size 64, 4-bit default with
  explicit 8-bit tensors; `jang_config.json` reports target 4.0 and actual
  4.66 bits. This is affine JANG, not JANGTQ/MXTQ.
- `generation_config.json` supplies no sampling defaults beyond EOS/cache.
- `chat_template.jinja` is 7594 bytes and contains native Qwen tools,
  reasoning-history rendering, and `enable_thinking`.
- Electron Start-button PID: initial `72441`, restarted `72753`.
- Exact argv used current project Python with:
  `--is-mllm --tool-call-parser qwen --reasoning-parser qwen3
  --use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000
  --enable-block-disk-cache --block-disk-cache-max-gb 10`.
- No load-error toast occurred. `/health` reported `model_loaded=true`.

Electron turns:

1. `R17-ORN-UI-T1`: Auto produced a separate 617-character Reasoning rail,
   rendered `\(47 \times 2 = 94\)` visually as math, and emitted the exact
   non-empty terminal answer. UI metrics: 221 tokens, 83.7 t/s, 57.5 pp/s,
   73 prompt tokens, 1.27 s TTFT, 4.0 s total.
2. `R17-ORN-UI-T2`: the model emitted `file_info(panel/package.json)`, but the
   tool failed truthfully because Built-in Coding Tools had no Working
   Directory. The UI did not hallucinate a tool result.
3. After setting the UI Working Directory to the current checkout,
   `R17-ORN-UI-T3` emitted one successful `file_info` call, two separate
   reasoning phases (141 and 223 characters), and the exact continuation
   `R17-ORN-UI-T3-DONE SIZE=5.2 KB`. UI metrics: 93 tokens, 62.8 t/s,
   2164.9 pp/s, 5326 prompt tokens, 0.37 s TTFT, 3.8 s total.

Gateway streaming:

- Three-generation Chat Completions sequence:
  - T1: 190 reasoning characters, exact visible answer, `finish=stop`.
  - T2: 130 reasoning characters, schema-valid required
    `lookup_value({"key":"panel-package"})`, `finish=tool_calls`, no premature
    visible answer.
  - T3: 139 reasoning characters, tool-result continuation, exact visible
    `R17-ORN-API-T3-DONE SIZE=5.2 KB`, `finish=stop`.
  - No `<think>`, `[THINK]`, or tool-control marker leaked to visible content.
  - T3 restored 2572 tokens from resident
    `paged+ssm+tq-native` state.
- Responses: 268 reasoning characters through
  `response.reasoning_summary_text.delta`, progressive output text, one
  `response.completed`, no marker leak.
- Anthropic: 530 reasoning characters through `thinking_delta`, progressive
  text, balanced blocks, one `message_stop`, no marker leak.
- Ollama: 231 thinking characters, progressive content, one terminal
  `done_reason=stop`, no marker leak.

Cache state and failing boundary:

- Before restart, health showed 19 TQ-native block-disk writes, 1109 attention
  KV tokens on disk, and 12437 typed SSM companion tokens on disk.
- The in-process three-generation Chat sequence recorded one accepted resident
  hit for 2572 tokens.
- After Electron `Save & Restart`, a changed-tail request found 39 persisted
  TQ-native attention blocks / 2496 tokens. Counters increased by
  `disk_hits=39` and `tq_native_hits=39`.
- The request did **not** receive cache-hit credit. Logs then reported:
  `SSM fetch MISS ... store_size=0` and
  `2496 KV blocks found but no SSM companion state — full prefill required`.
- Therefore Paged-On partial SSD lookup is proven, but safe usable hybrid
  partial restore is not. The current result is a correctness-safe full
  prefill fallback, not the requested partial SSD reuse.
- Paged-Off SSD-only changed-tail reuse, exact restart restore, new-chat reuse,
  new-session reuse, eviction/refault, corrupt/missing companion fallback, and
  media-salt behavior remain open.

Other live observations:

- Health selected q4 TurboQuant at the attention-KV storage boundary while
  preserving typed full-precision hybrid state.
- The bundle declares one MTP layer but `jang_config.drop_mtp=true` and has no
  MTP tensors. Health truthfully reported `metadata_inconsistent` and did not
  activate MTP.
- The M5 Max health record reported the affine MLX quantized-matmul path, but
  native-accelerator activation remained false with
  `host_not_known_na_capable`; no native-accelerator speed claim is closed.


### R17-004 restart-safe hybrid partial SSD checkpoint discovery

Status: `SOURCE+FOCUSED TEST PASS / ORNITH PAGED-ON+SSD-ONLY LIVE PASS / OTHER ARCHETYPES OPEN`

Source change and ownership:

- Pushed source head: `be6cc84979e57f910dc9d952c9222317ae721b7c`.
- `SSMCompanionDiskStore.candidate_lengths()` lazily indexes valid sidecar
  `num_tokens` boundaries after process restart.
- `SSMCompanionCache.fetch_longest_prefix()` merges those shorter L2
  boundaries with its L1 index only after an exact-boundary miss.
- Candidate discovery is not acceptance. The normal typed `fetch()` still
  recomputes the model/prefix key and validates record version, runtime cache
  fingerprint, safetensors header, tensor reconstruction, and completeness.
  Attention-KV-only hybrid hits still fall back safely.

Remote focused verification on `Erics-M5-Max.lan`:

- `tests/test_ssm_companion_cache.py` -> `54 passed`.
- `tests/test_mllm_scheduler_cache.py tests/test_cache_bypass.py` ->
  `167 passed, 4 skipped`.
- Python compile and `git diff --check` -> pass.

Live current-source Electron/API proof:

- Real Electron Save & Restart loaded pushed source into PIDs `74037` and
  `74256`; the local stale installed `1.6.9` app was not used.
- A unique 37,863-character system prefix established a real multi-turn cache
  lineage through the Electron gateway. The resident setup hit saved `9,279`
  tokens and exact-finaled the requested markers.
- To prove the shorter-checkpoint branch rather than another exact restore,
  the test-specific exact `9,279`-token SSM sidecar for that unique prefix was
  removed after restart while the `9,216`-token typed checkpoint and attention
  block L2 records remained. No unrelated cache root or entry was removed.
- Paged RAM on:
  - block L2 found `145` q4-TQ attention blocks / `9,279` tokens;
  - SSM L2 restored the shorter `9,216`-token checkpoint (`24` typed states);
  - hit credit was reduced from `9,279` to `9,216`;
  - only `109` tail tokens were prefetched;
  - visible SSE exact-finaled `R17-SSD-PARTIAL-C`, one `stop`, one `[DONE]`.
- Paged RAM off + Block Disk on:
  - health before and after showed `backend_mode=block_disk_only`,
    `paged_ram_enabled=false`, `disk_only=true`, `ram_tokens_cached=0`, and
    `l1_resident_bytes=0`;
  - block L2 again found `145` q4-TQ blocks / `9,279` tokens;
  - SSM L2 restored `9,216` tokens and prefetched only the `115`-token tail;
  - visible SSE exact-finaled `R17-SSD-ONLY-PARTIAL-D`, one `stop`, one
    `[DONE]`;
  - `candidate_length_scans=1`, one typed disk hit, and no unsafe full-credit
    KV-only acceptance.
- Electron settings were restored to supported Paged RAM on + Block Disk on;
  final engine PID is `74400`.

Retained evidence:

- `ornith-partial-ssd-restart.json` in this directory.

Boundary:

- This closes the hybrid Qwen/Ornith archetype for restart partial SSD
  discovery with Paged RAM both on and off. It does not inherit closure for
  Laguna/Gemma rotating state, MiniMax-M3 sparse/indexer state, DSV4 composite
  state, ZAYA CCA, or openPangu native prompt state.

### R17-005 canonical bundle generation-default resolver

Status: `SOURCE+FOCUSED TEST PASS / GEMMA LIVE PARITY PENDING`

Root cause and ownership:

- The Electron main process had two independent implementations of bundle
  sampling-default selection:
  `ipc/models.ts::readGenerationDefaults()` for Chat Settings and
  `sessions.ts::readBundleStartupDefaults()` for session startup/settings.
- Both intended the same precedence—JANG `chat.sampling_defaults` over
  `generation_config.json`—but already disagreed at one boundary:
  `readGenerationDefaults()` accepted any numeric `max_new_tokens`, while the
  startup path accepted only positive values.
- The duplicate logic also repeated disabled-`top_k`, `do_sample=false`, and
  mode-specific repetition-penalty selection, including DSV4's scalar-first
  direct-chat exception. Future edits could therefore make first-session UI,
  saved session state, and request defaults disagree.

Source correction:

- Added one pure
  `resolveBundleGenerationDefaults(generationConfig, jangConfig, modelConfig)`
  owner in `panel/src/shared/sessionGenerationDefaults.ts`.
- Both main-process readers now use that resolver. File I/O and
  thinking-budget capability detection remain in their owning main-process
  surfaces; only sampler/default selection was consolidated.
- Preserved field-by-field JANG precedence, explicit greedy
  `do_sample=false`, disabled negative `top_k -> 0`, DSV4 repetition
  precedence, and capability-only provenance.
- Invalid/non-positive `max_new_tokens` is no longer surfaced as a model-owned
  output default. This does not copy the model default into the server's
  `--max-tokens`; explicit chat/API output limits remain request-owned, while
  context remains `--max-prompt-tokens`.

Remote focused verification:

- Generation/session/effective-default tests: `35 passed`.
- Settings/reset/override flow tests: `318 passed`.
- Panel TypeScript `tsc --noEmit`: pass.
- `git diff --check`: pass.

Boundary:

- This is source/focused-test proof only. The next gate must use a different
  real bundle (Gemma 4) and verify initial Electron sliders, saved overrides,
  reset-to-bundle behavior, SQLite/session persistence, preview, engine argv,
  health, and direct/gateway request values. No cross-family settings claim is
  closed from these tests alone.

### R17-006 Gemma 4 vendored mixed-SWA prefix reconstruction

Status: `SOURCE+FOCUSED TEST+PAGED-ON LIVE PASS / SSD-ONLY+Q4+RESTART OPEN`

Root cause:

- The real Gemma 4 cache is created by vendored
  `mlx_vlm.models.cache.KVCache` and `RotatingKVCache` classes.
- `_hybrid_cache_layout()` and `_fix_hybrid_cache()` recognized only
  `mlx_lm` class identities. All 48 real slots were therefore classified as
  non-attention slots even though the bundle declares 40 sliding-attention and
  eight full-attention layers.
- After a valid paged prefix was reconstructed, `_fix_hybrid_cache()` compared
  48 restored entries with zero expected attention positions and replaced the
  whole prefix with `language_model.make_cache()`: a correctly shaped but
  entirely empty cache. The request retained cache-hit credit and decoded from
  empty state, producing the repeated-`1` warm corruption.

Source correction:

- Commit `9f5b1bde20089dc3b998a5522c43ef9e35805395` adds one structural
  attention-cache classifier for layout decisions across the `mlx_lm` and
  `mlx_vlm` namespaces.
- Mixed-SWA layouts no longer enter the unrelated SSM-companion path.
- A non-zero claimed hit that becomes an all-empty cache after reconstruction
  is rejected and safely re-prefilled instead of decoding corrupt state.
- `last_cache_execution.blocks` now reads the production
  `BlockTable.block_ids`; the old nonexistent `.blocks` read always displayed
  zero.
- Eight focused mixed-SWA/cache tests, Python compile, and `git diff --check`
  pass on the M5 Max.

Bundle and live provenance:

- Exact bundle:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/gemma-4-12B-it-qat-JANG_4M`.
- Retained SHA-256 values and model-derived fields are in
  `gemma-mixed-swa-paged-live.json`.
- Real Electron Stop -> Start loaded PID `95416` from the consolidation
  checkout and its project venv; no UI error alert appeared.
- Launch argv selected `gemma4` reasoning/tool parsers, Paged RAM with
  64-token blocks and 1,000 configured blocks, Block Disk L2 at 10 GB, and
  explicit KV cache quantization `none`.

Electron and protocol evidence:

- Three same-chat Electron turns visibly passed:
  1. separate 156-character reasoning rail, rendered math, exact visible final;
  2. separate 363-character rail, exactly one real
     `file_info(panel/package.json)` call/result, exact `5.2 KB` final;
  3. separate 484-character rail, no tool call, exact turn-one recall and
     arithmetic final.
- The different reasoning content/lengths and tasks rule out byte-identical
  stale replay. The retained screenshot is
  `gemma-mixed-swa-ui-3turn.png`.
- Raw Responses emitted 164 reasoning-summary deltas and 22 output-text
  deltas, preserved `\(47 \times 2 = 94\)` byte-for-byte, emitted one
  `response.completed`, and leaked no native markers.
- Raw Chat emitted reasoning only in `delta.reasoning_content`, visible text
  only in `delta.content`, then `stop`, usage, and `[DONE]`, with no native
  marker leakage.

Cache evidence:

- Final-source exact warm row restored 42 tokens as `paged+mixed_swa`,
  reconstructed successfully, reported one real block, and returned the same
  exact marker as cold.
- A deterministic 495-output-token cold/warm A/B both produced the exact
  integers 1 through 120 and were byte-identical; warm restored 52 tokens as
  `paged+mixed_swa`. This falsifies the proposed need to synchronously evaluate
  rotating cache state on every decode token, so no speculative performance
  change was added.
- The live disk cap also evicted 98 older blocks and returned to 7.79 GB below
  the configured 10 GB ceiling.

Boundary:

- This closes the current-source Paged-On, explicit-TQ-None Gemma JANG_4M
  corruption and telemetry row only.
- Paged-Off SSD-only exact/partial reuse, fresh-process restore, forced L1
  eviction/refault, q4 storage-boundary TQ, MXFP8, advertised media, and
  current-source Anthropic/Ollama rows remain open. This is not a Gemma family
  release pass and does not unlock v1.6.17 packaging.
