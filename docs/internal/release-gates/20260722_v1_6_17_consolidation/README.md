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

### R17-011 Gemma reasoning/tools/KaTeX and direct-audio boundary

Status:
`TEXT+PROTOCOL PASS / AUDIO TRANSPORT PASS / AUTO AUDIO QUALITY FAIL`.

Exact live bundle and process:

- `JANGQ-AI/gemma-4-12B-it-qat-JANG_4M`, affine JANG_4M group size 32.
- Real Electron Start-button PID `24290`, port `8001`, gateway `8088`.
- Bundle sampling defaults and the first-use Chat drawer agreed on
  temperature `1.0`, top-p `.95`, and top-k `64`.

Scoped live passes:

- Three Electron turns showed separate reasoning, one exact real
  `file_info(panel/package.json)` tool/result continuation, cache-aware
  follow-up, and a visibly rendered KaTeX fraction with zero KaTeX errors.
- Chat, Responses, Anthropic, and Ollama each streamed private reasoning and
  visible content separately and terminalized truthfully. Responses,
  Anthropic, and Ollama completed exact real tool loops.
- Gateway Chat preserved literal LaTeX bytes exactly. The renderer displayed
  `94/90 = 47/45` as stacked fractions.
- The retained WAV reached both direct and gateway Responses as
  `input_audio`. With thinking explicitly Off, both paths emitted 20
  progressive content deltas and the exact transcript.

Open failure:

- Default Auto did not preserve transcription quality in Electron. The
  existing chat overthought for 761 tokens and added spurious text; a fresh
  chat kept its 790 reasoning characters separate but misheard `Cobalt` as
  `code volt`. The attachment was present in SQLite and live model telemetry,
  so this is not classified as attachment loss or stream corruption.
- No forced-Off fallback, output rewrite, prompt coercion, or sampler clamp
  was added. Gemma Auto audio quality remains release-blocking.

Evidence:

- `gemma-reasoning-tools-math-audio-live.json`
- `gemma-ui-audio-auto-fresh.png`

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

Status: `SOURCE+FOCUSED TEST+GEMMA ELECTRON/DIRECT/GATEWAY LIVE PASS / CROSS-FAMILY OPEN`

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
- Added one engine-owned
  `server.py::_model_effective_defaults_status()` status contract. It keeps
  artifact-owned `sampling_defaults` separate from the effective defaults an
  omitted request receives after explicit server/session overrides and bounded
  fallback. `/health` and `/v1/capabilities` now call the same helper.
- Integer sampler fields are normalized on the status surface. The change is
  diagnostic only: it does not inject bundle defaults into argv or mutate
  explicit request values.

Remote focused verification:

- Generation/session/effective-default tests: `35 passed`.
- Settings/reset/override flow tests: `318 passed`.
- Panel TypeScript `tsc --noEmit`: pass.
- `git diff --check`: pass.
- New engine helper/endpoint and adjacent DSV4/greedy tests: `4 passed`.
- Broader runnable settings/health/capabilities selection: `57 passed`.
  Three pre-existing async tests could not execute because the project venv
  lacks `pytest-asyncio`; this is recorded as an environment limitation, not
  counted as a product pass or failure.

Gemma live parity:

- The real Electron Chat Settings surface initially displayed Auto,
  temperature `1.00`, top-p `0.95`, top-k `64`, min-p `0`, repetition penalty
  `1.00`, and blank model-owned output/thinking limits. SQLite session defaults
  contained corresponding metadata while the new chat sampler overrides were
  NULL.
- Saving `0.75/0.90/17/0.02/1.05/333` with Thinking On produced exactly those
  values in SQLite, the Electron request diagnostic, and final engine kwargs.
  New Chat returned to bundle defaults; Reset cleared sampler overrides.
- A saved `0.60/top-k 7/max 111/Off` survived Save & Restart, reached the
  engine exactly, produced no reasoning box, and reset cleanly afterward.
- After the source change, the real Server Settings Save & Restart button
  replaced PID `97058` with PID `97849`. Command, cwd, venv shebang, and
  Electron log all point at this consolidation checkout.
- Live `/health` and `/v1/capabilities` both report bundle defaults
  `1.0/0.95/64` and effective defaults
  `1.0/0.95/64/min-p 0/max-output 16384`. The blank UI max-output field is
  truthful: this Gemma bundle declares no `max_new_tokens`; `16384` is the
  engine's omitted-request reasoning fallback, not a hidden UI value.
- A fresh Electron Auto turn showed a separate 383-character reasoning box,
  exact non-empty final, and a 3,584-token `paged+mixed_swa+disk` hit. Raw
  direct Chat emitted 99 reasoning deltas, 14 content deltas, `stop`, and one
  `[DONE]`. Electron-gateway Responses emitted 187 reasoning-summary deltas,
  15 output-text deltas, and exactly one `response.completed`. Neither stream
  leaked control markers.
- Retained evidence:
  `gemma-settings-health-gateway-live.json`.

Boundary:

- This closes the current Gemma representative row through UI, SQLite,
  restart/argv, health/capabilities, direct Chat, and Electron-gateway
  Responses. It does not inherit closure for cross-family unusual top-k,
  mode-specific repetition penalties, bundle-owned max output, explicit
  context caps, model swaps, app restart, sleep/wake, or signed-app repetition.

### R17-006 Gemma 4 vendored mixed-SWA prefix reconstruction

Status: `SOURCE+FOCUSED TEST+PAGED-ON/OFF+SSD-RESTART LIVE PASS / Q4+MEDIA OPEN`

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

SSD-only, restart, eviction, and tier-promotion evidence:

- The real Server Settings drawer allowed `In-Memory Paged Cache (RAM)` Off
  while `Block Disk Cache (SSD / L2)` remained enabled and selectable.
- `Save & Restart` loaded PID `95934` with `--no-paged-cache`,
  `--enable-block-disk-cache`, and the 10 GB disk limit. Health reported
  `backend_mode=block_disk_only`, zero RAM-cached tokens, and zero L1 resident
  bytes.
- Same-process exact replay restored 11,058 of 11,059 input tokens from 173
  SSD blocks. A changed-tail request restored 11,008 of 11,065 tokens from 172
  SSD blocks. Both exact-finaled as `block-disk+mixed_swa`.
- Visible Stop -> Start loaded new PID `96185` with zero request hits and zero
  RAM residency while 492 SSD blocks remained. Exact restart replay restored
  11,064 of 11,065 tokens; changed-tail restart replay restored 11,008 of
  11,064. Both exact-finaled from SSD.
- The changed-tail write temporarily reached 10.191 GB. The configured 10 GB
  policy then evicted 106 blocks and returned the store to 387 blocks / 7.900
  GB. This is observed enforcement, not slider-presence inference.
- The UI was restored to Paged RAM On + Block Disk On and restarted as PID
  `96442`. Its first request promoted 173 persisted SSD blocks and reported
  `paged+mixed_swa+disk` plus 173 promotion hits; the second request used the
  resident tier as `paged+mixed_swa` with no additional disk hit.
- Full structured evidence is retained in
  `gemma-mixed-swa-ssd-restart-live.json`.

Boundary:

- This closes current-source Paged-On resident reuse, Paged-On SSD promotion,
  Paged-Off SSD-only exact/partial reuse, fresh-process restore, disk-cap
  eviction, and explicit-TQ-None Gemma JANG_4M corruption/telemetry.
- It also falsifies the broad claim that the production Gemma block-disk path
  currently uses an unusable MLLM key/offset contract. A separate legacy
  prompt-disk source discrepancy remains an unproven lead and was not
  rewritten without live failure evidence.
- q4 storage-boundary TQ, MXFP8, advertised media, and current-source
  Anthropic/Ollama rows remain open. This is not a Gemma family release pass
  and does not unlock v1.6.17 packaging.

### R17-007 DSV4 bundle-owned pool codec and math transport/rendering

Status: `VERIFIED-LIVE_SCOPED / DSV4 FAMILY MATRIX OPEN`.

Root causes and changes:

- The inspected DSV4 bundle declares
  `jang_config.cache.pool_quant_default=false`, but Electron creation/reset
  paths and direct CLI startup hard-coded the native CSA/HCA pool codec On.
- Model detection now exposes the stamped boolean. Create/reset/adopt/session
  hydration use the stamp when no saved explicit value exists; saved user
  values remain authoritative.
- Direct CLI/model loading now resolves explicit env first, bundle stamp
  second, and the historical enabled fallback only for unstamped legacy
  bundles.
- Completed inline math now cannot consume later reasoning paragraphs after an
  unmatched `\(`. KaTeX parse failures use escaped readable fallback instead
  of a visible `.katex-error` node.

Current-source live evidence:

- Real Electron Reset displayed native composite prefix On and DSV4 pool codec
  Off. SQLite stored prefix/paged/block-L2 On and pool codec Off.
- Real Electron Start loaded the 104.6 GB affine DSV4 bundle as PID `3423`
  from this checkout. Health reported native composite cache, 256-token pages,
  block SSD L2 On, generic TQ KV Off, and
  `native_cache.pool_quant={enabled:false,env:"0"}`.
- Live UI Auto produced a separate 2,364-character reasoning rail and a
  completed visible answer. Valid delimited math created KaTeX DOM; after the
  renderer fix the completed surface had zero `.katex-error` nodes.
- Raw Chat SSE with thinking Off reconstructed byte-exact
  `\(\frac{43}{17}\)`, then `stop` and one `[DONE]`.
- Raw Chat SSE with thinking On emitted 180 `reasoning_content` deltas and 61
  content deltas, then `stop` and one `[DONE]`, with no native marker leak.

Verification:

- Panel settings/model/cache/math selection: `441 passed`.
- Updated math renderer selection: `13 passed`.
- DSV4 Python cache/loader selection: `80 passed`.
- TypeScript typecheck: pass.
- Full bundled-Python verification and Electron main/preload/renderer build:
  pass; production renderer output includes KaTeX fonts/assets.

Evidence:

- `dsv4-bundle-pool-math-live.json`

Boundary:

- This does not close DSV4 prefix hit/restart/eviction, DSML agentic loops,
  long-output quality, or other families. Overall campaign status remains
  `PARTIAL / NOT RELEASE-READY`.

### R17-008 DSV4 exact composite reuse and DSML tool continuations

Status: `VERIFIED-LIVE_SCOPED_PARTIAL / CHANGED-TAIL TYPED REUSE OPEN`.

Snapshot/store finding:

- The earlier short DSV4 requests did not store cache because the source-owned
  prompt-boundary snapshot threshold is 256 tokens. This was expected policy,
  not evidence of a broken store.
- A deterministic 2,187-token direct Chat prompt cold-completed exactly in
  5.867 seconds and stored a clean 2,186-token, 43-layer native composite
  checkpoint. The estimated snapshot was 45,015,552 bytes.
- Its exact repeat restored all 2,186 tokens as `paged+dsv4`, exact-completed
  in 0.685 seconds, and retained nine SSD blocks / 2,186 SSD tokens.
- A same-length changed-tail request found matching earlier blocks but rejected
  them as a request-level hit. The live log explains the safety boundary:
  non-terminal DSV4 blocks carry `deepseek_v4_pending` local/SWA fragments;
  the terminal block alone owns the complete CSA/HCA composite state. Because
  the changed tail invalidated that terminal checkpoint, decoding from the
  partial chain would flatten/omit native state. The full prefill and exact
  final were correct, but safe DSV4 changed-tail reuse remains open.

Agentic protocol evidence on the same Electron-started PID:

- Direct streamed Chat emitted 65 separate reasoning characters, one
  schema-valid `file_info(panel/package.json)` call, `finish_reason=tool_calls`,
  and one `[DONE]`. After executing the real tool against the checkout, the
  continuation emitted 147 separate reasoning characters and exact visible
  `DSV4-DSML-DONE SIZE=5336`, then `stop` and one `[DONE]`.
- Electron-gateway streamed Responses emitted 258 reasoning-summary
  characters, one incremental function call, and exactly one
  `response.completed`. Its real 5,336-byte tool result continued with 352
  reasoning characters, exact visible `DSV4-RESP-DONE SIZE=5336`, and exactly
  one second `response.completed`. Neither pass leaked DSML or think markers.
- The real Electron Chat executed one visible `Info` tool card for
  `panel/package.json`, kept 271 reasoning characters in the rail, and
  continued to a non-empty final. The tool reports a human-readable `5.2 KB`;
  the model rendered that as `5200` and shortened the requested marker, so
  exact model instruction-following is retained as partial even though the
  UI/tool-loop transport completed.

Evidence:

- `dsv4-cache-dsml-live.json`

Boundary:

- Current DSV4 exact resident reuse and SSD write-through pass. Safe
  changed-tail partial reuse, Paged-Off SSD-only restoration, process-restart
  restoration, eviction/refault, long-output quality, and the remaining
  direct/gateway protocol combinations are still open. No release-ready claim
  is made.

### R17-009 DSV4 L2 owner thread and cache-equivalence fail-closed gate

Status: `OWNER-THREAD FIX VERIFIED / CACHE EQUIVALENCE FAIL / SAFE FALLBACK LIVE`.

Owner-thread root cause and repair:

- Block-disk promotion used to deserialize MLX arrays on the API
  `add_request` thread and later consume them on the scheduler's model worker.
  Restart restore therefore raised
  `There is no Stream(cpu, 3) in current thread`.
- `BlockDiskStore` now accepts the scheduler's model-worker executor and
  performs block reads there. Reconstruction failures retain a full traceback.
- Focused DSV4/cache verification passes `70 passed, 61 deselected`.
- A real Electron Save & Restart then restored 1,291 of 1,292 tokens from
  block L2 as `paged+dsv4+disk`, reconstructed successfully, emitted exact
  `DSV4-R17-L2-OWNER`, and showed no wrong-thread exception.

Semantic-equivalence failure:

- A three-turn Electron chat had a separate reasoning rail on every generated
  turn and one real tool/result continuation. Its third request found 269 of
  337 tokens from DSV4 block L2 but replayed the first turn's visible math
  answer instead of the requested marker.
- The identical 337-token Responses history with cache bypass returned exact
  `DSV4-UI-POST-L2-OK`, isolating the stale replay to cache reuse rather than
  renderer/history serialization alone.
- A later exact 336/337 SSD checkpoint was also unsafe: it looped for 2,647
  reasoning tokens with no visible answer before interruption. Exact N-1 shape
  is therefore not sufficient evidence of DSV4 CSA/HCA/SWA equivalence.

Correctness boundary:

- Commit `e9149f566` now rolls accepted DSV4 cache-hit credit back to zero,
  releases block refs, and full-prefills instead of consuming any DSV4
  paged/L2 checkpoint.
- Real Electron PID `11278` proved the gate: the log rejected the 336/337 hit,
  reported a paged miss, processed all 337 tokens, and the UI showed no cached
  token credit. The stale math replay disappeared.
- The full-prefill replay still entered a separate long reasoning loop and was
  interrupted after 2,616 tokens without visible content. That is retained as
  `FAIL`, not attributed to cache and not hidden by a forced closer or sampler
  override.
- DSV4 cache storage remains observable for investigation, but DSV4 cache
  reuse is disabled until exact and partial cold-vs-warm equivalence passes.
  This is a correctness checkpoint, not completion of the required partial
  SSD-reuse feature.

Evidence:

- `dsv4-l2-owner-and-equivalence-live.json`

Overall status remains `PARTIAL / NOT RELEASE-READY`.

### R17-010 Preserve reasoning mode across ordinary post-tool continuation

Status: `VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

Root cause and repair:

- The Electron Responses request builder used the same recovery helper for an
  ordinary exact-final pass after a successful tool and for the bounded
  recovery after an actually empty or incomplete post-tool answer.
- Both paths removed the completed tool, but both also forced
  `enable_thinking=false`, `thinking_mode=instruct`, and
  `chat_template_kwargs.enable_thinking=false`. A real tool continuation could
  therefore silently turn explicit Reasoning On or model-owned Auto into Off.
- `applyPostToolAnswerPolicy` now retires the completed tool for both paths,
  then preserves the requested reasoning mode for the ordinary follow-up.
  Only a true bounded empty-answer recovery may request instruct mode.

Current verification:

- Focused panel verification passes `22/22`; TypeScript typecheck passes.
- The real Electron Start button loaded
  `/Volumes/EricsLLMDrive/JANGQ-AI/Laguna-S-2.1-JANG_4M` from the consolidation
  checkout venv as PID `14831`, port `8003`, with no error toast.
- Before the fix, the initial tool pass resolved thinking On and the ordinary
  post-tool pass resolved Off.
- After the fix, both phases resolved thinking On. The UI executed exactly one
  real `file_info({"path":"panel/package.json"})`, retained its real `5.2 KB`
  result, and continued to
  `LAGUNA-UI-ON-TOOL-POSTFIX-DONE SIZE=5.2 KB`.
- That specific tool sample chose a direct visible path, so it is not counted
  as a non-empty reasoning-rail row merely because the route was enabled.
- A separate deterministic explicit-On Electron row persisted 325 reasoning
  characters in `reasoning_content`, displayed the distinct Reasoning rail,
  and produced non-empty visible content with zero raw markers, replacement
  characters, or KaTeX errors. It included extra visible calculation before
  the requested final, so exact instruction following remains partial.
- New Chat returned to Auto and the bundle-derived Laguna generation defaults
  instead of inheriting the saved explicit-On chat override.

Evidence:

- `laguna-post-tool-thinking-live.json`
- `laguna-ui-on-rail.png`

Remaining boundary:

- This closes the hidden post-tool Off override and a scoped Laguna UI rail/tool
  row. It does not close the required raw Chat/Responses/Anthropic/Ollama
  sequences, mixed-SWA Paged-On/Paged-Off SSD hierarchy, eviction/refault,
  broader parser families, full suites, installed app, or release gates.
- Overall status remains `PARTIAL / NOT RELEASE-READY`.

### R17-011 Canonical effective reasoning parser and UI/API LaTeX split

Status: `VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

Owning-layer repair:

- One shared resolver now owns effective reasoning-parser selection for session
  launch, command preview, Chat Settings, session shell/toolbar, chat IPC,
  Harmony routing, and Ollama capability reporting.
- Explicit `None` remains a literal engine opt-out even when bundle detection
  finds a parser. Auto uses the canonicalized detected parser. A model that
  explicitly declares thinking unsupported also launches with literal `none`.
- Laguna's stale saved `qwen3` parser migrates once to `deepseek_r1` only when
  current bundle detection independently proves that parser. Later explicit
  user choices survive the versioned migration.
- A persisted literal `reasoningParser: "none"` now renders as the visible
  None option rather than an unknown select value.

Current live Electron proof:

- The real Start button loaded the Laguna JANG_4M bundle from the consolidation
  venv with no error toast.
- Auto displayed `Auto (detected: deepseek_r1)`, persisted `auto`, launched PID
  `20507` with `--reasoning-parser deepseek_r1`, kept Chat Thinking controls
  enabled, and advertised Ollama `thinking`.
- Explicit None displayed the correct option, persisted the empty hard opt-out,
  launched PID `20077` with `--reasoning-parser none`, disabled all Chat
  Thinking controls, and removed Ollama `thinking`.
- Auto was restored through the real UI and Save & Restart before leaving the
  gate.
- A fresh New Chat matched current `/health` effective defaults:
  temperature `1.0`, top-p `1.0`, top-k `20`, min-p `0`, and max output
  `32768`. The older chat's `0.00` temperature was traced to its explicit
  persisted override rather than model-default drift.

LaTeX/UI versus API contract:

- The Electron answer stored literal `\(47 \times 2 = 94\)` and fraction
  commands, but the visible answer rendered two KaTeX nodes with zero errors.
- Raw gateway Chat SSE emitted progressive content deltas that reconstructed
  literal `\(19 \times 5 = 95\)` and `\(\frac{95}{5}=19\)` with one stop and
  one `[DONE]`. The gateway did not render or rewrite model bytes.

Proof-driver lifecycle:

- `uidrv.cjs` previously left its Playwright CDP connection alive despite the
  comment claiming process exit would disconnect it. This accumulated stale
  parent SSH clients and exhausted the remote SSH daemon.
- The driver now flushes stdout and exits explicitly without calling
  `browser.close()`, so it disconnects without terminating Electron.

Verification:

- Remote focused panel result: `6` files, `494` tests passed.
- TypeScript typecheck, `node --check scripts/uidrv.cjs`, and
  `git diff --check` passed.
- Evidence:
  - `laguna-parser-settings-math-live.json`
  - `laguna-parser-math-ui.png`

Remaining boundary:

- This closes one canonical parser-settings path and the scoped UI/API LaTeX
  split. It does not close Laguna's three-turn full protocol matrix,
  mixed-SWA Paged-On/Paged-Off SSD partial reuse, eviction/refault, other
  parser families, full suites, installed app, or release gates.
- Overall status remains `PARTIAL / NOT RELEASE-READY`.

### R17-012 Laguna Paged-On and Paged-Off partial SSD reuse

Status: `VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

Paged-On hierarchy:

- Real Electron PID `20507` launched with Paged RAM, 64-token blocks, 1,000
  blocks, Block Disk L2, and q4 native SSD storage for eligible full-attention
  KV while Laguna rotating-window state stayed native.
- A 2,983-token cold request wrote the shared prefix. `DELETE
  /v1/cache?type=ram` then reduced resident prompt tokens to zero while keeping
  150 SSD blocks / 9,320 tokens.
- The changed-tail request restored 2,944 tokens as
  `paged+disk+tq-native`, promoted 46 SSD blocks, and returned exact
  `R17-LAGUNA-SSD-B`. A second changed tail restored the same 2,944 tokens from
  resident `paged+tq-native`.

Paged-Off hierarchy:

- The real Electron settings UI disabled `In-Memory Paged Cache (RAM)` while
  leaving `Block Disk Cache (SSD / L2)` enabled. Save & Restart launched PID
  `22169` with `--no-paged-cache --enable-block-disk-cache`.
- Health reported `backend_mode=block_disk_only`,
  `paged_ram_enabled=false`, `disk_only=true`, and
  `ram_mirror_policy=disk_only`.
- The first post-restart gateway request reused 2,944 of 2,984 tokens written
  by the earlier Paged-On process as `block-disk+tq-native`. Direct and gateway
  changed-tail requests repeated the same partial reuse with exact outputs,
  one stop, and one `[DONE]`.
- Three distinct Electron chats were sent while Paged Off. The exact
  cross-chat replay restored 6,528 of 6,580 tokens from
  `block-disk+tq-native`, reduced TTFT from `4.55s` cold to `1.63s`, and
  visibly returned exact `R17-LAGUNA-UI-DISK-C`.

Counter truth:

- `scheduler_cache.total_tokens_cached` is the block index and reached 3,061
  in disk-only mode. It is not resident RAM.
- The owning aggregate simultaneously reported `ram_tokens_cached=0`,
  `l1_resident_bytes=0`, `l1_indexed_tokens=3061`, and 9,513 block-L2 tokens.
  Disk-only mode therefore did not retain a RAM payload mirror.

Exit state:

- The real UI restored the supported default before leaving the gate. PID
  `22853` launched with Paged RAM and Block Disk L2 enabled; health returned
  `backend_mode=paged`, `ram_mirror_policy=resident`, and retained 19,302 L2
  tokens.

Evidence:

- `laguna-paged-on-off-ssd-partial-live.json`
- `laguna-ui-paged-off-ssd-partial.png`

Remaining boundary:

- Current-source bounded SSD GB eviction/refault and missing/corrupt rotating
  companion fallback remain open.
- Gemma 4 and the other typed cache archetypes still require equivalent live
  Paged-On/Paged-Off proof.
- Overall status remains `PARTIAL / NOT RELEASE-READY`.

### R17-013 Laguna Block Disk GB capacity and eviction

Status: `VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

- The real Electron Block Cache Max slider changed from 10 GB to 1 GB. Save &
  Restart persisted `blockDiskCacheMaxGb=1` and launched PID `23420` with
  `--block-disk-cache-max-gb 1`.
- The existing Laguna namespace was 2.252 GB / 309 blocks at startup. One new
  q4 native block write triggered 156 LRU deletions and reduced the namespace
  to 0.790 GB / 198 blocks, below the configured cap.
- Replaying an older 2,987-token prefix then received zero cached-token credit,
  recorded 52 SSD misses, full-prefilled, and exact-finaled
  `R17-LAGUNA-EVICTED-REFILL`. Refill writes retained the 0.788 GB bounded
  size while cumulative SSD evictions reached 206.
- The test did not synthesize a hit after deletion. An evicted prefix safely
  became a miss and repopulated the cache.
- The real UI restored the 10 GB default. Current PID `23830` is healthy with
  the restored value.

Evidence:

- `laguna-ssd-capacity-eviction-live.json`

Remaining boundary:

- Equivalent capacity enforcement is still required for other typed cache
  archetypes. Corrupt/missing rotating-state companion fallback remains open.
- Overall status remains `PARTIAL / NOT RELEASE-READY`.

### R17-014 MiniMax M2.7 reasoning/tools and partial SSD cache-key repair

Status: `VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

Combined Electron and protocol proof:

- The real Electron Start button eagerly loaded
  `dealignai/MiniMax-M2.7-JANG_K-CRACK` without requiring a first message.
  Bundle truth resolved MiniMax reasoning/tool parsers, JANG_K affine weights,
  plain KV state, q4 TurboQuant prefix storage, and temperature/top-p/top-k
  defaults `1.0/0.95/40`.
- Three closely inspected UI turns produced non-empty visible answers, separate
  reasoning rails, one exact real `file_info(panel/package.json)` call/result,
  cross-turn `5.2 KB` recall, zero native-marker/replacement-character leaks,
  and a valid KaTeX-rendered `84/12` fraction.
- Direct and gateway Chat, Responses, Anthropic, and Ollama streams emitted
  reasoning separately from progressive content and truthful terminals.
  Gateway and direct Responses each completed a real reasoning -> tool ->
  result -> reasoning -> answer loop.

Global partial-prefix defect and repair:

- Pre-fix live Paged-Off changed-tail requests shared roughly 1,162 prompt
  tokens yet received zero cache credit. The replacement-style template path
  in `_generation_prompt_cache_extra_key` hashed the last 512 characters of
  the entire rendered prompt, including the changed user tail, into every block
  key.
- The repair now derives the textual discriminator only from content introduced
  after the with/without-generation renders diverge. Token suffixes remain the
  authoritative family-marker discriminator. A regression test pins equal keys
  across different user tails with the same generation suffix.

Live hierarchy after repair:

- Paged RAM Off + SSD L2 On: a 1,631-token changed-tail request restored 1,600
  tokens from `block-disk+tq-native`, prefilling only 31. Resident RAM stayed
  exactly zero tokens/bytes.
- After a real UI restart with Paged RAM and SSD both On, the first changed-tail
  request restored 1,600/1,634 tokens from `paged+disk+tq-native` and promoted
  them to RAM. The next changed tail restored the same 1,600 tokens from
  `paged+tq-native`; SSD hits did not increase.
- Focused remote verification: `132` generation-key/paged/cache-bypass tests
  passed; `git diff --check` passed.

Evidence:

- `minimax-m27-ui-api-cache-live.json`
- `minimax-m27-ui-cache.png`
- `minimax-m27-paged-off-ssd-on.png`
- `minimax-m27-paged-on-ssd-on.png`

Remaining boundary:

- The source correction applies globally, but this row is live proof for
  MiniMax M2.7 plain KV. Laguna mixed rotating/full-attention has a separate
  live row. Gemma mixed-SWA, hybrid SSM/GDN, M3 sparse/lightning, OpenPangu
  native prompt disk, and DSV4 composite caches still need architecture-specific
  changed-tail proof.
- DSV4 partial composite reuse remains deliberately fail-closed rather than
  reporting unsafe hit credit.
- Overall status remains `PARTIAL / NOT RELEASE-READY`.
