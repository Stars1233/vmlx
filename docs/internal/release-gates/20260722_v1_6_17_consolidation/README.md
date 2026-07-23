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
